"""Cost, capacity and latency sensitivity using clearly labelled assumptions."""
import csv,json,math
from pathlib import Path
from calculations import q1
from update_verified_catalog import updated_catalog as build
ROOT=Path(__file__).resolve().parent

def model_cost(m,x):
    a=m['input_usd_per_million'];b=m['output_usd_per_million'];c=m['cache_read_usd_per_million']
    raw=(x['monthly_input']*a+x['monthly_output']*b)/1e6
    attempts=x['monthly_calls'];prefix=1500
    eligible=c is not None and m['cache_minimum_tokens'] is not None and prefix>=m['cache_minimum_tokens']
    # One cache creation per hour as a conservative explicit refresh policy.
    # All other matching calls hit the cache. Assumes traffic is regular enough
    # to keep the prefix warm; actual cache hit fraction must be logged.
    writes=720 if eligible else 0
    hits=max(0,attempts-writes) if eligible else 0
    write_multiplier=m['cache_write_multiplier']
    # For OpenAI only 1,408 of 1,500 identical tokens lie on a 128-token boundary.
    cached_prefix=1408 if m['developer']=='OpenAI' else prefix
    reduction=hits*cached_prefix*(a-(c or 0))/1e6
    premium=writes*cached_prefix*a*(write_multiplier-1)/1e6
    storage=1500*720/1e6 if eligible and m['developer']=='Google' else 0
    cached=raw-reduction+premium+storage
    reasoning_unknown=m['model'] in ['Grok 4.7','gpt-oss-120b']
    return {'model':m['model'],'rate_evidence':m['evidence_status'],'uncached_usd':raw,
            'cache_scenario_usd':cached,'eligible_prefix':eligible,'billable_cached_prefix':cached_prefix if eligible else 0,
            'cache_creations_assumed':writes,'cache_hits_assumed':hits,'write_premium_usd':premium,
            'storage_usd':storage,'within_budget_uncached':None if reasoning_unknown and raw<=x['B'] else raw<=x['B'],
            'within_budget_cache_scenario':None if reasoning_unknown and cached<=x['B'] else cached<=x['B'],
            'reasoning_surcharge_unknown':reasoning_unknown}

def latency(x,ttft,tps,delay):
    terms=[]
    for s in x['steps']:
        # Context-sensitive TTFT here is an assumption, not a provider result.
        prefill=ttft+.00005*(s['input']-2000)
        seconds=prefill+s['output']/tps
        terms.append({**s,'ttft_assumed_seconds':prefill,'tps_assumed':tps,'llm_seconds_expected':seconds*s['attempts_expected']})
    return {'terms':terms,'seconds':sum(t['llm_seconds_expected'] for t in terms)+(x['N']-1)*delay,
            'tool_delay_assumed':delay,'status':'SCENARIO, NOT MEASURED'}

def analyze():
    x=q1();models=build();rows=[model_cost(m,x) for m in models]
    out=ROOT/'results';out.mkdir(exist_ok=True)
    with (out/'monthly_api_costs.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    timing_path=ROOT/'evidence'/'timing_summary.json'
    timing=json.loads(timing_path.read_text()) if timing_path.exists() else {}
    primary=timing.get('primary_balanced_measurements',{})
    delay=primary.get('pooled_median_seconds') or .5
    per_tool=primary.get('per_tool_median_seconds',{})
    remote=max(per_tool.get(name) or timing.get(name,{}).get('median_seconds') or .5 for name in ['weather','air_quality_observed'])
    local=sum(per_tool.get(name) or timing.get(name,{}).get('median_seconds') or .00005 for name in ['heat_index','knowledge_base'])
    lat={'GPT-4.1 nano':latency(x,.3,120,delay),'Gemini 2.5 Flash-Lite':latency(x,.2,180,delay)}
    # Architectural change: deterministic intent/slot router + parallel weather/AQ,
    # local HI and KB, one 120-token response generated from 2,500 input tokens.
    # This replaces the original seven-call protocol and changes its token budget.
    # Model call failures remain p with one successful retry for this scenario.
    newI=2500*(1+x['p']);newO=120*(1+x['p'])
    arch={name:{'seconds':remote+local+(.325+120/120)*(1+x['p']) if name.startswith('GPT') else remote+local+(.225+120/180)*(1+x['p']),
                'input_per_query':newI,'output_per_query':newO,'assumed_output_tokens':120,
                'non_llm_routing_validated':False} for name in lat}
    # Break-even is scoped to the ORIGINAL seven-call workload at an assumed
    # 60-second service time. Neither measured timing nor throughput is available.
    # Use cheaper uncached API to avoid assuming warm prefixes at tiny Q.
    api_unit=(newI*.1+newO*.4)/1e6
    mem=x['memory'];capacity=mem['concurrent_sequences_per_gpu'];lease=mem['gpu_usd_per_hour']*720
    holding=float(math.ceil(max(v['seconds'] for v in arch.values()))) # full resident-session scenario; not a throughput proof
    factor=(5*.35/120)*holding/60
    span=capacity/factor
    api_monthly_per_daily_q=(30+4*x['H'])*api_unit
    limit=500000
    cross=[]
    for g in range(1,math.ceil(limit/span)+1):
        lower=(g-1)*span; upper=g*span
        threshold=g*lease/api_monthly_per_daily_q
        if max(lower,threshold)<min(upper,limit):
            cross.append({'gpus':g,'daily_Q_strictly_greater_than':max(lower,threshold),'up_to_daily_Q':min(upper,limit)})
    volumes=sorted(set([0,1,x['Q']]+list(range(250,limit+1,250))))
    curve=[{'daily_Q':q,'monthly_api_usd':q*api_monthly_per_daily_q,
            'gpus':max(1,math.ceil(q*factor/capacity)),
            'monthly_self_host_usd':max(1,math.ceil(q*factor/capacity))*lease} for q in volumes]
    with (out/'break_even_curve.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(curve[0]));w.writeheader();w.writerows(curve)
    result={'monthly_costs':rows,'latency_scenarios':lat,'changed_architecture':arch,
            'tool_timing_basis':{'primary':primary,'pooled_seconds':delay,'parallel_remote_estimate_seconds':remote,'local_estimate_seconds':local,'caveat':'Dates and sample counts are recorded in primary. Parallel delay uses max of individual medians, not a measured parallel run. OpenAQ values are stale; valid for API timing only.'},
            'break_even':{'scope':f'Changed architecture, uncached cheaper API at .10/.40; {holding:g} s assumed resident-session occupancy','assumed_resident_session_seconds':holding,
              'api_usd_per_query':api_unit,'monthly_api_per_daily_Q':api_monthly_per_daily_q,'gpu_step_daily_queries':span,
              'lease_per_gpu_month':lease,'crossing_intervals':cross,'search_upper_daily_Q':limit,
              'asymptotic_self_host_per_daily_Q':lease/span,'no_crossing_at_any_volume':lease/span>=api_monthly_per_daily_q,
              'self_host_at_current_Q_usd':max(1,math.ceil(x['Q']*factor/capacity))*lease,
              'api_at_current_Q_usd':api_monthly_per_daily_q*x['Q']}}
    (out/'cost_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    r=analyze(); print(json.dumps({k:v for k,v in r.items() if k!='monthly_costs'},indent=2))
