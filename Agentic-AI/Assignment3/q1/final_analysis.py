"""Q1 estimates from sourced non-reasoning END-TO-END speed profiles."""
from pathlib import Path
import csv,json,math
ROOT=Path(__file__).resolve().parent

def reported_latency(x,ttft,tps,delay,slope=.00005):
    terms=[]
    for step in x['steps']:
        extra=slope*(step['input']-2000)
        # Reported TPS includes TTFT already. Do not add TTFT a second time.
        seconds=max(ttft,step['output']/tps)+extra
        terms.append({**step,'reported_ttft_seconds':ttft,'reported_end_to_end_tps':tps,
                      'context_increment_assumed_seconds':extra,'llm_seconds_expected':seconds*step['attempts_expected']})
    return {'terms':terms,'seconds':sum(t['llm_seconds_expected'] for t in terms)+(x['N']-1)*delay,
            'tool_delay_measured':delay,'status':'ESTIMATE from reported profile and measured tools; context slope assumed'}

def analyze():
    from calculations import q1
    from update_verified_catalog import updated_catalog
    from cost_analysis import model_cost
    x=q1();models=updated_catalog();rows=[model_cost(m,x) for m in models]
    out=ROOT/'results'
    with (out/'monthly_api_costs.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    timing=json.loads((ROOT/'evidence/timing_summary.json').read_text())
    primary=timing['primary_balanced_measurements'];delay=primary['pooled_median_seconds']
    per=primary['per_tool_median_seconds'];remote=max(per[n] for n in ['weather','air_quality_observed'])
    local=sum(per[n] for n in ['heat_index','knowledge_base'])
    candidates=[m for m in models if m['model'] in ['Gemini 2.5 Flash-Lite','Amazon Nova Lite']]
    lat={m['model']:reported_latency(x,m['ttft_seconds'],m['output_tokens_per_second'],delay) for m in candidates}
    newI=2500*(1+x['p']);newO=120*(1+x['p'])
    arch={}
    for m in candidates:
        seconds=remote+local+(max(m['ttft_seconds'],120/m['output_tokens_per_second'])+.00005*500)*(1+x['p'])
        unit=(newI*m['input_usd_per_million']+newO*m['output_usd_per_million'])/1e6
        arch[m['model']]={'seconds':seconds,'input_per_query':newI,'output_per_query':newO,
          'api_usd_per_query':unit,'monthly_usd':unit*x['monthly_queries'],'meets_T_estimate':seconds<=x['T'],
          'reported_ttft':m['ttft_seconds'],'reported_end_to_end_tps':m['output_tokens_per_second'],
          'input_rate':m['input_usd_per_million'],'output_rate':m['output_usd_per_million'],
          'non_llm_routing_validated':False}
    feasible=[name for name,v in arch.items() if v['meets_T_estimate']]
    chosen=min(feasible,key=lambda name:arch[name]['api_usd_per_query']) if feasible else min(arch,key=lambda name:arch[name]['seconds'])
    api_unit=arch[chosen]['api_usd_per_query']
    capacity=x['memory']['concurrent_sequences_per_gpu'];lease=x['memory']['gpu_usd_per_hour']*720
    holding=float(math.ceil(max(v['seconds'] for v in arch.values())))
    factor=(5*.35/120)*holding/60;span=capacity/factor
    api_slope=(30+4*x['H'])*api_unit
    curve=[{'daily_Q':q,'monthly_api_usd':api_slope*q,'gpus':math.ceil(q/span),'monthly_self_host_usd':math.ceil(q/span)*lease} for q in sorted(set([0,1,x['Q']]+list(range(250,500001,250))))]
    with (out/'break_even_curve.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(curve[0]));w.writeheader();w.writerows(curve)
    result={'monthly_costs':rows,'latency_scenarios':lat,'changed_architecture':arch,
      'candidate_choice':chosen,'speed_profile_note':'AA end-to-end output speed already includes TTFT. max(TTFT,O/rate) is an explicit transfer estimate, plus assumed context-growth increment; not a service guarantee.',
      'tool_timing_basis':{'primary':primary,'pooled_seconds':delay,'parallel_remote_estimate_seconds':remote,'local_estimate_seconds':local,'caveat':'max of two individual medians estimates parallel wait; not a measured parallel run. Historical observed values are valid timing evidence, not current pollution.'},
      'break_even':{'scope':'Changed architecture, cheaper candidate meeting T by estimate: '+chosen,'assumed_resident_session_seconds':holding,
        'api_usd_per_query':api_unit,'monthly_api_per_daily_Q':api_slope,'gpu_step_daily_queries':span,'lease_per_gpu_month':lease,
        'crossing_intervals':[],'search_upper_daily_Q':500000,'asymptotic_self_host_per_daily_Q':lease/span,
        'no_crossing_at_any_volume':lease/span>=api_slope,'self_host_at_current_Q_usd':math.ceil(x['Q']/span)*lease,'api_at_current_Q_usd':api_slope*x['Q']},
      'latency_sensitivity':{str(s):{m['model']:reported_latency(x,m['ttft_seconds'],m['output_tokens_per_second'],delay,s)['seconds'] for m in candidates} for s in [0,.00005,.0002]},
      'changed_peak':{'llm_rpm':x['peak_queries_per_minute']*(1+x['p']),'input_tpm':x['peak_queries_per_minute']*newI,'output_tpm':x['peak_queries_per_minute']*newO}}
    (out/'cost_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
if __name__=='__main__':
    d=analyze();print(json.dumps({k:v for k,v in d.items() if k not in ['monthly_costs','latency_scenarios']},indent=2))
