"""Q1 cache costing and final sourced analysis."""
import math

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


def analyze():
    from final_analysis import analyze as run
    return run()

if __name__=="__main__":
    import json
    print(json.dumps(analyze(),indent=2))
