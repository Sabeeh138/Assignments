"""Collect public source snapshots and honest, append-only tool timings.

Run timing collection on TWO distinct Karachi dates. No synthetic timings.
Open-Meteo air quality is modelled, not a ground-station observation.
An OpenAQ key can be supplied as OPENAQ_API_KEY for an observed-data probe.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, statistics, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / 'sources'
KARACHI = timezone(timedelta(hours=5))

def load_openaq_key():
    if os.environ.get('OPENAQ_API_KEY'):return
    candidates=[ROOT/'.secrets'/'openaq.key',ROOT.parent/'Assignment_Beta_Submission'/'q1'/'.secrets'/'openaq.key']
    for path in candidates:
        if path.is_file():
            os.environ['OPENAQ_API_KEY']=path.read_text(encoding='utf-8').strip()
            return

def stamp():
    return datetime.now(KARACHI).isoformat()

def diagnose_air():
    load_openaq_key()
    r=requests.get('https://api.openaq.org/v3/locations?coordinates=24.8607,67.0011&radius=25000&limit=100',headers={'X-API-Key':os.environ['OPENAQ_API_KEY']},timeout=30)
    r.raise_for_status();payload=r.json()
    out=ROOT/'evidence';out.mkdir(exist_ok=True)
    (out/'openaq_locations_public.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps([{'id':v['id'],'name':v.get('name'),'latest':v.get('datetimeLast'),'sensors':[{'parameter':s.get('parameter',{}).get('name'),'id':s.get('id'),'latest':s.get('datetimeLast')} for s in v.get('sensors',[])]} for v in payload.get('results',[])],indent=2))

def heat_index(celsius, humidity):
    """NWS Rothfusz regression, simple-formula gate and humidity adjustments."""
    t = celsius * 9 / 5 + 32
    r = humidity
    simple = (0.5 * (t + 61 + (t - 68) * 1.2 + r * .094) + t) / 2
    if simple < 80:
        return (simple - 32) * 5 / 9
    hi = (-42.379 + 2.04901523*t + 10.14333127*r - .22475541*t*r
          - .00683783*t*t - .05481717*r*r + .00122874*t*t*r
          + .00085282*t*r*r - .00000199*t*t*r*r)
    if r < 13 and 80 <= t <= 112:
        hi -= ((13-r)/4) * math.sqrt((17-abs(t-95))/17)
    elif r > 85 and 80 <= t <= 87:
        hi += ((r-85)/10) * ((87-t)/5)
    return (hi - 32) * 5 / 9

def fetch_source(item):
    name, url = item
    path = SOURCES / name
    try:
        r = requests.get(url, timeout=45, headers={'User-Agent': 'AssignmentResearch/1.0'})
        path.write_bytes(r.content)
        return {'file': name, 'url': url, 'accessed': stamp(), 'status': r.status_code,
                'sha256': hashlib.sha256(r.content).hexdigest()}
    except requests.RequestException as e:
        return {'file': name, 'url': url, 'accessed': stamp(), 'error': str(e)}

def snapshot():
    SOURCES.mkdir(parents=True, exist_ok=True)
    items = [
      ('openrouter_models.json', 'https://openrouter.ai/api/v1/models'),
      ('qwen7_config.json', 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json'),
      ('qwen7_card.md', 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/README.md'),
      ('qwen7_index.json', 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/model.safetensors.index.json'),
      ('together_pricing.html', 'https://www.together.ai/pricing'),
      ('rag_paper.html', 'https://www.researchgate.net/publication/397593400_A_Comparative_Evaluation_of_RAG_Architectures_for_Cross-Domain_LLM_Applications_Design_Implementation_and_Assessment'),
    ]
    ids = ['qwen/qwen-2.5-7b-instruct','qwen/qwen3-235b-a22b-2507',
           'meta-llama/llama-3.3-70b-instruct','meta-llama/llama-3.1-8b-instruct',
           'mistralai/mistral-small-3.2-24b-instruct','deepseek/deepseek-chat-v3-0324',
           'openai/gpt-4.1-mini','openai/gpt-4.1-nano',
           'anthropic/claude-haiku-4.5','anthropic/claude-sonnet-4.5',
           'google/gemini-2.5-flash-lite','x-ai/grok-4.7']
    items += [(s.replace('/','__')+'.json', f'https://openrouter.ai/api/v1/models/{s}/endpoints') for s in ids]
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(fetch_source, items))
    (SOURCES/'snapshot_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps([{'file':r['file'],'status':r.get('status'),'error':r.get('error')} for r in records],indent=2))

def timings(count):
    load_openaq_key()
    out = ROOT/'evidence'
    out.mkdir(exist_ok=True)
    path = out/'tool_latency_raw.csv'
    fields = ['timestamp_karachi','tool','iteration','elapsed_seconds','success','http_status','data_kind','detail','response_sha256']
    urls = {
      'weather': 'https://api.open-meteo.com/v1/forecast?latitude=24.8607&longitude=67.0011&current=temperature_2m,relative_humidity_2m,wind_speed_10m&timezone=Asia%2FKarachi',
      'air_quality_modelled': 'https://air-quality-api.open-meteo.com/v1/air-quality?latitude=24.8607&longitude=67.0011&current=pm10,pm2_5&timezone=Asia%2FKarachi'
    }
    observed_sensors={}
    if os.environ.get('OPENAQ_API_KEY'):
        discovery=requests.get('https://api.openaq.org/v3/locations?coordinates=24.8607,67.0011&radius=25000&limit=100',headers={'X-API-Key':os.environ['OPENAQ_API_KEY']},timeout=20)
        discovery.raise_for_status()
        locations=discovery.json().get('results',[])
        locations.sort(key=lambda loc:(loc.get('datetimeLast') or {}).get('utc',''),reverse=True)
        for loc in locations:
            sensors={s['parameter']['name']:s['id'] for s in loc.get('sensors',[]) if s.get('parameter',{}).get('name') in ['pm25','pm10']}
            if len(sensors)==2:
                observed_sensors=sensors;urls['air_quality_observed']=f"https://api.openaq.org/v3/locations/{loc['id']}/latest";break
        if not observed_sensors:raise RuntimeError('No OpenAQ location with both PM2.5 and PM10 found within 25 km. Do not substitute modelled data as observed data.')
    kb = json.loads((ROOT/'saafhawa_kb.json').read_text(encoding='utf-8'))
    with path.open('a',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        if path.stat().st_size == 0:
            writer.writeheader()
        for i in range(count):
            for tool, url in urls.items():
                row={'timestamp_karachi':stamp(),'tool':tool,'iteration':i+1,
                     'data_kind':'modelled' if 'modelled' in tool else 'forecast/current model'}
                start=time.perf_counter()
                try:
                    headers={'X-API-Key':os.environ['OPENAQ_API_KEY']} if tool=='air_quality_observed' else {}
                    r=requests.get(url,timeout=20,headers=headers)
                    row['elapsed_seconds']=time.perf_counter()-start
                    row.update(http_status=r.status_code,success=False,
                               response_sha256=hashlib.sha256(r.content).hexdigest())
                    r.raise_for_status()
                    payload=r.json()
                    values=payload.get('current',{})
                    expected=['pm10','pm2_5'] if 'air_quality' in tool else ['temperature_2m','relative_humidity_2m','wind_speed_10m']
                    if tool=='air_quality_observed':
                        latest={v['sensorsId']:v for v in payload.get('results',[])}
                        values={k:latest.get(sid) for k,sid in observed_sensors.items()}
                        expected=['pm25','pm10'];row['data_kind']='ground observations; retain measurement timestamps and inspect freshness'
                    row['success']=all(values.get(k) is not None for k in expected)
                    if tool=='air_quality_observed':
                        row['success']=all(isinstance(values.get(k,{}).get('value'),(float,int)) and math.isfinite(values[k]['value']) for k in expected)
                    row['detail']=json.dumps(values,ensure_ascii=False)
                    (out/f'{tool}_latest.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
                except requests.RequestException as e:
                    row.update(elapsed_seconds=time.perf_counter()-start,success=False,detail=str(e))
                writer.writerow(row); stream.flush()
            for tool, fn in [('heat_index',lambda:heat_index(38,65)),
                             ('knowledge_base',lambda:[r for r in kb if any(k in 'heat asthma child' for k in r['keywords'])])]:
                start=time.perf_counter(); result=fn(); elapsed=time.perf_counter()-start
                writer.writerow({'timestamp_karachi':stamp(),'tool':tool,'iteration':i+1,
                                 'elapsed_seconds':elapsed,'success':True,'data_kind':'local computation',
                                 'detail':json.dumps(result,ensure_ascii=False)})
            stream.flush()
            print(f'Timing round {i+1}/{count}',flush=True)
        # OpenAQ is an observed-data source; log authentication/availability honestly.
        row={'timestamp_karachi':stamp(),'tool':'openaq_observed_probe','iteration':1,'data_kind':'ground observations'}
        start=time.perf_counter()
        try:
            headers={'X-API-Key':os.environ['OPENAQ_API_KEY']} if os.environ.get('OPENAQ_API_KEY') else {}
            r=requests.get('https://api.openaq.org/v3/locations?coordinates=24.8607,67.0011&radius=25000&limit=5',headers=headers,timeout=20)
            row.update(elapsed_seconds=time.perf_counter()-start,http_status=r.status_code,
                       success=r.ok,detail=r.text[:300],response_sha256=hashlib.sha256(r.content).hexdigest())
        except requests.RequestException as e:
            row.update(elapsed_seconds=time.perf_counter()-start,success=False,detail=str(e))
        writer.writerow(row)
    summarize()

def summarize():
    path=ROOT/'evidence'/'tool_latency_raw.csv'
    rows=list(csv.DictReader(path.open(encoding='utf-8')))
    groups={}
    for r in rows:
        groups.setdefault(r['tool'],[]).append(r)
    stats={k:{'attempts':len(v),'successes':sum(r['success']=='True' for r in v),
              'successful_dates':sorted({r['timestamp_karachi'][:10] for r in v if r['success']=='True'}),
              'median_seconds':statistics.median([float(r['elapsed_seconds']) for r in v if r['success']=='True']) if any(r['success']=='True' for r in v) else None}
           for k,v in groups.items()}
    # Observed and modelled air data are different evidence streams. Failed-run
    # local calls must not overweight the pooled median; use equal sample counts.
    def balanced_pool(names):
        sets=[[float(r['elapsed_seconds']) for r in rows if r['tool']==name and r['success']=='True'] for name in names]
        count=min(map(len,sets))
        return {'samples_per_tool':count,'median_seconds':statistics.median([v for values in sets for v in values[-count:]]) if count else None}
    observed=balanced_pool(['weather','air_quality_observed','heat_index','knowledge_base'])
    modelled=balanced_pool(['weather','air_quality_modelled','heat_index','knowledge_base'])
    stats['balanced_observed_tool_pool']=observed
    stats['balanced_modelled_tool_pool_provisional_only']=modelled
    stats['pooled_local_and_network_median_seconds']=observed['median_seconds']
    stats['pooled_delay_usable']=observed['samples_per_tool']>=20
    stats['assignment_two_day_observed_requirement_complete']=all(stats.get(k,{}).get('successes',0)>=20 and len(stats.get(k,{}).get('successful_dates',[]))>=2 for k in ['weather','air_quality_observed'])
    latest_observed=[r for r in rows if r['tool']=='air_quality_observed' and r['success']=='True']
    if latest_observed:
        value=json.loads(latest_observed[-1]['detail'])
        ages={k:(datetime.now(timezone.utc)-datetime.fromisoformat(v['datetime']['utc'].replace('Z','+00:00'))).total_seconds()/3600 for k,v in value.items()}
        stats['latest_observed_age_hours']=ages
        stats['latest_observed_fresh_within_48_hours']=all(0<=age<=48 for age in ages.values())
        stats['freshness_note']='API timing success does not establish that observations are current. Retain original measurement dates.'
        for date in sorted({r['timestamp_karachi'][:10] for r in latest_observed}):
            names=['weather','air_quality_observed','heat_index','knowledge_base']
            groups_by_day={name:[r for r in rows if r['tool']==name and r['success']=='True' and r['timestamp_karachi'].startswith(date)] for name in names}
            if all(len(v)>=20 for v in groups_by_day.values()):
                primary=sorted([r for v in groups_by_day.values() for r in v[-20:]],key=lambda r:r['timestamp_karachi'])
                target=ROOT/'evidence'/f'tool_latency_primary_{date}.csv'
                with target.open('w',newline='',encoding='utf-8') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(primary)
        primary_rows=[]
        for path in (ROOT/'evidence').glob('tool_latency_primary_*.csv'):
            primary_rows.extend(csv.DictReader(path.open(encoding='utf-8')))
        stats['primary_balanced_measurements']={'n':len(primary_rows),'successful_dates':sorted({r['timestamp_karachi'][:10] for r in primary_rows}),'pooled_median_seconds':statistics.median(float(r['elapsed_seconds']) for r in primary_rows) if primary_rows else None,'per_tool_median_seconds':{name:statistics.median(float(r['elapsed_seconds']) for r in primary_rows if r['tool']==name) for name in names} if primary_rows else {},'selection':'Latest 20 successful calls/tool/date; all attempts retained in raw log. Four tools, equal weighting.'}
    (ROOT/'evidence'/'timing_summary.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--sources',action='store_true'); p.add_argument('--timings',action='store_true'); p.add_argument('--diagnose-air',action='store_true'); p.add_argument('--count',type=int,default=20); a=p.parse_args()
    if a.sources: snapshot()
    if a.timings: timings(a.count)
    if a.diagnose_air: diagnose_air()
