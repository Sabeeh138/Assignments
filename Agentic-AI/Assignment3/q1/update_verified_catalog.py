from pathlib import Path
import json,csv
from build_catalog import build
ROOT=Path(__file__).resolve().parent

def updated_catalog():
    rows=build()[:6]
    spec={
      'qwen/qwen3-32b':('Qwen3-32B','Alibaba Qwen','32.762B dense','Apache-2.0','Yes','enable_thinking=False'),
      'qwen/qwen3-235b-a22b-2507':('Qwen3-235B-A22B-Instruct-2507','Alibaba Qwen','235B total / 22B active MoE','Apache-2.0','Yes','Non-thinking instruct'),
      'meta-llama/llama-3.3-70b-instruct':('Llama-3.3-70B-Instruct','Meta','70B dense','Llama 3.3 Community','Conditional; review licence','Non-reasoning instruct'),
      'mistralai/mistral-small-3.2-24b-instruct':('Mistral-Small-3.2-24B-Instruct','Mistral AI','24B dense','Apache-2.0','Yes','Non-reasoning instruct'),
      'deepseek/deepseek-chat-v3.1':('DeepSeek-V3.1','DeepSeek','671B total / 37B active MoE','MIT','Yes','Non-thinking template'),
      'openai/gpt-oss-120b':('gpt-oss-120b','OpenAI','117B total / 5.1B active MoE','Apache-2.0','Yes','reasoning_effort=low; residual reasoning volume unknown')}
    for e in json.loads((ROOT/'research'/'verified_endpoints.json').read_text()):
        name,owner,params,lic,commercial,reason=spec[e['id']]
        row=dict(rows[0]);row.update(model=name,side='open',developer=owner,parameters=params,license=lic,
          commercial_self_host=commercial,reasoning_setting=reason,context_tokens=e['context'],max_output_tokens=e['max_output'],
          input_usd_per_million=e['input'],output_usd_per_million=e['output'],cache_read_usd_per_million=e['cached'],
          cache_minimum_tokens=None,cache_ttl='NR',cache_write_multiplier=1,rate_provider=e['provider']+' / '+e['tag'],
          source_url=e['source'],access_date=e['access_date'],tool_benchmark='NR',general_benchmark='NR',
          native_tools='Yes; endpoint tool-choice modes vary',structured_json='Yes; provider structured output',
          evidence_status='Retrieved public endpoint prices/context/output/features; cache policy/limits/speed NR')
        rows.append(row)
    for row in rows:
        if row['model']=='GPT-4.1 mini':row['tool_benchmark']='BFCL v4 FC: 50.45% [B1]'
        if row['model']=='GPT-4.1 nano':row['general_benchmark']='MMLU: 80.1% [B2]'
        if row['model']=='Claude Haiku 4.5':row['tool_benchmark']='BFCL v4 FC: 68.70% [B1]'
        if row['model']=='Llama-3.3-70B-Instruct':row.update(tool_benchmark='BFCL v2: 77.3% [B3]',general_benchmark='MMLU Pro CoT 5-shot: 68.9% [B3]')
        if row['model']=='Mistral-Small-3.2-24B-Instruct':row['general_benchmark']='MMLU: 80.50% [B4]'
        if row['model']=='Qwen3-235B-A22B-Instruct-2507':row.update(tool_benchmark='BFCL v3: 70.9% [B5]',general_benchmark='MMLU-Redux: 93.1% [B5]')
    override_file=ROOT/'research'/'verified_updates.json'
    if override_file.exists():
        updates=json.loads(override_file.read_text(encoding='utf-8'))
        for row in rows:row.update(updates.get(row['model'],{}))
    (ROOT/'research'/'model_catalog.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    with (ROOT/'research'/'model_catalog.csv').open('w',newline='',encoding='utf-8') as f:
        fields=list(dict.fromkeys(key for row in rows for key in row))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    return rows

if __name__=='__main__':updated_catalog()
