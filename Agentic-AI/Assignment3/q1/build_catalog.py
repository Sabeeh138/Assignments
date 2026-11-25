"""Build an explicitly qualified model research register.

NR means not retrieved, never zero. Open-model rate assumptions are scenario
inputs, not a claim to have verified the world's lowest current provider rate.
"""
import csv, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def build():
    rows=[]
    def add(name,side,owner,params,context,maxout,inp,out,cached,minimum,ttl,reasoning,license_,commercial,source,status,bench1='NR',bench2='NR'):
        rows.append(dict(model=name,side=side,developer=owner,parameters=params,context_tokens=context,max_output_tokens=maxout,
          input_usd_per_million=inp,output_usd_per_million=out,cache_read_usd_per_million=cached,
          cache_minimum_tokens=minimum,cache_ttl=ttl,cache_write_multiplier=1.25 if name=='Claude Sonnet 4.5' else 1,
          rate_provider='Anthropic' if name.startswith('Claude') else 'OpenAI' if name.startswith('GPT') else 'Google' if name.startswith('Gemini') else 'xAI' if name.startswith('Grok') else 'Illustrative public-provider rate scenario; not verified lowest',
          new_paid_rpm='NR: account/tier-specific',new_paid_tpm='NR: account/tier-specific',
          ttft_seconds=None,output_tokens_per_second=None,tool_benchmark=bench1,general_benchmark=bench2,
          native_tools='Yes' if side=='closed' else 'Model/provider-dependent; verify endpoint',
          structured_json='Yes' if side=='closed' else 'Provider constrained decoding required for guarantee',
          reasoning_setting=reasoning,license=license_,commercial_self_host=commercial,
          source_url=source,access_date='2026-11-25' if status.startswith('Retrieved') else 'Not retrieved',evidence_status=status))
    add('GPT-4.1 mini','closed','OpenAI','not disclosed',1047576,32768,.4,1.6,.1,1024,'5-10 min inactivity, up to 1 h',
        'Non-reasoning; no reasoning-effort mode','Proprietary','No',
        'https://developers.openai.com/api/docs/models/gpt-4.1-mini',
        'Retrieved official context/output/prices; rate limits, speed and benchmark scores NR')
    add('GPT-4.1 nano','closed','OpenAI','not disclosed',1047576,32768,.1,.4,.025,1024,'5-10 min inactivity, up to 1 h',
        'Non-reasoning; no reasoning-effort mode','Proprietary','No',
        'https://developers.openai.com/api/docs/models/gpt-4.1-nano',
        'Retrieved official model listing; price supported by GPT-4.1 launch table; remaining technical fields need recheck')
    add('Claude Haiku 4.5','closed','Anthropic','not disclosed',200000,64000,1,5,.1,4096,'5 min or 1 h',
        'thinking disabled; optional extended thinking','Proprietary','No',
        'https://platform.claude.com/docs/en/about-claude/pricing',
        'Retrieved official prices; other fields are documented configurations awaiting source recapture')
    add('Claude Sonnet 4.5','closed','Anthropic','not disclosed',200000,64000,3,15,.3,1024,'5 min or 1 h',
        'thinking disabled; optional extended thinking','Proprietary','No',
        'https://platform.claude.com/docs/en/about-claude/pricing',
        'Retrieved official prices; other fields are documented configurations awaiting source recapture')
    add('Gemini 2.5 Flash-Lite','closed','Google','not disclosed',1048576,65536,.1,.4,.01,1024,'Explicit default 1 h; configurable',
        'thinking_budget=0; optional thinking budget','Proprietary','No',
        'https://ai.google.dev/gemini-api/docs/pricing',
        'Retrieved public listing and pricing search result; caching/limits need recapture')
    add('Grok 4.7','closed','xAI','not disclosed',500000,None,2,6,None,None,'NR',
        'Lowest supported configurable reasoning; residual reasoning tokens NR','Proprietary','No',
        'https://docs.x.ai/developers/models/grok-4.7',
        'Retrieved official context/input/output prices; total cost is a lower bound until reasoning tokens measured')
    # These are explicitly hypothetical rate scenarios, because network recovery
    # failed. They keep all twelve cost calculations executable without posing
    # as current provider quotations.
    candidates=[
      ('Qwen2.5-7B-Instruct','Alibaba Qwen','7.61B dense',131072,8192,.30,.30,'Apache-2.0','Yes','https://huggingface.co/Qwen/Qwen2.5-7B-Instruct'),
      ('Qwen2.5-72B-Instruct','Alibaba Qwen','72.7B dense',131072,8192,.90,.90,'Qwen licence','Conditional; review terms','https://huggingface.co/Qwen/Qwen2.5-72B-Instruct'),
      ('Llama-3.3-70B-Instruct','Meta','70B dense',131072,None,.60,.80,'Llama 3.3 Community','Conditional; review terms','https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct'),
      ('Llama-3.1-8B-Instruct','Meta','8B dense',131072,None,.10,.10,'Llama 3.1 Community','Conditional; review terms','https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct'),
      ('Mistral-Small-3.2-24B-Instruct','Mistral AI','24B dense',128000,None,.10,.30,'Apache-2.0','Yes','https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506'),
      ('DeepSeek-V3-0324','DeepSeek','671B total / 37B active MoE',128000,None,.30,.90,'MIT','Yes','https://huggingface.co/deepseek-ai/DeepSeek-V3-0324')]
    for name,owner,params,ctx,maxout,ip,op,lic,commercial,source in candidates:
        add(name,'open',owner,params,ctx,maxout,ip,op,None,None,'NR',
            'Non-thinking instruct mode',lic,commercial,source,
            'UNVERIFIED SCENARIO RATES; current endpoint availability/lowest price/benchmarks NR')
    # The Qwen 7B model card was actually retrieved through web tools.
    rows[6]['evidence_status']='Retrieved official model-card parameters/context/output; prices remain a scenario, not a verified current quote'
    dest=ROOT/'research'; dest.mkdir(exist_ok=True)
    (dest/'model_catalog.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    with (dest/'model_catalog.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return rows

if __name__=='__main__':build()
