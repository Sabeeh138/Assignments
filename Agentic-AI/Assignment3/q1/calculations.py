"""Q1-only workload and memory calculations."""
import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ROLL="23K-0002"

def q1():
    digits=[int(c) for c in ROLL if c.isdigit()]; R=10*digits[-2]+digits[-1]; D=sum(digits)%10
    Q=1500+25*R; N=5+R%4; p=round(.04+.01*D,2); H=4+R%4; B=1200+10*R; T=10-R%3
    steps=[{'k':k,'input':2000+800*(k-1),'output':600 if k==N else 350,
            'attempts_expected':1 if k==N else 1+p} for k in range(1,N+1)]
    I=round(sum(s['input']*s['attempts_expected'] for s in steps),6)
    O=round(sum(s['output']*s['attempts_expected'] for s in steps),6)
    C=round(sum(s['attempts_expected'] for s in steps),6); M=Q*(30+4*H)
    peak_qpm=5*Q*.35/120
    cfgpath=ROOT/'sources'/'qwen3_32b_config.json'
    cfg=json.loads(cfgpath.read_text()) if cfgpath.exists() else {'num_hidden_layers':64,'num_key_value_heads':8,'hidden_size':5120,'num_attention_heads':64,'head_dim':128}
    layers=cfg['num_hidden_layers']; kv=cfg['num_key_value_heads']; dim=cfg.get('head_dim',cfg['hidden_size']//cfg['num_attention_heads'])
    params=32762123264  # BF16 safetensors metadata 65,524,246,528 bytes / 2.
    index=ROOT/'sources'/'qwen3_32b_metadata.json'
    if index.exists():
        data=json.loads(index.read_text())
        params=data['metadata']['total_size']//2
    seq=steps[-1]['input']+steps[-1]['output']
    cache_bytes=2*layers*kv*dim*seq*2
    # A100 80GB treated conservatively as 80e9 bytes, not 80GiB.
    capacity=math.floor((.9*80e9-params/2)/cache_bytes)
    baseline_latency_seconds=60.0  # Capacity scenario; never a measured serving latency.
    sessions=math.ceil(peak_qpm*baseline_latency_seconds/60)
    gpus=math.ceil(sessions/capacity)
    return {'student_roll':ROLL,'student_name':'Muhammad Sabeeh','R':R,'D':D,'Q':Q,'N':N,'p':p,'H':H,'B':B,'T':T,'steps':steps,
            'input_per_query':I,'output_per_query':O,'calls_per_query':C,'monthly_queries':M,
            'monthly_input':M*I,'monthly_output':M*O,'monthly_calls':M*C,
            'prefix_per_query':1500*C,'peak_queries_per_minute':peak_qpm,
            'peak_calls_per_minute':peak_qpm*C,'peak_input_tpm':peak_qpm*I,
            'peak_output_tpm':peak_qpm*O,'peak_total_tpm':peak_qpm*(I+O),
            'memory':{'parameter_count':params,'layers':layers,'kv_heads':kv,'head_dim':dim,'sequence_length':seq,
                      'fp16_weight_bytes':params*2,'int8_weight_bytes':params,'int4_weight_bytes':params/2,
                      'kv_bytes_per_sequence':cache_bytes,'gpu_bytes':80e9,'concurrent_sequences_per_gpu':capacity,
                      'assumed_session_seconds':baseline_latency_seconds,'peak_sessions':sessions,'gpu_count':gpus,
                      'gpu_usd_per_hour':1.59,'monthly_usd':gpus*720*1.59}}
