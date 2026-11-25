"""Reproducible Assignment Beta calculations; all dollars are USD.

No network access, GPU work or invented experimental results in this module.
"""
from __future__ import annotations
import csv, json, math, statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ROLL='23K-0002'

def softmax(z, temperature=1):
    a=[math.exp((v-max(z))/temperature) for v in z]
    return [v/sum(a) for v in a]

def distill_loss(z):
    p=softmax([5,3,2,0],2); q=softmax(z,2); q1=softmax(z)
    terms=[a*math.log(a/b) for a,b in zip(p,q)]
    kl=sum(terms); ce=-math.log(q1[0])
    return {'teacher_t1':softmax([5,3,2,0]),'student_t1':q1,
            'teacher_t2':p,'student_t2':q,'kl_terms':terms,'kl':kl,'ce':ce,'loss':2*kl+.5*ce}

def q3():
    z=[3,2.5,1,.5]; x=distill_loss(z)
    grad=[q-p+.5*(q1-y) for p,q,q1,y in zip(x['teacher_t2'],x['student_t2'],x['student_t1'],[1,0,0,0])]
    new=[v-.5*g for v,g in zip(z,grad)]
    # Central difference checks the entire mixed objective, including T^2.
    numeric=[]
    for i in range(4):
        a=z.copy(); b=z.copy(); a[i]+=1e-5; b[i]-=1e-5
        numeric.append((distill_loss(a)['loss']-distill_loss(b)['loss'])/2e-5)
    assert max(abs(a-b) for a,b in zip(grad,numeric))<1e-8
    assert abs(sum(grad))<1e-12
    return {**x,'gradient':grad,'finite_difference_gradient':numeric,'updated_logits':new,'after_step':distill_loss(new)}

def classification(m):
    total=sum(map(sum,m)); n=[sum(r) for r in m]; pred=[sum(r[j] for r in m) for j in range(3)]
    correct=sum(m[i][i] for i in range(3)); acc=correct/total
    z=1.959963984540054
    center=(acc+z*z/(2*total))/(1+z*z/total)
    half=z*math.sqrt(acc*(1-acc)/total+z*z/(4*total*total))/(1+z*z/total)
    classes=[]
    for i,label in enumerate(['Positive','Negative','Neutral']):
        tp=m[i][i]; p=tp/pred[i]; r=tp/n[i]; f=2*tp/(n[i]+pred[i])
        classes.append({'class':label,'tp':tp,'fp':pred[i]-tp,'fn':n[i]-tp,'support':n[i],'precision':p,'recall':r,'f1':f})
    return {'matrix':m,'correct':correct,'n':total,'accuracy':acc,'wilson_95':[center-half,center+half],
            'classes':classes,'macro_f1':statistics.mean(c['f1'] for c in classes),
            'weighted_f1':sum(c['f1']*c['support'] for c in classes)/total,'micro_f1':acc}

def ngrams(tokens,n):
    from collections import Counter
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def text_metrics():
    ref='the delivery was late but the product quality is good'.split()
    cand='the product quality is good but delivery was slow'.split()
    precisions=[]; overlaps=[]; rouge={}
    for n in range(1,5):
        a=ngrams(ref,n); b=ngrams(cand,n); match=sum((a&b).values())
        precisions.append(match/sum(b.values())); overlaps.append(match)
        if n in [1,2]:
            p=match/sum(b.values()); r=match/sum(a.values())
            rouge[f'rouge_{n}']={'overlap':match,'precision':p,'recall':r,'f1':2*p*r/(p+r)}
    dp=[[0]*(len(cand)+1) for _ in range(len(ref)+1)]
    for i,a in enumerate(ref,1):
        for j,b in enumerate(cand,1):
            dp[i][j]=dp[i-1][j-1]+1 if a==b else max(dp[i-1][j],dp[i][j-1])
    lcs=dp[-1][-1]; p=lcs/len(cand); r=lcs/len(ref)
    rouge['rouge_l']={'lcs':lcs,'precision':p,'recall':r,'f1':2*p*r/(p+r)}
    bp=math.exp(min(0,1-len(ref)/len(cand)))
    probs=[.60,.40,.50,.80,.70,.30,.25,.50,.20]
    exact_matches=[(0,5),(1,6),(2,7),(3,8),(4,9),(5,4),(6,1),(7,2)]
    chunks=1+sum(b[0]!=a[0]+1 or b[1]!=a[1]+1 for a,b in zip(exact_matches,exact_matches[1:]))
    mp=8/9; mr=8/10; harmonic=10*mp*mr/(mr+9*mp)
    exact_meteor=harmonic*(1-.5*(chunks/8)**3)
    return {'reference':ref,'candidate':cand,'overlaps_1_to_4':overlaps,'modified_precisions':precisions,
            'brevity_penalty':bp,'bleu4':bp*math.exp(sum(map(math.log,precisions))/4),
            **rouge,'meteor_exact_only':exact_meteor,'meteor_exact_chunks':chunks,
            'token_probabilities':probs,'sum_log_probabilities':sum(map(math.log,probs)),
            'mean_negative_log_probability':-statistics.mean(map(math.log,probs)),
            'perplexity':math.exp(-statistics.mean(map(math.log,probs)))}

def q4():
    a=classification([[100,5,15],[8,80,12],[20,10,50]])
    b=classification([[106,4,10],[6,84,10],[9,7,64]])
    runs=[.847,.840,.853,.837,.850]
    return {'stage1':a,'stage2':b,'accuracy_delta':b['accuracy']-a['accuracy'],
            'prompt_runs':runs,'prompt_mean':statistics.mean(runs),'prompt_sample_sd':statistics.stdev(runs),
            'prompt_population_sd':statistics.pstdev(runs),'text':text_metrics()}

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

def export():
    dest=ROOT/'results'; dest.mkdir(exist_ok=True)
    results={'q1':q1(),'q3':q3(),'q4':q4()}
    (dest/'calculations.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    with (dest/'classification_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['stage','class','tp','fp','fn','support','precision','recall','f1']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for stage in ['stage1','stage2']:
            for row in results['q4'][stage]['classes']: w.writerow({'stage':stage,**row})
    return results

if __name__=='__main__':
    print(json.dumps(export(),indent=2))
