"""Four-page Q2 report, populated only with actual saved GPU evidence."""
from pathlib import Path
import csv,json
from xml.sax.saxutils import escape

def build_q2_report(study_dir, output_pdf, require_complete=False):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
    root=Path(study_dir);output=Path(output_pdf);output.parent.mkdir(parents=True,exist_ok=True)
    idx=root/'export_index.json'; complete=idx.exists()
    if require_complete and not complete:raise RuntimeError('Complete all GPU experiments before exporting the measured report.')
    evidence=json.loads(idx.read_text()) if complete else None
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyQ',fontName='Helvetica',fontSize=9.4,leading=13.1,spaceAfter=8))
    styles.add(ParagraphStyle(name='SmallQ',fontName='Helvetica',fontSize=7.6,leading=10,spaceAfter=5))
    story=[]
    def p(s,style='BodyQ'):story.append(Paragraph(s,styles[style]))
    def h(s):p(s,'Heading2')
    def table(rows,widths=None):
        cells=[[Paragraph(escape(str(c)),styles['SmallQ']) for c in row] for row in rows]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eef5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#16334d')),('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5)]));story.append(t)
    p('Question 2 | PEFT deployment report','Title')
    p('Muhammad Sabeeh | 23K-0002 | Qwen2.5-1.5B-Instruct | 1,800 train / 200 test','SmallQ')
    if not complete:p('<b>Evidence status: GPU experiments have not run. This is an evidence-limited report, not a completed empirical comparison.</b> No experimental score, duration, inference speed or GPU memory is inferred from a tiny CPU test.')
    h('1. Production choice and deployment trade-offs')
    if complete:
        runs=evidence['all_base_results'][1:]
        winner=sorted(runs,key=lambda r:(-r['test']['exact_pct'],-r['test']['tool_pct'],-r['test']['json_pct'],r['peak_gpu_gib'],r['training_minutes']))[0]
        p(f"The strongest measured base comparison is <b>{escape(winner['method'])}, rank {winner['rank']}</b>: full match {winner['test']['exact_pct']:.2f}%, tool accuracy {winner['test']['tool_pct']:.2f}%, JSON validity {winner['test']['json_pct']:.2f}%, peak training memory {winner['peak_gpu_gib']:.3f} GiB, training {winner['training_minutes']:.2f} minutes, unmerged generation {winner['test']['tokens_per_second']:.2f} tokens/s. This post-hoc recommendation uses the reported test evidence; it is not additional training or a deployment safety certification.")
    else:
        p('My provisional engineering choice is rank-8 LoRA on a dense BF16/FP16 base, provided its measured full match is competitive and it fits the production GPU. Its small adapter and simple dense merge are attractive for a small team. This is a conditional design preference; the study does not yet identify an empirical winner. If dense storage is the binding constraint, NF4 QLoRA is the first alternative to evaluate.')
    p('Use full match and unnecessary-action errors ahead of JSON validity alone. A perfectly valid refund call can still be the wrong action. Compare training memory and duration separately from serving throughput: a training-memory advantage does not imply faster inference. Keep tool names and argument schemas constrained, validate order ownership and refund policy in backend code, and require confirmation for consequential actions.')
    p('A dense LoRA adapter can be merged into its dense base, removing the extra low-rank computation. Dense DoRA additionally learns magnitudes and can also be merged by supported PEFT implementations. NF4 QLoRA starts from quantized weights: merging may require dequantization or a supported quantized merge, and deployment may requantize. Rounding can change outputs and peak memory. No post-merge speed or quality gain is claimed without remeasurement.')
    h('Five-row comparison')
    rows=[['Method','Trainable / %','Peak GiB / min','JSON / tool / exact %']]
    if complete:
        for r in evidence['all_base_results']:
            fmt=lambda v:'N/A' if v is None else f'{v:.3f}'
            rows.append([r['method'],f"{r['trainable_parameters']} / {fmt(r['trainable_pct'])}",f"{fmt(r['peak_gpu_gib'])} / {fmt(r['training_minutes'])}",f"{r['test']['json_pct']:.1f} / {r['test']['tool_pct']:.1f} / {r['test']['exact_pct']:.1f}"])
    else:
        rows += [[m,'Unmeasured','N/A' if m=='Original' else 'Unmeasured','Unmeasured'] for m in ['Original','LoRA','QLoRA','DoRA','AdaLoRA']]
    table(rows,[75,119,100,188])
    story.append(PageBreak());h('2. Where it still breaks: five real errors')
    if complete:
        file=root/'tables'/'error_analysis.csv'; errors=list(csv.DictReader(file.open(encoding='utf-8')))
        summary=json.loads((root/'error_analysis_summary.json').read_text())
        p(f"Error-analysis row: {escape(str(summary['method']))}, rank {summary['rank']}. There are {summary['total_test_errors']} non-exact predictions in the 200 test records. This is a descriptive best-test-row selection, so its apparent performance is optimistic.")
        if not errors:p('The evaluated model made no full-match mistakes; five incorrect examples therefore do not exist in this test set.')
        for i,row in enumerate(errors[:5],1):
            p(f"<b>{i}. {escape(row['Error category'])}</b>",'SmallQ')
            for label,key in [('Message','Customer/user message'),('Correct','Correct function call'),('Actual','Actual generated response')]:
                p(f'<b>{label}:</b> {escape(row[key])}','SmallQ')
        p('Observed category counts: '+escape(json.dumps(summary['all_error_categories'])), 'SmallQ')
    else:
        p('Five real incorrect outputs cannot be selected until model generation runs. Fabricating customer messages and model responses would turn an error analysis into a simulation. The notebook automatically exports the first five non-exact responses from the descriptively best test row, including the full user message, expected call, raw response and error label.')
        table([['Evidence field','Collection rule'],['Customer message','Original held-out user message; preserve its wording.'],['Correct call','Frozen test JSON, including every argument and value type.'],['Actual response','Raw generated text; do not repair JSON before grading.'],['Error label','Broken JSON; invalid call structure; wrong tool; extra/missing argument; wrong value/type.'],['Pattern','Count all error categories across the entire 200-record set.']], [125,357])
        p('No pattern of actual model failures is claimed yet. The failure taxonomy specifically distinguishes syntactic reliability from semantic correctness: extra prose can break JSON, a hallucinated name can break dispatch, and a valid but unnecessary refund can break business policy. The last error requires policy-aware checks beyond exact match.')
    h('Protocol and leakage control')
    p('Freeze exactly 2,000 eligible unique Glaive examples with source/model revisions, then split 1,800/200 by seed. Duplicate normalized user messages cannot cross the split. Fit all 1,800 records with the same epochs, LR schedule, target modules, seed and masking. Use the fixed 180-example training-side subset to choose the rank-study method; it is in-sample and optimistic, not held-out validation. Never use the 200 test records to change training.')
    p('The exact four business tool names are not assumed to occur in Glaive. The notebook audits coverage and uses a general function-calling benchmark when the requested domain cannot supply 2,000 examples. It never relabels unrelated tools as support tools. Accordingly, this study cannot establish real refund authorization or Karachi-language performance.')
    story.append(PageBreak());h('3. The agentic angle: one base, several adapters')
    p('A router agent, refund agent and shipping agent can share one frozen base and load separate behavior adapters. The base is stored once, while each role stores only its small trainable update. Training also avoids gradients and optimizer state for the base. This cuts storage and optimizer memory relative to three fully fine-tuned copies, although activations, request KV caches and adapter-routing infrastructure still consume memory.')
    p('For one 4,096 by 4,096 matrix at rank 16, an adapter contains 131,072 weights versus 16,777,216 base weights. At FP16 this is 262,144 bytes (0.25 MiB), while the matrix alone occupies 33,554,432 bytes (32 MiB). Three base copies with one adapter each occupy 96.75 MiB; a shared base with three adapters occupies 32.75 MiB. This example excludes optimizer state and is not a whole-model memory benchmark.')
    p('A real live-swapping risk is request contamination: a shipping request may continue after the global active adapter is switched to refunds. Bind adapter identity and version to each request and decode operation; use a serving stack that isolates per-request adapters, or serialize access while switching. Cache keys, traces and rollback metadata must include adapter identity. Never merge one role permanently into the base and then treat the resulting weights as the common base for other roles.')
    h('Rank ablation: 4, 8, 16 and 32')
    if complete:
        rows=[['Rank','Tool %','Exact %','Peak GiB','Minutes']]
        for r in evidence['rank_runs']:rows.append([r['rank'],f"{r['test']['tool_pct']:.2f}",f"{r['test']['exact_pct']:.2f}",f"{r['peak_gpu_gib']:.3f}",f"{r['training_minutes']:.2f}"])
        table(rows,[55,95,95,105,132]);p(escape((root/'rank_summary.txt').read_text(encoding='utf-8')))
    else:
        p('The rank grid is fixed at 4, 8, 16 and 32. Rank 8 reuses the base comparison adapter, so the full experiment needs seven training runs plus the original-model evaluation. AdaLoRA uses an initial-rank policy ceil(1.5 times target), including 12 to target 8. Each run uses the same data and shared schedule. Plot the actual 200-record test tool accuracy against rank only after completion.')
        p('I cannot conclude that doubling rank helps without that graph. The required conclusion will compare percentage-point gains against additional trainable parameters, peak memory and training time. With only 200 test records, one changed decision is 0.5 percentage points; small changes may reflect sampling noise rather than a meaningful production advantage.')
    h('Measurement discipline')
    p('Use torch.cuda.max_memory_allocated() after resetting peak counters for training. CUDA synchronization brackets measured work. Record real elapsed training minutes, and score all 200 raw generations with strict whole-output JSON parsing and exact argument matching. Generation throughput is batch-one, greedy and unmerged; model loading and tokenization are excluded. These choices must be identical across methods.')
    story.append(PageBreak());h('4. What surprised me, and what follows')
    if complete:
        base=evidence['all_base_results'];orig=base[0];changes=sorted(base[1:],key=lambda r:abs(r['test']['tool_pct']-orig['test']['tool_pct']),reverse=True)
        r=changes[0];p(f"The largest observed change from the base model was {escape(r['method'])}: {r['test']['tool_pct']-orig['test']['tool_pct']:+.2f} percentage points in tool accuracy, alongside {r['test']['exact_pct']-orig['test']['exact_pct']:+.2f} points in full match. The difference between these two gains shows why tool-name accuracy alone cannot summarize argument correctness. This is the measured observation; no invented prior expectation is attributed to the student.")
    else:
        p('A concrete preparation finding was that the existing full-corpus validation found 112,960 Glaive conversations, 5,164 eligible unique examples under the fixed parser/token rules, and zero eligible examples for the exact four requested business tools. This evidence is preserved in authoring_validation.json and is not a fine-tuning result. It changes the interpretation of the experiment: a generic function-calling gain is not automatically a customer-support gain.')
        p('There is no training-result surprise to report yet. In particular, it would be unjustified to say that AdaLoRA beat LoRA, that QLoRA saved a measured amount of VRAM, or that rank 32 was wasteful. The notebook appends a measured report after the actual runs; this report keeps the missing evidence visible.')
    h('Shipping decision')
    p('For a real release, retain backend tool authorization, schema validation, user confirmation for transactions, and a human fallback even if an adapter scores highly. Evaluate a separate support set containing ambiguous requests, missing order IDs, unauthorized refund requests, Roman Urdu, Urdu script and English. The generated Glaive benchmark supplies a reproducible experiment, but it does not replace a representative deployment evaluation.')
    h('Limitations and completion gate')
    p('One seed, a small test set, train-side method selection and post-hoc best-test-row reporting limit inference. Core library versions and GPU identity are checked before resuming. The four methods share the optimizer and schedule; QLoRA differs in the required base quantization and AdaLoRA adds its intrinsic rank schedule/orthogonality behavior. No post-merge serving benchmark is included.')
    if not complete:p('<b>Remaining empirical work:</b> run the Colab notebook on a CUDA GPU, preserve all eight evaluated rows, export comparison/rank graphs and five genuine errors, then regenerate this report. The current machine has no CUDA device. The program deliberately refuses to manufacture the missing measurements.')
    h('References and AI declaration')
    for text in ['Hu et al., LoRA (2021), https://arxiv.org/abs/2106.09685',
                 'Dettmers et al., QLoRA (2023), https://arxiv.org/abs/2305.14314',
                 'Liu et al., DoRA (2024), https://arxiv.org/abs/2402.09353',
                 'Zhang et al., AdaLoRA (2023), https://arxiv.org/abs/2303.10512',
                 'Hugging Face PEFT documentation, https://huggingface.co/docs/peft/']:
        p(escape(text),'SmallQ')
    p('OpenAI Codex assisted with implementation, arithmetic, research and report writing. No AI-produced experimental score is presented as an observed result.','SmallQ')
    def footer(c,doc):
        c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#526779'));c.drawString(48,28,'Assignment Beta | Q2 | Muhammad Sabeeh | 23K-0002');c.drawRightString(A4[0]-48,28,str(doc.page))
    SimpleDocTemplate(str(output),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=38,bottomMargin=44).build(story,onFirstPage=footer,onLaterPages=footer)
    return str(output)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('study_dir');p.add_argument('output_pdf');p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    print(build_q2_report(a.study_dir,a.output_pdf,a.require_complete))
