"""Render the verified scratch comparison, including active training ancestry."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics


def read_rows(path):
    return [json.loads(x) for x in path.read_text().split('\n')[:-1] if x]


def moving(rows,key,window=20,weighted=False):
    values=[]
    for i in range(len(rows)):
        group=rows[max(0,i-window+1):i+1]
        values.append(sum(x[key]*x['supervised_tokens'] for x in group)/sum(x['supervised_tokens'] for x in group) if weighted else statistics.mean(x[key] for x in group))
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--health',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.root=a.root.resolve();a.output.mkdir(parents=True,exist_ok=True)
    health=json.loads(a.health.read_text());assert health['healthy'] and not health['errors']
    current={v:[x for x in read_rows(a.root/f'production-{v}-v5/train-metrics.jsonl') if x['step']<=health['variants'][v]['step']] for v in ['normal','simplicial']}
    history={v:read_rows(a.root/f'production-{v}-v5/prior-phase-metrics.jsonl')+current[v] for v in current}
    maps={v:{x['step']:x for x in current[v]} for v in current};shared=sorted(set(maps['normal'])&set(maps['simplicial']))[-20:];assert len(shared)==20 and shared[0]>30
    compare={}
    for v in current:
        rows=[maps[v][s] for s in shared];seconds=sum(x['wall_seconds'] for x in rows);targets=sum(x['supervised_tokens'] for x in rows);inputs=sum(x['input_tokens'] for x in rows)
        compare[v]={'matched_steps':shared,'supervised_tokens':targets,'input_tokens':inputs,'wall_seconds':seconds,'median_seconds_per_update':statistics.median(x['wall_seconds'] for x in rows),'supervised_tokens_per_second':targets/seconds,'input_tokens_per_second':inputs/seconds,'token_weighted_training_ce':sum(x['loss']*x['supervised_tokens'] for x in rows)/targets}
    assert compare['normal']['supervised_tokens']==compare['simplicial']['supervised_tokens']
    for step in shared:
        for key in ['supervised_tokens','input_tokens','consumed_supervised_tokens','window_cursor','learning_rate']:assert maps['normal'][step][key]==maps['simplicial'][step][key]
    ratio=compare['normal']['supervised_tokens_per_second']/compare['simplicial']['supervised_tokens_per_second'];result={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'health':health,'matched_throughput':compare,'normal_throughput_ratio':ratio,'source_commit':'a948e476c06b066556b5b4d3fd425d25aa54104a'}
    (a.output/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'normal':'#2166ac','simplicial':'#b2182b'};names={'normal':'Normal attention','simplicial':'2-simplicial'}
    fig,axes=plt.subplots(3,1,figsize=(10,10),sharex=True,constrained_layout=True)
    for v,rows in history.items():
        x=[z['consumed_supervised_tokens']/1e6 for z in rows];axes[0].plot(x,moving(rows,'loss',5,True),color=colors[v],label=names[v]);axes[1].plot(x,moving(rows,'router_load_cv_mean',20),color=colors[v],label=names[v])
        measured=[z for z in current[v] if z['router_usage_window_updates']==20];xx=[z['consumed_supervised_tokens']/1e6 for z in measured]
        axes[2].plot(xx,[100*(1-z['router_unused_fraction_window']) for z in measured],color=colors[v],label=names[v]+' — mean')
        axes[2].plot(xx,[100*(1-z['router_worst_unused_fraction_window']) for z in measured],color=colors[v],linestyle=':',alpha=.8,label=names[v]+' — worst layer')
    axes[0].set_ylabel('Training cross-entropy\n5-update token-weighted mean');axes[0].legend(loc='upper right');axes[0].set_title('DeepSeek V4.1 scratch comparison · width 640 · 20 layers · 8 B300 GPUs each')
    axes[1].set_ylabel('Expert-load CV\n20-update mean');axes[1].axhline(2.5,color='#888888',linestyle='--',linewidth=1)
    axes[2].set_ylabel('Experts used in last 20 updates (%)');axes[2].set_xlabel('Supervised training tokens (millions)');axes[2].set_ylim(60,101);axes[2].axhline(99,color='#aaaaaa',linestyle='--',linewidth=1);axes[2].legend(loc='lower right',fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2);ax.axvline(.577519,color='#999999',linestyle=':',linewidth=.8);ax.axvline(1.1758,color='#777777',linestyle=':',linewidth=.8)
    fig.savefig(a.output/'training-health.png',dpi=160);fig.savefig(a.output/'training-health.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    for v,rows in history.items():
        ax.plot([z['consumed_supervised_tokens']/1e6 for z in rows],moving(rows,'loss',5,True),color=colors[v],label=names[v],linewidth=2)
    ax.set(xlabel='Supervised training tokens (millions)',ylabel='Training cross-entropy (lower is better)',title='Scratch training · width 640 · 20 layers · 8 B300 GPUs per variant')
    ax.grid(alpha=.2);ax.legend();fig.text(.99,.005,'5-update token-weighted average · early training',ha='right',fontsize=9,color='#666666')
    fig.savefig(a.output/'learning-curve.png',dpi=160);fig.savefig(a.output/'learning-curve.pdf');plt.close(fig)
    lines=[f"Both reduced scratch-training variants passed sustained health checks through warmup at {health['utc']}. Each variant uses one 8-B300 node.",'',
    'Both models started from random weights. Hidden width is 640, depth 20, sequence length 2048; all model weights are trainable. The comparison uses the same native DeepSeek backbone with eight normal or 2-simplicial additive attention branches. Original routed-expert and Engram bucket counts are retained; total parameters are approximately 29.06B per model, including 24.58B Engram parameters. The GPU-resident factored optimizer uses no CPU weight/optimizer offload.','',
    'The shared dataset contains 10B DeepSeek-tokenized FineWeb-Edu training targets and a separate 1M-target validation set, preserving document boundaries. Initial held-out validation CE was exactly 11.9103568544 for both models. Training uses 64 document windows/update (microbatch 8, accumulation 1,8 GPUs), relative learning-rate warmup over 100 updates, peak 0.01, cosine floor 0.001, and global gradient clipping 1.0.','',
    '| Variant | Verified step | Verified tokens | Latest training CE | Mean router CV, 20 updates | Mean / worst-layer expert coverage, 20 updates |','|---|---:|---:|---:|---:|---:|']
    for v,d in health['variants'].items():lines.append(f"| {names[v]} | {d['step']} | {d['tokens']:,} | {d['loss']:.4f} | {d['mean_router_cv20']:.3f} | {100*d['expert_coverage20']:.2f}% / {100*d['worst_layer_expert_coverage20']:.2f}% |")
    lines+=['',f"Throughput uses the same 20 updates ({shared[0]}–{shared[-1]}) and identical input/target counts. Timings include data loading, forward/backward, optimizer, and router updates; checkpoint/validation pauses are outside this interval.",'','| Variant | Median seconds/update | Supervised targets/s | Input tokens/s |','|---|---:|---:|---:|']
    for v,d in compare.items():lines.append(f"| {names[v]} | {d['median_seconds_per_update']:.2f} | {d['supervised_tokens_per_second']:,.0f} | {d['input_tokens_per_second']:,.0f} |")
    lines += ['',f"Normal-attention throughput is {(ratio-1)*100:.1f}% higher on this matched interval. These are early training measurements; convergence and a quality ranking are not established.",'',
    'The admission checks covered all 8 ranks per variant: finite full-model gradients and parameter updates; distributed optimizer numerical checks; exact full-state checkpoint restoration and identical next-update replay; ragged 2048-token batches and empty ranks; matched data cursors and learning-rate schedules; completed production checkpoints; and sustained router-load and rolling expert-coverage checks through warmup.','',
    'During startup, checkpoint metadata was made JSON-stable and rank-invariant, and padding was excluded from routing/indexer objectives. Early expert concentration prompted a shared token-weighted routing auxiliary loss, larger microbatches with unchanged effective batch, and centered proportional bias correction. Active training ancestry is steps 1–10 from production-v3,11–20 from production-v4, and21 onward from production-v5. Newest-only retention preserves one completed checkpoint per production run; prototype runs are outside this controlled continuation. The active source is **a948e476c06b066556b5b4d3fd425d25aa54104a**; the pinned AutoModel source and container runtime were not edited.','',
    'The first corrected production checkpoints are at step 30 for both variants. Subsequent full checkpoints are scheduled every 500M additional training targets, with held-out validation every 100M. Both training runs remain active toward 10B targets.','']
    for v in current:lines.append(f"- [{names[v]} latest checkpoint]({health['variants'][v]['latest_checkpoint']}/COMPLETE.json)")
    lines += [f'- [Health verification]({a.health.resolve()})',f'- [Qualification and controlled migration]({a.root/"launch-router-correction-v4/QUALIFICATION_AND_MIGRATION_V5.json"})',f'- [Normal runtime contract]({a.root/"production-normal-v5/RUN_CONTRACT.json"})',f'- [2-simplicial runtime contract]({a.root/"production-simplicial-v5/RUN_CONTRACT.json"})','',f'![Training and expert health]({(a.output/"training-health.png").resolve()})','']
    (a.output/'REPORT.md').write_text('\n'.join(lines));print(json.dumps({'report':str((a.output/'REPORT.md').resolve()),'matched_throughput':compare,'normal_throughput_ratio':ratio},indent=2))


if __name__=='__main__':main()
