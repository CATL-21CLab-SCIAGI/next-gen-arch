"""Snapshot and plot paired V4.1 training ledgers without touching the runs."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

COLORS = {'simplicial': '#2866b3', 'normal': '#d97527'}
LABELS = {'simplicial': '2-simplicial', 'normal': 'Normal attention'}


def snapshot(path, destination):
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows = []
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines)-1 and not line.endswith(b'\n'):
                break
            raise
        if any(not math.isfinite(row[k]) for k in ('loss', 'seconds', 'gradient_norm_before_clip')):
            raise ValueError(f'nonfinite training metric: {path}')
        if row['seconds'] <= 0 or row['supervised_tokens'] <= 0:
            raise ValueError(f'invalid token or timing denominator: {path}')
        rows.append(row)
    if not rows or any(b['step'] != a['step']+1 for a, b in zip(rows, rows[1:])):
        raise ValueError(f'non-contiguous training ledger: {path}')
    destination.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return rows, {'source': str(path.resolve()), 'source_bytes_sha256': hashlib.sha256(raw).hexdigest(),
                  'rows': len(rows), 'source_mtime_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}


def arrays(rows, start):
    end = np.array([r['consumed_supervised_tokens'] for r in rows], dtype=np.float64)
    count = np.array([r['supervised_tokens'] for r in rows], dtype=np.float64)
    if not np.array_equal(np.diff(np.r_[start, end]), count):
        raise ValueError('token ledger does not match the consumed-target cursor')
    loss = np.array([r['loss'] for r in rows])
    return end, count, loss


def window_mean(rows, start, low, high):
    end, count, loss = arrays(rows, start)
    weight = np.maximum(0., np.minimum(end, high) - np.maximum(end-count, low))
    if weight.sum() <= 0:
        raise ValueError('empty token window')
    return float(np.dot(weight, loss) / weight.sum())


def smooth(rows, start, width):
    end, count, loss = arrays(rows, start)
    # The piecewise-constant per-update loss makes boundary-batch weighting
    # explicit. No individual token losses are inferred or interpolated.
    positions = np.r_[start, end]
    cumulative = np.r_[0., np.cumsum(count * loss)]
    beginning = end-width
    result = (cumulative[1:] - np.interp(beginning, positions, cumulative))/width
    result[beginning < start] = np.nan
    return end / 1e6, result


def style(ax):
    ax.set_facecolor('white')
    ax.grid(axis='y', color='#e5e7eb', linewidth=.7)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['bottom','left']].set_color('#c7ced8')
    ax.tick_params(colors='#465368', labelsize=9)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--smoothing-tokens', type=int, default=3000000)
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    output = args.output or args.root/f'curves-{now:%Y%m%dT%H%M%SZ}'
    output.mkdir(parents=True, exist_ok=False)
    runs, metadata, contracts, before = {}, {}, {}, {}
    for variant in LABELS:
        run = args.root/f'production-{variant}-v2'
        runs[variant], metadata[variant] = snapshot(run/'train-metrics.jsonl', output/f'{variant}-full-snapshot.jsonl')
        contracts[variant] = json.loads((run/'RUN_CONTRACT.json').read_text())
        old = Path(contracts[variant]['start_checkpoint']).parents[1]
        prior, prior_meta = snapshot(old/'train.jsonl', output/f'{variant}-adapter-snapshot.jsonl')
        cursor = json.loads((Path(contracts[variant]['start_checkpoint'])/'COMPLETE.json').read_text())['cursor']
        before[variant] = [r for r in prior if r['step'] <= cursor['step']]
        metadata[variant]['adapter_history'] = prior_meta
        metadata[variant]['start_cursor'] = cursor
        (output/f'{variant}-contract.json').write_text(json.dumps(contracts[variant], indent=2)+'\n')
    start = metadata['simplicial']['start_cursor']['supervised_tokens']
    assert start == metadata['normal']['start_cursor']['supervised_tokens']
    for key in ('project_commit','world_size','ep_size','accumulation','global_windows','optimizer','peak_relative_lr','warmup_phase_steps'):
        if contracts['simplicial'][key] != contracts['normal'][key]:
            raise ValueError(f'comparison contract differs: {key}')
    matched = min(map(len, runs.values()))
    for left, right in zip(runs['simplicial'][:matched], runs['normal'][:matched], strict=True):
        for key in ('step','phase_step','consumed_supervised_tokens','supervised_tokens','input_tokens','learning_rate'):
            if left[key] != right[key]:
                raise ValueError(f'paired training data/schedule differs: {key}')
    stop = runs['simplicial'][matched-1]['consumed_supervised_tokens']
    warmup = contracts['simplicial']['warmup_phase_steps']
    warmup_end = runs['simplicial'][warmup-1]['consumed_supervised_tokens']
    summary = {'snapshot_utc': now.isoformat(), 'matched_full_updates': matched,
               'matched_total_supervised_tokens': stop, 'matched_full_phase_tokens': stop-start,
               'start_supervised_tokens': start, 'smoothing_tokens': args.smoothing_tokens,
               'smoothing_method': 'token-weighted trailing mean of logged batch CE; boundary batches weighted proportionally',
               'timing_scope': 'matched updates after 20-step warmup, including optimizer; checkpoint I/O excluded',
               'evaluation': 'no held-out evaluation for this full-weight phase', 'variants': {}, 'sources': metadata}
    for variant, rows in runs.items():
        arrays(rows, start)
        exact = rows[:matched]
        timed = exact[warmup:]
        duration = sum(r['seconds'] for r in timed)
        last50 = exact[-50:]
        summary['variants'][variant] = {
            'latest_updates': len(rows), 'latest_total_tokens': rows[-1]['consumed_supervised_tokens'],
            'latest_full_phase_tokens': rows[-1]['consumed_supervised_tokens']-start,
            'mean_ce_matched_full_phase': window_mean(exact, start, start, stop),
            'mean_ce_matched_last_10m': window_mean(exact, start, max(start,stop-10000000), stop),
            'mean_ce_matched_last_5m': window_mean(exact, start, max(start,stop-5000000), stop),
            'mean_ce_exact_last_50_updates': sum(r['loss']*r['supervised_tokens'] for r in last50)/sum(r['supervised_tokens'] for r in last50),
            'exact_last_50_update_targets': sum(r['supervised_tokens'] for r in last50),
            'supervised_tokens_per_second': sum(r['supervised_tokens'] for r in timed)/duration,
            'mean_update_seconds': duration/len(timed), 'timed_matched_updates':len(timed),
            'last_gradient_norm':rows[-1]['gradient_norm_before_clip'],
            'peak_allocated_gib':max(r['max_memory_allocated_gib'] for r in rows)}
    a,b = [summary['variants'][v] for v in LABELS]
    summary['normal_throughput_speedup'] = b['supervised_tokens_per_second']/a['supervised_tokens_per_second']
    summary['normal_minus_simplicial_ce_last_10m'] = b['mean_ce_matched_last_10m']-a['mean_ce_matched_last_10m']
    summary['simplicial_relative_ce_advantage_last_10m_percent'] = 100*(b['mean_ce_matched_last_10m']-a['mean_ce_matched_last_10m'])/b['mean_ce_matched_last_10m']
    summary['normal_minus_simplicial_ce_whole_matched_phase'] = b['mean_ce_matched_full_phase']-a['mean_ce_matched_full_phase']
    (output/'comparison.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    with (output/'matched-updates.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['step','phase_step','total_supervised_tokens','batch_targets','input_tokens',
                         'simplicial_ce','normal_ce','normal_minus_simplicial_ce','simplicial_seconds','normal_seconds'])
        for left,right in zip(runs['simplicial'][:matched],runs['normal'][:matched],strict=True):
            writer.writerow([left['step'],left['phase_step'],left['consumed_supervised_tokens'],left['supervised_tokens'],left['input_tokens'],
                             left['loss'],right['loss'],right['loss']-left['loss'],left['seconds'],right['seconds']])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelcolor':'#344054',
                         'text.color':'#172238','axes.titleweight':'semibold','savefig.facecolor':'white'})
    figure = plt.figure(figsize=(12.4,8.4), layout='constrained')
    grid = figure.add_gridspec(2,3,height_ratios=[2,1.15])
    top = figure.add_subplot(grid[0,:])
    delta = figure.add_subplot(grid[1,:2])
    speed = figure.add_subplot(grid[1,2])
    for ax in (top,delta,speed): style(ax)
    maximum = max(rows[-1]['consumed_supervised_tokens'] for rows in runs.values())/1e6
    for variant,rows in runs.items():
        x,_,loss = arrays(rows,start)
        top.plot(x/1e6,loss,color=COLORS[variant],alpha=.13,lw=.7,rasterized=True)
        x,y = smooth(rows,start,args.smoothing_tokens)
        top.plot(x,y,color=COLORS[variant],lw=2.1,label=LABELS[variant])
    top.axvspan(start/1e6,warmup_end/1e6,color='#f4eac6',alpha=.6,zorder=0)
    top.text((start+warmup_end)/2e6,.99,'LR warmup',ha='center',va='top',transform=top.get_xaxis_transform(),fontsize=8.5,color='#816b34')
    top.axvline(stop/1e6,color='#7e8795',ls='--',lw=1)
    if maximum>stop/1e6:
        top.axvspan(stop/1e6,maximum+.5,color='#edf0f4',zorder=0)
        top.text((maximum+stop/1e6)/2,.99,'Normal-only\nextra progress',transform=top.get_xaxis_transform(),ha='center',va='top',fontsize=8.5,color='#616e82')
    top.set_xlim(start/1e6-.7,maximum+.5)
    all_losses=np.array([r['loss'] for rows in runs.values() for r in rows])
    top.set_ylim(all_losses.min()-.015,all_losses.max()+.025)
    top.set_ylabel('Training cross-entropy  ·  lower is better')
    top.set_xlabel('Total supervised tokens (millions)')
    top.set_title('Full-weight fine-tuning: the two loss curves closely overlap',loc='left',pad=13,fontsize=14)
    top.legend(loc='lower left',frameon=True,facecolor='white',edgecolor='#e5e7eb',ncol=2)
    top.xaxis.set_major_locator(MultipleLocator(10))
    paired_a=runs['simplicial'][:matched];paired_b=runs['normal'][:matched]
    x,ya=smooth(paired_a,start,args.smoothing_tokens);_,yb=smooth(paired_b,start,args.smoothing_tokens)
    difference=(yb-ya)*1000
    delta.axhline(0,color='#69778c',lw=1,ls='--')
    delta.plot(x,difference,color='#4b5568',lw=1.8)
    delta.fill_between(x,0,difference,where=difference>=0,color=COLORS['simplicial'],alpha=.2)
    delta.fill_between(x,0,difference,where=difference<0,color=COLORS['normal'],alpha=.2)
    delta.set_xlim(start/1e6-.7,stop/1e6+.5)
    delta.set_title('Paired loss difference is small',loc='left',pad=10,fontsize=12)
    delta.set_ylabel('Normal − 2-simplicial CE\n(× 0.001)')
    delta.set_xlabel('Total supervised tokens (millions)')
    delta.text(.02,.96,'Above zero: 2-simplicial has lower CE',transform=delta.transAxes,va='top',fontsize=8.5,color=COLORS['simplicial'])
    delta.text(.02,.04,'Below zero: normal has lower CE',transform=delta.transAxes,va='bottom',fontsize=8.5,color=COLORS['normal'])
    values=[summary['variants'][v]['supervised_tokens_per_second'] for v in LABELS]
    speed.barh([1,0],values,color=[COLORS[v] for v in LABELS],height=.43)
    speed.set_yticks([1,0],[LABELS[v] for v in LABELS],fontsize=9)
    speed.set_xlim(0,max(values)*1.24)
    speed.set_ylim(-.65,1.65)
    for y,value in zip([1,0],values):speed.text(value+max(values)*.03,y,f'{value:,.0f}',va='center',fontsize=10)
    speed.set_xlabel('Supervised tokens / second')
    speed.grid(axis='y',visible=False);speed.grid(axis='x',color='#edf0f4',lw=.7)
    speed.set_title(f'Normal is {(summary["normal_throughput_speedup"]-1)*100:.1f}% faster',loc='left',pad=10,fontsize=12)
    speed.text(.5,-.33,'Matched updates after warmup.\nCheckpoint I/O excluded.',ha='center',va='top',transform=speed.transAxes,fontsize=8.5,color='#616e82')
    figure.suptitle(f'DeepSeek V4.1 · 16 B300 GPUs per variant · snapshot {now:%Y-%m-%d %H:%M} UTC\n'
                   f'Faint: per-update CE. Bold: {args.smoothing_tokens/1e6:g}M-token weighted trailing mean. Matched comparison through {stop/1e6:.2f}M tokens.',
                   fontsize=10.5,color='#556278')
    for extension in ('png','pdf'):
        figure.savefig(output/f'full-finetune-curves.{extension}',dpi=190,bbox_inches='tight')
    plt.close(figure)
    history, ax = plt.subplots(figsize=(12.4,4.8),layout='constrained')
    style(ax)
    for variant in LABELS:
        arrays(before[variant],0)
        xp,yp=smooth(before[variant],0,args.smoothing_tokens)
        xf,yf=smooth(runs[variant],start,args.smoothing_tokens)
        ax.plot(xp,yp,color=COLORS[variant],lw=2,label=LABELS[variant])
        ax.plot(xf,yf,color=COLORS[variant],lw=2)
    ax.axvline(start/1e6,color='#6b7585',ls='--',lw=1.1)
    ax.axvspan(start/1e6,maximum+.5,color='#f0f3f8',alpha=.7,zorder=0)
    ax.text(start/1e6+1.1,.94,'Full-weight phase\nstarts at 50.12M',transform=ax.get_xaxis_transform(),ha='left',va='top',fontsize=9)
    ax.text(25,.94,'Adapter-only phase',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=10)
    ax.set_xlim(0,maximum+.5)
    ax.set_ylabel('Training cross-entropy')
    ax.set_xlabel('Total supervised tokens (millions)')
    ax.set_title('Training history · optimizer and backbone trainability change at 50.12M tokens',loc='left',fontsize=13,pad=12)
    ax.legend(loc='upper right',frameon=False)
    ax.text(.01,-.2,'3M-token weighted trailing means, smoothed separately within each phase. Adapter-only training beyond the selected 50M checkpoints is omitted.',transform=ax.transAxes,fontsize=8.5,color='#616e82')
    for extension in ('png','pdf'):
        history.savefig(output/f'full-training-history.{extension}',dpi=190,bbox_inches='tight')
    plt.close(history)
    text = (f'The training losses are practically tied through the matched {stop/1e6:.2f}M-token point.\n\n'
            f'Latest matched10M-token mean training CE: 2-simplicial {a["mean_ce_matched_last_10m"]:.6f}; '
            f'normal {b["mean_ce_matched_last_10m"]:.6f}. The difference is '
            f'{summary["normal_minus_simplicial_ce_last_10m"]:.6f} CE '
            f'({summary["simplicial_relative_ce_advantage_last_10m_percent"]:.3f}% relative, favoring2-simplicial).\n\n'
            f'Across the whole matched full-weight phase, the token-weighted means are '
            f'{a["mean_ce_matched_full_phase"]:.6f} and {b["mean_ce_matched_full_phase"]:.6f}, respectively. '
            'The small changes in which variant leads do not establish a robust quality advantage.\n\n'
            f'Normal attention is {(summary["normal_throughput_speedup"]-1)*100:.1f}% faster on matched post-warmup updates: '
            f'{b["supervised_tokens_per_second"]:,.0f} versus {a["supervised_tokens_per_second"]:,.0f} supervised tokens/s. '
            'These timings include optimizer work and exclude checkpoint I/O.\n\n'
            f'Current progress: 2-simplicial {a["latest_total_tokens"]/1e6:.2f}M total tokens; '
            f'normal {b["latest_total_tokens"]/1e6:.2f}M. The plot shades the unmatched normal-only tail.\n\n'
            'These are training losses, not held-out scores. Different batches have different difficulty; '
            'the moving mean describes the logged trajectory and does not by itself establish convergence. '
            'No training processes or training files were changed.\n')
    (output/'README.md').write_text(text.replace('matched10M','matched 10M').replace('favoring2','favoring 2'))
    print(json.dumps({'output':str(output.resolve()),'summary':summary},indent=2))


if __name__ == '__main__':
    main()
