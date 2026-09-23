"""Read-only operational and checkpoint audit of the paired full-weight runs."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess


def events(path):
    raw=path.read_text(errors='replace');decoder=json.JSONDecoder();offset=0;out=[]
    while True:
        offset=raw.find('{"event"',offset)
        if offset<0:return out
        try:
            value,end=decoder.raw_decode(raw,offset)
            out.append(value);offset=end
        except json.JSONDecodeError:
            offset+=1


def read_ledger(path,start,first_step):
    raw=path.read_bytes();lines=raw.splitlines(keepends=True);rows=[];consumed=start
    for i,line in enumerate(lines):
        try:r=json.loads(line)
        except json.JSONDecodeError:
            if i==len(lines)-1 and not line.endswith(b'\n'):break
            raise
        assert r['step']==first_step+len(rows)+1 and r['phase_step']==len(rows)+1
        for key in ('loss','seconds','gradient_norm_before_clip','max_memory_allocated_gib'):
            assert math.isfinite(r[key]),(path,key)
        assert r['supervised_tokens']>0 and r['input_tokens']>=r['supervised_tokens'] and r['seconds']>0
        consumed+=r['supervised_tokens'];assert consumed==r['consumed_supervised_tokens']
        assert math.isclose(r['learning_rate'],1e-4*min(1.,r['phase_step']/20),rel_tol=1e-12)
        assert r['changed_local_elements']>0
        assert all(math.isfinite(x) for x in r['indexer_kl_local'])
        rows.append(r)
    assert rows
    return rows,hashlib.sha256(raw).hexdigest()


def stats(rows):
    tokens=sum(r['supervised_tokens'] for r in rows);duration=sum(r['seconds'] for r in rows)
    return {'updates':len(rows),'targets':tokens,'token_weighted_ce':sum(r['loss']*r['supervised_tokens'] for r in rows)/tokens,
            'loss_range':[min(r['loss'] for r in rows),max(r['loss'] for r in rows)],
            'gradient_norm_range':[min(r['gradient_norm_before_clip'] for r in rows),max(r['gradient_norm_before_clip'] for r in rows)],
            'clipped_updates':sum(r['gradient_norm_before_clip']>1 for r in rows),
            'mean_update_seconds':duration/len(rows),'median_update_seconds':statistics.median(r['seconds'] for r in rows),
            'supervised_tokens_per_second':tokens/duration,
            'input_tokens_per_second':sum(r['input_tokens'] for r in rows)/duration,
            'indexer_kl_mean':statistics.mean(x for r in rows for x in r['indexer_kl_local'])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    now=datetime.now(timezone.utc)
    source=root/'source-v2'
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    clean=not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
    report={'as_of_utc':now.isoformat(),'scope':'current production v2; prior failed launches excluded',
            'source_commit':commit,'source_clean':clean,'training_files_modified':False,'variants':{}}
    ledgers={}
    for variant in ('simplicial','normal'):
        run=root/f'production-{variant}-v2';contract=json.loads((run/'RUN_CONTRACT.json').read_text())
        assert clean and contract['project_commit']==commit
        checked={}
        for relative,digest in contract['implementation_sha256'].items():
            checked[relative]=hashlib.sha256((source/'src/archlab'/relative).read_bytes()).hexdigest()==digest
        assert all(checked.values())
        old=Path(contract['start_checkpoint']);cursor=json.loads((old/'COMPLETE.json').read_text())['cursor']
        assert (old/'adapter-state.pt').is_file()
        rows,digest=read_ledger(run/'train-metrics.jsonl',cursor['supervised_tokens'],cursor['step'])
        ledgers[variant]=rows
        (out/f'{variant}-train-snapshot.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        log_events=[]
        for node in (0,1):log_events.extend(events(root/f'production-{variant}-v2-node{node}.log'))
        updates={rank:{} for rank in range(16)};saves={}
        for e in log_events:
            if e['event']=='full_train_update':updates[e['rank']][e['step']]=e
            if e['event']=='full_checkpoint_complete' and e['rank']==0:saves[e['step']]=e
        assert all(updates.values())
        newest_by_rank={rank:max(records) for rank,records in updates.items()}
        assert max(newest_by_rank.values())-min(newest_by_rank.values())<=1
        common=min(newest_by_rank.values());reference=updates[0][common]
        keys=('loss','supervised_tokens','input_tokens','consumed_supervised_tokens','learning_rate','gradient_norm_before_clip','max_memory_allocated_gib')
        for rank,records in updates.items():
            assert all(records[common][k]==reference[k] for k in keys)
            assert records[common]['changed_local_elements']>0
            assert records[common]['updated_parameter_tensors']==reference['updated_parameter_tensors']
        inventory=[]
        for checkpoint in sorted((run/'checkpoints').glob('step-*')):
            done=checkpoint/'COMPLETE.json'
            item={'path':str(checkpoint),'complete':done.exists(),'rank_manifests':len(list(checkpoint.glob('rank-*/MANIFEST.json')))}
            if done.exists():
                m=json.loads(done.read_text());c=m['cursor'];assert m['contract']==contract and m['world_size']==16 and item['rank_manifests']==16
                record=rows[c['phase_step']-1]
                assert (record['step'],record['consumed_supervised_tokens'])==(c['step'],c['supervised_tokens'])
                item.update(cursor=c,completed_at_utc=datetime.fromtimestamp(done.stat().st_mtime,timezone.utc).isoformat())
                if c['step'] in updates[0] and c['step'] in saves:
                    item['save_seconds']=saves[c['step']]['unix_time']-updates[0][c['step']]['unix_time']
            inventory.append(item)
        latest=rows[-1];recent=rows[-50:]
        complete=[i for i in inventory if i['complete']]
        intervals=[item['save_seconds'] for item in complete if 'save_seconds' in item]
        next_target=complete[-1]['cursor']['supervised_tokens']+20000000
        compute_remaining=max(0,next_target-latest['consumed_supervised_tokens'])/stats(recent)['supervised_tokens_per_second']
        matching_rows=[r for r in rows if r['step']>=rows[20]['step']]
        failures=sorted(f.name for f in run.glob('*failure.json'))
        assert not failures
        memory_peak=max(r['max_memory_allocated_gib'] for r in rows)
        first_peak=next(r for r in rows if r['max_memory_allocated_gib']==memory_peak)
        report['variants'][variant]={'latest':latest,'ledger_sha256':digest,'ledger_steps':len(rows),
            'ledger_age_seconds':now.timestamp()-(run/'train-metrics.jsonl').stat().st_mtime,
            'data_cursor_and_learning_rate_valid':True,'all_logged_metrics_finite':True,'failure_files':failures,
            'sealed_implementation_hashes_match':True,'hashed_implementation_files':len(checked),
            'all_rank_latest_steps':newest_by_rank,'all_rank_agreement_step':common,'all_rank_global_metrics_agree':True,
            'all_rank_parameter_updates_observed':True,'recent50':stats(recent),'whole_postwarm':stats(matching_rows),
            'peak_allocated_gib':memory_peak,'peak_first_reached_phase_step':first_peak['phase_step'],
            'updates_since_new_memory_peak':latest['phase_step']-first_peak['phase_step'],
            'checkpoints':inventory,'latest_checkpoint_token_lag':latest['consumed_supervised_tokens']-complete[-1]['cursor']['supervised_tokens'],
            'median_checkpoint_seconds':statistics.median(intervals) if intervals else None,
            'next_checkpoint_target_lower_bound':next_target,'estimated_seconds_to_next_checkpoint_start':compute_remaining,
            'estimated_seconds_to_next_checkpoint_complete':compute_remaining+(statistics.median(intervals) if intervals else 0),
            'original50m_adapter_checkpoint_available':True,'all_text_weights_unfrozen':contract['all_parameters_unfrozen'],
            'cpu_offload':contract['cpu_offload'],'runtime':contract['runtime']}
    count=min(map(len,ledgers.values()))
    for a,b in zip(ledgers['simplicial'][:count],ledgers['normal'][:count],strict=True):
        assert all(a[k]==b[k] for k in ('step','phase_step','consumed_supervised_tokens','supervised_tokens','input_tokens','learning_rate'))
    report['matched']={'full_updates':count,'through_total_tokens':ledgers['simplicial'][count-1]['consumed_supervised_tokens'],
                       'data_and_schedule_identical':True,'last50':{v:stats(r[count-50:count]) for v,r in ledgers.items()}}
    report['matched']['normal_speedup']=report['matched']['last50']['normal']['supervised_tokens_per_second']/report['matched']['last50']['simplicial']['supervised_tokens_per_second']
    sets={v:{x['cursor']['step'] for x in report['variants'][v]['checkpoints'] if x['complete']} for v in ledgers}
    matched_checkpoint=max(sets['simplicial']&sets['normal'])
    report['latest_matched_full_checkpoint_step']=matched_checkpoint
    report['latest_matched_full_checkpoint_tokens']=next(x['cursor']['supervised_tokens'] for x in report['variants']['simplicial']['checkpoints'] if x.get('cursor',{}).get('step')==matched_checkpoint)
    report['held_out_full_phase_evaluation_available']=False
    report['operational_checks_passed']=True
    (out/'training-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    brief={'as_of':report['as_of_utc'],'matched':report['matched'],'latest_matched_checkpoint_tokens':report['latest_matched_full_checkpoint_tokens'],'variants':{}}
    for v,r in report['variants'].items():
        brief['variants'][v]={k:r[k] for k in ('ledger_steps','recent50','peak_allocated_gib','updates_since_new_memory_peak','median_checkpoint_seconds','latest_checkpoint_token_lag','estimated_seconds_to_next_checkpoint_complete','all_rank_agreement_step')}
    print(json.dumps(brief,indent=2))


if __name__=='__main__':
    main()
