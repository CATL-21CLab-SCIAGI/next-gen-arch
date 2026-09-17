"""Import and follow existing JSONL training ledgers using the official MLflow client."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time


def atomic_json(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def read_rows(paths):
    rows=[]
    for path in paths:
        raw=Path(path).read_text()
        # A live writer may not yet have completed its last line.
        rows.extend(json.loads(line) for line in raw.split('\n')[:-1] if line.strip())
    if not rows:raise ValueError('No complete training records')
    for i,row in enumerate(rows):
        if not math.isfinite(row['loss']) or row['supervised_tokens']<=0:raise ValueError('Invalid training metric')
        if i:
            previous=rows[i-1]
            if row['step']!=previous['step']+1 or row['consumed_supervised_tokens']!=previous['consumed_supervised_tokens']+row['supervised_tokens']:
                raise ValueError('Training ledger is not contiguous')
    return rows


def point_values(row):
    seconds=row.get('wall_seconds',row['seconds'])
    if seconds<=0:raise ValueError('Nonpositive update duration')
    values={'train/cross_entropy':row['loss'],'train/supervised_tokens':row['consumed_supervised_tokens'],
            'perf/target_tokens_per_second':row['supervised_tokens']/seconds,'perf/seconds_per_update':seconds,
            'optim/learning_rate':row['learning_rate'],'optim/gradient_norm':row['gradient_norm_before_clip'],
            'memory/peak_allocated_gib':row['max_memory_allocated_gib']}
    if 'router_load_cv_mean' in row:values['router/load_cv']=row['router_load_cv_mean']
    if 'router_unused_fraction_window' in row:values['router/expert_coverage']=1-row['router_unused_fraction_window']
    if not all(math.isfinite(v) for v in values.values()):raise ValueError('Nonfinite scalar metric')
    return values


def run_params(spec, contract):
    runtime=contract['runtime'];text=runtime.get('geometry',{}).get('text_config',{})
    return {'variant':spec['variant'],'phase':spec['phase'],'gpu_type':'NVIDIA B300',
            'gpus':contract['world_size'],'parameters':contract.get('full_text_parameters',runtime.get('parameters')),
            'hidden_size':text.get('hidden_size',5120),'layers':text.get('num_hidden_layers',40),
            'source_commit':contract['project_commit'],'automodel_commit':runtime.get('automodel_commit'),
            'optimizer':contract['optimizer'],'cpu_offload':contract['cpu_offload'],
            'global_windows':contract.get('global_windows',64),'microbatch':contract.get('microbatch',contract.get('microbatch_size')),
            'accumulation':contract['accumulation'],'target_supervised_tokens':spec['budget'],
            'dataset':spec['dataset'],'timestamp_semantics':'ingestion time; optimizer-step and token axes are exact'}


class Sync:
    def __init__(self, client, state_path):
        self.client=client;self.state_path=state_path
        self.state=json.loads(state_path.read_text()) if state_path.exists() else {'runs':{}}

    def save(self):atomic_json(self.state_path,self.state)

    def ensure_run(self,spec,contract):
        from mlflow.entities import Param,RunTag
        local=self.state['runs'].get(spec['id'])
        if local:
            # Detect configuration drift rather than silently combining unrelated histories.
            if local['source_commit']!=contract['project_commit']:raise ValueError('Existing tracking run source changed')
            return local
        experiment=self.client.get_experiment_by_name(spec['experiment'])
        experiment_id=experiment.experiment_id if experiment else self.client.create_experiment(spec['experiment'],tags={'archlab.managed':'true'})
        matches=self.client.search_runs([experiment_id],filter_string=f"tags.archlab_source_id = '{spec['id']}'",max_results=2)
        if len(matches)>1:raise ValueError('Duplicate MLflow run identities')
        run=matches[0] if matches else self.client.create_run(experiment_id,run_name=spec['name'],tags={'archlab_source_id':spec['id'],'archlab.phase':spec['phase'],'archlab.variant':spec['variant'],'archlab.history_timestamps':'ingestion, not original wall-clock time'})
        local={'run_id':run.info.run_id,'experiment_id':experiment_id,'source_commit':contract['project_commit'],'last_step':None,'history_timestamp_ms':int(time.time()*1000),'aux_fingerprints':{}}
        if matches:
            history=self.client.get_metric_history(run.info.run_id,'train/cross_entropy')
            local['last_step']=max((x.step for x in history),default=None)
        self.client.log_batch(run.info.run_id,params=[Param(k,str(v)) for k,v in run_params(spec,contract).items() if v is not None],tags=[RunTag('archlab.variant',spec['variant'])])
        self.state['runs'][spec['id']]=local;self.save();return local

    def sync_run(self,spec):
        from mlflow.entities import Metric,RunTag
        root=Path(spec['path']);contract=json.loads((root/'RUN_CONTRACT.json').read_text());rows=read_rows(spec['history']);local=self.ensure_run(spec,contract);run_id=local['run_id'];last=rows[-1]
        if local['last_step'] is not None and local['last_step']>last['step']:raise ValueError('Tracking cursor is ahead of source')
        pending=[r for r in rows if local['last_step'] is None or r['step']>local['last_step']]
        for first in range(0,len(pending),60):
            group=pending[first:first+60];metrics=[]
            # One deterministic timestamp for a backfill batch, persisted before writes.
            saved_batch=local.get('pending_batch',{})
            timestamp=saved_batch.get('timestamp_ms') if saved_batch.get('first_step')==group[0]['step'] else None
            if timestamp is None:
                timestamp=local['history_timestamp_ms'] if len(pending)>60 else int(time.time()*1000)
                local['pending_batch']={'first_step':group[0]['step'],'timestamp_ms':timestamp};self.save()
            for row in group:
                metrics.extend(Metric(k,float(v),timestamp,row['step']) for k,v in point_values(row).items())
                metrics.append(Metric('train/cross_entropy_by_tokens',float(row['loss']),timestamp,row['consumed_supervised_tokens']))
            self.client.log_batch(run_id,metrics=metrics,synchronous=True);local['last_step']=group[-1]['step'];local.pop('pending_batch',None);self.save()
        aux=[];validation=root/'validation.jsonl'
        if validation.exists():
            mapping={r['step']:r['consumed_supervised_tokens'] for r in rows}
            for r in read_jsonl(validation):
                aux.extend([('eval/fineweb_cross_entropy',r['loss'],r['step']),('eval/fineweb_cross_entropy_by_tokens',r['loss'],mapping[r['step']])])
        if spec.get('evaluation'):
            file=Path(spec['evaluation'])
            if file.exists():
                result=json.loads(file.read_text());step=result['cursor']['step'];variant=spec['variant']
                for k,v in result['heldout_fineweb'][variant].items():
                    if k!='targets':aux.append(('eval/fineweb_'+k,v,step))
                aux.append(('eval/fineweb_cross_entropy_by_tokens',result['heldout_fineweb'][variant]['cross_entropy'],result['cursor']['supervised_tokens']))
                for task,d in result['multiple_choice'].items():
                    for key in ['accuracy','accuracy_norm']:aux.append((f'eval/{task}/{key}',d[key][variant+'_accuracy'],step))
        if aux:
            digest=hashlib.sha256(json.dumps(aux,sort_keys=True).encode()).hexdigest()
            if local['aux_fingerprints'].get('evaluation')!=digest:
                # Stable timestamps and exact points make recovery retries deterministic.
                self.client.log_batch(run_id,metrics=[Metric(k,float(v),local['history_timestamp_ms'],int(step)) for k,v,step in aux]);local['aux_fingerprints']['evaluation']=digest;self.save()
        checkpoints=sorted(root.glob('checkpoints/step-*/COMPLETE.json'))
        tags={'archlab.last_ingested_step':str(last['step']),'archlab.last_sync_utc':datetime.now(timezone.utc).isoformat()}
        if checkpoints:
            p=checkpoints[-1];marker=json.loads(p.read_text());tags.update({'checkpoint.latest_path':str(p.parent),'checkpoint.tokens':str(marker['cursor']['supervised_tokens']),'checkpoint.step':str(marker['cursor']['step'])})
        failures=list(root.glob('*failure.json'));completed=(root/'COMPLETE.json').exists()
        phase='failed' if failures else ('complete' if completed else spec.get('state','running'))
        status='FAILED' if failures else ('FINISHED' if phase in ('paused','complete') else 'RUNNING')
        tags.update({'archlab.state':phase,'archlab.source_age_seconds':str(int(time.time()-(root/'train-metrics.jsonl').stat().st_mtime))})
        self.client.log_batch(run_id,tags=[RunTag(k,v) for k,v in tags.items()])
        if local.get('status')!=status:
            if status=='RUNNING':self.client.update_run(run_id,status=status)
            else:self.client.set_terminated(run_id,status=status,end_time=int(time.time()*1000))
            local['status']=status;self.save()
        return {'id':spec['id'],'run_id':run_id,'experiment_id':local['experiment_id'],'step':last['step'],'tokens':last['consumed_supervised_tokens'],'new_updates':len(pending),'state':phase}


def read_jsonl(path):return [json.loads(line) for line in Path(path).read_text().split('\n')[:-1] if line.strip()]


def configure_client(credentials):
    os.environ['MLFLOW_DISABLE_AGENT_HINT']='1'
    from mlflow import MlflowClient
    config=json.loads(credentials.read_text());os.environ['MLFLOW_TRACKING_TOKEN']=config['token']
    for key in list(os.environ):
        if key.lower() in ('http_proxy','https_proxy','all_proxy'):os.environ.pop(key)
    os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT']='25';os.environ['MLFLOW_HTTP_REQUEST_MAX_RETRIES']='5'
    return MlflowClient(tracking_uri=config['tracking_uri'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--credentials',type=Path,required=True);parser.add_argument('--state',type=Path,required=True);parser.add_argument('--watch',action='store_true');parser.add_argument('--interval',type=int,default=30);args=parser.parse_args()
    args.state.parent.mkdir(parents=True,exist_ok=True)
    with args.state.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sync=Sync(configure_client(args.credentials),args.state)
        while True:
            summary={'utc':datetime.now(timezone.utc).isoformat(),'runs':[],'errors':[]}
            for spec in json.loads(args.config.read_text())['runs']:
                try:
                    row=sync.sync_run(spec);summary['runs'].append(row);print(json.dumps(row),flush=True)
                except Exception as error:
                    # Never persist credential-bearing HTTP headers or request objects.
                    summary['errors'].append({'run':spec['id'],'type':type(error).__name__});print(json.dumps(summary['errors'][-1]),flush=True)
                    if not args.watch:raise
            atomic_json(args.state.parent/'SYNC_STATUS.json',summary)
            if not args.watch:return
            time.sleep(args.interval)


if __name__=='__main__':main()
