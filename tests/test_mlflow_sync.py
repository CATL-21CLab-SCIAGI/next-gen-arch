import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from archlab.tracking.mlflow_sync import Sync,read_rows,point_values


def row(step):
    return {'step':step,'consumed_supervised_tokens':step*100,'supervised_tokens':100,'loss':1.0/step,'seconds':2.0,'learning_rate':.001,'gradient_norm_before_clip':.5,'max_memory_allocated_gib':10}


class Client:
    def __init__(self):self.metrics=[];self.created=0;self.fail_once=False
    def get_experiment_by_name(self,name):return SimpleNamespace(experiment_id='1')
    def search_runs(self,*args,**kwargs):return []
    def create_run(self,*args,**kwargs):self.created+=1;return SimpleNamespace(info=SimpleNamespace(run_id='test'))
    def log_batch(self,run_id,metrics=(),**kwargs):
        self.metrics.extend(metrics)
        if metrics and self.fail_once:self.fail_once=False;raise ConnectionError('simulated lost acknowledgement')
    def update_run(self,*args,**kwargs):pass
    def set_terminated(self,*args,**kwargs):pass


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.history=self.root/'train-metrics.jsonl';self.history.write_text(json.dumps(row(1))+'\n'+json.dumps(row(2))+'\n')
        contract={'runtime':{'parameters':123},'world_size':8,'project_commit':'abc','optimizer':'test','cpu_offload':False,'accumulation':1}
        (self.root/'RUN_CONTRACT.json').write_text(json.dumps(contract))
        self.spec={'id':'test','experiment':'test','name':'normal','phase':'scratch','variant':'normal','path':str(self.root),'history':[str(self.history)],'budget':1000,'dataset':'fixture','state':'paused'}

    def test_incremental_sync_does_not_reimport_history(self):
        client=Client();sync=Sync(client,self.root/'state.json');sync.sync_run(self.spec);count=len(client.metrics)
        sync.sync_run(self.spec);self.assertEqual(len(client.metrics),count);self.assertEqual(client.created,1)
        with self.history.open('a') as f:f.write(json.dumps(row(3))+'\n')
        sync.sync_run(self.spec);points=[m for m in client.metrics if m.key=='train/cross_entropy'];self.assertEqual([m.step for m in points],[1,2,3])
        token_points=[m for m in client.metrics if m.key=='train/cross_entropy_by_tokens'];self.assertEqual([m.step for m in token_points],[100,200,300])

    def test_partial_final_line_is_ignored(self):
        with self.history.open('a') as f:f.write('{"step":3')
        self.assertEqual(len(read_rows([self.history])),2)

    def test_discontinuous_source_rejected(self):
        self.history.write_text(json.dumps(row(1))+'\n'+json.dumps(row(3))+'\n')
        with self.assertRaises(ValueError):read_rows([self.history])

    def test_failed_batch_retries_identical_points(self):
        client=Client();client.fail_once=True;sync=Sync(client,self.root/'state.json')
        with self.assertRaises(ConnectionError):sync.sync_run(self.spec)
        first=[(m.key,m.step,m.timestamp,m.value) for m in client.metrics];self.assertIsNone(sync.state['runs']['test']['last_step'])
        sync.sync_run(self.spec);self.assertEqual(first,[(m.key,m.step,m.timestamp,m.value) for m in client.metrics[len(first):]])

    def test_nonfinite_or_zero_duration_rejected(self):
        r=row(1);r['seconds']=0
        with self.assertRaises(ValueError):point_values(r)


if __name__=='__main__':unittest.main()
