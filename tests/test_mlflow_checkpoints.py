import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch
from archlab.tracking.mlflow_checkpoints import make_bundle,CheckpointSync


class Client:
    def __init__(self):self.files={};self.bundle_uploads=0
    def log_artifacts(self,run_id,local_dir,artifact_path):
        self.bundle_uploads+=1
        for p in Path(local_dir).iterdir():self.files[artifact_path+'/'+p.name]=p.read_bytes()
    def log_artifact(self,run_id,local_path,artifact_path):self.files[artifact_path+'/'+Path(local_path).name]=Path(local_path).read_bytes()
    def download_artifacts(self,run_id,path,dst_path):
        p=Path(dst_path)/Path(path).name;p.write_bytes(self.files[path]);return str(p)
    def set_tag(self,*args):pass


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.checkpoint=self.root/'run/checkpoints/step-000010';self.checkpoint.mkdir(parents=True)
        marker={'format':'archlab-v41-full-sharded-v1','world_size':2,'contract':{'variant':'normal','project_commit':'abc'},'cursor':{'step':10,'supervised_tokens':1000},'manifests':['rank-00/MANIFEST.json','rank-01/MANIFEST.json']}
        (self.checkpoint/'COMPLETE.json').write_text(json.dumps(marker))
        for rank in range(2):
            folder=self.checkpoint/f'rank-{rank:02d}';folder.mkdir();manifest={**marker,'rank':rank,'tensors':[{'chunks':[{'file':'weights.pt','elements':3,'sha256':'fixture'}]}],'optimizer_states':['optimizer.pt']};(folder/'MANIFEST.json').write_text(json.dumps(manifest));(folder/'weights.pt').write_bytes(b'WEIGHTS MUST NOT BE UPLOADED')
        self.patch=patch('archlab.tracking.mlflow_checkpoints.storage_info',return_value={'mount':'/fixture','filesystem_type':'nfs'});self.patch.start();self.addCleanup(self.patch.stop)

    def test_bundle_excludes_payloads_and_preserves_hashes(self):
        out=self.root/'bundle';r=make_bundle(self.checkpoint,out,'normal');self.assertFalse(r['weights_uploaded']);self.assertEqual(r['rank_manifests'],2)
        with zipfile.ZipFile(out/'rank-manifests.zip') as archive:
            self.assertEqual(archive.namelist(),['rank-00/MANIFEST.json','rank-01/MANIFEST.json']);self.assertEqual(archive.read('rank-00/MANIFEST.json'),(self.checkpoint/'rank-00/MANIFEST.json').read_bytes())
        self.assertFalse(list(out.glob('*.pt')));self.assertEqual(r['checkpoint_marker_sha256'],hashlib.sha256((self.checkpoint/'COMPLETE.json').read_bytes()).hexdigest())

    def test_mismatched_rank_cursor_rejected(self):
        p=self.checkpoint/'rank-01/MANIFEST.json';m=json.loads(p.read_text());m['cursor']['step']=11;p.write_text(json.dumps(m))
        with self.assertRaises(ValueError):make_bundle(self.checkpoint,self.root/'bundle','normal')

    def test_manifest_traversal_rejected(self):
        p=self.checkpoint/'COMPLETE.json';m=json.loads(p.read_text());m['manifests'][0]='../outside.json';p.write_text(json.dumps(m))
        with self.assertRaises(ValueError):make_bundle(self.checkpoint,self.root/'bundle','normal')

    def test_unchanged_checkpoint_not_reuploaded_and_incomplete_skipped(self):
        (self.checkpoint.parent/'step-000011').mkdir();client=Client();sync=CheckpointSync(client,self.root/'state.json');spec={'id':'fixture','path':str(self.root/'run'),'variant':'normal'}
        first=sync.sync_run(spec,'runid');second=sync.sync_run(spec,'runid')
        self.assertEqual(client.bundle_uploads,1);self.assertEqual(first['new'],['step-000010']);self.assertFalse(second['new']);self.assertEqual(second['in_progress'],['step-000011'])
        self.assertTrue(json.loads(client.files['checkpoints/step-000010/ATTACHED.json'])['reference_readback_verified'])

    def test_pruned_reference_is_marked_unavailable(self):
        client=Client();sync=CheckpointSync(client,self.root/'state.json');spec={'id':'fixture','path':str(self.root/'run'),'variant':'normal'};sync.sync_run(spec,'runid');(self.checkpoint/'COMPLETE.json').unlink();sync.sync_run(spec,'runid');catalog=json.loads(client.files['checkpoints/CATALOG.json']);self.assertFalse(catalog['checkpoints'][0]['source_complete_marker_present'])


if __name__=='__main__':unittest.main()
