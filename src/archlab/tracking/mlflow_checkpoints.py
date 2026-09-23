"""Attach checkpoint references and compressed manifests to existing MLflow runs.

Only metadata is uploaded. Model, optimizer, and RNG payloads are never opened,
modified, deleted, or copied into the MLflow artifact store by this module.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
import zipfile
from functools import partial

from archlab.artifacts import atomic_write_json
from archlab.tracking.mlflow_sync import configure_client

atomic_json = partial(atomic_write_json, sort_keys=False, allow_nan=False, create_parents=False)


def sha(data):return hashlib.sha256(data).hexdigest()


def metadata_file(root,relative):
    name=Path(relative)
    if name.is_absolute() or '..' in name.parts:raise ValueError('Unsafe manifest path')
    path=root/name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):raise ValueError('Manifest escapes checkpoint')
    return path


def storage_info(path):
    path=path.resolve()
    result=subprocess.run(['findmnt','--json','-T',str(path),'-o','TARGET,FSTYPE'],capture_output=True,text=True,check=True)
    filesystem=json.loads(result.stdout)['filesystems'][0]
    return {'resolved_path':str(path),'mount':filesystem['target'],'filesystem_type':filesystem['fstype']}


def make_bundle(checkpoint, destination, expected_variant):
    checkpoint=checkpoint.resolve();raw=(checkpoint/'COMPLETE.json').read_bytes();marker=json.loads(raw)
    if marker['format']!='archlab-v41-full-sharded-v1' or marker['contract']['variant']!=expected_variant:raise ValueError('Wrong checkpoint format or variant')
    world=marker['world_size'];names=marker['manifests']
    if len(names)!=world or len(set(names))!=world:raise ValueError('Incomplete rank manifest set')
    manifests={};checksums={'COMPLETE.json':sha(raw)};chunks=0;optimizer_files=0
    for rank,relative in enumerate(names):
        data=metadata_file(checkpoint,relative).read_bytes();m=json.loads(data)
        if m['rank']!=rank or any(m[k]!=marker[k] for k in ('world_size','cursor','contract')):raise ValueError('Rank manifest identity mismatch')
        manifests[relative]=data;checksums[relative]=sha(data);chunks+=sum(len(t['chunks']) for t in m['tensors']);optimizer_files+=len(m['optimizer_states'])
    # Guard against pruning or replacing a checkpoint while metadata is collected.
    if (checkpoint/'COMPLETE.json').read_bytes()!=raw:raise ValueError('Checkpoint changed while reading metadata')
    destination.mkdir(parents=True,exist_ok=True);(destination/'COMPLETE.json').write_bytes(raw)
    with zipfile.ZipFile(destination/'rank-manifests.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for relative,data in manifests.items():
            info=zipfile.ZipInfo(relative,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(info,data)
    manifest_hash=sha((destination/'rank-manifests.zip').read_bytes())
    contract=marker['contract'];runtime=contract.get('runtime',{});storage=storage_info(checkpoint)
    reference={'format':'archlab-mlflow-checkpoint-reference-v1','attachment_mode':'reference-and-manifests','weights_uploaded':False,'optimizer_payloads_uploaded':False,'rng_payloads_uploaded':False,'source_path':str(checkpoint),'source_uri':checkpoint.as_uri(),'storage':storage,'checkpoint_format':marker['format'],'variant':expected_variant,'cursor':marker['cursor'],'world_size':world,'expert_parallel':contract.get('ep_size',runtime.get('ep_size')),'source_commit':contract.get('project_commit'),'checkpoint_marker_sha256':sha(raw),'rank_manifests':world,'tensor_payload_chunks_referenced':chunks,'optimizer_payload_files_referenced':optimizer_files,'rank_manifests_zip_sha256':manifest_hash,'metadata_validation':'rank identities, training contracts, and cursors checked; model payloads not reread','restore_requires':'Original payload files on the shared source mount and the trained runtime/mesh'}
    atomic_json(destination/'REFERENCE.json',reference);atomic_json(destination/'METADATA_SHA256.json',checksums)
    (destination/'RESTORE.md').write_text(f'''# Checkpoint reference: {checkpoint.name}

**Metadata only. The weight, optimizer, and RNG payloads remain on the shared filesystem.**

- Variant: **{expected_variant}**
- Optimizer step: **{marker['cursor']['step']:,}**
- Supervised tokens: **{marker['cursor']['supervised_tokens']:,}**
- Restore mesh: **{world} GPU ranks**
- Filesystem: **{storage['filesystem_type']}**, mounted at `{storage['mount']}`
- Source checkpoint: `{checkpoint}`
- Trained source commit: `{contract.get('project_commit')}`

Use a machine with this source mount and the exact trained container/source contract.
The scratch trainer accepts `--resume /path/to/checkpoint`; the full-finetuning
trainer accepts `--resume-full /path/to/checkpoint` and also requires its original
adapter initializer. Both restore functions verify tensor checksums. Normal and
2-simplicial checkpoint variants are not interchangeable.

`COMPLETE.json` records the full runtime/model contract and data cursor.
`rank-manifests.zip` contains every rank manifest, including weight-chunk SHA-256
hashes and optimizer filenames. `METADATA_SHA256.json` verifies the metadata files.
The metadata bundle by itself cannot restore the model; retain the source payloads.
Consult `checkpoints/CATALOG.json` for the most recently observed source availability.
''')
    return reference


class CheckpointSync:
    def __init__(self,client,state_path):
        self.client=client;self.state_path=state_path;self.state=json.loads(state_path.read_text()) if state_path.exists() else {'runs':{}}

    def sync_run(self,spec,run_id):
        local=self.state['runs'].setdefault(run_id,{'attached':{}});root=Path(spec['path'])/'checkpoints';available=[];pending=[];new=[]
        for checkpoint in sorted(root.glob('step-*')):
            if not checkpoint.is_dir():continue
            marker=checkpoint/'COMPLETE.json'
            if not marker.exists():pending.append(checkpoint.name);continue
            data=marker.read_bytes();saved=json.loads(data);digest=sha(data);entry={'name':checkpoint.name,'step':saved['cursor']['step'],'supervised_tokens':saved['cursor']['supervised_tokens'],'source_path':str(checkpoint.resolve()),'marker_sha256':digest};available.append(entry)
            known=local['attached'].get(checkpoint.name)
            if known:
                if known['marker_sha256']!=digest:raise ValueError('An attached checkpoint was replaced under the same name')
                continue
            with tempfile.TemporaryDirectory(prefix='archlab-checkpoint-reference-') as temporary:
                folder=Path(temporary);reference=make_bundle(checkpoint,folder,spec['variant']);artifact_path='checkpoints/'+checkpoint.name
                self.client.log_artifacts(run_id,str(folder),artifact_path=artifact_path)
                # Verify metadata readback before publishing the attachment receipt.
                with tempfile.TemporaryDirectory(prefix='archlab-checkpoint-readback-') as download:
                    returned=Path(self.client.download_artifacts(run_id,artifact_path+'/REFERENCE.json',dst_path=download))
                    if json.loads(returned.read_text())!=reference:raise ValueError('MLflow reference readback mismatch')
                receipt={'attached_at_utc':datetime.now(timezone.utc).isoformat(),'metadata_only':True,'reference_readback_verified':True,'checkpoint_marker_sha256':digest,'rank_manifests_zip_sha256':reference['rank_manifests_zip_sha256']}
                atomic_json(folder/'ATTACHED.json',receipt);self.client.log_artifact(run_id,str(folder/'ATTACHED.json'),artifact_path=artifact_path)
            local['attached'][checkpoint.name]={**entry,'marker_sha256':digest,'artifact_path':artifact_path};atomic_json(self.state_path,self.state);new.append(checkpoint.name)
            print(json.dumps({'event':'checkpoint_attached','run':spec['id'],'checkpoint':checkpoint.name,'tokens':entry['supervised_tokens'],'mode':'reference'}),flush=True)
        names={x['name'] for x in available};catalog={'attachment_mode':'reference-and-manifests','weights_uploaded':False,'checked_at_utc':datetime.now(timezone.utc).isoformat(),'checkpoints':[dict(v,source_complete_marker_present=k in names) for k,v in sorted(local['attached'].items())],'in_progress':pending,'latest_complete':available[-1]['name'] if available else None}
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);atomic_json(folder/'CATALOG.json',catalog);self.client.log_artifact(run_id,str(folder/'CATALOG.json'),artifact_path='checkpoints')
            if available:
                last=available[-1];atomic_json(folder/'LATEST.json',{'checkpoint':last['name'],'artifact_path':'checkpoints/'+last['name'],'source_path':last['source_path'],'step':last['step'],'supervised_tokens':last['supervised_tokens'],'metadata_only':True});self.client.log_artifact(run_id,str(folder/'LATEST.json'),artifact_path='checkpoints')
                self.client.set_tag(run_id,'checkpoint.artifacts','checkpoints/'+last['name']);self.client.set_tag(run_id,'checkpoint.attachment_mode','external-filesystem reference + manifests')
            else:
                atomic_json(folder/'LATEST.json',{'checkpoint':None,'source_available':False,'metadata_only':True});self.client.log_artifact(run_id,str(folder/'LATEST.json'),artifact_path='checkpoints');self.client.set_tag(run_id,'checkpoint.artifacts','unavailable')
        return {'id':spec['id'],'run_id':run_id,'attached':len(local['attached']),'new':new,'latest_complete':catalog['latest_complete'],'in_progress':pending}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--credentials',type=Path,required=True);parser.add_argument('--runs-state',type=Path,required=True);parser.add_argument('--state',type=Path,required=True);parser.add_argument('--watch',action='store_true');parser.add_argument('--interval',type=int,default=60);args=parser.parse_args();args.state.parent.mkdir(parents=True,exist_ok=True)
    with args.state.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);sync=CheckpointSync(configure_client(args.credentials),args.state)
        while True:
            ids=json.loads(args.runs_state.read_text())['runs'];report={'utc':datetime.now(timezone.utc).isoformat(),'runs':[],'errors':[]}
            for spec in json.loads(args.config.read_text())['runs']:
                try:report['runs'].append(sync.sync_run(spec,ids[spec['id']]['run_id']))
                except Exception as error:
                    report['errors'].append({'id':spec['id'],'type':type(error).__name__});print(json.dumps(report['errors'][-1]),flush=True)
                    if not args.watch:raise
            atomic_json(args.state.parent/'CHECKPOINT_SYNC_STATUS.json',report)
            if not args.watch:return
            time.sleep(args.interval)


if __name__=='__main__':main()
