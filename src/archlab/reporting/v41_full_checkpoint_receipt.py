"""Validate full checkpoint manifests, payload lengths and representative hashes."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def verify_checkpoint(path):
    import torch
    marker=json.loads((path/'COMPLETE.json').read_text())
    if marker['format']!='archlab-v41-full-sharded-v1' or marker['world_size']!=16:
        raise ValueError('unexpected checkpoint format or mesh')
    payload_bytes=0
    tensors=0
    tensor_chunks=0
    optimizer_files=0
    samples=[]
    names=None
    for rank,relative in enumerate(marker['manifests']):
        manifest_path=path/relative
        manifest=json.loads(manifest_path.read_text())
        if manifest['rank']!=rank or manifest['cursor']!=marker['cursor'] or manifest['contract']!=marker['contract']:
            raise ValueError(f'checkpoint manifest identity mismatch: {rank}')
        current_names=[e['name'] for e in manifest['tensors']]
        if names is not None and current_names!=names:
            raise ValueError('ranks have different parameter/buffer names')
        names=current_names
        for entry in manifest['tensors']:
            dtype=getattr(torch,entry['dtype'].removeprefix('torch.'))
            size=torch.empty(0,dtype=dtype).element_size()
            if sum(c['elements'] for c in entry['chunks'])!=math.prod(entry['shape']):
                raise ValueError(f'incomplete tensor chunks: {entry["name"]}')
            tensors+=1
            for chunk in entry['chunks']:
                file=manifest_path.parent/chunk['file']
                actual_bytes=file.stat().st_size
                raw_bytes=chunk['elements']*size
                if not raw_bytes <= actual_bytes < raw_bytes+65536:
                    raise ValueError(f'payload length mismatch: {file}')
                tensor_chunks+=1
                payload_bytes+=actual_bytes
        for filename in manifest['optimizer_states']:
            file=manifest_path.parent/filename
            if file.stat().st_size<=0:
                raise ValueError(f'empty optimizer state: {file}')
            optimizer_files+=1
            payload_bytes+=file.stat().st_size
        rng=torch.load(manifest_path.parent/'rng.pt',map_location='cpu',weights_only=True)
        if set(rng)!={'cpu_rng','cuda_rng'}:
            raise ValueError('incomplete RNG state')
        for filename in (manifest['optimizer_states'][0],manifest['optimizer_states'][-1]):
            state=torch.load(manifest_path.parent/filename,map_location='cpu',weights_only=True)
            if state['step']!=marker['cursor']['phase_step']:
                raise ValueError(f'optimizer step disagrees with cursor: {rank}')
        if rank==0:
            for name in ('model.embed_tokens.weight','lm_head.weight'):
                entry=next(e for e in manifest['tensors'] if e['name']==name)
                chunk=entry['chunks'][0]
                value=torch.load(manifest_path.parent/chunk['file'],map_location='cpu',weights_only=True)
                digest=hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                if digest!=chunk['sha256']:
                    raise ValueError(f'payload checksum mismatch: {name}')
                samples.append({'rank':rank,'name':name,'sha256':digest,'verified':True})
    return {'verified_at_utc':datetime.now(timezone.utc).isoformat(),'passed':True,'path':str(path.resolve()),
            'cursor':marker['cursor'],'rank_manifests':len(marker['manifests']),'tensor_entries':tensors,
            'tensor_payload_chunks':tensor_chunks,'optimizer_state_files':optimizer_files,
            'tensor_and_optimizer_payload_gib':payload_bytes/2**30,'representative_checksums':samples,
            'all_payload_lengths_verified':True,'all_rank_rng_states_present':True,
            'optimizer_counter_samples_match_cursor_on_all_ranks':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    receipt=verify_checkpoint(args.checkpoint)
    args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
