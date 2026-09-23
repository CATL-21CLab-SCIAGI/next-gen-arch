"""Qualify locally offloaded inference before benchmark scoring."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','weights','checkpoint','output','train-pilot','reference-receipts'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--contexts',type=int,nargs='+',default=[128,2048])
    args=p.parse_args()
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages
    select_container_kernel_packages(Path('/usr/local/lib/python3.12/dist-packages'))
    import torch
    from archlab.artifacts import atomic_write_json
    from archlab.automodel.deepseek_v41_data import MathPilot
    from archlab.automodel.deepseek_v41_local_inference import LocalV41Inference
    args.output.mkdir(parents=True,exist_ok=False)
    engine=LocalV41Inference(assets=args.assets,weights=args.weights,adapter_checkpoint=args.checkpoint,
                             context=max(2048,*args.contexts),max_batch_size=8,expert_cache_gib=96)
    atomic_write_json(args.output/'LOADING.json',engine.report)
    pilot=MathPilot(args.train_pilot,expected_split='train',expected_budget=1000000000)
    report={'loading':engine.report,'checks':[],'passed':False}
    for context in args.contexts:
        batches=[pilot.batch(rank,device='cpu',smoke_context=context,pad_to_full=True) for rank in range(8)]
        ids=torch.cat([b[0] for b in batches]);labels=torch.cat([b[1] for b in batches]).cuda()
        start=time.monotonic();_,hidden=engine.forward(ids,adapted=False,return_hidden=True)
        actual=[]
        for rank in range(8):
            totals=torch.zeros(2,device='cuda',dtype=torch.float64)
            for first in range(0,context,128):
                logits=engine.model.head(hidden[rank:rank+1,first:first+128],full_logits=True)
                target=labels[rank:rank+1,first:first+128];mask=target!=-100
                totals[0]+=torch.nn.functional.cross_entropy(logits[mask],target[mask],reduction='sum')
                totals[1]+=mask.sum()
            reference=json.loads((args.reference_receipts/f'rank{rank}.json').read_text())
            expected=next(row['reference_loss'] for row in reference['tests'] if row['context']==context)
            ce=float(totals[0]/totals[1]);delta=ce-expected
            actual.append({'rank_window':rank,'loss':ce,'reference_loss':expected,'delta_loss':delta,'passed':abs(delta)<.05})
        item={'context':context,'seconds':time.monotonic()-start,'per_window':actual,'cache_hits':engine.cache.hits,'cache_misses':engine.cache.misses,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30}
        report['checks'].append(item);atomic_write_json(args.output/'PROGRESS.json',report,allow_nan=False)
        print(json.dumps({'event':'local_forward_qualified',**item}),flush=True)
        if not all(r['passed'] for r in actual):raise ValueError('local base CE differs from qualified released reference')
    report['passed']=True;atomic_write_json(args.output/'COMPLETE.json',report,allow_nan=False)


if __name__=='__main__':main()
