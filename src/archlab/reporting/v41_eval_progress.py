"""Summarize evaluation progress without changing a worker or training run."""
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    root=args.root;out=root/'results-v1';decoder=json.JSONDecoder();offset=0;latest={};raw=(root/'results-v1-node0.log').read_text()
    while True:
        offset=raw.find('{"event"',offset)
        if offset<0:break
        try:event,end=decoder.raw_decode(raw,offset);offset=end
        except json.JSONDecodeError:offset+=1;continue
        if event.get('rank') not in (0,None):continue
        name=event['event']
        if name.startswith('eval_weight_restore'):latest['restore_'+event['variant']]=event
        elif name in ('eval_model_layout_start','eval_model_layout_complete','eval_forward_qualified','heldout_math_round','multiple_choice_round','paired_evaluation_complete'):
            latest[name+(('_'+event['variant']) if 'variant' in event else '')]=event
    state={'as_of_utc':datetime.now(timezone.utc).isoformat(),'latest':latest,'complete':(out/'COMPLETE.json').exists(),
           'failure_files':[f.name for f in out.glob('FAILED*')],'restore_receipts':len(list(out.glob('*restore.json')))}
    (root/'PROGRESS.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state))


if __name__=='__main__':main()
