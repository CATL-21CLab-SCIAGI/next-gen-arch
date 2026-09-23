"""Seal a bounded paired capability sample and exact adaptation-overlap audit."""

from __future__ import annotations
import argparse,hashlib,json,random,unicodedata
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def key(text):return hashlib.sha256(' '.join(unicodedata.normalize('NFC',text).split()).encode()).hexdigest()


def prepare(*,capability,piqa,train,validation,checkpoint,output,seed=20260914):
    from datasets import Dataset
    import yaml
    rows=[json.loads(x) for x in (capability/'cases.jsonl').read_text().split('\n') if x.strip()]
    source_manifest=json.loads((capability/'manifest.json').read_text())
    if digest(capability/'cases.jsonl')!=source_manifest['cases_sha256']:raise ValueError('capability cases changed')
    selected=[]
    for task,count in (('mmlu',57),('arc_challenge',32)):
        pool=[row for row in rows if row['task']==task];rng=random.Random(f'{seed}:{task}')
        if task=='mmlu':
            subjects=sorted({r['subject'] for r in pool})
            if len(subjects)!=57:raise ValueError('MMLU subject count changed')
            chosen=[rng.choice([r for r in pool if r['subject']==subject]) for subject in subjects]
        else:chosen=rng.sample(pool,count)
        selected+=sorted(chosen,key=lambda row:row['index'])
    ds=Dataset.from_file(str(piqa))
    if len(ds)!=1838:raise ValueError('use the public labeled PIQA validation split')
    for index in sorted(random.Random(f'{seed}:piqa').sample(range(len(ds)),64)):
        row=ds[index]
        selected.append({'task':'piqa','index':index,'id':f'piqa:{index}','question':row['goal'],'choices':[row['sol1'],row['sol2']],'answer':row['label'],'subject':None})
    training=[json.loads(x) for x in (train/'windows.jsonl').read_text().split('\n') if x.strip()]
    random.Random(2234).shuffle(training)
    cursor=json.loads((checkpoint/'COMPLETE.json').read_text())['cursor']
    consumed={r['problem_sha256'] for r in training[:cursor['step']*32]}
    planned={r['problem_sha256'] for r in training}
    for row in selected:
        variants=[row['question'],row['question']+'\n'+'\n'.join(f'{chr(65+i)}. {choice}' for i,choice in enumerate(row['choices']))]
        hashes={key(text) for text in variants}
        row['exact_overlap_consumed_training']=bool(hashes & consumed)
        row['exact_overlap_planned_training']=bool(hashes & planned)
    windows=[json.loads(x) for x in (validation/'windows.jsonl').read_text().split('\n') if x.strip()]
    random.Random(2234).shuffle(windows)
    chosen_math=[]
    for mode in ('low','medium','high'):
        indices=[i for i,row in enumerate(windows) if row['mode']==mode]
        indices=random.Random(f'{seed}:math:{mode}').sample(indices,4)
        for index in sorted(indices):
            row=windows[index]
            assert row['problem_sha256'] not in planned
            chosen_math.append({'pilot_index':index,**row})
    output.mkdir(parents=True,exist_ok=False)
    with (output/'cases.jsonl').open('w') as stream:
        for row in selected:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    (output/'math_windows.json').write_text(json.dumps(chosen_math,indent=2)+'\n')
    manifest={'seed':seed,'checkpoint_cursor':cursor,'counts':{task:sum(r['task']==task for r in selected) for task in ('mmlu','arc_challenge','piqa')},
              'math_windows':len(chosen_math),'math_supervised_targets':sum(r['targets'] for r in chosen_math),'math_lengths':[r['length'] for r in chosen_math],
              'cases_sha256':digest(output/'cases.jsonl'),'math_windows_sha256':digest(output/'math_windows.json'),
              'sources':source_manifest['sources'],'piqa':{'path':str(piqa),'sha256':digest(piqa),'split':'validation','rows':len(ds)},
              'train_manifest_sha256':digest(train/'PILOT_READY.json'),'validation_manifest_sha256':digest(validation/'PILOT_READY.json'),
              'exact_consumed_training_overlaps':[r['id'] for r in selected if r['exact_overlap_consumed_training']],
              'exact_planned_training_overlaps':[r['id'] for r in selected if r['exact_overlap_planned_training']],
              'overlap_limit':'Exact normalized question/lettered-choice hashes only; near-duplicates and base pretraining data not audited.',
              'pilot_not_full_benchmark':True}
    (output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('capability','piqa','train','validation','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();print(json.dumps(prepare(**vars(args)),indent=2))


if __name__=='__main__':main()
