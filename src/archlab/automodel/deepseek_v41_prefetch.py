"""Read ahead only selected frozen expert ranges into the host page cache."""
from __future__ import annotations
import argparse,concurrent.futures,json,os,re,struct,time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--first-layer',type=int,default=0)
    parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args();mapping=json.loads((args.weights/'model.safetensors.index.json').read_text())['weight_map']
    work=[]
    for name in sorted(set(mapping.values())):
        path=args.weights/name
        with path.open('rb') as stream:
            size=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(size))
        intervals=[]
        for key,value in header.items():
            match=re.match(r'layers\.(\d+)\.ffn\.experts\.',key)
            if match and int(match.group(1))>=args.first_layer:
                intervals.append((8+size+value['data_offsets'][0],8+size+value['data_offsets'][1]))
        merged=[]
        for lo,hi in sorted(intervals):
            if merged and lo<=merged[-1][1]+65536:merged[-1]=(merged[-1][0],max(hi,merged[-1][1]))
            else:merged.append((lo,hi))
        for lo,hi in merged:
            for start in range(lo,hi,32*1024**2):work.append((path,start,min(32*1024**2,hi-start)))
    def read(item):
        path,start,count=item
        with path.open('rb',buffering=0) as stream:
            data=os.pread(stream.fileno(),count,start)
        if len(data)!=count:raise IOError(f'short read: {path}:{start}')
        return count
    started=time.monotonic();done=0
    total=sum(item[2] for item in work)
    print(json.dumps({'event':'prefetch_start','bytes':total,'threads':args.threads}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(args.threads) as pool:
        for index,count in enumerate(pool.map(read,work)):
            done+=count
            if (index+1)%128==0:print(json.dumps({'event':'prefetch_progress','GiB':done/2**30,'seconds':time.monotonic()-started}),flush=True)
    report={'passed':True,'bytes':done,'seconds':time.monotonic()-started,'first_layer':args.first_layer,'threads':args.threads,'weights_modified':False}
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
