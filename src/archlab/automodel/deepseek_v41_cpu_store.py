"""Exact encoded expert weights resident in CPU RAM, with bounded parallel I/O."""
from __future__ import annotations
import concurrent.futures,json,math,os,struct,time
from pathlib import Path
import torch


class ResidentExpertStore:
    def __init__(self,root:Path,*,threads=8,pinned=True,reserve_gib=64):
        started=time.monotonic();root=Path(root)
        mapping=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
        self.entries={};self.buffers=[];plans=[];total=0
        types={'I8':torch.int8,'F8_E8M0':torch.float8_e8m0fnu}
        for filename in sorted({value for key,value in mapping.items() if key.startswith('layers.') and '.ffn.experts.' in key}):
            if Path(filename).name!=filename:raise ValueError('expert shard escapes model directory')
            path=root/filename
            with path.open('rb') as stream:
                length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length))
            selected=sorted(((name,value) for name,value in header.items() if name.startswith('layers.') and '.ffn.experts.' in name),key=lambda item:item[1]['data_offsets'][0])
            size=sum(value['data_offsets'][1]-value['data_offsets'][0] for _,value in selected)
            plans.append((path,8+length,selected,size));total+=size
        available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        if total+reserve_gib*2**30>available:raise MemoryError(f'expert CPU store needs {total/2**30:.1f}GiB plus64GiB reserve; only {available/2**30:.1f}GiB available')
        print(json.dumps({'event':'cpu_expert_store_allocate','GiB':total/2**30,'available_GiB':available/2**30,'pinned':pinned}),flush=True)
        work=[]
        for path,header_size,selected,size in plans:
            file_size=path.stat().st_size
            buffer=torch.empty(size,dtype=torch.uint8,device='cpu',pin_memory=pinned)
            self.buffers.append(buffer);raw=memoryview(buffer.numpy());destination=0;ranges=[]
            for name,meta in selected:
                lo,hi=meta['data_offsets'];count=hi-lo
                if meta['dtype'] not in types or count!=math.prod(meta['shape']):raise ValueError('unexpected expert encoding or extent')
                if header_size+hi>file_size:raise ValueError('expert source exceeds shard')
                self.entries[name]=buffer.narrow(0,destination,count).view(types[meta['dtype']]).reshape(meta['shape'])
                if ranges and ranges[-1][0]+ranges[-1][2]==header_size+lo and ranges[-1][1]+ranges[-1][2]==destination:
                    source,target,previous=ranges[-1];ranges[-1]=(source,target,previous+count)
                else:ranges.append((header_size+lo,destination,count))
                destination+=count
            for source,target,count in ranges:
                for offset in range(0,count,32*1024**2):
                    n=min(32*1024**2,count-offset)
                    work.append((path,source+offset,raw[target+offset:target+offset+n]))
        def read(item):
            path,offset,target=item;done=0
            fd=os.open(path,os.O_RDONLY)
            try:
                while done<len(target):
                    n=os.preadv(fd,[target[done:]],offset+done)
                    if n<=0:raise IOError(f'short expert read: {path}:{offset+done}')
                    done+=n
            finally:os.close(fd)
            return done
        copied=0
        with concurrent.futures.ThreadPoolExecutor(threads) as pool:
            for index,count in enumerate(pool.map(read,work)):
                copied+=count
                if (index+1)%256==0:print(json.dumps({'event':'cpu_expert_store_read','GiB':copied/2**30,'total_GiB':total/2**30,'seconds':time.monotonic()-started}),flush=True)
        if copied!=total:raise RuntimeError('incomplete CPU expert store')
        self.report={'bytes':total,'tensors':len(self.entries),'pinned':pinned,'seconds':time.monotonic()-started,'encoding_unchanged':True,'threads':threads}
        print(json.dumps({'event':'cpu_expert_store_ready',**self.report}),flush=True)

    def __contains__(self,name):return name in self.entries
    def get(self,name):return self.entries[name]
