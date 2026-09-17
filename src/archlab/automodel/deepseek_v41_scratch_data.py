"""Exact shared cursor over immutable, document-preserving pretraining chunks."""
from __future__ import annotations
import bisect
import hashlib
import json
from pathlib import Path
import time
import numpy as np


class ScratchData:
    def __init__(self,root,*,split='train',wait_seconds=21600):
        self.root=Path(root);self.split=split;self.wait_seconds=wait_seconds
        self.contract=json.loads((self.root/'CONTRACT.json').read_text())
        self.budget=self.contract['training_targets' if split=='train' else 'validation_targets']
        self.sequence=self.contract['sequence'];self.chunks=[];self.window_ends=[];self.target_ends=[];self.cache={}

    def _extend(self,index):
        while not self.window_ends or index>=self.window_ends[-1]:
            if self.target_ends and self.target_ends[-1]>=self.budget:return False
            path=self.root/'chunks'/f'{self.split}-{len(self.chunks):06d}';deadline=time.monotonic()+self.wait_seconds
            while not (path/'COMPLETE.json').exists():
                if time.monotonic()>=deadline:raise TimeoutError(f'waiting for sealed data: {path}')
                time.sleep(1)
            m=json.loads((path/'COMPLETE.json').read_text())
            if m['index']!=len(self.chunks) or m['split']!=self.split or m['sequence']!=self.sequence:raise ValueError('data chunk order or geometry differs')
            self.chunks.append((path,m));self.window_ends.append((self.window_ends[-1] if self.window_ends else 0)+m['windows']);self.target_ends.append((self.target_ends[-1] if self.target_ends else 0)+m['targets'])
        return True

    def window(self,index):
        if index<0:raise ValueError('negative data cursor')
        if not self._extend(index):return np.array([2,2],dtype=np.int64),0
        chunk=bisect.bisect_right(self.window_ends,index);path,m=self.chunks[chunk]
        if chunk not in self.cache:
            # Keep one data chunk mapped per reader. Every rank verifies its small immutable chunk once when opened.
            for filename,digest in [('tokens.u32',m['tokens_sha256']),('windows.npy',m['windows_sha256'])]:
                if hashlib.sha256((path/filename).read_bytes()).hexdigest()!=digest:raise ValueError(f'data checksum mismatch: {path/filename}')
            self.cache={chunk:(np.memmap(path/'tokens.u32',dtype='<u4',mode='r'),np.load(path/'windows.npy',mmap_mode='r',allow_pickle=False))}
        tokens,windows=self.cache[chunk];local=index-(self.window_ends[chunk-1] if chunk else 0)
        offset,length,before=map(int,windows[local]);global_before=before+(self.target_ends[chunk-1] if chunk else 0);count=max(0,min(length,self.budget-global_before))
        return np.array(tokens[offset:offset+length+1],dtype=np.int64),count

    def batch(self,indices,*,device='cuda'):
        import torch
        inputs=np.full((len(indices),self.sequence),self.contract['pad_id'],dtype=np.int64);labels=np.full_like(inputs,-100);count=0
        for row,index in enumerate(indices):
            tokens,n=self.window(index);length=min(self.sequence,len(tokens)-1);inputs[row,:length]=tokens[:length];labels[row,:n]=tokens[1:n+1];count+=n
        return torch.from_numpy(inputs).to(device),torch.from_numpy(labels).to(device),count

    def targets_before(self,index):
        if index==0:return 0
        self._extend(index-1)
        if index>self.window_ends[-1]:return self.budget
        chunk=bisect.bisect_right(self.window_ends,index-1);path,m=self.chunks[chunk];windows=np.load(path/'windows.npy',mmap_mode='r',allow_pickle=False);local=index-1-(self.window_ends[chunk-1] if chunk else 0);_,length,before=map(int,windows[local]);return min(self.budget,(self.target_ends[chunk-1] if chunk else 0)+before+length)
