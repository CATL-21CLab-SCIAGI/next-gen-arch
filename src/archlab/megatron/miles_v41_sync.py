"""Bounded BF16 Engram transfer and explicit full-update transactions."""

import json
import re
from pathlib import Path

import torch
import torch.distributed as dist
from miles.backends.megatron_utils.update_weight import hf_weight_iterator_direct as direct
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement

from archlab.megatron.miles_v41_checkpoint import source_names


class ArchlabWeightIterator(direct.HfWeightIteratorDirect):
    forced_placement = WeightUpdatePlacement(gather_pp=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tables = [info for batch in self._non_expert_batches for info in batch
                       if info.name.endswith("v41_engram.table")]
        self._non_expert_batches = [
            kept for batch in self._non_expert_batches
            if (kept := [info for info in batch if not info.name.endswith("v41_engram.table")])]
        self.version = 0
        config = json.loads((Path(self.args.hf_checkpoint) / "config.json").read_text())
        text = config.get("text_config", config)
        self.logical_rows = dict(zip(text["engram_layer_ids"], text["engram_num_embeddings"], strict=True))

    def _convert_to_hf_param_units(self, named_params):
        for name, param in named_params:
            if ".archlab_adapter." in name:
                yield [(source_names(self.args, name, param)[0], param)]
            else:
                yield from super()._convert_to_hf_param_units([(name, param)])

    def _iter_hf_param_units(self, weights, *, materialize):
        self.version += 1
        version = torch.tensor([self.version], device="cuda", dtype=torch.int64)
        if materialize:
            yield [("archlab_update_begin", version)]
        yield from super()._iter_hf_param_units(weights, materialize=materialize)
        for info in self.tables:
            yield from self._table_units(info, weights, materialize)
        if materialize:
            yield [("archlab_update_end", version)]

    def _table_units(self, info, weights, materialize):
        parallel = get_parallel_state()
        layer = int(re.search(r"decoder.layers.(\d+)", info.name)[1])
        rows, cols = info.shape
        chunk_rows = max(1, min(rows, (32 * 2**20) // (cols * 2)))
        tp_ranks = dist.get_process_group_ranks(parallel.tp.group)
        pp_ranks = dist.get_process_group_ranks(parallel.pp.group)
        local = weights[info.name] if dist.get_rank() == info.src_rank else None
        for owner in range(parallel.tp.size):
            for start in range(0, rows, chunk_rows):
                global_start = owner * rows + start
                count = min(chunk_rows, rows - start, self.logical_rows[layer] - global_start)
                if count <= 0:
                    continue
                buffer = torch.empty(count, cols, dtype=torch.bfloat16, device="cuda")
                if parallel.tp.rank == owner:
                    if local is not None:
                        buffer.copy_(local[start:start + count])
                    if info.src_rank in pp_ranks:
                        dist.broadcast(buffer, src=info.src_rank, group=parallel.pp.group)
                dist.broadcast(buffer, src=tp_ranks[owner], group=parallel.tp.group)
                if materialize:
                    yield [(f"layers.{layer}.engram.embed.weight.rows.{global_start}", buffer)]


def install():
    direct.HfWeightIteratorDirect = ArchlabWeightIterator
    import os
    if os.environ.get("ARCHLAB_RL_OFFLOAD_POLICY") == "forbidden":
        from archlab.megatron.miles_v41_weight_session import install as install_session
        install_session()
