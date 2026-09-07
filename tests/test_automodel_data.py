import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


@unittest.skipUnless(importlib.util.find_spec("nemo_automodel"), "requires pinned AutoModel source")
class OnePassDataTests(unittest.TestCase):
    def make_data(self, root):
        from nemo_automodel.components.datasets.llm.megatron.indexed_dataset import IndexedDatasetBuilder
        from archlab.automodel.data import OnePassTokenWindows

        prefixes = []
        for part, values in enumerate((torch.arange(13), torch.arange(13, 38))):
            prefix = root / f"part-{part}"
            builder = IndexedDatasetBuilder(str(prefix) + ".bin", dtype=np.int32)
            builder.add_item(values)
            builder.end_document()
            builder.finalize(str(prefix) + ".idx")
            prefixes.append(prefix)
        return OnePassTokenWindows(prefixes, 4)

    def test_boundaries_and_no_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.make_data(Path(directory))
            self.assertEqual(len(data), 9)
            for index in range(len(data)):
                torch.testing.assert_close(data[index], torch.arange(index * 4, index * 4 + 5))
            for invalid in (-1, len(data)):
                with self.assertRaises(IndexError):
                    data[invalid]

    def test_rank_partition_resume_and_tail_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.make_data(Path(directory))
            targets = []
            for cursor in range(data.full_microbatches(2, 2)):
                for rank in range(2):
                    batch = data.batch(cursor, rank=rank, world_size=2, micro_batch=2)
                    replay = data.batch(cursor, rank=rank, world_size=2, micro_batch=2)
                    torch.testing.assert_close(batch["input_ids"], replay["input_ids"])
                    targets.extend(batch["labels"].flatten().tolist())
            self.assertEqual(targets, list(range(1, 33)))
            self.assertEqual(data.accounting(2, 2), {
                "source_tokens": 38, "full_microbatches": 2, "consumed_target_tokens": 32,
                "unused_final_target_tokens": 5, "initial_context_only_tokens": 1, "wrapped_tokens": 0})
            with self.assertRaises(IndexError):
                data.batch(2, rank=0, world_size=2, micro_batch=2)
