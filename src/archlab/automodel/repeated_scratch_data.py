"""Repeat one sealed target-token prefix without losing tokens at epoch boundaries."""

import numpy as np

from archlab.automodel.deepseek_v41_scratch_data import ScratchData


class RepeatedScratchData(ScratchData):
    def __init__(self, root, *, unique_targets=100_000_000, epochs=10):
        if (
            type(epochs) is not int
            or epochs < 1
            or type(unique_targets) is not int
            or unique_targets < 1
        ):
            raise ValueError("repetition requires positive integer target and epoch counts")
        self.reader = ScratchData(root)
        if unique_targets > self.reader.budget:
            raise ValueError("repeated prefix exceeds the sealed training corpus")
        self.reader.budget = unique_targets
        self.unique_targets, self.epochs = unique_targets, epochs
        self.root, self.contract, self.sequence = (
            self.reader.root,
            self.reader.contract,
            self.reader.sequence,
        )
        self.budget = unique_targets * epochs
        self.reader._extend(0)
        while self.reader.target_ends[-1] < unique_targets:
            self.reader._extend(self.reader.window_ends[-1])
        path, _ = self.reader.chunks[-1]
        windows = np.load(path / "windows.npy", mmap_mode="r", allow_pickle=False)
        prior_targets = self.reader.target_ends[-2] if len(self.reader.chunks) > 1 else 0
        prior_windows = self.reader.window_ends[-2] if len(self.reader.chunks) > 1 else 0
        last = int(np.searchsorted(windows[:, 2] + windows[:, 1], unique_targets - prior_targets))
        self.windows_per_epoch = prior_windows + last + 1
        if self.reader.targets_before(self.windows_per_epoch) != unique_targets:
            raise ValueError("prefix does not contain the exact declared target count")

    def window(self, index):
        if index < 0:
            raise ValueError("negative repeated data cursor")
        epoch, local = divmod(index, self.windows_per_epoch)
        if epoch >= self.epochs:
            return np.array([2, 2], dtype=np.int64), 0
        return self.reader.window(local)

    def targets_before(self, index):
        if index < 0:
            raise ValueError("negative repeated data cursor")
        epoch, local = divmod(index, self.windows_per_epoch)
        return min(self.budget, epoch * self.unique_targets + self.reader.targets_before(local))
