"""A sealed shared prompt curriculum with full-distribution coverage."""

from __future__ import annotations

import random

from torch.utils.data import Sampler


class MathCurriculumSampler(Sampler):
    """Keep upstream GRPO group/reuse geometry; anneal shared solvable prompts.

    Absolute optimizer clocks can differ across resumed variants. The phase
    offset makes their new protocol see the same prompts at the same phase step.
    No answer text or heldout evidence is supplied to this sampler.
    """

    def __init__(self, dataset, curriculum, *, max_steps, phase_start, prompts_per_batch,
                 num_generations, repeat_count, seed=42, anneal_steps=128, initial_fraction=0.5):
        indices = {row["problem_sha256"]: index for index, row in enumerate(dataset)}
        keys = curriculum["problem_sha256"]
        if not keys or len(set(keys)) != len(keys) or not set(keys) <= indices.keys():
            raise ValueError("curriculum must contain unique training-only problem hashes")
        self.pool = [indices[key] for key in keys]
        self.all = list(range(len(dataset)))
        self.max_steps, self.phase_start = max_steps, phase_start
        self.batch = prompts_per_batch
        self.generations, self.repeat = num_generations, repeat_count
        self.seed, self.anneal = seed, anneal_steps
        self.initial_fraction = initial_fraction
        if min(max_steps, prompts_per_batch, num_generations, repeat_count, anneal_steps) < 1:
            raise ValueError("curriculum dimensions must be positive")
        if not 0 <= initial_fraction <= 0.5:
            raise ValueError("curriculum must retain at least half the full distribution")

    def __len__(self):
        return self.max_steps * self.batch * self.generations * self.repeat

    def __iter__(self):
        for step in range(self.max_steps):
            phase = max(0, step - self.phase_start)
            rng = random.Random(self.seed + phase)
            fraction = self.initial_fraction * max(0.0, 1.0 - phase / self.anneal)
            count = min(round(self.batch * fraction), len(self.pool))
            chosen = rng.sample(self.pool, count)
            available = [index for index in self.all if index not in chosen]
            chosen.extend(rng.sample(available, self.batch - len(chosen)))
            rng.shuffle(chosen)
            for _ in range(self.repeat):
                for index in chosen:
                    yield from [index] * self.generations
