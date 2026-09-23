"""Generate paired modular-search probes with exact answers and matched marginals.

The query T and two list entries define a three-way relation. This is an
exploratory language task, not an instantiation of a transformer lower bound.
Token-window eligibility must be audited with the actual native chat encoder.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

TEMPLATE_PATH = Path(__file__).parents[1] / "prompts" / "deepseek_v41_pair_search_templates_v1.yaml"


def witnesses(recent, earlier, target, modulus):
    """Return all matching index pairs using exact modular complements."""
    locations = {tuple(value): i for i, value in enumerate(earlier)}
    return [
        (i, locations[complement])
        for i, (x, y) in enumerate(recent)
        if (complement := ((target[0] - x) % modulus, (target[1] - y) % modulus)) in locations
    ]


def make_twins(rng, *, earlier_count, recent_count=2, modulus=97):
    """Return a NO instance and a unique-witness YES twin.

    Every recent entry has first-coordinate and second-coordinate matches in
    the NO instance, but never in the same earlier entry. Swapping just two
    earlier y coordinates creates one solution without changing either
    coordinate's marginal multiset. All earlier entries match at least one
    coordinate for some recent entry, so they are substantive distractors.
    """
    if not 1 <= recent_count < modulus:
        raise ValueError("require 1 <= recent_count < modulus")
    capacity = 2 * recent_count * modulus - recent_count**2 - recent_count
    if not 2 * recent_count <= earlier_count <= capacity:
        raise ValueError("earlier_count must allow two decoys per recent entry and fit the domain")
    for _ in range(1000):
        target = (rng.randrange(modulus), rng.randrange(modulus))
        recent = list(
            zip(rng.sample(range(modulus), recent_count), rng.sample(range(modulus), recent_count), strict=False)
        )
        complements = [((target[0] - x) % modulus, (target[1] - y) % modulus) for x, y in recent]
        forbidden = set(complements)
        earlier = []

        def add(value, forbidden=forbidden, earlier=earlier):
            if value not in forbidden and value not in earlier:
                earlier.append(value)

        # First guarantee both kinds of marginal evidence for every candidate.
        for x, y in complements:
            add((x, (y + rng.randrange(1, modulus)) % modulus))
            add(((x + rng.randrange(1, modulus)) % modulus, y))
        # Enumerate the finite pool to avoid rejection-sampling hangs.
        pool = sorted(
            {(x, v) for x, _ in complements for v in range(modulus)}
            | {(v, y) for _, y in complements for v in range(modulus)}
        )
        rng.shuffle(pool)
        for value in pool:
            if len(earlier) >= earlier_count:
                break
            add(value)
        rng.shuffle(earlier)
        swaps = [
            (i, j)
            for i in range(earlier_count)
            for j in range(i + 1, earlier_count)
            if (earlier[i][0], earlier[j][1]) in forbidden
            or (earlier[j][0], earlier[i][1]) in forbidden
        ]
        rng.shuffle(swaps)
        for i, j in swaps:
            positive = list(earlier)
            positive[i] = (earlier[i][0], earlier[j][1])
            positive[j] = (earlier[j][0], earlier[i][1])
            if (
                len(set(positive)) == earlier_count
                and len(witnesses(recent, positive, target, modulus)) == 1
            ):
                return recent, earlier, positive, target
    raise RuntimeError("could not construct a unique-witness twin; change the dimensions or seed")


def generate_suite(
    *, seed=20260920, sizes=(8, 16, 32, 64), pairs_per_size=1, recent_count=2, modulus=97
):
    if pairs_per_size < 1 or not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("require positive pairs_per_size and distinct nonempty sizes")
    template = yaml.safe_load(TEMPLATE_PATH.read_text())["templates"]["modular_pair"]
    rng = random.Random(seed)
    rows = []

    def fmt(values):
        return " ".join(f"{x},{y}" for x, y in values)

    for size in sizes:
        for number in range(pairs_per_size):
            recent, negative, positive, target = make_twins(
                rng, earlier_count=size, recent_count=recent_count, modulus=modulus
            )
            twins = [negative, positive]
            rng.shuffle(twins)
            pair_id = f"n{size}-pair{number}"
            for index, earlier in enumerate(twins):
                solution = witnesses(recent, earlier, target, modulus)
                rows.append(
                    {
                        "id": f"{pair_id}-case{index}",
                        "tags": ["modular_pair_search", "matched_marginals", f"earlier_{size}"],
                        "pair_id": pair_id,
                        "expected_answer": "YES" if solution else "NO",
                        "candidate_pairs": size * recent_count,
                        "oracle_witness_indices": [list(pair) for pair in solution],
                        "text": template.format(
                            modulus=modulus,
                            long_entries=fmt(earlier),
                            recent_entries=fmt(recent),
                            target=fmt([target]),
                        ),
                    }
                )
    return {
        "schema_version": 1,
        "generator": "archlab.evaluation.pair_search:v1",
        "seed": seed,
        "modulus": modulus,
        "recent_count": recent_count,
        "status": "exploratory; model difficulty and native token windows unverified",
        "prompts": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--pairs-per-size", type=int, default=1)
    parser.add_argument("--recent-count", type=int, default=2)
    parser.add_argument("--modulus", type=int, default=97)
    args = parser.parse_args()
    suite = generate_suite(
        seed=args.seed,
        sizes=tuple(args.sizes),
        pairs_per_size=args.pairs_per_size,
        recent_count=args.recent_count,
        modulus=args.modulus,
    )

    class PromptDumper(yaml.SafeDumper):
        pass

    PromptDumper.add_representer(
        str,
        lambda dumper, value: dumper.represent_scalar(
            "tag:yaml.org,2002:str", value, style="|" if "\n" in value else None
        ),
    )
    args.output.write_text(yaml.dump(suite, Dumper=PromptDumper, sort_keys=False, width=100))
    print(f"Wrote {len(suite['prompts'])} oracle-labelled prompts to {args.output}")


if __name__ == "__main__":
    main()
