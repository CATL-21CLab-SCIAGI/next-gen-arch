"""Check generated model-visible questions with an independent exhaustive oracle."""

from collections import Counter
import random
import unittest

from archlab.evaluation.pair_search import generate_suite, make_twins


def exhaustive(recent, earlier, target, modulus):
    return [(i, j) for i, a in enumerate(recent) for j, b in enumerate(earlier)
            if all((a[d] + b[d]) % modulus == target[d] for d in (0, 1))]


class PairSearchTests(unittest.TestCase):
    def test_twins_have_exact_labels_and_identical_marginals(self):
        for size in (8, 16, 32, 64, 128):
            for seed in range(20):
                with self.subTest(size=size, seed=seed):
                    recent, negative, positive, target = make_twins(random.Random(seed), earlier_count=size)
                    self.assertEqual(exhaustive(recent, negative, target, 97), [])
                    self.assertEqual(len(exhaustive(recent, positive, target, 97)), 1)
                    self.assertEqual(len(set(negative)), size)
                    self.assertEqual(len(set(positive)), size)
                    self.assertEqual(sum(a != b for a, b in zip(negative, positive)), 2)
                    for d in (0, 1):
                        self.assertEqual(Counter(v[d] for v in negative), Counter(v[d] for v in positive))
                        for a in recent:
                            self.assertTrue(any((a[d] + b[d]) % 97 == target[d] for b in negative))

    def test_visible_questions_match_labels_and_are_reproducible(self):
        suite = generate_suite(pairs_per_size=2)
        self.assertEqual(suite, generate_suite(pairs_per_size=2))
        self.assertNotEqual(suite, generate_suite(seed=20260921, pairs_per_size=2))
        labels = {}
        for row in suite["prompts"]:
            data = {}
            for line in row["text"].splitlines():
                if line.startswith(("A: ", "B: ", "T: ")):
                    key, values = line.split(": ")
                    data[key] = [tuple(map(int, v.split(","))) for v in values.split()]
            solution = exhaustive(data["A"], data["B"], data["T"][0], suite["modulus"])
            self.assertEqual(row["expected_answer"], "YES" if solution else "NO")
            self.assertEqual(row["oracle_witness_indices"], [list(v) for v in solution])
            self.assertIn("Answer with exactly YES or NO.", row["text"])
            labels.setdefault(row["pair_id"], []).append(row["expected_answer"])
        self.assertTrue(all(sorted(values) == ["NO", "YES"] for values in labels.values()))

    def test_invalid_dimensions_fail(self):
        for kwargs in ({"earlier_count": 3}, {"earlier_count": 10000},
                       {"earlier_count": 8, "recent_count": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_twins(random.Random(0), **kwargs)


if __name__ == "__main__":
    unittest.main()
