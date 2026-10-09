import json
import tempfile
import unittest
from pathlib import Path

from archlab.evaluation.deepseek_v41_compare_data import read_jsonl
from archlab.evaluation.deepseek_v41_compare_metrics import (
    math_summary,
    paired_binary,
    score_choices,
)


class EvaluationMetricsTests(unittest.TestCase):
    def test_jsonl_unicode_separators_stay_inside_strings(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "cases.jsonl"
            expected = [{"question": "first\u2028second\u0085third"}, {"question": "next"}]
            p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in expected))
            self.assertEqual(read_jsonl(p), expected)

    def test_character_normalization_changes_the_correct_choice(self):
        case = {"id": "test", "task": "piqa", "choices": ["x", "long"], "answer": 1}
        result = score_choices(case, {"simplicial": [-2.0, -3.5], "normal": [-5.0, -1.0]})
        self.assertEqual(result["simplicial"]["accuracy"], 0)
        self.assertEqual(result["simplicial"]["accuracy_norm"], 1)
        self.assertEqual(result["normal"]["accuracy"], 1)

    def test_paired_discordance_and_uncertainty(self):
        r = paired_binary([1, 0, 1, 0], [1, 1, 0, 0])
        self.assertEqual(r["normal_only_correct"], 1)
        self.assertEqual(r["simplicial_only_correct"], 1)
        self.assertEqual(r["delta_percentage_points"], 0)
        self.assertEqual(r["mcnemar_exact_two_sided_p"], 1)
        identical = paired_binary([1] * 20, [1] * 20)
        self.assertLess(identical["delta_95pct_interval_pp"][0], 0)
        self.assertGreater(identical["delta_95pct_interval_pp"][1], 0)

    def test_math_is_target_weighted_and_clustered_by_problem(self):
        records = []
        for problem, targets, ce in [("p1", 2, 1.0), ("p1", 3, 2.0), ("p2", 5, 3.0)]:
            s = {
                "nll": targets * ce,
                "top1": targets - 1,
                "top5": targets,
                "entropy": targets * 0.5,
            }
            n = {**s, "nll": targets * (ce + 0.1)}
            records.append(
                {
                    "targets": targets,
                    "simplicial": s,
                    "normal": n,
                    "mode": "high",
                    "has_tools": True,
                    "problem_sha256": problem,
                    "simplicial_to_normal_kl_sum": 0.1 * targets,
                    "normal_to_simplicial_kl_sum": 0.2 * targets,
                    "argmax_agreement_count": targets,
                }
            )
        r = math_summary(records, replicates=100)
        self.assertAlmostEqual(r["overall"]["variants"]["simplicial"]["cross_entropy"], 2.3)
        self.assertAlmostEqual(r["overall"]["normal_minus_simplicial_ce"], 0.1)
        self.assertEqual(r["paired_uncertainty"]["problem_clusters"], 2)
        for x in r["paired_uncertainty"]["cross_entropy_delta_95pct_interval"]:
            self.assertAlmostEqual(x, 0.1)
        self.assertEqual(r["overall"]["supervised_targets"], 10)


if __name__ == "__main__":
    unittest.main()
