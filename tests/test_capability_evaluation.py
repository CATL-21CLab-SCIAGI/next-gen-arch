"""Small numerical/protocol oracles; these are not full-model quality scores."""

import importlib.metadata
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from archlab.evaluation.capability import (
    EvaluationConfig,
    MathGrader,
    paired_statistics,
    select_cases,
)


def pinned_harness_available():
    try:
        return importlib.metadata.version("lm-eval") == "0.4.13"
    except importlib.metadata.PackageNotFoundError:
        return False


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(t) for t in tokens)


class TinyCausalModel(torch.nn.Module):
    """Fixed random transition matrix: exact, nonuniform next-token oracle."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "table", torch.randn(128, 128, generator=torch.Generator().manual_seed(42))
        )

    def forward(self, input_ids, logits_to_keep, **kwargs):
        return SimpleNamespace(logits=self.table[input_ids[:, -logits_to_keep:]])


class CapabilityTests(unittest.TestCase):
    def test_paired_direction_and_uncertainty(self):
        stats = paired_statistics([1, 1, 0, 0], [0, 1, 1, 1])
        self.assertEqual(
            (stats["gains"], stats["regressions"], stats["delta_percentage_points"]), (2, 1, 25)
        )
        equal = paired_statistics([1] * 8, [1] * 8)
        self.assertLess(equal["delta_95pct_interval_pp"][0], 0)
        self.assertGreater(equal["delta_95pct_interval_pp"][1], 0)
        self.assertEqual(equal["mcnemar_exact_two_sided_p"], 1)
        self.assertAlmostEqual(
            paired_statistics([0] * 5, [1] * 5)["mcnemar_exact_two_sided_p"], 0.0625
        )
        for base, adapted in (([], []), ([1], []), ([2], [1])):
            with self.assertRaises(ValueError):
                paired_statistics(base, adapted)

    def test_selection_and_budget_contract(self):
        rows = [
            {"id": str(i), "index": i, "subject": "ab"[i % 2], "task": "mmlu"} for i in range(20)
        ]
        a = select_cases(rows, limits={"mmlu": 2}, seed=42)
        self.assertEqual(a, select_cases(rows, limits={"mmlu": 2}, seed=42))
        self.assertEqual({r["subject"] for r in a}, {"a", "b"})
        with self.assertRaises(ValueError):
            EvaluationConfig(max_context=4096)


    @unittest.skipUnless(
        pinned_harness_available(), "requires installed lm-eval==0.4.13"
    )
    def test_multiple_choice_prompts_match_pinned_harness(self):
        import lm_eval
        import yaml
        from jinja2 import Environment, StrictUndefined

        root = Path(lm_eval.__file__).parent / "tasks"
        project = Path(__file__).resolve().parent.parent
        prompts = yaml.safe_load(
            (project / "src/archlab/prompts/capability_regression.yaml").read_text()
        )
        template = Environment(undefined=StrictUndefined)
        default = yaml.safe_load((root / "mmlu/default/_default_template_yaml").read_text())
        subjects = list((root / "mmlu/default").glob("mmlu_*.yaml"))
        self.assertEqual(len(subjects), 57)
        for path in subjects:
            task = yaml.safe_load(path.read_text())
            row = {
                "subject": task["dataset_name"],
                "question": " Example? ",
                "choices": ["a", "b", "c", "d"],
            }
            expected = task["description"] + template.from_string(default["doc_to_text"]).render(
                **row
            )
            self.assertEqual(
                template.from_string(prompts["multiple_choice"]["mmlu"]).render(**row), expected
            )
        arc = yaml.safe_load((root / "arc/arc_easy.yaml").read_text())
        self.assertEqual(
            template.from_string(prompts["multiple_choice"]["arc_challenge"]).render(
                question="Example?"
            ),
            template.from_string(arc["doc_to_text"]).render(question="Example?"),
        )

    @unittest.skipUnless(
        pinned_harness_available(), "requires installed lm-eval==0.4.13"
    )
    def test_final_answer_grading(self):
        import lm_eval

        grader = MathGrader(Path(lm_eval.__file__).parent.parent)
        gsm = {"task": "gsm8k", "answer": "1234"}
        self.assertEqual(
            grader.score(gsm, "thought 99</think>The answer is $1,234.")["metrics"][
                "flexible_extract"
            ],
            1,
        )
        self.assertEqual(grader.score(gsm, "The answer is 1234.")["metrics"]["flexible_extract"], 0)
        self.assertEqual(
            grader.score(gsm, "1234</think>The answer is 42.")["metrics"]["flexible_extract"], 0
        )
        self.assertEqual(
            grader.score({"task": "aime24", "answer": "42"}, "thinking</think>Thus \\boxed{42}.")[
                "metrics"
            ]["exact_match"],
            1,
        )
        self.assertEqual(
            grader.score({"task": "aime25", "answer": "42"}, "thinking \\boxed{42}")["metrics"][
                "exact_match"
            ],
            0,
        )


if __name__ == "__main__":
    unittest.main()
