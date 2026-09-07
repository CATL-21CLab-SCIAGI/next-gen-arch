"""Small numerical/protocol oracles; these are not full-model quality scores."""

from pathlib import Path
from types import SimpleNamespace
import importlib.util
import unittest

import torch

from archlab.benchmarks.capability import EvaluationConfig, MathGrader, paired_statistics, select_cases
from archlab.automodel.evaluate import added_modules, continuation_scores, encode_pair, math_completion


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(t) for t in tokens)


class TinyCausalModel(torch.nn.Module):
    """Fixed random transition matrix: exact, nonuniform next-token oracle."""

    def __init__(self):
        super().__init__()
        self.register_buffer("table", torch.randn(128, 128, generator=torch.Generator().manual_seed(42)))

    def forward(self, input_ids, logits_to_keep, **kwargs):
        return SimpleNamespace(logits=self.table[input_ids[:, -logits_to_keep:]])


class CapabilityTests(unittest.TestCase):
    def test_paired_direction_and_uncertainty(self):
        stats = paired_statistics([1, 1, 0, 0], [0, 1, 1, 1])
        self.assertEqual((stats["gains"], stats["regressions"], stats["delta_percentage_points"]), (2, 1, 25))
        equal = paired_statistics([1] * 8, [1] * 8)
        self.assertLess(equal["delta_95pct_interval_pp"][0], 0)
        self.assertGreater(equal["delta_95pct_interval_pp"][1], 0)
        self.assertEqual(equal["mcnemar_exact_two_sided_p"], 1)
        self.assertAlmostEqual(paired_statistics([0] * 5, [1] * 5)["mcnemar_exact_two_sided_p"], .0625)
        for base, adapted in (([], []), ([1], []), ([2], [1])):
            with self.assertRaises(ValueError):
                paired_statistics(base, adapted)

    def test_selection_and_budget_contract(self):
        rows = [{"id": str(i), "index": i, "subject": "ab"[i % 2], "task": "mmlu"} for i in range(20)]
        a = select_cases(rows, limits={"mmlu": 2}, seed=42)
        self.assertEqual(a, select_cases(rows, limits={"mmlu": 2}, seed=42))
        self.assertEqual({r["subject"] for r in a}, {"a", "b"})
        with self.assertRaises(ValueError):
            EvaluationConfig(max_context=4096)

    def test_loglikelihood_matches_tokenwise_oracle(self):
        model, tokenizer = TinyCausalModel(), CharacterTokenizer()
        for choices in (["A", "B"], [" A", " BB"]):
            result = continuation_scores(model, tokenizer, "Q:", choices, max_context=20, device="cpu")
            for index, choice in enumerate(choices):
                ids = tokenizer.encode("Q:" + choice)
                expected = sum(model.table[ids[j - 1]].log_softmax(-1)[ids[j]].item() for j in range(2, len(ids)))
                self.assertAlmostEqual(result["loglikelihoods"][index], expected, places=5)
                self.assertAlmostEqual(result["character_normalized_loglikelihoods"][index],
                                       expected / len(choice.removeprefix(" ")), places=5)
        self.assertEqual(encode_pair(tokenizer, "Q: ", "A"), (tokenizer.encode("Q:"), tokenizer.encode(" A")))
        with self.assertRaises(ValueError):
            continuation_scores(model, tokenizer, "Q:", ["long"], max_context=2, device="cpu")

    def test_generation_eos_and_cap(self):
        tokenizer, model = CharacterTokenizer(), TinyCausalModel()
        model.table.fill_(-10)
        model.table[:, ord("z")] = 10
        config = EvaluationConfig(max_context=20, max_new_tokens=4)
        result = math_completion(model, tokenizer, "Q", config=config, eos_ids={ord("z"), ord("x")}, device="cpu")
        self.assertEqual((result["stop_reason"], result["new_tokens"], result["completion"]), ("eos", 1, ""))
        result = math_completion(model, tokenizer, "Q", config=config, eos_ids={ord("x")}, device="cpu")
        self.assertEqual((result["stop_reason"], result["completion"]), ("token_cap", "zzzz"))

    @unittest.skipUnless(importlib.util.find_spec("lm_eval"), "requires existing pinned lm-eval source")
    def test_multiple_choice_prompts_match_pinned_harness(self):
        import lm_eval
        import yaml
        from jinja2 import Environment, StrictUndefined

        root = Path(lm_eval.__file__).parent / "tasks"
        project = Path(__file__).resolve().parent.parent
        prompts = yaml.safe_load((project / "src/archlab/prompts/capability_regression.yaml").read_text())
        template = Environment(undefined=StrictUndefined)
        default = yaml.safe_load((root / "mmlu/default/_default_template_yaml").read_text())
        subjects = list((root / "mmlu/default").glob("mmlu_*.yaml"))
        self.assertEqual(len(subjects), 57)
        for path in subjects:
            task = yaml.safe_load(path.read_text())
            row = {"subject": task["dataset_name"], "question": " Example? ", "choices": ["a", "b", "c", "d"]}
            expected = task["description"] + template.from_string(default["doc_to_text"]).render(**row)
            self.assertEqual(template.from_string(prompts["multiple_choice"]["mmlu"]).render(**row), expected)
        arc = yaml.safe_load((root / "arc/arc_easy.yaml").read_text())
        self.assertEqual(template.from_string(prompts["multiple_choice"]["arc_challenge"]).render(question="Example?"),
                         template.from_string(arc["doc_to_text"]).render(question="Example?"))

    @unittest.skipUnless(importlib.util.find_spec("lm_eval"), "requires existing pinned lm-eval source")
    def test_final_answer_grading(self):
        import lm_eval
        grader = MathGrader(Path(lm_eval.__file__).parent.parent)
        gsm = {"task": "gsm8k", "answer": "1234"}
        self.assertEqual(grader.score(gsm, "thought 99</think>The answer is $1,234.")["metrics"]["flexible_extract"], 1)
        self.assertEqual(grader.score(gsm, "The answer is 1234.")["metrics"]["flexible_extract"], 0)
        self.assertEqual(grader.score(gsm, "1234</think>The answer is 42.")["metrics"]["flexible_extract"], 0)
        self.assertEqual(grader.score({"task": "aime24", "answer": "42"},
                                     "thinking</think>Thus \\boxed{42}.")["metrics"]["exact_match"], 1)
        self.assertEqual(grader.score({"task": "aime25", "answer": "42"},
                                     "thinking \\boxed{42}")["metrics"]["exact_match"], 0)

    @unittest.skipUnless(importlib.util.find_spec("nemo_automodel"), "requires pinned AutoModel")
    def test_disabled_nonzero_additions_recover_original_exactly(self):
        from test_automodel_simplicial import AutoModelSimplicialTests
        from archlab.automodel.simplicial import AdditiveMoERead, install_simplicial_modules

        torch.manual_seed(138)
        model, config = AutoModelSimplicialTests().model_and_config()
        model.eval()
        tokens = torch.randint(2, 64, (1, 11))
        originals = {n: (p, p.detach().clone()) for n, p in model.named_parameters()}
        with torch.no_grad():
            before = model(input_ids=tokens, output_hidden_states=True)
        adapters = install_simplicial_modules(model, config, backend="reference")
        with torch.no_grad():
            for adapter in adapters.values():
                adapter.output.weight.normal_(std=.02)
            self.assertFalse(torch.equal(before.logits, model(input_ids=tokens).logits))
            with added_modules(model, enabled=False):
                after = model(input_ids=tokens, output_hidden_states=True)
            torch.testing.assert_close(before.logits, after.logits, atol=0, rtol=0)
            self.assertEqual(len(before.hidden_states), len(after.hidden_states))
            for a, b in zip(before.hidden_states, after.hidden_states):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        for name, (parameter, saved) in originals.items():
            self.assertIs(dict(model.named_parameters())[name], parameter)
            torch.testing.assert_close(parameter, saved, atol=0, rtol=0)
        with self.assertRaisesRegex(RuntimeError, "test restore"):
            with added_modules(model, enabled=False):
                raise RuntimeError("test restore")
        self.assertTrue(all(m.adapter_enabled for m in model.modules() if isinstance(m, AdditiveMoERead)))


if __name__ == "__main__":
    unittest.main()
