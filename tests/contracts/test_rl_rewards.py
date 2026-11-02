import unittest

from archlab.rl.rewards import (
    MAX_ANSWER_CHARACTERS,
    REWARD_BACKEND,
    canonical_math_answer,
    extract_final_answer,
    verify_math_answer,
)


class ExactMathRewardTests(unittest.TestCase):
    def test_exact_scalar_equivalence(self):
        examples = {
            "42": "42",
            "−42": "-42",
            " 1,234,567 ": "1234567",
            ".125": "1/8",
            "1.25e-2": "1/80",
            "2/4": "1/2",
            r"\frac{2}{4}": "1/2",
            r"-\dfrac{3}{\frac{9}{2}}": "-2/3",
            r"$\tfrac{ 1 }{ 2 }$": "1/2",
            r"\left(\frac{1}{2}+\frac{1}{3}\right)": "5/6",
            r"\frac{2^{3}}{4}": "2",
            "-2^2": "-4",
            "(-2)^2": "4",
            "25%": "1/4",
            r"25\%": "1/4",
            r"2\times3\div4": "3/2",
        }
        for answer, expected in examples.items():
            with self.subTest(answer=answer):
                self.assertEqual(canonical_math_answer(answer), expected)

    def test_final_answer_formats(self):
        completions = [
            r"After combining the terms, we obtain \boxed{\frac{2}{4}}.",
            r"Final answer: $\frac{2}{4}$.",
            r"The answer is: \frac{2}{4}.",
            "#### 0.5",
            "0.5",
            r"Final answer: \boxed{0.5}",
            r"<think>An intermediate \boxed{99} is not final.</think>\boxed{0.5}",
            r"reasoning from an open template</think>\boxed{0.5}",
        ]
        for completion in completions:
            with self.subTest(completion=completion):
                result = verify_math_answer(completion, "1/2")
                self.assertTrue(result.correct)
                self.assertEqual(result.reward, 1.0)
                self.assertEqual(result.reason, "correct")
                self.assertEqual(result.backend, REWARD_BACKEND)

    def test_no_reward_for_unsupported_or_adversarial_answer(self):
        completions = [
            "I considered 42 but the answer is unknown.",
            r"\boxed{42} or \boxed{43}",
            r"\boxed{42} actually the answer is 43",
            "Final answer: 42\nFinal answer: 43",
            "42 dollars",
            r"\boxed{42\text{ or 43}}",
            r"\boxed{42",
            r"\boxed42",
            r"\boxed{42}/0",
            r"Final answer: 43, but \boxed{42}",
            "4 2",
            r"4\,2",
            "1,2",
            "nan",
            "inf",
            "1/0",
            "0^0",
            r"<think>\boxed{42}",
            r"</think><think>\boxed{42}",
            "__import__('os').system('echo 42')",
            "42; 0",
            r"\boxed{42}\text{ meters}",
        ]
        for completion in completions:
            with self.subTest(completion=completion):
                self.assertIsNone(extract_final_answer(completion))
                self.assertEqual(verify_math_answer(completion, "42").reward, 0.0)

    def test_symbolic_or_units_never_rewarded_even_if_identical(self):
        for answer in [r"\sqrt{2}", "x+1", "x=2", "2 cm", "{1,2}", r"\pi"]:
            with self.subTest(answer=answer):
                result = verify_math_answer(answer, answer)
                self.assertEqual(result.reward, 0.0)
                self.assertEqual(result.reason, "unsupported_reference")

    def test_incorrect_is_distinct_from_parse_failure(self):
        wrong = verify_math_answer(r"\boxed{41}", "42")
        self.assertEqual(wrong.reason, "incorrect")
        self.assertEqual(wrong.canonical_answer, "41")
        self.assertEqual(wrong.canonical_reference, "42")
        missing = verify_math_answer("I do not know", "42")
        self.assertEqual(missing.reason, "missing_or_unsupported_final_answer")
        self.assertIsNone(missing.answer)

    def test_decimal_comparison_has_no_loose_tolerance(self):
        self.assertEqual(verify_math_answer("0.33333333333333", "1/3").reward, 0.0)
        self.assertEqual(verify_math_answer("42.000000001", "42").reward, 0.0)

    def test_resource_limits(self):
        for answer in ["9" * (MAX_ANSWER_CHARACTERS + 1), "1e999999", "9^999", "(" * 40 + "1" + ")" * 40]:
            with self.subTest(answer=answer):
                self.assertIsNone(canonical_math_answer(answer))
        self.assertIsNone(extract_final_answer("x" * 70000 + r"\boxed{42}"))


if __name__ == "__main__":
    unittest.main()
