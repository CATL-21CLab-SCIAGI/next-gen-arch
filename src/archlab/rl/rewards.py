"""Conservative final-answer rewards for unitless rational-valued math.

This is deliberately not a general symbolic verifier. The accepted grammar is
bounded numeric arithmetic, integers, finite decimals/scientific notation,
fractions (including brace-delimited LaTeX fractions), and percentages. Variables,
units, sets, equations, roots and arbitrary LaTeX are unsupported, even when two
unsupported strings are identical. No generated text is evaluated as Python.

The environment does not provide ``math_verify``. This small exact subset is
explicitly versioned so filtering and reward evaluation can use the same contract
without silently changing behavior when packages are later installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

REWARD_BACKEND = "exact-rational-scalar-v1"
MAX_ANSWER_CHARACTERS = 512
MAX_COMPLETION_CHARACTERS = 65536
_MAX_TOKENS = 128
_MAX_DEPTH = 16
_MAX_INTEGER_BITS = 4096
_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_TOKEN = re.compile(rf"{_NUMBER}|\\(?:frac|dfrac|tfrac)|[+*/^(){{}}%-]")
_BOX = re.compile(r"\\(?:boxed|fbox)\b")
_FINAL = re.compile(r"(?:^|\n)\s*(?:final\s+answer|answer)\s*:|\bthe\s+answer\s+is\s*:?|(?:^|\n)\s*####\s*", re.IGNORECASE)


class _Unsupported(ValueError):
    pass


def _checked(value: Fraction) -> Fraction:
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > _MAX_INTEGER_BITS:
        raise _Unsupported("numeric result too large")
    return value


class _RationalParser:
    """Recursive-descent parser using Fraction arithmetic, never eval/sympify."""

    def __init__(self, text: str):
        self.tokens: list[str] = []
        offset = 0
        while offset < len(text):
            if text[offset].isspace():
                offset += 1
                continue
            match = _TOKEN.match(text, offset)
            if match is None:
                raise _Unsupported("unsupported character or command")
            self.tokens.append(match.group())
            if len(self.tokens) > _MAX_TOKENS:
                raise _Unsupported("too many tokens")
            offset = match.end()
        self.index = 0
        self.depth = 0

    def peek(self) -> str | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def take(self, token: str | None = None) -> str:
        found = self.peek()
        if found is None or (token is not None and found != token):
            raise _Unsupported("missing token")
        self.index += 1
        return found

    def expression(self) -> Fraction:
        value = self.product()
        while self.peek() in ("+", "-"):
            operator = self.take()
            other = self.product()
            value = _checked(value + other if operator == "+" else value - other)
        return value

    def product(self) -> Fraction:
        value = self.unary()
        while self.peek() in ("*", "/"):
            operator = self.take()
            other = self.unary()
            value = _checked(value * other if operator == "*" else value / other)
        return value

    def unary(self) -> Fraction:
        self.depth += 1
        if self.depth > _MAX_DEPTH:
            raise _Unsupported("expression too deep")
        try:
            if self.peek() in ("+", "-"):
                sign = self.take()
                value = self.unary()
                return value if sign == "+" else -value
            return self.power()
        finally:
            self.depth -= 1

    def power(self) -> Fraction:
        value = self.primary()
        if self.peek() == "^":
            self.take("^")
            exponent = self.unary()
            if exponent.denominator != 1 or abs(exponent.numerator) > 32:
                raise _Unsupported("only small integer exponents are supported")
            if value == 0 and exponent <= 0:
                raise _Unsupported("undefined power")
            value = _checked(value ** exponent.numerator)
        if self.peek() == "%":
            self.take("%")
            value = _checked(value / 100)
        return value

    def primary(self) -> Fraction:
        token = self.peek()
        if token in ("(", "{"):
            opener = self.take()
            value = self.expression()
            self.take(")" if opener == "(" else "}")
            return value
        if token in (r"\frac", r"\dfrac", r"\tfrac"):
            self.take()
            self.take("{")
            numerator = self.expression()
            self.take("}")
            self.take("{")
            denominator = self.expression()
            self.take("}")
            return _checked(numerator / denominator)
        literal = self.take()
        if re.fullmatch(_NUMBER, literal) is None or len(literal) > 128:
            raise _Unsupported("expected bounded number")
        if "e" in literal.lower() and abs(int(literal.lower().split("e")[1])) > 128:
            raise _Unsupported("scientific exponent too large")
        return _checked(Fraction(literal))


def _math_wrapper(text: str) -> str:
    text = text.strip()
    for left, right in (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$")):
        if text.startswith(left) and text.endswith(right) and len(text) >= len(left + right):
            return text[len(left):-len(right)].strip()
    return text


def canonical_math_answer(answer: str) -> str | None:
    """Return exact ``numerator[/denominator]`` for the supported scalar subset.

    Input must be an answer, not a solution trace. Unsupported input, nonfinite
    numbers, invalid arithmetic and resource-bound violations all return None.
    Decimal comparisons are exact: no tolerance can turn a wrong number into a
    positive reward. This consequently rejects rounded approximations to a gold
    exact fraction unless they are mathematically identical.
    """
    if not isinstance(answer, str) or not answer or len(answer) > MAX_ANSWER_CHARACTERS:
        return None
    text = _math_wrapper(answer).replace("−", "-")
    text = re.sub(r"\\(?:left|right)\b", "", text)
    text = re.sub(r"\\(?:,|!|;|:)|\\(?:quad|qquad)\b", " ", text)
    text = text.replace(r"\times", "*").replace(r"\cdot", "*").replace(r"\div", "/")
    text = text.replace(r"\%", "%")
    # Only valid thousands groups are removed; lists such as '1,2' remain invalid.
    text = re.sub(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?![\d,])", lambda m: m.group().replace(",", ""), text)
    try:
        parser = _RationalParser(text)
        result = parser.expression()
        if parser.peek() is not None:
            return None
        return str(result)
    except (ValueError, ZeroDivisionError, OverflowError, RecursionError):
        return None


def extract_final_answer(completion: str) -> str | None:
    """Extract one unambiguous final answer without mining numbers from prose.

    A response may end in one boxed/fbox expression, an explicit final-answer
    marker, or consist entirely of a supported scalar. A completed think block
    is excluded. Multiple boxes/answer markers, text following the answer, and
    unfinished think blocks are rejected. This strict format prevents rewarding
    answer lists, copied intermediate values, and malformed boxes.
    """
    if not isinstance(completion, str) or len(completion) > MAX_COMPLETION_CHARACTERS:
        return None
    text = completion.strip()
    if "<think>" in text and "</think>" not in text:
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    if "<think>" in text or "</think>" in text:
        return None
    boxes = list(_BOX.finditer(text))
    if boxes:
        if len(boxes) != 1:
            return None
        markers = list(_FINAL.finditer(text[:boxes[0].start()]))
        if len(markers) > 1:
            return None
        if markers:
            between = text[markers[0].end():boxes[0].start()]
            if between.replace(r"\[", "").replace(r"\(", "").strip(" \t\r\n$"):
                return None
        start = boxes[0].end()
        while start < len(text) and text[start].isspace():
            start += 1
        if start == len(text) or text[start] != "{":
            return None
        depth = 1
        end = start + 1
        while end < len(text) and depth:
            depth += (text[end] == "{") - (text[end] == "}")
            if depth > _MAX_DEPTH:
                return None
            end += 1
        if depth:
            return None
        suffix = text[end:].replace(r"\]", "").replace(r"\)", "")
        if suffix.strip(" \t\r\n.$"):
            return None
        answer = text[start + 1:end - 1].strip()
    else:
        markers = list(_FINAL.finditer(text))
        if len(markers) > 1:
            return None
        answer = text[markers[0].end():].strip() if markers else text
        if answer.endswith("."):
            answer = answer[:-1].rstrip()
        answer = _math_wrapper(answer)
    return answer if canonical_math_answer(answer) is not None else None


@dataclass(frozen=True)
class MathReward:
    answer: str | None
    canonical_answer: str | None
    canonical_reference: str | None
    correct: bool
    reason: str
    backend: str = REWARD_BACKEND

    @property
    def reward(self) -> float:
        return float(self.correct)


def verify_math_answer(completion: str, reference: str) -> MathReward:
    """Binary verifiable outcome reward; never use a reference trace as a target."""
    expected = canonical_math_answer(reference)
    if expected is None:
        return MathReward(None, None, None, False, "unsupported_reference")
    answer = extract_final_answer(completion)
    if answer is None:
        return MathReward(None, None, expected, False, "missing_or_unsupported_final_answer")
    canonical = canonical_math_answer(answer)
    correct = canonical == expected
    return MathReward(answer, canonical, expected, correct, "correct" if correct else "incorrect")
