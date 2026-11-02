import copy

import pytest

from archlab.preprocessing.nemotron_rl_math import policy_messages, tokenize_row


def row():
    return dict(
        uuid="example",
        question="Compute 13 + 29.",
        expected_answer="42",
        responses_create_params=dict(input=[dict(role="user", content="Compute 13 + 29.")]),
    )


class Tokenizer:
    model_max_length = 10000

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict=False):
        assert add_generation_prompt
        assert not return_dict
        text = messages[0]["content"] + "<|im_start|>assistant\n"
        return self.encode(text, add_special_tokens=False) if tokenize else text

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return list(text.encode())


def test_only_prompt_is_encoded_and_gold_does_not_change_tokens():
    original = row()
    encoded = tokenize_row(original, Tokenizer())
    assert "42" not in encoded["prompt_text"]
    assert "expected_answer" not in encoded
    changed = copy.deepcopy(original)
    changed["expected_answer"] = "a different reward target"
    assert tokenize_row(changed, Tokenizer()) == encoded
    assert encoded["attention_mask"] == [1] * encoded["prompt_tokens"]


@pytest.mark.parametrize(
    "change", ["empty_question", "empty_answer", "placeholder", "assistant", "mismatch"]
)
def test_invalid_source_is_not_silently_tokenized(change):
    value = row()
    if change == "empty_question":
        value["question"] = ""
    elif change == "empty_answer":
        value["expected_answer"] = ""
    elif change == "placeholder":
        value["_hf_question_placeholder"] = {"row": 4}
    elif change == "assistant":
        value["responses_create_params"]["input"].append(dict(role="assistant", content="42"))
    else:
        value["responses_create_params"]["input"][0]["content"] = "Other question"
    with pytest.raises(ValueError):
        policy_messages(value)


def test_no_silent_truncation():
    tokenizer = Tokenizer()
    tokenizer.model_max_length = 4
    with pytest.raises(ValueError, match="context"):
        tokenize_row(row(), tokenizer)
