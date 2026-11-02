import copy

import pytest

from archlab.preprocessing.limite import exclusion_reason, render_row


@pytest.mark.parametrize('change', [
    {'tools': [{'name': 'python'}]},
    {'messages': [{'role': 'tool', 'content': '4'}]},
    {'messages': [{'role': 'assistant', 'tool_calls': [{'function': {'name': 'python'}}]}]},
    {'messages': [{'role': 'assistant', 'function_call': {'name': 'python'}}]},
])
def test_tools_explicitly_excluded(change):
    row = {'messages': [{'role': 'assistant', 'content': '4'}], **change}
    assert exclusion_reason(row) == 'tools_unsupported_by_native_template'
    with pytest.raises(ValueError, match='exclusion manifest'):
        render_row(None, row)


def test_empty_call_fields_removed_without_losing_reasoning_or_mutating_source():
    row = {'tools': [], 'messages': [
        {'role': 'user', 'content': '2+2?', 'tool_calls': []},
        {'role': 'assistant', 'content': '4', 'reasoning_content': 'Two plus two', 'tool_calls': []},
    ]}
    original = copy.deepcopy(row)

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert all('tool_calls' not in m for m in messages)
            assert messages[-1]['reasoning_content'] == 'Two plus two'
            assert kwargs == {'tokenize': False, 'add_generation_prompt': False}
            return 'native<|im_end|>\n'

    assert exclusion_reason(row) is None
    text, tools, _, _ = render_row(Tokenizer(), row)
    assert text.endswith('<|im_end|>\n')
    assert not tools
    assert row == original


def test_custom_system_is_excluded_without_rewriting_source():
    from jinja2.exceptions import TemplateError

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            if messages[0]['content'] != 'canonical':
                raise TemplateError('custom system messages are not supported')
            return 'native'

    row = {'messages': [{'role': 'system', 'content': 'Reasoning: high'}]}
    assert exclusion_reason(row, Tokenizer()) == 'system_prompt_unsupported_by_native_template'
    assert row['messages'][0]['content'] == 'Reasoning: high'
    assert exclusion_reason({'messages': [{'role': 'system', 'content': 'canonical'}]}, Tokenizer()) is None
