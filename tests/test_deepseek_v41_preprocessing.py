"""Small CPU tests; official asset integration tests are run separately."""

import copy
import os
import unittest

from archlab.preprocessing.deepseek_v41 import DeepSeekV41Renderer, prepare_messages, token_spans


class DeepSeekV41PreprocessingTests(unittest.TestCase):
    def test_tool_schema_attached_without_mutating_source(self):
        row = {"messages": [{"role": "user", "content": "question"},
                            {"role": "assistant", "content": "answer", "reasoning_content": "reason"}],
               "tools": [{"type": "function", "function": {"name": "python"}}]}
        before = copy.deepcopy(row)
        messages, has_tools = prepare_messages(row)
        self.assertTrue(has_tools)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["tools"], row["tools"])
        self.assertEqual(messages[-1]["reasoning_content"], "reason")
        self.assertEqual(row, before)

    def test_unfinished_trajectory_rejected(self):
        with self.assertRaises(ValueError):
            prepare_messages({"messages": [{"role": "assistant", "content": "x"},
                                           {"role": "tool", "content": "2"}]})

    def test_token_spans_preserve_multiple_turns(self):
        self.assertEqual(token_spans([(0, 2), (2, 4), (4, 5), (5, 8)], [(2, 4), (5, 8)]),
                         [[1, 2], [3, 4]])

    def test_boundary_crossing_rejected(self):
        with self.assertRaises(ValueError):
            token_spans([(0, 3), (3, 5)], [(2, 5)])


@unittest.skipUnless(os.environ.get("ARCHLAB_DEEPSEEK_V41_ASSETS"), "requires pinned upstream tokenizer assets")
class DeepSeekV41AssetTests(unittest.TestCase):
    def test_multiturn_tools_and_assistant_supervision(self):
        from tokenizers import Tokenizer

        root = os.environ["ARCHLAB_DEEPSEEK_V41_ASSETS"]
        renderer = DeepSeekV41Renderer(root)
        row = {"tools": [{"type": "function", "function": {
            "name": "python", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}},
        }}], "messages": [
            {"role": "user", "content": "Compute 2+2"},
            {"role": "assistant", "content": "", "reasoning_content": "FIRST_REASON",
             "tool_calls": [{"id": "call1", "type": "function", "function": {
                 "name": "python", "arguments": '{"code":"2+2"}',
             }}]},
            {"role": "tool", "tool_call_id": "call1", "content": "TOOL_RESULT_ONLY"},
            {"role": "assistant", "content": "4", "reasoning_content": "SECOND_REASON"},
        ]}
        before = copy.deepcopy(row)
        text, has_tools, messages, kwargs = renderer.render(row)
        self.assertTrue(has_tools)
        self.assertEqual(row, before)
        self.assertIn("Reasoning Effort: 75", text)
        self.assertIn("<｜DSML｜ calls>", text)
        tokenizer = Tokenizer.from_file(root + "/tokenizer.json")
        encoding = tokenizer.encode(text, add_special_tokens=False)
        spans = token_spans(encoding.offsets, kwargs["assistant_character_spans"])
        supervised = "".join(tokenizer.decode(encoding.ids[a:b], skip_special_tokens=False) for a, b in spans)
        self.assertIn("FIRST_REASON", supervised)
        self.assertIn("SECOND_REASON", supervised)
        self.assertIn("2+2", supervised)
        self.assertNotIn("TOOL_RESULT_ONLY", supervised)
        self.assertNotIn("Compute", supervised)
        self.assertEqual(len(spans), 2)
        self.assertEqual(encoding.ids[0], tokenizer.token_to_id(renderer.encoder.bos_token))
        self.assertEqual(encoding.ids[-1], tokenizer.token_to_id(renderer.encoder.eos_token))


if __name__ == "__main__":
    unittest.main()
