"""Small CPU tests; official asset integration tests are run separately."""

import copy
import importlib.util
import os
import unittest
from pathlib import Path

from archlab.preprocessing.deepseek_v41 import (
    DeepSeekV41Renderer,
    coalesce_reasoning_fragments,
    prepare_messages,
    token_spans,
)


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

    def test_reasoning_fragments_merge_without_source_mutation(self):
        row = {"messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "", "reasoning_content": "first  \n"},
            {"role": "assistant", "content": "", "reasoning_content": "second"},
            {"role": "assistant", "content": "answer", "reasoning_content": "third"},
        ]}
        before = copy.deepcopy(row)
        normalized, repairs = coalesce_reasoning_fragments(row)
        self.assertEqual(row, before)
        self.assertEqual(len(normalized["messages"]), 2)
        self.assertEqual(normalized["messages"][1]["reasoning_content"], "first  \n\n\nsecond\n\nthird")
        self.assertEqual(normalized["messages"][1]["content"], "answer")
        self.assertEqual(repairs[0]["source_message_indices"], [1, 2, 3])
        self.assertEqual(repairs[0]["reasoning_character_lengths"], [8, 6, 5])

    def test_completed_answers_calls_and_empty_fragments_are_never_merged(self):
        retained = [
            {"content": "answer", "reasoning_content": "thought"},
            {"tool_calls": [{"id": "unresolved"}], "reasoning_content": "thought"},
            {"reasoning_content": ""},
        ]
        for fragment in retained:
            with self.subTest(fragment=fragment):
                row = {"messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", **fragment},
                    {"role": "assistant", "content": "answer"},
                ]}
                normalized, events = coalesce_reasoning_fragments(row)
                self.assertEqual(normalized, row)
                self.assertFalse(events[0]["messages_merged"])

    def test_unknown_reasoning_fragment_fields_rejected(self):
        with self.assertRaises(ValueError):
            coalesce_reasoning_fragments({"messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "reasoning_content": "thought", "task": "classification"},
                {"role": "assistant", "content": "answer"},
            ]})

    def test_successful_old_domain_has_no_normalization(self):
        row = {"messages": [{"role": "user", "content": "q"},
                            {"role": "assistant", "content": "a", "reasoning_content": "r"}]}
        normalized, repairs = coalesce_reasoning_fragments(row)
        self.assertIs(normalized, row)
        self.assertEqual(repairs, [])


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

    def test_split_reasoning_before_tool_call_native_parity(self):
        from tokenizers import Tokenizer

        renderer = DeepSeekV41Renderer(os.environ["ARCHLAB_DEEPSEEK_V41_ASSETS"])
        row = {"messages": [
            {"role": "user", "content": "QUESTION_ONLY"},
            {"role": "assistant", "content": "", "reasoning_content": "FRAGMENT_ONE"},
            {"role": "assistant", "content": "", "reasoning_content": "FRAGMENT_TWO",
             "tool_calls": [{"id": "call1", "type": "function", "function": {
                 "name": "python", "arguments": '{"code":"2+2"}',
             }}]},
            {"role": "tool", "tool_call_id": "call1", "content": "RESULT_ONLY"},
            {"role": "assistant", "content": "4", "reasoning_content": "FINAL_REASON"},
        ]}
        before = copy.deepcopy(row)
        text, has_tools, messages, metadata = renderer.render(row)
        self.assertEqual(row, before)
        self.assertTrue(has_tools)
        self.assertEqual(messages[1]["reasoning_content"], "FRAGMENT_ONE\n\nFRAGMENT_TWO")
        self.assertEqual(len(messages[1]["tool_calls"]), 1)
        self.assertEqual(messages[2]["content"], "RESULT_ONLY")
        self.assertEqual(text, renderer.encode(messages))
        tokenizer = Tokenizer.from_file(os.environ["ARCHLAB_DEEPSEEK_V41_ASSETS"] + "/tokenizer.json")
        enc = tokenizer.encode(text, add_special_tokens=False)
        spans = token_spans(enc.offsets, metadata["assistant_character_spans"])
        supervised = "".join(tokenizer.decode(enc.ids[a:b], skip_special_tokens=False) for a, b in spans)
        self.assertIn("FRAGMENT_ONE\n\nFRAGMENT_TWO", supervised)
        self.assertIn("FINAL_REASON", supervised)
        self.assertNotIn("QUESTION_ONLY", supervised)
        self.assertNotIn("RESULT_ONLY", supervised)
        self.assertEqual(metadata["message_repairs"][0]["source_message_indices"], [1, 2])

    @unittest.skipUnless(os.environ.get("ARCHLAB_NEMOTRON_MATH_SOURCE") and os.environ.get("ARCHLAB_V41_LEGACY_RENDERER"),
                         "requires original corpus and v1 renderer snapshot")
    def test_real_failure_and_old_success_domain(self):
        import pyarrow.parquet as pq
        from tokenizers import Tokenizer

        spec = importlib.util.spec_from_file_location("legacy_v41_test", os.environ["ARCHLAB_V41_LEGACY_RENDERER"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assets = os.environ["ARCHLAB_DEEPSEEK_V41_ASSETS"]
        old, new = module.DeepSeekV41Renderer(assets), DeepSeekV41Renderer(assets)
        source = Path(os.environ["ARCHLAB_NEMOTRON_MATH_SOURCE"]) / "data"
        tokenizer = Tokenizer.from_file(assets + "/tokenizer.json")
        # Every source variant, including tool traces: exact text, tokens, masks,
        # messages and metadata equality across the old accepted domain.
        for name in ("high_part00", "high_part01", "high_part02", "medium", "low"):
            batch = next(pq.ParquetFile(source / (name + ".parquet")).iter_batches(batch_size=8, row_groups=[0]))
            for row in batch.to_pylist():
                self.assertEqual(old.render(row), new.render(row))
        row = pq.ParquetFile(source / "high_part00.parquet").read_row_group(184).slice(63, 1).to_pylist()[0]
        with self.assertRaisesRegex(ValueError, "consecutive-assistant"):
            old.render(row)
        text, _, messages, metadata = new.render(row)
        self.assertEqual(metadata["message_repairs"][0]["source_message_indices"], [121, 122])
        self.assertEqual(text, new.encode(messages))
        enc = tokenizer.encode(text, add_special_tokens=False)
        spans = token_spans(enc.offsets, metadata["assistant_character_spans"])
        self.assertGreater(len(enc.ids), 0)
        self.assertEqual(len(spans), sum(m["role"] == "assistant" for m in messages))
        originals = row["messages"]
        self.assertTrue(all(m.get("reasoning_content", "") in text for m in originals if m["role"] == "assistant"))
        self.assertEqual(sum(bool(m.get("tool_calls")) for m in originals), sum(bool(m.get("tool_calls")) for m in messages))

    def test_native_adjacent_calls_and_terminal_call_are_preserved(self):
        from tokenizers import Tokenizer

        renderer = DeepSeekV41Renderer(os.environ["ARCHLAB_DEEPSEEK_V41_ASSETS"])
        tokenizer = Tokenizer.from_file(os.environ["ARCHLAB_DEEPSEEK_V41_ASSETS"] + "/tokenizer.json")
        row = {"messages": [
            {"role": "user", "content": "PROMPT_ONLY"},
            {"role": "assistant", "reasoning_content": "REASON_ONE", "tool_calls": [
                {"id": "call0", "type": "function", "function": {"name": "python", "arguments": '{"code":"CALL_ZERO"}'}}]},
            {"role": "assistant", "reasoning_content": "REASON_TWO", "tool_calls": [
                {"id": "call1", "type": "function", "function": {"name": "python", "arguments": '{"code":"CALL_ONE"}'}}]},
            {"role": "tool", "tool_call_id": "call1", "content": "RESULT_ONLY"},
            {"role": "assistant", "reasoning_content": "REASON_LAST", "tool_calls": [
                {"id": "call2", "type": "function", "function": {"name": "python", "arguments": '{"code":"CALL_TWO"}'}}]},
        ]}
        before = copy.deepcopy(row)
        text, _, messages, metadata = renderer.render(row)
        self.assertEqual(row, before)
        native_messages, _ = prepare_messages(row)
        self.assertEqual(messages, native_messages)
        self.assertEqual(text, renderer.encode(native_messages))
        self.assertEqual(len(messages), 5)
        self.assertEqual(text.count(renderer.encoder.eos_token), 3)
        self.assertFalse(metadata["message_repairs"][-1]["complete_answer"])
        enc = tokenizer.encode(text, add_special_tokens=False)
        spans = token_spans(enc.offsets, metadata["assistant_character_spans"])
        self.assertEqual(len(spans), 3)
        supervised = "".join(tokenizer.decode(enc.ids[a:b], skip_special_tokens=False) for a, b in spans)
        for expected in ("REASON_ONE", "REASON_TWO", "REASON_LAST", "CALL_ZERO", "CALL_ONE", "CALL_TWO"):
            self.assertIn(expected, supervised)
        self.assertNotIn("PROMPT_ONLY", supervised)
        self.assertNotIn("RESULT_ONLY", supervised)

    @unittest.skipUnless(os.environ.get("ARCHLAB_NEMOTRON_MATH_SOURCE"), "requires original corpus")
    def test_real_adjacent_calls_and_terminal_calls(self):
        import pyarrow.parquet as pq
        from tokenizers import Tokenizer

        assets = os.environ["ARCHLAB_DEEPSEEK_V41_ASSETS"]
        renderer, tokenizer = DeepSeekV41Renderer(assets), Tokenizer.from_file(assets + "/tokenizer.json")
        source = Path(os.environ["ARCHLAB_NEMOTRON_MATH_SOURCE"]) / "data/high_part00.parquet"
        parquet = pq.ParquetFile(source)
        for rg, offset in ((185, 315), (186, 67), (187, 2), (192, 87), (198, 18)):
            row = parquet.read_row_group(rg).slice(offset, 1).to_pylist()[0]
            before = copy.deepcopy(row)
            text, _, messages, metadata = renderer.render(row)
            self.assertEqual(row, before)
            self.assertEqual(text, renderer.encode(messages))
            self.assertTrue(metadata["message_repairs"])
            old_calls = [call for m in row["messages"] for call in m.get("tool_calls") or []]
            calls = [call for m in messages for call in m.get("tool_calls") or []]
            self.assertEqual([c["id"] for c in old_calls], [c["id"] for c in calls])
            self.assertEqual([m["content"] for m in row["messages"] if m["role"] == "tool"],
                             [m["content"] for m in messages if m["role"] == "tool"])
            self.assertTrue(all(m.get("reasoning_content", "") in text for m in row["messages"] if m["role"] == "assistant"))
            enc = tokenizer.encode(text, add_special_tokens=False)
            spans = token_spans(enc.offsets, metadata["assistant_character_spans"])
            self.assertEqual(len(spans), sum(m["role"] == "assistant" for m in messages))


if __name__ == "__main__":
    unittest.main()
