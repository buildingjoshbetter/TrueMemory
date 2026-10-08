"""Real parser/extractor coverage with synthetic files and a fake provider only."""
from __future__ import annotations

import builtins
import hashlib
import json
import logging
import sys
import tempfile
import types
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_SECRET = "synthetic-private-diagnostic-sentinel"


class FakeLLMError(Exception):
    pass


def load_module(name: str) -> types.ModuleType:
    def forbidden_provider(*args: object, **kwargs: object) -> str:
        raise AssertionError("A test must install its own synthetic provider")

    models = types.SimpleNamespace(LLMConfig=object, LLMError=FakeLLMError, complete=forbidden_provider)

    def safe_import(name: str, globals: dict | None = None, locals: dict | None = None,
                    fromlist: tuple = (), level: int = 0) -> object:
        if name == "truememory.ingest.models":
            return models
        if name.startswith(("truememory", "torch", "numpy", "sentence_transformers")):
            raise AssertionError("Unexpected package or model import")
        return builtins.__import__(name, globals, locals, fromlist, level)

    module = types.ModuleType("synthetic_p01_" + name)
    module.__dict__["__builtins__"] = dict(vars(builtins), __import__=safe_import)
    path = ROOT / "truememory/ingest" / (name + ".py")
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


class TestTranscriptOutcomes(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = load_module("transcript")
        self.directory = tempfile.TemporaryDirectory(prefix="synthetic-ingestion-")
        self.addCleanup(self.directory.cleanup)
        self.directory_path = Path(self.directory.name)

    def write(self, data: bytes, name: str = "transcript.jsonl") -> Path:
        path = self.directory_path / name
        path.write_bytes(data)
        return path

    def test_valid_empty_and_known_nonconversation_records(self) -> None:
        inputs = ("", "  ", "[]", '{"type":"progress"}', '{"type":"file-history-snapshot"}',
                  '[{"type":"system","content":"synthetic metadata"}]',
                  '[{"role":"user","content":""}]')
        for text in inputs:
            with self.subTest(text=text):
                outcome = self.parser.parse_transcript_outcome(text)
                self.assertTrue(outcome.complete)
                self.assertEqual(outcome.messages, [])
                self.assertEqual(outcome.error_categories, ())
        self.assertTrue(self.parser.parse_transcript_outcome(self.write(b"")).complete)

    def test_explicit_missing_path_is_unreadable_but_string_remains_inline(self) -> None:
        path = self.directory_path / "missing.jsonl"
        outcome = self.parser.parse_transcript_outcome(path)
        self.assertEqual((outcome.status, outcome.messages, outcome.file_version), ("unreadable", [], None))
        self.assertEqual(outcome.error_categories, ("source_unreadable",))
        self.assertNotIn(str(path), repr(outcome))
        inline = self.parser.parse_transcript_outcome(str(path))
        self.assertTrue(inline.complete)
        self.assertEqual(inline.messages[0].content, str(path))
        self.assertEqual(self.parser.parse_transcript(path), [])

    def test_file_version_binds_captured_bytes_not_later_file(self) -> None:
        data = b'{"role":"user","content":"synthetic amber station"}\n'
        path = self.write(data)
        outcome = self.parser.parse_transcript_outcome(path)
        version = outcome.file_version
        self.assertTrue(outcome.complete)
        self.assertTrue(version.stable)
        self.assertEqual(version.byte_count, len(data))
        self.assertEqual(version.sha256, hashlib.sha256(data).hexdigest())
        self.assertEqual(version.before.inode, path.stat().st_ino)
        path.write_bytes(data + data)
        self.assertEqual(version.byte_count, len(data))
        self.assertNotEqual(version.after.size, path.stat().st_size)
        self.assertNotIn("synthetic amber station", repr(outcome))

    def test_file_change_during_capture_is_not_complete(self) -> None:
        path = self.write(b'{"role":"user","content":"synthetic station"}\n')
        real_open = Path.open

        class ChangingReader:
            def __enter__(self) -> ChangingReader:
                self.handle = real_open(path, "rb")
                return self

            def __exit__(self, *args: object) -> None:
                self.handle.close()

            def fileno(self) -> int:
                return self.handle.fileno()

            def read(self) -> bytes:
                data = self.handle.read()
                with real_open(path, "ab") as writer:
                    writer.write(b'{"role":"user","content":"synthetic later row"}\n')
                return data

        with patch.object(Path, "open", return_value=ChangingReader()):
            outcome = self.parser.parse_transcript_outcome(path)
        self.assertFalse(outcome.complete)
        self.assertFalse(outcome.file_version.stable)
        self.assertEqual(len(outcome.messages), 1)
        self.assertIn("source_changed_during_read", outcome.error_categories)

    def test_unreadable_and_invalid_utf8_do_not_leak_diagnostics(self) -> None:
        path = self.write(b"User: synthetic \xff station", "transcript.txt")
        outcome = self.parser.parse_transcript_outcome(path)
        self.assertEqual(outcome.status, "partial")
        self.assertIn("invalid_utf8", outcome.error_categories)
        self.assertEqual(outcome.file_version.sha256, hashlib.sha256(path.read_bytes()).hexdigest())
        with patch.object(Path, "open", side_effect=PermissionError(SYNTHETIC_SECRET)):
            unreadable = self.parser.parse_transcript_outcome(path)
        self.assertEqual(unreadable.status, "unreadable")
        self.assertNotIn(SYNTHETIC_SECRET, repr(unreadable))
        with patch.object(Path, "exists", side_effect=PermissionError(SYNTHETIC_SECRET)):
            denied_probe = self.parser.parse_transcript_outcome(str(path))
        self.assertEqual((denied_probe.status, denied_probe.messages), ("unreadable", []))

    def test_malformed_array_preserves_legacy_salvage_without_complete_claim(self) -> None:
        text = '[{"role":"user","content":"synthetic"}, BROKEN]'
        outcome = self.parser.parse_transcript_outcome(text)
        self.assertEqual(outcome.status, "salvaged")
        self.assertEqual(outcome.malformed_records, 1)
        self.assertEqual(outcome.messages, self.parser.parse_transcript(text))

    def test_jsonl_malformed_first_line_and_wrong_record_types_are_counted(self) -> None:
        path = self.write(b'BROKEN\n{"role":"user","content":"synthetic station"}\nnull\n')
        outcome = self.parser.parse_transcript_outcome(path)
        self.assertEqual((outcome.status, outcome.total_records, outcome.malformed_records), ("partial", 3, 2))
        self.assertEqual(len(outcome.messages), 1)
        for text in ('{"bad":"shape"}', '[17, {"role":"user","content":false}]', '{BROKEN',
                     '{"role":"user","content":[null,17]}'):
            with self.subTest(text=text):
                self.assertEqual(self.parser.parse_transcript_outcome(text).status, "malformed")

    def test_valid_supported_formats_match_legacy_messages(self) -> None:
        entries = [
            {"type": "file-history-snapshot"},
            {"type": "user", "message": {"content": "synthetic human message"}},
            {"type": "assistant", "message": {"content": [
                {"type": "thinking", "thinking": "synthetic excluded"},
                {"type": "text", "text": "synthetic reply"}]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "content": "synthetic tool"}]}},
        ]
        for text in (json.dumps(entries), "\n".join(map(json.dumps, entries)),
                     "User: synthetic station\nAssistant: synthetic reply"):
            with self.subTest(format=text[:1]):
                path = self.write(text.encode(), "supported.txt")
                outcome = self.parser.parse_transcript_outcome(path)
                self.assertTrue(outcome.complete)
                self.assertEqual(outcome.messages, self.parser.parse_transcript(path))

    def test_top_level_legacy_tool_call_retains_its_message(self) -> None:
        text = json.dumps({"type": "tool_use", "name": "synthetic-tool", "input": {"value": "synthetic"}})
        outcome = self.parser.parse_transcript_outcome(text)
        self.assertTrue(outcome.complete)
        self.assertEqual(len(outcome.messages), 1)
        self.assertEqual(outcome.messages, self.parser.parse_transcript(text))
        self.assertEqual(outcome.messages[0].role, "tool_use")

    def test_missing_text_is_malformed_but_recognized_empty_and_thinking_are_complete(self) -> None:
        for block in ({"type": "text"}, {"type": "text", "text": None}):
            with self.subTest(block=block):
                outcome = self.parser.parse_transcript_outcome(json.dumps({"role": "user", "content": [block]}))
                self.assertEqual((outcome.status, outcome.malformed_records), ("malformed", 1))
                self.assertEqual(outcome.messages, [])
        for block in ({"type": "text", "text": ""}, {"type": "thinking", "thinking": "synthetic excluded"}):
            with self.subTest(kind=block["type"]):
                outcome = self.parser.parse_transcript_outcome(json.dumps({"role": "assistant", "content": [block]}))
                self.assertTrue(outcome.complete)
                self.assertEqual(outcome.messages, [])

    def test_unknown_and_nontext_payloads_are_incomplete_without_guessing_text(self) -> None:
        for kind in ("image", "audio", "document", "synthetic-unknown"):
            with self.subTest(kind=kind):
                text = json.dumps({"role": "user", "content": [{"type": kind, "text": "synthetic unrecognized"}]})
                outcome = self.parser.parse_transcript_outcome(text)
                self.assertEqual(outcome.status, "partial")
                self.assertEqual(outcome.unrecognized_records, 1)
                self.assertEqual(outcome.error_categories, ("unrecognized_content",))
                self.assertEqual(outcome.messages, [])
                self.assertEqual(self.parser.parse_transcript(text), [])

    def test_mixed_blocks_retain_supported_text_and_count_record_loss_once(self) -> None:
        entries = [
            {"role": "user", "content": [{"type": "text", "text": "synthetic retained"},
                                          {"type": "text"}, {"type": "image"}, {"type": "unknown"}]},
            {"type": "user", "message": {"content": [{"type": "tool_result", "content": [
                {"type": "text", "text": "synthetic tool result"}, {"type": "text"}, {"type": "image"}]}]}},
        ]
        path = self.write(json.dumps(entries).encode(), "mixed.json")
        outcome = self.parser.parse_transcript_outcome(path)
        self.assertEqual(outcome.status, "partial")
        self.assertEqual((outcome.malformed_records, outcome.unrecognized_records), (2, 2))
        self.assertEqual([message.content for message in outcome.messages], ["synthetic retained", "synthetic tool result"])
        self.assertEqual(outcome.messages, self.parser.parse_transcript(path))
        self.assertEqual(outcome.messages[1].role, "tool_result")
        self.assertEqual(outcome.error_categories, ("malformed_records", "unrecognized_content"))


class TestExtractionOutcomes(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = load_module("extractor")
        self.config = types.SimpleNamespace(provider="synthetic")
        self.addCleanup(logging.disable, logging.NOTSET)
        logging.disable(logging.CRITICAL)

    def run_responses(self, responses: list, *, max_facts: int = 50, max_chunks: int = 20,
                      wrapper: bool = False) -> tuple:
        # One over-budget message stays intact, preserving the production
        # chunker's deliberate no-splitting policy.
        transcript = "\n\n".join("User: synthetic " + str(index) + "x" * 20001
                                   for index in range(len(responses)))
        fn = self.extractor.extract_facts if wrapper else self.extractor.extract_facts_outcome
        with patch.object(self.extractor, "complete", side_effect=responses) as provider:
            result = fn(transcript, self.config, max_facts=max_facts, max_chunks=max_chunks)
        return result, provider.call_count

    def test_valid_empty_responses_and_empty_input(self) -> None:
        for response in ("[]", '{"facts":[]}', '{"items":[]}', "```json\n[]\n```", "Facts: [] done"):
            with self.subTest(response=response):
                outcome, calls = self.run_responses([response])
                self.assertEqual((outcome.status, outcome.facts, calls), ("complete", [], 1))
                self.assertEqual((outcome.attempted_chunks, outcome.successful_chunks), (1, 1))
        with patch.object(self.extractor, "complete") as provider:
            empty = self.extractor.extract_facts_outcome("  ", self.config)
        self.assertTrue(empty.complete)
        self.assertEqual(empty.total_chunks, 0)
        provider.assert_not_called()

    def test_failed_response_is_distinct_from_successful_empty(self) -> None:
        for response in ("garbage", "{}", '{"unexpected":[]}', '[null,17,{}, {"content":" "}]', None):
            with self.subTest(response=response):
                outcome, calls = self.run_responses([response])
                self.assertEqual((outcome.status, calls, outcome.failed_chunks), ("failed", 1, 1))
                self.assertEqual(outcome.invalid_responses, 1)
                self.assertFalse(outcome.complete)
                self.assertTrue(outcome.error_categories)

    def test_mixed_provider_failure_and_valid_empty_is_partial(self) -> None:
        responses = ['[{"content":"synthetic amber"}]', FakeLLMError(SYNTHETIC_SECRET), "[]"]
        outcome, calls = self.run_responses(responses)
        self.assertEqual((outcome.status, calls), ("partial", 3))
        self.assertEqual((outcome.successful_chunks, outcome.partial_chunks, outcome.failed_chunks), (2, 0, 1))
        self.assertEqual([fact.content for fact in outcome.facts], ["synthetic amber"])
        self.assertEqual(outcome.error_categories, ("provider_error",))
        self.assertNotIn(SYNTHETIC_SECRET, repr(outcome))
        self.assertNotIn("synthetic amber", repr(outcome))
        legacy, legacy_calls = self.run_responses(responses, wrapper=True)
        self.assertEqual(legacy, outcome.facts)
        self.assertEqual(legacy_calls, calls)

    def test_all_provider_failures_are_failed_without_retry_or_error_text(self) -> None:
        logging.disable(logging.NOTSET)
        with self.assertLogs(self.extractor.log, level="ERROR") as captured:
            outcome, calls = self.run_responses([RuntimeError(SYNTHETIC_SECRET), FakeLLMError(SYNTHETIC_SECRET)])
        self.assertEqual((outcome.status, calls, outcome.failed_chunks), ("failed", 2, 2))
        self.assertNotIn(SYNTHETIC_SECRET, "\n".join(captured.output) + repr(outcome))
        self.assertEqual(outcome.error_categories, ("provider_exception", "provider_error"))

    def test_mixed_invalid_items_preserve_coercion_and_order(self) -> None:
        response = '[{"content":"synthetic first","category":"INVALID"},17,{}, {"content":42}]'
        outcome, _ = self.run_responses([response])
        self.assertEqual((outcome.status, outcome.invalid_items, outcome.partial_chunks), ("partial", 2, 1))
        self.assertEqual([fact.content for fact in outcome.facts], ["synthetic first", "42"])
        self.assertEqual(outcome.facts[0].category, "general")

    def test_salvage_is_visible_and_legacy_selection_is_preserved(self) -> None:
        response = '[{"content":"synthetic first"}, BROKEN, {"content":"synthetic second"}]'
        outcome, calls = self.run_responses([response], max_facts=1)
        self.assertEqual((outcome.status, calls, outcome.salvaged_responses), ("partial", 1, 1))
        self.assertEqual(outcome.merged_fact_omissions, 1)
        self.assertEqual([fact.content for fact in outcome.facts], ["synthetic first"])
        # The legacy response salvage is uncapped; the extraction merge then
        # enforces max_facts. Do not silently change this compatibility rule.
        self.assertEqual(len(self.extractor._parse_extraction_response(response, 1)), 2)

    def test_per_chunk_cap_counts_valid_omissions_before_slicing(self) -> None:
        response = '[{"content":"synthetic first"}, {}, {"content":"synthetic omitted"},17]'
        outcome, _ = self.run_responses([response], max_facts=1)
        self.assertEqual((outcome.omitted_items, outcome.chunk_fact_omissions), (3, 1))
        self.assertEqual(outcome.invalid_items, 2)
        self.assertFalse(outcome.complete)
        clean, _ = self.run_responses(['[{"content":"one"},{"content":"two"}]'], max_facts=1)
        self.assertEqual((clean.status, clean.chunk_fact_omissions), ("limited", 1))

    def test_merge_dedup_order_and_global_cap_match_legacy(self) -> None:
        responses = ['[{"content":"First"},{"content":"DUP"}]',
                     '[{"content":"dup"},{"content":"Last"}]']
        outcome, calls = self.run_responses(responses, max_facts=2)
        legacy, legacy_calls = self.run_responses(responses, max_facts=2, wrapper=True)
        self.assertEqual([fact.content for fact in outcome.facts], ["First", "DUP"])
        self.assertEqual((outcome.status, outcome.merged_fact_omissions, outcome.chunk_fact_omissions), ("limited", 1, 0))
        self.assertEqual((legacy, legacy_calls), (outcome.facts, calls))

    def test_default_chunk_limit_does_not_spend_tail_calls(self) -> None:
        outcome, calls = self.run_responses(["[]"] * 21)
        self.assertEqual((outcome.total_chunks, outcome.attempted_chunks, outcome.deferred_chunks), (21, 20, 1))
        self.assertEqual((outcome.status, calls), ("limited", 20))
        self.assertEqual(outcome.error_categories, ("chunk_limit",))

    def test_zero_and_negative_controls_preserve_legacy_slicing(self) -> None:
        for max_facts, max_chunks in ((0, 20), (-1, 20), (50, 0), (50, -1)):
            with self.subTest(max_facts=max_facts, max_chunks=max_chunks):
                responses = ['[{"content":"one"},{"content":"two"}]'] * 2
                outcome, calls = self.run_responses(responses, max_facts=max_facts, max_chunks=max_chunks)
                legacy, legacy_calls = self.run_responses(responses, max_facts=max_facts, max_chunks=max_chunks, wrapper=True)
                self.assertEqual((legacy, legacy_calls), (outcome.facts, calls))
                self.assertEqual(outcome.attempted_chunks + outcome.deferred_chunks, outcome.total_chunks)
                self.assertFalse(outcome.complete)

    def test_wrong_object_envelope_is_not_complete_even_when_legacy_finds_array(self) -> None:
        response = '{"unexpected":[{"content":"synthetic compatibility fact"}]}'
        outcome, _ = self.run_responses([response])
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.error_categories, ("response_shape",))
        self.assertEqual(len(outcome.facts), 1)
        self.assertEqual(self.extractor._parse_extraction_response(response, 50), outcome.facts)

    def test_fact_response_records_use_only_public_schema_fields(self) -> None:
        outcome, _ = self.run_responses(['[{"content":"synthetic","category":"PREFERENCE","extra":"ignored"}]'])
        self.assertEqual(asdict(outcome.facts[0]), {"content": "synthetic", "category": "preference",
                                                  "confidence": "medium", "source_role": "user"})


if __name__ == "__main__":
    unittest.main()
