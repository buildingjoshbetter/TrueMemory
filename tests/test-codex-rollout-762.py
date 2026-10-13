"""Synthetic native rollout contracts, without importing the application package."""
from __future__ import annotations

import ast
import builtins
import copy
import errno
import io
import json
from pathlib import Path
import sys
import types
import unittest
from bisect import bisect_left
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SENTINEL = "synthetic-private-content"
SCAFFOLD = "synthetic-scaffold-content"
TURN = "synthetic-turn"


def load_parser(source: str | None = None) -> types.ModuleType:
    module = types.ModuleType("synthetic_codex_parser")
    if source is None:
        source = (ROOT / "truememory/ingest/transcript.py").read_text(encoding="utf-8")

    def safe_import(name: str, globals: dict | None = None, locals: dict | None = None,
                    fromlist: tuple = (), level: int = 0) -> object:
        if name.startswith(("truememory", "torch", "numpy", "sentence_transformers")):
            raise AssertionError("Application imports are forbidden in parser tests")
        return builtins.__import__(name, globals, locals, fromlist, level)

    module.__dict__["__builtins__"] = dict(vars(builtins), __import__=safe_import)
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(compile(source, "synthetic-parser", "exec"), module.__dict__)
    return module


def response(role: str = "user", parts: tuple[str, ...] = (SENTINEL,), *,
             turn: str | None = TURN, phase: str | None = None, item: str = "synthetic-item") -> dict:
    payload = {"type": "message", "role": role, "id": item,
               "content": [{"type": "input_text" if role == "user" else "output_text", "text": text}
                           for text in parts]}
    if turn is not None:
        payload["internal_chat_message_metadata_passthrough"] = {"turn_id": turn}
    if phase is not None:
        payload["phase"] = phase
    return {"type": "response_item", "timestamp": "2026-01-01T00:00:00Z", "payload": payload}


def event(kind: str, **fields: object) -> dict:
    return {"type": "event_msg", "payload": {"type": kind, **fields}}


def start(turn: str = TURN) -> dict:
    return event("task_started", turn_id=turn, collaboration_mode_kind="default")


def terminal(text: object = "done", turn: str = TURN) -> dict:
    return event("task_complete", turn_id=turn, last_agent_message=text)


def completed(role: str, text: str, *, turn: str = TURN,
              item: str = "synthetic-item", phase: str | None = None) -> dict:
    value = {"type": "UserMessage" if role == "user" else "AgentMessage", "id": item,
             "content": [{"type": "text" if role == "user" else "Text", "text": text}]}
    if phase is not None:
        value["phase"] = phase
    return event("item_completed", turn_id=turn, thread_id="synthetic-thread", item=value)


def lifecycle(*, paginated: bool = False, parts: tuple[str, ...] = ("done",),
              phase: str | None = None, turn: str = TURN) -> list[dict]:
    user = response(turn=turn)
    assistant = response("assistant", parts, turn=turn, phase=phase)
    if paginated:
        user_echo = completed("user", SENTINEL, turn=turn)
        agent_echo = completed("assistant", "".join(parts), turn=turn, phase=phase)
    else:
        user_echo = event("user_message", message=SENTINEL, images=[], audio=None)
        agent_echo = event("agent_message", message="".join(parts), phase=phase)
    return [start(turn), user, user_echo, agent_echo, assistant, terminal("".join(parts), turn)]


def annotation(kind: str, **fields: object) -> dict:
    return {"provenance": {"type": kind, **fields}}


def annotate(record: dict, annotations: object = None, kinds: object = None) -> dict:
    metadata = record["payload"].setdefault("internal_chat_message_metadata_passthrough", {})
    metadata.update(content_item_metadata=annotations, content_item_kinds=kinds)
    return record


def load_stop_count(parser: types.ModuleType, text: str, source: str | None = None) -> tuple[object, list[Path]]:
    source = source or (ROOT / "truememory/ingest/hooks/stop.py").read_text(encoding="utf-8")
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "_has_enough_messages")
    calls: list[Path] = []

    def parse(path: Path) -> object:
        if not isinstance(path, Path):
            raise AssertionError("Stop must select the explicit path contract")
        calls.append(path)
        return parser._parse_transcript_text_outcome(text.strip())

    fake = types.SimpleNamespace(parse_transcript_outcome=parse, strip_truememory_blocks=parser.strip_truememory_blocks)

    def safe_import(name: str, globals: dict | None = None, locals: dict | None = None,
                    fromlist: tuple = (), level: int = 0) -> object:
        if name == "truememory.ingest.transcript":
            return fake
        raise AssertionError("Stop count imported an unexpected module")

    namespace = {"Path": Path, "json": json, "__builtins__": dict(vars(builtins), __import__=safe_import)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "synthetic-stop", "exec"), namespace)
    return namespace["_has_enough_messages"], calls


def counted_reconciliation(parser: types.ModuleType, records: list[dict]) -> tuple[object, int]:
    """Count key operations and binary-search reads in the actual outcome path."""
    operations = [0]

    class CountedText(str):
        def __hash__(self) -> int:
            operations[0] += 1
            return super().__hash__()

        def __eq__(self, other: object) -> bool:
            operations[0] += 1
            return super().__eq__(other)

    decode = parser._decode_codex_record

    def counted_decode(entry: dict) -> object:
        result = decode(entry)
        if result is not None:
            for key in ("projection", "phase", "turn_id", "item_id"):
                value = getattr(result, key)
                if isinstance(value, str):
                    setattr(result, key, CountedText(value))
        return result

    def counted_bisect(values: list[int], target: int) -> int:
        class Positions:
            def __len__(self) -> int:
                return len(values)

            def __getitem__(self, index: int) -> int:
                operations[0] += 1
                return values[index]

        return bisect_left(Positions(), target)

    with patch.object(parser, "_decode_codex_record", counted_decode), \
            patch.object(parser, "bisect_left", counted_bisect, create=True):
        result = parser._parse_transcript_text_outcome(json.dumps(records))
    return result, operations[0]


class NativeRolloutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = load_parser()

    def check(self, records: list, expected: list[tuple[str, str]], status: str = "complete",
              malformed: int = 0, unknown: int = 0) -> object:
        for text in (json.dumps(records), "\n".join(json.dumps(record) for record in records)):
            with patch.object(Path, "stat", side_effect=FileNotFoundError(errno.ENOENT, "synthetic missing")):
                messages = self.parser.parse_transcript(text)
                outcome = self.parser.parse_transcript_outcome(text)
            self.assertEqual([(message.role, message.content) for message in messages], expected)
            self.assertEqual(outcome.messages, messages)
            self.assertEqual((outcome.status, outcome.total_records, outcome.malformed_records,
                              outcome.unrecognized_records), (status, len(records), malformed, unknown))
            self.assertNotIn(SENTINEL, repr(outcome))
            self.assertNotIn(TURN, repr(outcome))
            self.assertNotIn(SCAFFOLD, repr(outcome))
        return outcome

    def test_full_legacy_and_paginated_lifecycles(self) -> None:
        for paginated in (False, True):
            with self.subTest(paginated=paginated):
                result = self.check(lifecycle(paginated=paginated), [("human", SENTINEL), ("assistant", "done")])
                self.assertEqual(self.parser.format_for_extraction(result.messages),
                                 "User: " + SENTINEL + "\n\nAssistant: done")

    def test_original_blocks_and_exact_private_projection(self) -> None:
        for parts in (("alpha", "beta"), (" a", "b "), ("", "alpha", "", "beta")):
            for paginated in (False, True):
                with self.subTest(parts=parts, paginated=paginated):
                    self.check(lifecycle(parts=parts, paginated=paginated),
                               [("human", SENTINEL), ("assistant", "\n".join(p for p in parts if p).strip())])
        records = lifecycle(parts=(" a", "b "))
        records[-1] = terminal("ab")
        self.check(records, [("human", SENTINEL), ("assistant", "a\nb")], "partial", unknown=1)
        records[-1] = terminal(" a\nb ")
        self.check(records, [("human", SENTINEL), ("assistant", "a\nb")], "partial", unknown=1)

    def test_each_phase_and_blank_tail_preserve_summary(self) -> None:
        for phase in (None, "commentary", "partial_answer", "final_answer"):
            with self.subTest(phase=phase):
                records = lifecycle(phase=phase)
                records[-1:-1] = [event("agent_message", message=" \n"), response("assistant", (" \n",))]
                self.check(records, [("human", SENTINEL), ("assistant", "done")])

    def test_duplicate_occurrences_and_ids_are_not_deduplicated(self) -> None:
        records = lifecycle()
        records[-1:-1] = [event("agent_message", message="done"), response("assistant", ("done",))]
        self.check(records, [("human", SENTINEL), ("assistant", "done"), ("assistant", "done")])
        self.check([response()] * 5, [("human", SENTINEL)] * 5)

    def test_extra_echo_cannot_borrow_terminal_credit(self) -> None:
        for paginated in (False, True):
            records = lifecycle(paginated=paginated)
            records.insert(-1, copy.deepcopy(records[3]))
            self.check(records, [("human", SENTINEL), ("assistant", "done")], "partial", unknown=1)

    def test_per_original_block_echoes_are_not_ordinary_finalization(self) -> None:
        records = lifecycle(parts=("alpha", "beta"))
        records[3:4] = [event("agent_message", message="alpha"), event("agent_message", message="beta")]
        self.check(records, [("human", SENTINEL), ("assistant", "alpha\nbeta")], "partial", unknown=2)

    def test_missing_echoes_and_monotonic_occurrence_matching(self) -> None:
        records = [start(), response("assistant", ("a",)), response("assistant", ("b",)),
                   event("agent_message", message="b"), terminal("b")]
        self.check(records, [("assistant", "a"), ("assistant", "b")])
        records.insert(-1, event("agent_message", message="a"))
        self.check(records, [("assistant", "a"), ("assistant", "b")], "partial", unknown=1)

    def test_summary_is_not_last_row_or_consumed_echo(self) -> None:
        records = lifecycle(parts=("a",))
        records[-1:-1] = [response("assistant", ("b",)), event("stream_error", message=SENTINEL)]
        self.check(records, [("human", SENTINEL), ("assistant", "a"), ("assistant", "b")])
        records[-1] = terminal(None)
        self.check(records, [("human", SENTINEL), ("assistant", "a"), ("assistant", "b")])

    def test_blank_echo_requires_canonical_occurrence(self) -> None:
        self.check([start(), event("agent_message", message=" \n"), response("assistant", (" \n",)), terminal(None)], [])
        self.check([start(), event("agent_message", message=" \n"), terminal(None)], [], "partial", unknown=1)
        for invalid in ("", " \n", 1, [], {}):
            with self.subTest(summary_type=type(invalid).__name__):
                self.check([start(), terminal(invalid)], [], "partial", unknown=1)

    def test_rust_blank_predicate_is_separate_from_public_strip(self) -> None:
        for nonblank in ("\u001c", "\u001d", "\u001e", "\u001f"):
            self.check([start(), response("assistant", (nonblank,)), terminal(nonblank)], [])
        for blank in ("\u0085", "\u00a0", "\u2003", "\u2028", "\u3000"):
            self.check([start(), response("assistant", (blank,)), terminal(blank)], [], "partial", unknown=1)

    def test_lifecycle_aliases_and_missing_legacy_response_ids(self) -> None:
        records = lifecycle()
        records[0]["payload"]["type"] = "turn_started"
        records[-1]["payload"]["type"] = "turn_complete"
        for index in (1, 4):
            records[index]["payload"].pop("internal_chat_message_metadata_passthrough")
            records[index]["payload"].pop("id")
        self.check(records, [("human", SENTINEL), ("assistant", "done")])

    def test_orphan_terminal_requires_explicit_matching_scope(self) -> None:
        self.check([response("assistant", ("done",)), terminal()], [("assistant", "done")])
        self.check([response("assistant", ("done",), turn=None), terminal()],
                   [("assistant", "done")], "partial", unknown=1)
        self.check([terminal(), response("assistant", ("done",))],
                   [("assistant", "done")], "partial", unknown=1)
        self.check([response("assistant", ("done",)), response("assistant", ("other",), turn="other"), terminal()],
                   [("assistant", "done"), ("assistant", "other")], "partial", unknown=1)

    def test_sessions_and_reused_turn_ids_are_occurrence_scopes(self) -> None:
        self.check(lifecycle() + lifecycle(), [("human", SENTINEL), ("assistant", "done")] * 2)
        self.check(lifecycle() + [{"type": "session_meta", "payload": {}}] + [start(), terminal()],
                   [("human", SENTINEL), ("assistant", "done")], "partial", unknown=1)
        self.check(lifecycle() + [start("second"), terminal("done", "second")],
                   [("human", SENTINEL), ("assistant", "done")], "partial", unknown=1)
        self.check([response("assistant", ("done",), turn=None), {"type": "session_meta", "payload": {}},
                    event("agent_message", message="done")], [("assistant", "done")], "partial", unknown=1)

    def test_conflicting_turn_phase_and_item_identity_stay_partial(self) -> None:
        for field, value in (("turn_id", "other-turn"), ("id", "other-item"), ("phase", "commentary")):
            records = lifecycle(paginated=True, phase="final_answer")
            target = records[3]["payload"] if field == "turn_id" else records[3]["payload"]["item"]
            target[field] = value
            result = self.parser._parse_transcript_text_outcome(json.dumps(records))
            self.assertEqual(result.status, "partial")
            self.assertEqual(len(result.messages), 2)
        records = lifecycle()
        records[4]["payload"]["internal_chat_message_metadata_passthrough"]["turn_id"] = "other-turn"
        self.check(records, [("human", SENTINEL), ("assistant", "done")], "partial", unknown=3)

    def test_duplicate_and_overlapping_terminals_do_not_manufacture_evidence(self) -> None:
        records = lifecycle() + [terminal()]
        self.check(records, [("human", SENTINEL), ("assistant", "done")], "partial", unknown=1)
        records = [start(), response("assistant", ("done",)), start("other"), terminal("done", "other")]
        result = self.parser._parse_transcript_text_outcome(json.dumps(records))
        self.assertEqual(result.status, "partial")
        self.assertIn("codex_unsupported_lifecycle", result.error_categories)

    def test_events_cannot_explain_each_other_or_rewrites(self) -> None:
        self.check([start(), event("agent_message", message="done"), terminal()], [], "partial", unknown=2)
        records = lifecycle()
        records[4] = response("assistant", ("original",))
        self.check(records, [("human", SENTINEL), ("assistant", "original")], "partial", unknown=2)

    def test_markup_and_mode_rewriting_are_explicit_coverage_gaps(self) -> None:
        for parts in (("x<oai-mem-", "citation>hidden</oai-mem-citation>y"),
                      ("before\n<proposed_plan>\nstep\n</proposed_plan>\nafter",)):
            for mode in ("plan", "default", None):
                records = lifecycle(parts=parts)
                records[0]["payload"]["collaboration_mode_kind"] = mode
                result = self.parser._parse_transcript_text_outcome(json.dumps(records))
                self.assertEqual(result.status, "partial")
                self.assertIn("codex_unsupported_projection", result.error_categories)
                self.assertEqual(len(result.messages), 2)

    def test_compaction_does_not_clear_original_evidence_or_emit_copies(self) -> None:
        records = lifecycle()
        records.insert(-1, {"type": "compacted", "payload": {"message": SCAFFOLD,
                               "replacement_history": [response("assistant", (SCAFFOLD,))["payload"]]}})
        self.check(records, [("human", SENTINEL), ("assistant", "done")])
        self.check([start(), records[-2], terminal(None)], [])

    def test_excluded_context_variants_do_not_unwrap_text(self) -> None:
        top = ("session_meta", "turn_context", "compacted", "token_usage_record", "world_state", "retained_context",
               "security_risk_score", "inter_agent_communication", "inter_agent_communication_metadata")
        variants = ("reasoning", "agent_message", "function_call", "function_call_output", "custom_tool_call",
                    "custom_tool_call_output", "compaction", "compaction_summary", "configuration_update")
        records = [{"type": kind, "payload": {"role": "user", "content": SENTINEL}} for kind in top]
        records += [{"type": "response_item", "payload": {"type": kind, "role": "user", "content": SENTINEL}}
                    for kind in variants]
        self.check(records, [])

    def test_outer_flags_are_strict_booleans(self) -> None:
        for key in ("compaction_output", "inherited_user_message"):
            for value in (0, 1, "false", "true", [], {}, None):
                record = response()
                record["metadata"] = {key: value}
                with self.subTest(key=key, value=value):
                    self.check([record], [], "malformed", malformed=1)
            record["metadata"] = {key: True}
            self.check([record], [])
            record["metadata"] = {key: False}
            self.check([record], [("human", SENTINEL)])

    def test_client_authored_does_not_override_scaffold(self) -> None:
        record = annotate(response(), [annotation("harness")])
        record["metadata"] = {"client_authored": True}
        self.check([record], [])

    def test_delivered_markers_exclude_text_and_unavailable_suffix(self) -> None:
        record = response("assistant")
        record["metadata"] = {"delivered_assistant_message": "codex:code-mode-delivery:v1:complete"}
        self.check([record], [])
        record["metadata"]["delivered_assistant_message"] = "codex:code-mode-delivery:v1:incomplete:" + SENTINEL
        result = self.check([record], [], "partial", unknown=1)
        self.assertIn("codex_delivery_unavailable", result.error_categories)
        record["metadata"]["delivered_assistant_message"] = "other"
        self.check([record], [], "partial", unknown=1)
        record["metadata"]["delivered_assistant_message"] = None
        self.check([record], [], "malformed", malformed=1)
        self.check([response("assistant", ("codex:code-mode-delivery:v1:complete",))],
                   [("assistant", "codex:code-mode-delivery:v1:complete")])

    def test_metadata_objects_and_array_alignment(self) -> None:
        for bad in (1, "bad", [], False):
            for outer in (True, False):
                record = response()
                if outer:
                    record["metadata"] = bad
                else:
                    record["payload"]["internal_chat_message_metadata_passthrough"] = bad
                self.check([record], [], "malformed", malformed=1)
        for key, bad in (("content_item_metadata", []), ("content_item_metadata", [None]),
                         ("content_item_metadata", [{}, {}]), ("content_item_metadata", {}),
                         ("content_item_kinds", [None]), ("content_item_kinds", []), ("content_item_kinds", "unknown")):
            record = response()
            record["payload"]["internal_chat_message_metadata_passthrough"][key] = bad
            self.check([record], [], "malformed", malformed=1)
        self.check([annotate(response(), None, None)], [("human", SENTINEL)])

    def test_per_index_provenance_does_not_shift_after_media(self) -> None:
        record = response(parts=(SCAFFOLD, SENTINEL, SCAFFOLD))
        record["payload"]["content"][0] = {"type": "input_image", "image_url": "synthetic-image"}
        annotate(record, [annotation("harness"), annotation("user"), annotation("agents_md")])
        result = self.check([record], [("human", SENTINEL)])
        self.assertNotIn(SCAFFOLD, self.parser.format_for_extraction(result.messages))
        record["payload"]["internal_chat_message_metadata_passthrough"]["content_item_metadata"][0] = annotation("user")
        self.check([record], [("human", SENTINEL)], "partial", unknown=1)

    def test_provenance_and_legacy_fields(self) -> None:
        exclusions = [annotation("developer_instructions", from_additional_requirements=False), annotation("skills"),
                      annotation("agents_md"), annotation("harness", hook=True), annotation("tool", namespace="functions"),
                      {"harness_injected": True}, {"source_tool_namespace": "functions", "harness_injected": False}]
        for attribution in exclusions:
            self.check([annotate(response(), [attribution])], [])
        for attribution in ({}, {"harness_injected": False}, {"source_tool_namespace": None}, annotation("user")):
            self.check([annotate(response(), [attribution])], [("human", SENTINEL)])
        for kind in ("unknown", "user_configured", "future"):
            self.check([annotate(response(), [annotation(kind)])], [], "partial", unknown=1)

    def test_nested_metadata_types_never_fall_back_to_user(self) -> None:
        values = [{"provenance": None}, {"provenance": "user"}, {"provenance": {"type": []}},
                  annotation("harness", hook=1), annotation("developer_instructions"),
                  annotation("developer_instructions", from_additional_requirements="false"),
                  annotation("tool", namespace=1), {"harness_injected": 1}, {"source_tool_namespace": []}]
        for attribution in values:
            self.check([annotate(response(), [attribution])], [], "malformed", malformed=1)

    def test_kinds_fallback_conflict_and_lookalike_prose(self) -> None:
        for kind in ("agents_md.instructions", "hooks.additional_context", "guardian.retained_instructions"):
            self.check([annotate(response(), None, [kind])], [])
            self.check([annotate(response(), [annotation("user")], [kind])], [], "partial", unknown=1)
        self.check([annotate(response(), None, ["user.future"])], [], "partial", unknown=1)
        self.check([annotate(response(), [annotation("harness")], ["future.fragment"])], [])
        self.check([response(parts=("# AGENTS.md ordinary quoted text",))], [("human", "# AGENTS.md ordinary quoted text")])
        for kind in ("", "unknown"):
            self.check([annotate(response(), None, [kind])], [("human", SENTINEL)])

    def test_excluded_equal_text_cannot_supply_echo_or_terminal_credit(self) -> None:
        excluded = [response("assistant", ("done",)) for _ in range(5)]
        excluded[0]["metadata"] = {"inherited_user_message": True}
        excluded[1]["metadata"] = {"compaction_output": True}
        excluded[2]["metadata"] = {"delivered_assistant_message": "codex:code-mode-delivery:v1:complete"}
        annotate(excluded[3], [annotation("harness")])
        annotate(excluded[4], [annotation("unknown")])
        for index, record in enumerate(excluded):
            self.check([start(), event("agent_message", message="done"), record, terminal()], [],
                       "partial", unknown=3 if index == 4 else 2)

    def test_record_level_counts_and_private_diagnostics(self) -> None:
        record = response()
        record["payload"]["content"] = [{"type": "future", "text": SENTINEL}, {"type": "input_image"},
                                       {"type": "input_text", "text": 3}, 5,
                                       {"type": "input_text", "text": SENTINEL}]
        record["timestamp"] = {SENTINEL: "bad"}
        result = self.check([record], [("human", SENTINEL)], "partial", malformed=1, unknown=1)
        self.assertEqual(result.messages[0].timestamp, "")
        self.assertEqual(result.total_records, 1)

    def test_unknown_native_and_events_are_not_empty_complete(self) -> None:
        for record in ({"type": "future", "payload": {"content": SENTINEL}},
                       {"type": "realtime_item", "payload": {"text": SENTINEL}},
                       {"type": "response_item", "payload": {"type": "future", "content": SENTINEL}},
                       event("future", text=SENTINEL), event("user_message", message=SENTINEL),
                       event("item_completed", item={"type": "Future", "text": SENTINEL})):
            self.check([record], [], "partial", unknown=1)
        for record in ({"type": "response_item", "payload": []},
                       {"type": "response_item", "payload": {"type": "message", "role": "user", "content": SENTINEL}},
                       {"type": "event_msg", "payload": {"type": []}}):
            self.check([record], [], "malformed", malformed=1)
        for invalid in ([], {}, 1, None):
            self.check([{"type": invalid, "payload": {}}], [], "malformed", malformed=1)

    def test_user_output_text_is_not_a_normal_user_projection(self) -> None:
        record = response()
        record["payload"]["content"][0]["type"] = "output_text"
        self.check([record], [("human", SENTINEL)])
        self.check([record, event("user_message", message=SENTINEL)], [("human", SENTINEL)], "partial", unknown=1)

    def test_event_media_and_unknown_typed_items_stay_partial(self) -> None:
        for key in ("images", "audio", "local_images", "file_ids"):
            records = lifecycle()
            records[2]["payload"][key] = [SENTINEL]
            self.check(records, [("human", SENTINEL), ("assistant", "done")], "partial", unknown=1)
        records = lifecycle(paginated=True)
        records[3]["payload"]["item"]["content"].append({"type": "Text", "text": "extra"})
        self.check(records, [("human", SENTINEL), ("assistant", "done")], "partial", unknown=1)

    def test_claude_and_generic_compatibility(self) -> None:
        records = [{"type": "user", "message": {"role": "user", "content": "hello"}},
                   {"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": SCAFFOLD},
                                                                 {"type": "text", "text": "answer"}]}},
                   {"type": "user", "message": {"content": [{"type": "tool_result", "content": "tool"}]}},
                   {"role": "user", "content": "generic"}, {"type": "progress"}]
        result = self.check(records, [("human", "hello"), ("assistant", "answer"), ("tool_result", "tool"), ("human", "generic")])
        self.assertEqual(self.parser.format_for_extraction(result.messages), "User: hello\n\nAssistant: answer\n\nUser: generic")

    def test_generic_records_keep_incidental_payload_fields(self) -> None:
        for payload in (None, {}, {"type": "message", "role": "assistant", "content": SCAFFOLD}):
            for type_fields in ({}, {"type": None}, {"type": ""}):
                records = [{"role": role, "content": text, "payload": payload, **type_fields}
                           for role, text in (("user", SENTINEL), ("assistant", "answer"))]
                self.check(records, [("human", SENTINEL), ("assistant", "answer")])
        self.check([{"type": "future_native", "payload": {}, "role": "user", "content": SENTINEL}],
                   [], "partial", unknown=1)
        self.check([{"role": "observer", "content": SENTINEL, "payload": None}], [("observer", SENTINEL)])

    def test_index_preserves_absent_and_explicit_identity_compatibility(self) -> None:
        first = response("assistant", ("same",), phase="commentary")
        first["payload"].pop("id")
        second = response("assistant", ("same",), phase="final_answer", item="second")
        third = response("assistant", ("same",), phase="final_answer", item="third")
        records = [start(), first, second, third,
                   completed("assistant", "same", item="second", phase="commentary"),
                   event("agent_message", message="same", phase="final_answer"),
                   completed("assistant", "same", item="third"), terminal("same")]
        self.check(records, [("assistant", "same")] * 3)
        records.insert(-1, completed("assistant", "same", item="third"))
        self.check(records, [("assistant", "same")] * 3, "partial", unknown=1)

    def test_legacy_exclusions_keep_incidental_payload_fields(self) -> None:
        for kind in sorted(self.parser._NON_CONVERSATION_TYPES):
            for selector in ({"type": kind}, {"role": kind}, {"type": None, "role": kind}):
                for extra in ({}, {"payload": None}, {"payload": {}},
                              {"payload": {"type": "message", "role": "assistant", "content": SCAFFOLD}}):
                    with self.subTest(kind=kind, selector=selector, payload_present="payload" in extra):
                        self.check([{**selector, **extra}], [])
        self.check([{"type": "future_native", "role": "summary", "payload": {}}],
                   [], "partial", unknown=1)
        self.check([{"type": "response_item", "role": "summary", "payload": {}}],
                   [], "malformed", malformed=1)

    def test_identity_filters_do_not_rewind_role_cursors(self) -> None:
        records = [response("assistant", ("same",), turn="first", item="first"),
                   response(parts=("same",), turn="second"),
                   response("assistant", ("same",), turn="second", item="second"),
                   completed("assistant", "same", turn="second", item="second"),
                   event("user_message", message="same"),
                   completed("assistant", "same", turn="first", item="first")]
        self.check(records, [("assistant", "same"), ("human", "same"), ("assistant", "same")], "partial", unknown=1)

    def test_unmatched_echo_work_is_bounded_including_metadata_misses(self) -> None:
        for mismatch in ("text", "phase", "item", "turn"):
            previous = None
            for size in (100, 200, 400):
                with self.subTest(mismatch=mismatch, size=size):
                    originals = [response("assistant", ("same",), phase="commentary") for _ in range(size)]
                    if mismatch == "text":
                        echo = event("agent_message", message="different")
                    elif mismatch == "phase":
                        echo = event("agent_message", message="same", phase="final_answer")
                    elif mismatch == "item":
                        echo = completed("assistant", "same", item="different")
                    else:
                        echo = completed("assistant", "same", turn="different")
                    result, operations = counted_reconciliation(self.parser, originals + [echo] * size)
                    self.assertEqual((len(result.messages), result.unrecognized_records), (size, size))
                    self.assertEqual(result.status, "partial")
                    # Eight entries per response; at most two indexed event
                    # queries. Allow linear key work plus binary-search reads.
                    self.assertLessEqual(operations, 96 * size + 4 * size * (size.bit_length() + 1))
                    if previous is not None:
                        self.assertLessEqual(operations, 3 * previous + 32)
                    previous = operations

    def test_syntax_accounting_is_not_replaced_by_native_counts(self) -> None:
        text = json.dumps(response()) + "\n{bad\n" + json.dumps(event("future"))
        result = self.parser._parse_transcript_text_outcome(text)
        self.assertEqual((result.status, result.total_records, result.malformed_records, result.unrecognized_records),
                         ("partial", 3, 1, 1))
        self.assertEqual(self.parser._parse_transcript_text_outcome("[bad").status, "salvaged")


class StopCountTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = load_parser()

    def check(self, text: str, expected: bool, threshold: int = 5) -> None:
        count, calls = load_stop_count(self.parser, text)
        with patch.object(Path, "read_text", side_effect=AssertionError("Stop must not read directly")), \
                patch.object(Path, "stat", side_effect=AssertionError("Stop must not stat directly")):
            self.assertIs(count("synthetic-transcript.jsonl", threshold), expected)
        self.assertEqual(calls, [Path("synthetic-transcript.jsonl")])

    def test_four_original_native_failures(self) -> None:
        five = [response()] * 5
        self.check(json.dumps(five), True)
        self.check("\n".join(json.dumps(item) for item in five), True)
        self.check(json.dumps(response(parts=(SENTINEL * 40,))), False)
        self.check(json.dumps({"type": "session_meta", "payload": {"metadata": SENTINEL * 40}}), False)

    def test_events_and_scaffolding_do_not_inflate_turns(self) -> None:
        self.check(json.dumps([response()] + [event("user_message", message=SENTINEL)] * 5), False)
        self.check(json.dumps([annotate(response(), [annotation("harness")])] * 5), False)
        self.check(json.dumps([response(parts=("<truememory-recall>synthetic</truememory-recall>",))] * 5), False)
        self.check(json.dumps([response(parts=(" \n",))] * 5), False)
        self.check(json.dumps([response(parts=("one", "two", "three", "four", "five"))]), False)

    def test_actual_threshold_duplicates_and_partial_supported_turns(self) -> None:
        self.check(json.dumps([response()] * 4), False)
        self.check(json.dumps([response()] * 5), True)
        self.check(json.dumps([response()] * 5 + [event("future")]), True)
        self.check(json.dumps([response()] * 5), False, threshold=6)
        self.check(json.dumps(lifecycle(paginated=True)), True, threshold=1)

    def test_claude_tools_plain_text_and_directives(self) -> None:
        self.check(json.dumps([{"type": "user", "message": {"content": [{"type": "tool_result", "content": "result"}]}}] * 5), False)
        self.check("Human: one\nAssistant: reply\nUser: two", True, threshold=2)
        self.check("ordinary unmarked text " * 100, False)
        self.check(json.dumps([response(parts=("Use the synthetic short format",))] * 5), True)

    def test_unreadable_is_false_even_at_zero_threshold(self) -> None:
        with patch.object(self.parser, "_parse_transcript_text_outcome", return_value=self.parser.TranscriptOutcome([], "unreadable")):
            self.check("", False, threshold=0)


class InMemoryPathTests(unittest.TestCase):
    def test_native_explicit_file_path_preserves_version_and_privacy(self) -> None:
        parser = load_parser()
        data = ("\n".join(json.dumps(record) for record in lifecycle())).encode("utf-8")
        info = types.SimpleNamespace(st_dev=1, st_ino=2, st_size=len(data), st_mtime_ns=3, st_ctime_ns=4)

        class Reader(io.BytesIO):
            def fileno(self) -> int:
                return 41

        with patch.object(Path, "open", side_effect=lambda *args, **kwargs: Reader(data)), \
                patch.object(parser.os, "fstat", return_value=info), \
                patch.object(Path, "stat", side_effect=AssertionError("Explicit path must not probe")):
            result = parser.parse_transcript_outcome(Path("synthetic-private-path.jsonl"))
        self.assertTrue(result.complete)
        self.assertTrue(result.file_version.stable)
        self.assertEqual(result.total_records, 6)
        self.assertEqual(len(result.messages), 2)
        self.assertNotIn("synthetic-private-path", repr(result))


class MutationControlTests(unittest.TestCase):
    """The witnesses must reject the specific earlier unsafe implementations."""

    def mutate(self, before: str, after: str) -> types.ModuleType:
        source = (ROOT / "truememory/ingest/transcript.py").read_text(encoding="utf-8")
        self.assertEqual(source.count(before), 1)
        return load_parser(source.replace(before, after))

    def outcome(self, parser: types.ModuleType, records: list[dict]) -> object:
        return parser._parse_transcript_text_outcome(json.dumps(records))

    def test_newline_or_trimmed_projection_breaks_ordinary_lifecycle(self) -> None:
        for replacement in ('raw = "\\n".join(parts)', 'raw = "".join(parts).strip()'):
            with self.subTest(replacement=replacement):
                parser = self.mutate('raw = "".join(parts)', replacement)
                self.assertEqual(self.outcome(parser, lifecycle(parts=(" a", "b "))).status, "partial")

    def test_removing_provenance_filter_leaks_scaffold(self) -> None:
        parser = self.mutate('if attribution == "excluded":', 'if False:')
        records = [annotate(response(parts=(SCAFFOLD,)), [annotation("harness")])]
        self.assertEqual([message.content for message in self.outcome(parser, records).messages], [SCAFFOLD])
        self.assertEqual(self.outcome(load_parser(), records).messages, [])

    def test_ignoring_event_reconciliation_falsely_certifies_event_only(self) -> None:
        parser = self.mutate('_reconcile_codex(native_records)', 'pass  # synthetic mutation')
        records = [start(), event("agent_message", message=SENTINEL), terminal(SENTINEL)]
        self.assertTrue(self.outcome(parser, records).complete)
        self.assertEqual(self.outcome(load_parser(), records).unrecognized_records, 2)

    def test_last_raw_row_only_rule_loses_valid_terminal(self) -> None:
        parser = self.mutate('for before, candidate in reversed(candidates)',
                             'for before, candidate in candidates[-1:]')
        records = lifecycle(parts=("a",))
        records.insert(-1, response("assistant", ("b",)))
        self.assertEqual(self.outcome(parser, records).status, "partial")
        self.assertTrue(self.outcome(load_parser(), records).complete)

    def test_echoes_must_not_match_only_preceding_responses(self) -> None:
        source = (ROOT / "truememory/ingest/transcript.py").read_text(encoding="utf-8")
        source = source.replace('for _, event in items:', 'for event_index, event in items:')
        target = 'position = _codex_next_echo(credit_index, event, cursors.get(event.role, 0))'
        self.assertEqual(source.count(target), 1)
        source = source.replace(target, target + '\n            if position is not None and credits[position][0] >= event_index:\n                position = None')
        parser = load_parser(source)
        self.assertEqual(self.outcome(parser, lifecycle()).status, "partial")
        self.assertTrue(self.outcome(load_parser(), lifecycle()).complete)

    def test_terminal_must_not_require_final_phase(self) -> None:
        parser = self.mutate('candidate.projection == terminal.projection and not candidate.malformed',
                             'candidate.phase == "final_answer" and candidate.projection == terminal.projection and not candidate.malformed')
        self.assertEqual(self.outcome(parser, lifecycle(phase="commentary")).status, "partial")
        self.assertTrue(self.outcome(load_parser(), lifecycle(phase="commentary")).complete)

    def test_session_reset_is_necessary_even_without_turn_boundaries(self) -> None:
        parser = self.mutate('if record.kind == "session_meta":', 'if False:')
        records = [response("assistant", ("done",), turn=None), {"type": "session_meta", "payload": {}},
                   event("agent_message", message="done")]
        self.assertTrue(self.outcome(parser, records).complete)
        self.assertEqual(self.outcome(load_parser(), records).status, "partial")

    def test_terminal_and_echo_credit_must_remain_independent(self) -> None:
        parser = self.mutate('cursors[event.role] = position + 1',
                             'cursors[event.role] = position + 1\n                credits[position][1].projection = None')
        self.assertEqual(self.outcome(parser, lifecycle()).status, "partial")
        self.assertTrue(self.outcome(load_parser(), lifecycle()).complete)

    def test_echo_credit_must_be_consumed_once(self) -> None:
        parser = self.mutate('cursors[event.role] = position + 1', 'cursors[event.role] = 0')
        records = lifecycle()
        records.insert(-1, event("agent_message", message="done"))
        self.assertTrue(self.outcome(parser, records).complete)
        self.assertEqual(self.outcome(load_parser(), records).status, "partial")


if __name__ == "__main__":
    unittest.main()
