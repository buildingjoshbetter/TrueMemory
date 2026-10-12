"""Inert synthetic controls for the pinned private Codex projection."""
from __future__ import annotations

import builtins
import copy
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
TURN = "synthetic-projection-turn"


def load_parser() -> types.ModuleType:
    module = types.ModuleType("synthetic_codex_projection_parser")

    def safe_import(name: str, globals: dict | None = None, locals: dict | None = None,
                    fromlist: tuple = (), level: int = 0) -> object:
        if name.startswith(("truememory", "torch", "numpy", "sentence_transformers")):
            raise AssertionError("Application imports are forbidden")
        return builtins.__import__(name, globals, locals, fromlist, level)

    module.__dict__["__builtins__"] = dict(vars(builtins), __import__=safe_import)
    source = (ROOT / "truememory/ingest/transcript.py").read_text(encoding="utf-8")
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(compile(source, "synthetic-projection-parser", "exec"), module.__dict__)
    return module


def event(kind: str, **fields: object) -> dict:
    return {"type": "event_msg", "payload": {"type": kind, **fields}}


def response(parts: tuple[str, ...], phase: str | None = None) -> dict:
    payload = {"type": "message", "role": "assistant", "id": "synthetic-item",
               "content": [{"type": "output_text", "text": text} for text in parts],
               "internal_chat_message_metadata_passthrough": {"turn_id": TURN}}
    if phase is not None:
        payload["phase"] = phase
    return {"type": "response_item", "timestamp": "2020-01-01T00:00:00Z", "payload": payload}


def context(mode: str = "plan", turn: str = TURN, **settings: object) -> dict:
    return {"type": "turn_context", "payload": {"turn_id": turn,
            "collaboration_mode": {"mode": mode, "settings": {"model": "synthetic-model", **settings}}}}


def plan_event(text: str, **fields: object) -> dict:
    return event("item_completed", turn_id=TURN, thread_id="synthetic-thread",
                 item={"type": "Plan", "id": TURN + "-plan", "text": text}, **fields)


def plan_start(timestamp: object = -1) -> dict:
    return event("item_started", turn_id=TURN, thread_id="synthetic-thread", started_at_ms=timestamp,
                 item={"type": "Plan", "id": TURN + "-plan", "text": ""})


def lifecycle(parts: tuple[str, ...], visible: str | None, *, mode: object = "plan",
              paginated: bool = False, phase: str | None = None) -> list[dict]:
    fields: dict = {"turn_id": TURN}
    if mode is not None:
        fields["collaboration_mode_kind"] = mode
    values = [event("task_started", **fields)]
    if visible is not None:
        if paginated:
            item = {"type": "AgentMessage", "id": "synthetic-item",
                    "content": [{"type": "Text", "text": visible}]}
            if phase is not None:
                item["phase"] = phase
            values.append(event("item_completed", turn_id=TURN,
                                thread_id="synthetic-thread", item=item))
        else:
            values.append(event("agent_message", message=visible, phase=phase))
    values.extend([response(parts, phase),
                   event("task_complete", turn_id=TURN, last_agent_message=visible)])
    return values


class ProjectionGrammarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = load_parser()

    def outcome(self, records: list[dict]) -> object:
        text = "\n".join(json.dumps(value) for value in records)
        with patch.object(self.parser, "_transcript_path", return_value=None):
            result = self.parser.parse_transcript_outcome(text)
            messages = self.parser.parse_transcript(text)
        self.assertEqual(messages, result.messages)
        return result

    def check_complete(self, parts: tuple[str, ...], visible: str | None, **fields: object) -> None:
        result = self.outcome(lifecycle(parts, visible, **fields))
        self.assertEqual(result.status, "complete", result.error_categories)
        self.assertEqual(result.error_categories, ())
        public = "\n".join(value for value in parts if value).strip()
        self.assertEqual([value.content for value in result.messages], [public] if public else [])
        if result.messages:
            self.assertEqual(result.messages[0].timestamp, "2020-01-01T00:00:00Z")

    def test_citation_boundaries_cross_blocks(self) -> None:
        parts = (" x<oai-mem-", "citation>hidden</oai-mem-", "citation>y ")
        for paginated in (False, True):
            self.check_complete(parts, " xy ", mode="default", paginated=paginated)

    def test_citation_literal_nonnested_first_close(self) -> None:
        self.check_complete(("a<oai-mem-citation>x<oai-mem-citation>y</oai-mem-citation>z</oai-mem-citation>",),
                            "az</oai-mem-citation>", mode="default")

    def test_open_citation_hides_through_eof(self) -> None:
        self.check_complete(("visible<oai-mem-citation>hidden</oai-mem-cit",), "visible")

    def test_partial_open_and_stray_close_stay_visible(self) -> None:
        text = "<oai-mem-cit </oai-mem-citation>"
        self.check_complete((text,), text, mode="default")

    def test_citation_only_has_no_invented_agent(self) -> None:
        self.check_complete(("<oai-mem-citation>hidden</oai-mem-citation>",), None)

    def test_multiple_citations_preserve_surrounding_text(self) -> None:
        self.check_complete(("a<oai-mem-citation>1</oai-mem-citation>b<oai-mem-citation>2</oai-mem-citation>c",),
                            "abc", mode=None)

    def test_whole_line_plan_known_mode(self) -> None:
        parts = ("before\n<proposed_", "plan>\nstep\n</proposed_plan>\nafter")
        for paginated in (False, True):
            self.check_complete(parts, "before\nafter", paginated=paginated)

    def test_plan_only_and_open_at_eof(self) -> None:
        self.check_complete(("<proposed_plan>\nstep\n",), None)

    def test_plan_crlf_and_surviving_endings_exact(self) -> None:
        self.check_complete(("before\r\n \t<proposed_plan>\r\nstep\r\n</proposed_plan>\r\nafter\r\n",),
                            "before\r\nafter\r\n")

    def test_plan_rust_white_space(self) -> None:
        for marker in ("\u0085", "\u00a0", "\u2007", "\u2028", "\u3000"):
            with self.subTest(marker=repr(marker)):
                text = f"before\n{marker}<proposed_plan>{marker}\nstep\n{marker}</proposed_plan>{marker}\nafter"
                self.check_complete((text,), "before\nafter")

    def test_python_extra_whitespace_is_literal(self) -> None:
        for marker in ("\x1c", "\x1d", "\x1e", "\x1f"):
            text = f"{marker}<proposed_plan>{marker}"
            self.check_complete((text,), text, mode="default")

    def test_inline_and_partial_plan_tags_visible(self) -> None:
        for text in ("x<proposed_plan>y</proposed_plan>z", "<proposed_", "<proposed_plan> extra"):
            self.check_complete((text,), text, mode="default")

    def test_nested_plan_open_is_literal_hidden_body(self) -> None:
        self.check_complete(("before\n<proposed_plan>\n<proposed_plan>\nstep\n</proposed_plan>\nafter\n</proposed_plan>",),
                            "before\nafter\n</proposed_plan>")

    def test_multiple_plan_blocks(self) -> None:
        self.check_complete(("a\n<proposed_plan>\n1\n</proposed_plan>\nb\n<proposed_plan>\n2\n</proposed_plan>\nc",),
                            "a\nb\nc")

    def test_citation_then_plan_order(self) -> None:
        self.check_complete(("a\n<oai-mem-citation>hidden</oai-mem-citation><proposed_plan>\nstep\n</proposed_plan>\nb",),
                            "a\nb")
        self.check_complete(("a<oai-mem-citation>\n<proposed_plan>\nstep\n</proposed_plan>\n</oai-mem-citation>b",),
                            "ab", mode=None)

    def test_default_unknown_plan_scope_remains_partial(self) -> None:
        text = "before\n<proposed_plan>\nstep\n</proposed_plan>\nafter"
        for mode in ("default", None, "unknown", True):
            for echo in (text, "before\nafter"):
                result = self.outcome(lifecycle((text,), echo, mode=mode))
                self.assertEqual(result.status, "partial")
                self.assertIn("codex_unsupported_projection", result.error_categories)
                self.assertEqual(result.messages[0].content, text)

    def test_mode_does_not_leak_across_lifecycles(self) -> None:
        text = "before\n<proposed_plan>\nstep\n</proposed_plan>\nafter"
        records = lifecycle(("plain",), "plain", mode="plan")
        records.extend(lifecycle((text,), text, mode=None))
        result = self.outcome(records)
        self.assertEqual(result.status, "partial")
        self.assertIn("codex_unsupported_projection", result.error_categories)

    def test_wrong_echo_and_terminal_never_fit_alternatives(self) -> None:
        text = "a<oai-mem-citation>hidden</oai-mem-citation>b"
        result = self.outcome(lifecycle((text,), text))
        self.assertEqual(result.status, "partial")
        self.assertIn("codex_unmatched_conversation_event", result.error_categories)
        self.assertIn("codex_unmatched_terminal", result.error_categories)

    def test_terminal_all_phases_use_same_exact_projection(self) -> None:
        for phase in (None, "commentary", "partial_answer", "final_answer"):
            self.check_complete(("a<oai-mem-citation>hidden</oai-mem-citation>b",), "ab", phase=phase)

    def test_unknown_tags_not_stripped(self) -> None:
        text = "a<unknown_tag>hidden</unknown_tag>b"
        self.check_complete((text,), text)

    def test_unexplained_structured_channels_remain_partial(self) -> None:
        records = lifecycle(("a<oai-mem-citation>hidden</oai-mem-citation>b",), "ab")
        records[1]["payload"]["memory_citation"] = {"entries": [], "rolloutIds": ["synthetic"]}
        self.assertEqual(self.outcome(records).status, "partial")
        records = lifecycle(("<proposed_plan>\nstep\n</proposed_plan>",), None)
        records.insert(1, event("item_completed", turn_id=TURN, thread_id="synthetic-thread",
                                item={"type": "Plan", "id": TURN + "-plan", "text": "wrong"}))
        self.assertEqual(self.outcome(records).status, "partial")

    def test_exact_citation_object_legacy_and_paginated(self) -> None:
        body = "<citation_entries>\nsynthetic/path:+9-+2|note=[ exact ]\r\n</citation_entries><rollout_ids>\na\nb\na\n</rollout_ids>"
        citation = {"entries": [{"path": "synthetic/path", "lineStart": 9, "lineEnd": 2, "note": "exact"}],
                    "rolloutIds": ["a", "b"]}
        parts = ("a<oai-mem-", "citation>" + body + "</oai-mem-citation>b")
        for paginated in (False, True):
            records = lifecycle(parts, "ab", paginated=paginated)
            target = records[1]["payload"]["item"] if paginated else records[1]["payload"]
            target["memory_citation"] = citation
            result = self.outcome(records)
            self.assertEqual(result.status, "complete", result.error_categories)
            self.assertEqual(result.messages[0].content, "\n".join(parts))

    def test_citation_opaque_values_order_and_first_complete_blocks(self) -> None:
        body1 = "<citation_entries>p:00000000000001-4294967295|note=[n]</citation_entries><citation_entries>wrong:1-2|note=[x]</citation_entries><rollout_ids>x\r\ny\u2028z\n</rollout_ids>"
        body2 = "<citation_entries>bad:-1-2|note=[x]\np:4294967296-1|note=[x]\np:1_0-2|note=[x]\np:١-2|note=[x]\np:1-2|note=[n]</citation_entries><rollout_ids>y\u2028z\nx</rollout_ids><thread_ids>unused</thread_ids>"
        fingerprint = self.parser._codex_derive_citation([body1, body2])
        self.assertEqual(fingerprint, ((("p", 1, 4294967295, "n"), ("p", 1, 2, "n")), ("x", "y\u2028z")))
        self.assertEqual(self.parser._codex_u32("+" + "0" * 5000 + "1"), 1)

    def test_citation_id_fallback_and_rust_trim(self) -> None:
        body = "<rollout_ids>incomplete<thread_ids> \u3000x\u3000 \ny\n</thread_ids>"
        self.assertEqual(self.parser._codex_derive_citation([body]), ((), ("x", "y")))
        self.assertEqual(self.parser._codex_derive_citation(["<rollout_ids></rollout_ids><thread_ids>x</thread_ids>"]), None)
        self.assertEqual(self.parser._codex_derive_citation(["<thread_ids>\x1cx\x1c</thread_ids>"]), ((), ("\x1cx\x1c",)))

    def test_supplied_citation_shapes_and_mismatch_reject(self) -> None:
        valid = {"entries": [{"path": "p", "lineStart": 1, "lineEnd": 2, "note": "n"}], "rolloutIds": []}
        body = "<citation_entries>p:1-2|note=[n]</citation_entries>"
        invalid = [[], {}, {"entries": []}, {"entries": [], "rolloutIds": [], "extra": True},
                   {"entries": [], "rolloutIds": []}, {"entries": [], "rolloutIds": [True]}]
        for key, value in (("lineStart", True), ("lineEnd", -1), ("lineEnd", 4294967296), ("note", "wrong")):
            item = copy.deepcopy(valid)
            item["entries"][0][key] = value
            invalid.append(item)
        for citation in invalid:
            records = lifecycle(("a<oai-mem-citation>" + body + "</oai-mem-citation>b",), "ab")
            records[1]["payload"]["memory_citation"] = citation
            result = self.outcome(records)
            self.assertEqual(result.status, "partial", citation)
            self.assertIn("codex_unsupported_projection", result.error_categories)

    def test_citation_absent_null_is_optional_compatibility(self) -> None:
        text = "a<oai-mem-citation><thread_ids>synthetic-id</thread_ids></oai-mem-citation>b"
        for supplied in (False, True):
            records = lifecycle((text,), "ab")
            if supplied:
                records[1]["payload"]["memory_citation"] = None
            self.assertEqual(self.outcome(records).status, "complete")
        records = lifecycle(("plain",), "plain")
        records[1]["payload"]["memory_citation"] = {"entries": [], "rolloutIds": []}
        self.assertEqual(self.outcome(records).status, "partial")

    def test_equal_visible_citation_credits_are_metadata_selective(self) -> None:
        records = [event("task_started", turn_id=TURN, collaboration_mode_kind="plan")]
        for value in ("a", "b"):
            records.append(response(("same<oai-mem-citation><thread_ids>" + value + "</thread_ids></oai-mem-citation>",)))
        for value in ("b", "a"):
            records.append(event("agent_message", message="same", memory_citation={"entries": [], "rolloutIds": [value]}))
        result = self.outcome(records)
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.unrecognized_records, 1)

    def test_context_plan_proof_precedes_supported_candidates(self) -> None:
        text = "before\n<proposed_plan>\nstep\n</proposed_plan>\nafter"
        records = lifecycle((text,), "before\nafter", mode=None)
        records.insert(1, context(reasoning_effort="synthetic-custom", developer_instructions=None))
        self.assertEqual(self.outcome(records).status, "complete")

    def test_context_future_previous_and_foreign_turn_not_borrowed(self) -> None:
        text = "before\n<proposed_plan>\nstep\n</proposed_plan>\nafter"
        for place, value in ((-1, context()), (0, context()), (1, context(turn="other"))):
            records = lifecycle((text,), "before\nafter", mode=None)
            records.insert(place, value)
            result = self.outcome(records)
            self.assertEqual(result.status, "partial")
            self.assertIn("codex_unsupported_projection", result.error_categories)

    def test_context_conflict_cannot_choose_favorable_mode(self) -> None:
        text = "before\n<proposed_plan>\nstep\n</proposed_plan>\nafter"
        for place in (1, -1):
            records = lifecycle((text,), "before\nafter", mode="plan")
            records.insert(place, context("default"))
            self.assertEqual(self.outcome(records).status, "partial")
        records = lifecycle((text,), "before\nafter", mode="default")
        records.insert(1, context())
        self.assertEqual(self.outcome(records).status, "partial")

    def test_foreign_unscoped_context_cannot_override_known_start_plan(self) -> None:
        text = "before\n<proposed_plan>\nstep\n</proposed_plan>\nafter"
        foreign = context("default", turn="foreign")
        unknown = context("default")
        unknown["payload"].pop("turn_id")
        malformed = context(turn="foreign", reasoning_effort=True)
        for value in (foreign, unknown, malformed):
            with self.subTest(value=value):
                records = lifecycle((text,), "before\nafter", mode="plan")
                records.insert(1, value)
                self.assertEqual(self.outcome(records).status, "complete")
                records = lifecycle((text,), "before\nafter", mode=None)
                records.insert(1, value)
                self.assertEqual(self.outcome(records).status, "partial")

    def test_explicit_invalid_mode_cannot_be_replaced_by_context(self) -> None:
        text = "<proposed_plan>\nstep\n</proposed_plan>"
        for mode in (True, "unknown", [], {}):
            records = lifecycle((text,), None, mode=mode)
            records.insert(1, context())
            self.assertEqual(self.outcome(records).status, "partial")

    def test_context_required_and_optional_shapes(self) -> None:
        text = "<proposed_plan>\nstep\n</proposed_plan>"
        for settings in ({}, {"reasoning_effort": None}, {"reasoning_effort": "custom"},
                         {"developer_instructions": "synthetic"}):
            records = lifecycle((text,), None, mode=None)
            records.insert(1, context(**settings))
            self.assertEqual(self.outcome(records).status, "complete")
        for settings in ({"model": True}, {"reasoning_effort": True}, {"reasoning_effort": ""},
                         {"developer_instructions": False}, {"unknown": "x"}):
            records = lifecycle((text,), None, mode=None)
            records.insert(1, context(**settings))
            self.assertEqual(self.outcome(records).status, "partial")

    def test_context_validation_only_material_to_plan(self) -> None:
        for value in (context(turn="foreign"), context(reasoning_effort=True), context("default"),
                      {"type": "turn_context", "payload": {"collaboration_mode": "unknown"}}):
            records = lifecycle(("plain",), "plain")
            records.insert(1, value)
            self.assertEqual(self.outcome(records).status, "complete")
            self.assertEqual(self.outcome([value]).status, "complete")

    def test_typed_plan_exact_body_separate_from_visible_echo(self) -> None:
        text = "before\n<proposed_plan>\nstep\n</proposed_plan>\nafter"
        records = lifecycle((text,), "before\nafter")
        records.insert(1, plan_event("step\n"))
        result = self.outcome(records)
        self.assertEqual(result.status, "complete", result.error_categories)
        self.assertEqual([value.content for value in result.messages], [text])

    def test_typed_plan_last_body_exact_crlf_eof_and_empty(self) -> None:
        for text, body in (("<proposed_plan>\nfirst\n</proposed_plan>\n<proposed_plan>\r\nsecond\r\n</proposed_plan>", "second\r\n"),
                           ("<proposed_plan>\nlast", "last"),
                           ("<proposed_plan>\n</proposed_plan>", "")):
            records = lifecycle((text,), None)
            records.insert(1, plan_event(body))
            self.assertEqual(self.outcome(records).status, "complete")
            records[1]["payload"]["item"]["text"] = body + "wrong"
            self.assertEqual(self.outcome(records).status, "partial")

    def test_typed_plan_body_citations_after_plan_extraction(self) -> None:
        text = "<oai-mem-citation>\n<proposed_plan>\nstep\n</proposed_plan>\n</oai-mem-citation>"
        records = lifecycle((text,), None)
        records.insert(1, plan_event("step\n"))
        self.assertEqual(self.outcome(records).status, "complete")
        text = "<proposed_plan>\nstep<oai-mem-citation>hidden</oai-mem-citation>\n</proposed_plan>"
        records = lifecycle((text,), None)
        records.insert(1, plan_event("step\n"))
        self.assertEqual(self.outcome(records).status, "complete")

    def test_typed_plan_output_only_not_input_text(self) -> None:
        text = "<proposed_plan>\nstep\n</proposed_plan>"
        records = lifecycle((text,), None)
        records[-2]["payload"]["content"][0]["type"] = "input_text"
        records.insert(1, plan_event("step\n"))
        self.assertEqual(self.outcome(records).status, "partial")

    def test_typed_plan_cannot_supply_canonical_or_agent_credit(self) -> None:
        self.assertEqual(self.outcome([event("task_started", turn_id=TURN, collaboration_mode_kind="plan"),
                                      plan_event("step\n")]).status, "partial")
        records = lifecycle(("<proposed_plan>\nstep\n</proposed_plan>",), None)
        records.insert(1, plan_event("step\n"))
        records.insert(2, event("agent_message", message="step\n"))
        self.assertEqual(self.outcome(records).status, "partial")

    def test_typed_plan_occurrence_credit_once_and_scope_boundaries(self) -> None:
        text = "<proposed_plan>\nstep\n</proposed_plan>"
        records = lifecycle((text,), None)
        records.insert(1, plan_event("step\n"))
        records.insert(2, plan_event("step\n"))
        self.assertEqual(self.outcome(records).status, "partial")
        records = lifecycle((text,), None)
        records.append(plan_event("step\n"))
        self.assertEqual(self.outcome(records).status, "partial")
        records = lifecycle((text,), None, mode=None)
        records.insert(1, plan_event("step\n"))
        records.insert(2, context())
        self.assertEqual(self.outcome(records).status, "partial")

    def test_typed_plan_wire_shape_and_time_types(self) -> None:
        text = "<proposed_plan>\nstep\n</proposed_plan>"
        for fields in ({}, {"started_at_ms": None}, {"started_at_ms": -(2**63), "completed_at_ms": 2**63 - 1}):
            records = lifecycle((text,), None)
            records.insert(1, plan_event("step\n", **fields))
            self.assertEqual(self.outcome(records).status, "complete")
        for fields in ({"completed_at_ms": None}, {"started_at_ms": True},
                       {"completed_at_ms": 2**63}, {"unknown": "x"}):
            records = lifecycle((text,), None)
            records.insert(1, plan_event("step\n", **fields))
            self.assertEqual(self.outcome(records).status, "partial")
        for key, value in (("id", "wrong"), ("text", []), ("unknown", "x")):
            records = lifecycle((text,), None)
            item = plan_event("step\n")
            item["payload"]["item"][key] = value
            records.insert(1, item)
            self.assertEqual(self.outcome(records).status, "partial")

    def test_plan_start_is_supported_context_without_credit(self) -> None:
        text = "before\n<proposed_plan>\nstep\n</proposed_plan>\nafter"
        records = lifecycle((text,), "before\nafter")
        records.insert(1, plan_start())
        records.insert(2, plan_event("step\n"))
        result = self.outcome(records)
        self.assertEqual(result.status, "complete", result.error_categories)
        self.assertEqual([value.content for value in result.messages], [text])
        records = lifecycle((text,), "before\nafter")
        records.insert(1, plan_start(-(2**63)))
        self.assertEqual(self.outcome(records).status, "complete")

    def test_plan_start_cannot_create_canonical_evidence(self) -> None:
        records = [event("task_started", turn_id=TURN, collaboration_mode_kind="plan"), plan_start()]
        self.assertEqual(self.outcome(records).status, "partial")
        records = lifecycle(("plain",), "plain")
        records.insert(1, plan_start())
        self.assertEqual(self.outcome(records).status, "partial")

    def test_plan_start_scope_mode_and_replay_are_strict(self) -> None:
        text = "<proposed_plan>\nstep\n</proposed_plan>"
        for mode in (None, "default"):
            records = lifecycle((text,), None, mode=mode)
            records.insert(1, plan_start())
            self.assertEqual(self.outcome(records).status, "partial")
        records = lifecycle((text,), None)
        records.insert(1, plan_start())
        records.insert(2, plan_start())
        self.assertEqual(self.outcome(records).status, "partial")
        records = lifecycle((text,), None)
        records.extend([plan_start()])
        self.assertEqual(self.outcome(records).status, "partial")
        records = lifecycle((text,), None)
        records.insert(1, plan_event("step\n"))
        records.insert(2, plan_start())
        self.assertEqual(self.outcome(records).status, "partial")

    def test_plan_start_required_signed_timestamp_and_empty_shape(self) -> None:
        text = "<proposed_plan>\nstep\n</proposed_plan>"
        for timestamp in (None, True, "1", -(2**63)-1, 2**63):
            records = lifecycle((text,), None)
            records.insert(1, plan_start(timestamp))
            self.assertEqual(self.outcome(records).status, "partial")
        for remove in ("started_at_ms", "thread_id", "turn_id"):
            records = lifecycle((text,), None)
            value = plan_start()
            value["payload"].pop(remove)
            records.insert(1, value)
            self.assertEqual(self.outcome(records).status, "partial")
        for key, value in (("text", "step"), ("id", "wrong"), ("unknown", "x")):
            records = lifecycle((text,), None)
            item = plan_start()
            item["payload"]["item"][key] = value
            records.insert(1, item)
            self.assertEqual(self.outcome(records).status, "partial")


if __name__ == "__main__":
    unittest.main()
