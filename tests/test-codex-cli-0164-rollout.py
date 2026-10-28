"""Synthetic codex-cli 0.164 rollouts: typed user input is a human turn, scaffold is not."""
from __future__ import annotations

import builtins
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCAFFOLD = "synthetic-scaffold-content"


def load_parser() -> types.ModuleType:
    module = types.ModuleType("synthetic_codex_0164_parser")

    def safe_import(name: str, globals: dict | None = None, locals: dict | None = None,
                    fromlist: tuple = (), level: int = 0) -> object:
        if name.startswith(("truememory", "torch", "numpy", "sentence_transformers")):
            raise AssertionError("Application imports are forbidden in parser tests")
        return builtins.__import__(name, globals, locals, fromlist, level)

    module.__dict__["__builtins__"] = dict(vars(builtins), __import__=safe_import)
    source = (ROOT / "truememory/ingest/transcript.py").read_text(encoding="utf-8")
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(compile(source, "synthetic-0164-parser", "exec"), module.__dict__)
    return module


def event(kind: str, **fields: object) -> dict:
    return {"timestamp": "2026-01-01T00:00:00Z", "type": "event_msg", "payload": {"type": kind, **fields}}


def message(role: str, blocks: list[tuple[str, str, dict | None]], *, item: str, turn: str,
            phase: str | None = None) -> dict:
    """blocks: (text, content_item_kind, provenance or None)."""
    payload: dict = {
        "type": "message", "role": role, "id": item,
        "content": [{"type": "input_text" if role == "user" else "output_text", "text": text}
                    for text, _, _ in blocks],
        "internal_chat_message_metadata_passthrough": {
            "turn_id": turn,
            "content_item_kinds": [kind for _, kind, _ in blocks],
            "content_item_metadata": [{"provenance": prov} if prov else {} for _, _, prov in blocks],
        },
    }
    if phase is not None:
        payload["phase"] = phase
    return {"timestamp": "2026-01-01T00:00:00Z", "type": "response_item", "payload": payload}


def turn_records(index: int, *, user_provenance: dict | None = None) -> list[dict]:
    turn, user_text, reply = f"synthetic-turn-{index}", f"synthetic question {index}", f"synthetic answer {index}"
    provenance = user_provenance or {"type": "user"}
    return [
        event("task_started", turn_id=turn, collaboration_mode_kind="default"),
        # Canonical id and event id differ, as codex-cli 0.164 writes them.
        message("user", [(user_text, "user.text", provenance)], item=f"msg_canonical-{index}", turn=turn),
        event("item_completed", thread_id="synthetic-thread", turn_id=turn,
              item={"type": "UserMessage", "id": f"event-{index}", "client_id": f"client-{index}",
                    "content": [{"type": "text", "text": user_text, "text_elements": []}]},
              started_at_ms=1, completed_at_ms=1),
        event("item_completed", thread_id="synthetic-thread", turn_id=turn,
              item={"type": "Extension", "kind": "web.search", "id": f"search-{index}", "query": "synthetic"},
              started_at_ms=1, completed_at_ms=1),
        message("assistant", [(reply, "unknown", None)], item=f"msg_reply-{index}", turn=turn, phase="final_answer"),
        event("item_completed", thread_id="synthetic-thread", turn_id=turn,
              item={"type": "AgentMessage", "id": f"msg_reply-{index}", "phase": "final_answer",
                    "content": [{"type": "Text", "text": reply}]},
              started_at_ms=1, completed_at_ms=1),
        event("task_complete", turn_id=turn, last_agent_message=reply),
    ]


def rollout(turns: int = 5, **options: object) -> list[dict]:
    records: list[dict] = [
        {"timestamp": "2026-01-01T00:00:00Z", "type": "session_meta", "payload": {"id": "synthetic-thread"}},
        event("thread_settings_applied", thread_id="synthetic-thread", thread_settings={"model": "synthetic"}),
        message("user", [(SCAFFOLD, "agents_md.instructions", {"type": "agents_md"}),
                         (SCAFFOLD, "environments.environment_context", {"type": "harness", "hook": False})],
                item="msg_scaffold", turn="synthetic-turn-0"),
    ]
    for index in range(turns):
        records += turn_records(index, **options)
    return records


class CodexCli0164RolloutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = load_parser()

    def outcome(self, records: list[dict]) -> object:
        return self.parser._parse_transcript_text_outcome("\n".join(json.dumps(r) for r in records))

    def test_typed_user_input_is_a_complete_human_turn(self) -> None:
        result = self.outcome(rollout())
        self.assertEqual(result.status, "complete")
        expected = []
        for index in range(5):
            expected += [("human", f"synthetic question {index}"), ("assistant", f"synthetic answer {index}")]
        self.assertEqual([(m.role, m.content) for m in result.messages], expected)
        self.assertNotIn(SCAFFOLD, repr(result))

    def test_stop_hook_counts_the_human_turns(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tm-synthetic-0164-") as directory:
            path = Path(directory) / "rollout-synthetic.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rollout()), encoding="utf-8")
            self.assertIs(self.parser.has_human_messages(path, 5), True)
            path.write_text("\n".join(json.dumps(r) for r in rollout(turns=4)), encoding="utf-8")
            self.assertIs(self.parser.has_human_messages(path, 5), False)

    def test_user_text_label_without_user_provenance_is_not_a_human_turn(self) -> None:
        for provenance in ({"type": "harness", "hook": False}, {"type": "agents_md"}, {"type": "synthetic-new"}):
            with self.subTest(provenance=provenance["type"]):
                result = self.outcome(rollout(turns=1, user_provenance=provenance))
                self.assertEqual(result.status, "partial")
                self.assertNotIn(("human", "synthetic question 0"),
                                 [(m.role, m.content) for m in result.messages])

    def test_user_event_text_must_still_match_the_canonical_turn(self) -> None:
        records = rollout(turns=1)
        user_event = next(r for r in records if r["type"] == "event_msg"
                          and r["payload"]["type"] == "item_completed"
                          and r["payload"]["item"]["type"] == "UserMessage")
        user_event["payload"]["item"]["content"][0]["text"] = "synthetic unmatched text"
        self.assertEqual(self.outcome(records).status, "partial")


if __name__ == "__main__":
    unittest.main()
