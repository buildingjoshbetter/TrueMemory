"""Synthetic controls for the bounded Hermes spool.

Every turn used to retain two full-history snapshots forever, so the spool
grew with the square of the turn count until capture and recall stopped.
Settled snapshots are now released, a snapshot whose messages begin a newer
one is superseded, and a full spool still answers recall.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_source(path: Path, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


BRIDGE = load_source(ROOT / "truememory/ingest/hooks/hermes.py", "synthetic_hermes_spool_retention")


@contextmanager
def collaborators() -> Iterator[None]:
    modules: dict[str, types.ModuleType] = {}
    for name in ("truememory", "truememory.ingest", "truememory.ingest.hooks", "truememory.hooks"):
        module = types.ModuleType(name)
        module.__path__ = []
        modules[name] = module
    shared = types.ModuleType("truememory.ingest.hooks._shared")
    shared.check_extraction_budget = lambda: True
    shared.refund_extraction_budget = lambda: None
    modules[shared.__name__] = shared
    with patch.dict(sys.modules, modules):
        yield


def payload(session: str, turn: int, history: list[dict[str, str]], *, post: bool = False) -> dict[str, object]:
    extra: dict[str, object] = {"turn_id": f"turn-{turn}", "user_message": f"Synthetic question {turn} " + "q" * 2000,
                                "conversation_history": history}
    if post:
        extra["assistant_response"] = f"Synthetic answer {turn} " + "a" * 2000
    return {"session_id": session, "extra": extra}


class HermesSpoolRetention(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="tm-synthetic-hermes-spool-")
        self.case = Path(self.directory.name).resolve()
        self.store = BRIDGE.Store(self.case / "owned", "synthetic-user", str(self.case / "synthetic.db"),
                                  "synthetic-session")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def run_worker(self) -> None:
        state = self.store.state()
        state["worker"] = {"token": "synthetic-token", "state": "reserved"}
        self.store.save(state)
        with collaborators(), patch.object(BRIDGE, "_owner", return_value={}), \
                patch.object(BRIDGE, "_recall", return_value="Synthetic recall"), \
                patch.object(BRIDGE, "_run_pipeline", return_value={"state": "extraction_complete"}):
            BRIDGE._worker(self.store, "synthetic-token")

    def turn(self, number: int, history: list[dict[str, str]]) -> list[dict[str, str]]:
        key = self.store.capture(payload(self.store.session, number, history), "pre_llm_call")
        self.run_worker()
        self.assertTrue(BRIDGE._emit_context(self.store, key, io.StringIO()))
        event = payload(self.store.session, number, history, post=True)
        self.store.capture(event, "post_llm_call")
        self.run_worker()
        return history + [{"role": "user", "content": event["extra"]["user_message"]},
                          {"role": "assistant", "content": event["extra"]["assistant_response"]}]

    def newest_snapshot_size(self) -> int:
        return max(path.stat().st_size for path in self.store.path.iterdir() if BRIDGE._SNAPSHOT.match(path.name)) \
            if any(BRIDGE._SNAPSHOT.match(path.name) for path in self.store.path.iterdir()) else 0

    def test_spool_stays_proportional_to_the_current_history(self) -> None:
        history: list[dict[str, str]] = []
        sizes = []
        for number in range(60):
            history = self.turn(number, history)
            sizes.append(self.store.usage()[0])
        state = self.store.state()
        self.assertTrue(all(item["state"] in ("extraction_complete", "superseded") for item in state["captures"].values()))
        self.assertTrue(all(item.get("released") for item in state["captures"].values()))
        self.assertEqual(len(state["turns"]), 60)
        # A retained turn costs O(history); the old spool kept 120 snapshots.
        last_history = len(json.dumps(history))
        self.assertLess(sizes[-1], 8 * last_history)
        self.assertLess(sizes[-1], sum(sizes) / len(sizes) * 4)
        self.assertFalse(list(self.store.path.glob("*.context.json")))

    def test_newer_snapshot_supersedes_only_an_exact_message_prefix(self) -> None:
        pre = self.store.capture(payload(self.store.session, 1, []), "pre_llm_call")
        self.store.capture(payload(self.store.session, 1, [], post=True), "post_llm_call")
        state = self.store.state()
        pre_name = state["turns"][pre]["snapshot"]
        self.assertEqual(state["captures"][pre_name]["state"], "superseded")
        # Its turn still awaits recall, so the bytes stay until the context is ready.
        self.assertTrue((self.store.path / pre_name).is_file())
        post_name = next(name for name, item in state["captures"].items() if item["state"] == "captured")
        compressed = [{"role": "user", "content": "Summary of earlier turns."}]
        self.store.capture(payload(self.store.session, 2, compressed, post=True), "post_llm_call")
        self.assertEqual(self.store.state()["captures"][post_name]["state"], "captured")
        self.run_worker()
        self.assertFalse((self.store.path / pre_name).exists())

    def test_full_account_admits_after_releasing_settled_snapshots(self) -> None:
        history: list[dict[str, str]] = []
        for number in range(3):
            history = self.turn(number, history)
        settled = self.store.usage()[0]
        other = BRIDGE.Store(self.store.root, self.store.user, self.store.db, "other-session")
        state = self.store.state()
        for item in state["captures"].values():
            item.pop("released", None)
        self.store.save(state)
        # Recreate settled bytes as if they had never been released.
        for name in state["captures"]:
            (self.store.path / name).write_bytes(b"x" * 20000)
            os.chmod(self.store.path / name, 0o600)
        with patch.object(BRIDGE, "MAX_SCOPE_BYTES", settled + 30000):
            other.capture(payload("other-session", 1, []), "pre_llm_call")
        self.assertEqual(len(other.state()["captures"]), 1)

    def test_full_spool_still_answers_recall(self) -> None:
        event = payload(self.store.session, 1, [])
        output = io.StringIO()
        with patch.dict(os.environ, {"TRUEMEMORY_HERMES_SPOOL_DIR": str(self.store.root)}), \
                patch.object(sys, "argv", ["hermes.py", "--event", "pre_llm_call", "--user", self.store.user,
                                          "--db", self.store.db]), \
                patch.object(sys, "stdin", io.StringIO(json.dumps(event))), patch.object(sys, "stdout", output), \
                patch.object(BRIDGE, "MAX_SCOPE_BYTES", 10), patch.object(BRIDGE, "_spawn") as spawn, \
                patch.object(BRIDGE, "_recall", return_value="Synthetic direct recall") as recall:
            self.assertEqual(BRIDGE.main(), 0)
        self.assertEqual(json.loads(output.getvalue()), {"context": "Synthetic direct recall"})
        self.assertEqual(recall.call_args.args[0]["native_user_message"], event["extra"]["user_message"])
        self.assertEqual(recall.call_args.args[1], 1)
        self.assertTrue(spawn.called)
        self.assertEqual(self.store.state()["captures"], {})

    def test_superseding_a_flushed_snapshot_keeps_its_extraction_queued(self) -> None:
        first = payload(self.store.session, 1, [], post=True)
        self.store.capture(first, "post_llm_call")
        history = [{"role": "user", "content": first["extra"]["user_message"]},
                   {"role": "assistant", "content": first["extra"]["assistant_response"]}]
        key = self.store.capture(payload(self.store.session, 2, history), "pre_llm_call")
        state = self.store.state()
        superseded = [item for item in state["captures"].values() if item["state"] == "superseded"]
        self.assertEqual(len(superseded), 1)
        self.assertTrue(superseded[0]["flush"])
        newest = state["captures"][state["turns"][key]["snapshot"]]
        # A crash before the next post must not lose the flushed turn.
        self.assertTrue(newest["flush"])
        self.assertTrue(BRIDGE._work_available(state))

    def test_long_session_without_delivered_contexts_stays_bounded(self) -> None:
        history: list[dict[str, str]] = []
        for number in range(150):
            key = self.store.capture(payload(self.store.session, number, history), "pre_llm_call")
            self.run_worker()  # the hook already gave up waiting; the context is never written
            event = payload(self.store.session, number, history, post=True)
            self.store.capture(event, "post_llm_call")
            self.run_worker()
            history = history[-4:] + [{"role": "user", "content": event["extra"]["user_message"]},
                                      {"role": "assistant", "content": event["extra"]["assistant_response"]}]
        state = self.store.state()
        self.assertLessEqual(len(list(self.store.path.glob("*.context.json"))), 1)
        self.assertEqual(state["turns"][key]["context_state"], "ready")
        self.assertLessEqual(len(state["captures"]), 2 * (BRIDGE.RETAINED_TURNS + 2))
        self.assertEqual(len(state["turns"]), 150)

    def test_finished_sessions_are_collected_when_the_account_is_full(self) -> None:
        sessions = []
        for index in range(6):
            store = BRIDGE.Store(self.store.root, self.store.user, self.store.db, f"finished-{index}")
            history: list[dict[str, str]] = []
            for number in range(3):
                store.capture(payload(store.session, number, history), "pre_llm_call")
                event = payload(store.session, number, history, post=True)
                store.capture(event, "post_llm_call")
                history += [{"role": "user", "content": event["extra"]["user_message"]},
                            {"role": "assistant", "content": event["extra"]["assistant_response"]}]
            self.store, saved = store, self.store
            self.run_worker()
            self.store = saved
            old = __import__("time").time() - 7200
            os.utime(store.state_path, (old, old))
            sessions.append(store)
        residual = self.store.usage()[0]
        self.assertGreater(residual, 0)
        live = BRIDGE.Store(self.store.root, self.store.user, self.store.db, "live-session")
        with patch.object(BRIDGE, "MAX_SCOPE_BYTES", 12000):
            live.capture(payload("live-session", 1, []), "pre_llm_call")
        self.assertEqual(len(live.state()["captures"]), 1)
        self.assertFalse(any(store.path.exists() for store in sessions))
        self.assertLessEqual(self.store.usage()[0], 12000)


if __name__ == "__main__":
    unittest.main()
