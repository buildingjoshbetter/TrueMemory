"""Exercise both real ingestion routes with synthetic SQLite and recall files."""
from __future__ import annotations

import ast
import builtins
import contextlib
import datetime
import enum
import json
import logging
import os
import re
import sqlite3
import sys
import tempfile
import time
import types
import unittest
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TEXT = "User: Synthetic station uses amber panels and stores reusable engineering notes."


class Action(enum.Enum):
    ADD = "add"
    UPDATE = "update"
    SKIP = "skip"


def selected_source(path: Path, names: set[str]) -> object:
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body
             if isinstance(node, ast.ImportFrom) and node.module == "__future__"
             or isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    return compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec")


def load_cache(directory: Path) -> types.ModuleType:
    module = types.ModuleType("synthetic_p02_cache")
    module.__dict__.update(Path=Path, json=json, os=os, time=time,
                           RECALL_CACHE_PATH=directory / "recall-cache.json",
                           RECALL_CACHE_TTL=300, _TRUEMEMORY_ROOT=directory)
    names = {"_secure_mkdir", "_atomic_write_text", "get_recall_cache", "set_recall_cache",
             "invalidate_recall_cache", "_normalize_db_path", "_recall_cache_key_prefix", "_recall_cache_key"}
    exec(selected_source(ROOT / "truememory/ingest/hooks/_shared.py", names), module.__dict__)
    return module


def load_pipeline(cache: types.ModuleType) -> types.ModuleType:
    parser = types.ModuleType("synthetic_p02_transcript")
    with patch.dict(sys.modules, {parser.__name__: parser}):
        path = ROOT / "truememory/ingest/transcript.py"
        exec(compile(path.read_text(), str(path), "exec"), parser.__dict__)

    def synthetic_import(name: str, globals: dict | None = None, locals: dict | None = None,
                         fromlist: tuple = (), level: int = 0) -> object:
        if name == "truememory.ingest.hooks._shared":
            return cache
        if name.startswith("truememory"):
            raise AssertionError("unexpected package/native import")
        return builtins.__import__(name, globals, locals, fromlist, level)

    module = types.ModuleType("synthetic_p02_pipeline")
    module.__dict__.update(
        contextlib=contextlib, datetime=datetime, json=json, logging=logging, re=re,
        sqlite3=sqlite3, time=time, Iterator=Iterator, dataclass=dataclass, field=field, Path=Path,
        log=logging.getLogger("synthetic.p02"), DedupAction=Action,
        _LOG_UNSAFE_RE=re.compile(r"[\x00-\x1f\x7f]"), _dedup_store_lock=contextlib.nullcontext,
        parse_transcript=parser.parse_transcript, format_for_extraction=parser.format_for_extraction,
        __builtins__=dict(vars(builtins), __import__=synthetic_import),
    )
    names = {"_safe_log", "IngestionResult", "_invalidate_recall_after_batch", "IngestionPipeline"}
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(selected_source(ROOT / "truememory/ingest/pipeline.py", names), module.__dict__)
    return module


class SyntheticEngine:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.before_add = None

    def add(self, content: str, **fields: object) -> dict:
        if self.before_add:
            self.before_add(content)
        cursor = self.conn.execute("INSERT INTO messages(content) VALUES (?)", (content,))
        self.conn.commit()
        return {"id": cursor.lastrowid, "content": content}


class SyntheticMemory:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._engine = SyntheticEngine(conn)
        self.before_update = None

    def add(self, content: str, **fields: object) -> dict:
        return self._engine.add(content)

    def update(self, memory_id: int, content: str) -> dict:
        if self.before_update:
            self.before_update(content)
        self._engine.conn.execute("UPDATE messages SET content=? WHERE id=?", (content, memory_id))
        self._engine.conn.commit()
        return {"id": memory_id, "content": content}


class TestIngestionRecallInvalidation(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-ingestion-cache-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = self.directory / "synthetic.sqlite"
        self.conn = sqlite3.connect(self.path, timeout=0)
        self.addCleanup(self.conn.close)
        self.conn.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY,content TEXT)")
        self.conn.commit()
        self.reader = sqlite3.connect(self.path, timeout=0)
        self.addCleanup(self.reader.close)
        self.transcript = self.directory / "synthetic-transcript.txt"
        self.transcript.write_text(TEXT)
        self.cache = load_cache(self.directory)
        self.module = load_pipeline(self.cache)
        self.pipeline = self.module.IngestionPipeline.__new__(self.module.IngestionPipeline)
        self.memory = SyntheticMemory(self.conn)
        self.pipeline.memory = self.memory
        self.pipeline.user_id = "synthetic-user"
        self.pipeline.llm_config = None
        self.pipeline.use_llm_dedup = False
        self.pipeline.gate_enabled = True
        self.pipeline.gate = types.SimpleNamespace(reset_batch=lambda: None, evaluate=self.gate)
        self.module.check_duplicate = self.dedup
        self.facts = []
        self.module.extract_facts_simple = lambda text: self.facts
        self.module.extract_facts = lambda text, config: self.facts
        self.invalidations = []
        self.cache_error = None
        invalidate = self.cache.invalidate_recall_cache

        def observed_invalidate(*args: object, **kwargs: object) -> None:
            # A second connection witnesses durable commits before invalidation.
            self.invalidations.append(self.reader.execute("SELECT content FROM messages ORDER BY id").fetchall())
            if self.cache_error:
                raise self.cache_error
            invalidate(*args, **kwargs)

        self.cache.invalidate_recall_cache = observed_invalidate

    def gate(self, content: str, category: str) -> types.SimpleNamespace:
        return types.SimpleNamespace(should_encode=not content.startswith("reject"), encoding_score=1.0,
                                     novelty=1.0, salience=1.0, prediction_error=1.0, reason="synthetic")

    def dedup(self, content: str, memory: object, **kwargs: object) -> types.SimpleNamespace:
        action = Action.SKIP if content.startswith("skip") else Action.UPDATE if content.startswith("replace") else Action.ADD
        return types.SimpleNamespace(action=action, fact=content, existing_id=1, reason="synthetic")

    def set_facts(self, *contents: str) -> None:
        self.facts = [types.SimpleNamespace(content=content, category="general", confidence="high") for content in contents]

    def seed_cache(self) -> None:
        self.cache.set_recall_cache("synthetic stale", str(self.path), "synthetic-user")
        self.assertEqual(self.cache.get_recall_cache(str(self.path), "synthetic-user"), "synthetic stale")

    def run_route(self, route: str) -> object:
        return (self.pipeline.ingest_text(TEXT, session_id="synthetic-session") if route == "text"
                else self.pipeline.ingest_transcript(self.transcript, session_id="synthetic-session"))

    def reset(self) -> None:
        self.reader.rollback()
        self.conn.rollback()
        self.conn.execute("DELETE FROM messages")
        self.conn.commit()
        self.invalidations.clear()
        self.memory._engine.before_add = None
        self.memory.before_update = None
        self.seed_cache()

    def test_both_routes_refresh_recall_after_committed_add(self) -> None:
        for route in ("text", "transcript"):
            with self.subTest(route=route):
                self.reset()
                self.set_facts("synthetic first fact", "synthetic second fact")
                other_db = str(self.directory / "other-synthetic.sqlite")
                self.cache.set_recall_cache("synthetic other scope", other_db, "other-synthetic-user", intensity="max", budget=2048)
                result = self.run_route(route)
                self.assertEqual((result.facts_extracted, result.facts_stored, result.facts_updated), (2, 2, 0))
                self.assertEqual(len(self.invalidations), 1)
                self.assertEqual(len(self.invalidations[0]), 2)
                self.assertIsNone(self.cache.get_recall_cache(str(self.path), "synthetic-user"))
                self.assertIsNone(self.cache.get_recall_cache(other_db, "other-synthetic-user", intensity="max", budget=2048))
                fresh = " | ".join(row[0] for row in self.reader.execute("SELECT content FROM messages ORDER BY id"))
                self.cache.set_recall_cache(fresh, str(self.path), "synthetic-user")
                self.assertEqual(self.cache.get_recall_cache(str(self.path), "synthetic-user"), fresh)

    def test_updates_invalidate_after_commit_on_both_routes(self) -> None:
        for route in ("text", "transcript"):
            with self.subTest(route=route):
                self.reset()
                self.conn.execute("INSERT INTO messages VALUES(1,'synthetic old')")
                self.conn.commit()
                self.set_facts("replace synthetic old")
                result = self.run_route(route)
                self.assertEqual((result.facts_stored, result.facts_updated), (0, 1))
                self.assertEqual(self.invalidations, [[("replace synthetic old",)]])
                self.assertIsNone(self.cache.get_recall_cache(str(self.path), "synthetic-user"))

    def test_noop_batches_do_not_invalidate(self) -> None:
        for route in ("text", "transcript"):
            for contents in ((), ("reject synthetic",), ("skip synthetic",), ("reject synthetic", "skip synthetic")):
                with self.subTest(route=route, size=len(contents)):
                    self.reset()
                    self.set_facts(*contents)
                    result = self.run_route(route)
                    self.assertEqual((result.facts_stored, result.facts_updated), (0, 0))
                    self.assertEqual(self.invalidations, [])
                    self.assertEqual(self.cache.get_recall_cache(str(self.path), "synthetic-user"), "synthetic stale")

    def test_mixed_batch_sqlite_lock_failure_invalidates_only_confirmed_writes(self) -> None:
        for route in ("text", "transcript"):
            with self.subTest(route=route):
                self.reset()
                self.conn.execute("INSERT INTO messages VALUES(1,'synthetic old')")
                self.conn.commit()
                self.set_facts("synthetic added", "replace synthetic updated", "skip synthetic", "reject synthetic", "synthetic locked")

                def lock_last(content: str) -> None:
                    if content == "synthetic locked":
                        self.reader.execute("BEGIN IMMEDIATE")

                self.memory._engine.before_add = lock_last
                with self.assertLogs("synthetic.p02", level="ERROR"):
                    result = self.run_route(route)
                self.reader.rollback()
                self.assertEqual((result.facts_extracted, result.facts_encoded, result.facts_stored,
                                  result.facts_updated, result.facts_skipped_gate, result.facts_skipped_dedup),
                                 (5, 4, 1, 1, 1, 1))
                self.assertEqual(len(self.invalidations), 1)
                self.assertEqual(len(self.invalidations[0]), 2)
                self.assertIsNone(self.cache.get_recall_cache(str(self.path), "synthetic-user"))
                self.assertEqual(self.conn.execute("SELECT count(*) FROM messages").fetchone(), (2,))

    def test_all_failed_adds_and_updates_leave_cache_unchanged(self) -> None:
        for route in ("text", "transcript"):
            with self.subTest(route=route):
                self.reset()
                self.conn.execute("INSERT INTO messages VALUES(1,'synthetic old')")
                self.conn.commit()
                self.reader.execute("BEGIN IMMEDIATE")
                self.set_facts("synthetic locked", "replace synthetic locked")
                with self.assertLogs("synthetic.p02", level="ERROR"):
                    result = self.run_route(route)
                self.reader.rollback()
                self.assertEqual((result.facts_stored, result.facts_updated), (0, 0))
                self.assertEqual(self.invalidations, [])
                self.assertEqual(self.cache.get_recall_cache(str(self.path), "synthetic-user"), "synthetic stale")

    def test_later_gate_or_dedup_exception_preserves_original_and_invalidates(self) -> None:
        for route in ("text", "transcript"):
            for step in ("gate", "dedup"):
                for failure in (RuntimeError("synthetic later failure"), KeyboardInterrupt("synthetic cancellation")):
                    with self.subTest(route=route, step=step, exception=type(failure).__name__):
                        self.reset()
                        self.set_facts("synthetic committed", "synthetic interrupted")

                        def gate(content: str, category: str) -> object:
                            if content == "synthetic interrupted" and step == "gate":
                                raise failure
                            return self.gate(content, category)

                        def dedup(content: str, memory: object, **kwargs: object) -> object:
                            if content == "synthetic interrupted" and step == "dedup":
                                raise failure
                            return self.dedup(content, memory, **kwargs)

                        self.pipeline.gate.evaluate = gate
                        self.module.check_duplicate = dedup
                        with self.assertRaises(type(failure)) as raised:
                            self.run_route(route)
                        self.assertIs(raised.exception, failure)
                        self.assertEqual(self.invalidations, [[("synthetic committed",)]])
                        self.assertIsNone(self.cache.get_recall_cache(str(self.path), "synthetic-user"))

    def test_cache_failure_is_diagnostic_and_does_not_replay_success(self) -> None:
        for route in ("text", "transcript"):
            with self.subTest(route=route):
                self.reset()
                self.set_facts("synthetic committed")
                self.cache_error = OSError("synthetic cache unavailable")
                with self.assertLogs("synthetic.p02", level="WARNING") as captured:
                    result = self.run_route(route)
                self.assertEqual(result.facts_stored, 1)
                self.assertEqual(self.conn.execute("SELECT count(*) FROM messages").fetchone(), (1,))
                self.assertEqual(len(self.invalidations), 1)
                self.assertEqual(len(captured.output), 1)
                self.assertIn("Recall cache invalidation failed", captured.output[0])
                self.assertNotIn("synthetic cache unavailable", captured.output[0])
                self.cache_error = None

    def test_cache_failure_cannot_replace_later_ingestion_exception(self) -> None:
        for route in ("text", "transcript"):
            with self.subTest(route=route):
                self.reset()
                self.set_facts("synthetic committed", "synthetic interrupted")
                original = RuntimeError("synthetic original failure")
                self.cache_error = OSError("synthetic cache unavailable")

                def gate(content: str, category: str) -> object:
                    if content == "synthetic interrupted":
                        raise original
                    return self.gate(content, category)

                self.pipeline.gate.evaluate = gate
                with self.assertLogs("synthetic.p02", level="WARNING"):
                    with self.assertRaises(RuntimeError) as raised:
                        self.run_route(route)
                self.assertIs(raised.exception, original)
                self.assertEqual(self.conn.execute("SELECT count(*) FROM messages").fetchone(), (1,))
                self.assertEqual(len(self.invalidations), 1)
                self.cache_error = None


if __name__ == "__main__":
    unittest.main()
