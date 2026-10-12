"""Synthetic actual-source landmark/temporal consumer controls, without startup.

The provenance accessor collaborator tests the locked interface. Its producer
and metadata validation are covered independently by the source-provenance
module tests; these controls do not certify native/model or archive acceptance.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import logging
import os
import sqlite3
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def load_functions(name: str, selected: set[str], namespace: dict) -> types.ModuleType:
    path = ROOT / "truememory" / (name + ".py")
    module = types.ModuleType("synthetic_landmark755_" + name)
    module.__dict__.update(namespace)
    tree = ast.parse(path.read_text(), filename=str(path))
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in selected:
            node.decorator_list = []
            body.append(node)
        elif isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name in selected:
                    method.decorator_list = []
                    body.append(method)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


STORAGE = types.ModuleType("synthetic_landmark755_storage")
exec(compile((ROOT / "truememory/storage.py").read_text(), "storage.py", "exec"), STORAGE.__dict__)
TEMPORAL = types.ModuleType("synthetic_landmark755_temporal")
TEMPORAL._row_to_dict = STORAGE._row_to_dict
TEMPORAL.select_message_cols = STORAGE.select_message_cols
tree = ast.parse((ROOT / "truememory/temporal.py").read_text())
tree.body = [node for node in tree.body if not (
    isinstance(node, ast.ImportFrom) and node.module == "truememory.storage"
)]
exec(compile(tree, "temporal.py", "exec"), TEMPORAL.__dict__)


def interface_intervals(metadata: object) -> tuple[list[tuple[str, str]], bool] | None:
    if not isinstance(metadata, dict) or "truememory_ingest" not in metadata:
        return None
    scope = metadata["truememory_ingest"]
    if not isinstance(scope, dict) or scope.get("version") != 1:
        return [], True
    intervals = []
    unknown = False
    for citation in scope.get("citations", []):
        source_time = citation["scope"]["source_time"]
        if source_time["status"] in {"known", "mixed"} and source_time.get("start") and source_time.get("end"):
            intervals.append((source_time["start"], source_time["end"]))
        unknown = unknown or source_time["status"] != "known"
    return intervals, unknown


def metadata_intervals(*intervals: tuple[str, str], unknown: bool = False) -> dict:
    citations = [{"scope": {"source_time": {
        "status": "known", "start": start, "end": end,
    }}} for start, end in intervals]
    if unknown:
        citations.append({"scope": {"source_time": {"status": "unknown", "start": None, "end": None}}})
    return {"truememory_ingest": {"version": 1, "timestamp_semantics": "source_scope_end", "citations": citations}}


@contextmanager
def provenance_interface():
    module = types.ModuleType("truememory.ingest.provenance")
    module.source_intervals = interface_intervals
    with patch.dict(sys.modules, {"truememory.ingest.provenance": module}):
        yield


class LandmarkControls(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.addCleanup(self.conn.close)
        self.conn.executescript(
            "CREATE TABLE messages(id INTEGER PRIMARY KEY,content TEXT,sender TEXT,recipient TEXT,"
            "timestamp TEXT,category TEXT DEFAULT '',modality TEXT DEFAULT '',directive INTEGER DEFAULT 0,metadata TEXT DEFAULT '{}');"
            "CREATE TABLE landmark_events(id INTEGER PRIMARY KEY,event_name TEXT,timestamp TEXT,"
            "event_type TEXT,related_entities TEXT DEFAULT '[]',source_message_id INTEGER);"
        )

    def event(self, phrase: str = "Demo Day", timestamp: str = "2025-06-15", sender: str = "synthetic-alice",
              metadata: dict | None = None) -> int:
        cursor = self.conn.execute(
            "INSERT INTO messages(content,sender,recipient,timestamp,metadata) VALUES(?,?,?,?,?)",
            (phrase + " arrived", sender, "", timestamp, json.dumps(metadata or {})),
        )
        source_id = cursor.lastrowid
        self.conn.execute(
            "INSERT INTO landmark_events(event_name,timestamp,event_type,related_entities,source_message_id) VALUES(?,?,?,?,?)",
            (phrase + " arrived", timestamp, "launch", json.dumps([sender]), source_id),
        )
        return source_id

    def intent(self, query: str = "What happened in the first month after Demo Day?", sender: str | None = None) -> dict:
        with provenance_interface():
            return TEMPORAL.resolve_temporal_intent(self.conn, query, sender)

    def test_named_event_resolves_live_source_and_existing_month(self) -> None:
        source_id = self.event()
        intent = self.intent()
        self.assertEqual((intent["after"], intent["before"]), ("2025-06-15", "2025-07-16"))
        diagnostic = intent["landmark_resolution"]
        self.assertEqual(diagnostic["status"], "resolved")
        self.assertEqual(diagnostic["candidates"][0]["source_message_id"], source_id)
        self.assertEqual(diagnostic["candidates"][0]["timestamp_basis"], "source_message_timestamp")

    def test_before_and_week_year_windows(self) -> None:
        self.event()
        for query, after, before in [
            ("What happened before Demo Day?", None, "2025-06-15"),
            ("What happened in the first week after Demo Day?", "2025-06-15", "2025-06-22"),
            ("What happened in the first year after Demo Day?", "2025-06-15", "2026-06-15"),
        ]:
            with self.subTest(query=query):
                intent = self.intent(query)
                self.assertEqual((intent["after"], intent["before"]), (after, before))

    def test_explicit_relative_date_precedence_no_landmark_lookup(self) -> None:
        self.event(timestamp="2020-01-01")
        for query in ["What happened after Demo Day (June 15, 2025)?", "What happened after January 15, 2026?",
                      "What happened after Demo Day last month?", "What happened yesterday?",
                      "What happened between March 5 2025 and March 5 2025?"]:
            with self.subTest(query=query):
                self.assertEqual(self.intent(query), TEMPORAL.detect_temporal_intent(query))

    def test_multiple_dates_and_same_date_sources_are_ambiguous(self) -> None:
        for second_date in ["2025-07-01", "2025-06-15"]:
            with self.subTest(second_date=second_date):
                self.conn.execute("DELETE FROM landmark_events")
                self.conn.execute("DELETE FROM messages")
                self.event()
                self.event(timestamp=second_date, sender="synthetic-bob")
                intent = self.intent()
                self.assertEqual(intent["landmark_resolution"]["status"], "ambiguous")
                self.assertIsNone(intent["after"])
                self.assertFalse(intent["has_temporal"])

    def test_sender_scope_and_possessive_qualifier(self) -> None:
        alice = self.event()
        self.event(timestamp="2025-07-01", sender="synthetic-bob")
        for query, sender in [("What happened after Demo Day?", "synthetic-alice"),
                              ("What happened after synthetic-alice's Demo Day?", None)]:
            with self.subTest(query=query):
                diagnostic = self.intent(query, sender)["landmark_resolution"]
                self.assertEqual(diagnostic["status"], "resolved")
                self.assertEqual([c["source_message_id"] for c in diagnostic["candidates"]], [alice])
        self.assertEqual(self.intent(sender="absent-sender")["landmark_resolution"]["candidates"], [])

    def test_literal_matching_quotes_whitespace_and_sql_metacharacters(self) -> None:
        self.event("Demo   Day")
        self.assertEqual(self.intent('What happened after "DEMO DAY"?')["landmark_resolution"]["status"], "resolved")
        self.conn.execute("DELETE FROM landmark_events")
        self.conn.execute("DELETE FROM messages")
        self.event("Cedar_%'s Launch")
        self.assertEqual(self.intent('What happened after "Cedar_%\'s Launch"?')["landmark_resolution"]["status"], "resolved")
        self.event("Cedar_% Launch")
        self.assertEqual(self.intent('What happened after "Cedar_% Launch"?')["landmark_resolution"]["status"], "resolved")
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 2)

    def test_whole_phrase_no_fuzzy_or_substring_match(self) -> None:
        self.event("SuperDemo Daylight")
        self.assertEqual(self.intent()["landmark_resolution"]["status"], "unresolved")
        self.assertEqual(self.intent("What happened after Demo Days?")["landmark_resolution"]["status"], "unresolved")

    def test_encoded_calendar_offsets_no_machine_timezone_conversion(self) -> None:
        for timestamp in ["2025-06-15T23:30:00-07:00", "2025-06-15T00:30:00+14:00", "2025-06-15T12:00:00Z"]:
            with self.subTest(timestamp=timestamp):
                self.conn.execute("DELETE FROM landmark_events")
                self.conn.execute("DELETE FROM messages")
                self.event(timestamp=timestamp)
                self.assertEqual(self.intent()["after"], "2025-06-15")

    def test_invalid_calendar_dates_and_non_iso_values_fail_open(self) -> None:
        for timestamp in ["2025-02-30", "2025-13-01", "20250615", "not-a-date", "2025-06-15garbage", "9999-12-31"]:
            with self.subTest(timestamp=timestamp):
                self.conn.execute("DELETE FROM landmark_events")
                self.conn.execute("DELETE FROM messages")
                self.event(timestamp=timestamp)
                intent = self.intent()
                self.assertIsNone(intent["after"])
                self.assertEqual(intent["landmark_resolution"]["status"], "unresolved")

    def test_all_extraction_scopes_reject_precise_anchor(self) -> None:
        examples = [metadata_intervals(("2025-06-15T00:00:00+00:00", "2025-06-15T00:00:00+00:00")),
                    metadata_intervals(("2025-06-15T00:00:00+00:00", "2025-06-15T23:00:00+00:00")),
                    metadata_intervals(("2025-06-14T00:00:00+00:00", "2025-06-15T00:00:00+00:00")),
                    metadata_intervals(unknown=True), {"truememory_ingest": {"version": 2}},
                    {"truememory_ingest": "malformed"}]
        for metadata in examples:
            with self.subTest(metadata=metadata):
                self.conn.execute("DELETE FROM landmark_events")
                self.conn.execute("DELETE FROM messages")
                self.event(metadata=metadata)
                intent = self.intent()
                self.assertIsNone(intent["after"])
                self.assertEqual(intent["landmark_resolution"]["reason"], "source_date_unresolved")

    def test_missing_source_stale_date_or_content_fail_open(self) -> None:
        for mutation in ["DELETE FROM messages", "UPDATE messages SET timestamp='2025-07-01'",
                         "UPDATE messages SET content='unrelated synthetic fact'"]:
            with self.subTest(mutation=mutation):
                self.conn.execute("DELETE FROM landmark_events")
                self.conn.execute("DELETE FROM messages")
                self.event()
                self.conn.execute(mutation)
                self.assertIsNone(self.intent()["after"])

    def test_missing_and_legacy_schema_fail_open(self) -> None:
        self.conn.execute("DROP TABLE landmark_events")
        self.assertEqual(self.intent()["landmark_resolution"]["reason"], "landmark_schema_unavailable")
        self.conn.execute("CREATE TABLE landmark_events(id INTEGER PRIMARY KEY,event_name TEXT,timestamp TEXT)")
        self.assertEqual(self.intent()["landmark_resolution"]["reason"], "landmark_schema_unavailable")

    def test_invalid_json_identity_and_oversized_matching_source_fail_open(self) -> None:
        for mutation in ["UPDATE landmark_events SET related_entities='invalid-json'",
                         "UPDATE landmark_events SET source_message_id=NULL",
                         "UPDATE messages SET metadata='invalid-json'",
                         "UPDATE messages SET content=content || printf('%5000s','padding')"]:
            with self.subTest(mutation=mutation):
                self.conn.execute("DELETE FROM landmark_events")
                self.conn.execute("DELETE FROM messages")
                self.event()
                self.conn.execute(mutation)
                self.assertIsNone(self.intent()["after"])

    def test_inventory_overflow_does_not_resolve_partial_subset(self) -> None:
        self.event()
        self.conn.executemany("INSERT INTO landmark_events(event_name,timestamp) VALUES(?,?)",
                              [("irrelevant", "2025-01-01")] * 1024)
        intent = self.intent()
        self.assertIsNone(intent["after"])
        self.assertEqual(intent["landmark_resolution"]["reason"], "landmark_inventory_limit")

    def test_candidate_truncation_never_resolves_and_duplicate_same_source_is_safe(self) -> None:
        source_id = self.event()
        self.conn.execute("INSERT INTO landmark_events(event_name,timestamp,source_message_id) VALUES('Demo Day arrived','2025-06-15',?)", (source_id,))
        self.assertEqual(self.intent()["landmark_resolution"]["status"], "resolved")
        for index in range(18):
            self.event(sender=f"synthetic-sender-{index}")
        diagnostic = self.intent()["landmark_resolution"]
        self.assertEqual(diagnostic["status"], "ambiguous")
        self.assertEqual(len(diagnostic["candidates"]), 16)
        self.assertTrue(diagnostic["candidates_truncated"])

    def test_directive_anchor_is_not_disclosed(self) -> None:
        self.event()
        self.conn.execute("UPDATE messages SET directive=1")
        self.assertEqual(self.intent()["landmark_resolution"]["candidates"], [])

    def test_unresolved_search_preserves_input_order_scores_and_metadata(self) -> None:
        rows = [{"id": 100, "content": "synthetic high", "score": 0.8, "timestamp": "2024-01-01", "metadata": {"keep": 1}},
                {"id": 101, "content": "synthetic low", "score": 0.2, "timestamp": "2026-01-01", "metadata": {}}]
        before = copy.deepcopy(rows)
        with provenance_interface():
            results = TEMPORAL.search_temporal(self.conn, "What happened after Missing Event?", fts_results=rows)
        self.assertEqual(rows, before)
        self.assertEqual([{k: v for k, v in row.items() if k != "temporal_resolution"} for row in results], before)

    def test_ordinary_intervals_overlap_individually_not_envelope(self) -> None:
        row = {"timestamp": "2025-09-01", "metadata": metadata_intervals(
            ("2025-01-01T00:00:00+00:00", "2025-01-02T00:00:00+00:00"),
            ("2025-09-01T00:00:00+00:00", "2025-09-02T00:00:00+00:00"))}
        with provenance_interface():
            self.assertTrue(TEMPORAL._temporal_row_matches(row, "2025-01-01", "2025-01-03"))
            self.assertFalse(TEMPORAL._temporal_row_matches(row, "2025-06-01", "2025-07-01"))
            self.assertTrue(TEMPORAL._temporal_row_matches(row, "2025-09-01", "2025-09-03"))

    def test_unknown_intervals_excluded_only_for_real_window(self) -> None:
        row = {"timestamp": "2025-06-15", "metadata": metadata_intervals(unknown=True)}
        with provenance_interface():
            self.assertFalse(TEMPORAL._temporal_row_matches(row, "2025-06-01", "2025-07-01"))
            self.assertTrue(TEMPORAL._temporal_row_matches(row, None, None))
            self.assertFalse(TEMPORAL._temporal_row_matches(
                {"timestamp": "2025-06-15", "metadata": {"truememory_ingest": {"version": 2}}}, "2025-06-01", "2025-07-01"))

    def test_timeline_fallback_finds_earlier_disjoint_interval(self) -> None:
        self.event("Synthetic evidence", "2025-09-01", metadata=metadata_intervals(
            ("2025-01-01T00:00:00+00:00", "2025-01-02T00:00:00+00:00"),
            ("2025-09-01T00:00:00+00:00", "2025-09-02T00:00:00+00:00")))
        with provenance_interface():
            early = TEMPORAL.get_timeline(self.conn, after="2025-01-01", before="2025-01-02")
            gap = TEMPORAL.get_timeline(self.conn, after="2025-06-01", before="2025-06-30")
        self.assertEqual(len(early), 1)
        self.assertEqual(gap, [])

    def test_timeline_exclusive_day_directive_entity_and_unknown_scope(self) -> None:
        self.event("Synthetic evidence", "2025-06-15", metadata=metadata_intervals(
            ("2025-06-16T00:00:00+00:00", "2025-06-16T00:00:00+00:00")))
        with provenance_interface():
            self.assertEqual(TEMPORAL.get_timeline(self.conn, after="2025-06-15", before="2025-06-15"), [])
            self.conn.execute("UPDATE messages SET metadata=?", (json.dumps(metadata_intervals(
                ("2025-06-15T23:59:59+00:00", "2025-06-15T23:59:59+00:00"))),))
            self.assertEqual(len(TEMPORAL.get_timeline(self.conn, entity="synthetic-alice", after="2025-06-15", before="2025-06-15")), 1)
            self.assertEqual(TEMPORAL.get_timeline(self.conn, entity="synthetic-bob", after="2025-06-15", before="2025-06-15"), [])
            self.conn.execute("UPDATE messages SET directive=1")
            self.assertEqual(TEMPORAL.get_timeline(self.conn, after="2025-06-15", before="2025-06-15"), [])

    def test_timeline_overflow_exposes_incomplete_coverage(self) -> None:
        metadata = json.dumps(metadata_intervals(("2025-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00")))
        self.conn.executemany("INSERT INTO messages(content,sender,timestamp,metadata) VALUES('synthetic','','2025-09-01',?)", [(metadata,)] * 1025)
        with provenance_interface():
            results = TEMPORAL.get_timeline(self.conn, after="2025-01-01", before="2025-01-01")
        self.assertEqual(len(results), 1024)
        self.assertEqual(results[0]["temporal_scope_coverage"]["status"], "incomplete")

    def test_fallback_caps_visited_raw_rows_before_namespace_filtering(self) -> None:
        self.conn.execute("INSERT INTO messages(content,sender,timestamp) VALUES('synthetic in-window','','2025-01-01')")
        self.conn.executemany("INSERT INTO messages(content,sender,timestamp) VALUES('synthetic legacy','','2025-09-01')", [()] * 1400)
        late = self.event("Synthetic late source", "2025-09-01", metadata=metadata_intervals(
            ("2025-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00")))
        with provenance_interface():
            results = TEMPORAL.get_timeline(self.conn, after="2025-01-01", before="2025-01-01")
        self.assertNotIn(late, {row["id"] for row in results})
        self.assertEqual(results[0]["temporal_scope_coverage"]["status"], "incomplete")

    def test_empty_client_status_preserves_raw_overflow_and_schema_unavailable(self) -> None:
        client = load_functions("client", {"resolve_temporal_query"}, {})
        self.conn.executemany("INSERT INTO messages(content,sender,timestamp) VALUES('synthetic legacy','','2025-09-01')", [()] * 1025)
        engine = types.SimpleNamespace(conn=self.conn, _ensure_connection=lambda: None)
        with patch.dict(sys.modules, {"truememory.temporal": TEMPORAL}):
            diagnostic = client.resolve_temporal_query(types.SimpleNamespace(_engine=engine), "What happened in January 2025?")
        self.assertEqual(diagnostic["source_interval_coverage"]["status"], "incomplete")
        self.assertEqual(diagnostic["source_interval_coverage"]["reason"], "source_interval_inventory_limit")
        self.conn.execute("ALTER TABLE messages RENAME COLUMN metadata TO synthetic_legacy_metadata")
        with patch.dict(sys.modules, {"truememory.temporal": TEMPORAL}):
            diagnostic = client.resolve_temporal_query(types.SimpleNamespace(_engine=engine), "What happened in January 2025?")
        self.assertEqual(diagnostic["source_interval_coverage"]["reason"], "source_interval_lookup_unavailable")

    def test_raw_lookup_sqlite_work_remains_bounded_with_5005_legacy_rows(self) -> None:
        self.conn.executemany("INSERT INTO messages(content,sender,timestamp) VALUES('synthetic legacy','','2025-09-01')", [()] * 5005)
        state = {"active": False, "steps": 0}

        def trace(sql: str) -> None:
            state["active"] = sql.startswith("SELECT id,substr(metadata")

        def progress() -> int:
            if state["active"]:
                state["steps"] += 1
            return 0

        self.conn.set_trace_callback(trace)
        self.conn.set_progress_handler(progress, 1)
        try:
            candidates, diagnostic = TEMPORAL._source_interval_candidates(self.conn)
        finally:
            self.conn.set_progress_handler(None, 0)
            self.conn.set_trace_callback(None)
        self.assertEqual(candidates, [])
        self.assertEqual(diagnostic["reason"], "source_interval_inventory_limit")
        self.assertEqual(diagnostic["candidate_limit"], 1024)
        self.assertEqual(diagnostic["visited_row_limit"], 1025)
        self.assertLess(state["steps"], 25 * (TEMPORAL._SOURCE_INTERVAL_ROW_LIMIT + 1))

    def test_unindexed_legacy_ids_do_not_trigger_unbounded_sort_or_lookup(self) -> None:
        self.conn.execute("ALTER TABLE messages RENAME TO synthetic_original_messages")
        self.conn.execute("CREATE TABLE messages(id INTEGER,content TEXT,timestamp TEXT,metadata TEXT)")
        self.conn.executemany("INSERT INTO messages VALUES(1,'synthetic','2025-01-01','{}')", [()] * 2000)
        statements = []
        self.conn.set_trace_callback(statements.append)
        try:
            candidates, diagnostic = TEMPORAL._source_interval_candidates(self.conn)
        finally:
            self.conn.set_trace_callback(None)
        self.assertEqual(candidates, [])
        self.assertEqual(diagnostic["reason"], "source_interval_lookup_unavailable")
        self.assertFalse(any(statement.startswith("SELECT") for statement in statements))

    def test_actual_mcp_empty_envelope_exposes_source_inventory_overflow(self) -> None:
        self.conn.executemany("INSERT INTO messages(content,sender,timestamp) VALUES('synthetic legacy','','2025-09-01')", [()] * 1025)
        client = load_functions("client", {"resolve_temporal_query"}, {})
        engine = types.SimpleNamespace(conn=self.conn, _ensure_connection=lambda: None)
        memory = types.SimpleNamespace(_engine=engine, search_deep=lambda *args, **kwargs: [])
        memory.resolve_temporal_query = lambda query, user_id=None: client.resolve_temporal_query(memory, query, user_id)

        @contextmanager
        def operation():
            yield memory, types.SimpleNamespace(reranker_id="unchanged", tier="free")

        tool = load_functions("mcp_server", {"truememory_search"}, {
            "json": json, "_touch_search_time": lambda: None, "_search_operation": operation,
            "_set_reranker": lambda _: None, "_SEARCH_INTERNAL_LIMIT": 100,
        })
        with patch.dict(sys.modules, {"truememory.temporal": TEMPORAL}):
            response = json.loads(tool.truememory_search("What happened in January 2025?", include_temporal_resolution=True))
        self.assertEqual(response["results"], [])
        self.assertEqual(response["temporal_resolution"]["source_interval_coverage"]["reason"], "source_interval_inventory_limit")

    def test_actual_engine_search_uses_anchor_and_preserves_ambiguity(self) -> None:
        self.event()
        rows = [{"id": 101, "content": "synthetic before", "sender": "synthetic-alice", "timestamp": "2025-05-01", "score": 0.2},
                {"id": 102, "content": "synthetic after", "sender": "synthetic-alice", "timestamp": "2025-06-20", "score": 0.2}]
        engine = load_functions("engine", {"search"}, {
            "_HAS_QUERY_CLASSIFIER": False, "search_fts": lambda *args, **kwargs: copy.deepcopy(rows),
            "resolve_temporal_intent": TEMPORAL.resolve_temporal_intent,
            "search_temporal": TEMPORAL.search_temporal, "attach_temporal_resolution": TEMPORAL.attach_temporal_resolution,
            "logger": logging.getLogger("synthetic-landmark"), "os": os,
        })
        instance = types.SimpleNamespace(conn=self.conn, _ensure_connection=lambda: None,
            _has_hybrid=False, _has_temporal=True, _has_personality=False, _has_consolidation=False,
            _has_salience=False, _has_reranker=False, _apply_surprise_boost=lambda rows: rows)
        with provenance_interface():
            resolved = engine.search(instance, "What happened in the first month after Demo Day?", limit=10)
            self.event(timestamp="2025-07-01", sender="synthetic-bob")
            ambiguous = engine.search(instance, "What happened in the first month after Demo Day?", limit=10)
        self.assertTrue(all(row["temporal_resolution"]["status"] == "resolved" for row in resolved))
        after = next(row for row in resolved if row["id"] == 102)
        self.assertEqual(after["score"], 0.2 * 1.3)
        self.assertEqual([(row["id"], row["score"]) for row in ambiguous], [(101, 0.2), (102, 0.2)])
        self.assertTrue(all(row["temporal_resolution"]["status"] == "ambiguous" for row in ambiguous))

    def test_actual_agentic_cleaner_retains_original_query_diagnostic(self) -> None:
        self.event()
        consumer = load_functions("engine", {"search_agentic"}, {
            "resolve_temporal_intent": TEMPORAL.resolve_temporal_intent,
            "attach_temporal_resolution": TEMPORAL.attach_temporal_resolution,
            "logger": logging.getLogger("synthetic-landmark"), "os": os,
        })
        agentic = load_functions("agentic_search", {"normalize_scores", "clean_results"}, {})
        query = "What happened in the first month after Demo Day?"
        rows = [{"id": 101, "content": "synthetic result", "timestamp": "2025-06-20", "score": 0.2, "source": "fts"}]
        instance = types.SimpleNamespace(conn=self.conn, _ensure_connection=lambda: None,
            _has_temporal=True, _has_reranker=False, _has_salience=False,
            search=Mock(return_value=copy.deepcopy(rows)), _entity_focused_search=lambda *args: [],
            _check_sufficiency=lambda rows: True, _apply_surprise_boost=lambda rows: rows,
            _clean_results=agentic.clean_results)
        with provenance_interface(), patch.dict(sys.modules, {"truememory.agentic_search": agentic}):
            result = consumer.search_agentic(instance, query, limit=10, use_hyde=False, use_clustering=False)
        self.assertEqual(result[0]["temporal_resolution"]["status"], "resolved")
        self.assertEqual(instance.search.call_args.kwargs["limit"], 30)
        self.assertEqual(result[0]["metadata"], {})

    def test_actual_agentic_entity_boost_uses_individual_intervals(self) -> None:
        consumer = load_functions("engine", {"search_agentic"}, {
            "resolve_temporal_intent": TEMPORAL.resolve_temporal_intent,
            "attach_temporal_resolution": TEMPORAL.attach_temporal_resolution,
            "logger": logging.getLogger("synthetic-landmark"), "os": os,
        })
        agentic = load_functions("agentic_search", {"normalize_scores", "clean_results"}, {})
        rows = [{"id": 101, "content": "synthetic source fact", "timestamp": "2025-09-01", "score": 0.2,
                 "source": "fts", "metadata": metadata_intervals(
                     ("2025-01-01T00:00:00+00:00", "2025-01-02T00:00:00+00:00"),
                     ("2025-09-01T00:00:00+00:00", "2025-09-02T00:00:00+00:00"))}]
        for month, score in [("January", 0.5 * 1.5), ("June", 0.5)]:
            with self.subTest(month=month):
                instance = types.SimpleNamespace(conn=self.conn, _ensure_connection=lambda: None,
                    _has_temporal=True, _has_reranker=False, _has_salience=False,
                    search=Mock(return_value=copy.deepcopy(rows)), _entity_focused_search=lambda *args: copy.deepcopy(rows),
                    _check_sufficiency=lambda rows: True, _apply_surprise_boost=lambda rows: rows,
                    _clean_results=agentic.clean_results)
                with provenance_interface(), patch.dict(sys.modules, {
                    "truememory.agentic_search": agentic, "truememory.temporal": TEMPORAL,
                }):
                    result = consumer.search_agentic(instance, "What happened in " + month + " 2025?", limit=10,
                                                     use_hyde=False, use_clustering=False)
                self.assertEqual(result[0]["score"], score)

    def test_actual_engine_rescope_rejects_disjoint_gap_and_unknown_coverage(self) -> None:
        self.event()
        primary = [{"id": 101, "content": "synthetic primary", "sender": "synthetic-alice",
                    "timestamp": "2025-06-20", "score": 0.2}]
        scoped = [{"id": 102, "content": "synthetic in-window", "timestamp": "2025-06-20", "score": 0.2,
                   "metadata": metadata_intervals(("2025-06-20T00:00:00+00:00", "2025-06-20T00:00:00+00:00"))},
                  {"id": 103, "content": "synthetic unrelated gap", "timestamp": "2025-06-20", "score": 0.2,
                   "metadata": metadata_intervals(("2025-01-01T00:00:00+00:00", "2025-01-02T00:00:00+00:00"),
                                                  ("2025-09-01T00:00:00+00:00", "2025-09-02T00:00:00+00:00"))},
                  {"id": 104, "content": "synthetic unknown scope", "timestamp": "2025-06-20", "score": 0.2,
                   "metadata": metadata_intervals(unknown=True)}]
        consumer = load_functions("engine", {"search"}, {
            "_HAS_QUERY_CLASSIFIER": False, "search_hybrid": lambda *args, **kwargs: copy.deepcopy(primary),
            "resolve_temporal_intent": TEMPORAL.resolve_temporal_intent,
            "search_temporal": TEMPORAL.search_temporal, "attach_temporal_resolution": TEMPORAL.attach_temporal_resolution,
            "_source_intervals": TEMPORAL._source_intervals, "_temporal_row_matches": TEMPORAL._temporal_row_matches,
            "_exclusive_upper_bound": TEMPORAL._exclusive_upper_bound,
            "logger": logging.getLogger("synthetic-landmark"), "os": os,
        })
        fts = types.ModuleType("truememory.fts_search")
        fts.search_fts_in_range = lambda *args, **kwargs: copy.deepcopy(scoped)
        instance = types.SimpleNamespace(conn=self.conn, _ensure_connection=lambda: None,
            _has_hybrid=True, _has_temporal=True, _has_personality=False, _has_consolidation=False,
            _has_salience=False, _has_reranker=False, _apply_surprise_boost=lambda rows: rows)
        with provenance_interface(), patch.dict(sys.modules, {"truememory.fts_search": fts}):
            result = consumer.search(instance, "What happened in the first month after Demo Day?", limit=10)
        self.assertIn(102, {row["id"] for row in result})
        self.assertNotIn(103, {row["id"] for row in result})
        self.assertNotIn(104, {row["id"] for row in result})


class BoundaryControls(unittest.TestCase):
    def test_original_parser_and_landmark_writer_ast_unchanged(self) -> None:
        current = (ROOT / "truememory/temporal.py").read_text()
        baseline_hashes = {
            "detect_temporal_intent": "2a752baf88d85db0eb61b91c087284c4e02dd3d4b89dece12bc5bf9e04ded287",
            "detect_landmark_events": "d3bbd201a06107765f0b80448f5cec8d2dde800d54f20e84cd442c137b91fe9c",
            "parse_date_reference": "8f8d13024c23b012ee9a960992060f7abf77d74939cb8000110f55f599df98ad",
        }
        for name, baseline_hash in baseline_hashes.items():
            with self.subTest(name=name):
                node = next(node for node in ast.parse(current).body if isinstance(node, ast.FunctionDef) and node.name == name)
                self.assertEqual(hashlib.sha256(ast.get_source_segment(current, node).encode()).hexdigest(), baseline_hash)

    def test_client_scope_threading_and_existing_limits(self) -> None:
        client = load_functions("client", {"search", "search_deep", "resolve_temporal_query"}, {})
        engine = types.SimpleNamespace(search=Mock(return_value=[]), search_agentic=Mock(return_value=[]))
        memory = types.SimpleNamespace(_engine=engine)
        client.search(memory, "after Demo Day", user_id="synthetic-alice", limit=10)
        self.assertEqual(engine.search.call_args.kwargs["limit"], 30)
        self.assertEqual(engine.search.call_args.kwargs["landmark_sender"], "synthetic-alice")
        client.search_deep(memory, "after Demo Day", limit=100)
        self.assertEqual(engine.search_agentic.call_args.kwargs["limit"], 100)
        self.assertNotIn("landmark_sender", engine.search_agentic.call_args.kwargs)

    def test_client_resolution_exposes_bounded_policy_without_retrieval(self) -> None:
        client = load_functions("client", {"resolve_temporal_query"}, {})
        connection = sqlite3.connect(":memory:")
        self.addCleanup(connection.close)
        engine = types.SimpleNamespace(conn=connection, _ensure_connection=lambda: None)
        with patch.dict(sys.modules, {"truememory.temporal": TEMPORAL}):
            intent = client.resolve_temporal_query(types.SimpleNamespace(_engine=engine), "What happened in January 2025?")
        self.assertEqual(intent["source_interval_lookup_policy"]["candidate_limit"], 1024)
        self.assertFalse(intent["source_interval_lookup_policy"]["complete_archive"])

    def test_mcp_default_lists_opt_in_empty_status_and_internal_limits(self) -> None:
        diagnostics = {"landmark_resolution": {"status": "ambiguous", "reason": "multiple_sources"}}
        memory = types.SimpleNamespace(search_deep=Mock(return_value=[]), resolve_temporal_query=Mock(return_value=diagnostics))
        operation = types.SimpleNamespace(reranker_id="unchanged", tier="free")

        @contextmanager
        def search_operation(**kwargs):
            yield memory, operation

        mcp = load_functions("mcp_server", {"truememory_search", "truememory_search_deep"}, {
            "json": json, "_touch_search_time": lambda: None, "_search_operation": search_operation,
            "_set_reranker": lambda _: None, "_SEARCH_INTERNAL_LIMIT": 100, "_DEEP_INTERNAL_LIMIT": 500,
            "_DEEP_RERANKER": "unchanged-deep", "_resolve_deepsearch_llm": lambda _: None,
            "_parallel_search": lambda *args, **kwargs: [],
        })
        for name, limit in [("truememory_search", 100), ("truememory_search_deep", 500)]:
            with self.subTest(name=name):
                function = getattr(mcp, name)
                self.assertEqual(json.loads(function("after Demo Day")), [])
                self.assertEqual(memory.search_deep.call_args.kwargs["limit"], limit)
                result = json.loads(function("after Demo Day", user_id="synthetic-alice", include_temporal_resolution=True))
                self.assertEqual(result["results"], [])
                self.assertEqual(result["temporal_resolution"]["landmark_resolution"]["status"], "ambiguous")
                self.assertEqual(memory.resolve_temporal_query.call_args.kwargs["user_id"], "synthetic-alice")
                self.assertEqual(len(json.loads(function("after Demo Day | before Launch", include_temporal_resolution=True))["temporal_resolutions"]), 2)

    def test_tools_count_and_empty_query_unchanged_by_default(self) -> None:
        source = (ROOT / "truememory/mcp_server.py").read_text()
        tree = ast.parse(source)
        tools = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("truememory_")]
        self.assertEqual(len(tools), 11)
        functions = load_functions("mcp_server", {"truememory_search", "truememory_search_deep"}, {"json": json})
        self.assertEqual(json.loads(functions.truememory_search("")), [])
        self.assertEqual(json.loads(functions.truememory_search_deep("", include_temporal_resolution=True))["results"], [])

    def test_engine_source_all_consumers_and_post_clean_diagnostics(self) -> None:
        tree = ast.parse((ROOT / "truememory/engine.py").read_text())
        engine = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrueMemoryEngine")
        methods = {node.name: node for node in engine.body if isinstance(node, ast.FunctionDef)}
        search_calls = [node for node in ast.walk(methods["search"]) if isinstance(node, ast.Call)]
        self.assertTrue(any(isinstance(node.func, ast.Name) and node.func.id == "resolve_temporal_intent" for node in search_calls))
        temporal_calls = [node for node in search_calls if isinstance(node.func, ast.Name) and node.func.id == "search_temporal"]
        self.assertEqual(len(temporal_calls), 2)
        self.assertTrue(all(any(keyword.arg == "resolved_intent" for keyword in node.keywords) for node in temporal_calls))
        agentic_calls = [node for node in ast.walk(methods["search_agentic"]) if isinstance(node, ast.Call)]
        self.assertTrue(any(isinstance(node.func, ast.Name) and node.func.id == "_temporal_row_matches" for node in agentic_calls))
        self.assertEqual(sum(isinstance(node.func, ast.Name) and node.func.id == "attach_temporal_resolution" for node in agentic_calls), 2)
        pool_assignment = next(node for node in ast.walk(methods["search_agentic"]) if isinstance(node, ast.Assign)
                               and any(isinstance(target, ast.Name) and target.id == "candidate_pool" for target in node.targets))
        self.assertIn("max(limit * 8, 100)", ast.unparse(pool_assignment))


if __name__ == "__main__":
    unittest.main()
