"""Safe architecture probes against frozen source; stdlib only, synthetic data.

AST-selected production functions run unchanged. Heavy modules are never imported.
External dependencies are explicitly faked where named. These probes establish
control flow and SQLite behavior, not model quality, device memory, or latency.
"""

from __future__ import annotations

import ast
import contextlib
import datetime
import hashlib
import json
import logging
import math
import pathlib
import re
import sqlite3
import struct
import sys
import tempfile
import time
import types
from collections import defaultdict
from typing import Callable

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else pathlib.Path.cwd()
COMMIT = "063e5b8844af735a52fde886217a5d26a0f13064"


def selected(path: str, names: set[str], namespace: dict[str, object]) -> dict[str, object]:
    tree = ast.parse((SOURCE / path).read_text())
    nodes = []
    for node in tree.body:
        assigned = {t.id for t in node.targets if isinstance(t, ast.Name)} if isinstance(node, ast.Assign) else set()
        if getattr(node, "name", None) in names or assigned & names:
            nodes.append(node)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE / path), "exec"), namespace)
    return namespace


def output(name: str, result: dict[str, object]) -> None:
    print(json.dumps({"probe": name, **result}, sort_keys=True))


def store_metadata_probe() -> None:
    ns = selected("truememory/ingest/pipeline.py", {"IngestionPipeline"}, {
        "datetime": datetime, "sqlite3": sqlite3, "log": logging.getLogger("repro"),
    })
    calls = []
    class Engine:
        def add(self, **kwargs):
            calls.append(kwargs)
    obj = ns["IngestionPipeline"].__new__(ns["IngestionPipeline"])
    obj.memory = types.SimpleNamespace(_engine=Engine())
    obj.user_id = "synthetic-user"
    fact = types.SimpleNamespace(category="preference", confidence=0.9)
    obj._store_fact("Prefers blue", fact, "synthetic-session-2020")
    record = calls[0]
    output("store_fact_metadata", {
        "content": record["content"], "sender": record["sender"],
        "timestamp_is_ingestion_year": str(record["timestamp"]).startswith(str(datetime.datetime.now(datetime.timezone.utc).year)),
        "passed_keys": sorted(record), "session_id_present": "session_id" in record,
        "metadata_present": "metadata" in record, "recipient_present": "recipient" in record,
    })


def preference_probe() -> None:
    queries = []
    class NeverRead:
        def execute(self, *args):
            queries.append(args)
            raise AssertionError("unexpected read")
    ns = selected("truememory/personality.py", {"extract_preferences"}, {"sqlite3": sqlite3})
    result = ns["extract_preferences"](NeverRead())
    output("scheduled_preferences_default", {"result": result, "sql_calls": len(queries)})


def style_probe() -> None:
    ns = selected("truememory/personality_style_vec.py", {
        "update_entity_style_vector_incremental", "mean_pool_vectors", "get_entity_style_vector",
    }, {"sqlite3": sqlite3, "json": json, "math": math, "datetime": datetime.datetime,
        "timezone": datetime.timezone, "DIM": 256})
    e1, e2 = [1.0, 0.0], [0.0, 1.0]
    def run(order):
        conn = sqlite3.connect(":memory:")
        for vec in order:
            ns["update_entity_style_vector_incremental"](conn, "alice", "synthetic", _pre_computed_vec=vec)
        result = ns["get_entity_style_vector"](conn, "alice")
        conn.close()
        return result
    batch = ns["mean_pool_vectors"]([e1, e2, e2])
    first, second = run([e1, e2, e2]), run([e2, e2, e1])
    output("incremental_style_mean", {"batch": batch, "incremental_e1_e2_e2": first,
        "incremental_e2_e2_e1": second, "max_difference": max(abs(a-b) for a,b in zip(batch, first)),
        "order_invariant": first == second})


def cluster_probe(temp: pathlib.Path) -> None:
    path = temp / "cluster.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE messages(id INTEGER PRIMARY KEY); CREATE TABLE probe(id INTEGER);")
    ns = selected("truememory/clustering.py", {"_CLUSTER_SCHEMA", "_init_cluster_tables", "cluster_messages"}, {
        "sqlite3": sqlite3, "defaultdict": defaultdict,
    })
    ns["_init_cluster_tables"](conn)
    conn.execute("INSERT INTO message_clusters VALUES (1, 9, 0)")
    conn.execute("INSERT INTO cluster_centroids(cluster_id, centroid) VALUES (9, X'00')")
    conn.commit()
    state = {}
    def probe_embeddings(active):
        state["in_transaction_during_embedding_read"] = active.in_transaction
        other = sqlite3.connect(path, timeout=0.02)
        try:
            other.execute("INSERT INTO probe VALUES (1)")
            other.commit()
            state["other_writer"] = "succeeded"
        except sqlite3.OperationalError as exc:
            state["other_writer"] = str(exc)
        finally:
            other.close()
        raise RuntimeError("synthetic compute failure")
    ns["_get_all_embeddings"] = probe_embeddings
    old = sys.modules.get("hdbscan")
    sys.modules["hdbscan"] = types.ModuleType("hdbscan")
    try:
        ns["cluster_messages"](conn)
    except RuntimeError:
        state["open_transaction_after_failure"] = conn.in_transaction
        conn.commit()  # engine.consolidate() ultimately commits after caught errors.
        state["cluster_rows_after_caller_commit"] = conn.execute("SELECT COUNT(*) FROM message_clusters").fetchone()[0]
    finally:
        if old is None:
            sys.modules.pop("hdbscan", None)
        else:
            sys.modules["hdbscan"] = old
        conn.close()
    output("cluster_transaction", state)


def episode_probe() -> None:
    ns = selected("truememory/storage.py", {"_SCHEMA_SQL"}, {})
    conn = sqlite3.connect(":memory:")
    conn.executescript(ns["_SCHEMA_SQL"])
    conn.executemany("INSERT INTO messages(content,sender,timestamp) VALUES (?,?,?)", [
        (f"Synthetic memory {n}", "alice", f"2026-01-01T{n:02d}:00:00") for n in range(12)
    ])
    conn.executescript("CREATE TABLE update_audit(same_content INTEGER); CREATE TRIGGER audit_updates AFTER UPDATE ON messages BEGIN INSERT INTO update_audit VALUES (new.content = old.content); END;")
    tns = selected("truememory/temporal.py", {"detect_episodes", "_parse_naive", "_TZ_SUFFIX_RE"}, {
        "datetime": datetime.datetime, "timedelta": datetime.timedelta, "re": re,
    })
    tns["detect_episodes"](conn)
    first = conn.execute("SELECT COUNT(*) FROM update_audit").fetchone()[0]
    tns["detect_episodes"](conn)
    total, same = conn.execute("SELECT COUNT(*), SUM(same_content) FROM update_audit").fetchone()
    output("episode_fts_write_amplification", {"message_rows": 12, "updates_first_pass": first,
        "updates_second_unchanged_pass": total-first, "content_unchanged_updates_total": same,
        "fts_rows_after": conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0],
        "fts_delete_insert_pairs_per_pass_by_unconditional_trigger": first})
    conn.close()


def surprise_probe(temp: pathlib.Path) -> None:
    path = temp / "surprise.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE messages(id INTEGER PRIMARY KEY, content TEXT, timestamp TEXT); CREATE TABLE surprise_scores(message_id INTEGER PRIMARY KEY,surprise REAL,fact_count INTEGER,new_fact_count INTEGER); CREATE TABLE probe(id INTEGER); INSERT INTO messages VALUES(1,'synthetic fact','2026-01-01'); INSERT INTO surprise_scores VALUES(1,0.5,1,1);")
    state = {}
    def score(content, existing):
        state["in_transaction_during_scoring"] = conn.in_transaction
        other = sqlite3.connect(path, timeout=0.02)
        try:
            other.execute("INSERT INTO probe VALUES(1)")
            other.commit()
            state["other_writer"] = "succeeded"
        except sqlite3.OperationalError as exc:
            state["other_writer"] = str(exc)
        finally:
            other.close()
        raise RuntimeError("synthetic scoring failure")
    ns = selected("truememory/predictive.py", {"build_surprise_index"}, {
        "_ensure_surprise_table": lambda active: None, "extract_facts": lambda text: {text},
        "compute_surprise_score": score,
    })
    try:
        ns["build_surprise_index"](conn)
    except RuntimeError:
        state["open_transaction_after_failure"] = conn.in_transaction
        conn.commit()
        state["surprise_rows_after_caller_commit"] = conn.execute("SELECT COUNT(*) FROM surprise_scores").fetchone()[0]
    conn.close()
    output("surprise_transaction", state)


def tier_retry_probe() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript("CREATE TABLE vec_messages_basepro(rowid INTEGER PRIMARY KEY,embedding BLOB); CREATE TABLE vec_messages_sep_basepro(rowid INTEGER PRIMARY KEY,embedding BLOB);")
    ns = selected("truememory/tier_switch/worker.py", {"RebuildWorker"}, {
        "sqlite3": sqlite3, "time": time, "log": logging.getLogger("repro"),
        "_HARD_TIMEOUT": 9000,
    })
    worker = ns["RebuildWorker"](conn, "base", "basepro", None)
    class Model:
        calls = 0
        def encode(self, texts, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("MPS backend out of memory")
            return [[1.0, 0.0] for _ in texts]
    model = Model()
    batch = [{"id": 1, "content": "synthetic"}]
    state = {"vector_table_substitute": "ordinary SQLite tables with unique rowid; sqlite-vec not loaded"}
    try:
        worker._process_batch(batch, model, "vec_messages_basepro", "vec_messages_sep_basepro", lambda vec: b"embedding", lambda *parts: " ".join(parts))
    except RuntimeError as exc:
        state["first_error_matches_worker_oom_branch"] = worker._is_oom_error(exc)
        state["in_transaction_after_oom"] = conn.in_transaction
    try:
        worker._process_batch(batch, model, "vec_messages_basepro", "vec_messages_sep_basepro", lambda vec: b"embedding", lambda *parts: " ".join(parts))
    except sqlite3.IntegrityError as exc:
        state["retry_error"] = str(exc)
    state["completion_rows"] = conn.execute("SELECT COUNT(*) FROM vec_messages_basepro").fetchone()[0]
    state["separation_rows"] = conn.execute("SELECT COUNT(*) FROM vec_messages_sep_basepro").fetchone()[0]
    conn.close()
    output("tier_partial_batch_retry", state)


def startup_and_update_probe() -> None:
    threads = []
    class FakeThread:
        def __init__(self, **kwargs):
            threads.append(kwargs)
        def start(self):
            pass
    ns = selected("truememory/engine.py", {"TrueMemoryEngine"}, {
        "threading": types.SimpleNamespace(Thread=FakeThread), "sqlite3": sqlite3,
    })
    cls = ns["TrueMemoryEngine"]
    conn = sqlite3.connect(":memory:")
    conn.executescript("CREATE TABLE messages(id INTEGER PRIMARY KEY,content TEXT,sender TEXT); CREATE TABLE message_clusters(message_id INTEGER); CREATE TABLE entity_profiles(entity TEXT, traits TEXT); INSERT INTO entity_profiles VALUES('alice','old-trait');")
    conn.executemany("INSERT INTO messages VALUES (?,?,?)", [(n, "old-fact", "alice") for n in range(25)])
    conn.commit()
    for unused in range(2):
        obj = cls.__new__(cls)
        obj.conn = conn
        obj._has_consolidation = True
        obj._auto_consolidate_threshold = 25
        obj._maybe_startup_consolidate()
    output("duplicate_startup_maintenance", {"engine_instances": 2, "scheduled_threads": len(threads), "actual_threads_started": 0})
    sns = selected("truememory/storage.py", {"update_message"}, {})
    ns["update_message"] = sns["update_message"]
    obj._ensure_connection = lambda: None
    obj._has_vectors = False
    obj._write_lock = contextlib.nullcontext()
    obj.get = lambda message_id: conn.execute("SELECT content,sender FROM messages WHERE id=?", (message_id,)).fetchone()
    result = obj.update(0, content="entirely-new-fact", sender="bob")
    output("derived_profile_after_update", {"updated_message": result, "profiles": conn.execute("SELECT entity,traits FROM entity_profiles").fetchall()})
    conn.close()


def temporal_landmark_probe() -> None:
    ns = selected("truememory/temporal.py", {"_MONTH_NAMES", "_MONTH_PATTERN", "parse_date_reference", "_end_of_month", "detect_temporal_intent"}, {
        "re": re, "datetime": datetime.datetime, "timedelta": datetime.timedelta,
    })
    unnamed = ns["detect_temporal_intent"]("What happened in the month after Demo Day?")
    named = ns["detect_temporal_intent"]("What happened in the month after Demo Day (June 15, 2025)?")
    output("temporal_landmark_resolution", {"named_event_without_date": unnamed, "same_event_with_date": named,
        "database_consulted": False})


def main() -> None:
    output("scope", {"commit": COMMIT, "heavy_dependencies_imported": False, "personal_data_accessed": False})
    store_metadata_probe()
    preference_probe()
    style_probe()
    startup_and_update_probe()
    temporal_landmark_probe()
    with tempfile.TemporaryDirectory(prefix="architecture-repro-") as directory:
        temp = pathlib.Path(directory)
        cluster_probe(temp)
        episode_probe()
        surprise_probe(temp)
        tier_retry_probe()
    output("finished", {"torch_imported": "torch" in sys.modules, "numpy_imported": "numpy" in sys.modules,
        "truememory_package_imported": "truememory" in sys.modules})


if __name__ == "__main__":
    main()
