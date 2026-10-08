"""A04 raw-sum regressions using stdlib source execution and in-memory SQLite."""
from __future__ import annotations

import ast
import builtins
import itertools
import json
import logging
import math
import random
import sqlite3
import struct
import sys
import threading
import types
import unittest
import uuid
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_stdlib_source(name: str) -> types.ModuleType:
    def safe_import(name: str, globals: dict | None = None, locals: dict | None = None,
                    fromlist: tuple[str, ...] = (), level: int = 0) -> object:
        if name.split(".", 1)[0] not in sys.stdlib_module_names:
            raise AssertionError("Non-stdlib import forbidden: " + name)
        return builtins.__import__(name, globals, locals, fromlist, level)

    module = types.ModuleType("synthetic_style_" + name)
    module.__dict__["__builtins__"] = dict(vars(builtins), __import__=safe_import)
    source = ROOT / "truememory" / (name + ".py")
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), module.__dict__)
    return module


STYLE = load_stdlib_source("personality_style_vec")
STORAGE = load_stdlib_source("storage")
LEGACY_SCHEMA = """CREATE TABLE entity_style_vectors (
    entity TEXT PRIMARY KEY, vector TEXT, message_count INTEGER DEFAULT 0, updated_at TEXT
)"""
SOURCES = """CREATE TABLE messages (
    id INTEGER PRIMARY KEY, sender TEXT, content TEXT, timestamp TEXT, directive INTEGER
); CREATE TABLE sentinel (value TEXT);"""

# Unchanged updater body from dependency base 8d7249c. It is executed with
# synthetic vectors to prove the old behavior fails the new numeric contract.
LEGACY_INCREMENTAL = '''def update_entity_style_vector_incremental(
    conn: sqlite3.Connection, entity: str, new_message: str,
    *, _pre_computed_vec: list[float] | None = None,
) -> None:
    """Incrementally update an entity's style vector with a new message.

    Uses a running weighted average: given the existing mean vector and
    its message count, the new mean is::

        new_mean = (existing * count + new_vec) / (count + 1)

    Then re-L2-normalized.

    Args:
        conn:        Open database connection.
        entity:      Entity name (sender).
        new_message: The new message text.
        _pre_computed_vec: If provided, skip compute_style_vector() call.
    """
    if not entity or not new_message:
        return

    # Normalize entity name to lowercase for case-insensitive matching (#467)
    entity = entity.lower()

    conn.execute(
        """CREATE TABLE IF NOT EXISTS entity_style_vectors (
            entity TEXT PRIMARY KEY,
            vector TEXT,
            message_count INTEGER DEFAULT 0,
            updated_at TEXT
        )"""
    )

    new_vec = _pre_computed_vec if _pre_computed_vec is not None else compute_style_vector(new_message)
    now = datetime.now(timezone.utc).isoformat()

    row = conn.execute(
        "SELECT vector, message_count FROM entity_style_vectors WHERE entity = ?",
        (entity,),
    ).fetchone()

    if row is None or row[0] is None:
        conn.execute(
            """INSERT OR REPLACE INTO entity_style_vectors
               (entity, vector, message_count, updated_at)
               VALUES (?, ?, ?, ?)""",
            (entity, json.dumps(new_vec), 1, now),
        )
    else:
        existing_vec = json.loads(row[0])
        count = row[1] or 0

        new_count = count + 1
        merged = [
            (existing_vec[i] * count + new_vec[i]) / new_count
            for i in range(len(existing_vec))
        ]

        norm = math.sqrt(sum(x * x for x in merged))
        if norm > 0:
            merged = [x / norm for x in merged]

        conn.execute(
            """INSERT OR REPLACE INTO entity_style_vectors
               (entity, vector, message_count, updated_at)
               VALUES (?, ?, ?, ?)""",
            (entity, json.dumps(merged), new_count, now),
        )
'''


def axis(index: int) -> list[float]:
    return [float(position == index) for position in range(256)]


def reference_mean(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return [0.0] * 256
    total = [0.0] * len(vectors[0])
    for vector in vectors:
        for index in range(len(total)):
            total[index] += vector[index]
    mean = [value / len(vectors) for value in total]
    norm = math.sqrt(sum(value * value for value in mean))
    return [value / norm for value in mean] if norm > 0 else mean


def allow_authorized_action(*_args: object) -> int:
    return sqlite3.SQLITE_OK


class StatementFailure:
    """Deny a single execution boundary, independent of SQLite's statement cache."""

    def __init__(self, conn: sqlite3.Connection, sql: str, denied_action: tuple[int, str, str | None],
                 *, occurrence: int = 1) -> None:
        self.conn = conn
        self.sql = sql
        self.denied_action = denied_action
        self.occurrence = occurrence
        self.matches = 0
        self.armed: list[str] = []

    def __getattr__(self, name: str) -> object:
        return getattr(self.conn, name)

    def authorize(self, action: int, first: str | None, second: str | None, *_rest: object) -> int:
        return sqlite3.SQLITE_DENY if (action, first, second) == self.denied_action else sqlite3.SQLITE_OK

    def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
        if sql == self.sql:
            self.matches += 1
            if self.matches == self.occurrence:
                self.armed.append(sql)
                # Install outside the pure callback so cached statements must
                # reauthorize; reset before the production rollback/RELEASE.
                self.conn.set_authorizer(self.authorize)
                try:
                    return self.conn.execute(sql, *args)
                finally:
                    self.conn.set_authorizer(allow_authorized_action)
        return self.conn.execute(sql, *args)


class StyleAccumulatorTests(unittest.TestCase):
    def connection(self, *, legacy: bool = False, full: bool = False, cached_statements: int = 100) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:", timeout=0, cached_statements=cached_statements)
        self.addCleanup(conn.close)
        if legacy:
            conn.execute(LEGACY_SCHEMA)
        conn.executescript(STORAGE._SCHEMA_SQL if full else SOURCES)
        if full:
            conn.execute("CREATE TABLE sentinel (value TEXT)")
            STORAGE._initialize_maintenance_tracking(conn)
        conn.commit()
        return conn

    def columns(self, conn: sqlite3.Connection) -> list[tuple]:
        return [tuple(row) for row in conn.execute("PRAGMA table_info(entity_style_vectors)")]

    def row(self, conn: sqlite3.Connection, entity: str = "synthetic") -> tuple | None:
        return conn.execute("SELECT * FROM entity_style_vectors WHERE entity=?", (entity,)).fetchone()

    def seed_legacy(self, conn: sqlite3.Connection) -> tuple:
        conn.execute("INSERT INTO entity_style_vectors(entity,vector,message_count,updated_at) VALUES (?,?,?,?)",
                     ("synthetic", json.dumps(axis(0)), 9, "synthetic-old-generation"))
        conn.commit()
        return self.row(conn)

    def append(self, conn: sqlite3.Connection, vector: list[float], *, entity: str = "synthetic") -> None:
        STYLE.update_entity_style_vector_incremental(conn, entity, "synthetic message", _pre_computed_vec=vector)

    def assert_close(self, expected: list[float], actual: list[float]) -> None:
        self.assertEqual(len(expected), 256)
        self.assertEqual(len(actual), 256)
        self.assertTrue(all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(expected, actual)))

    def test_permutations_match_batch_and_preserve_raw_sum(self) -> None:
        vectors = [axis(0), axis(1), axis(1)]
        for order in itertools.permutations(vectors):
            conn = self.connection()
            for vector in order:
                self.append(conn, vector)
            self.assert_close(reference_mean(vectors), STYLE.get_entity_style_vector(conn, "SYNTHETIC"))
            raw, count, version = conn.execute("SELECT vector_sum,message_count,accumulator_version FROM entity_style_vectors").fetchone()
            self.assertEqual((len(raw), count, version), (2048, 3, 1))
            self.assertEqual(struct.unpack("<256d", raw), (1.0, 2.0) + (0.0,) * 254)
            self.assertTrue(conn.in_transaction)

    def test_seeded_orders_match_reference(self) -> None:
        rng = random.Random(747)
        vectors = []
        for _ in range(128):
            vector = [rng.random() for _ in range(256)]
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector])
        for count in (1, 3, 31, 128):
            selected = vectors[:count]
            shuffled = list(selected)
            rng.shuffle(shuffled)
            for order in (selected, list(reversed(selected)), shuffled):
                conn = self.connection()
                for vector in order:
                    self.append(conn, vector)
                self.assert_close(reference_mean(selected), STYLE.get_entity_style_vector(conn, "synthetic"))

    def test_old_code_negative_controls(self) -> None:
        namespace = dict(STYLE.__dict__)
        exec(LEGACY_INCREMENTAL, namespace)
        old = namespace["update_entity_style_vector_incremental"]
        for vectors in ([axis(0), axis(1), axis(1)], [[0.0] * 256, axis(0), axis(1)]):
            conn = self.connection()
            for vector in vectors:
                old(conn, "synthetic", "synthetic message", _pre_computed_vec=vector)
            actual = STYLE.get_entity_style_vector(conn, "synthetic")
            self.assertGreater(max(abs(a - b) for a, b in zip(actual, reference_mean(vectors))), 0.05)
        conn = self.connection(legacy=True)
        before = self.seed_legacy(conn)
        old(conn, "synthetic", "synthetic message", _pre_computed_vec=axis(1))
        self.assertNotEqual(self.row(conn), before)

    def test_zero_and_noop_semantics(self) -> None:
        for vectors in ([[0.0] * 256], [[0.0] * 256, axis(0), axis(1)],
                        [axis(0), [0.0] * 256, axis(1)], [axis(0), axis(1), [0.0] * 256]):
            conn = self.connection()
            for vector in vectors:
                self.append(conn, vector)
            self.assert_close(reference_mean(vectors), STYLE.get_entity_style_vector(conn, "synthetic"))
            self.assertEqual(conn.execute("SELECT message_count FROM entity_style_vectors").fetchone()[0], len(vectors))
        conn = self.connection()
        STYLE.update_entity_style_vector_incremental(conn, "", "message")
        STYLE.update_entity_style_vector_incremental(conn, "synthetic", "", _pre_computed_vec=axis(0))
        self.assertEqual(self.columns(conn), [])
        self.assertFalse(conn.in_transaction)
        for text in ("a", "ab", " \t\n "):
            STYLE.update_entity_style_vector_incremental(conn, "synthetic", text)
        self.assertEqual(conn.execute("SELECT message_count FROM entity_style_vectors").fetchone()[0], 3)
        self.assertEqual(STYLE.get_entity_style_vector(conn, "synthetic"), [0.0] * 256)

    def test_backup_restart_between_additions(self) -> None:
        vectors = [axis(0), axis(1), axis(1)]
        conn = self.connection()
        for vector in vectors:
            self.append(conn, vector)
            conn.commit()
            before = self.row(conn)
            restored = sqlite3.connect(":memory:")
            self.addCleanup(restored.close)
            conn.backup(restored)
            conn.close()
            conn = restored
            self.assertEqual(self.row(conn), before)
            loaded_again = load_stdlib_source("personality_style_vec")
            self.assertEqual(loaded_again.get_entity_style_vector(conn, "synthetic"), json.loads(before[1]))
        self.assert_close(reference_mean(vectors), STYLE.get_entity_style_vector(conn, "synthetic"))

    def test_batch_arithmetic_and_append_after_rebuild(self) -> None:
        conn = self.connection(legacy=True)
        texts = ["please deploy the blue service", "I prefer quiet mornings and coffee", "another synthetic sentence"]
        for index, text in enumerate(texts):
            conn.execute("INSERT INTO messages VALUES (?,?,?,?,?)", (index + 1, "Synthetic", text, str(index), 0))
        conn.commit()
        result = STYLE.build_entity_style_vectors(conn)
        vectors = [STYLE.compute_style_vector(text) for text in texts]
        self.assertEqual(result["synthetic"], reference_mean(vectors))
        self.assertEqual(result["synthetic"], STYLE.mean_pool_vectors(vectors))
        self.assertFalse(conn.in_transaction)
        extra = STYLE.compute_style_vector("one more synthetic message")
        self.append(conn, extra)
        self.assert_close(reference_mean(vectors + [extra]), STYLE.get_entity_style_vector(conn, "synthetic"))

    def test_batch_directives_counts_and_case_folding(self) -> None:
        conn = self.connection()
        rows = [(1, "Synthetic", "abc", "1", 0), (2, "SYNTHETIC", "", "2", None),
                (3, "Synthetic", "a", "3", 0), (4, "Synthetic", "directive text", "4", 1),
                (5, "Directive-only", "directive text", "5", 1), (6, "", "no sender", "6", 0)]
        conn.executemany("INSERT INTO messages VALUES (?,?,?,?,?)", rows)
        conn.commit()
        result = STYLE.build_entity_style_vectors(conn)
        self.assertEqual(set(result), {"synthetic"})
        self.assertEqual(conn.execute("SELECT message_count FROM entity_style_vectors").fetchone()[0], 3)
        self.assertEqual(result["synthetic"], reference_mean([STYLE.compute_style_vector(text) for text in ("", "abc", "a")]))
        self.assertEqual(STYLE.get_entity_style_vector(conn, "SYNTHETIC"), result["synthetic"])

    def test_batch_minimal_source_schema_and_empty_generation(self) -> None:
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        conn.execute("CREATE TABLE messages(sender TEXT, content TEXT, timestamp TEXT)")
        conn.execute("INSERT INTO messages VALUES ('Synthetic','abc','1')")
        conn.commit()
        self.assertEqual(STYLE.build_entity_style_vectors(conn), {"synthetic": reference_mean([STYLE.compute_style_vector("abc")])})
        conn.execute("DELETE FROM messages")
        conn.commit()
        self.assertEqual(STYLE.build_entity_style_vectors(conn), {})
        self.assertEqual(conn.execute("SELECT count(*) FROM entity_style_vectors").fetchone()[0], 0)

    def test_legacy_append_preserves_row_schema_and_outer_writes(self) -> None:
        for caller in (False, True):
            conn = self.connection(legacy=True)
            before = self.seed_legacy(conn)
            columns = self.columns(conn)
            if caller:
                conn.execute("INSERT INTO sentinel VALUES ('pending')")
                conn.execute("INSERT INTO messages VALUES (1,'Synthetic','source survives','1',0)")
            with self.assertRaises(STYLE.StyleVectorRebuildRequired) as raised:
                self.append(conn, axis(1))
            self.assertEqual(raised.exception.reason, "legacy")
            self.assertEqual(self.row(conn), before)
            self.assertEqual(self.columns(conn), columns)
            self.assertEqual(conn.in_transaction, caller)
            conn.commit()
            self.assertEqual(conn.execute("SELECT count(*) FROM sentinel").fetchone()[0], int(caller))
            self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], int(caller))
            self.assertEqual(STYLE.get_entity_style_vector(conn, "synthetic"), axis(0))

    def test_bad_persisted_accumulators_defer_without_writes(self) -> None:
        raw = struct.pack("<256d", *axis(0))
        cases = [(None, 1, 0, "legacy"), (raw, 1, 0, "legacy"), (raw, 1, 2, "unsupported_version"),
                 (raw, 1, -1, "invalid"), (raw, 1, "bad", "invalid"), (None, 1, 1, "invalid"),
                 (raw[:-1], 1, 1, "invalid"), (raw.hex(), 1, 1, "invalid"), (raw, 0, 1, "invalid"),
                 (raw, None, 1, "invalid"), (raw, 1.5, 1, "invalid"), (raw, "bad", 1, "invalid"),
                 (struct.pack("<256d", float("nan"), *([0.0] * 255)), 1, 1, "invalid"),
                 (struct.pack("<256d", float("inf"), *([0.0] * 255)), 1, 1, "invalid"),
                 (struct.pack("<256d", 1e308, *([0.0] * 255)), 1, 1, "invalid")]
        for total, count, version, reason in cases:
            with self.subTest(count=count, version=version, reason=reason):
                conn = self.connection(full=True)
                conn.execute("INSERT INTO entity_style_vectors VALUES (?,?,?,?,?,?)",
                             ("synthetic", json.dumps(axis(0)), count, "unchanged", total, version))
                conn.commit()
                before = self.row(conn)
                with self.assertRaises(STYLE.StyleVectorRebuildRequired) as raised:
                    self.append(conn, axis(1))
                self.assertEqual(raised.exception.reason, reason)
                self.assertEqual(self.row(conn), before)
                self.assertFalse(conn.in_transaction)

    def test_invalid_input_fails_before_schema_or_writer(self) -> None:
        for vector in ([], [0.0] * 255, [0.0] * 257, [True] * 256, ["x"] * 256,
                       [float("nan")] * 256, [float("inf")] * 256, [10**1000] * 256):
            conn = self.connection()
            statements = []
            conn.set_trace_callback(statements.append)
            with self.assertRaises(ValueError):
                self.append(conn, vector)
            conn.set_trace_callback(None)
            self.assertEqual(statements, [])
            self.assertFalse(conn.in_transaction)
            self.assertEqual(self.columns(conn), [])

    def test_precomputed_vector_skips_hashing(self) -> None:
        conn = self.connection()
        with patch.object(STYLE, "compute_style_vector", side_effect=AssertionError("must use precomputed vector")):
            self.append(conn, axis(0))
        self.assertEqual(STYLE.get_entity_style_vector(conn, "synthetic"), axis(0))

    def test_append_success_retains_caller_commit_and_rollback(self) -> None:
        for caller in (False, True):
            conn = self.connection(legacy=True)
            columns = self.columns(conn)
            if caller:
                conn.execute("INSERT INTO sentinel VALUES ('pending')")
            self.append(conn, axis(0))
            self.assertTrue(conn.in_transaction)
            self.assertEqual(STYLE.get_entity_style_vector(conn, "synthetic"), axis(0))
            conn.rollback()
            self.assertEqual(self.columns(conn), columns)
            self.assertIsNone(self.row(conn))
            self.assertEqual(conn.execute("SELECT count(*) FROM sentinel").fetchone()[0], 0)

    def test_append_write_failure_rolls_back_its_generation(self) -> None:
        for caller in (False, True):
            conn = self.connection()
            self.append(conn, axis(0))
            conn.commit()
            before = self.row(conn)
            conn.execute("CREATE TRIGGER fail_style BEFORE INSERT ON entity_style_vectors BEGIN SELECT RAISE(ABORT,'synthetic failure'); END")
            if caller:
                conn.execute("INSERT INTO sentinel VALUES ('pending')")
            with self.assertRaisesRegex(sqlite3.IntegrityError, "synthetic failure"):
                self.append(conn, axis(1))
            self.assertEqual(conn.in_transaction, caller)
            conn.commit()
            self.assertEqual(self.row(conn), before)
            self.assertEqual(conn.execute("SELECT count(*) FROM sentinel").fetchone()[0], int(caller))

    def test_append_savepoint_and_release_failure_preserve_caller(self) -> None:
        for caller, operation, cache in itertools.product((False, True), ("BEGIN", "RELEASE"), (0, 5, 100)):
            conn = self.connection(cached_statements=cache)
            self.append(conn, axis(0))
            conn.commit()
            before = self.row(conn)
            if caller:
                conn.execute("INSERT INTO sentinel VALUES ('pending')")
            sql = ("SAVEPOINT" if operation == "BEGIN" else "RELEASE SAVEPOINT") + " truememory_style_append"
            probe = StatementFailure(conn, sql, (sqlite3.SQLITE_SAVEPOINT, operation, "truememory_style_append"))
            with self.subTest(caller=caller, operation=operation, cached_statements=cache):
                with self.assertRaisesRegex(sqlite3.DatabaseError, "not authorized"):
                    self.append(probe, axis(1))
                self.assertEqual(probe.armed, [sql])
            self.assertEqual(conn.in_transaction, caller)
            conn.commit()
            self.assertEqual(self.row(conn), before)
            self.assertEqual(conn.execute("SELECT count(*) FROM sentinel").fetchone()[0], int(caller))

    def test_storage_migration_preserves_legacy_and_source_counters(self) -> None:
        conn = self.connection(legacy=True, full=True)
        before = self.seed_legacy(conn)
        token = conn.execute("SELECT * FROM maintenance_source_state").fetchone()
        STORAGE._initialize_style_accumulator_schema(conn)
        self.assertEqual(self.row(conn), before + (None, 0))
        self.assertEqual(conn.execute("SELECT * FROM maintenance_source_state").fetchone(), token)
        self.assertFalse(conn.in_transaction)
        fields = {row[1]: row for row in self.columns(conn)}
        self.assertEqual(fields["vector_sum"][2:5], ("BLOB", 0, "NULL"))
        self.assertEqual(fields["accumulator_version"][2:5], ("INTEGER", 1, "0"))

    def test_create_db_initializes_only_format_for_legacy_profiles(self) -> None:
        conn = self.connection(legacy=True, full=True)
        before = self.seed_legacy(conn)
        conn.execute("INSERT INTO messages(content,sender) VALUES ('synthetic source','Synthetic')")
        conn.commit()
        token = conn.execute("SELECT * FROM maintenance_source_state").fetchone()
        with patch.object(STORAGE.sqlite3, "connect", return_value=conn):
            self.assertIs(STORAGE.create_db(":memory:"), conn)
        self.assertEqual(self.row(conn), before + (None, 0))
        self.assertEqual(conn.execute("SELECT * FROM maintenance_source_state").fetchone(), token)
        with self.assertRaises(STYLE.StyleVectorRebuildRequired) as raised:
            self.append(conn, axis(1))
        self.assertEqual(raised.exception.reason, "legacy")

    def test_partial_column_installation_uses_named_fields(self) -> None:
        for existing in ("vector_sum BLOB DEFAULT NULL", "accumulator_version INTEGER NOT NULL DEFAULT 0"):
            conn = self.connection(legacy=True)
            conn.execute("ALTER TABLE entity_style_vectors ADD COLUMN " + existing)
            conn.commit()
            STORAGE._initialize_style_accumulator_schema(conn)
            self.append(conn, axis(0))
            self.append(conn, axis(1))
            self.assertEqual(conn.execute("SELECT message_count,accumulator_version,length(vector_sum) FROM entity_style_vectors").fetchone(),
                             (2, 1, 2048))
            self.assert_close(reference_mean([axis(0), axis(1)]), STYLE.get_entity_style_vector(conn, "synthetic"))

    def test_repeated_schema_check_does_not_read_source_or_write(self) -> None:
        conn = self.connection(full=True)
        statements = []
        def authorize(action: int, one: str | None, *_rest: object) -> int:
            if action in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE, sqlite3.SQLITE_TRANSACTION):
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_READ and one in ("messages", "messages_fts", "entity_style_vectors"):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        conn.set_authorizer(authorize)
        conn.set_trace_callback(statements.append)
        try:
            STORAGE._initialize_style_accumulator_schema(conn)
        finally:
            conn.set_authorizer(allow_authorized_action)
            conn.set_trace_callback(None)
        self.assertEqual(statements, ["PRAGMA table_info(entity_style_vectors)"])

    def test_storage_schema_migration_rolls_back_with_caller(self) -> None:
        conn = self.connection(legacy=True)
        before = self.seed_legacy(conn)
        columns = self.columns(conn)
        conn.execute("INSERT INTO sentinel VALUES ('pending')")
        STORAGE._initialize_style_accumulator_schema(conn)
        self.assertTrue(conn.in_transaction)
        self.assertEqual(self.row(conn), before + (None, 0))
        conn.rollback()
        self.assertEqual(self.row(conn), before)
        self.assertEqual(self.columns(conn), columns)
        self.assertEqual(conn.execute("SELECT count(*) FROM sentinel").fetchone()[0], 0)

    def test_partial_schema_migration_failure_is_atomic(self) -> None:
        for caller, cache in itertools.product((False, True), (0, 5, 100)):
            conn = self.connection(legacy=True, cached_statements=cache)
            before = self.seed_legacy(conn)
            columns = self.columns(conn)
            if caller:
                conn.execute("INSERT INTO sentinel VALUES ('pending')")
            sql = "ALTER TABLE entity_style_vectors ADD COLUMN accumulator_version INTEGER NOT NULL DEFAULT 0"
            probe = StatementFailure(conn, sql, (sqlite3.SQLITE_ALTER_TABLE, "main", "entity_style_vectors"))
            statements: list[str] = []
            conn.set_trace_callback(statements.append)
            try:
                with self.subTest(caller=caller, cached_statements=cache):
                    with self.assertRaisesRegex(sqlite3.DatabaseError, "not authorized"):
                        STORAGE._initialize_style_accumulator_schema(probe)
                    self.assertEqual(probe.armed, [sql])
            finally:
                conn.set_trace_callback(None)
            self.assertIn("ALTER TABLE entity_style_vectors ADD COLUMN vector_sum BLOB DEFAULT NULL", statements)
            self.assertEqual(conn.in_transaction, caller)
            self.assertEqual(self.columns(conn), columns)
            conn.commit()
            self.assertEqual(self.row(conn), before)
            self.assertEqual(conn.execute("SELECT count(*) FROM sentinel").fetchone()[0], int(caller))

    def test_schema_commit_or_release_failure_restores_legacy_format(self) -> None:
        for caller, cache in itertools.product((False, True), (0, 5, 100)):
            conn = self.connection(legacy=True, cached_statements=cache)
            before = self.seed_legacy(conn)
            columns = self.columns(conn)
            if caller:
                conn.execute("INSERT INTO sentinel VALUES ('pending')")
            sql = "RELEASE SAVEPOINT truememory_style_schema" if caller else "COMMIT"
            denied_action = ((sqlite3.SQLITE_SAVEPOINT, "RELEASE", "truememory_style_schema") if caller else
                             (sqlite3.SQLITE_TRANSACTION, "COMMIT", None))
            probe = StatementFailure(conn, sql, denied_action)
            with self.subTest(caller=caller, cached_statements=cache):
                with self.assertRaisesRegex(sqlite3.DatabaseError, "not authorized"):
                    STORAGE._initialize_style_accumulator_schema(probe)
                self.assertEqual(probe.armed, [sql])
            self.assertEqual(conn.in_transaction, caller)
            self.assertEqual(self.columns(conn), columns)
            conn.commit()
            self.assertEqual(self.row(conn), before)
            self.assertEqual(conn.execute("SELECT count(*) FROM sentinel").fetchone()[0], int(caller))

    def test_batch_schema_creation_occurs_only_after_computation(self) -> None:
        conn = self.connection()
        conn.execute("INSERT INTO messages VALUES (1,'Synthetic','abc','1',0)")
        conn.commit()
        compute = STYLE.compute_style_vector
        observed = []
        def outside_writer(text: str) -> list[float]:
            observed.append((conn.in_transaction, self.columns(conn)))
            return compute(text)
        with patch.object(STYLE, "compute_style_vector", outside_writer):
            STYLE.build_entity_style_vectors(conn)
        self.assertEqual(observed, [(False, [])])
        self.assertEqual(conn.execute("SELECT accumulator_version,length(vector_sum) FROM entity_style_vectors").fetchone(), (1, 2048))

    def test_batch_partial_publication_rolls_back_schema_and_rows(self) -> None:
        for caller in (False, True):
            conn = self.connection(legacy=True)
            before = self.seed_legacy(conn)
            columns = self.columns(conn)
            conn.executemany("INSERT INTO messages VALUES (?,?,?,?,?)",
                             [(1, "a", "first content", "1", 0), (2, "b", "second content", "2", 0)])
            conn.commit()
            conn.execute("CREATE TRIGGER fail_style BEFORE INSERT ON entity_style_vectors WHEN new.entity='b' BEGIN SELECT RAISE(ABORT,'synthetic failure'); END")
            if caller:
                conn.execute("INSERT INTO sentinel VALUES ('pending')")
            with self.assertRaisesRegex(sqlite3.IntegrityError, "synthetic failure"):
                STYLE.build_entity_style_vectors(conn)
            self.assertEqual(conn.in_transaction, caller)
            self.assertEqual(self.columns(conn), columns)
            conn.commit()
            self.assertEqual(self.row(conn), before)
            self.assertEqual(conn.execute("SELECT count(*) FROM entity_style_vectors").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT count(*) FROM sentinel").fetchone()[0], int(caller))

    def test_batch_late_cancellation_retains_prior_accumulator_generation(self) -> None:
        conn = self.connection()
        conn.executemany("INSERT INTO messages VALUES (?,?,?,?,?)",
                         [(1, "a", "first", "1", 0), (2, "b", "second", "2", 0)])
        conn.commit()
        STYLE.build_entity_style_vectors(conn)
        before = conn.execute("SELECT * FROM entity_style_vectors ORDER BY entity").fetchall()
        conn.execute("INSERT INTO sentinel VALUES ('pending')")
        compute = STYLE.compute_style_vector
        calls = 0
        def cancelled(text: str) -> list[float]:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt("synthetic cancellation")
            return compute(text)
        with patch.object(STYLE, "compute_style_vector", cancelled):
            with self.assertRaises(KeyboardInterrupt):
                STYLE.build_entity_style_vectors(conn)
        conn.commit()
        self.assertEqual(conn.execute("SELECT * FROM entity_style_vectors ORDER BY entity").fetchall(), before)
        self.assertEqual(conn.execute("SELECT count(*) FROM sentinel").fetchone()[0], 1)

    def test_batch_source_change_rejects_before_schema_upgrade(self) -> None:
        conn = self.connection(legacy=True)
        before = self.seed_legacy(conn)
        columns = self.columns(conn)
        conn.execute("INSERT INTO messages VALUES (1,'Synthetic','original','1',0)")
        conn.commit()
        compute = STYLE.compute_style_vector
        def changed(text: str) -> list[float]:
            self.assertFalse(conn.in_transaction)
            conn.execute("UPDATE messages SET content='corrected' WHERE id=1")
            conn.commit()
            return compute(text)
        with patch.object(STYLE, "compute_style_vector", changed):
            with self.assertRaisesRegex(sqlite3.OperationalError, "L0 source changed"):
                STYLE.build_entity_style_vectors(conn)
        self.assertFalse(conn.in_transaction)
        self.assertEqual(self.columns(conn), columns)
        self.assertEqual(self.row(conn), before)

    def test_batch_commit_and_release_failure_restore_accumulator(self) -> None:
        for caller, cache in itertools.product((False, True), (0, 5, 100)):
            conn = self.connection(legacy=True, cached_statements=cache)
            before = self.seed_legacy(conn)
            columns = self.columns(conn)
            conn.execute("INSERT INTO messages VALUES (1,'Synthetic','replacement','1',0)")
            conn.commit()
            if caller:
                conn.execute("INSERT INTO sentinel VALUES ('pending')")
            sql = "RELEASE SAVEPOINT truememory_l0" if caller else "COMMIT"
            denied_action = ((sqlite3.SQLITE_SAVEPOINT, "RELEASE", "truememory_l0") if caller else
                             (sqlite3.SQLITE_TRANSACTION, "COMMIT", None))
            probe = StatementFailure(conn, sql, denied_action, occurrence=2)
            with self.subTest(caller=caller, cached_statements=cache):
                with self.assertRaisesRegex(sqlite3.DatabaseError, "not authorized"):
                    STYLE.build_entity_style_vectors(probe)
                self.assertEqual(probe.armed, [sql])
                self.assertEqual(probe.matches, 3 if caller else 2)
            self.assertEqual(conn.in_transaction, caller)
            self.assertEqual(self.columns(conn), columns)
            conn.commit()
            self.assertEqual(self.row(conn), before)
            self.assertEqual(conn.execute("SELECT count(*) FROM sentinel").fetchone()[0], int(caller))

    def test_batch_and_schema_do_not_advance_source_tracking(self) -> None:
        conn = self.connection(legacy=True, full=True)
        self.seed_legacy(conn)
        conn.execute("INSERT INTO messages(content,sender) VALUES ('ordinary content','Synthetic')")
        conn.commit()
        token = conn.execute("SELECT * FROM maintenance_source_state").fetchone()
        layers = conn.execute("SELECT * FROM maintenance_layers ORDER BY layer").fetchall()
        STORAGE._initialize_style_accumulator_schema(conn)
        STYLE.build_entity_style_vectors(conn)
        self.assertEqual(conn.execute("SELECT * FROM maintenance_source_state").fetchone(), token)
        self.assertEqual(conn.execute("SELECT * FROM maintenance_layers ORDER BY layer").fetchall(), layers)

    def test_two_connections_cannot_read_then_overwrite_one_another(self) -> None:
        uri = "file:synthetic-style-" + uuid.uuid4().hex + "?mode=memory&cache=shared"
        first = sqlite3.connect(uri, uri=True, timeout=0, cached_statements=100)
        second = sqlite3.connect(uri, uri=True, timeout=0, cached_statements=100)
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        self.append(first, axis(0))
        first.commit()
        self.append(first, axis(1))
        statements = []
        second.set_trace_callback(statements.append)
        with self.assertRaises(sqlite3.OperationalError):
            self.append(second, axis(1))
        second.set_trace_callback(None)
        self.assertFalse(second.in_transaction)
        self.assertFalse(any("SELECT vector_sum" in sql for sql in statements))
        first.commit()
        self.append(second, axis(1))
        second.commit()
        self.assertEqual(second.execute("SELECT message_count FROM entity_style_vectors").fetchone()[0], 3)
        self.assert_close(reference_mean([axis(0), axis(1), axis(1)]), STYLE.get_entity_style_vector(first, "synthetic"))

    def test_getter_still_reads_legacy_json_and_missing_rows(self) -> None:
        conn = self.connection(legacy=True)
        self.seed_legacy(conn)
        self.assertEqual(STYLE.get_entity_style_vector(conn, "SYNTHETIC"), axis(0))
        self.assertIsNone(STYLE.get_entity_style_vector(conn, "unknown"))
        self.assertEqual(len(self.columns(conn)), 4)

    def test_actual_engine_add_catches_style_deferral_and_keeps_source(self) -> None:
        rebuild = types.ModuleType("synthetic_rebuild")
        rebuild.__dict__.update(sqlite3=sqlite3, Iterator=Iterator, AbstractContextManager=AbstractContextManager,
                                contextmanager=contextmanager, nullcontext=nullcontext)
        tree = ast.parse((ROOT / "truememory/rebuild_source.py").read_text(encoding="utf-8"))
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "rebuild_transaction"]
        exec(compile(ast.Module(body=selected, type_ignores=[]), "rebuild_source.py", "exec"), rebuild.__dict__)

        maintenance = types.ModuleType("synthetic_append_maintenance")
        modules = {"truememory.personality_style_vec": STYLE, "truememory.rebuild_source": rebuild,
                   "truememory.maintenance": maintenance, "truememory.storage": STORAGE,
                   "truememory._platform": load_stdlib_source("_platform")}

        def safe_import(name: str, globals: dict | None = None, locals: dict | None = None,
                        fromlist: tuple[str, ...] = (), level: int = 0) -> object:
            if name in modules:
                return modules[name]
            if name.split(".", 1)[0] not in sys.stdlib_module_names:
                raise AssertionError("Unexpected import in engine.add: " + name)
            return builtins.__import__(name, globals, locals, fromlist, level)

        maintenance.__dict__["__builtins__"] = dict(vars(builtins), __import__=safe_import)
        path = ROOT / "truememory/maintenance.py"
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), maintenance.__dict__)
        tree = ast.parse((ROOT / "truememory/engine.py").read_text(encoding="utf-8"))
        engine_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrueMemoryEngine")
        add = next(node for node in engine_class.body if isinstance(node, ast.FunctionDef) and node.name == "add")
        namespace = {"__builtins__": dict(vars(builtins), __import__=safe_import), "MAX_CONTENT_LENGTH": 50000,
                     "insert_message": STORAGE.insert_message, "_update_style_vec": STYLE.update_entity_style_vector_incremental,
                     "logger": logging.getLogger("synthetic_style_engine")}
        exec(compile(ast.Module(body=[add], type_ignores=[]), "engine.py", "exec"), namespace)
        for caller, directive in itertools.product((False, True), (False, True)):
            conn = self.connection(legacy=True, full=True)
            before = self.seed_legacy(conn)
            engine = types.SimpleNamespace(conn=conn, _has_vectors=False, _has_personality=False, _has_style_vec=True,
                                           _write_lock=threading.Lock(), _ensure_connection=lambda: None,
                                           _maybe_auto_consolidate=lambda: None)
            compute = STYLE.compute_style_vector
            locks = []
            def outside_writer(text: str) -> list[float]:
                locks.append(engine._write_lock.locked())
                return compute(text)
            if caller:
                conn.execute("INSERT INTO sentinel VALUES ('pending')")
            with patch.object(STYLE, "compute_style_vector", outside_writer):
                result = namespace["add"](engine, "source survives legacy deferral", sender="Synthetic", directive=directive)
            self.assertEqual(locks, [False])
            self.assertEqual(self.row(conn), before)
            self.assertEqual(conn.in_transaction, caller)
            self.assertEqual(conn.execute("SELECT content,directive FROM messages WHERE id=?", (result["id"],)).fetchone(),
                             ("source survives legacy deferral", int(directive)))
            if caller:
                conn.rollback()
                self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
