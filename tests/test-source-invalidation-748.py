"""A05 stage 1: real stdlib L0 code and disposable SQLite, no package startup.

The two frozen references come from b97e586. Directive rows are removed only
from the reference input: all other formulas, ordering and returned values
must match exactly. No native dependencies, models or external data are read.
"""
from __future__ import annotations

import ast
import random
import sqlite3
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "truememory"


def _load_modules() -> tuple[types.ModuleType, types.ModuleType]:
    style = types.ModuleType("a05_style")
    exec(compile((SOURCE / "personality_style_vec.py").read_text(), "personality_style_vec.py", "exec"), style.__dict__)
    tree = ast.parse((SOURCE / "personality.py").read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and (node.module or "").startswith("truememory.")
    )]
    personality = types.ModuleType("a05_personality")
    for name in ("_l0_directive_filter", "_l0_source_schema", "_l0_transaction", "_l0_validate_source"):
        if hasattr(style, name):
            personality.__dict__[name] = getattr(style, name)
    exec(compile(tree, "personality.py", "exec"), personality.__dict__)
    return personality, style


SCHEMA = """
CREATE TABLE messages (
    id INTEGER PRIMARY KEY, content TEXT, sender TEXT, recipient TEXT,
    timestamp TEXT, category TEXT, modality TEXT, directive INTEGER
);
CREATE TABLE entity_profiles (
    entity TEXT PRIMARY KEY, message_count INTEGER, traits TEXT,
    communication_style TEXT, topics TEXT, relationships TEXT, updated_at TEXT
);
CREATE TABLE entity_style_vectors (
    entity TEXT PRIMARY KEY, vector TEXT, message_count INTEGER, updated_at TEXT
);
CREATE TABLE sentinel (value TEXT);
"""


def _db(path: str = ":memory:", *, directives: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=0, cached_statements=0)
    schema = SCHEMA if directives else SCHEMA.replace(", directive INTEGER", "")
    conn.executescript(schema)
    return conn


def _seed(conn: sqlite3.Connection) -> None:
    conn.executemany(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (9, "hey I write code and deploy the database", "Alice", "Bob", "2026-01-03", "work", "chat", 0),
            (3, "hello I enjoy coffee every morning", "ALICE", "Bob", "2026-01-01", "routine", "chat", None),
            (7, "I am worried about the server and customer pricing", "Bob", "Alice", "2026-01-03", "work", "chat", 0),
            (11, "directive-only sentinel should never become a profile", "Directive", "Alice", "2026-01-01", "control", "chat", 1),
            (12, "", "", "Bob", "2026-01-02", "", "chat", 0),
        ],
    )
    conn.commit()


def _allow(*_args: object) -> int:
    return sqlite3.SQLITE_OK


LEGACY_PROFILES = '''def _get_all_messages(conn: sqlite3.Connection) -> list[dict]:
    """Fetch all messages ordered by timestamp."""
    rows = conn.execute(
        "SELECT id, content, sender, recipient, timestamp, category, modality "
        "FROM messages ORDER BY timestamp"
    ).fetchall()
    return [
        {
            "id": r[0], "content": r[1], "sender": r[2],
            "recipient": r[3], "timestamp": r[4],
            "category": r[5], "modality": r[6],
        }
        for r in rows
    ]

def build_entity_profiles(conn: sqlite3.Connection) -> dict:
    """
    Analyze all messages and build personality profiles for key entities.

    For each entity (sender), extracts:

    - **message_count**: total messages sent.
    - **topics**: frequent themes (startup, health, food, etc.).
    - **communication_style**: average message length, emoji usage,
      formality level, and typical greeting.
    - **relationships**: who they message most and approximate topic focus
      per recipient.
    - **traits**: personality descriptors inferred from content analysis.

    Results are stored in the ``entity_profiles`` table and also returned
    as a dict keyed by entity name.

    Args:
        conn: Open database connection (from :func:`truememory.storage.create_db`).

    Returns:
        ``{entity: profile_dict}`` for every sender who has at least one
        message in the database.
    """
    all_msgs = _get_all_messages(conn)

    # Group messages by sender (normalized to lowercase for case-insensitive matching)
    by_sender: dict[str, list[dict]] = defaultdict(list)
    for msg in all_msgs:
        if msg["sender"]:
            by_sender[msg["sender"].lower()].append(msg)

    profiles: dict[str, dict] = {}

    for sender, messages in by_sender.items():
        # Communication style
        lengths = [len(m["content"]) for m in messages]
        avg_length = sum(lengths) / len(lengths) if lengths else 0.0
        uses_emoji = any(_detect_emoji(m["content"]) for m in messages)
        formality = _assess_formality(messages)
        greeting = _find_typical_greeting(messages)

        comm_style = {
            "avg_length": round(avg_length, 1),
            "uses_emoji": uses_emoji,
            "formality": formality,
            "typical_greeting": greeting,
        }

        # Relationships: who they talk to and rough topic per recipient
        recipient_counts: dict[str, int] = defaultdict(int)
        recipient_topics: dict[str, list[str]] = defaultdict(list)
        for msg in messages:
            recip = msg["recipient"]
            if recip:
                recipient_counts[recip] += 1
                # Quick topic tag for each message
                lower = msg["content"].lower()
                if any(w in lower for w in ("code", "deploy", "api",
                                            "database", "bug", "server")):
                    recipient_topics[recip].append("technical")
                elif any(w in lower for w in ("worried", "anxious",
                                              "scared", "stress")):
                    recipient_topics[recip].append("emotional")
                elif any(w in lower for w in ("dinner", "drinks",
                                              "hang out", "watch")):
                    recipient_topics[recip].append("social")
                elif any(w in lower for w in ("revenue", "investor",
                                              "pricing", "customer")):
                    recipient_topics[recip].append("business")

        relationships = {}
        for recip, count in sorted(recipient_counts.items(),
                                   key=lambda x: x[1], reverse=True):
            topic_freq: dict[str, int] = defaultdict(int)
            for t in recipient_topics.get(recip, []):
                topic_freq[t] += 1
            top_topic = max(topic_freq, key=topic_freq.get) if topic_freq else "general"
            relationships[recip] = {
                "message_count": count,
                "primary_topic": top_topic,
            }

        with _warnings_ctx():
            topics = _extract_topics(messages)
            traits = _extract_traits(messages)

        profile = {
            "message_count": len(messages),
            "topics": topics,
            "communication_style": comm_style,
            "relationships": relationships,
            "traits": traits,
        }
        profiles[sender] = profile

        # Store in database
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT OR REPLACE INTO entity_profiles
               (entity, message_count, traits, communication_style,
                topics, relationships, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                sender,
                len(messages),
                json.dumps(traits),
                json.dumps(comm_style),
                json.dumps(topics),
                json.dumps(relationships),
                now,
            ),
        )

    conn.commit()
    return profiles'''

LEGACY_STYLE = '''def build_entity_style_vectors(conn: sqlite3.Connection) -> dict[str, list[float]]:
    """Batch-build style vectors for every entity (sender) in the database.

    For each sender:
        1. Compute ``compute_style_vector(msg.content)`` for each message.
        2. Mean-pool all per-message vectors via ``mean_pool_vectors``.
        3. Store the result in the ``entity_style_vectors`` table.

    Args:
        conn: Open database connection (from :func:`truememory.storage.create_db`).

    Returns:
        ``{entity: vector}`` for every sender.
    """
    conn.execute(
        """CREATE TABLE IF NOT EXISTS entity_style_vectors (
            entity TEXT PRIMARY KEY,
            vector TEXT,
            message_count INTEGER DEFAULT 0,
            updated_at TEXT
        )"""
    )

    rows = conn.execute(
        "SELECT sender, content FROM messages WHERE sender != '' ORDER BY sender, timestamp"
    ).fetchall()

    from collections import defaultdict
    by_sender: dict[str, list[str]] = defaultdict(list)
    for sender, content in rows:
        by_sender[sender.lower()].append(content)

    result: dict[str, list[float]] = {}
    now = datetime.now(timezone.utc).isoformat()

    for sender, contents in by_sender.items():
        vecs = [compute_style_vector(c) for c in contents]
        mean_vec = mean_pool_vectors(vecs)
        result[sender] = mean_vec

        conn.execute(
            """INSERT OR REPLACE INTO entity_style_vectors
               (entity, vector, message_count, updated_at)
               VALUES (?, ?, ?, ?)""",
            (sender, json.dumps(mean_vec), len(contents), now),
        )

    conn.commit()
    return result'''

class TestSourceInvalidation748(unittest.TestCase):
    def setUp(self) -> None:
        self.personality, self.style = _load_modules()
        self.temp = tempfile.TemporaryDirectory(prefix="a05-748-")
        self.addCleanup(self.temp.cleanup)
        self.connections: list[sqlite3.Connection] = []
        self.addCleanup(self._close)

    def _close(self) -> None:
        for conn in self.connections:
            conn.set_authorizer(_allow)
            conn.close()

    def db(self, *, file: bool = False, directives: bool = True) -> sqlite3.Connection:
        path = str(Path(self.temp.name) / f"source-{len(self.connections)}.db") if file else ":memory:"
        conn = _db(path, directives=directives)
        self.connections.append(conn)
        if directives:
            _seed(conn)
        return conn

    def builders(self) -> list[tuple[types.ModuleType, str, str]]:
        return [
            (self.personality, "build_entity_profiles", "entity_profiles"),
            (self.style, "build_entity_style_vectors", "entity_style_vectors"),
        ]

    def generation(self, conn: sqlite3.Connection, table: str, *, stamps: bool = True) -> list[tuple]:
        columns = (
            "entity,message_count,traits,communication_style,topics,relationships"
            if table == "entity_profiles" else "entity,vector,message_count"
        )
        if stamps:
            columns += ",updated_at"
        return [tuple(row) for row in conn.execute(f"SELECT {columns} FROM {table} ORDER BY entity")]

    def reference(self, conn: sqlite3.Connection, kind: str) -> tuple[dict, list[tuple]]:
        ref = _db()
        try:
            rows = conn.execute("SELECT * FROM messages WHERE directive = 0 OR directive IS NULL").fetchall()
            ref.executemany("INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [tuple(r) for r in rows])
            ref.commit()
            module = self.personality if kind == "entity_profiles" else self.style
            namespace = dict(module.__dict__)
            exec(LEGACY_PROFILES if kind == "entity_profiles" else LEGACY_STYLE, namespace)
            name = "build_entity_profiles" if kind == "entity_profiles" else "build_entity_style_vectors"
            result = namespace[name](ref)
            return result, self.generation(ref, kind, stamps=False)
        finally:
            ref.close()

    def assert_reference(self, conn: sqlite3.Connection, module: types.ModuleType, name: str, table: str) -> None:
        expected, stored = self.reference(conn, table)
        self.assertEqual(getattr(module, name)(conn), expected)
        self.assertEqual(self.generation(conn, table, stamps=False), stored)

    def test_unchanged_formulas_and_tied_order_match_frozen_reference(self) -> None:
        rng = random.Random(748)
        phrases = ["hello friend coffee morning", "please deploy the code", "worried about pricing", "", "a", "gym every day"]
        for size in (0, 1, 5, 51, 103):
            with self.subTest(size=size):
                conn = self.db()
                conn.execute("DELETE FROM messages")
                for index in range(size):
                    conn.execute("INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
                        index * 3 - 5, rng.choice(phrases), rng.choice(["Alice", "ALICE", "Bob", "Änne", ""]),
                        rng.choice(["Alice", "Bob", "BOB", ""]), rng.choice(["", "2026-01-01", "2026-01-03"]),
                        "fixture", "chat", rng.choice([0, 0, 1, None]),
                    ))
                conn.commit()
                for module, name, table in self.builders():
                    self.assert_reference(conn, module, name, table)
                    self.assertFalse(conn.in_transaction)

    def test_edits_deletes_replacement_and_empty_match_fresh_reference(self) -> None:
        changes = [
            "UPDATE messages SET content='hello new ordinary content', recipient='Carol' WHERE id=9",
            "UPDATE messages SET sender='Carol', timestamp='2025-12-01' WHERE id=3",
            "DELETE FROM messages WHERE sender='Bob'",
            "INSERT OR REPLACE INTO messages VALUES (9,'replaced content','Dana','Carol','2025-01-01','','chat',0)",
            "DELETE FROM messages",
        ]
        for module, name, table in self.builders():
            conn = self.db()
            getattr(module, name)(conn)
            for sql in changes:
                with self.subTest(builder=name, change=sql.split()[0]):
                    conn.execute(sql)
                    conn.commit()
                    self.assert_reference(conn, module, name, table)
            self.assertEqual(self.generation(conn, table), [])

    def test_directive_index_does_not_reorder_tied_ordinary_rows(self) -> None:
        conn = self.db()
        conn.execute("DELETE FROM messages")
        conn.executescript(
            "CREATE INDEX idx_messages_sender ON messages(sender);"
            "CREATE INDEX idx_messages_timestamp ON messages(timestamp);"
            "CREATE INDEX idx_messages_directive ON messages(directive);"
        )
        rows = [
            (1, "hello one code", "Alice", "Bob", "2026-01-01", "", "", None),
            (2, "hey two coffee", "Alice", "Carol", "2026-01-01", "", "", 0),
        ] + [(100 + i, "ignored", "Control", "", "2026-01-01", "", "", 1) for i in range(100)]
        conn.executemany("INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        conn.commit()
        for module, name, table in self.builders():
            self.assert_reference(conn, module, name, table)

    def test_directive_toggle_and_directive_only_generation(self) -> None:
        for module, name, table in self.builders():
            conn = self.db()
            self.assert_reference(conn, module, name, table)
            self.assertNotIn("directive", dict((r[0], r) for r in self.generation(conn, table)))
            for flag in (1, 0, None):
                conn.execute("UPDATE messages SET directive=? WHERE sender != ''", (flag,))
                conn.commit()
                self.assert_reference(conn, module, name, table)
                if flag == 1:
                    self.assertEqual(self.generation(conn, table), [])

    def test_standalone_schema_without_maintenance_or_directive(self) -> None:
        for module, name, table in self.builders():
            conn = self.db(directives=False)
            conn.execute("INSERT INTO messages VALUES (1,'hello code','Alice','Bob','2026-01-01','','chat')")
            conn.commit()
            result = getattr(module, name)(conn)
            self.assertEqual(set(result), {"alice"})
            conn.execute("DELETE FROM messages")
            conn.commit()
            self.assertEqual(getattr(module, name)(conn), {})
            self.assertEqual(self.generation(conn, table), [])

    def test_row_factory_and_autocommit_compatibility(self) -> None:
        for module, name, table in self.builders():
            conn = self.db()
            conn.row_factory = sqlite3.Row
            conn.isolation_level = None
            self.assert_reference(conn, module, name, table)
            self.assertIsNone(conn.isolation_level)
            self.assertFalse(conn.in_transaction)

    def test_style_builder_accepts_its_original_minimal_source_schema(self) -> None:
        conn = sqlite3.connect(":memory:")
        self.connections.append(conn)
        conn.execute("CREATE TABLE messages (sender TEXT, content TEXT, timestamp TEXT)")
        conn.execute("INSERT INTO messages VALUES ('Alice','hello ordinary content','2026-01-01')")
        conn.commit()
        result = self.style.build_entity_style_vectors(conn)
        self.assertEqual(result, {"alice": self.style.compute_style_vector("hello ordinary content")})
        self.assertEqual(conn.execute("SELECT message_count FROM entity_style_vectors").fetchone()[0], 1)

    def test_failed_source_read_does_not_leave_owned_transaction(self) -> None:
        for module, name, _table in self.builders():
            conn = sqlite3.connect(":memory:")
            self.connections.append(conn)
            with self.assertRaises(sqlite3.OperationalError):
                getattr(module, name)(conn)
            self.assertFalse(conn.in_transaction)

    def test_style_table_creation_is_after_compute_and_atomic(self) -> None:
        conn = self.db()
        conn.execute("DROP TABLE entity_style_vectors")
        original = self.style.compute_style_vector
        observed = []
        def compute(text: str) -> list[float]:
            observed.append(conn.execute("SELECT count(*) FROM sqlite_master WHERE name='entity_style_vectors'").fetchone()[0])
            self.assertFalse(conn.in_transaction)
            return original(text)
        with patch.object(self.style, "compute_style_vector", compute):
            self.style.build_entity_style_vectors(conn)
        self.assertTrue(observed)
        self.assertEqual(set(observed), {0})
        self.assertTrue(self.generation(conn, "entity_style_vectors"))

    def test_publication_preserves_caller_transaction_and_rollback(self) -> None:
        for module, name, table in self.builders():
            conn = self.db()
            getattr(module, name)(conn)
            before = self.generation(conn, table)
            conn.execute("INSERT INTO sentinel VALUES ('uncommitted')")
            conn.execute("UPDATE messages SET sender='Carol' WHERE sender='Bob'")
            result = getattr(module, name)(conn)
            self.assertIn("carol", result)
            self.assertTrue(conn.in_transaction)
            conn.rollback()
            self.assertEqual(self.generation(conn, table), before)
            self.assertEqual(conn.execute("SELECT count(*) FROM sentinel").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT sender FROM messages WHERE id=7").fetchone()[0], "Bob")

    def test_publication_retains_caller_writes_until_explicit_commit(self) -> None:
        for module, name, table in self.builders():
            conn = self.db(file=True)
            path = conn.execute("PRAGMA database_list").fetchone()[2]
            other = sqlite3.connect(path, timeout=0)
            self.connections.append(other)
            conn.execute("INSERT INTO sentinel VALUES ('uncommitted')")
            getattr(module, name)(conn)
            self.assertEqual(other.execute("SELECT count(*) FROM sentinel").fetchone()[0], 0)
            self.assertEqual(other.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 0)
            conn.commit()
            self.assertEqual(other.execute("SELECT count(*) FROM sentinel").fetchone()[0], 1)
            self.assertGreater(other.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 0)

    def test_late_compute_failure_and_cancellation_leave_previous_generation(self) -> None:
        for module, name, table in self.builders():
            for error_type in (RuntimeError, KeyboardInterrupt):
                with self.subTest(builder=name, error=error_type.__name__):
                    conn = self.db()
                    getattr(module, name)(conn)
                    before = self.generation(conn, table)
                    conn.execute("UPDATE messages SET content=content || ' corrected'")
                    conn.commit()
                    conn.execute("INSERT INTO sentinel VALUES ('pending')")
                    function = "_extract_topics" if table == "entity_profiles" else "compute_style_vector"
                    original = getattr(module, function)
                    calls = 0
                    error = error_type("synthetic computation failure")
                    def compute(*args: object, **kwargs: object) -> object:
                        nonlocal calls
                        calls += 1
                        if calls == 3:
                            raise error
                        return original(*args, **kwargs)
                    # Profile fixture has two ordinary senders; add a third to fail late.
                    conn.execute("INSERT INTO messages VALUES (20,'new content','Carol','','2026-01-04','','chat',0)")
                    with patch.object(module, function, compute):
                        with self.assertRaises(error_type) as raised:
                            getattr(module, name)(conn)
                    self.assertIs(raised.exception, error)
                    self.assertGreaterEqual(calls, 3)
                    conn.commit()
                    self.assertEqual(self.generation(conn, table), before)

    def test_partial_insert_failure_rolls_back_replacement_not_caller(self) -> None:
        for module, name, table in self.builders():
            for caller in (False, True):
                with self.subTest(builder=name, caller=caller):
                    conn = self.db()
                    getattr(module, name)(conn)
                    before = self.generation(conn, table)
                    conn.execute("UPDATE messages SET content=content || ' corrected'")
                    conn.commit()
                    conn.execute(f"CREATE TRIGGER fail_l0 BEFORE INSERT ON {table} WHEN new.entity='bob' BEGIN SELECT RAISE(ABORT,'synthetic write failure'); END")
                    if caller:
                        conn.execute("INSERT INTO sentinel VALUES ('pending')")
                    with self.assertRaisesRegex(sqlite3.IntegrityError, "synthetic write failure"):
                        getattr(module, name)(conn)
                    self.assertEqual(conn.in_transaction, caller)
                    conn.commit()
                    self.assertEqual(self.generation(conn, table), before)
                    self.assertEqual(conn.execute("SELECT count(*) FROM sentinel").fetchone()[0], int(caller))

    def test_publication_commit_or_release_failure_rolls_back(self) -> None:
        for module, name, table in self.builders():
            for caller in (False, True):
                with self.subTest(builder=name, caller=caller):
                    conn = self.db()
                    getattr(module, name)(conn)
                    before = self.generation(conn, table)
                    conn.execute("UPDATE messages SET content=content || ' corrected'")
                    conn.commit()
                    if caller:
                        conn.execute("INSERT INTO sentinel VALUES ('pending')")
                    writing = False
                    denied = False
                    def authorize(action: int, one: str | None, two: str | None, *_rest: object) -> int:
                        nonlocal writing, denied
                        if action == sqlite3.SQLITE_DELETE and one == table:
                            writing = True
                        commit = action == sqlite3.SQLITE_TRANSACTION and one == "COMMIT"
                        release = action == sqlite3.SQLITE_SAVEPOINT and one == "RELEASE"
                        if writing and not denied and (release if caller else commit):
                            denied = True
                            return sqlite3.SQLITE_DENY
                        return sqlite3.SQLITE_OK
                    conn.set_authorizer(authorize)
                    try:
                        with self.assertRaises(sqlite3.DatabaseError):
                            getattr(module, name)(conn)
                    finally:
                        conn.set_authorizer(_allow)
                    self.assertTrue(denied)
                    self.assertEqual(conn.in_transaction, caller)
                    conn.commit()
                    self.assertEqual(self.generation(conn, table), before)
                    self.assertEqual(conn.execute("SELECT count(*) FROM sentinel").fetchone()[0], int(caller))

    def test_rollback_failure_never_releases_partial_publication(self) -> None:
        for module, name, table in self.builders():
            conn = self.db()
            getattr(module, name)(conn)
            before = self.generation(conn, table)
            conn.execute(f"CREATE TRIGGER fail_l0 BEFORE INSERT ON {table} WHEN new.entity='bob' BEGIN SELECT RAISE(ABORT,'synthetic write failure'); END")
            conn.execute("INSERT INTO sentinel VALUES ('pending')")
            blocked = False
            releases_after_failure = []
            def authorize(action: int, one: str | None, two: str | None, *_rest: object) -> int:
                nonlocal blocked
                if action == sqlite3.SQLITE_SAVEPOINT and one == "ROLLBACK":
                    blocked = True
                    return sqlite3.SQLITE_DENY
                if blocked and action == sqlite3.SQLITE_SAVEPOINT and one == "RELEASE":
                    releases_after_failure.append(two)
                return sqlite3.SQLITE_OK
            conn.set_authorizer(authorize)
            try:
                with self.assertRaises(sqlite3.DatabaseError):
                    getattr(module, name)(conn)
            finally:
                conn.set_authorizer(_allow)
            self.assertTrue(blocked)
            self.assertTrue(conn.in_transaction)
            self.assertEqual(releases_after_failure, [])
            conn.rollback()
            self.assertEqual(self.generation(conn, table), before)
            self.assertEqual(conn.execute("SELECT count(*) FROM sentinel").fetchone()[0], 0)

    def test_foreground_source_changes_commit_during_compute_and_fence_publish(self) -> None:
        changes = [
            "UPDATE messages SET content='concurrent new text' WHERE id=9",
            "UPDATE messages SET sender='Carol' WHERE id=9",
            "UPDATE messages SET timestamp='2025-01-01' WHERE id=9",
            "UPDATE messages SET directive=1 WHERE id=9",
            "DELETE FROM messages WHERE id=9",
            "INSERT INTO messages VALUES (2,'backdated ordinary text','Alice','Bob','2024-01-01','','chat',0)",
            "DELETE FROM messages",
            "ALTER TABLE messages ADD COLUMN extra TEXT",
        ]
        for module, name, table in self.builders():
            for sql in changes:
                with self.subTest(builder=name, mutation=sql.split()[0]):
                    conn = self.db(file=True)
                    getattr(module, name)(conn)
                    before = self.generation(conn, table)
                    path = conn.execute("PRAGMA database_list").fetchone()[2]
                    other = sqlite3.connect(path, timeout=0)
                    self.connections.append(other)
                    function = "_extract_topics" if table == "entity_profiles" else "compute_style_vector"
                    original = getattr(module, function)
                    calls = 0
                    def compute(*args: object, **kwargs: object) -> object:
                        nonlocal calls
                        calls += 1
                        self.assertFalse(conn.in_transaction)
                        if calls == 1:
                            other.execute(sql)
                            other.commit()
                        return original(*args, **kwargs)
                    with patch.object(module, function, compute):
                        with self.assertRaisesRegex(sqlite3.OperationalError, "L0 source changed"):
                            getattr(module, name)(conn)
                    self.assertGreater(calls, 0)
                    self.assertFalse(conn.in_transaction)
                    conn.commit()
                    self.assertEqual(self.generation(conn, table), before)

    def test_stale_caller_wal_snapshot_cannot_publish_or_restart(self) -> None:
        for module, name, table in self.builders():
            conn = self.db(file=True)
            conn.execute("PRAGMA journal_mode=WAL")
            getattr(module, name)(conn)
            before = self.generation(conn, table)
            path = conn.execute("PRAGMA database_list").fetchone()[2]
            other = sqlite3.connect(path, timeout=0)
            self.connections.append(other)
            conn.execute("BEGIN")
            function = "_extract_topics" if table == "entity_profiles" else "compute_style_vector"
            original = getattr(module, function)
            changed = False
            def compute(*args: object, **kwargs: object) -> object:
                nonlocal changed
                self.assertTrue(conn.in_transaction)
                if not changed:
                    other.execute("UPDATE messages SET content='concurrent corrected source' WHERE id=9")
                    other.commit()
                    changed = True
                return original(*args, **kwargs)
            with patch.object(module, function, compute):
                with self.assertRaises(sqlite3.OperationalError):
                    getattr(module, name)(conn)
            self.assertTrue(changed)
            self.assertTrue(conn.in_transaction)
            self.assertEqual(self.generation(conn, table), before)
            self.assertNotEqual(conn.execute("SELECT content FROM messages WHERE id=9").fetchone()[0], "concurrent corrected source")
            conn.rollback()
            self.assertEqual(conn.execute("SELECT content FROM messages WHERE id=9").fetchone()[0], "concurrent corrected source")
            self.assertEqual(self.generation(conn, table), before)

    def test_validation_holds_writer_ownership(self) -> None:
        for module, name, table in self.builders():
            conn = self.db(file=True)
            path = conn.execute("PRAGMA database_list").fetchone()[2]
            other = sqlite3.connect(path, timeout=0)
            self.connections.append(other)
            original = module._l0_validate_source
            blocked = []
            def validate(*args: object, **kwargs: object) -> None:
                original(*args, **kwargs)
                try:
                    with self.assertRaises(sqlite3.OperationalError):
                        other.execute("UPDATE messages SET sender='Racing' WHERE id=9")
                    blocked.append(True)
                finally:
                    other.rollback()
            with patch.object(module, "_l0_validate_source", validate):
                getattr(module, name)(conn)
            self.assertEqual(blocked, [True])

    def test_cancellation_inside_publication_preserves_caller_generation(self) -> None:
        class CancelConnection(sqlite3.Connection):
            cancel_table: str | None = None

            def executemany(self, sql: str, parameters: object) -> sqlite3.Cursor:
                if self.cancel_table and f"INSERT INTO {self.cancel_table}" in sql:
                    rows = iter(parameters)
                    super().execute(sql, next(rows))
                    raise KeyboardInterrupt("synthetic publication cancellation")
                return super().executemany(sql, parameters)

        for module, name, table in self.builders():
            conn = sqlite3.connect(":memory:", factory=CancelConnection)
            self.connections.append(conn)
            conn.executescript(SCHEMA)
            _seed(conn)
            getattr(module, name)(conn)
            before = self.generation(conn, table)
            conn.execute("INSERT INTO sentinel VALUES ('pending')")
            conn.execute("UPDATE messages SET content=content || ' changed'")
            conn.cancel_table = table
            with self.assertRaises(KeyboardInterrupt):
                getattr(module, name)(conn)
            self.assertTrue(conn.in_transaction)
            conn.commit()
            self.assertEqual(self.generation(conn, table), before)
            self.assertEqual(conn.execute("SELECT count(*) FROM sentinel").fetchone()[0], 1)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        unittest.main()
