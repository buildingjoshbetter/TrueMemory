"""Tests for truememory.storage — schema creation, CRUD, FTS sync, indexes."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from truememory.storage import (
    create_db,
    insert_message,
    delete_message,
    get_message,
    get_message_count,
    bulk_replace_messages,
    update_message,
    DEFAULT_BUSY_TIMEOUT_MS,
)


@pytest.fixture
def db():
    path = tempfile.mktemp(suffix=".db")
    conn = create_db(path)
    yield conn
    conn.close()
    Path(path).unlink(missing_ok=True)


class TestCreateDb:
    def test_creates_messages_table(self, db):
        tables = {r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "messages" in tables

    def test_creates_fts_table(self, db):
        tables = {r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "messages_fts" in tables

    def test_creates_entity_tables(self, db):
        tables = {r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        for t in ("entity_profiles", "fact_timeline", "summaries", "episodes"):
            assert t in tables, f"Missing table: {t}"

    def test_wal_mode_enabled(self, db):
        mode = db.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"

    def test_busy_timeout_set(self, db):
        timeout = db.execute("PRAGMA busy_timeout").fetchone()[0]
        assert timeout == DEFAULT_BUSY_TIMEOUT_MS

    def test_synchronous_normal(self, db):
        val = db.execute("PRAGMA synchronous").fetchone()[0]
        assert val == 1  # NORMAL = 1

    def test_cache_size(self, db):
        val = db.execute("PRAGMA cache_size").fetchone()[0]
        assert val == -64000

    def test_mmap_size(self, db):
        val = db.execute("PRAGMA mmap_size").fetchone()[0]
        assert val == 268435456

    def test_indexes_created(self, db):
        indexes = {r[1] for r in db.execute("PRAGMA index_list('messages')").fetchall()}
        assert "idx_messages_sender" in indexes
        assert "idx_messages_timestamp" in indexes

    def test_legacy_messages_table_adds_metadata_column(self, tmp_path):
        path = tmp_path / "legacy.db"
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE messages ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "content TEXT NOT NULL, "
            "sender TEXT DEFAULT '', "
            "recipient TEXT DEFAULT '', "
            "timestamp TEXT DEFAULT '', "
            "category TEXT DEFAULT '', "
            "modality TEXT DEFAULT '', "
            "directive INTEGER DEFAULT 0"
            ")"
        )
        conn.execute("INSERT INTO messages (content, sender) VALUES (?, ?)", ("legacy", "alice"))
        conn.commit()
        conn.close()

        migrated = create_db(path)
        try:
            cols = {row[1] for row in migrated.execute("PRAGMA table_info(messages)").fetchall()}
            assert "metadata" in cols
            row = get_message(migrated, 1)
            assert row["metadata"] == {}
        finally:
            migrated.close()


class TestInsertMessage:
    def test_basic_insert_and_retrieve(self, db):
        msg = {"content": "hello world", "sender": "alice"}
        new_id = insert_message(db, msg)
        db.commit()
        assert new_id > 0
        row = get_message(db, new_id)
        assert row is not None
        assert row["content"] == "hello world"
        assert row["sender"] == "alice"

    def test_empty_content_rejected(self, db):
        with pytest.raises(ValueError, match="content cannot be empty"):
            insert_message(db, {"content": ""})

    def test_whitespace_content_rejected(self, db):
        with pytest.raises(ValueError, match="content cannot be empty"):
            insert_message(db, {"content": "   "})

    def test_fts_sync_on_insert(self, db):
        insert_message(db, {"content": "unique_canary_token_xyz"})
        db.commit()
        rows = db.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?",
            ("unique_canary_token_xyz",),
        ).fetchall()
        assert len(rows) == 1

    def test_sql_injection_safe(self, db):
        msg = {"content": "Robert'); DROP TABLE messages;--"}
        new_id = insert_message(db, msg)
        db.commit()
        row = get_message(db, new_id)
        assert row["content"] == "Robert'); DROP TABLE messages;--"
        assert get_message_count(db) >= 1

    def test_metadata_round_trip(self, db):
        metadata = {"session_id": "sess-1", "source": "test"}
        new_id = insert_message(db, {"content": "with metadata", "metadata": metadata})
        db.commit()
        row = get_message(db, new_id)
        assert row["metadata"] == metadata

    def test_metadata_must_be_dict(self, db):
        with pytest.raises(TypeError, match="metadata must be a dict or None"):
            insert_message(db, {"content": "bad metadata", "metadata": ["bad"]})

    def test_metadata_must_be_json_serializable(self, db):
        with pytest.raises(TypeError, match="metadata must be JSON-serializable"):
            insert_message(db, {"content": "bad metadata", "metadata": {"bad": object()}})


class TestUpdateMessage:
    def test_update_metadata(self, db):
        new_id = insert_message(db, {"content": "update metadata"})
        db.commit()
        assert update_message(db, new_id, metadata={"stage": "updated"}) is True
        row = get_message(db, new_id)
        assert row["metadata"] == {"stage": "updated"}


class TestDeleteMessage:
    def test_delete_existing(self, db):
        new_id = insert_message(db, {"content": "to delete"})
        db.commit()
        assert delete_message(db, new_id) is True
        assert get_message(db, new_id) is None

    def test_delete_nonexistent(self, db):
        assert delete_message(db, 999999) is False

    def test_fts_cleaned_on_delete(self, db):
        new_id = insert_message(db, {"content": "fts_delete_test_token"})
        db.commit()
        delete_message(db, new_id)
        db.commit()
        rows = db.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?",
            ("fts_delete_test_token",),
        ).fetchall()
        assert len(rows) == 0


class TestBulkReplace:
    def test_replaces_all(self, db):
        insert_message(db, {"content": "old message"})
        db.commit()
        messages = [
            {"content": "new msg 1", "sender": "a", "metadata": {"batch": 1}},
            {"content": "new msg 2", "sender": "b"},
        ]
        count = bulk_replace_messages(db, messages)
        assert count == 2
        assert get_message_count(db) == 2
        first_id = db.execute(
            "SELECT id FROM messages WHERE content = ?", ("new msg 1",)
        ).fetchone()[0]
        first = get_message(db, first_id)
        assert first["metadata"] == {"batch": 1}

    def test_fts_synced_after_replace(self, db):
        messages = [{"content": "bulk_unique_token_abc"}]
        bulk_replace_messages(db, messages)
        rows = db.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?",
            ("bulk_unique_token_abc",),
        ).fetchall()
        assert len(rows) == 1


class TestConcurrentAccess:
    def test_two_connections_read_write(self):
        path = tempfile.mktemp(suffix=".db")
        conn1 = create_db(path)
        conn2 = sqlite3.connect(path, check_same_thread=False)
        conn2.execute("PRAGMA journal_mode=WAL")
        conn2.execute(f"PRAGMA busy_timeout={DEFAULT_BUSY_TIMEOUT_MS}")

        insert_message(conn1, {"content": "from conn1"})
        conn1.commit()

        row = conn2.execute("SELECT content FROM messages WHERE id = 1").fetchone()
        assert row[0] == "from conn1"

        conn1.close()
        conn2.close()
        Path(path).unlink(missing_ok=True)
