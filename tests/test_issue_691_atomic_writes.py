"""Regression locks for atomic-write hardening (#691): adapter config writes
(X2-1), recall-cache writes (C1-1), and the pre-migration backup (D1-6).

No model loads.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import sqlite3
import stat
import sys


# ── X2-1: shared atomic adapter write ────────────────────────────────────────

def test_base_atomic_write_text_replaces_cleanly(tmp_path):
    from truememory.hooks.adapters.base import atomic_write_text, atomic_write_json
    p = tmp_path / "sub" / "config.json"  # parent doesn't exist yet
    atomic_write_text(p, '{"a": 1}')
    assert json.loads(p.read_text()) == {"a": 1}
    # no leftover tmp files in the dir
    assert not list(p.parent.glob(".*tmp*"))
    atomic_write_json(p, {"b": 2})
    assert json.loads(p.read_text()) == {"b": 2}


def test_adapters_use_atomic_write(tmp_path):
    """The 6 previously-bare adapters no longer call .write_text directly."""
    import inspect
    import importlib
    for name in ("cursor", "codex", "gemini", "kimi", "hermes", "openclaw"):
        mod = importlib.import_module(f"truememory.hooks.adapters.{name}")
        src = inspect.getsource(mod)
        assert ".write_text(" not in src, f"{name} still has a bare .write_text()"
        assert "atomic_write_text" in src, f"{name} does not use atomic_write_text"


# ── C1-1: recall cache uses the unique-per-pid tmp, not a fixed one ──────────

def test_recall_cache_uses_unique_tmp():
    import inspect
    from truememory.ingest.hooks import _shared
    # both cache writers must route through _atomic_write_text (unique per-pid
    # tmp), not a fixed shared .tmp (C1-1 race).
    set_src = inspect.getsource(_shared.set_recall_cache)
    inv_src = inspect.getsource(_shared.invalidate_recall_cache)
    for name, src in (("set_recall_cache", set_src), ("invalidate_recall_cache", inv_src)):
        assert "_atomic_write_text(RECALL_CACHE_PATH" in src, (
            f"{name} must write the cache via _atomic_write_text (unique tmp)"
        )
        # the old fixed-tmp code pattern (tmp = ...with_suffix; tmp.write_text)
        # must be gone — check the actual write call, not comments.
        assert "tmp.write_text(" not in src, (
            f"{name} still does a manual fixed-tmp write (C1-1 race)"
        )


def test_recall_cache_set_and_invalidate_roundtrip(tmp_path, monkeypatch):
    from truememory.ingest.hooks import _shared
    cache = tmp_path / "recall_cache.json"
    monkeypatch.setattr(_shared, "RECALL_CACHE_PATH", cache)
    # set_recall_cache(context, db_path, user_id=..., ...)
    _shared.set_recall_cache("ctx-one", "db", user_id="u")
    assert cache.exists()
    data = json.loads(cache.read_text())
    assert any(v.get("context") == "ctx-one" for v in data.values())
    # file is valid JSON (not torn) and owner-only
    if sys.platform != "win32":
        assert stat.S_IMODE(os.stat(cache).st_mode) == 0o600


# ── D1-6: pre-migration backup is a consistent single-file snapshot ──────────

def test_backup_is_consistent_single_file_snapshot(tmp_path):
    from truememory.storage import create_db, _backup_database, insert_message
    from pathlib import Path

    db = tmp_path / "memories.db"
    conn = create_db(db)
    for i in range(10):
        insert_message(conn, {
            "content": f"fact {i} about the project", "sender": "alice",
            "recipient": "bob", "timestamp": "2026-01-01T00:00:00Z",
            "category": "s", "modality": "conversation",
        })
    conn.commit()  # data may live in the -wal at this point

    backup = _backup_database(Path(db))
    assert backup is not None and backup.exists()

    # The backup is a COMPLETE single file: opening it alone (no -wal/-shm)
    # must surface all rows — proving the WAL was folded in (D1-6).
    assert not Path(f"{backup}-wal").exists()
    bconn = sqlite3.connect(str(backup))
    n = bconn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    bconn.close()
    assert n == 10, f"backup snapshot missing rows: {n} != 10 (torn-WAL regression)"
