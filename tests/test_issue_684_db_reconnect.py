"""Regression lock: the engine reconnects after its DB handle is poisoned
(SRE-01 / issue #684).

Pre-fix: _ensure_connection short-circuited on ``self.conn is not None`` and
``self.conn`` was reset only by close(), so a transient disk I/O error (stray
-wal deletion, FS blip) left a poisoned-but-non-None handle and EVERY
subsequent add/search/recall failed until the process was manually restarted.
Post-fix: _ensure_connection probes the cached handle and reconnects on failure.

FTS-only / no model loads.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sqlite3

import pytest

import truememory.engine as _engine_mod
from truememory.client import Memory


@pytest.fixture(autouse=True)
def _fts_only():
    """Force FTS-only so no sqlite-vec / embedding model is needed."""
    saved = _engine_mod._HAS_VECTOR
    _engine_mod._HAS_VECTOR = False
    yield
    _engine_mod._HAS_VECTOR = saved


class _FlakyConn:
    """Wraps a real sqlite3.Connection; raises a disk-I/O error on the FIRST
    ``PRAGMA schema_version`` probe, then delegates normally."""

    def __init__(self, real):
        self._real = real
        self._raised = False

    def execute(self, sql, *args, **kwargs):
        if not self._raised and sql.strip().upper().startswith("PRAGMA SCHEMA_VERSION"):
            self._raised = True
            raise sqlite3.OperationalError("disk I/O error")
        return self._real.execute(sql, *args, **kwargs)

    def close(self):
        return self._real.close()

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_reconnects_after_handle_closed(tmp_path):
    """A closed handle (poisoned connection) is detected and replaced; the next
    operation succeeds and prior data is intact."""
    db = tmp_path / "reconnect.db"
    m = Memory(str(db))
    m.add("the deploy succeeded on friday", user_id="u")

    # Poison: close the underlying connection behind the engine's back.
    m._engine.conn.close()
    assert m._engine.conn is not None  # the pre-fix short-circuit would return this dead handle

    res = m.add("the rollback happened on saturday", user_id="u")
    assert res.get("id") is not None
    assert any("rollback" in r.get("content", "") for r in m.search("rollback", user_id="u", limit=5))
    # prior data survived (same DB file, fresh handle)
    assert any("deploy" in r.get("content", "") for r in m.search("deploy", user_id="u", limit=5))


def test_reconnects_after_disk_io_error(tmp_path):
    """A handle whose probe raises OperationalError('disk I/O error') is replaced."""
    db = tmp_path / "io.db"
    m = Memory(str(db))
    m.add("first fact about coffee", user_id="u")

    # Wrap the live handle so the next probe raises a transient I/O error once.
    m._engine.conn = _FlakyConn(m._engine.conn)

    res = m.add("second fact about tea", user_id="u")
    assert res.get("id") is not None
    assert any("tea" in r.get("content", "") for r in m.search("tea", user_id="u", limit=5))


def test_healthy_handle_is_reused(tmp_path):
    """A healthy handle passes the probe and is reused across many ops."""
    db = tmp_path / "ok.db"
    m = Memory(str(db))
    for i in range(5):
        m.add(f"fact number {i} about widgets", user_id="u")
    assert len(m.search("widgets", user_id="u", limit=10)) >= 1
