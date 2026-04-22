"""Regression locks for Hunter F01 / F02 / F32 — embedder-identity tracking.

These tests exercise the upgrade path where a stored DB was built with a
different embedder than the one currently configured. The code under test must:

- F01: raise `TrueMemoryMigrationError` when `init_vec_table` finds an existing
  `vec_messages` at a different dim.
- F02: persist `(embed_model, embed_dim)` in a `metadata` key/value table on
  every `build_vectors` / `build_separation_vectors`, and raise on model drift
  at matching dim (the silent-quality-collapse case).
- F32: refuse a silent auto-rebuild in `engine.open()` when metadata names a
  different model than the currently-configured tier.
"""
from __future__ import annotations

import logging
import sqlite3

import pytest

from truememory import vector_search
from truememory.storage import create_db
from truememory.vector_search import (
    EMBEDDING_MODEL,
    TrueMemoryMigrationError,
    _check_embedder_compatibility,
    _check_rebuild_allowed,
    _read_embedder_metadata,
    _write_embedder_metadata,
    build_vectors,
    init_vec_table,
)


def _fresh_conn(tmp_path) -> sqlite3.Connection:
    """Create a DB with full schema but no vec tables yet."""
    return create_db(tmp_path / "upgrade.db")


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    import sqlite_vec
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


# ---------------------------------------------------------------------------
# F01 — dim mismatch on existing vec_messages is fatal
# ---------------------------------------------------------------------------


def test_dim_mismatch_raises_migration_error(tmp_path, monkeypatch):
    """Simulates a v0.3.0 Pro (1024d) DB being opened on v0.4.0 (256d)."""
    conn = _fresh_conn(tmp_path)
    _load_sqlite_vec(conn)
    conn.execute("CREATE VIRTUAL TABLE vec_messages USING vec0(embedding float[1024])")
    conn.commit()

    # Current tier advertises 256d (Edge / Base / Pro on v0.4.0 all = 256d)
    monkeypatch.setattr(vector_search, "_embedding_dim", 256)

    with pytest.raises(TrueMemoryMigrationError) as excinfo:
        init_vec_table(conn)
    msg = str(excinfo.value)
    assert "1024d" in msg and "256d" in msg
    assert "truememory_configure" in msg  # actionable hint present


# ---------------------------------------------------------------------------
# F02 — metadata is written on build and read on init
# ---------------------------------------------------------------------------


def test_metadata_written_on_build_vectors(tmp_path):
    """`build_vectors` must persist (embed_model, embed_dim) so later opens
    can detect drift."""
    conn = _fresh_conn(tmp_path)
    init_vec_table(conn)

    conn.execute("INSERT INTO messages(content) VALUES (?)", ("hello world",))
    conn.execute("INSERT INTO messages(content) VALUES (?)", ("another message",))
    conn.commit()

    n = build_vectors(conn)
    assert n == 2

    model, dim = _read_embedder_metadata(conn)
    assert model == EMBEDDING_MODEL
    assert dim is not None and dim > 0


def test_model_change_raises_migration_error_at_matching_dim(
    tmp_path, monkeypatch
):
    """Simulates Model2Vec 256d → Qwen3 256d without re-embed (F02's core scenario).

    Matching dims would otherwise mask a silent vector-space mismatch.
    """
    conn = _fresh_conn(tmp_path)
    init_vec_table(conn)
    # Pretend this DB was built with model2vec previously
    _write_embedder_metadata(conn)
    assert _read_embedder_metadata(conn)[0] == EMBEDDING_MODEL

    # Now simulate the user switching tier (same dim, different model) WITHOUT
    # going through truememory_configure()
    monkeypatch.setattr(vector_search, "EMBEDDING_MODEL", "qwen3_256")
    # dim stays the same

    # init_vec_table must reject this silent drift
    with pytest.raises(TrueMemoryMigrationError) as excinfo:
        init_vec_table(conn)
    msg = str(excinfo.value)
    assert "truememory_configure" in msg


def test_legacy_v030_db_warns_without_raising(tmp_path, caplog):
    """v0.3.0 DBs have `vec_messages` but no metadata row. At matching dim,
    we must warn (not raise) — letting the user keep working until they
    re-embed intentionally.
    """
    conn = _fresh_conn(tmp_path)
    _load_sqlite_vec(conn)
    # Pre-create vec_messages at current dim with NO metadata row (legacy shape).
    conn.execute(
        f"CREATE VIRTUAL TABLE vec_messages USING vec0(embedding float[{vector_search._embedding_dim}])"
    )
    # Drop the metadata table that create_db gave us so we faithfully model
    # a pre-F02 DB.
    conn.execute("DROP TABLE IF EXISTS metadata")
    conn.commit()

    with caplog.at_level(logging.WARNING, logger="truememory.vector_search"):
        # Should NOT raise — just warn.
        _check_embedder_compatibility(conn)

    assert any(
        "without embedder metadata" in rec.message
        for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# fresh-DB / same-model — must not trip either check
# ---------------------------------------------------------------------------


def test_fresh_db_init_vec_table_no_error(tmp_path):
    """A brand-new DB has no vec table and no metadata — init must succeed."""
    conn = _fresh_conn(tmp_path)
    init_vec_table(conn)
    # Idempotent: second call also fine.
    init_vec_table(conn)


def test_same_model_no_raise(tmp_path):
    """Re-opening with the same model must not raise."""
    conn = _fresh_conn(tmp_path)
    init_vec_table(conn)
    conn.execute("INSERT INTO messages(content) VALUES ('x')")
    conn.commit()
    build_vectors(conn)
    # Second init should see matching metadata and pass cleanly.
    init_vec_table(conn)


# ---------------------------------------------------------------------------
# F32 — engine.open() must refuse silent rebuild on model drift
# ---------------------------------------------------------------------------


def test_check_rebuild_allowed_raises_on_model_drift(tmp_path, monkeypatch):
    conn = _fresh_conn(tmp_path)
    _write_embedder_metadata(conn)  # records current EMBEDDING_MODEL

    # Simulate live tier switch without a re-embed
    monkeypatch.setattr(vector_search, "EMBEDDING_MODEL", "qwen3_256")

    with pytest.raises(TrueMemoryMigrationError) as excinfo:
        _check_rebuild_allowed(conn)
    msg = str(excinfo.value)
    assert "truememory_configure" in msg
    assert "silent auto-rebuild" in msg.lower() or "refusing" in msg.lower()


def test_check_rebuild_allowed_noop_without_metadata(tmp_path):
    """No metadata = pre-F02 DB. Rebuild is allowed; metadata will be written
    on the first build_vectors call."""
    conn = _fresh_conn(tmp_path)
    # Drop the metadata table so we're modelling a v0.3.0 DB
    conn.execute("DROP TABLE IF EXISTS metadata")
    conn.commit()
    # Must not raise.
    _check_rebuild_allowed(conn)
