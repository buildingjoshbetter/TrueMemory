"""Regression lock: the L2->cosine vector migration must not lose vectors when
a prior run crashed mid-staging (D1-2 / issue #686).

Pre-fix: Phase 1 created+committed an empty staging table; if the process died
before the rows were staged, the resume path trusted ANY existing stage and
swapped it over the intact L2 table -> all vectors silently lost (0 rows).
Post-fix: staging writes a "done" marker in the same commit as the rows, so an
interrupted stage is detected and the migration re-stages from the intact
original instead of promoting an empty stage.

Synthetic vectors / no model loads. Gated on sqlite-vec loadability.
"""
from __future__ import annotations

import sqlite3

import pytest


def _can_load_vec() -> bool:
    try:
        import sqlite_vec
        c = sqlite3.connect(":memory:")
        c.enable_load_extension(True)
        sqlite_vec.load(c)
        c.close()
        return True
    except Exception:
        return False


_HAS_VEC = _can_load_vec()
pytestmark = pytest.mark.skipif(not _HAS_VEC, reason="sqlite-vec cannot load")

DIM = 8


def _conn(tmp_path):
    import sqlite_vec
    c = sqlite3.connect(str(tmp_path / "vec.db"))
    c.enable_load_extension(True)
    sqlite_vec.load(c)
    c.enable_load_extension(False)
    return c


def _make_l2_table(c, n=5):
    """Create an OLD-format (L2-default, no distance_metric=cosine) vec0 table."""
    from truememory.vector_search import serialize_f32
    c.execute(f"CREATE VIRTUAL TABLE vec_messages USING vec0(embedding float[{DIM}])")
    for i in range(1, n + 1):
        vec = [float((i + j) % 5 + 1) for j in range(DIM)]  # nonzero, finite
        c.execute("INSERT INTO vec_messages(rowid, embedding) VALUES (?, ?)", (i, serialize_f32(vec)))
    c.commit()


def test_happy_path_migrates_and_preserves_vectors(tmp_path):
    from truememory.vector_search import migrate_to_cosine_metric, _table_uses_cosine
    c = _conn(tmp_path)
    _make_l2_table(c, n=5)
    assert not _table_uses_cosine(c, "vec_messages")

    assert migrate_to_cosine_metric(c) is True
    assert _table_uses_cosine(c, "vec_messages")
    n = c.execute("SELECT COUNT(*) FROM vec_messages").fetchone()[0]
    assert n == 5, f"expected 5 vectors carried over, got {n}"
    c.close()


def test_interrupted_stage_does_not_lose_vectors(tmp_path):
    """The D1-2 scenario: an empty stage left by a mid-copy crash must NOT be
    promoted over the intact original."""
    from truememory.vector_search import migrate_to_cosine_metric, _table_uses_cosine
    c = _conn(tmp_path)
    _make_l2_table(c, n=5)

    # Simulate a crash DURING Phase 1: the stage table exists but is empty and
    # has NO done marker (the row inserts + marker never committed). The
    # original L2 table is still fully intact.
    c.execute("CREATE TABLE vec_messages_cos_stage (rowid INTEGER PRIMARY KEY, embedding BLOB)")
    c.commit()

    assert migrate_to_cosine_metric(c) is True
    # Vectors must survive (re-staged from the intact original), NOT be wiped.
    n = c.execute("SELECT COUNT(*) FROM vec_messages").fetchone()[0]
    assert n == 5, f"interrupted stage lost vectors: got {n}, expected 5 (D1-2 regression)"
    assert _table_uses_cosine(c, "vec_messages")
    # the partial stage + its (absent) marker are cleaned up
    leftover = c.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'vec_messages_cos_stage%'"
    ).fetchone()[0]
    assert leftover == 0
    c.close()


def test_complete_stage_resume_finishes_swap(tmp_path):
    """A stage with rows + a done marker (durable crash AFTER staging) is
    promoted to finish the swap."""
    from truememory.vector_search import migrate_to_cosine_metric, serialize_f32, _table_uses_cosine
    c = _conn(tmp_path)
    _make_l2_table(c, n=3)

    # Simulate a crash AFTER staging completed (rows + done marker present) but
    # before the swap. Build the stage exactly as Phase 1 leaves it.
    c.execute("CREATE TABLE vec_messages_cos_stage (rowid INTEGER PRIMARY KEY, embedding BLOB)")
    c.execute("CREATE TABLE vec_messages_cos_stage_done (done INTEGER PRIMARY KEY)")
    for i in range(1, 4):
        v = [1.0 / (DIM ** 0.5)] * DIM  # unit vector
        c.execute("INSERT INTO vec_messages_cos_stage(rowid, embedding) VALUES (?, ?)", (i, serialize_f32(v)))
    c.execute("INSERT INTO vec_messages_cos_stage_done(done) VALUES (1)")
    c.commit()

    assert migrate_to_cosine_metric(c) is True
    assert _table_uses_cosine(c, "vec_messages")
    n = c.execute("SELECT COUNT(*) FROM vec_messages").fetchone()[0]
    assert n == 3, f"complete-stage resume should carry 3, got {n}"
    c.close()
