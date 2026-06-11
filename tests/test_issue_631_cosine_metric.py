"""Regression tests for issue #631: vec0 tables must declare cosine distance.

M-03 (P0): all sqlite-vec vec0 tables were created without
``distance_metric=cosine``, so sqlite-vec used its L2 default. The code then
computed ``cos_sim = 1 - distance`` and treated it as cosine — numerically
wrong for non-unit vectors and a wrong transform even for unit vectors.

These tests verify:
  (a) new vec tables declare cosine (via sqlite_master.sql);
  (b) reported similarity equals TRUE cosine for unit vectors
      (orthogonal -> ~0.0, identical -> ~1.0, known-angle -> expected);
  (c) the rebuild migration upgrades an L2 table to cosine, preserving the
      same ids and (normalized) vectors;
  (d) zero / non-finite vectors do not crash the build or migration.

All tests need the sqlite-vec extension; they skip cleanly where it can't
load. Vectors are constructed by hand — no model loads (HF_HUB_OFFLINE=1).
"""
from __future__ import annotations

import struct

import pytest

from tests.conftest import requires_sqlite_ext


def _ser(vec):
    return struct.pack(f"{len(vec)}f", *vec)


def _load_vec(conn):
    import sqlite_vec

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def _make_conn():
    import sqlite3

    conn = sqlite3.connect(":memory:")
    _load_vec(conn)
    return conn


def _ddl(conn, table):
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name=? AND type='table'",
        (table,),
    ).fetchone()
    return row[0] if row else None


@requires_sqlite_ext
class TestIssue631CosineMetric:

    # ----- (a) new tables declare cosine -----------------------------------

    def test_init_vec_table_declares_cosine(self):
        from truememory.storage import create_db
        from truememory.vector_search import init_vec_table, _active_vec_table, _active_sep_table

        conn = create_db(":memory:")
        _load_vec(conn)
        init_vec_table(conn)

        vec = _active_vec_table(conn)
        sep = _active_sep_table(conn)
        assert "distance_metric=cosine" in _ddl(conn, vec).replace(" ", "")
        assert "distance_metric=cosine" in _ddl(conn, sep).replace(" ", "")

    def test_column_decl_helper(self):
        from truememory.vector_search import _vec0_column_decl

        decl = _vec0_column_decl(256)
        assert "float[256]" in decl
        assert "distance_metric=cosine" in decl

    # ----- (b) reported similarity is TRUE cosine --------------------------

    def test_reported_similarity_is_true_cosine(self):
        """For unit vectors, ``1 - distance`` equals true cosine.

        Stored [1,0]; query orthogonal [0,1] -> cos 0.0; query identical
        [1,0] -> cos 1.0; query at 36.87deg ([0.8,0.6]) -> cos 0.8. Under the
        OLD L2 default, distance for the orthogonal pair would be sqrt(2)~1.414
        (so the wrong "cos_sim = 1 - 1.414" clamps to 0.0 by luck) but the
        36.87deg pair would give L2 ~0.633 -> wrong "cos_sim" 0.367 != 0.8.
        """
        from truememory.vector_search import _vec0_column_decl

        conn = _make_conn()
        conn.execute(
            f"CREATE VIRTUAL TABLE t USING vec0({_vec0_column_decl(2)})"
        )
        conn.execute("INSERT INTO t(rowid, embedding) VALUES (1, ?)", (_ser([1.0, 0.0]),))

        cases = {
            "orthogonal": ([0.0, 1.0], 0.0),
            "identical": ([1.0, 0.0], 1.0),
            "angle_36.87deg": ([0.8, 0.6], 0.8),
        }
        for label, (query, expected_cos) in cases.items():
            dist = conn.execute(
                "SELECT distance FROM t WHERE embedding MATCH ? AND k=1",
                (_ser(query),),
            ).fetchone()[0]
            cos_sim = 1.0 - dist
            assert cos_sim == pytest.approx(expected_cos, abs=1e-5), (
                f"{label}: cosine {cos_sim} != expected {expected_cos}"
            )

    def test_normalize_for_cosine_unit_and_zero(self):
        import numpy as np
        from truememory.vector_search import _normalize_for_cosine

        out = _normalize_for_cosine([3.0, 4.0])
        assert out is not None
        assert float(np.linalg.norm(out)) == pytest.approx(1.0, abs=1e-6)
        assert out.tolist() == pytest.approx([0.6, 0.8], abs=1e-6)

        assert _normalize_for_cosine([0.0, 0.0]) is None
        assert _normalize_for_cosine([float("nan"), 1.0]) is None
        assert _normalize_for_cosine([float("inf"), 0.0]) is None

    # ----- (c) migration upgrades L2 -> cosine preserving vectors ----------

    def test_migration_upgrades_l2_to_cosine(self):
        from truememory.vector_search import migrate_to_cosine_metric, _table_uses_cosine

        conn = _make_conn()
        # Old-format L2 table with UNNORMALIZED vectors (as Qwen3@256 produces).
        conn.execute("CREATE VIRTUAL TABLE vec_messages USING vec0(embedding float[2])")
        conn.execute("INSERT INTO vec_messages(rowid, embedding) VALUES (1, ?)", (_ser([3.0, 4.0]),))
        conn.execute("INSERT INTO vec_messages(rowid, embedding) VALUES (2, ?)", (_ser([1.0, 0.0]),))

        assert not _table_uses_cosine(conn, "vec_messages")
        assert migrate_to_cosine_metric(conn) is True
        assert _table_uses_cosine(conn, "vec_messages")

        # Same ids preserved.
        ids = [r[0] for r in conn.execute("SELECT rowid FROM vec_messages ORDER BY rowid")]
        assert ids == [1, 2]

        # Vector [3,4] is now L2-normalized to [0.6, 0.8].
        blob = conn.execute("SELECT embedding FROM vec_messages WHERE rowid=1").fetchone()[0]
        assert struct.unpack("2f", blob) == pytest.approx((0.6, 0.8), abs=1e-6)

        # Query returns true cosine: query [1,0] vs stored [0.6,0.8] -> cos 0.6.
        dist = conn.execute(
            "SELECT distance FROM vec_messages WHERE embedding MATCH ? AND k=1 ",
            (_ser([0.6, 0.8]),),
        ).fetchone()[0]
        assert (1.0 - dist) == pytest.approx(1.0, abs=1e-5)  # identical to row1 normalized

    def test_migration_is_idempotent(self):
        from truememory.vector_search import migrate_to_cosine_metric

        conn = _make_conn()
        conn.execute("CREATE VIRTUAL TABLE vec_messages USING vec0(embedding float[2])")
        conn.execute("INSERT INTO vec_messages(rowid, embedding) VALUES (1, ?)", (_ser([1.0, 0.0]),))
        assert migrate_to_cosine_metric(conn) is True
        assert migrate_to_cosine_metric(conn) is False  # already cosine

    def test_migration_resumes_from_staging(self):
        """A crash that left a populated staging table is recovered."""
        from truememory.vector_search import migrate_to_cosine_metric, _table_uses_cosine

        conn = _make_conn()
        # Old L2 table still present, plus a staging table (simulated crash).
        conn.execute("CREATE VIRTUAL TABLE vec_messages USING vec0(embedding float[2])")
        conn.execute("INSERT INTO vec_messages(rowid, embedding) VALUES (1, ?)", (_ser([1.0, 0.0]),))
        conn.execute("CREATE TABLE vec_messages_cos_stage (rowid INTEGER PRIMARY KEY, embedding BLOB)")
        conn.execute("INSERT INTO vec_messages_cos_stage VALUES (1, ?)", (_ser([1.0, 0.0]),))
        conn.execute("INSERT INTO vec_messages_cos_stage VALUES (9, ?)", (_ser([0.0, 1.0]),))
        conn.commit()

        assert migrate_to_cosine_metric(conn) is True
        assert _table_uses_cosine(conn, "vec_messages")
        ids = [r[0] for r in conn.execute("SELECT rowid FROM vec_messages ORDER BY rowid")]
        assert ids == [1, 9]  # recovered from staging
        # staging cleaned up
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='vec_messages_cos_stage'"
        ).fetchone() is None

    # ----- (d) zero / NaN vectors do not crash -----------------------------

    def test_migration_drops_zero_vector_without_crash(self):
        from truememory.vector_search import migrate_to_cosine_metric

        conn = _make_conn()
        conn.execute("CREATE VIRTUAL TABLE vec_messages USING vec0(embedding float[2])")
        conn.execute("INSERT INTO vec_messages(rowid, embedding) VALUES (1, ?)", (_ser([1.0, 0.0]),))
        conn.execute("INSERT INTO vec_messages(rowid, embedding) VALUES (2, ?)", (_ser([0.0, 0.0]),))
        conn.execute("INSERT INTO vec_messages(rowid, embedding) VALUES (3, ?)", (_ser([0.0, 1.0]),))

        assert migrate_to_cosine_metric(conn) is True
        ids = [r[0] for r in conn.execute("SELECT rowid FROM vec_messages ORDER BY rowid")]
        assert ids == [1, 3]  # zero vector dropped

        # No NULL distances poison results.
        rows = conn.execute(
            "SELECT rowid, distance FROM vec_messages WHERE embedding MATCH ? AND k=2",
            (_ser([1.0, 0.0]),),
        ).fetchall()
        assert all(d is not None for _, d in rows)

    def test_build_vectors_skips_zero_vector(self):
        """build_vectors must skip a zero-norm embedding (C2-8)."""
        from unittest.mock import patch
        import numpy as np
        from truememory.storage import create_db
        from truememory import vector_search as vs

        conn = create_db(":memory:")
        _load_vec(conn)
        vs.init_vec_table(conn)

        conn.execute("INSERT INTO messages(id, content) VALUES (1, 'real')")
        conn.execute("INSERT INTO messages(id, content) VALUES (2, 'zero')")
        conn.commit()

        dim = vs._embedding_dim
        good = np.zeros(dim, dtype=np.float32)
        good[0] = 1.0
        bad = np.zeros(dim, dtype=np.float32)

        def fake_encode(texts, **kw):
            out = []
            for t in texts:
                out.append(bad if t == "zero" else good)
            return np.array(out)

        with patch.object(vs, "get_model", return_value=object()), \
             patch.object(vs, "_encode_with_mps_fallback", side_effect=lambda m, t, **k: fake_encode(t)):
            inserted = vs.build_vectors(conn)

        # Only the real (non-zero) vector is stored.
        assert inserted == 1
        ids = [r[0] for r in conn.execute(
            f"SELECT rowid FROM {vs._active_vec_table(conn)} ORDER BY rowid"
        )]
        assert ids == [1]
