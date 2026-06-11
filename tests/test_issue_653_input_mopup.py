"""Regression locks for issue #653 — input/platform mop-up.

Covers:
- M-33: create_db() rejects a non-path object (e.g. a sqlite3.Connection)
  with TypeError instead of stringifying it into a stray file.
- M-60: Engine.add() enforces the 50KB content cap at the engine level.
- M-90: a transcript_path outside the expected transcripts root is rejected
  by the compact hook (never read into the store).
- M-95: the compact hook tolerates non-string stdin fields (int session_id /
  transcript_path) without raising TypeError.
- M-70: empty-timestamp rows are excluded by the temporal filter.
- M-97: the param-safety shim matches context-window-suffixed IDs like
  ``claude-fable-5[1m]``.

All tests use tmp dirs and never load models (HF_HUB_OFFLINE handled by the
suite). No network, no embeddings required for the assertions.
"""
from __future__ import annotations

import io
import os
import sqlite3
import sys

import pytest


# ---------------------------------------------------------------------------
# M-33 — create_db type guard
# ---------------------------------------------------------------------------

def test_create_db_rejects_connection_object(tmp_path):
    from truememory.storage import create_db

    conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(TypeError):
            create_db(conn)
    finally:
        conn.close()

    # No stray "<sqlite3.Connection object at 0x...>" file was created.
    strays = [p for p in os.listdir(".") if p.startswith("<sqlite3.Connection")]
    assert not strays, f"stray connection file(s) created: {strays}"


def test_create_db_accepts_str_and_pathlike(tmp_path):
    from truememory.storage import create_db

    p = tmp_path / "ok.db"
    conn = create_db(str(p))
    conn.close()
    conn2 = create_db(p)  # PathLike
    conn2.close()
    assert p.exists()


# ---------------------------------------------------------------------------
# M-60 — Engine.add content cap
# ---------------------------------------------------------------------------

def test_engine_add_rejects_oversize_content(tmp_path):
    from truememory.engine import TrueMemoryEngine, MAX_CONTENT_LENGTH

    eng = TrueMemoryEngine(db_path=str(tmp_path / "m.db"))
    big = "x" * (MAX_CONTENT_LENGTH + 1)
    with pytest.raises(ValueError):
        eng.add(content=big)


def test_engine_add_accepts_boundary_content(tmp_path):
    from truememory.engine import TrueMemoryEngine, MAX_CONTENT_LENGTH

    eng = TrueMemoryEngine(db_path=str(tmp_path / "m.db"))
    ok = "y" * MAX_CONTENT_LENGTH
    res = eng.add(content=ok)
    assert res.get("id")


# ---------------------------------------------------------------------------
# M-90 / M-95 — compact hook transcript validation + non-string stdin
# ---------------------------------------------------------------------------

def test_compact_rejects_transcript_outside_root(tmp_path, monkeypatch):
    from truememory.ingest.hooks import compact

    # An attacker-chosen file outside the transcripts root.
    evil = tmp_path / "secret.txt"
    evil.write_text("password=hunter2", encoding="utf-8")
    # Point the root somewhere else entirely.
    monkeypatch.setenv("TRUEMEMORY_TRANSCRIPT_DIR", str(tmp_path / "allowed"))
    assert compact._is_allowed_transcript(str(evil)) is False


def test_compact_allows_transcript_inside_root(tmp_path, monkeypatch):
    from truememory.ingest.hooks import compact

    root = tmp_path / "allowed"
    root.mkdir()
    good = root / "t.jsonl"
    good.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("TRUEMEMORY_TRANSCRIPT_DIR", str(root))
    assert compact._is_allowed_transcript(str(good)) is True


def test_compact_hook_tolerates_non_string_stdin(monkeypatch):
    from truememory.ingest.hooks import compact

    # int session_id and transcript_path would crash a naive str/Path op.
    payload = '{"session_id": 12345, "transcript_path": 67890}'
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))
    # Must not raise TypeError; non-existent transcript path -> early return.
    compact.main()


# ---------------------------------------------------------------------------
# M-70 — empty-timestamp rows excluded by temporal filter
# ---------------------------------------------------------------------------

def test_temporal_filter_excludes_empty_timestamp(tmp_path):
    from truememory.storage import create_db, insert_message
    from truememory.fts_search import search_fts_in_range

    conn = create_db(str(tmp_path / "fts.db"))
    try:
        insert_message(conn, {
            "content": "alpha widget report",
            "sender": "u", "recipient": "", "timestamp": "2026-01-15",
            "category": "", "modality": "", "directive": False, "metadata": None,
        })
        insert_message(conn, {
            "content": "alpha widget summary",
            "sender": "u", "recipient": "", "timestamp": "",
            "category": "", "modality": "", "directive": False, "metadata": None,
        })
        conn.commit()

        # Upper-bound-only filter: the empty-timestamp row must NOT pass.
        out = search_fts_in_range(conn, "alpha", after=None, before="2026-12-31")
        ts = [r["timestamp"] for r in out]
        assert "" not in ts, f"empty-timestamp row leaked through: {ts}"
        assert "2026-01-15" in ts

        # Lower-bound-only filter: still excluded.
        out2 = search_fts_in_range(conn, "alpha", after="2026-01-01", before=None)
        assert "" not in [r["timestamp"] for r in out2]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# M-97 — param shim matches context-window-suffixed model IDs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_id", [
    "claude-fable-5[1m]",
    "claude-fable-5",
    "claude-fable-5-20260101",
    "anthropic/claude-fable-5[1m]",
    "claude-opus-4-8[1m]",
])
def test_param_shim_strips_for_strict_models(model_id):
    from truememory.ingest.models import sanitize_model_params

    out = sanitize_model_params(model_id, {"temperature": 0.5, "top_p": 0.9})
    assert "temperature" not in out
    assert "top_p" not in out


def test_param_shim_passthrough_for_other_models():
    from truememory.ingest.models import sanitize_model_params

    out = sanitize_model_params("gpt-4o-mini", {"temperature": 0.5})
    assert out.get("temperature") == 0.5
