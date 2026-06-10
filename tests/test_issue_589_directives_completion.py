"""Regression suite for issue #589: directives completion (lands the WIP safely).

Covers, in finding order:

- D-2  — legacy (pre-directive, v0.7.5.0-shaped) DBs must upgrade cleanly, and a
         failed pre-migration backup must NOT leave the DB unmigrated or crash
         open. Open must NEVER raise ``no such column: directive``.
- D-8  — the encoding gate must exclude directives BEFORE caching search
         results, so prediction-error and ``similar_memory`` never see them.
- D-4  — session-start directive injection is capped, sender-scoping must not
         hide ``sender=''`` directives, and load errors are logged, not
         swallowed.
- D-6  — ``delete_message`` cascades into the ``custom`` tier vector tables.
- D-7  — Gemini/Cursor/Codex adapter templates + the base.py fallback + user
         docs all carry directive guidance.
- WIP  — full lifecycle: store directive -> search returns it flagged ->
         stats counts it -> forget removes messages + FTS + vec rows.

The legacy schema below is the verbatim messages/FTS section of
``git show v0.7.5.0:truememory/storage.py`` (the last release before the
``directive`` column existed).
"""

from __future__ import annotations

import logging
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# v0.7.5.0 legacy schema (pre-directive) — extracted from the tagged release
# ---------------------------------------------------------------------------

PRE_DIRECTIVE_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    sender TEXT DEFAULT '',
    recipient TEXT DEFAULT '',
    timestamp TEXT DEFAULT '',
    category TEXT DEFAULT '',
    modality TEXT DEFAULT '',
    episode_id INTEGER DEFAULT NULL,
    emotional_valence REAL DEFAULT 0.0,
    embedding_separation BLOB DEFAULT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content, sender, recipient, category, modality,
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content, sender, recipient, category, modality)
    VALUES (new.id, new.content, new.sender, new.recipient, new.category, new.modality);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
    INSERT INTO messages_fts(rowid, content, sender, recipient, category, modality)
    VALUES (new.id, new.content, new.sender, new.recipient, new.category, new.modality);
END;
"""


def _make_legacy_db(db_path, rows=()):
    conn = sqlite3.connect(str(db_path))
    conn.executescript(PRE_DIRECTIVE_SCHEMA)
    for content in rows:
        conn.execute("INSERT INTO messages (content) VALUES (?)", (content,))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# D-2: migration / upgrade path
# ---------------------------------------------------------------------------


def test_issue_589_pre_directive_db_upgrades_cleanly(tmp_path):
    """Happy path (pins existing behavior): a v0.7.5.0 DB opens, the
    directive column is added by migration, legacy rows survive, directive
    ops work, and a pre-migration backup is created."""
    from truememory.storage import create_db, insert_message

    db = tmp_path / "old.db"
    _make_legacy_db(db, rows=["legacy fact one", "legacy fact two", "legacy fact three"])

    conn = create_db(db)  # must not raise
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()}
        assert "directive" in cols
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 3
        insert_message(conn, {"content": "always respond in lowercase", "directive": True})
        assert (
            conn.execute("SELECT COUNT(*) FROM messages WHERE directive = 1").fetchone()[0]
            == 1
        )
    finally:
        conn.close()
    assert list(tmp_path.glob("old.db.backup-pre-migration-*")), (
        "pre-migration backup must be created on the happy path"
    )


def test_issue_589_skipped_migration_cannot_crash_open(tmp_path, monkeypatch, caplog):
    """Exact failing scenario (repro 5d): when the pre-migration backup fails,
    the migration must PROCEED anyway (loud warning), so open succeeds and
    directive ops still work. The old behavior skipped the migration entirely,
    which the WIP's directive index turned into a hard crash at open."""
    from truememory.storage import create_db, insert_message

    db = tmp_path / "old.db"
    _make_legacy_db(db, rows=["legacy fact"])

    monkeypatch.setattr("truememory.storage._backup_database", lambda p: None)

    with caplog.at_level(logging.WARNING, logger="truememory.storage"):
        conn = create_db(db)  # must not raise — D-2
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()}
        assert "directive" in cols, (
            "migration must proceed even when the backup fails (D-2)"
        )
        # The proceed-anyway path must be loud.
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("backup" in w.lower() for w in warnings), (
            f"expected a loud backup-failure warning, got: {warnings}"
        )
        # Directive ops work on the migrated DB.
        insert_message(conn, {"content": "always run tests", "directive": True})
        assert (
            conn.execute("SELECT COUNT(*) FROM messages WHERE directive = 1").fetchone()[0]
            == 1
        )
        # Legacy rows intact.
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM messages WHERE content = 'legacy fact'"
            ).fetchone()[0]
            == 1
        )
    finally:
        conn.close()


def test_issue_589_open_never_raises_no_such_column(tmp_path, monkeypatch):
    """Defense in depth: even if the migration cannot run AT ALL (simulated by
    no-opping it), opening a legacy DB must never raise
    ``no such column: directive`` — the directive index must be guarded by a
    column-existence check, not baked unconditionally into _SCHEMA_SQL."""
    from truememory import storage

    db = tmp_path / "old.db"
    _make_legacy_db(db, rows=["legacy fact"])

    monkeypatch.setattr(storage, "_migrate_messages_schema", lambda conn, path: None)

    conn = storage.create_db(db)  # must not raise, even degraded
    try:
        # Degraded-but-openable: reads of legacy data still work.
        assert (
            conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# D-6: forget cascade for the custom tier group
# ---------------------------------------------------------------------------


def test_issue_589_forget_directive_cascades_custom_tier(tmp_path):
    """delete_message must clear vec_messages_custom / vec_messages_sep_custom
    (plain tables stand in for the vec0 virtual tables — DELETE semantics are
    identical)."""
    from truememory.storage import create_db, delete_message, insert_message

    conn = create_db(tmp_path / "custom.db")
    try:
        mid = insert_message(conn, {"content": "always use tabs", "directive": True})
        conn.execute(
            "CREATE TABLE IF NOT EXISTS vec_messages_custom "
            "(rowid INTEGER PRIMARY KEY, embedding BLOB)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS vec_messages_sep_custom "
            "(rowid INTEGER PRIMARY KEY, embedding BLOB)"
        )
        conn.execute(
            "INSERT INTO vec_messages_custom(rowid, embedding) VALUES (?, x'00')", (mid,)
        )
        conn.execute(
            "INSERT INTO vec_messages_sep_custom(rowid, embedding) VALUES (?, x'00')",
            (mid,),
        )
        conn.commit()

        assert delete_message(conn, mid) is True

        assert (
            conn.execute(
                "SELECT COUNT(*) FROM vec_messages_custom WHERE rowid = ?", (mid,)
            ).fetchone()[0]
            == 0
        ), "custom-tier vector row must be cascaded on delete (D-6)"
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM vec_messages_sep_custom WHERE rowid = ?", (mid,)
            ).fetchone()[0]
            == 0
        ), "custom-tier separation vector row must be cascaded on delete (D-6)"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# D-4: injection cap + sender scoping + logged load errors
# ---------------------------------------------------------------------------


def test_issue_589_directive_visible_regardless_of_user_scope(tmp_path):
    """A directive stored with the MCP default ``user_id=''`` (sender='') must
    still be injected when the session hook runs with --user scoping."""
    from truememory import Memory
    from truememory.ingest.hooks.session_start import _load_directives

    m = Memory(path=str(tmp_path / "scope.db"))
    try:
        m.add("always reply in lowercase", directive=True)  # sender=''
        m.add("always run the linter", user_id="josh", directive=True)
        m.add("directive for someone else", user_id="sam", directive=True)

        got = _load_directives(m, user_id="josh")
        contents = {d["content"] for d in got}
        assert "always reply in lowercase" in contents, (
            "sender='' directives must be visible under --user scoping (D-4)"
        )
        assert "always run the linter" in contents
        assert "directive for someone else" not in contents, (
            "other users' directives must stay scoped out"
        )

        # Unscoped load still returns everything.
        all_got = _load_directives(m)
        assert len(all_got) == 3
    finally:
        m.close()


def test_issue_589_directive_injection_cap(tmp_path, caplog):
    """More directives than the cap -> only DIRECTIVE_LIMIT are injected, a
    warning is logged, and the block carries an overflow note pointing at
    truememory_directives."""
    from truememory.storage import create_db, insert_message

    db = tmp_path / "cap.db"
    conn = create_db(db)
    for i in range(60):
        insert_message(conn, {"content": f"directive number {i:03d}", "directive": True})
    conn.commit()
    conn.close()

    from truememory.ingest.hooks.session_start import DIRECTIVE_LIMIT, recall_memories

    assert DIRECTIVE_LIMIT == 50

    with caplog.at_level(logging.WARNING):
        ctx = recall_memories({}, db_path=str(db))

    assert "<truememory-directives>" in ctx
    block = ctx.split("<truememory-directives>")[1].split("</truememory-directives>")[0]
    bullets = [ln for ln in block.splitlines() if ln.startswith("- directive number")]
    assert len(bullets) == DIRECTIVE_LIMIT, (
        f"expected exactly {DIRECTIVE_LIMIT} injected directives, got {len(bullets)}"
    )
    assert "truememory_directives" in block, (
        "overflow note must point the agent at truememory_directives"
    )
    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("directive" in w.lower() for w in warnings), (
        f"cap truncation must be logged, got: {warnings}"
    )


def test_issue_589_directive_load_errors_are_logged(tmp_path, caplog, monkeypatch):
    """_load_directives must log failures instead of silently returning []."""
    from truememory import Memory
    from truememory.ingest.hooks.session_start import _load_directives

    m = Memory(path=str(tmp_path / "err.db"))
    try:
        m.add("always reply in lowercase", directive=True)

        def _boom(*a, **kw):
            raise sqlite3.OperationalError("no such column: directive")

        monkeypatch.setattr(m._engine, "_ensure_connection", _boom)
        with caplog.at_level(logging.WARNING):
            got = _load_directives(m)
        assert got == []
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("directive" in w.lower() for w in warnings), (
            f"directive load failure must be logged, got: {warnings}"
        )
    finally:
        m.close()


# ---------------------------------------------------------------------------
# D-8: encoding gate must not feed directives into PE / similar_memory
# ---------------------------------------------------------------------------


class _StubMemory:
    """Memory stub whose only search hit is a near-identical directive."""

    def __init__(self, results):
        self._results = results

    def search_vectors(self, fact, limit=10):
        return [dict(r) for r in self._results]

    def search(self, query, user_id=None, limit=5):
        return [dict(r) for r in self._results]


def test_issue_589_gate_excludes_directives_from_pe():
    from truememory.ingest.encoding_gate import EncodingGate

    directive_row = {
        "id": 1,
        "content": "always respond to emails in lowercase",
        "directive": True,
        "score": 0.99,
    }
    gate = EncodingGate(memory=_StubMemory([directive_row]))
    decision = gate.evaluate("respond to emails in lowercase always")

    assert all(not r.get("directive") for r in gate._last_search_results), (
        "_last_search_results must never contain directives (D-8): "
        f"{gate._last_search_results}"
    )
    assert decision.similar_memory == "", (
        "similar_memory must not expose directive text (D-8)"
    )
    # With the directive filtered out, memory is effectively empty: PE must
    # match the empty-memory baseline (0.0) and novelty the empty-memory max.
    assert decision.prediction_error == 0.0
    assert decision.novelty == 1.0


def test_issue_589_gate_still_sees_real_facts():
    """Sanity: non-directive results still flow into the gate's cache."""
    from truememory.ingest.encoding_gate import EncodingGate

    fact_row = {"id": 2, "content": "user works at Anthropic", "score": 0.9}
    gate = EncodingGate(memory=_StubMemory([fact_row]))
    gate.evaluate("user works at Anthropic as an engineer")
    assert gate._last_search_results, "real facts must remain visible to the gate"
    assert gate._last_search_results[0]["content"] == "user works at Anthropic"


# ---------------------------------------------------------------------------
# WIP surface: full directive lifecycle (store -> search -> stats -> forget)
# ---------------------------------------------------------------------------


def test_issue_589_directive_full_lifecycle(tmp_path):
    """Embeddings use the house stub (``vector_search.get_model`` patched to a
    deterministic MagicMock, as in test_issue_493_cascade_delete.py /
    test_issue_462_delete_cascade.py) so the test is embedder-agnostic: it
    behaves identically on the no-network CI gate (where the real model cannot
    load and ``engine.add`` would degrade to storing without an embedding) and
    on dev machines with cached models."""
    import numpy as np

    from truememory import Memory

    fake_embedding = np.random.rand(256).astype(np.float32)
    mock_model = MagicMock()
    mock_model.encode = lambda texts, **kw: np.array([fake_embedding] * len(texts))

    with patch("truememory.vector_search.get_model", return_value=mock_model):
        m = Memory(path=str(tmp_path / "life.db"))
        try:
            stored = m.add("always sign commits with GPG", directive=True)
            mid = stored["id"]
            m.add("user prefers vim keybindings")

            # Search returns the directive flagged.
            # When sqlite-vec is unavailable the engine falls back to FTS-only
            # mode.  The full search() pipeline (salience guard, quality
            # self-check, etc.) can legitimately return [] for short DBs with
            # only two rows, so we fall back to a raw FTS probe when vectors
            # are missing.
            if m._engine._has_vectors:
                results = m._engine.search(
                    "sign commits GPG", limit=5,
                    _skip_reranker=True, include_directives=True,
                )
                hit = next((r for r in results if r.get("id") == mid), None)
                assert hit is not None, f"directive not returned by search: {results}"
                assert hit.get("directive") is True, (
                    "search results must carry the directive flag (WIP propagation)"
                )
                default_results = m._engine.search(
                    "sign commits GPG", limit=5, _skip_reranker=True,
                )
                assert all(
                    r.get("directive") is not True for r in default_results
                ), "default search must exclude directive rows"
                fact_hits = m._engine.search("vim keybindings", limit=5, _skip_reranker=True)
                assert fact_hits and all(
                    r.get("directive") is False for r in fact_hits if r.get("id") != mid
                ), "regular facts must be flagged directive=False"
            else:
                # FTS-only fallback: verify the directive flag propagates
                # through direct FTS retrieval (bypasses salience/quality
                # stages that may drop results in a tiny corpus).
                from truememory.fts_search import search_fts

                results = search_fts(m._engine.conn, "sign commits GPG", limit=5, include_directives=True)
                hit = next((r for r in results if r.get("id") == mid), None)
                assert hit is not None, (
                    f"directive not returned by FTS search (FTS-only mode): {results}"
                )
                assert hit.get("directive") is True, (
                    "FTS results must carry the directive flag (WIP propagation)"
                )

            # Stats counts it.
            stats = m._engine.get_stats()
            assert stats.get("directive_count") == 1, (
                f"get_stats must report directive_count, got: {stats.keys()}"
            )

            conn = m._engine.conn
            # Only the vec0 virtual tables themselves — not their _info/
            # _chunks/_rowids shadow tables, which hold table-level metadata,
            # not per-message rows.
            vec_tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE name LIKE 'vec_messages%' AND sql LIKE '%USING vec0%'"
                ).fetchall()
            ]
            if m._engine._has_vectors:
                # With the stubbed model, add() embeds deterministically even
                # on the no-network gate (sqlite-vec is a local pip package).
                pre = sum(
                    conn.execute(
                        f"SELECT COUNT(*) FROM {t} WHERE rowid = ?", (mid,)
                    ).fetchone()[0]
                    for t in vec_tables
                )
                assert pre >= 1, "directive must be embedded at add time"
            assert conn.execute(
                "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'gpg'"
            ).fetchone()[0] >= 1

            # Forget removes messages + FTS + vec rows.
            assert m.delete(mid) is True
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE id = ?", (mid,)
                ).fetchone()[0]
                == 0
            )
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'gpg'"
                ).fetchone()[0]
                == 0
            ), "FTS row must be removed on forget"
            for t in vec_tables:
                assert (
                    conn.execute(
                        f"SELECT COUNT(*) FROM {t} WHERE rowid = ?", (mid,)
                    ).fetchone()[0]
                    == 0
                ), f"vec row must be removed from {t} on forget"

            stats = m._engine.get_stats()
            assert stats.get("directive_count") == 0
        finally:
            m.close()


# ---------------------------------------------------------------------------
# D-7: adapter/docs parity
# ---------------------------------------------------------------------------


def _adapter_templates():
    from truememory.hooks.adapters import base as base_mod
    from truememory.hooks.adapters.codex import _AGENTS_TEMPLATE
    from truememory.hooks.adapters.cursor import (
        _SYSTEM_PROMPT_TEMPLATE as cursor_template,
    )
    from truememory.hooks.adapters.gemini import (
        _SYSTEM_PROMPT_TEMPLATE as gemini_template,
    )

    return {
        "gemini": gemini_template,
        "cursor": cursor_template,
        "codex": _AGENTS_TEMPLATE,
        "base-fallback": base_mod._FALLBACK_PROMPT,
    }


@pytest.mark.parametrize("adapter", ["gemini", "cursor", "codex", "base-fallback"])
def test_issue_589_all_adapters_mention_directives(adapter):
    text = _adapter_templates()[adapter]
    lower = text.lower()
    assert "directive" in lower, f"{adapter} template must mention directives (D-7)"
    assert "directive=true" in lower, (
        f"{adapter} template must show the directive=True store call (D-7)"
    )
    assert "always" in lower and "never" in lower, (
        f"{adapter} template must include the always/never trigger phrases (D-7)"
    )


def test_issue_589_base_fallback_is_served_when_template_missing(monkeypatch):
    """The hardened fallback (with directive guidance) is what non-Claude CLIs
    get when CLAUDE_TEMPLATE.md is unreadable."""
    from pathlib import Path

    from truememory.hooks.adapters import base as base_mod

    monkeypatch.setattr(Path, "exists", lambda self: False)
    prompt = base_mod.get_generic_system_prompt()
    assert "directive" in prompt.lower()


def test_issue_589_user_docs_mention_directives():
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    for doc in ("README.md", "docs/mcp-tools.md"):
        text = (repo / doc).read_text(encoding="utf-8").lower()
        assert "directive" in text, f"{doc} must document directives (D-7)"
    mcp_doc = (repo / "docs/mcp-tools.md").read_text(encoding="utf-8")
    assert "truememory_directives" in mcp_doc, (
        "docs/mcp-tools.md must document the truememory_directives tool (D-7)"
    )
