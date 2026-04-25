"""MEMORIST-L4 regression tests.

Ensures the `build_entity_summary_sheets` function is disabled by default
in `TrueMemoryEngine.consolidate()`, that legacy `period='entity_profile'`
rows are purged on `open()`, and that the escape-hatch env var
`TRUEMEMORY_ENTITY_SHEETS=1` re-enables the function.

Context: MEMORIST-L4 research session (2026-04-23) found that the function
produces fat profile rows that saturate top-1 retrieval by keyword match
and leak superseded facts into contradiction scoring. Disabling produced
+5.3 pts on the composite L4 probe metric. See
``_working/memorist/l4_consolidation/REPORT.md`` §3, §10.7.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import tempfile
import warnings
from pathlib import Path

import pytest


def _seed_messages(tmp_path):
    """Build a convo JSON with enough per-sender messages (>=5) to
    trigger build_entity_summary_sheets when enabled."""
    tmp_json = tmp_path / "convo.json"
    messages = [
        {"content": "I live in Boston and work at Stripe.",
         "sender": "alice", "recipient": "bob",
         "timestamp": "2026-01-05T10:00:00Z", "category": "session_1",
         "modality": "conversation"},
        {"content": "Boston winters are tough.",
         "sender": "alice", "recipient": "bob",
         "timestamp": "2026-01-06T10:00:00Z", "category": "session_1",
         "modality": "conversation"},
        {"content": "Thinking about leaving Stripe soon.",
         "sender": "alice", "recipient": "bob",
         "timestamp": "2026-01-10T10:00:00Z", "category": "session_1",
         "modality": "conversation"},
        {"content": "Got an offer from a new startup.",
         "sender": "alice", "recipient": "bob",
         "timestamp": "2026-02-01T10:00:00Z", "category": "session_2",
         "modality": "conversation"},
        {"content": "I accepted the offer. Moving to Austin.",
         "sender": "alice", "recipient": "bob",
         "timestamp": "2026-02-15T10:00:00Z", "category": "session_2",
         "modality": "conversation"},
        {"content": "Austin is hot but I love it here.",
         "sender": "alice", "recipient": "bob",
         "timestamp": "2026-03-01T10:00:00Z", "category": "session_3",
         "modality": "conversation"},
    ]
    tmp_json.write_text(json.dumps(messages))
    return tmp_json


@pytest.fixture
def seeded_engine(tmp_path, monkeypatch):
    """Fresh DB ingested via TrueMemoryEngine.ingest() — which runs the
    full consolidation pipeline including (default) the L4-disable path."""
    from truememory.engine import TrueMemoryEngine

    monkeypatch.delenv("TRUEMEMORY_ENTITY_SHEETS", raising=False)
    tmp_json = _seed_messages(tmp_path)
    db_path = tmp_path / "l4_test.db"

    eng = TrueMemoryEngine(db_path)
    stats = eng.ingest(str(tmp_json))
    eng.close()
    return db_path, stats


def test_disabled_by_default_no_entity_profile_rows(seeded_engine):
    """Default v0.6.0 behavior: ingest() must NOT write any summaries
    rows with period='entity_profile'."""
    db_path, _stats = seeded_engine
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT COUNT(*) FROM summaries WHERE period = 'entity_profile'"
    ).fetchone()[0]
    conn.close()
    assert rows == 0, (
        f"Expected 0 entity_profile rows after default ingest(), got {rows}. "
        "build_entity_summary_sheets should be disabled by default; see "
        "MEMORIST-L4 REPORT.md."
    )


def test_stats_reports_disabled_string(seeded_engine):
    """The ingest() stats dict must explicitly flag the feature as
    DISABLED so observability tools can surface the state."""
    _db_path, stats = seeded_engine
    assert "entity_summary_sheets" in stats
    assert "DISABLED" in stats["entity_summary_sheets"]
    assert "TRUEMEMORY_ENTITY_SHEETS" in stats["entity_summary_sheets"]


def test_env_var_re_enables(tmp_path, monkeypatch):
    """TRUEMEMORY_ENTITY_SHEETS=1 re-enables the (deprecated) function."""
    monkeypatch.setenv("TRUEMEMORY_ENTITY_SHEETS", "1")

    from truememory.engine import TrueMemoryEngine

    tmp_json = _seed_messages(tmp_path)
    db_path = tmp_path / "reenabled_test.db"

    # Suppress the deprecation warning during intentional re-enable.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        eng = TrueMemoryEngine(db_path)
        stats = eng.ingest(str(tmp_json))
        eng.close()

    assert "re-enabled via TRUEMEMORY_ENTITY_SHEETS=1" in stats["entity_summary_sheets"]

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT COUNT(*) FROM summaries WHERE period = 'entity_profile'"
    ).fetchone()[0]
    conn.close()
    assert rows > 0, (
        "With TRUEMEMORY_ENTITY_SHEETS=1, entity_profile summary rows "
        "should be written."
    )


def test_startup_migration_purges_legacy_rows(tmp_path, monkeypatch):
    """Upgraders arriving with an existing DB that has
    period='entity_profile' rows (produced by v0.5.0) must have those
    rows purged on open() so the MEMORIST-L4 benefit is immediate."""
    from truememory.engine import TrueMemoryEngine
    from truememory.storage import create_db

    # Simulate a v0.5.0 database with legacy rows.
    db_path = tmp_path / "legacy.db"
    conn = create_db(db_path)
    # Ensure summaries table exists with schema.
    conn.execute(
        "INSERT INTO summaries (entity, period, start_date, end_date, summary, message_ids) "
        "VALUES ('alice', 'entity_profile', '2026-01-01', '2026-03-01', "
        "'Entity Profile: alice. Lives in Boston, moved to Austin.', '[1,2,3,4,5]')"
    )
    conn.commit()
    conn.close()

    # Confirm the legacy row exists before open().
    conn2 = sqlite3.connect(str(db_path))
    pre = conn2.execute(
        "SELECT COUNT(*) FROM summaries WHERE period='entity_profile'"
    ).fetchone()[0]
    conn2.close()
    assert pre == 1, "Test setup: legacy row should exist before open()"

    # Open and let the migration run. Default = env not set.
    monkeypatch.delenv("TRUEMEMORY_ENTITY_SHEETS", raising=False)
    eng = TrueMemoryEngine(db_path).open(rebuild_vectors=False)
    eng.close()

    # Legacy row should be gone.
    conn3 = sqlite3.connect(str(db_path))
    post = conn3.execute(
        "SELECT COUNT(*) FROM summaries WHERE period='entity_profile'"
    ).fetchone()[0]
    conn3.close()
    assert post == 0, (
        f"MEMORIST-L4 migration should have purged legacy entity_profile "
        f"rows on open(); got {post} remaining."
    )


def test_startup_migration_skipped_when_re_enabled(tmp_path, monkeypatch):
    """If the user explicitly re-enables the feature, the purge should
    NOT run — their next consolidate() will rewrite the rows anyway."""
    from truememory.engine import TrueMemoryEngine
    from truememory.storage import create_db

    db_path = tmp_path / "reenabled.db"
    conn = create_db(db_path)
    conn.execute(
        "INSERT INTO summaries (entity, period, start_date, end_date, summary, message_ids) "
        "VALUES ('alice', 'entity_profile', '2026-01-01', '2026-03-01', "
        "'Entity Profile: alice.', '[1,2,3]')"
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("TRUEMEMORY_ENTITY_SHEETS", "1")
    eng = TrueMemoryEngine(db_path).open(rebuild_vectors=False)
    eng.close()

    conn2 = sqlite3.connect(str(db_path))
    rows = conn2.execute(
        "SELECT COUNT(*) FROM summaries WHERE period='entity_profile'"
    ).fetchone()[0]
    conn2.close()
    assert rows == 1, (
        "With TRUEMEMORY_ENTITY_SHEETS=1, migration should not purge; "
        "got %d rows (expected 1 preserved)." % rows
    )


def test_deprecation_warning_on_direct_call():
    """Calling build_entity_summary_sheets directly must emit a
    DeprecationWarning so future developers see the MEMORIST-L4 context
    without having to read the REPORT."""
    from truememory.consolidation import build_entity_summary_sheets
    from truememory.storage import create_db

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "warn.db"
        conn = create_db(db_path)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                build_entity_summary_sheets(conn)
            except Exception:
                # The function may raise on empty DB, but the warning
                # must already have been emitted at the top of the body.
                pass

        conn.close()

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep_warnings, (
        "build_entity_summary_sheets must emit DeprecationWarning when "
        "called directly (MEMORIST-L4)."
    )
    msg = str(dep_warnings[0].message)
    assert "MEMORIST-L4" in msg
    assert "TRUEMEMORY_ENTITY_SHEETS" in msg


# ── Rustle-the-feathers remediation tests (PR 77 follow-up) ──────────────

def _seed_legacy_db(tmp_path, n_rows: int):
    """Create a v0.5.0-shape DB with N legacy entity_profile rows."""
    from truememory.storage import create_db

    db_path = tmp_path / f"legacy_{n_rows}.db"
    conn = create_db(db_path)
    for i in range(n_rows):
        conn.execute(
            "INSERT INTO summaries (entity, period, start_date, end_date, "
            "summary, message_ids) VALUES (?, 'entity_profile', "
            "'2026-01-01', '2026-03-01', ?, '[1,2,3]')",
            (f"entity_{i}", f"Profile for entity_{i}"),
        )
    conn.commit()
    conn.close()
    return db_path


def test_migration_log_fires_on_purge(tmp_path, monkeypatch, caplog):
    """The INFO log line must include the actual purged count."""
    from truememory.engine import TrueMemoryEngine

    monkeypatch.delenv("TRUEMEMORY_ENTITY_SHEETS", raising=False)
    db_path = _seed_legacy_db(tmp_path, n_rows=3)

    with caplog.at_level(logging.INFO):
        eng = TrueMemoryEngine(db_path).open(rebuild_vectors=False)
        eng.close()

    text = caplog.text
    assert "MEMORIST-L4 migration" in text, (
        f"Expected migration log, got: {text!r}"
    )
    assert "purged 3" in text, (
        f"Expected 'purged 3' in log; got: {text!r}"
    )


def test_migration_idempotent_across_opens(tmp_path, monkeypatch, caplog):
    """Second open must skip the DELETE and not re-emit the log."""
    from truememory.engine import TrueMemoryEngine

    monkeypatch.delenv("TRUEMEMORY_ENTITY_SHEETS", raising=False)
    db_path = _seed_legacy_db(tmp_path, n_rows=4)

    # First open: should purge.
    with caplog.at_level(logging.INFO):
        eng1 = TrueMemoryEngine(db_path).open(rebuild_vectors=False)
        eng1.close()
    first_text = caplog.text
    assert "purged 4" in first_text

    # Verify metadata flag set.
    conn = sqlite3.connect(str(db_path))
    flag = conn.execute(
        "SELECT value FROM metadata WHERE key = ?",
        ("l4_entity_profile_migration_done",),
    ).fetchone()
    conn.close()
    assert flag is not None and flag[0] == "1", (
        f"Migration flag should be set after first open; got {flag!r}"
    )

    # Second open: must NOT re-emit the purge log.
    caplog.clear()
    with caplog.at_level(logging.INFO):
        eng2 = TrueMemoryEngine(db_path).open(rebuild_vectors=False)
        eng2.close()
    assert "MEMORIST-L4 migration: purged" not in caplog.text, (
        f"Idempotent open should skip the migration log; got: {caplog.text!r}"
    )

    # Final state: no entity_profile rows.
    conn2 = sqlite3.connect(str(db_path))
    remaining = conn2.execute(
        "SELECT COUNT(*) FROM summaries WHERE period='entity_profile'"
    ).fetchone()[0]
    conn2.close()
    assert remaining == 0


@pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "YES", "on", "On"])
def test_env_var_accepts_variants(tmp_path, monkeypatch, value):
    """All accepted variants must enable the writer."""
    from truememory.engine import TrueMemoryEngine

    monkeypatch.setenv("TRUEMEMORY_ENTITY_SHEETS", value)
    tmp_json = _seed_messages(tmp_path)
    db_path = tmp_path / f"variant_{value.strip().lower()}.db"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        eng = TrueMemoryEngine(db_path)
        stats = eng.ingest(str(tmp_json))
        eng.close()

    assert "sheets in" in stats["entity_summary_sheets"], (
        f"Variant {value!r} should enable writer; got: {stats['entity_summary_sheets']!r}"
    )
    assert "DISABLED" not in stats["entity_summary_sheets"]


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "random", "  "])
def test_env_var_rejects_invalid(tmp_path, monkeypatch, value):
    """Non-truthy / unknown values must leave the writer disabled."""
    from truememory.engine import TrueMemoryEngine

    if value == "":
        monkeypatch.delenv("TRUEMEMORY_ENTITY_SHEETS", raising=False)
    else:
        monkeypatch.setenv("TRUEMEMORY_ENTITY_SHEETS", value)
    tmp_json = _seed_messages(tmp_path)
    db_path = tmp_path / f"reject_{abs(hash(value))}.db"

    eng = TrueMemoryEngine(db_path)
    stats = eng.ingest(str(tmp_json))
    eng.close()

    assert "DISABLED" in stats["entity_summary_sheets"], (
        f"Value {value!r} should leave writer DISABLED; got: {stats['entity_summary_sheets']!r}"
    )


def test_other_consolidation_stages_preserved(tmp_path, monkeypatch):
    """Sibling consolidation stages must still produce summaries rows
    after the L4 disable. Specifically: monthly summaries (build_summaries)
    and structured facts (build_structured_facts) are independent of the
    entity_profile writer."""
    from truememory.engine import TrueMemoryEngine

    monkeypatch.delenv("TRUEMEMORY_ENTITY_SHEETS", raising=False)

    # Build a corpus large enough to trigger monthly summarization
    # (>= 30 messages spanning >= 2 calendar months).
    messages = []
    base_senders = ["alice", "bob", "carol"]
    # Month 1: Jan 2026.
    for i in range(15):
        messages.append({
            "content": f"Jan message {i}: discussing project status with team.",
            "sender": base_senders[i % 3],
            "recipient": base_senders[(i + 1) % 3],
            "timestamp": f"2026-01-{(i % 28) + 1:02d}T10:00:00Z",
            "category": "work",
            "modality": "conversation",
        })
    # Month 2: Feb 2026.
    for i in range(15):
        messages.append({
            "content": f"Feb message {i}: follow-up notes on the same project.",
            "sender": base_senders[i % 3],
            "recipient": base_senders[(i + 1) % 3],
            "timestamp": f"2026-02-{(i % 28) + 1:02d}T10:00:00Z",
            "category": "work",
            "modality": "conversation",
        })
    tmp_json = tmp_path / "corpus.json"
    tmp_json.write_text(json.dumps(messages))

    db_path = tmp_path / "siblings.db"
    eng = TrueMemoryEngine(db_path)
    eng.ingest(str(tmp_json))
    eng.close()

    conn = sqlite3.connect(str(db_path))
    by_period = dict(conn.execute(
        "SELECT period, COUNT(*) FROM summaries GROUP BY period"
    ).fetchall())
    conn.close()

    # The whole point of the migration: zero entity_profile rows.
    assert by_period.get("entity_profile", 0) == 0, (
        f"Expected 0 entity_profile rows; got {by_period.get('entity_profile')}"
    )
    # Monthly summaries must still produce something.
    assert by_period.get("monthly", 0) > 0, (
        f"build_summaries should produce monthly rows; counts={by_period!r}"
    )
    # At least one other sibling stage produced rows (entity_monthly or
    # structured_fact). Both are corpus-dependent — assert union.
    sibling_total = (
        by_period.get("entity_monthly", 0)
        + by_period.get("structured_fact", 0)
    )
    assert sibling_total > 0, (
        f"At least one of entity_monthly/structured_fact should produce "
        f"rows on this corpus; counts={by_period!r}"
    )


def test_rowcount_not_minus_one(tmp_path, monkeypatch, caplog):
    """Regression for the cursor-scope bug: .rowcount must reflect the
    actual delete count, not -1 from a GC'd cursor."""
    from truememory.engine import TrueMemoryEngine

    monkeypatch.delenv("TRUEMEMORY_ENTITY_SHEETS", raising=False)
    db_path = _seed_legacy_db(tmp_path, n_rows=5)

    with caplog.at_level(logging.INFO):
        eng = TrueMemoryEngine(db_path).open(rebuild_vectors=False)
        eng.close()

    text = caplog.text
    assert "purged 5" in text, (
        f"Expected exact 'purged 5'; cursor-scope bug would show -1 or 0. "
        f"Got: {text!r}"
    )
    assert "purged -1" not in text
    assert "purged 0" not in text
