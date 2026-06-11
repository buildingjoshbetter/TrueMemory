"""Tests for issue #649 — dedup × gate coherence + lock/txn hygiene.

Three findings:

* M-13: the encoding gate's contradiction vocabulary and dedup's update
  vocabulary diverged, so a correction ("Correction: X", "that's
  incorrect", ...) passed the gate but was then SKIPped by dedup as a
  near-duplicate before LLM arbitration. Fixed by a SHARED marker module
  (``truememory.ingest.markers``) used by both, plus threading
  ``fact.category`` into ``check_duplicate``.

* M-31: the ingest dedup-store lock TTL-stole the lock even when the
  holder PID was alive, splitting the lock across inodes → two holders.
  Fixed: TTL only applies when the holder PID is dead; a live holder is
  never stale.

* M-32: ``detect_contradictions`` / ``build_structured_facts`` committed
  the CALLER's open transaction before their own ``BEGIN IMMEDIATE``,
  persisting writes the caller might roll back. Fixed: refuse to run with a
  caller transaction open (raise) and re-read inside the write txn.

FTS-only / in-memory, no model loads.
"""

import os
import sqlite3
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import truememory.ingest.markers as markers
import truememory.ingest.dedup as dedup
import truememory.ingest.encoding_gate as gate_mod
import truememory.ingest.pipeline as pipeline
import truememory.consolidation as consolidation
from truememory.ingest.dedup import DedupAction, check_duplicate


# ── M-13: shared markers + correction routing ────────────────────────────

def _make_memory(results: list[dict]) -> MagicMock:
    mem = MagicMock()
    mem.search_vectors.return_value = results
    return mem


def test_gate_and_dedup_share_one_marker_source():
    """The gate and dedup must consume the SAME marker vocabulary (#649)."""
    # Gate's contradiction markers ARE the shared module's UPDATE_MARKERS.
    assert gate_mod._CONTRADICTION_MARKERS is markers.UPDATE_MARKERS
    # Both stages use the same predicate function object.
    assert gate_mod._has_update_markers is markers.has_update_markers
    assert dedup._has_update_markers is markers.has_update_markers
    # Dedup's regex patterns are the shared compiled list.
    assert dedup._UPDATE_MARKER_PATTERNS is markers.UPDATE_MARKER_PATTERNS


@pytest.mark.parametrize("correction", [
    "Correction: I use bun now, not npm",
    "That's incorrect — I live in Austin",
    "actually I switched to Postgres",
    "no longer at the old company",
])
def test_correction_recognized_by_both_gate_and_dedup(correction):
    """A correction must be a contradiction to the gate AND an update to dedup."""
    assert gate_mod._is_contradiction(correction) is True
    assert markers.has_update_markers(correction) is True


def test_high_similarity_correction_not_silently_skipped_by_category():
    """A category='correction' fact at >0.92 cosine reaches arbitration, not SKIP.

    No LLM config → heuristic path → must route to UPDATE (supersede),
    never SKIP.
    """
    memory = _make_memory([
        {
            "id": 7,
            "content": "User uses npm as their package manager",
            "score": 0.97,            # near-exact: old code would SKIP
            "score_space": "cosine",
            "directive": False,
        },
    ])
    decision = check_duplicate(
        "User uses npm as their package manager",  # no marker words at all
        memory,
        category="correction",                     # category alone must route it
    )
    assert decision.action != DedupAction.SKIP, (
        f"correction was silently SKIPped: {decision.reason}"
    )
    assert decision.action == DedupAction.UPDATE


def test_high_similarity_correction_not_skipped_by_markers():
    """A marker-bearing correction at >0.92 cosine is not SKIPped (#576/#649)."""
    memory = _make_memory([
        {
            "id": 3,
            "content": "User lives in Seattle",
            "score": 0.95,
            "score_space": "cosine",
            "directive": False,
        },
    ])
    decision = check_duplicate(
        "Correction: user moved to Austin",
        memory,
    )
    assert decision.action != DedupAction.SKIP
    assert decision.action == DedupAction.UPDATE


def test_plain_duplicate_still_skipped():
    """A genuine non-correction near-duplicate is still SKIPped (no regression)."""
    memory = _make_memory([
        {
            "id": 9,
            "content": "User likes coffee",
            "score": 0.98,
            "score_space": "cosine",
            "directive": False,
        },
    ])
    decision = check_duplicate("User likes coffee", memory, category="preference")
    assert decision.action == DedupAction.SKIP


# ── M-31: lock not stolen from a live holder ──────────────────────────────

def _write_lock(path, pid: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{pid}\n", encoding="utf-8")


def test_live_holder_lock_is_not_stale(tmp_path, monkeypatch):
    """A lock held by a LIVE pid must not be considered stale, even if old."""
    lock = tmp_path / "ingest.lock"
    _write_lock(lock, os.getpid())  # our own pid → definitely alive
    # Force the file's mtime far past the TTL.
    old = 1.0  # epoch+1s, ~decades old
    os.utime(lock, (old, old))
    assert pipeline._is_lock_stale(lock) is False


def test_dead_pid_lock_is_reclaimed(tmp_path, monkeypatch):
    """A stale lock from a DEAD pid IS reclaimable."""
    lock = tmp_path / "ingest.lock"
    dead_pid = 2_000_000_000  # implausibly high → not alive
    _write_lock(lock, dead_pid)
    monkeypatch.setattr(pipeline, "_pid_is_alive", lambda pid: pid != dead_pid)
    assert pipeline._is_lock_stale(lock) is True


def test_empty_lock_falls_back_to_ttl(tmp_path):
    """A corrupt/empty lock (no attributable holder) is stale past the TTL."""
    lock = tmp_path / "ingest.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("", encoding="utf-8")
    os.utime(lock, (1.0, 1.0))  # ancient
    assert pipeline._is_lock_stale(lock) is True


@pytest.mark.skipif(not pipeline._HAS_FCNTL, reason="requires fcntl")
def test_live_flock_blocks_second_acquirer(tmp_path, monkeypatch):
    """While one holder flocks the lock, a second acquirer must not steal it.

    We hold the lock via the real context manager (which flocks the inode)
    and assert a second non-blocking acquire of the same path cannot take an
    exclusive lock — i.e. mutual exclusion holds and no inode split occurs.
    """
    import fcntl

    lock = tmp_path / "ingest.lock"
    monkeypatch.setattr(pipeline, "_LOCK_PATH", lock)

    with pipeline._dedup_store_lock():
        # Holder is live and flocking. A second process-like attempt to grab
        # an exclusive flock on the SAME path must fail immediately.
        fd = os.open(str(lock), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            with pytest.raises(OSError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)


# ── M-32: detect_contradictions must not commit the caller's txn ─────────

def _make_timeline_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY,
            content TEXT,
            sender TEXT,
            recipient TEXT,
            timestamp TEXT,
            category TEXT,
            modality TEXT,
            directive INTEGER DEFAULT 0
        );
        CREATE TABLE fact_timeline (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT,
            fact TEXT,
            source_message_id INTEGER,
            timestamp TEXT,
            entity_scope TEXT,
            valid_from TEXT,
            valid_to TEXT,
            superseded_by INTEGER,
            status TEXT DEFAULT 'active'
        );
        CREATE TABLE canary (id INTEGER PRIMARY KEY, note TEXT);
        """
    )
    return conn


def test_detect_contradictions_does_not_commit_callers_txn():
    """The caller's open, uncommitted write must remain rollback-able (#649 M-32).

    The old code did ``conn.commit()`` before its own ``BEGIN IMMEDIATE``,
    which silently persisted the caller's in-flight rows so a later
    ``rollback()`` could not undo them. With the SAVEPOINT-based write the
    caller's transaction is never committed on their behalf.
    """
    conn = _make_timeline_db()
    conn.execute("BEGIN")
    conn.execute("INSERT INTO canary (note) VALUES ('uncommitted')")
    assert conn.in_transaction

    # Runs fine even with the caller's txn open (nests via SAVEPOINT).
    result = consolidation.detect_contradictions(conn)
    assert isinstance(result, list)

    # The caller's write must still be rollback-able — proof we did NOT
    # commit it.
    conn.rollback()
    rows = conn.execute("SELECT COUNT(*) FROM canary").fetchone()[0]
    assert rows == 0, "caller's write was committed despite rollback — txn leaked"
    # And our fact_timeline rows were rolled back with the caller's txn too.
    assert conn.execute("SELECT COUNT(*) FROM fact_timeline").fetchone()[0] == 0


def test_detect_contradictions_runs_without_caller_txn():
    """With no explicit caller txn, detect_contradictions runs and commits."""
    conn = _make_timeline_db()
    conn.execute(
        "INSERT INTO messages (id, content, sender, timestamp, directive) "
        "VALUES (1, 'price is now $50 instead of $30', 'u', '2024-01-01', 0)"
    )
    conn.commit()
    result = consolidation.detect_contradictions(conn)
    assert isinstance(result, list)


def test_detect_contradictions_preserves_isolation_level():
    """isolation_level must not be mutated/leaked by the write (#649 M-32)."""
    conn = _make_timeline_db()
    conn.commit()
    before = conn.isolation_level
    consolidation.detect_contradictions(conn)
    assert conn.isolation_level == before


def test_build_structured_facts_does_not_commit_callers_txn():
    """build_structured_facts has the same SAVEPOINT hygiene (#649 M-32)."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY, content TEXT, sender TEXT, recipient TEXT,
            timestamp TEXT, category TEXT, modality TEXT, directive INTEGER DEFAULT 0
        );
        CREATE TABLE summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT, period TEXT, start_date TEXT,
            end_date TEXT, entity TEXT, summary TEXT, key_facts TEXT,
            message_ids TEXT, created_at TEXT
        );
        CREATE TABLE canary (id INTEGER PRIMARY KEY, note TEXT);
        """
    )
    conn.execute("BEGIN")
    conn.execute("INSERT INTO canary (note) VALUES ('uncommitted')")
    consolidation.build_structured_facts(conn)
    conn.rollback()
    assert conn.execute("SELECT COUNT(*) FROM canary").fetchone()[0] == 0
