"""Regression locks for issue #644 — backlog drain + maintenance hardening.

Covers the concurrency / lifecycle findings:

  M-14  A corrupt JSON backlog marker is quarantined to ``.corrupt`` (not
        recycled to ``.json``), so a few poison pills cannot permanently
        consume every ``_DRAIN_CAP`` slot and starve later sessions.
  M-15  The stale-session scanner does NOT re-queue a session that already has
        an in-flight ``.processing`` claim (which would race two workers onto
        one transcript). The Stop hook's ``_queue_to_backlog`` also refuses to
        clobber a live ``.processing`` claim.
  M-34  Drain/cascade write an optimistic EXTRACTED_DIR marker (tagged with the
        worker PID) right after spawn, so ``should_extract_session`` sees the
        in-flight worker and the Stop hook does not spawn a parallel ingest.
  M-37  The scan watermark is not advanced past the cap window, so a deferred
        stale session is re-checked on the next scan.
  M-71  The extraction-budget slot is refunded when the spawn gate denies.

All tests use tmp dirs and fake PIDs; no model loads.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import pytest


@contextmanager
def _gate_allows():
    yield True


@contextmanager
def _gate_denies():
    yield False


def _write_marker(backlog: Path, session_id: str, transcript: Path) -> Path:
    backlog.mkdir(parents=True, exist_ok=True)
    marker = backlog / f"{session_id}.json"
    marker.write_text(
        json.dumps(
            {
                "transcript_path": str(transcript),
                "session_id": session_id,
                "user_id": "",
                "db_path": "",
            }
        ),
        encoding="utf-8",
    )
    return marker


# ---------------------------------------------------------------------------
# M-14: corrupt marker quarantine
# ---------------------------------------------------------------------------


def test_corrupt_marker_quarantined_not_recycled(monkeypatch, tmp_path):
    """A marker with invalid JSON is renamed to ``.corrupt`` and NOT restored
    to ``.json`` by the drainer. Pre-fix the generic ``except`` recycled it.
    """
    from truememory.ingest.hooks import session_start as ss
    from truememory.ingest.hooks import _shared as shared_mod
    from truememory.hooks import core as core_mod

    backlog = tmp_path / "backlog"
    backlog.mkdir(parents=True)
    bad = backlog / "sess-corrupt.json"
    bad.write_text("{ this is not valid json", encoding="utf-8")

    monkeypatch.setattr(ss, "BACKLOG_DIR", backlog)
    monkeypatch.setattr(shared_mod, "check_extraction_budget", lambda: True)
    monkeypatch.setattr(core_mod, "spawn_gate", _gate_allows)
    monkeypatch.setattr(core_mod, "register_spawned_pid", lambda pid: None)

    ss._drain_backlog()

    assert not (backlog / "sess-corrupt.json").exists(), "corrupt marker recycled to .json"
    assert not (backlog / "sess-corrupt.processing").exists(), "corrupt marker left as claim"
    assert (backlog / "sess-corrupt.corrupt").exists(), "corrupt marker not quarantined"


def test_corrupt_markers_do_not_starve_drain_slots(monkeypatch, tmp_path):
    """Poison-pill scenario: a full ``_DRAIN_CAP`` worth of corrupt markers must
    be quarantined (freeing the slots) rather than recycled forever. After one
    drain pass a healthy session queued later must be drainable.
    """
    from truememory.ingest.hooks import session_start as ss
    from truememory.ingest.hooks import _shared as shared_mod
    from truememory.hooks import core as core_mod

    backlog = tmp_path / "backlog"
    backlog.mkdir(parents=True)
    # Fill every drain slot with corrupt markers (names sort before "z-good").
    for i in range(ss._DRAIN_CAP):
        (backlog / f"sess-bad-{i}.json").write_text("{bad", encoding="utf-8")

    monkeypatch.setattr(ss, "BACKLOG_DIR", backlog)
    monkeypatch.setattr(shared_mod, "check_extraction_budget", lambda: True)
    monkeypatch.setattr(core_mod, "spawn_gate", _gate_allows)
    monkeypatch.setattr(core_mod, "register_spawned_pid", lambda pid: None)
    monkeypatch.setattr(shared_mod, "EXTRACTED_DIR", tmp_path / "extracted")

    ss._drain_backlog()

    # All corrupt markers quarantined, none left as .json poison pills.
    assert sorted(p.name for p in backlog.glob("*.json")) == []
    assert len(list(backlog.glob("*.corrupt"))) == ss._DRAIN_CAP


# ---------------------------------------------------------------------------
# M-15: scanner skips sessions with an existing claim
# ---------------------------------------------------------------------------


def test_stale_scan_skips_existing_processing_claim(monkeypatch, tmp_path):
    """A session whose transcript still has a live ``.processing`` claim must
    NOT be re-queued by the stale scanner (would race two workers).
    """
    from truememory.ingest.hooks import session_start as ss
    from truememory.ingest.hooks import _shared as shared_mod

    # Fake ~/.claude/projects with one large recent transcript.
    home = tmp_path / "home"
    proj = home / ".claude" / "projects" / "proj-a"
    proj.mkdir(parents=True)
    sid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    transcript = proj / f"{sid}.jsonl"
    transcript.write_text("x" * 6000, encoding="utf-8")

    backlog = tmp_path / "backlog"
    backlog.mkdir(parents=True)
    extracted = tmp_path / "extracted"
    extracted.mkdir(parents=True)
    scan_marker = tmp_path / "scan_marker"

    # Pre-existing in-flight claim for this exact session.
    claim = backlog / f"{shared_mod._safe_session_id(sid)}.processing"
    claim.write_text(json.dumps({"claimed_pid": os.getpid()}), encoding="utf-8")

    monkeypatch.setattr(ss, "BACKLOG_DIR", backlog)
    monkeypatch.setattr(shared_mod, "BACKLOG_DIR", backlog)
    monkeypatch.setattr(shared_mod, "EXTRACTED_DIR", extracted)
    monkeypatch.setattr(ss, "_SCAN_MARKER", scan_marker)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    # Force a scan (no prior watermark / interval gate).
    monkeypatch.setattr(ss, "_SCAN_INTERVAL", 0)

    from truememory.ingest.hooks import stop as stop_mod
    queued: list = []
    monkeypatch.setattr(stop_mod, "_queue_to_backlog", lambda *a, **k: queued.append(a))

    ss._scan_stale_sessions()

    assert queued == [], "scanner re-queued a session with a live .processing claim"


def test_queue_to_backlog_refuses_to_clobber_live_claim(monkeypatch, tmp_path):
    """``_queue_to_backlog`` must not overwrite an in-flight ``.processing``
    claim with a fresh ``.json`` (POSIX rename-overwrite race / M-15).
    """
    from truememory.ingest.hooks import stop as stop_mod

    backlog = tmp_path / "backlog"
    backlog.mkdir(parents=True)
    sid = "sess-live"
    claim = backlog / f"{sid}.processing"
    claim.write_text(json.dumps({"claimed_pid": os.getpid()}), encoding="utf-8")

    monkeypatch.setattr(stop_mod, "BACKLOG_DIR", backlog)

    transcript = tmp_path / "t.jsonl"
    transcript.write_text("x" * 100, encoding="utf-8")

    stop_mod._queue_to_backlog(str(transcript), sid, "", "", reason="test")

    assert not (backlog / f"{sid}.json").exists(), "clobbered a live .processing claim"
    assert claim.exists()


# ---------------------------------------------------------------------------
# M-14 / atomicity: markers are written via tmp + os.replace (no torn read)
# ---------------------------------------------------------------------------


def test_atomic_write_uses_replace_no_partial_json(monkeypatch, tmp_path):
    """The atomic writer must never leave a partially-written ``.json`` visible.

    We simulate a crash mid-write by making the tmp file's write raise after
    creating the tmp but before replace; the destination must not appear.
    """
    from truememory.ingest.hooks import _shared as shared_mod

    target = tmp_path / "marker.json"
    shared_mod._atomic_write_text(target, json.dumps({"a": 1}))
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    # No leftover tmp files.
    assert list(tmp_path.glob("*.tmp")) == []

    # Simulate failure during the tmp write: destination stays untouched.
    orig_replace = os.replace

    def _boom(src, dst):
        raise OSError("simulated crash before replace")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError):
        shared_mod._atomic_write_text(tmp_path / "marker2.json", "partial")
    assert not (tmp_path / "marker2.json").exists(), "partial write became visible"
    # tmp cleaned up on failure.
    monkeypatch.setattr(os, "replace", orig_replace)
    assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# M-71: budget slot refunded when spawn gate denies
# ---------------------------------------------------------------------------


def test_budget_refunded_when_gate_denies(monkeypatch, tmp_path):
    """When the spawn gate denies, the consumed extraction-budget slot must be
    refunded so a denied spawn does not permanently burn budget.
    """
    from truememory.ingest.hooks import session_start as ss
    from truememory.ingest.hooks import _shared as shared_mod
    from truememory.hooks import core as core_mod

    backlog = tmp_path / "backlog"
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("x" * 100, encoding="utf-8")
    _write_marker(backlog, "sess-refund", transcript)

    # Real budget file in tmp so check/refund operate on it.
    monkeypatch.setattr(shared_mod, "_BUDGET_FILE", tmp_path / ".extraction_budget")
    monkeypatch.setattr(shared_mod, "_MAX_EXTRACTIONS_PER_HOUR", 5)
    monkeypatch.setattr(ss, "BACKLOG_DIR", backlog)
    monkeypatch.setattr(core_mod, "spawn_gate", _gate_denies)
    monkeypatch.setattr(core_mod, "register_spawned_pid", lambda pid: None)

    # Budget starts empty (count 0).
    ss._drain_backlog()

    # The gate denied → the marker is recycled and the budget count is back to 0.
    raw = (tmp_path / ".extraction_budget").read_text(encoding="utf-8").strip()
    data = json.loads(raw) if raw else {"count": 0}
    assert data.get("count", 0) == 0, "budget slot not refunded on gate denial"
    # Marker recycled (not lost) for the next hour.
    assert (backlog / "sess-refund.json").exists()


def test_refund_extraction_budget_direct(monkeypatch, tmp_path):
    """Unit test the refund helper: consume then refund returns to baseline."""
    from truememory.ingest.hooks import _shared as shared_mod

    monkeypatch.setattr(shared_mod, "_BUDGET_FILE", tmp_path / ".budget")
    monkeypatch.setattr(shared_mod, "_MAX_EXTRACTIONS_PER_HOUR", 3)

    assert shared_mod.check_extraction_budget() is True
    after_consume = json.loads((tmp_path / ".budget").read_text())["count"]
    assert after_consume == 1

    shared_mod.refund_extraction_budget()
    after_refund = json.loads((tmp_path / ".budget").read_text())["count"]
    assert after_refund == 0

    # Refund never goes negative.
    shared_mod.refund_extraction_budget()
    assert json.loads((tmp_path / ".budget").read_text())["count"] == 0


# ---------------------------------------------------------------------------
# M-34: optimistic extracted marker after spawn
# ---------------------------------------------------------------------------


def test_drain_writes_optimistic_extracted_marker(monkeypatch, tmp_path):
    """After spawning, the drainer records an EXTRACTED_DIR marker tagged with
    the worker PID so should_extract_session() sees the in-flight worker.
    """
    from truememory.ingest.hooks import session_start as ss
    from truememory.ingest.hooks import _shared as shared_mod
    from truememory.hooks import core as core_mod

    backlog = tmp_path / "backlog"
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("x" * 100, encoding="utf-8")
    _write_marker(backlog, "sess-m34", transcript)

    extracted = tmp_path / "extracted"
    monkeypatch.setattr(ss, "BACKLOG_DIR", backlog)
    monkeypatch.setattr(shared_mod, "EXTRACTED_DIR", extracted)
    monkeypatch.setattr(shared_mod, "check_extraction_budget", lambda: True)
    monkeypatch.setattr(core_mod, "spawn_gate", _gate_allows)
    monkeypatch.setattr(core_mod, "register_spawned_pid", lambda pid: None)

    worker_pid = 4242

    class _Proc:
        pid = worker_pid

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _Proc())

    ss._drain_backlog()

    marker = extracted / shared_mod._safe_session_id("sess-m34")
    assert marker.exists(), "no optimistic extracted marker written after spawn"
    data = json.loads(marker.read_text(encoding="utf-8"))
    assert data.get("pid") == worker_pid, "extracted marker not tagged with worker PID"


# ---------------------------------------------------------------------------
# M-37: watermark is not advanced past the cap window
# ---------------------------------------------------------------------------


def test_watermark_held_at_cutoff_when_cap_hit(monkeypatch, tmp_path):
    """When the scan hits _SCAN_CAP, the watermark must NOT jump to ``now`` —
    deferred candidates beyond the cap would then fall outside the next scan's
    window. The watermark is held at the scan cutoff instead.
    """
    from truememory.ingest.hooks import session_start as ss
    from truememory.ingest.hooks import _shared as shared_mod
    from truememory.ingest.hooks import stop as stop_mod

    home = tmp_path / "home"
    proj = home / ".claude" / "projects" / "proj-a"
    proj.mkdir(parents=True)
    # Create more candidate transcripts than the cap.
    for i in range(ss._SCAN_CAP + 2):
        sid = f"{i:08d}-bbbb-cccc-dddd-eeeeeeeeeeee"
        (proj / f"{sid}.jsonl").write_text("x" * 6000, encoding="utf-8")

    backlog = tmp_path / "backlog"
    backlog.mkdir(parents=True)
    extracted = tmp_path / "extracted"
    extracted.mkdir(parents=True)
    scan_marker = tmp_path / "scan_marker"

    monkeypatch.setattr(ss, "BACKLOG_DIR", backlog)
    monkeypatch.setattr(shared_mod, "BACKLOG_DIR", backlog)
    monkeypatch.setattr(shared_mod, "EXTRACTED_DIR", extracted)
    monkeypatch.setattr(ss, "_SCAN_MARKER", scan_marker)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(ss, "_SCAN_INTERVAL", 0)
    monkeypatch.setattr(stop_mod, "_queue_to_backlog", lambda *a, **k: None)

    now = time.time()
    ss._scan_stale_sessions()

    written = float(scan_marker.read_text(encoding="utf-8").strip())
    # Cap was hit → watermark held at cutoff (first-run cutoff = now - 86400),
    # well below ``now``. Pre-fix it would have been advanced to ~now.
    assert written < now - 1000, "watermark advanced past cap window (M-37)"


def test_watermark_advances_to_now_when_window_drained(monkeypatch, tmp_path):
    """When the scan fully drains the window (cap not hit), the watermark
    advances to ``now`` so the next scan is O(new).
    """
    from truememory.ingest.hooks import session_start as ss
    from truememory.ingest.hooks import _shared as shared_mod
    from truememory.ingest.hooks import stop as stop_mod

    home = tmp_path / "home"
    proj = home / ".claude" / "projects" / "proj-a"
    proj.mkdir(parents=True)
    # One candidate only — well under the cap.
    sid = "11111111-bbbb-cccc-dddd-eeeeeeeeeeee"
    (proj / f"{sid}.jsonl").write_text("x" * 6000, encoding="utf-8")

    backlog = tmp_path / "backlog"
    backlog.mkdir(parents=True)
    extracted = tmp_path / "extracted"
    extracted.mkdir(parents=True)
    scan_marker = tmp_path / "scan_marker"

    monkeypatch.setattr(ss, "BACKLOG_DIR", backlog)
    monkeypatch.setattr(shared_mod, "BACKLOG_DIR", backlog)
    monkeypatch.setattr(shared_mod, "EXTRACTED_DIR", extracted)
    monkeypatch.setattr(ss, "_SCAN_MARKER", scan_marker)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(ss, "_SCAN_INTERVAL", 0)
    monkeypatch.setattr(stop_mod, "_queue_to_backlog", lambda *a, **k: None)

    now = time.time()
    ss._scan_stale_sessions()

    written = float(scan_marker.read_text(encoding="utf-8").strip())
    assert written >= now - 5, "watermark did not advance to now on a drained window"


# ---------------------------------------------------------------------------
# M-38: maintenance child guard honored
# ---------------------------------------------------------------------------


def test_maintenance_child_does_not_respawn(monkeypatch, tmp_path):
    """A process running as TRUEMEMORY_MAINTENANCE_CHILD must not spawn another
    maintenance subprocess (recursion guard / M-38).
    """
    from truememory.ingest.hooks import session_start as ss

    monkeypatch.setenv("TRUEMEMORY_MAINTENANCE_CHILD", "1")
    spawned = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: spawned.append(a))

    ss._run_maintenance_background()

    assert spawned == [], "maintenance child re-spawned maintenance (no guard)"
