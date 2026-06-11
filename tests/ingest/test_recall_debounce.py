"""Tests for the first-message recall debounce (issue #561).

SessionStart already searches TrueMemory and injects up to 25 memories. The
UserPromptSubmit hook's per-message auto-recall is therefore redundant on the
first prompt of a session. These tests cover the one-shot, time-windowed
debounce marker that lets UserPromptSubmit skip that redundant recall.

Covers:
  - mark_recall_injected / consume_recall_injected round-trip and semantics
  - _try_auto_recall short-circuits (no recall work) when the marker is fresh
"""
from __future__ import annotations

import importlib
import io
import json
import os
import time

import pytest

from truememory.ingest.hooks import _shared
from truememory.ingest.hooks import session_start as ss
from truememory.ingest.hooks import user_prompt_submit as ups


@pytest.fixture
def marker_dir(monkeypatch, tmp_path):
    """Point the recall-marker store at an isolated temp dir."""
    d = tmp_path / "recall_markers"
    monkeypatch.setattr(_shared, "RECALL_MARKER_DIR", d)
    return d


class TestRecallMarker:
    def test_mark_then_consume_is_true(self, marker_dir):
        _shared.mark_recall_injected("session-abc")
        assert _shared.consume_recall_injected("session-abc") is True

    def test_consume_is_one_shot(self, marker_dir):
        """Only the first prompt is debounced; the marker is consumed."""
        _shared.mark_recall_injected("session-abc")
        assert _shared.consume_recall_injected("session-abc") is True
        assert _shared.consume_recall_injected("session-abc") is False

    def test_consume_without_marker_is_false(self, marker_dir):
        assert _shared.consume_recall_injected("never-marked") is False

    def test_stale_marker_is_not_fresh_and_is_cleaned(self, marker_dir):
        _shared.mark_recall_injected("session-old")
        marker = marker_dir / "session-old"
        marker.write_text(str(time.time() - 10_000), encoding="utf-8")
        assert _shared.consume_recall_injected("session-old", within_seconds=60) is False
        # Stale markers are still removed so the dir self-cleans.
        assert not marker.exists()

    def test_within_seconds_zero_disables_debounce(self, marker_dir):
        _shared.mark_recall_injected("session-abc")
        assert _shared.consume_recall_injected("session-abc", within_seconds=0) is False

    def test_empty_session_id_is_safe(self, marker_dir):
        assert _shared.consume_recall_injected("") is False
        # Must not raise even when there is nothing to mark.
        _shared.mark_recall_injected("")

    def test_corrupt_marker_is_false(self, marker_dir):
        marker_dir.mkdir(parents=True, exist_ok=True)
        (marker_dir / "session-bad").write_text("not-a-timestamp", encoding="utf-8")
        assert _shared.consume_recall_injected("session-bad") is False


class TestAutoRecallGate:
    def test_fresh_marker_short_circuits_recall(self, marker_dir, monkeypatch):
        """A consumed debounce marker makes _try_auto_recall return None before
        doing any recall work (no detection, no Memory load).

        Issue #636 (M-41): the marker is now consumed once in main() and the
        result passed in as ``debounced`` (so the proactive and auto-recall
        paths can't double-consume it). The short-circuit semantics are
        unchanged — only who reads the marker moved upstream.
        """
        def _boom(*_a, **_k):
            raise AssertionError("recall work must not run when marker is fresh")

        monkeypatch.setattr(ups, "_detect_recall", _boom)

        result = ups._try_auto_recall(
            "what's my favorite editor", "", "", session_id="session-first",
            debounced=True,
        )
        assert result is None

    def test_no_marker_allows_recall_detection(self, marker_dir, monkeypatch):
        """Without a marker, the gate falls through to normal recall detection."""
        called = {}

        def _detect(prompt):
            called["prompt"] = prompt
            return False  # short-circuit before Memory load

        monkeypatch.setattr(ups, "_detect_recall", _detect)
        result = ups._try_auto_recall(
            "what's my favorite editor", "", "", session_id="session-fresh"
        )
        assert result is None
        assert called["prompt"] == "what's my favorite editor"


def _run_session_start(monkeypatch, session_id: str, recall_context: str,
                       update_notice: str = "") -> str:
    """Run session_start.main() hermetically and return its stdout."""
    monkeypatch.setattr(ss, "_drain_backlog", lambda: None)
    monkeypatch.setattr(ss, "_scan_stale_sessions", lambda: None)
    monkeypatch.setattr(ss, "_is_first_run", lambda: False)
    monkeypatch.setattr(ss, "recall_memories", lambda *a, **k: recall_context)
    monkeypatch.setattr(ss, "_check_for_update", lambda: update_notice)
    monkeypatch.setattr(ss, "_check_email_needed", lambda: "")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"session_id": session_id})))
    captured = io.StringIO()
    monkeypatch.setattr("sys.stdout", captured)
    ss.main()
    return captured.getvalue().strip()


class TestSessionStartMarker:
    """The session_start.py half of the debounce (issue #561)."""

    def test_issue_561_marker_not_written_when_recall_empty(self, marker_dir, monkeypatch):
        """An empty/failed recall injects nothing, so the marker must NOT be
        written — otherwise the first prompt's targeted recall is suppressed
        too and the session's first turn gets zero recall, silently."""
        out = _run_session_start(monkeypatch, "sess-empty", "")
        assert out == ""
        assert not (marker_dir / "sess-empty").exists()

    def test_issue_561_marker_written_when_context_nonempty(self, marker_dir, monkeypatch):
        """Happy path: recall produced context, it was emitted, marker written."""
        out = _run_session_start(monkeypatch, "sess-full", "<truememory-recall>x</truememory-recall>")
        data = json.loads(out)
        assert "additionalContext" in data
        assert (marker_dir / "sess-full").exists()
        assert _shared.consume_recall_injected("sess-full") is True

    def test_issue_561_notices_alone_do_not_mark(self, marker_dir, monkeypatch):
        """Update/email notices can make the emitted context truthy even when
        recall returned nothing; that must not debounce the first prompt."""
        out = _run_session_start(monkeypatch, "sess-notice", "", update_notice="update available")
        data = json.loads(out)
        assert "update available" in data["additionalContext"]
        assert not (marker_dir / "sess-notice").exists()


class TestRecallMarkerEdges:
    def test_issue_561_per_session_isolation(self, marker_dir):
        """Marking one session must never debounce another."""
        _shared.mark_recall_injected("session-a")
        assert _shared.consume_recall_injected("session-b") is False
        assert _shared.consume_recall_injected("session-a") is True

    def test_issue_561_stale_markers_pruned_on_write(self, marker_dir):
        """Sessions that never send a prompt never consume their marker; the
        next write opportunistically sweeps anything well past the window."""
        _shared.mark_recall_injected("session-abandoned")
        stale = marker_dir / "session-abandoned"
        two_hours_ago = time.time() - 7200
        os.utime(stale, (two_hours_ago, two_hours_ago))
        _shared.mark_recall_injected("session-new")
        assert not stale.exists()
        assert (marker_dir / "session-new").exists()

    def test_issue_561_fresh_markers_survive_the_sweep(self, marker_dir):
        _shared.mark_recall_injected("session-a")
        _shared.mark_recall_injected("session-b")
        assert (marker_dir / "session-a").exists()
        assert (marker_dir / "session-b").exists()

    def test_issue_561_short_first_prompt_does_not_strand_marker(self, marker_dir, monkeypatch):
        """A first prompt under the min-length gate must still consume the
        marker, so the second (real) prompt is not debounced."""
        _shared.mark_recall_injected("sess-short")
        monkeypatch.delenv("TRUEMEMORY_EXTRACTION", raising=False)
        monkeypatch.setattr(
            "sys.stdin", io.StringIO(json.dumps({"prompt": "hi", "session_id": "sess-short"}))
        )
        ups.main()
        assert not (marker_dir / "sess-short").exists()
        assert _shared.consume_recall_injected("sess-short") is False

    def test_issue_561_invalid_env_var_falls_back_to_default(self, tmp_path):
        """TRUEMEMORY_RECALL_DEBOUNCE_SECONDS=abc must not crash the hooks.

        The knob is parsed at import of _shared, which both SessionStart and
        UserPromptSubmit import — an unguarded float() kills every hook."""
        old = os.environ.get("TRUEMEMORY_RECALL_DEBOUNCE_SECONDS")
        os.environ["TRUEMEMORY_RECALL_DEBOUNCE_SECONDS"] = "abc"
        try:
            importlib.reload(_shared)  # must not raise
            assert _shared._RECALL_DEBOUNCE_SECONDS == 60.0
            # And the marker round-trip still works on the reloaded module.
            _shared.RECALL_MARKER_DIR = tmp_path / "recall_markers"
            _shared.mark_recall_injected("sess-env")
            assert _shared.consume_recall_injected("sess-env") is True
        finally:
            if old is None:
                os.environ.pop("TRUEMEMORY_RECALL_DEBOUNCE_SECONDS", None)
            else:
                os.environ["TRUEMEMORY_RECALL_DEBOUNCE_SECONDS"] = old
            importlib.reload(_shared)
