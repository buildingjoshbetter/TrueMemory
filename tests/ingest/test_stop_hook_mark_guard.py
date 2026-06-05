"""Regression lock for #400 — stop hook must not mark a session extracted
when no ingest process was actually spawned.

Pre-fix: stop.main() called mark_session_extracted unconditionally after
_run_background_ingestion. That function returns 0 when the spawn cap denies
it (after queueing to the backlog) or when Popen fails. Marking the session
extracted at its current size then made should_extract_session() skip it until
the transcript grew >1KB, silently dropping the session if the backlog drain
never completed.

Fix: only mark when spawned_pid > 0 (a real ingest process started). The same
guard is applied in compact.py and user_prompt_submit.py.
"""
from __future__ import annotations

import io
import json


def _make_transcript(tmp_path, n=6):
    p = tmp_path / "transcript.jsonl"
    p.write_text(
        "\n".join(
            json.dumps({"type": "user", "message": {"content": f"message number {i}"}})
            for i in range(n)
        ),
        encoding="utf-8",
    )
    return p


def _drive_stop(monkeypatch, tmp_path, transcript, return_pid):
    """Run stop.main() with _run_background_ingestion stubbed to return_pid.
    Returns the path where the extracted marker would be written."""
    from truememory.ingest.hooks import stop as stop_mod
    from truememory.ingest.hooks import _shared

    monkeypatch.delenv("TRUEMEMORY_EXTRACTION", raising=False)
    extracted_dir = tmp_path / "extracted"
    monkeypatch.setattr(_shared, "EXTRACTED_DIR", extracted_dir)
    monkeypatch.setattr(stop_mod, "TRACE_DIR", tmp_path / "traces")
    monkeypatch.setattr(stop_mod, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(stop_mod, "BACKLOG_DIR", tmp_path / "backlog")
    monkeypatch.setattr(
        stop_mod, "_run_background_ingestion", lambda *a, **k: return_pid
    )

    stdin_payload = json.dumps(
        {"transcript_path": str(transcript), "session_id": "sess-400"}
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin_payload))
    stop_mod.main()
    return _shared._safe_session_id("sess-400"), extracted_dir


def test_not_marked_when_spawn_denied(monkeypatch, tmp_path):
    """pid == 0 (queued to backlog / Popen failure): no extracted marker."""
    transcript = _make_transcript(tmp_path)
    safe_id, extracted_dir = _drive_stop(monkeypatch, tmp_path, transcript, return_pid=0)
    marker = extracted_dir / safe_id
    assert not marker.exists(), (
        "#400 regression: session marked extracted despite spawn denial (pid=0)"
    )


def test_marked_when_real_spawn(monkeypatch, tmp_path):
    """pid > 0 (real ingest process): marker written so concurrent dedup works."""
    transcript = _make_transcript(tmp_path)
    safe_id, extracted_dir = _drive_stop(
        monkeypatch, tmp_path, transcript, return_pid=999_999
    )
    marker = extracted_dir / safe_id
    assert marker.exists(), "session not marked extracted after a real spawn (pid>0)"
    data = json.loads(marker.read_text(encoding="utf-8"))
    assert data["pid"] == 999_999
