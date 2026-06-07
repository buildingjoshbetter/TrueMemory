"""Regression lock for issue #422 — the MCP drainer copy must not unlink the
``.processing`` claim on spawn, and must spawn the ingest CLI with the raw
session_id via ``--session``.

There are two near-identical backlog drainers in the codebase: the Stop/
SessionStart hook copy (``truememory.ingest.hooks.session_start._drain_backlog``,
already locked by ``tests/ingest/test_backlog_drain_no_unlink_422.py``) and the
MCP server copy (``truememory.mcp_server._drain_batch_from_backlog``). The MCP
copy runs in the long-lived server process and is the one that drains the
backlog while a session is active. It must satisfy the same #422 contract:

  1. After spawning a worker, the ``.processing`` claim is LEFT in place — the
     spawned ingest CLI removes it on confirmed success, and a crashed worker's
     surviving claim is what ``cleanup_stale_processing`` recovers. Unlinking on
     spawn silently drops sessions whose worker exits non-zero.
  2. The spawned command carries the *raw* (unsanitized) session_id under
     ``--session`` so the CLI's success path can call
     ``clear_backlog_processing(args.session)`` against the right claim.
"""
from __future__ import annotations

import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _gate_allows():
    yield True


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


def _patch_drain_deps(monkeypatch, captured_cmd):
    """Patch the helpers the MCP drainer imports inside the function body.

    ``_drain_batch_from_backlog`` does ``import subprocess as _subprocess`` and
    imports ``spawn_gate`` / ``register_spawned_pid`` from
    ``truememory.hooks.core`` and the budget / stale helpers from
    ``truememory.ingest.hooks._shared`` at call time, so patching the source
    module attributes is sufficient.
    """
    from truememory.hooks import core as core_mod
    from truememory.ingest.hooks import _shared as shared_mod

    monkeypatch.setattr(shared_mod, "check_extraction_budget", lambda: True)
    monkeypatch.setattr(core_mod, "spawn_gate", _gate_allows)
    monkeypatch.setattr(core_mod, "register_spawned_pid", lambda pid: None)
    monkeypatch.setattr(
        shared_mod, "record_stale_processing_pid", lambda path, pid: None
    )
    # No-op stale cleanup so the test controls the claim's fate.
    monkeypatch.setattr(shared_mod, "cleanup_stale_processing", lambda d: None)

    class _DummyProc:
        pid = os.getpid()  # alive PID so any stale logic would leave the claim

    def _fake_popen(cmd, *a, **k):
        captured_cmd.append(list(cmd))
        return _DummyProc()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)


def test_mcp_drain_leaves_processing_claim_and_passes_raw_session(
    monkeypatch, tmp_path
):
    """The MCP drainer must keep the .processing claim after spawn and spawn
    the CLI with the raw session_id under --session.

    Pre-fix (unlink-on-spawn) the claim assertion fails; a regression that
    sanitizes or drops the session_id before --session fails the command
    assertion.
    """
    import truememory.mcp_server as ms

    raw_session_id = "proj/foo:bar baz-1"  # would change under sanitization

    backlog = tmp_path / "backlog"
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("x" * 100, encoding="utf-8")
    marker = _write_marker(backlog, "sess-mcp-422", transcript)
    # Overwrite payload to carry a raw session_id distinct from the filename.
    marker.write_text(
        json.dumps(
            {
                "transcript_path": str(transcript),
                "session_id": raw_session_id,
                "user_id": "",
                "db_path": "",
            }
        ),
        encoding="utf-8",
    )

    captured_cmd: list[list[str]] = []
    _patch_drain_deps(monkeypatch, captured_cmd)

    ms._drain_batch_from_backlog([marker])

    claim = backlog / "sess-mcp-422.processing"
    json_marker = backlog / "sess-mcp-422.json"

    # (a) The claim must survive the spawn (worker hasn't confirmed success).
    assert claim.exists(), (
        "MCP drainer unlinked .processing on spawn (issue #422)"
    )
    assert not json_marker.exists()

    # (b) The CLI was spawned with the RAW session_id under --session.
    assert len(captured_cmd) == 1, "expected exactly one ingest spawn"
    cmd = captured_cmd[0]
    assert "--session" in cmd, "drainer must pass --session to the ingest CLI"
    assert cmd[cmd.index("--session") + 1] == raw_session_id, (
        "--session must carry the raw, unsanitized session_id"
    )
    # Sanity: it's invoking the ingest CLI ingest subcommand.
    assert "truememory.ingest.cli" in cmd
    assert "ingest" in cmd
