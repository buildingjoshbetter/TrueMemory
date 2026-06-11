"""Regression locks: the M-90 transcript-path allowlist must guard the stop
hook and the SessionStart backlog drain, not just the compact hook (A1-2).

Pre-fix: #653/M-90 added ``_is_allowed_transcript`` to compact.py ONLY; stop.py
and the drain gated solely on ``Path.exists()``, so a crafted hook stdin /
backlog marker with ``transcript_path=/etc/passwd`` (or any readable file) was
parsed into the memory store via parse_transcript's plaintext fallback. These
tests fail pre-fix, pass post-fix.

FTS-only / no model loads.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from truememory.ingest.hooks._shared import is_allowed_transcript, _transcript_roots


def test_allowed_transcript_rejects_outside_root(tmp_path, monkeypatch):
    # Constrain the allowed root to a temp dir.
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setenv("TRUEMEMORY_TRANSCRIPT_DIR", str(root))

    # An arbitrary readable file outside the root is rejected.
    evil = tmp_path / "secret.txt"
    evil.write_text("user: root\npassword: hunter2\n" * 50)
    assert is_allowed_transcript(str(evil)) is False

    # A real /etc/passwd-style absolute path is rejected.
    assert is_allowed_transcript("/etc/passwd") is False

    # Empty / missing is rejected (not crash).
    assert is_allowed_transcript("") is False


def test_allowed_transcript_accepts_inside_root(tmp_path, monkeypatch):
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setenv("TRUEMEMORY_TRANSCRIPT_DIR", str(root))
    good = root / "session-abc.jsonl"
    good.write_text("{}")
    assert is_allowed_transcript(str(good)) is True


def test_allowed_transcript_rejects_symlink_escape(tmp_path, monkeypatch):
    """A symlink INSIDE the root pointing OUTSIDE must be rejected (resolve())."""
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setenv("TRUEMEMORY_TRANSCRIPT_DIR", str(root))
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    link = root / "innocent.jsonl"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        return  # platform without symlink support
    assert is_allowed_transcript(str(link)) is False


def test_allowed_transcript_rejects_dotdot_escape(tmp_path, monkeypatch):
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setenv("TRUEMEMORY_TRANSCRIPT_DIR", str(root))
    escape = root / ".." / "secret.txt"
    (tmp_path / "secret.txt").write_text("x")
    assert is_allowed_transcript(str(escape)) is False


def test_stop_hook_imports_and_uses_guard():
    """stop.py main() calls is_allowed_transcript (guard is wired in)."""
    import inspect
    from truememory.ingest.hooks import stop
    src = inspect.getsource(stop)
    assert "is_allowed_transcript" in src, "stop hook must gate on the transcript allowlist"


def test_drain_imports_and_uses_guard():
    """session_start drain calls is_allowed_transcript (guard is wired in)."""
    import inspect
    from truememory.ingest.hooks import session_start
    src = inspect.getsource(session_start)
    assert "is_allowed_transcript" in src, "drain must gate on the transcript allowlist"


def test_compact_shim_still_works(tmp_path, monkeypatch):
    """#653 backward-compat: compact._is_allowed_transcript still resolves (delegates)."""
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setenv("TRUEMEMORY_TRANSCRIPT_DIR", str(root))
    from truememory.ingest.hooks import compact
    assert compact._is_allowed_transcript("/etc/passwd") is False
    good = root / "s.jsonl"
    good.write_text("{}")
    assert compact._is_allowed_transcript(str(good)) is True
    assert _transcript_roots()  # shared roots non-empty
