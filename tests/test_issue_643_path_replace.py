"""Issue #643 (M-49): atomic-write sites must use os.replace, not Path.rename.

On POSIX, ``Path.rename(dst)`` silently overwrites an existing destination, so
the bug was latent. On Windows, ``Path.rename`` raises ``FileExistsError`` when
the destination exists, which broke every atomic tmp-write-then-swap site
(update notices, config writes, telemetry, spawn-cap state, buffer rotation).
``os.replace`` / ``Path.replace`` overwrites atomically on *both* platforms.

These tests assert the replace-onto-existing-destination semantics the fix
relies on, and guard the converted source sites against regressing back to
``.rename``. Backlog-claim sites (``marker -> .processing``) intentionally keep
``.rename`` for their fail-if-exists exclusive-claim guarantee and are NOT
covered here.
"""
from __future__ import annotations

import re
from pathlib import Path


def test_path_replace_overwrites_existing_destination(tmp_path):
    """The primitive the fix depends on: replace tolerates an existing dst."""
    src = tmp_path / "src.tmp"
    dst = tmp_path / "dst"
    src.write_text("new", encoding="utf-8")
    dst.write_text("old", encoding="utf-8")

    # Path.replace must NOT raise when dst exists (Path.rename would on Windows).
    src.replace(dst)

    assert dst.read_text(encoding="utf-8") == "new"
    assert not src.exists()


def _source(rel: str) -> str:
    root = Path(__file__).resolve().parent.parent / "truememory"
    return (root / rel).read_text(encoding="utf-8")


def test_atomic_write_sites_use_replace_not_rename():
    """The converted atomic-write sites must use .replace (M-49 regression guard)."""
    checks = {
        "telemetry.py": r"_tmp\.replace\(_p\)",
        "mcp_server.py": r"_CONFIG_PATH\.replace\(backup\)",
        "ingest/hooks/session_start.py": r"tmp\.replace\(update_path\)",
        "ingest/hooks/user_prompt_submit.py": r"tmp\.replace\(config_path\)",
        "hooks/core.py": r"tmp\.replace\(_SPAWN_CAP_STATE_PATH\)",
    }
    for rel, pattern in checks.items():
        src = _source(rel)
        assert re.search(pattern, src), f"{rel}: expected atomic .replace ({pattern})"


def test_buffer_rotation_uses_replace():
    """Buffer rotation (timestamped dest) also uses replace for Windows safety."""
    for rel in ("hooks/core.py", "ingest/hooks/user_prompt_submit.py"):
        src = _source(rel)
        assert "buffer_file.replace(rotated)" in src, f"{rel}: buffer rotation must use .replace"


def test_claim_sites_still_use_rename():
    """Backlog-claim sites MUST keep .rename (fail-if-exists exclusive claim).

    Converting these to .replace would silently overwrite a live worker's
    .processing claim — reintroducing the M-15 duplicate-ingest race.
    """
    for rel in ("ingest/cli.py", "ingest/hooks/_shared.py"):
        src = _source(rel)
        assert ".replace(claimed_path)" not in src, f"{rel}: claim site must NOT use .replace"
        assert ".replace(marker_path)" not in src, f"{rel}: claim restore must NOT use .replace"
