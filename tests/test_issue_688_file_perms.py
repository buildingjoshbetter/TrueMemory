"""Regression lock: at-rest files/dirs holding memories or prompt text must be
owner-only (S1-2 / issue #688).

Pre-fix: the DB (all memories), recall cache, buffers, and several
~/.truememory mkdirs were created world-readable (0644 files / 0755 dirs), so
another local user could read stored memories on a shared host.

POSIX-only — skipped on Windows where chmod modes don't apply.
"""
import os
import sys
import stat

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes only")


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def test_db_file_is_owner_only(tmp_path):
    from truememory.storage import create_db
    db = tmp_path / "memories.db"
    conn = create_db(db)
    conn.execute("CREATE TABLE IF NOT EXISTS t (x)")
    conn.commit()
    conn.close()
    assert _mode(db) == 0o600, f"DB file mode {oct(_mode(db))} != 0o600"
    # WAL/SHM, if present, must also be owner-only.
    for suffix in ("-wal", "-shm"):
        p = tmp_path / f"memories.db{suffix}"
        if p.exists():
            assert _mode(p) == 0o600, f"{p.name} mode {oct(_mode(p))} != 0o600"


def test_atomic_write_text_is_owner_only(tmp_path):
    from truememory.ingest.hooks._shared import _atomic_write_text
    p = tmp_path / "marker.json"
    _atomic_write_text(p, '{"x": 1}')
    assert _mode(p) == 0o600, f"atomic-written file mode {oct(_mode(p))} != 0o600"
    # explicit mode honored
    p2 = tmp_path / "marker2.json"
    _atomic_write_text(p2, "data", mode=0o644)
    assert _mode(p2) == 0o644


def test_secure_mkdir_sets_0700_on_leaf_and_root(tmp_path, monkeypatch):
    import truememory.ingest.hooks._shared as shared
    # Point the "root" at a temp dir so we don't touch the real ~/.truememory.
    fake_root = tmp_path / ".truememory"
    monkeypatch.setattr(shared, "_TRUEMEMORY_ROOT", fake_root)
    leaf = fake_root / "extracted"
    shared._secure_mkdir(leaf)
    assert leaf.is_dir()
    assert _mode(leaf) == 0o700, f"leaf mode {oct(_mode(leaf))} != 0o700"
    assert _mode(fake_root) == 0o700, f"root mode {oct(_mode(fake_root))} != 0o700"


def test_secure_mkdir_repairs_loose_root(tmp_path, monkeypatch):
    """If ~/.truememory already exists world-traversable, it is tightened."""
    import truememory.ingest.hooks._shared as shared
    fake_root = tmp_path / ".truememory"
    fake_root.mkdir()
    os.chmod(fake_root, 0o755)  # the "hook-first ordering left it 0755" state
    monkeypatch.setattr(shared, "_TRUEMEMORY_ROOT", fake_root)
    shared._secure_mkdir(fake_root / "recall_markers")
    assert _mode(fake_root) == 0o700, "loose root not tightened to 0700"
