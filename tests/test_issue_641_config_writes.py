"""Regression tests for issue #641 — atomic + cross-process-locked config writes.

M-26 (P1): two config writers truncated config.json in place via bare
    ``write_text`` — ``telemetry._save_user_id`` (runs at EVERY server start when
    user_id is missing) and ``ingest/cli._save_truememory_config``. A concurrent
    ``_load_config`` reading the truncated window renamed config.json to
    ``.corrupt.<ts>`` and the writer finished into the orphaned inode -> tier +
    API keys vanished. Fix: route BOTH through the shared atomic ``_save_config``
    (mkstemp + os.replace).

M-58 (P2): ``_save_config``'s lock was in-process only;
    ``tier_switch.manager._apply_config_switch`` did read-modify-replace with NO
    cross-process lock -> lost updates. Fix: hold a cross-process fcntl/msvcrt
    file lock (``_config_file_lock``) around the read-modify-write in both paths.

All writes go to a tmp dir; the real ~/.truememory is never touched.
"""
from __future__ import annotations

import json
import threading

import pytest


@pytest.fixture
def tm_dir(monkeypatch, tmp_path):
    """Isolate every module's config + lock paths at a fresh tmp .truememory dir.

    All four writers compute their config/lock paths from module-level constants
    at import time. We repoint them at one shared tmp dir so the cross-process
    file lock (which keys off ``mcp_server._CONFIG_LOCK_PATH``) actually guards
    the same config.json that ``tier_switch`` writes.
    """
    d = tmp_path / ".truememory"
    d.mkdir(parents=True, exist_ok=True)
    cfg = d / "config.json"

    import truememory.mcp_server as ms

    monkeypatch.setattr(ms, "_TRUEMEMORY_DIR", d)
    monkeypatch.setattr(ms, "_CONFIG_PATH", cfg)
    monkeypatch.setattr(ms, "_CONFIG_LOCK_PATH", d / "config.json.lock")
    ms._config_cache = None
    ms._config_cache_mtime = 0.0
    ms._config_cache_time = 0.0

    import truememory.ingest.cli as cli
    monkeypatch.setattr(cli, "_TRUEMEMORY_CONFIG_PATH", cfg)

    import truememory.tier_switch.manager as mgr
    monkeypatch.setattr(mgr, "_TRUEMEMORY_DIR", d)

    return d


def _read_cfg(tm_dir) -> dict:
    return json.loads((tm_dir / "config.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# M-26: telemetry._save_user_id goes through the atomic path
# ---------------------------------------------------------------------------

def test_save_user_id_uses_atomic_save_config(tm_dir, monkeypatch):
    """``_save_user_id`` must delegate to ``mcp_server._save_config`` (atomic),
    not bare ``write_text``."""
    import truememory.mcp_server as ms
    import truememory.telemetry as tel

    called = {}

    real_save = ms._save_config

    def spy(config):
        called["config"] = dict(config)
        return real_save(config)

    monkeypatch.setattr(ms, "_save_config", spy)

    tel._save_user_id({"tier": "pro", "user_id": "abc", "anthropic_api_key": "sk-keep"})

    assert called["config"]["user_id"] == "abc"
    # Written atomically and readable
    on_disk = _read_cfg(tm_dir)
    assert on_disk["user_id"] == "abc"
    assert on_disk["anthropic_api_key"] == "sk-keep"
    assert on_disk["tier"] == "pro"


def test_save_user_id_never_leaves_truncated_config(tm_dir):
    """Even under repeated writes, config.json is always complete/parseable —
    the os.replace path is atomic so no reader ever sees an empty file."""
    import truememory.telemetry as tel

    for i in range(50):
        tel._save_user_id({"tier": "base", "user_id": f"u{i}", "anthropic_api_key": "sk"})
        # Every observation is a complete JSON object
        data = _read_cfg(tm_dir)
        assert data["user_id"] == f"u{i}"
        assert data["anthropic_api_key"] == "sk"

    # No corrupt-rename backups produced
    assert not list(tm_dir.glob("config.json.corrupt.*"))


# ---------------------------------------------------------------------------
# M-58: tier switch preserves unrelated fields written by another path
# ---------------------------------------------------------------------------

def test_tier_switch_preserves_api_key(tm_dir):
    """A tier-switch config write must not drop an api_key set by another
    writer (read-modify-write must compose, not clobber)."""
    import truememory.telemetry as tel
    import truememory.tier_switch.manager as mgr

    # Another path persists the API key + user_id (via the atomic writer).
    tel._save_user_id({"user_id": "u1", "anthropic_api_key": "sk-secret", "tier": "edge"})

    # Tier switch flips the tier.
    m = mgr.RebuildManager.__new__(mgr.RebuildManager)
    mgr.RebuildManager._apply_config_switch(m, "pro", None)

    data = _read_cfg(tm_dir)
    assert data["tier"] == "pro"
    # The unrelated fields survive the tier write.
    assert data["anthropic_api_key"] == "sk-secret"
    assert data["user_id"] == "u1"


# ---------------------------------------------------------------------------
# Interleaved writers do not corrupt config.json (cross-process lock + atomic)
# ---------------------------------------------------------------------------

def test_interleaved_writers_never_corrupt(tm_dir):
    """Many threads hammering the two writers + a reader: config.json stays a
    valid, complete JSON object throughout and at the end."""
    import truememory.mcp_server as ms
    import truememory.telemetry as tel
    import truememory.tier_switch.manager as mgr

    # Seed an api_key that tier switches must never drop.
    tel._save_user_id({"user_id": "seed", "anthropic_api_key": "sk-seed", "tier": "edge"})

    errors: list[Exception] = []
    stop = threading.Event()

    def writer_userid():
        for i in range(40):
            try:
                cfg = ms._load_config()
                cfg["user_id"] = f"w{i}"
                tel._save_user_id(cfg)
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)

    def writer_tier():
        m = mgr.RebuildManager.__new__(mgr.RebuildManager)
        for i in range(40):
            try:
                mgr.RebuildManager._apply_config_switch(
                    m, "pro" if i % 2 else "base", None,
                )
            except Exception as e:  # pragma: no cover
                errors.append(e)

    def reader():
        while not stop.is_set():
            try:
                txt = (tm_dir / "config.json").read_text(encoding="utf-8")
                if txt:
                    obj = json.loads(txt)
                    assert isinstance(obj, dict)
            except FileNotFoundError:
                pass
            except Exception as e:  # pragma: no cover - this is the bug we fixed
                errors.append(e)

    threads = [
        threading.Thread(target=writer_userid),
        threading.Thread(target=writer_tier),
        threading.Thread(target=reader),
    ]
    for t in threads:
        t.start()
    threads[0].join()
    threads[1].join()
    stop.set()
    threads[2].join()

    assert not errors, errors[:3]
    # Final state is a complete, valid object that still carries the api_key
    # (no lost update wiped it out).
    data = _read_cfg(tm_dir)
    assert isinstance(data, dict)
    assert data["anthropic_api_key"] == "sk-seed"
    # No corrupt backups were ever created.
    assert not list(tm_dir.glob("config.json.corrupt.*"))
