"""Regression lock for Hunter F04 — corrupt ~/.truememory/config.json.

Prior behavior: `_load_config()` silently swallowed `JSONDecodeError` /
`OSError` and returned `{}`, losing the user's stored tier and API keys
with no visible signal. Fix renames the corrupt file to
`.corrupt.<unix-ts>` (preserving recovery of API keys) and writes a
stderr warning so the user knows something happened.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Redirect `_CONFIG_PATH` / `_TRUEMEMORY_DIR` inside tmp_path.

    Note: we deliberately do NOT monkeypatch ``HOME`` or reload the module.
    ``huggingface_hub`` caches its cache-dir constant at import time — a
    HOME swap pollutes that constant, and later tests that load real
    embedding models fail with ``LocalEntryNotFoundError``.
    """
    home = tmp_path / "home"
    home.mkdir()
    (home / ".truememory").mkdir()
    import truememory.mcp_server as ms
    monkeypatch.setattr(ms, "_TRUEMEMORY_DIR", home / ".truememory")
    monkeypatch.setattr(ms, "_CONFIG_PATH", home / ".truememory" / "config.json")
    yield home, ms


def test_corrupt_json_backed_up_and_warned(fake_home, capsys):
    home, ms = fake_home
    cfg = home / ".truememory" / "config.json"
    cfg.write_text('{"tier": "pro", "anthr', encoding="utf-8")

    result = ms._load_config()

    assert result == {}
    # Original corrupt file must be renamed, not deleted — user may want to
    # recover the partial API key from the backup.
    remaining = sorted(p.name for p in (home / ".truememory").iterdir())
    assert "config.json" not in remaining, "corrupt file should have been moved aside"
    backups = [n for n in remaining if n.startswith("config.json.corrupt.")]
    assert len(backups) == 1, f"expected one backup file, got {remaining!r}"

    captured = capsys.readouterr()
    assert "corrupt" in captured.err.lower()
    assert "config.json.corrupt" in captured.err
    assert "api key" in captured.err.lower() or "recoverable" in captured.err.lower()


def test_osreror_warned_but_no_backup(fake_home, capsys, monkeypatch):
    home, ms = fake_home
    cfg = home / ".truememory" / "config.json"
    cfg.write_text('{"tier": "edge"}', encoding="utf-8")

    # Force read_text to raise OSError
    original_read_text = Path.read_text

    def flaky_read_text(self, *args, **kwargs):
        if str(self) == str(cfg):
            raise PermissionError("simulated: cannot read config.json")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)

    result = ms._load_config()

    assert result == {}
    captured = capsys.readouterr()
    assert "could not read" in captured.err.lower()
    assert "config.json" in captured.err
    # File must remain in place (nothing to back up — we couldn't even read it)
    assert cfg.exists()


def test_missing_config_file_silent(fake_home, capsys):
    """Absent config.json is the normal first-run case — no warning."""
    home, ms = fake_home
    # No config.json written
    result = ms._load_config()
    assert result == {}
    captured = capsys.readouterr()
    assert captured.err == ""


def test_valid_json_roundtrips(fake_home):
    home, ms = fake_home
    cfg = home / ".truememory" / "config.json"
    cfg.write_text('{"tier": "base", "anthropic_api_key": "sk-ant-real"}', encoding="utf-8")
    result = ms._load_config()
    assert result == {"tier": "base", "anthropic_api_key": "sk-ant-real"}
