"""Antigravity adapter -- MCP config for the Antigravity AI CLI.

Antigravity reads its MCP config from a JSON file named ``mcp_config.json``
(the filename its docs and "View raw config" UI use), following the same
mcpServers format as Claude Desktop / ChatGPT Desktop. Earlier versions of
this adapter wrote ``~/.antigravity/mcp.json`` — a filename the host does not
read — and verify() re-read that same file, so a broken install still reported
success. We now write/verify ``mcp_config.json``.

NOTE (uncertainty): community docs also place this file under ``~/.gemini/``
(e.g. ``~/.gemini/config/mcp_config.json`` or ``~/.gemini/antigravity/``)
rather than ``~/.antigravity/``. The directory is left as ``~/.antigravity``
pending confirmation; the filename fix is the high-confidence change. See PR
notes for M-66.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

from truememory.hooks.adapters.base import CLIAdapter, get_generic_system_prompt

_ANTIGRAVITY_DIR = Path.home() / ".antigravity"
_MCP_CONFIG = _ANTIGRAVITY_DIR / "mcp_config.json"

_TRUEMEMORY_MARKER = "truememory"


def _atomic_write(path: Path, text: str) -> None:
    """Write text to path atomically: tempfile in the same dir + os.replace."""
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except OSError:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _backup_config(config_path: Path) -> Path:
    """Copy an unparseable config aside before overwriting it."""
    original = config_path.read_bytes()
    backup = config_path.with_name(
        f"{config_path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        f"-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    fd = os.open(str(backup), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(original)
    return backup


class AntigravityAdapter(CLIAdapter):
    """Adapter for the Antigravity AI CLI."""

    @property
    def name(self) -> str:
        return "Antigravity"

    @property
    def cli_id(self) -> str:
        return "antigravity"

    @property
    def config_path(self) -> Path:
        return _MCP_CONFIG

    def detect(self) -> bool:
        return _ANTIGRAVITY_DIR.is_dir() or shutil.which("antigravity") is not None

    def is_configured(self) -> bool:
        return self._has_mcp_entry()

    def install_mcp(self, python_path: str | None = None) -> None:
        py = python_path or sys.executable

        existing: dict = {}
        if _MCP_CONFIG.exists():
            try:
                data = json.loads(_MCP_CONFIG.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = None
            if isinstance(data, dict):
                existing = data
            else:
                try:
                    backup = _backup_config(_MCP_CONFIG)
                except OSError as e:
                    raise RuntimeError(
                        f"Existing {_MCP_CONFIG} is not valid JSON and could "
                        f"not be backed up ({e}); refusing to overwrite it."
                    ) from e
                print(
                    f"\033[33m⚠ Existing {_MCP_CONFIG} is not valid JSON; "
                    f"backed it up to {backup} and starting fresh.\033[0m",
                    file=sys.stderr,
                )

        servers = existing.setdefault("mcpServers", {})
        if not isinstance(servers, dict):
            servers = {}
            existing["mcpServers"] = servers

        servers["truememory"] = {
            "command": py,
            "args": ["-m", "truememory.mcp_server"],
        }

        _MCP_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(_MCP_CONFIG, json.dumps(existing, indent=2))

    def install_hooks(
        self,
        python_path: str | None = None,
        user_id: str = "",
        db_path: str = "",
    ) -> None:
        # Antigravity exposes MCP tools but does not support lifecycle hooks yet.
        del python_path, user_id, db_path

    def uninstall(self) -> None:
        config_path = _MCP_CONFIG
        if not config_path.exists():
            return
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(
                f"\033[33m⚠ {config_path} could not be parsed ({e}); "
                f"leaving it untouched -- remove the truememory entry "
                f"manually if present.\033[0m",
                file=sys.stderr,
            )
            return
        if not isinstance(data, dict):
            print(
                f"\033[33m⚠ {config_path} is not a JSON object; "
                f"leaving it untouched.\033[0m",
                file=sys.stderr,
            )
            return
        servers = data.get("mcpServers", {})
        if isinstance(servers, dict) and _TRUEMEMORY_MARKER in servers:
            del servers[_TRUEMEMORY_MARKER]
            _atomic_write(config_path, json.dumps(data, indent=2))

    def verify(self) -> bool:
        return self._has_mcp_entry()

    def get_system_prompt_path(self) -> Path | None:
        return _ANTIGRAVITY_DIR / "ANTIGRAVITY.md"

    def get_system_prompt_content(self) -> str:
        # Antigravity has no lifecycle hooks: pass capability flags so the
        # prompt does not falsely promise auto-loaded directives or SessionEnd
        # transcript capture.
        return get_generic_system_prompt(
            has_hooks=self.has_hooks,
            has_session_start=self.has_session_start,
        )

    def _has_mcp_entry(self) -> bool:
        if not _MCP_CONFIG.exists():
            return False
        try:
            data = json.loads(_MCP_CONFIG.read_text(encoding="utf-8"))
            return isinstance(data, dict) and _TRUEMEMORY_MARKER in data.get("mcpServers", {})
        except (json.JSONDecodeError, OSError):
            return False
