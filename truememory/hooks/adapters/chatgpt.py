"""ChatGPT Desktop adapter - local MCP config."""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

from truememory.hooks.adapters.base import CLIAdapter

if sys.platform == "darwin":
    _CHATGPT_DIR = Path.home() / "Library" / "Application Support" / "com.openai.chat"
    _CONFIG_PATH = _CHATGPT_DIR / "mcp.json"
elif sys.platform == "win32":
    _base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    _CHATGPT_DIR = Path(_base) / "com.openai.chat" if _base else Path.home() / "AppData" / "Local" / "com.openai.chat"
    _CONFIG_PATH = _CHATGPT_DIR / "mcp.json"
else:
    _CHATGPT_DIR = Path.home() / ".config" / "com.openai.chat"
    _CONFIG_PATH = _CHATGPT_DIR / "mcp.json"

_TRUEMEMORY_MARKER = "truememory"


class ChatGPTAdapter(CLIAdapter):
    """Adapter for ChatGPT Desktop MCP servers."""

    @property
    def name(self) -> str:
        return "ChatGPT Desktop"

    @property
    def cli_id(self) -> str:
        return "chatgpt"

    @property
    def config_path(self) -> Path:
        return _CONFIG_PATH

    def detect(self) -> bool:
        return _CHATGPT_DIR.is_dir() or _CONFIG_PATH.exists()

    def is_configured(self) -> bool:
        return self._has_mcp_entry()

    def install_mcp(self, python_path: str | None = None) -> None:
        py = python_path or sys.executable

        existing: dict = {}
        if _CONFIG_PATH.exists():
            try:
                data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = None
            if isinstance(data, dict):
                existing = data
            else:
                # Unparseable (or non-object) config: never silently destroy
                # it. Back it up first; if the backup fails, refuse.
                backup = _CONFIG_PATH.with_name(
                    f"{_CONFIG_PATH.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
                )
                try:
                    shutil.copy2(_CONFIG_PATH, backup)
                except OSError as e:
                    raise RuntimeError(
                        f"Existing {_CONFIG_PATH} is not valid JSON and could "
                        f"not be backed up ({e}); refusing to overwrite it."
                    ) from e
                print(
                    f"\033[33m⚠ Existing {_CONFIG_PATH} is not valid JSON; "
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

        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CONFIG_PATH.write_text(
            json.dumps(existing, indent=2),
            encoding="utf-8",
        )

    def install_hooks(
        self,
        python_path: str | None = None,
        user_id: str = "",
        db_path: str = "",
    ) -> None:
        # ChatGPT Desktop exposes MCP tools, but not TrueMemory lifecycle hooks.
        del python_path, user_id, db_path

    def uninstall(self) -> None:
        # Match the Cursor house pattern: never write unless the file parsed
        # cleanly AND our entry is actually present. A corrupt config must be
        # left untouched (it may be user-recoverable).
        if not _CONFIG_PATH.exists():
            return
        try:
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            servers = data.get("mcpServers", {})
            if isinstance(servers, dict) and _TRUEMEMORY_MARKER in servers:
                del servers[_TRUEMEMORY_MARKER]
                _CONFIG_PATH.write_text(
                    json.dumps(data, indent=2), encoding="utf-8",
                )
        except (json.JSONDecodeError, OSError):
            pass

    def verify(self) -> bool:
        return self._has_mcp_entry()

    def get_system_prompt_path(self) -> Path | None:
        return None

    def get_system_prompt_content(self) -> str:
        return ""

    def _has_mcp_entry(self) -> bool:
        data = self._read_config()
        servers = data.get("mcpServers", {})
        return isinstance(servers, dict) and _TRUEMEMORY_MARKER in servers

    @staticmethod
    def _read_config() -> dict:
        if not _CONFIG_PATH.exists():
            return {}
        try:
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return data if isinstance(data, dict) else {}
