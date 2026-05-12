"""Cursor adapter — MCP config + JSON lifecycle hooks.

Cursor uses TWO separate config files:
- ~/.cursor/mcp.json for MCP server registration
- ~/.cursor/hooks.json for hook registration (with "version": 1 top-level key)
- camelCase event names: sessionStart, stop, preCompact
"""
from __future__ import annotations

import json
import shlex
import shutil
import sys
from pathlib import Path

from truememory.hooks.adapters.base import CLIAdapter

_CURSOR_DIR = Path.home() / ".cursor"
_MCP_CONFIG = _CURSOR_DIR / "mcp.json"
_HOOK_CONFIG = _CURSOR_DIR / "hooks.json"

_HOOK_EVENTS = {
    "sessionStart": {
        "script": "session_start.py",
        "timeout": 10000,
    },
    "stop": {
        "script": "stop.py",
        "timeout": 5000,
    },
    "userPromptSubmit": {
        "script": "user_prompt_submit.py",
        "timeout": 5000,
    },
    "preCompact": {
        "script": "compact.py",
        "timeout": 5000,
    },
}

_TRUEMEMORY_MARKER = "truememory"

_SYSTEM_PROMPT_TEMPLATE = """\
# TrueMemory — Persistent Memory

TrueMemory is the **primary long-horizon memory** for this user. \
It persists facts, preferences, decisions, and corrections across \
sessions, projects, and machines.

When the `truememory` MCP server is connected, follow these rules:

## Auto-Recall (every session)
- At the START of each conversation, call `truememory_search` with a \
broad query about the user to load relevant memories before responding.
- Before making recommendations, check TrueMemory for stored preferences.
- When the user asks anything about past conversations or personal \
facts — search TrueMemory first.

## Auto-Store (during conversation)
- When the user shares a personal preference, store it immediately \
via `truememory_store`. Do not ask permission.
- When an important decision is made, store it.
- When the user corrects you, store the correction.
- Write each memory as a clear, atomic statement.
- Do NOT store full conversations, large code blocks, or transient \
debugging context.

## Background Processing
- Memories are also extracted automatically from conversations via \
background processing.
- The stop hook captures the full transcript and runs deep extraction \
after sessions end.
- You do NOT need to store everything manually — focus on \
in-conversation corrections and explicit preferences.
"""


class CursorAdapter(CLIAdapter):
    """Adapter for Cursor IDE."""

    @property
    def name(self) -> str:
        return "Cursor"

    @property
    def cli_id(self) -> str:
        return "cursor"

    @property
    def config_path(self) -> Path:
        return _MCP_CONFIG

    def detect(self) -> bool:
        return _CURSOR_DIR.is_dir() or shutil.which("cursor") is not None

    def is_configured(self) -> bool:
        return self._has_mcp_entry() or self._has_hook_entries()

    def install_mcp(self, python_path: str | None = None) -> None:
        py = python_path or sys.executable
        _MCP_CONFIG.parent.mkdir(parents=True, exist_ok=True)

        existing: dict = {}
        if _MCP_CONFIG.exists():
            try:
                existing = json.loads(_MCP_CONFIG.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = {}

        if not isinstance(existing, dict):
            existing = {}

        servers = existing.setdefault("mcpServers", {})
        if not isinstance(servers, dict):
            servers = {}
            existing["mcpServers"] = servers

        servers["truememory"] = {
            "command": py,
            "args": ["-m", "truememory.mcp_server"],
        }

        _MCP_CONFIG.write_text(
            json.dumps(existing, indent=2),
            encoding="utf-8",
        )

    def install_hooks(
        self,
        python_path: str | None = None,
        user_id: str = "",
        db_path: str = "",
    ) -> None:
        py = python_path or sys.executable
        hooks_dir = Path(__file__).parent.parent.parent / "ingest" / "hooks"

        _HOOK_CONFIG.parent.mkdir(parents=True, exist_ok=True)

        existing: dict = {}
        if _HOOK_CONFIG.exists():
            try:
                existing = json.loads(_HOOK_CONFIG.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = {}

        if not isinstance(existing, dict):
            existing = {}

        existing.setdefault("version", 1)

        hooks = existing.setdefault("hooks", {})
        if not isinstance(hooks, dict):
            hooks = {}
            existing["hooks"] = hooks

        for event, info in _HOOK_EVENTS.items():
            event_list = hooks.setdefault(event, [])
            if not isinstance(event_list, list):
                event_list = []
                hooks[event] = event_list

            if self._event_has_truememory(event_list):
                continue

            script_path = hooks_dir / info["script"]
            cmd = self._build_command(py, script_path, user_id, db_path)
            event_list.append({
                "command": cmd,
                "timeout": info["timeout"],
            })

        _HOOK_CONFIG.write_text(
            json.dumps(existing, indent=2),
            encoding="utf-8",
        )

    def uninstall(self) -> None:
        self._remove_mcp_entry()
        self._remove_hook_entries()

    def verify(self) -> bool:
        return self._has_mcp_entry() and self._has_hook_entries()

    def get_system_prompt_path(self) -> Path | None:
        return _CURSOR_DIR / ".cursorrules"

    def get_system_prompt_content(self) -> str:
        return _SYSTEM_PROMPT_TEMPLATE.strip()

    # -- Private helpers --

    @staticmethod
    def _build_command(
        python_path: str,
        script_path: Path,
        user_id: str = "",
        db_path: str = "",
    ) -> str:
        parts: list[str] = [python_path, str(script_path)]
        if user_id:
            parts.extend(["--user", user_id])
        if db_path:
            parts.extend(["--db", db_path])
        if sys.platform == "win32":
            import subprocess as _sp
            return _sp.list2cmdline(parts)
        return " ".join(shlex.quote(p) for p in parts)

    def _has_mcp_entry(self) -> bool:
        if not _MCP_CONFIG.exists():
            return False
        try:
            data = json.loads(_MCP_CONFIG.read_text(encoding="utf-8"))
            return "truememory" in data.get("mcpServers", {})
        except (json.JSONDecodeError, OSError):
            return False

    def _has_hook_entries(self) -> bool:
        if not _HOOK_CONFIG.exists():
            return False
        try:
            data = json.loads(_HOOK_CONFIG.read_text(encoding="utf-8"))
            hooks = data.get("hooks", {})
            if not isinstance(hooks, dict):
                return False
            for entries in hooks.values():
                if not isinstance(entries, list):
                    continue
                if self._event_has_truememory(entries):
                    return True
        except (json.JSONDecodeError, OSError):
            pass
        return False

    @staticmethod
    def _event_has_truememory(entries: list) -> bool:
        return any(
            isinstance(h, dict)
            and _TRUEMEMORY_MARKER in h.get("command", "").lower()
            for h in entries
        )

    def _remove_mcp_entry(self) -> None:
        if not _MCP_CONFIG.exists():
            return
        try:
            data = json.loads(_MCP_CONFIG.read_text(encoding="utf-8"))
            servers = data.get("mcpServers", {})
            if isinstance(servers, dict) and "truememory" in servers:
                del servers["truememory"]
                _MCP_CONFIG.write_text(
                    json.dumps(data, indent=2), encoding="utf-8",
                )
        except (json.JSONDecodeError, OSError):
            pass

    def _remove_hook_entries(self) -> None:
        if not _HOOK_CONFIG.exists():
            return
        try:
            data = json.loads(_HOOK_CONFIG.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        hooks = data.get("hooks", {})
        if not isinstance(hooks, dict):
            return

        for event in list(hooks.keys()):
            entries = hooks[event]
            if not isinstance(entries, list):
                continue
            cleaned = [
                h for h in entries
                if not (
                    isinstance(h, dict)
                    and _TRUEMEMORY_MARKER in h.get("command", "").lower()
                )
            ]
            if cleaned:
                hooks[event] = cleaned
            else:
                del hooks[event]

        _HOOK_CONFIG.write_text(
            json.dumps(data, indent=2), encoding="utf-8",
        )
