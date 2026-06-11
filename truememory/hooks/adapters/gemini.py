"""Gemini CLI adapter -- MCP config + JSON lifecycle hooks.

Gemini CLI uses:
- ~/.gemini/settings.json for BOTH MCP server registration AND hook registration
- MCP under mcpServers key (Claude Desktop-compatible format)
- Hooks under hooks key with PascalCase event names, nested HookDefinition format
- Same JSON stdin/stdout hook protocol as Claude Code

Hook format (from google-gemini/gemini-cli TypeScript types):
  HookDefinition: { matcher?: string, sequential?: bool, hooks: HookConfig[] }
  CommandHookConfig: { type: "command", command: string, name?: string, timeout?: number }
"""
from __future__ import annotations

import json
import shlex
import shutil
import sys
from pathlib import Path

from truememory.hooks.adapters.base import CLIAdapter, atomic_write_text

_GEMINI_DIR = Path.home() / ".gemini"
_CONFIG_PATH = _GEMINI_DIR / "settings.json"

_HOOK_EVENTS = {
    "SessionStart": {
        "script": "session_start.py",
        "timeout": 10000,
    },
    "SessionEnd": {
        "script": "stop.py",
        "timeout": 5000,
    },
    "BeforeAgent": {
        "script": "user_prompt_submit.py",
        "timeout": 5000,
    },
    "PreCompress": {
        "script": "compact.py",
        "timeout": 5000,
    },
}

_TRUEMEMORY_MARKER = "truememory"

_SYSTEM_PROMPT_TEMPLATE = """\
# TrueMemory -- Persistent Memory

TrueMemory is the **primary long-horizon memory** for this user. \
It persists facts, preferences, decisions, and corrections across \
sessions, projects, and machines.

When the `truememory` MCP server is connected, follow these rules:

## Auto-Recall (every session)
- At the START of each conversation, call `truememory_search` with a \
broad query about the user to load relevant memories before responding.
- Directives are automatically injected at session start -- you do not \
need to search for them.
- Before making recommendations, check TrueMemory for stored preferences.
- When the user asks anything about past conversations or personal \
facts -- search TrueMemory first.

## Auto-Store (during conversation)
- When the user shares a personal preference, store it immediately \
via `truememory_store`. Do not ask permission.
- When an important decision is made, store it.
- When the user corrects you, store the correction.
- When the user gives a standing instruction ("always do X", \
"never do Y", "from now on..."), store it as a directive: \
`truememory_store(content="...", directive=True)`. Directives \
auto-load at the start of every session -- regular memories do not.
- Write each memory as a clear, atomic statement.
- Do NOT store full conversations, large code blocks, or transient \
debugging context.

## Background Processing
- Memories are also extracted automatically from conversations via \
background processing.
- The SessionEnd hook captures the full transcript and runs deep extraction \
after sessions end.
- You do NOT need to store everything manually -- focus on \
in-conversation corrections and explicit preferences.
"""


class GeminiAdapter(CLIAdapter):
    """Adapter for Google Gemini CLI."""

    @property
    def has_hooks(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "Gemini CLI"

    @property
    def cli_id(self) -> str:
        return "gemini"

    @property
    def config_path(self) -> Path:
        return _CONFIG_PATH

    def detect(self) -> bool:
        return _GEMINI_DIR.is_dir() or shutil.which("gemini") is not None

    def is_configured(self) -> bool:
        return self._has_mcp_entry() or self._has_hook_entries()

    def install_mcp(self, python_path: str | None = None) -> None:
        py = python_path or sys.executable
        settings = self._read_config()

        servers = settings.setdefault("mcpServers", {})
        if not isinstance(servers, dict):
            servers = {}
            settings["mcpServers"] = servers

        servers["truememory"] = {
            "command": py,
            "args": ["-m", "truememory.mcp_server"],
        }

        self._write_config(settings)

    def install_hooks(
        self,
        python_path: str | None = None,
        user_id: str = "",
        db_path: str = "",
    ) -> None:
        py = python_path or sys.executable
        hooks_dir = Path(__file__).parent.parent.parent / "ingest" / "hooks"

        settings = self._read_config()
        hooks = settings.setdefault("hooks", {})
        if not isinstance(hooks, dict):
            hooks = {}
            settings["hooks"] = hooks

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
                "hooks": [{
                    "type": "command",
                    "command": cmd,
                    "name": f"truememory-{event.lower()}",
                    "timeout": info["timeout"],
                }],
            })

        self._write_config(settings)

    def uninstall(self) -> None:
        if not _CONFIG_PATH.exists():
            return
        settings = self._read_config()

        servers = settings.get("mcpServers", {})
        if isinstance(servers, dict) and "truememory" in servers:
            del servers["truememory"]

        hooks = settings.get("hooks", {})
        if isinstance(hooks, dict):
            for event in list(hooks.keys()):
                entries = hooks[event]
                if not isinstance(entries, list):
                    continue
                cleaned = [
                    h for h in entries
                    if not self._definition_has_truememory(h)
                ]
                if cleaned:
                    hooks[event] = cleaned
                else:
                    del hooks[event]

        self._write_config(settings)

    def verify(self) -> bool:
        if not _CONFIG_PATH.exists():
            return False
        return self._has_mcp_entry() and self._has_hook_entries()

    def get_system_prompt_path(self) -> Path | None:
        return _GEMINI_DIR / "GEMINI.md"

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

    def _read_config(self) -> dict:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if _CONFIG_PATH.exists():
            try:
                data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _write_config(self, settings: dict) -> None:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(_CONFIG_PATH, json.dumps(settings, indent=2))

    def _has_mcp_entry(self) -> bool:
        if not _CONFIG_PATH.exists():
            return False
        try:
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            return "truememory" in data.get("mcpServers", {})
        except (json.JSONDecodeError, OSError):
            return False

    def _has_hook_entries(self) -> bool:
        if not _CONFIG_PATH.exists():
            return False
        try:
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
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
    def _definition_has_truememory(hook_def: object) -> bool:
        """Check if a HookDefinition contains a TrueMemory hook config.

        Handles both the correct nested format:
            {"hooks": [{"type": "command", "command": "...truememory..."}]}
        and the legacy flat format (for migration):
            {"command": "...truememory...", "timeout": 10000}
        """
        if not isinstance(hook_def, dict):
            return False
        # Correct nested format: check inside hooks sub-array
        inner_hooks = hook_def.get("hooks", [])
        if isinstance(inner_hooks, list):
            for hc in inner_hooks:
                if (
                    isinstance(hc, dict)
                    and _TRUEMEMORY_MARKER in hc.get("command", "").lower()
                ):
                    return True
        # Legacy flat format fallback
        if _TRUEMEMORY_MARKER in hook_def.get("command", "").lower():
            return True
        return False

    @staticmethod
    def _event_has_truememory(entries: list) -> bool:
        """Check if any HookDefinition in the event list is a TrueMemory hook."""
        return any(
            GeminiAdapter._definition_has_truememory(h)
            for h in entries
        )
