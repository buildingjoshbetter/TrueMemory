"""Kimi CLI adapter — MCP config + TOML lifecycle hooks.

Kimi CLI uses:
- ~/.kimi/mcp.json for MCP server registration (Claude Desktop-compatible format)
- ~/.kimi/config.toml for hook registration ([[hooks]] entries)
- Same JSON stdin/stdout hook protocol as Claude Code
"""
from __future__ import annotations

import json
import logging
import shlex
import shutil
import sys
from pathlib import Path

from truememory.hooks.adapters.base import CLIAdapter

log = logging.getLogger(__name__)

_KIMI_DIR = Path.home() / ".kimi"
_MCP_CONFIG = _KIMI_DIR / "mcp.json"
_HOOK_CONFIG = _KIMI_DIR / "config.toml"

_HOOK_EVENTS = {
    "SessionStart": {
        "script": "session_start.py",
        "timeout": 10,  # seconds (Kimi CLI convention)
    },
    "Stop": {
        "script": "stop.py",
        "timeout": 5,  # seconds
    },
    "UserPromptSubmit": {
        "script": "user_prompt_submit.py",
        "timeout": 5,  # seconds
    },
    "PreCompact": {
        "script": "compact.py",
        "timeout": 5,  # seconds
    },
}

_TRUEMEMORY_MARKER = "truememory"


class KimiAdapter(CLIAdapter):
    """Adapter for Kimi CLI (Moonshot AI)."""

    @property
    def has_hooks(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "Kimi CLI"

    @property
    def cli_id(self) -> str:
        return "kimi"

    @property
    def config_path(self) -> Path:
        return _HOOK_CONFIG

    def detect(self) -> bool:
        return _KIMI_DIR.is_dir() or shutil.which("kimi") is not None

    def is_configured(self) -> bool:
        mcp_ok = self._has_mcp_entry()
        hooks_ok = self._has_hook_entries()
        return mcp_ok or hooks_ok

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

        existing_text = ""
        if _HOOK_CONFIG.exists():
            try:
                existing_text = _HOOK_CONFIG.read_text(encoding="utf-8")
            except OSError:
                existing_text = ""

        existing_hooks = self._parse_existing_hooks(existing_text)

        lines_to_append: list[str] = []
        for event, info in _HOOK_EVENTS.items():
            if self._event_already_registered(existing_hooks, event):
                continue

            script_path = hooks_dir / info["script"]
            cmd = self._build_command(py, script_path, user_id, db_path)

            lines_to_append.append("")
            lines_to_append.append("[[hooks]]")
            lines_to_append.append(f'event = "{event}"')
            lines_to_append.append(f'command = "{cmd}"')
            lines_to_append.append(f'timeout = {info["timeout"]}')

        if lines_to_append:
            new_text = existing_text.rstrip() + "\n" + "\n".join(lines_to_append) + "\n"
            _HOOK_CONFIG.write_text(new_text, encoding="utf-8")

    def uninstall(self) -> None:
        self._remove_mcp_entry()
        self._remove_hook_entries()

    def verify(self) -> bool:
        return self._has_mcp_entry() and self._has_hook_entries()

    def get_system_prompt_path(self) -> Path | None:
        return Path.home() / ".kimi" / "truememory_prompt.md"

    def get_system_prompt_content(self) -> str:
        from truememory.hooks.adapters.base import get_generic_system_prompt
        return get_generic_system_prompt(
            has_hooks=self.has_hooks,
            has_session_start=self.has_session_start,
        )

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

    @staticmethod
    def _parse_existing_hooks(toml_text: str) -> list[dict]:
        if not toml_text.strip():
            return []
        try:
            import tomllib
            data = tomllib.loads(toml_text)
            return data.get("hooks", [])
        except ModuleNotFoundError:
            import re
            hooks: list[dict] = []
            for block in re.split(r'(?=^\[\[hooks\]\])', toml_text, flags=re.MULTILINE):
                if not block.strip().startswith("[[hooks]]"):
                    continue
                entry: dict[str, str] = {}
                for line in block.splitlines()[1:]:
                    line = line.strip()
                    if not line or line.startswith("["):
                        break
                    m = re.match(r'(\w+)\s*=\s*"([^"]*)"', line)
                    if m:
                        entry[m.group(1)] = m.group(2)
                    else:
                        m = re.match(r'(\w+)\s*=\s*(\d+)', line)
                        if m:
                            entry[m.group(1)] = m.group(2)
                if entry:
                    hooks.append(entry)
            return hooks
        except Exception:
            return []

    @staticmethod
    def _event_already_registered(hooks: list[dict], event: str) -> bool:
        for h in hooks:
            if (
                h.get("event") == event
                and _TRUEMEMORY_MARKER in h.get("command", "").lower()
            ):
                return True
        return False

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
            text = _HOOK_CONFIG.read_text(encoding="utf-8")
            hooks = self._parse_existing_hooks(text)
            return any(
                _TRUEMEMORY_MARKER in h.get("command", "").lower()
                for h in hooks
            )
        except OSError:
            return False

    def _remove_mcp_entry(self) -> None:
        if not _MCP_CONFIG.exists():
            return
        try:
            data = json.loads(_MCP_CONFIG.read_text(encoding="utf-8"))
            servers = data.get("mcpServers", {})
            if "truememory" in servers:
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
            text = _HOOK_CONFIG.read_text(encoding="utf-8")
        except OSError:
            return

        lines = text.splitlines(keepends=True)
        cleaned: list[str] = []

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if stripped == "[[hooks]]":
                block_lines = [line]
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("["):
                    block_lines.append(lines[i])
                    i += 1

                block_text = "".join(block_lines)
                if _TRUEMEMORY_MARKER not in block_text.lower():
                    cleaned.extend(block_lines)
                continue

            cleaned.append(line)
            i += 1

        _HOOK_CONFIG.write_text("".join(cleaned), encoding="utf-8")
