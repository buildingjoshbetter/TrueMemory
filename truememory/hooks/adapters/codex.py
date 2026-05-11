"""Codex CLI adapter — MCP config + TOML lifecycle hooks.

Codex CLI uses:
- ~/.codex/config.toml for BOTH MCP server registration AND hook registration
- MCP under [mcp_servers.name] sections
- Hooks as [[hooks]] entries
- Same JSON stdin/stdout hook protocol as Claude Code
"""
from __future__ import annotations

import logging
import re
import shlex
import shutil
import sys
from pathlib import Path

from truememory.hooks.adapters.base import CLIAdapter

log = logging.getLogger(__name__)

_CODEX_DIR = Path.home() / ".codex"
_CONFIG_PATH = _CODEX_DIR / "config.toml"

_HOOK_EVENTS = {
    "SessionStart": {
        "script": "session_start.py",
        "timeout": 10000,
    },
    "Stop": {
        "script": "stop.py",
        "timeout": 5000,
    },
    "UserPromptSubmit": {
        "script": "user_prompt_submit.py",
        "timeout": 5000,
    },
}

_TRUEMEMORY_MARKER = "truememory"

_AGENTS_TEMPLATE = """\
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
- The Stop hook captures the full transcript and runs deep extraction \
after sessions end.
- You do NOT need to store everything manually — focus on \
in-conversation corrections and explicit preferences.
"""


class CodexAdapter(CLIAdapter):
    """Adapter for OpenAI Codex CLI."""

    @property
    def name(self) -> str:
        return "Codex CLI"

    @property
    def cli_id(self) -> str:
        return "codex"

    @property
    def config_path(self) -> Path:
        return _CONFIG_PATH

    def detect(self) -> bool:
        return _CODEX_DIR.is_dir() or shutil.which("codex") is not None

    def is_configured(self) -> bool:
        return self._has_mcp_entry() or self._has_hook_entries()

    def install_mcp(self, python_path: str | None = None) -> None:
        py = python_path or sys.executable
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

        existing_text = ""
        if _CONFIG_PATH.exists():
            try:
                existing_text = _CONFIG_PATH.read_text(encoding="utf-8")
            except OSError:
                existing_text = ""

        if self._has_mcp_entry_in_text(existing_text):
            return

        section = (
            "\n[mcp_servers.truememory]\n"
            f'command = "{py}"\n'
            'args = ["-m", "truememory.mcp_server"]\n'
        )

        new_text = existing_text.rstrip() + "\n" + section
        _CONFIG_PATH.write_text(new_text, encoding="utf-8")

    def install_hooks(
        self,
        python_path: str | None = None,
        user_id: str = "",
        db_path: str = "",
    ) -> None:
        py = python_path or sys.executable
        hooks_dir = Path(__file__).parent.parent.parent / "ingest" / "hooks"

        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

        existing_text = ""
        if _CONFIG_PATH.exists():
            try:
                existing_text = _CONFIG_PATH.read_text(encoding="utf-8")
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
            _CONFIG_PATH.write_text(new_text, encoding="utf-8")

    def uninstall(self) -> None:
        if not _CONFIG_PATH.exists():
            return
        try:
            text = _CONFIG_PATH.read_text(encoding="utf-8")
        except OSError:
            return

        text = self._remove_mcp_section(text)
        text = self._remove_hook_blocks(text)
        _CONFIG_PATH.write_text(text, encoding="utf-8")

    def verify(self) -> bool:
        if not _CONFIG_PATH.exists():
            return False
        try:
            text = _CONFIG_PATH.read_text(encoding="utf-8")
        except OSError:
            return False
        return (
            self._has_mcp_entry_in_text(text)
            and self._has_hook_entries_in_text(text)
        )

    def get_system_prompt_path(self) -> Path | None:
        return Path.home() / ".codex" / "AGENTS.md"

    def get_system_prompt_content(self) -> str:
        return _AGENTS_TEMPLATE.strip()

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

    @staticmethod
    def _has_mcp_entry_in_text(text: str) -> bool:
        return "[mcp_servers.truememory]" in text

    def _has_mcp_entry(self) -> bool:
        if not _CONFIG_PATH.exists():
            return False
        try:
            text = _CONFIG_PATH.read_text(encoding="utf-8")
            return self._has_mcp_entry_in_text(text)
        except OSError:
            return False

    def _has_hook_entries_in_text(self, text: str) -> bool:
        hooks = self._parse_existing_hooks(text)
        return any(
            _TRUEMEMORY_MARKER in h.get("command", "").lower()
            for h in hooks
        )

    def _has_hook_entries(self) -> bool:
        if not _CONFIG_PATH.exists():
            return False
        try:
            text = _CONFIG_PATH.read_text(encoding="utf-8")
            return self._has_hook_entries_in_text(text)
        except OSError:
            return False

    @staticmethod
    def _remove_mcp_section(text: str) -> str:
        lines = text.splitlines(keepends=True)
        cleaned: list[str] = []
        skip = False

        for line in lines:
            stripped = line.strip()
            if stripped == "[mcp_servers.truememory]":
                skip = True
                continue
            if skip:
                if stripped.startswith("[") and stripped != "[mcp_servers.truememory]":
                    skip = False
                    cleaned.append(line)
                elif not stripped or "=" in stripped:
                    continue
                else:
                    skip = False
                    cleaned.append(line)
            else:
                cleaned.append(line)

        return "".join(cleaned)

    @staticmethod
    def _remove_hook_blocks(text: str) -> str:
        lines = text.splitlines(keepends=True)
        cleaned: list[str] = []

        i = 0
        while i < len(lines):
            stripped = lines[i].strip()

            if stripped == "[[hooks]]":
                block_lines = [lines[i]]
                i += 1
                while (
                    i < len(lines)
                    and lines[i].strip()
                    and not lines[i].strip().startswith("[[")
                    and not lines[i].strip().startswith("[")
                ):
                    block_lines.append(lines[i])
                    i += 1

                block_text = "".join(block_lines)
                if _TRUEMEMORY_MARKER not in block_text.lower():
                    cleaned.extend(block_lines)
                continue

            cleaned.append(lines[i])
            i += 1

        return "".join(cleaned)
