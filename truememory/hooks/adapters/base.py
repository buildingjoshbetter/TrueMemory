"""Abstract base class for CLI adapters.

Each supported CLI (Claude Code, Kimi, Hermes, OpenClaw) implements
this interface to handle its config format, hook registration, and
MCP server setup.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* atomically (tmp-in-same-dir + os.replace).

    X2-1 (#691): adapters write the user's live editor/CLI config. A bare
    ``write_text`` can truncate that config if the process dies mid-write.
    Writing to a unique per-process tmp in the same directory and atomically
    renaming means a reader (or a crash) sees either the old file or the fully
    written new one — never a torn one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, data, indent: int = 2) -> None:
    """Atomically write *data* as JSON (see :func:`atomic_write_text`)."""
    atomic_write_text(path, json.dumps(data, indent=indent))


class CLIAdapter(ABC):
    """Base interface for CLI-specific TrueMemory integration."""

    # --- Capability flags -------------------------------------------------
    # Hook-less adapters (Antigravity, ChatGPT, Cursor, ...) expose MCP tools
    # but cannot register lifecycle hooks. Their system prompt MUST NOT promise
    # auto-loading directives or SessionEnd transcript capture, because nothing
    # delivers those. Adapters that DO install hooks override these to True.
    @property
    def has_hooks(self) -> bool:
        """True if install_hooks registers real lifecycle hooks."""
        return False

    @property
    def has_session_start(self) -> bool:
        """True if a SessionStart hook injects directives every session."""
        return self.has_hooks

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable CLI name (e.g. 'Claude Code')."""

    @property
    @abstractmethod
    def cli_id(self) -> str:
        """Machine identifier (e.g. 'claude', 'kimi', 'hermes', 'openclaw')."""

    @property
    @abstractmethod
    def config_path(self) -> Path:
        """Path to the CLI's main config file."""

    @abstractmethod
    def detect(self) -> bool:
        """Return True if this CLI is installed on the system."""

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if TrueMemory is already wired into this CLI."""

    @abstractmethod
    def install_mcp(self, python_path: str | None = None) -> None:
        """Register the TrueMemory MCP server in the CLI's config."""

    @abstractmethod
    def install_hooks(
        self,
        python_path: str | None = None,
        user_id: str = "",
        db_path: str = "",
    ) -> None:
        """Register TrueMemory lifecycle hooks in the CLI's config."""

    @abstractmethod
    def uninstall(self) -> None:
        """Remove all TrueMemory entries from the CLI's config."""

    @abstractmethod
    def verify(self) -> bool:
        """Smoke-test the installation (config exists, paths resolve)."""

    @abstractmethod
    def get_system_prompt_path(self) -> Path | None:
        """Return the path to the CLI's system prompt file, or None."""

    @abstractmethod
    def get_system_prompt_content(self) -> str:
        """Return the TrueMemory system prompt content for this CLI."""


# Served when CLAUDE_TEMPLATE.md is missing or unreadable. Must carry the
# same directive guidance as the template (issue #589, D-7) so directives
# stay discoverable even on broken installs. The auto-load / store-manually
# sentence is appended per the host's hook capability (issue #651, M-62).
_FALLBACK_PROMPT = (
    "# TrueMemory — Persistent Memory\n\n"
    "You have access to TrueMemory MCP tools for persistent memory.\n"
    "- Use `truememory_store` to save user facts, preferences, and decisions.\n"
    '- When the user gives a standing instruction ("always do X", "never do Y", '
    '"from now on..."), store it as a directive: '
    '`truememory_store(content="...", directive=True)`.\n'
    "- Use `truememory_search` to recall stored memories before answering.\n"
    "- Search TrueMemory FIRST on any 'do you remember' question.\n"
)

# Appended only for hosts with a SessionStart hook — they get directives
# injected automatically. Hook-less hosts must be told to fetch them.
_FALLBACK_AUTOLOAD = (
    "- Directives are injected automatically at the start of every session — "
    "you do not need to search for them.\n"
)
_FALLBACK_NO_AUTOLOAD = (
    "- This host has no session-start hook, so directives are NOT auto-injected. "
    "At the start of a session, call `truememory_search` for standing "
    "instructions and follow any directives it returns.\n"
)


def get_generic_system_prompt(
    has_hooks: bool = False,
    has_session_start: bool | None = None,
) -> str:
    """Return the TrueMemory system prompt for non-Claude CLIs.

    The base template (CLAUDE_TEMPLATE.md) is written for Claude Code, which
    installs SessionStart/SessionEnd hooks and uses MEMORY.md. Hook-less
    adapters (Antigravity, ChatGPT) must NOT inherit those promises — there is
    no hook to auto-load directives or capture the transcript, and they have no
    MEMORY.md. Pass the adapter's capability flags so the prompt is honest.
    """
    if has_session_start is None:
        has_session_start = has_hooks

    template = Path(__file__).parent.parent.parent / "ingest" / "CLAUDE_TEMPLATE.md"
    if has_hooks and template.exists():
        # Hook-capable non-Claude host: the template's hook-based guarantees
        # hold, just relabel Claude-specific bits.
        try:
            content = template.read_text(encoding="utf-8").strip()
            content = content.replace("Claude Code's built-in auto-memory", "The host CLI's built-in memory")
            content = content.replace("(`MEMORY.md` files under `~/.claude/projects/*/memory/`)", "")
            return content
        except OSError:
            pass

    # Hook-less host (or unreadable template): build an honest prompt that does
    # not claim auto-load / SessionEnd capture / MEMORY.md.
    prompt = _FALLBACK_PROMPT
    prompt += _FALLBACK_AUTOLOAD if has_session_start else _FALLBACK_NO_AUTOLOAD
    if not has_hooks:
        prompt += (
            "- This host does NOT capture the transcript at session end. "
            "TrueMemory cannot extract memories automatically here, so store "
            "important facts, preferences, decisions, and corrections yourself "
            "with `truememory_store` as the conversation happens.\n"
        )
    return prompt
