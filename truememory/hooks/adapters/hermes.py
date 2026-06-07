"""Hermes Agent adapter — YAML config for MCP + shell hooks.

Hermes Agent (Nous Research) uses:
- ~/.hermes/config.yaml for both MCP server registration (mcp_servers: key)
  and shell hook registration (hooks: key)
- Shell hooks are list-of-dicts under each event name with command/matcher/timeout
- Same JSON stdin/stdout wire protocol as Claude Code for shell hooks
- 19 valid hook events defined in hermes_cli/plugins.py VALID_HOOKS

Hook config format (from hermes-agent source — agent/shell_hooks.py):
  hooks:
    <event_name>:
      - command: "path/to/script.sh"   # REQUIRED, ~ expanded, shlex-split
        matcher: "regex_pattern"        # OPTIONAL, only pre/post_tool_call
        timeout: 10                     # OPTIONAL, seconds (default 60, max 300)

Valid events: pre_tool_call, post_tool_call, transform_terminal_output,
  transform_tool_result, transform_llm_output, pre_llm_call, post_llm_call,
  pre_api_request, post_api_request, api_request_error, on_session_start,
  on_session_end, on_session_finalize, on_session_reset, subagent_start,
  subagent_stop, pre_gateway_dispatch, pre_approval_request,
  post_approval_response
"""
from __future__ import annotations

import logging
import shlex
import shutil
import sys
from pathlib import Path

from truememory.hooks.adapters.base import CLIAdapter

log = logging.getLogger(__name__)

_HERMES_DIR = Path.home() / ".hermes"
_CONFIG = _HERMES_DIR / "config.yaml"

# Map TrueMemory hook scripts to Hermes hook events.
# Hermes uses a single config.yaml for both mcp_servers and hooks.
#
# NOTE on pre_llm_call -> user_prompt_submit.py:
#   Hermes fires pre_llm_call on every LLM API call, including continuation
#   and retry calls — not just on new user prompts.  user_prompt_submit.py is
#   designed to be idempotent (buffers are append-only, recall is read-only,
#   extraction is guarded by should_extract_session), so repeated invocations
#   are harmless but wasteful.  Hermes does not expose a dedicated
#   "user_prompt_submitted" event; pre_llm_call is the closest approximation.
#   The script exits early if stdin provides no prompt or a prompt < 3 chars,
#   which naturally filters most continuation/retry calls where the prompt
#   field is empty.
_HOOK_ENTRIES = {
    "on_session_start": {
        "script": "session_start.py",
        "timeout": 30,
    },
    "on_session_end": {
        "script": "stop.py",
        "timeout": 30,
    },
    "pre_llm_call": {
        "script": "user_prompt_submit.py",
        "timeout": 10,
    },
    "on_session_finalize": {
        "script": "compact.py",
        "timeout": 15,
    },
}

_TRUEMEMORY_MARKER = "truememory"


def _yaml_safe_load(text: str) -> dict:
    try:
        import yaml
    except ImportError:
        return {}
    data = yaml.safe_load(text)
    return data if isinstance(data, dict) else {}


def _yaml_safe_dump(data: dict) -> str:
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for Hermes integration")
    return yaml.safe_dump(data, default_flow_style=False, sort_keys=False)


class HermesAdapter(CLIAdapter):
    """Adapter for Hermes Agent (Nous Research)."""

    @property
    def name(self) -> str:
        return "Hermes Agent"

    @property
    def cli_id(self) -> str:
        return "hermes"

    @property
    def config_path(self) -> Path:
        return _CONFIG

    def detect(self) -> bool:
        return _HERMES_DIR.is_dir() or shutil.which("hermes") is not None

    def is_configured(self) -> bool:
        return self._has_mcp_entry() or self._has_hook_entries()

    def install_mcp(self, python_path: str | None = None) -> None:
        py = python_path or sys.executable
        _CONFIG.parent.mkdir(parents=True, exist_ok=True)

        existing: dict = {}
        if _CONFIG.exists():
            try:
                existing = _yaml_safe_load(_CONFIG.read_text(encoding="utf-8"))
            except OSError:
                existing = {}

        servers = existing.setdefault("mcp_servers", {})
        if not isinstance(servers, dict):
            servers = {}
            existing["mcp_servers"] = servers

        servers["truememory"] = {
            "command": py,
            "args": ["-m", "truememory.mcp_server"],
        }

        _CONFIG.write_text(_yaml_safe_dump(existing), encoding="utf-8")

    def install_hooks(
        self,
        python_path: str | None = None,
        user_id: str = "",
        db_path: str = "",
    ) -> None:
        """Register TrueMemory shell hooks in ~/.hermes/config.yaml.

        Hermes hooks live under the ``hooks:`` key in config.yaml (the same
        file used for ``mcp_servers:``).  Each event maps to a **list** of
        entries.  Each entry has ``command`` (required), ``timeout`` (optional,
        seconds), and ``matcher`` (optional, only for pre/post_tool_call).
        """
        py = python_path or sys.executable
        hooks_dir = Path(__file__).parent.parent.parent / "ingest" / "hooks"

        _CONFIG.parent.mkdir(parents=True, exist_ok=True)

        existing: dict = {}
        if _CONFIG.exists():
            try:
                existing = _yaml_safe_load(_CONFIG.read_text(encoding="utf-8"))
            except OSError:
                existing = {}

        hooks = existing.setdefault("hooks", {})
        if not isinstance(hooks, dict):
            hooks = {}
            existing["hooks"] = hooks

        for event, info in _HOOK_ENTRIES.items():
            script_path = hooks_dir / info["script"]
            cmd = self._build_command(py, script_path, user_id, db_path)

            # Ensure the event key is a list
            event_list = hooks.setdefault(event, [])
            if not isinstance(event_list, list):
                event_list = []
                hooks[event] = event_list

            # Skip if a truememory command is already registered for this event
            already = any(
                isinstance(entry, dict)
                and _TRUEMEMORY_MARKER in entry.get("command", "").lower()
                for entry in event_list
            )
            if already:
                continue

            entry: dict = {"command": cmd}
            if info.get("timeout"):
                entry["timeout"] = info["timeout"]
            event_list.append(entry)

        _CONFIG.write_text(_yaml_safe_dump(existing), encoding="utf-8")

    def uninstall(self) -> None:
        self._remove_mcp_entry()
        self._remove_hook_entries()

    def verify(self) -> bool:
        return self._has_mcp_entry() and self._has_hook_entries()

    def get_system_prompt_path(self) -> Path | None:
        return Path.home() / ".hermes" / "truememory_prompt.md"

    def get_system_prompt_content(self) -> str:
        from truememory.hooks.adapters.base import get_generic_system_prompt
        return get_generic_system_prompt()

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
        if not _CONFIG.exists():
            return False
        try:
            data = _yaml_safe_load(_CONFIG.read_text(encoding="utf-8"))
            return "truememory" in data.get("mcp_servers", {})
        except OSError:
            return False

    def _has_hook_entries(self) -> bool:
        """Check if any truememory hook commands exist under hooks:."""
        if not _CONFIG.exists():
            return False
        try:
            data = _yaml_safe_load(_CONFIG.read_text(encoding="utf-8"))
            hooks = data.get("hooks", {})
            if not isinstance(hooks, dict):
                return False
            for _event, entries in hooks.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if (
                        isinstance(entry, dict)
                        and _TRUEMEMORY_MARKER in entry.get("command", "").lower()
                    ):
                        return True
            return False
        except OSError:
            return False

    def _remove_mcp_entry(self) -> None:
        if not _CONFIG.exists():
            return
        try:
            data = _yaml_safe_load(_CONFIG.read_text(encoding="utf-8"))
            servers = data.get("mcp_servers", {})
            if isinstance(servers, dict) and "truememory" in servers:
                del servers["truememory"]
                _CONFIG.write_text(_yaml_safe_dump(data), encoding="utf-8")
        except OSError:
            pass

    def _remove_hook_entries(self) -> None:
        """Remove all truememory entries from the hooks: config."""
        if not _CONFIG.exists():
            return
        try:
            data = _yaml_safe_load(_CONFIG.read_text(encoding="utf-8"))
            hooks = data.get("hooks", {})
            if not isinstance(hooks, dict):
                return
            changed = False
            for event in list(hooks.keys()):
                entries = hooks[event]
                if not isinstance(entries, list):
                    continue
                cleaned = [
                    entry for entry in entries
                    if not (
                        isinstance(entry, dict)
                        and _TRUEMEMORY_MARKER in entry.get("command", "").lower()
                    )
                ]
                if len(cleaned) != len(entries):
                    changed = True
                    if cleaned:
                        hooks[event] = cleaned
                    else:
                        del hooks[event]
            if changed:
                _CONFIG.write_text(_yaml_safe_dump(data), encoding="utf-8")
        except OSError:
            pass
