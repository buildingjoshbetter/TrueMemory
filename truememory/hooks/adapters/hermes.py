"""Hermes Agent adapter — YAML config for MCP + shell hooks.

Hermes Agent (Nous Research) uses:
- ~/.hermes/config.yaml for both MCP server registration (mcp_servers: key)
  and shell hook registration (hooks: key)
- Shell hooks are list-of-dicts under each event name with command/matcher/timeout
- Native session_id plus extra content on stdin; top-level context on stdout
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
import re
import shlex
import shutil
import sys
from pathlib import Path

from truememory.hooks.adapters.base import CLIAdapter, atomic_write_text

log = logging.getLogger(__name__)

_HERMES_DIR = Path.home() / ".hermes"
_CONFIG = _HERMES_DIR / "config.yaml"

# Pinned Hermes main and v0.21.6 emit pre/post once per user turn.
# The fixed wrapper translates native content without transcript admission.
_HOOK_ENTRIES = {
    "on_session_start": {
        "script": "hermes.py",
        "timeout": 30,
    },
    "on_session_end": {
        "script": "hermes.py",
        "timeout": 30,
    },
    "pre_llm_call": {
        "script": "hermes.py",
        "timeout": 10,
    },
    "post_llm_call": {
        "script": "hermes.py",
        "timeout": 30,
    },
    "on_session_finalize": {
        "script": "hermes.py",
        "timeout": 15,
    },
}

_LEGACY_SCRIPTS = {"on_session_start": "session_start.py", "on_session_end": "stop.py",
                   "pre_llm_call": "user_prompt_submit.py", "on_session_finalize": "compact.py"}


def _owned_command(command: object, event: str) -> bool:
    """Only replace a simple Python invocation of this installed event hook."""
    if not isinstance(command, str) or any(char in command for char in ("\0", "\n", "\r")):
        return False
    try:
        lexer = shlex.shlex(command, posix=False, punctuation_chars=";&|<>()`")
        lexer.whitespace_split = True
        for token in lexer:
            if token and all(char in ";&|<>()`" for char in token):
                return False
            if not token.startswith("'") and any(part in token for part in ("$(", "${", "`")):
                return False
        tokens = shlex.split(command, comments=False, posix=sys.platform != "win32")
    except ValueError:
        return False
    if sys.platform == "win32":
        tokens = [token[1:-1] if token.startswith('"') and token.endswith('"') else token
                  for token in tokens]
    if len(tokens) < 2 or re.fullmatch(r"python(?:\d+(?:\.\d+)*)?(?:\.exe)?",
                                     tokens[0].replace("\\", "/").rsplit("/", 1)[-1], re.I) is None:
        return False
    index = 1
    while index < len(tokens) and tokens[index] in {"-B", "-u", "-E", "-I", "-s", "-S", "-O", "-OO"}:
        index += 1
    if index >= len(tokens):
        return False
    script = tokens[index]
    index += 1
    if script == "-m":
        if index >= len(tokens):
            return False
        module = tokens[index]
        index += 1
        prefix = "truememory.ingest.hooks."
        script_name = module.removeprefix(prefix) + ".py" if module.startswith(prefix) else ""
    else:
        hook_dir = Path(__file__).parent.parent.parent / "ingest" / "hooks"
        candidates = {str(hook_dir / name): name for name in
                      {"hermes.py", *_LEGACY_SCRIPTS.values()}}
        script_name = candidates.get(script, "")
    native = script_name == "hermes.py"
    if not native and script_name != _LEGACY_SCRIPTS.get(event):
        return False
    flags: dict[str, str] = {}
    rest = tokens[index:]
    if len(rest) % 2:
        return False
    for flag, value in zip(rest[::2], rest[1::2]):
        if flag not in {"--user", "--db", "--host", "--event"} or flag in flags:
            return False
        flags[flag] = value
    if native:
        return event in _HOOK_ENTRIES and flags.get("--event") == event and "--host" not in flags
    return flags.get("--host", "claude") == "claude" and "--event" not in flags


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
    def has_hooks(self) -> bool:
        return True

    @property
    def has_session_start(self) -> bool:
        # Hermes discards observer output; first-turn recall uses pre_llm_call.
        return False

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

        atomic_write_text(_CONFIG, _yaml_safe_dump(existing))

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
            cmd = self._build_command(py, script_path, user_id, db_path, event=event)

            # Ensure the event key is a list
            event_list = hooks.setdefault(event, [])
            if not isinstance(event_list, list):
                raise ValueError(f"Hermes {event} hooks must be a list")
            event_list[:] = [entry for entry in event_list
                             if not (isinstance(entry, dict) and _owned_command(entry.get("command"), event))]

            entry: dict = {"command": cmd}
            if info.get("timeout"):
                entry["timeout"] = info["timeout"]
            event_list.append(entry)

        atomic_write_text(_CONFIG, _yaml_safe_dump(existing))

    def uninstall(self) -> None:
        self._remove_mcp_entry()
        self._remove_hook_entries()

    def verify(self) -> bool:
        return self._has_mcp_entry() and self._has_hook_entries()

    def get_system_prompt_path(self) -> Path | None:
        return Path.home() / ".hermes" / "truememory_prompt.md"

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
        *,
        event: str = "",
    ) -> str:
        parts: list[str] = [python_path, str(script_path)]
        if event:
            parts.extend(["--event", event])
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
                        and _owned_command(entry.get("command"), _event)
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
                atomic_write_text(_CONFIG, _yaml_safe_dump(data))
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
                        and _owned_command(entry.get("command"), event)
                    )
                ]
                if len(cleaned) != len(entries):
                    changed = True
                    if cleaned:
                        hooks[event] = cleaned
                    else:
                        del hooks[event]
            if changed:
                atomic_write_text(_CONFIG, _yaml_safe_dump(data))
        except OSError:
            pass
