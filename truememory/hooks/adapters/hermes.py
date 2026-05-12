"""Hermes Agent adapter — YAML config + plugin/gateway hooks.

Hermes Agent uses:
- ~/.hermes/config.yaml for MCP server registration (mcp_servers: key)
- ~/.hermes/cli-config.yaml for plugin hooks (plugins: key, CLI mode)
- ~/.hermes/hooks/<name>/ for gateway hooks (Telegram/Discord/etc.)
- Same JSON stdin/stdout hook protocol as Claude Code for plugin hooks
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
_MCP_CONFIG = _HERMES_DIR / "config.yaml"
_CLI_CONFIG = _HERMES_DIR / "cli-config.yaml"
_GATEWAY_HOOKS_DIR = _HERMES_DIR / "hooks"

_PLUGIN_HOOKS = {
    "on_session_start": {
        "name": "truememory-session-start",
        "script": "session_start.py",
    },
    "on_session_end": {
        "name": "truememory-session-end",
        "script": "stop.py",
    },
    "on_user_prompt": {
        "name": "truememory-user-prompt",
        "script": "user_prompt_submit.py",
    },
    "on_pre_compact": {
        "name": "truememory-pre-compact",
        "script": "compact.py",
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
        return _MCP_CONFIG

    def detect(self) -> bool:
        return _HERMES_DIR.is_dir() or shutil.which("hermes") is not None

    def is_configured(self) -> bool:
        return self._has_mcp_entry() or self._has_plugin_entries()

    def install_mcp(self, python_path: str | None = None) -> None:
        py = python_path or sys.executable
        _MCP_CONFIG.parent.mkdir(parents=True, exist_ok=True)

        existing: dict = {}
        if _MCP_CONFIG.exists():
            try:
                existing = _yaml_safe_load(_MCP_CONFIG.read_text(encoding="utf-8"))
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

        _MCP_CONFIG.write_text(_yaml_safe_dump(existing), encoding="utf-8")

    def install_hooks(
        self,
        python_path: str | None = None,
        user_id: str = "",
        db_path: str = "",
    ) -> None:
        py = python_path or sys.executable
        hooks_dir = Path(__file__).parent.parent.parent / "ingest" / "hooks"

        _CLI_CONFIG.parent.mkdir(parents=True, exist_ok=True)

        existing: dict = {}
        if _CLI_CONFIG.exists():
            try:
                existing = _yaml_safe_load(_CLI_CONFIG.read_text(encoding="utf-8"))
            except OSError:
                existing = {}

        plugins = existing.setdefault("plugins", [])
        if not isinstance(plugins, list):
            plugins = []
            existing["plugins"] = plugins

        existing_names = {
            p.get("name", "") for p in plugins if isinstance(p, dict)
        }

        for event, info in _PLUGIN_HOOKS.items():
            if info["name"] in existing_names:
                continue

            script_path = hooks_dir / info["script"]
            cmd = self._build_command(py, script_path, user_id, db_path)

            plugins.append({
                "name": info["name"],
                "event": event,
                "command": cmd,
            })

        _CLI_CONFIG.write_text(_yaml_safe_dump(existing), encoding="utf-8")

    def uninstall(self) -> None:
        self._remove_mcp_entry()
        self._remove_plugin_entries()

    def verify(self) -> bool:
        return self._has_mcp_entry() and self._has_plugin_entries()

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
        if not _MCP_CONFIG.exists():
            return False
        try:
            data = _yaml_safe_load(_MCP_CONFIG.read_text(encoding="utf-8"))
            return "truememory" in data.get("mcp_servers", {})
        except OSError:
            return False

    def _has_plugin_entries(self) -> bool:
        if not _CLI_CONFIG.exists():
            return False
        try:
            data = _yaml_safe_load(_CLI_CONFIG.read_text(encoding="utf-8"))
            plugins = data.get("plugins", [])
            if not isinstance(plugins, list):
                return False
            return any(
                isinstance(p, dict)
                and _TRUEMEMORY_MARKER in p.get("name", "").lower()
                for p in plugins
            )
        except OSError:
            return False

    def _remove_mcp_entry(self) -> None:
        if not _MCP_CONFIG.exists():
            return
        try:
            data = _yaml_safe_load(_MCP_CONFIG.read_text(encoding="utf-8"))
            servers = data.get("mcp_servers", {})
            if isinstance(servers, dict) and "truememory" in servers:
                del servers["truememory"]
                _MCP_CONFIG.write_text(_yaml_safe_dump(data), encoding="utf-8")
        except OSError:
            pass

    def _remove_plugin_entries(self) -> None:
        if not _CLI_CONFIG.exists():
            return
        try:
            data = _yaml_safe_load(_CLI_CONFIG.read_text(encoding="utf-8"))
            plugins = data.get("plugins", [])
            if not isinstance(plugins, list):
                return
            cleaned = [
                p for p in plugins
                if not (
                    isinstance(p, dict)
                    and _TRUEMEMORY_MARKER in p.get("name", "").lower()
                )
            ]
            data["plugins"] = cleaned
            _CLI_CONFIG.write_text(_yaml_safe_dump(data), encoding="utf-8")
        except OSError:
            pass
