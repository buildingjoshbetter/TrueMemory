"""OpenClaw adapter — JSON config + JS plugin system.

OpenClaw uses:
- ~/.openclaw/openclaw.json (JSON5-compatible) for MCP server registration
  under ``mcp.servers`` (nested, NOT top-level ``mcpServers``).
  The runtime reads ``sourceConfig.mcp?.servers`` in mcp-config.ts and all
  write-back functions (setConfiguredMcpServer, etc.) write to
  ``next.mcp = { ...next.mcp, servers }``.
- ~/.openclaw/plugins/<name>/ for plugins (openclaw.plugin.json + index.js)
  using ``definePluginEntry()`` from ``openclaw/plugin-sdk/plugin-entry``
- Plugin lifecycle hooks: session_start, session_end, before_tool_call,
  before_compaction (registered via ``api.on("event", handler)`` inside
  ``register(api)`` callback)
- Container-based skills can be enabled/disabled in the ``skills:`` section
  of the main config

Config format (from OpenClaw source, mcp-config.ts + configuration-reference.md):
  {
    "mcp": {
      "servers": {
        "name": {"command": "...", "args": [...]}
      }
    }
  }
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

from truememory.hooks.adapters.base import CLIAdapter

log = logging.getLogger(__name__)

_OPENCLAW_DIR = Path.home() / ".openclaw"
_CONFIG_PATH = _OPENCLAW_DIR / "openclaw.json"
_PLUGINS_DIR = _OPENCLAW_DIR / "plugins"
_PLUGIN_NAME = "truememory"

_TEMPLATE_DIR = Path(__file__).parent.parent / "templates" / "openclaw"


def _read_json_config(path: Path) -> dict:
    """Read a JSON config file, tolerating minor JSON5 features."""
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
        return json.loads(text)
    except json.JSONDecodeError:
        stripped = _strip_json5_comments(text)
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            log.warning("Cannot parse %s as JSON — skipping", path)
            return {}
    except OSError:
        return {}


def _strip_json5_comments(text: str) -> str:
    """Best-effort removal of single-line // comments and trailing commas.

    Uses a state-aware parser — ``//`` and trailing commas inside
    ``"..."`` are preserved.
    """
    result: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == '"':
            j = i + 1
            while j < len(text):
                if text[j] == '\\':
                    j += 2
                    continue
                if text[j] == '"':
                    j += 1
                    break
                j += 1
            result.append(text[i:j])
            i = j
        elif text[i:i+2] == '//':
            nl = text.find('\n', i)
            i = nl if nl != -1 else len(text)
        elif text[i] == ',':
            j = i + 1
            while j < len(text):
                if text[j] in ' \t\n\r':
                    j += 1
                elif text[j:j+2] == '//':
                    nl = text.find('\n', j)
                    j = nl + 1 if nl != -1 else len(text)
                else:
                    break
            if j < len(text) and text[j] in '}]':
                i = j
            else:
                result.append(text[i])
                i += 1
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)


class OpenClawAdapter(CLIAdapter):
    """Adapter for OpenClaw agent gateway."""

    @property
    def name(self) -> str:
        return "OpenClaw"

    @property
    def cli_id(self) -> str:
        return "openclaw"

    @property
    def config_path(self) -> Path:
        return _CONFIG_PATH

    def detect(self) -> bool:
        return _OPENCLAW_DIR.is_dir() or shutil.which("openclaw") is not None

    def is_configured(self) -> bool:
        return self._has_mcp_entry() or self._has_plugin()

    def install_mcp(self, python_path: str | None = None) -> None:
        py = python_path or sys.executable
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

        existing = _read_json_config(_CONFIG_PATH)

        # OpenClaw reads MCP config from mcp.servers (nested).
        # The runtime reads ``sourceConfig.mcp?.servers`` (mcp-config.ts)
        # and all write-back functions use ``next.mcp = { ...next.mcp, servers }``.
        mcp = existing.setdefault("mcp", {})
        if not isinstance(mcp, dict):
            mcp = {}
            existing["mcp"] = mcp
        servers = mcp.setdefault("servers", {})
        if not isinstance(servers, dict):
            servers = {}
            mcp["servers"] = servers

        servers[_PLUGIN_NAME] = {
            "command": py,
            "args": ["-m", "truememory.mcp_server"],
        }

        # Migration: remove stale entries from incorrect top-level mcpServers
        # (written by earlier buggy versions of this adapter)
        old_top = existing.get("mcpServers", {})
        if isinstance(old_top, dict) and _PLUGIN_NAME in old_top:
            del old_top[_PLUGIN_NAME]
            if not old_top:
                del existing["mcpServers"]

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
        plugin_dir = _PLUGINS_DIR / _PLUGIN_NAME
        plugin_dir.mkdir(parents=True, exist_ok=True)

        for template_file in ("openclaw.plugin.json", "package.json", "index.js"):
            src = _TEMPLATE_DIR / template_file
            dst = plugin_dir / template_file
            if src.exists():
                content = src.read_text(encoding="utf-8")
                if python_path:
                    content = content.replace(
                        'process.env.TRUEMEMORY_PYTHON || "python3"',
                        f'process.env.TRUEMEMORY_PYTHON || "{python_path}"',
                    )
                dst.write_text(content, encoding="utf-8")

    def uninstall(self) -> None:
        self._remove_mcp_entry()
        self._remove_plugin()

    def verify(self) -> bool:
        return self._has_mcp_entry() and self._has_plugin()

    def get_system_prompt_path(self) -> Path | None:
        return _OPENCLAW_DIR / "truememory_prompt.md"

    def get_system_prompt_content(self) -> str:
        from truememory.hooks.adapters.base import get_generic_system_prompt
        return get_generic_system_prompt()

    # -- Private helpers --

    def _has_mcp_entry(self) -> bool:
        data = _read_json_config(_CONFIG_PATH)
        # Check correct mcp.servers path
        mcp = data.get("mcp", {})
        if isinstance(mcp, dict):
            servers = mcp.get("servers", {})
            if isinstance(servers, dict) and _PLUGIN_NAME in servers:
                return True
        # Also check legacy top-level mcpServers for detection
        # (written by earlier buggy versions of this adapter)
        servers = data.get("mcpServers", {})
        if isinstance(servers, dict) and _PLUGIN_NAME in servers:
            return True
        return False

    def _has_plugin(self) -> bool:
        plugin_dir = _PLUGINS_DIR / _PLUGIN_NAME
        return (
            plugin_dir.is_dir()
            and (plugin_dir / "openclaw.plugin.json").exists()
            and (plugin_dir / "index.js").exists()
        )

    def _remove_mcp_entry(self) -> None:
        if not _CONFIG_PATH.exists():
            return
        try:
            data = _read_json_config(_CONFIG_PATH)
            changed = False

            # Remove from correct mcp.servers path
            mcp = data.get("mcp", {})
            if isinstance(mcp, dict):
                servers = mcp.get("servers", {})
                if isinstance(servers, dict) and _PLUGIN_NAME in servers:
                    del servers[_PLUGIN_NAME]
                    changed = True

            # Also clean up legacy top-level mcpServers
            old_top = data.get("mcpServers", {})
            if isinstance(old_top, dict) and _PLUGIN_NAME in old_top:
                del old_top[_PLUGIN_NAME]
                changed = True

            if changed:
                _CONFIG_PATH.write_text(
                    json.dumps(data, indent=2), encoding="utf-8",
                )
        except OSError:
            pass

    def _remove_plugin(self) -> None:
        plugin_dir = _PLUGINS_DIR / _PLUGIN_NAME
        if plugin_dir.is_dir():
            try:
                shutil.rmtree(plugin_dir)
            except OSError as e:
                log.warning("Failed to remove plugin dir %s: %s", plugin_dir, e)
