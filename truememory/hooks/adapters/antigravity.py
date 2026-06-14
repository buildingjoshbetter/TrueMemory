"""Antigravity adapter -- MCP config for the Antigravity AI CLI.

Antigravity reads its MCP config from a JSON file named ``mcp_config.json``
and its lifecycle hooks from ``hooks.json``, both located in ``~/.gemini/config/``.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

from truememory.hooks.adapters.base import CLIAdapter, get_generic_system_prompt

_ANTIGRAVITY_DIR = Path.home() / ".gemini" / "config"
_MCP_CONFIG = _ANTIGRAVITY_DIR / "mcp_config.json"
_HOOKS_CONFIG = _ANTIGRAVITY_DIR / "hooks.json"

_TRUEMEMORY_MARKER = "truememory"


def _atomic_write(path: Path, text: str) -> None:
    """Write text to path atomically: tempfile in the same dir + os.replace."""
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except OSError:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _backup_config(config_path: Path) -> Path:
    """Copy an unparseable config aside before overwriting it."""
    original = config_path.read_bytes()
    backup = config_path.with_name(
        f"{config_path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        f"-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    fd = os.open(str(backup), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(original)
    return backup


class AntigravityAdapter(CLIAdapter):
    """Adapter for the Antigravity AI CLI."""

    @property
    def name(self) -> str:
        return "Antigravity"

    @property
    def cli_id(self) -> str:
        return "antigravity"

    @property
    def config_path(self) -> Path:
        return _MCP_CONFIG

    def detect(self) -> bool:
        return _ANTIGRAVITY_DIR.is_dir() or shutil.which("antigravity") is not None or shutil.which("agy") is not None

    def is_configured(self) -> bool:
        return self._has_mcp_entry()

    def install_mcp(self, python_path: str | None = None) -> None:
        py = python_path or sys.executable

        existing: dict = {}
        if _MCP_CONFIG.exists() and _MCP_CONFIG.stat().st_size > 0:
            try:
                data = json.loads(_MCP_CONFIG.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = None
            if isinstance(data, dict):
                existing = data
            else:
                try:
                    backup = _backup_config(_MCP_CONFIG)
                except OSError as e:
                    raise RuntimeError(
                        f"Existing {_MCP_CONFIG} is not valid JSON and could "
                        f"not be backed up ({e}); refusing to overwrite it."
                    ) from e
                print(
                    f"\033[33m⚠ Existing {_MCP_CONFIG} is not valid JSON; "
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

        _MCP_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(_MCP_CONFIG, json.dumps(existing, indent=2))

    def install_hooks(
        self,
        python_path: str | None = None,
        user_id: str = "",
        db_path: str = "",
    ) -> None:
        py = python_path or sys.executable
        import truememory

        hooks_dir = Path(truememory.__file__).parent / "ingest" / "hooks"

        existing: dict = {}
        if _HOOKS_CONFIG.exists() and _HOOKS_CONFIG.stat().st_size > 0:
            try:
                data = json.loads(_HOOKS_CONFIG.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    existing = data
            except (json.JSONDecodeError, OSError):
                pass

        hook_entries = [
            ("pre_invocation_hooks", "session_start.py", "truememory-sessionstart", 10000),
            ("pre_invocation_hooks", "user_prompt_submit.py", "truememory-beforeagent", 5000),
            ("post_invocation_hooks", "compact.py", "truememory-precompress", 5000),
            ("stop_hooks", "stop.py", "truememory-sessionend", 5000),
        ]
        
        # Remove version key if it exists, as JSONHookSpec does not support it
        existing.pop("version", None)
        # Remove enable_json_hooks if it exists, as it causes unmarshal bool errors
        existing.pop("enable_json_hooks", None)

        for event_name, script_name, hook_id, timeout in hook_entries:
            # Each event maps directly to a JSONHookSpec object in Cortex
            if event_name not in existing:
                existing[event_name] = {"hooks": []}
            elif not isinstance(existing[event_name], dict) or "hooks" not in existing[event_name]:
                existing[event_name] = {"hooks": []}
                
            hook_spec = existing[event_name]
            hooks_list = hook_spec["hooks"]
            
            # Remove existing TrueMemory hook for this event
            cleaned_list = [h for h in hooks_list if isinstance(h, dict) and "truememory" not in h.get("command", "").lower()]
            hooks_list[:] = cleaned_list
            
            cmd = f"{py} {hooks_dir / script_name}"
            hooks_list.append({
                "type": "command",
                "command": cmd,
                "name": hook_id,
                "timeout": timeout
            })

        _HOOKS_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(_HOOKS_CONFIG, json.dumps(existing, indent=2))

    def uninstall(self) -> None:
        # Remove MCP
        if _MCP_CONFIG.exists():
            try:
                data = json.loads(_MCP_CONFIG.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    servers = data.get("mcpServers", {})
                    if isinstance(servers, dict) and _TRUEMEMORY_MARKER in servers:
                        del servers[_TRUEMEMORY_MARKER]
                        _atomic_write(_MCP_CONFIG, json.dumps(data, indent=2))
            except (json.JSONDecodeError, OSError):
                pass
                
        # Remove Hooks
        if _HOOKS_CONFIG.exists():
            try:
                data = json.loads(_HOOKS_CONFIG.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "hooks" in data:
                    hooks_dict = data["hooks"]
                    for event_list in hooks_dict.values():
                        if isinstance(event_list, list):
                            event_list[:] = [
                                h for h in event_list
                                if not (isinstance(h, dict) and "truememory" in h.get("command", ""))
                            ]
                    _atomic_write(_HOOKS_CONFIG, json.dumps(data, indent=2))
            except (json.JSONDecodeError, OSError):
                pass

    def verify(self) -> bool:
        return self._has_mcp_entry()

    def get_system_prompt_path(self) -> Path | None:
        return Path.home() / ".gemini" / "antigravity-cli" / "skills" / "truememory" / "SKILL.md"

    def get_system_prompt_content(self) -> str:
        return get_generic_system_prompt(
            has_hooks=True,
            has_session_start=True,
        )

    def _has_mcp_entry(self) -> bool:
        if not _MCP_CONFIG.exists() or _MCP_CONFIG.stat().st_size == 0:
            return False
        try:
            data = json.loads(_MCP_CONFIG.read_text(encoding="utf-8"))
            return isinstance(data, dict) and _TRUEMEMORY_MARKER in data.get("mcpServers", {})
        except (json.JSONDecodeError, OSError):
            return False

