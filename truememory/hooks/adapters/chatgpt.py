"""ChatGPT Desktop adapter - local MCP config (EXPERIMENTAL).

IMPORTANT: ChatGPT Desktop does not currently load local MCP servers.
As of mid-2026, OpenAI supports only remote HTTPS MCP connectors, enabled
via developer mode in the ChatGPT *web* UI — and the macOS desktop app
cannot enable developer mode at all. There is no ChatGPT Desktop for Linux.

This adapter pre-stages a local stdio MCP config for when/if OpenAI enables
local MCP support in the desktop app. It refuses to run (and refuses to
claim success) unless the actual ChatGPT Desktop app is installed, and it
prints an explicit experimental warning whenever it writes config.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

from truememory.hooks.adapters.base import CLIAdapter

if sys.platform == "darwin":
    _CHATGPT_DIR = Path.home() / "Library" / "Application Support" / "com.openai.chat"
    _CONFIG_PATH = _CHATGPT_DIR / "mcp.json"
elif sys.platform == "win32":
    _base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    _CHATGPT_DIR = Path(_base) / "com.openai.chat" if _base else Path.home() / "AppData" / "Local" / "com.openai.chat"
    _CONFIG_PATH = _CHATGPT_DIR / "mcp.json"
else:
    # No ChatGPT Desktop exists for Linux; placeholder so the module imports.
    _CHATGPT_DIR = Path.home() / ".config" / "com.openai.chat"
    _CONFIG_PATH = _CHATGPT_DIR / "mcp.json"

_TRUEMEMORY_MARKER = "truememory"

_EXPERIMENTAL_WARNING = """\
\033[33m⚠ EXPERIMENTAL: ChatGPT Desktop does not currently load local MCP servers.
  OpenAI supports only remote HTTPS MCP connectors, enabled via developer mode
  in the ChatGPT web UI; the macOS desktop app cannot enable it (as of mid-2026).
  This adapter pre-stages a local MCP config for when/if OpenAI enables local
  MCP support. TrueMemory will NOT appear in ChatGPT until then.\033[0m"""

_APP_NOT_FOUND_MESSAGE = (
    "ChatGPT Desktop app not found — refusing to write config for an app that "
    "is not installed. Note: ChatGPT Desktop does not currently load local MCP "
    "servers anyway (OpenAI supports only remote HTTPS connectors via developer "
    "mode on the web UI). See docs/setup-chatgpt.md for details and manual setup."
)


def _app_installed() -> bool:
    """Return True only if the actual ChatGPT Desktop app is present.

    The Application Support / config directory is NOT sufficient evidence:
    our own install would create it, making detection self-fulfilling.
    """
    if sys.platform == "darwin":
        candidates = (
            Path("/Applications/ChatGPT.app"),
            Path.home() / "Applications" / "ChatGPT.app",
        )
        return any(p.exists() for p in candidates)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            return False
        packages = Path(base) / "Packages"
        try:
            return any(packages.glob("OpenAI.ChatGPT*"))
        except OSError:
            return False
    # No ChatGPT Desktop for Linux.
    return False


def _atomic_write(path: Path, text: str) -> None:
    """Write text to path atomically: tempfile in the same dir + os.replace.

    A crash mid-write must never leave a truncated config behind — the old
    file stays intact until the replace, which is atomic on POSIX and Windows.
    """
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
    """Copy an unparseable config aside before overwriting it.

    The backup name carries timestamp + pid + random suffix and is opened
    with O_EXCL (exclusive create), so concurrent installs can never
    overwrite each other's backups. Raises OSError on any failure — the
    caller must refuse to overwrite the original in that case.
    """
    original = config_path.read_bytes()
    backup = config_path.with_name(
        f"{config_path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        f"-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    fd = os.open(str(backup), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(original)
    return backup


class ChatGPTAdapter(CLIAdapter):
    """Adapter for ChatGPT Desktop MCP servers (experimental, forward-looking)."""

    @property
    def name(self) -> str:
        return "ChatGPT Desktop"

    @property
    def cli_id(self) -> str:
        return "chatgpt"

    @property
    def config_path(self) -> Path:
        return _CONFIG_PATH

    def detect(self) -> bool:
        return _app_installed()

    def is_configured(self) -> bool:
        return self._has_mcp_entry()

    def install_mcp(self, python_path: str | None = None) -> None:
        # Order matters: refuse (app absent) BEFORE any side effect — no
        # backup files, no directories, no writes on any refuse path.
        if not _app_installed():
            raise RuntimeError(_APP_NOT_FOUND_MESSAGE)

        config_path = _CONFIG_PATH
        py = python_path or sys.executable

        existing: dict = {}
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = None
            if isinstance(data, dict):
                existing = data
            else:
                # Unparseable (or non-object) config: never silently destroy
                # it. Back it up first; if the backup fails, refuse.
                try:
                    backup = _backup_config(config_path)
                except OSError as e:
                    raise RuntimeError(
                        f"Existing {config_path} is not valid JSON and could "
                        f"not be backed up ({e}); refusing to overwrite it."
                    ) from e
                print(
                    f"\033[33m⚠ Existing {config_path} is not valid JSON; "
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

        config_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(config_path, json.dumps(existing, indent=2))
        print(_EXPERIMENTAL_WARNING, file=sys.stderr)

    def install_hooks(
        self,
        python_path: str | None = None,
        user_id: str = "",
        db_path: str = "",
    ) -> None:
        # ChatGPT Desktop exposes MCP tools, but not TrueMemory lifecycle hooks.
        del python_path, user_id, db_path

    def uninstall(self) -> None:
        # Never write unless the file parsed cleanly AND our entry is
        # actually present. A corrupt config must be left untouched (it may
        # be user-recoverable) — but never silently: warn before no-op'ing.
        config_path = _CONFIG_PATH
        if not config_path.exists():
            return
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(
                f"\033[33m⚠ {config_path} could not be parsed ({e}); "
                f"leaving it untouched — remove the truememory entry "
                f"manually if present.\033[0m",
                file=sys.stderr,
            )
            return
        if not isinstance(data, dict):
            print(
                f"\033[33m⚠ {config_path} is not a JSON object; "
                f"leaving it untouched.\033[0m",
                file=sys.stderr,
            )
            return
        servers = data.get("mcpServers", {})
        if isinstance(servers, dict) and _TRUEMEMORY_MARKER in servers:
            del servers[_TRUEMEMORY_MARKER]
            _atomic_write(config_path, json.dumps(data, indent=2))

    def verify(self) -> bool:
        return self._has_mcp_entry()

    def get_system_prompt_path(self) -> Path | None:
        return None

    def get_system_prompt_content(self) -> str:
        return ""

    def _has_mcp_entry(self) -> bool:
        data = self._read_config()
        servers = data.get("mcpServers", {})
        return isinstance(servers, dict) and _TRUEMEMORY_MARKER in servers

    @staticmethod
    def _read_config() -> dict:
        if not _CONFIG_PATH.exists():
            return {}
        try:
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return data if isinstance(data, dict) else {}
