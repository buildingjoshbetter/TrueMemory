"""CLI detection, config path mapping, and install state tracking.

Discovers which CLI tools are installed and tracks which ones have
TrueMemory configured.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from truememory.hooks.adapters.base import CLIAdapter

log = logging.getLogger(__name__)

STATE_FILE = Path.home() / ".truememory" / "integrations.json"


def _get_all_adapters() -> list[CLIAdapter]:
    """Return instances of all known CLI adapters.

    Note: HermesAdapter and OpenClawAdapter were removed in June 2026
    because the target products (NousResearch/hermes-agent, openclaw/openclaw)
    do not exist. See the adapter module docstrings for details.
    """
    from truememory.hooks.adapters.claude import ClaudeAdapter
    from truememory.hooks.adapters.codex import CodexAdapter
    from truememory.hooks.adapters.cursor import CursorAdapter
    from truememory.hooks.adapters.gemini import GeminiAdapter
    from truememory.hooks.adapters.kimi import KimiAdapter
    return [ClaudeAdapter(), CodexAdapter(), CursorAdapter(), GeminiAdapter(), KimiAdapter()]


def detect_installed() -> list[CLIAdapter]:
    """Return adapters for CLIs that are installed on this system."""
    return [a for a in _get_all_adapters() if a.detect()]


def detect_configured() -> list[CLIAdapter]:
    """Return adapters for CLIs that have TrueMemory configured."""
    return [a for a in _get_all_adapters() if a.is_configured()]


def get_adapter(cli_id: str) -> CLIAdapter | None:
    """Get a specific adapter by its CLI identifier."""
    for a in _get_all_adapters():
        if a.cli_id == cli_id:
            return a
    return None


def load_state() -> dict:
    """Load the integration state file."""
    if not STATE_FILE.exists():
        return {"configured": [], "configured_at": {}, "version": ""}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"configured": [], "configured_at": {}, "version": ""}


def save_state(state: dict) -> None:
    """Save the integration state file."""
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(
            json.dumps(state, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        log.warning("Failed to save integration state: %s", e)


def mark_configured(cli_id: str) -> None:
    """Record that a CLI has been configured."""
    from truememory import __version__
    state = load_state()
    configured = state.get("configured", [])
    if cli_id not in configured:
        configured.append(cli_id)
    state["configured"] = configured
    state.setdefault("configured_at", {})[cli_id] = (
        datetime.now(timezone.utc).isoformat()
    )
    state["version"] = __version__
    save_state(state)


def mark_unconfigured(cli_id: str) -> None:
    """Record that a CLI has been unconfigured."""
    state = load_state()
    configured = state.get("configured", [])
    if cli_id in configured:
        configured.remove(cli_id)
    state["configured"] = configured
    state.get("configured_at", {}).pop(cli_id, None)
    save_state(state)
