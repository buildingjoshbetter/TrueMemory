"""CLI entry points for multi-CLI hook setup.

Provides functions that the main `truememory-ingest setup` command can
delegate to for per-CLI hook installation and management.
"""
from __future__ import annotations

import logging
from pathlib import Path

from truememory.hooks.adapters.base import CLIAdapter
from truememory.hooks.registry import (
    detect_installed,
    get_adapter,
    mark_configured,
    mark_unconfigured,
)

log = logging.getLogger(__name__)


def install_cli(
    cli_id: str,
    python_path: str | None = None,
    user_id: str = "",
    db_path: str = "",
) -> bool:
    """Install TrueMemory hooks and MCP server for a specific CLI.

    Returns True on success, False if the CLI is unknown or install fails.
    """
    adapter = get_adapter(cli_id)
    if adapter is None:
        log.error("Unknown CLI: %s", cli_id)
        return False

    try:
        adapter.install_hooks(
            python_path=python_path,
            user_id=user_id,
            db_path=db_path,
        )
        adapter.install_mcp(python_path=python_path)
        mark_configured(cli_id)
        return True
    except Exception as e:
        log.error("Failed to install for %s: %s", cli_id, e)
        return False


def uninstall_cli(cli_id: str) -> bool:
    """Remove TrueMemory from a specific CLI's config.

    Returns True on success, False if the CLI is unknown or uninstall fails.
    """
    adapter = get_adapter(cli_id)
    if adapter is None:
        log.error("Unknown CLI: %s", cli_id)
        return False

    try:
        adapter.uninstall()
        mark_unconfigured(cli_id)
        return True
    except Exception as e:
        log.error("Failed to uninstall from %s: %s", cli_id, e)
        return False


def verify_cli(cli_id: str) -> bool:
    """Verify TrueMemory installation for a specific CLI."""
    adapter = get_adapter(cli_id)
    if adapter is None:
        return False
    return adapter.verify()
