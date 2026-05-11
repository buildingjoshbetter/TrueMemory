"""Abstract base class for CLI adapters.

Each supported CLI (Claude Code, Kimi, Hermes, OpenClaw) implements
this interface to handle its config format, hook registration, and
MCP server setup.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class CLIAdapter(ABC):
    """Base interface for CLI-specific TrueMemory integration."""

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
