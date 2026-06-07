"""Hermes Agent adapter -- REMOVED.

This adapter was removed because the target product ("NousResearch/hermes-agent")
does not exist. NousResearch produces Hermes LLM fine-tunes (Hermes 2, Hermes 3,
OpenHermes), not a CLI agent framework. The config schema (YAML with
``mcp_servers`` and ``plugins`` keys, event names ``on_session_start``,
``on_pre_compact``, etc.) could not be verified against any real product and
appears to have been derived from Claude Code's hook system with cosmetic
renaming.

Removed in June 2026 after cross-lab verification confirmed no such product
exists in NousResearch's public releases or documentation.

If NousResearch or another party ships a real Hermes Agent CLI in the future,
a new adapter can be written from scratch against the actual config format.
"""
from __future__ import annotations

from pathlib import Path

from truememory.hooks.adapters.base import CLIAdapter

_REMOVED_REASON = (
    "The Hermes Agent adapter was removed because the target product does not exist. "
    "NousResearch makes Hermes LLM models, not a CLI agent framework."
)


class HermesAdapter(CLIAdapter):
    """Stub for the removed Hermes adapter.

    Kept as a subclass of ``CLIAdapter`` for type-compatibility.
    Read-only / detection methods return safe defaults (``False`` / ``None``).
    Installation and uninstallation methods raise ``NotImplementedError``
    to prevent silent misconfiguration.
    """

    @property
    def name(self) -> str:
        return "Hermes Agent (REMOVED)"

    @property
    def cli_id(self) -> str:
        return "hermes"

    @property
    def config_path(self) -> Path:
        return Path.home() / ".hermes" / "config.yaml"

    def detect(self) -> bool:
        return False

    def is_configured(self) -> bool:
        return False

    def install_mcp(self, python_path: str | None = None) -> None:
        raise NotImplementedError(_REMOVED_REASON)

    def install_hooks(
        self,
        python_path: str | None = None,
        user_id: str = "",
        db_path: str = "",
    ) -> None:
        raise NotImplementedError(_REMOVED_REASON)

    def uninstall(self) -> None:
        raise NotImplementedError(_REMOVED_REASON)

    def verify(self) -> bool:
        return False

    def get_system_prompt_path(self) -> Path | None:
        return None

    def get_system_prompt_content(self) -> str:
        return ""
