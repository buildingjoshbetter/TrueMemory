"""Hermes Agent adapter -- REMOVED.

This adapter was removed because the target product ("NousResearch/hermes-agent")
does not exist. NousResearch produces Hermes LLM fine-tunes (Hermes 2, Hermes 3,
OpenHermes), not a CLI agent framework. The config schema (YAML with
``mcp_servers`` and ``plugins`` keys, event names ``on_session_start``,
``on_pre_compact``, etc.) could not be verified against any real product and
appears to have been derived from Claude Code's hook system with cosmetic
renaming.

Removed in June 2026 after 7-model cross-lab verification (Anthropic, OpenAI,
DeepSeek, Alibaba/Qwen, Google) unanimously confirmed the product is
non-existent.

If NousResearch or another party ships a real Hermes Agent CLI in the future,
a new adapter can be written from scratch against the actual config format.
"""
from __future__ import annotations

_REMOVED_REASON = (
    "The Hermes Agent adapter was removed because the target product does not exist. "
    "NousResearch makes Hermes LLM models, not a CLI agent framework."
)


class HermesAdapter:
    """Stub for the removed Hermes adapter.

    Raises ``NotImplementedError`` on any method call to prevent silent
    misconfiguration. Kept as a stub so that existing code importing
    ``HermesAdapter`` gets a clear error rather than an ``ImportError``.
    """

    @property
    def name(self) -> str:
        return "Hermes Agent (REMOVED)"

    @property
    def cli_id(self) -> str:
        return "hermes"

    def detect(self) -> bool:
        return False

    def is_configured(self) -> bool:
        return False

    def install_mcp(self, python_path: str | None = None) -> None:
        raise NotImplementedError(_REMOVED_REASON)

    def install_hooks(self, **kwargs) -> None:  # type: ignore[override]
        raise NotImplementedError(_REMOVED_REASON)

    def uninstall(self) -> None:
        raise NotImplementedError(_REMOVED_REASON)

    def verify(self) -> bool:
        return False

    def get_system_prompt_path(self):  # type: ignore[return]
        return None

    def get_system_prompt_content(self) -> str:
        return ""
