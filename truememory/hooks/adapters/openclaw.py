"""OpenClaw adapter -- REMOVED.

This adapter was removed because the target product ("openclaw/openclaw")
does not exist as described. The name "OpenClaw" refers to an unrelated
reimplementation of the 1997 game Captain Claw, not an AI agent gateway.
The claimed 377,304 GitHub stars would place it in the top 3 repositories
on all of GitHub. The config schema (JSON with ``mcp.servers`` key, JS plugin
system with ``before_agent_run``/``before_compress`` events) could not be
verified against any real product and appears to have been derived from
Claude Code's hook system with cosmetic renaming.

Removed in June 2026 after 7-model cross-lab verification (Anthropic, OpenAI,
DeepSeek, Alibaba/Qwen, Google) unanimously confirmed the product is
non-existent.

If a real "OpenClaw" AI agent platform ships in the future, a new adapter can
be written from scratch against the actual config format.
"""
from __future__ import annotations

_REMOVED_REASON = (
    "The OpenClaw adapter was removed because the target product does not exist. "
    "'OpenClaw' refers to a Captain Claw game reimplementation, not an AI agent."
)


# Keep the JSON5 comment stripper -- it is a useful, well-tested utility
# that other code may import.

def _strip_json5_comments(text: str) -> str:
    """Best-effort removal of single-line // comments and trailing commas.

    Uses a state-aware parser -- ``//`` and trailing commas inside
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


class OpenClawAdapter:
    """Stub for the removed OpenClaw adapter.

    Raises ``NotImplementedError`` on any method call to prevent silent
    misconfiguration. Kept as a stub so that existing code importing
    ``OpenClawAdapter`` gets a clear error rather than an ``ImportError``.
    """

    @property
    def name(self) -> str:
        return "OpenClaw (REMOVED)"

    @property
    def cli_id(self) -> str:
        return "openclaw"

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
