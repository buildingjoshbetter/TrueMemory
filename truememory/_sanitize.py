"""Shared sanitizer for content interpolated into injected prompt blocks.

TrueMemory injects directives and recalled memories into the agent at session
start and on each prompt, wrapped in ``<truememory-directives>`` /
``<truememory-context>`` / ``<truememory-recall>`` blocks. Any *content* placed
inside those blocks is attacker-influenced (a user — or a poisoned upstream
source — controls memory/directive text). Without neutralization, crafted
content can:

  * close its wrapper (``</truememory-context>``) and forge a sibling block,
  * open a foreign authoritative-looking block (``<system-directive>`` /
    ``<system-reminder>``) that an agent may treat as system framing.

Issue #638 (M-28) escaped the ``<truememory-*>`` wrapper tokens for the
*directive* block only. This module hoists that logic into one place and
broadens it to also neutralize ``<system…>`` framing, so every injection site
(directive block + the three memory-render blocks) shares one defense.

The escape is intentionally narrow — it rewrites only the leading ``<`` of the
specific framing-tag vocabulary, leaving ordinary angle brackets in memory text
(code snippets, ``<email>``, math) intact.
"""
import re

# Control / ANSI characters that have no place in injected text (G2-2).
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Framing tokens that must never be openable/closable from inside a block:
#   - truememory-*  : our own wrapper tokens (breakout / sibling-forge)
#   - system\b      : <system>, <system-reminder>, <system-directive>, etc.
#                     (the `\b` matches <system>, <system-foo>; NOT <systematic>)
_FRAMING_TOKEN_RE = re.compile(r"(?i)<(/?(?:truememory-|system\b))")


def sanitize_injection_content(content: str) -> str:
    """Neutralize *content* before interpolating it into an injected block.

    Strips control/ANSI chars, then escapes the leading ``<`` of any
    ``<truememory-…>`` or ``<system…>`` framing token (and their closing forms)
    so the content can neither break out of its wrapper nor forge an
    authoritative block. The result is inert, human-readable text.
    """
    if not content:
        return content
    content = CONTROL_CHARS_RE.sub("", content)
    content = _FRAMING_TOKEN_RE.sub(r"&lt;\1", content)
    return content
