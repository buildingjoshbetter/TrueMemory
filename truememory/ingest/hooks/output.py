"""Event-specific context output for installed recall hooks."""
from __future__ import annotations

import json
import logging
import sys
from typing import TextIO

log = logging.getLogger(__name__)

HOOK_HOSTS = ("claude", "cursor", "gemini", "codex")
_CONTEXT_EVENTS = {
    "claude": frozenset({"SessionStart", "UserPromptSubmit"}),
    "cursor": frozenset({"sessionStart"}),
    "gemini": frozenset({"SessionStart", "BeforeAgent"}),
    "codex": frozenset({"SessionStart", "UserPromptSubmit"}),
}


def supports_context(host: str, event: str) -> bool:
    """Use trusted installer flags, never host identity supplied on stdin."""
    return event in _CONTEXT_EVENTS.get(host, ())


def emit_context(
    context: str, *, host: str, event: str, stream: TextIO | None = None,
) -> bool:
    """Return success only after the supported JSON response is flushed.

    This confirms transport emission, not native host/model acceptance.
    Cursor beforeSubmitPrompt has no documented context output contract.
    """
    if not isinstance(context, str) or not context or not supports_context(host, event):
        return False

    output: dict[str, object]
    if host == "claude":
        output = {"additionalContext": context}
    elif host == "cursor":
        output = {"additional_context": context}
    else:
        output = {
            "hookSpecificOutput": {
                "hookEventName": event,
                "additionalContext": context,
            },
        }

    target = stream if stream is not None else sys.stdout
    try:
        response = json.dumps(output) + "\n"
        if target.write(response) != len(response):
            return False
        target.flush()
    except (OSError, ValueError, TypeError):
        log.warning("Recall context emission failed for host=%s event=%s", host, event)
        return False
    return True
