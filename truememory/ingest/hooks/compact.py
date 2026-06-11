#!/usr/bin/env python3
"""
Compact Hook — Pre-Compression Snapshot
=========================================

Fires before Claude Code compresses its context window. This is a
critical moment — information from earlier in the conversation is
about to be lost. We capture a summary snapshot and store it as a
memory so important context survives compression.

This is analogous to the brain's "rehearsal" mechanism — replaying
important information to strengthen encoding before it fades from
working memory.

Input (stdin JSON):
    {"session_id": "...", "transcript_path": "..."}

Output: None (stores snapshot directly)
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


# M-90 transcript allowlist now lives in truememory.ingest.hooks._shared
# (hoisted so stop.py and the SessionStart backlog drain share the same guard).
# These thin shims keep the original names working for callers/tests.

def _transcript_roots() -> list[Path]:
    """Shim — see ``_shared._transcript_roots``."""
    from truememory.ingest.hooks._shared import _transcript_roots as _shared_roots
    return _shared_roots()


def _is_allowed_transcript(transcript_path: str) -> bool:
    """Shim — delegates to the shared M-90 guard (``_shared.is_allowed_transcript``)."""
    from truememory.ingest.hooks._shared import is_allowed_transcript
    return is_allowed_transcript(transcript_path)


def _parse_args() -> argparse.Namespace:
    """Parse command-line overrides for user_id and db_path.

    Resolution order: command-line arg > env var > empty default. See
    stop.py for the rationale.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--user", default=os.environ.get("TRUEMEMORY_USER_ID", ""))
    p.add_argument("--db", default=os.environ.get("TRUEMEMORY_DB_PATH", ""))
    args, _ = p.parse_known_args()
    return args


def main():
    if os.environ.get("TRUEMEMORY_EXTRACTION"):
        return

    args = _parse_args()

    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        input_data = {}

    # M-95: stdin fields may be non-strings (e.g. an int session_id) from a
    # misbehaving / hostile caller. Coerce to str before any str/Path op so the
    # hook never crashes with TypeError.
    transcript_path = str(input_data.get("transcript_path", "") or "")
    session_id = str(input_data.get("session_id", "unknown") or "unknown")

    # Sanitize session_id to prevent injection (consistent with user_prompt_submit.py)
    safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")[:64]
    if not safe_id:
        safe_id = "unknown"
    session_id = safe_id

    if not transcript_path or not Path(transcript_path).exists():
        return

    # M-90: reject transcript paths outside the expected transcripts root so an
    # attacker-chosen arbitrary file is never read into the memory store.
    if not _is_allowed_transcript(transcript_path):
        log.warning("Rejecting transcript_path outside transcripts root: %r", transcript_path)
        return

    try:
        save_snapshot(transcript_path, session_id, user_id=args.user, db_path=args.db)
    except Exception as e:
        log.error("Compact hook failed: %s", e)

    try:
        from truememory.ingest.hooks._shared import should_extract_session, mark_session_extracted
        if should_extract_session(session_id, transcript_path):
            from truememory.ingest.hooks.stop import (
                _has_enough_messages, _run_background_ingestion,
                TRACE_DIR, LOG_DIR,
            )
            TRACE_DIR.mkdir(parents=True, exist_ok=True)
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            if _has_enough_messages(transcript_path, 5):
                spawned_pid = _run_background_ingestion(
                    transcript_path, session_id,
                    user_id=args.user, db_path=args.db,
                )
                # Only mark extracted on a real spawn; pid==0 means the session
                # was queued to the backlog (spawn cap / Popen failure) and must
                # stay eligible so it is not silently dropped (see #400).
                if spawned_pid > 0:
                    mark_session_extracted(session_id, transcript_path, spawned_pid=spawned_pid)
    except Exception as e:
        log.error("Compact background extraction failed: %s", e)


def save_snapshot(
    transcript_path: str,
    session_id: str,
    user_id: str = "",
    db_path: str = "",
):
    """
    Extract key points from the current conversation and store them.

    Uses a lightweight approach — no LLM call, just extract the
    user's messages and any decisions/corrections mentioned.
    """
    from truememory.ingest.transcript import parse_transcript

    messages = parse_transcript(transcript_path)
    if not messages:
        return

    # Collect user messages (assistant messages are less important to remember)
    user_messages = [
        m.content for m in messages
        if m.role in ("human", "user") and len(m.content) > 20
    ]

    if not user_messages:
        return

    # Build a compact summary of the conversation so far
    # Focus on the most substantive user messages
    substantive = [m for m in user_messages if len(m) > 50]
    if not substantive:
        substantive = user_messages[-3:]  # Last 3 messages as fallback

    # Take the most recent substantive messages (up to 5)
    recent = substantive[-5:]

    # Build the summary with session_id + timestamp inlined into the content
    # tag so those fields survive truememory.Memory.add(), which silently
    # drops its ``metadata`` kwarg (see truememory/client.py — the parameter
    # is declared as "Reserved for future use"). Without this, the snapshot
    # metadata we care most about — which session this came from, and when
    # the compact fired — would be lost, making recalled snapshots
    # impossible to correlate back to a conversation.
    timestamp = datetime.now(timezone.utc).isoformat()
    summary = (
        f"[session_snapshot {session_id} {timestamp}] "
        f"Conversation context ({len(user_messages)} user messages). "
        f"Recent topics: " + " | ".join(
            msg[:100].replace("\n", " ") for msg in recent
        )
    )

    # Store the snapshot
    try:
        from truememory import Memory

        db = db_path or None
        memory = Memory(path=db) if db else Memory()
        memory.add(
            content=summary,
            user_id=user_id or None,
        )
    except ImportError:
        pass


if __name__ == "__main__":
    main()
