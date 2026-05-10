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
    args = _parse_args()

    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        input_data = {}

    transcript_path = input_data.get("transcript_path", "")
    session_id = input_data.get("session_id", "unknown")

    # Sanitize session_id to prevent injection (consistent with user_prompt_submit.py)
    safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")[:64]
    if not safe_id:
        safe_id = "unknown"
    session_id = safe_id

    if not transcript_path or not Path(transcript_path).exists():
        return

    try:
        save_snapshot(transcript_path, session_id, user_id=args.user, db_path=args.db)
    except Exception as e:
        log.error("Compact hook failed: %s", e)

    # Background extraction: context is about to be compressed, so run
    # the full extraction pipeline on the transcript before it's lost.
    # The snapshot above is a lightweight breadcrumb; this does deep
    # LLM-based fact extraction. Uses the shared timestamp marker to
    # coordinate with the UserPromptSubmit incremental trigger.
    try:
        from truememory.ingest.hooks._shared import should_extract, mark_extracted
        if should_extract(interval=0):
            from truememory.ingest.hooks.stop import (
                _has_enough_messages, _run_background_ingestion,
            )
            if _has_enough_messages(transcript_path, 5):
                _run_background_ingestion(
                    transcript_path, session_id,
                    user_id=args.user, db_path=args.db,
                )
                mark_extracted()
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
