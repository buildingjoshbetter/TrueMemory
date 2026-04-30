#!/usr/bin/env python3
"""
SessionStart Hook — Memory Injection
=====================================

Fires when a new Claude Code session begins. Searches TrueMemory for
relevant memories and injects them as additionalContext so Claude
has full context from the start.

This is the "recall" phase — the hippocampus retrieving relevant
memories to inform the current experience.

Input (stdin JSON):
    {"session_id": "...", "cwd": "...", "transcript_path": "..."}

Output (stdout JSON):
    {"additionalContext": "<truememory-context>...</truememory-context>"}
"""

import argparse
import json
import logging
import os
import sys

log = logging.getLogger(__name__)

# Configuration via environment or defaults
MEMORY_LIMIT = int(os.environ.get("TRUEMEMORY_RECALL_LIMIT", "15"))


def _parse_args() -> argparse.Namespace:
    """Parse command-line overrides for user_id and db_path.

    Resolution order: command-line arg > env var > empty default. See
    stop.py for the rationale — hooks must accept both sources so
    multi-profile installs work regardless of whether Claude Code
    passes env vars or argv.
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

    try:
        context = recall_memories(input_data, user_id=args.user, db_path=args.db)
        if context:
            output = {"additionalContext": context}
            print(json.dumps(output))
    except Exception as e:
        # Hooks must not crash — log and exit cleanly
        log.error("SessionStart hook failed: %s", e)


def recall_memories(input_data: dict, user_id: str = "", db_path: str = "") -> str:
    """Search TrueMemory and format relevant memories for injection."""
    try:
        from truememory import Memory
    except ImportError:
        return ""

    db = db_path or None
    memory = Memory(path=db) if db else Memory()

    queries = [
        "user preferences favorites likes dislikes",
        "personal facts name location job role",
        "recent decisions and commitments",
        "corrections and updates to prior information",
        "relationships family friends coworkers",
    ]

    all_results = []
    seen_ids = set()

    for query in queries:
        try:
            if user_id:
                results = memory.search(query, user_id=user_id, limit=MEMORY_LIMIT)
            else:
                results = memory.search(query, limit=MEMORY_LIMIT)

            for r in results:
                rid = r.get("id")
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    all_results.append(r)
        except Exception:
            continue

    if not all_results:
        return ""

    lines = [
        "<truememory-context>",
        "## TrueMemory — What You Know About This User",
        "These are facts from TrueMemory (the primary long-horizon memory system).",
        "Use these to answer user questions. Search TrueMemory for more if needed.",
        "",
    ]
    for r in all_results[:MEMORY_LIMIT]:
        content = r.get("content", "").strip()
        if content:
            lines.append(f"- {content}")

    lines.append("</truememory-context>")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
