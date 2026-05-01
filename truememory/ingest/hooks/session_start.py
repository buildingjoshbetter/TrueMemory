#!/usr/bin/env python3
"""
SessionStart Hook — Memory Injection + First-Run Onboarding
=============================================================

Fires when a new Claude Code session begins. Two modes:

1. **First run** (no ~/.truememory/.onboarded marker):
   Injects the TrueMemory banner and guided setup instructions so
   Claude walks the user through tier selection on first launch.

2. **Normal run** (marker exists):
   Searches TrueMemory for relevant memories and injects them as
   additionalContext so Claude has full context from the start.

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
from pathlib import Path

log = logging.getLogger(__name__)

MEMORY_LIMIT = int(os.environ.get("TRUEMEMORY_RECALL_LIMIT", "15"))
ONBOARDED_MARKER = Path.home() / ".truememory" / ".onboarded"

BANNER = r"""
████████╗██████╗ ██╗   ██╗███████╗    ███╗   ███╗███████╗███╗   ███╗ ██████╗ ██████╗ ██╗   ██╗
╚══██╔══╝██╔══██╗██║   ██║██╔════╝    ████╗ ████║██╔════╝████╗ ████║██╔═══██╗██╔══██╗╚██╗ ██╔╝
   ██║   ██████╔╝██║   ██║█████╗      ██╔████╔██║█████╗  ██╔████╔██║██║   ██║██████╔╝ ╚████╔╝
   ██║   ██╔══██╗██║   ██║██╔══╝      ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗  ╚██╔╝
   ██║   ██║  ██║╚██████╔╝███████╗    ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║   ██║
   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝
""".strip()

SETUP_GUIDE = """
Welcome to TrueMemory — persistent memory for AI agents.

TrueMemory needs a quick one-time setup. IMPORTANT: Present this setup guide to the user NOW, before responding to anything else. Walk them through these steps:

1. **Choose a tier** — ask the user to pick one:
   - **Edge** — fastest, lightweight. Model2Vec embeddings (8M params), MiniLM reranker. Best for: local-only, low-resource machines.
   - **Base** — balanced. Qwen3 embeddings (256d), gte-reranker-modernbert. Best for: most users. Recommended.
   - **Pro** — maximum accuracy. Qwen3 + HyDE query expansion. Requires an API key (Anthropic, OpenRouter, or OpenAI).

2. **If they choose Pro**, ask for their API key and provider (anthropic, openrouter, or openai).

3. **Call `truememory_configure`** with their choices:
   - Edge: `truememory_configure(tier="edge")`
   - Base: `truememory_configure(tier="base")`
   - Pro: `truememory_configure(tier="pro", api_key="...", api_provider="...")`

4. **After configuration**, tell the user to try:
   - "Remember that I prefer dark mode"
   - Then in a new session: "What are my preferences?"

5. **Done!** TrueMemory will now automatically remember facts, preferences, and decisions across all sessions.
""".strip()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--user", default=os.environ.get("TRUEMEMORY_USER_ID", ""))
    p.add_argument("--db", default=os.environ.get("TRUEMEMORY_DB_PATH", ""))
    args, _ = p.parse_known_args()
    return args


def _is_first_run() -> bool:
    return not ONBOARDED_MARKER.exists()


def main():
    args = _parse_args()

    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        input_data = {}

    try:
        if _is_first_run():
            context = _first_run_context()
        else:
            context = recall_memories(input_data, user_id=args.user, db_path=args.db)

        if context:
            output = {"additionalContext": context}
            print(json.dumps(output))
    except Exception as e:
        log.error("SessionStart hook failed: %s", e)


def _first_run_context() -> str:
    lines = [
        "<truememory-first-run>",
        BANNER,
        "",
        SETUP_GUIDE,
        "</truememory-first-run>",
    ]
    return "\n".join(lines)


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
