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

MEMORY_LIMIT = int(os.environ.get("TRUEMEMORY_RECALL_LIMIT", "25"))
ONBOARDED_MARKER = Path.home() / ".truememory" / ".onboarded"

BANNER = r"""
████████╗██████╗ ██╗   ██╗███████╗    ███╗   ███╗███████╗███╗   ███╗ ██████╗ ██████╗ ██╗   ██╗
╚══██╔══╝██╔══██╗██║   ██║██╔════╝    ████╗ ████║██╔════╝████╗ ████║██╔═══██╗██╔══██╗╚██╗ ██╔╝
   ██║   ██████╔╝██║   ██║█████╗      ██╔████╔██║█████╗  ██╔████╔██║██║   ██║██████╔╝ ╚████╔╝
   ██║   ██╔══██╗██║   ██║██╔══╝      ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗  ╚██╔╝
   ██║   ██║  ██║╚██████╔╝███████╗    ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║   ██║
   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝
                                  a sauron company
""".strip()

SETUP_GUIDE = """
Welcome to TrueMemory — persistent memory for AI agents.

TrueMemory needs a quick one-time setup. IMPORTANT: Present this setup guide to the user NOW, before responding to anything else. Walk them through these steps:

1. **Choose a tier** — ask the user to pick one:
   - **Edge** — fastest, lightweight. Model2Vec embeddings (8M params), MiniLM reranker. Best for: local-only, low-resource machines.
   - **Base** — balanced. Qwen3 embeddings (256d), gte-reranker-modernbert. Best for: most users. Recommended.
   - **Pro** — maximum accuracy. Qwen3 + HyDE query expansion. Requires an API key (Anthropic, OpenRouter, or OpenAI).

2. **If they choose Pro**, ask for their API key and provider (anthropic, openrouter, or openai).

3. **Ask for their email** — ask: "What's your email? We'll use it to send you important updates." Always include it in the configure call if provided.

4. **Call `truememory_configure`** with their choices:
   - Edge: `truememory_configure(tier="edge")` or `truememory_configure(tier="edge", email="user@example.com")`
   - Base: `truememory_configure(tier="base")` or with email
   - Pro: `truememory_configure(tier="pro", api_key="...", api_provider="...", email="...")`

5. **After configuration**, tell the user to try:
   - "Remember that I prefer dark mode"
   - Then in a new session: "What are my preferences?"

6. **Done!** TrueMemory will now automatically remember facts, preferences, and decisions across all sessions.
""".strip()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--user", default=os.environ.get("TRUEMEMORY_USER_ID", ""))
    p.add_argument("--db", default=os.environ.get("TRUEMEMORY_DB_PATH", ""))
    args, _ = p.parse_known_args()
    return args


def _is_first_run() -> bool:
    return not ONBOARDED_MARKER.exists()


def _check_for_update() -> str:
    """Check if the MCP server wrote an update notice during startup."""
    try:
        update_path = Path.home() / ".truememory" / ".update_available"
        if not update_path.exists():
            return ""
        data = json.loads(update_path.read_text(encoding="utf-8"))
        # Delete the file so we only show the notice once
        update_path.unlink(missing_ok=True)
        if data.get("update_available"):
            return (
                "<truememory-update>\n"
                f"A new version of TrueMemory is available: v{data.get('latest_version', '?')}. "
                f"Tell the user: \"{data.get('message', 'Run: uv tool upgrade truememory')}\"\n"
                "</truememory-update>"
            )
    except Exception:
        pass
    return ""


def _check_email_needed() -> str:
    """Prompt for email if the user hasn't provided one yet."""
    try:
        config_path = Path.home() / ".truememory" / "config.json"
        if not config_path.exists():
            return ""
        config = json.loads(config_path.read_text(encoding="utf-8"))
        # Only prompt if onboarded but no email
        if config.get("tier") and not config.get("email"):
            return (
                "<truememory-email-request>\n"
                "TrueMemory doesn't have your email yet. Ask the user: "
                "\"What's your email? We use it to send important updates about TrueMemory.\" "
                "If they provide one, call truememory_configure with their current tier and the email: "
                f"truememory_configure(tier=\"{config.get('tier', 'edge')}\", email=\"their@email.com\"). "
                "If they decline, respect that and don't ask again this session.\n"
                "</truememory-email-request>"
            )
    except Exception:
        pass
    return ""


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

        # Check for available updates
        update_notice = _check_for_update()
        if update_notice:
            context = (context or "") + "\n\n" + update_notice

        # Prompt for email if not set yet
        email_notice = _check_email_needed()
        if email_notice:
            context = (context or "") + "\n\n" + email_notice

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

    per_query_limit = max(1, MEMORY_LIMIT // len(queries))

    all_results = []
    seen_ids = set()
    seen_content = set()

    for query in queries:
        added_this_query = 0
        try:
            results = memory._engine.search(query, limit=per_query_limit * 3, _skip_reranker=True)
            if user_id:
                results = [r for r in results if r.get("sender", "") == user_id]

            for r in results:
                if added_this_query >= per_query_limit:
                    break
                rid = r.get("id")
                if rid in seen_ids:
                    continue
                content = r.get("content", "").strip()
                if not content:
                    continue
                # Content-based dedup: normalize and check for near-duplicates
                normalized = content.lower().strip().rstrip(".")
                if normalized in seen_content:
                    continue
                # Check for substring containment (catches rephrased duplicates)
                is_dup = False
                for existing in seen_content:
                    if normalized in existing or existing in normalized:
                        is_dup = True
                        break
                if is_dup:
                    continue
                seen_ids.add(rid)
                seen_content.add(normalized)
                all_results.append(r)
                added_this_query += 1
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
