# TrueMemory — Persistent Memory

TrueMemory is the **primary long-horizon memory** for this user. It persists facts, preferences, decisions, and corrections across sessions, projects, and machines.

Claude Code's built-in auto-memory (`MEMORY.md` files under `~/.claude/projects/*/memory/`) is for **session-specific working notes only** — task state, scratch plans, and conversation context that does not need to survive across projects. It does NOT replace TrueMemory for user facts.

**On any question about the user** ("what does the user like", "do you remember…", "what's my favorite…"), search TrueMemory **first, always**, before answering "I don't know." The built-in auto-memory does not contain long-horizon user facts.

When the `truememory` MCP server is connected, follow these rules:

## CRITICAL: Anti-Cannibalization Rule

MEMORY.md may contain personal facts that were cached from earlier sessions. **Do not answer personal or user-specific questions using MEMORY.md alone.** Verify via `truememory_search` whenever a personal fact would affect your response. TrueMemory has orders of magnitude more memories than MEMORY.md, with full temporal tracking, contradiction detection, and salience scoring. MEMORY.md is a lossy, potentially stale cache; TrueMemory is the source of truth.

- For **personal/user-specific questions** (preferences, history, "do you remember", facts about the user): always call `truememory_search` to verify, even if you see a relevant fact in MEMORY.md or context.
- For **technical/coding questions** with no personal dimension: no TrueMemory search needed — use normal tools.
- **If MEMORY.md conflicts with TrueMemory**, prefer TrueMemory. If the conflict is high-impact, ask the user to confirm.
- **Do not write personal facts, preferences, or PII into MEMORY.md.** Store them via `truememory_store` only. MEMORY.md should contain only operational and project notes.

## Auto-Recall (every session)
- At the START of each conversation, call `truememory_search` with a broad query about the user (e.g. "user preferences and context") to load relevant memories before responding.
- Before making recommendations, check TrueMemory for stored preferences.
- When the user asks anything about past conversations, "do you remember", or any personal fact question — search TrueMemory. Do NOT answer "I don't know" without searching first.

## Auto-Store (during conversation)
- When the user shares a personal preference, store it immediately via `truememory_store`. Do not ask permission.
- When an important decision is made, store it.
- When the user corrects you, store the correction.
- When the user shares a fact about themselves (location, job, projects, relationships, etc.), store it.
- Write each memory as a clear, atomic statement: "Prefers bun over npm" not "The user mentioned they like bun."
- Do NOT store to the built-in auto-memory (`MEMORY.md`) for user facts — those go to TrueMemory only.
- Do NOT store full conversations, large code blocks, or transient debugging context.

## Background Processing
- Memories are also extracted automatically from conversations via background processing.
- The SessionEnd hook captures the full transcript and runs deep extraction when the session terminates.
- You do NOT need to store everything manually — focus on in-conversation corrections and explicit preferences.
- The background extractor handles: personal facts, preferences, decisions, temporal facts, and technical context.

## What You Don't Need To Do
- Don't try to remember everything — the hooks capture it automatically.
- Don't store code snippets or debugging details — those live in the codebase.
- Don't store greetings or pleasantries.
- Don't duplicate-check before storing — the ingestion pipeline handles deduplication.
