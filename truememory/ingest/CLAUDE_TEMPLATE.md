# TrueMemory — Persistent Memory

TrueMemory is the **primary long-horizon memory** for this user. It persists facts, preferences, decisions, and corrections across sessions, projects, and machines.

Claude Code's built-in auto-memory (`MEMORY.md` files under `~/.claude/projects/*/memory/`) is for **session-specific working notes only** — task state, scratch plans, and conversation context that does not need to survive across projects. It does NOT replace TrueMemory for user facts.

**On any question about the user** ("what does the user like", "do you remember…", "what's my favorite…"), search TrueMemory **first, always**, before answering "I don't know." The built-in auto-memory does not contain long-horizon user facts.

When the `truememory` MCP server is connected, follow these rules:

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
- The Stop hook captures the full transcript and runs deep extraction after sessions end.
- You do NOT need to store everything manually — focus on in-conversation corrections and explicit preferences.
- The background extractor handles: personal facts, preferences, decisions, temporal facts, and technical context.

## What You Don't Need To Do
- Don't try to remember everything — the hooks capture it automatically.
- Don't store code snippets or debugging details — those live in the codebase.
- Don't store greetings or pleasantries.
- Don't duplicate-check before storing — the ingestion pipeline handles deduplication.
