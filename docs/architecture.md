# Multi-CLI Architecture

## How TrueMemory Connects to CLIs

```
┌─────────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────┐
│ Claude Code │  │ Kimi CLI │  │ Hermes Agent │  │ OpenClaw │
└──────┬──────┘  └────┬─────┘  └──────┬───────┘  └────┬─────┘
       │              │               │                │
       ▼              ▼               ▼                ▼
┌──────────────────────────────────────────────────────────────┐
│                    Hook Adapters                              │
│  claude.py    kimi.py    hermes.py    openclaw.py            │
│  (JSON)       (TOML)     (YAML)       (JSON5+JS)            │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                     Core Hook Logic                           │
│  recall_memories()  buffer_message()  run_background_ingestion() │
│  save_snapshot()    prune_old_buffers()                       │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    TrueMemory Engine                          │
│  Memory.add()   Memory.search()   Encoding Gate              │
│  Vector Search  Reranker          HyDE Query Expansion       │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
                 ~/.truememory/memories.db
```

## Memory Lifecycle

1. **Recall** (session start): The SessionStart hook searches TrueMemory for relevant memories and injects them as `additionalContext` so the AI has full user context from the start.

2. **Buffer** (during session): The UserPromptSubmit hook (Claude Code only) appends user messages to a per-session buffer file for diagnostics.

3. **Snapshot** (pre-compact): The PreCompact hook (Claude Code, Kimi) saves a lightweight snapshot of the conversation before context compression.

4. **Extract** (session end): The SessionEnd hook launches a background ingestion process that parses the transcript, runs the encoding gate on each fact, and stores high-quality memories.

## Package Structure

```
truememory/hooks/
├── __init__.py
├── core.py              # CLI-agnostic logic (recall, buffer, extract)
├── cli.py               # install_cli(), uninstall_cli(), verify_cli()
├── registry.py          # CLI detection, state tracking
├── adapters/
│   ├── base.py          # CLIAdapter abstract base class
│   ├── claude.py        # Wraps existing install logic
│   ├── kimi.py          # TOML + JSON config
│   ├── hermes.py        # YAML config
│   └── openclaw.py      # JSON5 config + JS plugin
└── templates/
    └── openclaw/        # JS plugin files
        ├── plugin.json
        └── index.js
```

## Adapter Interface

Every CLI adapter implements `CLIAdapter`:

- `detect()` — is this CLI installed?
- `is_configured()` — is TrueMemory already wired in?
- `install_mcp()` — register the MCP server
- `install_hooks()` — register lifecycle hooks
- `uninstall()` — clean removal
- `verify()` — smoke test

## State Tracking

`~/.truememory/integrations.json` tracks which CLIs are configured:

```json
{
  "configured": ["claude", "kimi"],
  "configured_at": {
    "claude": "2026-05-09T12:00:00+00:00",
    "kimi": "2026-05-09T12:05:00+00:00"
  },
  "version": "0.7.0"
}
```
