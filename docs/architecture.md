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

1. **Recall** (session start): The SessionStart hook searches stored memories and injects selected relevant results as `additionalContext`.

2. **Buffer** (during session): The UserPromptSubmit hook (Claude Code and Codex CLI) appends user-only diagnostic records to a per-session buffer. Buffers are rotated and pruned, with a default retention of 7 days. They are not a complete searchable conversation archive.

3. **Snapshot** (pre-compact): The pre-compaction hooks (Claude Code, Cursor, Gemini CLI, Kimi) store a text snapshot containing up to 5 selected recent user messages, taking the first 100 characters of each, before context compression. It does not preserve the full conversation.

4. **Extract** (session end): The SessionEnd hook launches background ingestion that parses and formats the transcript, extracts atomic facts through an LLM or heuristic fallback, then gates and deduplicates those facts before storing or updating selected memories.

The installed hook scripts are [`user_prompt_submit.py`](../truememory/ingest/hooks/user_prompt_submit.py) and [`compact.py`](../truememory/ingest/hooks/compact.py). The separate helpers in [`hooks/core.py`](../truememory/hooks/core.py) have different bounds: `buffer_message` keeps the first 10,000 characters of each prompt, and `save_snapshot` takes at most 500 source characters from each of up to 5 selected user messages. Availability of each lifecycle hook depends on the configured adapter.

## Input routes and coverage

The stored content depends on the entry point:

| Entry point | Stored representation | Coverage |
|-------------|-----------------------|----------|
| Bulk JSON import through `TrueMemoryEngine.ingest()` | Message records with the supplied `content` strings and supported fields. | Replaces the target dataset. It bypasses transcript fact extraction; preservation applies to the input supplied to this API. |
| Direct `Memory.add()` or `TrueMemoryEngine.add()` | Caller-supplied text, subject to API validation and applicable directive deduplication. | Stores what the caller provides, without automatically importing the surrounding conversation. |
| Automatic transcript ingestion through `IngestionPipeline.ingest_transcript()` | Extracted facts that survive gating and deduplication, with category tags when applicable. | Selected, potentially rewritten information from the formatted transcript, rather than every original event. |

The bulk route is implemented by [`engine.py`](../truememory/engine.py) and `load_messages_from_file` / `bulk_replace_messages` in [`storage.py`](../truememory/storage.py). Direct storage is exposed by [`client.py`](../truememory/client.py). Automatic capture uses [`ingest/pipeline.py`](../truememory/ingest/pipeline.py).

Before extraction, [`format_for_extraction`](../truememory/ingest/transcript.py) excludes tool and system messages, strips TrueMemory-injected context blocks, truncates assistant text after 500 characters, and omits message timestamps from the formatted text. The fact writer stores selected fact text with an ingestion timestamp; it does not persist a durable reference from each fact to its original transcript span. Source transcript files maintained by a CLI may still exist independently. This pipeline does not establish their retention or complete archival coverage.

The TrueMemory benchmark scripts use message-level inputs: [LoCoMo](../benchmarks/locomo/scripts/bench_truememory_pro.py) and [LongMemEval](../benchmarks/longmemeval/bench_truememory_pro.py) call bulk import, while [BEAM](../benchmarks/beam/bench_truememory_pro_beam1m.py) adds messages directly. The LoCoMo parser resolves relative dates before import. Thus, preserving the supplied benchmark content does not mean preserving every byte of the original dataset or evaluating the automatic fact-extraction route.

Ordinary search and Deep Search operate on stored database memories. [`Memory.search_deep()`](../truememory/client.py) delegates to the engine's agentic search, and the [MCP Deep Search tool](../truememory/mcp_server.py) uses that stored-memory path. It does not search external transcript archives or recover details that were never stored. Complete archive search with durable source references and coverage semantics remains future scope.

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
│   ├── chatgpt.py       # ChatGPT Desktop MCP config (experimental)
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
