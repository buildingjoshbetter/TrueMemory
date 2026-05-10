# Getting Started

## Install

**Mac / Linux:**
```bash
curl -LsSf https://raw.githubusercontent.com/buildingjoshbetter/TrueMemory/main/install.sh | sh
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/buildingjoshbetter/TrueMemory/main/install.ps1 | iex
```

The installer handles everything: uv, Python 3.12, TrueMemory, MCP server registration, hooks, and model downloads.

## First session

1. Quit Claude completely and reopen it
2. Type **"Set up TrueMemory"**
3. Choose Edge, Base, or Pro
4. Done. TrueMemory remembers your conversations automatically from here.

## How it works

TrueMemory captures memories at three points:
- **Session end** — when you close a Claude session, the full conversation is processed
- **Every 4 hours** — long-running sessions trigger incremental extraction automatically
- **Before context compression** — when Claude is about to compress its context, TrueMemory captures memories before they're lost

The extraction pipeline:
1. Reads the conversation transcript
2. Extracts atomic facts (preferences, decisions, corrections)
3. Filters through the encoding gate (novelty + salience + prediction error)
4. Deduplicates against existing memories
5. Stores what passes the gate

Next session, TrueMemory searches your memories and injects relevant context before your first message.

## Try it

In a Claude session:
```
"Remember that I prefer dark mode and TypeScript"
```

Close the session. Open a new one:
```
"What are my preferences?"
```

Claude will recall the fact from the previous session.

## Switch tiers

From the terminal:
```bash
truememory-ingest upgrade-tier pro
```

Or tell Claude: "Switch to Pro tier."

## Uninstall

```bash
uv tool uninstall truememory
```

## Python SDK

For embedding TrueMemory in your own applications:

```python
from truememory import Memory

m = Memory()
m.add("Prefers dark mode", user_id="alice")
results = m.search("preferences", user_id="alice")
print(results[0]["content"])
# "Prefers dark mode"
```

See [Python API Reference](../python-api.md) for full details.

## Telemetry

TrueMemory collects anonymous usage telemetry (tool calls, session counts, platform info). No memory content, queries, or API keys are ever sent. To opt out:

```bash
export TRUEMEMORY_TELEMETRY=off
```

See the [Environment Variables](../env-vars.md) reference for details.

## Contributing

Want to contribute? See [CONTRIBUTING.md](../../CONTRIBUTING.md) for the development setup, branching conventions, and our code review process.
