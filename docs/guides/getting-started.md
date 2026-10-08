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

Usage telemetry is enabled by default and is not anonymous. Every event carries a persistent user UUID and timestamp. Session starts include tier, app version, platform, architecture, Python version, a stable hashed device ID, and your email if configured. Email is sent on each session start when present, as well as during registration. Tool events include the tool name, duration and success or failure; tier changes include the previous and new tier.

The user UUID persists in the configuration. The device ID is a truncated SHA-256 hash of the operating system's machine identifier, so it can remain the same after reinstalling the app. Hashing does not make email-linked usage anonymous. Built-in telemetry events do not include memory content, query text, tool arguments or results, exception text, file paths, API keys, or credentials.

To opt out, set this in the environment used to launch each TrueMemory MCP server:

```bash
export TRUEMEMORY_TELEMETRY=off
```

Restart existing server processes after changing the setting. The values `false`, `0`, and `no` also disable telemetry, case-insensitively. Alternatively, add `"telemetry": false` to your existing `~/.truememory/config.json`, preserving the other fields, and restart. Disabled startup does not enqueue telemetry or start its transport threads. This also stops new telemetry-based update checks; an already cached notice may still appear.

See the [complete telemetry disclosure](../../README.md#faq) for the destination and field table, and the [Environment Variables](../env-vars.md) reference. These controls apply to usage telemetry; optional cloud features have separate data flows.

## Contributing

Want to contribute? See [CONTRIBUTING.md](../../CONTRIBUTING.md) for the development setup, branching conventions, and our code review process.
