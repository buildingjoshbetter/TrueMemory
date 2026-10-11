# Cursor Setup

## Prerequisites

- Cursor installed (`~/.cursor/` exists)
- TrueMemory installed: `uv tool install truememory`
- Python 3.10+

## Automatic Setup

```bash
truememory-ingest setup --cli cursor
```

Or during the interactive setup wizard:

```bash
truememory-ingest setup
# Select Cursor when prompted
```

## Manual Setup

### 1. MCP Server

Merge this server entry into `~/.cursor/mcp.json`, preserving other servers and settings:

```json
{
  "mcpServers": {
    "truememory": {
      "command": "/path/to/python",
      "args": ["-m", "truememory.mcp_server"]
    }
  }
}
```

### 2. Lifecycle Hooks

Merge these event entries into `~/.cursor/hooks.json`, preserving other handlers. Replace the placeholder paths with absolute installed paths:

```json
{
  "version": 1,
  "hooks": {
    "sessionStart": [
      {
        "command": "/path/to/python /path/to/truememory/ingest/hooks/session_start.py --host cursor --event sessionStart",
        "timeout": 10
      }
    ],
    "stop": [
      {
        "command": "/path/to/python /path/to/truememory/ingest/hooks/stop.py --host cursor --event stop",
        "timeout": 5
      }
    ],
    "beforeSubmitPrompt": [
      {
        "command": "/path/to/python /path/to/truememory/ingest/hooks/user_prompt_submit.py --host cursor --event beforeSubmitPrompt",
        "timeout": 5
      }
    ],
    "preCompact": [
      {
        "command": "/path/to/python /path/to/truememory/ingest/hooks/compact.py --host cursor --event preCompact",
        "timeout": 5
      }
    ]
  }
}
```

Cursor's `timeout` is in seconds. `sessionStart` accepts `additional_context`; `beforeSubmitPrompt` accepts `continue` and `user_message`, so TrueMemory registers it for prompt capture and does not inject per-prompt recall there. `stop` marks agent completion, and `preCompact` observes compaction. See the [official Cursor hooks contract](https://prod.cursor.com/docs/hooks).

These are local user hooks. Cursor cloud agents do not load `~/.cursor/hooks.json`. Native delivery, capture and model-visible recall remain unverified here; see [compatibility limits](compatibility.md#validation-status).

Find your Python path: `python3 -c "import sys; print(sys.executable)"`

Find hook paths: `python3 -c "from pathlib import Path; import truememory; print(Path(truememory.__file__).parent / 'ingest' / 'hooks')"`

### 3. .cursorrules (Optional)

For auto-recall/auto-store instructions, run:

```bash
truememory-ingest setup --cli cursor
```

The adapter writes `~/.cursor/.cursorrules`. This documentation check does not establish that the current Cursor release loads that legacy location as rules.

## Verification

```bash
truememory-ingest status
```

This checks configuration presence, not event execution or model-visible context. No native Cursor version was exercised by the documentation parity checks.

## Troubleshooting

- **MCP not connecting**: Verify the Python path in `mcp.json` points to the environment with TrueMemory installed.
- **Existing config**: TrueMemory uses additive merges — your existing Cursor config is preserved.
- **Hooks not firing**: Check exact camelCase event names (`sessionStart`, `stop`, `beforeSubmitPrompt`, `preCompact`), installed paths and the Python environment. Quote paths inside `command` if they contain spaces.
- **Two config files**: MCP lives in `~/.cursor/mcp.json`, hooks live in `~/.cursor/hooks.json`. They are separate files.
- **Version key**: `hooks.json` requires `"version": 1` at the top level.
- **Windows Defender ASR**: If commands are blocked, use `python -m` form instead. See [debugging guide](guides/debugging.md#windows-defender-asr-blocks-truememory-mcpexe).
