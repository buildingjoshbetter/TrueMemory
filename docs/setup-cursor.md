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

Add to `~/.cursor/mcp.json`:

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

Add to `~/.cursor/hooks.json`:

```json
{
  "version": 1,
  "hooks": {
    "sessionStart": [
      {
        "command": "/path/to/python /path/to/truememory/ingest/hooks/session_start.py",
        "timeout": 10000
      }
    ],
    "stop": [
      {
        "command": "/path/to/python /path/to/truememory/ingest/hooks/stop.py",
        "timeout": 5000
      }
    ],
    "preCompact": [
      {
        "command": "/path/to/python /path/to/truememory/ingest/hooks/compact.py",
        "timeout": 5000
      }
    ]
  }
}
```

Find your Python path: `python3 -c "import sys; print(sys.executable)"`

Find hook paths: `python3 -c "from pathlib import Path; import truememory; print(Path(truememory.__file__).parent / 'ingest' / 'hooks')"`

### 3. .cursorrules (Optional)

For auto-recall/auto-store instructions, run:

```bash
truememory-ingest setup --cli cursor
```

This creates `~/.cursor/.cursorrules` with TrueMemory system prompt instructions.

## Verification

```bash
truememory-ingest status
```

## Troubleshooting

- **MCP not connecting**: Verify the Python path in `mcp.json` points to the environment with TrueMemory installed.
- **Existing config**: TrueMemory uses additive merges — your existing Cursor config is preserved.
- **Hooks not firing**: Check that event names are camelCase (`sessionStart`, `stop`, `preCompact`). PascalCase will silently fail.
- **Two config files**: MCP lives in `~/.cursor/mcp.json`, hooks live in `~/.cursor/hooks.json`. They are separate files.
- **Version key**: `hooks.json` requires `"version": 1` at the top level.
