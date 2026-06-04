# Gemini CLI Setup

## Prerequisites

- Gemini CLI installed (`~/.gemini/` exists)
- TrueMemory installed: `uv tool install truememory`
- Python 3.10+

## Automatic Setup

```bash
truememory-ingest setup --cli gemini
```

Or during the interactive setup wizard:

```bash
truememory-ingest setup
# Select Gemini CLI when prompted
```

## Manual Setup

### 1. MCP Server + Lifecycle Hooks

Both MCP and hooks live in `~/.gemini/settings.json`. Add the following:

```json
{
  "mcpServers": {
    "truememory": {
      "command": "/path/to/python",
      "args": ["-m", "truememory.mcp_server"]
    }
  },
  "hooks": {
    "SessionStart": [
      {
        "command": "/path/to/python /path/to/truememory/ingest/hooks/session_start.py",
        "timeout": 10000
      }
    ],
    "SessionEnd": [
      {
        "command": "/path/to/python /path/to/truememory/ingest/hooks/session_end.py",
        "timeout": 5000
      }
    ],
    "PreCompress": [
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

### 2. GEMINI.md (Optional)

For auto-recall/auto-store instructions, run:

```bash
truememory-ingest setup --cli gemini
```

This creates `~/.gemini/GEMINI.md` with TrueMemory system prompt instructions.

## Verification

```bash
truememory-ingest status
```

## Troubleshooting

- **MCP not connecting**: Verify the Python path in `settings.json` points to the environment with TrueMemory installed.
- **Existing config**: TrueMemory uses additive merges — your existing Gemini config is preserved.
- **Hooks not firing**: Check that the hook script paths are absolute and the Python executable is correct. Event names are PascalCase: `SessionStart`, `SessionEnd`, `PreCompress`.
- **Windows Defender ASR**: If commands are blocked, use `python -m` form instead. See [debugging guide](guides/debugging.md#windows-defender-asr-blocks-truememory-mcpexe).
