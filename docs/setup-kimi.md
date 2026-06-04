# Kimi CLI Setup

## Prerequisites

- Kimi CLI installed (`~/.kimi/` exists)
- TrueMemory installed: `uv tool install truememory`
- Python 3.10+

## Automatic Setup

```bash
truememory-ingest setup --cli kimi
```

Or during the interactive setup wizard:

```bash
truememory-ingest setup
# Select Kimi CLI when prompted
```

## Manual Setup

### 1. MCP Server

Add to `~/.kimi/mcp.json`:

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

Find your Python path: `python3 -c "import sys; print(sys.executable)"`

### 2. Lifecycle Hooks

Add to `~/.kimi/config.toml`:

```toml
[[hooks]]
event = "SessionStart"
command = "/path/to/python /path/to/truememory/ingest/hooks/session_start.py"
timeout = 10000

[[hooks]]
event = "Stop"
command = "/path/to/python /path/to/truememory/ingest/hooks/session_end.py"
timeout = 5000

[[hooks]]
event = "PreCompact"
command = "/path/to/python /path/to/truememory/ingest/hooks/compact.py"
timeout = 5000
```

Find hook paths: `python3 -c "from pathlib import Path; import truememory; print(Path(truememory.__file__).parent / 'ingest' / 'hooks')"`

## Verification

```bash
truememory-ingest status
```

## Troubleshooting

- **Hooks not firing**: Kimi hooks are in beta. Check `kimi --version` for compatibility.
- **MCP not connecting**: Verify the Python path in `mcp.json` points to the environment with TrueMemory installed.
- **Existing config**: TrueMemory uses additive merges — your existing Kimi config is preserved.
- **Windows Defender ASR**: If commands are blocked, use `python -m` form instead. See [debugging guide](guides/debugging.md#windows-defender-asr-blocks-truememory-mcpexe).
