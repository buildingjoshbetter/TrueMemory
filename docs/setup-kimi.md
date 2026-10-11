# Kimi CLI Setup

## Prerequisites

- Kimi CLI installed (`~/.kimi/` exists)
- TrueMemory installed: `uv tool install truememory`
- Python 3.10+

This guide targets the legacy `MoonshotAI/kimi-cli` configuration at `~/.kimi/`, which matches the current TrueMemory adapter. Newer Kimi Code documentation uses `~/.kimi-code/`; automatic discovery and installation for that location are unverified. See [legacy config source](https://github.com/MoonshotAI/kimi-cli/blob/main/docs/en/configuration/config-files.md) and [current Kimi Code hooks](https://www.kimi.ai/help/kimi-code/cli-hooks).

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

Merge this server entry into `~/.kimi/mcp.json`, preserving other servers and settings:

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

Append these TrueMemory entries to `~/.kimi/config.toml`, preserving other settings and hook entries. Replace the placeholder paths with absolute installed paths:

```toml
[[hooks]]
event = "SessionStart"
command = "/path/to/python /path/to/truememory/ingest/hooks/session_start.py"
timeout = 10

[[hooks]]
event = "Stop"
command = "/path/to/python /path/to/truememory/ingest/hooks/stop.py"
timeout = 5

[[hooks]]
event = "UserPromptSubmit"
command = "/path/to/python /path/to/truememory/ingest/hooks/user_prompt_submit.py"
timeout = 5

[[hooks]]
event = "PreCompact"
command = "/path/to/python /path/to/truememory/ingest/hooks/compact.py"
timeout = 5
```

Kimi hook timeouts are seconds. The flat `[[hooks]]` array is the Kimi contract; it differs from Codex's nested event groups. These entries match the current adapter's registration. Native event delivery, host-specific output and transcript capture remain unverified; no native Kimi version was exercised by this documentation check. See [compatibility limits](compatibility.md#validation-status).

Find hook paths: `python3 -c "from pathlib import Path; import truememory; print(Path(truememory.__file__).parent / 'ingest' / 'hooks')"`

## Verification

```bash
truememory-ingest status
```

This checks configuration presence. It does not establish lifecycle execution in either legacy Kimi CLI or current Kimi Code.

## Troubleshooting

- **Hooks not firing**: Kimi hooks are in beta. Check `kimi --version` for compatibility.
- **MCP not connecting**: Verify the Python path in `mcp.json` points to the environment with TrueMemory installed.
- **Existing config**: TrueMemory uses additive merges — your existing Kimi config is preserved.
- **Paths with spaces**: Quote each path inside the hook `command` string.
- **Windows Defender ASR**: If commands are blocked, use `python -m` form instead. See [debugging guide](guides/debugging.md#windows-defender-asr-blocks-truememory-mcpexe).
