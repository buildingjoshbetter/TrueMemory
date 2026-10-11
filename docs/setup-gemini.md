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

Both MCP and hooks live in `~/.gemini/settings.json`. Merge the server entry and append the TrueMemory hook definitions below, preserving other settings and handlers. Replace the placeholder paths with absolute installed paths:

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
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/python /path/to/truememory/ingest/hooks/session_start.py --host gemini --event SessionStart",
            "name": "truememory-sessionstart",
            "timeout": 10000
          }
        ]
      }
    ],
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/python /path/to/truememory/ingest/hooks/stop.py --host gemini --event SessionEnd",
            "name": "truememory-sessionend",
            "timeout": 5000
          }
        ]
      }
    ],
    "BeforeAgent": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/python /path/to/truememory/ingest/hooks/user_prompt_submit.py --host gemini --event BeforeAgent",
            "name": "truememory-beforeagent",
            "timeout": 5000
          }
        ]
      }
    ],
    "PreCompress": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/python /path/to/truememory/ingest/hooks/compact.py --host gemini --event PreCompress",
            "name": "truememory-precompress",
            "timeout": 5000
          }
        ]
      }
    ]
  }
}
```

Each event contains HookDefinition groups, each with a `hooks` array of `type: "command"` handlers. Gemini's `timeout` is in milliseconds: `10000 / 1000 = 10` seconds and `5000 / 1000 = 5` seconds. See the [official Gemini hooks reference](https://geminicli.com/docs/hooks/reference/).

`SessionStart` and `BeforeAgent` use `hookSpecificOutput.additionalContext` with the matching `hookEventName`. Installed commands supply the trusted host and event; stdin alone does not select the host. Native delivery, Gemini transcript handling and model-visible context remain unverified here; see [compatibility limits](compatibility.md#validation-status).

Find your Python path: `python3 -c "import sys; print(sys.executable)"`

Find hook paths: `python3 -c "from pathlib import Path; import truememory; print(Path(truememory.__file__).parent / 'ingest' / 'hooks')"`

### 2. GEMINI.md (Optional)

For auto-recall/auto-store instructions, run:

```bash
truememory-ingest setup --cli gemini
```

This adds TrueMemory instructions to `~/.gemini/GEMINI.md`; preserve unrelated instructions when editing manually.

## Verification

```bash
truememory-ingest status
```

This checks configuration presence, not event execution or model-visible context. No native Gemini CLI version was exercised by the documentation parity checks.

## Troubleshooting

- **MCP not connecting**: Verify the Python path in `settings.json` points to the environment with TrueMemory installed.
- **Existing config**: TrueMemory uses additive merges — your existing Gemini config is preserved.
- **Hooks not firing**: Check absolute paths and the Python environment. Event names are PascalCase: `SessionStart`, `SessionEnd`, `BeforeAgent`, `PreCompress`. Quote paths inside `command` if they contain spaces.
- **Windows Defender ASR**: If commands are blocked, use `python -m` form instead. See [debugging guide](guides/debugging.md#windows-defender-asr-blocks-truememory-mcpexe).
