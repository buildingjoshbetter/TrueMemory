# OpenClaw Setup

## Prerequisites

- OpenClaw installed (`~/.openclaw/` exists)
- TrueMemory installed: `uv tool install truememory`
- Python 3.10+
- Node.js (for the OpenClaw plugin runtime)

## Automatic Setup

```bash
truememory-ingest setup --cli openclaw
```

Or during the interactive setup wizard:

```bash
truememory-ingest setup
# Select OpenClaw when prompted
```

## Manual Setup

### 1. MCP Server

Add to `~/.openclaw/openclaw.json`:

```json
{
  "mcp": {
    "servers": {
      "truememory": {
        "command": "/path/to/python",
        "args": ["-m", "truememory.mcp_server"]
      }
    }
  }
}
```

### 2. Plugin Installation

Copy the TrueMemory plugin to `~/.openclaw/plugins/truememory/`:

```bash
python3 -c "
from pathlib import Path
import shutil, truememory
src = Path(truememory.__file__).parent / 'hooks' / 'templates' / 'openclaw'
dst = Path.home() / '.openclaw' / 'plugins' / 'truememory'
dst.mkdir(parents=True, exist_ok=True)
for f in ('plugin.json', 'index.js'):
    shutil.copy2(src / f, dst / f)
print(f'Plugin installed to {dst}')
"
```

The plugin registers two event handlers:
- `before_agent_run` — recalls relevant memories and injects them as context
- `agent_end` — captures the transcript and triggers background extraction

## Multi-Surface Memory

OpenClaw connects to WhatsApp, Telegram, Slack, Discord, and more. Memories stored from any surface are available on all others — TrueMemory's `user_id` scoping ensures a single memory database serves all platforms.

## Verification

```bash
truememory-ingest status
```

## Troubleshooting

- **Plugin not loading**: Verify `~/.openclaw/plugins/truememory/plugin.json` exists and is valid JSON.
- **Python not found**: Set the `TRUEMEMORY_PYTHON` environment variable to your Python path.
- **JSON5 config**: OpenClaw uses JSON5. TrueMemory reads it but writes standard JSON — comments may be stripped on config updates.
- **Windows Defender ASR**: If commands are blocked, use `python -m` form instead. See [debugging guide](guides/debugging.md#windows-defender-asr-blocks-truememory-mcpexe).
