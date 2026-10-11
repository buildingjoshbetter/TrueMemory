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

The current adapter stages plugin files under `~/.openclaw/plugins/truememory/`. Current OpenClaw documentation describes discovery under `~/.openclaw/extensions/` or explicit `plugins.load.paths`. Automatic staging does not establish discovery or lifecycle support; the adapter path and plugin runtime remain tracked under [OpenClaw integration #764](https://github.com/buildingjoshbetter/TrueMemory/issues/764).

Or during the interactive setup wizard:

```bash
truememory-ingest setup
# Select OpenClaw when prompted
```

## Manual Setup

### 1. MCP Server

Merge this server entry into `~/.openclaw/openclaw.json`, preserving other servers and settings:

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

Copy the shipped package to the documented global discovery directory `~/.openclaw/extensions/truememory/`. The script refuses to replace an existing package; inspect an existing installation before updating it.

```bash
python3 - <<'PY'
from pathlib import Path
import shutil, truememory
src = Path(truememory.__file__).parent / 'hooks' / 'templates' / 'openclaw'
dst = Path.home() / '.openclaw' / 'extensions' / 'truememory'
files = ('openclaw.plugin.json', 'package.json', 'index.js')
for f in files:
    if not (src / f).is_file():
        raise FileNotFoundError(src / f)
dst.mkdir(parents=True, exist_ok=False)
for f in files:
    shutil.copy2(src / f, dst / f)
print(f'Plugin installed to {dst}')
PY
```

`openclaw.plugin.json` contains the plugin ID and configuration schema. `package.json` declares `type: "module"` and `openclaw.extensions: ["./index.js"]`, which selects the JavaScript entrypoint. All three files are required by this package. See [OpenClaw plugin installation and discovery](https://docs.openclaw.ai/cli/plugins/install) and [manifest versus package metadata](https://docs.openclaw.ai/plugins/manifest/package-json).

OpenClaw policy also determines activation. If you already use `plugins.allow`, add `truememory` to that list while preserving its other entries. Keep existing `plugins.deny`, `plugins.enabled` and per-plugin policy intact. Use `openclaw plugins list` to inspect discovery and activation before expecting execution.

The shipped entrypoint registers `session_start`, `session_end`, `before_tool_call` and `before_compaction`. These are source registrations, not a verified mapping to every installed OpenClaw version. Its payload handling and context delivery remain unverified under #764; copying the package does not prove native recall or extraction.

## Multi-Surface Memory

Cross-surface memory requires every surface to use the same TrueMemory database and user identity. The current plugin installer does not propagate its `user_id` or `db_path` arguments into the JavaScript hook commands, so explicit identity consistency remains unverified.

## Verification

```bash
truememory-ingest status
```

This checks the adapter's config and staging location. It does not verify the manual `extensions` package, native discovery, event execution or model-visible context. No native OpenClaw version was exercised by the documentation parity checks.

## Troubleshooting

- **Plugin not loading**: Verify `~/.openclaw/extensions/truememory/openclaw.plugin.json`, `package.json` and `index.js` exist, then inspect `openclaw plugins list` and plugin activation policy. The automatic adapter's `plugins` directory differs from this documented discovery location.
- **Python not found**: Set the `TRUEMEMORY_PYTHON` environment variable to your Python path.
- **JSON5 config**: OpenClaw uses JSON5. TrueMemory reads it but writes standard JSON — comments may be stripped on config updates.
- **Windows Defender ASR**: If commands are blocked, use `python -m` form instead. See [debugging guide](guides/debugging.md#windows-defender-asr-blocks-truememory-mcpexe).
