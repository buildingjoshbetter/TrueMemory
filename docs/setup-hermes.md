# Hermes Agent Setup

## Prerequisites

- Hermes Agent installed (`~/.hermes/` exists)
- TrueMemory installed: `uv tool install truememory`
- Python 3.10+

## Automatic Setup

```bash
truememory-ingest setup --cli hermes
```

Or during the interactive setup wizard:

```bash
truememory-ingest setup
# Select Hermes Agent when prompted
```

## Manual Setup

### 1. MCP Server

Merge this server entry into `~/.hermes/config.yaml`, preserving other servers and settings:

```yaml
mcp_servers:
  truememory:
    command: /path/to/python
    args:
      - -m
      - truememory.mcp_server
```

### 2. Shell hooks

Merge this `hooks` map into the same `~/.hermes/config.yaml`, preserving other event handlers. Each event contains a list of command entries. Replace the placeholder paths with absolute installed paths:

```yaml
hooks:
  on_session_start:
    - command: /path/to/python /path/to/truememory/ingest/hooks/session_start.py
      timeout: 30
  on_session_end:
    - command: /path/to/python /path/to/truememory/ingest/hooks/stop.py
      timeout: 30
  pre_llm_call:
    - command: /path/to/python /path/to/truememory/ingest/hooks/user_prompt_submit.py
      timeout: 10
  on_session_finalize:
    - command: /path/to/python /path/to/truememory/ingest/hooks/compact.py
      timeout: 15
```

Hermes shell-hook timeouts are in seconds. Shell hooks are supported by Hermes in CLI and gateway sessions, as well as its other chat surfaces. They are distinct from gateway-only `HOOK.yaml`/`handler.py` extensions. See the [official Hermes hook documentation](https://hermes-agent.nousresearch.com/docs/user-guide/features/hooks/).

The entries above match the current TrueMemory adapter. `pre_llm_call` can fire repeatedly for continuations and retries, so it is not an exact user-prompt boundary. `on_session_finalize` is not a pre-compaction event. Native payload translation, context injection and transcript capture remain unverified under [Hermes integration #765](https://github.com/buildingjoshbetter/TrueMemory/issues/765).

Find your Python path: `python3 -c "import sys; print(sys.executable)"`

Find hook paths: `python3 -c "from pathlib import Path; import truememory; print(Path(truememory.__file__).parent / 'ingest' / 'hooks')"`

## Verification

```bash
truememory-ingest status
```

This checks configuration presence. Hermes's `hermes hooks list` also shows its configured events and consent status; neither establishes that TrueMemory's output reached the model. No native Hermes version was exercised by the documentation parity checks.

## Troubleshooting

- **Hooks not loading**: Check that `config.yaml` is valid YAML and that each event under `hooks` maps to a list. Review the hook consent status in `hermes hooks list`.
- **MCP not connecting**: Verify the Python path in `config.yaml`.
- **Existing config**: Merge only the TrueMemory server and hook entries. Keep unrelated YAML keys and handlers.
- **Paths with spaces**: Quote each path inside the hook `command` string.
- **Windows Defender ASR**: If commands are blocked, use `python -m` form instead. See [debugging guide](guides/debugging.md#windows-defender-asr-blocks-truememory-mcpexe).
