# Codex CLI Setup

## Prerequisites

- Codex CLI installed (`~/.codex/` exists)
- TrueMemory installed: `uv tool install truememory`
- Python 3.10+

## Automatic Setup

```bash
truememory-ingest setup --cli codex
```

Or during the interactive setup wizard:

```bash
truememory-ingest setup
# Select Codex CLI when prompted
```

## Manual Setup

### 1. MCP Server + Lifecycle Hooks

Both MCP and inline hooks live in `~/.codex/config.toml`. Merge the MCP entry and append the TrueMemory matcher groups below, preserving other settings and hook handlers. Replace the placeholder paths with absolute paths in the environment where TrueMemory is installed.

```toml
[mcp_servers.truememory]
command = "/path/to/python"
args = ["-m", "truememory.mcp_server"]

[[hooks.SessionStart]]

[[hooks.SessionStart.hooks]]
type = "command"
command = "/path/to/python /path/to/truememory/ingest/hooks/session_start.py --host codex --event SessionStart"
timeout = 10

[[hooks.Stop]]

[[hooks.Stop.hooks]]
type = "command"
command = "/path/to/python /path/to/truememory/ingest/hooks/stop.py --host codex --event Stop"
timeout = 5

[[hooks.UserPromptSubmit]]

[[hooks.UserPromptSubmit.hooks]]
type = "command"
command = "/path/to/python /path/to/truememory/ingest/hooks/user_prompt_submit.py --host codex --event UserPromptSubmit"
timeout = 5

[[hooks.PreCompact]]

[[hooks.PreCompact.hooks]]
type = "command"
command = "/path/to/python /path/to/truememory/ingest/hooks/compact.py --host codex --event PreCompact"
timeout = 5
```

Codex uses event groups with nested command handlers, and `timeout` is in seconds. Keep one hook representation per configuration layer to avoid duplicate execution. Review and trust the new definitions through Codex's `/hooks` command before expecting execution. See the [official OpenAI hooks documentation](https://learn.chatgpt.com/docs/hooks).

The installed command identifies the host and exact event independently of stdin. `SessionStart` and `UserPromptSubmit` use `hookSpecificOutput.additionalContext` with the matching `hookEventName`. `Stop` is a turn completion event, not proof that the session ended. Native execution, transcript capture and model-visible context remain unverified here; see [compatibility limits](compatibility.md#validation-status).

Find your Python path: `python3 -c "import sys; print(sys.executable)"`

Find hook paths: `python3 -c "from pathlib import Path; import truememory; print(Path(truememory.__file__).parent / 'ingest' / 'hooks')"`

### 2. AGENTS.md (Optional)

For auto-recall/auto-store instructions, run:

```bash
truememory-ingest setup --cli codex
```

This adds TrueMemory instructions to `~/.codex/AGENTS.md`; preserve unrelated instructions when editing manually.

## Verification

```bash
truememory-ingest status
```

This checks configuration presence. It does not prove that a Codex event ran or that its output reached the model. No native Codex version was exercised by the documentation parity checks.

## Troubleshooting

- **MCP not connecting**: Verify the Python path in `config.toml` points to the environment with TrueMemory installed.
- **Existing config**: TrueMemory uses additive merges — your existing Codex config is preserved.
- **Hooks not firing**: Check absolute paths, the Python environment and `/hooks` trust status. Quote each path inside `command` if it contains spaces.
- **Windows Defender ASR**: If commands are blocked, use `python -m` form instead. See [debugging guide](guides/debugging.md#windows-defender-asr-blocks-truememory-mcpexe).
