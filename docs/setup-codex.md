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

Both MCP and hooks live in `~/.codex/config.toml`. Add the following:

```toml
[mcp_servers.truememory]
command = "/path/to/python"
args = ["-m", "truememory.mcp_server"]

[[hooks]]
event = "SessionStart"
command = "/path/to/python /path/to/truememory/ingest/hooks/session_start.py"
timeout = 10000

[[hooks]]
event = "Stop"
command = "/path/to/python /path/to/truememory/ingest/hooks/stop.py"
timeout = 5000

[[hooks]]
event = "UserPromptSubmit"
command = "/path/to/python /path/to/truememory/ingest/hooks/user_prompt_submit.py"
timeout = 5000
```

Find your Python path: `python3 -c "import sys; print(sys.executable)"`

Find hook paths: `python3 -c "from pathlib import Path; import truememory; print(Path(truememory.__file__).parent / 'ingest' / 'hooks')"`

### 2. AGENTS.md (Optional)

For auto-recall/auto-store instructions, run:

```bash
truememory-ingest setup --cli codex
```

This creates `~/.codex/AGENTS.md` with TrueMemory system prompt instructions.

## Verification

```bash
truememory-ingest status
```

## Troubleshooting

- **MCP not connecting**: Verify the Python path in `config.toml` points to the environment with TrueMemory installed.
- **Existing config**: TrueMemory uses additive merges — your existing Codex config is preserved.
- **Hooks not firing**: Check that the hook script paths are absolute and the Python executable is correct.
- **Windows (Defender ASR Block mode)**: if `truememory-ingest setup --cli codex` is blocked by ASR rule `01443614` ("Block executable files from running unless they meet a prevalence, age, or trusted list criteria"), use the module form — the signed `python.exe` wrapper passes ASR: `python -m truememory.ingest.cli setup --cli codex`. The same applies to every other `truememory-ingest` / `truememory-mcp` command; see [docs/guides/debugging.md](guides/debugging.md#windows-risky-action-blocked--defender-asr-kills-truememory-mcpexe) for the full list.
