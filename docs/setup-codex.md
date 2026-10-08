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
- **Windows Defender ASR**: If commands are blocked, use `python -m` form instead. See [debugging guide](guides/debugging.md#windows-defender-asr-blocks-truememory-mcpexe).

## Native rollout parsing

The parser supports Codex JSONL and JSON arrays containing native
`response_item` message envelopes. User and assistant text blocks retain file
order and repeated occurrences. Legacy `user_message` / `agent_message`
events and Paginated `item_completed` events are coverage evidence, not extra
conversation turns. Ordinary plain-text lifecycle records, including
`task_started` / `task_complete` and their `turn_*` aliases, are reconciled
against the canonical responses. A terminal summary uses its identified turn
and never supplies another echo credit.

This subset follows the [pinned Codex producer](https://github.com/openai/codex/tree/0b755b1945bf4a31560df2ce1469aeb044dccbc9/codex-rs).
It excludes explicit harness, skill, AGENTS, tool, inherited, compaction and
retained-delivery context. Positional provenance must align with the original
content array. Explicit unknown provenance cannot authorize text. Older
records without attribution retain role/text compatibility; JSON metadata
does not authenticate authorship or prove that earlier history is present.

The detailed parser reports partial coverage for unmatched events, ambiguous
lifecycles, unknown variants, media, unavailable delivery and unsupported
assistant rewriting. Citation and proposed-plan markup projections are not
implemented in this subset. Supported canonical text remains available for
inspection, with categorical diagnostics that omit input text and identifiers.
`complete` describes the supported supplied bytes, not the success of the
model's task or an authenticated record of the whole session.

The stop admission check counts parsed human messages after removing
TrueMemory-injected blocks. The default threshold of five means five nonempty
human occurrences. A long message, metadata, tool results or repeated event
echoes cannot meet that threshold through byte length. Role-marked plain text
still counts; unmarked prose does not. Supported human messages can count in
a partial transcript; admission is not a completion verdict.

Parser support alone does not establish automatic capture. Native hook
delivery and path admission remain separate integration requirements. Capture
currently uses the list parser API; detailed coverage does not yet control
durable success receipts. The #768 completion-policy work must preserve
partial coverage before automatic-capture completeness can be claimed.
