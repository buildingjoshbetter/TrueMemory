# Multi-CLI Compatibility Matrix

TrueMemory integrates with multiple AI CLI tools. Each CLI has different config formats and hook systems, but TrueMemory provides a unified adapter layer.

## Feature Support

| Feature | Claude Code | Codex CLI | Cursor | Gemini CLI | Kimi CLI | Hermes Agent | OpenClaw |
|---------|:-----------:|:---------:|:------:|:----------:|:--------:|:------------:|:--------:|
| MCP server | JSON | TOML | JSON | JSON | JSON | YAML | JSON |
| Auto-recall at session start | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Auto-extract at session end | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Mid-session extraction (PreCompact) | Yes | No | Yes | Yes | Yes | No | No |
| Message buffering (UserPromptSubmit) | Yes | Yes | No | No | No | No | No |
| System prompt injection | Yes | Yes | Yes | Yes | No | No | No |
| Hook protocol | JSON stdin/stdout | JSON stdin/stdout | JSON stdin/stdout | JSON stdin/stdout | JSON stdin/stdout | JSON stdin/stdout | JS plugin API |
| Config format | JSON | TOML | JSON (2 files) | JSON | TOML + JSON | YAML | JSON5 + JS |
| Non-interactive install | `--cli claude` | `--cli codex` | `--cli cursor` | `--cli gemini` | `--cli kimi` | `--cli hermes` | `--cli openclaw` |

## Config Locations

| CLI | MCP Config | Hook Config | Detection Path |
|-----|-----------|------------|----------------|
| Claude Code | `~/.claude/settings.json` | `~/.claude/settings.json` | `~/.claude/` |
| Codex CLI | `~/.codex/config.toml` | `~/.codex/config.toml` | `~/.codex/` |
| Cursor | `~/.cursor/mcp.json` | `~/.cursor/hooks.json` | `~/.cursor/` |
| Gemini CLI | `~/.gemini/settings.json` | `~/.gemini/settings.json` | `~/.gemini/` |
| Kimi CLI | `~/.kimi/mcp.json` | `~/.kimi/config.toml` | `~/.kimi/` |
| Hermes Agent | `~/.hermes/config.yaml` | `~/.hermes/cli-config.yaml` | `~/.hermes/` |
| OpenClaw | `~/.openclaw/openclaw.json` | `~/.openclaw/plugins/truememory/` | `~/.openclaw/` |

## Hook Events

| Event | Claude Code | Codex CLI | Cursor | Gemini CLI | Kimi CLI | Hermes Agent | OpenClaw |
|-------|------------|-----------|--------|------------|----------|-------------|----------|
| Session start | `SessionStart` | `SessionStart` | `sessionStart` | `SessionStart` | `SessionStart` | `on_session_start` | `before_agent_run` |
| Session end | `Stop` | `Stop` | `stop` | `SessionEnd` | `Stop` | `on_session_end` | `agent_end` |
| Pre-compact | `PreCompact` | — | `preCompact` | `PreCompress` | `PreCompact` | — | — |
| User message | `UserPromptSubmit` | `UserPromptSubmit` | — | — | — | — | — |

## Shared Memory

All CLIs share the same TrueMemory database (`~/.truememory/memories.db`). Memories stored from one CLI are available in all others. User scoping via `--user` flag works across CLIs.

## Known Limitations

- **Codex CLI**: MCP and hooks share a single `config.toml`. TrueMemory uses additive merges to avoid overwriting other settings.
- **Cursor**: MCP (`mcp.json`) and hooks (`hooks.json`) are separate files. Event names are camelCase. `hooks.json` requires `"version": 1` top-level key.
- **Gemini CLI**: MCP and hooks share a single `settings.json`. Uses different event names than Claude Code (`SessionEnd` instead of `Stop`, `PreCompress` instead of `PreCompact`).
- **Kimi CLI**: Hooks are in beta. Event availability may change.
- **Hermes Agent**: Gateway hooks (Telegram/Discord/etc.) require separate `handler.py` setup.
- **OpenClaw**: Uses a JS plugin system — requires Node.js at runtime for the plugin.
- **All CLIs**: The first search after model download may be slow (model loading).
