# Multi-CLI Compatibility Matrix

TrueMemory integrates with multiple AI CLI tools. Each CLI has different config formats and hook systems, but TrueMemory provides a unified adapter layer.

## Feature Support

| Feature | Claude Code | Kimi CLI | Hermes Agent | OpenClaw |
|---------|:-----------:|:--------:|:------------:|:--------:|
| MCP server | JSON | JSON | YAML | JSON |
| Auto-recall at session start | Yes | Yes | Yes | Yes |
| Auto-extract at session end | Yes | Yes | Yes | Yes |
| Mid-session extraction (PreCompact) | Yes | Yes | No | No |
| Message buffering (UserPromptSubmit) | Yes | No | No | No |
| System prompt injection | Yes | No | No | No |
| Hook protocol | JSON stdin/stdout | JSON stdin/stdout | JSON stdin/stdout | JS plugin API |
| Config format | JSON | TOML + JSON | YAML | JSON5 + JS |
| Non-interactive install | `--cli claude` | `--cli kimi` | `--cli hermes` | `--cli openclaw` |

## Config Locations

| CLI | MCP Config | Hook Config | Detection Path |
|-----|-----------|------------|----------------|
| Claude Code | `~/.claude/settings.json` | `~/.claude/settings.json` | `~/.claude/` |
| Kimi CLI | `~/.kimi/mcp.json` | `~/.kimi/config.toml` | `~/.kimi/` |
| Hermes Agent | `~/.hermes/config.yaml` | `~/.hermes/cli-config.yaml` | `~/.hermes/` |
| OpenClaw | `~/.openclaw/openclaw.json` | `~/.openclaw/plugins/truememory/` | `~/.openclaw/` |

## Hook Events

| Event | Claude Code | Kimi CLI | Hermes Agent | OpenClaw |
|-------|------------|----------|-------------|----------|
| Session start | `SessionStart` | `SessionStart` | `on_session_start` | `before_agent_run` |
| Session end | `Stop` | `Stop` | `on_session_end` | `agent_end` |
| Pre-compact | `PreCompact` | `PreCompact` | — | — |
| User message | `UserPromptSubmit` | — | — | — |

## Shared Memory

All CLIs share the same TrueMemory database (`~/.truememory/memories.db`). Memories stored from one CLI are available in all others. User scoping via `--user` flag works across CLIs.

## Known Limitations

- **Kimi CLI**: Hooks are in beta. Event availability may change.
- **Hermes Agent**: Gateway hooks (Telegram/Discord/etc.) require separate `handler.py` setup.
- **OpenClaw**: Uses a JS plugin system — requires Node.js at runtime for the plugin.
- **All CLIs**: The first search after model download may be slow (model loading).
