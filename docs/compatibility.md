# Multi-CLI Compatibility Matrix

TrueMemory integrates with multiple AI CLI tools. Each CLI has different config formats and hook systems, but TrueMemory provides a unified adapter layer.

## Configuration support

This table describes what the adapters register or write. A `Registered` entry is not native lifecycle acceptance. See [validation status](#validation-status) before relying on automatic recall or extraction.

| Feature | Claude Code | ChatGPT Desktop* | Codex CLI | Cursor | Gemini CLI | Kimi CLI | Hermes Agent | OpenClaw |
|---------|:-----------:|:-------:|:---------:|:------:|:----------:|:--------:|:------------:|:--------:|
| MCP server | JSON | JSON (experimental) | TOML | JSON | JSON | JSON | YAML | JSON |
| Session-start hook | Registered | No | Registered | Registered | Registered | Registered | Registered | Staged JS |
| Extraction boundary | Registered | No | Turn stop | Turn stop | Session end | Turn stop | Session end | Staged JS |
| Compaction/finalization hook | Registered | No | PreCompact | preCompact | PreCompress | PreCompact | Session finalize | Staged JS |
| Prompt/call hook | Registered | No | UserPromptSubmit | beforeSubmitPrompt | BeforeAgent | UserPromptSubmit | pre_llm_call | before_tool_call |
| Instruction file written | Yes | No | AGENTS.md | .cursorrules* | GEMINI.md | truememory_prompt.md* | truememory_prompt.md* | truememory_prompt.md* |
| Hook wire format | JSON stdin/stdout | None | JSON stdin/stdout | JSON stdin/stdout | JSON stdin/stdout | JSON stdin/stdout | JSON stdin/stdout | JS plugin API |
| Config format | JSON | JSON | TOML | JSON (2 files) | JSON | TOML + JSON | YAML | JSON5 + JS |
| Non-interactive install | `--cli claude` | `--cli chatgpt` | `--cli codex` | `--cli cursor` | `--cli gemini` | `--cli kimi` | `--cli hermes` | `--cli openclaw` |

\* **ChatGPT Desktop is experimental and forward-looking:** ChatGPT Desktop does **not** currently load local MCP servers. OpenAI supports only remote HTTPS MCP connectors, enabled via developer mode in the ChatGPT *web* UI, and the macOS desktop app cannot enable developer mode (as of mid-2026). The adapter pre-stages `mcp.json` for when/if OpenAI enables local MCP support — TrueMemory will not appear in ChatGPT until then. See [ChatGPT Setup](setup-chatgpt.md).

\* An instruction file being written does not establish that the host loads that file. Current loading of the adapter's legacy `.cursorrules` path and its generic `truememory_prompt.md` files has not been established by these checks.

## Config Locations

| CLI | MCP Config | Hook Config | Detection Path |
|-----|-----------|------------|----------------|
| Claude Code | `~/.claude/settings.json` | `~/.claude/settings.json` | `~/.claude/` |
| ChatGPT Desktop | `~/Library/Application Support/com.openai.chat/mcp.json` (experimental, not yet read by ChatGPT) | Not supported | `/Applications/ChatGPT.app` |
| Codex CLI | `~/.codex/config.toml` | `~/.codex/config.toml` | `~/.codex/` |
| Cursor | `~/.cursor/mcp.json` | `~/.cursor/hooks.json` | `~/.cursor/` |
| Gemini CLI | `~/.gemini/settings.json` | `~/.gemini/settings.json` | `~/.gemini/` |
| Kimi CLI (legacy adapter target) | `~/.kimi/mcp.json` | `~/.kimi/config.toml` | `~/.kimi/` |
| Hermes Agent | `~/.hermes/config.yaml` | `~/.hermes/config.yaml` under `hooks` | `~/.hermes/` |
| OpenClaw | `~/.openclaw/openclaw.json` | Manual: `~/.openclaw/extensions/truememory/`; adapter staging: `~/.openclaw/plugins/truememory/` | `~/.openclaw/` |

## Hook Events

| Event | Claude Code | ChatGPT Desktop | Codex CLI | Cursor | Gemini CLI | Kimi CLI | Hermes Agent | OpenClaw |
|-------|------------|---------|-----------|--------|------------|----------|-------------|----------|
| Session start | `SessionStart` | None | `SessionStart` | `sessionStart` | `SessionStart` | `SessionStart` | `on_session_start` | `session_start` |
| Extraction trigger | `SessionEnd` | None | `Stop` (turn) | `stop` (turn) | `SessionEnd` | `Stop` (turn) | `on_session_end` | `session_end` |
| Compaction/finalization | `PreCompact` | None | `PreCompact` | `preCompact` | `PreCompress` | `PreCompact` | `on_session_finalize` | `before_compaction` |
| Prompt/call registration | `UserPromptSubmit` | None | `UserPromptSubmit` | `beforeSubmitPrompt` | `BeforeAgent` | `UserPromptSubmit` | `pre_llm_call` | `before_tool_call` |

Hermes `pre_llm_call` and OpenClaw `before_tool_call` are not exact user-message boundaries. OpenClaw names above are the shipped JavaScript registrations, whose native payload and lifecycle contract remain unverified.

## Hook schemas and timeout units

| Host | Registration shape | Session-start timeout | Other registered timeouts | Unit |
| --- | --- | ---: | --- | --- |
| Codex | `hooks.Event[]` groups with nested `hooks[]` and `type = "command"` | 10 | 5 | Seconds |
| Cursor | `version: 1`, camelCase event arrays | 10 | 5 | Seconds |
| Gemini | Event arrays of HookDefinition groups with nested `hooks[]` and `type: "command"` | 10000 | 5000 | Milliseconds |
| Kimi legacy CLI | Flat `[[hooks]]` entries with `event` | 10 | 5 | Seconds |
| Hermes | `config.yaml` event lists under `hooks` | 30 | Session end 30, pre-LLM 10, finalize 15 | Seconds |
| OpenClaw | `openclaw.plugin.json` + `package.json` entrypoint + `index.js` | 10000 | Sync prompt 5000, compaction 5000 | JavaScript subprocess milliseconds |

For second-based hooks, `10000 / 60 = 166.67` minutes and `5000 / 60 = 83.33` minutes, which are the old unintended values. The corrected values are 10 and 5 seconds. Gemini's native conversion is `10000 / 1000 = 10` seconds and `5000 / 1000 = 5` seconds.

## Validation status

Configuration contracts were reviewed against the adapters and official host documentation on 2026-10-10. Documentation checks parse the snippets and use synthetic isolated configurations; they do not launch a native host. Native tested versions for Codex, Cursor, Gemini, Kimi, Hermes and OpenClaw: none in this parity validation.

Codex and Gemini recall commands identify both `--host` and the exact `--event`; their context output uses `hookSpecificOutput.additionalContext` and the corresponding `hookEventName`. Cursor `sessionStart` uses `additional_context`. Cursor `beforeSubmitPrompt` supports `continue`/`user_message`, so per-prompt recall injection is unsupported there; its hook remains registered for capture. These contracts depend on the host-aware hook changes in [#761](https://github.com/buildingjoshbetter/TrueMemory/issues/761) and [#763](https://github.com/buildingjoshbetter/TrueMemory/issues/763).

Config presence and `truememory-ingest status` do not prove execution, payload translation, transcript capture or model-visible recall. [C05/#761](https://github.com/buildingjoshbetter/TrueMemory/issues/761), [C07/#763](https://github.com/buildingjoshbetter/TrueMemory/issues/763), [C08/#764](https://github.com/buildingjoshbetter/TrueMemory/issues/764) and [C09/#765](https://github.com/buildingjoshbetter/TrueMemory/issues/765) remain incomplete. The separately implemented [C06/#762 Codex rollout reader](https://github.com/buildingjoshbetter/TrueMemory/issues/762) does not establish the other host lifecycles. C10/#766 corrects manual configuration only.

Contract references: [official OpenAI hooks](https://learn.chatgpt.com/docs/hooks), [Cursor hooks](https://prod.cursor.com/docs/hooks), [Gemini hooks](https://geminicli.com/docs/hooks/reference/), [legacy Kimi source](https://github.com/MoonshotAI/kimi-cli/blob/main/docs/en/customization/hooks.md), [current Kimi Code hooks](https://www.kimi.ai/help/kimi-code/cli-hooks), [Hermes shell hooks](https://hermes-agent.nousresearch.com/docs/user-guide/features/hooks/), [OpenClaw discovery](https://docs.openclaw.ai/cli/plugins/install) and [package entrypoints](https://docs.openclaw.ai/plugins/manifest/package-json).

## Shared Memory

All CLIs share the same TrueMemory database (`~/.truememory/memories.db`). Memories stored from one CLI are available in all others. User scoping via `--user` flag works across CLIs.

## Known Limitations

- **Codex CLI**: MCP and inline hooks share `config.toml`. Hook handlers are nested under event groups, use second-based timeouts and require Codex hook trust. `Stop` is a turn boundary.
- **ChatGPT Desktop**: Local MCP configs are **not currently loaded by ChatGPT Desktop** — OpenAI supports only remote HTTPS MCP connectors via developer mode on the web UI (the macOS app cannot enable it as of mid-2026). The adapter pre-stages config for a possible future and requires the actual app to be installed. Lifecycle hooks and system prompt file injection are not supported. There is no ChatGPT Desktop for Linux.
- **Cursor**: MCP (`mcp.json`) and hooks (`hooks.json`) are separate files. Event names are camelCase, timeouts are seconds and `hooks.json` requires `"version": 1`. Local user hooks are unavailable to cloud agents. Per-prompt recall injection is unsupported by `beforeSubmitPrompt`.
- **Gemini CLI**: MCP and nested hook definitions share `settings.json`; timeouts are milliseconds. Its `SessionEnd`, `BeforeAgent` and `PreCompress` events differ from Claude Code.
- **Kimi CLI**: The adapter targets legacy `~/.kimi/` and its beta hooks. Current Kimi Code documentation uses `~/.kimi-code/`; support for that location is unverified.
- **Hermes Agent**: MCP and shell hooks share `config.yaml`. Hermes supports shell hooks on CLI and gateway surfaces, but TrueMemory's native translation remains unverified. Separate gateway extensions use `HOOK.yaml`/`handler.py`.
- **OpenClaw**: Requires a JavaScript runtime and all three package files. The adapter's `plugins` staging directory differs from documented `extensions` discovery. Plugin user/database propagation, native payloads and context delivery remain unverified.
- **All CLIs**: The first search after model download may be slow (model loading).
