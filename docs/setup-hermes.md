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

Add to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  truememory:
    command: /path/to/python
    args:
      - -m
      - truememory.mcp_server
```

### 2. Plugin Hooks (CLI Mode)

Add to `~/.hermes/cli-config.yaml`:

```yaml
plugins:
  - name: truememory-session-start
    event: on_session_start
    command: /path/to/python /path/to/truememory/ingest/hooks/session_start.py
  - name: truememory-session-end
    event: on_session_end
    command: /path/to/python /path/to/truememory/ingest/hooks/stop.py
```

## Hermes Learning Loop

Hermes has a built-in self-improving learning loop. TrueMemory is complementary:
- **TrueMemory**: Cross-session persistent memory (facts, preferences, decisions)
- **Hermes learning loop**: Within-session self-improvement

Both work together — TrueMemory provides long-term context while Hermes optimizes per-session.

## Verification

```bash
truememory-ingest status
```

## Troubleshooting

- **Plugins not loading**: Check that `cli-config.yaml` is valid YAML (no tab characters).
- **MCP not connecting**: Verify the Python path in `config.yaml`.
- **Gateway users**: For Telegram/Discord/etc., a separate `handler.py` gateway hook is needed (see architecture docs).
- **Windows (Defender ASR Block mode)**: if `truememory-ingest setup --cli hermes` is blocked by ASR rule `01443614` ("Block executable files from running unless they meet a prevalence, age, or trusted list criteria"), use the module form — the signed `python.exe` wrapper passes ASR: `python -m truememory.ingest.cli setup --cli hermes`. The same applies to every other `truememory-ingest` / `truememory-mcp` command; see [docs/guides/debugging.md](guides/debugging.md#windows-risky-action-blocked--defender-asr-kills-truememory-mcpexe) for the full list.
