# Debugging Guide

## Check if TrueMemory is set up correctly

```bash
truememory-ingest status
```

This verifies: truememory is importable, hooks are installed, MCP server is registered, LLM backend is available, and the database exists.

## View ingestion logs

```bash
truememory-ingest logs              # most recent session
truememory-ingest logs --tail 100   # more lines
truememory-ingest logs --list       # list all log files
```

Logs are stored at `~/.truememory/logs/`. Each session gets its own log file named by session ID.

## View encoding gate decisions

```bash
truememory-ingest trace             # most recent session
truememory-ingest trace --raw       # raw JSON
```

Shows every fact the extractor found and whether the gate accepted or rejected it, with scores for novelty, salience, and prediction error.

## View stored facts

```bash
truememory-ingest facts             # most recent session
truememory-ingest facts --all       # include rejected facts
truememory-ingest facts --category personal
```

## Common issues

### "truememory is not installed"

You're running the command from a shell that doesn't have the uv tool directory on PATH. Either:
- Open a new terminal window
- Or run: `export PATH="$HOME/.local/bin:$PATH"`

### Memories not being stored

1. Check that the session had at least 5 user messages (configurable via `TRUEMEMORY_MIN_MESSAGES`)
2. Check the encoding gate threshold: `truememory-ingest trace` — if all facts show "SKIP", the threshold may be too high
3. Verify the stop hook is firing: `truememory-ingest logs` should show ingestion output

### Search returns wrong results

1. Verify you're on the right tier: `truememory-ingest status`
2. If you recently switched tiers, ensure re-embedding completed: run `truememory-ingest upgrade-tier <tier> --force`
3. Try `truememory_search_deep` for harder queries

### Hooks not firing

1. Run `truememory-ingest status` — check the hooks section
2. Reinstall hooks: `truememory-ingest install`
3. Verify Claude Code settings: check `~/.claude/settings.json` for hook entries

## File locations

| Path | Contents |
|------|----------|
| `~/.truememory/memories.db` | SQLite database (all memories) |
| `~/.truememory/config.json` | Tier, API keys, user ID |
| `~/.truememory/logs/` | Per-session ingestion logs |
| `~/.truememory/traces/` | Per-session encoding gate traces |
| `~/.truememory/buffers/` | Per-session user message buffers |
| `~/.truememory/backlog/` | Queued ingestions (failed spawns) |
| `~/.truememory/.onboarded` | First-run marker |
| `~/.claude/settings.json` | Hook registrations |
| `~/.claude/CLAUDE.md` | System prompt (managed section) |

## Backup and restore

```bash
# Backup
cp ~/.truememory/memories.db ~/backup_memories.db

# Restore
cp ~/backup_memories.db ~/.truememory/memories.db

# Migrate to a new machine
scp ~/.truememory/memories.db newmachine:~/.truememory/
scp ~/.truememory/config.json newmachine:~/.truememory/
```

The database is a single SQLite file. Portable, backupable, `cp` it anywhere.
