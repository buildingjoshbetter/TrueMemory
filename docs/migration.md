# Migration Guide

## Switching Between CLIs

TrueMemory stores all memories in a single database at `~/.truememory/memories.db`. This means:

- **No migration needed** — memories are automatically shared across all configured CLIs
- Adding a new CLI is additive — just run `truememory-ingest setup --cli <name>`
- Removing a CLI doesn't affect your memories

## Adding a New CLI

```bash
# Add Kimi alongside existing Claude Code setup
truememory-ingest setup --cli kimi

# Or add multiple at once
truememory-ingest setup --cli kimi,hermes
```

## Removing a CLI

Currently, use the CLI's own config tools to remove TrueMemory entries. The adapter's `uninstall()` method handles this programmatically:

```python
from truememory.hooks.registry import get_adapter
adapter = get_adapter("kimi")
adapter.uninstall()
```

## Upgrading TrueMemory

After upgrading TrueMemory (`uv tool upgrade truememory`), re-run setup to update hook paths:

```bash
truememory-ingest setup
# Re-select your CLIs — existing config is preserved, paths are updated
```

## User ID Scoping

If you use `--user` flags, ensure the same user ID is used across all CLIs for consistent memory retrieval. The `--user` flag is passed through the hook commands in each CLI's config.

## Database Location

Default: `~/.truememory/memories.db`

Override with `--db` flag or `TRUEMEMORY_DB_PATH` environment variable. If you override, ensure all CLIs point to the same database.
