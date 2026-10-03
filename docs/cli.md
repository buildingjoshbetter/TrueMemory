# CLI Reference

TrueMemory provides two CLI tools: `truememory-mcp` (MCP server) and `truememory-ingest` (ingestion and management).

> **Windows Defender ASR**: If commands are blocked, use `python -m truememory.mcp_server` / `python -m truememory.ingest.cli` instead. See [debugging guide](guides/debugging.md#windows-defender-asr-blocks-truememory-mcpexe).

---

## truememory-mcp

The MCP server process. Invoked automatically by Claude Code/Desktop.

```bash
truememory-mcp              # run the MCP server (stdio transport)
truememory-mcp --setup      # auto-configure Claude Code + Claude Desktop
truememory-mcp --help       # show help
truememory-mcp --version    # show version
```

---

## truememory-ingest

Ingestion pipeline and management CLI.

### ingest

Ingest a conversation transcript and extract memories.

```bash
truememory-ingest ingest /path/to/transcript.json
truememory-ingest ingest /path/to/transcript.json --user alice
truememory-ingest ingest /path/to/transcript.json --threshold 0.25
```

| Flag | Default | Description |
|------|---------|-------------|
| `--user` | `""` | User ID for memory scoping |
| `--db` | `~/.truememory/memories.db` | Path to database |
| `--threshold` | `0.30` | Encoding gate threshold (0.0-1.0) |
| `--provider` | `auto` | LLM provider (auto/ollama/claude_cli/openrouter/anthropic/openai) |
| `--model` | `""` | LLM model name |
| `--session` | `""` | Session identifier for tracing |
| `--trace` | `None` | Save decision trace to file |
| `-v` | | Verbose logging |

### install

Install lifecycle hooks into Claude Code settings.

```bash
truememory-ingest install
truememory-ingest install --user alice --db /path/to/db
truememory-ingest install --dry-run    # preview without writing
```

### setup

Interactive first-time setup wizard. Guides through tier selection, API key entry, and hook installation.

```bash
truememory-ingest setup
truememory-ingest setup --non-interactive    # use defaults + env vars
truememory-ingest setup --cli chatgpt   # experimental, see docs/setup-chatgpt.md
```

### upgrade-tier

Switch embedding tier without re-running the full setup wizard.

```bash
truememory-ingest upgrade-tier base
truememory-ingest upgrade-tier pro
truememory-ingest upgrade-tier edge
truememory-ingest upgrade-tier base --force    # re-embed even if already on base
```

### status

Check whether TrueMemory is set up correctly.

```bash
truememory-ingest status
```

### uninstall

Remove TrueMemory hooks from Claude Code settings.

```bash
truememory-ingest uninstall
truememory-ingest uninstall --dry-run
```

### logs

View recent ingestion log files.

```bash
truememory-ingest logs                    # tail most recent log
truememory-ingest logs --tail 100         # show 100 lines
truememory-ingest logs --session abc123   # specific session
truememory-ingest logs --list             # list available log files
```

### trace

Show the decision trace for a session (what the encoding gate decided for each fact).

```bash
truememory-ingest trace                   # most recent session
truememory-ingest trace abc123            # specific session
truememory-ingest trace --raw             # raw JSON output
```

### facts

Show facts stored during a session with per-fact gate decisions.

```bash
truememory-ingest facts                   # most recent session
truememory-ingest facts abc123            # specific session
truememory-ingest facts --all             # include skipped facts
truememory-ingest facts --category personal
```

### stats

Show ingestion statistics from a trace file.

```bash
truememory-ingest stats /path/to/trace.json
```

### backup

Create an encrypted cloud backup of the database. Opt-in: requires
`TRUEMEMORY_BACKUP_ENABLED=1` plus bucket and encryption key configuration.
See [cloud-backup.md](cloud-backup.md).

```bash
truememory-ingest backup                  # snapshot, encrypt, upload
truememory-ingest backup --db /path/memories.db
truememory-ingest backup --key-file /path/key.txt
truememory-ingest backup --generate-key    # print a new encryption key and exit
```

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `~/.truememory/memories.db` | Path to database |
| `--key-file` | | File containing the base64 encryption key (alternative to `TRUEMEMORY_BACKUP_KEY`) |
| `--generate-key` | | Print a fresh random key to stdout and exit |

### restore

Restore the database from an encrypted cloud backup. Replaces the current
database; a `*.backup-pre-restore-*` safety snapshot is created first. Asks for
confirmation unless `--yes` is passed (required when stdin is not a TTY).

```bash
truememory-ingest restore                 # interactive confirmation
truememory-ingest restore --yes           # non-interactive
truememory-ingest restore --db /path/memories.db --key-file /path/key.txt
```

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `~/.truememory/memories.db` | Path to database |
| `--key-file` | | File containing the base64 encryption key |
| `-y`, `--yes` | | Skip the interactive confirmation |
