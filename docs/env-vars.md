# Environment Variables

All TrueMemory environment variables and their defaults.

## Core

| Variable | Default | Description |
|----------|---------|-------------|
| `TRUEMEMORY_DB_PATH` | `~/.truememory/memories.db` | Path to the SQLite database |
| `TRUEMEMORY_DB` | (alias for DB_PATH) | Legacy name, still supported |
| `TRUEMEMORY_EMBED_MODEL` | `edge` | Embedding model tier (`edge`, `base`, `pro`) |
| `TRUEMEMORY_USER_ID` | `""` | Default user ID for hook-based ingestion |

## Encoding Gate

| Variable | Default | Description |
|----------|---------|-------------|
| `TRUEMEMORY_GATE_THRESHOLD` | `0.30` | Minimum score to store a fact (0.0-1.0) |
| `TRUEMEMORY_GATE_ENABLED` | `1` | Set to `0` to disable the gate entirely |
| `TRUEMEMORY_GATE_W_NOVELTY` | `0.25` | Weight for compression novelty signal |
| `TRUEMEMORY_GATE_W_SALIENCE` | `0.20` | Weight for speech-act salience signal |
| `TRUEMEMORY_GATE_W_PE` | `0.30` | Weight for prediction error signal |
| `TRUEMEMORY_GATE_SALIENCE_FLOOR` | `0.10` | Minimum salience to consider encoding |

## Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `TRUEMEMORY_ALPHA_SURPRISE` | `0.2` | L5 surprise boost coefficient |
| `TRUEMEMORY_L0_SCORE_SCALE` | `0.9` | Personality layer score scaling |
| `TRUEMEMORY_MIN_SALIENCE` | (auto) | Override minimum salience in search |
| `TRUEMEMORY_ENTITY_SHEETS` | `0` | Set to `1` to enable entity profile summaries |

## Hooks

| Variable | Default | Description |
|----------|---------|-------------|
| `TRUEMEMORY_RECALL_LIMIT` | `25` | Max memories injected at session start |
| `TRUEMEMORY_MIN_MESSAGES` | `5` | Minimum user messages before ingestion triggers |
| `TRUEMEMORY_INGEST_SPAWN_CAP` | `2` | Max concurrent ingestion processes |
| `TRUEMEMORY_INCREMENTAL_INTERVAL` | `14400` | Seconds between incremental extractions (default: 4 hours) |
| `TRUEMEMORY_BUFFER_RETENTION_DAYS` | `7` | Days to keep diagnostic buffer files |
| `TRUEMEMORY_BUFFER_MAX_BYTES` | `10485760` | Max buffer file size before rotation (10 MB) |

## Directories

| Variable | Default | Description |
|----------|---------|-------------|
| `TRUEMEMORY_TRACE_DIR` | `~/.truememory/traces` | Decision trace output directory |
| `TRUEMEMORY_LOG_DIR` | `~/.truememory/logs` | Ingestion log directory |
| `TRUEMEMORY_BACKLOG_DIR` | `~/.truememory/backlog` | Queued ingestion markers |
| `TRUEMEMORY_BUFFER_DIR` | `~/.truememory/buffers` | User message buffer directory |

## Telemetry

| Variable | Default | Description |
|----------|---------|-------------|
| `TRUEMEMORY_TELEMETRY` | (enabled) | Set to `off`, `false`, `0`, or `no` to disable telemetry |

## API Keys

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key for HyDE (Pro tier) |
| `OPENROUTER_API_KEY` | OpenRouter API key for HyDE |
| `OPENAI_API_KEY` | OpenAI API key for HyDE |
