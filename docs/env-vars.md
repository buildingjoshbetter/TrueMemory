# Environment Variables

All TrueMemory environment variables and their defaults.

## Core

| Variable | Default | Description |
|----------|---------|-------------|
| `TRUEMEMORY_DB_PATH` | `~/.truememory/memories.db` | Path to the SQLite database |
| `TRUEMEMORY_DB` | (alias for DB_PATH) | Legacy name, still supported |
| `TRUEMEMORY_EMBED_MODEL` | `edge` | Embedding model tier (`edge`, `base`, `pro`) |
| `TRUEMEMORY_USER_ID` | `""` | Default user ID for hook-based ingestion |

## Model Server

| Variable | Default | Description |
|----------|---------|-------------|
| `TRUEMEMORY_DEVICE` | `auto` | Inference device for embedding and reranker models: `cpu`, `mps`, `cuda`, or `auto`. Honored by the shared model server (embed + rerank), the local-fallback reranker, and the adaptive throttler. `cpu` is the escape hatch for MPS out-of-memory retry storms on memory-constrained Macs. If the requested accelerator is unavailable (or the value is invalid), a warning is logged and auto-detection (`cuda` → `mps` → `cpu`) is used. Note: the model server reads this at startup — after changing the value, restart the server (it exits on its own after the idle timeout, 300s by default). After an MPS OOM the affected model is degraded to CPU for the server's lifetime regardless of this setting. |

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

## Ingestion

| Variable | Default | Description |
|----------|---------|-------------|
| `TRUEMEMORY_INGEST_LOCK` | `~/.truememory/ingest.lock` | Path to cross-process ingestion lock file |

## Hooks

| Variable | Default | Description |
|----------|---------|-------------|
| `TRUEMEMORY_RECALL_LIMIT` | `25` | Max memories injected at session start |
| `TRUEMEMORY_MIN_MESSAGES` | `5` | Minimum user messages before ingestion triggers |
| `TRUEMEMORY_INGEST_SPAWN_CAP` | `2` | Max concurrent ingestion processes |
| `TRUEMEMORY_INCREMENTAL_INTERVAL` | `14400` | Seconds between incremental extractions (default: 4 hours) |
| `TRUEMEMORY_BUFFER_RETENTION_DAYS` | `7` | Days to keep diagnostic buffer files |
| `TRUEMEMORY_BUFFER_MAX_BYTES` | `10485760` | Max buffer file size before rotation (10 MB) |
| `TRUEMEMORY_RECALL_DEBOUNCE_SECONDS` | `60` | Window after SessionStart recall during which the first prompt's auto-recall is skipped (`0` or negative disables) |
| `TRUEMEMORY_HOOK_RECALL_TIMEOUT` | `5` | Per-request model-server deadline (seconds) for hook recall searches. On expiry the search falls back to FTS-only retrieval instead of blocking the hook. `0` or negative disables (legacy 120s behavior) |

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

## Cloud Backup

Opt-in encrypted cloud backup (issue #199, Phase 1). Disabled unless
`TRUEMEMORY_BACKUP_ENABLED` is truthy. See [cloud-backup.md](cloud-backup.md).

| Variable | Default | Description |
|----------|---------|-------------|
| `TRUEMEMORY_BACKUP_ENABLED` | (disabled) | Set to `1`/`true`/`yes`/`on` to enable cloud backup |
| `TRUEMEMORY_BACKUP_BUCKET` | | S3-compatible bucket name (required when enabled) |
| `TRUEMEMORY_BACKUP_KEY` | | Base64 32-byte encryption key, user-controlled (required when enabled). Generate with `truememory-ingest backup --generate-key` |
| `TRUEMEMORY_BACKUP_ENDPOINT` | AWS endpoint | Custom endpoint for R2/MinIO/other S3-compatible providers |
| `TRUEMEMORY_BACKUP_REGION` | `us-east-1` | Region name (usually ignored by S3-compatible providers) |
| `TRUEMEMORY_BACKUP_PREFIX` | `truememory/` | Object key prefix |
| `TRUEMEMORY_BACKUP_OBJECT` | `<prefix><db filename>.tmbackup` | Explicit object key override |
| `TRUEMEMORY_BACKUP_ACCESS_KEY_ID` | provider chain | Optional explicit access key; otherwise boto3's standard chain applies |
| `TRUEMEMORY_BACKUP_SECRET_ACCESS_KEY` | provider chain | Optional secret for the access key above |

## API Keys

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key for HyDE query expansion and LLM-based fact extraction |
| `OPENROUTER_API_KEY` | OpenRouter API key for HyDE and fact extraction |
| `OPENAI_API_KEY` | OpenAI API key for HyDE and fact extraction |
