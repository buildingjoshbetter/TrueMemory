# Changelog

## [0.6.0] — 2026-04-XX

### Added
- **L5 surprise rerank boost** — retrieval now reweights candidates by
  `(1 + α · surprise)` using the `surprise_scores` table populated at
  ingest. Default α=0.3. Override via `Memory(alpha_surprise=…)` or
  `TRUEMEMORY_ALPHA_SURPRISE` env var. Set to `0` to disable.

### Changed
- **L4 `build_entity_summary_sheets` disabled by default** per
  MEMORIST-L4 research (2026-04-23): the function produced monolithic
  per-entity profile rows that saturated top-1 retrieval and leaked
  superseded facts into contradiction scoring. Disabling is Pareto-
  dominant (+5.3 pts composite probe metric, +12.9 pts contradiction
  accuracy, −15 ms wall-clock, −1 KB/persona storage).

  - **Escape hatch:** set `TRUEMEMORY_ENTITY_SHEETS=1` (also accepts
    `true`, `yes`, `on`, case-insensitive) **before first engine
    `open()`** to retain legacy behavior. Setting the var after the
    initial upgrade-open will not restore already-purged rows — the
    next `consolidate()` will write them fresh.
  - **One-time migration on `open()`** purges legacy
    `period='entity_profile'` summary rows for upgraders. Guarded by a
    `l4_entity_profile_migration_done` flag in the `metadata` table so
    subsequent opens skip the scan.
  - **DeprecationWarning** now emitted when `build_entity_summary_sheets`
    is called directly (e.g., by a user re-enabling via env var).
  - **Failure visibility:** the migration's exception path now logs at
    WARNING (was DEBUG) so silent failures on a destructive operation
    surface in production logs.

## [0.5.0] - 2026-04-23

Post-v0.4.0 hardening release. 40 findings closed from a structured audit ("Hunter v1" — see tracker #44), shipped across 11 PRs + 3 direct-to-main commits. v0.4.0 was a code-only tier-realignment release that never reached PyPI; v0.5.0 is the first published release to include it.

Backward compatibility: no public API was removed. The one name change (`load_messages` → `bulk_replace_messages`) ships with a `DeprecationWarning`-emitting alias for one release.

### Added — new public API

- **`TrueMemoryMigrationError`** (in `truememory.vector_search`). Raised on DB open when the stored embedder doesn't match the active tier — catches the v0.3.0 Pro @ 1024d → v0.4.0 @ 256d upgrade-path crash with an actionable migration hint instead of a raw `sqlite3.OperationalError: Dimension mismatch`. Also catches same-dim / different-model drift (e.g., Model2Vec 256d → Qwen3 256d) via a new `metadata` key-value table that `build_vectors` / `build_separation_vectors` stamp with `(embed_model, embed_dim)` on every rebuild.
- **`bulk_replace_messages(conn, messages)`** (in `truememory.storage`). The non-deprecated name for the destructive DELETE-then-INSERT helper. `load_messages` is kept as a deprecated alias that emits `DeprecationWarning`.
- **`truememory_stats()["health"]`** (MCP tool). New payload with `reranker`, `hyde_llm`, and `vectors` subsystem status, each reporting `{status: "ok"|"degraded", last_error, ...}`. MCP clients can now diagnose "why is search bad?" without digging through server logs. `health.hyde_llm` additionally reports `active_provider` and `last_error_by_provider` (anthropic / openrouter / openai).
- **`get_vectors_load_error()`** helper (in `truememory.engine`) for programmatic access to the sqlite-vec load-failure state.
- **`__all__`** in `truememory/__init__.py` now enumerates all 80 re-exported names (was 3). `from truememory import *` and IDE autocomplete now see the full public surface.
- **`Memory.add("")`** / whitespace-only content now emits a `UserWarning` and returns a skip-marker (`{id: None, created_at: None, ...}`) instead of silently inserting a useless row.

### Changed — user-visible behavior

- **`_load_config`** corruption handling: corrupt `~/.truememory/config.json` is now renamed to `config.json.corrupt.<unix-ts>` (preserving any API keys for recovery) with a stderr warning, instead of silently returning `{}` and losing the user's tier + keys.
- **LLM client construction** in MCP server: bad API key / rate-limit / import failure now logs at WARNING and stores `_llm_last_error[provider]` (consumed by `stats.health.hyde_llm`) instead of silently falling through to no-HyDE mode. Pro tier no longer silently degrades to Base without signal.
- **Reranker load failure** in MCP server: now logs at WARNING with an install hint (`pip install truememory[gpu]`) and stores `_reranker_last_error`. Throttled to log once per distinct error to avoid log spam on the per-search code path.
- **`truememory_configure` re-embed**: now rebuilds BOTH `vec_messages` AND `vec_messages_sep` (previously only the completion table was rebuilt, leaving separation search silently empty after any tier switch). Exceptions are surfaced as `rebuild_error` + `warning` fields in the response payload instead of being swallowed. `_memory = None` now lives in a `finally` so the next call always gets a fresh instance, even on failure.
- **`HF_HUB_OFFLINE`** environment restoration in `truememory_configure` is now wrapped in `try/finally` so any raise mid-configure (e.g., `set_embedding_model` on a removed model) doesn't leak offline-mode-disabled state for the process lifetime.
- **`sqlite-vec` load failure** in `engine.open()` upgraded from DEBUG to WARNING with a link to platform notes. FTS-only fallback is unchanged — just no longer silent.
- **`_setup_claude`**: all four `claude mcp` CLI calls now have `timeout=30` and report `TimeoutExpired` to stderr instead of hanging forever if the binary stalls (auth prompt, blocked network, deadlock).
- **`pip install truememory[gpu]`** subprocess in the setup wizard now has `timeout=600` (10 min) with a stderr message and Edge-tier fallback on timeout.
- **Stop hook**: now bounds concurrent ingest spawns via `TRUEMEMORY_INGEST_SPAWN_CAP` (default 2). Over-cap events queue to `~/.truememory/backlog/<session_id>.json` for a later session to drain. When Popen itself fails, the hook writes a backlog marker instead of falling back to synchronous inline ingestion (which had been blocking Claude Code's shutdown for 10–60s).
- **`PRAGMA busy_timeout`** is now a single source of truth (`storage.DEFAULT_BUSY_TIMEOUT_MS = 10_000`) across `create_db` and `pipeline._set_busy_timeout`. Pre-fix the two paths used 5 s and 10 s asymmetrically, surfacing as sporadic `database is locked` under contention.
- **Claude Desktop config path** in `_setup_claude` now resolves per-platform (macOS / Windows / Linux+BSD) instead of hard-coding the macOS `~/Library/Application Support/` path. Linux and Windows users with Desktop installed are no longer reported as "not detected".
- **Windows config-file secrets warning**: when saving an API key on Windows, `_save_config` prints a stderr warning pointing to the env-var route. POSIX `chmod(0o600)` calls are no-ops on Windows, so the file inherits parent-directory ACL and may be readable by other local users.

### Added — CI + packaging

- New **`build-check`** job in `.github/workflows/ci.yml` running `python -m build && twine check --strict` on every PR (single Python 3.12 cell). Packaging regressions (missing `py.typed`, broken README rendering, sdist contents) now fail CI instead of release day.
- New **`.github/dependabot.yml`** with weekly pip + github-actions scans.
- sdist now excludes `benchmarks/` entirely (was shipping ~60 KB of Modal bench source, amplifying doc-drift findings to PyPI source-download users).

### Fixed — concurrency / correctness

- **`_get_llm_fn`** now uses double-checked locking around first-call construction (`_llm_cache_lock`). Prevents duplicate LLM-client construction on parallel first-searches.
- **`_parallel_search._run_query`** now uses `with Memory(path=db_path) as thread_m:` instead of manual `try/finally m.close()`. Closes an interrupt-safety hole on KeyboardInterrupt between construction and the `try:` block.

### Dependencies

- `sentence-transformers>=3.0.0,<4.0` → `>=3.0.0,<6.0` (was forcing 3.4.1 when 5.4.1 was current; CrossEncoder API is stable across the range).
- `pytest>=7.0,<9.0` → `>=7.0,<10.0` (dev-only; latest is 9.0.3).

### Documentation

- `vector_search.py` module docstring rewritten from "Model2Vec-only" to the v0.4.0 tier-aware resolution story.
- `bench_truememory_base.py` header corrected (was a copy-paste from Pro claiming "Pro Tier +HyDE"; now accurately describes the Base tier @ 91.5% with HyDE off).
- README chart-regeneration caveat moved above the first benchmark chart so it's seen before the stale visuals.
- README `benchmarks/` / `install.sh` / `CLAUDE.md.example` / `LICENSE` links rewritten from relative paths to absolute `github.com/buildingjoshbetter/TrueMemory/blob/main/...` URLs (were 404'ing on PyPI's renderer).
- `BENCHMARK_RESULTS.md` no longer cites gitignored `_working/` paths as authoritative sources.
- `_nm_` → `_tm_` identifier cleanup in 4 Modal bench scripts (pre-rebrand neuromem namespace).
- CHANGELOG rewrite for the branded migration-error in the v0.4.0 upgrade notes.

### Security

- Replaced dead `security@sauronlabs.ai` contact with the repo's maintainer address.

### Deprecated

- **`load_messages(conn, messages)`** in `truememory.storage`. Use `bulk_replace_messages` for the same (destructive) behavior, or `insert_message` per-row if you actually want to append. The deprecated alias will be removed in a future release.

### Migration notes

- **If you're upgrading a v0.3.0 Pro database**: the first open now raises `TrueMemoryMigrationError` (was: raw `OperationalError: Dimension mismatch`). Options are `truememory_configure(tier=...)` to re-embed in place, or delete `~/.truememory/memories.db` to start fresh. The error message includes both options.
- **If you call `load_messages`**: start moving to `bulk_replace_messages`. The old name still works for now but emits `DeprecationWarning`.
- **If you pattern-match on `truememory_stats()` response shape**: the new `health` key is additive; existing keys are unchanged.

## [0.4.0] - 2026-04-21

Paper-aligned Edge / Base / Pro tier realignment. Pro no longer uses the cherry-picked Qwen3 1024d + mxbai-rerank-large-v1 configuration. The three tiers now match the paper §2.0 spec exactly:

| Tier | Embedder | Reranker | HyDE | LoCoMo target |
|------|----------|----------|------|---------------|
| Edge | Model2Vec potion-base-8M @ 256d | `cross-encoder/ms-marco-MiniLM-L-6-v2` | off | 90.1% |
| Base (Default) | `Qwen/Qwen3-Embedding-0.6B` @ 256d Matryoshka | `Alibaba-NLP/gte-reranker-modernbert-base` | off | 91.5% |
| Pro (+HyDE) | `Qwen/Qwen3-Embedding-0.6B` @ 256d Matryoshka | `Alibaba-NLP/gte-reranker-modernbert-base` | on | 91.8% |

### Breaking changes
- **Pro tier reconfigured.** The v0.3.0 "Pro" combo (Qwen3 @ native 1024d + `mixedbread-ai/mxbai-rerank-large-v1` + HyDE on) is replaced with the paper-§2.0 +HyDE combo (Qwen3 @ 256d Matryoshka + `Alibaba-NLP/gte-reranker-modernbert-base` + HyDE on). The authoritative 56-grid sweep measured the v0.3.0 Pro config at 90.7% — below the v0.4.0 Base tier (91.5%, HyDE off). The v0.4.0 Pro reaches 91.8% with HyDE on.
- **`TRUEMEMORY_EMBED_MODEL=qwen3` removed.** The bare internal name `qwen3` (which meant "Qwen3 at native 1024d") is gone. Setting it — via env var or `set_embedding_model("qwen3")` — raises `ValueError` at startup. Migrate to `TRUEMEMORY_EMBED_MODEL=pro` (tier alias) or `=qwen3_256` (internal name). Both map to the same paper-aligned Qwen3 @ 256d Matryoshka config.
- **Base tier meaning changed.** In v0.3.0, "Base" meant Model2Vec + MiniLM-L-6-v2 at 88.2% LoCoMo (the old leaderboard number — the same config scores 90.1% on the authoritative 56-grid harness used for v0.4.0). That config is now called **Edge**. The new **Base** tier is Qwen3 @ 256d Matryoshka + gte-reranker-modernbert (HyDE off) at 91.5%.

### Added
- **Edge tier** formalized (was previously called Base in v0.3.0). CPU-only, ~30 MB install, ~30M total parameters, 90.1% LoCoMo target. Runs on any machine with Python 3.10+ and 512 MB RAM.
- **Base tier** (middle tier, GPU recommended): same embedder + reranker as Pro, HyDE off. 91.5% LoCoMo target. No LLM API key required.
- Matryoshka truncation support for Qwen3-Embedding-0.6B via `SentenceTransformer(..., truncate_dim=256)` — this is what the paper-§2.0 Base and Pro tiers use under the hood.
- New bench scripts `benchmarks/locomo/scripts/bench_truememory_edge.py` (Edge), `bench_truememory_base.py` (Base, new content), and an updated `bench_truememory_pro.py`.
- New unit tests in `tests/test_tier_aliases.py` covering all three aliases plus a negative test asserting the `qwen3` internal name is gone.

### Removed
- Internal embedding model name `qwen3` (1024d native). Use `pro` (tier alias) or `qwen3_256` (internal name) instead.
- Default reranker `mixedbread-ai/mxbai-rerank-large-v1` for the Pro tier. Users who explicitly set it via `get_reranker(model_name="...")` can continue to; only the Pro tier's built-in default has changed.

### Migration guide

If you were using TrueMemory 0.3.0:

1. **Upgrading the package.** `pip install -U truememory`. The first run will download ~1.5 GB of model weights (Qwen3-Embedding-0.6B + gte-reranker-modernbert) if you pick Base or Pro. Edge remains ~30 MB.
2. **If you had `TRUEMEMORY_EMBED_MODEL=qwen3` set.** Change it to `TRUEMEMORY_EMBED_MODEL=pro` (recommended) or `TRUEMEMORY_EMBED_MODEL=qwen3_256`. The old value now raises `ValueError` on startup.
3. **If you picked "Base" in v0.3.0 expecting Model2Vec.** That tier is now called **Edge**. Set `TRUEMEMORY_EMBED_MODEL=edge` to preserve the old behavior, or pick Edge at the first-run setup prompt.
4. **Embedding table shape.** All three v0.4.0 tiers produce 256-dim vectors, so the sqlite-vec virtual table layout is unchanged versus an Edge-tier v0.3.0 database. Upgrading from a v0.3.0 Pro (1024d) database to v0.4.0 Pro (256d) requires a fresh ingestion — the vector dimensions no longer match.
5. **Benchmark reproduction.** Three scripts replace the old two: `bench_truememory_edge.py`, `bench_truememory_base.py`, `bench_truememory_pro.py`. Each is self-contained for Modal. Smoke-run them with `--smoke` before the full 1540-question run.

## [0.3.0] - 2026-04-11

### Changed
- **Renamed the package from `neuromem` / `neuromem-core` to `truememory`.** The import path, PyPI dist name, console scripts (`truememory-mcp`, `truememory-ingest`), environment variables (`TRUEMEMORY_*`), runtime data directory (`~/.truememory/`), MCP server slug (`truememory`), and wire-format tags (`<truememory-context>`) all moved to the new name. See MIGRATION notes below if you're upgrading from 0.2.x.
- Moved `mcp[cli]` and `httpx` from the `[mcp]` optional extra into the core `dependencies` list so `pip install truememory && truememory-mcp --setup` works on the first run. The `[mcp]` extra is kept as a no-op alias for backwards compatibility.

### Fixed
- Aligned version string across `pyproject.toml`, `truememory/__init__.py`, `truememory/ingest/__init__.py`, `CITATION.cff`, and the README bibtex. Previously four of these were stuck at 0.2.0 while the code was tagged 0.2.2.
- Rebranded the `_SAURON_BANNER` ASCII splash in `truememory/ingest/cli.py` that was missed by the initial sed pass (its letters were separated by spaces, which evaded the contiguous `neuromem` regex).
- Rebranded all 10 chart PNGs in `assets/charts/` (hero-banner, leaderboard, accuracy-vs-cost, category-radar, category-heatmap, category-grouped-bars, cost-per-answer, latency-comparison, hardware-matrix, eval-pipeline). Re-rendered from the original design HTML sources with TrueMemory branding; coloring, typography, grid, grain, and layout preserved exactly.
- Fixed two label overlaps in the parallel-category-coordinates chart (Temporal axis EverMemOS label was colliding with TrueMemory Pro's dot; Single-hop axis Mem0 label was sitting on the descending line toward Multi-hop).

### Migration from 0.2.x (`neuromem-core`)
- Uninstall the old package: `pip uninstall neuromem-core`
- Install the new one: `pip install truememory`
- Update imports: `from neuromem import Memory` → `from truememory import Memory`
- Update class references: `NeuromemEngine` → `TrueMemoryEngine`
- Update environment variables: `NEUROMEM_*` → `TRUEMEMORY_*`
- Your existing data at `~/.neuromem/` is not automatically migrated — either move it manually to `~/.truememory/` or start fresh
- Re-register the MCP server in Claude Code: `claude mcp remove neuromem && truememory-mcp --setup`

## [0.2.0] - 2026-04-03

### Added
- 9 data visualizations (hero banner, leaderboard bar chart, accuracy vs cost scatter, cost per answer, category radar, latency, hardware matrix, eval pipeline diagram, per-category grouped bars)
- `assets/charts/` directory with chart HTML sources and rendered PNGs
- `benchmarks/` directory with full LoCoMo evaluation against 8 memory systems
- Independent benchmark scripts for each competitor (self-contained, reproducible on Modal)
- Complete result JSONs with per-question answers, judge votes, and latency data
- BENCHMARK_RESULTS.md with cost analysis, latency comparison, and hardware requirements
- LICENSE file (Apache 2.0)
- CHANGELOG.md

### Changed
- Visual README overhaul: hero banner, emoji section headers, highlight badges, embedded charts
- License changed from MIT to Apache 2.0
- Updated README benchmark section: 8 competitors (was 4), best scores across runs
- TrueMemory Pro: 91.5% on LoCoMo
- TrueMemory Base: 88.2% on LoCoMo

### Benchmark Results
- 8 systems evaluated on LoCoMo (1,540 questions each, 12,320 total) with identical answer model, judge, scoring, top-k, and prompt
- TrueMemory Pro: 91.5%, TrueMemory Base: 88.2%
- All runs completed with zero API errors

## [0.1.3] - 2026-03-28

### Added
- TRUEMEMORY_EMBED_MODEL environment variable for tier selection
- GPU optional dependency (`pip install truememory[gpu]`)

## [0.1.2] - 2026-03-27

### Added
- Incremental entity profile building for MCP/add() workflow

## [0.1.1] - 2026-03-26

### Added
- Initial release of truememory
- 6-layer memory pipeline: FTS5, vector search, temporal, salience, personality, consolidation
- Base tier (Model2Vec) and Pro tier (Qwen3) embedding support
- MCP server for Claude integration
- Simple Memory API (Mem0-compatible interface)
