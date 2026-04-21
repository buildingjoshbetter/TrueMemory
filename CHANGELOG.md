# Changelog

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
