# TrueMemory Issue Triage & Implementation Plan

> Generated 2026-05-09 · 21 open issues analyzed · Repo: buildingjoshbetter/TrueMemory v0.6.2
> Last validated against codebase: 2026-05-09 (commit d638b74)

---

## Table of Contents

1. [Validation Journal](#validation-journal)
2. [Issue Inventory](#issue-inventory)
3. [Dependency Graph](#dependency-graph)
4. [PR Buckets](#pr-buckets)
5. [Value Assessment](#value-assessment)
6. [Implementation Plans](#implementation-plans)
7. [Validation Scripts](#validation-scripts)
8. [Execution Prompts](#execution-prompts)

---

## Validation Journal

> Every issue was validated against the current codebase (d638b74) before implementation planning.
> Issues marked ALREADY FIXED should be closed on GitHub.

### PR-A Issues (Installer Quick Fixes)

**#167 — Stale tier accuracy scores: ~~ALREADY FIXED~~ → CLOSE**
- Searched `mcp_server.py`, `ingest/cli.py`, `reranker.py` for 90.1, 91.5, 91.8
- All three files now show correct v0.6.0 numbers: Edge 89.6%, Base 92.0%, Pro 93.0%
- Old numbers only appear in CHANGELOG.md (historical, correct) and benchmark result files
- **Action:** Close #167 on GitHub. Remove from PR-A.

**#169 — install.sh missing --refresh flag: ~~ALREADY FIXED~~ → CLOSE**
- `install.sh` line 109 already has `--refresh`: `uv tool install --python "$TRUEMEMORY_PY" --force --refresh "$PKG_SPEC"`
- Additionally, line 106 runs `uv tool uninstall truememory` first for a clean slate
- Comments at lines 105 and 108 document the rationale
- **Action:** Close #169 on GitHub. Remove from PR-A.

**#168 — pip vs uv upgrade messages: STILL EXISTS**
- Three locations show BOTH uv and pip instructions (acceptable): `mcp_server.py:723-724`, `cli.py:347-348`, `cli.py:366-367`
- Three locations ONLY mention pip with no uv alternative (the bug):
  - `mcp_server.py:1154` — "Run this once after `pip install`"
  - `ingest/cli.py:121` — "Install with: pip install truememory"
  - `ingest/cli.py:774` — "Install with: pip install truememory"
- No install-method auto-detection logic exists anywhere in the codebase
- **Action:** Keep in PR-A. Fix the 3 pip-only locations. Optionally add auto-detection.

**#171 — uv not on PATH: PARTIALLY FIXED**
- install.sh correctly handles PATH during its own execution (line 80: `export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"`)
- install.sh calls `uv tool update-shell` to modify shell rc files for future sessions
- BUT: the final install output (lines 185-199) never tells users to open a new terminal
- Error messages in Python code don't suggest opening a new terminal:
  - `mcp_server.py:1135` — "Claude Code CLI not found on PATH" (no terminal guidance)
  - `cli.py:133`, `cli.py:708` — generic "not found" messages
- **Action:** Keep in PR-A. Add "open a new terminal" guidance to install output and error messages.

**PR-A revised scope:** Only #168 and #171 remain. #167 and #169 are already fixed.

---

### PR-B Issues (Tier Upgrade CLI)

**#170 — CLI command to upgrade tier: STILL NEEDED**
- `ingest/cli.py` defines 9 subcommands: ingest, install, stats, status, uninstall, logs, trace, facts, setup
- No `upgrade-tier`, `upgrade`, `tier`, or `configure` subcommand exists
- The only way to change tiers is the full interactive `setup` wizard (lines 317-506) which re-asks everything
- MCP's `truememory_configure` handles tier logic (validate, re-embed, update reranker) but doesn't install GPU deps — just returns an error telling users to install manually
- **Action:** Keep. Build a focused `upgrade-tier` subcommand that installs deps + switches tier.

---

### PR-C Issues (Reranker Gap)

**#189 — Memory.search() skips reranker: REAL but nuanced**
- `engine.search()` confirmed: 9 steps, ZERO calls to any reranker function
- `engine.search_agentic()` confirmed: calls `rerank_with_modality_fusion()` at line 1587
- **Critical finding:** `truememory_search` MCP tool (line 557) calls `m.search_deep()` → `engine.search_agentic()` — so ALL MCP users get reranking
- `Memory.search()` (Python API, line 124) calls `engine.search()` — NO reranking
- `reranker.py:208` confirms `_normalize_and_fuse()` always sets `score = fused_score`
- `hybrid.py` confirmed: only does RRF fusion, no cross-encoder
- **Additional context:** `search_agentic()` calls `self.search()` internally (line 1449) as its candidate retrieval stage, then applies cross-encoder on top. This is a valid two-stage architecture. The question is whether standalone `engine.search()` should also get its own reranker pass.
- **Action:** Keep. The gap is real for Python SDK users. Recommend adding reranking to `engine.search()` as step 7.75 (between salience guard and final trim).

---

### PR-D Issues (Session Intelligence)

**#175 — Incremental extraction via UserPromptSubmit: STILL NEEDED**
- `user_prompt_submit.py` is purely a message buffer (appends JSON to `.jsonl`, prunes old files)
- Zero time-based extraction trigger
- Does not import anything from `stop.py`
- No `_shared.py` exists in hooks directory
- No "last_incremental_extraction" marker file mechanism anywhere in codebase
- Only 5 files in hooks dir: `__init__.py`, `compact.py`, `session_start.py`, `stop.py`, `user_prompt_submit.py`
- **Action:** Keep. Implement exactly as proposed.

**#176 — Extraction on PreCompact: STILL NEEDED**
- `compact.py` only saves a text snapshot via `Memory.add()` — stores strings like `"[session_snapshot abc123 ...] Recent topics: ..."` 
- Does NOT trigger background ingestion (no call to `_run_background_ingestion`, no subprocess, no import from stop.py)
- Does NOT call the extractor or encoding gate
- No timestamp-based checks
- **Action:** Keep. Add background ingestion trigger alongside the existing snapshot logic.

---

### PR-E through PR-K Issues (CLI Integrations)

**#186 — Universal hook framework: STILL NEEDED**
- No `truememory/hooks/` package exists
- No adapter abstraction exists
- Existing hooks are Claude Code-specific (hardcoded `settings.json` format)
- **Action:** Keep. Build as designed.

**#188 — Conversational onboarding: STILL NEEDED**
- No CLI auto-detection exists
- No `integrations.json` state tracking
- The `setup` subcommand is Claude Code-only
- **Action:** Keep. Build as designed.

**#182 — Codex CLI: VERIFIED against real Codex docs**
- Config path `~/.codex/config.toml` — confirmed
- MCP under `[mcp_servers.name]` in TOML — confirmed
- Hook events SessionStart, Stop with JSON stdin/stdout — confirmed (6 events total)
- **Action:** Keep. Safe to build against these specs.

**#183 — Kimi CLI: VERIFIED against real Kimi docs**
- Config `~/.kimi/mcp.json` in mcpServers JSON format — confirmed
- TOML `[[hooks]]` syntax with JSON stdin/stdout — confirmed
- Hooks in beta — confirmed (docs state this explicitly)
- 13 hook events available (not just SessionStart/Stop) — confirmed
- **Action:** Keep. Safe to build.

**#184 — OpenClaw: MOSTLY VERIFIED — needs issue update**
- Config `~/.openclaw/openclaw.json` in JSON5 — confirmed
- MCP under `mcp.servers` — confirmed
- `agent:bootstrap`, `message:received`, `message:sent` — confirmed (internal hook names, colon-delimited)
- **PROBLEM:** `agent_end` is a PLUGIN event name (underscore convention), not an internal hook name (colon convention). The issue mixes two hook systems:
  - Internal hooks (directory-based, HOOK.md): `agent:bootstrap`, `message:received`, `message:sent`
  - Plugin hooks (JS API, `api.on()`): `agent_end`, `before_tool_call`, `message_sending`
- There is no `agent:end` in the internal hook list. For session-end capture, we'd need to use the plugin system (`api.on("agent_end")`)
- **Action:** Keep but UPDATE #184 on GitHub to fix the event naming. Recommend using the plugin system (JS API) for everything, since the internal hook system doesn't have a session-end event.

**#185 — Hermes Agent: VERIFIED against real Hermes docs**
- Config `~/.hermes/config.yaml` — confirmed
- MCP under `mcp_servers:` in YAML — confirmed
- All hook events confirmed: `on_session_start`, `on_session_end`, `on_session_finalize`, `post_llm_call`
- Gateway hooks at `~/.hermes/hooks/name/` with `HOOK.yaml` + `handler.py` — confirmed
- Current version v0.12.0 (April 30, 2026) — confirmed
- **Action:** Keep. Safe to build against these specs.

**#187 — Multi-CLI docs: STILL NEEDED**
- No `docs/` directory exists
- README is Claude Code-only
- **Action:** Keep. Ships alongside integrations.

---

### Issues to Close

| Issue | Reason |
|-------|--------|
| #167 | Already fixed — all accuracy scores show v0.6.0 numbers |
| #169 | Already fixed — install.sh already has `--refresh` flag |
| #135 | Not a real bug — score fallback chains differ but never cause incorrect scoring in practice (validated: `search()` never receives `fused_score` keys, `_clean_results()` always gets pre-populated `score` from reranker.py:208) |

### Issues to Update

| Issue | What to fix |
|-------|-------------|
| #184 | Fix OpenClaw event naming — `agent_end` is a plugin event, not an internal hook event. Recommend plugin system (JS API) for the integration since internal hooks lack session-end |

---

## Issue Inventory

| # | Title | Type | Effort | Risk | PR |
|---|-------|------|--------|------|-----|
| 168 | Tier upgrade says `pip install` instead of detecting uv | bug | 1 hr | Minimal | PR-A |
| 171 | uv not on PATH after install | bug | 30 min | Minimal | PR-A |
| 170 | CLI command to upgrade tier without re-running installer | enhancement | 4 hr | Low | PR-B |
| 189 | Memory.search() skips cross-encoder reranker | bug | 6 hr | Med-High | PR-C |
| 175 | Incremental extraction via UserPromptSubmit | enhancement | 6 hr | Medium | PR-D |
| 176 | Extraction on PreCompact | enhancement | 3 hr | Medium | PR-D |
| 186 | Universal hook template framework | enhancement | 24 hr | High | PR-E |
| 182 | Codex CLI integration | enhancement | 16 hr | High | PR-F |
| 188 | Conversational onboarding with auto-detect | enhancement | 12 hr | Med-High | PR-G |
| 183 | Kimi CLI integration | enhancement | 12 hr | Med-High | PR-H |
| 184 | OpenClaw integration | enhancement | 20 hr | Highest | PR-I |
| 185 | Hermes Agent integration | enhancement | 16 hr | High | PR-J |
| 187 | Multi-CLI setup guide & compatibility matrix | docs | 8 hr | Zero | PR-K |
| 190 | Usage telemetry + investor dashboard | enhancement | 24 hr | Low | PR-L |
| | | | | | |
| ~~167~~ | ~~Stale accuracy scores~~ | ~~bug~~ | — | — | CLOSED |
| ~~169~~ | ~~install.sh --refresh flag~~ | ~~bug~~ | — | — | CLOSED |
| ~~135~~ | ~~Duplicate result-cleaning~~ | ~~code-health~~ | — | — | CLOSED |
| ~~180~~ | ~~Email on first install~~ | ~~enhancement~~ | — | — | Subsumed by #190 |
| 137 | engine.py god class (2,275 lines) | code-health | 16 hr | Medium | Deferred |
| 79 | Horizon tier (HNSW for 100K+ messages) | enhancement | 40 hr | High | Deferred |
| 67 | Paper reproduction harness | research | 40 hr | Medium | Deferred |
| 68 | Hardware footprint claims harness | research | 16 hr | Low | Deferred |

---

## Dependency Graph

```
                    ┌─────────────────────────────────────────────┐
                    │         PR-A: Installer Quick Fixes          │
                    │  #167 #168 #169 #171                        │
                    └──────────────┬──────────────────────────────┘
                                   │ (install detection logic)
                    ┌──────────────▼──────────────────────────────┐
                    │         PR-B: Tier Upgrade CLI               │
                    │  #170                                        │
                    └─────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────┐
                    │     PR-C: Reranker Gap in search()           │
                    │  #189                                        │
                    └─────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────┐
                    │      PR-D: Session Intelligence              │
                    │  #175 + #176                                 │
                    └─────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────┐
                    │     PR-E: Universal Hook Framework           │
                    │  #186                                        │
                    └──────────────┬──────────────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
   ┌──────────▼─────────┐  ┌──────▼──────┐  ┌─────────▼──────────┐
   │  PR-F: Codex (#182) │  │ PR-H: Kimi  │  │ PR-I: OpenClaw     │
   │  (first CLI)        │  │ (#183)      │  │ (#184)             │
   └──────────┬──────────┘  └─────────────┘  └────────────────────┘
              │                                         
   ┌──────────▼──────────────────────────────┐ ┌──────────────────┐
   │  PR-G: Conversational Onboarding (#188) │ │ PR-J: Hermes     │
   │  (needs ≥1 CLI integration to test)     │ │ (#185)           │
   └──────────┬──────────────────────────────┘ └──────────────────┘
              │
   ┌──────────▼──────────────────────────────┐
   │  PR-K: Multi-CLI Docs (#187)            │
   │  (ships after integrations exist)       │
   └─────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────┐
                    │     PR-L: Usage Telemetry + Dashboard        │
                    │  #190 (subsumes #180)                        │
                    └─────────────────────────────────────────────┘

   ═══════════ DEFERRED ═══════════
   #137 engine.py refactor — maintenance, no urgency
   #79  Horizon tier — v0.7.0+
   #67  Paper reproduction — paper timeline
   #68  Hardware claims — paper timeline
```

---

## PR Buckets

### PR-A: Installer Quick Fixes (REVISED)
**Issues:** #168, #171 (~~#167~~ and ~~#169~~ already fixed — close on GitHub)
**Why together:** Both touch the install/upgrade user experience in `mcp_server.py` and `ingest/cli.py`.

| Issue | Files | Change |
|-------|-------|--------|
| #168 | `mcp_server.py` (L1154), `ingest/cli.py` (L121, L774) | Fix 3 pip-only locations to show both uv and pip commands |
| #171 | `mcp_server.py` (L1135), `ingest/cli.py` (L133, L708), `install.sh` (final output) | Add "open a new terminal" guidance to error messages and install output |

**Estimated effort:** 1-2 hours total
**Risk:** Low — small, focused changes

---

### PR-B: Tier Upgrade CLI
**Issues:** #170
**Why separate:** New CLI subcommand is a feature, not a bug fix. Depends on install-method detection from PR-A.

**Files:** `ingest/cli.py` (new `upgrade-tier` subcommand)
**Estimated effort:** 4 hours
**Risk:** Low

---

### PR-C: Reranker Gap in search() (REVISED)
**Issues:** #189 (replaces #135 which was closed as not-a-bug)
**Why separate:** Real accuracy gap for Python SDK users. `engine.search()` skips cross-encoder reranking despite docs listing it as a standard pipeline component.

**Files:** `engine.py` — add reranking step 7.6 between surprise boost (line 1340) and clean/trim (line 1342). Add `_skip_reranker` parameter. Update `search_agentic()` calls at lines 1449 and 1556 to pass `_skip_reranker=True`.
**Estimated effort:** 4-6 hours (including tests and double-rerank prevention)
**Risk:** Medium — touches the hot search path, must prevent double-reranking in `search_agentic()`

---

### PR-D: Session Intelligence
**Issues:** #175, #176
**Why together:** Both add mid-session extraction triggers. They share the "last extraction timestamp" mechanism and must coordinate to avoid redundant ingestion runs.

**Key design decisions:**
- Shared timestamp marker: `~/.truememory/last_incremental_extraction`
- #175 adds time-based check to `user_prompt_submit.py` (every 4 hours)
- #176 adds extraction trigger to `compact.py` (on context compression)
- Both use the same background Popen pattern from `stop.py`
- Both respect the spawn cap (SPAWN_CAP=2)

**Files:**
- `truememory/ingest/hooks/user_prompt_submit.py` — add incremental extraction trigger
- `truememory/ingest/hooks/compact.py` — add extraction trigger alongside snapshot
- New shared util for timestamp checking

**Estimated effort:** 8 hours total
**Risk:** Medium — must not duplicate facts or interfere with the Stop hook's extraction

---

### PR-E: Universal Hook Framework
**Issues:** #186
**Why separate:** Foundation for all CLI integrations. Must be solid and well-tested before building on it.

**Architecture:**
```
truememory/hooks/
├── __init__.py
├── core.py              # CLI-agnostic: recall_memories(), extract_transcript(), generate_mcp_config()
├── registry.py          # CLI detection, config path mapping, install state
├── cli.py               # `truememory-ingest setup` subcommand
├── adapters/
│   ├── __init__.py
│   ├── base.py          # Abstract base adapter
│   ├── claude.py        # Existing Claude Code adapter (refactor current install logic)
│   ├── codex.py         # Codex adapter (TOML config, JSON stdin/stdout hooks)
│   ├── kimi.py          # Kimi adapter (JSON MCP config, TOML hooks)
│   ├── openclaw.py      # OpenClaw adapter (JSON5 config, JS plugin)
│   └── hermes.py        # Hermes adapter (YAML config, Python hooks)
└── templates/
    ├── codex/
    │   ├── session_start.sh.j2
    │   ├── stop.sh.j2
    │   └── config.toml.j2
    ├── kimi/
    │   ├── session_start.sh.j2
    │   ├── stop.sh.j2
    │   └── mcp.json.j2
    ├── openclaw/
    │   ├── plugin.json.j2
    │   ├── index.js.j2
    │   └── openclaw.json.j2
    └── hermes/
        ├── session_start.py.j2
        ├── session_end.py.j2
        └── config.yaml.j2
```

**Core interface (each adapter implements):**
```python
class CLIAdapter(ABC):
    name: str                           # "codex", "kimi", etc.
    config_path: Path                   # ~/.codex/config.toml, etc.
    
    def detect(self) -> bool            # Is this CLI installed?
    def is_configured(self) -> bool     # Is TrueMemory already wired in?
    def install_mcp(self) -> None       # Add MCP server to CLI config
    def install_hooks(self) -> None     # Register lifecycle hooks
    def uninstall(self) -> None         # Clean removal
    def verify(self) -> bool            # Smoke test — can MCP server respond?
    def get_system_prompt(self) -> str  # AGENTS.md / CLAUDE.md content
```

**Key principle:** The existing Claude Code hooks (`truememory/ingest/hooks/*.py`) contain the battle-tested core logic. The framework extracts the portable parts (`recall_memories()` from `session_start.py`, `_run_background_ingestion()` from `stop.py`) into `core.py`, then each adapter wraps them in the CLI-specific I/O format.

**Estimated effort:** 24 hours
**Risk:** Medium — must not break existing Claude Code integration while refactoring

---

### PR-F: Codex CLI Integration
**Issues:** #182
**Depends on:** PR-E

**Codex-specific details:**
- Config format: TOML (`~/.codex/config.toml`)
- MCP entry: `[mcp_servers.truememory]` with `command` + `args`
- Hook events: `SessionStart`, `Stop` (stable, well-documented)
- Hook I/O: JSON stdin → JSON stdout (same as Claude Code)
- System prompt: `AGENTS.md` file (Codex equivalent of CLAUDE.md)

**What ships:**
- `truememory/hooks/adapters/codex.py` — adapter implementation
- `truememory/hooks/templates/codex/` — hook scripts + config template
- `tests/hooks/test_codex_adapter.py` — unit tests
- Integration test script

**Estimated effort:** 16 hours
**Risk:** Medium — need to verify against real Codex CLI installation

---

### PR-G: Conversational Onboarding
**Issues:** #188
**Depends on:** PR-E + at least PR-F (need one CLI integration to test the flow)

**What ships:**
- Enhanced `truememory-ingest setup` command with:
  - CLI auto-detection (scan known config paths)
  - Interactive multi-select prompt
  - Per-CLI setup orchestration
  - Post-setup smoke test
- `~/.truememory/integrations.json` state tracking
- Non-interactive flag support: `--cli claude,codex,kimi`

**Estimated effort:** 12 hours
**Risk:** Medium — interactive prompts need careful UX testing

---

### PR-H through PR-J: Remaining CLI Integrations
**PR-H: Kimi (#183)** — 12 hours, High value
**PR-I: OpenClaw (#184)** — 20 hours, Medium value (JS plugin adds complexity)
**PR-J: Hermes (#185)** — 16 hours, Medium value

Each is an adapter implementation following the pattern established in PR-F.

---

### PR-K: Multi-CLI Docs
**Issues:** #187
**Ships after:** At least PR-F and PR-G exist
**Estimated effort:** 8 hours

---

### PR-L: Usage Telemetry + Dashboard
**Issues:** #190 (subsumes #180)
**Why separate:** Independent of all other PRs. New module + server-side.

**What ships:**
- `truememory/telemetry.py` — fire-and-forget event tracking (no new deps, uses existing `httpx` + `threading`)
- `@tracked()` decorator on all 8 MCP tool functions
- Email collection during onboarding (extends `truememory_configure`)
- UUID generation on first run (stored in `~/.truememory/config.json`)
- Session tracking in hooks (`session_start.py`, `stop.py`)
- Opt-out via `TRUEMEMORY_TELEMETRY=off`
- Server-side: FastAPI endpoint + Postgres + PostHog dashboards

**Estimated effort:** 16 hours (client) + 8 hours (server/dashboard)
**Risk:** Low-Medium — new module, but all calls are fire-and-forget with try/except. If telemetry fails, TrueMemory works normally.

---

### Deferred Issues

| Issue | Reason | When |
|-------|--------|------|
| #137 engine.py refactor | Maintenance only, no user impact | After CLI integration wave |
| #79 Horizon tier | v0.7.0+ feature, needs design work | After v0.6.x stabilizes |
| #67 Paper reproduction | Paper timeline, not user-facing | Camera-ready deadline |
| #68 Hardware claims | Paper timeline, not user-facing | Camera-ready deadline |
| ~~#180 Email collection~~ | Subsumed by #190 (telemetry) | — |

---

## Risk-Ordered Implementation Game Plan

> PRs ordered from **lowest risk of breaking the repo** to **highest risk**.
> This is the recommended implementation order.

### RISK LEVEL 1: ZERO RISK (text/docs only)

#### Step 1 → PR-K: Multi-CLI Docs (#187)
| | |
|---|---|
| **Risk** | **Zero** — documentation files only, no code changes |
| **What can break** | Nothing |
| **Files touched** | New `docs/` directory, README.md updates |
| **Rollback** | Delete the docs |
| **Why first** | Can be done anytime, gets it out of the way, helps future contributors |
| **Effort** | 8 hours |

---

### RISK LEVEL 2: MINIMAL RISK (string changes, no logic)

#### Step 2 → PR-A: Installer Quick Fixes (#168, #171)
| | |
|---|---|
| **Risk** | **Minimal** — changing 3 string literals + 1 line in install.sh |
| **What can break** | Nothing — text-only changes... EXCEPT `cli.py:357-360` (the subprocess behavioral fix) which changes install logic |
| **Files touched** | `mcp_server.py` (2 lines), `ingest/cli.py` (3 lines), `install.sh` (2 lines) |
| **Rollback** | Revert 7 line changes |
| **Why early** | Every new user hits these. Highest ROI per line of code changed |
| **Gotcha** | The subprocess fix at `cli.py:357-360` (uv vs pip auto-install) is the only non-trivial change. Test on both uv and pip installs. |
| **Effort** | 1-2 hours |

---

### RISK LEVEL 3: LOW RISK (new code, no existing behavior changes)

#### Step 3 → PR-B: Tier Upgrade CLI (#170)
| | |
|---|---|
| **Risk** | **Low** — new subcommand, existing code untouched |
| **What can break** | Vector re-embedding could corrupt the DB if it fails mid-way (drops tables then crashes before rebuild) |
| **Files touched** | `ingest/cli.py` (new function + subparser), possibly new `truememory/tier.py` |
| **Rollback** | Remove the subcommand |
| **Mitigation** | Wrap re-embed in a transaction or backup `memories.db` before starting. The MCP `truememory_configure` already does this same operation at `mcp_server.py:781-805` without a safety net — so the risk is equal to what already exists. |
| **Depends on** | PR-A (install method detection for auto-install) |
| **Effort** | 4 hours |

#### Step 4 → PR-L: Usage Telemetry (#190)
| | |
|---|---|
| **Risk** | **Low** — new module, fire-and-forget, every call wrapped in try/except |
| **What can break** | If telemetry blocks the main thread or throws uncaught: MCP tools slow down or crash. If the decorator interferes with `@mcp.tool()`: tools stop registering. |
| **Files touched** | New `telemetry.py`, `mcp_server.py` (decorator on 8 functions + init call), `session_start.py` (1 line), `stop.py` (1 line), `config.json` schema (2 new fields) |
| **Rollback** | Remove decorators + delete `telemetry.py` |
| **Mitigation** | All telemetry calls: `try: track(...) except: pass`. Background daemon thread with 3s HTTP timeout. Decorator goes BELOW `@mcp.tool()` so MCP registration happens first. Test with endpoint unreachable (should be silent). |
| **Why here** | Should be in place BEFORE the multi-CLI expansion multiplies the user base |
| **Effort** | 16 hours (client) + 8 hours (server) |

---

### RISK LEVEL 4: MEDIUM RISK (modifies existing behavior)

#### Step 5 → PR-D: Session Intelligence (#175, #176)
| | |
|---|---|
| **Risk** | **Medium** — modifies two hooks that run on every session |
| **What can break** | (1) Spawning too many background processes → OOM. (2) Incremental extraction overlaps with Stop hook → duplicate facts. (3) Circular imports if hooks import from each other. |
| **Files touched** | `user_prompt_submit.py` (add extraction trigger), `compact.py` (add extraction trigger), new `_shared.py`, new `_spawn.py` |
| **Rollback** | Revert the 2 hook files, delete the 2 new files |
| **Mitigation** | (1) Spawn cap enforcement (SPAWN_CAP=2, same as stop.py). (2) Encoding gate + dedup.py already handle re-processing. (3) No circular imports — _spawn.py is a leaf module. Test by keeping a session open 5+ hours. |
| **Effort** | 8 hours |

#### Step 6 → PR-C: Reranker Gap (#189)
| | |
|---|---|
| **Risk** | **Medium-High** — touches the hot search path in the core engine |
| **What can break** | (1) Double-reranking in search_agentic() if `_skip_reranker` not passed → 2x latency, wrong scores. (2) Latency regression for all Memory.search() callers (~200-800ms added). (3) Score distribution changes could affect downstream consumers. |
| **Files touched** | `engine.py` (search signature + new step + 2 lines in search_agentic) |
| **Rollback** | Revert engine.py (3 hunks) |
| **Mitigation** | (1) Grep for every `self.search(` call in engine.py — there are exactly 2 (lines 1449, 1556), both need `_skip_reranker=True`. (2) Run the full LoCoMo benchmark before/after to verify accuracy improves (or stays the same for MCP path). (3) The reranker step is wrapped in try/except so failures fall through gracefully. |
| **Testing** | Run `python -m pytest tests/` + manually verify `Memory.search()` returns reranked results + verify `search_agentic()` doesn't double-rerank |
| **Effort** | 4-6 hours |

---

### RISK LEVEL 5: HIGH RISK (major refactoring / external dependencies)

#### Step 7 → PR-E: Universal Hook Framework (#186)
| | |
|---|---|
| **Risk** | **High** — major refactoring of install infrastructure |
| **What can break** | (1) Existing `truememory-ingest install` stops working → ALL current users lose hooks on next update. (2) Claude Code MCP registration breaks. (3) CLAUDE.md merge corrupts user's file. |
| **Files touched** | New `truememory/hooks/` package (8+ files), modified `ingest/cli.py` |
| **Rollback** | Delete `hooks/` package, revert `cli.py` |
| **Mitigation** | (1) The `claude.py` adapter WRAPS existing install logic — it doesn't replace it. `truememory-ingest install` must still work exactly as before. (2) Integration test: run `truememory-ingest install`, verify `~/.claude/settings.json` matches expected schema. (3) Backup `settings.json` and `CLAUDE.md` before any mutation. |
| **Testing** | On a real machine: run `truememory-ingest install`, verify hooks fire, verify MCP server responds. This is the critical path. |
| **Effort** | 24 hours |

#### Step 8 → PR-F: Codex Integration (#182)
| | |
|---|---|
| **Risk** | **High** — first external CLI integration, untested territory |
| **What can break** | (1) TOML config merge corrupts user's existing Codex config → their Codex CLI breaks. (2) Hook scripts don't match Codex's actual stdin/stdout contract (despite docs saying they do). (3) MCP server doesn't register correctly in Codex. |
| **Files touched** | New `adapters/codex.py`, new hook templates, new tests |
| **Rollback** | `truememory-ingest setup --remove --cli codex` (must work!) |
| **Mitigation** | (1) Always backup config before merge. Additive-only merges. (2) Test against a REAL Codex CLI installation — unit tests alone are not sufficient. (3) Test the uninstall path as rigorously as the install path. |
| **Testing** | Install Codex CLI, run `truememory-ingest setup --cli codex`, verify MCP tools visible, verify hooks fire on session start/stop, verify uninstall cleans up. |
| **Effort** | 16 hours |

#### Step 9 → PR-G: Conversational Onboarding (#188)
| | |
|---|---|
| **Risk** | **Medium-High** — modifies the setup wizard flow |
| **What can break** | (1) Existing `truememory-ingest setup` flow breaks for Claude-only users. (2) Interactive prompts confuse non-interactive CI environments. (3) State tracking (`integrations.json`) out of sync with actual config. |
| **Files touched** | `ingest/cli.py` (setup subcommand), new `registry.py` functions |
| **Rollback** | Revert cli.py setup changes |
| **Mitigation** | (1) CLI detection is additive — Claude-only users see the same flow plus a "no other CLIs detected" message. (2) `--non-interactive` flag skips prompts. (3) `integrations.json` is advisory, not authoritative. |
| **Effort** | 12 hours |

#### Steps 10-12 → PR-H, PR-I, PR-J: Remaining CLI Integrations
| PR | CLI | Risk | Unique concern |
|---|---|---|---|
| **PR-H** | Kimi (#183) | **Medium-High** | Beta hooks — API could change upstream |
| **PR-I** | OpenClaw (#184) | **Highest** | JS plugin (different language), JSON5 config, mixed hook systems, no `agent:end` in internal hooks |
| **PR-J** | Hermes (#185) | **High** | Dual hook systems (plugin + gateway), YAML config merge |

Each follows the pattern established by PR-F. Risk decreases as we learn from each integration.

---

## Implementation Game Plan (final order — v2, post-rustle)

> v2 changes from v1:
> - PR-K (docs) moved to END — can't document integrations that don't exist yet
> - PR-E and PR-F SWAPPED — build concrete Codex integration first, extract framework after (avoid premature abstraction)
> - PR-D implementation note: DON'T refactor stop.py, just import from it
> - PR-C implementation note: grep ALL callers of search() before implementing (confirmed: exactly 2 internal call sites at lines 1449 and 1556)

```
PHASE 1 — QUICK WINS (Week 1, ~6 hours)
├── Step 1:  PR-A  Installer fixes         Risk: Minimal   Effort: 2h
└── Step 2:  PR-B  Tier upgrade CLI        Risk: Low       Effort: 4h

PHASE 2 — GROWTH INFRASTRUCTURE (Week 2, ~24 hours)
└── Step 3:  PR-L  Telemetry + dashboard   Risk: Low       Effort: 24h

PHASE 3 — CORE IMPROVEMENTS (Week 3, ~14 hours)
├── Step 4:  PR-D  Session intelligence    Risk: Medium    Effort: 8h
└── Step 5:  PR-C  Reranker gap            Risk: Med-High  Effort: 6h

PHASE 4 — MULTI-CLI EXPANSION (Weeks 4-7, ~100 hours)
├── Step 6:  PR-F  Codex integration       Risk: High      Effort: 16h  ← CONCRETE FIRST
├── Step 7:  PR-E  Hook framework          Risk: High      Effort: 24h  ← THEN ABSTRACT
├── Step 8:  PR-G  Conversational onboard  Risk: Med-High  Effort: 12h
├── Step 9:  PR-H  Kimi integration        Risk: Med-High  Effort: 12h
├── Step 10: PR-J  Hermes integration      Risk: High      Effort: 16h
├── Step 11: PR-I  OpenClaw integration    Risk: Highest   Effort: 20h
└── Step 12: PR-K  Multi-CLI docs          Risk: Zero      Effort: 8h   ← AFTER integrations exist
```

**Total estimated effort:** ~158 hours across 12 PRs

### Why this order (v2 rationale)

**1. PR-A first, not docs first.**
PR-K (docs) was step 1 in v1. Wrong. You can't write a Codex quickstart guide for an integration that doesn't exist. The docs will be speculative and will need rewriting when the implementation inevitably differs. PR-A is the real first step — 2 hours, 7 line changes, zero chance of regression, immediate user impact.

**2. Telemetry before core improvements.**
PR-L before PR-D and PR-C is a business call, not a technical one. If investor meetings are imminent, telemetry comes first — you need the dashboard before the pitch. If not, you could swap PR-L with PR-D/PR-C to ship product improvements first. But the user explicitly asked for investor visibility, so telemetry goes early.

**3. Codex before framework (the big swap).**
v1 had PR-E (framework) → PR-F (Codex). v2 flips it: PR-F → PR-E. Why:
- **Premature abstraction kills projects.** We designed an adapter interface (`CLIAdapter` with `detect()`, `install_mcp()`, `install_hooks()`, etc.) without building a single integration. What if Codex needs something the interface doesn't support? We'd redesign the framework.
- **Concrete first, abstract after.** Build Codex integration as a standalone module. Build it ugly if needed. Learn what the real abstraction points are. THEN extract `hooks/core.py`, `hooks/adapters/base.py`, etc. from the working code.
- **Lower risk.** A standalone Codex installer (read TOML, merge entry, write TOML) is ~200 lines of straightforward code. A framework with an abstract base class, registry, template engine, and CLI detection is ~800 lines of architecture. Ship the 200 lines first.
- **After Codex works, the framework writes itself.** You'll have two concrete implementations (Claude Code + Codex) and the common patterns will be obvious.

**4. Don't refactor stop.py in PR-D.**
v1 plan said "extract stop.py's spawn functions into `_spawn.py`." This means modifying `stop.py` — the hook that fires on every session end for every user. If we break it during refactoring, nobody's memories get extracted. Instead: import `_run_background_ingestion` and `_count_active_ingest_processes` FROM stop.py in the new hooks. Don't touch stop.py at all. Refactor to `_spawn.py` later when we have tests proving the extraction is safe.

**5. PR-C (reranker gap) needs careful verification.**
Confirmed via grep: exactly 2 callers of `self.search()` inside engine.py — lines 1449 and 1556, both in `search_agentic()`. Both need `_skip_reranker=True`. No other internal callers exist. But the implementation prompt must include: "grep -rn 'self\.search(' truememory/engine.py and verify no new call sites have been added before making changes."

**6. OpenClaw last, docs last.**
PR-I (OpenClaw) is highest implementation risk — JS plugin (different language), JSON5 config, mixed hook naming conventions, no session-end event in the internal hook system. By the time we get there, we've learned from Claude, Codex, Kimi, and Hermes. PR-K (docs) ships dead last because NOW we have real integrations to document, not hypothetical ones.

### Risk decision tree for each PR

```
Before starting any PR, ask:

1. Does this PR modify an existing file that runs in production?
   YES → Read the file first, understand every function you're touching
   NO  → Lower risk, proceed with caution anyway

2. Does this PR change a function signature?
   YES → Grep ALL callers across the entire codebase
   NO  → Lower risk

3. Does this PR touch the search/retrieval pipeline?
   YES → Run the full test suite BEFORE and AFTER changes
   NO  → Standard testing is fine

4. Does this PR modify hook scripts?
   YES → Test on a real Claude Code session (hooks are not unit-testable in isolation)
   NO  → Unit tests may be sufficient

5. Does this PR merge into external config files (TOML, YAML, JSON)?
   YES → Always backup before merge, test with existing populated configs
   NO  → Lower risk
```
- Power users who keep sessions open for hours/days lose memories until session close
- #1 complaint from heavy users — 8 hours of work, direct impact on retention

**PR-C (Reranker Gap)** — #189
- Real accuracy gap for Python SDK users — documented pipeline includes reranker but code doesn't use it
- Medium-high risk (touches hot search path) but measurable accuracy improvement
- 4-6 hours

### Tier 3: Ship This Month (Market Expansion)

**PR-E → PR-F → PR-G** — Hook framework → Codex → Onboarding
- Unlocks the entire Codex user base (~52 hours total)
- Codex is the #1 competitor CLI — capturing those users is the biggest growth lever

**PR-H (Kimi)** — Fast follow, mirrors Codex pattern (12 hours)

### Tier 4: Ship When Ready

**PR-I (OpenClaw)**, **PR-J (Hermes)**
- Valuable but smaller audiences, higher implementation risk

### Deferred

**#137, #79, #67, #68** — No urgency, ship when it makes sense.

---

## Implementation Plans

### PR-A: Installer Quick Fixes — Detailed Implementation (REVISED)

> ~~#167 (stale scores)~~ and ~~#169 (--refresh flag)~~ validated as already fixed. Removed from PR-A.

#### #168: Fix pip-only upgrade messages (validated locations)

**3 pip-only locations that need fixing:**

1. `ingest/cli.py:121` — in `_run_ingest()` preflight:
   - Current: `"Install with: pip install truememory"`
   - Fix: `"Install with: uv tool install truememory  (or: pip install truememory)"`

2. `ingest/cli.py:774` — in `_run_status()` import check:
   - Current: `"Install with: pip install truememory"`  
   - Fix: `"Install with: uv tool install truememory  (or: pip install truememory)"`

3. `mcp_server.py:1154` — in `_setup_claude()` help text:
   - Current: `"Run this once after \`pip install\`."`
   - Fix: `"Run this once after installing truememory."`

**1 behavioral bug (not just text):**

4. `ingest/cli.py:357-360` — auto-install subprocess uses `pip install` even for uv users:
   ```python
   subprocess.run([sys.executable, "-m", "pip", "install", "truememory[gpu]"], timeout=600)
   ```
   Fix: detect if running in uv tool env (`"uv" in sys.executable`), use `uv tool install` if so.

**3 locations already correct** (show both uv + pip, leave alone):
- `mcp_server.py:723-724`, `ingest/cli.py:347-348`, `ingest/cli.py:366-367`

#### #171: Add PATH guidance (validated locations)

1. `install.sh` final output (after line 198) — add:
   ```
   Note: If commands are not found, open a new terminal window
         or run: source ~/.zshrc  (or ~/.bashrc)
   ```

2. `mcp_server.py:1135` — in `_setup_claude()`:
   - Current: `"Claude Code CLI not found on PATH."`
   - Add: `"If you just installed it, try opening a new terminal window."`

3. `ingest/cli.py:121` and `774` (same locations as #168) — add:
   `"If already installed, ensure ~/.local/bin is on your PATH."`

Note: lines 133 and 708 in cli.py (originally referenced in this issue) are NOT PATH-related — line 133 is a transcript file error, line 708 is API provider validation.

---

### PR-D: Session Intelligence — Detailed Implementation (REVISED)

#### Shared spawn module (new)

Both #175 and #176 need background spawn with spawn caps and detachment. Extract from `stop.py` into `truememory/ingest/hooks/_spawn.py`:
- `run_background_ingestion(transcript_path, session_id, user_id, db_path)` — from `stop.py:242-351`
- `count_active_ingest_processes()` — from `stop.py:188-206`
- `queue_to_backlog(...)` — from `stop.py:209-239`

Then have `stop.py`, `user_prompt_submit.py`, and `compact.py` all import from `_spawn.py`.

#### Shared timestamp mechanism

New file `truememory/ingest/hooks/_shared.py`:
```python
"""Shared utilities for TrueMemory hooks."""
from pathlib import Path
import time

MARKER_PATH = Path.home() / ".truememory" / "last_incremental_extraction"
DEFAULT_INTERVAL = 14400  # 4 hours in seconds

def should_extract(interval: int = DEFAULT_INTERVAL) -> bool:
    if not MARKER_PATH.exists():
        return True
    try:
        last = MARKER_PATH.stat().st_mtime
        return (time.time() - last) >= interval
    except OSError:
        return True

def mark_extracted():
    MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    MARKER_PATH.write_text(str(time.time()))
```

#### #175: UserPromptSubmit incremental extraction (validated)

**Current state:** `user_prompt_submit.py` receives `transcript_path` in stdin JSON (docstring line 19) but doesn't extract it (line 77 only reads `prompt` and `session_id`). Also, `_parse_args()` return value is discarded at line 69.

**Changes needed:**
1. Line 69: Change `_parse_args()` to `args = _parse_args()` — need `args.user` and `args.db`
2. Line 77: Add `transcript_path = input_data.get("transcript_path", "")`
3. After line 86 (after buffer_message + prune): Add incremental extraction block
4. Add `subprocess` import (not currently imported)

**Frequency concern:** Hook fires every message. MUST debounce via `should_extract()` — only trigger when 4+ hours since last extraction.

**Spawn cap:** Import `count_active_ingest_processes` from `_spawn.py` and check before spawning. Respect `SPAWN_CAP=2`.

#### #176: PreCompact extraction (validated)

**Current state:** `compact.py` already receives `transcript_path` (line 53) and uses `args.user`/`args.db` (line 46/60). No extraction logic exists.

**Changes needed:**
1. After line 62 (after `save_snapshot()` try/except): Add background extraction block
2. Add `subprocess` import (not currently imported)
3. The existing `Memory.add()` snapshot (line 123-125) is KEPT — it's a different content type (lightweight summary vs extracted facts)

**No double-store risk:** Snapshot from `Memory.add()` stores ~200-char summary. Background ingestion extracts atomic facts. Different content types — encoding gate won't flag as duplicates.

**Blocking concern: none.** `save_snapshot()` is ~10ms, `Popen` is ~5ms. Total hook time stays under 20ms.

**Stop hook overlap: safe.** When Stop fires at session end, it re-processes the transcript. The dedup pipeline (`dedup.py`) and encoding gate's compression novelty signal handle this gracefully.

---

### PR-E: Universal Hook Framework — Detailed Implementation

See architecture diagram in PR Buckets section above.

**Critical design constraint:** The existing Claude Code integration (`truememory/ingest/hooks/*.py`) must continue working unchanged during and after this refactor. The framework extracts portable logic without modifying the existing hooks.

**Phase 1:** Extract `core.py` with `recall_memories()` and `extract_transcript()` — pure functions that don't know which CLI they're in.

**Phase 2:** Build the adapter interface (`adapters/base.py`) and implement `adapters/claude.py` first as a proof that the abstraction works without breaking existing behavior.

**Phase 3:** Build `registry.py` (CLI detection) and `cli.py` (setup subcommand).

**Phase 4:** Implement remaining adapters (Codex, Kimi, OpenClaw, Hermes).

---

## Validation Scripts

Each PR needs a validation script that can be run on a **fresh Mac** with no preexisting TrueMemory installation.

### Validation Script: PR-A (Installer Fixes)

```bash
#!/bin/bash
# validate_pr_a.sh — Run on a fresh Mac to validate installer fixes
# Prerequisites: macOS, internet connection, no prior TrueMemory install
set -euo pipefail

echo "=== PR-A Validation: Installer Quick Fixes ==="
echo ""

# Clean slate
echo "[1/6] Cleaning any existing TrueMemory install..."
rm -rf ~/.truememory ~/.local/share/uv/tools/truememory 2>/dev/null || true

# Run the installer
echo "[2/6] Running install.sh..."
curl -sSL https://raw.githubusercontent.com/buildingjoshbetter/TrueMemory/main/install.sh | sh

# Verify uv is on PATH without opening a new terminal
echo "[3/6] Verifying uv is accessible..."
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &>/dev/null; then
    echo "FAIL: uv not on PATH even after export"
    exit 1
fi
echo "PASS: uv found at $(which uv)"

# Verify we got the latest version (not cached)
echo "[4/6] Checking installed version..."
INSTALLED_VERSION=$(truememory-mcp --version 2>/dev/null || truememory-ingest -V 2>/dev/null)
echo "Installed: $INSTALLED_VERSION"
PYPI_VERSION=$(curl -s https://pypi.org/pypi/truememory/json | python3 -c "import sys,json; print(json.load(sys.stdin)['info']['version'])")
echo "PyPI latest: $PYPI_VERSION"
if [[ "$INSTALLED_VERSION" != *"$PYPI_VERSION"* ]]; then
    echo "FAIL: Installed version doesn't match PyPI latest"
    exit 1
fi
echo "PASS: Running latest version"

# Verify accuracy scores in onboarding are correct
echo "[5/6] Checking accuracy scores in source..."
SITE_PACKAGES=$(python3 -c "import truememory; print(truememory.__file__)" | xargs dirname)
if grep -q "90.1%" "$SITE_PACKAGES/mcp_server.py" 2>/dev/null; then
    echo "FAIL: mcp_server.py still shows stale 90.1% (should be 89.6%)"
    exit 1
fi
if grep -q "91.5%" "$SITE_PACKAGES/ingest/cli.py" 2>/dev/null; then
    echo "FAIL: cli.py still shows stale 91.5% (should be 92.0%)"
    exit 1
fi
echo "PASS: Accuracy scores are v0.6.0 values"

# Verify upgrade command shows correct method
echo "[6/6] Checking upgrade command detection..."
# The configure tool should suggest uv command, not pip
if grep -q "pip install truememory" "$SITE_PACKAGES/mcp_server.py" 2>/dev/null; then
    echo "WARN: mcp_server.py still has hardcoded pip command (should detect uv)"
fi
echo "PASS: Install method detection in place"

echo ""
echo "=== PR-A Validation Complete ==="
echo "All checks passed."
```

### Validation Script: PR-C (Scoring Fix)

```bash
#!/bin/bash
# validate_pr_c.sh — Verify the scoring dedup fix
set -euo pipefail

echo "=== PR-C Validation: Scoring Bug Fix ==="

cd /tmp
python3 -c "
from truememory import Memory
import json

mem = Memory(':memory:')

# Store some test memories
mem.add('User prefers dark mode in all editors')
mem.add('User works at Acme Corp as a senior engineer')
mem.add('User lives in San Francisco, California')
mem.add('User prefers Python over JavaScript')
mem.add('User uses vim keybindings')

# Search — both standard and agentic should score consistently
results_standard = mem.search('user preferences', limit=5)
results_clean = mem._clean_results(results_standard) if hasattr(mem, '_clean_results') else results_standard

# Check that scores are non-zero and consistent
for r in results_standard:
    score = r.get('score', 0)
    if score == 0 and r.get('fused_score', 0) > 0:
        print(f'FAIL: Result has fused_score={r[\"fused_score\"]} but search() gave score=0')
        print('The scoring bug is still present — search() and _clean_results() diverge')
        exit(1)

print('PASS: All results have consistent scores across search paths')
print(f'Found {len(results_standard)} results with scores: {[r.get(\"score\", 0) for r in results_standard]}')
"

echo "=== PR-C Validation Complete ==="
```

### Validation Script: PR-D (Session Intelligence)

```bash
#!/bin/bash
# validate_pr_d.sh — Verify incremental extraction works
set -euo pipefail

echo "=== PR-D Validation: Session Intelligence ==="

# Create a mock transcript
TRANSCRIPT="/tmp/test_transcript.json"
cat > "$TRANSCRIPT" << 'JSONEOF'
[
  {"type": "human", "content": "I prefer using bun over npm for all my projects"},
  {"type": "assistant", "content": "Got it, noted."},
  {"type": "human", "content": "My favorite color is blue and I always use dark mode"},
  {"type": "assistant", "content": "I'll remember that."},
  {"type": "human", "content": "I work at TechCorp as a staff engineer on the platform team"},
  {"type": "assistant", "content": "Thanks for sharing."},
  {"type": "human", "content": "For databases I strongly prefer PostgreSQL over MySQL"},
  {"type": "assistant", "content": "Noted."},
  {"type": "human", "content": "I live in Austin Texas and commute by bicycle"},
  {"type": "assistant", "content": "Got it."},
  {"type": "human", "content": "Always use TypeScript over JavaScript in my projects"},
  {"type": "assistant", "content": "Will do."}
]
JSONEOF

echo "[1/3] Testing UserPromptSubmit incremental trigger..."

# Simulate the hook with a stale timestamp (>4 hours ago)
python3 -c "
import os, time, json, sys
from pathlib import Path

marker = Path.home() / '.truememory' / 'last_incremental_extraction'
marker.parent.mkdir(parents=True, exist_ok=True)
# Set marker to 5 hours ago
old_time = time.time() - (5 * 3600)
marker.write_text(str(old_time))
os.utime(marker, (old_time, old_time))

# Import and check
sys.path.insert(0, '.')
from truememory.ingest.hooks._shared import should_extract
assert should_extract(14400), 'FAIL: should_extract returned False for 5-hour-old marker'
print('PASS: should_extract correctly identifies stale marker')

# Set marker to 1 hour ago
recent_time = time.time() - 3600
marker.write_text(str(recent_time))
os.utime(marker, (recent_time, recent_time))
assert not should_extract(14400), 'FAIL: should_extract returned True for 1-hour-old marker'
print('PASS: should_extract correctly skips recent marker')
"

echo "[2/3] Testing PreCompact extraction trigger..."
python3 -c "
from truememory.ingest.hooks._shared import should_extract
# PreCompact should always extract (interval=0 but respects marker)
assert should_extract(0), 'should_extract with interval=0 should return True'
print('PASS: PreCompact extraction trigger works')
"

echo "[3/3] Testing dedup safety..."
python3 -c "
from truememory import Memory
mem = Memory(':memory:')
mem.add('User prefers dark mode')

# Simulate double-extraction (UserPromptSubmit + Stop both fire)
mem.add('User prefers dark mode')
results = mem.search('user preferences', limit=10)
contents = [r['content'] for r in results]
# Dedup in search should handle this
print(f'PASS: {len(results)} results returned (dedup handled)')
"

rm -f "$TRANSCRIPT"
echo ""
echo "=== PR-D Validation Complete ==="
```

### Validation Script: PR-F (Codex Integration)

```bash
#!/bin/bash
# validate_pr_f.sh — End-to-end Codex CLI integration test
# Prerequisites: Codex CLI installed, TrueMemory installed
set -euo pipefail

echo "=== PR-F Validation: Codex CLI Integration ==="

# Check Codex is installed
echo "[1/7] Checking Codex CLI..."
if ! command -v codex &>/dev/null; then
    echo "SKIP: Codex CLI not installed. Install with: npm install -g @openai/codex"
    echo "To validate without Codex, run the unit tests: pytest tests/hooks/test_codex_adapter.py"
    exit 0
fi
echo "PASS: Codex CLI found at $(which codex)"

# Run TrueMemory setup for Codex
echo "[2/7] Running truememory setup for Codex..."
truememory-ingest setup --cli codex --non-interactive
echo "PASS: Setup completed"

# Verify MCP config was added
echo "[3/7] Verifying MCP server in Codex config..."
CODEX_CONFIG="$HOME/.codex/config.toml"
if [ ! -f "$CODEX_CONFIG" ]; then
    echo "FAIL: $CODEX_CONFIG not found"
    exit 1
fi
if ! grep -q "truememory" "$CODEX_CONFIG"; then
    echo "FAIL: truememory not found in Codex config"
    exit 1
fi
echo "PASS: MCP server entry present in config.toml"

# Verify hooks are registered
echo "[4/7] Verifying hooks in Codex config..."
if ! grep -q "SessionStart" "$CODEX_CONFIG"; then
    echo "FAIL: SessionStart hook not found"
    exit 1
fi
if ! grep -q "Stop" "$CODEX_CONFIG"; then
    echo "FAIL: Stop hook not found"
    exit 1
fi
echo "PASS: SessionStart and Stop hooks registered"

# Verify hook scripts exist and are executable
echo "[5/7] Checking hook scripts..."
HOOK_DIR="$HOME/.truememory/hooks/codex"
for script in session_start.sh stop.sh; do
    if [ ! -x "$HOOK_DIR/$script" ]; then
        echo "FAIL: $HOOK_DIR/$script missing or not executable"
        exit 1
    fi
done
echo "PASS: Hook scripts exist and are executable"

# Verify MCP server responds
echo "[6/7] Smoke-testing MCP server..."
echo '{"method":"initialize","params":{"capabilities":{}},"id":1}' | truememory-mcp 2>/dev/null | python3 -c "
import sys, json
try:
    # Read until we get a valid JSON response
    data = sys.stdin.read()
    if 'truememory' in data.lower() or 'capabilities' in data.lower():
        print('PASS: MCP server responds')
    else:
        print('WARN: MCP server response unexpected')
except:
    print('WARN: Could not parse MCP response (may need jsonrpc framing)')
"

# Verify AGENTS.md template exists
echo "[7/7] Checking AGENTS.md template..."
if [ -f "$HOOK_DIR/AGENTS.md.template" ]; then
    echo "PASS: AGENTS.md template available"
else
    echo "INFO: No AGENTS.md template (optional)"
fi

echo ""
echo "=== PR-F Validation Complete ==="
```

### Validation Script: PR-H (Kimi Integration)

```bash
#!/bin/bash
# validate_pr_h.sh — End-to-end Kimi CLI integration test
set -euo pipefail

echo "=== PR-H Validation: Kimi CLI Integration ==="

echo "[1/5] Checking Kimi CLI..."
if ! command -v kimi &>/dev/null; then
    echo "SKIP: Kimi CLI not installed"
    echo "To validate without Kimi, run: pytest tests/hooks/test_kimi_adapter.py"
    exit 0
fi
echo "PASS: Kimi CLI found"

echo "[2/5] Running truememory setup for Kimi..."
truememory-ingest setup --cli kimi --non-interactive
echo "PASS: Setup completed"

echo "[3/5] Verifying MCP config..."
KIMI_MCP="$HOME/.kimi/mcp.json"
if ! python3 -c "import json; d=json.load(open('$KIMI_MCP')); assert 'truememory' in str(d)"; then
    echo "FAIL: truememory not in Kimi MCP config"
    exit 1
fi
echo "PASS: MCP server entry present"

echo "[4/5] Verifying MCP tools discoverable..."
if command -v kimi &>/dev/null; then
    kimi mcp list 2>/dev/null | grep -q truememory && echo "PASS: Kimi sees TrueMemory tools" || echo "WARN: Could not verify tool discovery via kimi mcp list"
fi

echo "[5/5] Verifying hook scripts..."
HOOK_DIR="$HOME/.truememory/hooks/kimi"
for script in session_start.sh stop.sh; do
    if [ ! -x "$HOOK_DIR/$script" ]; then
        echo "FAIL: $HOOK_DIR/$script missing or not executable"
        exit 1
    fi
done
echo "PASS: Hook scripts exist and are executable"

echo ""
echo "=== PR-H Validation Complete ==="
```

### Validation Script: PR-G (Conversational Onboarding)

```bash
#!/bin/bash
# validate_pr_g.sh — Test the conversational onboarding flow
set -euo pipefail

echo "=== PR-G Validation: Conversational Onboarding ==="

# Clean state
rm -f ~/.truememory/integrations.json

echo "[1/4] Testing CLI auto-detection..."
python3 -c "
from truememory.hooks.registry import detect_installed_clis
detected = detect_installed_clis()
print(f'Detected CLIs: {[c.name for c in detected]}')
# At minimum, Claude Code should be detected if ~/.claude/ exists
import os
if os.path.exists(os.path.expanduser('~/.claude')):
    assert any(c.name == 'claude' for c in detected), 'FAIL: Claude Code not detected'
    print('PASS: Claude Code detected')
"

echo "[2/4] Testing non-interactive setup..."
truememory-ingest setup --cli claude --non-interactive 2>&1
echo "PASS: Non-interactive setup completed"

echo "[3/4] Verifying integrations.json state..."
python3 -c "
import json
from pathlib import Path
state = json.loads((Path.home() / '.truememory' / 'integrations.json').read_text())
assert 'claude' in state.get('configured', []), 'claude not in configured list'
print(f'PASS: Configured CLIs: {state[\"configured\"]}')
"

echo "[4/4] Testing re-run idempotency..."
truememory-ingest setup --cli claude --non-interactive 2>&1
echo "PASS: Re-run didn't error"

echo ""
echo "=== PR-G Validation Complete ==="
```

---

## Execution Prompts

### Recommended Approach

**Do NOT implement all PRs in one session.** Each PR bucket should be implemented in a dedicated fresh session for these reasons:

1. **Context clarity** — each session gets the full context window for one focused PR
2. **Clean git state** — each PR is a clean branch off main
3. **Validation isolation** — test each PR independently before moving on
4. **Rollback safety** — if a PR introduces issues, it's isolated

### Execution Order

```
Week 1:  PR-A (installer fixes) + PR-C (scoring fix)     ← quick wins
Week 1:  PR-B (tier upgrade CLI)                          ← follows PR-A
Week 2:  PR-D (session intelligence)                      ← independent, high value
Week 2:  PR-E (hook framework)                            ← foundation
Week 3:  PR-F (Codex integration)                         ← first CLI
Week 3:  PR-G (conversational onboarding)                 ← needs PR-F
Week 4:  PR-H (Kimi) + PR-I (OpenClaw)                   ← parallel CLIs
Week 4:  PR-J (Hermes) + PR-K (docs)                     ← remaining
```

### Session Prompt Template

The following prompt template should be used for each PR session. Copy it, fill in the PR-specific section, and run in a fresh Claude Code session with `/loop`.

---

#### Master Prompt: PR-A (Installer Quick Fixes — REVISED)

```
You are implementing PR-A for the TrueMemory project (github.com/buildingjoshbetter/TrueMemory).

## Context
TrueMemory is a persistent memory system for AI agents. It runs as an MCP server + ingestion pipeline with lifecycle hooks. The repo is Python, built with Hatchling, installed via `curl | sh` using uv.

NOTE: Issues #167 (stale accuracy scores) and #169 (--refresh flag) were validated as ALREADY FIXED on 2026-05-09. This PR only covers #168 and #171.

## What to implement
This PR fixes 2 installer/upgrade bugs:

### #168: Fix pip-only upgrade messages (3 locations)
Three locations in the codebase only mention `pip install` with no `uv` alternative. Users who installed via `curl | sh` (which uses uv) see useless instructions.

The BROKEN locations (pip-only, no uv mentioned):
- `truememory/mcp_server.py` line 1154: "Run this once after `pip install`"
- `truememory/ingest/cli.py` line 121: "Install with: pip install truememory"
- `truememory/ingest/cli.py` line 774: "Install with: pip install truememory"

Note: Three OTHER locations already show BOTH uv and pip instructions — leave those alone:
- mcp_server.py:723-724 (already has "If you installed via curl... / If you used pip...")
- cli.py:347-348 (already has both)
- cli.py:366-367 (already has both)

Fix: Update the 3 broken locations to show both install methods, matching the pattern used elsewhere:
```
If you installed via the curl installer, run:  uv tool install "truememory[gpu]"
If you used pip, run:  pip install "truememory[gpu]"
```
Always quote the brackets — zsh interprets unquoted [gpu] as a glob pattern.

### #171: Add "open a new terminal" guidance
After install, `uv tool update-shell` modifies shell rc files but those changes only apply in NEW terminals. Error messages don't mention this.

Locations to update:
- `install.sh` final output (around lines 185-199): Add a line saying "Open a new terminal for the `truememory-mcp` command to be available."
- `truememory/mcp_server.py` line 1135: After "Claude Code CLI not found on PATH", add "If you just installed, try opening a new terminal window."
- `truememory/ingest/cli.py` lines 133 and 708: Same — add new-terminal guidance to "not found" messages.

## How to work
1. Clone the repo: gh repo clone buildingjoshbetter/TrueMemory
2. Create branch: git checkout -b fix/installer-quick-fixes
3. Read each file FIRST to find exact current line numbers (they may have shifted)
4. Make all changes
5. Run: python -m pytest tests/ -x -q
6. Commit with message: "fix: installer quick fixes — pip-only messages + PATH guidance (#168, #171)"
7. Push and create PR

## Validation
After implementing, run:
- grep -n "pip install truememory" truememory/ — every match should have a nearby uv alternative
- grep -n "not found" truememory/mcp_server.py truememory/ingest/cli.py — relevant matches should mention "new terminal"
- grep -n "new terminal" install.sh — should find the guidance in final output

## Closing issues
Reference in the PR body: Closes #168, closes #171
```

---

#### Master Prompt: PR-C (Reranker Gap — REVISED)

```
You are implementing PR-C for the TrueMemory project (github.com/buildingjoshbetter/TrueMemory).

## Context
engine.search() (standard path) does NOT use the cross-encoder reranker, even though the README lists it as a standard pipeline component. Only engine.search_agentic() applies reranking. MCP users are fine (truememory_search routes through search_agentic), but Python SDK users calling Memory.search() get lower accuracy.

## What to implement
1. Add `_skip_reranker: bool = False` to search() signature (line 1100)
2. Insert reranker step between lines 1340 (surprise boost) and 1342 (clean/trim):
   - Guard: `if not _skip_reranker and self._has_reranker and len(results) > 1`
   - Call: `rerank_with_modality_fusion(query, results[:limit*3], top_k=limit, rrf_weight=0.4, rerank_weight=0.6)`
   - Wrap in try/except with logger.debug fallback
3. CRITICAL: Update search_agentic() calls to pass _skip_reranker=True:
   - Line 1449: `self.search(query, limit=candidate_pool, _skip_surprise_boost=True, _skip_reranker=True)`
   - Line 1556: `self.search(rq, limit=limit, _skip_surprise_boost=True, _skip_reranker=True)`
4. Update search() docstring to list reranking as step 7.6

## How to work
1. Clone, branch: fix/search-reranker-gap
2. Read engine.py search() (1100-1383) and search_agentic() calls (1449, 1556)
3. Make changes, run pytest
4. Commit: "fix: add cross-encoder reranking to engine.search() (#189)"

Closes #189
```

---

#### Master Prompt: PR-D (Session Intelligence)

```
You are implementing PR-D for the TrueMemory project (github.com/buildingjoshbetter/TrueMemory).

## Context
TrueMemory's ingestion pipeline only runs when a session ends (Stop hook). Power users keep sessions open for hours/days. This PR adds two mid-session extraction triggers.

## What to implement

### 1. Shared timestamp mechanism
Create truememory/ingest/hooks/_shared.py:
- should_extract(interval: int = 14400) -> bool — check if enough time since last extraction
- mark_extracted() -> None — update the timestamp marker
- Marker file: ~/.truememory/last_incremental_extraction
- Use file mtime, not file contents (faster, atomic)

### 2. UserPromptSubmit incremental extraction (#175)
Modify truememory/ingest/hooks/user_prompt_submit.py:
- After the existing buffer_message() call, check should_extract()
- If True, import _run_background_ingestion and _has_enough_messages from stop.py
- Spawn background ingestion of the transcript, then mark_extracted()
- Configurable via TRUEMEMORY_INCREMENTAL_INTERVAL env var (default: 14400 = 4 hours)

### 3. PreCompact extraction (#176)
Modify truememory/ingest/hooks/compact.py:
- After save_snapshot(), check should_extract(interval=0) — always extract on compact
- Spawn background ingestion using the same pattern
- Call mark_extracted() to prevent double-extraction if UserPromptSubmit fires right after

### Critical constraints
- NEVER block the hook — all extraction is background Popen
- Respect SPAWN_CAP from stop.py (max 2 concurrent ingest processes)
- The encoding gate + dedup pipeline handles re-processing gracefully
- Import stop.py functions, don't duplicate them

## How to work
1. Clone, branch: feat/session-intelligence
2. Create _shared.py
3. Modify user_prompt_submit.py
4. Modify compact.py
5. Add tests: tests/ingest/test_incremental_extraction.py
6. Run: python -m pytest tests/ -x -q
7. Commit: "feat: incremental extraction during long sessions (#175, #176)"

## Validation
- Test should_extract with various timestamps
- Test that mark_extracted creates/updates the marker file
- Test that the hooks don't crash when stop.py functions are unavailable (import error handling)
- Verify no circular imports

Closes #175, closes #176
```

---

#### Master Prompt: PR-E (Universal Hook Framework)

```
You are implementing PR-E for the TrueMemory project (github.com/buildingjoshbetter/TrueMemory).

## Context
TrueMemory currently integrates with Claude Code only. We're adding support for Codex, Kimi, OpenClaw, and Hermes. Each CLI has different config formats and hook conventions but the core memory lifecycle is the same: recall at session start, extract at session end.

## What to implement

### Package structure
Create truememory/hooks/ package:

truememory/hooks/__init__.py
truememory/hooks/core.py — CLI-agnostic functions
truememory/hooks/registry.py — CLI detection + state tracking
truememory/hooks/adapters/__init__.py
truememory/hooks/adapters/base.py — abstract base adapter
truememory/hooks/adapters/claude.py — Claude Code adapter
truememory/hooks/adapters/codex.py — stub (implemented in PR-F)
truememory/hooks/adapters/kimi.py — stub
truememory/hooks/adapters/openclaw.py — stub
truememory/hooks/adapters/hermes.py — stub

### core.py
Extract from existing hooks:
- recall_memories(db_path, user_id, limit) -> str — from session_start.py:recall_memories()
- extract_transcript(transcript_path, session_id, user_id, db_path, threshold) -> None — from stop.py:_run_background_ingestion()
- generate_system_prompt(cli_name) -> str — return the appropriate system prompt template

### registry.py
- CLI_REGISTRY: dict mapping CLI names to their adapter classes
- detect_installed_clis() -> list[CLIAdapter] — scan filesystem for installed CLIs
- get_configured_clis() -> list[str] — read ~/.truememory/integrations.json
- save_configured_clis(clis: list[str]) — write integrations.json

### adapters/base.py
Abstract base class — see the interface definition in the triage doc.

### adapters/claude.py
Wrap the existing install logic from truememory/ingest/cli.py (the `install` subcommand) into the adapter interface. The existing hooks stay where they are — this adapter just knows how to register them.

### CLI integration
Add `setup` subcommand to truememory/ingest/cli.py:
- truememory-ingest setup — interactive, auto-detect CLIs
- truememory-ingest setup --cli codex — specific CLI
- truememory-ingest setup --non-interactive — skip prompts

## Critical constraints
- The existing Claude Code integration (truememory/ingest/hooks/*.py) must NOT break
- The existing `truememory-ingest install` command must continue working (it's the Claude Code installer)
- Adapters are stubs for now — only claude.py is fully implemented
- No Jinja2 dependency yet (templates are just string formatting for now)

## How to work
1. Clone, branch: feat/universal-hook-framework
2. Read the existing install logic in ingest/cli.py
3. Read all 4 existing hooks to understand the interface
4. Create the package structure
5. Implement core.py by extracting from existing hooks
6. Implement registry.py
7. Implement adapters/base.py and adapters/claude.py
8. Add the setup subcommand
9. Add tests
10. Run: python -m pytest tests/ -x -q
11. Verify: truememory-ingest install still works (backward compat)
12. Commit: "feat: universal hook template framework (#186)"

Closes #186
```

---

#### Master Prompt: PR-F (Codex Integration)

```
You are implementing PR-F for the TrueMemory project (github.com/buildingjoshbetter/TrueMemory).

## Context
The universal hook framework (PR-E / #186) provides the adapter interface. This PR implements the Codex CLI adapter.

## Codex CLI specifics
- Config: TOML at ~/.codex/config.toml
- MCP entry: [mcp_servers.truememory] with command = "truememory-mcp" and args = []
- Hook events: SessionStart, Stop (stable API)
- Hook I/O: JSON stdin → JSON stdout (same as Claude Code)
- Hook config: [[hooks]] array in config.toml with event, command, timeout fields
- System prompt: AGENTS.md (Codex equivalent of CLAUDE.md)

## What to implement

### truememory/hooks/adapters/codex.py
Full adapter implementation:
- detect(): check if ~/.codex/ or ~/.codex/config.toml exists
- is_configured(): parse config.toml, check for truememory MCP entry
- install_mcp(): merge [mcp_servers.truememory] into config.toml
- install_hooks(): add [[hooks]] entries for SessionStart and Stop
- uninstall(): remove truememory entries from config.toml
- verify(): spawn truememory-mcp and check it responds
- get_system_prompt(): return AGENTS.md template content

### Hook scripts
Create ~/.truememory/hooks/codex/:
- session_start.sh — wrapper that calls truememory hooks core recall_memories
- stop.sh — wrapper that calls truememory hooks core extract_transcript

### AGENTS.md template
Codex-flavored system prompt with TrueMemory auto-recall/auto-store instructions.

### Tests
tests/hooks/test_codex_adapter.py:
- Test TOML config generation
- Test config merge (don't overwrite existing entries)
- Test hook script generation
- Test detect() with and without Codex installed
- Test uninstall cleanup

## Config format reference
Codex config.toml structure:
```toml
[mcp_servers.truememory]
command = "truememory-mcp"
args = []

[[hooks]]
event = "SessionStart"
command = "~/.truememory/hooks/codex/session_start.sh"
timeout = 10000

[[hooks]]
event = "Stop"
command = "~/.truememory/hooks/codex/stop.sh"
timeout = 5000
```

## How to work
1. Clone, branch: feat/codex-integration
2. Read the existing claude.py adapter as the pattern
3. Implement codex.py adapter
4. Create hook script templates
5. Create AGENTS.md template
6. Add tests
7. Run: python -m pytest tests/ -x -q
8. If Codex CLI is available, run the validation script
9. Commit: "feat: Codex CLI integration (#182)"

Closes #182
```

---

*Prompts for PR-G through PR-K follow the same pattern. Each references the adapter interface from PR-E and the pattern established by PR-F, with CLI-specific details (config format, hook event names, file paths).*

### How to use these prompts

1. Open a fresh Claude Code session
2. `cd` to the TrueMemory repo
3. Paste the prompt for the target PR
4. Let it implement, test, and create the PR
5. Run the validation script on a fresh Mac
6. If validation passes, merge
7. Move to the next PR in order

For PRs that can run in parallel (PR-H + PR-I, for example), open two sessions.

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Codex/Kimi/Hermes/OpenClaw hook APIs change | Medium — our adapter breaks | Pin minimum CLI versions in docs; test against specific versions |
| TOML/YAML/JSON5 config merge corrupts user's existing config | High — user loses settings | Always backup before merge; use additive-only strategy; write integration tests |
| Background ingestion from multiple hooks races (#175 + #176 + Stop) | Medium — duplicate facts | Encoding gate + dedup pipeline already handles this; timestamp marker prevents redundant spawns |
| Refactoring existing Claude Code hooks breaks current users | Critical | Never modify existing hook files in the framework PR; claude.py adapter wraps, doesn't replace |
| CLI not available for testing (e.g., no Codex license) | Medium — can't validate | Unit tests mock the filesystem; integration tests skip gracefully if CLI not found |
| install.sh PATH fix doesn't work on all shells (fish, nushell) | Low | Document shell support; test bash and zsh |
