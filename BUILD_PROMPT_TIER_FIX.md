# BUILD_PROMPT: Fix TrueMemory Tier Resolution Bug

## Mission

TrueMemory's vector health shows `"degraded"` for Base/Pro users on every MCP server restart. The symptom is a `TrueMemoryMigrationError` saying the DB was built with `qwen3_256` but the current model is `model2vec`. The user's config says `tier: base` but the runtime resolves to Edge embeddings.

Previous fix attempts were pushed to main without full validation and didn't solve the problem. This prompt starts from scratch with a proper investigation.

**Rules:**
- No direct commits to main. Everything goes through a PR branch.
- No `git push --force` or destructive git operations.
- No code changes until the investigation is complete and you have a plan.
- All fixes must be proven locally before creating a PR.
- Do NOT merge the PR — leave it for Josh.

---

## PHASE 0: DEEP INVESTIGATION

**You must complete this entire phase before writing ANY code. No exceptions.**

### 0.1 — Establish the current state

```bash
cd /Users/j/Desktop/TrueMemory

# What's on main right now? Previous fix attempts may have been pushed.
git log --oneline -10

# What has changed since the last known-good version (v0.7.1)?
git diff a9f9250..HEAD --stat

# What does the user's config say?
cat ~/.truememory/config.json | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin), indent=2))" 2>/dev/null | head -10

# What does the DB metadata say about the embedding model?
/Users/j/Desktop/TrueMemory/.venv/bin/python3.12 -c "
import sqlite3
conn = sqlite3.connect(str(__import__('pathlib').Path.home() / '.truememory/memories.db'))
try:
    rows = conn.execute('SELECT key, value FROM metadata').fetchall()
    for k, v in rows:
        print(f'{k}: {v}')
except Exception as e:
    print(f'No metadata table or error: {e}')
"
```

### 0.2 — Reproduce the bug

Simulate exactly what happens when Claude Code starts the MCP server (`python -m truememory.mcp_server`):

```bash
# Clear the env var to simulate a fresh process
/Users/j/Desktop/TrueMemory/.venv/bin/python3.12 -c "
import os
os.environ.pop('TRUEMEMORY_EMBED_MODEL', None)

# This triggers the same import chain as python -m truememory.mcp_server:
# 1. Python loads truememory package -> __init__.py runs
# 2. __init__.py imports vector_search at top level
# 3. vector_search reads TRUEMEMORY_EMBED_MODEL at module level
# 4. THEN mcp_server.py code executes

from truememory.vector_search import EMBEDDING_MODEL
print(f'EMBEDDING_MODEL resolved to: {EMBEDDING_MODEL}')
print(f'Expected for Base tier: qwen3_256')
print(f'Bug present: {EMBEDDING_MODEL != \"qwen3_256\"}'  )
"
```

### 0.3 — Trace the full import chain

Read these files IN ORDER and understand the dependency chain. Do not skim — read the actual module-level code that executes on import:

1. **`truememory/__init__.py`** — Read lines 1-55. What does it import at the top level? Which of those imports trigger `vector_search.py`?

2. **`truememory/vector_search.py`** — Read lines 1-120. How is `EMBEDDING_MODEL` set? What does `_resolve_model_name()` do? What happens when `TRUEMEMORY_EMBED_MODEL` is not in the environment?

3. **`truememory/mcp_server.py`** — Read lines 1-135. When does the config loading happen relative to the `truememory` imports? Why can't the env var be set in time?

4. **`truememory/engine.py`** — Read lines 95-110 and 325-410 and 880-910. Where does `_vectors_load_error` get set? What triggers the `TrueMemoryMigrationError`? Does engine.py read `TRUEMEMORY_EMBED_MODEL` directly or use `EMBEDDING_MODEL` from vector_search?

5. **`truememory/reranker.py`** — Read lines 75-105. This file already solved the same problem correctly. How? Can the pattern be reused?

6. **`truememory/model_server.py`** — Read lines 65-120. The model_server is a separate process. How does it resolve the tier? Does it have the same bug?

7. **`truememory/ingest/cli.py`** — Grep for `TRUEMEMORY_EMBED_MODEL`. These are entry points that set the env var explicitly. Understand why they work but the MCP server doesn't.

### 0.4 — Map every env var read

```bash
grep -rn 'TRUEMEMORY_EMBED_MODEL' truememory/ --include="*.py" | grep -v __pycache__ | sort
```

For each occurrence, classify it:
- **SETTER** (sets the env var) — these are fine
- **READER with correct fallback** (reads env var, falls back to config.json) — these are fine
- **READER with "edge" default** (reads env var, defaults to "edge") — these are the bug

### 0.5 — Understand the model_server architecture

The model_server is a separate process (communicates via Unix socket at `~/.truememory/model.sock`). Read:
- How it starts (check `model_server.py` for the `main()` or `__main__` block)
- How it resolves the tier
- Whether it caches the model and what happens when it restarts
- Whether it has the same "edge" default bug

```bash
ps aux | grep TrueMemory | grep -v grep
cat ~/.truememory/model_server.pid
```

### 0.6 — Understand what previous fix attempts did

```bash
# Show all changes from previous fix attempts
git log --oneline a9f9250..HEAD
git diff a9f9250..HEAD -- truememory/vector_search.py truememory/mcp_server.py truememory/engine.py
```

Assess each change:
- Is it correct but incomplete?
- Is it wrong and needs to be reverted?
- Is it correct and should be kept?

### 0.7 — Write your findings

Before proceeding to Phase 1, write a summary of:
1. The root cause (be specific — which file, which line, which import chain)
2. Every file that needs to change and why
3. Every file that does NOT need to change and why not
4. Whether previous fix attempts should be kept, extended, or reverted
5. The exact approach you'll take (centralized function? per-file fix? something else?)
6. Any risks or edge cases you've identified

Print this summary to stdout. Do not proceed until you're confident in the analysis.

---

## PHASE 1: IMPLEMENT THE FIX

Based on your Phase 0 findings, implement the fix. Guidelines:

- Prefer a centralized tier-resolution function over duplicating logic
- The resolver should: check env var → read config.json → default to "edge"
- `reranker.py` already has a working implementation — consider whether you can reuse it or need to create a new one (watch for circular imports)
- If previous fix attempts left partial changes on main, decide whether to build on them or revert and start clean. Document your reasoning.
- Every `os.environ.get("TRUEMEMORY_EMBED_MODEL", "edge")` pattern must be addressed

---

## PHASE 2: LOCAL VALIDATION

Run ALL of these tests. Every one must pass.

### Core resolution tests

```bash
PYTHON=/Users/j/Desktop/TrueMemory/.venv/bin/python3.12

# Test 1: Base tier resolves correctly without env var
$PYTHON -c "
import os; os.environ.pop('TRUEMEMORY_EMBED_MODEL', None)
from truememory.vector_search import EMBEDDING_MODEL
assert EMBEDDING_MODEL == 'qwen3_256', f'FAIL: {EMBEDDING_MODEL}'
print('TEST 1 PASS: Base tier -> qwen3_256')
"

# Test 2: Engine init has no migration error
$PYTHON -c "
import os; os.environ.pop('TRUEMEMORY_EMBED_MODEL', None)
from truememory.engine import TrueMemoryEngine, _vectors_load_error
e = TrueMemoryEngine(); e._ensure_connection()
assert _vectors_load_error is None or 'model2vec' not in str(_vectors_load_error), f'FAIL: {_vectors_load_error}'
print('TEST 2 PASS: No model2vec migration error')
"

# Test 3: Edge tier still works
TRUEMEMORY_EMBED_MODEL=edge $PYTHON -c "
from truememory.vector_search import EMBEDDING_MODEL
assert EMBEDDING_MODEL == 'model2vec', f'FAIL: {EMBEDDING_MODEL}'
print('TEST 3 PASS: Edge -> model2vec')
"

# Test 4: Env var takes precedence over config
TRUEMEMORY_EMBED_MODEL=pro $PYTHON -c "
from truememory.vector_search import EMBEDDING_MODEL
assert EMBEDDING_MODEL == 'qwen3_256', f'FAIL: {EMBEDDING_MODEL}'
print('TEST 4 PASS: Env var overrides config')
"

# Test 5: No config.json -> falls back to edge
$PYTHON -c "
import os, tempfile; os.environ.pop('TRUEMEMORY_EMBED_MODEL', None)
# Temporarily rename config to simulate missing config
from pathlib import Path
cfg = Path.home() / '.truememory/config.json'
backup = cfg.with_suffix('.json.test_backup')
cfg.rename(backup)
try:
    # Force reimport
    import importlib
    import truememory.vector_search as vs
    importlib.reload(vs)
    assert vs.EMBEDDING_MODEL == 'model2vec', f'FAIL: {vs.EMBEDDING_MODEL}'
    print('TEST 5 PASS: No config -> edge fallback')
finally:
    backup.rename(cfg)
"

# Test 6: Existing test suite
cd /Users/j/Desktop/TrueMemory && $PYTHON -m pytest tests/ -x -q 2>&1 | tail -20
```

If ANY test fails, debug and fix before proceeding.

---

## PHASE 3: RUSTLE THE FEATHERS — OpenRouter Multi-Model Review

After ALL Phase 2 tests pass, run adversarial review.

### Round 1: Code review (7 models)

Generate the full diff, then send to these models via OpenRouter API (`OPENROUTER_API_KEY` is set in the environment):

**Models:**
1. `openai/gpt-5.5`
2. `openai/gpt-5.2-codex`
3. `google/gemini-2.5-pro`
4. `x-ai/grok-4.20`
5. `deepseek/deepseek-r1-0528`
6. `qwen/qwen3-235b-a22b`
7. `anthropic/claude-opus-4.7`

Write a Python script using `urllib.request` (NOT curl, NOT jq — they break on control chars in LLM responses) that sends to all 7 in parallel using `concurrent.futures.ThreadPoolExecutor`. Max tokens: 800. Temperature: 0.3. Timeout: 120s per model.

**Round 1 prompt:**

```
You are performing a deep adversarial code review of a Python fix. The change fixes a bug where TrueMemory's embedding model silently defaults to "edge"/model2vec when the TRUEMEMORY_EMBED_MODEL env var is not set, ignoring the user's configured tier stored in ~/.truememory/config.json.

Analyze the diff for:
1. Circular import risk — trace the full import chain
2. Module initialization side effects and ordering
3. Config read failures blocking startup
4. Environment variable precedence correctness (env var > config.json > "edge" default)
5. Any remaining code path that still defaults to "edge" without config fallback
6. Regression risk for Edge tier users (env var not set, no config = should get edge)
7. Test breakage from changed behavior
8. Process forking / multiprocessing env var propagation
9. Model server (separate process) compatibility

DIFF:
{diff}

Respond with:
- VERDICT: APPROVE or REJECT
- CONFIDENCE: HIGH / MEDIUM / LOW
- RISK SUMMARY: one sentence, the highest-risk concern
- If REJECT: what specifically is wrong and how to fix it
```

**Consensus: 5/7 must APPROVE. If any REJECT, analyze their concern. If valid, fix and re-run.**

### Round 2: Scenario-based adversarial testing (same 7 models)

```
Given this tier-resolution fix, analyze these adversarial scenarios:

1. config.json exists but is being written by another process (partial JSON)
2. config.json has tier="" (empty string)
3. config.json has tier="EDGE" (uppercase)
4. config.json has tier="qwen3_256" (internal model name, not tier alias)
5. config.json is deleted after initial setup
6. Two MCP server processes start simultaneously and both read config
7. model_server (separate process) starts before config exists
8. User runs truememory_configure(tier=base) — does the in-process state update correctly?
9. User upgrades from 0.7.1 to the fixed version — is migration seamless?

For each: does it break? Is it worse than the status quo (always defaulting to edge)?

{diff}
```

**Consensus: no scenario can produce a WORSE outcome than the status quo.**

---

## PHASE 4: CREATE PR

Only after Phase 2 (all tests pass) AND Phase 3 (consensus achieved).

```bash
cd /Users/j/Desktop/TrueMemory
git checkout -b fix/tier-resolution-centralized
git add <changed files>
git commit -m "fix: centralize tier resolution to read config.json when env var is unset

<description of what was changed and why>"
git push -u origin fix/tier-resolution-centralized
gh pr create --title "fix: centralize tier resolution for Base/Pro users" --body "$(cat <<'PREOF'
## Summary
<what was wrong, what was changed, which files>

## Test results
<paste all 6 test results>

## OpenRouter consensus
<paste model verdicts from both rounds>

## Files changed
<list>
PREOF
)"
```

**Do NOT merge.** Print the PR URL and stop.
