# Contributing to TrueMemory

Thanks for your interest in contributing. This document covers everything you need to know to submit a high-quality pull request.

---

## Architecture Overview

TrueMemory uses a 6-layer retrieval pipeline plus an encoding gate. Each layer is a standalone module in `truememory/`:

| Layer | Name | Module(s) |
|-------|------|-----------|
| L0 | Personality Engram | `personality.py`, `personality_style_vec.py` |
| L1 | Working Memory | Deferred (not yet implemented) |
| L2 | Episodic | `fts_search.py` |
| L3 | Semantic | `vector_search.py`, `hybrid.py` |
| L4 | Salience Guard | `salience.py` |
| L5 | Consolidation | `consolidation.py`, `predictive.py` |
| + | Reranker | `reranker.py` |

The orchestrator (`engine.py`) ties all layers together with graceful degradation. If any module is missing or fails, the engine falls back to whatever layers are available.

### Ingest Pipeline

The `truememory/ingest/` subpackage handles automatic memory capture:

| Module | What it does |
|--------|-------------|
| `extractor.py` | LLM-based fact extraction from conversation transcripts |
| `encoding_gate.py` | Three-signal filter (novelty + salience + prediction error) that decides what gets stored |
| `encoding_salience.py` | Speech-act salience scoring for the encoding gate |
| `dedup.py` | Deduplication via vector similarity + word-overlap Jaccard matching |
| `pipeline.py` | End-to-end ingestion orchestrator |
| `hooks/` | Claude Code lifecycle hook scripts (SessionStart, Stop, UserPromptSubmit, PreCompact) |

### Storage

Everything lives in a single SQLite file (`~/.truememory/memories.db`). The storage layer is in `storage.py`. WAL mode is enabled by default for concurrent access.

---

## Getting Started

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/TrueMemory.git
cd TrueMemory

# 2. Create a virtualenv
python -m venv .venv && source .venv/bin/activate

# 3. Install in dev mode with all extras
pip install -e ".[all,dev]"

# 4. Verify everything works
pytest tests/ -v
ruff check truememory/ tests/
```

Both commands must pass before you submit anything.

---

## Branching

Create a branch from `main` with a descriptive name:

```
fix/gate-85-cosine-novelty
feat/gate-108-encoding-salience
docs/readme-v2
chore/cleanup-stale-results
```

Format: `type/short-description` or `type/issue-number-short-description`

---

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): short description
```

**Types:**

| Type | When to use |
|------|-------------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `chore` | Maintenance, cleanup, dependency updates |
| `refactor` | Code restructuring without behavior change |
| `test` | Adding or updating tests |

**Examples:**
```
feat(gate): encoding-specific salience scorer (closes #108)
fix(gate): use cosine similarity for novelty instead of RRF (closes #85)
docs: update scores to 3-run means with standard deviation
chore: update CITATION.cff to v0.6.0/AGPL-3.0
```

Reference issue numbers with `(closes #N)` or `(#N)` in the commit message.

---

## Pull Requests

### One PR, one purpose

Every PR should do one thing. A bug fix is a bug fix. A feature is a feature. Don't bundle unrelated changes. If you find a separate issue while working on something, file it as a new issue and address it in a separate PR.

### PR structure

Every PR should follow this format:

```markdown
## Summary
What changed and why. 1-3 sentences.

## Changes
Detailed breakdown. Use a table for multi-file changes:

| File | Change |
|------|--------|
| `engine.py` | Added `search_vectors_raw()` public method |
| `client.py` | Uses `search_vectors_raw()` instead of private attrs |

## Test plan
- [x] All tests pass (`pytest tests/ -v`)
- [x] Lint clean (`ruff check truememory/ tests/`)
- [ ] CI green
```

For bug fixes, include the issue number in the title:

```
fix: deduplicate _fts_search + expose search_vectors_raw (closes #145, closes #136)
```

### Before submitting

1. `pytest tests/ -v` passes with zero failures
2. `ruff check truememory/ tests/` is clean
3. No unused imports or dead code introduced
4. Public functions have type hints
5. No secrets, credentials, or PII in the diff

---

## Validation: "Rustle the Feathers"

We use a multi-perspective adversarial review process called "rustle the feathers" before merging significant changes. The goal is to catch problems that a single-perspective review would miss.

### How it works

Before a PR is merged, review the changes from at least 3 of these perspectives:

| Perspective | What it asks |
|-------------|-------------|
| **Redundancy Skeptic** | Does this duplicate existing functionality? Is there already a way to do this? |
| **Context Skeptic** | Does this change make sense given the surrounding code? Will it break assumptions elsewhere? |
| **Overfitting Skeptic** | Is this too specific to one case? Will it generalize? Are we optimizing for the benchmark instead of real usage? |
| **Practitioner** | Would a developer actually use this? Is the API intuitive? Are the defaults sensible? |
| **Theorist** | Does the approach make sense from first principles? Is the algorithm correct? |

### When to rustle

- Before merging any PR that touches retrieval accuracy, the encoding gate, or scoring logic
- Before running benchmarks (catch config drift, missing patches, or inconsistencies)
- On implementation prompts before executing them
- On PR diffs by cross-referencing against the original research or plan

You don't need to rustle a typo fix or a one-line config change.

---

## Workflow: Phased PR

For high-blast-radius changes (anything that touches multiple layers, changes scoring, or could affect benchmark results), we use the Phased PR workflow:

### Steps

1. **Scope document.** Write a formal spec with explicit lists of files that may be modified and files that must not be touched. This prevents scope creep.

2. **Rustle the spec.** Review the spec from multiple adversarial perspectives before writing any code.

3. **Lock decisions.** Once the spec is reviewed and approved, the decisions are locked. No ad-hoc changes during implementation.

4. **Implement.** Execute the spec in a single focused commit. If you discover something outside the scope, file it as a separate issue.

5. **Test-first for bug fixes.** Write the failing test first. Verify it fails on `main`. Implement the fix. Verify the test passes. This order matters.

6. **Rustle the PR.** Before requesting review, run rustle-the-feathers on the diff. Cross-reference the changes against the original spec.

7. **Stop gate.** No PR is merged without review and approval. Stop at each stage to validate.

### Example

For the encoding gate work (PRs #103-#123), each signal was developed as a separate PR:
- `fix/gate-85-cosine-novelty` (compression novelty)
- `feat/gate-108-encoding-salience` (speech-act salience)
- `feat/gate-109-prediction-error` (embedding pair-diff PE)

Each PR was independently tested, rustled, and merged before starting the next.

---

## Testing

### Running tests

```bash
# Full test suite
pytest tests/ -v

# Specific test file
pytest tests/test_encoding_gate_threshold.py -v

# Tests matching a keyword
pytest tests/ -v -k "salience"
```

### Writing tests

- Test files go in `tests/` (flat structure) or `tests/ingest/` for ingest-specific tests.
- Name test files `test_<feature>.py`.
- Test the behavior, not the implementation. If you refactor internals without changing behavior, existing tests should still pass.
- For encoding gate changes, include tests that verify the gate's accept/reject behavior with known inputs.
- For bug fixes, always write the failing test first, verify it fails, then implement the fix.

### CI

CI runs on every push and PR across Python 3.10, 3.11, 3.12, and 3.13. It runs `pytest` and `ruff check`. A separate `build-check` job verifies the package builds and passes `twine check --strict`.

All CI checks must pass before merge.

---

## Code Style

- **Type hints** on all public function signatures.
- **`logging`** instead of `print` for diagnostic output.
- **One layer per file.** Keep modules focused.
- **No star imports** (`from module import *`).
- **No comments unless the "why" is non-obvious.** Well-named identifiers should explain the "what". Don't reference issues, callers, or current tasks in comments.
- **No em dashes in documentation.** Use periods, commas, or colons instead.

### What not to do

- Don't add error handling for scenarios that can't happen. Trust internal code and framework guarantees.
- Don't introduce abstractions beyond what the task requires. Three similar lines is better than a premature abstraction.
- Don't add backwards-compatibility shims. If something is unused, delete it.
- Don't add feature flags or half-finished implementations.
- Don't commit benchmark result JSONs, research docs, or sweep artifacts. Those stay local. Only reproduction scripts and published result files belong on GitHub.

---

## Benchmarks

TrueMemory's claims are backed by reproducible benchmarks on LoCoMo and BEAM-1M. If your change could affect retrieval accuracy:

1. **Rustle before running.** Check for config drift, missing patches, or inconsistencies with prior validated runs.
2. **Run one variant first** to validate your setup, then run 2-3 additional variants with adjustments based on initial results.
3. **Include before/after scores** in your PR description.
4. **If accuracy drops,** explain why the tradeoff is worth it.

Benchmark scripts are in `benchmarks/locomo/scripts/` and `benchmarks/beam/`. They are self-contained and run on [Modal](https://modal.com).

---

## Reporting Issues

Use [GitHub Issues](https://github.com/buildingjoshbetter/TrueMemory/issues). Include:

- Python version and OS
- TrueMemory version (`pip show truememory`)
- Steps to reproduce
- Expected vs actual behavior
- Full error traceback if applicable

---

## License

By contributing, you agree that your contributions will be licensed under the [AGPL-3.0 License](LICENSE). Free for personal and research use. Commercial use requires a separate license.
