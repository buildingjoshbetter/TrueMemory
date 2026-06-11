# TrueMemory Phase 1: Memory Efficiency Bug Fixes

## Context

TrueMemory is consuming 10.5 GB across 9 processes on a 24 GB Mac. The root cause is a
silent fallback chain: PR #343 implemented a shared model server that works correctly, but
three layers of error suppression hide connection failures, causing every MCP process to
load 2-3 GB of models locally instead of using the shared server.

The full audit report is at `/Users/j/Desktop/TRUEMEMORY_MEMORY_AUDIT.html`.

The repo is at `/Users/j/Desktop/TrueMemory/` — work directly on the `main` branch.

## Before Starting

```bash
cd /Users/j/Desktop/TrueMemory
git checkout main
git pull origin main
```

## What You Must Do

Fix exactly 5 bugs across 5 files. Every change is surgical — no refactoring, no new
features, no architectural changes. The shared model server architecture from PR #343 is
correct. We just need to make it stop failing silently.

After all fixes, commit, push to GitHub, and verify.

---

## Bug 1: model_server.py — Bare `import psutil` crashes the server

**File:** `truememory/model_server.py`
**Problem:** Line 11 has `import psutil` at the top level. If psutil is missing or fails
to import (common in `uv tool install` environments), the entire model server crashes
before it can create its socket. Since stderr is swallowed (Bug 2), this is completely
silent.

**Fix:** Wrap the psutil import and `_set_mps_memory_cap()` in a try/except so the model
server can still start without psutil, using a safe default memory ratio.

**Current code (lines 10-31):**
```python
import os
import psutil


def _set_mps_memory_cap():
    """Set MPS memory cap BEFORE torch is imported."""
    if os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"):
        return
    total_gb = psutil.virtual_memory().total / (1024**3)
    if total_gb >= 32:
        ratio = "0.55"
    else:
        ratio = "0.50"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = ratio
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")


_set_mps_memory_cap()
```

**Change to:**
```python
import os

try:
    import psutil
except ImportError:
    psutil = None


def _set_mps_memory_cap():
    """Set MPS memory cap BEFORE torch is imported."""
    if os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"):
        return
    if psutil is not None:
        total_gb = psutil.virtual_memory().total / (1024**3)
        ratio = "0.55" if total_gb >= 32 else "0.50"
    else:
        ratio = "0.50"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = ratio
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")


_set_mps_memory_cap()
```

---

## Bug 2: model_client.py — stderr goes to /dev/null

**File:** `truememory/model_client.py`
**Problem:** `_start_server()` at lines 155 and 166 uses `stderr=subprocess.DEVNULL` when
spawning the model server. When the model server crashes (e.g., from Bug 1), the crash
traceback is completely swallowed. Zero diagnostic output.

**Fix:** Route stderr to a log file at `~/.truememory/model_server.stderr` instead of
DEVNULL. This makes model server crashes diagnosable.

**Find this code block (the primary Popen, around line 154-161):**
```python
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
```

**Change to:**
```python
        _stderr_path = _TRUEMEMORY_DIR / "model_server.stderr"
        _stderr_fh = open(_stderr_path, "a")
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=_stderr_fh,
            start_new_session=True,
            env=env,
        )
```

**Also find the fallback Popen (around line 166-170):**
```python
                subprocess.Popen(
                    [sys.executable, "-m", "truememory.model_server"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
```

**Change to:**
```python
                _stderr_fh2 = open(_stderr_path, "a")
                subprocess.Popen(
                    [sys.executable, "-m", "truememory.model_server"],
                    stdout=subprocess.DEVNULL,
                    stderr=_stderr_fh2,
                    start_new_session=True,
                )
```

---

## Bug 3: vector_search.py — Silent fallback loads 2-3 GB per process

**File:** `truememory/vector_search.py`
**Problem:** `get_model()` at line 201 has `except Exception: pass` which silently falls
through to loading the full SentenceTransformer locally when the model server is
unavailable. This is the single biggest cause of memory bloat — every MCP process quietly
becomes a 2-3 GB model host.

**Fix:** Log a loud warning when falling back to local loading. Do NOT silently pass.

**Current code (lines 196-203):**
```python
        from truememory.model_client import use_model_server, get_embedding_proxy
        if use_model_server():
            try:
                proxy = get_embedding_proxy(tier=EMBEDDING_MODEL)
                _model = proxy
                return _model
            except Exception:
                pass  # Fall through to local loading
```

**Change to:**
```python
        from truememory.model_client import use_model_server, get_embedding_proxy
        if use_model_server():
            try:
                proxy = get_embedding_proxy(tier=EMBEDDING_MODEL)
                _model = proxy
                return _model
            except Exception:
                log.warning(
                    "Model server available but embedding proxy failed — "
                    "falling back to local model loading (high memory cost). "
                    "Check ~/.truememory/model_server.stderr for details."
                )
```

Make sure `log` is available at the top of the file. Check if there's already a
`log = logging.getLogger(...)` near the top — there likely is. If not, add:
```python
import logging
log = logging.getLogger(__name__)
```

---

## Bug 4: reranker.py — Same silent fallback as Bug 3

**File:** `truememory/reranker.py`
**Problem:** `get_reranker()` at line 185 has the identical `except Exception: pass`
pattern. The original fix commit (5857d93) never even touched this file — it only fixed
vector_search.py before being reverted.

**Fix:** Same as Bug 3 — log a loud warning instead of silently passing.

**Current code (lines 178-186):**
```python
        from truememory.model_client import use_model_server, get_reranker_proxy
        if use_model_server():
            try:
                proxy = get_reranker_proxy(model_name=name)
                _model = proxy
                _model_name = name
                return _model
            except Exception:
                pass  # Fall through to local loading
```

**Change to:**
```python
        from truememory.model_client import use_model_server, get_reranker_proxy
        if use_model_server():
            try:
                proxy = get_reranker_proxy(model_name=name)
                _model = proxy
                _model_name = name
                return _model
            except Exception:
                log.warning(
                    "Model server available but reranker proxy failed — "
                    "falling back to local model loading (high memory cost). "
                    "Check ~/.truememory/model_server.stderr for details."
                )
```

Same as Bug 3 — make sure `log` is available. Check for an existing logger at the top.

---

## Bug 5: mcp_server.py — _unload_models() doesn't flush MPS GPU memory

**File:** `truememory/mcp_server.py`
**Problem:** `_unload_models()` at lines 991-1003 sets model references to None and calls
`gc.collect()`, but never calls `torch.mps.empty_cache()`. On Apple Silicon, this leaks
~1 GB of MPS/GPU memory per process per unload cycle. The correct code already exists in
`model_server.py:214-223` (`_flush_mps_cache`) but is never called from the unload path.

**Fix:** Add MPS cache flushing before `gc.collect()`.

**Current code (lines 991-1004):**
```python
def _unload_models() -> None:
    try:
        from truememory.vector_search import unload_model
        unload_model()
    except Exception:
        pass
    try:
        from truememory.reranker import unload_reranker
        unload_reranker()
    except Exception:
        pass
    gc.collect()
    log.info("Models unloaded (idle timeout). RSS=%.0f MB", _get_rss_mb())
```

**Change to:**
```python
def _unload_models() -> None:
    try:
        from truememory.vector_search import unload_model
        unload_model()
    except Exception:
        pass
    try:
        from truememory.reranker import unload_reranker
        unload_reranker()
    except Exception:
        pass
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
    except Exception:
        pass
    gc.collect()
    log.info("Models unloaded (idle timeout). RSS=%.0f MB", _get_rss_mb())
```

---

## After All Fixes

### 1. Verify the changes compile
```bash
cd /Users/j/Desktop/TrueMemory
.venv/bin/python -c "import truememory" 2>&1 && echo "OK" || echo "IMPORT FAILED"
```

### 2. Run any existing tests
```bash
cd /Users/j/Desktop/TrueMemory
.venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -20
```
If tests fail on unrelated issues, note them but don't block on them.

### 3. Commit and push
Create a single commit with a clear message explaining all 5 fixes. Push to main.

Commit message should be:
```
fix: eliminate silent model server fallback causing 2-3GB per MCP process

5 bugs fixed:
1. model_server.py: psutil import wrapped in try/except (prevents silent crash)
2. model_client.py: stderr routed to log file instead of /dev/null (crashes now diagnosable)
3. vector_search.py: silent fallback now logs warning (no more invisible 2GB loads)
4. reranker.py: same silent fallback fix (this file was missed in the original fix attempt)
5. mcp_server.py: _unload_models() now calls torch.mps.empty_cache() (fixes 1GB MPS leak)

Root cause: PR #343 implemented a shared model server, but three layers of error
suppression hid connection failures. The correct fix (commit 5857d93) was reverted
in 86f405b — this re-applies those fixes plus the reranker.py fix that was missed
and the MPS cache leak fix (#338).

Closes #338.
```

### 4. Verify model server usage
After pushing, to test:
1. Kill ALL TrueMemory processes: `pkill -f truememory`
2. Start a fresh Claude Code session
3. Run a truememory_search query
4. Check Activity Monitor — you should see:
   - "TrueMemory" (model server) at ~1.5-2.5 GB (the ONE shared copy)
   - "TrueMemory MCP" at ~50-80 MB (thin proxy, no models loaded)
5. Check `~/.truememory/model_server.stderr` — should be empty or have clean startup logs

Expected total memory: ~2.5-3 GB for the model server + ~50 MB per MCP session.
Down from ~10.5 GB.

---

## What NOT to Do

- Do NOT refactor surrounding code
- Do NOT add new features
- Do NOT change model loading logic beyond the error handling
- Do NOT touch the model server architecture (PR #343 is correct)
- Do NOT modify pyproject.toml or dependencies
- Do NOT change the idle timeout values
- Do NOT try to implement ONNX migration (that's Phase 2)
- Do NOT create new files
- Do NOT change the Unix domain socket protocol
