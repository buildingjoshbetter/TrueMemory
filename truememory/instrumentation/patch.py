"""
Runtime monkey-patching for instrumentation.

Design invariants:
1. Every wrapper swallows its own exceptions (try/except Exception: pass).
   A telemetry bug must NEVER break engine.add(), engine.search(), etc.
2. Each method gets ONE unified wrapper (no double-wrapping).
3. Per-method sentinel attribute (__truememory_instrumented__) prevents
   double-wrapping from partial re-installs, module reloads, or tests.
4. Thread-safe install latch via threading.Lock.
5. No MCP server tool patches (dead code -- FastMCP stores refs at
   decoration time).
6. No surprise.py import (deleted -- uses gate's native prediction_error).
7. No WAL checkpoint in hot path (SQLite manages via wal_autocheckpoint).
"""

from __future__ import annotations

import functools
import threading
import time

from truememory.instrumentation.log import is_enabled, _lock_enabled, dlog
from truememory.instrumentation import signals, writer

_WRAPPED_SENTINEL = "__truememory_instrumented__"

_install_lock = threading.Lock()
_installed = False

# Store original methods for uninstall
_originals: dict[str, object] = {}


def _is_wrapped(obj, attr: str) -> bool:
    """Check if a method already has our instrumentation sentinel."""
    method = getattr(obj, attr, None)
    if method is None:
        return False
    return getattr(method, _WRAPPED_SENTINEL, False)


def _mark_wrapped(fn):
    """Mark a wrapper function with our sentinel."""
    setattr(fn, _WRAPPED_SENTINEL, True)
    return fn


# ─────────────────────────────────────────────────────────────────────
# Wrapper factories — each method gets ONE dedicated wrapper that
# combines timing + signal emission.  The telemetry-emission block is
# always inside try/except Exception: pass.
# ─────────────────────────────────────────────────────────────────────

def _make_engine_add_wrapper(original_fn):
    """Wrap TrueMemoryEngine.add with timing + category signal."""

    @functools.wraps(original_fn)
    def _patched(self, content, sender="", recipient="",
                 timestamp="", category="", metadata=None):
        if not is_enabled():
            return original_fn(self, content, sender, recipient,
                               timestamp, category, metadata)
        t0 = time.perf_counter()
        try:
            result = original_fn(self, content, sender, recipient,
                                 timestamp, category, metadata)
        except Exception:
            try:
                dur_ms = (time.perf_counter() - t0) * 1000.0
                signals.emit_timing("engine.add", dur_ms, extra={"error": True})
            except Exception:
                pass
            raise
        try:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            signals.emit_timing("engine.add", dur_ms)
            if category:
                signals.emit_category(category)
        except Exception:
            pass  # telemetry failure is invisible
        return result

    return _mark_wrapped(_patched)


def _make_engine_search_wrapper(original_fn, label: str):
    """Wrap TrueMemoryEngine.search or search_agentic."""

    @functools.wraps(original_fn)
    def _patched(self, query, *args, **kwargs):
        if not is_enabled():
            return original_fn(self, query, *args, **kwargs)
        t0 = time.perf_counter()
        try:
            results = original_fn(self, query, *args, **kwargs)
        except Exception:
            try:
                dur_ms = (time.perf_counter() - t0) * 1000.0
                signals.emit_timing(label, dur_ms, extra={"error": True})
            except Exception:
                pass
            raise
        try:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            n_results = len(results) if isinstance(results, list) else 0
            signals.emit_timing(label, dur_ms, n_results=n_results)
            # Emit search distance metrics (metadata only, no query text)
            if isinstance(results, list) and results:
                scores = []
                for r in results:
                    s = r.get("score")
                    if isinstance(s, (int, float)):
                        scores.append(float(s))
                if scores:
                    top_score = max(scores)
                    min_score = min(scores)
                    mean_score = sum(scores) / len(scores)
                    spread = top_score - min_score
                    signals.emit_search_distance(
                        spread=spread,
                        top_score=top_score,
                        min_score=min_score,
                        mean_score=mean_score,
                        n_results=len(scores),
                    )
                # Emit memory_returned for each result
                for rank, r in enumerate(results):
                    mem_id = r.get("id")
                    if mem_id is not None and isinstance(mem_id, int) and mem_id > 0:
                        signals.emit_memory_returned(
                            memory_id=mem_id,
                            rank=rank,
                        )
        except Exception:
            pass  # telemetry failure is invisible
        return results

    return _mark_wrapped(_patched)


def _make_engine_delete_wrapper(original_fn):
    """Wrap TrueMemoryEngine.delete with timing signal."""

    @functools.wraps(original_fn)
    def _patched(self, memory_id):
        if not is_enabled():
            return original_fn(self, memory_id)
        t0 = time.perf_counter()
        try:
            result = original_fn(self, memory_id)
        except Exception:
            try:
                dur_ms = (time.perf_counter() - t0) * 1000.0
                signals.emit_timing("engine.delete", dur_ms, extra={"error": True})
            except Exception:
                pass
            raise
        try:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            signals.emit_timing("engine.delete", dur_ms, extra={
                "deleted": bool(result),
                "memory_id": int(memory_id),
            })
        except Exception:
            pass  # telemetry failure is invisible
        return result

    return _mark_wrapped(_patched)


def _make_memory_add_wrapper(original_fn):
    """Wrap Memory.add (client-level)."""

    @functools.wraps(original_fn)
    def _patched(self, content, user_id=None, metadata=None):
        if not is_enabled():
            return original_fn(self, content, user_id, metadata)
        t0 = time.perf_counter()
        try:
            result = original_fn(self, content, user_id, metadata)
        except Exception:
            try:
                dur_ms = (time.perf_counter() - t0) * 1000.0
                signals.emit_timing("Memory.add", dur_ms, extra={"error": True})
            except Exception:
                pass
            raise
        try:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            signals.emit_timing("Memory.add", dur_ms)
        except Exception:
            pass
        return result

    return _mark_wrapped(_patched)


def _make_memory_search_wrapper(original_fn, label: str):
    """Wrap Memory.search or Memory.search_deep."""

    @functools.wraps(original_fn)
    def _patched(self, query, *args, **kwargs):
        if not is_enabled():
            return original_fn(self, query, *args, **kwargs)
        t0 = time.perf_counter()
        try:
            results = original_fn(self, query, *args, **kwargs)
        except Exception:
            try:
                dur_ms = (time.perf_counter() - t0) * 1000.0
                signals.emit_timing(label, dur_ms, extra={"error": True})
            except Exception:
                pass
            raise
        try:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            n = len(results) if isinstance(results, list) else 0
            signals.emit_timing(label, dur_ms, n_results=n)
        except Exception:
            pass
        return results

    return _mark_wrapped(_patched)


def _make_memory_delete_wrapper(original_fn):
    """Wrap Memory.delete."""

    @functools.wraps(original_fn)
    def _patched(self, memory_id):
        if not is_enabled():
            return original_fn(self, memory_id)
        t0 = time.perf_counter()
        try:
            result = original_fn(self, memory_id)
        except Exception:
            try:
                dur_ms = (time.perf_counter() - t0) * 1000.0
                signals.emit_timing("Memory.delete", dur_ms, extra={"error": True})
            except Exception:
                pass
            raise
        try:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            signals.emit_timing("Memory.delete", dur_ms, extra={
                "deleted": bool(result),
            })
        except Exception:
            pass
        return result

    return _mark_wrapped(_patched)


def _make_gate_evaluate_wrapper(original_fn):
    """Wrap EncodingGate.evaluate with gate_decision + surprise signals."""

    @functools.wraps(original_fn)
    def _patched(self, fact, category=""):
        if not is_enabled():
            return original_fn(self, fact, category)
        try:
            decision = original_fn(self, fact, category)
        except Exception:
            raise
        try:
            # Emit gate decision with structured reason_code
            reason_code = getattr(decision, "reason_code", "")
            if not reason_code:
                # Derive reason_code from decision state if not set
                if decision.salience < getattr(self, "salience_floor", 0.10):
                    reason_code = "salience_floor"
                elif not decision.should_encode:
                    reason_code = "below_threshold"
                else:
                    reason_code = "pass"
            signals.emit_gate_decision(
                should_encode=decision.should_encode,
                encoding_score=decision.encoding_score,
                novelty=decision.novelty,
                salience=decision.salience,
                prediction_error=decision.prediction_error,
                reason_code=reason_code,
                category=category,
            )
            # Emit native prediction_error as the surprise signal
            signals.emit_surprise(score=float(decision.prediction_error))
            # Emit salience
            signals.emit_salience(score=float(decision.salience))
        except Exception:
            pass  # telemetry failure is invisible
        return decision

    return _mark_wrapped(_patched)


# ─────────────────────────────────────────────────────────────────────
# Install / Uninstall
# ─────────────────────────────────────────────────────────────────────

def install() -> None:
    """Install instrumentation patches.

    Requires ``TRUEMEMORY_INSTRUMENTATION=1`` (or truthy).
    Thread-safe, idempotent (won't double-wrap).
    """
    if not is_enabled():
        return

    global _installed
    with _install_lock:
        if _installed:
            return
        _installed = True

    # Lock the enabled state so hot-path checks skip os.environ
    _lock_enabled(True)

    dlog("installing instrumentation patches")

    # Bootstrap: prune old telemetry on startup
    try:
        writer.prune_now()
    except Exception:
        pass

    # Patch TrueMemoryEngine methods
    try:
        from truememory.engine import TrueMemoryEngine

        if not _is_wrapped(TrueMemoryEngine, "add"):
            _originals["TrueMemoryEngine.add"] = TrueMemoryEngine.add
            TrueMemoryEngine.add = _make_engine_add_wrapper(TrueMemoryEngine.add)

        if not _is_wrapped(TrueMemoryEngine, "search"):
            _originals["TrueMemoryEngine.search"] = TrueMemoryEngine.search
            TrueMemoryEngine.search = _make_engine_search_wrapper(
                TrueMemoryEngine.search, "engine.search"
            )

        if not _is_wrapped(TrueMemoryEngine, "search_agentic"):
            _originals["TrueMemoryEngine.search_agentic"] = TrueMemoryEngine.search_agentic
            TrueMemoryEngine.search_agentic = _make_engine_search_wrapper(
                TrueMemoryEngine.search_agentic, "engine.search_agentic"
            )

        if not _is_wrapped(TrueMemoryEngine, "delete"):
            _originals["TrueMemoryEngine.delete"] = TrueMemoryEngine.delete
            TrueMemoryEngine.delete = _make_engine_delete_wrapper(TrueMemoryEngine.delete)

        dlog("patched TrueMemoryEngine methods")
    except Exception as exc:
        dlog("failed to patch TrueMemoryEngine: %s", type(exc).__name__)

    # Patch Memory (client) methods
    try:
        from truememory.client import Memory

        if not _is_wrapped(Memory, "add"):
            _originals["Memory.add"] = Memory.add
            Memory.add = _make_memory_add_wrapper(Memory.add)

        if not _is_wrapped(Memory, "search"):
            _originals["Memory.search"] = Memory.search
            Memory.search = _make_memory_search_wrapper(Memory.search, "Memory.search")

        if not _is_wrapped(Memory, "search_deep"):
            _originals["Memory.search_deep"] = Memory.search_deep
            Memory.search_deep = _make_memory_search_wrapper(
                Memory.search_deep, "Memory.search_deep"
            )

        if not _is_wrapped(Memory, "delete"):
            _originals["Memory.delete"] = Memory.delete
            Memory.delete = _make_memory_delete_wrapper(Memory.delete)

        dlog("patched Memory methods")
    except Exception as exc:
        dlog("failed to patch Memory: %s", type(exc).__name__)

    # Patch EncodingGate.evaluate
    try:
        from truememory.ingest.encoding_gate import EncodingGate

        if not _is_wrapped(EncodingGate, "evaluate"):
            _originals["EncodingGate.evaluate"] = EncodingGate.evaluate
            EncodingGate.evaluate = _make_gate_evaluate_wrapper(EncodingGate.evaluate)

        dlog("patched EncodingGate.evaluate")
    except Exception as exc:
        dlog("failed to patch EncodingGate: %s", type(exc).__name__)

    dlog("instrumentation installed")


def uninstall() -> None:
    """Remove instrumentation patches and close the writer."""
    global _installed
    with _install_lock:
        _installed = False

    _lock_enabled(None)

    # Restore original methods
    for key, original in _originals.items():
        try:
            parts = key.split(".")
            if len(parts) == 2:
                cls_name, method_name = parts
                if cls_name == "TrueMemoryEngine":
                    from truememory.engine import TrueMemoryEngine
                    setattr(TrueMemoryEngine, method_name, original)
                elif cls_name == "Memory":
                    from truememory.client import Memory
                    setattr(Memory, method_name, original)
                elif cls_name == "EncodingGate":
                    from truememory.ingest.encoding_gate import EncodingGate
                    setattr(EncodingGate, method_name, original)
        except Exception:
            pass

    _originals.clear()
    writer.close()
    dlog("instrumentation uninstalled")
