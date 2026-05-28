"""Runtime monkey-patches that wire telemetry onto the TrueMemory engine.

This is the "sits on top" core of the overlay. ``install()`` replaces a handful
of engine + MCP-server methods with thin wrappers that call the ``signals``
emitters and then delegate to the original. There are NO emit calls woven into
the engine source — everything is applied here at runtime.

Robustness contract:

- **Opt-in.** ``install()`` is a no-op unless ``TRUEMEMORY_INSTRUMENTATION=1``
  (checked via ``is_enabled()``). Default off → nothing is patched.
- **Idempotent.** A module-level latch means a second ``install()`` call does
  nothing.
- **Independently isolated.** Each patch is applied inside its own
  ``try/except``. If an upstream method has been renamed or moved (Josh changes
  the core), that one patch logs a miss and is skipped — it can neither break
  the other patches nor disturb the host. A patched method that moves simply
  stops being wrapped; the engine keeps working.

Patches stay applied for the life of the process; there is no uninstall hook.
"""
from __future__ import annotations

import functools
import time
import os

from truememory.instrumentation.log import dlog, is_enabled
from truememory.instrumentation import signals

_installed = False


def _timed_phase(label: str):
    """Decorator: log ENTER + EXIT/ERROR with elapsed ms. Falls through when
    disabled."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not is_enabled():
                return fn(*args, **kwargs)
            t = time.time()
            dlog(f"{label} ENTER")
            try:
                result = fn(*args, **kwargs)
                dlog(f"{label} EXIT total {(time.time() - t) * 1000:.0f}ms")
                return result
            except Exception as exc:
                dlog(
                    f"{label} ERROR after {(time.time() - t) * 1000:.0f}ms: "
                    f"{type(exc).__name__}: {exc}"
                )
                raise

        return wrapper

    return decorator


def _wrap_engine_add(original_add):
    """Wrap TrueMemoryEngine.add with phase timing, WAL-checkpoint logging, and
    a per-memory category signal."""

    @functools.wraps(original_add)
    def patched_add(self, content, sender="", recipient="", timestamp="", category="", metadata=None):
        if not is_enabled():
            return original_add(
                self, content, sender=sender, recipient=recipient,
                timestamp=timestamp, category=category, metadata=metadata,
            )
        t_total = time.time()
        dlog(f"engine.add ENTER content_len={len(content)} sender={sender!r}")
        try:
            result = original_add(
                self, content, sender=sender, recipient=recipient,
                timestamp=timestamp, category=category, metadata=metadata,
            )
            elapsed = (time.time() - t_total) * 1000
            new_id = result.get("id") if isinstance(result, dict) else None
            dlog(f"engine.add EXIT total {elapsed:.0f}ms id={new_id}")
            # WAL checkpoint snapshot AFTER commit (commit happens inside original_add)
            try:
                signals.log_wal_checkpoint(self.conn)
            except Exception as exc:
                dlog(f"wal_checkpoint hook failed: {type(exc).__name__}: {exc}")
            # category signal — fires once per stored memory that has a category.
            # salience/surprise/gate_decision are emitted from the encoding gate
            # patch below (they're computed there, before a memory_id exists).
            if new_id is not None and category:
                try:
                    signals.emit_category(new_id, category)
                except Exception as exc:
                    dlog(f"emit_category failed: {type(exc).__name__}: {exc}")
            return result
        except Exception as exc:
            elapsed = (time.time() - t_total) * 1000
            dlog(
                f"engine.add ERROR after {elapsed:.0f}ms: "
                f"{type(exc).__name__}: {exc}"
            )
            raise

    return patched_add


def _wrap_embed_single(original_embed):
    """Wrap vector_search.embed_single with cold-vs-warm classification."""

    @functools.wraps(original_embed)
    def patched_embed(*args, **kwargs):
        if not is_enabled():
            return original_embed(*args, **kwargs)
        cold = signals.is_model_cold()
        t = time.time()
        try:
            result = original_embed(*args, **kwargs)
            elapsed = (time.time() - t) * 1000
            dlog(f"embed_single cold={cold} done in {elapsed:.0f}ms")
            if cold:
                signals.mark_model_warm()
            return result
        except Exception as exc:
            elapsed = (time.time() - t) * 1000
            dlog(
                f"embed_single ERROR cold={cold} after {elapsed:.0f}ms: "
                f"{type(exc).__name__}: {exc}"
            )
            raise

    return patched_embed


def install() -> None:
    """Apply all telemetry monkey-patches to TrueMemory. Idempotent, opt-in.

    No-op unless ``TRUEMEMORY_INSTRUMENTATION=1``. Each patch is independently
    exception-isolated, so a renamed/moved upstream method only disables that
    one patch — never the host.
    """
    # Opt-in gate FIRST — a disabled call must not trip the idempotency latch,
    # so a later enabled call in the same process (e.g. a test) still installs.
    if not is_enabled():
        return

    global _installed
    if _installed:
        return
    _installed = True

    dlog(
        f"=== instrumentation installing (pid={os.getpid()}, "
        f"ppid={os.getppid()}) ==="
    )
    signals.log_spawner_classification()
    signals.install_hf_hub_intercept()

    # Engine: replace add() with phase-timed version that also logs WAL state +
    # emits the category signal.
    try:
        from truememory.engine import TrueMemoryEngine
        TrueMemoryEngine.add = _wrap_engine_add(TrueMemoryEngine.add)
        TrueMemoryEngine._ensure_connection = _timed_phase("engine._ensure_connection")(
            TrueMemoryEngine._ensure_connection
        )
        dlog("patched truememory.engine.TrueMemoryEngine.add + _ensure_connection")
    except Exception as exc:
        dlog(f"engine patch failed: {type(exc).__name__}: {exc}")

    # Memory client class methods — the layer MCP tools delegate to. Patching
    # at the class level catches every call regardless of how FastMCP stores
    # its tool references, because instance attribute lookup always resolves
    # through the class.
    try:
        from truememory.client import Memory
        for name in ("add", "search", "search_deep", "get", "delete"):
            if hasattr(Memory, name):
                fn = getattr(Memory, name)
                setattr(Memory, name, _timed_phase(f"Memory.{name}")(fn))
        dlog("patched truememory.client.Memory methods (add/search/search_deep/get/delete)")
    except Exception as exc:
        dlog(f"Memory patch failed: {type(exc).__name__}: {exc}")

    # MCP server tool functions — best-effort. May or may not fire depending on
    # whether FastMCP stored a direct reference at decoration time. Harmless if
    # it doesn't fire; the Memory.* patches above provide redundant coverage.
    try:
        from truememory import mcp_server as _mcp
        tool_names = (
            "truememory_store",
            "truememory_search",
            "truememory_search_deep",
            "truememory_get",
            "truememory_forget",
            "truememory_stats",
            "truememory_entity_profile",
            "truememory_configure",
        )
        for name in tool_names:
            if hasattr(_mcp, name):
                fn = getattr(_mcp, name)
                setattr(_mcp, name, _timed_phase(f"mcp.{name}")(fn))
        dlog(f"patched truememory.mcp_server tool functions (best-effort): {tool_names}")
    except Exception as exc:
        dlog(f"mcp_server patch failed: {type(exc).__name__}: {exc}")

    # vector_search.embed_single — cold-vs-warm tracking
    try:
        from truememory import vector_search as _vs
        if hasattr(_vs, "embed_single"):
            _vs.embed_single = _wrap_embed_single(_vs.embed_single)
            dlog("patched truememory.vector_search.embed_single")
    except Exception as exc:
        dlog(f"vector_search patch failed: {type(exc).__name__}: {exc}")

    # ---- Model-lifecycle + semantic-signal patches (telemetry table writes) ----
    _install_model_lifecycle_patches()
    _install_semantic_signal_patches()

    # Bootstrap: emit instrumentation_start so the telemetry table is created
    # immediately, even before any store/search call fires. (Renamed from the
    # overlay's "diag_install" — there is no diag install upstream.)
    try:
        from truememory.instrumentation import writer
        writer.emit("instrumentation_start", context={"pid": os.getpid()})
        dlog("telemetry writer bootstrap emit done (table created)")
    except Exception as exc:
        dlog(f"telemetry writer bootstrap failed: {type(exc).__name__}: {exc}")

    # Close the instrumentation connection cleanly at process exit so we don't
    # leak the WAL handle. atexit is best-effort but works for normal shutdown.
    try:
        import atexit
        from truememory.instrumentation import writer
        atexit.register(writer.close)
    except Exception as exc:
        dlog(f"telemetry writer atexit hook failed: {type(exc).__name__}: {exc}")

    dlog("=== instrumentation installed ===")


def _install_model_lifecycle_patches() -> None:
    """Wire the preload / degraded / unload lifecycle telemetry.

    Each patch is independent and exception-isolated — if the upstream function
    doesn't exist (older TrueMemory, or a future rename), we log the miss and
    continue. The rest of the overlay stays functional.
    """
    # _preload_models entry: emit preload_start so the dashboard can pair it
    # with a later preload_complete row.
    try:
        from truememory import mcp_server as _mcp
        if hasattr(_mcp, "_preload_models"):
            _original_preload = _mcp._preload_models

            @functools.wraps(_original_preload)
            def _patched_preload(*args, **kwargs):
                try:
                    signals.emit_preload_start()
                except Exception as exc:
                    dlog(f"emit_preload_start failed: {type(exc).__name__}: {exc}")
                return _original_preload(*args, **kwargs)

            _mcp._preload_models = _patched_preload
            dlog("patched truememory.mcp_server._preload_models (+ emit_preload_start)")
    except Exception as exc:
        dlog(f"_preload_models patch failed: {type(exc).__name__}: {exc}")

    # reranker.get_reranker: emit preload_complete on cold load (when the
    # module-level singleton transitions from None to loaded).
    try:
        from truememory import reranker as _rr
        if hasattr(_rr, "get_reranker"):
            _original_get_reranker = _rr.get_reranker

            @functools.wraps(_original_get_reranker)
            def _patched_get_reranker(*args, **kwargs):
                was_cold = getattr(_rr, "_model", None) is None
                t = time.time()
                result = _original_get_reranker(*args, **kwargs)
                if was_cold and getattr(_rr, "_model", None) is not None:
                    load_ms = (time.time() - t) * 1000
                    try:
                        signals.emit_preload_complete("reranker", load_ms)
                    except Exception as exc:
                        dlog(
                            f"emit_preload_complete failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                return result

            _rr.get_reranker = _patched_get_reranker
            dlog("patched truememory.reranker.get_reranker (+ emit_preload_complete)")
    except Exception as exc:
        dlog(f"get_reranker patch failed: {type(exc).__name__}: {exc}")

    # model_client.get_reranker_proxy: the observable "reranker degraded" event
    # in the current architecture is the shared model-server proxy failing and
    # forcing an expensive local fallback load. reranker.get_reranker re-imports
    # this name on every call, catches the proxy's Exception, and loads
    # CrossEncoder locally. We patch the proxy getter to emit reranker_degraded
    # the moment it raises — then re-raise so get_reranker's own fallback path
    # runs exactly as before. Fully isolated: any failure here is swallowed.
    try:
        from truememory import model_client as _mc
        if hasattr(_mc, "get_reranker_proxy"):
            _original_get_reranker_proxy = _mc.get_reranker_proxy

            @functools.wraps(_original_get_reranker_proxy)
            def _patched_get_reranker_proxy(*args, **kwargs):
                try:
                    return _original_get_reranker_proxy(*args, **kwargs)
                except Exception as exc:
                    try:
                        signals.emit_reranker_degraded(
                            f"model_server_proxy_failed:{type(exc).__name__}"
                        )
                    except Exception as exc2:
                        dlog(
                            f"emit_reranker_degraded failed: "
                            f"{type(exc2).__name__}: {exc2}"
                        )
                    raise

            _mc.get_reranker_proxy = _patched_get_reranker_proxy
            dlog(
                "patched truememory.model_client.get_reranker_proxy "
                "(+ emit_reranker_degraded on proxy-fail fallback)"
            )
        else:
            dlog(
                "model_client.get_reranker_proxy not found — reranker_degraded "
                "stays unwired (model-server path unavailable in this build)."
            )
    except Exception as exc:
        dlog(f"get_reranker_proxy patch failed: {type(exc).__name__}: {exc}")

    # reranker.unload_reranker: emit model_unload when the idle timer fires.
    try:
        from truememory import reranker as _rr
        if hasattr(_rr, "unload_reranker"):
            _original_unload = _rr.unload_reranker

            @functools.wraps(_original_unload)
            def _patched_unload(*args, **kwargs):
                try:
                    signals.emit_model_unload("reranker")
                except Exception as exc:
                    dlog(
                        f"emit_model_unload failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                return _original_unload(*args, **kwargs)

            _rr.unload_reranker = _patched_unload
            dlog("patched truememory.reranker.unload_reranker (+ emit_model_unload)")
    except Exception as exc:
        dlog(f"unload_reranker patch failed: {type(exc).__name__}: {exc}")


def _install_semantic_signal_patches() -> None:
    """Wire the per-memory semantic telemetry signals.

    Wired signals:
    - user_forget     — Memory.delete (1 row per explicit delete call)
    - gate_decision   — EncodingGate.evaluate (1 row per candidate fact, pass or reject)
    - salience        — EncodingGate.evaluate (1 row per candidate fact)
    - surprise        — EncodingGate.evaluate (1 row per candidate fact; the
                        instrumentation layer's novelty-ratio heuristic, NOT
                        the engine's consolidation-based surprise_scores)
    - search_distance — Memory.search + Memory.search_deep (1 row per call,
                        score spread + n_results)
    - memory_returned — Memory.search + Memory.search_deep (1 row per result in
                        the returned list, 0-based rank)
    """
    # Memory.delete: emit user_forget.
    try:
        from truememory.client import Memory
        if hasattr(Memory, "delete"):
            _original_delete = Memory.delete

            @functools.wraps(_original_delete)
            def _patched_delete(self, memory_id: int, *args, **kwargs):
                result = _original_delete(self, memory_id, *args, **kwargs)
                try:
                    # match_count is 1 if delete returned True (single-id delete),
                    # 0 if False.
                    match_count = 1 if result else 0
                    signals.emit_user_forget(
                        memory_id=memory_id if result else None,
                        requested_text=f"id={memory_id}",
                        match_count=match_count,
                    )
                except Exception as exc:
                    dlog(
                        f"emit_user_forget failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                return result

            Memory.delete = _patched_delete
            dlog("patched truememory.client.Memory.delete (+ emit_user_forget)")
    except Exception as exc:
        dlog(f"Memory.delete patch failed: {type(exc).__name__}: {exc}")

    # EncodingGate.evaluate: emit gate_decision + salience + surprise.
    # One telemetry row per candidate fact, fires before a memory_id exists.
    try:
        from truememory.ingest.encoding_gate import EncodingGate
        if hasattr(EncodingGate, "evaluate"):
            _original_evaluate = EncodingGate.evaluate

            @functools.wraps(_original_evaluate)
            def _patched_evaluate(self, fact, category=""):
                decision = _original_evaluate(self, fact, category)
                try:
                    outcome = "pass" if decision.should_encode else "reject"
                    reason_text = (decision.reason or "").lower()
                    if "salience" in reason_text and "floor" in reason_text:
                        reason_code = "salience_floor"
                    elif "novelty" in reason_text and (
                        "low" in reason_text or "too" in reason_text
                    ):
                        reason_code = "novelty_too_low"
                    elif "pred" in reason_text and "low" in reason_text:
                        reason_code = "pred_error_low"
                    elif outcome == "reject":
                        reason_code = "combined_below_threshold"
                    else:
                        reason_code = "pass"
                    signals.emit_gate_decision(
                        memory_id=None,
                        score=float(decision.encoding_score),
                        outcome=outcome,
                        reason_code=reason_code,
                        context={
                            "novelty": float(decision.novelty),
                            "salience": float(decision.salience),
                            "prediction_error": float(decision.prediction_error),
                            "category": category or "",
                            "fact_preview": (fact or "")[:60],
                        },
                    )
                    signals.emit_salience(
                        memory_id=None,
                        score=float(decision.salience),
                    )
                except Exception as exc:
                    dlog(
                        f"emit gate_decision/salience failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                # surprise signal — the instrumentation layer's novelty-ratio
                # heuristic, computed here because the engine's native
                # surprise_scores requires consolidate() which may be off.
                try:
                    from truememory.instrumentation import surprise as _surprise
                    surprise_score = _surprise.compute_surprise(fact)
                    signals.emit_surprise(
                        memory_id=None,
                        score=float(surprise_score),
                    )
                except Exception as exc:
                    dlog(f"emit_surprise failed: {type(exc).__name__}: {exc}")
                return decision

            EncodingGate.evaluate = _patched_evaluate
            dlog(
                "patched truememory.ingest.encoding_gate.EncodingGate.evaluate"
                " (+ emit_gate_decision + emit_salience + emit_surprise)"
            )
        else:
            dlog("EncodingGate.evaluate not found — encoding_gate patches skipped")
    except Exception as exc:
        dlog(f"EncodingGate.evaluate patch failed: {type(exc).__name__}: {exc}")

    # Memory.search + Memory.search_deep: emit search_distance + memory_returned.
    # search_distance: 1 row per call. memory_returned: 1 row per result.
    # Results are dicts with an "id" key.
    def _wrap_search(original_fn, label: str):
        """Factory so both search and search_deep share identical emit logic."""
        @functools.wraps(original_fn)
        def _patched(self, query: str, *args, **kwargs):
            results = original_fn(self, query, *args, **kwargs)
            try:
                # search_distance value_num = SPREAD of the returned score
                # distribution (top - bottom). After the reranker's per-call
                # min-max normalization the top is pinned to ~1.0, so spread —
                # not top — is the real discrimination signal. Raw top/min/mean
                # go into context_json for absolute plotting.
                scores = [
                    float(r.get("score", 0.0))
                    for r in results
                    if isinstance(r.get("score"), (int, float))
                ]
                if scores:
                    top_score = scores[0]            # results are pre-sorted desc
                    min_score = min(scores)
                    mean_score = sum(scores) / len(scores)
                    spread = top_score - min_score   # the informative signal
                else:
                    top_score = min_score = mean_score = spread = 0.0
                signals.emit_search_distance(
                    query_text=query,
                    spread=spread,
                    top_score=top_score,
                    min_score=min_score,
                    mean_score=mean_score,
                    n_results=len(results),
                )
                for rank, result in enumerate(results):
                    mem_id = result.get("id")
                    if mem_id is not None:
                        signals.emit_memory_returned(
                            memory_id=int(mem_id),
                            rank=rank,
                            query_text=query,
                        )
            except Exception as exc:
                dlog(
                    f"emit search_distance/memory_returned ({label}) failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            return results
        return _patched

    try:
        from truememory.client import Memory
        if hasattr(Memory, "search"):
            Memory.search = _wrap_search(Memory.search, "search")
            dlog(
                "patched truememory.client.Memory.search"
                " (+ emit_search_distance + emit_memory_returned)"
            )
        else:
            dlog("Memory.search not found — search patches skipped")
        if hasattr(Memory, "search_deep"):
            Memory.search_deep = _wrap_search(Memory.search_deep, "search_deep")
            dlog(
                "patched truememory.client.Memory.search_deep"
                " (+ emit_search_distance + emit_memory_returned)"
            )
        else:
            dlog("Memory.search_deep not found — search_deep patch skipped")
    except Exception as exc:
        dlog(f"Memory.search patch failed: {type(exc).__name__}: {exc}")
