"""Regression tests for issue #577: model server device override, sticky-CPU
OOM degradation outside the global lock, rerank OOM handling, single-text
fast lane, and client deadline differentiation.

Design consensus (7-model panel):
1. Shared ``resolve_device`` helper honoring ``TRUEMEMORY_DEVICE`` at all
   load sites (server embed, server reranker, local reranker, throttler).
2. MPS OOM marks the affected model sticky-CPU for the server's lifetime;
   the expensive re-encode runs OUTSIDE the global request lock; the model
   is never re-promoted to MPS.
3. The rerank path gets the same OOM handler (previously none — raw
   RuntimeError reached the client, 3/3 rerank deaths in baseline probe).
4. Single-text encode requests (hook recall) never queue behind batch work.
5. ``EmbeddingProxy.encode`` / ``RerankerProxy.predict`` accept a per-call
   ``timeout``; expiry fast-fails with TimeoutError (no autostart retry);
   hook recall paths set a short process-wide deadline (~5s).

All tests are device-mockable: CI has no MPS/CUDA. MPS OOM is faked with a
string-matched RuntimeError that satisfies ``mps_utils.is_mps_oom`` (same
pattern as tests/test_issue_459_mps_oom.py). Thread governor: each test
spawns at most 2 extra threads.
"""
from __future__ import annotations

import socket
import threading
import types
from unittest.mock import MagicMock

import numpy as np
import pytest


def _make_mps_oom_error():
    """Create a RuntimeError that looks like an MPS OOM (house pattern)."""
    return RuntimeError(
        "MPS backend out of memory (MPS allocated: 2.16 GiB, "
        "other allocations: 5.69 MiB, max allowed: 1.95 GiB). "
        "Tried to allocate 3.00 KiB on private pool."
    )


def _fake_sentence_transformers(captured: dict):
    """Build a fake ``sentence_transformers`` module that records the
    ``device`` argument passed to CrossEncoder / SentenceTransformer."""
    mod = types.ModuleType("sentence_transformers")

    class FakeCrossEncoder:
        def __init__(self, name, device=None, **kwargs):
            captured["cross_encoder_device"] = device
            captured["cross_encoder_name"] = name

        def predict(self, pairs, **kwargs):
            return np.zeros(len(pairs), dtype=np.float32)

    class FakeSentenceTransformer:
        def __init__(self, name, device=None, **kwargs):
            captured["st_device"] = device
            captured["st_name"] = name

        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 4), dtype=np.float32)

    mod.CrossEncoder = FakeCrossEncoder
    mod.SentenceTransformer = FakeSentenceTransformer
    return mod


@pytest.fixture()
def no_flush(monkeypatch):
    """Stub out the MPS cache flush so OOM tests never import torch."""
    from truememory import mps_utils
    monkeypatch.setattr(mps_utils, "flush_mps_cache", lambda: None)


# ---------------------------------------------------------------------------
# 1. TRUEMEMORY_DEVICE override
# ---------------------------------------------------------------------------

class TestIssue577DeviceEnvOverride:
    def test_issue_577_device_env_override_cpu_resolve_helper(self, monkeypatch):
        """resolve_device honors TRUEMEMORY_DEVICE=cpu regardless of torch."""
        from truememory.mps_utils import resolve_device
        monkeypatch.setenv("TRUEMEMORY_DEVICE", "cpu")
        assert resolve_device("mps") == "cpu"
        assert resolve_device(None) == "cpu"

    def test_issue_577_device_env_override_unset_uses_default_auto(self, monkeypatch):
        """Unset / 'auto' returns the caller's default_auto unchanged."""
        from truememory.mps_utils import resolve_device
        monkeypatch.delenv("TRUEMEMORY_DEVICE", raising=False)
        assert resolve_device("mps") == "mps"
        assert resolve_device(None) is None
        monkeypatch.setenv("TRUEMEMORY_DEVICE", "auto")
        assert resolve_device("cuda:0") == "cuda:0"

    def test_issue_577_device_env_override_invalid(self, monkeypatch, caplog):
        """Invalid TRUEMEMORY_DEVICE warns and falls back to default_auto."""
        from truememory.mps_utils import resolve_device
        monkeypatch.setenv("TRUEMEMORY_DEVICE", "tpu")
        with caplog.at_level("WARNING", logger="truememory.mps_utils"):
            assert resolve_device("mps") == "mps"
        assert any("TRUEMEMORY_DEVICE" in r.message for r in caplog.records), (
            "invalid TRUEMEMORY_DEVICE value must log a warning"
        )

    def test_issue_577_device_env_override_cpu_server_reranker(self, monkeypatch):
        """The server reranker load site honors TRUEMEMORY_DEVICE=cpu."""
        import sys
        from truememory.model_server import ModelServer

        captured: dict = {}
        monkeypatch.setitem(
            sys.modules, "sentence_transformers", _fake_sentence_transformers(captured)
        )
        monkeypatch.setenv("TRUEMEMORY_DEVICE", "cpu")

        server = ModelServer()
        server._get_reranker("test-model")
        assert captured["cross_encoder_device"] == "cpu"

    def test_issue_577_device_env_override_cpu_server_embed(self, monkeypatch):
        """The server embed load site passes the resolved device through."""
        import sys
        from truememory.model_server import ModelServer

        captured: dict = {}
        monkeypatch.setitem(
            sys.modules, "sentence_transformers", _fake_sentence_transformers(captured)
        )
        server = ModelServer()

        # No override and no sticky degradation: framework picks (device=None).
        monkeypatch.delenv("TRUEMEMORY_DEVICE", raising=False)
        assert server._embed_device() is None
        server._build_embed_model("qwen3_256", server._embed_device())
        assert captured["st_device"] is None

        # Env override is honored at load time.
        monkeypatch.setenv("TRUEMEMORY_DEVICE", "cpu")
        assert server._embed_device() == "cpu"
        server._build_embed_model("qwen3_256", server._embed_device())
        assert captured["st_device"] == "cpu"

        # Sticky-CPU degradation wins even over the env var.
        monkeypatch.setenv("TRUEMEMORY_DEVICE", "mps")
        server._sticky_cpu.add("embed")
        assert server._embed_device() == "cpu"

    def test_issue_577_device_env_override_cpu_local_reranker(self, monkeypatch):
        """reranker.get_reranker local fallback honors TRUEMEMORY_DEVICE=cpu."""
        import sys
        from truememory import reranker

        captured: dict = {}
        monkeypatch.setitem(
            sys.modules, "sentence_transformers", _fake_sentence_transformers(captured)
        )
        monkeypatch.setenv("TRUEMEMORY_DEVICE", "cpu")
        monkeypatch.setenv("TRUEMEMORY_NO_MODEL_SERVER", "1")
        monkeypatch.setattr(reranker, "_model", None)
        monkeypatch.setattr(reranker, "_model_name", "")

        model = reranker.get_reranker(model_name="test-model")
        assert captured["cross_encoder_device"] == "cpu"
        assert model is not None

        # Explicit device= parameter takes precedence over the env var.
        monkeypatch.setattr(reranker, "_model", None)
        monkeypatch.setattr(reranker, "_model_name", "")
        monkeypatch.setenv("TRUEMEMORY_DEVICE", "cpu")
        reranker.get_reranker(model_name="other-model", device="mps")
        assert captured["cross_encoder_device"] == "mps"

    def test_issue_577_device_env_override_cpu_throttler(self, monkeypatch):
        """The throttler's device pick honors TRUEMEMORY_DEVICE=cpu."""
        from truememory.model_server import ModelServer
        from truememory.tier_switch import throttler as throttler_mod

        class FakeThrottler:
            def __init__(self, device="cpu"):
                self.device = device

        monkeypatch.setattr(throttler_mod, "DynamicThrottler", FakeThrottler)
        monkeypatch.setenv("TRUEMEMORY_DEVICE", "cpu")

        server = ModelServer()
        server._activate_throttler()
        assert server._throttler is not None
        assert server._throttler.device == "cpu"

        # Sticky-CPU embed degradation also forces the throttler to CPU.
        server2 = ModelServer()
        monkeypatch.delenv("TRUEMEMORY_DEVICE", raising=False)
        server2._sticky_cpu.add("embed")
        server2._activate_throttler()
        assert server2._throttler.device == "cpu"


# ---------------------------------------------------------------------------
# 2. Sticky-CPU OOM degradation (embed path)
# ---------------------------------------------------------------------------

class TestIssue577StickyCpuOom:
    def test_issue_577_oom_marks_sticky_cpu(self, no_flush, caplog):
        """MPS OOM moves the model to CPU permanently: no MPS re-promotion,
        loud log exactly once."""
        from truememory.model_server import ModelServer

        server = ModelServer()
        to_calls: list[str] = []
        encode_calls: list[int] = []

        def encode(texts, **kwargs):
            encode_calls.append(len(texts))
            # OOM on the 1st and 3rd encode attempts (3rd = 2nd request).
            if len(encode_calls) in (1, 3):
                raise _make_mps_oom_error()
            return np.zeros((len(texts), 4), dtype=np.float32)

        model = MagicMock()
        model.encode = encode
        model.to = lambda device: to_calls.append(device)
        from truememory.model_server import _EmbedState
        server._embed_state = _EmbedState(model=model, tier="", model_id="model2vec")

        req = {"op": "embed", "texts": ["a", "b"], "tier": ""}
        with caplog.at_level("ERROR", logger="truememory.model_server"):
            resp = server.handle_request(req)
        assert resp["ok"] is True
        assert "embed" in server._sticky_cpu
        assert to_calls == ["cpu"], "model must move to CPU and NEVER back to MPS"
        sticky_logs = [r for r in caplog.records if "CPU" in r.message]
        assert len(sticky_logs) == 1, "sticky degradation must log loudly once"

        # Second OOM: still recovers, never re-promotes, no second loud log.
        caplog.clear()
        with caplog.at_level("ERROR", logger="truememory.model_server"):
            resp2 = server.handle_request(req)
        assert resp2["ok"] is True
        assert "mps" not in to_calls, "no re-promotion to MPS after recovery"
        assert all(d == "cpu" for d in to_calls)
        assert not [r for r in caplog.records if r.levelname == "ERROR"], (
            "sticky degradation must only be logged on the first OOM"
        )

    def test_issue_577_oom_retry_not_under_lock(self, no_flush):
        """The CPU re-encode after an MPS OOM must not hold the global lock:
        another thread can acquire it while recovery is in progress."""
        from truememory.model_server import ModelServer

        server = ModelServer()
        retry_started = threading.Event()
        release_retry = threading.Event()
        calls: list[int] = []

        def encode(texts, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise _make_mps_oom_error()
            retry_started.set()
            assert release_retry.wait(10), "test deadlock: retry never released"
            return np.zeros((len(texts), 4), dtype=np.float32)

        model = MagicMock()
        model.encode = encode
        from truememory.model_server import _EmbedState
        server._embed_state = _EmbedState(model=model, tier="", model_id="model2vec")

        results: list[dict] = []
        t = threading.Thread(
            target=lambda: results.append(
                server.handle_request({"op": "embed", "texts": ["a", "b"], "tier": ""})
            ),
            daemon=True,
        )
        t.start()
        assert retry_started.wait(10), "OOM retry never started"

        acquired = server._lock.acquire(timeout=2)
        try:
            assert acquired, (
                "global request lock held during OOM recovery — every other "
                "client (hook recall, rerank) starves behind the re-encode"
            )
        finally:
            if acquired:
                server._lock.release()

        release_retry.set()
        t.join(10)
        assert results and results[0]["ok"] is True

    def test_issue_577_non_oom_error_still_propagates(self, no_flush):
        """Non-MPS RuntimeErrors must not be swallowed by the sticky handler."""
        from truememory.model_server import ModelServer

        server = ModelServer()
        model = MagicMock()
        model.encode = MagicMock(
            side_effect=RuntimeError("CUDA error: device-side assert triggered")
        )
        from truememory.model_server import _EmbedState
        server._embed_state = _EmbedState(model=model, tier="", model_id="model2vec")

        with pytest.raises(RuntimeError, match="CUDA"):
            server.handle_request({"op": "embed", "texts": ["a", "b"], "tier": ""})
        assert "embed" not in server._sticky_cpu


# ---------------------------------------------------------------------------
# 3. Rerank path OOM handling
# ---------------------------------------------------------------------------

class TestIssue577RerankOom:
    def test_issue_577_rerank_oom_handled(self, no_flush, monkeypatch):
        """MPS OOM in rerank must not reach the client: sticky-CPU reload
        produces a real score response (baseline: 3/3 rerank deaths)."""
        from truememory.model_server import ModelServer

        class FakeReranker:
            def __init__(self, fail):
                self.fail = fail

            def predict(self, pairs, **kwargs):
                if self.fail:
                    raise _make_mps_oom_error()
                return np.full(len(pairs), 0.5, dtype=np.float32)

        def fake_get_reranker(self, model_name=None):
            # Before sticky degradation the loaded model is on MPS and OOMs;
            # after, the reload is CPU-resident and succeeds.
            return FakeReranker(fail="rerank" not in self._sticky_cpu)

        monkeypatch.setattr(ModelServer, "_get_reranker", fake_get_reranker)
        server = ModelServer()

        resp = server.handle_request(
            {"op": "rerank", "pairs": [["q", "d1"], ["q", "d2"]], "model_name": "m"}
        )
        assert resp["ok"] is True, "rerank OOM must be recovered, not surfaced"
        assert len(resp["scores"]) == 2
        assert "rerank" in server._sticky_cpu

    def test_issue_577_rerank_sticky_reloads_on_cpu(self, monkeypatch):
        """After sticky degradation, _get_reranker loads with device=cpu even
        if TRUEMEMORY_DEVICE requests an accelerator."""
        import sys
        from truememory.model_server import ModelServer

        captured: dict = {}
        monkeypatch.setitem(
            sys.modules, "sentence_transformers", _fake_sentence_transformers(captured)
        )
        monkeypatch.setenv("TRUEMEMORY_DEVICE", "cpu")

        server = ModelServer()
        server._sticky_cpu.add("rerank")
        server._get_reranker("test-model")
        assert captured["cross_encoder_device"] == "cpu"

    def test_issue_577_rerank_non_oom_error_propagates(self, no_flush, monkeypatch):
        """Non-OOM rerank errors keep the existing propagate-to-client path."""
        from truememory.model_server import ModelServer

        broken = MagicMock()
        broken.predict = MagicMock(side_effect=RuntimeError("tokenizer exploded"))
        monkeypatch.setattr(
            ModelServer, "_get_reranker", lambda self, model_name=None: broken
        )
        server = ModelServer()
        with pytest.raises(RuntimeError, match="tokenizer"):
            server.handle_request({"op": "rerank", "pairs": [["q", "d"]], "model_name": "m"})
        assert "rerank" not in server._sticky_cpu


# ---------------------------------------------------------------------------
# 4. Single-text fast lane
# ---------------------------------------------------------------------------

class TestIssue577FastLane:
    def test_issue_577_single_text_fast_lane(self, monkeypatch):
        """A single-text encode (hook recall) completes while a batch holds
        the global lock, and never trips the sustained-workload throttler."""
        from truememory.model_server import ModelServer

        server = ModelServer()

        main_model = MagicMock()
        main_model.encode = MagicMock(
            side_effect=AssertionError(
                "fast lane must not use the locked main model while a batch runs"
            )
        )
        from truememory.model_server import _EmbedState
        server._embed_state = _EmbedState(model=main_model, tier="", model_id="model2vec")

        fast_calls: list[list[str]] = []

        class FakeFastEncoder:
            def encode(self, texts, **kwargs):
                fast_calls.append(list(texts))
                return np.zeros((len(texts), 4), dtype=np.float32)

        monkeypatch.setattr(
            ModelServer, "_get_fast_encoder", lambda self, tier: FakeFastEncoder()
        )

        # Simulate an in-flight ingestion batch occupying the heavy path.
        assert server._lock.acquire(timeout=1)
        try:
            result: dict = {}
            t = threading.Thread(
                target=lambda: result.update(
                    server.handle_request(
                        {"op": "embed", "texts": ["hook recall query"], "tier": ""}
                    )
                ),
                daemon=True,
            )
            t.start()
            t.join(5)
            assert not t.is_alive(), (
                "single-text request queued behind the batch lock — hook "
                "recall starves behind ingestion (issue #577)"
            )
            assert result.get("ok") is True
            assert fast_calls == [["hook recall query"]]
            assert server._embed_timestamps == [], (
                "fast-lane requests must not feed the sustained-workload "
                "throttler (C-7: hook bursts tripped batch=1 ramp)"
            )
        finally:
            server._lock.release()

    def test_issue_577_fast_lane_model_dim_parity(self, monkeypatch):
        """Panel rounds 1-2, items 3/4: the fast-lane encoder's
        (model_id, dim) must equal the main path's — including under
        sticky-CPU degradation and global tier drift — and must DECLINE
        (fall through) when no snapshot exists for the tier or the model is
        custom. Stubs at the loader level (_build_embed_model) so the real
        parity logic runs."""
        import truememory.vector_search as vs
        from truememory.model_server import ModelServer, _EmbedState

        built: list[tuple] = []

        class FakeModel:
            _DIMS = {"model2vec": 256, "qwen3_256": 256}

            def __init__(self, model_id, device):
                self.model_id = model_id
                self.device = device
                self.dim = self._DIMS.get(model_id, 999)

            def encode(self, texts, **kwargs):
                return np.zeros((len(texts), self.dim), dtype=np.float32)

        def fake_build(model_id, device):
            built.append((model_id, device))
            return FakeModel(model_id, device)

        monkeypatch.setattr(
            ModelServer, "_build_embed_model", staticmethod(fake_build)
        )

        server = ModelServer()
        server._sticky_cpu.add("embed")  # simulate sticky-CPU degradation
        monkeypatch.setattr(vs, "EMBEDDING_MODEL", "base")

        # No snapshot yet: the fast lane must DECLINE (round 2 semantics —
        # the one-time load happens exactly once, on the main path).
        assert server._lock.acquire(timeout=1)
        try:
            assert server._handle_fast_embed(["q"], "") is None, (
                "fast lane must decline when no main model is loaded yet"
            )
        finally:
            server._lock.release()

        # Main path loads for tier="" -> resolves via the global ("base"
        # -> qwen3_256) and records the loaded identity in the snapshot.
        resp = server.handle_request({"op": "embed", "texts": ["a", "b"], "tier": ""})
        assert resp["ok"] is True
        assert server._embed_state.model_id == "qwen3_256"
        assert built[-1] == ("qwen3_256", "cpu"), "sticky degradation must load on CPU"

        # Global drift AFTER the main load: a fresh tier="" resolution would
        # now pick model2vec — a DIFFERENT vector space. The fast lane must
        # follow the snapshot's actual loaded identity instead.
        monkeypatch.setattr(vs, "EMBEDDING_MODEL", "edge")

        assert server._lock.acquire(timeout=1)  # force the contended branch
        try:
            result = server._handle_fast_embed(["hook query"], "")
        finally:
            server._lock.release()
        assert result is not None and result["ok"] is True
        assert server._fast_model_id == server._embed_state.model_id == "qwen3_256", (
            "fast-lane encoder drifted from the main path's loaded model — "
            "single-text embeddings would silently mismatch the vec table"
        )
        assert built[-1] == ("qwen3_256", "cpu")

        main_dim = server._embed_state.model.encode(["x"]).shape[1]
        fast_dim = server._fast_encoder.encode(["x"]).shape[1]
        assert main_dim == fast_dim, "fast-lane embeddings in a different space"

        # Custom model: dim is re-read from config.json at build time, so
        # parity cannot be guaranteed -> the fast lane must decline and the
        # request must fall through to the main (locked) path.
        server._embed_state = _EmbedState(
            model=FakeModel("acme/custom-encoder", "cpu"),
            tier="custom",
            model_id="acme/custom-encoder",
        )
        assert server._lock.acquire(timeout=1)
        try:
            assert server._handle_fast_embed(["q"], "custom") is None, (
                "fast lane must decline custom models instead of risking a "
                "vector-space mismatch"
            )
        finally:
            server._lock.release()

        # Allowlist honesty (round 2 item 5): ids without an explicit
        # _build_embed_model branch (minilm/bge-small legacy fall-through)
        # must also decline.
        server._embed_state = _EmbedState(
            model=FakeModel("minilm", "cpu"), tier="", model_id="minilm",
        )
        assert server._lock.acquire(timeout=1)
        try:
            assert server._handle_fast_embed(["q"], "") is None, (
                "minilm has no explicit builder branch — fast lane must decline"
            )
        finally:
            server._lock.release()

    def test_issue_577_fast_lane_no_global_mutation(self, monkeypatch):
        """Panel rounds 1-2, item 2: the fast lane never resolves tiers at
        all — it must never call set_embedding_model, which force-unloads
        vector_search's model singleton and rewrites
        EMBEDDING_MODEL/_embedding_dim while the main path may be
        mid-encode."""
        import truememory.vector_search as vs
        from truememory.model_server import ModelServer

        def fake_build(model_id, device):
            model = MagicMock()
            model.encode = lambda texts, **kw: np.zeros(
                (len(texts), 4), dtype=np.float32
            )
            return model

        monkeypatch.setattr(
            ModelServer, "_build_embed_model", staticmethod(fake_build)
        )

        calls: list[str] = []
        monkeypatch.setattr(vs, "set_embedding_model", lambda name: calls.append(name))
        monkeypatch.setattr(vs, "EMBEDDING_MODEL", "edge")

        server = ModelServer()
        # Arm the snapshot via a main-path load (the historical mutating
        # sync happens exactly here, under the lock).
        resp = server.handle_request({"op": "embed", "texts": ["a", "b"], "tier": "base"})
        assert resp["ok"] is True
        assert calls == ["base"]
        calls.clear()

        # Contended fast lane: serves from the snapshot, zero mutation.
        assert server._lock.acquire(timeout=1)
        try:
            result = server._handle_fast_embed(["q"], "base")
        finally:
            server._lock.release()
        assert result is not None and result["ok"] is True
        assert calls == [], (
            "fast lane mutated vector_search globals (set_embedding_model) "
            "outside the request lock"
        )
        assert vs.EMBEDDING_MODEL == "edge"

    def test_issue_577_fast_lane_concurrent_tier_load_parity(self, monkeypatch):
        """Panel round 2, item 6: interleave a main-path tier load (blocked
        mid-build, holding the global lock) with a fast-lane single-text in
        another thread, plus a mid-load global drift. The single-text result
        must be parity (same model id as the main load) or an explicit
        decline that falls through to the main path — never a fresh
        resolution into a different vector space. Loader-level stub;
        3 threads total."""
        import truememory.vector_search as vs
        from truememory.model_server import ModelServer

        built: list[tuple] = []
        build_started = threading.Event()
        release_build = threading.Event()

        class FakeModel:
            def __init__(self, model_id):
                self.model_id = model_id
                self.marker = float(abs(hash(model_id)) % 997)

            def encode(self, texts, **kwargs):
                return np.full((len(texts), 8), self.marker, dtype=np.float32)

        def fake_build(model_id, device):
            built.append((model_id, device, threading.current_thread().name))
            if threading.current_thread().name == "main-loader":
                build_started.set()
                assert release_build.wait(10), "test deadlock: build never released"
            return FakeModel(model_id)

        monkeypatch.setattr(
            ModelServer, "_build_embed_model", staticmethod(fake_build)
        )
        monkeypatch.setattr(vs, "EMBEDDING_MODEL", "base")

        server = ModelServer()
        results_a: list[dict] = []
        results_b: list[dict] = []

        ta = threading.Thread(
            name="main-loader",
            target=lambda: results_a.append(
                server.handle_request({"op": "embed", "texts": ["a", "b"], "tier": ""})
            ),
            daemon=True,
        )
        ta.start()
        assert build_started.wait(10)  # A is mid-load, holding the lock

        # Global drift WHILE the load is in flight: a fresh tier=""
        # resolution would now pick model2vec instead of qwen3_256.
        monkeypatch.setattr(vs, "EMBEDDING_MODEL", "edge")

        tb = threading.Thread(
            name="fast-lane",
            target=lambda: results_b.append(
                server.handle_request({"op": "embed", "texts": ["hook q"], "tier": ""})
            ),
            daemon=True,
        )
        tb.start()
        tb.join(2)
        # Correct behavior: B declined (no snapshot yet) and is blocked
        # behind the main load — it must NOT have answered from a freshly
        # resolved (drifted) model.
        release_build.set()
        ta.join(10)
        tb.join(10)
        assert not ta.is_alive() and not tb.is_alive()
        assert results_a and results_a[0]["ok"] is True
        assert results_b and results_b[0]["ok"] is True

        # Every encoder built during the interleaving resolved to the SAME
        # model id — no vector-space fork.
        ids = {m for (m, _d, _t) in built}
        assert ids == {"qwen3_256"}, (
            f"vector-space fork during concurrent tier load: built {ids}"
        )
        assert results_a[0]["vectors"][0, 0] == results_b[0]["vectors"][0, 0], (
            "single-text vectors in a different space than the batch vectors"
        )

        # Phase 2: with the snapshot armed and the lock held by a batch,
        # the contended fast lane serves from the snapshot identity.
        results_c: list[dict] = []
        assert server._lock.acquire(timeout=1)
        try:
            tc = threading.Thread(
                name="fast-lane-2",
                target=lambda: results_c.append(
                    server.handle_request({"op": "embed", "texts": ["q2"], "tier": ""})
                ),
                daemon=True,
            )
            tc.start()
            tc.join(5)
            assert not tc.is_alive(), "fast lane blocked despite armed snapshot"
        finally:
            server._lock.release()
        assert results_c and results_c[0]["ok"] is True
        assert results_c[0]["vectors"][0, 0] == results_a[0]["vectors"][0, 0]
        assert server._fast_model_id == "qwen3_256"

    def test_issue_577_single_text_uses_main_path_when_idle(self, monkeypatch):
        """With no contention the fast lane uses the main model (no extra
        CPU encoder is loaded)."""
        from truememory.model_server import ModelServer

        server = ModelServer()
        main_model = MagicMock()
        main_model.encode = MagicMock(
            return_value=np.ones((1, 4), dtype=np.float32)
        )
        from truememory.model_server import _EmbedState
        server._embed_state = _EmbedState(model=main_model, tier="", model_id="model2vec")

        resp = server.handle_request({"op": "embed", "texts": ["q"], "tier": ""})
        assert resp["ok"] is True
        assert main_model.encode.called
        assert server._fast_encoder is None, (
            "idle single-text requests must not pay for a duplicate CPU encoder"
        )


# ---------------------------------------------------------------------------
# 5. Client deadline kwarg + hook wiring
# ---------------------------------------------------------------------------

class TestIssue577ClientDeadline:
    def test_issue_577_client_deadline_kwarg(self, monkeypatch):
        """encode/predict accept timeout=; expiry fast-fails with TimeoutError
        and never falls into the autostart + full-timeout retry cycle."""
        from truememory import model_client

        send_calls: list[float | None] = []

        def fake_send(request, timeout=None):
            send_calls.append(timeout)
            raise socket.timeout("timed out")

        def forbidden_start(wait_timeout=None):
            raise AssertionError("autostart must not run after deadline expiry")

        monkeypatch.setattr(model_client, "_send_request", fake_send)
        monkeypatch.setattr(model_client, "_start_server", forbidden_start)

        with pytest.raises(TimeoutError, match="deadline"):
            model_client.EmbeddingProxy().encode(["q"], timeout=0.05)
        assert send_calls == [0.05], "per-call timeout must reach the socket layer"

        with pytest.raises(TimeoutError, match="deadline"):
            model_client.RerankerProxy().predict([("q", "d")], timeout=0.05)

    def test_issue_577_client_default_timeout_unchanged(self, monkeypatch):
        """timeout=None keeps the legacy 120s default + autostart retry."""
        from truememory import model_client

        assert model_client._REQUEST_TIMEOUT == 120.0

        attempts: list[int] = []

        def flaky_send(request, timeout=None):
            attempts.append(1)
            if len(attempts) == 1:
                raise socket.timeout("timed out")
            return {"ok": True, "vectors": np.zeros((1, 4), dtype=np.float32)}

        monkeypatch.setattr(model_client, "_send_request", flaky_send)
        monkeypatch.setattr(model_client, "_start_server", lambda wait_timeout=None: True)

        out = model_client.EmbeddingProxy().encode(["q"])
        assert out.shape == (1, 4)
        assert len(attempts) == 2, "legacy path must still autostart + retry"

    def test_issue_577_process_default_deadline(self, monkeypatch):
        """set_request_timeout() gives hook processes a short deadline for
        every embed/rerank without threading kwargs through engine.search."""
        from truememory import model_client

        def fake_send(request, timeout=None):
            raise socket.timeout("timed out")

        monkeypatch.setattr(model_client, "_send_request", fake_send)
        monkeypatch.setattr(
            model_client,
            "_start_server",
            lambda wait_timeout=None: pytest.fail("autostart after deadline expiry"),
        )

        model_client.set_request_timeout(0.05)
        try:
            with pytest.raises(TimeoutError, match="deadline"):
                model_client.EmbeddingProxy().encode(["q"])
        finally:
            model_client.set_request_timeout(None)

    def test_issue_577_hooks_pass_short_deadline(self, monkeypatch):
        """Hook recall paths arm the ~5s deadline so FTS-only fallback can
        actually trigger (C-1: worst case was 5 x 120s)."""
        import inspect

        from truememory.ingest.hooks import _shared, session_start, user_prompt_submit

        assert _shared.get_recall_deadline() == 5.0
        monkeypatch.setenv("TRUEMEMORY_HOOK_RECALL_TIMEOUT", "2.5")
        assert _shared.get_recall_deadline() == 2.5
        monkeypatch.setenv("TRUEMEMORY_HOOK_RECALL_TIMEOUT", "0")
        assert _shared.get_recall_deadline() is None  # 0/negative disables
        monkeypatch.setenv("TRUEMEMORY_HOOK_RECALL_TIMEOUT", "banana")
        assert _shared.get_recall_deadline() == 5.0

        recall_src = inspect.getsource(session_start.recall_memories)
        assert "set_request_timeout" in recall_src, (
            "session_start recall must arm the short model-server deadline"
        )
        ups_src = inspect.getsource(user_prompt_submit._try_auto_recall)
        assert "set_request_timeout" in ups_src, (
            "user_prompt_submit auto-recall must arm the short deadline"
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
