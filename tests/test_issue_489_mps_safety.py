"""Regression tests for MPS safety group (H3 #489, H4 #490, H8 #494, M21 #518).

H3: hybrid.py unprotected model.encode() crashes on MPS OOM
H4: model_server.py never restores model to MPS after CPU fallback
H8: Concurrent device transitions cause race conditions
M21: Inconsistent OOM detection patterns across modules
"""

import threading
import unittest
from unittest.mock import MagicMock, patch


class TestMpsUtils(unittest.TestCase):
    """Tests for the shared mps_utils module (M21, H8)."""

    def test_is_mps_oom_detects_standard_message(self):
        from truememory.mps_utils import is_mps_oom
        exc = RuntimeError("MPS backend out of memory")
        self.assertTrue(is_mps_oom(exc))

    def test_is_mps_oom_detects_allocated_exceed(self):
        from truememory.mps_utils import is_mps_oom
        exc = RuntimeError("MPS: allocated 2.5GB, would exceed 3.0GB limit")
        self.assertTrue(is_mps_oom(exc))

    def test_is_mps_oom_rejects_unrelated_error(self):
        from truememory.mps_utils import is_mps_oom
        exc = RuntimeError("CUDA device not found")
        self.assertFalse(is_mps_oom(exc))

    def test_is_mps_oom_case_insensitive(self):
        from truememory.mps_utils import is_mps_oom
        exc = RuntimeError("mps BACKEND Out Of Memory")
        self.assertTrue(is_mps_oom(exc))

    def test_encode_with_mps_fallback_normal_path(self):
        from truememory.mps_utils import encode_with_mps_fallback
        model = MagicMock()
        model.encode.return_value = [[0.1, 0.2]]
        result = encode_with_mps_fallback(model, ["hello"])
        self.assertEqual(result, [[0.1, 0.2]])
        model.encode.assert_called_once_with(["hello"])

    def test_encode_with_mps_fallback_oom_falls_to_cpu_and_restores(self):
        """H4+H8: On OOM, should move to CPU, encode, then restore to MPS."""
        from truememory.mps_utils import encode_with_mps_fallback
        model = MagicMock()
        model.encode.side_effect = [
            RuntimeError("MPS backend out of memory"),
            [[0.1, 0.2]],
        ]
        with patch("truememory.mps_utils.flush_mps_cache"):
            with patch("torch.backends.mps.is_available", return_value=True):
                result = encode_with_mps_fallback(model, ["hello"])
        self.assertEqual(result, [[0.1, 0.2]])
        to_calls = [c.args[0] for c in model.to.call_args_list]
        self.assertIn("cpu", to_calls)
        self.assertIn("mps", to_calls)
        cpu_idx = to_calls.index("cpu")
        mps_idx = to_calls.index("mps")
        self.assertLess(cpu_idx, mps_idx)

    def test_encode_with_mps_fallback_reraises_non_oom(self):
        from truememory.mps_utils import encode_with_mps_fallback
        model = MagicMock()
        model.encode.side_effect = RuntimeError("something unrelated")
        with self.assertRaises(RuntimeError):
            encode_with_mps_fallback(model, ["hello"])

    def test_device_lock_serializes_concurrent_transitions(self):
        """H8: Concurrent calls should not interleave device transitions."""
        from truememory.mps_utils import _device_lock
        results = []

        def acquire_lock(label):
            with _device_lock:
                results.append(f"{label}-start")
                results.append(f"{label}-end")

        t1 = threading.Thread(target=acquire_lock, args=("a",))
        t2 = threading.Thread(target=acquire_lock, args=("b",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        # Each start-end pair must be adjacent (no interleaving)
        for i in range(0, len(results), 2):
            label = results[i].split("-")[0]
            self.assertEqual(results[i], f"{label}-start")
            self.assertEqual(results[i + 1], f"{label}-end")


class TestHybridMpsSafety(unittest.TestCase):
    """H3: hybrid.py must use MPS-safe encoding."""

    def test_hybrid_uses_mps_fallback_import(self):
        """Verify hybrid.py imports encode_with_mps_fallback."""
        import inspect
        from truememory import hybrid
        source = inspect.getsource(hybrid)
        self.assertIn("encode_with_mps_fallback", source)
        self.assertNotIn("_q_model.encode([query])[0]", source)


class TestModelServerMpsRestore(unittest.TestCase):
    """H4: model_server must restore model to MPS after CPU fallback."""

    def test_model_server_uses_shared_is_mps_oom(self):
        """M21: model_server should use shared is_mps_oom, not inline string matching."""
        import inspect
        from truememory import model_server
        source = inspect.getsource(model_server)
        self.assertIn("from truememory.mps_utils import is_mps_oom", source)

    def test_model_server_sticky_cpu_no_mps_restore(self):
        """H4 (superseded by #577): after CPU fallback the model must STAY
        on CPU for the server's lifetime.

        The original H4 fix restored the model to MPS after recovery, but
        the MPS pool never drops below the watermark cap after an OOM, so
        re-promotion guaranteed the next OOM (measured retry storms stalling
        every client 23-112s — issue #577). The policy is now sticky-CPU
        degradation: model.to("cpu") with NO model.to("mps") re-promotion.
        """
        import inspect
        from truememory import model_server
        handler_source = inspect.getsource(model_server.ModelServer.handle_request)
        cpu_idx = handler_source.find('model.to("cpu")')
        mps_idx = handler_source.find('model.to("mps")')
        self.assertGreater(cpu_idx, -1, "Should have model.to('cpu')")
        self.assertEqual(
            mps_idx, -1,
            "model.to('mps') re-promotion must NOT exist — sticky-CPU "
            "degradation (issue #577) replaces the H4 restore behavior",
        )
        self.assertIn(
            "_mark_sticky_cpu", handler_source,
            "OOM recovery must mark the model sticky-CPU (issue #577)",
        )


if __name__ == "__main__":
    unittest.main()
