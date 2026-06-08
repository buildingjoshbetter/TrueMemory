"""Regression tests for model server group (M17 #514, M18 #515, M19 #516, M20 #517).

M17: dtype whitelist in JSON deserialization
M18: Outbound response size check
M19: PermissionError in PID check
M20: flock-based rebuild lock
"""

import inspect
import unittest


class TestM17DtypeWhitelist(unittest.TestCase):
    """M17: Only allowed dtypes should be accepted."""

    def test_rejects_disallowed_dtype(self):
        from truememory.model_server import _json_object_hook
        import base64
        obj = {
            "__ndarray__": base64.b64encode(b"\x00" * 4).decode(),
            "dtype": "object",
            "shape": [1],
        }
        with self.assertRaises(ValueError):
            _json_object_hook(obj)

    def test_accepts_float32(self):
        from truememory.model_server import _json_object_hook
        import base64, struct
        data = struct.pack("f", 1.0)
        obj = {
            "__ndarray__": base64.b64encode(data).decode(),
            "dtype": "float32",
            "shape": [1],
        }
        result = _json_object_hook(obj)
        self.assertAlmostEqual(float(result[0]), 1.0)

    def test_whitelist_exists(self):
        from truememory.model_server import _ALLOWED_DTYPES
        self.assertIn("float32", _ALLOWED_DTYPES)
        self.assertNotIn("object", _ALLOWED_DTYPES)


class TestM18OutboundSizeCheck(unittest.TestCase):
    """M18: Response size should be checked."""

    def test_send_response_has_size_check(self):
        from truememory.model_server import ModelServer
        source = inspect.getsource(ModelServer._send_response)
        self.assertIn("_MAX_MESSAGE_SIZE", source,
                       "_send_response should check against _MAX_MESSAGE_SIZE")


class TestM19PermissionError(unittest.TestCase):
    """M19: PID check should handle PermissionError (via pid_is_alive)."""

    def test_pid_check_handles_permission_error(self):
        from truememory._platform import pid_is_alive
        source = inspect.getsource(pid_is_alive)
        self.assertIn("PermissionError", source,
                       "pid_is_alive() should catch PermissionError")

    def test_main_uses_pid_is_alive(self):
        from truememory.model_server import main
        source = inspect.getsource(main)
        self.assertIn("pid_is_alive", source,
                       "main() should use pid_is_alive for cross-platform PID check")


class TestM20RebuildLock(unittest.TestCase):
    """M20: Rebuild should use flock-based locking."""

    def test_rebuild_uses_flock(self):
        from truememory.tier_switch.manager import RebuildManager
        source = inspect.getsource(RebuildManager.run_rebuild_sync)
        self.assertIn("fcntl.flock", source,
                       "run_rebuild_sync should use fcntl.flock")

    def test_flock_imported(self):
        import truememory.tier_switch.manager as mgr
        source = inspect.getsource(mgr)
        self.assertIn("import fcntl", source)


if __name__ == "__main__":
    unittest.main()
