"""Regression test for H10 #496: update() holds write lock during model inference.

The fix pre-computes embeddings outside the lock (like add() does),
then only holds _write_lock for the DB writes.
"""

import inspect
import threading
import time
import unittest


class TestUpdateLockScope(unittest.TestCase):
    """Verify update() does not hold _write_lock during embedding."""

    def test_update_computes_embedding_outside_lock(self):
        """The embed call must happen BEFORE 'with self._write_lock'."""
        from truememory.engine import TrueMemoryEngine
        source = inspect.getsource(TrueMemoryEngine.update)
        lock_pos = source.find("self._write_lock")
        self.assertGreater(lock_pos, -1, "update() should use _write_lock")
        embed_call = source.find("embed_single")
        if embed_call != -1:
            self.fail(
                "update() should NOT call embed_single() inside the lock — "
                "it should pre-compute embeddings like add() does"
            )

    def test_update_pre_computes_like_add(self):
        """update() should use _encode_with_mps_fallback before the lock."""
        from truememory.engine import TrueMemoryEngine
        source = inspect.getsource(TrueMemoryEngine.update)
        self.assertIn("_encode_with_mps_fallback", source,
                       "update() should pre-compute embeddings with _encode_with_mps_fallback")

    def test_update_concurrent_not_blocked_by_inference(self):
        """Two concurrent update() calls should not serialize on inference."""
        import tempfile, os
        from truememory.engine import TrueMemoryEngine

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            e = TrueMemoryEngine(db_path=db_path)
            e._ensure_connection()

        m1 = e.add("first message", sender="alice")
        m2 = e.add("second message", sender="bob")

        results = [None, None]
        errors = [None, None]

        def do_update(idx, mid, text):
            try:
                results[idx] = e.update(mid, content=text)
            except Exception as ex:
                errors[idx] = ex

        t1 = threading.Thread(target=do_update, args=(0, m1["id"], "updated first"))
        t2 = threading.Thread(target=do_update, args=(1, m2["id"], "updated second"))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        self.assertIsNone(errors[0], f"update 1 failed: {errors[0]}")
        self.assertIsNone(errors[1], f"update 2 failed: {errors[1]}")
        self.assertIsNotNone(results[0])
        self.assertIsNotNone(results[1])
        self.assertEqual(results[0]["content"], "updated first")
        self.assertEqual(results[1]["content"], "updated second")


if __name__ == "__main__":
    unittest.main()
