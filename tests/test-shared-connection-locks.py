"""Statements on a connection shared between threads run under that connection's lock.

sqlite3 shares cached statements per connection: two threads stepping one
corrupt each other (IndexError, "another row available"), and on SQLite 3.53
a statement waiting for another thread's step freezes the whole process.
"""

from __future__ import annotations

import ast
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class Guarded:
    """A connection whose statements must run while ``held()`` is true."""

    def __init__(self, conn, held):
        self._conn, self._held, self.statements = conn, held, 0

    def execute(self, *args, **kwargs):
        if not self._held():
            raise AssertionError("statement outside the connection lock")
        self.statements += 1
        return self._conn.execute(*args, **kwargs)

    def commit(self):
        if not self._held():
            raise AssertionError("commit outside the connection lock")
        return self._conn.commit()

    def __getattr__(self, name):
        return getattr(self._conn, name)


class SessionStartDirectives(unittest.TestCase):
    def test_directive_queries_hold_the_engine_lock(self):
        # Review v5 L-v5-4: the session-start hook read directives on the
        # engine connection after a search may have started the backfill thread.
        import tempfile
        from truememory import Memory
        from truememory.ingest.hooks.session_start import _count_directives, _load_directives
        with tempfile.TemporaryDirectory() as directory:
            memory = Memory(path=str(Path(directory) / "directives.db"))
            try:
                memory.add("always answer in plain words", directive=True)
                engine = memory._engine
                real = engine.conn
                engine.conn = guarded = Guarded(real, engine._write_lock.owned_by_current_thread)
                try:
                    with patch.object(engine, "_ensure_connection", lambda: None):
                        counted = _count_directives(memory)
                        loaded = [item["content"] for item in _load_directives(memory)]
                finally:
                    engine.conn = real
            finally:
                memory.close()
        self.assertEqual((counted, loaded, guarded.statements), (1, ["always answer in plain words"], 2))


class EngineConnectionOutsideTheEngine(unittest.TestCase):
    def test_no_module_closes_the_engine_connection_directly(self):
        # The MCP shutdown closed _engine.conn without the engine's locks while
        # a backfill pass could still be finishing; Engine.close takes them.
        found = []
        for path in sorted((ROOT / "truememory").rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            for node in ast.walk(ast.parse(source)):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "close"
                        and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "conn"
                        and "_engine" in ast.get_source_segment(source, node.func.value)):
                    found.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual(found, [])


class InstrumentationWriter(unittest.TestCase):
    def test_telemetry_statements_hold_the_writer_lock(self):
        import os
        import tempfile
        from truememory.instrumentation import writer
        lock = threading.RLock()
        holders = []

        class Tracking:
            def __enter__(self):
                lock.acquire()
                holders.append(threading.get_ident())

            def __exit__(self, *exc):
                holders.pop()
                lock.release()

        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"TRUEMEMORY_INSTRUMENTATION_DB": str(Path(directory) / "telemetry.db")}):
            writer.reset()
            try:
                real = writer._get_connection()
                guarded = Guarded(real, lambda: threading.get_ident() in holders)
                with patch.object(writer, "_statement_lock", Tracking(), create=True), \
                        patch.object(writer, "_conn", guarded), patch.object(writer, "_PRUNE_EVERY", 1):
                    writer.emit("synthetic.signal", {"n": 1})
                    writer.prune_now()
                count = real.execute("SELECT count(*) FROM telemetry").fetchone()[0]
            finally:
                writer.reset()
        self.assertEqual(count, 1)
        self.assertEqual(guarded.statements, 3)  # insert, emit's prune, prune_now


if __name__ == "__main__":
    unittest.main()
