"""SQLite WAL-reset guard: autocheckpoint off on affected builds, checkpoints only under the write lock."""

from __future__ import annotations

import ast
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from truememory import storage

ROOT = Path(__file__).resolve().parents[1]


def _child(source: str, *args: str, timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(source), *args],
                          capture_output=True, text=True, timeout=timeout)


class VersionRule(unittest.TestCase):
    def test_affected_versions_follow_the_fixed_releases_and_backports(self):
        cases = {(3, 7, 0): True, (3, 44, 5): True, (3, 44, 6): False, (3, 45, 0): True, (3, 46, 1): True,
                 (3, 47, 1): True, (3, 50, 4): True, (3, 50, 7): False, (3, 51, 0): True, (3, 51, 2): True,
                 (3, 51, 3): False, (3, 53, 3): False}
        self.assertEqual({version: storage.wal_reset_affected(version) for version in cases}, cases)


class WalLocks(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = str(Path(directory.name) / "guard.db")
        conn = sqlite3.connect(self.path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t(v BLOB)")
        conn.close()

    def test_write_lock_holder_keeps_other_processes_from_committing(self):
        # The guard's premise: while BEGIN IMMEDIATE is held, no writer anywhere
        # can append to (or restart) the WAL; a commit waits or fails busy.
        holder = sqlite3.connect(self.path, isolation_level=None)
        holder.execute("BEGIN IMMEDIATE")
        try:
            refused = _child("""
                import sqlite3, sys
                conn = sqlite3.connect(sys.argv[1], timeout=0.3, isolation_level=None)
                try:
                    conn.execute("BEGIN IMMEDIATE"); conn.execute("INSERT INTO t VALUES (x'00')"); conn.execute("COMMIT")
                    print("committed")
                except sqlite3.OperationalError as error:
                    print(error)
            """, self.path)
            self.assertEqual(refused.stdout.strip(), "database is locked")
            waiting = subprocess.Popen([sys.executable, "-c", textwrap.dedent("""
                import sqlite3, sys, time
                conn = sqlite3.connect(sys.argv[1], timeout=30, isolation_level=None)
                conn.execute("INSERT INTO t VALUES (x'01')")
                print(time.time(), flush=True)
            """), self.path], stdout=subprocess.PIPE, text=True)
            time.sleep(1.0)
            self.assertIsNone(waiting.poll())  # Still waiting for the lock.
            released = time.time()
        finally:
            holder.execute("ROLLBACK")
            holder.close()
        committed = float(waiting.communicate(timeout=30)[0])
        self.assertGreaterEqual(committed, released)
        conn = sqlite3.connect(self.path)
        self.assertEqual(conn.execute("SELECT count(*) FROM t").fetchone()[0], 1)
        conn.close()

    def test_close_checkpoints_only_as_the_last_connection(self):
        # SQLite checkpoints on close only after taking EXCLUSIVE on the
        # database file; any other open connection (here another process)
        # prevents it, so a close-time checkpoint never meets a writer.
        writer = sqlite3.connect(self.path, isolation_level=None)
        writer.execute("PRAGMA wal_autocheckpoint=0")
        other = subprocess.Popen([sys.executable, "-c", textwrap.dedent("""
            import sqlite3, sys
            conn = sqlite3.connect(sys.argv[1]); conn.execute("SELECT count(*) FROM t").fetchone()
            print("open", flush=True); sys.stdin.readline(); conn.close()
        """), self.path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.assertEqual(other.stdout.readline().strip(), "open")
        writer.execute("INSERT INTO t VALUES (zeroblob(100000))")
        writer.close()
        wal = Path(self.path + "-wal")
        self.assertTrue(wal.exists())
        self.assertGreater(wal.stat().st_size, 100000)  # Not checkpointed or removed.
        other.communicate("close\n", timeout=30)
        self.assertEqual(other.returncode, 0)
        last = sqlite3.connect(self.path)
        last.execute("SELECT count(*) FROM t").fetchone()
        last.close()
        self.assertFalse(wal.exists())  # The last close checkpointed and removed it.
        raw = sqlite3.connect(f"file:{self.path}?mode=ro&immutable=1", uri=True)
        self.assertEqual(raw.execute("SELECT length(v) FROM t").fetchall(), [(100000,)])
        raw.close()

    def test_protected_checkpoint_skips_a_busy_lock_and_never_keeps_it(self):
        conn = storage.configure_connection(sqlite3.connect(self.path, isolation_level=None))
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("INSERT INTO t VALUES (zeroblob(200000))")
        before = storage.wal_checkpoint_stats()
        busy = sqlite3.connect(self.path, isolation_level=None)
        busy.execute("BEGIN IMMEDIATE")
        try:
            self.assertFalse(storage.protected_checkpoint(self.path, hold_wait=0.05))
        finally:
            busy.execute("ROLLBACK")
            busy.close()
        self.assertEqual(storage.wal_checkpoint_stats()["skipped_busy"], before["skipped_busy"] + 1)
        raw = sqlite3.connect(f"file:{self.path}?mode=ro&immutable=1", uri=True)  # The database file alone.
        self.assertEqual(raw.execute("SELECT count(*) FROM t").fetchone()[0], 0)
        raw.close()
        self.assertTrue(storage.protected_checkpoint(self.path))
        raw = sqlite3.connect(f"file:{self.path}?mode=ro&immutable=1", uri=True)
        self.assertEqual(raw.execute("SELECT count(*) FROM t").fetchone()[0], 1)
        raw.close()
        # The write lock is free again at once.
        probe = sqlite3.connect(self.path, timeout=0, isolation_level=None)
        probe.execute("BEGIN IMMEDIATE")
        probe.execute("INSERT INTO t VALUES (x'02')")
        probe.execute("COMMIT")
        probe.close()
        conn.close()
        self.assertEqual(storage.wal_checkpoint_stats()["checkpoints"], before["checkpoints"] + 1)


class Openers(unittest.TestCase):
    ALLOWED = {
        # Read-only or never-writing connections cannot autocheckpoint.
        ("truememory/instrumentation/reader.py", 'f"file:{db_path}?mode=ro", uri=True'),
        ("truememory/tier_switch/manager.py", 'path.absolute().as_uri() + "?mode=ro", uri=True'),
        ("truememory/tier_switch/runtime.py", 'path.resolve().as_uri() + "?mode=ro", uri=True'),
        ("truememory/tier_switch/manager.py", '":memory:"'),
        ("truememory/ingest/cli.py", "str(path)"),  # PRAGMA quick_check only.
        ("truememory/storage.py", "str(backup)"),  # The new backup file; its only connection.
        ("truememory/storage.py", "uri, uri=True"),  # The guard's own lock holder and checkpointer.
        ("truememory/tier_switch/manager.py", "str(dest)"),  # Likewise for backup_db.
    }

    def test_every_database_opener_is_configured_by_the_guard(self):
        # Each sqlite3.connect is wrapped as configure(sqlite3.connect(...)) unless it never writes.
        unwrapped = set()
        for path in sorted((ROOT / "truememory").rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "connect"
                        and isinstance(node.func.value, ast.Name) and node.func.value.id == "sqlite3"):
                    continue
                parent = parents.get(node)
                while isinstance(parent, ast.IfExp):
                    parent = parents.get(parent)
                if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id == "configure_connection":
                    continue
                unwrapped.add((relative, ", ".join(ast.get_source_segment(source, arg) for arg in node.args)
                               + "".join(f", {k.arg}={ast.get_source_segment(source, k.value)}" for k in node.keywords
                                         if k.arg == "uri")))
        self.assertEqual(unwrapped, self.ALLOWED)

    def test_openers_disable_autocheckpoint_only_on_affected_builds(self):
        from truememory.storage import create_db
        from truememory.tier_switch.manager import _open_db
        from truememory.tier_switch.runtime import open_serving_connection
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "openers.db"
            create_db(path).close()
            for affected, expected in ((True, 0), (False, 1000)):
                with self.subTest(affected=affected), patch.object(storage, "WAL_RESET_GUARD", affected):
                    for name, conn in (("configure", storage.configure_connection(sqlite3.connect(path))), ("create_db", create_db(path)),
                                       ("tier _open_db", _open_db(path)),
                                       ("serving", open_serving_connection(path, create_db))):
                        try:
                            self.assertEqual(conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0], expected, name)
                        finally:
                            conn.close()



class DatabaseFileAccess(unittest.TestCase):
    # Opening and closing a database file outside SQLite drops every POSIX lock
    # the process holds on it; on SQLite 3.47/3.50 another process's close then
    # deleted the WAL in use (GPUBox D1/D2). Copies also miss WAL content.
    # Copies of files that are not databases (MEMORY.md before its migration).
    ALLOWED_COPIES = {("truememory/ingest/migrate_memory_md.py", "memory_md_path")}
    COPIES = {"copy", "copy2", "copyfile", "copytree", "move"}
    OPENS = {"open", "read_bytes", "read_text", "write_bytes", "write_text"}
    DATABASE_NAME = re.compile(r"\b_?(db_path|database_path|DB_PATH|db_file)\b|memories\.db|-wal\b|-shm\b")

    def test_no_module_copies_moves_or_opens_database_files(self):
        # Every module, every import style (import shutil as x, from shutil
        # import copy2 as y), and direct opens whose target names a database.
        found = []
        for path in sorted((ROOT / "truememory").rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            modules, functions = set(), set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.update(alias.asname or alias.name for alias in node.names if alias.name == "shutil")
                elif isinstance(node, ast.ImportFrom) and node.module == "shutil":
                    functions.update(alias.asname or alias.name for alias in node.names if alias.name in self.COPIES)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                target = ast.get_source_segment(source, node.args[0]) if node.args else ""
                copy = ((isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                         and func.value.id in modules and func.attr in self.COPIES)
                        or (isinstance(func, ast.Name) and func.id in functions))
                if copy and (relative, target) not in self.ALLOWED_COPIES:
                    found.append(f"{relative}:{node.lineno} copies {target}")
                opened = (isinstance(func, ast.Name) and func.id == "open") or (
                    isinstance(func, ast.Attribute) and func.attr in self.OPENS)
                if opened:
                    receiver = ast.get_source_segment(source, func.value) if isinstance(func, ast.Attribute) else ""
                    if self.DATABASE_NAME.search(f"{receiver} {target}"):
                        found.append(f"{relative}:{node.lineno} opens {receiver or target}")
        self.assertEqual(found, [])

    def test_static_scan_catches_aliases_and_direct_header_reads(self):
        # The two mutants review v5 found slipped past the previous guards.
        mutants = {"from shutil import copy2 as _copy\n_copy(str(source), str(dest))\n": "copies",
                   "import shutil as _sh\n_sh.copy2(db_path, backup)\n": "copies",
                   "header = open(_DB_PATH, 'rb').read(16)\n": "opens"}
        for mutant, kind in mutants.items():
            with self.subTest(mutant=mutant), tempfile.TemporaryDirectory() as directory:
                package = Path(directory) / "truememory"
                package.mkdir()
                (package / "mutant.py").write_text(mutant, encoding="utf-8")
                with patch.dict(globals(), {"ROOT": Path(directory)}):
                    with self.assertRaises(AssertionError) as caught:
                        self.test_no_module_copies_moves_or_opens_database_files()
                self.assertIn(kind, str(caught.exception))

    def test_suite_guard_flags_product_code_opening_a_connected_database(self):
        # conftest's audit hook fails any test whose truememory code opens the
        # file of a database this process is connected to.
        guard = next(module for name, module in sys.modules.items()
                     if name.endswith("conftest") and hasattr(module, "_database_file_opens"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "connected.db"
            conn = sqlite3.connect(path)
            conn.execute("CREATE TABLE t(v)")
            namespace = {}
            reader = os.path.join(directory, "truememory", "status.py")
            exec(compile("def header(path):\n    with open(path, 'rb') as stream:\n        return stream.read(16)\n",
                         reader, "exec"), namespace)
            try:
                namespace["header"](str(path))
                caught = list(guard._database_file_opens)
            finally:
                del guard._database_file_opens[:]
                conn.close()
        self.assertEqual(len(caught), 1)
        self.assertIn("connected.db from " + reader, caught[0])

    def test_tier_backup_includes_committed_wal_content(self):
        from truememory.tier_switch import manager
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "backup-source.db"
            conn = sqlite3.connect(path, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA wal_autocheckpoint=0")
            conn.execute("CREATE TABLE t(v)")
            conn.execute("INSERT INTO t VALUES ('only in the WAL')")
            try:
                with patch.object(manager, "_TRUEMEMORY_DIR", Path(directory)):
                    backup = manager.backup_db(path)
                self.assertTrue(Path(f"{path}-wal").exists())
            finally:
                conn.close()
            copy = sqlite3.connect(backup)
            self.assertEqual(copy.execute("SELECT v FROM t").fetchall(), [("only in the WAL",)])
            copy.close()


STRESS_WORKER = """
import os, random, sqlite3, sys, time
sys.path.insert(0, sys.argv[1])
from truememory import storage
path, index, seconds, mode = sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
r = random.Random(os.getpid())
db = storage.configure_connection(sqlite3.connect(path, isolation_level=None, timeout=2.0))
db.execute("PRAGMA synchronous=NORMAL")
if mode == "plain":
    db.execute("PRAGMA wal_autocheckpoint=200")
committed = seq = checkpoints = 0
problem = None
stop = time.time() + seconds
while time.time() < stop and problem is None:
    roll = 0 if mode == "saturate" else r.randrange(100)
    try:
        if roll < 70:
            seq += 1
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT INTO t(writer, seq, blob) VALUES (?, ?, randomblob(?))", (index, seq, r.randint(64, 1663)))
            db.execute("UPDATE ctr SET v=? WHERE writer=?", (seq, index))
            db.execute("UPDATE churn SET blob=randomblob(400) WHERE id IN (?, ?, ?)", tuple(r.randint(1, 300) for _ in range(3)))
            db.execute("COMMIT")
            committed = seq
            if mode == "saturate":
                storage.checkpoint_if_due(path)  # What the engine's idle hook does after each commit.
        elif roll < 90:
            if mode == "plain":
                m = r.randrange(100)
                db.execute("PRAGMA wal_checkpoint(%s)" % ("PASSIVE" if m < 70 else "RESTART" if m < 85 else "TRUNCATE" if m < 95 else "FULL")).fetchone()
            else:
                storage.protected_checkpoint(path)
            checkpoints += 1
        elif db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            problem = "quick_check"
    except sqlite3.OperationalError as error:
        if db.in_transaction:
            db.execute("ROLLBACK")
        if "locked" not in str(error) and "busy" not in str(error):
            problem = str(error)
    except sqlite3.DatabaseError as error:
        problem = str(error)
    if mode != "saturate" and r.randrange(8) == 0:
        time.sleep(r.randrange(500) / 1e6)
autocheckpoint = db.execute("PRAGMA wal_autocheckpoint").fetchone()[0]
db.close()
print(committed, checkpoints, autocheckpoint, storage.wal_checkpoint_stats()["max_wal_bytes"], problem or "", sep="\t")
"""


def wal_stress(directory: str, *, processes: int = 4, seconds: float = 8.0, mode: str = "guard") -> dict:
    """Antithesis-shaped workload (github.com/antithesishq/sqlite, 3.51.2-instrumented): small
    write transactions, frequent checkpoints and quick checks from several processes at once."""
    path = str(Path(directory) / "stress.db")
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA page_size=4096")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE t(id INTEGER PRIMARY KEY, writer INT, seq INT, blob BLOB);
        CREATE INDEX t_ws ON t(writer, seq);
        CREATE TABLE ctr(writer INTEGER PRIMARY KEY, v INT);
        CREATE TABLE churn(id INTEGER PRIMARY KEY, blob BLOB);
    """)
    conn.executemany("INSERT INTO ctr VALUES (?, 0)", [(i,) for i in range(processes)])
    conn.execute("WITH RECURSIVE s(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM s WHERE x < 300) "
                 "INSERT INTO churn(blob) SELECT randomblob(400) FROM s")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    env = dict(os.environ, TRUEMEMORY_WAL_GUARD="force") if mode in ("guard", "saturate") else dict(os.environ)
    workers = [subprocess.Popen([sys.executable, "-c", STRESS_WORKER, str(ROOT), path, str(i), str(seconds), mode],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
               for i in range(processes)]
    rows = []
    for worker in workers:
        out, err = worker.communicate(timeout=seconds + 120)
        if worker.returncode:
            raise AssertionError(err[-2000:])
        committed, checkpoints, autocheckpoint, max_wal, problem = out.rstrip("\n").split("\t")
        rows.append((int(committed), int(checkpoints), int(autocheckpoint), int(max_wal), problem))
    conn = sqlite3.connect(path)
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        counters = dict(conn.execute("SELECT writer, v FROM ctr").fetchall())
    except sqlite3.DatabaseError as error:
        integrity, counters = str(error), {}
    conn.close()
    return {"integrity": integrity, "problems": [row[4] for row in rows if row[4]],
            "lost": {i: (row[0], counters.get(i)) for i, row in enumerate(rows) if counters.get(i) != row[0]},
            "commits": sum(row[0] for row in rows), "checkpoints": sum(row[1] for row in rows),
            "autocheckpoint": {row[2] for row in rows}, "max_wal_bytes": max(row[3] for row in rows)}


class GuardedCheckpointStress(unittest.TestCase):
    def test_guarded_writers_and_checkpoints_keep_the_database_intact(self):
        # A regression smoke test on any SQLite: with the guard forced on, every
        # checkpoint runs under the write lock, nothing is lost or corrupted.
        # Run unguarded (mode="plain") on SQLite 3.47.1 it found nothing either
        # in our measurements: the race needs schedule fuzzing to hit.
        with tempfile.TemporaryDirectory() as directory:
            result = wal_stress(directory)
        self.assertEqual((result["integrity"], result["problems"], result["lost"]), ("ok", [], {}))
        self.assertEqual(result["autocheckpoint"], {0})
        self.assertGreater(result["commits"], 100)
        self.assertGreater(result["checkpoints"], 20)

    def test_saturated_writers_keep_the_guarded_wal_bounded(self):
        # Review v5 L-v5-7: with checkpoints at most once a second, saturated
        # writers grew the guarded WAL to 237-336 MiB (441 MiB here on 3.47.1).
        with tempfile.TemporaryDirectory() as directory:
            result = wal_stress(directory, seconds=5.0, mode="saturate")
        self.assertEqual((result["integrity"], result["problems"], result["lost"]), ("ok", [], {}))
        self.assertGreater(result["commits"], 1000)
        self.assertLess(result["max_wal_bytes"], 32 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
