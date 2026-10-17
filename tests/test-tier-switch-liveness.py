"""Real create_db databases through certified tier switching, with synthetic models.

Each scenario runs in a fresh interpreter: tier selection, model slots and the
maintenance registry are process-wide, so isolation keeps scenarios honest.
Embedding leases, frozen targets and rerankers are deterministic stand-ins; the
database, journal, job marker, source fences and serving gate are real.
"""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _vector_extension_available() -> bool:
    try:
        import sqlite_vec
        conn = sqlite3.connect(":memory:")
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        finally:
            conn.close()
        return True
    except (ImportError, AttributeError, OSError, sqlite3.Error):
        return False


if not _vector_extension_available():
    raise unittest.SkipTest("Tier switching needs sqlite-vec with extension loading")

PRELUDE = r'''
import json, os, random, sqlite3, struct, sys, threading, time, logging
from contextlib import contextmanager
from pathlib import Path
logging.basicConfig(level=logging.CRITICAL)
HOME = Path(os.environ["HOME"]); DB = os.environ["TRUEMEMORY_DB_PATH"]
CONFIG = HOME / ".truememory" / "config.json"

def emit(**values):
    print("RESULT " + json.dumps(values, sort_keys=True, default=str), flush=True)

def vec(seed, dim=256):
    r = random.Random(seed)
    v = [r.uniform(-1, 1) for _ in range(dim)]
    n = sum(x * x for x in v) ** 0.5
    return struct.pack(f"{dim}f", *[x / n for x in v])

def load_vec(conn):
    import sqlite_vec
    conn.enable_load_extension(True); sqlite_vec.load(conn); conn.enable_load_extension(False)

def connect():
    conn = sqlite3.connect(DB); load_vec(conn); return conn

def seed(n=40, group="basepro", tables=None, path=None):
    from truememory.storage import create_db
    conn = create_db(path or DB); load_vec(conn)
    model = "qwen3_256" if group == "basepro" else "model2vec"
    tables = tables or (f"vec_messages_{group}", f"vec_messages_sep_{group}")
    conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES ('embed_model',?)", (model,))
    conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES ('embed_dim','256')")
    conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES ('qwen3_nan_fix_applied','1')")
    conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES ('l4_entity_profile_migration_done','1')")
    for t in tables:
        conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {t} USING vec0(embedding float[256] distance_metric=cosine)")
    for i in range(n):
        mid = conn.execute("INSERT INTO messages(content,sender,recipient,timestamp) VALUES (?,?,?,?)",
                           (f"synthetic fact number {i}", "alice", "", "2026-01-01T00:00:00")).lastrowid
        conn.execute(f"INSERT INTO {tables[0]}(rowid,embedding) VALUES (?,?)", (mid, vec(mid)))
        conn.execute(f"INSERT INTO {tables[1]}(rowid,embedding) VALUES (?,?)", (mid, vec(-mid)))
    if tables[0] != "vec_messages":
        conn.execute("INSERT OR REPLACE INTO vector_cache_registry(tier_group,vec_table,sep_table,last_embedded_id,"
                     "vector_count,model_name,embedding_dim) VALUES (?,?,?,?,?,?,256)",
                     (group, tables[0], tables[1], n, n, "Qwen3-Embedding-0.6B" if group == "basepro" else "potion-base-8M"))
    conn.commit(); conn.close()

class FakeLease:
    def __init__(self, target, delay):
        self.target, self.delay = target, delay
    def encode(self, texts, timeout=None, batch_size=32):
        import numpy as np
        if self.delay: time.sleep(self.delay)
        out = np.zeros((len(texts), self.target.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            r = random.Random(text)
            out[i] = [r.uniform(-1, 1) for _ in range(self.target.dimension)]
        return out

class FakeEncoder:
    def encode(self, texts, **kwargs):
        import numpy as np
        texts = [texts] if isinstance(texts, str) else list(texts)
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, text in enumerate(texts):
            r = random.Random(text)
            out[i] = [r.uniform(-1, 1) for _ in range(256)]
        return out

class FakeReranker:
    def predict(self, pairs, **kwargs):
        return [0.5 for _ in pairs]

def stub_models(delay=0.0, rerankers_available=lambda name: True, embeddings_available=lambda: True,
                cached=lambda repository: True):
    from truememory import vector_search, reranker
    from truememory.tier_switch import preflight
    preflight.cached_snapshot = cached
    @contextmanager
    def prepare(target, *, timeout=None):
        yield FakeLease(target, delay)
    vector_search.prepare_embedding_target = prepare
    def load(target, timeout=None):
        if not embeddings_available():
            raise OSError("synthetic: embedding weights unavailable offline")
        return FakeEncoder()
    vector_search._load_frozen_embedding_target = load
    def get_reranker(model_name=None, device=None, *, deadline=None, cancelled=None, certified=False):
        name = model_name or reranker.get_current_reranker_name()
        if not rerankers_available(name):
            raise OSError("synthetic: reranker weights unavailable offline")
        reranker._model = FakeReranker(); reranker._model_name = name; reranker._model_certified = True
        return reranker._model
    reranker.get_reranker = get_reranker

def state():
    from truememory.tier_switch.activation import read_activation_state
    from truememory.tier_switch.job import read_selected_job_marker
    conn = sqlite3.connect(DB)
    try:
        st = read_activation_state(conn)
        marker = read_selected_job_marker(conn)
        rows = conn.execute("SELECT id, status FROM rebuild_status ORDER BY id").fetchall()
    finally:
        conn.close()
    result = st.selection or st.legacy_policy
    return {"selected": None if result is None else result.target.tier,
            "acknowledged": None if result is None else result.config_acknowledged,
            "marker": None if marker is None else marker.state, "status_rows": rows}

def switch(tier, *, join=True):
    from truememory.tier_switch.manager import RebuildManager
    manager = RebuildManager()
    try:
        status_id = manager.start_rebuild(tier, db_path=Path(DB), config_path=CONFIG)
    except Exception as error:
        return {"error": type(error).__name__, "message": str(error)}
    if join and manager._active_thread is not None:
        manager._active_thread.join(120)
    return {"status_id": status_id, "outcome": manager._last_outcome, "manager": manager}

def drain():
    from truememory.maintenance import get_coordinator
    get_coordinator(DB).wait(60)
'''


def run_scenario(body: str, *, tier: str = "base", timeout: float = 240) -> list[dict]:
    with tempfile.TemporaryDirectory() as home:
        tm = Path(home) / ".truememory"
        tm.mkdir()
        (tm / "config.json").write_text(json.dumps({"tier": tier}), encoding="utf-8")
        env = {key: value for key, value in os.environ.items() if not key.startswith("TRUEMEMORY_")}
        env.update(
            HOME=home, TRUEMEMORY_NO_MODEL_SERVER="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            TRUEMEMORY_TELEMETRY="off", TRUEMEMORY_EMBED_MODEL=tier, TRUEMEMORY_PRELOAD_MODELS="0",
            TRUEMEMORY_DB_PATH=str(tm / "memories.db"),
            PYTHONPATH=os.pathsep.join(filter(None, (str(ROOT), os.environ.get("PYTHONPATH")))),
        )
        completed = subprocess.run(
            [sys.executable, "-c", PRELUDE + "PRELUDE_SOURCE = " + repr(PRELUDE) + "\n" + textwrap.dedent(body)],
            env=env, cwd=home,
            capture_output=True, text=True, timeout=timeout,
        )
    results = [json.loads(line[7:]) for line in completed.stdout.splitlines() if line.startswith("RESULT ")]
    if completed.returncode != 0 or not results:
        raise AssertionError(f"scenario failed ({completed.returncode})\n{completed.stdout[-3000:]}\n{completed.stderr[-6000:]}")
    return results


class CertifiedSwitchOnCreatedDatabase(unittest.TestCase):
    def test_rebuild_switch_activates_on_real_create_db_schema(self):
        (result,) = run_scenario("""
            seed(40)
            stub_models()
            outcome = switch("edge")
            current = state()
            conn = connect()
            edge = conn.execute("SELECT count(*) FROM vec_messages_edge").fetchone()[0]
            registry = conn.execute("SELECT vec_table FROM vector_cache_registry WHERE tier_group='edge'").fetchone()
            status = conn.execute("SELECT status FROM rebuild_status WHERE id=?", (outcome["status_id"],)).fetchone()
            emit(outcome=outcome["outcome"], state=current, edge=edge, registry=registry, status=status,
                 config=json.loads(CONFIG.read_text())["tier"])
        """)
        self.assertEqual(result["outcome"], "complete")
        self.assertEqual(result["state"]["selected"], "edge")
        self.assertTrue(result["state"]["acknowledged"])
        self.assertEqual(result["edge"], 40)
        self.assertEqual(result["registry"], ["vec_messages_edge"])
        self.assertEqual(result["status"], ["complete"])
        self.assertEqual(result["config"], "edge")



class RebuildKeepsDatabaseAvailable(unittest.TestCase):
    def test_fresh_opens_and_foreign_commits_during_rebuild(self):
        (result,) = run_scenario("""
            import subprocess
            seed(30)
            stub_models(delay=0.15)
            outcome = switch("edge", join=False)
            time.sleep(0.6)
            from truememory.engine import TrueMemoryEngine
            engine = TrueMemoryEngine(DB)
            added = engine.add("written by a fresh engine during the rebuild", sender="bob")["id"]
            listed = len(engine.get_all(limit=100))
            drain()  # automatic consolidation is a different owner; this test is about the rebuild
            child = subprocess.run([sys.executable, "-c",
                "import logging; logging.basicConfig(level=logging.CRITICAL)\\n"
                "from truememory.engine import TrueMemoryEngine\\n"
                "print(len(TrueMemoryEngine(%r).get_all(limit=100)))" % DB], capture_output=True, text=True)
            alive = outcome["manager"]._active_thread.is_alive()
            engine.close(); drain()
            outcome["manager"]._active_thread.join(120)
            conn = connect()
            emit(added=added, listed=listed, child=child.stdout.strip(), child_error=child.stderr[-300:],
                 alive_during_writes=alive, outcome=outcome["manager"]._last_outcome, state=state(),
                 edge=conn.execute("SELECT count(*) FROM vec_messages_edge").fetchone()[0],
                 covered=conn.execute("SELECT count(*) FROM vec_messages_edge WHERE rowid=?", (added,)).fetchone()[0])
        """)
        self.assertEqual(result["added"], 31)
        self.assertEqual(result["listed"], 31)
        self.assertEqual(result["child"], "31", result["child_error"])
        self.assertTrue(result["alive_during_writes"])
        self.assertEqual(result["outcome"], "complete")
        self.assertEqual(result["state"]["selected"], "edge")
        self.assertEqual(result["edge"], 31)
        self.assertEqual(result["covered"], 1)

    def test_correction_during_rebuild_restarts_the_generation(self):
        (result,) = run_scenario("""
            seed(30)
            stub_models(delay=0.1)
            outcome = switch("edge", join=False)
            time.sleep(0.6)
            conn = sqlite3.connect(DB)
            conn.execute("DELETE FROM messages WHERE id=5"); conn.commit(); conn.close()
            outcome["manager"]._active_thread.join(120)
            conn = connect()
            emit(outcome=outcome["manager"]._last_outcome, state=state(),
                 edge=conn.execute("SELECT count(*) FROM vec_messages_edge").fetchone()[0],
                 ghost=conn.execute("SELECT count(*) FROM vec_messages_edge WHERE rowid=5").fetchone()[0])
        """)
        self.assertEqual(result["outcome"], "complete")
        self.assertEqual(result["state"]["selected"], "edge")
        self.assertEqual((result["edge"], result["ghost"]), (29, 0))

    @unittest.skipIf(sys.platform == "win32", "SIGKILL is unavailable on Windows")
    def test_killed_worker_reports_interrupted_and_retry_recovers(self):
        (result,) = run_scenario("""
            import signal, subprocess
            seed(40)
            code = PRELUDE_SOURCE + "\\nstub_models(delay=0.2)\\nswitch('edge', join=False)\\nprint('started', flush=True)\\ntime.sleep(60)\\n"
            child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
            child.stdout.readline(); time.sleep(1.0)
            os.kill(child.pid, signal.SIGKILL); child.wait()
            before = state()
            from truememory.tier_switch.manager import RebuildManager
            status = RebuildManager().get_status(1)["status"]
            stub_models()
            outcome = switch("edge")
            emit(before=before, status=status, outcome=outcome["outcome"], after=state())
        """)
        self.assertEqual(result["before"]["marker"], "selected")
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(result["outcome"], "complete")
        self.assertEqual(result["after"]["selected"], "edge")
        self.assertEqual(result["after"]["status_rows"][0], [1, "interrupted"])

    def test_unbound_selected_marker_can_be_cancelled(self):
        (result,) = run_scenario("""
            seed(10)
            from truememory.embedding_target import EmbeddingTarget
            from truememory.rebuild_source import ensure_rebuild_tracking
            from truememory.tier_switch.job import select_tier_job
            from truememory.tier_switch.manager import RebuildManager
            conn = sqlite3.connect(DB)
            ensure_rebuild_tracking(conn)
            job = select_tier_job(conn, EmbeddingTarget.capture("edge"))
            conn.close()
            try:
                RebuildManager().cancel(7, expected_job_id="0" * 32, db_path=Path(DB))
                wrong = None
            except Exception as error:
                wrong = type(error).__name__
            cancelled = RebuildManager().cancel(7, expected_job_id=job.job_id, db_path=Path(DB))
            emit(wrong=wrong, cancelled=cancelled, state=state())
        """)
        self.assertEqual(result["wrong"], "TierSwitchUnsupportedError")
        self.assertEqual(result["cancelled"]["status"], "cancelled")
        self.assertEqual(result["state"]["marker"], "cancelled")



class UnusableInactivePairsAreReset(unittest.TestCase):
    def test_released_switch_back_cache_is_discarded_and_rebuilt(self):
        (result,) = run_scenario("""
            seed(30)
            conn = connect()
            for t in ("vec_messages_edge", "vec_messages_sep_edge"):
                conn.execute(f"CREATE VIRTUAL TABLE {t} USING vec0(embedding float[256] distance_metric=cosine)")
            for i in range(1, 21):
                conn.execute("INSERT INTO vec_messages_edge(rowid,embedding) VALUES (?,?)", (i, vec(i)))
                conn.execute("INSERT INTO vec_messages_sep_edge(rowid,embedding) VALUES (?,?)", (i, vec(-i)))
            conn.execute("INSERT INTO vector_cache_registry(tier_group,vec_table,sep_table,last_embedded_id,vector_count,"
                         "model_name,embedding_dim) VALUES ('edge','vec_messages_edge','vec_messages_sep_edge',20,20,'potion-base-8M',256)")
            conn.commit(); conn.close()
            stub_models()
            outcome = switch("edge")
            conn = connect()
            emit(outcome=outcome["outcome"], state=state(),
                 edge=conn.execute("SELECT count(*) FROM vec_messages_edge").fetchone()[0],
                 basepro=conn.execute("SELECT count(*) FROM vec_messages_basepro").fetchone()[0],
                 registry=conn.execute("SELECT model_name, vector_count FROM vector_cache_registry WHERE tier_group='edge'").fetchone())
        """)
        self.assertEqual(result["outcome"], "complete")
        self.assertEqual(result["state"]["selected"], "edge")
        self.assertEqual(result["edge"], 30)
        self.assertEqual(result["basepro"], 30)
        self.assertEqual(result["registry"], ["model2vec", 30])

    def test_round_trip_returns_to_previously_served_groups(self):
        (result,) = run_scenario("""
            seed(25)
            stub_models()
            first = switch("edge")["outcome"]
            from truememory.engine import TrueMemoryEngine
            engine = TrueMemoryEngine(DB)
            appended = engine.add("written while edge serves", sender="alice")["id"]
            engine.close(); drain()
            back = switch("base")["outcome"]
            after_back = state()
            again = switch("edge")["outcome"]
            conn = connect()
            emit(first=first, back=back, after_back=after_back, again=again, state=state(),
                 edge=conn.execute("SELECT count(*) FROM vec_messages_edge").fetchone()[0],
                 basepro=conn.execute("SELECT count(*) FROM vec_messages_basepro").fetchone()[0],
                 appended=appended)
        """)
        self.assertEqual((result["first"], result["back"], result["again"]), ("complete", "complete", "complete"))
        self.assertEqual(result["after_back"]["selected"], "base")
        self.assertEqual(result["state"]["selected"], "edge")
        self.assertEqual((result["edge"], result["basepro"]), (26, 26))

    def test_partial_pair_invalidated_by_a_forget_is_rebuilt(self):
        (result,) = run_scenario("""
            seed(30)
            stub_models(delay=0.1)
            outcome = switch("edge", join=False)
            for _ in range(400):
                conn = connect()
                published = conn.execute("SELECT count(*) FROM vec_messages_edge").fetchone()[0]
                conn.close()
                if published:
                    break
                time.sleep(0.05)
            outcome["manager"].cancel()
            outcome["manager"]._active_thread.join(60)
            conn = connect()
            partial = conn.execute("SELECT count(*) FROM vec_messages_edge").fetchone()[0]
            conn.execute("DELETE FROM messages WHERE id=12"); conn.commit(); conn.close()
            stub_models()
            retried = switch("edge")["outcome"]
            conn = connect()
            emit(cancelled=outcome["manager"]._last_outcome, partial=partial, retried=retried, state=state(),
                 edge=conn.execute("SELECT count(*) FROM vec_messages_edge").fetchone()[0])
        """)
        self.assertEqual(result["cancelled"], "cancelled")
        self.assertGreater(result["partial"], 0)
        self.assertEqual(result["retried"], "complete")
        self.assertEqual(result["state"]["selected"], "edge")
        self.assertEqual(result["edge"], 29)



class ActivationFailuresKeepPreviousServing(unittest.TestCase):
    def test_unloadable_reranker_refuses_before_the_durable_commit(self):
        (result,) = run_scenario("""
            seed(20)
            stub_models(rerankers_available=lambda name: "MiniLM" not in name)
            outcome = switch("edge")
            emit(outcome=outcome["outcome"], state=state(), config=json.loads(CONFIG.read_text())["tier"])
        """)
        self.assertEqual(result["outcome"], "interrupted")
        self.assertIsNone(result["state"]["selected"])
        self.assertEqual(result["state"]["marker"], "failed")
        self.assertEqual(result["state"]["status_rows"], [[1, "failed"]])
        self.assertEqual(result["config"], "base")

    def test_projection_fault_after_commit_restores_previous_selection(self):
        (result,) = run_scenario("""
            seed(20)
            loadable = {"ok": False}
            stub_models(embeddings_available=lambda: loadable["ok"])
            outcome = switch("edge")
            restored = state()
            config = json.loads(CONFIG.read_text())
            from truememory.engine import TrueMemoryEngine
            engine = TrueMemoryEngine(DB)
            added = engine.add("legacy serving still accepts writes", sender="alice")["id"]
            engine.close(); drain()
            loadable["ok"] = True
            retried = switch("edge")
            emit(outcome=outcome["outcome"], served=outcome["manager"]._served_tier, restored=restored,
                 config=config, added=added, retried=retried["outcome"], final=state())
        """)
        self.assertEqual(result["outcome"], "failed")
        self.assertEqual(result["served"], "base")
        self.assertIsNone(result["restored"]["selected"])
        self.assertEqual(result["restored"]["marker"], "failed")
        self.assertEqual(result["restored"]["status_rows"], [[1, "failed"]])
        self.assertEqual(result["config"]["tier"], "base")
        self.assertEqual(result["added"], 21)
        self.assertEqual(result["retried"], "complete")
        self.assertEqual(result["final"]["selected"], "edge")



CHILD_WITHOUT_MODELS = """
stub_models(embeddings_available=lambda: False, rerankers_available=lambda name: False)
from truememory.engine import TrueMemoryEngine
engine = TrueMemoryEngine(DB)
out = {"get_all": len(engine.get_all(limit=100)), "get": engine.get(1)["content"]}
out["add"] = engine.add("stored without vectors", sender="alice")["id"]
out["delete"] = engine.delete(2)
engine._ensure_connection()
out["directives"] = engine.conn.execute("SELECT count(*) FROM messages WHERE directive=1").fetchone()[0]
try:
    out["search"] = [row.get("id") for row in engine.search("synthetic fact number 3", limit=3)]
except Exception as error:
    out["search"] = type(error).__name__
engine.close(); drain()
emit(**out)
"""


class ControlledDatabasesWithoutModels(unittest.TestCase):
    def run_child(self, switch_to):
        (result,) = run_scenario("""
            import subprocess
            seed(10)
            stub_models()
            switched = switch(%r)["outcome"]
            child = subprocess.run([sys.executable, "-c", PRELUDE_SOURCE + %r], capture_output=True, text=True)
            lines = [json.loads(line[7:]) for line in child.stdout.splitlines() if line.startswith("RESULT ")]
            conn = connect()
            table = "vec_messages_edge" if %r == "edge" else "vec_messages_basepro"
            emit(switched=switched, child=lines[0] if lines else child.stderr[-3000:], state=state(),
                 vector=conn.execute(f"SELECT count(*) FROM {table} WHERE rowid=11").fetchone()[0],
                 count=conn.execute("SELECT count(*) FROM messages").fetchone()[0])
        """ % (switch_to, CHILD_WITHOUT_MODELS, switch_to))
        return result

    def assert_source_operations(self, result, selected):
        self.assertEqual(result["switched"], "complete")
        self.assertEqual(result["state"]["selected"], selected)
        child = result["child"]
        self.assertIsInstance(child, dict, child)
        self.assertEqual(child["get_all"], 10)
        self.assertEqual(child["get"], "synthetic fact number 0")
        self.assertEqual(child["add"], 11)
        self.assertTrue(child["delete"])
        self.assertEqual(child["directives"], 0)
        # Keyword (FTS) results, exactly like a legacy store without its model.
        self.assertIsInstance(child["search"], list, child["search"])
        self.assertIn(4, child["search"])
        self.assertEqual((result["vector"], result["count"]), (0, 10))

    def test_policy_controlled_database(self):
        self.assert_source_operations(self.run_child("pro"), "pro")

    def test_certified_selection(self):
        self.assert_source_operations(self.run_child("edge"), "edge")



CHILD_OPEN = """
from truememory.engine import TrueMemoryEngine
engine = TrueMemoryEngine(DB)
added = engine.add("written after reopening", sender="alice")["id"]
listed = len(engine.get_all(limit=200))
engine.close(); drain()
conn = sqlite3.connect(DB)
tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
emit(added=added, listed=listed, state=state(),
     maintenance_tables=sorted(t for t in tables if t in {"maintenance_source_state", "maintenance_layers",
                                                          "dunbar_generated_relationships"}),
     retired=sorted(row[0] for row in conn.execute("SELECT key FROM metadata WHERE key LIKE '%_retired'")),
     config=json.loads(CONFIG.read_text()))
"""



STRIP = """
def strip_upgrade_schema(conn):
    for name in ("maintenance_source_state", "maintenance_layers", "dunbar_generated_relationships"):
        conn.execute(f"DROP TABLE IF EXISTS {name}")
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND "
                                "(name LIKE 'messages_maintenance_%' OR name LIKE 'dunbar_relationships_%')").fetchall():
        conn.execute(f"DROP TRIGGER {name}")
"""


class ControlledDatabasesReconcileAtOpen(unittest.TestCase):
    def reopen(self, setup: str) -> dict:
        (result,) = run_scenario(f"""
            import subprocess
            {setup}
            child = subprocess.run([sys.executable, "-c", PRELUDE_SOURCE + "stub_models()\\n" + {CHILD_OPEN!r}],
                                   capture_output=True, text=True)
            lines = [json.loads(line[7:]) for line in child.stdout.splitlines() if line.startswith("RESULT ")]
            emit(child=lines[0] if lines else child.stderr[-3000:])
        """.replace("\n            ", "\n"))
        self.assertIsInstance(result["child"], dict, result["child"])
        return result["child"]

    def test_first_policy_switch_after_upgrade_and_controlled_reopen_migrate(self):
        child = self.reopen(STRIP + """
seed(10)
conn = sqlite3.connect(DB); strip_upgrade_schema(conn); conn.commit(); conn.close()
stub_models()
assert switch("pro")["outcome"] == "complete"
conn = sqlite3.connect(DB)
migrated = conn.execute("SELECT 1 FROM sqlite_master WHERE name='maintenance_layers'").fetchone()
assert migrated, "the switch itself migrates before publishing the policy"
strip_upgrade_schema(conn); conn.commit(); conn.close()
""")
        self.assertEqual(child["state"]["selected"], "pro")
        self.assertEqual(child["maintenance_tables"],
                         ["dunbar_generated_relationships", "maintenance_layers", "maintenance_source_state"])
        self.assertEqual((child["added"], child["listed"]), (11, 11))

    def test_older_release_tier_switch_retires_the_journal(self):
        child = self.reopen("""
seed(20)
stub_models()
assert switch("edge")["outcome"] == "complete"
conn = connect()
mid = conn.execute("INSERT INTO messages(content,sender) VALUES ('added by 0.7.6.1 after its switch','alice')").lastrowid
conn.execute("INSERT INTO vec_messages_basepro(rowid,embedding) VALUES (?,?)", (mid, vec(mid)))
conn.execute("INSERT INTO vec_messages_sep_basepro(rowid,embedding) VALUES (?,?)", (mid, vec(-mid)))
conn.execute("INSERT OR REPLACE INTO vector_cache_registry(tier_group,vec_table,sep_table,last_embedded_id,vector_count,"
             "model_name,embedding_dim,last_updated,created) VALUES ('basepro','vec_messages_basepro',"
             "'vec_messages_sep_basepro',?,?,'Qwen3-Embedding-0.6B',256,0,0)", (mid, mid))
conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES ('embed_model','qwen3_256')")
conn.commit(); conn.close()
config = json.loads(CONFIG.read_text()); config["tier"] = "base"; CONFIG.write_text(json.dumps(config))
""")
        self.assertIsNone(child["state"]["selected"])
        self.assertEqual(child["retired"], ["tier_activation_v1_retired", "tier_selected_v1_retired"])
        self.assertEqual((child["added"], child["listed"]), (22, 22))

    def test_policy_on_generic_pair_stranded_by_older_release_is_retired(self):
        child = self.reopen("""
seed(10, tables=("vec_messages", "vec_messages_sep"))
from truememory.embedding_target import EmbeddingTarget
from truememory.tier_switch.activation import commit_config_only_transition
from truememory.tier_switch.projection import capture_config_guard
conn = connect()
commit_config_only_transition(conn, expected_selection_generation=None, expected_intent_id=None,
                              target=EmbeddingTarget.capture("pro"), reranker_id="Alibaba-NLP/gte-reranker-modernbert-base",
                              expected_config=capture_config_guard(config_path=CONFIG),
                              legacy_pair=("vec_messages", "vec_messages_sep"))
for name in ("vec_messages_basepro", "vec_messages_sep_basepro"):
    conn.execute(f"CREATE VIRTUAL TABLE {name} USING vec0(embedding float[256] distance_metric=cosine)")
conn.execute("INSERT INTO vec_messages_basepro SELECT * FROM vec_messages")
conn.execute("INSERT INTO vec_messages_sep_basepro SELECT * FROM vec_messages_sep")
conn.execute("DROP TABLE vec_messages"); conn.execute("DROP TABLE vec_messages_sep")
conn.execute("INSERT INTO vector_cache_registry(tier_group,vec_table,sep_table,last_embedded_id,vector_count,model_name,"
             "embedding_dim) VALUES ('basepro','vec_messages_basepro','vec_messages_sep_basepro',10,10,'Qwen3-Embedding-0.6B',256)")
conn.commit(); conn.close()
""")
        self.assertIsNone(child["state"]["selected"])
        self.assertEqual(child["retired"], ["tier_activation_v1_retired"])
        self.assertEqual((child["added"], child["listed"]), (11, 11))

    def test_policy_switch_migrates_a_populated_generic_pair_first(self):
        (result,) = run_scenario("""
            seed(10, tables=("vec_messages", "vec_messages_sep"))
            stub_models()
            outcome = switch("pro")["outcome"]
            from truememory.tier_switch.activation import read_activation_state
            conn = connect()
            emit(outcome=outcome, tables=read_activation_state(conn).legacy_policy.tables,
                 generic=conn.execute("SELECT count(*) FROM sqlite_master WHERE name='vec_messages'").fetchone()[0],
                 rows=conn.execute("SELECT count(*) FROM vec_messages_basepro").fetchone()[0])
        """)
        self.assertEqual(result["outcome"], "complete")
        self.assertEqual(result["tables"], ["vec_messages_basepro", "vec_messages_sep_basepro"])
        self.assertEqual((result["generic"], result["rows"]), (0, 10))

    def test_unacknowledged_selection_is_mirrored_on_open(self):
        child = self.reopen("""
seed(10)
stub_models()
assert switch("edge")["outcome"] == "complete"
conn = sqlite3.connect(DB)
selected = json.loads(conn.execute("SELECT value FROM metadata WHERE key='tier_selected_v1'").fetchone()[0])
intent = json.loads(conn.execute("SELECT value FROM metadata WHERE key='tier_activation_v1'").fetchone()[0])
selected["config_acknowledged"] = False; intent["state"] = "db_selected"
conn.execute("UPDATE metadata SET value=? WHERE key='tier_selected_v1'", (json.dumps(selected, sort_keys=True, separators=(",", ":")),))
conn.execute("UPDATE metadata SET value=? WHERE key='tier_activation_v1'", (json.dumps(intent, sort_keys=True, separators=(",", ":")),))
conn.commit(); conn.close()
config = json.loads(CONFIG.read_text())
config["tier"] = intent["expected_config"]["tier"]
config["tier_activation_generation"] = intent["expected_config"]["generation"]
CONFIG.write_text(json.dumps(config))
""")
        self.assertEqual(child["state"]["selected"], "edge")
        self.assertTrue(child["state"]["acknowledged"])
        self.assertEqual(child["config"]["tier"], "edge")



class RerankerOverridesDoNotFailOtherOperations(unittest.TestCase):
    def test_store_during_deep_search_waits_and_completes(self):
        (result,) = run_scenario("""
            seed(5)
            stub_models()
            from truememory.engine import TrueMemoryEngine
            from truememory.tier_switch.runtime import engine_serving_operation
            engine = TrueMemoryEngine(DB)
            engine.get_all(limit=1); drain()
            held, release = threading.Event(), threading.Event()
            def deep():
                with engine_serving_operation(engine, reranker_override="BAAI/bge-reranker-v2-m3"):
                    held.set(); release.wait(10)
            worker = threading.Thread(target=deep); worker.start(); held.wait(10)
            threading.Timer(1.0, release.set).start()
            started = time.monotonic()
            added = engine.add("stored while a deep search runs", sender="alice")["id"]
            waited = time.monotonic() - started
            worker.join(10)
            engine.close(); drain()
            emit(added=added, waited=waited)
        """)
        self.assertEqual(result["added"], 6)
        self.assertGreaterEqual(result["waited"], 0.8)


class ActivationLockOrder(unittest.TestCase):
    def test_activation_and_relaunched_consolidation_do_not_deadlock(self):
        # Consolidation owns maintenance and then a serving reader. Activation
        # must take the same order; the reverse froze the process (H1).
        (result,) = run_scenario("""
            seed(30)
            stub_models(delay=0.05)
            from truememory.engine import TrueMemoryEngine
            from truememory.maintenance import get_coordinator
            from truememory.tier_switch import runtime, serving
            from truememory.tier_switch.manager import RebuildManager
            engine = TrueMemoryEngine(DB); engine.get_all(limit=1); drain()
            finalize = RebuildManager._finalize_rebuild
            activating = threading.Event()
            def marked(self, *args, **kwargs):
                activating.set()
                return finalize(self, *args, **kwargs)
            RebuildManager._finalize_rebuild = marked
            opened = runtime.open_serving_connection
            def open_when_activation_starts(path, factory):
                # The coordinator owns maintenance from launch; it reaches its
                # serving reader exactly when the worker starts activating.
                started = time.monotonic()
                while not (activating.is_set() or serving._DEFAULT_GATE._writers) and time.monotonic() - started < 30:
                    time.sleep(0.01)
                time.sleep(0.2)
                return opened(path, factory)
            runtime.open_serving_connection = open_when_activation_starts
            manager = switch("edge", join=False)["manager"]
            worker = manager._active_thread
            time.sleep(0.3)
            launched = get_coordinator(DB).request_layers(threshold=1, include_layers=True)
            worker.join(45)
            finished = not worker.is_alive()
            served = threading.Event()
            threading.Thread(target=lambda: (engine.get_all(limit=1), served.set()), daemon=True).start()
            responsive = served.wait(5)
            if not finished:
                manager.cancel(); worker.join(30)  # Breaks a deadlock so the scenario can report it.
            served.wait(30)
            engine.close(); drain()
            emit(launched=launched, finished=finished, responsive=responsive, outcome=manager._last_outcome, state=state())
        """)
        self.assertTrue(result["launched"])
        self.assertTrue(result["finished"], result)
        self.assertTrue(result["responsive"])
        self.assertEqual(result["outcome"], "complete")
        self.assertEqual(result["state"]["selected"], "edge")


def steps(*parts: str) -> str:
    return "".join(textwrap.dedent(part) for part in parts)


def run_child(source: str) -> str:
    """Scenario code that runs ``source`` in a fresh interpreter and binds its RESULT to ``child``."""
    return ("_run = subprocess.run([sys.executable, '-c', PRELUDE_SOURCE + %r], capture_output=True, text=True)\n"
            "_lines = [json.loads(l[7:]) for l in _run.stdout.splitlines() if l.startswith('RESULT ')]\n"
            "child = _lines[0] if _lines else {'error': _run.stderr[-2000:], 'returncode': _run.returncode}\n" % source)


# Drains keep automatic consolidation (another maintenance owner) out of the way.
LONG_RUNNING = """
stub_models()
from truememory.engine import TrueMemoryEngine
engine = TrueMemoryEngine(DB)
engine.get_all(limit=1); drain()
print("OPEN", flush=True)
sys.stdin.readline()
window_id = engine.add("second server stores during the activation window", sender="alice")["id"]
drain()
print("ADDED", window_id, flush=True)
sys.stdin.readline()
out = {"window": window_id}
for name, call in (("get_all", lambda: len(engine.get_all(limit=100))),
                   ("add", lambda: engine.add("second server after the other process changed tier", sender="alice")["id"]),
                   ("search", lambda: [row.get("id") for row in engine.search("second server", limit=3)])):
    try:
        out[name] = call()
    except Exception as error:
        out[name] = type(error).__name__ + ": " + str(error)[:120]
engine.close(); drain()
emit(**out)
"""

CRASH_IN_RESTORE = """
stub_models(embeddings_available=lambda: False)
from truememory.tier_switch import projection
from truememory.tier_switch.manager import RebuildManager
if CRASH_POINT == "config":
    projection.revert_activation_mirror = lambda *args, **kwargs: os._exit(9)
else:
    RebuildManager._restore_journal_rows = lambda *args, **kwargs: os._exit(9)
switch("edge")
"""

OPEN_AFTER_CRASH = """
stub_models()
from truememory.engine import TrueMemoryEngine
from truememory.tier_switch.runtime import engine_serving_operation
engine = TrueMemoryEngine(DB)
added = engine.add("after a crash inside the restore", sender="alice")["id"]
with engine_serving_operation(engine) as operation:
    served = list(operation.tables)
engine.close(); drain()
emit(added=added, served=served, back=switch("base").get("outcome"), state=state())
"""

ADD_IN_CALLER_TRANSACTION = """
stub_models()
from truememory.engine import TrueMemoryEngine
engine = TrueMemoryEngine(DB)
engine._ensure_connection()
engine.conn.execute("BEGIN")
try:
    borrowed = engine.add("inside a caller transaction", sender="alice")["id"]
except Exception as error:
    borrowed = type(error).__name__
engine.conn.rollback()
clean = engine.add("after the caller transaction", sender="alice")["id"]
engine.close(); drain()
emit(borrowed=borrowed, clean=clean)
"""

ADD_WITHOUT_MODELS = """
stub_models(embeddings_available=lambda: False, rerankers_available=lambda name: False)
from truememory.engine import TrueMemoryEngine
engine = TrueMemoryEngine(DB)
ids = [engine.add(f"captured while models were down {i}", sender="alice")["id"] for i in range(3)]
coverage = engine.get_stats()["vector_coverage"]
engine.close(); drain()
emit(ids=ids, coverage=coverage)
"""

ADD_WITH_MODELS = """
stub_models()
from truememory.engine import TrueMemoryEngine
engine = TrueMemoryEngine(DB)
added = engine.add("models are back", sender="alice")["id"]
coverage = engine.get_stats()["vector_coverage"]
engine.close(); drain()
emit(added=added, coverage=coverage)
"""

HOLD_MAINTENANCE = """
from truememory.maintenance import MaintenanceBusyError, maintenance_owner
sys.stdin.readline()
while True:
    try:
        owner = maintenance_owner(DB)
        owner.__enter__()
        break
    except MaintenanceBusyError:
        time.sleep(0.05)
print("HELD", flush=True)
sys.stdin.readline()
owner.__exit__(None, None, None)
"""

ADD_ACROSS_COMMIT = """
stub_models()
from truememory.engine import TrueMemoryEngine
from truememory.tier_switch import writer
engine = TrueMemoryEngine(DB); engine.get_all(limit=1); drain()
capture = writer.capture_writer_selection
def capture_before_the_commit(conn):
    # The selection is captured under the legacy admission; the activation
    # commit lands before the writer opens.
    captured = capture(conn)
    probe = sqlite3.connect(DB); started = time.monotonic()
    while time.monotonic() - started < 60 and not probe.execute(
            "SELECT 1 FROM metadata WHERE key='tier_selected_v1'").fetchone():
        time.sleep(0.05)
    probe.close()
    return captured
writer.capture_writer_selection = capture_before_the_commit
print("READY", flush=True)
try:
    added = engine.add("hook capture racing the activation commit", sender="alice")["id"]
except Exception as error:
    added = type(error).__name__ + ": " + str(error)[:120]
engine.close(); drain()
emit(added=added)
"""

RETIRE_BY_OPENING = """
stub_models()
from truememory.engine import TrueMemoryEngine
engine = TrueMemoryEngine(DB); engine.get_all(limit=1); engine.close(); drain()
emit(opened=True)
"""


class ProjectionFaultsAfterCommit(unittest.TestCase):
    def test_used_selection_is_kept_and_other_processes_keep_working(self):
        (result,) = run_scenario("""
            import subprocess
            seed(20)
            stub_models()
            other = subprocess.Popen([sys.executable, "-c", PRELUDE_SOURCE + %r], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            assert other.stdout.readline().startswith("OPEN")
            from truememory import vector_search
            loaded = vector_search._load_frozen_embedding_target
            seen = {}
            def load(target, timeout=None):
                if "fault" not in seen:
                    # Another process writes under the committed selection, then
                    # this projection fails (memory pressure, missing cache).
                    seen["fault"] = True
                    other.stdin.write("go\\n"); other.stdin.flush()
                    seen["added"] = other.stdout.readline().strip()
                    raise OSError("synthetic: frozen target could not load after the commit")
                return loaded(target, timeout)
            vector_search._load_frozen_embedding_target = load
            switched = switch("edge")
            outcome = switched.get("outcome", switched)
            out, err = other.communicate("go\\n", timeout=120)
            results = [json.loads(line[7:]) for line in out.splitlines() if line.startswith("RESULT ")]
            window = int(seen["added"].split()[1])
            conn = connect()
            emit(outcome=outcome, state=state(), other=results[0] if results else err[-2000:],
                 window_in_edge=conn.execute("SELECT count(*) FROM vec_messages_edge WHERE rowid=?", (window,)).fetchone()[0])
        """ % LONG_RUNNING)
        self.assertEqual(result["outcome"], "activation_pending")
        self.assertEqual(result["state"]["selected"], "edge")
        self.assertEqual(result["window_in_edge"], 1)
        other = result["other"]
        self.assertIsInstance(other, dict, other)
        self.assertEqual((other["window"], other["get_all"], other["add"]), (21, 21, 22))
        self.assertIsInstance(other["search"], list, other["search"])

    def test_kept_selection_still_allows_switching_back(self):
        # Review v5 M-v5-1: after a post-commit projection failure keeps the new
        # selection, switching back was refused ("The committed tier must be
        # reconciled before another switch") and the job's row stayed running.
        (result,) = run_scenario("""
            import subprocess
            seed(20)
            stub_models()
            other = subprocess.Popen([sys.executable, "-c", PRELUDE_SOURCE + %r], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            assert other.stdout.readline().startswith("OPEN")
            from truememory import vector_search
            loaded = vector_search._load_frozen_embedding_target
            seen = {}
            def load(target, timeout=None):
                if "fault" not in seen:
                    seen["fault"] = True
                    other.stdin.write("go\\n"); other.stdin.flush()
                    seen["added"] = other.stdout.readline().strip()
                    raise OSError("synthetic: frozen target could not load after the commit")
                return loaded(target, timeout)
            vector_search._load_frozen_embedding_target = load
            kept = switch("edge")["outcome"]
            other.communicate("go\\n", timeout=120)
            pending = state()
            back = switch("base")
            emit(kept=kept, pending=pending, back=back.get("outcome", back.get("message")), state=state(),
                 config=json.loads(CONFIG.read_text())["tier"])
        """ % LONG_RUNNING)
        self.assertEqual(result["kept"], "activation_pending")
        self.assertEqual(result["pending"]["selected"], "edge")
        self.assertEqual(result["pending"]["status_rows"], [[1, "activation_pending"]])
        self.assertEqual(result["back"], "complete")
        self.assertEqual((result["state"]["selected"], result["state"]["acknowledged"], result["config"]), ("base", True, "base"))

    def test_unused_selection_is_restored(self):
        (result,) = run_scenario("""
            seed(20)
            stub_models(embeddings_available=lambda: False)
            outcome = switch("edge")["outcome"]
            stub_models()
            from truememory.engine import TrueMemoryEngine
            from truememory.tier_switch.runtime import engine_serving_operation
            engine = TrueMemoryEngine(DB)
            added = engine.add("after the restore", sender="alice")["id"]
            with engine_serving_operation(engine) as operation:
                served = list(operation.tables)
            engine.close(); drain()
            emit(outcome=outcome, state=state(), config=json.loads(CONFIG.read_text())["tier"], added=added, served=served)
        """)
        self.assertEqual(result["outcome"], "failed")
        self.assertIsNone(result["state"]["selected"])
        self.assertEqual(result["state"]["marker"], "failed")
        self.assertEqual(result["config"], "base")
        self.assertEqual(result["added"], 21)
        self.assertEqual(result["served"], ["vec_messages_basepro", "vec_messages_sep_basepro"])

    def test_crash_inside_the_restore_leaves_a_servable_store(self):
        for point in ("config", "journal"):
            with self.subTest(crash_before=point):
                self.assert_servable_after_crash(point)

    def assert_servable_after_crash(self, point):
        (result,) = run_scenario(steps("""
            import subprocess
            seed(20)
            os.environ["TRUEMEMORY_EMBED_MODEL"] = ""
            crasher = "CRASH_POINT = %r\\n" + %r
            crashed = subprocess.run([sys.executable, "-c", PRELUDE_SOURCE + crasher]).returncode
        """ % (point, CRASH_IN_RESTORE), run_child(OPEN_AFTER_CRASH), """
            conn = connect()
            emit(crashed=crashed, opener=child,
                 added_vectors={group: conn.execute(f"SELECT count(*) FROM vec_messages_{group} WHERE rowid=21").fetchone()[0]
                                for group in ("edge", "basepro")})
        """))
        self.assertEqual(result["crashed"], 9)
        opener = result["opener"]
        self.assertEqual(opener.get("served"), ["vec_messages_edge", "vec_messages_sep_edge"], opener)
        self.assertEqual(opener["added"], 21)
        self.assertEqual(opener["back"], "complete")
        self.assertEqual(opener["state"]["selected"], "base")
        self.assertEqual(result["added_vectors"], {"edge": 1, "basepro": 1})


class ConfigChangesFromOlderReleases(unittest.TestCase):
    def test_same_space_config_change_retires_the_policy(self):
        # 0.7.6.1 switches base and pro by rewriting config.json only.
        (result,) = run_scenario("""
            seed(10)
            os.environ["TRUEMEMORY_EMBED_MODEL"] = ""
            stub_models()
            assert switch("pro")["outcome"] == "complete"
            policy = state()
            config = json.loads(CONFIG.read_text()); config["tier"] = "base"
            CONFIG.write_text(json.dumps(config))
            from truememory.engine import TrueMemoryEngine
            from truememory.tier_switch.runtime import engine_serving_operation
            engine = TrueMemoryEngine(DB)
            with engine_serving_operation(engine) as operation:
                served = operation.tier
            added = engine.add("after the older release changed tier", sender="alice")["id"]
            engine.close(); drain()
            emit(policy=policy, served=served, added=added, state=state(), switched=switch("edge").get("outcome"),
                 final=state())
        """)
        self.assertEqual((result["policy"]["selected"], result["policy"]["acknowledged"]), ("pro", True))
        self.assertEqual(result["served"], "base")
        self.assertEqual(result["added"], 11)
        self.assertIsNone(result["state"]["selected"])
        self.assertEqual(result["switched"], "complete")
        self.assertEqual(result["final"]["selected"], "edge")

    def test_cross_space_config_edit_is_restored_to_the_selection(self):
        (result,) = run_scenario("""
            seed(10)
            stub_models()
            assert switch("edge")["outcome"] == "complete"
            config = json.loads(CONFIG.read_text()); config["tier"] = "pro"  # Edited; nothing was rebuilt.
            CONFIG.write_text(json.dumps(config))
            switched = switch("base").get("outcome")
            emit(switched=switched, state=state(), config=json.loads(CONFIG.read_text())["tier"])
        """)
        self.assertEqual(result["switched"], "complete")
        self.assertEqual(result["state"]["selected"], "base")
        self.assertEqual(result["config"], "base")


class VectorlessMemories(unittest.TestCase):
    def test_add_in_caller_transaction_refuses_instead_of_dropping_vectors(self):
        (result,) = run_scenario(steps("""
            import subprocess
            seed(10)
            stub_models()
            assert switch("edge")["outcome"] == "complete"
        """, run_child(ADD_IN_CALLER_TRANSACTION), """
            conn = connect()
            emit(child=child, messages=conn.execute("SELECT count(*) FROM messages").fetchone()[0],
                 clean_vector=conn.execute("SELECT count(*) FROM vec_messages_edge WHERE rowid=?",
                                           (child.get("clean", -1),)).fetchone()[0])
        """))
        self.assertEqual(result["child"].get("borrowed"), "TierRuntimeError", result["child"])
        self.assertEqual((result["child"]["clean"], result["clean_vector"], result["messages"]), (11, 1, 11))

    def test_vectorless_memories_are_counted_and_backfilled(self):
        (result,) = run_scenario(steps("""
            import subprocess
            seed(10)
            stub_models()
            assert switch("edge")["outcome"] == "complete"
        """, run_child(ADD_WITHOUT_MODELS), """
            offline = child
        """, run_child(ADD_WITH_MODELS), """
            conn = connect()
            emit(offline=offline, online=child,
                 vectors=[conn.execute(f"SELECT count(*) FROM {table} WHERE rowid IN (11, 12, 13, 14)").fetchone()[0]
                          for table in ("vec_messages_edge", "vec_messages_sep_edge")])
        """))
        offline, online = result["offline"], result["online"]
        self.assertEqual(offline.get("ids"), [11, 12, 13], offline)
        self.assertEqual((offline["coverage"]["vectorless"], offline["coverage"]["backfill_from"]), (3, "11"))
        self.assertEqual(online.get("added"), 14, online)
        self.assertEqual((online["coverage"]["vectorless"], online["coverage"]["backfill_from"]), (0, None))
        self.assertEqual(result["vectors"], [4, 4])


class ActivationEdgeCases(unittest.TestCase):
    def test_cancel_while_waiting_for_maintenance_ends_the_job(self):
        # Waiting for maintenance must not hold this process's serving writer.
        (result,) = run_scenario("""
            import subprocess
            seed(20)
            stub_models(delay=0.02)
            from truememory.engine import TrueMemoryEngine
            from truememory.tier_switch.manager import RebuildManager
            engine = TrueMemoryEngine(DB); engine.get_all(limit=1); drain()
            holder = subprocess.Popen([sys.executable, "-c", PRELUDE_SOURCE + %r], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True)
            finalize = RebuildManager._finalize_rebuild
            activating = threading.Event()
            def finalize_after_maintenance_is_taken(self, *args, **kwargs):
                # Another process starts maintenance just as activation begins.
                holder.stdin.write("acquire\\n"); holder.stdin.flush()
                holder.stdout.readline()
                activating.set()
                return finalize(self, *args, **kwargs)
            RebuildManager._finalize_rebuild = finalize_after_maintenance_is_taken
            manager = switch("edge", join=False)["manager"]
            worker = manager._active_thread
            activating.wait(60); time.sleep(0.5)
            served = threading.Event()
            def serve():
                engine.get_all(limit=5)
                engine.add("stored while activation waits for maintenance", sender="bob")
                served.set()
            threading.Thread(target=serve, daemon=True).start()
            responsive = served.wait(5)
            manager.cancel()
            worker.join(60)
            stopped = not worker.is_alive()
            holder.communicate("release\\n", timeout=30)
            served.wait(30); engine.close(); drain()
            emit(responsive=responsive, stopped=stopped, outcome=manager._last_outcome, state=state())
        """ % HOLD_MAINTENANCE)
        self.assertTrue(result["responsive"])
        self.assertTrue(result["stopped"])
        self.assertNotEqual(result["outcome"], "activation_pending")
        self.assertIsNone(result["state"]["selected"])
        self.assertEqual(result["state"]["marker"], "failed")
        self.assertEqual(result["state"]["status_rows"], [[1, "failed"]])

    def test_add_racing_the_activation_commit_is_retried_and_kept(self):
        (result,) = run_scenario("""
            import subprocess
            seed(20)
            stub_models(delay=0.02)
            process = subprocess.Popen([sys.executable, "-c", PRELUDE_SOURCE + %r], stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True)
            assert process.stdout.readline().startswith("READY")
            outcome = switch("edge")["outcome"]
            out, err = process.communicate(timeout=120)
            lines = [json.loads(line[7:]) for line in out.splitlines() if line.startswith("RESULT ")]
            conn = connect()
            emit(outcome=outcome, child=lines[0] if lines else err[-2000:],
                 messages=conn.execute("SELECT count(*) FROM messages").fetchone()[0],
                 vector=conn.execute("SELECT count(*) FROM vec_messages_edge WHERE rowid=21").fetchone()[0])
        """ % ADD_ACROSS_COMMIT)
        self.assertEqual(result["outcome"], "complete")
        self.assertEqual(result["child"], {"added": 21})
        self.assertEqual((result["messages"], result["vector"]), (21, 1))

    def test_process_serving_a_retired_journal_returns_to_legacy(self):
        (result,) = run_scenario(steps("""
            import subprocess
            seed(10)
            stub_models()
            assert switch("edge")["outcome"] == "complete"
            process = subprocess.Popen([sys.executable, "-c", PRELUDE_SOURCE + %r], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            assert process.stdout.readline().startswith("OPEN")
            process.stdin.write("go\\n"); process.stdin.flush()
            process.stdout.readline()
            # 0.7.6.1 switches back to base with exactly these database and config effects.
            conn = connect()
            mid = conn.execute("INSERT INTO messages(content,sender) VALUES ('added by 0.7.6.1 after its switch','a')").lastrowid
            conn.execute("INSERT INTO vec_messages_basepro(rowid,embedding) VALUES (?,?)", (mid, vec(mid)))
            conn.execute("INSERT INTO vec_messages_sep_basepro(rowid,embedding) VALUES (?,?)", (mid, vec(-mid)))
            conn.execute("INSERT OR REPLACE INTO vector_cache_registry(tier_group,vec_table,sep_table,last_embedded_id,"
                         "vector_count,model_name,embedding_dim,last_updated,created) VALUES ('basepro','vec_messages_basepro',"
                         "'vec_messages_sep_basepro',?,?,'Qwen3-Embedding-0.6B',256,0,0)", (mid, mid))
            conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES ('embed_model','qwen3_256')")
            conn.commit(); conn.close()
            config = json.loads(CONFIG.read_text()); config["tier"] = "base"; CONFIG.write_text(json.dumps(config))
        """ % LONG_RUNNING, run_child(RETIRE_BY_OPENING), """
            out, err = process.communicate("go\\n", timeout=120)
            lines = [json.loads(line[7:]) for line in out.splitlines() if line.startswith("RESULT ")]
            emit(retired=child, state=state(), other=lines[0] if lines else err[-2000:])
        """))
        self.assertEqual(result["retired"], {"opened": True})
        self.assertIsNone(result["state"]["selected"])
        other = result["other"]
        self.assertIsInstance(other, dict, other)
        self.assertEqual((other["window"], other["get_all"], other["add"]), (11, 12, 13))
        self.assertIsInstance(other["search"], list, other["search"])


# A long-running process pinned to its startup tier (the MCP server pins
# TRUEMEMORY_EMBED_MODEL) that switches to base itself, then waits.
STALE_ENV_PROCESS = """
os.environ["TRUEMEMORY_EMBED_MODEL"] = "edge"
stub_models()
import model2vec, sentence_transformers
model2vec.StaticModel.from_pretrained = staticmethod(lambda *args, **kwargs: FakeEncoder())
sentence_transformers.SentenceTransformer = lambda *args, **kwargs: FakeEncoder()
from truememory.engine import TrueMemoryEngine
from truememory.tier_switch.runtime import engine_serving_operation
switched = switch("base")["outcome"]
engine = TrueMemoryEngine(DB)
engine.add("written while base is selected", sender="alice")
drain()
print("READY", switched, flush=True)
sys.stdin.readline()
out = {}
try:
    with engine_serving_operation(engine) as operation:
        out["served"] = [operation.tier, list(operation.tables)]
except Exception as error:
    out["served"] = type(error).__name__
try:
    out["add"] = engine.add("written after the selection was retired elsewhere", sender="alice")["id"]
except Exception as error:
    out["add"] = type(error).__name__
out["env"] = os.environ["TRUEMEMORY_EMBED_MODEL"]
engine.close(); drain()
emit(**out)
"""

NAN_ENCODER = """
_finite_encode = FakeEncoder.encode
def _encode_with_one_nan(self, texts, **kwargs):
    out = _finite_encode(self, texts, **kwargs)
    for i, text in enumerate([texts] if isinstance(texts, str) else list(texts)):
        if "POISON" in text:
            out[i][0] = float("nan")  # The fp16 NaN class tracked by issues 465/485.
    return out
FakeEncoder.encode = _encode_with_one_nan
"""

ADD_POISON = NAN_ENCODER + """
stub_models()
from truememory.engine import TrueMemoryEngine
engine = TrueMemoryEngine(DB)
poison = engine.add("POISON input that the model turns into NaN", sender="alice")["id"]
engine.close(); drain()
emit(poison=poison)
"""

ADDS_AFTER_POISON = NAN_ENCODER + """
stub_models()
from truememory import vector_search
from truememory.engine import TrueMemoryEngine
encoded = [0]
_encode = vector_search._encode_with_mps_fallback
def counting(model, texts, **kwargs):
    encoded[0] += len(texts)
    return _encode(model, texts, **kwargs)
vector_search._encode_with_mps_fallback = counting
engine = TrueMemoryEngine(DB)
vectorless = []
for i in range(4):
    engine.add(f"healthy add {i}", sender="alice")
    vectorless.append(engine.get_vector_coverage()["vectorless"])
coverage = engine.get_vector_coverage()
engine.close(); drain()
emit(vectorless=vectorless, encoded=encoded[0], coverage=coverage)
"""

ADD_RACING_THE_BACKFILL = """
stub_models()
from truememory import vector_search
from truememory.engine import TrueMemoryEngine
raced = []
_encode = vector_search._encode_with_mps_fallback
def encode_while_another_process_adds(model, texts, **kwargs):
    if len(texts) > 1 and not raced:
        # A process without models commits a vectorless add during this pass.
        other = sqlite3.connect(DB)
        raced.append(other.execute("INSERT INTO messages(content,sender,recipient,timestamp) "
                                   "VALUES ('raced vectorless add','bob','','')").lastrowid)
        vector_search.note_vectorless(other, raced[0])
        other.commit(); other.close()
    return _encode(model, texts, **kwargs)
vector_search._encode_with_mps_fallback = encode_while_another_process_adds
engine = TrueMemoryEngine(DB)
engine.add("healthy add runs a backfill pass", sender="alice")
after_pass = engine.get_vector_coverage()
engine.add("second healthy add", sender="alice")
after_second = engine.get_vector_coverage()
engine.close(); drain()
emit(raced=raced, after_pass=after_pass, after_second=after_second)
"""

SEARCH_ONLY = """
stub_models()
from truememory.engine import TrueMemoryEngine
engine = TrueMemoryEngine(DB)
hits = engine.search("captured while models were down", limit=3)
started = time.monotonic()  # The background pass runs; close() would stop it.
coverage = {}
while coverage.get("vectorless") != 0 and time.monotonic() - started < 30:
    time.sleep(0.05)
    coverage = engine.get_vector_coverage() or {}
engine.close(); drain()
emit(hits=len(hits), coverage=coverage)
"""

TARGET_IDENTITY_CHANGED = """
stub_models()
from truememory.embedding_target import EmbeddingTarget, EmbeddingTargetError
def changed(self):
    raise EmbeddingTargetError("Prepared target configuration changed; capture a new target")
EmbeddingTarget.check_configuration = changed  # A certified custom tier edited afterwards.
from truememory.engine import TrueMemoryEngine
engine = TrueMemoryEngine(DB)
out = {}
for name, call in (("search", lambda: engine.search("synthetic fact number 3", limit=3)),
                   ("add", lambda: engine.add("stored under a changed target identity", sender="alice"))):
    try:
        call()
        out[name] = "ok"
    except Exception as error:
        out[name] = type(error).__name__
engine.close(); drain()
emit(**out)
"""

# Consolidation relaunching on pending adds: 2 s runs back to back. A relaunch
# re-acquires at once; that is modelled without any release gap (no timing
# luck either way), releasing only when the product's acquisition rule would
# refuse the relaunch. Releases older code never makes, so it can only starve.
BACK_TO_BACK_MAINTENANCE = """
import truememory.maintenance as maintenance
path = maintenance.canonical_database_path(DB)
refuses_relaunch = getattr(maintenance, "_wanted_by_other_locked", None)
end = time.monotonic() + 60
fd, announced = None, False
while time.monotonic() < end:
    if fd is None:
        fd = maintenance._acquire_owner(path)
        if fd is None:
            time.sleep(0.001)
            continue
        if not announced:
            announced = True
            print("HELD", flush=True)
    time.sleep(2.0)
    with maintenance._registry_lock:
        refused = refuses_relaunch is not None and refuses_relaunch(path)
    if refused:
        maintenance._release_owner(fd)
        fd = None
"""

FRESH_OPEN = """
stub_models()
from truememory.engine import TrueMemoryEngine
started = time.monotonic()
try:
    result = len(TrueMemoryEngine(DB).get_all(limit=100))
except Exception as error:
    result = type(error).__name__
emit(result=result, seconds=round(time.monotonic() - started, 1))
"""


SLOW_BACKGROUND_PASS = """
stub_models()
from truememory import vector_search
from truememory.engine import TrueMemoryEngine
_encode = vector_search._encode_with_mps_fallback
def slow_in_background(model, texts, **kwargs):
    if threading.current_thread().name == "truememory-vector-backfill":
        time.sleep(3)  # One long encode call, like Qwen3 on CPU.
    return _encode(model, texts, **kwargs)
vector_search._encode_with_mps_fallback = slow_in_background
engine = TrueMemoryEngine(DB)
engine.search("captured while models were down", limit=3)
time.sleep(0.5)
started = time.monotonic()
engine.close()
closed = round(time.monotonic() - started, 2)
time.sleep(4)  # The pass finishes its encode call, then must write nothing.
drain()
emit(closed=closed)
"""

CUSTOM_IDENTITY_CHANGED = """
os.environ["TRUEMEMORY_CUSTOM_EMBED_MODEL"] = "synthetic/embed-b"  # Edited after certification.
stub_models()
from truememory.engine import TrueMemoryEngine
engine = TrueMemoryEngine(DB)
listed = len(engine.get_all(limit=100))
try:
    engine.search("synthetic fact number 3", limit=3)
    search = "ok"
except Exception as error:
    search = type(error).__name__
report = engine.get_tier_identity()
engine.close(); drain()
emit(listed=listed, search=search, report=report)
"""

LONG_MAINTENANCE_OWNER = """
from truememory.maintenance import maintenance_owner
with maintenance_owner(DB):  # A long consolidation run.
    print("HELD", flush=True)
    time.sleep(60)
"""


class RetiredSelectionInPinnedProcess(unittest.TestCase):
    def run_retire(self, config_tier):
        (result,) = run_scenario(steps("""
            import subprocess
            from truememory.maintenance import maintenance_owner
            from truememory.tier_switch.activation import retire_incoherent_activation
            seed(15, group="edge")
            process = subprocess.Popen([sys.executable, "-c", PRELUDE_SOURCE + %r], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            ready = process.stdout.readline().strip()
            # 0.7.6.1 effects: its rebuild rewrites the registry row, and config names the tier.
            conn = connect()
            conn.execute("UPDATE vector_cache_registry SET model_name='Qwen3-Embedding-0.6B' WHERE tier_group='basepro'")
            conn.commit(); conn.close()
            config = json.loads(CONFIG.read_text()); config["tier"] = %r; CONFIG.write_text(json.dumps(config))
            with maintenance_owner(DB):  # As any open elsewhere does.
                conn = connect(); retired = retire_incoherent_activation(conn); conn.close()
            out, err = process.communicate("go\\n", timeout=120)
            lines = [json.loads(line[7:]) for line in out.splitlines() if line.startswith("RESULT ")]
            conn = connect()
            emit(ready=ready, retired=retired, process=lines[0] if lines else err[-2000:],
                 messages=conn.execute("SELECT count(*) FROM messages").fetchone()[0],
                 vectors=conn.execute("SELECT count(*) FROM vec_messages_basepro WHERE rowid=17").fetchone()[0])
        """ % (STALE_ENV_PROCESS, config_tier)), tier="edge")
        self.assertEqual((result["ready"], result["retired"]), ("READY complete", True))
        self.assertIsInstance(result["process"], dict, result["process"])
        return result

    def test_process_resets_to_the_stores_configured_tier(self):
        result = self.run_retire("pro")
        process = result["process"]
        self.assertEqual(process["served"], ["pro", ["vec_messages_basepro", "vec_messages_sep_basepro"]])
        self.assertEqual((process["add"], process["env"]), (17, "pro"))
        self.assertEqual((result["messages"], result["vectors"]), (17, 1))

    def test_process_refuses_a_configured_tier_whose_pair_is_stale(self):
        # Config names edge, but the edge pair predates the base selection.
        result = self.run_retire("edge")
        process = result["process"]
        self.assertEqual((process["served"], process["add"]), ("TierRuntimeError", "TierRuntimeError"))
        self.assertEqual(result["messages"], 16)


class BackfillRobustness(unittest.TestCase):
    def test_nonfinite_embedding_is_skipped_once_and_reported(self):
        (result,) = run_scenario(steps("""
            import subprocess
            seed(10)
            stub_models()
            assert switch("edge")["outcome"] == "complete"
        """, run_child(ADD_POISON), """
            poison = child
        """, run_child(ADD_WITHOUT_MODELS), """
            offline = child
        """, run_child(ADDS_AFTER_POISON), """
            conn = connect()
            emit(poison=poison, offline=offline, online=child,
                 cursor=conn.execute("SELECT value FROM metadata WHERE key='vector_backfill_from'").fetchone())
        """))
        self.assertEqual(result["poison"], {"poison": 11})
        self.assertEqual(result["offline"].get("ids"), [12, 13, 14], result["offline"])
        online = result["online"]
        self.assertEqual(online.get("vectorless"), [1, 1, 1, 1], online)
        # 4 adds x 2 texts, plus one pass over 4 rows x 2 texts: no row is re-encoded.
        self.assertEqual(online["encoded"], 16)
        self.assertEqual((online["coverage"]["skipped_nonfinite"], online["coverage"]["first_vectorless_id"]), (1, None))
        self.assertIsNone(result["cursor"])

    def test_vectorless_add_during_a_pass_keeps_its_cursor(self):
        (result,) = run_scenario(steps("""
            import subprocess
            seed(10)
            stub_models()
            assert switch("edge")["outcome"] == "complete"
        """, run_child(ADD_WITHOUT_MODELS), run_child(ADD_RACING_THE_BACKFILL), """
            emit(child=child)
        """))
        child = result["child"]
        self.assertEqual(child.get("raced"), [15], child)
        self.assertEqual((child["after_pass"]["vectorless"], child["after_pass"]["backfill_from"]), (1, "15"))
        self.assertEqual((child["after_second"]["vectorless"], child["after_second"]["backfill_from"]), (0, None))

    def test_search_backfills_an_idle_store(self):
        (result,) = run_scenario(steps("""
            import subprocess
            seed(10)
            stub_models()
            assert switch("edge")["outcome"] == "complete"
        """, run_child(ADD_WITHOUT_MODELS), run_child(SEARCH_ONLY), """
            conn = connect()
            emit(child=child, cursor=conn.execute("SELECT value FROM metadata WHERE key='vector_backfill_from'").fetchone(),
                 vectors=[conn.execute(f"SELECT count(*) FROM {table} WHERE rowid IN (11, 12, 13)").fetchone()[0]
                          for table in ("vec_messages_edge", "vec_messages_sep_edge")])
        """))
        self.assertEqual(result["child"].get("hits"), 3, result["child"])
        self.assertEqual(result["vectors"], [3, 3], result["child"])
        self.assertIsNone(result["cursor"])

    def test_target_identity_error_is_surfaced_not_degraded(self):
        (result,) = run_scenario(steps("""
            import subprocess
            seed(10)
            stub_models()
            assert switch("edge")["outcome"] == "complete"
        """, run_child(TARGET_IDENTITY_CHANGED), """
            conn = connect()
            emit(child=child, messages=conn.execute("SELECT count(*) FROM messages").fetchone()[0],
                 cursor=conn.execute("SELECT value FROM metadata WHERE key='vector_backfill_from'").fetchone())
        """))
        self.assertEqual(result["child"], {"search": "EmbeddingTargetError", "add": "EmbeddingTargetError"})
        self.assertEqual((result["messages"], result["cursor"]), (10, None))


    def test_close_stops_a_background_pass_within_a_second(self):
        (result,) = run_scenario(steps("""
            import subprocess
            seed(10)
            stub_models()
            assert switch("edge")["outcome"] == "complete"
        """, run_child(ADD_WITHOUT_MODELS), run_child(SLOW_BACKGROUND_PASS), """
            conn = connect()
            emit(child=child, cursor=conn.execute("SELECT value FROM metadata WHERE key='vector_backfill_from'").fetchone(),
                 vectors=conn.execute("SELECT count(*) FROM vec_messages_edge WHERE rowid IN (11, 12, 13)").fetchone()[0])
        """))
        self.assertLess(result["child"].get("closed", 99), 1.5, result["child"])
        self.assertEqual((result["vectors"], result["cursor"]), (0, ["11"]))


class TierIdentityStatus(unittest.TestCase):
    def test_status_names_a_custom_tier_identity_mismatch(self):
        (result,) = run_scenario(steps("""
            import subprocess
            seed(10)
            os.environ.update(TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD="1", TRUEMEMORY_CUSTOM_EMBED_MODEL="synthetic/embed-a",
                              TRUEMEMORY_CUSTOM_EMBED_DIM="256")
            stub_models()
            assert switch("custom")["outcome"] == "complete"
        """, run_child(CUSTOM_IDENTITY_CHANGED), """
            emit(child=child)
        """))
        child = result["child"]
        self.assertEqual((child.get("listed"), child.get("search")), (10, "EmbeddingTargetError"), child)
        report = child["report"]
        self.assertEqual((report["status"], report["error"]), ("degraded", "EmbeddingTargetError"))
        self.assertEqual(report["selected"], {"tier": "custom", "model_id": "synthetic/embed-a", "dimension": 256})
        self.assertEqual(report["configured"], {"tier": "custom", "model_id": "synthetic/embed-b", "dimension": 256})
        self.assertIn("TRUEMEMORY_CUSTOM_EMBED_MODEL=synthetic/embed-a", report["remedy"])


class ActivationFairness(unittest.TestCase):
    def test_activation_is_not_starved_by_back_to_back_maintenance(self):
        (result,) = run_scenario("""
            import subprocess
            seed(20)
            stub_models(delay=0.02)
            from truememory.tier_switch.manager import RebuildManager
            holding, activating = threading.Event(), threading.Event()
            finalize = RebuildManager._finalize_rebuild
            def finalize_once_maintenance_is_busy(self, *args, **kwargs):
                holding.wait(60)  # Deterministic: the holder owns maintenance before activation.
                activating.set()
                return finalize(self, *args, **kwargs)
            RebuildManager._finalize_rebuild = finalize_once_maintenance_is_busy
            manager = switch("edge", join=False)["manager"]
            worker = manager._active_thread
            holder = subprocess.Popen([sys.executable, "-c", PRELUDE_SOURCE + %r], stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True)
            assert holder.stdout.readline().startswith("HELD")
            holding.set()
            activating.wait(60)
            started = time.monotonic()
            worker.join(12)
            finished, seconds = not worker.is_alive(), round(time.monotonic() - started, 1)
            if not finished:
                manager.cancel(); worker.join(30)
            holder.kill(); holder.wait()
            emit(finished=finished, seconds=seconds, outcome=manager._last_outcome, state=state())
        """ % BACK_TO_BACK_MAINTENANCE)
        self.assertTrue(result["finished"], result)
        self.assertLess(result["seconds"], 12)
        self.assertEqual(result["outcome"], "complete")
        self.assertEqual(result["state"]["selected"], "edge")

    def test_reader_drain_never_outlasts_fresh_opens_elsewhere(self):
        # A 12 s reader in the worker's process (a long tool call) at activation.
        (result,) = run_scenario("""
            import subprocess
            seed(30)
            stub_models(delay=0.02)
            from truememory.engine import TrueMemoryEngine
            from truememory.tier_switch.manager import RebuildManager
            from truememory.tier_switch.runtime import engine_serving_operation
            engine = TrueMemoryEngine(DB); engine.get_all(limit=1); drain()
            held = threading.Event()
            def long_reader():
                with engine_serving_operation(engine):
                    held.set(); time.sleep(12)
            finalize = RebuildManager._finalize_rebuild
            def finalize_during_a_long_read(self, *args, **kwargs):
                threading.Thread(target=long_reader, daemon=True).start()
                held.wait(10)
                return finalize(self, *args, **kwargs)
            RebuildManager._finalize_rebuild = finalize_during_a_long_read
            manager = switch("edge", join=False)["manager"]
            worker = manager._active_thread
            held.wait(60); time.sleep(0.5)
            local = {}
            def local_read():
                started = time.monotonic()
                try:
                    local["result"] = len(engine.get_all(limit=100))
                except Exception as error:
                    local["result"] = type(error).__name__
                local["seconds"] = round(time.monotonic() - started, 1)
            reader = threading.Thread(target=local_read); reader.start()
            fresh = subprocess.run([sys.executable, "-c", PRELUDE_SOURCE + %r], capture_output=True, text=True)
            reader.join(30); worker.join(60)
            lines = [json.loads(line[7:]) for line in fresh.stdout.splitlines() if line.startswith("RESULT ")]
            engine.close(); drain()
            emit(fresh=lines[0] if lines else fresh.stderr[-2000:], local=local, outcome=manager._last_outcome)
        """ % FRESH_OPEN)
        self.assertEqual(result["fresh"].get("result"), 30, result["fresh"])
        self.assertEqual(result["local"].get("result"), 30, result["local"])
        self.assertLess(result["local"]["seconds"], 8)
        self.assertEqual(result["outcome"], "complete")



    def test_fresh_open_keeps_waiting_through_a_drain_gap(self):
        # A fresh open finds maintenance busy during the reader drain and checks
        # for an activation exactly when the drain hands maintenance back. The
        # marker must still be held then (6ea1635), or the open fails.
        (result,) = run_scenario("""
            seed(30)
            stub_models(delay=0.02)
            from truememory.engine import TrueMemoryEngine
            from truememory.tier_switch import job, serving
            from truememory.tier_switch import manager as manager_module
            from truememory.tier_switch.manager import RebuildManager
            from truememory.tier_switch.runtime import engine_serving_operation
            engine = TrueMemoryEngine(DB); engine.get_all(limit=1); drain()
            held, in_gap, opened = threading.Event(), threading.Event(), threading.Event()
            worker = [None]
            def reader():
                with engine_serving_operation(engine):
                    held.set(); time.sleep(12)  # Outlasts one drain attempt, so a gap follows.
            finalize = RebuildManager._finalize_rebuild
            def finalize_during_a_long_read(self, *args, **kwargs):
                threading.Thread(target=reader, daemon=True).start()
                held.wait(10)
                return finalize(self, *args, **kwargs)
            RebuildManager._finalize_rebuild = finalize_during_a_long_read
            class GapClock:
                def __getattr__(self, name):
                    return getattr(time, name)
                def sleep(self, seconds):
                    if seconds == 0.2 and threading.current_thread() is worker[0]:
                        in_gap.set()  # Maintenance is released between drain attempts.
                        opened.wait(10)
                    time.sleep(seconds)
            manager_module.time = GapClock()
            in_progress = job.tier_activation_in_progress
            def checked_in_the_gap(db_path):
                in_gap.wait(30)
                return in_progress(db_path)
            job.tier_activation_in_progress = checked_in_the_gap
            manager = switch("edge", join=False)["manager"]
            worker[0] = manager._active_thread
            held.wait(60)
            started = time.monotonic()
            while not serving._DEFAULT_GATE._waiting_writers and time.monotonic() - started < 30:
                time.sleep(0.01)  # The activation owns maintenance and drains the reader.
            fresh = {}
            def fresh_open():
                begun = time.monotonic()
                try:
                    fresh["rows"] = len(TrueMemoryEngine(DB).get_all(limit=100))
                except Exception as error:
                    fresh["rows"] = type(error).__name__
                fresh["seconds"] = round(time.monotonic() - begun, 1)
                opened.set()
            opener = threading.Thread(target=fresh_open); opener.start(); opener.join(60)
            worker[0].join(60)
            engine.close(); drain()
            emit(fresh=fresh, outcome=manager._last_outcome)
        """)
        self.assertEqual(result["fresh"].get("rows"), 30, result["fresh"])
        self.assertEqual(result["outcome"], "complete")

    def test_reads_queued_behind_the_activation_writer_are_retried(self):
        (result,) = run_scenario("""
            seed(30)
            stub_models(delay=0.02)
            from truememory.engine import TrueMemoryEngine
            from truememory.tier_switch import serving
            from truememory.tier_switch.manager import RebuildManager
            from truememory.tier_switch.runtime import engine_serving_operation
            # One engine (connection) per queued call: this is about admission,
            # not about threads sharing one sqlite connection.
            engine, *readers = [TrueMemoryEngine(DB) for _ in range(5)]
            for each in (engine, *readers):
                each.get_all(limit=1)
            drain()
            held = threading.Event()
            def reader():
                with engine_serving_operation(engine):
                    held.set(); time.sleep(2)
            finalize = RebuildManager._finalize_rebuild
            def finalize_during_a_read(self, *args, **kwargs):
                threading.Thread(target=reader, daemon=True).start()
                held.wait(10)
                return finalize(self, *args, **kwargs)
            RebuildManager._finalize_rebuild = finalize_during_a_read
            manager = switch("edge", join=False)["manager"]
            worker = manager._active_thread
            held.wait(60)
            started = time.monotonic()
            while not serving._DEFAULT_GATE._waiting_writers and time.monotonic() - started < 30:
                time.sleep(0.005)
            queued = bool(serving._DEFAULT_GATE._waiting_writers)
            results = {}
            def call(name, function):
                try:
                    results[name] = function()
                except Exception as error:
                    results[name] = type(error).__name__
            calls = {"get_all": lambda: len(readers[0].get_all(limit=100)), "get": lambda: readers[1].get(1)["id"],
                     "get_stats": lambda: readers[2].get_stats()["message_count"],
                     "search_simple": lambda: len(readers[3].search_simple("synthetic fact", limit=3))}
            threads = [threading.Thread(target=call, args=item) for item in calls.items()]
            [thread.start() for thread in threads]; [thread.join(60) for thread in threads]
            worker.join(60)
            for each in (engine, *readers):
                each.close()
            drain()
            emit(queued=queued, results=results, outcome=manager._last_outcome)
        """)
        self.assertTrue(result["queued"])
        self.assertEqual(result["results"], {"get_all": 30, "get": 1, "get_stats": 30, "search_simple": 3})
        self.assertEqual(result["outcome"], "complete")

    def test_fresh_open_fails_at_once_behind_a_long_owner_while_activation_waits(self):
        (result,) = run_scenario("""
            import subprocess
            seed(20)
            stub_models(delay=0.02)
            from contextlib import contextmanager
            from truememory import maintenance
            from truememory.tier_switch.manager import RebuildManager
            holding, waiting = threading.Event(), threading.Event()
            finalize = RebuildManager._finalize_rebuild
            def finalize_behind_the_owner(self, *args, **kwargs):
                holding.wait(60)
                return finalize(self, *args, **kwargs)
            RebuildManager._finalize_rebuild = finalize_behind_the_owner
            wanted = maintenance.maintenance_wanted
            @contextmanager
            def noted_wait(path):
                with wanted(path) as asked:
                    waiting.set()  # The activation now waits behind the owner.
                    yield asked
            maintenance.maintenance_wanted = noted_wait
            manager = switch("edge", join=False)["manager"]
            worker = manager._active_thread
            owner = subprocess.Popen([sys.executable, "-c", PRELUDE_SOURCE + %r], stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
            assert owner.stdout.readline().startswith("HELD")
            holding.set()
            assert waiting.wait(30)
            fresh = subprocess.run([sys.executable, "-c", PRELUDE_SOURCE + %r], capture_output=True, text=True)
            owner.kill(); owner.wait()
            worker.join(60)
            lines = [json.loads(line[7:]) for line in fresh.stdout.splitlines() if line.startswith("RESULT ")]
            emit(fresh=lines[0] if lines else fresh.stderr[-2000:], outcome=manager._last_outcome)
        """ % (LONG_MAINTENANCE_OWNER, FRESH_OPEN))
        # Unchanged behaviour behind a long owner: refuse at once, not after 10 s.
        self.assertEqual(result["fresh"].get("result"), "MaintenanceBusyError", result["fresh"])
        self.assertLess(result["fresh"]["seconds"], 2)
        self.assertEqual(result["outcome"], "complete")


class ProcessProjectionScope(unittest.TestCase):
    def test_other_stores_are_served_beside_a_controlled_store(self):
        # A library process keeps a controlled store and opens others: SQL-only
        # calls and a store in the same embedding space need no reload; a store
        # in another space is served by reloading, logged once.
        (result,) = run_scenario("""
            seed(10)
            stub_models()
            import model2vec, sentence_transformers
            from truememory import vector_search
            from truememory.tier_switch import runtime
            loads = {"legacy": 0, "frozen": 0}
            def legacy_load(*args, **kwargs):
                loads["legacy"] += 1
                return FakeEncoder()
            model2vec.StaticModel.from_pretrained = staticmethod(legacy_load)
            sentence_transformers.SentenceTransformer = legacy_load
            frozen = vector_search._load_frozen_embedding_target
            def frozen_load(target, timeout=None):
                loads["frozen"] += 1
                return frozen(target, timeout)
            vector_search._load_frozen_embedding_target = frozen_load
            notes = []
            class Notes(logging.Handler):
                def emit(self, record):
                    notes.append(record.getMessage())
            runtime_log = logging.getLogger("truememory.tier_switch.runtime")
            runtime_log.addHandler(Notes()); runtime_log.setLevel(logging.INFO)
            assert switch("edge")["outcome"] == "complete"
            same, other = str(HOME / "same.db"), str(HOME / "other.db")
            seed(5, group="edge", path=same)
            seed(5, group="basepro", path=other)
            from truememory.engine import TrueMemoryEngine
            controlled, shared, foreign = (TrueMemoryEngine(path) for path in (DB, same, other))
            controlled.search("synthetic fact", limit=3)
            def counted(calls):
                before = dict(loads)
                values = [call() for call in calls]
                return values, {name: loads[name] - before[name] for name in loads}
            out = {"sql_only": counted([lambda: len(foreign.get_all(limit=100)), lambda: foreign.delete(2),
                                        lambda: len(shared.get_all(limit=100))])}
            out["same_space"] = counted([lambda: len(shared.search("synthetic fact", limit=3)),
                                         lambda: shared.add("same space", sender="alice")["id"],
                                         lambda: len(controlled.search("synthetic fact", limit=3))] * 3)
            out["other_space"] = counted([lambda: len(foreign.search("synthetic fact", limit=3)),
                                          lambda: foreign.add("other space", sender="alice")["id"],
                                          lambda: len(controlled.search("synthetic fact", limit=3)),
                                          lambda: controlled.add("controlled", sender="alice")["id"]] * 2)
            for engine in (controlled, shared, foreign):
                engine.close()
            drain()
            vectors = {}
            for name, path, group, rows in (("same", same, "edge", (6, 7, 8)), ("other", other, "basepro", (2, 6, 7)),
                                            ("controlled", DB, "edge", (11, 12))):
                conn = sqlite3.connect(path); load_vec(conn)
                vectors[name] = [conn.execute(f"SELECT count(*) FROM vec_messages_{group} WHERE rowid=?", (row,)).fetchone()[0]
                                 for row in rows]
                conn.close()
            emit(out=out, vectors=vectors, reload_notes=sum("another embedding model" in note for note in notes))
        """)
        out = result["out"]
        self.assertEqual(out["sql_only"], [[5, True, 5], {"legacy": 0, "frozen": 0}])
        self.assertEqual(out["same_space"][0], [3, 6, 3, 3, 7, 3, 3, 8, 3])
        self.assertEqual(out["same_space"][1], {"legacy": 0, "frozen": 0})
        self.assertEqual(out["other_space"][0], [3, 6, 3, 11, 3, 7, 3, 12])
        self.assertEqual(out["other_space"][1], {"legacy": 2, "frozen": 2})  # One reload per switch.
        # The deleted row's vectors are gone from its own pair; new rows have vectors in their own space.
        self.assertEqual(result["vectors"], {"same": [1, 1, 1], "other": [0, 1, 1], "controlled": [1, 1]})
        self.assertEqual(result["reload_notes"], 1)


class SharedEngineThreads(unittest.TestCase):
    def test_threads_sharing_one_engine_keep_its_statements_intact(self):
        # sqlite3 shares cached statements per connection: two threads stepping
        # one engine's connection raised IndexError, SystemError or "no such table".
        (result,) = run_scenario("""
            import collections
            import model2vec
            model2vec.StaticModel.from_pretrained = staticmethod(lambda *args, **kwargs: FakeEncoder())
            from truememory.engine import TrueMemoryEngine
            engine = TrueMemoryEngine(DB)
            for i in range(30):
                engine.add(f"seed memory {i}", sender="alice")
            ops, errors = collections.Counter(), collections.Counter()
            stop = time.monotonic() + 6
            def loop(kind, call):
                n = 0
                while time.monotonic() < stop:
                    n += 1
                    try:
                        call(n)
                        ops[kind] += 1
                    except Exception as error:
                        errors[f"{kind}: {type(error).__name__}: {str(error)[:80]}"] += 1
            calls = {"get": lambda n: engine.get(random.randint(1, 30)),
                     "get_all": lambda n: engine.get_all(limit=20),
                     "search": lambda n: engine.search("seed memory", limit=5),
                     "add": lambda n: engine.add(f"concurrent memory {n}", sender="alice"),
                     "delete": lambda n: engine.delete(random.randint(31, 2000))}
            threads = [threading.Thread(target=loop, args=item) for item in calls.items()]
            [thread.start() for thread in threads]
            [thread.join() for thread in threads]
            engine.close()
            emit(ops=sorted(kind for kind, count in ops.items() if count), errors=errors)
        """, tier="edge")
        self.assertEqual(result["errors"], {})
        self.assertEqual(result["ops"], ["add", "delete", "get", "get_all", "search"])

    def test_a_search_frees_the_engine_during_model_work(self):
        # Review v5 L-v5-1: the whole search, query encoding included, ran under
        # the engine lock, so a get waited up to 6.2 s behind back-to-back searches.
        (result,) = run_scenario("""
            import model2vec
            slow = {"on": False, "encodes": 0}
            class SlowEncoder(FakeEncoder):
                def encode(self, texts, **kwargs):
                    if slow["on"]:
                        slow["encodes"] += 1
                        time.sleep(0.25)
                    return super().encode(texts, **kwargs)
            model2vec.StaticModel.from_pretrained = staticmethod(lambda *args, **kwargs: SlowEncoder())
            stub_models()
            from truememory.engine import TrueMemoryEngine
            engine = TrueMemoryEngine(DB)
            for i in range(10):
                engine.add(f"memory number {i} about the garden", sender="alice")
            drain()
            slow["on"] = True
            stop = time.monotonic() + 3
            searches = []
            def searcher():
                while time.monotonic() < stop:  # A new query each time: nothing is cached.
                    searches.append(len(engine.search(f"garden memory {len(searches)}", limit=5)))
            thread = threading.Thread(target=searcher)
            thread.start()
            latencies = []
            while time.monotonic() < stop:
                started = time.perf_counter()
                engine.get(1)
                latencies.append(time.perf_counter() - started)
                time.sleep(0.01)
            thread.join()
            engine.close()
            emit(gets=len(latencies), worst=max(latencies), searches=len(searches), encodes=slow["encodes"])
        """, tier="edge")
        self.assertGreaterEqual(result["encodes"], 6)  # Searches kept encoding throughout.
        self.assertGreaterEqual(result["gets"], 50)
        self.assertLess(result["worst"], 0.5)

    def test_a_write_waiting_on_another_connection_does_not_freeze_the_process(self):
        # The waiting writer holds SQLite's connection mutex; a second thread that
        # read the connection then held the GIL until the wait failed, so the
        # other connection could never commit and the write failed as locked.
        (result,) = run_scenario("""
            import model2vec
            model2vec.StaticModel.from_pretrained = staticmethod(lambda *args, **kwargs: FakeEncoder())
            from truememory.engine import TrueMemoryEngine
            engine = TrueMemoryEngine(DB)
            engine.add("first memory", sender="alice")
            drain()
            engine.conn.execute("PRAGMA busy_timeout=4000")
            other = sqlite3.connect(DB, isolation_level=None, check_same_thread=False)
            other.execute("BEGIN IMMEDIATE")
            results = {}
            def call(name, body):
                try:
                    results[name] = body()
                except Exception as error:
                    results[name] = f"{type(error).__name__}: {error}"
            writer = threading.Thread(target=call, args=("add", lambda: engine.add("waits for the commit", sender="alice")["id"]))
            writer.start()
            time.sleep(0.5)
            started = time.monotonic()
            reader = threading.Thread(target=call, args=("get", lambda: engine.get(1)["content"]))
            reader.start()
            time.sleep(0.2)
            other.execute("COMMIT")
            committed = time.monotonic() - started
            writer.join(10); reader.join(10)
            engine.close(); other.close()
            emit(results=results, committed_within_a_second=committed < 1.0)
        """, tier="edge")
        self.assertEqual(result["results"], {"add": 2, "get": "first memory"})
        self.assertTrue(result["committed_within_a_second"])



class SwitchStaging(unittest.TestCase):
    def test_switch_waits_a_bounded_time_for_maintenance_elsewhere(self):
        # Another process's consolidation pass refused the switch at once; it now
        # waits for the pass (new owners yield), and names the cause past the bound.
        (result,) = run_scenario("""
            import subprocess
            seed(20)
            stub_models()
            from truememory.tier_switch import manager as manager_module
            holder = subprocess.Popen([sys.executable, "-c", PRELUDE_SOURCE + %r], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True)
            holder.stdin.write("acquire\\n"); holder.stdin.flush()
            holder.stdout.readline()
            manager_module._STAGING_MAINTENANCE_SECONDS = 1.0
            bounded = switch("edge")
            manager_module._STAGING_MAINTENANCE_SECONDS = 60.0
            def release():
                time.sleep(3)
                holder.communicate("release\\n", timeout=30)
            releaser = threading.Thread(target=release)
            releaser.start()
            started = time.monotonic()
            waited = switch("edge")
            seconds = time.monotonic() - started
            releaser.join(); drain()
            emit(bounded=bounded, outcome=waited.get("outcome", waited.get("error")), waited=seconds >= 2.5, state=state())
        """ % HOLD_MAINTENANCE)
        self.assertEqual(result["bounded"]["error"], "MaintenanceBusyError")
        self.assertIn("retry the tier switch", result["bounded"]["message"])
        self.assertEqual(result["outcome"], "complete")
        self.assertTrue(result["waited"])
        self.assertEqual(result["state"]["selected"], "edge")

    def test_cancel_of_a_durable_marker_waits_for_maintenance_elsewhere(self):
        # Cancelling an interrupted job failed at once while another process ran
        # a consolidation pass; it now waits a bounded time like staging does.
        (result,) = run_scenario("""
            import subprocess
            seed(10)
            from truememory.embedding_target import EmbeddingTarget
            from truememory.rebuild_source import ensure_rebuild_tracking
            from truememory.tier_switch.job import select_tier_job
            from truememory.tier_switch.manager import RebuildManager
            conn = sqlite3.connect(DB)
            ensure_rebuild_tracking(conn)
            job = select_tier_job(conn, EmbeddingTarget.capture("edge"))
            conn.close()
            holder = subprocess.Popen([sys.executable, "-c", PRELUDE_SOURCE + %r], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True)
            holder.stdin.write("acquire\\n"); holder.stdin.flush()
            holder.stdout.readline()
            def release():
                time.sleep(2)
                holder.communicate("release\\n", timeout=30)
            releaser = threading.Thread(target=release)
            releaser.start()
            started = time.monotonic()
            try:
                outcome = RebuildManager().cancel(7, expected_job_id=job.job_id, db_path=Path(DB))["status"]
            except Exception as error:
                outcome = type(error).__name__
            waited = time.monotonic() - started
            releaser.join()
            emit(outcome=outcome, waited=waited >= 1.5, marker=state()["marker"])
        """ % HOLD_MAINTENANCE)
        self.assertEqual(result, {"outcome": "cancelled", "waited": True, "marker": "cancelled"})

    def test_switch_without_cached_models_fails_before_any_work(self):
        # With downloads off, a missing reranker surfaced only at activation,
        # after every memory had been re-embedded for the target.
        (result,) = run_scenario("""
            seed(20)
            missing = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            stub_models(cached=lambda repository: repository != missing,
                        rerankers_available=lambda name: name != missing)
            encodes = []
            encode = FakeLease.encode
            def counted(self, texts, **kwargs):
                encodes.append(len(texts))
                return encode(self, texts, **kwargs)
            FakeLease.encode = counted
            refused = switch("edge")
            emit(error=refused.get("error"), message=refused.get("message"), encodes=len(encodes), state=state())
        """)
        self.assertEqual(result["error"], "TierSwitchUnsupportedError")
        self.assertIn("models that are not downloaded: cross-encoder/ms-marco-MiniLM-L-6-v2.", result["message"])
        self.assertEqual(result["encodes"], 0)
        self.assertEqual(result["state"], {"selected": None, "acknowledged": None, "marker": None, "status_rows": []})

    def test_status_row_reports_the_whole_store_and_live_throughput(self):
        # Background switches wrote no live metrics, and the final row counted
        # only the last catch-up round (2/2 after a two-row tail).
        (result,) = run_scenario("""
            seed(40)
            stub_models()
            encode = FakeLease.encode
            appended = []
            def append_once(self, texts, **kwargs):
                if not appended:
                    conn = sqlite3.connect(DB)
                    for i in range(2):
                        appended.append(conn.execute(
                            "INSERT INTO messages(content,sender,recipient,timestamp) VALUES (?,?,?,?)",
                            (f"appended during the rebuild {i}", "bob", "", "2026-01-02T00:00:00")).lastrowid)
                    conn.commit(); conn.close()
                return encode(self, texts, **kwargs)
            FakeLease.encode = append_once
            outcome = switch("edge")["outcome"]
            conn = sqlite3.connect(DB)
            row = conn.execute("SELECT status, processed_messages, total_messages, batch_size, "
                               "last_heartbeat > started_at FROM rebuild_status").fetchone()
            count = conn.execute("SELECT count(*) FROM messages").fetchone()[0]
            conn.close()
            emit(outcome=outcome, appended=len(appended), row=row, count=count)
        """)
        self.assertEqual(result["outcome"], "complete")
        self.assertEqual(result["appended"], 2)
        status, processed, total, batch_size, heartbeat = result["row"]
        self.assertEqual((status, processed, total), ("complete", 42, 42))
        self.assertEqual(result["count"], 42)
        self.assertGreaterEqual(batch_size, 1)
        self.assertEqual(heartbeat, 1)



class FailedSwitchRecovery(unittest.TestCase):
    def test_store_switches_back_after_a_failed_activation(self):
        # GPUBox D2: a killed edge job, its retry, then edge->base with a delete
        # mid-rebuild and a failed activation. Asking for edge again was refused
        # ("Policy projection requires an explicit same-space public target"),
        # so the store looked stuck on a failed switch.
        (result,) = run_scenario("""
            import signal, subprocess
            seed(40)
            code = PRELUDE_SOURCE + "\\nstub_models(delay=0.2)\\nswitch('edge', join=False)\\nprint('started', flush=True)\\ntime.sleep(60)\\n"
            child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
            child.stdout.readline(); time.sleep(1.0)
            os.kill(child.pid, signal.SIGKILL); child.wait()
            stub_models()
            retried = switch("edge")["outcome"]
            stub_models(rerankers_available=lambda name: "gte-reranker" not in name)
            from truememory.engine import TrueMemoryEngine
            engine = TrueMemoryEngine(DB)
            encode, deleted = FakeLease.encode, []
            def delete_once(self, texts, **kwargs):
                if not deleted:
                    deleted.append(engine.delete(7) and 7)
                return encode(self, texts, **kwargs)
            FakeLease.encode = delete_once
            failed = switch("base")["outcome"]
            FakeLease.encode = encode
            back = switch("edge")
            served = len(engine.search("synthetic fact", limit=3))
            stub_models()
            again = switch("base")["outcome"]
            engine.close()
            conn = connect()
            kept = conn.execute("SELECT count(*) FROM vec_messages_basepro WHERE rowid=7").fetchone()[0]
            check = conn.execute("PRAGMA integrity_check").fetchone()[0]
            conn.close()
            emit(retried=retried, failed=failed, back=back.get("outcome", back.get("message")), served=served,
                 again=again, deleted=deleted, kept=kept, check=check, state=state())
        """)
        self.assertEqual(result["retried"], "complete")
        self.assertEqual(result["failed"], "interrupted")
        self.assertEqual(result["back"], "complete")
        self.assertEqual(result["served"], 3)
        self.assertEqual(result["again"], "complete")
        self.assertEqual((result["deleted"], result["kept"], result["check"]), ([7], 0, "ok"))
        self.assertEqual(result["state"]["selected"], "base")



class MaintenanceUnderForegroundLoad(unittest.TestCase):
    def test_a_burst_of_adds_leaves_no_failed_layer(self):
        # Round v6 M6-1: the in-process maintenance thread raced 200 back-to-back
        # adds; its read-then-write transactions hit SQLITE_BUSY at once and
        # layers ended failed (on 6e1e459: 27 aborted passes in 44). Since round
        # v7 a build that sees a correction, or style input, change under it
        # defers to the next pass; nothing fails. The seeded memories keep the
        # first pass off HDBSCAN's minimum sample count, which fails on fewer.
        (result,) = run_scenario("""
            import collections
            seed(30, group="edge")
            import model2vec
            model2vec.StaticModel.from_pretrained = staticmethod(lambda *args, **kwargs: FakeEncoder())
            stub_models()
            from truememory import maintenance
            errors, outcomes = collections.Counter(), collections.Counter()
            run_layers = maintenance.run_layers
            def observed(*args, **kwargs):
                try:
                    results = run_layers(*args, **kwargs)
                except BaseException as error:
                    errors[type(error).__name__ + ": " + str(error)[:60]] += 1
                    raise
                for item in results:
                    if item[2] == "failed" or item[2] == "deferred" and item[5] not in (
                            "ModelBusy", "SourceChanged", "StyleSourceChanged", "WriterBusy"):
                        outcomes[f"{item[0]} {item[2]} {item[5]}"] += 1
                return results
            maintenance.run_layers = observed
            from truememory.engine import TrueMemoryEngine
            engine = TrueMemoryEngine(DB)
            words = "alice bob carol dave moved to austin started job at acme likes coffee hates rain said that".split()
            latencies = []
            for i in range(200):
                r = random.Random(i)
                started = time.perf_counter()
                engine.add(f"{r.choice(words).title()} {' '.join(r.choice(words) for _ in range(12))} {i}",
                           sender=r.choice(["alice", "bob", "carol"]))
                latencies.append(time.perf_counter() - started)
            engine._get_maintenance_coordinator().wait(120)
            rows = engine.conn.execute("SELECT layer, outcome FROM maintenance_layers "
                                       "WHERE outcome IN ('failed', 'running')").fetchall()
            engine.close()
            latencies.sort()
            emit(errors=errors, outcomes=outcomes, rows=rows, p95=latencies[int(len(latencies) * 0.95)])
        """, tier="edge")
        self.assertEqual((result["errors"], result["outcomes"], result["rows"]), ({}, {}, []))
        self.assertLess(result["p95"], 0.25)

    def test_a_build_slower_than_the_busy_timeout_never_fails_an_add(self):
        # Review v6 H-v6-1: each layer was built under SQLite's writer, so on a
        # 10k-memory store an add arriving during a build longer than the busy
        # timeout failed with "database is locked". Here the busy timeout is
        # 1 s and the surprise index takes 2.5 s to compute.
        (result,) = run_scenario("""
            import collections
            seed(30, group="edge")
            import model2vec
            model2vec.StaticModel.from_pretrained = staticmethod(lambda *args, **kwargs: FakeEncoder())
            stub_models()
            from truememory import engine as engine_module, maintenance, predictive, storage
            storage.DEFAULT_BUSY_TIMEOUT_MS = engine_module.DEFAULT_BUSY_TIMEOUT_MS = 1000
            compute, builds = predictive._compute_surprise_rows, []
            def slow(rows):
                builds.append(len(rows))
                time.sleep(2.5)
                return compute(rows)
            predictive._compute_surprise_rows = slow
            failed = collections.Counter()
            run_layers = maintenance.run_layers
            def observed(*args, **kwargs):
                results = run_layers(*args, **kwargs)
                failed.update(f"{item[0]} {item[5]}" for item in results if item[2] == "failed")
                return results
            maintenance.run_layers = observed
            engine = engine_module.TrueMemoryEngine(DB)
            latencies, errors = [], collections.Counter()
            for i in range(150):
                started = time.perf_counter()
                try:
                    engine.add(f"synthetic fact {i} about alice moving to austin", sender="alice")
                except Exception as error:
                    errors[type(error).__name__ + ": " + str(error)[:60]] += 1
                latencies.append(time.perf_counter() - started)
                time.sleep(0.03)
            engine._get_maintenance_coordinator().wait(120)
            surprise = engine.conn.execute("SELECT outcome FROM maintenance_layers WHERE layer='surprise'").fetchone()
            engine.close()
            latencies.sort()
            emit(errors=errors, failed=failed, builds=len(builds), surprise=surprise,
                 p95=latencies[int(len(latencies) * 0.95)])
        """, tier="edge")
        self.assertEqual((result["errors"], result["failed"]), ({}, {}))
        self.assertGreaterEqual(result["builds"], 2)
        self.assertEqual(result["surprise"], ["success"])
        self.assertLess(result["p95"], 0.25)


class ForegroundWriteRetry(unittest.TestCase):
    def test_writes_outlast_a_writer_held_past_the_busy_timeout_then_give_up(self):
        # Round v8 (review v6 L-v6-3): under saturated writers SQLite can starve
        # one writer past its busy timeout. A foreground write retries the busy
        # BEGIN, nothing written yet, for at most 30 s in all, then raises.
        (result,) = run_scenario("""
            seed(5, group="edge")
            import model2vec
            model2vec.StaticModel.from_pretrained = staticmethod(lambda *args, **kwargs: FakeEncoder())
            stub_models()
            from truememory import engine as engine_module, storage
            storage.DEFAULT_BUSY_TIMEOUT_MS = engine_module.DEFAULT_BUSY_TIMEOUT_MS = 200
            from truememory.client import Memory
            memory = Memory(DB)  # What the MCP server and the hooks store through.
            engine = memory._engine
            first = memory.add("synthetic warm-up", user_id="alice")
            def held(write, seconds=1.5):
                holder = sqlite3.connect(DB, isolation_level=None, check_same_thread=False)
                holder.execute("BEGIN IMMEDIATE")
                timer = threading.Timer(seconds, lambda: (holder.rollback(), holder.close()))
                timer.start()
                started = time.monotonic()
                try:
                    outcome = write()
                except sqlite3.OperationalError as error:
                    outcome = str(error)
                waited = time.monotonic() - started
                timer.join()
                return outcome, waited
            count = lambda: engine.conn.execute("SELECT count(*) FROM messages").fetchone()[0]
            before = count()
            added, add_wait = held(lambda: memory.add("synthetic stored after the wait", user_id="alice"))
            updated, update_wait = held(lambda: memory.update(added["id"], "synthetic updated after the wait"))
            deleted, delete_wait = held(lambda: memory.delete(first["id"]))
            content = engine.get(added["id"])["content"]
            stored = count() - before
            from truememory.tier_switch import writer
            writer.FOREGROUND_BUSY_RETRY_SECONDS = 0.5
            outcome, gave_up = held(lambda: memory.add("synthetic given up", user_id="alice") and "stored", 2.5)
            after = count() - before
            memory.add("synthetic stored again", user_id="alice")
            emit(waits=[add_wait, update_wait, delete_wait], content=content, deleted=deleted, stored=stored,
                 outcome=outcome, gave_up=gave_up, after=after, final=count() - before)
            engine.close()
        """, tier="edge")
        self.assertTrue(all(wait > 1.0 for wait in result["waits"]), result)
        self.assertEqual((result["content"], result["deleted"]), ("synthetic updated after the wait", True))
        self.assertEqual(result["stored"], 0)  # One added, one deleted.
        self.assertEqual(result["outcome"], "database is locked")
        self.assertLess(result["gave_up"], 2.0)
        self.assertEqual((result["after"], result["final"]), (0, 1))


class DatabaseFileAccess(unittest.TestCase):
    def test_store_and_switch_never_open_database_files_outside_sqlite(self):
        # Every Python-level open (open, os.open, shutil copies) raises an "open"
        # audit event; SQLite's own opens do not. Closing a descriptor opened
        # outside SQLite drops the process's POSIX locks on the file (D1/D2).
        (result,) = run_scenario("""
            import traceback
            watched = {os.path.realpath(DB + suffix) for suffix in ("", "-wal", "-shm")}
            opened = []
            def audit(event, args):
                if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
                    name = os.fsdecode(args[0])
                    if name.endswith(("memories.db", "-wal", "-shm")) and os.path.realpath(name) in watched:
                        opened.append(name + " " + traceback.format_stack(limit=4)[0].strip().replace(chr(10), " "))
            sys.addaudithook(audit)
            seed(30)
            stub_models()
            import model2vec
            model2vec.StaticModel.from_pretrained = staticmethod(lambda *args, **kwargs: FakeEncoder())
            from truememory.engine import TrueMemoryEngine
            from truememory.tier_switch.manager import RebuildManager, backup_db
            engine = TrueMemoryEngine(DB)
            added = engine.add("stored before the switch", sender="alice")["id"]
            engine.search("synthetic fact", limit=3); engine.get(added); engine.get_all(limit=5)
            outcome = switch("edge")["outcome"]
            engine.add("stored after the switch", sender="alice"); engine.delete(added)
            drain()
            backup_db(Path(DB))
            RebuildManager().get_status()
            engine.close()
            emit(outcome=outcome, opened=opened)
        """)
        self.assertEqual(result["outcome"], "complete")
        self.assertEqual(result["opened"], [])


if __name__ == "__main__":
    unittest.main()
