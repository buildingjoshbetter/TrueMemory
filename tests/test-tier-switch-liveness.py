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

def seed(n=40, group="basepro", tables=None):
    from truememory.storage import create_db
    conn = create_db(DB); load_vec(conn)
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

def stub_models(delay=0.0, rerankers_available=lambda name: True, embeddings_available=lambda: True):
    from truememory import vector_search, reranker
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
    engine.search("fact", limit=3)
    out["search"] = "ok"
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
        self.assertNotEqual(child["search"], "ok")
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


class ActivationEdgeCases(unittest.TestCase):
    def test_cancel_while_waiting_for_maintenance_ends_the_job(self):
        (result,) = run_scenario("""
            import subprocess
            seed(20)
            stub_models(delay=0.02)
            from truememory.tier_switch.manager import RebuildManager
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
            manager.cancel()
            worker.join(60)
            stopped = not worker.is_alive()
            holder.communicate("release\\n", timeout=30)
            emit(stopped=stopped, outcome=manager._last_outcome, state=state())
        """ % HOLD_MAINTENANCE)
        self.assertTrue(result["stopped"])
        self.assertNotEqual(result["outcome"], "activation_pending")
        self.assertIsNone(result["state"]["selected"])
        self.assertEqual(result["state"]["marker"], "failed")
        self.assertEqual(result["state"]["status_rows"], [[1, "failed"]])

if __name__ == "__main__":
    unittest.main()
