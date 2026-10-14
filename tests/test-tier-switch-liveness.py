"""Real create_db databases through certified tier switching, with synthetic models.

Each scenario runs in a fresh interpreter: tier selection, model slots and the
maintenance registry are process-wide, so isolation keeps scenarios honest.
Embedding leases, frozen targets and rerankers are deterministic stand-ins; the
database, journal, job marker, source fences and serving gate are real.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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
            [sys.executable, "-c", PRELUDE + textwrap.dedent(body)], env=env, cwd=home,
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


if __name__ == "__main__":
    unittest.main()
