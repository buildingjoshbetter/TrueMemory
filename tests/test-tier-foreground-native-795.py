"""Opt-in Linux sqlite-vec publication under another thread's maintenance owner."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import struct
import sys
import threading
from types import ModuleType

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or os.environ.get("TRUEMEMORY_TEST_NATIVE_VEC") != "1",
    reason="Native foreground regression requires Linux and TRUEMEMORY_TEST_NATIVE_VEC=1",
)


class SyntheticEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    @staticmethod
    def vector(text: str) -> tuple[float, ...]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # 256 * (1 / 16) ** 2 = 1, including the deterministic sign changes.
        return tuple(1 / 16 if digest[index // 8] & (1 << (index % 8)) else -1 / 16
                     for index in range(256))

    def encode(self, texts: list[str], **kwargs: object) -> object:
        import numpy as np

        self.calls.append(tuple(texts))
        return np.asarray([self.vector(text) for text in texts], dtype=np.float32).reshape(len(texts), 256)


@pytest.fixture
def native_engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[tuple[object, ModuleType, SyntheticEncoder]]:
    # Keep the guard before every application/native import and database open.
    if sys.platform != "linux" or os.environ.get("TRUEMEMORY_TEST_NATIVE_VEC") != "1":
        pytest.skip("Native foreground regression is opt-in on Linux")
    monkeypatch.setenv("TRUEMEMORY_EMBED_MODEL", "edge")
    monkeypatch.setenv("TRUEMEMORY_MODEL_SERVER", "off")
    from truememory import reranker, vector_search
    from truememory.engine import TrueMemoryEngine
    from truememory.maintenance import get_coordinator

    vector_search.set_embedding_model("edge")
    reranker.set_active_tier("edge")
    encoder = SyntheticEncoder()
    monkeypatch.setattr(vector_search, "_model", encoder)
    engine = TrueMemoryEngine(tmp_path / "synthetic-foreground.sqlite")
    coordinator = get_coordinator(engine.db_path)
    try:
        # Setup alone suppresses scheduling; all tested operations run normally.
        engine._ensure_connection(_suppress_maintenance=True)
        assert engine.ready and engine._has_vectors
        assert encoder.calls == []
        assert engine.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        for table in (vector_search._active_vec_table(engine.conn), vector_search._active_sep_table(engine.conn)):
            sql = engine.conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()[0]
            assert "using vec0" in sql.lower()
            assert engine.conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] == 0
        yield engine, vector_search, encoder
    finally:
        # close() only detaches the Engine. Stop and join its shared worker first.
        coordinator.cancel()
        try:
            assert coordinator.wait(10), "Shared maintenance teardown did not finish"
        finally:
            engine.close()


@contextmanager
def held_maintenance(path: Path) -> Iterator[threading.Thread]:
    from truememory.maintenance import MaintenanceBusyError, maintenance_owner

    acquired = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def hold() -> None:
        try:
            with maintenance_owner(path):
                acquired.set()
                if not release.wait(30):
                    raise AssertionError("Foreground operation outlived the maintenance-owner deadline")
        except BaseException as error:
            errors.append(error)
            acquired.set()

    worker = threading.Thread(target=hold, name="synthetic-foreground-maintenance-owner")
    worker.start()
    try:
        assert acquired.wait(10), "Maintenance owner was not acquired"
        assert not errors, errors
        with pytest.raises(MaintenanceBusyError):
            with maintenance_owner(path):
                pytest.fail("A second thread acquired the existing maintenance ownership")
        yield worker
        assert worker.is_alive(), "Maintenance ownership ended before foreground assertions"
    finally:
        release.set()
        worker.join(10)
        assert not worker.is_alive(), "Maintenance owner did not join"
        assert not errors, errors


def assert_publication(engine: object, vector: ModuleType, encoder: SyntheticEncoder,
                       message_id: int, content: str) -> tuple[bytes, bytes]:
    row = engine.conn.execute(
        "SELECT content,sender,recipient,timestamp FROM messages WHERE id=?", (message_id,),
    ).fetchone()
    assert row is not None and row[0] == content
    separation = vector._build_sep_text(row[1], row[2], row[3], content)
    blobs = []
    for table, text in ((vector._active_vec_table(engine.conn), content),
                        (vector._active_sep_table(engine.conn), separation)):
        rows = engine.conn.execute(f'SELECT rowid,embedding FROM "{table}" ORDER BY rowid').fetchall()
        assert len(rows) == 1 and rows[0][0] == message_id
        blob = rows[0][1]
        assert len(blob) == 256 * 4
        values = struct.unpack("256f", blob)
        assert values == encoder.vector(text)
        assert sum(value * value for value in values) == 1
        blobs.append(blob)
    metadata = dict(engine.conn.execute(
        "SELECT key,value FROM metadata WHERE key IN ('embed_model','embed_dim')",
    ).fetchall())
    assert metadata == {"embed_model": vector.EMBEDDING_MODEL, "embed_dim": "256"}
    assert not engine.conn.in_transaction
    return blobs[0], blobs[1]


def seed_source(engine: object, content: str) -> int:
    cursor = engine.conn.execute(
        "INSERT INTO messages(content,sender,recipient,timestamp) VALUES(?,?,?,?)",
        (content, "synthetic-sender", "synthetic-recipient", "2026-01-02"),
    )
    engine.conn.commit()
    return cursor.lastrowid


def test_first_add_publishes_vectors_and_metadata_under_maintenance(native_engine: tuple) -> None:
    engine, vector, encoder = native_engine
    with held_maintenance(engine.db_path):
        result = engine.add("synthetic-first-add", sender="synthetic-sender",
                            recipient="synthetic-recipient", timestamp="2026-01-02")
        assert_publication(engine, vector, encoder, result["id"], "synthetic-first-add")
        assert len(encoder.calls) == 2


def test_update_replaces_both_vectors_under_maintenance(native_engine: tuple) -> None:
    engine, vector, encoder = native_engine
    message_id = seed_source(engine, "synthetic-before-update")
    vector.embed_single(engine.conn, message_id, "synthetic-before-update")
    before = assert_publication(engine, vector, encoder, message_id, "synthetic-before-update")
    call_count = len(encoder.calls)
    with held_maintenance(engine.db_path):
        result = engine.update(message_id, content="synthetic-after-update")
        assert result is not None and result["id"] == message_id
        after = assert_publication(engine, vector, encoder, message_id, "synthetic-after-update")
        assert before[0] != after[0] and before[1] != after[1]
        assert len(encoder.calls) == call_count + 2


def test_direct_embed_single_publishes_known_row_under_maintenance(native_engine: tuple) -> None:
    engine, vector, encoder = native_engine
    message_id = seed_source(engine, "synthetic-direct-embed")
    with held_maintenance(engine.db_path):
        vector.embed_single(engine.conn, message_id, "synthetic-direct-embed")
        assert_publication(engine, vector, encoder, message_id, "synthetic-direct-embed")
        assert len(encoder.calls) == 2


def test_fresh_engine_opens_during_actual_automatic_worker(
    native_engine: tuple, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from model2vec import StaticModel
    from truememory import maintenance
    from truememory.engine import TrueMemoryEngine

    first, vector, encoder = native_engine
    content = "synthetic-automatic-worker-open"
    message_id = seed_source(first, content)
    vector.embed_single(first.conn, message_id, content)
    before = assert_publication(first, vector, encoder, message_id, content)
    encode_count = len(encoder.calls)
    coordinator = first._get_maintenance_coordinator()
    second = TrueMemoryEngine(first.db_path)
    entered = threading.Event()
    release = threading.Event()
    failures: list[str] = []
    completed: list[bool] = []
    constructor_calls: list[bool] = []
    real_run = maintenance.run_engine_maintenance

    def blocked_run(conn: object, owner: object, **kwargs: object) -> object:
        try:
            assert conn is not first.conn
            assert owner is coordinator
            assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
            entered.set()
            if not release.wait(30):
                raise AssertionError("Fresh Engine outlived the automatic-worker barrier")
            result = real_run(conn, owner, **kwargs)
            completed.append(True)
            return result
        except BaseException as error:
            failures.append(type(error).__name__)
            entered.set()
            raise

    def forbidden_constructor(*args: object, **kwargs: object) -> object:
        constructor_calls.append(True)
        raise AssertionError("Opening a validated database must not construct an embedding model")

    def value_only(value: object) -> bool:
        if type(value) is tuple:
            return all(value_only(item) for item in value)
        return type(value) in (type(None), bool, int, float, str)

    monkeypatch.setattr(maintenance, "run_engine_maintenance", blocked_run)
    try:
        first._maybe_startup_consolidate()
        assert entered.wait(10), "Automatic worker did not reach its opened-connection barrier"
        assert not failures, failures
        assert coordinator.snapshot()["active"]
        with pytest.raises(maintenance.MaintenanceBusyError):
            with maintenance.maintenance_owner(first.db_path):
                pytest.fail("Automatic worker did not retain real maintenance ownership")

        database_path = first.db_path.resolve()
        with maintenance._active_initialization_receipt(database_path) as offered:
            receipt = None if offered is None else offered[2]
        assert type(receipt) is tuple and value_only(receipt)
        assert receipt == first._reconnect_receipt

        with monkeypatch.context() as construction_guard:
            construction_guard.setattr(StaticModel, "from_pretrained", forbidden_constructor)
            second._ensure_connection()
            assert second.ready and second._has_vectors
            assert second.conn is not first.conn
            assert assert_publication(second, vector, encoder, message_id, content) == before
            assert len(encoder.calls) == encode_count
            assert vector._model is encoder
            assert constructor_calls == []
            assert coordinator.snapshot()["active"] and not release.is_set()
            assert not failures, failures
    finally:
        release.set()
        try:
            assert coordinator.wait(15), "Automatic worker did not finish after barrier release"
            assert not failures, failures
            assert completed, "The wrapper never called the real maintenance implementation"
        finally:
            coordinator.cancel()
            try:
                assert coordinator.wait(10), "Automatic maintenance teardown did not finish"
            finally:
                second.close()


def test_fresh_public_search_waits_for_committed_style_enrollment(
    native_engine: tuple, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sentence_transformers
    from model2vec import StaticModel
    from truememory import maintenance, reranker, storage
    from truememory.client import Memory
    from truememory.tier_switch import runtime

    first, vector, encoder = native_engine
    content = "synthetic workshop inventory record"
    query = "workshop inventory"
    message_id = seed_source(first, content)
    vector.embed_single(first.conn, message_id, content)
    before = assert_publication(first, vector, encoder, message_id, content)
    encode_count = len(encoder.calls)
    assert first._has_style_vec
    receipt = first._reconnect_receipt
    assert receipt is not None
    schema_before = first.conn.execute("PRAGMA schema_version").fetchone()[0]
    assert schema_before == receipt[3]
    expected_triggers = {
        name: " ".join(sql.split()) for name, sql in storage._STYLE_OUTPUT_TRIGGERS.items()
    }
    assert len(expected_triggers) == 3
    trigger_query = "SELECT name,sql FROM sqlite_master WHERE type='trigger' AND name IN (?,?,?)"
    assert first.conn.execute(trigger_query, tuple(expected_triggers)).fetchall() == []

    coordinator = first._get_maintenance_coordinator()
    fresh = Memory(path=first.db_path)
    enrolled = threading.Event()
    release = threading.Event()
    wait_entered = threading.Event()
    search_done = threading.Event()
    preparation_lock = threading.Lock()
    preparations: list[bool] = []
    enrolled_schemas: list[int] = []
    worker_errors: list[BaseException] = []
    search_errors: list[BaseException] = []
    results: list[dict] = []
    constructor_calls: list[bool] = []
    reranker_constructor_calls: list[bool] = []
    reranker_pairs: list[tuple[tuple[str, str], ...]] = []
    reranker_predictions: list[tuple[float, ...]] = []
    expected_reranker_name = reranker.get_current_reranker_name()
    real_prepare = maintenance._prepare_routed_style_maintenance
    real_wait = maintenance.wait_for_automatic_owner

    class SyntheticReranker:
        def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> list[float]:
            scores = [1.0 if document == content else 0.25 for _, document in pairs]
            reranker_pairs.append(tuple(pairs))
            reranker_predictions.append(tuple(scores))
            return scores

    synthetic_reranker = SyntheticReranker()

    def prepared_then_blocked(conn: object) -> None:
        with preparation_lock:
            first_preparation = not preparations
            preparations.append(True)
        try:
            real_prepare(conn)
            if not first_preparation:
                return
            assert conn is not first.conn
            assert not conn.in_transaction
            installed = {
                name: " ".join(sql.strip().rstrip(";").split())
                for name, sql in conn.execute(trigger_query, tuple(expected_triggers))
            }
            assert installed == expected_triggers
            schema_after = conn.execute("PRAGMA schema_version").fetchone()[0]
            assert schema_after == schema_before + len(expected_triggers)
            enrolled_schemas.append(schema_after)
            enrolled.set()
            if not release.wait(30):
                raise AssertionError("Public warm search outlived the style-enrollment barrier")
        except BaseException as error:
            worker_errors.append(error)
            enrolled.set()
            raise

    def observed_wait(error: object, *, deadline: float) -> object:
        assert runtime.current_runtime_operation() is None
        assert fresh._engine.conn is None and not fresh._engine.ready
        assert len(encoder.calls) == encode_count
        wait_entered.set()
        return real_wait(error, deadline=deadline)

    def search_publicly() -> None:
        try:
            results.extend(fresh.search(query))
        except BaseException as error:
            search_errors.append(error)
        finally:
            search_done.set()

    def forbidden_constructor(*args: object, **kwargs: object) -> object:
        constructor_calls.append(True)
        raise AssertionError("Warm public search must reuse its synthetic embedding model")

    def forbidden_reranker_constructor(*args: object, **kwargs: object) -> object:
        reranker_constructor_calls.append(True)
        raise AssertionError("Warm public search must reuse its synthetic reranker")

    search_worker = threading.Thread(target=search_publicly, name="synthetic-style-warm-search")
    monkeypatch.setattr(maintenance, "_prepare_routed_style_maintenance", prepared_then_blocked)
    monkeypatch.setattr(maintenance, "wait_for_automatic_owner", observed_wait)
    with monkeypatch.context() as construction_guard:
        construction_guard.setattr(StaticModel, "from_pretrained", forbidden_constructor)
        construction_guard.setattr(sentence_transformers, "CrossEncoder", forbidden_reranker_constructor)
        construction_guard.setattr(reranker, "_model", synthetic_reranker)
        construction_guard.setattr(reranker, "_model_name", expected_reranker_name)
        construction_guard.setattr(reranker, "_model_certified", False)
        try:
            first._maybe_startup_consolidate()
            assert enrolled.wait(10), "Actual canonical style preparation did not complete"
            assert not worker_errors, worker_errors
            assert enrolled_schemas == [schema_before + 3]
            assert first.conn.execute("PRAGMA schema_version").fetchone()[0] == enrolled_schemas[0]
            assert coordinator.snapshot()["active"]
            with pytest.raises(maintenance.MaintenanceBusyError):
                with maintenance.maintenance_owner(first.db_path):
                    pytest.fail("Committed style enrollment lost its automatic maintenance owner")

            search_worker.start()
            assert wait_entered.wait(10), "Fresh public search did not reach automatic-owner waiting"
            assert not search_done.is_set() and search_worker.is_alive()
            assert not search_errors, search_errors
            assert fresh._engine.conn is None and not fresh._engine.ready
            assert len(encoder.calls) == encode_count
            assert constructor_calls == []
            assert reranker_constructor_calls == [] and reranker_pairs == []
            assert coordinator.snapshot()["active"] and not release.is_set()
            assert fresh._engine._write_lock.acquire(blocking=False), "Public wait retained the Engine writer lock"
            fresh._engine._write_lock.release()

            release.set()
            search_worker.join(15)
            assert not search_worker.is_alive() and search_done.is_set()
            assert not search_errors, search_errors
            assert coordinator.wait(15), "Automatic worker did not finish after style barrier release"
            assert not worker_errors, worker_errors
            assert fresh._engine.ready and fresh._engine._has_vectors
            assert any(result["id"] == message_id and "vec" in result["source"] for result in results)
            assert assert_publication(fresh._engine, vector, encoder, message_id, content) == before
            search_calls = encoder.calls[encode_count:]
            assert 1 <= len(search_calls) <= 2
            assert all(texts == (query,) for texts in search_calls)
            assert vector._model is encoder and constructor_calls == []
            assert reranker._model is synthetic_reranker and reranker._model_name == expected_reranker_name
            assert reranker_constructor_calls == [] and reranker_pairs
            assert all(pair_query == query for pairs in reranker_pairs for pair_query, _ in pairs)
            assert any(document == content for pairs in reranker_pairs for _, document in pairs)
            assert all(
                len(pairs) == len(scores)
                and scores == tuple(1.0 if document == content else 0.25 for _, document in pairs)
                for pairs, scores in zip(reranker_pairs, reranker_predictions)
            )
        finally:
            release.set()
            try:
                if search_worker.ident is not None:
                    search_worker.join(15)
                    assert not search_worker.is_alive(), "Public search thread did not join"
            finally:
                coordinator.cancel()
                try:
                    assert coordinator.wait(10), "Style maintenance teardown did not finish"
                finally:
                    fresh._engine.close()


def test_native_mcp_deep_children_keep_final_tables_and_defer_maintenance(
    native_engine: tuple, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from model2vec import StaticModel
    import sentence_transformers
    from truememory import Memory, maintenance, reranker
    from truememory.engine import TrueMemoryEngine
    from truememory.tier_switch import runtime, serving

    monkeypatch.setenv("TRUEMEMORY_PRELOAD_MODELS", "0")
    from truememory import mcp_server

    first, vector, encoder = native_engine
    content = "synthetic workshop inventory record"
    queries = ("workshop inventory", "synthetic inventory")
    message_id = seed_source(first, content)
    vector.embed_single(first.conn, message_id, content)
    before = assert_publication(first, vector, encoder, message_id, content)
    encode_count = len(encoder.calls)
    source_row = first.conn.execute(
        "SELECT sender,recipient,timestamp FROM messages WHERE id=?", (message_id,),
    ).fetchone()
    separation = vector._build_sep_text(*source_row, content)
    coordinator = first._get_maintenance_coordinator()
    parent = Memory(path=first.db_path)
    expected_tables = ("vec_messages_edge", "vec_messages_sep_edge")
    expected_reranker = mcp_server._DEEP_RERANKER
    both_constructed = threading.Event()
    both_opened = threading.Event()
    query_barrier = threading.Barrier(2)
    body_finished = threading.Event()
    worker_entered = threading.Event()
    worker_release = threading.Event()
    observations_lock = threading.Lock()
    children: list[Memory] = []
    opening: list[object] = []
    opened: list[object] = []
    active_queries: list[str] = []
    concurrent_queries: list[int] = []
    completed_queries: list[str] = []
    parent_keys: list[tuple] = []
    child_keys: list[tuple] = []
    requests: list[bool] = []
    worker_runs: list[bool] = []
    errors: list[BaseException] = []
    constructor_calls: list[str] = []
    reranker_pairs: list[tuple[tuple[str, str], ...]] = []
    real_open = TrueMemoryEngine._open_connection_handle
    real_deep = Memory.search_deep
    real_request = coordinator.request_layers
    real_run = maintenance.run_engine_maintenance

    class SyntheticDeepReranker:
        def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> list[float]:
            reranker_pairs.append(tuple(pairs))
            return [1.0 if document == content else 0.25 for _, document in pairs]

    synthetic_reranker = SyntheticDeepReranker()

    def make_child(*args: object, **kwargs: object) -> Memory:
        child = Memory(*args, **kwargs)
        with observations_lock:
            children.append(child)
            if len(children) == 2:
                both_constructed.set()
        return child

    def observed_open(engine: object) -> None:
        with observations_lock:
            is_child = any(child._engine is engine for child in children)
            first_handle = is_child and engine.conn is None
            if first_handle:
                opening.append(engine)
                ordinal = len(opening)
        if not first_handle:
            return real_open(engine)
        try:
            assert both_constructed.wait(10), "Both real child Memory objects were not constructed"
            with observations_lock:
                assert len(opening) == ordinal and len(opened) == ordinal - 1
            real_open(engine)
            assert engine.conn is not None and not engine.ready
            with observations_lock:
                opened.append(engine)
                if len(opened) == 2:
                    both_opened.set()
        except BaseException as error:
            errors.append(error)
            raise

    def observed_deep(memory: Memory, query: str, *args: object, **kwargs: object) -> list[dict]:
        try:
            assert both_opened.wait(10), "A child search retained the sibling opening lock"
            assert kwargs == {"user_id": None, "limit": 10, "llm_fn": None} and args == ()
            operation = runtime.current_operation(memory._engine.conn)
            assert operation is not None and operation._defer_maintenance
            assert operation.tables == expected_tables and operation.key == parent_keys[0]
            assert operation.reranker_id == expected_reranker
            assert reranker.get_current_reranker_name() == expected_reranker
            assert reranker.get_reranker() is synthetic_reranker
            assert requests == [] and not worker_entered.is_set()
            with observations_lock:
                child_keys.append(operation.key)
                active_queries.append(query)
                concurrent_queries.append(len(active_queries))
            query_barrier.wait(10)
            result = real_deep(memory, query, *args, **kwargs)
            assert memory._engine.ready and memory._engine._has_vectors
            assert runtime.current_operation(memory._engine.conn) is operation
            assert (vector._active_vec_table(memory._engine.conn),
                    vector._active_sep_table(memory._engine.conn)) == expected_tables
            assert assert_publication(memory._engine, vector, encoder, message_id, content) == before
            assert any(item["id"] == message_id and "vec" in item["source"] for item in result)
            assert requests == [] and not worker_entered.is_set()
            completed_queries.append(query)
            return result
        except BaseException as error:
            errors.append(error)
            query_barrier.abort()
            raise
        finally:
            with observations_lock:
                if query in active_queries:
                    active_queries.remove(query)

    def observed_request(**kwargs: object) -> bool:
        requests.append(True)
        try:
            assert body_finished.is_set() and len(requests) == 1
            assert runtime.current_runtime_operation() is None
            assert not serving.current_thread_admitted()
            assert len(children) == 2 and all(child._engine.conn is None for child in children)
            assert sorted(completed_queries) == sorted(queries)
            return real_request(**kwargs)
        except BaseException as error:
            errors.append(error)
            raise

    def observed_run(conn: object, owner: object, **kwargs: object) -> object:
        try:
            assert owner is coordinator and body_finished.is_set()
            worker_runs.append(True)
            if len(worker_runs) == 1:
                worker_entered.set()
                if not worker_release.wait(30):
                    raise AssertionError("MCP post-parent worker outlived its observation barrier")
            return real_run(conn, owner, **kwargs)
        except BaseException as error:
            errors.append(error)
            worker_entered.set()
            raise

    def forbidden_embedding_constructor(*args: object, **kwargs: object) -> object:
        constructor_calls.append("embedding")
        raise AssertionError("MCP child searches must retain the synthetic embedding model")

    def forbidden_reranker_constructor(*args: object, **kwargs: object) -> object:
        constructor_calls.append("reranker")
        raise AssertionError("MCP Deep children must retain the synthetic explicit reranker")

    with monkeypatch.context() as observed:
        observed.setattr(mcp_server, "_memory", parent)
        observed.setattr(mcp_server, "Memory", make_child)
        observed.setattr(TrueMemoryEngine, "_open_connection_handle", observed_open)
        observed.setattr(Memory, "search_deep", observed_deep)
        observed.setattr(coordinator, "request_layers", observed_request)
        observed.setattr(maintenance, "run_engine_maintenance", observed_run)
        observed.setattr(StaticModel, "from_pretrained", forbidden_embedding_constructor)
        observed.setattr(sentence_transformers, "CrossEncoder", forbidden_reranker_constructor)
        observed.setattr(reranker, "_model", synthetic_reranker)
        observed.setattr(reranker, "_model_name", expected_reranker)
        observed.setattr(reranker, "_model_certified", False)
        try:
            with mcp_server._search_operation(reranker_id=expected_reranker) as (memory, operation):
                assert memory is parent and parent._engine.ready
                assert operation.tables == expected_tables and operation._defer_maintenance
                assert operation.reranker_id == expected_reranker
                parent_keys.append(operation.key)
                assert assert_publication(parent._engine, vector, encoder, message_id, content) == before
                assert parent._engine.conn.execute(
                    "SELECT name FROM sqlite_master WHERE name IN ('vec_messages','vec_messages_sep')",
                ).fetchall() == []
                assert len(encoder.calls) == encode_count and constructor_calls == []
                assert requests == [] and not worker_entered.is_set()
                results = mcp_server._parallel_search(
                    queries, None, 10, None, 10, memory=memory, operation=operation,
                )
                assert not errors, errors
                assert len(opening) == len(opened) == 2 and both_opened.is_set()
                assert child_keys == [operation.key, operation.key]
                assert max(concurrent_queries) == 2 and active_queries == []
                assert sorted(completed_queries) == sorted(queries)
                assert all(child._engine.conn is None for child in children)
                assert any(item["id"] == message_id and "vec" in item["source"] for item in results)
                assert requests == [] and not worker_entered.is_set()
                assert assert_publication(parent._engine, vector, encoder, message_id, content) == before
                body_finished.set()
            assert requests == [True]
            assert worker_entered.wait(10), "Deferred automatic maintenance did not start after the parent"
            assert not errors, errors
            assert coordinator.snapshot()["active"] and not worker_release.is_set()
            assert constructor_calls == [] and vector._model is encoder
            assert reranker._model is synthetic_reranker and reranker._model_name == expected_reranker
            assert encoder.calls[encode_count:]
            assert all(text not in (content, separation)
                       for texts in encoder.calls[encode_count:] for text in texts)
            assert all(query in queries for pairs in reranker_pairs for query, _ in pairs)
            worker_release.set()
            assert coordinator.wait(15), "MCP post-parent maintenance did not finish"
            assert not errors, errors
            assert requests == [True] and constructor_calls == []
            assert assert_publication(parent._engine, vector, encoder, message_id, content) == before
        finally:
            worker_release.set()
            query_barrier.abort()
            coordinator.cancel()
            try:
                assert coordinator.wait(10), "MCP shared maintenance teardown did not finish"
            finally:
                for child in children:
                    child.close()
                parent.close()
