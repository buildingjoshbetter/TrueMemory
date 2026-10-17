"""Engine.close() stops this process's background writes; synthetic databases only.

Every wait here is on an Event set by the code under test or on a join, never
a sleep. A run that outlives the close wait is fenced, so these tests set the
wait to zero to reach that branch deterministically.
"""

import shutil
import sqlite3
import threading
from pathlib import Path

import pytest

from truememory import maintenance
from truememory.engine import TrueMemoryEngine
from truememory.storage import create_db
from truememory.tier_switch import runtime


def _engine(path: Path) -> TrueMemoryEngine:
    engine = TrueMemoryEngine(path)
    engine.conn = create_db(path)
    return engine


def _rows(path: Path) -> int:
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        return conn.execute("SELECT count(*) FROM messages").fetchone()[0]
    finally:
        conn.close()


def _blocked_worker(coordinator: maintenance.MaintenanceCoordinator, outcome: dict) -> threading.Event:
    """Start a worker that has read its database and waits; then it tries one write."""
    entered, release = threading.Event(), threading.Event()

    def work(conn: sqlite3.Connection, cancel: threading.Event) -> None:
        conn.execute("SELECT count(*) FROM messages").fetchone()
        entered.set()
        assert release.wait(10)
        try:
            conn.execute("INSERT INTO messages(content) VALUES ('synthetic write after close')")
            conn.commit()
        except maintenance.MaintenanceClosedError:
            outcome["write"] = "refused"
            raise
        outcome["write"] = "committed"

    assert coordinator.request(work)
    assert entered.wait(10)
    return release


@pytest.fixture
def store(tmp_path: Path) -> Path:
    path = tmp_path / "store" / "synthetic.sqlite"
    path.parent.mkdir()
    create_db(path).close()
    return path


def test_last_close_waits_for_a_worker_that_stops_at_its_phase_boundary(store: Path) -> None:
    engine = _engine(store)
    coordinator = maintenance.get_coordinator(store)
    entered = threading.Event()

    def work(conn: sqlite3.Connection, cancel: threading.Event) -> None:
        entered.set()
        cancel.wait(10)  # The phase boundary: the next step observes close().

    assert coordinator.request(work)
    assert entered.wait(10)
    engine.close()
    # The worker had finished, its connection and owner lock closed, before close() returned.
    assert coordinator.wait(0)
    assert coordinator.status == ("cancelled", None)
    shutil.rmtree(store.parent)
    assert not store.parent.exists()


def test_last_close_fences_a_worker_that_outlives_the_wait(store: Path, monkeypatch) -> None:
    engine = _engine(store)
    coordinator = maintenance.get_coordinator(store)
    outcome = {}
    release = _blocked_worker(coordinator, outcome)
    monkeypatch.setattr(maintenance, "_CLOSE_WAIT_SECONDS", 0.0, raising=False)
    engine.close()
    release.set()
    assert coordinator.wait(10)
    assert outcome == {"write": "refused"}
    assert _rows(store) == 0
    assert coordinator.status == ("cancelled", None)


def test_close_then_delete_leaves_nothing_for_a_fenced_worker_to_recreate(store: Path, monkeypatch) -> None:
    engine = _engine(store)
    coordinator = maintenance.get_coordinator(store)
    outcome = {}
    release = _blocked_worker(coordinator, outcome)
    monkeypatch.setattr(maintenance, "_CLOSE_WAIT_SECONDS", 0.0, raising=False)
    engine.close()
    shutil.rmtree(store.parent)
    release.set()
    assert coordinator.wait(10)
    assert outcome == {"write": "refused"}
    assert not store.parent.exists()


def test_worker_open_never_recreates_a_deleted_store(store: Path, monkeypatch) -> None:
    coordinator = maintenance.get_coordinator(store)
    opened, release = threading.Event(), threading.Event()
    real_open = runtime.open_serving_connection

    def paused_open(db_path, legacy_factory):
        opened.set()
        assert release.wait(10)
        return real_open(db_path, legacy_factory)

    monkeypatch.setattr(runtime, "open_serving_connection", paused_open)
    ran = []
    assert coordinator.request(lambda conn, cancel: ran.append(conn))
    assert opened.wait(10)
    for suffix in ("", "-wal", "-shm"):
        Path(str(store) + suffix).unlink(missing_ok=True)
    release.set()
    assert coordinator.wait(10)
    assert ran == []
    assert coordinator.status[0] == "failed"
    assert not store.exists()


def test_background_opens_never_create_a_missing_store(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sqlite"
    with maintenance.background_writes(maintenance.get_coordinator(missing)):
        with pytest.raises(sqlite3.OperationalError):
            runtime.open_serving_connection(missing, create_db)
        with pytest.raises(sqlite3.OperationalError):
            create_db(missing)
    # Only the maintenance owner's lock file; no database, journal or WAL.
    assert [item.name for item in tmp_path.iterdir()] == ["missing.sqlite.maintenance.lock"]


def test_last_close_fences_other_background_runs_and_refuses_new_ones(store: Path, monkeypatch) -> None:
    # The Qwen3 NaN re-embed thread runs its work this way.
    engine = _engine(store)
    coordinator = maintenance.get_coordinator(store)
    entered, release = threading.Event(), threading.Event()
    outcome = {}

    def reembed() -> None:
        with maintenance.background_writes(coordinator):
            conn = runtime.open_serving_connection(store, create_db)
            try:
                conn.execute("SELECT count(*) FROM messages").fetchone()
                entered.set()
                assert release.wait(10)
                conn.execute("INSERT INTO messages(content) VALUES ('synthetic re-embed after close')")
                conn.commit()
                outcome["write"] = "committed"
            except maintenance.MaintenanceClosedError:
                outcome["write"] = "refused"
            finally:
                conn.close()

    thread = threading.Thread(target=reembed, name="synthetic-reembed")
    thread.start()
    assert entered.wait(10)
    monkeypatch.setattr(maintenance, "_CLOSE_WAIT_SECONDS", 0.0, raising=False)
    engine.close()
    release.set()
    thread.join(10)
    assert not thread.is_alive()
    assert outcome == {"write": "refused"}
    assert _rows(store) == 0
    with pytest.raises(maintenance.MaintenanceClosedError):
        with maintenance.background_writes(coordinator):
            pass
    # An Engine opening the database again lifts the refusal.
    reopened = _engine(store)
    assert maintenance.get_coordinator(store) is coordinator
    with maintenance.background_writes(coordinator):
        pass
    reopened.close()


def test_background_backfill_never_reopens_a_closed_engine(store: Path, monkeypatch) -> None:
    engine = _engine(store)
    engine.ready = True
    entered, release = threading.Event(), threading.Event()
    real_open = runtime.open_engine_for_operation

    def paused_open(target):
        entered.set()
        assert release.wait(10)
        return real_open(target)

    monkeypatch.setattr(runtime, "open_engine_for_operation", paused_open)
    # The backfill thread's body, past its stop check when close() runs.
    backfill = threading.Thread(target=engine._background_backfill, name="synthetic-backfill")
    backfill.start()
    assert entered.wait(10)
    engine.close()
    shutil.rmtree(store.parent)
    release.set()
    backfill.join(30)
    assert not backfill.is_alive()
    assert engine.conn is None
    assert not store.parent.exists()


def test_shared_worker_survives_one_close_and_stops_at_the_last(store: Path, monkeypatch) -> None:
    first, second = _engine(store), _engine(store)
    coordinator = maintenance.get_coordinator(store)

    shared = {}
    release = _blocked_worker(coordinator, shared)
    first.close()
    # The other Engine still has the database open: the worker keeps running.
    assert not coordinator.wait(0)
    assert coordinator.snapshot()["active"]
    assert second.conn.execute("SELECT count(*) FROM messages").fetchone() == (0,)
    release.set()
    assert coordinator.wait(10)
    assert shared == {"write": "committed"}
    assert coordinator.status == ("success", None)

    # Maintenance continues for the open Engine: a new wake starts a worker.
    report = maintenance.MaintenanceReport((), "SKIPPED (synthetic)")
    monkeypatch.setattr(maintenance, "run_engine_maintenance", lambda conn, owner, **kwargs: report)
    assert coordinator.request_layers(threshold=25)
    assert coordinator.wait(10)
    assert coordinator.status == ("success", None)

    last = {}
    release = _blocked_worker(coordinator, last)
    monkeypatch.setattr(maintenance, "_CLOSE_WAIT_SECONDS", 0.0, raising=False)
    second.close()
    release.set()
    assert coordinator.wait(10)
    assert last == {"write": "refused"}
    assert _rows(store) == 1
    # Nothing starts after the last close until an Engine opens the database again.
    assert not coordinator.request_layers(threshold=25)
