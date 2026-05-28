"""Tests for the opt-in telemetry instrumentation overlay.

Covers the two halves of the contract:

1. With ``TRUEMEMORY_INSTRUMENTATION=1`` set, ``install()`` monkey-patches the
   engine and the documented signals land in the ``telemetry`` table:
     - a gate evaluation  -> gate_decision + salience + surprise rows
     - a search           -> search_distance + memory_returned rows
     - a delete           -> user_forget row
     - the table schema matches the dashboard-mirror contract
     - install() emits an instrumentation_start bootstrap row

2. With the env var UNSET, ``install()`` is a no-op: no methods are patched and
   the ``telemetry`` table is never created.

Reload-safety: other test modules in the suite call ``importlib.reload`` on
engine / encoding-gate modules, which swaps the *class objects*. So this file
NEVER caches class references at import time. Every fixture and test resolves
``Memory`` / ``EncodingGate`` / ``TrueMemoryEngine`` LIVE from their modules and
snapshots the pristine method of whatever class is current. That keeps the
assertions correct regardless of suite ordering — the class ``install()``
patches is always the same class object the test inspects.
"""
from __future__ import annotations

import importlib
import sqlite3

import pytest

from truememory import Memory
from truememory.instrumentation import patch as patch_mod
from truememory.instrumentation import writer as writer_mod
from truememory.instrumentation import log as log_mod

# Method names install() may replace, keyed by the module + attribute that owns
# them. Resolved live (never cached) so a reload elsewhere can't desync us.
_PATCH_TARGETS = (
    ("truememory.client", "Memory", ("add", "search", "search_deep", "get", "delete")),
    ("truememory.ingest.encoding_gate", "EncodingGate", ("evaluate",)),
    ("truememory.engine", "TrueMemoryEngine", ("add", "_ensure_connection")),
)


def _current_class(module_name: str, class_name: str):
    """Return the class object that ``install()`` will actually patch right now."""
    return getattr(importlib.import_module(module_name), class_name)


def _snapshot_pristine() -> dict[tuple[str, str], object]:
    """Snapshot the current method objects for every patch target, live."""
    snap: dict[tuple[str, str], object] = {}
    for module_name, class_name, methods in _PATCH_TARGETS:
        cls = _current_class(module_name, class_name)
        for meth in methods:
            if hasattr(cls, meth):
                snap[(class_name, meth)] = getattr(cls, meth)
    return snap


def _restore(snapshot: dict[tuple[str, str], object]) -> None:
    """Restore every snapshotted method onto its current class + reset state."""
    for module_name, class_name, methods in _PATCH_TARGETS:
        cls = _current_class(module_name, class_name)
        for meth in methods:
            key = (class_name, meth)
            if key in snapshot:
                setattr(cls, meth, snapshot[key])
    patch_mod._installed = False
    writer_mod.close()  # drop any cached connection so a fresh DB path is honored


@pytest.fixture
def clean_overlay():
    """Snapshot pristine methods, guarantee an unpatched + latch-reset overlay
    before and after each test. Yields the pristine snapshot so tests can do
    identity comparisons against the exact objects they started from."""
    snapshot = _snapshot_pristine()
    _restore(snapshot)
    yield snapshot
    _restore(snapshot)


@pytest.fixture
def enabled_db(tmp_path, monkeypatch, clean_overlay):
    """Enable the overlay and point both the engine and the telemetry writer at
    one shared temp memories.db. Returns (db_path, pristine_snapshot)."""
    db_path = tmp_path / "memories.db"
    monkeypatch.setenv("TRUEMEMORY_INSTRUMENTATION", "1")
    monkeypatch.setenv("TRUEMEMORY_DB_PATH", str(db_path))
    assert log_mod.is_enabled() is True
    return db_path, clean_overlay


def _read_signals(db_path) -> dict[str, int]:
    """Return {signal: count} from the telemetry table, or {} if no table."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "telemetry" not in tables:
            return {}
        rows = conn.execute(
            "SELECT signal, COUNT(*) FROM telemetry GROUP BY signal"
        ).fetchall()
        return {signal: count for signal, count in rows}
    finally:
        conn.close()


def _telemetry_columns(db_path) -> list[str]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return [row[1] for row in conn.execute("PRAGMA table_info(telemetry)")]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# ENABLED — install() patches and signals land
# ---------------------------------------------------------------------------


def test_install_patches_methods(enabled_db):
    """install() replaces the targeted class methods when enabled."""
    _, pristine = enabled_db
    patch_mod.install()
    # Compare against the live classes install() actually patched.
    gate = _current_class("truememory.ingest.encoding_gate", "EncodingGate")
    mem = _current_class("truememory.client", "Memory")
    eng = _current_class("truememory.engine", "TrueMemoryEngine")
    assert gate.evaluate is not pristine[("EncodingGate", "evaluate")]
    assert mem.search is not pristine[("Memory", "search")]
    assert mem.delete is not pristine[("Memory", "delete")]
    assert eng.add is not pristine[("TrueMemoryEngine", "add")]


def test_install_is_idempotent(enabled_db):
    """A second install() does not re-wrap (latch holds)."""
    patch_mod.install()
    gate = _current_class("truememory.ingest.encoding_gate", "EncodingGate")
    after_first = gate.evaluate
    patch_mod.install()
    assert gate.evaluate is after_first


def test_install_emits_bootstrap_row(enabled_db):
    """install() writes an instrumentation_start row, creating the table."""
    db_path, _ = enabled_db
    patch_mod.install()
    signals = _read_signals(db_path)
    assert signals.get("instrumentation_start", 0) >= 1


def test_telemetry_schema_matches_contract(enabled_db):
    """The telemetry table columns are byte-for-byte the dashboard contract."""
    db_path, _ = enabled_db
    patch_mod.install()
    cols = _telemetry_columns(db_path)
    assert cols == [
        "id",
        "ts",
        "pid",
        "signal",
        "memory_id",
        "value_num",
        "value_text",
        "context_json",
    ]


def test_telemetry_indices_exist(enabled_db):
    """Both documented indices are created."""
    db_path, _ = enabled_db
    patch_mod.install()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        idx = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
    finally:
        conn.close()
    assert "idx_telemetry_signal_ts" in idx
    assert "idx_telemetry_memory_id" in idx


def test_gate_eval_emits_gate_salience_surprise(enabled_db):
    """A gate evaluation emits gate_decision + salience + surprise."""
    db_path, _ = enabled_db
    patch_mod.install()

    EncodingGate = _current_class("truememory.ingest.encoding_gate", "EncodingGate")

    class _FakeMemory:
        def search(self, query, limit=5, user_id=None):
            return []

    gate = EncodingGate(_FakeMemory(), threshold=0.30)
    decision = gate.evaluate("Alice moved to Seattle in March 2026", category="personal")
    # The wrapped evaluate must still return the genuine decision unchanged.
    assert hasattr(decision, "should_encode")
    assert hasattr(decision, "salience")

    signals = _read_signals(db_path)
    assert signals.get("gate_decision", 0) >= 1
    assert signals.get("salience", 0) >= 1
    assert signals.get("surprise", 0) >= 1


def test_search_emits_distance_and_returned(enabled_db):
    """A search that returns hits emits search_distance + memory_returned."""
    db_path, _ = enabled_db
    patch_mod.install()
    m = Memory(path=str(db_path))
    try:
        m.add("Alice likes espresso in the morning", user_id="alice")
        m.add("Alice enjoys a flat white after lunch", user_id="alice")
        results = m.search("coffee espresso", user_id="alice")
    finally:
        m.close()

    signals = _read_signals(db_path)
    # search_distance fires once per search call (even if 0 results).
    assert signals.get("search_distance", 0) >= 1
    # memory_returned fires once per returned hit; we stored two relevant rows.
    if results:
        assert signals.get("memory_returned", 0) >= 1


def test_delete_emits_user_forget(enabled_db):
    """An explicit delete emits a user_forget row."""
    db_path, _ = enabled_db
    patch_mod.install()
    m = Memory(path=str(db_path))
    try:
        added = m.add("Temporary fact to be forgotten", user_id="alice")
        assert added["id"] is not None
        m.delete(added["id"])
    finally:
        m.close()

    signals = _read_signals(db_path)
    assert signals.get("user_forget", 0) >= 1


def test_engine_add_emits_category(enabled_db):
    """Storing a memory with a category emits a category row."""
    db_path, _ = enabled_db
    patch_mod.install()
    m = Memory(path=str(db_path))
    try:
        # Drive engine.add directly with a category so the category signal fires
        # (Memory.add does not forward a category, but engine.add accepts one).
        m._engine.add(
            content="Bob switched jobs to Acme Corp",
            sender="bob",
            category="decision",
        )
    finally:
        m.close()

    signals = _read_signals(db_path)
    assert signals.get("category", 0) >= 1


# ---------------------------------------------------------------------------
# DISABLED — install() is a no-op
# ---------------------------------------------------------------------------


def test_disabled_install_is_noop(tmp_path, monkeypatch, clean_overlay):
    """With the env var unset, install() patches nothing and writes nothing."""
    pristine = clean_overlay
    db_path = tmp_path / "memories.db"
    monkeypatch.delenv("TRUEMEMORY_INSTRUMENTATION", raising=False)
    monkeypatch.setenv("TRUEMEMORY_DB_PATH", str(db_path))
    assert log_mod.is_enabled() is False

    patch_mod.install()

    # No methods were replaced (compare against the live classes).
    gate = _current_class("truememory.ingest.encoding_gate", "EncodingGate")
    mem = _current_class("truememory.client", "Memory")
    eng = _current_class("truememory.engine", "TrueMemoryEngine")
    assert mem.search is pristine[("Memory", "search")]
    assert mem.delete is pristine[("Memory", "delete")]
    assert gate.evaluate is pristine[("EncodingGate", "evaluate")]
    assert eng.add is pristine[("TrueMemoryEngine", "add")]


def test_disabled_no_telemetry_table(tmp_path, monkeypatch, clean_overlay):
    """With the env var unset, ordinary engine use never creates a telemetry table."""
    db_path = tmp_path / "memories.db"
    monkeypatch.delenv("TRUEMEMORY_INSTRUMENTATION", raising=False)
    monkeypatch.setenv("TRUEMEMORY_DB_PATH", str(db_path))

    patch_mod.install()  # no-op

    m = Memory(path=str(db_path))
    try:
        added = m.add("A fact stored with instrumentation off", user_id="alice")
        m.search("fact", user_id="alice")
        if added["id"] is not None:
            m.delete(added["id"])
    finally:
        m.close()

    # The DB file exists (engine created it) but holds no telemetry table.
    assert db_path.exists()
    assert _read_signals(db_path) == {}


def test_disabled_install_does_not_trip_latch(tmp_path, monkeypatch, clean_overlay):
    """A disabled install() must not burn the idempotency latch — a later
    enabled install() in the same process still patches."""
    pristine = clean_overlay
    db_path = tmp_path / "memories.db"
    monkeypatch.delenv("TRUEMEMORY_INSTRUMENTATION", raising=False)
    monkeypatch.setenv("TRUEMEMORY_DB_PATH", str(db_path))
    patch_mod.install()  # disabled no-op
    assert patch_mod._installed is False

    # Now enable and install for real.
    monkeypatch.setenv("TRUEMEMORY_INSTRUMENTATION", "1")
    patch_mod.install()
    assert patch_mod._installed is True
    gate = _current_class("truememory.ingest.encoding_gate", "EncodingGate")
    assert gate.evaluate is not pristine[("EncodingGate", "evaluate")]
