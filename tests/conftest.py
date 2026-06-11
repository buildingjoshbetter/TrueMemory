"""Shared pytest fixtures — test isolation for the whole suite.

Issue #426 (flaky CI): several tests mutate process-global state that leaks
into later tests *and* into the subprocesses spawned by the CLI tests
(``tests/test_cli_help.py`` runs ``truememory-*`` via ``subprocess`` and
inherits ``os.environ``). The classic symptom is
``test_cli_help::test_ingest_version_flag_exits_cleanly`` passing in isolation
but failing in full-suite order, and ``tests/test_upgrade_path.py`` depending
on the ambient ``~/.truememory/config.json`` tier (which decides
``vector_search.EMBEDDING_MODEL`` at import time).

Two autouse fixtures restore determinism without weakening any assertion:

1. ``_isolate_environ`` snapshots ``os.environ`` before each test and restores
   it after, so direct ``os.environ[...] = ...`` writes (which are NOT
   monkeypatch and therefore not auto-reverted) cannot leak across tests or
   into spawned subprocesses.
2. ``_isolate_vector_search_globals`` snapshots the module-level embedder
   globals (``EMBEDDING_MODEL``, ``_embedding_dim``, ``_model``) and restores
   them after each test, so a test that switches tiers does not poison the
   next one.
"""
from __future__ import annotations

import os
import sqlite3

import pytest

# V-cigate-1 (#697): offline mode is set ONCE here, in the single conftest that
# pytest imports before any test module, instead of each test module doing its
# own module-level os.environ.setdefault("HF_HUB_OFFLINE", "1"). Those scattered
# import-time mutations were the §3.5 class behind the original network-tests
# leak (#654): pytest imports every module during collection, so an offline
# setdefault leaked into the online job. setdefault (not [...]=) so the
# network-tests job's explicit HF_HUB_OFFLINE="0" still wins. A guard test
# (test_ci_isolation) fails if a test module reintroduces the antipattern.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _can_load_sqlite_vec() -> bool:
    """True if sqlite-vec can be loaded into a connection."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(conn)
        return True
    except (AttributeError, ImportError, OSError):
        return False
    finally:
        conn.close()


can_load_extensions = _can_load_sqlite_vec()

requires_sqlite_ext = pytest.mark.skipif(
    not can_load_extensions,
    reason="sqlite-vec not available (missing enable_load_extension or sqlite_vec package)",
)


@pytest.fixture(scope="session", autouse=True)
def _isolate_truememory_home(tmp_path_factory):
    """Point the model-server socket/state and ``$HOME`` at a session tmp dir.

    Issue #654 (M-54): ``model_client.SOCK_PATH`` / ``model_server.SOCK_PATH``
    are computed at import time from ``Path.home() / ".truememory"`` — i.e. the
    contributor's REAL user-global socket. Without isolation, a test that
    probes ``_server_ready()`` / autostart can connect to (or stomp on)
    whatever live daemon the developer is running, producing order- and
    machine-dependent flakes.

    This session-scoped autouse fixture redirects the module-level path
    attributes on both modules at a hermetic tmp ``.truememory`` dir and sets
    ``$HOME`` so any *new* code that recomputes ``Path.home()`` also lands in
    the sandbox. Per-test ``patch.object(..., "SOCK_PATH", ...)`` calls still
    override these baselines within their own scope, so existing socket tests
    are unaffected.
    """
    home = tmp_path_factory.mktemp("tm_home")
    tm_dir = home / ".truememory"
    tm_dir.mkdir(parents=True, exist_ok=True)

    # Set HOME first so it is captured by the per-test ``_isolate_environ``
    # snapshot and therefore persists across the whole session.
    os.environ["HOME"] = str(home)

    for mod_name in ("truememory.model_client", "truememory.model_server"):
        try:
            mod = __import__(mod_name, fromlist=["_TRUEMEMORY_DIR"])
        except Exception:
            continue
        if hasattr(mod, "_TRUEMEMORY_DIR"):
            mod._TRUEMEMORY_DIR = tm_dir
        for attr, name in (
            ("SOCK_PATH", "model.sock"),
            ("PID_PATH", "model_server.pid"),
            ("PORT_PATH", "model_server.port"),
            ("TOKEN_PATH", "model_server.token"),
            ("LOCK_PATH", "model_server.lock"),
        ):
            if hasattr(mod, attr):
                setattr(mod, attr, tm_dir / name)
    yield


@pytest.fixture(autouse=True)
def _isolate_environ():
    """Snapshot/restore ``os.environ`` around every test.

    Plain ``os.environ[...] = ...`` writes in some env-driven tests are not
    monkeypatch-managed, so without this they persist for the rest of the
    session and are inherited by subprocess-based CLI tests.
    """
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


@pytest.fixture(autouse=True)
def _isolate_vector_search_globals():
    """Snapshot/restore ``truememory.vector_search`` embedder globals.

    These are process-global (module-level) and several tests mutate them via
    ``set_embedding_model`` or ``monkeypatch.setattr``. monkeypatch reverts the
    ones it owns, but a defensive snapshot here keeps the embedder identity
    deterministic regardless of test order or ambient config.json.
    """
    try:
        from truememory import vector_search as _vs
    except Exception:
        # vector_search may be unimportable in minimal-dep environments; nothing
        # to isolate in that case.
        yield
        return

    saved = (
        getattr(_vs, "EMBEDDING_MODEL", None),
        getattr(_vs, "_embedding_dim", None),
        getattr(_vs, "_model", None),
    )
    try:
        yield
    finally:
        _vs.EMBEDDING_MODEL, _vs._embedding_dim, _vs._model = saved


@pytest.fixture(autouse=True)
def _isolate_model_client_timeout():
    """Snapshot/restore ``truememory.model_client._default_request_timeout``.

    Hook recall paths arm a process-wide model-server deadline via
    ``set_request_timeout`` (issue #577). In production the hooks are
    short-lived standalone processes, but in the test suite any test that
    exercises a recall path would otherwise leak the 5s deadline into later
    tests that assert the legacy 120s autostart-retry behavior.
    """
    try:
        from truememory import model_client
    except Exception:
        yield
        return

    saved = model_client._default_request_timeout
    try:
        yield
    finally:
        model_client._default_request_timeout = saved
