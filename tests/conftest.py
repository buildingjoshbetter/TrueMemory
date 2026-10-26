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
import sys
import threading
import weakref

import pytest


def _establish_test_home() -> object:
    """Isolate HOME before any truememory import; guard the real home.

    Modules capture paths and read the tier config at import, and pytest
    imports every test module during collection, before any fixture runs.
    A session fixture therefore set HOME too late: vector_search read the
    user's real ~/.truememory/config.json at collection. HOME is set here,
    at conftest import, and subprocesses inherit it. The Hugging Face cache
    (read offline) and the user site stay where they were.

    ``tests.conftest`` imports load this file a second time, so the state
    lives on ``sys`` and is created once per process.
    """
    state = getattr(sys, "_truememory_test_isolation", None)
    if state is not None:
        return state
    import atexit
    import shutil
    import site
    import tempfile
    import types

    homes = {os.path.realpath(os.path.expanduser("~"))}
    try:
        import pwd
        homes.add(os.path.realpath(pwd.getpwuid(os.getuid()).pw_dir))
    except (ImportError, KeyError, OSError):
        pass
    protected = tuple(sorted(os.path.join(home, name) for home in homes
                             for name in (".truememory", ".claude", ".codex")))
    real_cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    os.environ.setdefault("HF_HOME", os.path.join(real_cache, "huggingface"))
    os.environ.setdefault("PYTHONUSERBASE", site.getuserbase())
    home = tempfile.mkdtemp(prefix="tmh-")
    atexit.register(shutil.rmtree, home, True)
    os.environ["HOME"] = home
    for name, relative in (("XDG_CONFIG_HOME", ".config"), ("XDG_CACHE_HOME", ".cache"),
                           ("XDG_DATA_HOME", ".local/share"), ("XDG_STATE_HOME", ".local/state")):
        os.environ[name] = os.path.join(home, relative)
    state = types.SimpleNamespace(home=home, protected=protected if os.name != "nt" else (),
                                  touches=[], phase="collection")
    sys._truememory_test_isolation = state
    sys.addaudithook(_guard_real_home)
    return state


# Audited operations that read, write, list or connect to a path.
_PATH_EVENTS = {
    "open": 1, "sqlite3.connect": 1, "os.listdir": 1, "os.scandir": 1, "os.mkdir": 1,
    "os.rmdir": 1, "os.remove": 1, "os.chmod": 1, "os.chown": 1, "os.truncate": 1,
    "os.utime": 1, "shutil.rmtree": 1, "os.rename": 2, "os.link": 2, "os.symlink": 2,
    "shutil.copyfile": 2, "shutil.copytree": 2, "shutil.move": 2,
}


def _protected_path(value: object, protected: tuple[str, ...]) -> str | None:
    if isinstance(value, int) or value is None:
        return None
    try:
        path = os.fsdecode(os.fspath(value))
    except TypeError:
        return None
    if path.startswith("file:"):
        path = path[len("file:"):].split("?", 1)[0]
    if not path or path == ":memory:":
        return None
    path = os.path.abspath(path)
    for root in protected:
        if path == root or path.startswith(root + os.sep):
            return path
    return None


def _guard_real_home(event: str, args: tuple) -> None:
    state = getattr(sys, "_truememory_test_isolation", None)
    if state is None or not state.protected:
        return
    if event in ("socket.connect", "socket.bind"):
        values = args[1:2] if len(args) > 1 and isinstance(args[1], (str, bytes)) else ()
    else:
        count = _PATH_EVENTS.get(event)
        if count is None:
            return
        values = args[:count]
    for value in values:
        path = _protected_path(value, state.protected)
        if path is not None:
            # Refuse before the operation runs, and record it in case the
            # caller swallows the error: the per-test check fails the test.
            state.touches.append((state.phase, event, path))
            raise PermissionError(f"Test isolation: {event} on {path} is inside the real home")


_ISOLATION = _establish_test_home()
TEST_HOME = _ISOLATION.home

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


@pytest.fixture(autouse=True)
def _forbid_real_home(request):
    """Fail any test that opens, writes or connects under the real home."""
    start = len(_ISOLATION.touches)
    _ISOLATION.phase = request.node.nodeid
    try:
        yield
    finally:
        _ISOLATION.phase = "between tests"
    touched = _ISOLATION.touches[start:]
    if touched:
        pytest.fail("Test touched the real home: " + "; ".join(f"{event} {path}" for _, event, path in touched))


@pytest.fixture(scope="session", autouse=True)
def _isolate_truememory_home():
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
    from pathlib import Path

    # HOME was isolated at conftest import; keep this session on the same dir.
    home = Path(TEST_HOME)
    tm_dir = home / ".truememory"
    tm_dir.mkdir(parents=True, exist_ok=True)
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
            ("START_LOCK_PATH", "model_server.start.lock"),
            ("START_STATE_PATH", "model_server.start.json"),
        ):
            if hasattr(mod, attr):
                setattr(mod, attr, tm_dir / name)
        if mod_name == "truememory.model_client":
            mod._APP_BUNDLE_PATH = tm_dir / "TrueMemory.app"
            mod._APP_EXECUTABLE = mod._APP_BUNDLE_PATH / "Contents" / "MacOS" / "TrueMemory"
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
    """Snapshot/restore the complete process-local serving identity.

    These are process-global (module-level) and several tests mutate them via
    ``set_embedding_model`` or ``monkeypatch.setattr``. monkeypatch reverts the
    ones it owns, but policy projection also changes embedding/reranker identity
    and runtime acknowledgement. Restore those fields together without resetting
    serving leases, runtime locals or background workers.
    """
    try:
        from truememory import vector_search as _vs
        from truememory import reranker as _rr
        from truememory.tier_switch import runtime as _runtime
    except Exception:
        # vector_search may be unimportable in minimal-dep environments; nothing
        # to isolate in that case.
        yield
        return

    saved_vector = (
        getattr(_vs, "EMBEDDING_MODEL", None),
        getattr(_vs, "_embedding_dim", None),
        getattr(_vs, "_model", None),
        getattr(_vs, "_model_generation", 0),
        getattr(_vs, "_frozen_embedding_target", None),
        getattr(_vs, "_runtime_policy_tier", None),
    )
    saved_reranker = (_rr._model, _rr._model_name, _rr._model_certified,
                      _rr._active_tier, _rr._frozen_reranker_id)
    saved_acknowledgement = _runtime._acknowledged
    try:
        yield
    finally:
        (_vs.EMBEDDING_MODEL, _vs._embedding_dim, _vs._model, _vs._model_generation,
         _vs._frozen_embedding_target, _vs._runtime_policy_tier) = saved_vector
        (_rr._model, _rr._model_name, _rr._model_certified,
         _rr._active_tier, _rr._frozen_reranker_id) = saved_reranker
        _runtime._acknowledged = saved_acknowledgement


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


# Opening and closing a file that a live SQLite connection in this process also
# has open drops every POSIX lock SQLite holds on it; on SQLite 3.47/3.50 another
# process's close then deleted the WAL still in use (GPUBox D1/D2). Product code
# must reach the database, -wal and -shm only through SQLite. One process-wide
# audit hook (hooks cannot be removed) records the databases this process
# connects to and every Python-level open of them from truememory/ code; the
# fixture below fails the test that did it.
_connected_databases: set[str] = set()
_database_file_opens: list[str] = []
_PRODUCT = os.sep + "truememory" + os.sep

# Every connection product code opens must pass through
# storage.configure_connection (no automatic checkpoints on SQLite builds with
# the WAL-reset bug) unless it never writes. A connection opened from
# truememory/ stays pending until configure_connection's audit event names it;
# the fixture below fails the test that left one pending.
_unconfigured_connections: dict[int, tuple[str, "weakref.ref | None"]] = {}
_opening_database = threading.local()
_RAW_CONNECTIONS = {  # (module, function): writers that are not TrueMemory stores.
    ("storage.py", "protected_checkpoint"),  # The guard's own lock holder and checkpointer.
    ("storage.py", "_backup_database"),  # The new backup file; its only connection.
    (os.path.join("tier_switch", "manager.py"), "backup_db"),  # Likewise.
    (os.path.join("ingest", "cli.py"), "_preflight_writable_target"),  # PRAGMA quick_check only.
}


def _product_caller():
    import sys
    frame = sys._getframe(2)
    while frame is not None and frame.f_code.co_filename.endswith(("conftest.py", "<frozen importlib._bootstrap>")):
        frame = frame.f_back
    if frame is None or _PRODUCT not in frame.f_code.co_filename or os.sep + "tests" + os.sep in frame.f_code.co_filename:
        return None
    return frame


def _database_target(database: object) -> str | None:
    try:
        name = os.fsdecode(database)
    except TypeError:
        return None
    if name.startswith("file:"):
        from urllib.parse import unquote, urlsplit
        parts = urlsplit(name)
        name = unquote(parts.path) if parts.scheme == "file" else ""
    if not name or name == ":memory:" or name.startswith(":"):
        return None
    return os.path.realpath(name)


def _audit_database_files(event: str, args: tuple) -> None:
    if event == "sqlite3.connect":
        target = _database_target(args[0]) if args else None
        if target is not None:
            _connected_databases.add(target)
        name = os.fsdecode(args[0]) if args and isinstance(args[0], (str, bytes, os.PathLike)) else ""
        _opening_database.writable = target is not None and "mode=ro" not in name
    elif event == "sqlite3.connect/handle" and args and getattr(_opening_database, "writable", False):
        _opening_database.writable = False
        frame = _product_caller()
        if frame is not None:
            module = frame.f_code.co_filename.rsplit(_PRODUCT, 1)[1]
            if (module, frame.f_code.co_name) not in _RAW_CONNECTIONS:
                try:
                    handle = weakref.ref(args[0])
                except TypeError:  # A plain sqlite3.Connection takes no weak reference.
                    handle = None
                _unconfigured_connections[id(args[0])] = (f"{frame.f_code.co_filename}:{frame.f_lineno}", handle)
    elif event == "truememory.configure_connection" and args:
        _unconfigured_connections.pop(id(args[0]), None)
    elif event == "open" and _connected_databases and args and isinstance(args[0], (str, bytes, os.PathLike)):
        name = os.fsdecode(args[0])
        base = name[:-4] if name.endswith(("-wal", "-shm")) else name
        if os.path.basename(base) not in {os.path.basename(path) for path in _connected_databases}:
            return
        if os.path.realpath(base) not in _connected_databases:
            return
        import sys
        frame = sys._getframe(1)
        while frame is not None and frame.f_code.co_filename.endswith(("conftest.py", "<frozen importlib._bootstrap>")):
            frame = frame.f_back
        if frame is not None and _PRODUCT in frame.f_code.co_filename and os.sep + "tests" + os.sep not in frame.f_code.co_filename:
            _database_file_opens.append(f"{name} from {frame.f_code.co_filename}:{frame.f_lineno}")


import sys as _sys  # noqa: E402
_sys.addaudithook(_audit_database_files)


def _still_open(conn: object) -> bool:
    if conn is None:
        return False
    try:
        conn.total_changes  # type: ignore[attr-defined]
    except sqlite3.ProgrammingError:
        return False
    return True


@pytest.fixture(autouse=True)
def _forbid_database_file_opens_outside_sqlite():
    """Fail any test whose product code opened a connected database's files directly."""
    _connected_databases.clear()
    del _database_file_opens[:]
    _unconfigured_connections.clear()
    yield
    found = list(_database_file_opens)
    del _database_file_opens[:]
    # A background run fenced by a close before its connection was configured
    # closes that connection at once; only a connection still open counts.
    unconfigured = sorted({location for location, handle in _unconfigured_connections.values()
                           if handle is None or _still_open(handle())})
    _unconfigured_connections.clear()
    if found:
        pytest.fail("Database files opened outside SQLite (drops this process's POSIX locks): "
                    + "; ".join(found[:5]), pytrace=False)
    if unconfigured:
        pytest.fail("Database connections opened without storage.configure_connection: "
                    + "; ".join(unconfigured[:5]), pytrace=False)
