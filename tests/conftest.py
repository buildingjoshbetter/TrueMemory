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

import pytest


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
