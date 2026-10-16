"""The suite never reads, writes or connects inside the developer's real home.

Running the suite on a developer machine must not touch the live store in
~/.truememory, nor ~/.claude or ~/.codex. conftest isolates HOME at import,
before collection imports truememory, and an audit hook refuses and records
any access under the real home. These tests fail on a tree that isolates HOME
only from a session fixture: vector_search then reads the real
~/.truememory/config.json while pytest collects test modules.
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ISOLATION = getattr(sys, "_truememory_test_isolation", None)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="The real-home guard covers POSIX homes")


def _without_recorded(start: int) -> None:
    # This test touched the real home on purpose; the guard refused it.
    del ISOLATION.touches[start:]


def test_isolation_is_established_at_conftest_import():
    assert ISOLATION is not None, "conftest must isolate HOME before collection"
    assert ISOLATION.protected and all(not path.startswith(ISOLATION.home) for path in ISOLATION.protected)
    assert os.environ["HOME"] == ISOLATION.home == str(Path.home())


def test_collection_never_touched_the_real_home():
    touched = [touch for touch in ISOLATION.touches if touch[0] == "collection"]
    assert touched == [], f"Importing test modules touched the real home: {touched}"


def test_import_time_paths_resolve_inside_the_isolated_home():
    from truememory import client, mcp_server, vector_search
    from truememory.hooks import core
    from truememory.ingest.hooks import _shared

    for path in (client._DEFAULT_DB, mcp_server._TRUEMEMORY_DIR, Path(mcp_server._DB_PATH), core.BACKLOG_DIR,
                 _shared.RECALL_CACHE_PATH):
        # Tests may repoint these to their own temp dirs; none may reach the real home.
        assert not any(os.path.abspath(path).startswith(root) for root in ISOLATION.protected), path
    # The tier read at import came from the isolated home, not the user's config.
    assert vector_search._raw_env == "edge"


def test_real_home_access_is_refused_before_it_happens():
    start = len(ISOLATION.touches)
    target = os.path.join(ISOLATION.protected[0], "isolation-probe-never-created")
    try:
        with pytest.raises(PermissionError, match="real home"):
            open(target, "rb")
        with pytest.raises(PermissionError, match="real home"):
            sqlite3.connect(target)
        with pytest.raises(PermissionError, match="real home"):
            sqlite3.connect(f"file:{target}?mode=ro", uri=True)
        assert [event for _, event, _ in ISOLATION.touches[start:]] == ["open", "sqlite3.connect", "sqlite3.connect"]
    finally:
        _without_recorded(start)
    assert not os.path.exists(target)


def test_subprocesses_inherit_the_isolated_home():
    probe = ("import json, os, pathlib, site; print(json.dumps([str(pathlib.Path.home()), os.environ['HF_HOME'], "
             "site.getusersitepackages()]))")
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, check=True)
    import json
    home, hf_home, user_site = json.loads(result.stdout)
    assert home == ISOLATION.home
    assert not hf_home.startswith(ISOLATION.home)  # the offline model cache stays readable
    assert not user_site.startswith(ISOLATION.home)  # user-installed packages stay importable


def test_a_developer_run_imports_truememory_inside_the_isolated_home():
    """Replay collection order with the developer's real HOME: conftest, then truememory."""
    real_home = os.path.dirname(ISOLATION.protected[0])
    root = str(Path(__file__).resolve().parents[1])
    probe = ("import json, sys; sys.path.insert(0, sys.argv[1]); import tests.conftest; "
             "import truememory.vector_search as vs; "
             "print(json.dumps([sys._truememory_test_isolation.touches, vs._raw_env]))")
    env = {**os.environ, "HOME": real_home}
    for name in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"):
        env.pop(name, None)
    result = subprocess.run([sys.executable, "-c", probe, root], capture_output=True, text=True,
                            timeout=120, env=env, cwd=root)
    assert result.returncode == 0, result.stderr[-2000:]
    import json
    touches, tier = json.loads(result.stdout.splitlines()[-1])
    assert touches == [], f"Collection-order imports touched the real home: {touches}"
    assert tier == "edge"
