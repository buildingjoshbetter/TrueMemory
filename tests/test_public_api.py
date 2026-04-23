"""Regression lock for Hunter F37 — `__all__` must enumerate the actual
re-exports in `truememory/__init__.py`.

Pre-fix, `__all__` declared 3 symbols while the package re-exported 79;
IDE auto-import, Sphinx autodoc, and `from truememory import *` all saw
a misleadingly small public API. This file locks the expanded list in
place so future re-exports can't silently drift again.

The contract:
- Every non-underscore name exposed at the top of `truememory` is in
  `__all__`.
- `__all__` has no dead entries (every name resolves).
- `from truememory import *` surfaces exactly the set declared by `__all__`.
"""
from __future__ import annotations


def _public_names_of_truememory() -> set[str]:
    """Names available on the top-level `truememory` module that are
    treated as public (don't start with an underscore).

    Python's `__version__` convention lives in `__all__` but is filtered
    out by this helper because it starts with `_`. Tests handle it
    explicitly.

    Uses a fresh subprocess so other tests that imported `truememory.ingest`
    or `truememory.mcp_server` don't pollute the `dir()` (Python registers
    submodule imports on the parent package — a real "fresh install"
    `import truememory` doesn't see them until they're explicitly accessed).
    """
    import subprocess
    import sys
    import json as _json

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, truememory\n"
                "print(json.dumps([n for n in dir(truememory) "
                "if not n.startswith('_')]))"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"subprocess failed: {result.stderr}"
    )
    return set(_json.loads(result.stdout.strip()))


def test_all_is_declared_as_a_list_of_strings():
    import truememory
    assert hasattr(truememory, "__all__")
    assert isinstance(truememory.__all__, list)
    assert all(isinstance(n, str) for n in truememory.__all__)
    # No duplicates (would mask drift during maintenance)
    assert len(truememory.__all__) == len(set(truememory.__all__)), (
        "F37 regression: duplicate entries in __all__"
    )


def test_all_entries_resolve_to_real_attributes():
    """Every name in `__all__` must actually exist on the module —
    dead entries would confuse autodoc and IDE tooling."""
    import truememory
    missing = [n for n in truememory.__all__ if not hasattr(truememory, n)]
    assert missing == [], (
        f"F37 regression: __all__ declares names that don't exist on the "
        f"module: {missing}"
    )


def test_no_public_drift_between_dir_and_all():
    """Every non-underscore name in `dir(truememory)` must be in `__all__`.

    This is the primary F37 invariant. If a contributor adds
    `from truememory.newmod import frob` to `__init__.py`, they must
    also add `"frob"` to `__all__`, or this test fails loudly.
    """
    import truememory
    public = _public_names_of_truememory()
    declared = set(truememory.__all__)
    # `__version__` is in __all__ but not in `public` because it starts
    # with underscore. That's the ONE exception.
    drift_in_public_not_declared = public - declared
    assert drift_in_public_not_declared == set(), (
        f"F37 regression: {len(drift_in_public_not_declared)} public name(s) "
        f"exposed by truememory/__init__.py are NOT in __all__. Either add "
        f"them to __all__ or move the import out of __init__.py. "
        f"Missing: {sorted(drift_in_public_not_declared)}"
    )


def test_all_minus_version_equals_public_dir_set():
    """The inverse check: `__all__` must not declare names that don't
    actually appear in `dir()` (bar `__version__`). A declared-but-
    not-exposed name means a typo or a stale entry."""
    import truememory
    public = _public_names_of_truememory()
    declared = set(truememory.__all__)
    declared_minus_version = declared - {"__version__"}
    extra = declared_minus_version - public
    assert extra == set(), (
        f"F37 regression: __all__ declares non-underscore names that aren't "
        f"actually exposed: {sorted(extra)}"
    )


def test_star_import_surfaces_exactly_all():
    """`from truememory import *` must surface exactly the `__all__` set.

    Runs in a subprocess so other tests' lazy submodule imports
    (`truememory.ingest`, `truememory.mcp_server`) don't pollute the
    result — `import *` would pick those up otherwise even though they're
    intentionally not in `__all__`.
    """
    import subprocess
    import sys
    import json as _json
    import truememory

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, truememory\n"
                "ns = {}\n"
                "exec('from truememory import *', ns)\n"
                "print(json.dumps([k for k in ns if k != '__builtins__']))"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    star = set(_json.loads(result.stdout.strip()))
    expected = set(truememory.__all__)
    assert star == expected, (
        f"F37 regression: `from truememory import *` surfaced {len(star)} "
        f"names but __all__ has {len(expected)}. "
        f"Surfaced-not-declared: {sorted(star - expected)}. "
        f"Declared-not-surfaced: {sorted(expected - star)}."
    )


def test_core_symbols_present():
    """Sanity: the three originally-declared names must still be in `__all__`."""
    import truememory
    for n in ("__version__", "Memory", "TrueMemoryEngine"):
        assert n in truememory.__all__, f"F37 regression: {n!r} dropped from __all__"


def test_submodules_accessible_via_star_import():
    """Submodule names were listed in __all__ specifically so that
    `from truememory import vector_search` works under `import *`
    semantics. Verify those submodules resolve to the expected packages."""
    import truememory
    submodules = [
        "client", "engine", "storage", "vector_search", "fts_search",
        "hybrid", "temporal", "salience", "personality", "consolidation",
        "predictive", "query_classifier", "reranker", "hyde", "clustering",
    ]
    for m in submodules:
        assert m in truememory.__all__
        assert hasattr(truememory, m)
        mod = getattr(truememory, m)
        assert mod.__name__ == f"truememory.{m}", (
            f"F37 regression: truememory.{m} doesn't resolve to the package"
        )


def test_ingest_submodule_is_intentionally_lazy():
    """`truememory.ingest` is NOT in `__all__` — it's a lazy import via
    `__getattr__`. Confirm the public contract: attribute access works,
    but `import *` does NOT pull it in."""
    import truememory
    # Access via attribute works (triggers __getattr__ lazy-load)
    ingest = truememory.ingest
    assert ingest.__name__ == "truememory.ingest"
    # But it's NOT in __all__, so `import *` must NOT surface it
    assert "ingest" not in truememory.__all__
