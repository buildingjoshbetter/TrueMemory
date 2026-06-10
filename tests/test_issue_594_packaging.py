"""Regression test for issue #594 — dashboard packaging fork.

pyproject.toml declared a ``truememory-dashboard`` entry point whose target
module (``truememory.dashboard.cli:main``) never existed in the repo.  This
shipped a broken console-script in every wheel.

The fix removes the phantom entry point (and its optional-dependency extra).
This test locks the invariant: **every ``[project.scripts]`` entry point must
resolve to an importable module + callable**.
"""
from __future__ import annotations

import importlib
import pathlib
import re


def _parse_entry_points() -> list[tuple[str, str, str]]:
    """Return ``[(name, module, attr), ...]`` from pyproject.toml ``[project.scripts]``."""
    toml_path = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = toml_path.read_text()

    in_scripts = False
    results: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[project.scripts]":
            in_scripts = True
            continue
        if in_scripts:
            if stripped.startswith("["):
                break
            m = re.match(r'^([\w-]+)\s*=\s*"([\w.]+):(\w+)"', stripped)
            if m:
                results.append((m.group(1), m.group(2), m.group(3)))
    return results


def test_all_entry_points_resolve():
    """Every console-script declared in pyproject.toml must point to an
    importable module that exposes the named callable."""
    entry_points = _parse_entry_points()
    assert entry_points, "pyproject.toml should declare at least one entry point"

    for name, module_path, attr in entry_points:
        mod = importlib.import_module(module_path)
        obj = getattr(mod, attr, None)
        assert obj is not None, (
            f"Entry point {name!r} references {module_path}:{attr} "
            f"but {attr!r} was not found in module {module_path!r}"
        )
        assert callable(obj), (
            f"Entry point {name!r}: {module_path}:{attr} exists but is not callable"
        )


def test_no_dashboard_entry_point():
    """The phantom ``truememory-dashboard`` entry point must not reappear."""
    entry_points = _parse_entry_points()
    names = {name for name, _, _ in entry_points}
    assert "truememory-dashboard" not in names, (
        "truememory-dashboard entry point must not exist — "
        "the dashboard module was never committed (issue #594)"
    )


def test_no_dashboard_optional_extra():
    """The ``dashboard`` optional-dependency extra must not reappear."""
    toml_path = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = toml_path.read_text()

    in_optional = False
    extras: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[project.optional-dependencies]":
            in_optional = True
            continue
        if in_optional:
            if stripped.startswith("["):
                break
            m = re.match(r"^(\w+)\s*=", stripped)
            if m:
                extras.append(m.group(1))

    assert "dashboard" not in extras, (
        "dashboard optional-dependency extra must not exist — "
        "the dashboard module was never committed (issue #594)"
    )
