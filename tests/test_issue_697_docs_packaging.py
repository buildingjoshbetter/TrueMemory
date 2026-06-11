"""Regression locks for #697: docs accuracy (X1), sdist hygiene (A2-4), and the
test-module import-time env-leak class (V-cigate-1 / §3.5).

No model loads.
"""
import glob
import os
import re
from pathlib import Path
# NOTE: offline mode is set in conftest.py (V-cigate-1 / #697), not here.
# NOTE: pyproject is parsed as text, not tomllib — tomllib is 3.11+ and the CI
# matrix includes 3.10.

_ROOT = Path(__file__).resolve().parents[1]


# ── X1: docs report the real MCP tool count (11) ─────────────────────────────

def _actual_tool_count() -> int:
    src = (_ROOT / "truememory" / "mcp_server.py").read_text(encoding="utf-8")
    return len(set(re.findall(r"def (truememory_[a-z_]+)\(", src)))


def test_docs_report_correct_tool_count():
    n = _actual_tool_count()
    assert n == 11, f"expected 11 truememory_* tools, found {n} — update this test + the docs"
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    mcp_doc = (_ROOT / "docs" / "mcp-tools.md").read_text(encoding="utf-8")
    assert "8 MCP tools" not in readme, "README still claims 8 MCP tools"
    assert "exposes 9 tools" not in mcp_doc, "docs/mcp-tools.md still claims 9 tools"
    assert f"{n} MCP tools" in readme or f"All {n} MCP tools" in readme
    assert f"exposes {n} tools" in mcp_doc


def test_python_api_doc_metadata_not_reserved():
    doc = (_ROOT / "docs" / "python-api.md").read_text(encoding="utf-8")
    assert "Reserved for future use" not in doc, "metadata is a live field, not reserved"
    assert "directive" in doc, "m.add docs should mention the directive param"


# ── A2-4: sdist excludes tests/ ──────────────────────────────────────────────

def test_sdist_excludes_tests():
    text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # the hatch sdist exclude list must contain "/tests"
    assert '"/tests"' in text, "sdist must exclude /tests"


# ── V-cigate-1 / §3.5: no test module sets offline mode at import ────────────

def test_no_test_module_sets_hf_offline_at_import():
    """Offline mode is centralized in conftest.py. A test module doing its own
    module-level os.environ.setdefault("HF_HUB_OFFLINE", ...) is the §3.5 leak
    class (#654) — pytest imports every module during collection, so it would
    leak offline-mode into the online network-tests job."""
    offenders = []
    pat = re.compile(r'(?m)^os\.environ\.setdefault\(\s*["\'](?:HF_HUB_OFFLINE|TRANSFORMERS_OFFLINE)["\']')
    for f in glob.glob(str(_ROOT / "tests" / "**" / "*.py"), recursive=True):
        name = os.path.basename(f)
        if name == "conftest.py":
            continue  # conftest is the sanctioned single source
        if pat.search(Path(f).read_text(encoding="utf-8")):
            offenders.append(os.path.relpath(f, _ROOT))
    assert not offenders, (
        "these test modules set offline mode at import (move it to conftest.py): " + ", ".join(offenders)
    )
