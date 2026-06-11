"""Regression locks for #696: hdbscan declared as an optional extra (A2-1) and
log sanitization of untrusted content (S1-3).

No model loads.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import tomllib
from pathlib import Path



# ── A2-1: hdbscan declared as an extra ───────────────────────────────────────

def test_hdbscan_declared_as_extra():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    extras = data["project"]["optional-dependencies"]
    assert "clustering" in extras, "a 'clustering' extra must exist"
    assert any("hdbscan" in dep for dep in extras["clustering"]), (
        "the clustering extra must declare hdbscan"
    )


# ── S1-3: log sanitization ───────────────────────────────────────────────────

def test_safe_log_strips_control_chars():
    from truememory.ingest.pipeline import _safe_log
    # newline / CR / ANSI / NUL must not survive into a log line
    out = _safe_log("line one\nFORGED LOG LINE\r\x1b[31mred\x00\ttab")
    assert "\n" not in out and "\r" not in out
    assert "\x1b" not in out and "\x00" not in out and "\t" not in out
    # the visible text is preserved (control chars replaced with spaces)
    assert "line one" in out and "FORGED LOG LINE" in out and "red" in out


def test_safe_log_handles_non_str():
    from truememory.ingest.pipeline import _safe_log
    assert _safe_log(None) == "None"
    assert _safe_log(123) == "123"


def test_pipeline_log_calls_use_safe_log():
    """The user-content log args (fact.content / dedup.fact) route through _safe_log."""
    import inspect
    from truememory.ingest import pipeline
    src = inspect.getsource(pipeline)
    # no RAW user-content slice remains as a bare log arg
    assert "_safe_log(fact.content[:50])" in src
    assert "_safe_log(dedup.fact[:80])" in src
    assert "_safe_log(dedup.fact[:120])" in src
