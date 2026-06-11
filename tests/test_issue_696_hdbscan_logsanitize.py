"""Regression locks for #696: hdbscan declared as an optional extra (A2-1) and
log sanitization of untrusted content (S1-3).

No model loads. Offline mode is set in conftest.py (#697), not here.
"""
from pathlib import Path



# ── A2-1: hdbscan declared as an extra ───────────────────────────────────────

def test_hdbscan_declared_as_extra():
    # Text-based (no tomllib — that's 3.11+, but the matrix includes 3.10).
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert "clustering = [" in text, "a 'clustering' extra must exist"
    # the clustering extra line must name hdbscan
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("clustering = ["))
    assert "hdbscan" in line, "the clustering extra must declare hdbscan"


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
