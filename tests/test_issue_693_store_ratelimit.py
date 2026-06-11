"""Regression lock: the per-exchange store at store_intensity=max must be
rate-limited so a prompt flood can't pile up unbounded synchronous embed/dedup
work (SRE-02 / issue #693).

Pre-fix: _try_per_exchange_store ran a full Memory-open + search-embed +
EncodingGate + check_duplicate + add on EVERY storable prompt with no spacing.
Post-fix: a per-session time debounce (_store_debounced) skips the eager store
when one ran within the window — the Stop-hook background extraction still
captures everything from the transcript, so no data is lost.

No model loads — exercises the debounce gate directly.
"""


import time


def test_store_debounce_skips_rapid_second_store(tmp_path, monkeypatch):
    import truememory.ingest.hooks.user_prompt_submit as ups
    monkeypatch.setattr(ups, "_STORE_MARKER_DIR", tmp_path / "store_markers")
    monkeypatch.setattr(ups, "_STORE_DEBOUNCE_SECONDS", 2)

    # First store for the session is allowed; an immediate second is debounced.
    assert ups._store_debounced("sess-A") is False  # allowed (records time)
    assert ups._store_debounced("sess-A") is True    # within window -> skip
    assert ups._store_debounced("sess-A") is True    # still within window


def test_store_debounce_is_per_session(tmp_path, monkeypatch):
    import truememory.ingest.hooks.user_prompt_submit as ups
    monkeypatch.setattr(ups, "_STORE_MARKER_DIR", tmp_path / "store_markers")
    monkeypatch.setattr(ups, "_STORE_DEBOUNCE_SECONDS", 5)
    assert ups._store_debounced("sess-A") is False
    # A different session is independent (not debounced by sess-A's store).
    assert ups._store_debounced("sess-B") is False


def test_store_debounce_window_expires(tmp_path, monkeypatch):
    import truememory.ingest.hooks.user_prompt_submit as ups
    monkeypatch.setattr(ups, "_STORE_MARKER_DIR", tmp_path / "store_markers")
    monkeypatch.setattr(ups, "_STORE_DEBOUNCE_SECONDS", 1)
    assert ups._store_debounced("sess-C") is False
    assert ups._store_debounced("sess-C") is True
    time.sleep(1.1)
    assert ups._store_debounced("sess-C") is False  # window elapsed -> allowed again


def test_store_debounce_disabled_when_zero(tmp_path, monkeypatch):
    import truememory.ingest.hooks.user_prompt_submit as ups
    monkeypatch.setattr(ups, "_STORE_MARKER_DIR", tmp_path / "store_markers")
    monkeypatch.setattr(ups, "_STORE_DEBOUNCE_SECONDS", 0)
    # disabled: never debounces
    assert ups._store_debounced("sess-D") is False
    assert ups._store_debounced("sess-D") is False
