"""Regression lock for Hunter F34 — `load_messages` → `bulk_replace_messages`
rename with a deprecated alias.

Pre-fix, `load_messages` (destructive: DELETE then INSERT) shared the
naming convention with `insert_message` (appending). A caller writing
`load_messages(conn, [new_msg])` believing it appended destroyed their
DB. The destructive semantics now live under `bulk_replace_messages`;
`load_messages` is preserved for one release as a deprecated alias.
"""
from __future__ import annotations

import warnings


def _make_db(tmp_path):
    from truememory.storage import create_db
    return create_db(tmp_path / "t.db")


def _sample_messages(n=3):
    return [
        {"content": f"msg {i}", "sender": "alice", "recipient": "bob"}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# bulk_replace_messages — the non-deprecated API
# ---------------------------------------------------------------------------


def test_bulk_replace_messages_is_public():
    """The new name must be importable from `truememory` AND
    `truememory.storage`, and be in `truememory.__all__`."""
    import truememory
    from truememory.storage import bulk_replace_messages as from_storage
    from truememory import bulk_replace_messages as from_top
    assert from_top is from_storage
    assert "bulk_replace_messages" in truememory.__all__


def test_bulk_replace_messages_inserts_without_warning(tmp_path):
    conn = _make_db(tmp_path)
    from truememory.storage import bulk_replace_messages, get_message_count
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            n = bulk_replace_messages(conn, _sample_messages(3))
        # No DeprecationWarning on the new name
        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep_warnings == []
        assert n == 3
        assert get_message_count(conn) == 3
    finally:
        conn.close()


def test_bulk_replace_messages_is_destructive(tmp_path):
    """The core invariant: bulk_replace WIPES existing rows before insert."""
    conn = _make_db(tmp_path)
    from truememory.storage import bulk_replace_messages, get_message_count
    try:
        bulk_replace_messages(conn, _sample_messages(5))
        assert get_message_count(conn) == 5
        # Second call wipes the first batch
        bulk_replace_messages(conn, _sample_messages(2))
        assert get_message_count(conn) == 2
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# load_messages — deprecated alias
# ---------------------------------------------------------------------------


def test_load_messages_emits_deprecation_warning(tmp_path):
    """The old name must still work but emit DeprecationWarning. Callers
    migrating in the deprecation window should see the hint in their
    CI / test logs."""
    conn = _make_db(tmp_path)
    from truememory.storage import load_messages
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_messages(conn, _sample_messages(2))
        dep_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert len(dep_warnings) == 1
        msg = str(dep_warnings[0].message)
        # Must name the replacement function
        assert "bulk_replace_messages" in msg
        # Must also hint at insert_message for the append case
        assert "insert_message" in msg
    finally:
        conn.close()


def test_load_messages_same_behavior_as_bulk_replace(tmp_path):
    """The deprecated alias must produce identical final state."""
    from truememory.storage import bulk_replace_messages, load_messages, get_message_count

    # Run via the new name
    (tmp_path / "a").mkdir()
    conn_a = _make_db(tmp_path / "a")
    try:
        bulk_replace_messages(conn_a, _sample_messages(4))
        count_new = get_message_count(conn_a)
    finally:
        conn_a.close()

    # Run via the deprecated alias (suppress the warning for this assertion)
    (tmp_path / "b").mkdir()
    conn_b = _make_db(tmp_path / "b")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            load_messages(conn_b, _sample_messages(4))
        count_old = get_message_count(conn_b)
    finally:
        conn_b.close()

    assert count_old == count_new == 4


def test_load_messages_from_file_does_not_emit_deprecation(tmp_path):
    """`load_messages_from_file` is NOT deprecated — its internal impl
    was rewired to call `bulk_replace_messages` directly so users who
    never called the bare `load_messages` don't see our own
    DeprecationWarning bleeding into their logs."""
    import json as _json
    conn = _make_db(tmp_path)
    data_path = tmp_path / "msgs.json"
    data_path.write_text(_json.dumps(_sample_messages(3)))

    from truememory.storage import load_messages_from_file
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            n = load_messages_from_file(conn, data_path)
        dep_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert dep_warnings == [], (
            "F34 regression: load_messages_from_file is emitting a "
            "DeprecationWarning because it's still calling the deprecated "
            "load_messages internally. Rewire it to call bulk_replace_messages."
        )
        assert n == 3
    finally:
        conn.close()


def test_load_messages_stacklevel_attributes_to_caller(tmp_path):
    """`stacklevel=2` in the deprecation warning means the warning's
    filename points at the CALLER, not at storage.py. This makes the
    migration hint actionable — users see their own code in the
    warning line, not a library internal."""
    conn = _make_db(tmp_path)
    from truememory.storage import load_messages
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_messages(conn, _sample_messages(1))
        dep_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert len(dep_warnings) == 1
        # Filename should be the test file (caller), not storage.py
        assert "test_load_messages_rename.py" in dep_warnings[0].filename
    finally:
        conn.close()
