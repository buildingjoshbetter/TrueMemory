"""Regression lock: single-memory forget must not leave the forgotten content
behind in derived aggregate tables (S1-1 / issue #685).

Pre-fix: delete_message removed the messages row + FK-linked vec/FTS + a few
message-keyed tables, but NOT the entity-keyed aggregates summaries /
entity_profiles / entity_style_vectors / entity_relationships, so a forgotten
fact survived verbatim inside a consolidated summary or entity profile.
Post-fix: delete_message also purges the involved entities' derived rows (and
any summary that lists the message id).

FTS-only / no model loads — exercises delete_message directly against
hand-seeded derived rows.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json

from truememory.storage import create_db, insert_message, delete_message


def _seed(conn):
    mid = insert_message(conn, {
        "content": "my social security number is 555-00-1234",
        "sender": "josh", "recipient": "assistant",
        "timestamp": "2026-01-01T00:00:00Z", "category": "s", "modality": "conversation",
    })
    # Hand-build derived aggregates that reference the message / entity.
    conn.execute(
        "INSERT INTO summaries (period, entity, summary, key_facts, message_ids, created_at) "
        "VALUES ('all', 'josh', ?, ?, ?, '2026-01-01T00:00:00Z')",
        ("josh's SSN is 555-00-1234", json.dumps(["555-00-1234"]), json.dumps([mid])),
    )
    conn.execute(
        "INSERT INTO entity_profiles (entity, message_count, traits) VALUES ('josh', 1, ?)",
        (json.dumps({"ssn": "555-00-1234"}),),
    )
    conn.execute(
        "INSERT INTO entity_style_vectors (entity, vector, message_count) VALUES ('josh', '[0.1]', 1)"
    )
    conn.execute(
        "INSERT INTO entity_relationships (entity_a, entity_b, relationship_type) "
        "VALUES ('josh', 'assistant', 'self')"
    )
    conn.commit()
    return mid


def test_forget_purges_derived_aggregates(tmp_path):
    conn = create_db(tmp_path / "forget.db")
    mid = _seed(conn)

    # Sanity: the forgotten content is present in derived tables before delete.
    assert conn.execute("SELECT COUNT(*) FROM summaries WHERE entity='josh'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM entity_profiles WHERE entity='josh'").fetchone()[0] == 1

    assert delete_message(conn, mid) is True

    # The message and every entity-keyed aggregate for 'josh' are gone.
    assert conn.execute("SELECT COUNT(*) FROM messages WHERE id=?", (mid,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM summaries WHERE entity='josh'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM entity_profiles WHERE entity='josh'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM entity_style_vectors WHERE entity='josh'").fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM entity_relationships WHERE entity_a='josh' OR entity_b='josh'"
    ).fetchone()[0] == 0

    # The SSN string survives nowhere in summaries/profiles.
    leaked = conn.execute(
        "SELECT COUNT(*) FROM summaries WHERE summary LIKE '%555-00-1234%' OR key_facts LIKE '%555-00-1234%'"
    ).fetchone()[0]
    assert leaked == 0
    conn.close()


def test_forget_summary_by_message_id_even_for_other_entity(tmp_path):
    """A summary that lists the message id is purged even if its `entity` differs."""
    conn = create_db(tmp_path / "forget2.db")
    mid = insert_message(conn, {
        "content": "secret fact", "sender": "alice", "recipient": "bob",
        "timestamp": "2026-01-01T00:00:00Z", "category": "s", "modality": "conversation",
    })
    # Summary attributed to a THIRD entity but built from this message id.
    conn.execute(
        "INSERT INTO summaries (period, entity, summary, message_ids, created_at) "
        "VALUES ('all', 'carol', 'mentions secret fact', ?, '2026-01-01T00:00:00Z')",
        (json.dumps([mid]),),
    )
    conn.commit()
    assert delete_message(conn, mid) is True
    assert conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0] == 0
    conn.close()


def test_forget_unrelated_entity_summary_preserved(tmp_path):
    """Deleting josh's message must NOT wipe an unrelated entity's summary."""
    conn = create_db(tmp_path / "forget3.db")
    mid = insert_message(conn, {
        "content": "josh fact", "sender": "josh", "recipient": "assistant",
        "timestamp": "2026-01-01T00:00:00Z", "category": "s", "modality": "conversation",
    })
    conn.execute(
        "INSERT INTO summaries (period, entity, summary, message_ids, created_at) "
        "VALUES ('all', 'dylan', 'dylan likes climbing', '[]', '2026-01-01T00:00:00Z')"
    )
    conn.commit()
    assert delete_message(conn, mid) is True
    # dylan's unrelated summary is untouched.
    assert conn.execute("SELECT COUNT(*) FROM summaries WHERE entity='dylan'").fetchone()[0] == 1
    conn.close()
