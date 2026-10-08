"""
TrueMemory L0 — Character N-gram Style Vectors
================================================

Per-entity writing-style profiles using 256-dimensional hashed character
n-gram vectors (char-(3,4,5)-grams, L2-normalized, mean-pooled across
a persona's messages).

This module implements the C3c candidate from the MEMORIST-L0 research
session (2026-04-23).  On the hand-authored multi-persona probe set,
C3c scored 0.686 accuracy vs 0.271 for the shipping hand-tuned keyword
extractor -- a 2.5x improvement that also beats the no-L0 baseline
(0.371).

See: ``_working/memorist/l0_personality/REPORT.md``
See: ``benchmarks/gate_eval/candidates/l0_personality/c3c_char_ngram.py``

All functions operate on stdlib only (no external dependencies).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone


def _stable_hash(s: str) -> int:
    """Stable hash unaffected by PYTHONHASHSEED — safe for cross-process vector compatibility."""
    return struct.unpack("<I", hashlib.md5(s.encode("utf-8"), usedforsecurity=False).digest()[:4])[0]

DIM = 256
NGRAM_SIZES = (3, 4, 5)
_STYLE_ACCUMULATOR_VERSION = 1
_STYLE_SUM = struct.Struct("<256d")


class StyleVectorRebuildRequired(RuntimeError):
    """The stored profile remains readable but cannot accept exact appends."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Style accumulator rebuild required: {reason}")


def compute_style_vector(text: str) -> list[float]:
    """Compute a 256-d L2-normalized char-n-gram hash vector from text.

    Algorithm:
        1. Lowercase and normalize whitespace.
        2. Extract all character n-grams for n in (3, 4, 5).
        3. Hash each n-gram to a bucket in [0, 256) via ``hash(ng) % DIM``.
        4. L2-normalize the resulting count vector.

    Args:
        text: Input text (any length, any language).

    Returns:
        List of 256 floats forming a unit vector (L2 norm ~1.0).
        Returns a zero vector if text is empty or only whitespace.
    """
    vec = [0.0] * DIM
    if not text or not text.strip():
        return vec
    normalized = re.sub(r"\s+", " ", text.lower())
    for n in NGRAM_SIZES:
        for i in range(len(normalized) - n + 1):
            ng = normalized[i:i + n]
            h = _stable_hash(ng) % DIM
            vec[h] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def mean_pool_vectors(vectors: list[list[float]]) -> list[float]:
    """Average a list of vectors and re-normalize to unit length.

    Args:
        vectors: List of equal-length float vectors.

    Returns:
        L2-normalized mean vector.  Returns a zero vector if *vectors*
        is empty.
    """
    if not vectors:
        return [0.0] * DIM
    dim = len(vectors[0])
    out = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            out[i] += v[i]
    out = [x / len(vectors) for x in out]
    norm = math.sqrt(sum(x * x for x in out))
    return [x / norm for x in out] if norm > 0 else out


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors.

    Safe for zero vectors (returns 0.0).

    Args:
        a: First vector.
        b: Second vector (same length as *a*).

    Returns:
        Similarity in [-1.0, 1.0].  For L2-normalized inputs this
        equals the dot product.
    """
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return num / (na * nb)


@contextmanager
def _l0_transaction(conn: sqlite3.Connection, *, write: bool = False) -> Iterator[None]:
    """Own only this phase, retaining any transaction opened by the caller."""
    owned = not conn.in_transaction
    if owned:
        conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
    else:
        conn.execute("SAVEPOINT truememory_l0")
    completed = False
    try:
        if write and not owned:
            # Upgrade before validation: a stale WAL reader must not publish.
            conn.execute("UPDATE messages SET sender = sender WHERE 0")
        yield
        conn.execute("COMMIT" if owned else "RELEASE SAVEPOINT truememory_l0")
        completed = True
    finally:
        if not completed and conn.in_transaction:
            if owned:
                conn.rollback()
            else:
                # If rollback fails, never RELEASE potentially partial writes.
                conn.execute("ROLLBACK TO SAVEPOINT truememory_l0")
                conn.execute("RELEASE SAVEPOINT truememory_l0")


def _l0_source_schema(conn: sqlite3.Connection) -> tuple[tuple, ...]:
    return tuple(tuple(row) for row in conn.execute("PRAGMA table_info(messages)"))


def _l0_directive_filter(schema: tuple[tuple, ...]) -> str:
    # Public builders also accept older standalone schemas without directives.
    if any(column[1] == "directive" for column in schema):
        return "(directive = 0 OR directive IS NULL)"
    return "1"


def _l0_validate_source(
    conn: sqlite3.Connection,
    schema: tuple[tuple, ...],
    query: str,
    source_rows: list[tuple],
) -> None:
    """Compare exact relevant source values while holding writer ownership."""
    error = "L0 source changed during computation; retry the rebuild"
    if _l0_source_schema(conn) != schema:
        raise sqlite3.OperationalError(error)
    current = conn.execute(query)
    try:
        for expected in source_rows:
            row = current.fetchone()
            if row is None or tuple(row) != expected:
                raise sqlite3.OperationalError(error)
        if current.fetchone() is not None:
            raise sqlite3.OperationalError(error)
    finally:
        current.close()


def _compute_entity_style_vectors(
    rows: list[tuple],
) -> tuple[dict[str, list[float]], list[tuple]]:
    from collections import defaultdict

    by_sender: dict[str, list[str]] = defaultdict(list)
    for sender, content, _timestamp, ordinary in rows:
        if ordinary:
            by_sender[sender.lower()].append(content)

    result: dict[str, list[float]] = {}
    stored_rows: list[tuple] = []
    now = datetime.now(timezone.utc).isoformat()
    for sender, contents in by_sender.items():
        total = [0.0] * DIM
        for content in contents:
            vec = _validated_style_vector(compute_style_vector(content))
            for index in range(DIM):
                total[index] += vec[index]
        mean_vec = _style_profile_from_sum(total, len(contents))
        result[sender] = mean_vec
        stored_rows.append((sender, json.dumps(mean_vec), len(contents), now,
                            _STYLE_SUM.pack(*total), _STYLE_ACCUMULATOR_VERSION))
    return result, stored_rows


def _validated_style_vector(vector: list[float]) -> list[float]:
    if not isinstance(vector, (list, tuple)) or len(vector) != DIM:
        raise ValueError("Style vector must have 256 finite components")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in vector):
        raise ValueError("Style vector must have 256 finite components")
    try:
        values = [float(value) for value in vector]
    except (OverflowError, ValueError) as error:
        raise ValueError("Style vector must have 256 finite components") from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Style vector must have 256 finite components")
    return values


def _style_profile_from_sum(total: list[float], count: int) -> list[float]:
    # Retain mean_pool_vectors' divide-then-normalize arithmetic exactly.
    mean = [value / count for value in total]
    norm = math.sqrt(sum(value * value for value in mean))
    if not math.isfinite(norm):
        raise ValueError("Style profile magnitude is not finite")
    return [value / norm for value in mean] if norm > 0 else mean


def _ensure_style_accumulator_schema(conn: sqlite3.Connection) -> None:
    """Add format columns inside the caller's publication transaction."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS entity_style_vectors (
            entity TEXT PRIMARY KEY,
            vector TEXT,
            message_count INTEGER DEFAULT 0,
            updated_at TEXT,
            vector_sum BLOB DEFAULT NULL,
            accumulator_version INTEGER NOT NULL DEFAULT 0
        )"""
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(entity_style_vectors)")}
    if "vector_sum" not in columns:
        conn.execute("ALTER TABLE entity_style_vectors ADD COLUMN vector_sum BLOB DEFAULT NULL")
    if "accumulator_version" not in columns:
        conn.execute("ALTER TABLE entity_style_vectors ADD COLUMN accumulator_version INTEGER NOT NULL DEFAULT 0")


def _decode_style_accumulator(row: tuple) -> tuple[list[float], int]:
    raw, count, version = row
    if type(version) is not int or version < 0:
        raise StyleVectorRebuildRequired("invalid")
    if version == 0:
        raise StyleVectorRebuildRequired("legacy")
    if version != _STYLE_ACCUMULATOR_VERSION:
        raise StyleVectorRebuildRequired("unsupported_version")
    if not isinstance(raw, bytes) or len(raw) != _STYLE_SUM.size or type(count) is not int or count < 1:
        raise StyleVectorRebuildRequired("invalid")
    total = list(_STYLE_SUM.unpack(raw))
    if not all(math.isfinite(value) for value in total):
        raise StyleVectorRebuildRequired("invalid")
    try:
        _style_profile_from_sum(total, count)
    except ValueError as error:
        raise StyleVectorRebuildRequired("invalid") from error
    return total, count


@contextmanager
def _style_append_transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """Serialize read-modify-write without committing an append for its caller."""
    owned = not conn.in_transaction
    if owned:
        conn.execute("BEGIN IMMEDIATE")
    savepoint = False
    completed = False
    try:
        conn.execute("SAVEPOINT truememory_style_append")
        savepoint = True
        yield
        conn.execute("RELEASE SAVEPOINT truememory_style_append")
        completed = True
    finally:
        if not completed and conn.in_transaction:
            if owned:
                conn.rollback()
            elif savepoint:
                conn.execute("ROLLBACK TO SAVEPOINT truememory_style_append")
                conn.execute("RELEASE SAVEPOINT truememory_style_append")


def build_entity_style_vectors(conn: sqlite3.Connection) -> dict[str, list[float]]:
    """Batch-build style vectors for every entity (sender) in the database.

    For each sender:
        1. Compute ``compute_style_vector(msg.content)`` for each message.
        2. Mean-pool with the same arithmetic as ``mean_pool_vectors``.
        3. Store the profile and its raw sum in ``entity_style_vectors``.

    Args:
        conn: Open database connection (from :func:`truememory.storage.create_db`).

    Returns:
        ``{entity: vector}`` for every sender.

    Only ordinary messages contribute. Computation starts no write transaction;
    any transaction already held by the caller remains theirs. Publication
    atomically replaces the complete table after checking the source snapshot.
    A changed source raises ``sqlite3.OperationalError`` without retrying.
    """
    with _l0_transaction(conn):
        schema = _l0_source_schema(conn)
        query = (
            f"SELECT sender, content, timestamp, {_l0_directive_filter(schema)} AS ordinary "
            "FROM messages WHERE sender != '' "
            "ORDER BY sender, timestamp"
        )
        rows = [tuple(row) for row in conn.execute(query)]
    result, stored_rows = _compute_entity_style_vectors(rows)

    with _l0_transaction(conn, write=True):
        _l0_validate_source(conn, schema, query, rows)
        _ensure_style_accumulator_schema(conn)
        conn.execute("DELETE FROM entity_style_vectors")
        conn.executemany(
            """INSERT INTO entity_style_vectors
               (entity, vector, message_count, updated_at, vector_sum, accumulator_version)
               VALUES (?, ?, ?, ?, ?, ?)""",
            stored_rows,
        )
    return result


def update_entity_style_vector_incremental(
    conn: sqlite3.Connection, entity: str, new_message: str,
    *, _pre_computed_vec: list[float] | None = None,
) -> None:
    """Incrementally update an entity's style vector with a new message.

    Persist the sum of per-message vectors and normalize only the exposed
    profile. A legacy, unsupported or invalid accumulator raises
    ``StyleVectorRebuildRequired`` without changing its previous profile.
    Rebuild explicitly with ``build_entity_style_vectors`` before appending.
    This function never commits; successful writes remain the caller's.

    Args:
        conn:        Open database connection.
        entity:      Entity name (sender).
        new_message: The new message text.
        _pre_computed_vec: If provided, skip compute_style_vector() call.
    """
    if not entity or not new_message:
        return

    # Normalize entity name to lowercase for case-insensitive matching (#467)
    entity = entity.lower()

    new_vec = _validated_style_vector(
        _pre_computed_vec if _pre_computed_vec is not None else compute_style_vector(new_message)
    )
    now = datetime.now(timezone.utc).isoformat()
    with _style_append_transaction(conn):
        _ensure_style_accumulator_schema(conn)
        # Existing callers may hold only a read snapshot. Upgrade before
        # reading the accumulator so another connection cannot lose an append.
        conn.execute("UPDATE entity_style_vectors SET entity=entity WHERE 0")
        row = conn.execute(
            "SELECT vector_sum, message_count, accumulator_version FROM entity_style_vectors WHERE entity = ?",
            (entity,),
        ).fetchone()
        if row is None:
            total, count = new_vec, 1
        else:
            previous, count = _decode_style_accumulator(row)
            total = [previous[index] + new_vec[index] for index in range(DIM)]
            count += 1
        profile = _style_profile_from_sum(total, count)
        conn.execute(
            """INSERT OR REPLACE INTO entity_style_vectors
               (entity, vector, message_count, updated_at, vector_sum, accumulator_version)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (entity, json.dumps(profile), count, now, _STYLE_SUM.pack(*total), _STYLE_ACCUMULATOR_VERSION),
        )


def get_entity_style_vector(
    conn: sqlite3.Connection, entity: str,
) -> list[float] | None:
    """Retrieve the stored style vector for an entity.

    Args:
        conn:   Open database connection.
        entity: Entity name (case-insensitive match).

    Returns:
        256-element float list, or ``None`` if no vector is stored.
    """
    # Normalize entity name to lowercase for case-insensitive matching (#467)
    entity = entity.lower()
    try:
        row = conn.execute(
            "SELECT vector FROM entity_style_vectors WHERE entity = ?",
            (entity,),
        ).fetchone()
    except sqlite3.OperationalError:
        return None

    if row is None or row[0] is None:
        return None
    return json.loads(row[0])
