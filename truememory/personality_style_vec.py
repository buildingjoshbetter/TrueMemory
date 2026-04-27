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

import json
import math
import re
import sqlite3
from datetime import datetime, timezone

DIM = 256
NGRAM_SIZES = (3, 4, 5)


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
            h = hash(ng) % DIM
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


def build_entity_style_vectors(conn: sqlite3.Connection) -> dict[str, list[float]]:
    """Batch-build style vectors for every entity (sender) in the database.

    For each sender:
        1. Compute ``compute_style_vector(msg.content)`` for each message.
        2. Mean-pool all per-message vectors via ``mean_pool_vectors``.
        3. Store the result in the ``entity_style_vectors`` table.

    Args:
        conn: Open database connection (from :func:`truememory.storage.create_db`).

    Returns:
        ``{entity: vector}`` for every sender.
    """
    conn.execute(
        """CREATE TABLE IF NOT EXISTS entity_style_vectors (
            entity TEXT PRIMARY KEY,
            vector TEXT,
            message_count INTEGER DEFAULT 0,
            updated_at TEXT
        )"""
    )

    rows = conn.execute(
        "SELECT sender, content FROM messages WHERE sender != '' ORDER BY sender, timestamp"
    ).fetchall()

    from collections import defaultdict
    by_sender: dict[str, list[str]] = defaultdict(list)
    for sender, content in rows:
        by_sender[sender].append(content)

    result: dict[str, list[float]] = {}
    now = datetime.now(timezone.utc).isoformat()

    for sender, contents in by_sender.items():
        vecs = [compute_style_vector(c) for c in contents]
        mean_vec = mean_pool_vectors(vecs)
        result[sender] = mean_vec

        conn.execute(
            """INSERT OR REPLACE INTO entity_style_vectors
               (entity, vector, message_count, updated_at)
               VALUES (?, ?, ?, ?)""",
            (sender, json.dumps(mean_vec), len(contents), now),
        )

    conn.commit()
    return result


def update_entity_style_vector_incremental(
    conn: sqlite3.Connection, entity: str, new_message: str,
) -> None:
    """Incrementally update an entity's style vector with a new message.

    Uses a running weighted average: given the existing mean vector and
    its message count, the new mean is::

        new_mean = (existing * count + new_vec) / (count + 1)

    Then re-L2-normalized.

    Args:
        conn:        Open database connection.
        entity:      Entity name (sender).
        new_message: The new message text.
    """
    if not entity or not new_message:
        return

    conn.execute(
        """CREATE TABLE IF NOT EXISTS entity_style_vectors (
            entity TEXT PRIMARY KEY,
            vector TEXT,
            message_count INTEGER DEFAULT 0,
            updated_at TEXT
        )"""
    )

    new_vec = compute_style_vector(new_message)
    now = datetime.now(timezone.utc).isoformat()

    row = conn.execute(
        "SELECT vector, message_count FROM entity_style_vectors WHERE entity = ?",
        (entity,),
    ).fetchone()

    if row is None or row[0] is None:
        conn.execute(
            """INSERT OR REPLACE INTO entity_style_vectors
               (entity, vector, message_count, updated_at)
               VALUES (?, ?, ?, ?)""",
            (entity, json.dumps(new_vec), 1, now),
        )
    else:
        existing_vec = json.loads(row[0])
        count = row[1] or 0

        new_count = count + 1
        merged = [
            (existing_vec[i] * count + new_vec[i]) / new_count
            for i in range(len(existing_vec))
        ]

        norm = math.sqrt(sum(x * x for x in merged))
        if norm > 0:
            merged = [x / norm for x in merged]

        conn.execute(
            """INSERT OR REPLACE INTO entity_style_vectors
               (entity, vector, message_count, updated_at)
               VALUES (?, ?, ?, ?)""",
            (entity, json.dumps(merged), new_count, now),
        )

    conn.commit()


def get_entity_style_vector(
    conn: sqlite3.Connection, entity: str,
) -> list[float] | None:
    """Retrieve the stored style vector for an entity.

    Args:
        conn:   Open database connection.
        entity: Entity name (case-sensitive match).

    Returns:
        256-element float list, or ``None`` if no vector is stored.
    """
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
