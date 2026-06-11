"""Issue #654 (M-52): real-embedding smoke test for cosine-correct ranking.

Every other build_vectors / cosine test in the suite stubs the encoder, so a
regression in the *real* embedding path (the L2-vs-cosine bug from M-03, a
model-resolution break, or a build_vectors/search_vector mismatch) can still
merge green. This test closes that gap by running the genuine edge-tier
Model2Vec embedder (``minishlab/potion-base-8M``) end to end and asserting the
ranking is cosine-correct: a query semantically close to one stored message
must out-rank a semantically distant one.

It is marked ``network`` because the real model is downloaded from HuggingFace
(or read from the local HF cache). It therefore runs in the dedicated
``network-tests`` CI job — NOT the offline gate (which sets HF_HUB_OFFLINE=1
and does not vendor the model). Adding it to the gate would introduce a flaky
network dependency, which issue #654 explicitly forbids; instead the
``network-tests`` job is made to fail (not silently continue-on-error) on a
real regression. See .github/workflows/ci.yml.
"""
from __future__ import annotations

import pytest

from tests.conftest import requires_sqlite_ext


def _load_vec(conn):
    import sqlite_vec

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


@pytest.mark.network
@requires_sqlite_ext
def test_real_model2vec_cosine_ranking():
    """Real embeddings must rank a semantically-near message above a far one.

    This is the assertion the L2-vs-cosine bug (M-03) would have failed: with a
    broken metric, the distractor could out-rank or tie the true match.
    """
    from truememory import vector_search as vs
    from truememory.storage import create_db

    # Force the edge-tier (Model2Vec) embedder so the test is deterministic and
    # CPU-only regardless of the contributor's ambient tier.
    vs.set_embedding_model("model2vec")

    conn = create_db(":memory:")
    _load_vec(conn)
    vs.init_vec_table(conn)

    rows = [
        (1, "The cat sat on the warm windowsill in the afternoon sun."),
        (2, "Quarterly revenue grew on strong enterprise software sales."),
    ]
    for mid, content in rows:
        conn.execute("INSERT INTO messages(id, content) VALUES (?, ?)", (mid, content))
    conn.commit()

    inserted = vs.build_vectors(conn)
    assert inserted == 2, f"expected 2 real vectors, built {inserted}"

    results = vs.search_vector(conn, "a kitten relaxing by the sunny window", limit=2)
    assert results, "real-embedder search returned no results"

    by_id = {r["id"]: r["score"] for r in results}
    assert 1 in by_id and 2 in by_id, f"missing ids in {by_id}"

    # Cosine-correct: the feline/window message must out-rank the finance one.
    assert by_id[1] > by_id[2], (
        f"cosine ranking regression: near-match score {by_id[1]:.4f} did not "
        f"beat distractor {by_id[2]:.4f}"
    )
    # Scores are normalized to (0, 1]; sanity-check the transform.
    assert 0.0 < by_id[1] <= 1.0 and 0.0 < by_id[2] <= 1.0, by_id

    conn.close()
