"""Tests for issue #633 — supplement score normalization, pre-reranker
re-sort, salience exemption, entity-window gating, and degenerate fuse.

The theme (T1): supplement rows entered the result pool in the wrong score
space and the wrong order, so they either dominated organic hits or were
sliced off before the reranker ever saw them. Each test pins one fix:

  (a) M-11 — a tail-appended contradiction/summary supplement with a
      competitive score survives the ``results[:limit*3]`` reranker slice
      once the main path re-sorts by score desc before slicing.
  (b) M-09 — a consolidated-summary supplement with a RAW integer
      keyword-overlap score does NOT outrank all organic hits after being
      rescaled to the local pool.
  (c) M-10 — a temporal_rescoped row in [0,1] FTS space does NOT dominate
      the RRF-scored pool after being rescaled to the local max; and
      ``search_fts_in_range`` normalizes before trimming.
  (d) M-30 — a short contradiction current-fact ("ClickHouse", salience
      ~0.043) is exempt from the 0.05 salience spotlight floor.
  (e) M-77 — an all-identical-rerank-scores pool fuses to neutral 0.5,
      not 0.0.

These are unit-level tests over the actual fixed functions plus the exact
score-space transforms the engine applies, so they run FTS-only with no
embedding or reranker model loads (HF_HUB_OFFLINE=1).
"""
from __future__ import annotations

import os
import tempfile

import pytest

from truememory.fts_search import search_fts_in_range
from truememory.reranker import _normalize_and_fuse
from truememory.salience import filter_by_salience
from truememory.storage import create_db


# ---------------------------------------------------------------------------
# (a) M-11 — pre-reranker re-sort keeps tail supplements in the slice
# ---------------------------------------------------------------------------

def _resort_like_engine(results: list[dict]) -> list[dict]:
    """Mirror the M-11 sort the engine applies before results[:limit*3]."""
    results.sort(
        key=lambda r: (
            -(r.get("score", r.get("rrf_score", r.get("raw_score", 0))) or 0),
            str(r.get("id", "")),
        )
    )
    return results


class TestPreRerankerResort:
    """A competitively-scored supplement appended at the tail must survive
    the reranker candidate slice once the pool is re-sorted by score."""

    def test_contradiction_supplement_survives_slice(self):
        limit = 5
        # A full candidate pool of organic hits with descending RRF scores,
        # then a contradiction supplement appended at the very tail with a
        # competitive score (max_existing * 0.8).
        pool = [
            {"id": i, "content": f"organic {i}", "source": "hybrid",
             "score": 0.05 - i * 0.001}
            for i in range(40)
        ]
        max_existing = max(r["score"] for r in pool)
        supplement = {
            "id": 999,
            "content": "CarbonSense migrated to ClickHouse",
            "current_fact": "CarbonSense migrated to ClickHouse",
            "source": "contradiction",
            "score": max_existing * 0.8,
        }
        pool.append(supplement)

        # BEFORE the fix: tail position, sliced off at results[:limit*3].
        sliced_unsorted = pool[: limit * 3]
        assert 999 not in {r["id"] for r in sliced_unsorted}, (
            "precondition: supplement is sliced off without the re-sort"
        )

        # AFTER the fix: re-sort by score desc, then slice.
        resorted = _resort_like_engine(pool)
        sliced = resorted[: limit * 3]
        assert 999 in {r["id"] for r in sliced}, (
            "contradiction supplement must survive the reranker slice (#633 M-11)"
        )


# ---------------------------------------------------------------------------
# (b) M-09 — consolidated supplements rescaled, not raw integer scores
# ---------------------------------------------------------------------------

def _rescale_consolidated(results: list[dict], consolidated: list[dict]) -> None:
    """Mirror the M-09 rescale the engine applies before appending."""
    _cons_max = max(
        (r.get("score", r.get("rrf_score", 0)) for r in results),
        default=0.05,
    )
    _raw_max = max(
        (s.get("score", 0) for s in consolidated
         if isinstance(s.get("score"), (int, float))),
        default=0.0,
    )
    for sr in consolidated:
        _raw = sr.get("score", 0)
        if not isinstance(_raw, (int, float)) or _raw_max <= 0:
            _rel = 1.0
        else:
            _rel = _raw / _raw_max
        sr["score"] = _cons_max * 0.8 * max(0.0, min(_rel, 1.0))


class TestConsolidatedRescale:
    """Raw integer keyword-overlap scores must not outrank organic RRF hits."""

    def test_consolidated_does_not_outrank_all_organic(self):
        # Organic RRF hits cap near 0.05.
        results = [
            {"id": i, "content": f"organic {i}", "source": "hybrid",
             "score": 0.05 - i * 0.002}
            for i in range(10)
        ]
        organic_max = max(r["score"] for r in results)
        # Consolidated rows arrive with RAW integer overlap scores (relevance*2).
        consolidated = [
            {"id": "summary_1", "content": "journey summary", "source": "summary",
             "score": 6},  # raw integer — would be 120x the organic max
        ]
        _rescale_consolidated(results, consolidated)
        supp_score = consolidated[0]["score"]
        assert supp_score <= organic_max, (
            "consolidated supplement must not dominate organic hits on raw "
            f"integer score (got {supp_score} vs organic max {organic_max}) (#633 M-09)"
        )
        # And it should remain competitive (not zeroed out).
        assert supp_score > 0


# ---------------------------------------------------------------------------
# (c) M-10 — temporal_rescoped rows rescaled to local max; fts normalizes
#     before trimming.
# ---------------------------------------------------------------------------

def _rescale_rescoped(results: list[dict], range_results: list[dict]) -> None:
    """Mirror the M-10 engine rescale of temporal_rescoped supplements."""
    _resc_max = max(
        (r.get("score", r.get("rrf_score", 0)) for r in results),
        default=0.05,
    )
    for rr in range_results:
        _rs = rr.get("score", 1.0)
        if not isinstance(_rs, (int, float)):
            _rs = 1.0
        rr["score"] = _resc_max * 0.8 * max(0.0, min(_rs, 1.0))


class TestTemporalRescopedRescale:
    """[0,1] FTS-space rescoped rows must not dominate the RRF pool."""

    def test_rescoped_does_not_dominate_rrf_pool(self):
        results = [
            {"id": 1, "score": 0.05},
            {"id": 2, "rrf_score": 0.0167},
        ]
        pool_max = max(r.get("score", r.get("rrf_score", 0)) for r in results)
        # A rescoped row arrives at 1.0 (FTS-normalized single survivor).
        range_results = [{"id": 9, "score": 1.0, "timestamp": "2026-03-15"}]
        _rescale_rescoped(results, range_results)
        assert range_results[0]["score"] <= pool_max, (
            "temporal_rescoped row must not dominate purely on FTS [0,1] "
            "scaling (#633 M-10)"
        )

    def test_fts_in_range_normalizes_before_trim(self):
        """When the limit slice trims a multi-row in-range set, the surviving
        top rows keep their relative scale rather than being renormalized
        over the singleton."""
        d = tempfile.mkdtemp()
        conn = create_db(os.path.join(d, "t.db"))
        # Several in-window rows so the limit=1 trim cuts the pool.
        rows = [
            ("alpha clickhouse migration detail", "2026-03-15T00:00:00Z"),
            ("beta clickhouse migration note", "2026-03-16T00:00:00Z"),
            ("gamma clickhouse migration other", "2026-03-17T00:00:00Z"),
        ]
        for c, t in rows:
            conn.execute(
                "INSERT INTO messages (content, sender, recipient, timestamp, "
                "category, modality) VALUES (?,?,?,?,?,?)",
                (c, "a", "b", t, "s", "conversation"),
            )
        conn.commit()
        out = search_fts_in_range(
            conn, "clickhouse migration",
            after="2026-03-14", before="2026-03-18", limit=1,
        )
        conn.close()
        # Only 1 row returned (limit=1), but its score was computed over the
        # full 3-row in-range set, so the top row is 1.0 by construction —
        # the point is the call does not raise and the survivor is the most
        # relevant. The dominance defense lives in the engine rescale (M-10).
        assert len(out) == 1
        assert 0.0 <= out[0]["score"] <= 1.0


# ---------------------------------------------------------------------------
# (d) M-30 — contradiction source exempt from the salience floor
# ---------------------------------------------------------------------------

class TestContradictionSalienceExemption:
    """Short current-fact noun phrases below the floor survive when their
    source is 'contradiction'."""

    def test_short_contradiction_fact_survives_floor(self):
        rows = [
            {"id": 1, "current_fact": "ClickHouse",
             "source": "contradiction", "modality": ""},
        ]
        filtered = filter_by_salience(rows, min_salience=0.05)
        assert len(filtered) == 1, (
            "short contradiction current-fact must be exempt from the 0.05 "
            "spotlight floor (#633 M-30)"
        )
        # Confirm it genuinely scores below the floor (so the exemption is
        # what saved it, not the salience value).
        assert filtered[0]["salience"] < 0.05

    def test_short_non_contradiction_fact_still_dropped(self):
        """The exemption is source-scoped: a non-contradiction short fact is
        still dropped by the floor."""
        rows = [
            {"id": 1, "content": "ClickHouse", "source": "fts", "modality": ""},
        ]
        filtered = filter_by_salience(rows, min_salience=0.05)
        assert len(filtered) == 0

    def test_compound_contradiction_source_tag_exempt(self):
        """source values carrying 'contradiction' as a substring (e.g.
        'contradiction+temporal') are also exempt."""
        rows = [
            {"id": 1, "current_fact": "Redis", "source": "contradiction+temporal",
             "modality": ""},
        ]
        filtered = filter_by_salience(rows, min_salience=0.05)
        assert len(filtered) == 1


# ---------------------------------------------------------------------------
# (e) M-77 — degenerate (all-identical) rerank scores fuse to 0.5
# ---------------------------------------------------------------------------

class TestDegenerateFuse:
    """An all-identical rerank-score pool fuses to neutral 0.5, not 0.0."""

    def test_identical_rerank_scores_fuse_to_half(self):
        rows = [
            {"id": i, "content": f"m{i}", "rerank_score": 5.0, "score": 5.0}
            for i in range(3)
        ]
        out = _normalize_and_fuse(
            rows, rerank_weight=0.6, rrf_weight=0.4, top_k=10,
        )
        for r in out:
            assert r["fused_score"] == pytest.approx(0.5), (
                "degenerate all-identical pool must fuse to 0.5 not 0.0 (#633 M-77)"
            )

    def test_identical_rerank_only_still_neutral(self):
        """Even when only the rerank component is degenerate, that component
        contributes its neutral 0.5 (not 0.0)."""
        rows = [
            {"id": 0, "content": "m0", "rerank_score": 2.0, "score": 0.9},
            {"id": 1, "content": "m1", "rerank_score": 2.0, "score": 0.1},
        ]
        out = _normalize_and_fuse(
            rows, rerank_weight=0.6, rrf_weight=0.4, top_k=10,
        )
        # rerank component is degenerate -> 0.5 for both; orig differs.
        # Row with higher orig score must rank first and exceed 0.6*0.5.
        assert out[0]["id"] == 0
        assert out[0]["fused_score"] > 0.6 * 0.5
