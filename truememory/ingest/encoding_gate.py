"""
Encoding Gate (Inspired by Hippocampal Novelty Detection)
==========================================================

A memory filter that decides which candidate facts get stored. Three
signals — novelty, salience, and prediction error — are computed and
combined into a weighted score; facts above the threshold are encoded.

**Honest framing**: This module is *inspired by* neuroscience, not a
faithful computational model of it. The three-signal gating architecture
mirrors the *function* of hippocampal / amygdala / prefrontal circuits,
not their *mechanism*. What you see in this code is a pragmatic proxy:

- **Novelty** is vector-similarity inversion via truememory's hybrid
  search. A novel fact is one whose content is dissimilar to existing
  memories. Real CA1 novelty detection involves CA3→CA1 pattern
  completion, oscillatory dynamics, and sparse coding we don't model.

- **Salience** delegates to truememory's existing
  `salience.compute_message_salience` (which scores length, numbers,
  dates, emotional markers, life events) and adds a category weight
  from the LLM extractor's classification. Real amygdala modulation is
  norepinephrine release affecting LTP threshold, not a weighted sum.

- **Prediction error** uses embedding pair-difference scoring: embed
  the (message, nearest_memory) pair and compare to the (memory, memory)
  self-pair. When the pair embedding diverges from the self-pair
  embedding, the message says something different about the same topic.
  Validated in 200-variant sweep (v1+v2): AUC 0.730 standalone, gate
  AUC 0.796. Real predictive coding is Bayesian error propagation up a
  hierarchical generative model.

The delegation to `truememory.salience` for salience scoring is
intentional: it already implements the scoring truememory uses for
retrieval weighting. If the module is unavailable (older truememory
or partial install) a warning is logged at import time and the gate
falls back to internal heuristics. Prediction error uses an
embedding-based scorer that is independent of L5's surprise module.

**What a skeptical reader should know**: the final encoding decision is
`0.25 * novelty + 0.20 * salience + 0.30 * prediction_error >= 0.26`
(with a salience floor of 0.10 — messages below the floor are rejected
regardless of gate score).
The neuroscience names describe what each term is *inspired by*, not a
claim that this is how the brain works.

References (for inspiration, not literal modeling):
- Lisman & Grace, 2005 — hippocampal novelty detection
- McGaugh, 2004 — amygdala memory modulation
- Rao & Ballard, 1999 — predictive coding in visual cortex
- McClelland, McNaughton & O'Reilly, 1995 — complementary learning systems
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


# Try to import truememory's existing scoring modules. If they're not
# available (older truememory or partial install), fall back to
# internal heuristics.
_HAS_TRUEMEMORY_SALIENCE = False
try:
    from truememory.salience import compute_message_salience as _tm_salience
    _HAS_TRUEMEMORY_SALIENCE = True
except ImportError:
    _tm_salience = None

if not _HAS_TRUEMEMORY_SALIENCE:
    log.warning(
        "truememory.salience.compute_message_salience not available; "
        "using fallback salience heuristic. Install/upgrade truememory "
        "for the full salience signal."
    )

# Reuse the salience scorer's noise set for PE — avoids maintaining two
# copies of the same word list. Falls back to a minimal set if the
# salience module isn't available.
try:
    from truememory.ingest.encoding_salience import _NOISE_EXACT_V23 as _PE_NOISE
except ImportError:
    _PE_NOISE = frozenset({
        "ok", "okay", "k", "kk", "yes", "yeah", "yep", "yup",
        "no", "nah", "nope", "lol", "lmao", "haha", "hahaha",
        "nice", "cool", "thanks", "thx", "ty", "got it", "gotcha",
        "sounds good", "sure", "bet", "word", "same", "mood", "idk",
        "gn", "gm", "brb", "ttyl", "damn", "dude", "bro", "ugh", "wow",
    })


@dataclass
class EncodingDecision:
    """Result of the encoding gate evaluation."""
    should_encode: bool
    encoding_score: float
    novelty: float           # 0 = fully familiar, 1 = completely novel
    salience: float          # 0 = noise, 1 = critical personal information
    prediction_error: float  # 0 = expected, 1 = contradicts existing knowledge
    reason: str = ""         # Human-readable explanation
    similar_memory: str = "" # Most similar existing memory (if any)


# Category-level weights derived from the LLM extractor's classification.
# These are not "the amygdala" — they are a pragmatic boost for fact types
# that matter more for future retrieval (corrections > decisions > technical).
_CATEGORY_SALIENCE_BOOST = {
    "correction": 0.40,
    "decision": 0.30,
    "personal": 0.25,
    "preference": 0.25,
    "relationship": 0.20,
    "temporal": 0.15,
    "technical": 0.10,
    "general": 0.05,
}


class EncodingGate:
    """
    Encoding gate that filters candidate facts through three signals.

    The gate computes a weighted sum of novelty, salience, and prediction
    error. Facts whose score exceeds the threshold are encoded; others
    are filtered out. All three signals delegate to truememory modules
    where possible, so the gate stays consistent with retrieval behavior.

    Args:
        memory: A truememory Memory instance for searching existing memories.
        threshold: Minimum encoding score to pass the gate (0.0 - 1.0).
        w_novelty: Weight for the novelty signal.
        w_salience: Weight for the salience signal.
        w_prediction_error: Weight for the prediction error signal.
        salience_floor: Minimum salience to even consider encoding (0.0 - 1.0).
        user_id: Optional user scope for memory searches.
    """

    def __init__(
        self,
        memory,
        threshold: float = 0.26,
        w_novelty: float | None = None,
        w_salience: float | None = None,
        w_prediction_error: float | None = None,
        salience_floor: float | None = None,
        user_id: str = "",
    ):
        self.memory = memory
        self.threshold = threshold
        if w_novelty is None:
            w_novelty = float(os.environ.get("TRUEMEMORY_GATE_W_NOVELTY", "0.25"))
        if w_salience is None:
            w_salience = float(os.environ.get("TRUEMEMORY_GATE_W_SALIENCE", "0.20"))
        if w_prediction_error is None:
            w_prediction_error = float(os.environ.get("TRUEMEMORY_GATE_W_PE", "0.30"))
        self.w_novelty = w_novelty
        self.w_salience = w_salience
        self.w_prediction_error = w_prediction_error
        if salience_floor is None:
            salience_floor = float(os.environ.get("TRUEMEMORY_GATE_SALIENCE_FLOOR", "0.10"))
        self.salience_floor = salience_floor
        self.user_id = user_id
        # Normalized weights so the final score lands in [0, 1]
        total = w_novelty + w_salience + w_prediction_error
        self._norm = total if total > 0 else 1.0
        self._last_search_results: list[dict] = []
        self._embed_model = None
        self._batch_scores: list[float] = []
        self._batch_novelties: list[float] = []
        self._batch_saliences: list[float] = []
        self._batch_pes: list[float] = []

    def evaluate(self, fact: str, category: str = "") -> EncodingDecision:
        """
        Pass a candidate fact through the encoding gate.

        Returns an EncodingDecision with the full signal breakdown.
        """
        novelty = self._compute_novelty(fact)
        salience = self._compute_salience(fact, category)
        pred_error = self._compute_prediction_error(fact)

        # Weighted sum, normalized to [0, 1]
        raw = (
            novelty * self.w_novelty
            + salience * self.w_salience
            + pred_error * self.w_prediction_error
        )
        score = max(0.0, min(1.0, raw / self._norm))

        # Salience floor: reject messages the salience scorer considers
        # pure noise, regardless of how novel or surprising they are.
        # Prevents high-novelty off-topic chatter from passing the gate.
        floored = salience < self.salience_floor
        if floored:
            should_encode = False
        else:
            should_encode = score >= self.threshold
        reason = self._explain(novelty, salience, pred_error, score, should_encode, floored)

        verdict = "ENCODE" if should_encode else "SKIP"
        log.debug(
            "gate: fact=%r n=%.2f s=%.2f p=%.2f score=%.3f thr=%.2f -> %s",
            fact[:60], novelty, salience, pred_error, score,
            self.threshold, verdict,
        )

        self._batch_scores.append(score)
        self._batch_novelties.append(novelty)
        self._batch_saliences.append(salience)
        self._batch_pes.append(pred_error)

        # Get the most similar existing memory for context (only if moderately similar)
        similar = ""
        if 0.1 < novelty < 0.7:
            if self._last_search_results:
                similar = self._last_search_results[0].get("content", "")
            else:
                results = self._search(fact, limit=1)
                if results:
                    similar = results[0].get("content", "")

        return EncodingDecision(
            should_encode=should_encode,
            encoding_score=round(score, 3),
            novelty=round(novelty, 3),
            salience=round(salience, 3),
            prediction_error=round(pred_error, 3),
            reason=reason,
            similar_memory=similar,
        )

    # ------------------------------------------------------------------
    # Signal 1: Novelty — vector-similarity proxy for CA1 comparator
    # ------------------------------------------------------------------

    def _compute_novelty(self, fact: str) -> float:
        """
        Compression-based novelty detection.

        Measures how much NEW information this message adds to stored
        memories using gzip compression cost. Novel information compresses
        poorly against a memory-trained model; redundant information
        compresses cheaply.

        Formula: (gzip(memory + msg) - gzip(memory)) / gzip(msg)
        High ratio = message contains information not in memory = novel.
        Low ratio = message is redundant with memory = not novel.

        This replaced cosine similarity inversion (PR #105) because
        embedding distance is anti-correlated with novelty in conversational
        data — noise like "ok" is semantically distant from factual memories
        while important updates are semantically close. Compression measures
        statistical redundancy, which is a better proxy for information
        novelty. Validated in 120-variant sweep: AUC 0.788 vs 0.484 for
        cosine baseline. See issue #107.

        Falls back to cosine similarity when memory is empty or on error.
        """
        import gzip

        # Build memory text from stored results
        # Use cached search results if available, otherwise search
        results = None
        if hasattr(self.memory, "search_vectors"):
            try:
                results = self.memory.search_vectors(fact, limit=10)
            except Exception:
                pass
        if results is None:
            results = self._search(fact, limit=10)

        self._last_search_results = results

        if not results:
            return 1.0  # Empty memory = maximum novelty

        # Concatenate memory contents for compression comparison
        memory_text = " ".join(
            r.get("content", "") for r in results[:10] if r.get("content")
        )

        if not memory_text.strip():
            return 1.0

        try:
            fact_bytes = fact.encode("utf-8")
            memory_bytes = memory_text.encode("utf-8")
            combined_bytes = memory_bytes + b" " + fact_bytes

            c_memory = len(gzip.compress(memory_bytes, compresslevel=6))
            c_combined = len(gzip.compress(combined_bytes, compresslevel=6))
            c_fact = len(gzip.compress(fact_bytes, compresslevel=6))

            if c_fact < 10:
                return 0.05  # Trivially short messages (noise)

            # Conditional compression: how much does adding this message
            # increase the compressed size of memory?
            compression_cost = (c_combined - c_memory) / c_fact

            # Normalize to [0, 1] — compression_cost typically in [0.3, 1.2]
            # Values near 0 mean the message compresses away (redundant)
            # Values near 1+ mean the message is incompressible (novel)
            novelty = max(0.0, min(1.0, compression_cost))

            return max(0.05, novelty)

        except Exception as e:
            log.warning("Compression novelty failed: %s — returning neutral 0.5", e)
            return 0.5

    # ------------------------------------------------------------------
    # Signal 2: Salience — delegates to truememory.salience when available
    # ------------------------------------------------------------------

    def _compute_salience(self, fact: str, category: str = "") -> float:
        """
        Encoding-specific salience: "is this worth remembering?"

        Uses a hybrid scorer (encoding_salience_d) that applies a
        rule-based, length-independent scorer for short messages
        (<=50 chars) and L3's retrieval scorer for longer text.
        Falls back to L3 + category boost if the encoding salience
        module is unavailable.
        """
        try:
            from truememory.ingest.encoding_salience import encoding_salience_d
            return encoding_salience_d(fact, category)
        except ImportError:
            if _HAS_TRUEMEMORY_SALIENCE and _tm_salience is not None:
                try:
                    base = float(_tm_salience(fact, "chat"))
                except Exception as e:
                    log.debug("truememory salience failed, using fallback: %s", e)
                    base = self._fallback_salience(fact)
            else:
                base = self._fallback_salience(fact)
            cat = (category or "").strip().lower()
            boost = _CATEGORY_SALIENCE_BOOST.get(cat, 0.05)
            return max(0.0, min(1.0, base + boost))

    @staticmethod
    def _fallback_salience(fact: str) -> float:
        """Minimal heuristic salience when truememory isn't available."""
        if not fact:
            return 0.0
        length = len(fact)
        if length < 10:
            return 0.1
        if length < 30:
            return 0.25
        if length < 100:
            return 0.40
        return 0.50

    # ------------------------------------------------------------------
    # Signal 3: Prediction error — embedding pair-difference scorer
    # ------------------------------------------------------------------

    def _compute_prediction_error(self, fact: str) -> float:
        """
        Embedding pair-difference prediction error (v044).

        Embeds the (message, nearest_memory) pair as a single text and
        compares it to the (memory, memory) self-pair embedding. When a
        message says something DIFFERENT about the same topic, the pair
        embedding diverges from the self-pair — that divergence is PE.

        Validated in 200-variant sweep across 10 paradigms: AUC 0.730
        standalone, gate AUC 0.796 in three-signal combination. Runs in
        0.3ms/msg using the existing embedding model (no extra download).

        Independent of novelty (r=0.30) and salience (r=0.23).
        """
        if not fact or fact.lower().strip().rstrip("!?.… ") in _PE_NOISE:
            return 0.0
        if len(fact.strip()) < 3:
            return 0.0

        if not self._last_search_results:
            return 0.0

        if self._embed_model is None:
            try:
                from truememory.vector_search import get_model
                self._embed_model = get_model()
            except Exception:
                return 0.0
        model = self._embed_model

        nearest = self._last_search_results[0]
        mem_content = nearest.get("content", "")
        if not mem_content:
            return 0.0

        try:
            embeddings = model.encode([fact, mem_content,
                                       fact + " [SEP] " + mem_content,
                                       mem_content + " [SEP] " + mem_content])
            emb_fact = embeddings[0]
            emb_mem = embeddings[1]

            norm_f = float(np.linalg.norm(emb_fact))
            norm_m = float(np.linalg.norm(emb_mem))
            if norm_f < 1e-10 or norm_m < 1e-10:
                return 0.0
            sim = float(np.dot(emb_fact, emb_mem)) / (norm_f * norm_m)
            if sim < 0.2:
                return 0.0

            pair_emb = embeddings[2]
            self_emb = embeddings[3]

            norm_p = float(np.linalg.norm(pair_emb))
            norm_s = float(np.linalg.norm(self_emb))
            if norm_p < 1e-10 or norm_s < 1e-10:
                return 0.0
            pair_sim = float(np.dot(pair_emb, self_emb)) / (norm_p * norm_s)

            pe = max(0.0, min(1.0, 1.0 - pair_sim))
            return pe

        except Exception as e:
            log.debug("PE embedding scorer failed: %s", e)
            return 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _search(self, query: str, limit: int = 5) -> list[dict]:
        """Search existing memories, scoped to user if set."""
        try:
            if self.user_id:
                return self.memory.search(query, user_id=self.user_id, limit=limit)
            else:
                return self.memory.search(query, limit=limit)
        except Exception as e:
            log.warning("Memory search failed during encoding gate: %s", e)
            return []

    def _explain(
        self,
        novelty: float,
        salience: float,
        pred_error: float,
        score: float,
        encode: bool,
        floored: bool = False,
    ) -> str:
        """Human-readable explanation of the encoding decision."""
        parts = []

        if floored:
            parts.append(f"salience below floor ({self.salience_floor:.2f})")

        if novelty > 0.7:
            parts.append("novel")
        elif novelty < 0.2:
            parts.append("familiar")
        else:
            parts.append(f"partially novel ({novelty:.0%})")

        if salience > 0.6:
            parts.append("high salience")
        elif salience < 0.3:
            parts.append("low salience")

        if pred_error > 0.6:
            parts.append("high prediction error")
        elif pred_error > 0.3:
            parts.append("moderate surprise")

        verdict = "ENCODE" if encode else "SKIP"
        return (
            f"{verdict} score={score:.2f} "
            f"(n={novelty:.2f}, s={salience:.2f}, p={pred_error:.2f}) "
            f"threshold={self.threshold:.2f} — {', '.join(parts)}"
        )

    def log_batch_summary(self) -> dict:
        """Log summary statistics for the current batch and return them."""
        n = len(self._batch_scores)
        if n == 0:
            return {"evaluated": 0}

        passed = sum(1 for s in self._batch_scores if s >= self.threshold)
        blocked = n - passed

        def _stats(vals: list[float]) -> str:
            mn = min(vals)
            mx = max(vals)
            avg = sum(vals) / len(vals)
            return f"[{mn:.2f}, {mx:.2f}, mean={avg:.2f}]"

        log.info(
            "gate summary: %d evaluated, %d passed (%d%%), %d blocked. "
            "score_range=%s threshold=%.2f",
            n, passed, round(passed / n * 100) if n else 0, blocked,
            _stats(self._batch_scores), self.threshold,
        )
        log.debug(
            "gate signals: novelty=%s salience=%s pe=%s",
            _stats(self._batch_novelties),
            _stats(self._batch_saliences),
            _stats(self._batch_pes),
        )

        return {
            "evaluated": n,
            "passed": passed,
            "blocked": blocked,
            "score_min": min(self._batch_scores),
            "score_max": max(self._batch_scores),
        }

    def reset_batch(self):
        """Clear the batch-level caches (call between transcripts)."""
        self._last_search_results = []
        self._batch_scores.clear()
        self._batch_novelties.clear()
        self._batch_saliences.clear()
        self._batch_pes.clear()
