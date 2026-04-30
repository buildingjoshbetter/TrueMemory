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

- **Prediction error** delegates to truememory's existing
  `predictive.compute_surprise_score` which computes an information-
  theoretic surprise signal by comparing extracted facts against prior
  context. Real predictive coding is Bayesian error propagation up a
  hierarchical generative model.

The delegation to `truememory.salience` and `truememory.predictive` is
intentional: those modules already implement the surprise and salience
scoring that truememory uses for retrieval weighting. Reusing them
keeps the encoding gate consistent with the retrieval layer and avoids
code duplication. If either module is unavailable (older truememory
or partial install) a warning is logged at import time and the gate
falls back to internal heuristics.

**What a skeptical reader should know**: the final encoding decision is
`0.40 * novelty + 0.35 * salience + 0.25 * prediction_error >= 0.30`.
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

log = logging.getLogger(__name__)


# Try to import truememory's existing scoring modules. If they're not
# available (older truememory or partial install), fall back to
# internal heuristics.
_HAS_TRUEMEMORY_SALIENCE = False
_HAS_TRUEMEMORY_PREDICTIVE = False
try:
    from truememory.salience import compute_message_salience as _tm_salience
    _HAS_TRUEMEMORY_SALIENCE = True
except ImportError:
    _tm_salience = None

try:
    from truememory.predictive import compute_surprise_score as _tm_surprise
    from truememory.predictive import extract_facts as _tm_extract_facts
    _HAS_TRUEMEMORY_PREDICTIVE = True
except ImportError:
    _tm_surprise = None
    _tm_extract_facts = None

# Log fallback mode loudly at import time so users know when they're
# running degraded — previously this was silent and users couldn't tell
# whether the delegation to truememory's scoring was active or not.
if not _HAS_TRUEMEMORY_SALIENCE:
    log.warning(
        "truememory.salience.compute_message_salience not available; "
        "using fallback salience heuristic. Install/upgrade truememory "
        "for the full salience signal."
    )
if not _HAS_TRUEMEMORY_PREDICTIVE:
    log.warning(
        "truememory.predictive.compute_surprise_score not available; "
        "using fallback prediction-error heuristic. Install/upgrade "
        "truememory for the full prediction-error signal."
    )


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
        user_id: Optional user scope for memory searches.
    """

    def __init__(
        self,
        memory,
        threshold: float = 0.30,
        w_novelty: float | None = None,
        w_salience: float | None = None,
        w_prediction_error: float | None = None,
        user_id: str = "",
    ):
        self.memory = memory
        self.threshold = threshold
        if w_novelty is None:
            w_novelty = float(os.environ.get("TRUEMEMORY_GATE_W_NOVELTY", "0.40"))
        if w_salience is None:
            w_salience = float(os.environ.get("TRUEMEMORY_GATE_W_SALIENCE", "0.35"))
        if w_prediction_error is None:
            w_prediction_error = float(os.environ.get("TRUEMEMORY_GATE_W_PE", "0.25"))
        self.w_novelty = w_novelty
        self.w_salience = w_salience
        self.w_prediction_error = w_prediction_error
        self.user_id = user_id
        # Normalized weights so the final score lands in [0, 1]
        total = w_novelty + w_salience + w_prediction_error
        self._norm = total if total > 0 else 1.0
        # Cache of extracted facts from prior candidate facts in the same
        # batch — used so that prediction error can detect contradictions
        # within the batch, not just against stored memories
        self._batch_facts: set[str] = set()
        self._last_search_results: list[dict] = []
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
        pred_error = self._compute_prediction_error(fact, novelty)

        # Weighted sum, normalized to [0, 1]
        raw = (
            novelty * self.w_novelty
            + salience * self.w_salience
            + pred_error * self.w_prediction_error
        )
        score = max(0.0, min(1.0, raw / self._norm))

        should_encode = score >= self.threshold
        reason = self._explain(novelty, salience, pred_error, score, should_encode)

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

        # Add this fact's fingerprint to the batch cache so subsequent
        # facts in the same transcript can detect duplicates/contradictions
        if _HAS_TRUEMEMORY_PREDICTIVE and _tm_extract_facts is not None:
            try:
                self._batch_facts.update(_tm_extract_facts(fact))
            except Exception:
                pass

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
    # Signal 3: Prediction error — delegates to truememory.predictive
    # ------------------------------------------------------------------

    def _compute_prediction_error(self, fact: str, novelty: float) -> float:
        """
        Prediction error proxy.

        Delegates to truememory's `predictive.compute_surprise_score` when
        available, which computes information-theoretic surprise by
        comparing extracted facts (dates, numbers, proper nouns, event
        keywords) against prior context.

        If the predictive module isn't available, falls back to a
        novelty-shaped heuristic.

        Real predictive coding is Bayesian error propagation up a
        hierarchical generative model. This is not that.

        NOTE: Previously this function gated on the novelty score
        (returning fixed values for novelty > 0.9 or < 0.05). That
        coupling was removed (issue #110) so PE can be evaluated
        independently. The linear combination handles signal interaction.
        """
        # Use truememory's surprise score if available
        if _HAS_TRUEMEMORY_PREDICTIVE and _tm_surprise is not None:
            try:
                # Seed with the batch cache of already-processed facts so that
                # within-batch repetition gets penalized
                surprise = float(_tm_surprise(fact, self._batch_facts))
                # Surprise score is 0-1; treat it as prediction error
                # but weight by novelty to prevent totally new topics from
                # dominating (we already captured that in the novelty signal)
                return max(0.0, min(1.0, surprise * 0.9))
            except Exception as e:
                log.debug("truememory surprise failed, using fallback: %s", e)

        # Fallback: use the moderate-similarity heuristic
        return self._fallback_prediction_error(fact)

    def _fallback_prediction_error(self, fact: str) -> float:
        """Fallback prediction error when truememory.predictive isn't available."""
        results = self._last_search_results[:3] if self._last_search_results else self._search(fact, limit=3)
        if not results:
            return 0.3

        top_score = float(results[0].get("score", 0) or 0)
        top_content = results[0].get("content", "")

        # Explicit contradiction check
        if self._looks_like_update(fact, top_content):
            return 0.9

        if 0.3 < top_score < 0.7:
            return 0.6
        elif 0.7 <= top_score < 0.85:
            return 0.2
        else:
            return 0.1

    @staticmethod
    def _looks_like_update(new_fact: str, existing: str) -> bool:
        """Quick heuristic for whether new_fact supersedes existing."""
        new_lower = new_fact.lower()
        old_lower = existing.lower()

        update_verbs = [
            "lives in", "works at", "uses", "prefers", "switched to",
            "moved to", "changed to", "now uses", "started using",
            "is located", "runs on", "deployed to",
        ]
        for verb in update_verbs:
            if verb in new_lower and verb in old_lower:
                new_after = new_lower.split(verb, 1)[-1].strip()[:30]
                old_after = old_lower.split(verb, 1)[-1].strip()[:30]
                if new_after and old_after and new_after != old_after:
                    return True

        if any(m in new_lower for m in [
            "no longer", "not anymore", "stopped", "quit",
            "actually", "correction",
        ]):
            return True

        return False

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
    ) -> str:
        """Human-readable explanation of the encoding decision."""
        parts = []

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
        """Clear the batch-level fact cache (call between transcripts)."""
        self._batch_facts.clear()
        self._last_search_results = []
        self._batch_scores.clear()
        self._batch_novelties.clear()
        self._batch_saliences.clear()
        self._batch_pes.clear()
