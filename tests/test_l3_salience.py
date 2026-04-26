"""MEMORIST-L3 regression tests.

Ensures the learned logistic regression salience scorer:

1. Loads weights from l3_weights.json at import time.
2. Produces scores in [0, 1] for all inputs.
3. Ranks factual content above noise.
4. Fixes wrong signs from the hand-tuned scorer (arousal, date, newline).
5. Respects length dominance (~30x weight vs hand-tuned).
6. Falls back to the legacy scorer when weights are missing.
7. Preserves the public API surface.

Context: MEMORIST-L3 research (10-fold LOCO-CV, n=5882) found C4
(logistic regression) significantly outperforms the hand-tuned baseline
(+0.045 AUC, p=0.012). See ``_working/memorist/l3_salience/REPORT.md``.
"""
from __future__ import annotations

from math import log

import pytest

from truememory.salience import (
    _L3_BIAS,
    _L3_WEIGHTS,
    _NOISE_EXACT,
    _extract_features,
    apply_salience_guard,
    compute_message_salience,
    detect_entities,
    filter_by_entity,
    filter_by_salience,
)


# ---------------------------------------------------------------------------
# Test message constants
# ---------------------------------------------------------------------------

FACTUAL_LONG = (
    "We raised $2M in our seed round from Sequoia and are hiring "
    "3 engineers in San Francisco. The team will start on January 15th "
    "and we expect to ship the beta by March. Our burn rate is $150K/month."
)
NOISE_SHORT = "ok"
NOISE_LOL = "lol"
AROUSAL_MSG = "oh that's amazing and incredible, truly thrilling wow"
PLAIN_FACTUAL_SIMILAR_LEN = (
    "The quarterly revenue was forty two million dollars from enterprise"
)
DATE_MSG = "Let's meet on January 15th or February 20th to discuss"
NEWLINE_MSG = (
    "First item on the agenda\nSecond item on the agenda\n"
    "Third item on the agenda\nFourth item"
)
PLAIN_FACTUAL_LONG = (
    "The company reported strong earnings with revenue growing twenty percent "
    "year over year driven by enterprise customer expansion"
)
SHORT_FACTUAL = "Revenue grew by 20%."
LONG_FACTUAL = (
    "Our company completed a strategic acquisition of DataTech Corp for $45M, "
    "bringing their team of 120 engineers into our organization. The deal closed "
    "on March 15th after six months of negotiations. We expect this acquisition "
    "to increase our annual recurring revenue by $12M within the first year. "
    "The integration plan includes merging their cloud infrastructure with ours "
    "over the next quarter, consolidating three data centers into our primary "
    "facility in Austin, and cross-training both engineering teams on the "
    "combined platform. CEO Jane Smith will lead the integration effort."
)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

class TestWeightLoading:
    def test_weights_loaded(self):
        assert _L3_WEIGHTS is not None
        assert _L3_BIAS is not None

    def test_weights_count(self):
        assert len(_L3_WEIGHTS) == 13

    def test_weights_are_floats(self):
        for w in _L3_WEIGHTS:
            assert isinstance(w, (int, float))
        assert isinstance(_L3_BIAS, (int, float))


# ---------------------------------------------------------------------------
# Basic ranking
# ---------------------------------------------------------------------------

class TestBasicRanking:
    def test_factual_beats_noise(self):
        factual = compute_message_salience(FACTUAL_LONG)
        noise_ok = compute_message_salience(NOISE_SHORT)
        noise_lol = compute_message_salience(NOISE_LOL)
        assert factual > noise_ok
        assert factual > noise_lol

    def test_factual_score_substantial(self):
        assert compute_message_salience(FACTUAL_LONG) > 0.5

    def test_noise_score_low(self):
        assert compute_message_salience(NOISE_SHORT) < 0.3
        assert compute_message_salience(NOISE_LOL) < 0.3


# ---------------------------------------------------------------------------
# Output range
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "",
    "   ",
    "ok",
    "lol",
    FACTUAL_LONG,
    AROUSAL_MSG,
    "\U0001F600\U0001F601\U0001F602",
    "x" * 10000,
    "ALL CAPS WORDS HERE NOW TODAY",
    "got married and got promoted and having a baby",
    "$100 and $200 and $300",
    "January 15th and February 20th and March 3rd",
    "line one\nline two\nline three\nline four\nline five and more text here",
], ids=[
    "empty", "whitespace", "ok", "lol", "factual_long", "arousal",
    "emoji_only", "very_long", "all_caps", "life_events", "money_heavy",
    "date_heavy", "newline_heavy",
])
def test_output_range(text):
    score = compute_message_salience(text)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    def test_feature_count(self):
        feats = _extract_features("hello world")
        assert len(feats) == 13

    def test_all_features_numeric(self):
        feats = _extract_features(FACTUAL_LONG, "email")
        for f in feats:
            assert isinstance(f, (int, float))

    def test_length_feature(self):
        for text in ["hi", "hello world", FACTUAL_LONG]:
            feats = _extract_features(text)
            expected = log(1 + len(text.strip())) / 7.0
            assert abs(feats[2] - expected) < 1e-10

    def test_noise_feature(self):
        assert _extract_features("ok")[0] == 1.0
        assert _extract_features("lol")[0] == 1.0
        assert _extract_features(FACTUAL_LONG)[0] == 0.0

    def test_modality_feature(self):
        assert _extract_features("test", "email")[6] == 1.0
        assert _extract_features("test", "ocr")[6] == 1.0
        assert _extract_features("test", "imessage")[6] == 0.0
        assert _extract_features("test", "")[6] == 0.0

    def test_money_feature(self):
        feats = _extract_features("We raised $2M in funding")
        assert feats[4] > 0.0

    def test_life_event_feature(self):
        feats = _extract_features("She got married last year")
        assert feats[12] > 0.0


# ---------------------------------------------------------------------------
# Wrong signs fixed (key regression tests)
# ---------------------------------------------------------------------------

class TestWrongSignsFixed:
    def test_arousal_does_not_dominate(self):
        arousal = compute_message_salience(AROUSAL_MSG)
        plain = compute_message_salience(PLAIN_FACTUAL_SIMILAR_LEN)
        assert plain > arousal

    def test_dates_do_not_inflate(self):
        date_score = compute_message_salience(DATE_MSG)
        plain_score = compute_message_salience(PLAIN_FACTUAL_LONG)
        assert date_score <= plain_score + 0.05

    def test_newlines_do_not_inflate(self):
        nl_score = compute_message_salience(NEWLINE_MSG)
        plain_score = compute_message_salience(PLAIN_FACTUAL_LONG)
        assert nl_score <= plain_score + 0.05


# ---------------------------------------------------------------------------
# Length dominance
# ---------------------------------------------------------------------------

def test_long_much_higher_than_short():
    short = compute_message_salience(SHORT_FACTUAL)
    long = compute_message_salience(LONG_FACTUAL)
    assert long - short > 0.15


# ---------------------------------------------------------------------------
# Fallback behavior
# ---------------------------------------------------------------------------

class TestFallback:
    def test_fallback_returns_valid_score(self, monkeypatch):
        import truememory.salience as sal
        monkeypatch.setattr(sal, "_L3_WEIGHTS", None)
        monkeypatch.setattr(sal, "_L3_BIAS", None)
        score = sal.compute_message_salience("test message with content")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_fallback_handles_noise(self, monkeypatch):
        import truememory.salience as sal
        monkeypatch.setattr(sal, "_L3_WEIGHTS", None)
        monkeypatch.setattr(sal, "_L3_BIAS", None)
        assert sal.compute_message_salience("ok") < 0.3


# ---------------------------------------------------------------------------
# API compatibility
# ---------------------------------------------------------------------------

class TestAPICompatibility:
    def test_apply_salience_guard_callable(self):
        assert callable(apply_salience_guard)

    def test_detect_entities_callable(self):
        assert callable(detect_entities)

    def test_filter_by_entity_callable(self):
        assert callable(filter_by_entity)

    def test_filter_by_salience_callable(self):
        assert callable(filter_by_salience)


# ---------------------------------------------------------------------------
# Noise set cleanup
# ---------------------------------------------------------------------------

class TestNoiseSet:
    @pytest.mark.parametrize("punct", ["?", "??", "???", "!", "!!", "!!!"])
    def test_unreachable_punctuation_removed(self, punct):
        assert punct not in _NOISE_EXACT

    @pytest.mark.parametrize("word", ["ok", "lol", "yeah", "thanks", "brb"])
    def test_real_noise_still_present(self, word):
        assert word in _NOISE_EXACT
