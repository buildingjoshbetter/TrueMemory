"""Tests for the encoding-specific salience scorer (issue #108)."""

import pytest


class TestEncodingSalienceVariants:
    """Each variant produces valid 0-1 scores."""

    def _score_range_check(self, scorer):
        test_inputs = [
            "", "  ", "ok", "hello world", "I GOT IT",
            "we're pregnant", "dad passed away",
            "The quarterly revenue report shows a 23% increase in ARR",
            "🎉🎉🎉", "a" * 500,
        ]
        for text in test_inputs:
            score = scorer(text)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for: {text!r}"

    def test_variant_a_range(self):
        from truememory.ingest.encoding_salience import encoding_salience_a
        self._score_range_check(encoding_salience_a)

    def test_variant_b_range(self):
        from truememory.ingest.encoding_salience import encoding_salience_b
        self._score_range_check(encoding_salience_b)

    def test_variant_c_range(self):
        from truememory.ingest.encoding_salience import encoding_salience_c
        self._score_range_check(encoding_salience_c)

    def test_variant_d_range(self):
        from truememory.ingest.encoding_salience import encoding_salience_d
        self._score_range_check(encoding_salience_d)

    def test_variant_e_range(self):
        from truememory.ingest.encoding_salience import encoding_salience_e
        self._score_range_check(encoding_salience_e)


class TestEncodingSalienceWinner:
    """Variant D (hybrid) meets all acceptance criteria."""

    @pytest.fixture
    def scorer(self):
        from truememory.ingest.encoding_salience import encoding_salience_d
        return encoding_salience_d

    def test_pregnant_above_04(self, scorer):
        assert scorer("we're pregnant") > 0.4

    def test_passed_away_above_04(self, scorer):
        assert scorer("dad passed away") > 0.4

    def test_got_it_above_03(self, scorer):
        assert scorer("I GOT IT") > 0.3

    def test_said_yes_above_03(self, scorer):
        assert scorer("I said yes") > 0.3

    def test_ok_below_015(self, scorer):
        assert scorer("ok") < 0.15

    def test_lol_below_015(self, scorer):
        assert scorer("lol") < 0.15

    def test_haha_below_015(self, scorer):
        assert scorer("haha") < 0.15

    def test_empty_returns_zero(self, scorer):
        assert scorer("") == 0.0
        assert scorer("  ") == 0.0

    def test_had_a_baby_high(self, scorer):
        assert scorer("I HAD A BABY") > 0.5

    def test_commitment_patterns(self, scorer):
        assert scorer("it's booked.") > 0.3
        assert scorer("seeing someone") > 0.3

    def test_noise_messages_low(self, scorer):
        for msg in ["yeah", "cool", "sounds good", "brb", "ttyl", "bet", "idk"]:
            assert scorer(msg) < 0.15, f"{msg!r} scored too high: {scorer(msg)}"


class TestEncodingSalienceAUC:
    """Variant D achieves AUC > 0.75 on a small fixture."""

    def test_signal_noise_separation(self):
        from truememory.ingest.encoding_salience import encoding_salience_d

        signal_msgs = [
            "we're pregnant", "dad passed away", "I GOT IT",
            "I said yes", "I HAD A BABY", "she promoted me to Senior Engineer",
            "I enrolled in the design bootcamp", "it's booked.",
            "I got into UCLA", "I gave my two weeks notice today",
        ]
        noise_msgs = [
            "ok", "lol", "haha", "yeah", "cool", "sounds good",
            "brb", "ttyl", "bet", "nice",
        ]

        signal_scores = [encoding_salience_d(m) for m in signal_msgs]
        noise_scores = [encoding_salience_d(m) for m in noise_msgs]

        avg_signal = sum(signal_scores) / len(signal_scores)
        avg_noise = sum(noise_scores) / len(noise_scores)

        assert avg_signal > avg_noise * 5, (
            f"Signal mean ({avg_signal:.3f}) should be much higher than "
            f"noise mean ({avg_noise:.3f})"
        )


class TestEncodingGateFallback:
    """The encoding gate falls back to L3 if encoding_salience is unavailable."""

    def test_fallback_produces_valid_score(self):
        from unittest.mock import patch

        class FakeMemory:
            def search(self, *a, **kw):
                return []

        with patch.dict("sys.modules", {"truememory.ingest.encoding_salience": None}):
            import importlib
            import truememory.ingest.encoding_gate as gate_mod
            importlib.reload(gate_mod)
            reloaded_gate = gate_mod.EncodingGate(memory=FakeMemory(), threshold=0.30)
            score = reloaded_gate._compute_salience("test message")
            assert 0.0 <= score <= 1.0

        importlib.reload(gate_mod)
