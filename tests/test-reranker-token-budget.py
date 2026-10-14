"""Reranker input cap and token-budgeted batches bound attention scratch."""
from __future__ import annotations

import numpy as np
import pytest

from truememory import rerank_budget
from truememory.rerank_budget import (
    RERANK_MAX_TOKENS, RERANK_TOKEN_BUDGET, budget_batch_size, budgeted_predict,
    budgeted_slices, cap_reranker_length, pair_token_lengths, rerank_token_budget,
)


class FakeCrossEncoder:
    """Word count stands in for tokens; scores depend only on each pair."""

    def __init__(self, max_seq_length: int | None = 1024) -> None:
        self.max_seq_length = max_seq_length
        self.calls: list[tuple[int, int]] = []

    def tokenizer(self, first, second, truncation, max_length):
        assert truncation == "longest_first"
        return {"input_ids": [[0] * min(len(a.split()) + len(b.split()) + 3, max_length)
                              for a, b in zip(first, second)]}

    @staticmethod
    def _input_length(pair) -> int:
        return len(pair[0]) + len(pair[1])

    def predict(self, pairs, batch_size, show_progress_bar=False):
        self.calls.append((len(pairs), batch_size))
        return np.array([len(b.split()) / 1000 + len(a) / 1e6 for a, b in pairs], dtype=np.float32)


def _pairs(*word_counts: int) -> list[list[str]]:
    return [["synthetic query", " ".join(["word"] * count)] for count in word_counts]


def test_cap_lowers_long_limits_only():
    for current, expected in ((8192, RERANK_MAX_TOKENS), (None, RERANK_MAX_TOKENS), (512, 512)):
        model = FakeCrossEncoder(current)
        cap_reranker_length(model)
        assert model.max_seq_length == expected


def test_budget_shrinks_under_memory_pressure():
    assert rerank_token_budget(16 * 1024 ** 3) == RERANK_TOKEN_BUDGET
    assert rerank_token_budget(4 * 1024 ** 3) == RERANK_TOKEN_BUDGET // 2
    assert rerank_token_budget(1024 ** 3) == RERANK_TOKEN_BUDGET // 4


def test_default_batches_kept_for_pairs_up_to_512_tokens():
    assert budget_batch_size(512, 64, RERANK_TOKEN_BUDGET) == 64
    assert budgeted_slices([512, 40, 300] * 30, 64, RERANK_TOKEN_BUDGET) is None


def test_long_pairs_go_longest_first_in_budgeted_slices():
    lengths = [50] * 80 + [1024] * 20
    slices = budgeted_slices(lengths, 64, RERANK_TOKEN_BUDGET)
    assert sorted(index for group in slices for index in group) == list(range(100))
    for group in slices:
        assert len(group) * max(lengths[index] for index in group) ** 2 <= RERANK_TOKEN_BUDGET or len(group) == 1
        assert len(group) <= 64
    assert [len(group) for group in slices] == [16, 16, 64, 4]


def test_short_search_is_one_unchanged_predict_call():
    model = FakeCrossEncoder()
    pairs = _pairs(*range(10, 110))
    scores = budgeted_predict(model, pairs, 64)
    assert model.calls == [(100, 64)]
    assert scores.shape == (100,)


def test_long_search_scores_match_and_respect_budget(monkeypatch):
    monkeypatch.setattr(rerank_budget, "rerank_token_budget", lambda: RERANK_TOKEN_BUDGET)
    model = FakeCrossEncoder()
    pairs = _pairs(*([20] * 70 + [2000] * 30))
    scores = budgeted_predict(model, pairs, 64)
    expected = FakeCrossEncoder().predict(pairs, batch_size=len(pairs))
    assert np.array_equal(scores, expected)
    assert sum(size for size, _ in model.calls) == 100
    lengths = pair_token_lengths(model, pairs)
    assert max(lengths) == RERANK_MAX_TOKENS
    assert all(batch == size and size <= 64 for size, batch in model.calls)
    assert model.calls[0][0] == RERANK_TOKEN_BUDGET // RERANK_MAX_TOKENS ** 2


def test_unmeasurable_model_keeps_single_call():
    class Proxy:
        calls = []

        def predict(self, pairs, batch_size, show_progress_bar=False):
            self.calls.append(len(pairs))
            return np.zeros(len(pairs), dtype=np.float32)

    proxy = Proxy()
    budgeted_predict(proxy, _pairs(5000, 5000), 64)
    assert proxy.calls == [2]


def test_model_server_rerank_uses_budgeted_slices(monkeypatch):
    from truememory.model_server import ModelServer

    fake = FakeCrossEncoder()
    monkeypatch.setattr(ModelServer, "_get_reranker", lambda self, model_name=None, deadline=None: fake)
    monkeypatch.setattr(rerank_budget, "rerank_token_budget", lambda: RERANK_TOKEN_BUDGET)
    pairs = _pairs(*([30] * 50 + [1500] * 10))
    response = ModelServer().handle_request({"op": "rerank", "pairs": pairs, "model_name": "m"})
    assert response["ok"] is True
    expected = FakeCrossEncoder().predict(pairs, batch_size=len(pairs))
    assert np.allclose(response["scores"], expected, rtol=0, atol=0)
    assert max(size for size, _ in fake.calls) <= 64
    assert fake.calls[0] == (RERANK_TOKEN_BUDGET // RERANK_MAX_TOKENS ** 2, RERANK_TOKEN_BUDGET // RERANK_MAX_TOKENS ** 2)


def test_model_server_short_rerank_unchanged(monkeypatch):
    from truememory.model_server import ModelServer

    fake = FakeCrossEncoder()
    monkeypatch.setattr(ModelServer, "_get_reranker", lambda self, model_name=None, deadline=None: fake)
    response = ModelServer().handle_request({"op": "rerank", "pairs": _pairs(*range(5, 45)), "model_name": "m"})
    assert response["ok"] is True
    assert fake.calls == [(40, 64)]


@pytest.mark.parametrize("length", [1, 512, 513, 1024])
def test_batch_size_never_zero_or_above_limit(length):
    assert 1 <= budget_batch_size(length, 64, RERANK_TOKEN_BUDGET) <= 64
