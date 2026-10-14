"""Embedding input cap and token-budgeted batches bound attention memory."""
from __future__ import annotations

import numpy as np
import pytest

from truememory import _embed_budget as embed_budget
from truememory._embed_budget import (
    EMBED_MAX_TOKENS, EMBED_TOKEN_BUDGET, budgeted_encode, cap_embed_length, embed_token_budget,
    text_token_lengths,
)


class FakeEmbedder:
    """Word count stands in for tokens; each vector depends only on its text."""

    def __init__(self, max_seq_length: int | None = 32768) -> None:
        self.max_seq_length = max_seq_length
        self.calls: list[tuple[int, int]] = []

    def tokenizer(self, texts, truncation, max_length):
        assert truncation is True
        return {"input_ids": [[0] * min(len(text.split()) + 2, max_length) for text in texts]}

    def encode(self, texts, batch_size=32, show_progress_bar=False, **kwargs):
        self.calls.append((len(texts), batch_size))
        return np.array([[len(text.split()), len(text) % 7] for text in texts], dtype=np.float32)


def _texts(*word_counts: int) -> list[str]:
    return [" ".join(["word"] * count) for count in word_counts]


def test_cap_lowers_long_limits_only():
    for current, expected in ((32768, EMBED_MAX_TOKENS), (512, 512), (None, None)):
        model = FakeEmbedder(current)
        cap_embed_length(model)
        assert model.max_seq_length == expected
    cap_embed_length(object())


def test_budget_shrinks_under_memory_pressure():
    assert embed_token_budget(16 * 1024 ** 3) == EMBED_TOKEN_BUDGET
    assert embed_token_budget(4 * 1024 ** 3) == EMBED_TOKEN_BUDGET // 2
    assert embed_token_budget(1024 ** 3) == EMBED_TOKEN_BUDGET // 4


def test_texts_up_to_1024_tokens_keep_one_unchanged_call(monkeypatch):
    monkeypatch.setattr(embed_budget, "embed_token_budget", lambda: EMBED_TOKEN_BUDGET)
    model = FakeEmbedder(EMBED_MAX_TOKENS)
    texts = _texts(*([1000] * 40 + [5] * 40))
    values = budgeted_encode(model, texts, batch_size=32, show_progress_bar=False)
    assert model.calls == [(80, 32)]
    assert values.shape == (80, 2)


def test_long_texts_split_longest_first_and_keep_order(monkeypatch):
    monkeypatch.setattr(embed_budget, "embed_token_budget", lambda: EMBED_TOKEN_BUDGET)
    model = FakeEmbedder(EMBED_MAX_TOKENS)
    texts = _texts(*([20] * 60 + [5000] * 12))
    values = budgeted_encode(model, texts, batch_size=32, show_progress_bar=False)
    assert np.array_equal(values, FakeEmbedder().encode(texts))
    assert sum(size for size, _ in model.calls) == 72
    assert model.calls[0] == (EMBED_TOKEN_BUDGET // EMBED_MAX_TOKENS ** 2,) * 2
    assert max(text_token_lengths(model, texts)) == EMBED_MAX_TOKENS
    assert all(batch == size <= 32 for size, batch in model.calls)


def test_tensor_output_and_static_models_keep_single_call():
    model = FakeEmbedder(EMBED_MAX_TOKENS)
    budgeted_encode(model, _texts(5000, 5000), batch_size=32, convert_to_tensor=True)
    assert model.calls == [(2, 32)]

    class Static:
        calls = []

        def encode(self, texts, **kwargs):
            self.calls.append(len(texts))
            return np.zeros((len(texts), 2), dtype=np.float32)

    static = Static()
    from truememory.mps_utils import encode_with_model_ownership
    encode_with_model_ownership(static, _texts(5000, 5000), batch_size=32)
    assert static.calls == [2]


def test_local_encode_wrappers_use_the_budget(monkeypatch):
    monkeypatch.setattr(embed_budget, "embed_token_budget", lambda: EMBED_TOKEN_BUDGET)
    from truememory.mps_utils import encode_with_model_ownership, encode_with_mps_fallback
    for wrapper in (encode_with_model_ownership, encode_with_mps_fallback):
        model = FakeEmbedder(EMBED_MAX_TOKENS)
        texts = _texts(*([10] * 30 + [4000] * 9))
        values = wrapper(model, texts, batch_size=32, show_progress_bar=False)
        assert np.array_equal(values, FakeEmbedder().encode(texts))
        assert model.calls[0][0] == EMBED_TOKEN_BUDGET // EMBED_MAX_TOKENS ** 2


def test_model_server_embed_uses_budgeted_slices(monkeypatch):
    from truememory.model_server import ModelServer, _EmbedState

    monkeypatch.setattr(embed_budget, "embed_token_budget", lambda: EMBED_TOKEN_BUDGET)
    server = ModelServer()
    model = FakeEmbedder(EMBED_MAX_TOKENS)
    server._embed_state = _EmbedState(model=model, tier="", model_id="model2vec")
    texts = _texts(*([30] * 50 + [3000] * 10))
    response = server.handle_request({"op": "embed", "texts": texts, "tier": ""})
    assert response["ok"] is True
    assert np.array_equal(np.asarray(response["vectors"]), FakeEmbedder().encode(texts))
    assert model.calls[0] == (EMBED_TOKEN_BUDGET // EMBED_MAX_TOKENS ** 2,) * 2
    assert all(size <= 32 for size, _ in model.calls)


def test_model_server_short_embed_unchanged():
    from truememory.model_server import ModelServer, _EmbedState

    server = ModelServer()
    model = FakeEmbedder(EMBED_MAX_TOKENS)
    server._embed_state = _EmbedState(model=model, tier="", model_id="model2vec")
    response = server.handle_request({"op": "embed", "texts": _texts(*range(5, 45)), "tier": ""})
    assert response["ok"] is True
    assert model.calls == [(32, 32), (8, 32)]


@pytest.mark.parametrize("length", [1, 1024, 1025, 2048])
def test_batch_size_bounds(length):
    from truememory._rerank_budget import budget_batch_size
    assert 1 <= budget_batch_size(length, 32, EMBED_TOKEN_BUDGET) <= 32
