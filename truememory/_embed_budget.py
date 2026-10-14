"""Embedding input cap and token-budgeted batches that bound attention memory."""
from __future__ import annotations

from truememory._rerank_budget import budgeted_slices, memory_pressure_divisor

# Above the longest stored memory, so existing vectors are unchanged; only
# oversized texts such as pasted documents are truncated.
EMBED_MAX_TOKENS = 2048
# Keeps the default 32-text batch for every text up to 1,024 tokens.
EMBED_TOKEN_BUDGET = 32 * 1024 * 1024


def cap_embed_length(model: object) -> None:
    """Lower an embedder's integer input limit to the cap; static models are untouched."""
    current = getattr(model, "max_seq_length", None)
    if type(current) is int and current > EMBED_MAX_TOKENS:
        model.max_seq_length = EMBED_MAX_TOKENS


def embed_token_budget(available: int | None = None) -> int:
    return EMBED_TOKEN_BUDGET // memory_pressure_divisor(available)


def measurable(model: object) -> bool:
    return getattr(model, "tokenizer", None) is not None and type(getattr(model, "max_seq_length", None)) is int


def text_token_lengths(model: object, texts: object) -> list[int] | None:
    """Truncated token length of each text, or None when it cannot be measured."""
    if not measurable(model) or not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        return None
    if not texts:
        return []
    encoded = model.tokenizer(texts, truncation=True, max_length=model.max_seq_length)
    return [len(ids) for ids in encoded["input_ids"]]


def budgeted_encode(model: object, texts: object, **kwargs: object):
    """SentenceTransformer.encode with long texts embedded in smaller batches.

    Inputs whose default batches already fit keep the original single call.
    """
    batch_size = kwargs.get("batch_size", 32)
    plain = (type(batch_size) is int and batch_size > 0 and not kwargs.get("convert_to_tensor")
             and kwargs.get("output_value", "sentence_embedding") == "sentence_embedding")
    lengths = text_token_lengths(model, texts) if plain else None
    slices = None if lengths is None else budgeted_slices(lengths, batch_size, embed_token_budget())
    if slices is None:
        return model.encode(texts, **kwargs)
    import numpy as np
    results: list = [None] * len(texts)
    for indices in slices:
        values = model.encode([texts[index] for index in indices], **{**kwargs, "batch_size": len(indices)})
        for position, index in enumerate(indices):
            results[index] = values[position]
    return np.stack(results)
