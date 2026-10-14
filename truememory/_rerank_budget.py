"""Reranker input cap and token-budgeted batches that bound attention scratch."""
from __future__ import annotations


# Longer pairs are truncated longest side first. Stored memories are far
# shorter; the cap bounds oversized queries.
RERANK_MAX_TOKENS = 1024
# Attention scratch grows with batch x length^2. This budget keeps the default
# 64-pair batch for every pair up to 512 tokens.
RERANK_TOKEN_BUDGET = 64 * 512 * 512


def _length_attribute(model: object) -> str | None:
    # sentence-transformers 5.5+ exposes max_seq_length. Older CrossEncoders
    # expose max_length; 3.x leaves it None, meaning the tokenizer's limit.
    if type(getattr(model, "max_seq_length", None)) is int:
        return "max_seq_length"
    if hasattr(model, "max_length"):
        return "max_length"
    return None


def token_limit(model: object) -> int | None:
    """The cross-encoder's effective input token limit, if it can be read."""
    name = _length_attribute(model)
    value = None if name is None else getattr(model, name, None)
    if value is None and name == "max_length":
        value = getattr(getattr(model, "tokenizer", None), "model_max_length", None)
    return value if type(value) is int and value > 0 else None


def cap_reranker_length(model: object) -> None:
    """Bound a cross-encoder's input length without raising a smaller limit."""
    name = _length_attribute(model)
    current = token_limit(model)
    if name is not None and (current is None or current > RERANK_MAX_TOKENS):
        setattr(model, name, RERANK_MAX_TOKENS)


def memory_pressure_divisor(available: int | None = None) -> int:
    """1 normally, 2 below 6 GiB available and 4 below 2 GiB."""
    if available is None:
        try:
            import psutil
            available = psutil.virtual_memory().available
        except (ImportError, OSError):
            return 1
    if available < 2 * 1024 ** 3:
        return 4
    if available < 6 * 1024 ** 3:
        return 2
    return 1


def rerank_token_budget(available: int | None = None) -> int:
    """Per-batch attention budget, reduced while the host is short of memory."""
    return RERANK_TOKEN_BUDGET // memory_pressure_divisor(available)


def pair_token_lengths(model: object, pairs: list) -> list[int] | None:
    """Truncated token length of each pair, or None when it cannot be measured."""
    tokenizer = getattr(model, "tokenizer", None)
    limit = token_limit(model)
    if (tokenizer is None or limit is None
            or not all(isinstance(pair, (list, tuple)) and len(pair) == 2
                       and all(isinstance(text, str) for text in pair) for pair in pairs)):
        return None
    if not pairs:
        return []
    encoded = tokenizer([pair[0] for pair in pairs], [pair[1] for pair in pairs],
                        truncation="longest_first", max_length=limit)
    return [len(ids) for ids in encoded["input_ids"]]


def budget_batch_size(length: int, limit: int, budget: int) -> int:
    return max(1, min(limit, budget // max(1, length) ** 2))


def budgeted_slices(lengths: list[int], limit: int, budget: int) -> list[list[int]] | None:
    """Longest-first index groups sized to the budget.

    None means every default batch already fits, so callers keep their
    original single call and its exact results.
    """
    if budget_batch_size(max(lengths, default=0), limit, budget) >= limit:
        return None
    order = sorted(range(len(lengths)), key=lambda index: -lengths[index])
    slices, start = [], 0
    while start < len(order):
        size = budget_batch_size(lengths[order[start]], limit, budget)
        slices.append(order[start:start + size])
        start += size
    return slices


def budgeted_predict(model: object, pairs: list, batch_size: int):
    """CrossEncoder.predict with long pairs scored in smaller batches."""
    lengths = (pair_token_lengths(model, pairs)
               if type(batch_size) is int and batch_size > 0 else None)
    slices = None if lengths is None else budgeted_slices(lengths, batch_size, rerank_token_budget())
    if slices is None:
        return model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    import numpy as np
    results: list = [None] * len(pairs)
    for indices in slices:
        values = model.predict([pairs[index] for index in indices], batch_size=len(indices),
                               show_progress_bar=False)
        for position, index in enumerate(indices):
            results[index] = values[position]
    return np.stack(results)
