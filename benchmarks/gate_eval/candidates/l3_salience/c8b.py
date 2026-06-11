"""C8b — SmolLM2-135M per-message perplexity as salience.

Scoring: compute mean-per-token NLL on the message text. Higher NLL
(more surprising) → higher salience. Normalize using training-fold
mean and std (fit by the harness call to fit()).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


_HF_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ez = math.exp(x)
    return ez / (1.0 + ez)


class Candidate:
    name = "C8b"
    tier = "base_pro"
    model_ids = [_HF_MODEL_ID]

    def __init__(self):
        self._model = None
        self._tok = None
        self._mu = None
        self._sigma = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(_HF_MODEL_ID)
        # Some tokenizers have no pad token; add one.
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            _HF_MODEL_ID, dtype=torch.float32
        )
        self._model.eval()

    def _nll(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0
        torch = self._torch
        with torch.no_grad():
            ids = self._tok(text, return_tensors="pt", truncation=True, max_length=256).input_ids
            if ids.shape[1] < 2:
                return 0.0
            out = self._model(ids, labels=ids)
            return float(out.loss.item())

    def fit(self, messages_train):
        self._load()
        # Compute NLL on a subsample for calibration (too expensive on full).
        # Deterministic seed via index modulo.
        subsample = messages_train[::max(1, len(messages_train) // 300)][:300]
        nlls = []
        for m in subsample:
            nlls.append(self._nll(m.get("content", "") or ""))
        nlls = np.asarray(nlls, dtype=float)
        self._mu = float(nlls.mean())
        self._sigma = float(nlls.std() + 1e-6)

    def score(self, msg) -> float:
        self._load()
        nll = self._nll(msg.get("content", "") or "")
        mu = self._mu if self._mu is not None else 5.0
        sigma = self._sigma if self._sigma is not None else 1.0
        z = (nll - mu) / sigma
        return float(_sigmoid(z))
