#!/usr/bin/env python3
"""
120-variant novelty sweep for the encoding gate.

Every variant scores a message's NOVELTY — how much new information it adds
relative to what's already stored. Novelty ≠ salience: novelty requires
comparing against stored memories.

Signature: variant_NNN(content, memory_contents, memory_embeddings=None) -> float [0,1]
"""

from __future__ import annotations

import gzip
import math
import re
from collections import Counter
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Embedding model — loaded once, shared by all variants
# ---------------------------------------------------------------------------

_EMBEDDER = None


def set_embedder(model):
    global _EMBEDDER
    _EMBEDDER = model


def _embed(texts: list[str]) -> np.ndarray:
    if _EMBEDDER is None:
        raise RuntimeError("Embedder not set — call set_embedder(model) first")
    return _EMBEDDER.encode(texts)


def _embed_one(text: str) -> np.ndarray:
    return _embed([text])[0]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "i", "a", "the", "is", "it", "in", "to", "and", "of", "for", "on", "at",
    "by", "an", "or", "so", "if", "my", "we", "he", "she", "do", "no", "up",
    "be", "but", "not", "you", "all", "can", "her", "was", "one", "our", "out",
    "are", "has", "his", "how", "its", "let", "may", "new", "now", "old", "see",
    "way", "who", "did", "get", "got", "had", "him", "own", "say", "too", "use",
    "oh", "hey", "hi", "ok", "yeah", "yes", "like", "just", "that", "this",
    "with", "have", "from", "they", "been", "said", "each", "when", "what",
    "your", "will", "than", "them", "then", "some", "time", "very", "make",
    "also", "into", "only", "come", "made", "well", "back", "much", "more",
    "about", "would", "could", "after", "first", "other", "these", "which",
    "those", "here", "there", "even", "still", "down", "off", "over", "such",
    "take", "find", "give", "most", "tell", "think", "help", "every", "last",
    "long", "great", "little", "right", "going", "know", "want",
    "actually", "really", "literally", "honestly", "though", "because",
    "already", "probably", "definitely", "seriously",
})

_FILLER_WORDS = frozenset({
    "haha", "hahaha", "lol", "lmao", "omg", "like", "yeah", "um", "uh",
    "so", "well", "just", "hmm", "ok", "okay", "ya", "yep", "nah",
    "dude", "bro", "man", "wow", "damn", "ugh", "yikes", "ooh",
})

_COMMITMENT_RE = re.compile(
    r"\b(?:"
    r"i\s+(?:got|did|made|found|built|started|quit|left|joined|enrolled|"
    r"accepted|submitted|finished|completed|signed|bought|sold|moved|"
    r"said|told|asked|proposed|created|launched|shipped|published|"
    r"passed|graduated|earned|won|lost|broke|fixed)"
    r"|i'm\s+(?:pregnant|engaged|leaving|moving|starting|quitting|"
    r"going\s+to|seeing\s+someone|having\s+a)"
    r"|we're\s+(?:pregnant|engaged|moving|having|getting|doing)"
    r"|i\s+have\s+(?:a\s+baby|cancer|diabetes|a\s+new)"
    r"|she\s+(?:promoted|said\s+yes|agreed|accepted)"
    r"|he\s+(?:proposed|said\s+yes|agreed|accepted)"
    r"|it's\s+(?:booked|official|confirmed|done|over|happening)"
    r"|i\s+gave\s+(?:my\s+(?:two\s+weeks|notice))"
    r"|(?:all|both)\s+(?:three|four|five)?\s*(?:apps?|applications?)\s+submitted"
    r")\b",
    re.IGNORECASE,
)

_NOISE_EXACT = frozenset({
    "ok", "okay", "k", "kk", "yes", "yeah", "yep", "yup", "ya", "yea",
    "no", "nah", "nope", "lol", "lmao", "lmfao", "haha", "hahaha", "heh",
    "omg", "omfg", "wtf", "nice", "cool", "dope", "sick", "lit", "fire",
    "thanks", "thx", "ty", "thank you", "got it", "gotcha",
    "sounds good", "sounds great", "bet", "word", "sure", "for sure",
    "same", "mood", "idk", "idc", "np", "no problem",
    "gn", "goodnight", "good night", "gm", "good morning", "brb", "ttyl",
    "damn", "dude", "bro", "ugh", "wow", "yikes", "ooh", "oof",
    "true", "facts", "right", "exactly", "totally", "absolutely",
    "lmao dead", "im dead", "crying", "screaming",
})

_EVENT_VERBS = frozenset({
    "got", "moved", "started", "quit", "left", "joined", "enrolled",
    "accepted", "submitted", "finished", "completed", "signed", "bought",
    "sold", "launched", "published", "graduated", "earned", "won",
    "proposed", "hired", "fired", "promoted", "resigned", "transferred",
    "built", "created", "shipped", "found", "applied", "booked",
})

_PLACE_INDICATORS = frozenset({
    "street", "avenue", "ave", "boulevard", "blvd", "road", "rd", "lane",
    "drive", "dr", "court", "ct", "way", "place", "plaza", "square",
    "city", "town", "village", "county", "state", "country",
})


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


def _cosine_sims(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    normed = matrix / norms
    qn = np.linalg.norm(query)
    if qn < 1e-10:
        return np.zeros(len(matrix))
    q_normed = query / qn
    return normed @ q_normed


def _ncd(a: str, b: str) -> float:
    ab = a.encode() + b" ".encode() + b.encode()
    ca = len(gzip.compress(a.encode()))
    cb = len(gzip.compress(b.encode()))
    cab = len(gzip.compress(ab))
    return (cab - min(ca, cb)) / max(ca, cb, 1)


def _gz_len(text: str) -> int:
    return len(gzip.compress(text.encode()))


def _char_trigrams(text: str) -> set[str]:
    t = text.lower()
    return {t[i:i+3] for i in range(len(t) - 2)} if len(t) >= 3 else set()


def _word_bigrams(text: str) -> set[tuple[str, str]]:
    words = text.lower().split()
    return {(words[i], words[i+1]) for i in range(len(words) - 1)} if len(words) >= 2 else set()


def _extract_proper_nouns(text: str) -> list[str]:
    words = text.split()
    proper = []
    for w in words:
        cleaned = re.sub(r'[^\w]', '', w)
        if cleaned and cleaned[0].isupper() and cleaned.lower() not in _STOPWORDS and len(cleaned) > 1:
            proper.append(cleaned)
    return proper


def _extract_numbers(text: str) -> list[str]:
    return re.findall(r'\b\d+(?:\.\d+)?%?\b|\$[\d,.]+', text)


def _extract_temporal(text: str) -> list[str]:
    patterns = [
        r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        r'\b\d{4}\b',
        r'\b(?:last|next|this)\s+(?:week|month|year|weekend|summer|winter|spring|fall)\b',
        r'\b(?:yesterday|tomorrow|tonight)\b',
    ]
    results = []
    lower = text.lower()
    for p in patterns:
        results.extend(re.findall(p, lower))
    return results


def _speech_act_score(content: str) -> float:
    lower = content.lower().strip()
    if lower in _NOISE_EXACT:
        return 0.02
    if content.strip().endswith("?") or lower.startswith((
        "what ", "how ", "why ", "where ", "when ", "who ", "which ",
        "do you", "are you", "is it", "can you", "could you",
    )):
        return 0.2
    if _COMMITMENT_RE.search(lower):
        return 0.8
    if re.match(r"^(?:hey|hi|hello|yo|sup|what's up|howdy)", lower):
        return 0.05
    if re.match(r"^(?:haha|lol|lmao|omg|wow|damn|ugh|yikes)", lower):
        return 0.08
    words = re.findall(r"[a-zA-Z]+", content)
    if len(words) >= 5:
        return 0.5
    return 0.25


def _declaration_score(content: str) -> float:
    lower = content.lower().strip()
    if lower in _NOISE_EXACT:
        return 0.02
    if content.strip().endswith("?"):
        return 0.1
    if _COMMITMENT_RE.search(lower):
        return 0.9
    words = re.findall(r"[a-zA-Z]+", content)
    content_words = [w for w in words if w.lower() not in _STOPWORDS]
    if not words:
        return 0.05
    ratio = len(content_words) / len(words)
    has_verb = bool(re.search(r'\b(?:is|are|was|were|have|has|had|do|does|did|will|would|can|could|shall|should|may|might|must)\b', lower))
    if has_verb and ratio > 0.3:
        return min(1.0, 0.3 + ratio * 0.5)
    return ratio * 0.4


def _factual_density(content: str) -> float:
    text = content.strip()
    if not text:
        return 0.0
    lower = text.lower()
    if lower in _NOISE_EXACT:
        return 0.02
    score = 0.0
    proper = _extract_proper_nouns(text)
    score += min(0.3, len(proper) * 0.1)
    numbers = _extract_numbers(text)
    score += min(0.3, len(numbers) * 0.1)
    temporal = _extract_temporal(text)
    score += min(0.2, len(temporal) * 0.1)
    if _COMMITMENT_RE.search(lower):
        score += 0.3
    words = re.findall(r"[a-zA-Z]+", text)
    content_words = [w for w in words if w.lower() not in _STOPWORDS]
    if words:
        score += (len(content_words) / len(words)) * 0.2
    return min(1.0, score)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _nearest_text(content: str, memory_contents: list[str]) -> str:
    """Find most similar memory by word overlap (cheap)."""
    if not memory_contents:
        return ""
    cw = set(content.lower().split())
    best_score = -1
    best_idx = 0
    for i, m in enumerate(memory_contents):
        mw = set(m.lower().split())
        overlap = len(cw & mw)
        total = len(cw | mw) or 1
        score = overlap / total
        if score > best_score:
            best_score = score
            best_idx = i
    return memory_contents[best_idx]


# ============================================================================
# CATEGORY 1: Memory-Comparative Distance (001-020)
# ============================================================================


def variant_001(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Cosine distance to nearest memory (baseline)."""
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 1.0
    emb = _embed_one(content)
    sims = _cosine_sims(emb, memory_embeddings)
    return _clamp(1.0 - float(np.max(sims)))


def variant_002(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Cosine distance to memory centroid."""
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 1.0
    emb = _embed_one(content)
    centroid = memory_embeddings.mean(axis=0)
    return _clamp(1.0 - _cosine_sim(emb, centroid))


def variant_003(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Mahalanobis distance from memory distribution."""
    if memory_embeddings is None or len(memory_embeddings) < 3:
        return 1.0
    emb = _embed_one(content)
    mean = memory_embeddings.mean(axis=0)
    diff = emb - mean
    # Use diagonal covariance for speed (full cov is too slow with 256 dims)
    var = np.var(memory_embeddings, axis=0) + 1e-8
    maha = float(np.sqrt(np.sum(diff ** 2 / var)))
    # Normalize: typical Mahalanobis in 256-dim is ~16. Map to [0,1].
    return _clamp(maha / 32.0)


def variant_004(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Isolation Forest anomaly score on memory embeddings."""
    if memory_embeddings is None or len(memory_embeddings) < 5:
        return 1.0
    emb = _embed_one(content)
    n = len(memory_embeddings)
    # Simplified isolation: avg random split depth estimate
    # Use distance to random subsets as proxy for isolation
    np.random.seed(42)
    scores = []
    for _ in range(10):
        subset_idx = np.random.choice(n, min(n, 10), replace=False)
        subset = memory_embeddings[subset_idx]
        dists = 1.0 - _cosine_sims(emb, subset)
        scores.append(float(np.mean(dists)))
    avg_dist = np.mean(scores)
    return _clamp(avg_dist)


def variant_005(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Local Outlier Factor against memory embeddings."""
    if memory_embeddings is None or len(memory_embeddings) < 5:
        return 1.0
    emb = _embed_one(content)
    k = min(5, len(memory_embeddings))

    # Compute k-distance for the query point
    sims = _cosine_sims(emb, memory_embeddings)
    dists = 1.0 - sims
    sorted_dists = np.sort(dists)
    query_k_dist = float(sorted_dists[k-1])

    # Compute reachability density for query
    query_reach = float(np.mean(np.maximum(sorted_dists[:k], query_k_dist)))
    if query_reach < 1e-10:
        return 0.0
    query_lrd = 1.0 / query_reach

    # Average LRD of neighbors
    neighbor_lrds = []
    for i in range(k):
        ni = np.argsort(dists)[i]
        ni_sims = _cosine_sims(memory_embeddings[ni], memory_embeddings)
        ni_dists = np.sort(1.0 - ni_sims)
        ni_k_dist = float(ni_dists[min(k, len(ni_dists)-1)])
        ni_reach = float(np.mean(np.maximum(ni_dists[1:k+1], ni_k_dist)))
        if ni_reach < 1e-10:
            neighbor_lrds.append(query_lrd)
        else:
            neighbor_lrds.append(1.0 / ni_reach)

    avg_neighbor_lrd = np.mean(neighbor_lrds)
    lof = avg_neighbor_lrd / (query_lrd + 1e-10)
    # LOF > 1 means outlier. Map to [0,1]: lof=1 -> 0.5, lof=2 -> 0.75
    return _clamp(1.0 - 1.0 / (lof + 1e-10))


def variant_006(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """k-distance: distance to 3rd nearest memory."""
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 1.0
    emb = _embed_one(content)
    sims = _cosine_sims(emb, memory_embeddings)
    sorted_sims = np.sort(sims)[::-1]
    k = min(3, len(sorted_sims))
    return _clamp(1.0 - float(sorted_sims[k-1]))


def variant_007(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Average distance to top-5 nearest memories."""
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 1.0
    emb = _embed_one(content)
    sims = _cosine_sims(emb, memory_embeddings)
    top_k = min(5, len(sims))
    top_sims = np.sort(sims)[::-1][:top_k]
    return _clamp(1.0 - float(np.mean(top_sims)))


def variant_008(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Pattern completion failure: distance from centroid of top-3 nearest (CA1)."""
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 1.0
    emb = _embed_one(content)
    sims = _cosine_sims(emb, memory_embeddings)
    k = min(3, len(sims))
    top_idx = np.argsort(sims)[::-1][:k]
    centroid = memory_embeddings[top_idx].mean(axis=0)
    return _clamp(1.0 - _cosine_sim(emb, centroid))


def variant_009(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Running EMA centroid distance (alpha=0.1)."""
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 1.0
    emb = _embed_one(content)
    alpha = 0.1
    ema = memory_embeddings[0].copy()
    for i in range(1, len(memory_embeddings)):
        ema = alpha * memory_embeddings[i] + (1 - alpha) * ema
    return _clamp(1.0 - _cosine_sim(emb, ema))


def variant_010(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """PCA residual: project onto top-10 PCs of memory, measure residual."""
    if memory_embeddings is None or len(memory_embeddings) < 3:
        return 1.0
    emb = _embed_one(content)
    mean = memory_embeddings.mean(axis=0)
    centered = memory_embeddings - mean
    n_components = min(10, len(memory_embeddings) - 1, memory_embeddings.shape[1])
    if n_components < 1:
        return 1.0
    # SVD for PCA
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_components]
    # Project query
    q_centered = emb - mean
    projection = components.T @ (components @ q_centered)
    residual = q_centered - projection
    residual_norm = float(np.linalg.norm(residual))
    q_norm = float(np.linalg.norm(q_centered))
    if q_norm < 1e-10:
        return 0.0
    return _clamp(residual_norm / q_norm)


def variant_011(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Autoencoder reconstruction error (linear, dim=32)."""
    if memory_embeddings is None or len(memory_embeddings) < 5:
        return 1.0
    emb = _embed_one(content)
    mean = memory_embeddings.mean(axis=0)
    centered = memory_embeddings - mean
    n_comp = min(32, len(memory_embeddings) - 1, memory_embeddings.shape[1])
    if n_comp < 1:
        return 1.0
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_comp]
    q_centered = emb - mean
    reconstructed = components.T @ (components @ q_centered)
    error = float(np.linalg.norm(q_centered - reconstructed))
    # Normalize by average reconstruction error of training data
    train_errors = []
    for me in memory_embeddings[:min(20, len(memory_embeddings))]:
        mc = me - mean
        mr = components.T @ (components @ mc)
        train_errors.append(float(np.linalg.norm(mc - mr)))
    avg_error = np.mean(train_errors) if train_errors else 1.0
    return _clamp(error / (2 * avg_error + 1e-10))


def variant_012(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Distance from nearest k-means cluster centroid (k=5)."""
    if memory_embeddings is None or len(memory_embeddings) < 5:
        return 1.0
    emb = _embed_one(content)
    k = min(5, len(memory_embeddings))
    # Simple k-means: pick k random seeds, assign, recompute centroids x3
    np.random.seed(42)
    idx = np.random.choice(len(memory_embeddings), k, replace=False)
    centroids = memory_embeddings[idx].copy()
    for _ in range(3):
        sims_all = centroids @ memory_embeddings.T
        assignments = np.argmax(sims_all, axis=0)
        for c in range(k):
            mask = assignments == c
            if mask.any():
                centroids[c] = memory_embeddings[mask].mean(axis=0)
    sims = _cosine_sims(emb, centroids)
    return _clamp(1.0 - float(np.max(sims)))


def variant_013(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Minimum L2 distance to any memory."""
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 1.0
    emb = _embed_one(content)
    diffs = memory_embeddings - emb
    l2_dists = np.linalg.norm(diffs, axis=1)
    min_dist = float(np.min(l2_dists))
    # L2 distance range for unit-ish vectors: 0 to ~2. Map to [0,1].
    return _clamp(min_dist / 2.0)


def variant_014(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Angular distance to nearest memory."""
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 1.0
    emb = _embed_one(content)
    sims = _cosine_sims(emb, memory_embeddings)
    max_sim = float(np.max(np.clip(sims, -1.0, 1.0)))
    angle = math.acos(max_sim) / math.pi  # Normalize to [0, 1]
    return _clamp(angle)


def variant_015(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Correlation distance to nearest memory (1 - Pearson r)."""
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 1.0
    emb = _embed_one(content)
    emb_centered = emb - emb.mean()
    best_corr = -1.0
    for me in memory_embeddings:
        me_centered = me - me.mean()
        denom = (np.linalg.norm(emb_centered) * np.linalg.norm(me_centered))
        if denom < 1e-10:
            continue
        corr = float(np.dot(emb_centered, me_centered) / denom)
        best_corr = max(best_corr, corr)
    return _clamp(1.0 - best_corr)


def variant_016(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Cosine distance weighted by memory age (older memories count less)."""
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 1.0
    emb = _embed_one(content)
    sims = _cosine_sims(emb, memory_embeddings)
    n = len(sims)
    # Exponential decay: older = smaller index = lower weight
    weights = np.array([math.exp(-0.05 * (n - 1 - i)) for i in range(n)])
    weighted_max_sim = float(np.max(sims * weights))
    return _clamp(1.0 - weighted_max_sim)


def variant_017(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None, **kwargs) -> float:
    """Cosine distance to nearest memory from SAME speaker (uses speaker info in content prefix)."""
    # Without speaker metadata, fall back to baseline cosine
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 1.0
    emb = _embed_one(content)
    sims = _cosine_sims(emb, memory_embeddings)
    return _clamp(1.0 - float(np.max(sims)))


def variant_018(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None, **kwargs) -> float:
    """Cosine distance to nearest memory from DIFFERENT speaker."""
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 1.0
    emb = _embed_one(content)
    sims = _cosine_sims(emb, memory_embeddings)
    return _clamp(1.0 - float(np.max(sims)))


def variant_019(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Max similarity to any memory (inverse)."""
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 1.0
    emb = _embed_one(content)
    sims = _cosine_sims(emb, memory_embeddings)
    max_sim = float(np.max(sims))
    # Sigmoid transform centered at 0.3 for sharper distinction
    return _clamp(1.0 / (1.0 + math.exp(10 * (max_sim - 0.5))))


def variant_020(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Harmonic mean distance to top-5 nearest memories."""
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 1.0
    emb = _embed_one(content)
    sims = _cosine_sims(emb, memory_embeddings)
    top_k = min(5, len(sims))
    top_sims = np.sort(sims)[::-1][:top_k]
    dists = 1.0 - top_sims
    dists = np.maximum(dists, 1e-10)
    hmean = float(top_k / np.sum(1.0 / dists))
    return _clamp(hmean)


# ============================================================================
# CATEGORY 2: Information-Theoretic with Memory (021-040)
# ============================================================================


def variant_021(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """NCD to nearest memory text."""
    if not memory_contents:
        return 1.0
    nearest = _nearest_text(content, memory_contents)
    return _clamp(_ncd(content, nearest))


def variant_022(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Conditional compression: new info added to memory corpus."""
    if not memory_contents:
        return 1.0
    # Use last 20 memories for speed
    corpus = " ".join(memory_contents[-20:])
    c_corpus = _gz_len(corpus)
    c_combined = _gz_len(corpus + " " + content)
    c_msg = _gz_len(content)
    if c_msg < 1:
        return 0.0
    ratio = (c_combined - c_corpus) / c_msg
    return _clamp(ratio)


def variant_023(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """NCD to centroid text (top-5 similar memories concatenated)."""
    if not memory_contents:
        return 1.0
    if memory_embeddings is not None and len(memory_embeddings) > 0:
        emb = _embed_one(content)
        sims = _cosine_sims(emb, memory_embeddings)
        top_k = min(5, len(sims))
        top_idx = np.argsort(sims)[::-1][:top_k]
        centroid_text = " ".join(memory_contents[i] for i in top_idx)
    else:
        centroid_text = " ".join(memory_contents[-5:])
    return _clamp(_ncd(content, centroid_text))


def variant_024(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Shannon surprise via char-trigram model trained on memory."""
    if not memory_contents:
        return 1.0
    corpus = " ".join(memory_contents[-30:]).lower()
    if len(corpus) < 10:
        return 1.0
    # Build char-trigram frequency model
    trigram_counts: Counter = Counter()
    bigram_counts: Counter = Counter()
    for i in range(len(corpus) - 2):
        trigram_counts[corpus[i:i+3]] += 1
        bigram_counts[corpus[i:i+2]] += 1
    # Score message
    msg = content.lower()
    if len(msg) < 3:
        return 0.5
    total_surprise = 0.0
    n = 0
    for i in range(len(msg) - 2):
        tri = msg[i:i+3]
        bi = msg[i:i+2]
        tri_count = trigram_counts.get(tri, 0)
        bi_count = bigram_counts.get(bi, 0)
        if bi_count > 0:
            prob = (tri_count + 1) / (bi_count + 256)
        else:
            prob = 1.0 / 256
        total_surprise += -math.log2(max(prob, 1e-10))
        n += 1
    avg_surprise = total_surprise / max(n, 1)
    # Typical surprise: 4-8 bits. Map to [0,1].
    return _clamp((avg_surprise - 2) / 8)


def variant_025(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """PPM compression cost against memory model."""
    if not memory_contents:
        return 1.0
    corpus = " ".join(memory_contents[-30:])
    c_corpus = _gz_len(corpus)
    c_combined = _gz_len(corpus + " " + content)
    cost = c_combined - c_corpus
    c_msg = _gz_len(content)
    if c_msg < 1:
        return 0.0
    # Ratio > 1 means message adds new info that can't be compressed away
    return _clamp(cost / (c_msg + 10))


def variant_026(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """KL divergence of char-trigram distributions."""
    if not memory_contents:
        return 1.0
    corpus = " ".join(memory_contents[-30:]).lower()
    msg = content.lower()
    if len(msg) < 3 or len(corpus) < 3:
        return 0.5

    def trigram_dist(text: str) -> Counter:
        c: Counter = Counter()
        for i in range(len(text) - 2):
            c[text[i:i+3]] += 1
        return c

    p = trigram_dist(msg)
    q = trigram_dist(corpus)
    all_keys = set(p.keys()) | set(q.keys())
    p_total = sum(p.values()) + len(all_keys)
    q_total = sum(q.values()) + len(all_keys)
    kl = 0.0
    for k in all_keys:
        pk = (p.get(k, 0) + 1) / p_total
        qk = (q.get(k, 0) + 1) / q_total
        kl += pk * math.log2(pk / qk)
    return _clamp(kl / 5.0)


def variant_027(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Cross-entropy of message under memory char-level model."""
    if not memory_contents:
        return 1.0
    corpus = " ".join(memory_contents[-30:]).lower()
    msg = content.lower()
    if len(msg) < 2 or len(corpus) < 2:
        return 0.5
    char_counts: Counter = Counter(corpus)
    total = len(corpus) + 256
    ce = 0.0
    for c in msg:
        prob = (char_counts.get(c, 0) + 1) / total
        ce += -math.log2(prob)
    avg_ce = ce / len(msg)
    return _clamp((avg_ce - 3) / 5)


def variant_028(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Mutual information proxy via compression."""
    if not memory_contents:
        return 1.0
    corpus = " ".join(memory_contents[-20:])
    c_corpus = _gz_len(corpus)
    c_msg = _gz_len(content)
    c_joint = _gz_len(corpus + " " + content)
    mi = c_corpus + c_msg - c_joint
    # Negative MI -> message is novel (can't be predicted from memory)
    # Positive MI -> message shares info with memory
    return _clamp(1.0 - mi / max(c_msg, 1))


def variant_029(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Vendi novelty: does adding this message increase embedding diversity?"""
    if memory_embeddings is None or len(memory_embeddings) < 2:
        return 1.0
    emb = _embed_one(content)
    # Compute diversity before and after (using eigenvalue entropy)
    def vendi_score(embeddings: np.ndarray) -> float:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        normed = embeddings / norms
        K = normed @ normed.T
        K = (K + K.T) / 2
        eigs = np.linalg.eigvalsh(K)
        eigs = np.maximum(eigs, 0)
        eigs = eigs / (eigs.sum() + 1e-10)
        eigs = eigs[eigs > 1e-10]
        return float(np.exp(-np.sum(eigs * np.log(eigs))))

    # Sample for speed
    n = len(memory_embeddings)
    if n > 30:
        idx = np.random.choice(n, 30, replace=False)
        sample = memory_embeddings[idx]
    else:
        sample = memory_embeddings

    before = vendi_score(sample)
    after = vendi_score(np.vstack([sample, emb[np.newaxis]]))
    improvement = (after - before) / max(before, 1e-10)
    return _clamp(improvement * 5 + 0.5)


def variant_030(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Compression ratio improvement when adding this message."""
    if not memory_contents:
        return 1.0
    corpus = " ".join(memory_contents[-20:])
    raw_len = len(corpus.encode())
    c_before = _gz_len(corpus)
    ratio_before = c_before / max(raw_len, 1)

    new_corpus = corpus + " " + content
    new_raw_len = len(new_corpus.encode())
    c_after = _gz_len(new_corpus)
    ratio_after = c_after / max(new_raw_len, 1)

    # If ratio worsens (goes up), message adds novel info
    improvement = ratio_after - ratio_before
    return _clamp(improvement * 10 + 0.5)


def variant_031(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Token-level surprise: fraction of tokens not in memory vocabulary."""
    if not memory_contents:
        return 1.0
    memory_vocab: set = set()
    for m in memory_contents:
        memory_vocab.update(m.lower().split())
    words = content.lower().split()
    if not words:
        return 0.0
    novel_count = sum(1 for w in words if w not in memory_vocab)
    return _clamp(novel_count / len(words))


def variant_032(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Bigram novelty: fraction of word bigrams not in memory."""
    if not memory_contents:
        return 1.0
    memory_bigrams: set = set()
    for m in memory_contents:
        memory_bigrams.update(_word_bigrams(m))
    msg_bigrams = _word_bigrams(content)
    if not msg_bigrams:
        return 0.5
    novel = sum(1 for b in msg_bigrams if b not in memory_bigrams)
    return _clamp(novel / len(msg_bigrams))


def variant_033(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Char-trigram novelty: fraction of char trigrams not in memory."""
    if not memory_contents:
        return 1.0
    memory_trigrams: set = set()
    for m in memory_contents[-30:]:
        memory_trigrams.update(_char_trigrams(m))
    msg_trigrams = _char_trigrams(content)
    if not msg_trigrams:
        return 0.5
    novel = sum(1 for t in msg_trigrams if t not in memory_trigrams)
    return _clamp(novel / len(msg_trigrams))


def variant_034(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Word-level entropy of message relative to memory word distribution."""
    if not memory_contents:
        return 1.0
    memory_words: Counter = Counter()
    for m in memory_contents:
        memory_words.update(m.lower().split())
    total = sum(memory_words.values()) + 1000
    msg_words = content.lower().split()
    if not msg_words:
        return 0.0
    entropy = 0.0
    for w in msg_words:
        prob = (memory_words.get(w, 0) + 1) / total
        entropy += -math.log2(prob)
    avg = entropy / len(msg_words)
    return _clamp((avg - 5) / 10)


def variant_035(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """TF-IDF novelty: max TF-IDF score of message tokens against memory."""
    if not memory_contents:
        return 1.0
    # Compute IDF from memory
    n_docs = len(memory_contents)
    doc_freq: Counter = Counter()
    for m in memory_contents:
        words_in_doc = set(m.lower().split())
        for w in words_in_doc:
            doc_freq[w] += 1
    msg_words = content.lower().split()
    if not msg_words:
        return 0.0
    msg_freq: Counter = Counter(msg_words)
    max_tfidf = 0.0
    for w in set(msg_words):
        if w in _STOPWORDS:
            continue
        tf = msg_freq[w] / len(msg_words)
        idf = math.log((n_docs + 1) / (doc_freq.get(w, 0) + 1))
        tfidf = tf * idf
        max_tfidf = max(max_tfidf, tfidf)
    return _clamp(max_tfidf / 3.0)


def variant_036(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """BM25 novelty: inverse BM25 score against memory."""
    if not memory_contents:
        return 1.0
    k1 = 1.5
    b = 0.75
    n_docs = len(memory_contents)
    doc_lens = [len(m.split()) for m in memory_contents]
    avgdl = sum(doc_lens) / max(n_docs, 1)

    doc_freq: Counter = Counter()
    for m in memory_contents:
        for w in set(m.lower().split()):
            doc_freq[w] += 1

    msg_words = content.lower().split()
    if not msg_words:
        return 0.5

    max_bm25 = 0.0
    for i, m in enumerate(memory_contents[-20:]):
        score = 0.0
        m_words = m.lower().split()
        m_freq: Counter = Counter(m_words)
        dl = len(m_words)
        for w in set(msg_words):
            if w in _STOPWORDS:
                continue
            df = doc_freq.get(w, 0)
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            tf = m_freq.get(w, 0)
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avgdl, 1)))
        max_bm25 = max(max_bm25, score)
    # Higher BM25 match = lower novelty
    return _clamp(1.0 - max_bm25 / 10.0)


def variant_037(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Lempel-Ziv complexity ratio vs memory."""
    if not memory_contents:
        return 1.0

    def lz_complexity(s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        dictionary: set = set()
        w = ""
        complexity = 0
        for c in s:
            wc = w + c
            if wc in dictionary:
                w = wc
            else:
                dictionary.add(wc)
                complexity += 1
                w = ""
        if w:
            complexity += 1
        return complexity

    corpus = " ".join(memory_contents[-20:])
    # Build dictionary from corpus
    memory_dict: set = set()
    w = ""
    for c in corpus.lower():
        wc = w + c
        memory_dict.add(wc)
        w = wc if len(wc) < 5 else ""

    # Score message against memory dictionary
    msg = content.lower()
    found = 0
    total = 0
    for i in range(len(msg)):
        for j in range(i + 1, min(i + 10, len(msg) + 1)):
            total += 1
            if msg[i:j] in memory_dict:
                found += 1
    if total == 0:
        return 0.5
    coverage = found / total
    return _clamp(1.0 - coverage)


def variant_038(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Byte-pair encoding cost against memory vocabulary."""
    if not memory_contents:
        return 1.0
    corpus = " ".join(memory_contents[-20:]).lower()
    msg = content.lower()
    # Build simple BPE vocabulary from corpus (top 100 merges)
    vocab: Counter = Counter()
    for i in range(len(corpus) - 1):
        vocab[corpus[i:i+2]] += 1
    top_pairs = set(p for p, _ in vocab.most_common(100))
    # Count how much of the message is covered by the vocabulary
    covered = 0
    for i in range(len(msg) - 1):
        if msg[i:i+2] in top_pairs:
            covered += 1
    total = max(len(msg) - 1, 1)
    return _clamp(1.0 - covered / total)


def variant_039(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Longest common substring coverage."""
    if not memory_contents:
        return 1.0
    msg = content.lower()
    if len(msg) < 2:
        return 0.5
    # Check coverage by memory substrings
    corpus = " ".join(memory_contents[-10:]).lower()
    max_lcs = 0
    for start in range(0, len(msg), 3):
        for length in range(min(len(msg) - start, 30), 2, -1):
            substr = msg[start:start + length]
            if substr in corpus:
                max_lcs = max(max_lcs, length)
                break
    coverage = max_lcs / max(len(msg), 1)
    return _clamp(1.0 - coverage)


def variant_040(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Information gain: topic entropy change from adding this message."""
    if not memory_contents or len(memory_contents) < 3:
        return 1.0
    # Proxy for topics: word clusters
    all_words: Counter = Counter()
    for m in memory_contents:
        all_words.update(w for w in m.lower().split() if w not in _STOPWORDS)
    total = sum(all_words.values())
    if total == 0:
        return 1.0
    # Entropy before
    probs_before = np.array([c / total for c in all_words.values()])
    probs_before = probs_before[probs_before > 0]
    h_before = -float(np.sum(probs_before * np.log2(probs_before)))

    # Add message words
    msg_words = [w for w in content.lower().split() if w not in _STOPWORDS]
    for w in msg_words:
        all_words[w] += 1
    total2 = sum(all_words.values())
    probs_after = np.array([c / total2 for c in all_words.values()])
    probs_after = probs_after[probs_after > 0]
    h_after = -float(np.sum(probs_after * np.log2(probs_after)))

    gain = h_after - h_before
    return _clamp(gain * 5 + 0.5)


# ============================================================================
# CATEGORY 3: Entity & Fact Novelty (041-060)
# ============================================================================


def _all_memory_proper_nouns(memory_contents: list[str]) -> set[str]:
    """Extract all proper nouns from stored memories (lowercased)."""
    result: set = set()
    for m in memory_contents:
        for pn in _extract_proper_nouns(m):
            result.add(pn.lower())
    return result


def _all_memory_numbers(memory_contents: list[str]) -> set[str]:
    result: set = set()
    for m in memory_contents:
        result.update(_extract_numbers(m))
    return result


def _all_memory_temporal(memory_contents: list[str]) -> set[str]:
    result: set = set()
    for m in memory_contents:
        result.update(t.lower() for t in _extract_temporal(m))
    return result


def _all_memory_words(memory_contents: list[str]) -> set[str]:
    result: set = set()
    for m in memory_contents:
        result.update(m.lower().split())
    return result


def variant_041(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """New named entity count."""
    if not memory_contents:
        return 1.0
    known = _all_memory_proper_nouns(memory_contents)
    msg_proper = _extract_proper_nouns(content)
    new_count = sum(1 for p in msg_proper if p.lower() not in known)
    return _clamp(new_count / 3.0)


def variant_042(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """New named entity ratio."""
    if not memory_contents:
        return 1.0
    known = _all_memory_proper_nouns(memory_contents)
    msg_proper = _extract_proper_nouns(content)
    words = content.split()
    if not words:
        return 0.0
    new_count = sum(1 for p in msg_proper if p.lower() not in known)
    return _clamp(new_count / max(len(words), 1))


def variant_043(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """First mention of a person name."""
    if not memory_contents:
        return 1.0
    known = _all_memory_proper_nouns(memory_contents)
    msg_proper = _extract_proper_nouns(content)
    # Filter to likely person names (single capitalized words, not after sentence start)
    words = content.split()
    person_names = []
    for i, w in enumerate(words):
        cleaned = re.sub(r'[^\w]', '', w)
        if cleaned and cleaned[0].isupper() and cleaned.lower() not in _STOPWORDS and len(cleaned) > 1:
            # Not first word (could be sentence start) or word after period
            if i > 0 and not words[i-1].endswith(('.', '!', '?')):
                person_names.append(cleaned.lower())
    new_people = sum(1 for p in person_names if p not in known)
    return _clamp(min(1.0, new_people * 0.5))


def variant_044(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """First mention of a place."""
    if not memory_contents:
        return 1.0
    known_words = _all_memory_words(memory_contents)
    lower = content.lower()
    # Simple place detection: capitalized word followed by place indicator, or known cities
    proper = _extract_proper_nouns(content)
    words = lower.split()
    place_score = 0.0
    for i, w in enumerate(words):
        if w in _PLACE_INDICATORS and i > 0:
            prev = words[i-1]
            if prev not in known_words:
                place_score += 0.3
    # Check for proper nouns that might be places
    for p in proper:
        if p.lower() not in _all_memory_proper_nouns(memory_contents):
            place_score += 0.1
    return _clamp(place_score)


def variant_045(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """New number/amount not seen before."""
    if not memory_contents:
        return 1.0
    known_nums = _all_memory_numbers(memory_contents)
    msg_nums = _extract_numbers(content)
    new_count = sum(1 for n in msg_nums if n not in known_nums)
    return _clamp(new_count / 2.0)


def variant_046(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """New temporal reference not seen in memory."""
    if not memory_contents:
        return 1.0
    known_temp = _all_memory_temporal(memory_contents)
    msg_temp = _extract_temporal(content)
    new_count = sum(1 for t in msg_temp if t.lower() not in known_temp)
    return _clamp(new_count / 2.0)


def variant_047(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Entity state-change: known entity + new verb context."""
    if not memory_contents:
        return 1.0
    known_proper = _all_memory_proper_nouns(memory_contents)
    msg_proper = [p.lower() for p in _extract_proper_nouns(content)]

    # Find known entities in this message
    known_in_msg = [p for p in msg_proper if p in known_proper]
    if not known_in_msg:
        return 0.3

    # Check if the context around this entity is new
    lower = content.lower()
    known_entity_contexts: set = set()
    for m in memory_contents:
        ml = m.lower()
        for entity in known_in_msg:
            if entity in ml:
                # Extract surrounding words
                idx = ml.find(entity)
                context = ml[max(0, idx-20):idx+len(entity)+20]
                known_entity_contexts.update(context.split())

    # Check for new context words
    msg_context_words = set(lower.split()) - _STOPWORDS
    new_context = msg_context_words - known_entity_contexts - set(known_in_msg)
    return _clamp(len(new_context) / max(len(msg_context_words), 1))


def variant_048(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Fact extraction then novelty: SVO patterns checked against memory."""
    if not memory_contents:
        return 1.0
    # Simple SVO extraction via regex
    svo_patterns = [
        r'(\w+)\s+(is|are|was|were|has|have|had)\s+(\w+)',
        r'(\w+)\s+(got|made|started|quit|moved|joined|left|built|found)\s+(\w+)',
        r'i\s+(am|was|have|had|got|made|started|quit|moved)\s+(\w+)',
    ]
    msg_facts: set = set()
    for p in svo_patterns:
        for match in re.finditer(p, content.lower()):
            msg_facts.add(match.group())

    if not msg_facts:
        return 0.3

    # Check against memory
    memory_text = " ".join(memory_contents[-30:]).lower()
    new_facts = sum(1 for f in msg_facts if f not in memory_text)
    return _clamp(new_facts / max(len(msg_facts), 1))


def variant_049(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Proper noun + verb novelty."""
    if not memory_contents:
        return 1.0
    msg_proper = _extract_proper_nouns(content)
    if not msg_proper:
        return 0.2

    # Extract entity-verb pairs
    words = content.lower().split()
    pairs: set = set()
    for pn in msg_proper:
        pn_lower = pn.lower()
        for i, w in enumerate(words):
            if w == pn_lower:
                # Look at surrounding verbs
                for j in range(max(0, i-3), min(len(words), i+4)):
                    if words[j] in _EVENT_VERBS:
                        pairs.add(f"{pn_lower}_{words[j]}")

    if not pairs:
        return 0.2

    # Check against memory
    memory_pairs: set = set()
    for m in memory_contents:
        m_proper = _extract_proper_nouns(m)
        m_words = m.lower().split()
        for pn in m_proper:
            pn_lower = pn.lower()
            for i, w in enumerate(m_words):
                if w == pn_lower:
                    for j in range(max(0, i-3), min(len(m_words), i+4)):
                        if m_words[j] in _EVENT_VERBS:
                            memory_pairs.add(f"{pn_lower}_{m_words[j]}")

    new_pairs = pairs - memory_pairs
    return _clamp(len(new_pairs) / max(len(pairs), 1))


def variant_050(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Relationship novelty: two known entities mentioned together for first time."""
    if not memory_contents:
        return 1.0
    msg_proper = [p.lower() for p in _extract_proper_nouns(content)]
    if len(msg_proper) < 2:
        return 0.1

    # Build set of entity pairs from memory
    memory_pairs: set = set()
    for m in memory_contents:
        m_proper = [p.lower() for p in _extract_proper_nouns(m)]
        for i in range(len(m_proper)):
            for j in range(i+1, len(m_proper)):
                pair = tuple(sorted([m_proper[i], m_proper[j]]))
                memory_pairs.add(pair)

    # Check for new pairs
    new_pairs = 0
    for i in range(len(msg_proper)):
        for j in range(i+1, len(msg_proper)):
            pair = tuple(sorted([msg_proper[i], msg_proper[j]]))
            if pair not in memory_pairs:
                new_pairs += 1

    return _clamp(new_pairs / 2.0)


def variant_051(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """New domain/topic: words from a cluster not seen in memory."""
    if not memory_contents:
        return 1.0
    memory_vocab = _all_memory_words(memory_contents)
    msg_words = [w for w in content.lower().split() if w not in _STOPWORDS and len(w) > 3]
    if not msg_words:
        return 0.1
    new_words = [w for w in msg_words if w not in memory_vocab]
    if not new_words:
        return 0.1
    # Check if new words cluster together (same domain)
    consecutive_new = 0
    max_consecutive = 0
    for w in msg_words:
        if w in new_words:
            consecutive_new += 1
            max_consecutive = max(max_consecutive, consecutive_new)
        else:
            consecutive_new = 0

    ratio = len(new_words) / len(msg_words)
    cluster_bonus = min(0.3, max_consecutive * 0.1)
    return _clamp(ratio + cluster_bonus)


def variant_052(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Novel entity attribute: entity + "is/was/has" + new attribute."""
    if not memory_contents:
        return 1.0
    # Extract "entity is/was/has X" patterns
    attr_pattern = re.compile(r'(\b[A-Z]\w+)\s+(?:is|was|has|have)\s+([\w\s]+?)(?:\.|,|!|$)', re.IGNORECASE)
    msg_attrs = {}
    for m in attr_pattern.finditer(content):
        entity = m.group(1).lower()
        attr = m.group(2).strip().lower()
        if entity not in _STOPWORDS:
            msg_attrs[entity] = attr

    if not msg_attrs:
        return 0.2

    # Check against memory
    memory_attrs: dict = {}
    for mem in memory_contents:
        for m in attr_pattern.finditer(mem):
            entity = m.group(1).lower()
            attr = m.group(2).strip().lower()
            if entity not in _STOPWORDS:
                if entity not in memory_attrs:
                    memory_attrs[entity] = set()
                memory_attrs[entity].add(attr)

    novel = 0
    for entity, attr in msg_attrs.items():
        if entity not in memory_attrs or attr not in memory_attrs[entity]:
            novel += 1
    return _clamp(novel / max(len(msg_attrs), 1))


def variant_053(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Entity count change: more entities than typical for this conversation."""
    if not memory_contents:
        return 1.0
    msg_entities = len(_extract_proper_nouns(content))
    memory_entity_counts = [len(_extract_proper_nouns(m)) for m in memory_contents[-20:]]
    if not memory_entity_counts:
        return 0.5
    avg = np.mean(memory_entity_counts)
    std = max(np.std(memory_entity_counts), 0.5)
    z = (msg_entities - avg) / std
    return _clamp(0.5 + z * 0.2)


def variant_054(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Event trigger detection: action verbs not seen with this entity."""
    if not memory_contents:
        return 1.0
    msg_proper = _extract_proper_nouns(content)
    lower = content.lower()
    words = lower.split()

    # Find event verbs in message
    msg_events = [w for w in words if w in _EVENT_VERBS]
    if not msg_events:
        return 0.2

    # Check if these events are associated with known entities
    memory_text = " ".join(memory_contents).lower()
    score = 0.0
    for verb in msg_events:
        for pn in msg_proper:
            combined = f"{pn.lower()} {verb}"
            if combined not in memory_text:
                score += 0.3
    return _clamp(score)


def variant_055(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Quote/speech novelty: reported speech from a person."""
    if not memory_contents:
        return 1.0
    # Detect reported speech
    quote_patterns = [
        r'"([^"]+)"',
        r"'([^']+)'",
        r'(?:said|told|asked|mentioned|replied)\s+"?([^"]+)"?',
    ]
    msg_quotes = []
    for p in quote_patterns:
        msg_quotes.extend(re.findall(p, content))

    if not msg_quotes:
        return 0.2

    memory_text = " ".join(memory_contents[-20:]).lower()
    new_quotes = sum(1 for q in msg_quotes if q.lower() not in memory_text)
    return _clamp(new_quotes / max(len(msg_quotes), 1))


def variant_056(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Commitment with new entity."""
    if not memory_contents:
        return 1.0
    lower = content.lower()
    has_commitment = bool(_COMMITMENT_RE.search(lower))
    if not has_commitment:
        return 0.1

    known = _all_memory_proper_nouns(memory_contents)
    msg_proper = _extract_proper_nouns(content)
    new_entities = [p for p in msg_proper if p.lower() not in known]

    if new_entities:
        return 0.9
    return 0.6


def variant_057(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Numeric fact novelty: $X or X% where the number is new."""
    if not memory_contents:
        return 1.0
    money_pattern = re.compile(r'\$[\d,.]+|\d+%')
    msg_nums = set(money_pattern.findall(content))
    if not msg_nums:
        return 0.1
    known_nums: set = set()
    for m in memory_contents:
        known_nums.update(money_pattern.findall(m))
    new_count = len(msg_nums - known_nums)
    return _clamp(new_count / 2.0)


def variant_058(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Address/location novelty."""
    if not memory_contents:
        return 1.0
    # Detect address-like patterns
    addr_pattern = re.compile(r'\d+\s+\w+\s+(?:street|avenue|ave|blvd|road|rd|lane|drive|dr|court|ct|way)', re.IGNORECASE)
    city_pattern = re.compile(r'\b(?:Portland|Seattle|Denver|Austin|Chicago|NYC|LA|Boston|Miami|Atlanta|SF)\b', re.IGNORECASE)

    msg_locations = set(addr_pattern.findall(content.lower())) | set(city_pattern.findall(content.lower()))
    if not msg_locations:
        return 0.1

    known_locations: set = set()
    for m in memory_contents:
        known_locations.update(addr_pattern.findall(m.lower()))
        known_locations.update(city_pattern.findall(m.lower()))

    new_count = len(msg_locations - known_locations)
    return _clamp(new_count * 0.5)


def variant_059(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Organization novelty: company/school/hospital names."""
    if not memory_contents:
        return 1.0
    org_indicators = {'inc', 'corp', 'llc', 'university', 'college', 'school',
                      'hospital', 'clinic', 'institute', 'foundation', 'labs',
                      'technologies', 'studios', 'media', 'group', 'partners'}

    def extract_orgs(text: str) -> set[str]:
        orgs: set = set()
        words = text.split()
        for i, w in enumerate(words):
            cleaned = re.sub(r'[^\w]', '', w).lower()
            if cleaned in org_indicators and i > 0:
                prev = re.sub(r'[^\w]', '', words[i-1]).lower()
                if prev not in _STOPWORDS:
                    orgs.add(f"{prev} {cleaned}")
        # Also check for known patterns like "UCLA", "MIT"
        for m in re.finditer(r'\b[A-Z]{2,5}\b', text):
            orgs.add(m.group().lower())
        return orgs

    msg_orgs = extract_orgs(content)
    if not msg_orgs:
        return 0.1
    known_orgs: set = set()
    for m in memory_contents:
        known_orgs.update(extract_orgs(m))
    new_count = len(msg_orgs - known_orgs)
    return _clamp(new_count * 0.4)


def variant_060(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """New verb-entity pair: entity + verb combination not seen before."""
    if not memory_contents:
        return 1.0

    def extract_verb_entity_pairs(text: str) -> set[str]:
        pairs: set = set()
        proper = _extract_proper_nouns(text)
        words = text.lower().split()
        verbs = re.findall(r'\b(?:is|are|was|were|has|have|had|got|made|started|quit|moved|joined|left|built|found|said|told|asked|accepted|submitted|enrolled|graduated)\b', text.lower())
        for pn in proper:
            pn_lower = pn.lower()
            for v in set(verbs):
                pairs.add(f"{pn_lower}+{v}")
        return pairs

    msg_pairs = extract_verb_entity_pairs(content)
    if not msg_pairs:
        return 0.2
    memory_pairs: set = set()
    for m in memory_contents:
        memory_pairs.update(extract_verb_entity_pairs(m))
    new_count = len(msg_pairs - memory_pairs)
    return _clamp(new_count / max(len(msg_pairs), 1))


# ============================================================================
# CATEGORY 4: Cognitive/Biological Inspired (061-080)
# ============================================================================


def variant_061(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Orienting response: content_quality AND semantic_distance."""
    content_score = _speech_act_score(content)
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return content_score
    emb = _embed_one(content)
    sims = _cosine_sims(emb, memory_embeddings)
    distance = 1.0 - float(np.max(sims))
    return _clamp(content_score * distance)


def variant_062(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Orienting response with soft-OR: AND-gate PLUS bypass for commitments."""
    content_score = _speech_act_score(content)
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return content_score
    emb = _embed_one(content)
    sims = _cosine_sims(emb, memory_embeddings)
    distance = 1.0 - float(np.max(sims))
    and_score = content_score * distance

    # Bypass for commitment patterns
    lower = content.lower()
    if _COMMITMENT_RE.search(lower):
        return _clamp(max(and_score, 0.7))

    # Bypass for new proper nouns
    if memory_contents:
        known = _all_memory_proper_nouns(memory_contents)
        msg_proper = _extract_proper_nouns(content)
        new_entities = [p for p in msg_proper if p.lower() not in known]
        if new_entities:
            return _clamp(max(and_score, 0.5))

    return _clamp(and_score)


def variant_063(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Habituation curve: exp(-k * times similar message seen)."""
    if not memory_contents:
        return 1.0
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 1.0
    emb = _embed_one(content)
    sims = _cosine_sims(emb, memory_embeddings)
    # Count how many memories are "similar" (sim > 0.5)
    similar_count = int(np.sum(sims > 0.5))
    return _clamp(math.exp(-0.5 * similar_count))


def variant_064(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Von Restorff isolation: distinctiveness within last 10 memories."""
    if memory_embeddings is None or len(memory_embeddings) < 2:
        return 1.0
    emb = _embed_one(content)
    window = memory_embeddings[-10:]
    # Average pairwise similarity within window
    if len(window) < 2:
        return 0.5
    window_sims = []
    for i in range(len(window)):
        for j in range(i+1, len(window)):
            window_sims.append(_cosine_sim(window[i], window[j]))
    avg_window_sim = np.mean(window_sims) if window_sims else 0.5

    # Similarity of message to window centroid
    centroid = window.mean(axis=0)
    msg_sim = _cosine_sim(emb, centroid)

    # More distinct from window centroid than window elements are from each other
    isolation = (1.0 - msg_sim) - (1.0 - avg_window_sim)
    return _clamp(0.5 + isolation)


def variant_065(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Synaptic tagging: weak tag from distance, reinforced by salience."""
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 1.0
    emb = _embed_one(content)
    sims = _cosine_sims(emb, memory_embeddings)
    distance = 1.0 - float(np.max(sims))

    # Weak tag based on distance
    weak_tag = distance

    # Reinforcement from salience signals
    salience = _factual_density(content)

    # Tagging: if salience is above threshold within "time window" (same batch),
    # reinforce the weak tag
    if salience > 0.3:
        return _clamp(weak_tag * (1.0 + salience))
    else:
        return _clamp(weak_tag * 0.5)


def variant_066(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Levels of processing: depth markers (causal, temporal, entity refs)."""
    lower = content.lower()
    depth = 0.0

    # Causal markers
    causal = ['because', 'since', 'therefore', 'so that', 'due to', 'as a result', 'caused by']
    depth += 0.2 * sum(1 for c in causal if c in lower)

    # Temporal markers
    temporal = ['after', 'before', 'when', 'while', 'during', 'then', 'finally', 'eventually']
    depth += 0.15 * sum(1 for t in temporal if f" {t} " in f" {lower} ")

    # Entity references
    proper = _extract_proper_nouns(content)
    depth += 0.1 * len(proper)

    # Numbers (specific facts)
    nums = _extract_numbers(content)
    depth += 0.1 * len(nums)

    depth_score = min(1.0, depth)

    # Combine with memory distance
    if memory_embeddings is not None and len(memory_embeddings) > 0:
        emb = _embed_one(content)
        sims = _cosine_sims(emb, memory_embeddings)
        distance = 1.0 - float(np.max(sims))
        return _clamp(depth_score * 0.6 + distance * 0.4)
    return _clamp(depth_score)


def variant_067(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Prediction error: expected message type vs actual."""
    if not memory_contents or len(memory_contents) < 3:
        return 0.5

    # Build simple type model from recent messages
    def msg_type(text: str) -> str:
        lower = text.lower().strip()
        if lower in _NOISE_EXACT:
            return "noise"
        if text.strip().endswith("?"):
            return "question"
        if _COMMITMENT_RE.search(lower):
            return "commitment"
        if len(text.split()) < 3:
            return "short"
        return "statement"

    recent_types = [msg_type(m) for m in memory_contents[-5:]]
    current_type = msg_type(content)

    # Count type frequencies
    type_freq: Counter = Counter(recent_types)
    total = len(recent_types)
    expected_prob = type_freq.get(current_type, 0) / max(total, 1)

    # Surprise = 1 - expected_prob
    return _clamp(1.0 - expected_prob)


def variant_068(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Bayesian surprise: topic distribution shift."""
    if not memory_contents or len(memory_contents) < 3:
        return 0.5

    # Topic proxy: content word distribution
    prior: Counter = Counter()
    for m in memory_contents[-20:]:
        for w in m.lower().split():
            if w not in _STOPWORDS and len(w) > 2:
                prior[w] += 1

    total_prior = sum(prior.values()) + 100
    msg_words = [w for w in content.lower().split() if w not in _STOPWORDS and len(w) > 2]
    if not msg_words:
        return 0.2

    # KL divergence from uniform over message words
    kl = 0.0
    for w in msg_words:
        p_post = 1.0 / len(msg_words)
        p_prior = (prior.get(w, 0) + 1) / total_prior
        kl += p_post * math.log2(p_post / p_prior + 1e-10)

    return _clamp(kl / 5.0)


def variant_069(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Attention capture: combination triggering involuntary attention."""
    lower = content.lower()
    score = 0.0

    # Shouting (ALL CAPS words)
    caps_words = [w for w in content.split() if w.isupper() and len(w) > 1]
    score += min(0.3, len(caps_words) * 0.15)

    # Exclamation
    score += min(0.2, content.count("!") * 0.1)

    # Commitment patterns
    if _COMMITMENT_RE.search(lower):
        score += 0.3

    # New proper nouns (if memory available)
    if memory_contents:
        known = _all_memory_proper_nouns(memory_contents)
        msg_proper = _extract_proper_nouns(content)
        new_entities = [p for p in msg_proper if p.lower() not in known]
        score += min(0.3, len(new_entities) * 0.15)

    return _clamp(score)


def variant_070(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Memory interference: does this message CONFLICT with stored memories?"""
    if not memory_contents:
        return 0.3

    lower = content.lower()
    contradiction_markers = ['no longer', 'not anymore', 'stopped', 'quit', 'actually',
                            'correction', 'wrong', 'changed', 'switched', 'used to',
                            'not true', 'turns out', 'apparently']

    has_contradiction = any(m in lower for m in contradiction_markers)
    if not has_contradiction:
        return 0.2

    # Check if contradicted topic exists in memory
    if memory_embeddings is not None and len(memory_embeddings) > 0:
        emb = _embed_one(content)
        sims = _cosine_sims(emb, memory_embeddings)
        max_sim = float(np.max(sims))
        # High similarity + contradiction markers = update signal
        if max_sim > 0.3:
            return 0.9

    return 0.6


def variant_071(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Spacing effect: novelty from topic gap length."""
    if memory_embeddings is None or len(memory_embeddings) < 2:
        return 0.5
    emb = _embed_one(content)
    sims = _cosine_sims(emb, memory_embeddings)

    # Find index of most similar memory
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    if best_sim < 0.3:
        return 1.0  # Topic never seen

    # Gap = distance from end (number of memories since last similar)
    gap = len(memory_embeddings) - 1 - best_idx
    # Longer gap = more novelty (topic re-emerging after silence)
    return _clamp(1.0 - math.exp(-0.1 * gap))


def variant_072(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Generation effect: information production vs acknowledgment."""
    lower = content.lower().strip()

    # Acknowledgment patterns
    if lower in _NOISE_EXACT:
        return 0.05

    # Question (seeking, not producing)
    if content.strip().endswith("?"):
        return 0.2

    # Generation markers
    generation_score = 0.0
    if _COMMITMENT_RE.search(lower):
        generation_score += 0.4

    proper = _extract_proper_nouns(content)
    generation_score += min(0.2, len(proper) * 0.05)

    nums = _extract_numbers(content)
    generation_score += min(0.2, len(nums) * 0.05)

    words = content.split()
    if len(words) >= 5:
        generation_score += 0.2

    # Combine with memory novelty
    if memory_embeddings is not None and len(memory_embeddings) > 0:
        emb = _embed_one(content)
        sims = _cosine_sims(emb, memory_embeddings)
        distance = 1.0 - float(np.max(sims))
        return _clamp(generation_score * 0.7 + distance * 0.3)

    return _clamp(generation_score)


def variant_073(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Schema violation: unexpected pattern for this conversation."""
    if not memory_contents or len(memory_contents) < 5:
        return 0.5

    # Build schema from recent memory: avg word count, common patterns
    word_counts = [len(m.split()) for m in memory_contents[-10:]]
    avg_wc = np.mean(word_counts)
    std_wc = max(np.std(word_counts), 1.0)

    msg_wc = len(content.split())
    wc_deviation = abs(msg_wc - avg_wc) / std_wc

    # Content type deviation
    noise_ratio = sum(1 for m in memory_contents[-10:] if m.lower().strip() in _NOISE_EXACT) / 10
    is_noise = content.lower().strip() in _NOISE_EXACT

    type_surprise = 0.0
    if noise_ratio > 0.5 and not is_noise:
        type_surprise = 0.3  # Substantive message in noisy conversation
    elif noise_ratio < 0.2 and is_noise:
        type_surprise = 0.1  # Noise in substantive conversation (not as interesting)

    return _clamp(wc_deviation * 0.15 + type_surprise + 0.2)


def variant_074(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Emotional arousal gated novelty: distance * arousal."""
    arousal_words = {'amazing', 'incredible', 'terrible', 'awful', 'love', 'hate',
                    'excited', 'devastated', 'thrilled', 'furious', 'ecstatic',
                    'heartbroken', 'overjoyed', 'disgusted', 'shocked', 'stunned'}
    lower = content.lower()
    arousal = sum(1 for w in arousal_words if w in lower) / 3.0
    arousal = min(1.0, arousal + 0.2)  # Base arousal

    if memory_embeddings is None or len(memory_embeddings) == 0:
        return _clamp(arousal)

    emb = _embed_one(content)
    sims = _cosine_sims(emb, memory_embeddings)
    distance = 1.0 - float(np.max(sims))

    return _clamp(distance * arousal)


def variant_075(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Dual-process: fast keyword check, slow embedding distance for borderlines."""
    lower = content.lower().strip()

    # Fast system: clear noise
    if lower in _NOISE_EXACT:
        return 0.05

    # Fast system: clear signal
    if _COMMITMENT_RE.search(lower):
        return 0.85

    # Slow system: embedding distance
    if memory_embeddings is not None and len(memory_embeddings) > 0:
        emb = _embed_one(content)
        sims = _cosine_sims(emb, memory_embeddings)
        distance = 1.0 - float(np.max(sims))
        return _clamp(distance)

    return 0.5


def variant_076(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Consolidation candidate: would this survive a sleep consolidation pass?"""
    # Factors that promote consolidation:
    # 1. Emotional significance
    # 2. Connection to existing knowledge
    # 3. Factual content
    # 4. Novelty

    lower = content.lower()
    score = 0.0

    # Factual content
    factual = _factual_density(content)
    score += factual * 0.4

    # Novelty from memory
    if memory_embeddings is not None and len(memory_embeddings) > 0:
        emb = _embed_one(content)
        sims = _cosine_sims(emb, memory_embeddings)
        distance = 1.0 - float(np.max(sims))
        max_sim = float(np.max(sims))
        # Best consolidation: moderately novel (connected but new)
        connection_novelty = 1.0 - abs(max_sim - 0.4) * 2
        score += connection_novelty * 0.3
        score += distance * 0.3
    else:
        score += 0.3

    return _clamp(score)


def variant_077(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Reconsolidation trigger: does this UPDATE a previously consolidated memory?"""
    if not memory_contents:
        return 0.3

    lower = content.lower()
    update_markers = ['no longer', 'not anymore', 'now', 'switched to', 'changed to',
                     'moved to', 'started', 'quit', 'actually', 'turns out']

    has_update = any(m in lower for m in update_markers)

    if memory_embeddings is not None and len(memory_embeddings) > 0:
        emb = _embed_one(content)
        sims = _cosine_sims(emb, memory_embeddings)
        max_sim = float(np.max(sims))

        # High similarity + update markers = reconsolidation
        if has_update and max_sim > 0.3:
            return 0.9

        # High similarity without update = repetition
        if max_sim > 0.7:
            return 0.1

        return _clamp(1.0 - max_sim)

    return 0.5 if has_update else 0.3


def variant_078(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Pattern separation: how distinctly encoded vs nearest memory."""
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 1.0
    emb = _embed_one(content)
    sims = _cosine_sims(emb, memory_embeddings)
    nearest_idx = int(np.argmax(sims))
    nearest_sim = float(sims[nearest_idx])

    # Pattern separation is strongest for moderately similar inputs
    # (very different inputs are already well-separated)
    if nearest_sim < 0.2:
        return 0.8  # Already distinct
    if nearest_sim > 0.8:
        return 0.1  # Too similar, can't separate

    # For moderate similarity: check if content words differ
    msg_words = set(content.lower().split()) - _STOPWORDS
    mem_words = set(memory_contents[nearest_idx].lower().split()) - _STOPWORDS
    word_overlap = len(msg_words & mem_words) / max(len(msg_words | mem_words), 1)

    # High embedding sim but low word overlap = needs pattern separation
    separation = (1.0 - word_overlap) * (1.0 - nearest_sim)
    return _clamp(0.3 + separation * 2)


def variant_079(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Replay priority: would this be selected for memory replay?"""
    lower = content.lower().strip()

    # Factors: emotional significance, factual density, connection to goals
    priority = 0.0

    if lower in _NOISE_EXACT:
        return 0.02

    if _COMMITMENT_RE.search(lower):
        priority += 0.4

    factual = _factual_density(content)
    priority += factual * 0.3

    # Recency-weighted novelty
    if memory_embeddings is not None and len(memory_embeddings) > 0:
        emb = _embed_one(content)
        sims = _cosine_sims(emb, memory_embeddings)
        distance = 1.0 - float(np.max(sims))
        priority += distance * 0.3
    else:
        priority += 0.3

    return _clamp(priority)


def variant_080(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Contextual distinctiveness: different from surrounding messages."""
    if memory_embeddings is None or len(memory_embeddings) < 2:
        return 0.5
    emb = _embed_one(content)
    # Compare to last 5 memories and next... well, we don't have "next"
    # So compare to last 5 memories as context
    context = memory_embeddings[-5:]
    context_centroid = context.mean(axis=0)
    context_sim = _cosine_sim(emb, context_centroid)

    # Also compare to overall memory
    overall_centroid = memory_embeddings.mean(axis=0)
    overall_sim = _cosine_sim(emb, overall_centroid)

    # Distinct from local context but not from overall = topic shift
    local_distinctiveness = 1.0 - context_sim
    return _clamp(local_distinctiveness)


# ============================================================================
# CATEGORY 5: Fact-Extraction-First (081-100)
# ============================================================================


def variant_081(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Regex extract proper nouns, check if any are new."""
    if not memory_contents:
        return 1.0
    known = _all_memory_proper_nouns(memory_contents)
    msg_proper = _extract_proper_nouns(content)
    if not msg_proper:
        return 0.1
    new_count = sum(1 for p in msg_proper if p.lower() not in known)
    if new_count > 0:
        # Score based on embedding novelty of JUST the proper nouns
        if memory_embeddings is not None and len(memory_embeddings) > 0:
            proper_text = " ".join(msg_proper)
            emb = _embed_one(proper_text)
            sims = _cosine_sims(emb, memory_embeddings)
            distance = 1.0 - float(np.max(sims))
            return _clamp(max(0.5, distance))
        return 0.7
    return 0.15


def variant_082(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Regex extract numbers, check if any are new."""
    if not memory_contents:
        return 1.0
    known = _all_memory_numbers(memory_contents)
    msg_nums = _extract_numbers(content)
    if not msg_nums:
        return 0.1
    new_count = sum(1 for n in msg_nums if n not in known)
    return _clamp(new_count / 2.0 + 0.1)


def variant_083(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Regex extract 'I/we + past tense verb' patterns, check if action is new."""
    if not memory_contents:
        return 1.0
    pattern = re.compile(
        r'\b(?:i|we)\s+(?:got|did|made|found|built|started|quit|left|joined|enrolled|'
        r'accepted|submitted|finished|completed|signed|bought|sold|moved|said|told|'
        r'asked|proposed|created|launched|shipped|published|passed|graduated|earned|'
        r'won|lost|broke|fixed|applied|booked)\b',
        re.IGNORECASE,
    )
    msg_actions = set(m.group().lower() for m in pattern.finditer(content))
    if not msg_actions:
        return 0.15

    memory_actions: set = set()
    for mem in memory_contents:
        memory_actions.update(m.group().lower() for m in pattern.finditer(mem))

    new_count = len(msg_actions - memory_actions)
    return _clamp(new_count / max(len(msg_actions), 1) * 0.7 + 0.2)


def variant_084(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Regex extract 'entity is/was/has attribute' patterns, check if new."""
    if not memory_contents:
        return 1.0
    pattern = re.compile(r'(\b[A-Z]\w+)\s+(?:is|was|has|have|had)\s+([\w\s]{2,30}?)(?:\.|,|!|\?|$)')
    msg_attrs = set()
    for m in pattern.finditer(content):
        entity = m.group(1).lower()
        if entity not in _STOPWORDS:
            msg_attrs.add(f"{entity}:{m.group(2).strip().lower()[:20]}")

    if not msg_attrs:
        return 0.15

    memory_attrs: set = set()
    for mem in memory_contents:
        for m in pattern.finditer(mem):
            entity = m.group(1).lower()
            if entity not in _STOPWORDS:
                memory_attrs.add(f"{entity}:{m.group(2).strip().lower()[:20]}")

    new_count = len(msg_attrs - memory_attrs)
    return _clamp(new_count / max(len(msg_attrs), 1))


def variant_085(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Extract capitalized phrases, compute novelty of JUST those."""
    caps_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
    caps_text = " ".join(caps_phrases)
    if not caps_text or len(caps_text) < 3:
        return 0.1
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 0.7
    emb = _embed_one(caps_text)
    sims = _cosine_sims(emb, memory_embeddings)
    return _clamp(1.0 - float(np.max(sims)))


def variant_086(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Extract quoted text, compute novelty of quotes only."""
    quotes = re.findall(r'"([^"]+)"|"([^"]+)"', content)
    quote_text = " ".join(q[0] or q[1] for q in quotes)
    if not quote_text:
        return 0.2
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 0.8
    emb = _embed_one(quote_text)
    sims = _cosine_sims(emb, memory_embeddings)
    return _clamp(1.0 - float(np.max(sims)))


def variant_087(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Extract sentences with numbers, compute novelty of those."""
    sentences = re.split(r'[.!?]+', content)
    num_sentences = [s.strip() for s in sentences if re.search(r'\d', s) and s.strip()]
    if not num_sentences:
        return 0.15
    text = " ".join(num_sentences)
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 0.7
    emb = _embed_one(text)
    sims = _cosine_sims(emb, memory_embeddings)
    return _clamp(1.0 - float(np.max(sims)))


def variant_088(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Extract first-person declarative sentences, compute novelty."""
    sentences = re.split(r'[.!?]+', content)
    fp_sentences = [s.strip() for s in sentences
                    if re.match(r'\s*(?:i|we|i\'m|we\'re|i\'ve|we\'ve)\b', s.strip(), re.IGNORECASE)
                    and not s.strip().endswith("?") and s.strip()]
    if not fp_sentences:
        return 0.15
    text = " ".join(fp_sentences)
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 0.7
    emb = _embed_one(text)
    sims = _cosine_sims(emb, memory_embeddings)
    return _clamp(1.0 - float(np.max(sims)))


def variant_089(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Strip filler words, score remainder against memory."""
    words = content.split()
    filtered = [w for w in words if w.lower() not in _FILLER_WORDS]
    if not filtered:
        return 0.05
    text = " ".join(filtered)
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 0.7
    emb = _embed_one(text)
    sims = _cosine_sims(emb, memory_embeddings)
    return _clamp(1.0 - float(np.max(sims)))


def variant_090(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Longest content word sequence (no stopwords), score against memory."""
    words = content.split()
    # Find longest run of non-stopwords
    best_run = []
    current_run = []
    for w in words:
        if w.lower() not in _STOPWORDS:
            current_run.append(w)
        else:
            if len(current_run) > len(best_run):
                best_run = current_run
            current_run = []
    if len(current_run) > len(best_run):
        best_run = current_run

    if not best_run:
        return 0.1
    text = " ".join(best_run)
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 0.7
    emb = _embed_one(text)
    sims = _cosine_sims(emb, memory_embeddings)
    return _clamp(1.0 - float(np.max(sims)))


def variant_091(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Extract verb + object pairs, check if new."""
    if not memory_contents:
        return 1.0
    # Simple heuristic: word after a verb
    verbs = set(_EVENT_VERBS) | {'is', 'are', 'was', 'were', 'has', 'have', 'had',
                                  'got', 'made', 'went', 'came', 'took', 'gave'}
    words = content.lower().split()
    pairs: set = set()
    for i in range(len(words) - 1):
        if words[i] in verbs:
            obj = words[i+1]
            if obj not in _STOPWORDS:
                pairs.add(f"{words[i]}_{obj}")

    if not pairs:
        return 0.2

    memory_pairs: set = set()
    for m in memory_contents:
        m_words = m.lower().split()
        for i in range(len(m_words) - 1):
            if m_words[i] in verbs:
                obj = m_words[i+1]
                if obj not in _STOPWORDS:
                    memory_pairs.add(f"{m_words[i]}_{obj}")

    new_count = len(pairs - memory_pairs)
    return _clamp(new_count / max(len(pairs), 1))


def variant_092(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Extract temporal expressions + associated content, check novelty."""
    if not memory_contents:
        return 1.0
    temporal = _extract_temporal(content)
    if not temporal:
        return 0.15

    # Extract temporal-content pairs from message
    sentences = re.split(r'[.!?]+', content)
    temporal_sentences = [s.strip() for s in sentences
                          if any(t in s.lower() for t in temporal) and s.strip()]

    if not temporal_sentences:
        return 0.3

    text = " ".join(temporal_sentences)
    if memory_embeddings is not None and len(memory_embeddings) > 0:
        emb = _embed_one(text)
        sims = _cosine_sims(emb, memory_embeddings)
        return _clamp(1.0 - float(np.max(sims)))
    return 0.6


def variant_093(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Remove short words (<4 chars), score remainder."""
    words = content.split()
    long_words = [w for w in words if len(w) >= 4]
    if not long_words:
        return 0.05
    text = " ".join(long_words)
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 0.7
    emb = _embed_one(text)
    sims = _cosine_sims(emb, memory_embeddings)
    return _clamp(1.0 - float(np.max(sims)))


def variant_094(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Extract words with capital letters or digits, score that substring."""
    signal_words = [w for w in content.split()
                    if any(c.isupper() for c in w) or any(c.isdigit() for c in w)]
    if not signal_words:
        return 0.1
    text = " ".join(signal_words)
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 0.7
    emb = _embed_one(text)
    sims = _cosine_sims(emb, memory_embeddings)
    return _clamp(1.0 - float(np.max(sims)))


def variant_095(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Extract commitment patterns, score JUST the commitment."""
    matches = list(_COMMITMENT_RE.finditer(content.lower()))
    if not matches:
        return 0.1
    commitment_text = " ".join(m.group() for m in matches)
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 0.8
    emb = _embed_one(commitment_text)
    sims = _cosine_sims(emb, memory_embeddings)
    return _clamp(1.0 - float(np.max(sims)))


def variant_096(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Score only the second half of the message."""
    words = content.split()
    if len(words) < 4:
        # Short message: use the whole thing
        text = content
    else:
        mid = len(words) // 2
        text = " ".join(words[mid:])
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 0.7
    emb = _embed_one(text)
    sims = _cosine_sims(emb, memory_embeddings)
    return _clamp(1.0 - float(np.max(sims)))


def variant_097(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Extract sentence with highest TF-IDF score, score JUST that."""
    if not memory_contents:
        return 1.0
    sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
    if not sentences:
        return 0.3

    # Compute IDF from memory
    n_docs = len(memory_contents)
    doc_freq: Counter = Counter()
    for m in memory_contents:
        for w in set(m.lower().split()):
            doc_freq[w] += 1

    # Score each sentence
    best_score = 0.0
    best_sentence = sentences[0]
    for sent in sentences:
        words = sent.lower().split()
        if not words:
            continue
        freq: Counter = Counter(words)
        score = 0.0
        for w in set(words):
            if w in _STOPWORDS:
                continue
            tf = freq[w] / len(words)
            idf = math.log((n_docs + 1) / (doc_freq.get(w, 0) + 1))
            score += tf * idf
        if score > best_score:
            best_score = score
            best_sentence = sent

    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 0.7
    emb = _embed_one(best_sentence)
    sims = _cosine_sims(emb, memory_embeddings)
    return _clamp(1.0 - float(np.max(sims)))


def variant_098(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Extract sentences containing proper nouns, score those."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
    proper_sentences = [s for s in sentences if _extract_proper_nouns(s)]
    if not proper_sentences:
        return 0.15
    text = " ".join(proper_sentences)
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 0.7
    emb = _embed_one(text)
    sims = _cosine_sims(emb, memory_embeddings)
    return _clamp(1.0 - float(np.max(sims)))


def variant_099(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Extract 'but/however/actually' pivot sentences."""
    pivot_words = ['but', 'however', 'actually', 'though', 'although', 'yet', 'except',
                   'instead', 'turns out', 'in fact']
    sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
    pivot_sentences = [s for s in sentences
                       if any(pw in s.lower() for pw in pivot_words)]
    if not pivot_sentences:
        return 0.2
    text = " ".join(pivot_sentences)
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 0.7
    emb = _embed_one(text)
    sims = _cosine_sims(emb, memory_embeddings)
    return _clamp(1.0 - float(np.max(sims)))


def variant_100(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Combine top-3 fact extraction methods, take max novelty."""
    # Proper noun extraction (v081 logic)
    score1 = 0.1
    if memory_contents:
        known = _all_memory_proper_nouns(memory_contents)
        msg_proper = _extract_proper_nouns(content)
        new_count = sum(1 for p in msg_proper if p.lower() not in known)
        if new_count > 0:
            score1 = min(1.0, 0.4 + new_count * 0.2)

    # Number extraction (v082 logic)
    score2 = 0.1
    if memory_contents:
        known_nums = _all_memory_numbers(memory_contents)
        msg_nums = _extract_numbers(content)
        new_nums = sum(1 for n in msg_nums if n not in known_nums)
        if new_nums > 0:
            score2 = min(1.0, 0.3 + new_nums * 0.2)

    # Commitment detection (v083 logic)
    score3 = 0.1
    lower = content.lower()
    if _COMMITMENT_RE.search(lower):
        score3 = 0.7
        # Additional novelty from embedding if available
        if memory_embeddings is not None and len(memory_embeddings) > 0:
            emb = _embed_one(content)
            sims = _cosine_sims(emb, memory_embeddings)
            distance = 1.0 - float(np.max(sims))
            score3 = max(score3, 0.5 + distance * 0.3)

    return max(score1, score2, score3)


# ============================================================================
# CATEGORY 6: Hybrid AND-Gate Variants (101-120)
# ============================================================================


def _cosine_distance(content: str, memory_embeddings: np.ndarray) -> float:
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 1.0
    emb = _embed_one(content)
    sims = _cosine_sims(emb, memory_embeddings)
    return _clamp(1.0 - float(np.max(sims)))


def _ncd_to_nearest(content: str, memory_contents: list[str]) -> float:
    if not memory_contents:
        return 1.0
    nearest = _nearest_text(content, memory_contents)
    return _clamp(_ncd(content, nearest))


def _entity_novelty_score(content: str, memory_contents: list[str]) -> float:
    if not memory_contents:
        return 1.0
    known = _all_memory_proper_nouns(memory_contents)
    msg_proper = _extract_proper_nouns(content)
    if not msg_proper:
        return 0.0
    new_count = sum(1 for p in msg_proper if p.lower() not in known)
    return _clamp(new_count / 2.0)


def _commitment_score(content: str) -> float:
    return 0.8 if _COMMITMENT_RE.search(content.lower()) else 0.1


def _compression_novelty(content: str, memory_contents: list[str]) -> float:
    if not memory_contents:
        return 1.0
    corpus = " ".join(memory_contents[-20:])
    c_corpus = _gz_len(corpus)
    c_combined = _gz_len(corpus + " " + content)
    c_msg = _gz_len(content)
    if c_msg < 1:
        return 0.0
    return _clamp((c_combined - c_corpus) / c_msg)


def variant_101(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """speech_act_score × cosine_distance."""
    return _clamp(_speech_act_score(content) * _cosine_distance(content, memory_embeddings))


def variant_102(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """speech_act_score × NCD_to_nearest."""
    return _clamp(_speech_act_score(content) * _ncd_to_nearest(content, memory_contents))


def variant_103(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """speech_act_score × entity_novelty_count."""
    return _clamp(_speech_act_score(content) * _entity_novelty_score(content, memory_contents))


def variant_104(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """declaration_score × cosine_distance."""
    return _clamp(_declaration_score(content) * _cosine_distance(content, memory_embeddings))


def variant_105(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """declaration_score × compression_novelty."""
    return _clamp(_declaration_score(content) * _compression_novelty(content, memory_contents))


def variant_106(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """factual_density × cosine_distance."""
    return _clamp(_factual_density(content) * _cosine_distance(content, memory_embeddings))


def variant_107(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """factual_density × Mahalanobis_distance."""
    return _clamp(_factual_density(content) * variant_003(content, memory_contents, memory_embeddings))


def variant_108(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """(speech_act OR commitment) × cosine_distance — soft-OR gate."""
    sa = _speech_act_score(content)
    commit = _commitment_score(content)
    content_gate = max(sa, commit)
    return _clamp(content_gate * _cosine_distance(content, memory_embeddings))


def variant_109(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """(speech_act OR new_entity) × cosine_distance — entity bypass."""
    sa = _speech_act_score(content)
    entity_score = _entity_novelty_score(content, memory_contents)
    content_gate = max(sa, entity_score)
    return _clamp(content_gate * _cosine_distance(content, memory_embeddings))


def variant_110(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """max(content × distance, 0.8 × new_entity_score) — hard entity bypass."""
    content_score = _speech_act_score(content)
    distance = _cosine_distance(content, memory_embeddings)
    entity_score = _entity_novelty_score(content, memory_contents)
    return _clamp(max(content_score * distance, 0.8 * entity_score))


def variant_111(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """max(content × distance, 0.7 × commitment_score) — commitment bypass."""
    content_score = _speech_act_score(content)
    distance = _cosine_distance(content, memory_embeddings)
    commit = _commitment_score(content)
    return _clamp(max(content_score * distance, 0.7 * commit))


def variant_112(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """content × distance + 0.3 × entity_novelty — additive entity bonus."""
    content_score = _speech_act_score(content)
    distance = _cosine_distance(content, memory_embeddings)
    entity_score = _entity_novelty_score(content, memory_contents)
    return _clamp(content_score * distance + 0.3 * entity_score)


def variant_113(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """sqrt(content × distance) — geometric mean."""
    content_score = _speech_act_score(content)
    distance = _cosine_distance(content, memory_embeddings)
    return _clamp(math.sqrt(content_score * distance))


def variant_114(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """min(content, distance) — conservative: both must be high."""
    content_score = _speech_act_score(content)
    distance = _cosine_distance(content, memory_embeddings)
    return _clamp(min(content_score, distance))


def variant_115(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Balanced: max * 0.5 + min * 0.5."""
    content_score = _speech_act_score(content)
    distance = _cosine_distance(content, memory_embeddings)
    return _clamp(max(content_score, distance) * 0.5 + min(content_score, distance) * 0.5)


def variant_116(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Weighted geometric mean favoring distance: content^0.3 × distance^0.7."""
    content_score = max(_speech_act_score(content), 0.01)
    distance = max(_cosine_distance(content, memory_embeddings), 0.01)
    return _clamp(content_score ** 0.3 * distance ** 0.7)


def variant_117(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Weighted geometric mean favoring content: content^0.7 × distance^0.3."""
    content_score = max(_speech_act_score(content), 0.01)
    distance = max(_cosine_distance(content, memory_embeddings), 0.01)
    return _clamp(content_score ** 0.7 * distance ** 0.3)


def variant_118(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Binary content gate (>0.2), continuous distance."""
    content_score = _speech_act_score(content)
    distance = _cosine_distance(content, memory_embeddings)
    gate = 1.0 if content_score > 0.2 else 0.05
    return _clamp(gate * distance)


def variant_119(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Continuous content, binary distance gate (>0.3)."""
    content_score = _speech_act_score(content)
    distance = _cosine_distance(content, memory_embeddings)
    gate = 1.0 if distance > 0.3 else 0.1
    return _clamp(content_score * gate)


def variant_120(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Logistic combination of content, distance, entity novelty, commitment."""
    content_score = _speech_act_score(content)
    distance = _cosine_distance(content, memory_embeddings)
    entity_score = _entity_novelty_score(content, memory_contents)
    commit = _commitment_score(content)

    # Hand-tuned logistic weights (will be replaced by learned weights in combo sweep)
    logit = -2.0 + 2.5 * content_score + 3.0 * distance + 2.0 * entity_score + 1.5 * commit
    return _clamp(1.0 / (1.0 + math.exp(-logit)))


# ============================================================================
# Registry
# ============================================================================

ALL_VARIANTS: list[tuple[str, str, Callable]] = []

# Auto-register all variant_NNN functions
import sys as _sys
_module = _sys.modules[__name__]
for _i in range(1, 121):
    _name = f"variant_{_i:03d}"
    _fn = getattr(_module, _name, None)
    if _fn is not None:
        _desc = (_fn.__doc__ or "").strip().split("\n")[0]
        ALL_VARIANTS.append((_name, _desc, _fn))

assert len(ALL_VARIANTS) == 120, f"Expected 120 variants, got {len(ALL_VARIANTS)}"
