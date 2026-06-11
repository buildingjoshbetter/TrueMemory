#!/usr/bin/env python3
"""
100-variant prediction error sweep v2 — 10 computational paradigms.

Unlike v1 (keyword matching), v2 uses genuinely different computational
substrates: NLI models, embedding geometry, triple extraction, knowledge
graphs, cross-encoders, discourse analysis, stylometry, sequence alignment,
change-point detection, and masked slot-filling.

Signature: variant_NNN(content, memory_contents, memory_embeddings=None) -> float [0,1]
"""

from __future__ import annotations

import difflib
import gzip
import math
import re
from collections import Counter, defaultdict
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Embedding model — shared by all paradigms
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
# Model loading — NLI, Cross-Encoder, MLM (with fallbacks)
# ---------------------------------------------------------------------------

_NLI_MODEL = None
_NLI_TOKENIZER = None
_CROSS_ENCODER = None
_MLM_MODEL = None
_MLM_TOKENIZER = None
_HAS_NLI = False
_HAS_CROSS_ENCODER = False
_HAS_MLM = False

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _nli_name = "cross-encoder/nli-MiniLM2-L6-H768"
    _NLI_TOKENIZER = AutoTokenizer.from_pretrained(_nli_name)
    _NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(_nli_name)
    _NLI_MODEL.eval()
    _HAS_NLI = True
    print(f"NLI model loaded: {_nli_name}")
except Exception as e:
    print(f"NLI model not available: {e}")

try:
    from sentence_transformers import CrossEncoder as _CE
    _CROSS_ENCODER = _CE("cross-encoder/stsb-distilroberta-base")
    _HAS_CROSS_ENCODER = True
    print("Cross-encoder loaded: cross-encoder/stsb-distilroberta-base")
except Exception as e:
    print(f"Cross-encoder not available: {e}")

try:
    import torch as _torch2
    from transformers import AutoModelForMaskedLM, AutoTokenizer as _MLMTok

    _mlm_name = "distilbert-base-uncased"
    _MLM_TOKENIZER = _MLMTok.from_pretrained(_mlm_name)
    _MLM_MODEL = AutoModelForMaskedLM.from_pretrained(_mlm_name)
    _MLM_MODEL.eval()
    _HAS_MLM = True
    print(f"MLM loaded: {_mlm_name}")
except Exception as e:
    print(f"MLM not available: {e}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


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
    "lmao dead", "im dead", "crying", "screaming", "yo", "heyyy", "hey",
    "hi", "hello", "sup", "what's up",
})

_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")

_COMMON_PROPER = frozenset({
    "I", "The", "A", "An", "Is", "It", "He", "She", "We", "They",
    "My", "Your", "His", "Her", "Our", "This", "That", "What",
    "When", "Where", "Who", "How", "Why", "But", "And", "Or",
    "So", "If", "Not", "No", "Yes", "Can", "Will", "Just",
    "Do", "Did", "Has", "Have", "Had", "Was", "Were", "Are",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "Going", "Looking", "Don", "Let", "Also", "Still", "Even",
    "Actually", "Really", "Honestly", "Apparently", "Oh",
})

_LOCATION_RE = re.compile(
    r"\b(?:in|to|from|at|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
)

_NUMBER_RE = re.compile(
    r"\$[\d,.]+[KMBkmb]?|\d+\.?\d*\s*%|\d{1,3}(?:,\d{3})+|\d+\.?\d*"
)

_POSITIVE_WORDS = frozenset({
    "love", "great", "amazing", "awesome", "wonderful", "fantastic",
    "excellent", "happy", "enjoy", "best", "perfect", "good",
    "thrilled", "excited", "glad", "pleased", "delighted",
    "beautiful", "incredible", "brilliant", "superb", "terrific",
})

_NEGATIVE_WORDS = frozenset({
    "hate", "terrible", "awful", "horrible", "worst", "bad",
    "disgusting", "miserable", "unhappy", "disappointed", "angry",
    "frustrated", "annoyed", "furious", "devastated", "depressed",
    "broke", "broken", "failed", "dead", "died", "lost", "lonely",
})

_FUNCTIONAL_RELATIONS = frozenset({
    "lives_in", "works_at", "dating", "married_to", "job_title",
    "salary", "school", "major", "city", "uses_editor",
})


def _is_noise(text: str) -> bool:
    return text.lower().strip().rstrip("!?.… ") in _NOISE_EXACT or len(text.strip()) < 3


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOPWORDS and len(w) > 1}


def _extract_entities(text: str) -> set[str]:
    entities = set()
    for m in _PROPER_NOUN_RE.finditer(text):
        noun = m.group(0)
        if noun not in _COMMON_PROPER and len(noun) > 1:
            entities.add(noun.lower())
    return entities


def _extract_numbers(text: str) -> list[str]:
    return [m.group(0).replace(",", "").strip() for m in _NUMBER_RE.finditer(text)]


def _extract_locations(text: str) -> set[str]:
    locs = set()
    for m in _LOCATION_RE.finditer(text):
        loc = m.group(1)
        if loc not in _COMMON_PROPER:
            locs.add(loc.lower())
    return locs


def _word_overlap(a: str, b: str) -> float:
    wa = _content_words(a)
    wb = _content_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _entity_overlap(msg: str, mem: str) -> float:
    e1 = _extract_entities(msg)
    e2 = _extract_entities(mem)
    cw1 = _content_words(msg)
    cw2 = _content_words(mem)
    entity_jac = _jaccard(e1, e2) if (e1 and e2) else 0.0
    word_jac = _jaccard(cw1, cw2)
    return max(entity_jac, word_jac * 0.5)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b)) / (na * nb)


def _cosine_sims(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    normed = matrix / norms
    qn = np.linalg.norm(query)
    if qn < 1e-10:
        return np.zeros(len(matrix))
    return normed @ (query / qn)


def _nearest_by_embedding(emb: np.ndarray, memory_embeddings: np.ndarray) -> tuple[int, float]:
    sims = _cosine_sims(emb, memory_embeddings)
    idx = int(np.argmax(sims))
    return idx, float(sims[idx])


def _nearest_by_word_overlap(content: str, memory_contents: list[str]) -> tuple[int, float]:
    if not memory_contents:
        return -1, 0.0
    best_idx = 0
    best_score = -1.0
    for i, m in enumerate(memory_contents):
        score = _word_overlap(content, m)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx, best_score


def _nearest_by_entity(content: str, memory_contents: list[str]) -> tuple[int, float]:
    if not memory_contents:
        return -1, 0.0
    best_idx = 0
    best_score = -1.0
    for i, m in enumerate(memory_contents):
        score = _entity_overlap(content, m)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx, best_score


def _sentiment_score(text: str) -> float:
    words = set(text.lower().split())
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def _nli_score(premise: str, hypothesis: str) -> tuple[float, float, float]:
    """Returns (entailment, neutral, contradiction) probabilities."""
    if not _HAS_NLI:
        return _nli_fallback(premise, hypothesis)
    inputs = _NLI_TOKENIZER(premise, hypothesis, return_tensors="pt",
                             truncation=True, max_length=128)
    with torch.no_grad():
        logits = _NLI_MODEL(**inputs).logits[0]
    probs = torch.softmax(logits, dim=0).cpu().numpy()
    return float(probs[0]), float(probs[1]), float(probs[2])


def _nli_fallback(premise: str, hypothesis: str) -> tuple[float, float, float]:
    """Embedding-based NLI approximation when transformer model unavailable."""
    e1 = _embed_one(premise)
    e2 = _embed_one(hypothesis)
    sim = _cosine_sim(e1, e2)
    neg_premise = "not " + premise.lower()
    e_neg = _embed_one(neg_premise)
    neg_sim = _cosine_sim(e2, e_neg)
    if sim > 0.8:
        return (sim, 0.1, 0.05)
    elif neg_sim > sim + 0.1:
        return (0.1, 0.3, min(0.9, neg_sim))
    else:
        return (max(0, sim - 0.3), 0.5, max(0, 0.3 - sim))


def _cross_encoder_score(text_a: str, text_b: str) -> float:
    """Returns similarity score [0, 1] from cross-encoder."""
    if not _HAS_CROSS_ENCODER:
        return _cross_encoder_fallback(text_a, text_b)
    raw = _CROSS_ENCODER.predict([(text_a, text_b)])[0]
    return _clamp(float(raw) / 5.0)


def _cross_encoder_fallback(text_a: str, text_b: str) -> float:
    e1 = _embed_one(text_a)
    e2 = _embed_one(text_b)
    return _clamp((_cosine_sim(e1, e2) + 1.0) / 2.0)


def _mlm_predict_mask(context: str, mask_pos: int = -1) -> dict:
    """Predict masked token. Returns {token: prob} for top-10."""
    if not _HAS_MLM:
        return {}
    tokens = _MLM_TOKENIZER(context, return_tensors="pt", truncation=True, max_length=128)
    input_ids = tokens["input_ids"][0]
    mask_id = _MLM_TOKENIZER.mask_token_id
    mask_positions = (input_ids == mask_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        return {}
    pos = mask_positions[0] if mask_pos < 0 else mask_positions[min(mask_pos, len(mask_positions) - 1)]
    with _torch2.no_grad():
        logits = _MLM_MODEL(**tokens).logits[0, pos]
    probs = _torch2.softmax(logits, dim=0)
    top_k = _torch2.topk(probs, 10)
    result = {}
    for idx, prob in zip(top_k.indices.tolist(), top_k.values.tolist()):
        token = _MLM_TOKENIZER.decode([idx]).strip()
        result[token] = prob
    return result


# ---------------------------------------------------------------------------
# Per-conversation state for stateful paradigms
# ---------------------------------------------------------------------------

_CONV_STATE: dict = {}


def reset_conversation_state():
    """Call between conversations to reset stateful paradigm data."""
    global _CONV_STATE
    _CONV_STATE = {
        "entity_embeddings": defaultdict(list),  # P2, P9
        "entity_ema": {},                         # P2: embedding EMA per entity
        "entity_velocity": {},                    # P2: velocity tracking
        "entity_sentiments": defaultdict(list),   # P6, P9
        "entity_messages": defaultdict(list),     # P6, P7, P9
        "entity_stances": defaultdict(list),      # P6
        "commitments": [],                        # P6
        "kg": defaultdict(lambda: defaultdict(list)),  # P4
        "entity_word_sets": defaultdict(set),     # P7
        "entity_pronoun_history": defaultdict(list),  # P7
        "entity_style_baselines": defaultdict(lambda: {
            "emphasis_count": 0, "msg_count": 0,
            "avg_len": 0, "hedge_count": 0, "certain_count": 0
        }),
    }


reset_conversation_state()


def _update_entity_state(content: str, memory_contents: list[str]):
    """Update per-conversation state with new message."""
    entities = _extract_entities(content)
    emb = _embed_one(content)
    sentiment = _sentiment_score(content)
    words = _content_words(content)

    for ent in entities:
        _CONV_STATE["entity_embeddings"][ent].append(emb)
        _CONV_STATE["entity_sentiments"][ent].append(sentiment)
        _CONV_STATE["entity_messages"][ent].append(content)
        _CONV_STATE["entity_word_sets"][ent].update(words)

        # Update EMA
        alpha = 0.3
        if ent in _CONV_STATE["entity_ema"]:
            old_ema = _CONV_STATE["entity_ema"][ent]
            _CONV_STATE["entity_ema"][ent] = alpha * emb + (1 - alpha) * old_ema
        else:
            _CONV_STATE["entity_ema"][ent] = emb.copy()

        # Update velocity
        embs = _CONV_STATE["entity_embeddings"][ent]
        if len(embs) >= 2:
            _CONV_STATE["entity_velocity"][ent] = embs[-1] - embs[-2]

        # Update style baselines
        baseline = _CONV_STATE["entity_style_baselines"][ent]
        baseline["msg_count"] += 1
        caps = sum(1 for c in content if c.isupper())
        excl = content.count("!")
        baseline["emphasis_count"] += caps + excl
        baseline["avg_len"] = (baseline["avg_len"] * (baseline["msg_count"] - 1) + len(content.split())) / baseline["msg_count"]

        # Pronoun tracking
        lower = content.lower()
        if " we " in lower or lower.startswith("we ") or " we'" in lower:
            _CONV_STATE["entity_pronoun_history"][ent].append("we")
        elif " i " in lower or lower.startswith("i ") or " i'" in lower:
            _CONV_STATE["entity_pronoun_history"][ent].append("i")

    # Commitment tracking
    commit_patterns = [
        r"\bi will\b", r"\bi'm going to\b", r"\bi decided\b",
        r"\bi plan to\b", r"\bwe will\b", r"\bwe're going to\b",
    ]
    for pat in commit_patterns:
        if re.search(pat, content, re.IGNORECASE):
            _CONV_STATE["commitments"].append(content)
            break


# ============================================================================
# PARADIGM 1: NLI Contradiction Classifiers (001-010)
# ============================================================================


def variant_001(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """NLI contradiction against nearest memory (word overlap retrieval)."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    idx, overlap = _nearest_by_word_overlap(content, memory_contents)
    if overlap < 0.05:
        return 0.0
    _, _, contradiction = _nli_score(memory_contents[idx], content)
    return _clamp(contradiction)


def variant_002(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """NLI contradiction against nearest memory (embedding retrieval)."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    idx, sim = _nearest_by_embedding(emb, memory_embeddings)
    if sim < 0.2:
        return 0.0
    _, _, contradiction = _nli_score(memory_contents[idx], content)
    return _clamp(contradiction)


def variant_003(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Max NLI contradiction across top-5 memories."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    sims = _cosine_sims(emb, memory_embeddings)
    top_indices = np.argsort(sims)[-5:]
    max_c = 0.0
    for idx in top_indices:
        if sims[idx] < 0.15:
            continue
        _, _, c = _nli_score(memory_contents[idx], content)
        max_c = max(max_c, c)
    return _clamp(max_c)


def variant_004(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """NLI contradiction weighted by topic overlap."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    idx, sim = _nearest_by_embedding(emb, memory_embeddings)
    if sim < 0.2:
        return 0.0
    overlap = _entity_overlap(content, memory_contents[idx])
    _, _, contradiction = _nli_score(memory_contents[idx], content)
    return _clamp(contradiction * (0.3 + 0.7 * overlap))


def variant_005(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """NLI with memory chunking — split long memories into sentences."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    idx, sim = _nearest_by_embedding(emb, memory_embeddings)
    if sim < 0.15:
        return 0.0
    mem = memory_contents[idx]
    sentences = [s.strip() for s in re.split(r'[.!?]+', mem) if s.strip() and len(s.strip()) > 5]
    if not sentences:
        sentences = [mem]
    max_c = 0.0
    for sent in sentences[:5]:
        _, _, c = _nli_score(sent, content)
        max_c = max(max_c, c)
    return _clamp(max_c)


def variant_006(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """NLI entailment deficit — low entailment despite same topic = PE."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    idx, sim = _nearest_by_embedding(emb, memory_embeddings)
    if sim < 0.2:
        return 0.0
    ent, neu, con = _nli_score(memory_contents[idx], content)
    topic_overlap = _entity_overlap(content, memory_contents[idx])
    if topic_overlap < 0.1:
        return 0.0
    expected_ent = 0.5 + 0.4 * topic_overlap
    deficit = max(0, expected_ent - ent)
    return _clamp(deficit + con * 0.5)


def variant_007(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """NLI contradiction-to-neutral ratio."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    idx, sim = _nearest_by_embedding(emb, memory_embeddings)
    if sim < 0.2:
        return 0.0
    _, neu, con = _nli_score(memory_contents[idx], content)
    ratio = con / (neu + 0.1)
    return _clamp(ratio)


def variant_008(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Bidirectional NLI — max contradiction in both directions."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    idx, sim = _nearest_by_embedding(emb, memory_embeddings)
    if sim < 0.2:
        return 0.0
    mem = memory_contents[idx]
    _, _, c_fwd = _nli_score(mem, content)
    _, _, c_bwd = _nli_score(content, mem)
    return _clamp(max(c_fwd, c_bwd))


def variant_009(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """NLI with context window — 3 most recent entity-related memories."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        if memory_embeddings is not None:
            emb = _embed_one(content)
            idx, sim = _nearest_by_embedding(emb, memory_embeddings)
            if sim < 0.2:
                return 0.0
            _, _, c = _nli_score(memory_contents[idx], content)
            return _clamp(c)
        return 0.0
    related = []
    for i, mem in enumerate(memory_contents):
        mem_entities = _extract_entities(mem)
        if msg_entities & mem_entities:
            related.append(i)
    if not related:
        return 0.0
    context_mems = related[-3:]
    combined_mem = " ".join(memory_contents[i] for i in context_mems)
    _, _, c = _nli_score(combined_mem, content)
    return _clamp(c)


def variant_010(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """NLI ensemble — average of v001, v002, v008."""
    s1 = variant_001(content, memory_contents, memory_embeddings)
    s2 = variant_002(content, memory_contents, memory_embeddings)
    s3 = variant_008(content, memory_contents, memory_embeddings)
    return _clamp((s1 + s2 + s3) / 3.0)


# ============================================================================
# PARADIGM 2: Embedding Direction Geometry (011-020)
# ============================================================================


def variant_011(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Cosine of difference vectors — opposite directions in entity space = PE."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        return 0.0
    emb_msg = _embed_one(content)
    max_pe = 0.0
    for ent in msg_entities:
        ent_emb = _embed_one(ent)
        for i, mem in enumerate(memory_contents):
            if ent.lower() in mem.lower():
                diff_mem = memory_embeddings[i] - ent_emb
                diff_msg = emb_msg - ent_emb
                cos = _cosine_sim(diff_mem, diff_msg)
                pe = (1.0 - cos) / 2.0
                max_pe = max(max_pe, pe)
    return _clamp(max_pe)


def variant_012(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Projection onto memory axis — negative projection = PE."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        return 0.0
    emb_msg = _embed_one(content)
    max_pe = 0.0
    for ent in msg_entities:
        ent_emb = _embed_one(ent)
        for i, mem in enumerate(memory_contents):
            if ent.lower() in mem.lower():
                axis = memory_embeddings[i] - ent_emb
                axis_norm = np.linalg.norm(axis)
                if axis_norm < 1e-10:
                    continue
                axis_unit = axis / axis_norm
                msg_vec = emb_msg - ent_emb
                projection = float(np.dot(msg_vec, axis_unit))
                if projection < 0:
                    pe = min(1.0, abs(projection) / (axis_norm + 1e-10))
                    max_pe = max(max_pe, pe)
    return _clamp(max_pe)


def variant_013(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Angular distance in entity-centered space."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        return 0.0
    emb_msg = _embed_one(content)
    max_pe = 0.0
    for ent in msg_entities:
        ent_emb = _embed_one(ent)
        for i, mem in enumerate(memory_contents):
            if ent.lower() in mem.lower():
                vec_mem = memory_embeddings[i] - ent_emb
                vec_msg = emb_msg - ent_emb
                cos = _cosine_sim(vec_mem, vec_msg)
                cos = max(-1.0, min(1.0, cos))
                angle = math.acos(cos)
                pe = angle / math.pi
                max_pe = max(max_pe, pe)
    return _clamp(max_pe)


def variant_014(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Embedding velocity reversal — direction change from EMA trend."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        return 0.0
    emb_msg = _embed_one(content)
    max_pe = 0.0
    for ent in msg_entities:
        if ent in _CONV_STATE["entity_velocity"]:
            velocity = _CONV_STATE["entity_velocity"][ent]
            ent_emb = _embed_one(ent)
            new_direction = emb_msg - ent_emb
            cos = _cosine_sim(velocity, new_direction)
            pe = (1.0 - cos) / 2.0
            max_pe = max(max_pe, pe)
    return _clamp(max_pe)


def variant_015(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Centroid displacement direction — large displacement in new direction."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        return 0.0
    emb_msg = _embed_one(content)
    max_pe = 0.0
    for ent in msg_entities:
        embs = _CONV_STATE["entity_embeddings"].get(ent, [])
        if len(embs) < 2:
            continue
        centroid = np.mean(embs, axis=0)
        centroid_dir = centroid / (np.linalg.norm(centroid) + 1e-10)
        displacement = emb_msg - centroid
        disp_mag = np.linalg.norm(displacement)
        if disp_mag < 1e-10:
            continue
        disp_dir = displacement / disp_mag
        cos = _cosine_sim(centroid_dir, disp_dir)
        pe = disp_mag * (1.0 - cos) / 2.0
        max_pe = max(max_pe, min(1.0, pe))
    return _clamp(max_pe)


def variant_016(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Principal component violation — message opposes entity's PC1."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        return 0.0
    emb_msg = _embed_one(content)
    max_pe = 0.0
    for ent in msg_entities:
        embs = _CONV_STATE["entity_embeddings"].get(ent, [])
        if len(embs) < 3:
            continue
        mat = np.array(embs)
        mean = mat.mean(axis=0)
        centered = mat - mean
        try:
            _, s, vt = np.linalg.svd(centered, full_matrices=False)
            pc1 = vt[0]
            msg_centered = emb_msg - mean
            proj = float(np.dot(msg_centered, pc1))
            mean_proj = float(np.mean(centered @ pc1))
            if proj * mean_proj < 0:
                pe = min(1.0, abs(proj - mean_proj) / (s[0] / len(embs) + 1e-10))
                max_pe = max(max_pe, pe)
        except Exception:
            pass
    return _clamp(max_pe)


def variant_017(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Semantic differential — positive/negative anchor axis projection."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    pos_anchor = _embed_one("good happy together love")
    neg_anchor = _embed_one("bad sad apart hate broke")
    axis = neg_anchor - pos_anchor
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        return 0.0
    axis_unit = axis / axis_norm
    emb_msg = _embed_one(content)
    msg_proj = float(np.dot(emb_msg, axis_unit))
    max_pe = 0.0
    for i, mem in enumerate(memory_contents):
        overlap = _entity_overlap(content, mem)
        if overlap < 0.1:
            continue
        mem_proj = float(np.dot(memory_embeddings[i], axis_unit))
        if (msg_proj > 0 and mem_proj < 0) or (msg_proj < 0 and mem_proj > 0):
            pe = abs(msg_proj - mem_proj) / (2 * axis_norm + 1e-10)
            max_pe = max(max_pe, min(1.0, pe * 2))
    return _clamp(max_pe)


def variant_018(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Relative direction change — angle > 90° between entity→memory and entity→message."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        return 0.0
    emb_msg = _embed_one(content)
    max_pe = 0.0
    for ent in msg_entities:
        ent_emb = _embed_one(ent)
        for i, mem in enumerate(memory_contents):
            if ent.lower() in mem.lower():
                dir_mem = memory_embeddings[i] - ent_emb
                dir_msg = emb_msg - ent_emb
                cos = _cosine_sim(dir_mem, dir_msg)
                cos = max(-1.0, min(1.0, cos))
                angle_deg = math.degrees(math.acos(cos))
                if angle_deg > 90:
                    pe = (angle_deg - 90) / 90.0
                    max_pe = max(max_pe, pe)
    return _clamp(max_pe)


def variant_019(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Embedding trajectory curvature — high curvature = sudden direction change."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        return 0.0
    emb_msg = _embed_one(content)
    max_pe = 0.0
    for ent in msg_entities:
        embs = _CONV_STATE["entity_embeddings"].get(ent, [])
        if len(embs) < 2:
            continue
        v_prev = embs[-1] - embs[-2]
        v_curr = emb_msg - embs[-1]
        cos = _cosine_sim(v_prev, v_curr)
        curvature = (1.0 - cos) / 2.0
        max_pe = max(max_pe, curvature)
    return _clamp(max_pe)


def variant_020(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Geometric PE ensemble — weighted mean of v011, v014, v017."""
    s1 = variant_011(content, memory_contents, memory_embeddings)
    s2 = variant_014(content, memory_contents, memory_embeddings)
    s3 = variant_017(content, memory_contents, memory_embeddings)
    return _clamp(0.4 * s1 + 0.3 * s2 + 0.3 * s3)


# ============================================================================
# PARADIGM 3: Structured Triple Extraction (021-030)
# ============================================================================

_TRIPLE_TEMPLATES = [
    (r"(?:i|he|she|we|they|[A-Z]\w+)\s+(?:live|lives|lived)\s+(?:in|at)\s+(.+?)(?:\.|,|$|!|\?)", "lives_in"),
    (r"(?:i|he|she|we|they|[A-Z]\w+)\s+(?:work|works|worked)\s+(?:at|for)\s+(.+?)(?:\.|,|$|!|\?)", "works_at"),
    (r"(?:i|he|she|we|they|[A-Z]\w+)\s+(?:use|uses|used)\s+(.+?)(?:\.|,|$|!|\?)", "uses"),
    (r"(?:i|he|she|we|they|[A-Z]\w+)\s+(?:prefer|prefers|preferred)\s+(.+?)(?:\.|,|$|!|\?)", "prefers"),
    (r"(?:i|he|she|we|they|[A-Z]\w+)\s+(?:study|studies|studied)\s+(?:at|in)\s+(.+?)(?:\.|,|$|!|\?)", "studies_at"),
    (r"(?:i|he|she|we|they|[A-Z]\w+)\s+(?:moved|moving)\s+to\s+(.+?)(?:\.|,|$|!|\?)", "lives_in"),
    (r"(?:i|he|she|we|they|[A-Z]\w+)\s+(?:switched|switching)\s+to\s+(.+?)(?:\.|,|$|!|\?)", "uses"),
    (r"(?:i|he|she|we|they|[A-Z]\w+)\s+(?:am|is|are|was|were|'m|'s|'re)\s+(?:a|an)\s+(.+?)(?:\.|,|$|!|\?)", "is_a"),
    (r"(?:i|he|she|we|they|[A-Z]\w+)\s+(?:am|is|are|was|were|'m|'s|'re)\s+(?:dating|married to|engaged to|seeing)\s+(.+?)(?:\.|,|$|!|\?)", "partner"),
    (r"(?:i|he|she|we|they|[A-Z]\w+)\s+(?:earn|earns|earned|make|makes|made)\s+\$?([\d,.]+[KMBkmb]?)(?:\s|$|\.|\?|!)", "salary"),
    (r"(?:i|he|she|we|they|[A-Z]\w+)\s+(?:drive|drives|drove)\s+(?:a|an)?\s*(.+?)(?:\.|,|$|!|\?)", "drives"),
    (r"(?:i|he|she|we|they|[A-Z]\w+)\s+(?:got|getting|get)\s+(?:a|an|the)?\s*(.+?)(?:\.|,|$|!|\?)", "acquired"),
    (r"(?:i|he|she|we|they|[A-Z]\w+)\s+(?:joined|joining)\s+(.+?)(?:\.|,|$|!|\?)", "works_at"),
    (r"(?:i|he|she|we|they|[A-Z]\w+)\s+(?:quit|quitting|left|leaving)\s+(.+?)(?:\.|,|$|!|\?)", "quit"),
    (r"(?:i|he|she|we|they|[A-Z]\w+)\s+(?:broke up|breaking up|divorced|split|separated)", "single"),
]

_COMPILED_TRIPLES = [(re.compile(pat, re.IGNORECASE), rel) for pat, rel in _TRIPLE_TEMPLATES]


def _extract_triples(text: str) -> list[tuple[str, str, str]]:
    triples = []
    entities = _extract_entities(text)
    subject = list(entities)[0] if entities else "speaker"
    for regex, relation in _COMPILED_TRIPLES:
        match = regex.search(text)
        if match:
            value = match.group(1).strip().lower() if match.lastindex else ""
            triples.append((subject, relation, value))
    return triples


def _triple_conflict_score(msg_triples: list[tuple], mem_triples: list[tuple]) -> float:
    if not msg_triples or not mem_triples:
        return 0.0
    conflicts = 0
    for ms, mr, mv in msg_triples:
        for es, er, ev in mem_triples:
            same_subj = ms == es or _jaccard(set(ms.split()), set(es.split())) > 0.5
            same_rel = mr == er or (mr == "quit" and er == "works_at") or (mr == "single" and er == "partner")
            if same_subj and same_rel:
                if mr in ("quit", "single"):
                    conflicts += 1
                elif mv != ev and mv and ev:
                    conflicts += 1
    return min(1.0, conflicts / max(len(msg_triples), 1))


def variant_021(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Regex template triples — conflicting subject+relation with different values."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_triples = _extract_triples(content)
    if not msg_triples:
        return 0.0
    max_score = 0.0
    for mem in memory_contents:
        mem_triples = _extract_triples(mem)
        score = _triple_conflict_score(msg_triples, mem_triples)
        max_score = max(max_score, score)
    return _clamp(max_score)


def variant_022(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Verb-argument structure — same subject+verb but different object."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    svo_re = re.compile(
        r"(?:i|he|she|we|they|[A-Z]\w+)\s+(\w+(?:ed|ing|s)?)\s+(?:at|in|to|for|with)?\s*(.+?)(?:\.|,|$|!|\?)",
        re.IGNORECASE
    )
    msg_svos = svo_re.findall(content)
    if not msg_svos:
        return 0.0
    max_pe = 0.0
    for mem in memory_contents:
        if _entity_overlap(content, mem) < 0.1:
            continue
        mem_svos = svo_re.findall(mem)
        for mv, mo in msg_svos:
            for ev, eo in mem_svos:
                if mv.lower().rstrip("sed") == ev.lower().rstrip("sed"):
                    if mo.lower().strip() != eo.lower().strip() and mo.strip() and eo.strip():
                        max_pe = max(max_pe, 0.7)
    return _clamp(max_pe)


def variant_023(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Attribute extraction — entity+attribute pairs changing."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    msg_locs = _extract_locations(content)
    msg_nums = _extract_numbers(content)
    if not msg_entities:
        return 0.0
    max_pe = 0.0
    for mem in memory_contents:
        mem_entities = _extract_entities(mem)
        shared = msg_entities & mem_entities
        if not shared:
            continue
        mem_locs = _extract_locations(mem)
        mem_nums = _extract_numbers(mem)
        if msg_locs and mem_locs and msg_locs != mem_locs:
            max_pe = max(max_pe, 0.8)
        if msg_nums and mem_nums and set(msg_nums) != set(mem_nums):
            max_pe = max(max_pe, 0.7)
    return _clamp(max_pe)


def variant_024(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Possessive relation change — X's Y patterns with different values."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    poss_re = re.compile(r"(\w+)'s\s+(\w+)", re.IGNORECASE)
    has_re = re.compile(r"(\w+)\s+(?:has|have|had)\s+(?:a|an|the)?\s*(\w+)", re.IGNORECASE)
    msg_poss = poss_re.findall(content) + has_re.findall(content)
    if not msg_poss:
        return 0.0
    max_pe = 0.0
    for mem in memory_contents:
        mem_poss = poss_re.findall(mem) + has_re.findall(mem)
        for mo, ms in msg_poss:
            for eo, es in mem_poss:
                if mo.lower() == eo.lower() and ms.lower() != es.lower():
                    max_pe = max(max_pe, 0.6)
    return _clamp(max_pe)


def variant_025(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Temporal slot update — same event, different time."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    time_re = re.compile(
        r"(?:on|at|by|before|after|until)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}(?::\d{2})?(?:\s*[ap]m)?|\d{1,2}/\d{1,2})",
        re.IGNORECASE
    )
    msg_times = [m.group(1).lower() for m in time_re.finditer(content)]
    if not msg_times:
        return 0.0
    max_pe = 0.0
    for mem in memory_contents:
        if _entity_overlap(content, mem) < 0.1:
            continue
        mem_times = [m.group(1).lower() for m in time_re.finditer(mem)]
        if mem_times and msg_times and set(msg_times) != set(mem_times):
            max_pe = max(max_pe, 0.6)
    return _clamp(max_pe)


def variant_026(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Numeric slot update — PE proportional to magnitude of change."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_nums = _extract_numbers(content)
    if not msg_nums:
        return 0.0
    max_pe = 0.0
    for mem in memory_contents:
        if _entity_overlap(content, mem) < 0.1:
            continue
        mem_nums = _extract_numbers(mem)
        if not mem_nums:
            continue
        for mn in msg_nums:
            for en in mem_nums:
                try:
                    mv = float(mn.replace("$", "").replace("k", "000").replace("K", "000")
                              .replace("m", "000000").replace("M", "000000").replace("%", ""))
                    ev = float(en.replace("$", "").replace("k", "000").replace("K", "000")
                              .replace("m", "000000").replace("M", "000000").replace("%", ""))
                    if ev > 0 and mv > 0 and mv != ev:
                        ratio = max(mv, ev) / min(mv, ev)
                        pe = min(1.0, math.log(ratio) / math.log(3))
                        max_pe = max(max_pe, pe)
                except (ValueError, ZeroDivisionError):
                    pass
    return _clamp(max_pe)


def variant_027(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Stance/opinion extraction — conflicting sentiment about same topic."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_sent = _sentiment_score(content)
    if msg_sent == 0.0:
        return 0.0
    max_pe = 0.0
    for mem in memory_contents:
        overlap = _entity_overlap(content, mem)
        if overlap < 0.1:
            continue
        mem_sent = _sentiment_score(mem)
        if mem_sent == 0.0:
            continue
        if (msg_sent > 0 and mem_sent < 0) or (msg_sent < 0 and mem_sent > 0):
            pe = abs(msg_sent - mem_sent) / 2.0
            max_pe = max(max_pe, pe * overlap * 2)
    return _clamp(max_pe)


def variant_028(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Dependency-like comparison via regex verb extraction."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    dep_re = re.compile(r"(\w+)\s+(is|are|was|were|has|have|had|does|do|did|will|would|can|could)\s+(\w+)", re.IGNORECASE)
    msg_deps = [(m.group(1).lower(), m.group(2).lower(), m.group(3).lower()) for m in dep_re.finditer(content)]
    if not msg_deps:
        return 0.0
    max_pe = 0.0
    for mem in memory_contents:
        mem_deps = [(m.group(1).lower(), m.group(2).lower(), m.group(3).lower()) for m in dep_re.finditer(mem)]
        for ms, mv, mo in msg_deps:
            for es, ev, eo in mem_deps:
                if ms == es and mv == ev and mo != eo:
                    max_pe = max(max_pe, 0.6)
                elif ms == es and mo == eo and mv != ev:
                    max_pe = max(max_pe, 0.4)
    return _clamp(max_pe)


def variant_029(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Coreference-aware triples — resolve pronouns then extract triples."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    resolved = content
    lower = content.lower()
    pronouns = {"she", "he", "they", "her", "him", "them"}
    found_pronouns = {w for w in lower.split() if w in pronouns}
    if found_pronouns:
        for mem in reversed(memory_contents[-5:]):
            mem_entities = _extract_entities(mem)
            if mem_entities:
                main_entity = list(mem_entities)[0]
                for p in found_pronouns:
                    resolved = re.sub(r'\b' + p + r'\b', main_entity, resolved, flags=re.IGNORECASE, count=1)
                break
    msg_triples = _extract_triples(resolved)
    if not msg_triples:
        return 0.0
    max_score = 0.0
    for mem in memory_contents:
        mem_triples = _extract_triples(mem)
        score = _triple_conflict_score(msg_triples, mem_triples)
        max_score = max(max_score, score)
    return _clamp(max_score)


def variant_030(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Triple extraction ensemble — max of v021, v026, v027."""
    s1 = variant_021(content, memory_contents, memory_embeddings)
    s2 = variant_026(content, memory_contents, memory_embeddings)
    s3 = variant_027(content, memory_contents, memory_embeddings)
    return max(s1, s2, s3)


# ============================================================================
# PARADIGM 4: Knowledge Graph Conflict Detection (031-040)
# ============================================================================


def _kg_add_and_detect(content: str, memory_contents: list[str]) -> tuple[int, int, list[tuple]]:
    """Add triples to KG, return (conflicts, total_edges, conflicting_triples)."""
    kg = _CONV_STATE["kg"]
    msg_triples = _extract_triples(content)
    conflicts = 0
    conflicting = []
    for subj, rel, val in msg_triples:
        existing = kg[subj].get(rel, [])
        for old_val, _, _ in existing:
            if old_val != val and old_val and val:
                if rel in ("quit", "single") or rel in _FUNCTIONAL_RELATIONS:
                    conflicts += 1
                    conflicting.append((subj, rel, old_val, val))
                elif old_val != val:
                    conflicts += 1
                    conflicting.append((subj, rel, old_val, val))
        kg[subj][rel].append((val, len(memory_contents), content))
    total_edges = sum(len(rels) for rels in kg.values() for rels in rels.values())
    return conflicts, total_edges, conflicting


def variant_031(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Simple KG edge conflict — same entity+relation, different value."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    conflicts, _, _ = _kg_add_and_detect(content, memory_contents)
    return _clamp(min(1.0, conflicts * 0.5))


def variant_032(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Edge conflict weighted by node degree."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    kg = _CONV_STATE["kg"]
    msg_triples = _extract_triples(content)
    if not msg_triples:
        return 0.0
    max_pe = 0.0
    for subj, rel, val in msg_triples:
        existing = kg[subj].get(rel, [])
        degree = sum(len(v) for v in kg[subj].values())
        for old_val, _, _ in existing:
            if old_val != val and old_val and val:
                pe = min(1.0, 0.5 * math.log(degree + 1 + 1e-10))
                max_pe = max(max_pe, pe)
    return _clamp(max_pe)


def variant_033(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Cascade impact — count downstream edges invalidated by change."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    kg = _CONV_STATE["kg"]
    msg_triples = _extract_triples(content)
    if not msg_triples:
        return 0.0
    cascade_count = 0
    for subj, rel, val in msg_triples:
        existing = kg[subj].get(rel, [])
        for old_val, _, _ in existing:
            if old_val != val and old_val and val:
                for other_rel, entries in kg[subj].items():
                    if other_rel != rel:
                        for entry_val, _, entry_msg in entries:
                            if old_val.lower() in entry_msg.lower():
                                cascade_count += 1
                if old_val in kg:
                    cascade_count += sum(len(v) for v in kg[old_val].values())
    return _clamp(min(1.0, cascade_count * 0.2))


def variant_034(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Temporal edge decay — recent conflicts weighted higher."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    kg = _CONV_STATE["kg"]
    msg_triples = _extract_triples(content)
    if not msg_triples:
        return 0.0
    current_idx = len(memory_contents)
    max_pe = 0.0
    for subj, rel, val in msg_triples:
        existing = kg[subj].get(rel, [])
        for old_val, old_idx, _ in existing:
            if old_val != val and old_val and val:
                recency = 1.0 / (1.0 + (current_idx - old_idx) * 0.1)
                max_pe = max(max_pe, 0.6 * recency)
    return _clamp(max_pe)


def variant_035(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Functional dependency violation — single-valued relations get higher PE."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    kg = _CONV_STATE["kg"]
    msg_triples = _extract_triples(content)
    if not msg_triples:
        return 0.0
    max_pe = 0.0
    for subj, rel, val in msg_triples:
        existing = kg[subj].get(rel, [])
        for old_val, _, _ in existing:
            if old_val != val and old_val and val:
                if rel in _FUNCTIONAL_RELATIONS:
                    max_pe = max(max_pe, 0.9)
                else:
                    max_pe = max(max_pe, 0.4)
    return _clamp(max_pe)


def variant_036(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Type constraint violation."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    type_checks = {
        "is_a": lambda v: v.strip().split()[0] if v else "",
    }
    kg = _CONV_STATE["kg"]
    msg_triples = _extract_triples(content)
    if not msg_triples:
        return 0.0
    max_pe = 0.0
    for subj, rel, val in msg_triples:
        if rel == "is_a" and subj in kg:
            old_types = kg[subj].get("is_a", [])
            for old_val, _, _ in old_types:
                if old_val and val and old_val != val:
                    max_pe = max(max_pe, 0.8)
    return _clamp(max_pe)


def variant_037(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Subgraph replacement score — BFS from changed node."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    kg = _CONV_STATE["kg"]
    msg_triples = _extract_triples(content)
    if not msg_triples:
        return 0.0
    max_pe = 0.0
    for subj, rel, val in msg_triples:
        existing = kg[subj].get(rel, [])
        for old_val, _, _ in existing:
            if old_val != val and old_val and val:
                visited = {subj}
                queue = [subj]
                while queue:
                    node = queue.pop(0)
                    for r, entries in kg.get(node, {}).items():
                        for v, _, _ in entries:
                            if v and v not in visited:
                                visited.add(v)
                                if v in kg:
                                    queue.append(v)
                total = sum(len(v) for rels in kg.values() for v in rels.values())
                pe = len(visited) / max(total, 1)
                max_pe = max(max_pe, pe)
    return _clamp(max_pe)


def variant_038(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Belief revision (AGM-inspired) — minimal retraction set."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    kg = _CONV_STATE["kg"]
    msg_triples = _extract_triples(content)
    if not msg_triples:
        return 0.0
    retracted = 0
    total = sum(len(entries) for rels in kg.values() for entries in rels.values())
    for subj, rel, val in msg_triples:
        existing = kg[subj].get(rel, [])
        for old_val, _, _ in existing:
            if old_val != val and old_val and val:
                retracted += 1
    if total == 0:
        return 0.0
    return _clamp(retracted / total)


def variant_039(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Graph embedding conflict — TransE energy of new edge."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    msg_triples = _extract_triples(content)
    if not msg_triples:
        return 0.0
    max_pe = 0.0
    for subj, rel, val in msg_triples:
        if not val:
            continue
        try:
            h = _embed_one(subj)
            r = _embed_one(rel.replace("_", " "))
            t = _embed_one(val)
            energy = float(np.linalg.norm(h + r - t))
            if subj in _CONV_STATE["kg"] and rel in _CONV_STATE["kg"][subj]:
                for old_val, _, _ in _CONV_STATE["kg"][subj][rel]:
                    t_old = _embed_one(old_val)
                    old_energy = float(np.linalg.norm(h + r - t_old))
                    if energy > old_energy * 1.2:
                        pe = min(1.0, (energy - old_energy) / (old_energy + 1e-10))
                        max_pe = max(max_pe, pe)
        except Exception:
            pass
    return _clamp(max_pe)


def variant_040(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """KG PE ensemble — max of v031, v033, v035."""
    s1 = variant_031(content, memory_contents, memory_embeddings)
    s2 = variant_033(content, memory_contents, memory_embeddings)
    s3 = variant_035(content, memory_contents, memory_embeddings)
    return max(s1, s2, s3)


# ============================================================================
# PARADIGM 5: Cross-Encoder Pairwise Scoring (041-050)
# ============================================================================


def variant_041(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Semantic similarity deficit — high entity overlap but low similarity = PE."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    idx, sim = _nearest_by_embedding(emb, memory_embeddings)
    entity_ov = _entity_overlap(content, memory_contents[idx])
    if entity_ov < 0.1:
        return 0.0
    expected_sim = 0.3 + 0.6 * entity_ov
    actual_sim = _cross_encoder_score(content, memory_contents[idx])
    deficit = max(0, expected_sim - actual_sim)
    return _clamp(deficit * 2.0)


def variant_042(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Cross-encoder with entity masking — compare predicate structure only."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    idx, sim = _nearest_by_embedding(emb, memory_embeddings)
    if sim < 0.2:
        return 0.0
    entities = _extract_entities(content) | _extract_entities(memory_contents[idx])
    masked_msg = content
    masked_mem = memory_contents[idx]
    for ent in entities:
        masked_msg = re.sub(re.escape(ent), "X", masked_msg, flags=re.IGNORECASE)
        masked_mem = re.sub(re.escape(ent), "X", masked_mem, flags=re.IGNORECASE)
    score = _cross_encoder_score(masked_msg, masked_mem)
    return _clamp(1.0 - score)


def variant_043(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Paraphrase anomaly — same topic but not paraphrasing = PE."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    idx, sim = _nearest_by_embedding(emb, memory_embeddings)
    entity_ov = _entity_overlap(content, memory_contents[idx])
    if entity_ov < 0.15:
        return 0.0
    para_score = _cross_encoder_score(content, memory_contents[idx])
    pe = entity_ov * (1.0 - para_score)
    return _clamp(pe)


def variant_044(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Cross-encoder embedding difference — pair vs self-pair distance."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    idx, sim = _nearest_by_embedding(emb, memory_embeddings)
    if sim < 0.2:
        return 0.0
    mem = memory_contents[idx]
    pair_emb = _embed_one(content + " [SEP] " + mem)
    self_emb = _embed_one(mem + " [SEP] " + mem)
    dist = 1.0 - _cosine_sim(pair_emb, self_emb)
    return _clamp(dist)


def variant_045(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Element-wise embedding difference L2 norm."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    idx, sim = _nearest_by_embedding(emb, memory_embeddings)
    entity_ov = _entity_overlap(content, memory_contents[idx])
    if entity_ov < 0.1:
        return 0.0
    diff = np.abs(emb - memory_embeddings[idx])
    l2 = float(np.linalg.norm(diff))
    pe = entity_ov * min(1.0, l2 / 2.0)
    return _clamp(pe)


def variant_046(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Negation injection — is message more similar to negated memory?"""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    idx, sim = _nearest_by_embedding(emb, memory_embeddings)
    if sim < 0.2:
        return 0.0
    mem = memory_contents[idx]
    negated = _negate_text(mem)
    sim_original = _cross_encoder_score(content, mem)
    sim_negated = _cross_encoder_score(content, negated)
    if sim_negated > sim_original:
        return _clamp((sim_negated - sim_original) * 2.0)
    return 0.0


def _negate_text(text: str) -> str:
    aux_verbs = ["is", "are", "was", "were", "has", "have", "had", "will", "would", "can", "could", "do", "does", "did"]
    lower = text.lower()
    for aux in aux_verbs:
        pattern = r'\b(' + aux + r')\b'
        if re.search(pattern, lower):
            return re.sub(pattern, aux + " not", text, count=1, flags=re.IGNORECASE)
    words = text.split()
    if len(words) >= 2:
        return words[0] + " does not " + " ".join(words[1:])
    return "not " + text


def variant_047(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Multiple cross-encoder scales — full + sentence + phrase level."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    idx, sim = _nearest_by_embedding(emb, memory_embeddings)
    if sim < 0.15:
        return 0.0
    mem = memory_contents[idx]
    full_score = _cross_encoder_score(content, mem)
    msg_sents = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
    mem_sents = [s.strip() for s in re.split(r'[.!?]+', mem) if s.strip()]
    sent_score = full_score
    for ms in msg_sents[:3]:
        for es in mem_sents[:3]:
            s = _cross_encoder_score(ms, es)
            sent_score = min(sent_score, s)
    pe = 1.0 - min(full_score, sent_score)
    entity_ov = _entity_overlap(content, mem)
    return _clamp(pe * entity_ov)


def variant_048(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Cross-encoder temporal — add temporal markers to input."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    idx, sim = _nearest_by_embedding(emb, memory_embeddings)
    if sim < 0.2:
        return 0.0
    mem = memory_contents[idx]
    temporal_input_a = f"[BEFORE] {mem}"
    temporal_input_b = f"[NOW] {content}"
    score = _cross_encoder_score(temporal_input_b, temporal_input_a)
    entity_ov = _entity_overlap(content, mem)
    pe = (1.0 - score) * entity_ov
    return _clamp(pe)


def variant_049(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Cross-encoder with calibrated scoring."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    idx, sim = _nearest_by_embedding(emb, memory_embeddings)
    if sim < 0.2:
        return 0.0
    raw = _cross_encoder_score(content, memory_contents[idx])
    entity_ov = _entity_overlap(content, memory_contents[idx])
    calibrated = (1.0 - raw) * entity_ov
    calibrated = 1.0 / (1.0 + math.exp(-10 * (calibrated - 0.3)))
    return _clamp(calibrated)


def variant_050(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Cross-encoder PE ensemble — average of v041, v043, v046."""
    s1 = variant_041(content, memory_contents, memory_embeddings)
    s2 = variant_043(content, memory_contents, memory_embeddings)
    s3 = variant_046(content, memory_contents, memory_embeddings)
    return _clamp((s1 + s2 + s3) / 3.0)


# ============================================================================
# PARADIGM 6: Conversation Flow / Discourse Analysis (051-060)
# ============================================================================

_CORRECTION_DISCOURSE = [
    "actually", "turns out", "i realized", "i was wrong", "apparently",
    "in fact", "to be honest", "correction", "i meant", "wait no",
    "scratch that", "nvm", "never mind", "not really",
]

_CONTRAST_DISCOURSE = [
    "but", "however", "instead", "on the other hand", "rather than",
    "although", "despite", "even though", "yet", "nevertheless",
]

_UPDATE_DISCOURSE = [
    "now", "anymore", "no longer", "not anymore", "finally",
    "just", "recently", "starting to", "stopped",
]


def variant_051(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Topic continuity break — embedding shift from entity centroid."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        return 0.0
    emb_msg = _embed_one(content)
    max_pe = 0.0
    for ent in msg_entities:
        embs = _CONV_STATE["entity_embeddings"].get(ent, [])
        if len(embs) < 3:
            continue
        centroid = np.mean(embs[-5:], axis=0)
        recent_stability = np.mean([_cosine_sim(embs[i], embs[i-1]) for i in range(max(1, len(embs)-3), len(embs))])
        dist = 1.0 - _cosine_sim(emb_msg, centroid)
        if recent_stability > 0.7:
            pe = dist * 2.0
            max_pe = max(max_pe, pe)
    return _clamp(max_pe)


def variant_052(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Discourse relation classification — correction/contrast markers."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    correction_score = sum(1 for m in _CORRECTION_DISCOURSE if m in lower) * 0.3
    contrast_score = sum(1 for m in _CONTRAST_DISCOURSE if m in lower) * 0.15
    has_entity_overlap = any(_entity_overlap(content, m) > 0.1 for m in memory_contents[-10:])
    if not has_entity_overlap and correction_score < 0.3:
        return 0.0
    return _clamp(correction_score + contrast_score)


def variant_053(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """RST scoring — concession and antithesis relations."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    rst_patterns = [
        (r"although\s+.+?,\s*(?:actually|really|in fact)\s+", 0.8),
        (r"despite\s+.+?,\s+", 0.5),
        (r"even though\s+.+?,\s+", 0.5),
        (r"(?:but|however)\s+(?:actually|really|in fact)\s+", 0.7),
        (r"(?:i know|you think)\s+.+?(?:but|however)\s+", 0.6),
    ]
    max_score = 0.0
    for pattern, weight in rst_patterns:
        if re.search(pattern, lower):
            max_score = max(max_score, weight)
    return _clamp(max_score)


def variant_054(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Stance shift detection — sentiment flip about same entity."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        return 0.0
    msg_sent = _sentiment_score(content)
    max_pe = 0.0
    for ent in msg_entities:
        sents = _CONV_STATE["entity_sentiments"].get(ent, [])
        if len(sents) < 1:
            continue
        last_sent = sents[-1]
        if (msg_sent > 0.2 and last_sent < -0.2) or (msg_sent < -0.2 and last_sent > 0.2):
            shift = abs(msg_sent - last_sent)
            max_pe = max(max_pe, min(1.0, shift))
    return _clamp(max_pe)


def variant_055(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Narrative expectation violation — embedding-based next-step prediction."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        return 0.0
    emb_msg = _embed_one(content)
    max_pe = 0.0
    for ent in msg_entities:
        embs = _CONV_STATE["entity_embeddings"].get(ent, [])
        if len(embs) < 3:
            continue
        predicted = 2 * embs[-1] - embs[-2]
        pred_sim = _cosine_sim(emb_msg, predicted)
        actual_sim = _cosine_sim(emb_msg, embs[-1])
        pe = max(0, actual_sim - pred_sim) * 2.0
        if pe < 0.1:
            pe = (1.0 - pred_sim) * 0.5
        max_pe = max(max_pe, pe)
    return _clamp(max_pe)


def variant_056(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Information structure change — given/new mismatch."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    definite_re = re.compile(r'\b(the|that|this|my|our|his|her|their)\s+(\w+)', re.IGNORECASE)
    msg_givens = {m.group(2).lower() for m in definite_re.finditer(content)}
    if not msg_givens:
        return 0.0
    max_pe = 0.0
    for mem in memory_contents[-10:]:
        mem_givens = {m.group(2).lower() for m in definite_re.finditer(mem)}
        shared_topics = msg_givens & mem_givens
        if shared_topics:
            overlap = _entity_overlap(content, mem)
            word_ov = _word_overlap(content, mem)
            if overlap > 0.1 and word_ov < 0.4:
                max_pe = max(max_pe, overlap * 0.7)
    return _clamp(max_pe)


def variant_057(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Speaker commitment tracking — contradicting prior commitments."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    commitments = _CONV_STATE["commitments"]
    if not commitments:
        return 0.0
    lower = content.lower()
    negation_markers = ["didn't", "don't", "won't", "can't", "not going to",
                        "changed my mind", "decided not", "cancelled", "canceled",
                        "backed out", "called off"]
    has_negation = any(m in lower for m in negation_markers)
    if not has_negation:
        return 0.0
    max_pe = 0.0
    for commit in commitments:
        overlap = _entity_overlap(content, commit)
        word_ov = _word_overlap(content, commit)
        if overlap > 0.1 or word_ov > 0.2:
            max_pe = max(max_pe, 0.7 + 0.3 * overlap)
    return _clamp(max_pe)


def variant_058(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Turn-level surprise — unexpected response type."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    is_announcement = any(m in lower for m in ["guess what", "big news", "i need to tell you", "omg so"])
    is_correction = any(m in lower for m in _CORRECTION_DISCOURSE[:5])
    is_sudden = any(m in lower for m in ["suddenly", "out of nowhere", "just like that", "boom"])
    if not memory_contents:
        return 0.0
    last_mem = memory_contents[-1].lower()
    is_question_prev = "?" in last_mem
    if is_announcement or is_correction or is_sudden:
        return _clamp(0.3 + 0.3 * (is_correction) + 0.2 * (is_sudden))
    return 0.0


def variant_059(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Emotional arc disruption — valence reversal about entity."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    msg_sent = _sentiment_score(content)
    if abs(msg_sent) < 0.1:
        return 0.0
    max_pe = 0.0
    for ent in msg_entities:
        sents = _CONV_STATE["entity_sentiments"].get(ent, [])
        if len(sents) < 2:
            continue
        trend = sum(s for s in sents[-3:]) / len(sents[-3:])
        if (trend > 0.1 and msg_sent < -0.1) or (trend < -0.1 and msg_sent > 0.1):
            reversal = abs(msg_sent - trend)
            max_pe = max(max_pe, min(1.0, reversal * 1.5))
    return _clamp(max_pe)


def variant_060(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Discourse PE ensemble — max of v052, v054, v057."""
    s1 = variant_052(content, memory_contents, memory_embeddings)
    s2 = variant_054(content, memory_contents, memory_embeddings)
    s3 = variant_057(content, memory_contents, memory_embeddings)
    return max(s1, s2, s3)


# ============================================================================
# PARADIGM 7: Forensic Stylometry (061-070)
# ============================================================================

_EPISTEMIC_MARKERS = [
    "actually", "turns out", "i realized", "i was wrong", "apparently",
    "in fact", "to be honest", "i forgot to mention", "come to find out",
    "believe it or not", "truth is", "real talk", "not gonna lie",
    "tbh", "ngl", "fun fact", "plot twist",
]

_HEDGE_WORDS = frozenset({
    "might", "maybe", "perhaps", "probably", "possibly", "considering",
    "thinking about", "not sure", "debating", "wondering", "could",
})

_CERTAINTY_WORDS = frozenset({
    "definitely", "absolutely", "confirmed", "official",
    "for sure", "100%", "decided", "committed", "done deal",
    "i did", "i'm doing", "it's done", "yes we are",
})

_TEMPORAL_ADVERBS = [
    "now", "anymore", "finally", "just", "recently", "no longer",
    "still", "already", "soon", "eventually", "never", "always",
    "currently", "formerly", "previously", "once",
]


def variant_061(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Epistemic marker density — linguistic fingerprints of belief revision."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    count = sum(1 for m in _EPISTEMIC_MARKERS if m in lower)
    word_count = max(len(lower.split()), 1)
    density = count / word_count
    has_overlap = any(_entity_overlap(content, m) > 0.05 for m in memory_contents[-10:])
    if not has_overlap and count < 2:
        return 0.0
    return _clamp(min(1.0, density * 10 + count * 0.2))


def variant_062(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Hedging shift — certainty level change about same entity."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    msg_hedge = sum(1 for h in _HEDGE_WORDS if h in lower)
    msg_certain = sum(1 for c in _CERTAINTY_WORDS if c in lower)
    if msg_hedge == 0 and msg_certain == 0:
        return 0.0
    max_pe = 0.0
    for mem in memory_contents[-10:]:
        overlap = _entity_overlap(content, mem)
        if overlap < 0.05:
            continue
        mem_lower = mem.lower()
        mem_hedge = sum(1 for h in _HEDGE_WORDS if h in mem_lower)
        mem_certain = sum(1 for c in _CERTAINTY_WORDS if c in mem_lower)
        if (msg_hedge > 0 and mem_certain > 0) or (msg_certain > 0 and mem_hedge > 0):
            max_pe = max(max_pe, 0.5 + 0.3 * overlap)
    return _clamp(max_pe)


def variant_063(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Formality shift — average word length and complexity change."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        return 0.0
    words = content.split()
    msg_formality = sum(len(w) for w in words) / max(len(words), 1)
    max_pe = 0.0
    for ent in msg_entities:
        msgs = _CONV_STATE["entity_messages"].get(ent, [])
        if len(msgs) < 2:
            continue
        prev_formalities = []
        for m in msgs[-5:]:
            w = m.split()
            prev_formalities.append(sum(len(x) for x in w) / max(len(w), 1))
        avg_form = sum(prev_formalities) / len(prev_formalities)
        shift = abs(msg_formality - avg_form) / max(avg_form, 1)
        max_pe = max(max_pe, min(1.0, shift * 2))
    return _clamp(max_pe)


def variant_064(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Sentence structure anomaly — punchiness for announcements."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        lower = content.lower()
        if any(w in lower for w in ["i got", "we broke", "i did", "i made"]):
            msg_len = len(content.split())
            if msg_len <= 5:
                return _clamp(0.5)
        return 0.0
    msg_len = len(content.split())
    max_pe = 0.0
    for ent in msg_entities:
        baseline = _CONV_STATE["entity_style_baselines"].get(ent)
        if not baseline or baseline["msg_count"] < 2:
            continue
        avg_len = baseline["avg_len"]
        if avg_len > 0:
            ratio = msg_len / avg_len
            if ratio < 0.4 or ratio > 2.5:
                max_pe = max(max_pe, min(1.0, abs(1 - ratio) * 0.5))
    return _clamp(max_pe)


def variant_065(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Emphasis marker surge — CAPS, !, intensifiers above baseline."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    caps = sum(1 for c in content if c.isupper())
    excl = content.count("!")
    intensifiers = sum(1 for w in ["really", "literally", "seriously", "absolutely", "completely", "totally"]
                       if w in content.lower())
    emphasis = caps * 0.1 + excl * 0.3 + intensifiers * 0.3
    if emphasis < 0.3:
        return 0.0
    has_overlap = any(_entity_overlap(content, m) > 0.05 for m in memory_contents[-10:])
    if has_overlap:
        return _clamp(min(1.0, emphasis * 0.5))
    return _clamp(min(0.5, emphasis * 0.3))


def variant_066(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Disfluency markers — self-repairs signal belief revision."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    repairs = ["i mean", "wait", "no i meant", "sorry i said", "no wait",
               "actually no", "hold on", "let me rephrase", "what i meant was"]
    count = sum(1 for r in repairs if r in lower)
    if count == 0:
        return 0.0
    has_overlap = any(_entity_overlap(content, m) > 0.05 for m in memory_contents[-10:])
    return _clamp(count * 0.3 + 0.2 * has_overlap)


def variant_067(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Temporal adverb density — state transition signals."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    count = sum(1 for t in _TEMPORAL_ADVERBS if t in lower)
    word_count = max(len(lower.split()), 1)
    density = count / word_count
    has_overlap = any(_entity_overlap(content, m) > 0.05 for m in memory_contents[-10:])
    if not has_overlap and count < 2:
        return 0.0
    return _clamp(density * 8 + count * 0.15)


def variant_068(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Pronoun shift — we→I or I→we signals relationship change."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        return 0.0
    lower = content.lower()
    msg_pronoun = None
    if " we " in lower or lower.startswith("we ") or " we'" in lower:
        msg_pronoun = "we"
    elif " i " in lower or lower.startswith("i ") or " i'" in lower:
        msg_pronoun = "i"
    if not msg_pronoun:
        return 0.0
    max_pe = 0.0
    for ent in msg_entities:
        history = _CONV_STATE["entity_pronoun_history"].get(ent, [])
        if len(history) < 2:
            continue
        recent = history[-3:]
        if all(p != msg_pronoun for p in recent):
            max_pe = max(max_pe, 0.6)
    return _clamp(max_pe)


def variant_069(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Lexical novelty within entity context — new vocabulary about familiar entity."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        return 0.0
    msg_words = _content_words(content)
    if not msg_words:
        return 0.0
    max_pe = 0.0
    for ent in msg_entities:
        known_words = _CONV_STATE["entity_word_sets"].get(ent, set())
        if len(known_words) < 5:
            continue
        novel = msg_words - known_words - _STOPWORDS
        novelty_frac = len(novel) / max(len(msg_words), 1)
        max_pe = max(max_pe, min(1.0, novelty_frac * 1.5))
    return _clamp(max_pe)


def variant_070(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Stylometry PE ensemble — max of v061, v062, v065."""
    s1 = variant_061(content, memory_contents, memory_embeddings)
    s2 = variant_062(content, memory_contents, memory_embeddings)
    s3 = variant_065(content, memory_contents, memory_embeddings)
    return max(s1, s2, s3)


# ============================================================================
# PARADIGM 8: Bioinformatic Sequence Alignment (071-080)
# ============================================================================


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _token_type(token: str) -> str:
    if token[0].isupper() and token not in _COMMON_PROPER:
        return "proper"
    if re.match(r"^\$?[\d,.]+[KMBkmb%]?$", token):
        return "number"
    if token in _STOPWORDS:
        return "stop"
    return "content"


def variant_071(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Token-level edit distance with typed operations — substitutions = PE."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    idx, overlap = _nearest_by_entity(content, memory_contents)
    if overlap < 0.1:
        return 0.0
    mem = memory_contents[idx]
    msg_tokens = _tokenize(content)
    mem_tokens = _tokenize(mem)
    sm = difflib.SequenceMatcher(None, mem_tokens, msg_tokens)
    subs = 0
    total = max(len(msg_tokens), len(mem_tokens), 1)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "replace":
            subs += max(i2 - i1, j2 - j1)
    return _clamp(subs / total)


def variant_072(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Weighted edit distance — proper noun substitution > function word."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    idx, overlap = _nearest_by_entity(content, memory_contents)
    if overlap < 0.1:
        return 0.0
    mem = memory_contents[idx]
    msg_tokens = content.split()
    mem_tokens = mem.split()
    sm = difflib.SequenceMatcher(None, [t.lower() for t in mem_tokens], [t.lower() for t in msg_tokens])
    weighted_subs = 0.0
    total_weight = 0.0
    weights = {"proper": 1.0, "number": 0.8, "content": 0.5, "stop": 0.1}
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "replace":
            for k in range(i1, i2):
                if k < len(mem_tokens):
                    tt = _token_type(mem_tokens[k])
                    weighted_subs += weights.get(tt, 0.3)
        for k in range(i1, i2):
            if k < len(mem_tokens):
                total_weight += weights.get(_token_type(mem_tokens[k]), 0.3)
        if op == "equal":
            for k in range(i1, i2):
                if k < len(mem_tokens):
                    total_weight += weights.get(_token_type(mem_tokens[k]), 0.3)
    if total_weight < 0.1:
        return 0.0
    return _clamp(weighted_subs / total_weight)


def variant_073(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Smith-Waterman local alignment score deficit."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    idx, overlap = _nearest_by_entity(content, memory_contents)
    if overlap < 0.1:
        return 0.0
    mem = memory_contents[idx]
    msg_tokens = _tokenize(content)
    mem_tokens = _tokenize(mem)
    n, m_len = len(mem_tokens), len(msg_tokens)
    if n == 0 or m_len == 0:
        return 0.0
    H = [[0] * (m_len + 1) for _ in range(n + 1)]
    max_score = 0
    match_score = 2
    mismatch_penalty = -1
    gap_penalty = -1
    for i in range(1, n + 1):
        for j in range(1, m_len + 1):
            match = match_score if mem_tokens[i-1] == msg_tokens[j-1] else mismatch_penalty
            H[i][j] = max(0, H[i-1][j-1] + match, H[i-1][j] + gap_penalty, H[i][j-1] + gap_penalty)
            max_score = max(max_score, H[i][j])
    max_possible = min(n, m_len) * match_score
    if max_possible == 0:
        return 0.0
    alignment_quality = max_score / max_possible
    pe = (1.0 - alignment_quality) * overlap
    return _clamp(pe)


def variant_074(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """LCS inversion — tokens not in LCS at expected positions = substitutions."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    idx, overlap = _nearest_by_entity(content, memory_contents)
    if overlap < 0.1:
        return 0.0
    mem = memory_contents[idx]
    msg_tokens = _tokenize(content)
    mem_tokens = _tokenize(mem)
    sm = difflib.SequenceMatcher(None, mem_tokens, msg_tokens)
    matching = sm.get_matching_blocks()
    total_matching = sum(block.size for block in matching)
    total = max(len(msg_tokens), len(mem_tokens), 1)
    non_lcs_ratio = (total - total_matching) / total
    return _clamp(non_lcs_ratio * overlap * 2)


def variant_075(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """N-gram substitution — bigrams with 1 token differing."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    idx, overlap = _nearest_by_entity(content, memory_contents)
    if overlap < 0.1:
        return 0.0
    mem = memory_contents[idx]
    msg_tokens = _tokenize(content)
    mem_tokens = _tokenize(mem)
    if len(msg_tokens) < 2 or len(mem_tokens) < 2:
        return 0.0
    msg_bigrams = [(msg_tokens[i], msg_tokens[i+1]) for i in range(len(msg_tokens)-1)]
    mem_bigrams = [(mem_tokens[i], mem_tokens[i+1]) for i in range(len(mem_tokens)-1)]
    subs = 0
    for mb in mem_bigrams:
        for ab in msg_bigrams:
            shared = sum(1 for a, b in zip(mb, ab) if a == b)
            if shared == 1:
                subs += 1
                break
    return _clamp(subs / max(len(mem_bigrams), 1) * overlap * 2)


def variant_076(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Semantic edit distance — embed-aligned token substitutions."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    idx, overlap = _nearest_by_entity(content, memory_contents)
    if overlap < 0.1:
        return 0.0
    mem = memory_contents[idx]
    msg_words = [w for w in content.split() if w.lower() not in _STOPWORDS and len(w) > 1]
    mem_words = [w for w in mem.split() if w.lower() not in _STOPWORDS and len(w) > 1]
    if not msg_words or not mem_words:
        return 0.0
    msg_embs = _embed(msg_words)
    mem_embs = _embed(mem_words)
    misaligned = 0
    for i, me in enumerate(mem_embs):
        sims = np.array([_cosine_sim(me, ae) for ae in msg_embs])
        best_sim = float(np.max(sims)) if len(sims) > 0 else 0
        if best_sim < 0.5:
            misaligned += 1
    return _clamp(misaligned / max(len(mem_words), 1) * overlap)


def variant_077(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Template alignment — same template, different fillers = PE."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    def templatize(text):
        t = text
        for ent in _extract_entities(text):
            t = t.replace(ent, "[ENTITY]")
        for num in _extract_numbers(text):
            t = t.replace(num, "[NUMBER]")
        for loc in _extract_locations(text):
            t = re.sub(re.escape(loc), "[PLACE]", t, flags=re.IGNORECASE)
        return t

    msg_template = templatize(content)
    max_pe = 0.0
    for mem in memory_contents:
        mem_template = templatize(mem)
        sm = difflib.SequenceMatcher(None, mem_template.lower().split(), msg_template.lower().split())
        ratio = sm.ratio()
        if ratio > 0.6:
            msg_fillers = set(re.findall(r'\[(\w+)\]', msg_template))
            mem_fillers_text = [w for w in mem.split() if w not in mem_template.split()]
            msg_fillers_text = [w for w in content.split() if w not in msg_template.split()]
            if msg_fillers_text and mem_fillers_text:
                filler_overlap = _jaccard(set(w.lower() for w in msg_fillers_text),
                                         set(w.lower() for w in mem_fillers_text))
                if filler_overlap < 0.5:
                    max_pe = max(max_pe, ratio * (1 - filler_overlap))
    return _clamp(max_pe)


def variant_078(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Diff-match-patch — modification ratio via Myers diff."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    idx, overlap = _nearest_by_entity(content, memory_contents)
    if overlap < 0.1:
        return 0.0
    mem = memory_contents[idx]
    sm = difflib.SequenceMatcher(None, mem, content)
    mod_chars = 0
    total_chars = max(len(content), len(mem), 1)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "replace":
            mod_chars += max(i2 - i1, j2 - j1)
    return _clamp(mod_chars / total_chars * overlap * 3)


def variant_079(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Alignment with gap penalties — mid-alignment substitutions weighted higher."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    idx, overlap = _nearest_by_entity(content, memory_contents)
    if overlap < 0.1:
        return 0.0
    mem = memory_contents[idx]
    msg_tokens = _tokenize(content)
    mem_tokens = _tokenize(mem)
    sm = difflib.SequenceMatcher(None, mem_tokens, msg_tokens)
    opcodes = sm.get_opcodes()
    total_ops = len(opcodes)
    weighted_subs = 0.0
    for i, (op, i1, i2, j1, j2) in enumerate(opcodes):
        if op == "replace":
            position_weight = 1.0 - abs(2.0 * i / max(total_ops, 1) - 1.0)
            weighted_subs += (max(i2 - i1, j2 - j1)) * (0.5 + position_weight)
    total = max(len(msg_tokens), len(mem_tokens), 1)
    return _clamp(weighted_subs / total * overlap * 2)


def variant_080(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Sequence alignment PE ensemble — max of v072, v077, v078."""
    s1 = variant_072(content, memory_contents, memory_embeddings)
    s2 = variant_077(content, memory_contents, memory_embeddings)
    s3 = variant_078(content, memory_contents, memory_embeddings)
    return max(s1, s2, s3)


# ============================================================================
# PARADIGM 9: Change-Point Detection (081-090)
# ============================================================================


def variant_081(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """CUSUM on entity embeddings — cumulative deviation from running mean."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        return 0.0
    emb_msg = _embed_one(content)
    max_pe = 0.0
    for ent in msg_entities:
        embs = _CONV_STATE["entity_embeddings"].get(ent, [])
        if len(embs) < 3:
            continue
        running_mean = np.mean(embs, axis=0)
        deviation = 1.0 - _cosine_sim(emb_msg, running_mean)
        deviations = [1.0 - _cosine_sim(e, running_mean) for e in embs]
        avg_dev = sum(deviations) / len(deviations) if deviations else 0.1
        cusum = max(0, deviation - avg_dev - 0.05)
        max_pe = max(max_pe, min(1.0, cusum * 5))
    return _clamp(max_pe)


def variant_082(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """CUSUM on sentiment — emotional state change detection."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    msg_sent = _sentiment_score(content)
    max_pe = 0.0
    for ent in msg_entities:
        sents = _CONV_STATE["entity_sentiments"].get(ent, [])
        if len(sents) < 3:
            continue
        running_mean = sum(sents) / len(sents)
        deviation = abs(msg_sent - running_mean)
        avg_dev = sum(abs(s - running_mean) for s in sents) / len(sents)
        cusum = max(0, deviation - avg_dev - 0.1)
        max_pe = max(max_pe, min(1.0, cusum * 3))
    return _clamp(max_pe)


def variant_083(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Bayesian Online Change-Point Detection (BOCPD)."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        return 0.0
    emb_msg = _embed_one(content)
    max_pe = 0.0
    hazard_rate = 1.0 / 50.0
    for ent in msg_entities:
        embs = _CONV_STATE["entity_embeddings"].get(ent, [])
        if len(embs) < 3:
            continue
        sims = [_cosine_sim(emb_msg, e) for e in embs]
        avg_sim = sum(sims) / len(sims)
        std_sim = max(0.01, (sum((s - avg_sim) ** 2 for s in sims) / len(sims)) ** 0.5)
        surprise = max(0, (1 - _cosine_sim(emb_msg, embs[-1]) - avg_sim) / std_sim)
        cp_prob = hazard_rate * max(0, 1 - math.exp(-surprise))
        n = len(embs)
        cp_prob *= min(1.0, n / 5.0)
        max_pe = max(max_pe, min(1.0, cp_prob * 5))
    return _clamp(max_pe)


def variant_084(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Moving window distribution test — recent vs historical difference."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        return 0.0
    emb_msg = _embed_one(content)
    max_pe = 0.0
    for ent in msg_entities:
        embs = _CONV_STATE["entity_embeddings"].get(ent, [])
        if len(embs) < 4:
            continue
        recent = embs[-3:]
        historical = embs[:-3] if len(embs) > 3 else embs[:1]
        recent_mean = np.mean(recent, axis=0)
        hist_mean = np.mean(historical, axis=0)
        msg_to_recent = _cosine_sim(emb_msg, recent_mean)
        msg_to_hist = _cosine_sim(emb_msg, hist_mean)
        dist_shift = abs(msg_to_recent - msg_to_hist)
        max_pe = max(max_pe, min(1.0, dist_shift * 3))
    return _clamp(max_pe)


def variant_085(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """EMA deviation — distance from exponential moving average."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        return 0.0
    emb_msg = _embed_one(content)
    max_pe = 0.0
    for ent in msg_entities:
        ema = _CONV_STATE["entity_ema"].get(ent)
        if ema is None:
            continue
        if len(_CONV_STATE["entity_embeddings"].get(ent, [])) < 3:
            continue
        deviation = 1.0 - _cosine_sim(emb_msg, ema)
        max_pe = max(max_pe, min(1.0, deviation * 3))
    return _clamp(max_pe)


def variant_086(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Variance spike detection — increasing variance = inconsistent story."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        return 0.0
    emb_msg = _embed_one(content)
    max_pe = 0.0
    for ent in msg_entities:
        embs = _CONV_STATE["entity_embeddings"].get(ent, [])
        if len(embs) < 4:
            continue
        centroid = np.mean(embs, axis=0)
        old_var = np.mean([(1 - _cosine_sim(e, centroid)) ** 2 for e in embs])
        all_embs = embs + [emb_msg]
        new_centroid = np.mean(all_embs, axis=0)
        new_var = np.mean([(1 - _cosine_sim(e, new_centroid)) ** 2 for e in all_embs])
        if new_var > old_var * 1.5 and old_var > 0:
            pe = min(1.0, (new_var - old_var) / (old_var + 1e-10))
            max_pe = max(max_pe, pe)
    return _clamp(max_pe)


def variant_087(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Topic cluster shift — message falls in different cluster than recent."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        return 0.0
    emb_msg = _embed_one(content)
    max_pe = 0.0
    for ent in msg_entities:
        embs = _CONV_STATE["entity_embeddings"].get(ent, [])
        if len(embs) < 4:
            continue
        from sklearn.cluster import KMeans
        n_clusters = min(3, len(embs))
        mat = np.array(embs)
        km = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
        labels = km.fit_predict(mat)
        recent_cluster = labels[-1]
        msg_cluster = km.predict(emb_msg.reshape(1, -1))[0]
        if msg_cluster != recent_cluster:
            recent_labels = labels[-min(3, len(labels)):]
            if all(l == recent_cluster for l in recent_labels):
                max_pe = max(max_pe, 0.7)
            else:
                max_pe = max(max_pe, 0.4)
    return _clamp(max_pe)


def variant_088(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Autoregressive prediction error — AR(1) residual."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        return 0.0
    emb_msg = _embed_one(content)
    max_pe = 0.0
    for ent in msg_entities:
        embs = _CONV_STATE["entity_embeddings"].get(ent, [])
        if len(embs) < 3:
            continue
        alpha = 0.7
        predicted = alpha * embs[-1] + (1 - alpha) * embs[-2]
        residual = np.linalg.norm(emb_msg - predicted)
        baseline_residuals = []
        for i in range(2, len(embs)):
            pred = alpha * embs[i-1] + (1 - alpha) * embs[i-2]
            baseline_residuals.append(np.linalg.norm(embs[i] - pred))
        if baseline_residuals:
            sigma = max(0.01, np.std(baseline_residuals))
            z_score = (residual - np.mean(baseline_residuals)) / sigma
            pe = min(1.0, max(0, z_score) / 3.0)
            max_pe = max(max_pe, pe)
    return _clamp(max_pe)


def variant_089(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Entropy rate change via compression."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        return 0.0
    max_pe = 0.0
    for ent in msg_entities:
        msgs = _CONV_STATE["entity_messages"].get(ent, [])
        if len(msgs) < 3:
            continue
        old_text = " ".join(msgs)
        new_text = old_text + " " + content
        old_compressed = len(gzip.compress(old_text.encode()))
        new_compressed = len(gzip.compress(new_text.encode()))
        added_raw = len(content.encode())
        if added_raw > 0:
            compression_ratio = (new_compressed - old_compressed) / added_raw
            if compression_ratio > 0.8:
                max_pe = max(max_pe, min(1.0, (compression_ratio - 0.5) * 2))
    return _clamp(max_pe)


def variant_090(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Change-point PE ensemble — max of v081, v083, v085."""
    s1 = variant_081(content, memory_contents, memory_embeddings)
    s2 = variant_083(content, memory_contents, memory_embeddings)
    s3 = variant_085(content, memory_contents, memory_embeddings)
    return max(s1, s2, s3)


# ============================================================================
# PARADIGM 10: Masked Slot-Filling Surprise (091-100)
# ============================================================================


def _find_key_token(text: str) -> tuple[str, int]:
    """Find the most informative token to mask — returns (token, word_index)."""
    entities = _extract_entities(text)
    if entities:
        ent = list(entities)[0]
        words = text.split()
        for i, w in enumerate(words):
            if w.lower() == ent.lower() or ent.lower() in w.lower():
                return w, i
    numbers = _extract_numbers(text)
    if numbers:
        words = text.split()
        for i, w in enumerate(words):
            if any(n in w for n in numbers):
                return w, i
    locations = _extract_locations(text)
    if locations:
        loc = list(locations)[0]
        words = text.split()
        for i, w in enumerate(words):
            if w.lower() == loc.lower():
                return w, i
    return "", -1


def _mlm_surprise(context_with_mask: str, actual_token: str) -> float:
    """Compute surprise = 1 - P(actual_token | masked context)."""
    if not _HAS_MLM:
        return _mlm_fallback_surprise(context_with_mask, actual_token)
    predictions = _mlm_predict_mask(context_with_mask)
    if not predictions:
        return 0.3
    actual_lower = actual_token.lower()
    prob = predictions.get(actual_lower, 0.0)
    for token, p in predictions.items():
        if token.lower() == actual_lower:
            prob = max(prob, p)
    return _clamp(1.0 - prob)


def _mlm_fallback_surprise(context_with_mask: str, actual_token: str) -> float:
    """Fallback: use embedding distance as surprise proxy."""
    context_clean = context_with_mask.replace("[MASK]", "something")
    emb_context = _embed_one(context_clean)
    emb_actual = _embed_one(actual_token)
    sim = _cosine_sim(emb_context, emb_actual)
    return _clamp(1.0 - (sim + 1) / 2)


def variant_091(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Single-slot masking — mask key token in memory, surprise if message disagrees."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    idx, sim = _nearest_by_embedding(emb, memory_embeddings)
    if sim < 0.2:
        return 0.0
    mem = memory_contents[idx]
    if _entity_overlap(content, mem) < 0.1:
        return 0.0
    key_token, key_idx = _find_key_token(mem)
    if not key_token:
        return 0.0
    words = mem.split()
    masked = list(words)
    if key_idx < len(masked):
        masked[key_idx] = "[MASK]"
    masked_text = " ".join(masked)
    msg_key, _ = _find_key_token(content)
    if not msg_key:
        return 0.0
    surprise = _mlm_surprise(masked_text, msg_key)
    return _clamp(surprise)


def variant_092(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Multi-slot masking — mask ALL informative tokens."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    idx, sim = _nearest_by_embedding(emb, memory_embeddings)
    if sim < 0.2:
        return 0.0
    mem = memory_contents[idx]
    if _entity_overlap(content, mem) < 0.1:
        return 0.0
    mem_entities = _extract_entities(mem)
    mem_numbers = set(_extract_numbers(mem))
    mem_locs = _extract_locations(mem)
    informative = mem_entities | mem_numbers | mem_locs
    if not informative:
        return 0.0
    msg_entities = _extract_entities(content)
    msg_numbers = set(_extract_numbers(content))
    msg_locs = _extract_locations(content)
    msg_info = msg_entities | msg_numbers | msg_locs
    if not msg_info:
        return 0.0
    surprises = []
    for info_token in list(informative)[:3]:
        words = mem.split()
        masked = []
        for w in words:
            if w.lower() == info_token.lower() or info_token.lower() in w.lower():
                masked.append("[MASK]")
            else:
                masked.append(w)
        masked_text = " ".join(masked)
        for msg_token in msg_info:
            s = _mlm_surprise(masked_text, msg_token)
            surprises.append(s)
    if surprises:
        return _clamp(sum(surprises) / len(surprises))
    return 0.0


def variant_093(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Contrastive slot filling — P(memory_filler) vs P(message_filler)."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    idx, sim = _nearest_by_embedding(emb, memory_embeddings)
    if sim < 0.2:
        return 0.0
    mem = memory_contents[idx]
    if _entity_overlap(content, mem) < 0.1:
        return 0.0
    mem_key, mem_idx = _find_key_token(mem)
    msg_key, _ = _find_key_token(content)
    if not mem_key or not msg_key or mem_key.lower() == msg_key.lower():
        return 0.0
    words = mem.split()
    masked = list(words)
    if mem_idx < len(masked):
        masked[mem_idx] = "[MASK]"
    masked_text = " ".join(masked)
    surprise_msg = _mlm_surprise(masked_text, msg_key)
    surprise_mem = _mlm_surprise(masked_text, mem_key)
    pe = max(0, surprise_msg - surprise_mem)
    return _clamp(pe * 2)


def variant_094(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Bidirectional slot filling — both directions confident but different."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    idx, sim = _nearest_by_embedding(emb, memory_embeddings)
    if sim < 0.2:
        return 0.0
    mem = memory_contents[idx]
    if _entity_overlap(content, mem) < 0.1:
        return 0.0
    mem_key, mem_idx = _find_key_token(mem)
    msg_key, msg_idx = _find_key_token(content)
    if not mem_key or not msg_key or mem_key.lower() == msg_key.lower():
        return 0.0
    words_mem = mem.split()
    masked_mem = list(words_mem)
    if mem_idx < len(masked_mem):
        masked_mem[mem_idx] = "[MASK]"
    p_mem = 1.0 - _mlm_surprise(" ".join(masked_mem), mem_key)
    words_msg = content.split()
    masked_msg = list(words_msg)
    if msg_idx < len(masked_msg):
        masked_msg[msg_idx] = "[MASK]"
    p_msg = 1.0 - _mlm_surprise(" ".join(masked_msg), msg_key)
    if p_mem > 0.3 and p_msg > 0.3:
        return _clamp(min(p_mem, p_msg))
    return 0.0


def variant_095(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Relation-specific masking — mask based on detected relation type."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    idx, sim = _nearest_by_embedding(emb, memory_embeddings)
    if sim < 0.2:
        return 0.0
    mem = memory_contents[idx]
    if _entity_overlap(content, mem) < 0.1:
        return 0.0
    mem_triples = _extract_triples(mem)
    msg_triples = _extract_triples(content)
    if not mem_triples or not msg_triples:
        return variant_091(content, memory_contents, memory_embeddings)
    max_pe = 0.0
    for ms, mr, mv in msg_triples:
        for es, er, ev in mem_triples:
            if mr == er or (mr == "quit" and er == "works_at"):
                if mv and ev and mv != ev:
                    words = mem.split()
                    masked = []
                    for w in words:
                        if w.lower() in ev.lower().split() or ev.lower() in w.lower():
                            masked.append("[MASK]")
                        else:
                            masked.append(w)
                    masked_text = " ".join(masked)
                    surprise = _mlm_surprise(masked_text, mv.split()[0] if mv else "")
                    max_pe = max(max_pe, surprise)
    return _clamp(max_pe)


def variant_096(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Memory-conditioned prediction — concatenate memory with masked message."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    idx, sim = _nearest_by_embedding(emb, memory_embeddings)
    if sim < 0.2:
        return 0.0
    mem = memory_contents[idx]
    if _entity_overlap(content, mem) < 0.1:
        return 0.0
    msg_key, msg_idx = _find_key_token(content)
    if not msg_key:
        return 0.0
    words = content.split()
    masked = list(words)
    if msg_idx < len(masked):
        masked[msg_idx] = "[MASK]"
    conditioned = mem + " [SEP] " + " ".join(masked)
    surprise = _mlm_surprise(conditioned, msg_key)
    return _clamp(surprise)


def variant_097(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Top-k prediction divergence — is actual token in top-10?"""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    idx, sim = _nearest_by_embedding(emb, memory_embeddings)
    if sim < 0.2:
        return 0.0
    mem = memory_contents[idx]
    if _entity_overlap(content, mem) < 0.1:
        return 0.0
    mem_key, mem_idx = _find_key_token(mem)
    msg_key, _ = _find_key_token(content)
    if not mem_key or not msg_key:
        return 0.0
    words = mem.split()
    masked = list(words)
    if mem_idx < len(masked):
        masked[mem_idx] = "[MASK]"
    masked_text = " ".join(masked)
    if _HAS_MLM:
        predictions = _mlm_predict_mask(masked_text)
        if not predictions:
            return 0.3
        top_tokens = list(predictions.keys())
        msg_lower = msg_key.lower()
        if msg_lower in [t.lower() for t in top_tokens[:1]]:
            return 0.0
        elif msg_lower in [t.lower() for t in top_tokens[:5]]:
            return 0.3
        elif msg_lower in [t.lower() for t in top_tokens]:
            return 0.5
        else:
            return 0.8
    else:
        return _mlm_fallback_surprise(masked_text, msg_key)


def variant_098(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Entropy-weighted surprise — high confidence + wrong = high PE."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    idx, sim = _nearest_by_embedding(emb, memory_embeddings)
    if sim < 0.2:
        return 0.0
    mem = memory_contents[idx]
    if _entity_overlap(content, mem) < 0.1:
        return 0.0
    mem_key, mem_idx = _find_key_token(mem)
    msg_key, _ = _find_key_token(content)
    if not mem_key or not msg_key:
        return 0.0
    words = mem.split()
    masked = list(words)
    if mem_idx < len(masked):
        masked[mem_idx] = "[MASK]"
    masked_text = " ".join(masked)
    if _HAS_MLM:
        predictions = _mlm_predict_mask(masked_text)
        if not predictions:
            return 0.3
        probs = list(predictions.values())
        entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
        max_entropy = math.log(len(probs) + 1e-10)
        confidence = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0)
        surprise = _mlm_surprise(masked_text, msg_key)
        return _clamp(confidence * surprise)
    else:
        return _mlm_fallback_surprise(masked_text, msg_key)


def variant_099(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Template-conditioned masking — template match + slot comparison."""
    if _is_noise(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = _embed_one(content)
    idx, sim = _nearest_by_embedding(emb, memory_embeddings)
    if sim < 0.2:
        return 0.0
    mem = memory_contents[idx]
    if _entity_overlap(content, mem) < 0.1:
        return 0.0

    def templatize_with_slots(text):
        template = text
        slots = []
        for ent in _extract_entities(text):
            template = template.replace(ent, "[MASK]")
            slots.append(ent)
        for num in _extract_numbers(text):
            template = template.replace(num, "[MASK]")
            slots.append(num)
        return template, slots

    mem_template, mem_slots = templatize_with_slots(mem)
    msg_template, msg_slots = templatize_with_slots(content)
    template_sim = difflib.SequenceMatcher(None, mem_template.lower(), msg_template.lower()).ratio()
    if template_sim < 0.5 or not mem_slots or not msg_slots:
        return 0.0
    slot_match = _jaccard(set(s.lower() for s in mem_slots), set(s.lower() for s in msg_slots))
    pe = template_sim * (1.0 - slot_match)
    return _clamp(pe)


def variant_100(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Masked slot-filling PE ensemble — average of v091, v096, v098."""
    s1 = variant_091(content, memory_contents, memory_embeddings)
    s2 = variant_096(content, memory_contents, memory_embeddings)
    s3 = variant_098(content, memory_contents, memory_embeddings)
    return _clamp((s1 + s2 + s3) / 3.0)


# ============================================================================
# ALL_VARIANTS registry
# ============================================================================

ALL_VARIANTS: dict[str, Callable] = {}
for _i in range(1, 101):
    _name = f"variant_{_i:03d}"
    _fn = globals().get(_name)
    if _fn is not None:
        ALL_VARIANTS[_name] = _fn

print(f"PE v2 sweep: {len(ALL_VARIANTS)} variants registered")
