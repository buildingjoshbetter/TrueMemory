#!/usr/bin/env python3
"""
100-variant prediction error sweep for the encoding gate.

Every variant scores PREDICTION ERROR — does this message CONTRADICT, UPDATE,
or RESOLVE something already stored in memory? PE is context-dependent: the
same message can score HIGH or LOW depending on what's in memory.

PE ≠ novelty (is this new?) and PE ≠ salience (is this worth remembering?).
PE = does this CHANGE what I already believe?

Signature: variant_NNN(content, memory_contents, memory_embeddings=None) -> float [0,1]
"""

from __future__ import annotations

import gzip
import math
import re
from collections import Counter, defaultdict
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

_STATE_CHANGE_VERBS = frozenset({
    "switched", "changed", "moved", "converted", "upgraded", "downgraded",
    "replaced", "broke", "fixed", "traded", "swapped", "transferred",
    "transitioned", "shifted", "pivoted", "migrated", "adopted",
})

_UPDATE_VERBS = frozenset({
    "started", "stopped", "quit", "joined", "left", "enrolled", "dropped",
    "began", "ended", "finished", "completed", "got", "lost", "bought",
    "sold", "graduated", "hired", "fired", "promoted", "resigned",
    "launched", "shipped", "published", "submitted", "accepted", "rejected",
    "married", "divorced", "engaged", "proposed",
})

_CORRECTION_MARKERS = frozenset({
    "actually", "correction", "i was wrong", "turns out", "i meant",
    "sorry i said", "i lied", "my bad", "wait no", "nvm",
    "scratch that", "forget what i said", "not really",
})

_NEGATION_PHRASES = frozenset({
    "no longer", "not anymore", "stopped", "quit", "gave up",
    "don't anymore", "doesn't anymore", "isn't anymore",
    "not any more", "never again", "done with",
})

_HEDGE_WORDS = frozenset({
    "might", "maybe", "perhaps", "probably", "possibly", "considering",
    "thinking about", "not sure", "debating", "wondering", "could",
    "might be", "i guess", "i think", "kind of", "sort of",
})

_CERTAINTY_WORDS = frozenset({
    "definitely", "absolutely", "confirmed", "it's official",
    "for sure", "100%", "decided", "committed", "done deal",
    "i did", "i'm doing", "it's done", "yes we are",
})

_RELATIONSHIP_TERMS = frozenset({
    "dating", "engaged", "married", "divorced", "broke up", "breaking up",
    "single", "together", "separated", "seeing someone", "in a relationship",
    "boyfriend", "girlfriend", "wife", "husband", "partner", "fiancé",
    "fiancée", "ex", "widow", "widower",
})

_POSITIVE_WORDS = frozenset({
    "love", "great", "amazing", "awesome", "wonderful", "fantastic",
    "excellent", "happy", "enjoy", "best", "perfect", "good",
    "thrilled", "excited", "glad", "pleased", "delighted",
})

_NEGATIVE_WORDS = frozenset({
    "hate", "terrible", "awful", "horrible", "worst", "bad",
    "disgusting", "miserable", "unhappy", "disappointed", "angry",
    "frustrated", "annoyed", "furious", "devastated",
})

_LOCATION_RE = re.compile(
    r"\b(?:in|to|from|at|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
)

_NUMBER_RE = re.compile(
    r"\$[\d,.]+[KMBkmb]?|\d+\.?\d*\s*%|\d{1,3}(?:,\d{3})+|\d+\.?\d*"
)

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
})

_ANTONYM_PAIRS = {
    "love": "hate", "like": "dislike", "start": "stop", "begin": "end",
    "join": "leave", "buy": "sell", "hire": "fire", "accept": "reject",
    "open": "close", "win": "lose", "pass": "fail", "married": "divorced",
    "together": "separated", "happy": "sad", "alive": "dead",
    "employed": "unemployed", "healthy": "sick", "rich": "poor",
    "stay": "leave", "arrive": "depart", "gain": "lose",
}
_ANTONYM_REVERSE = {v: k for k, v in _ANTONYM_PAIRS.items()}
_ALL_ANTONYMS = {**_ANTONYM_PAIRS, **_ANTONYM_REVERSE}

_QUANTIFIER_OPPOSITES = {
    "all": "none", "always": "never", "everyone": "nobody",
    "everything": "nothing", "everywhere": "nowhere",
    "every": "no", "many": "few", "most": "least",
}
_QUANTIFIER_REVERSE = {v: k for k, v in _QUANTIFIER_OPPOSITES.items()}
_ALL_QUANTIFIERS = {**_QUANTIFIER_OPPOSITES, **_QUANTIFIER_REVERSE}

_ROLE_TITLES = frozenset({
    "intern", "junior", "senior", "lead", "manager", "director", "vp",
    "cto", "ceo", "cfo", "engineer", "developer", "designer", "analyst",
    "student", "graduate", "professor", "teacher", "doctor", "nurse",
    "chef", "lawyer", "freelancer", "consultant", "founder",
})

_TECH_TERMS = frozenset({
    "vim", "emacs", "vscode", "vs code", "sublime", "atom", "neovim",
    "python", "javascript", "typescript", "rust", "go", "java", "c++",
    "react", "vue", "angular", "svelte", "next", "nuxt",
    "npm", "yarn", "pnpm", "bun", "deno", "node",
    "postgres", "mysql", "sqlite", "mongodb", "redis", "dynamodb",
    "aws", "gcp", "azure", "vercel", "netlify", "heroku",
    "docker", "kubernetes", "terraform", "linux", "mac", "windows",
    "chrome", "firefox", "safari", "edge",
    "slack", "discord", "teams", "zoom",
    "iphone", "android", "pixel", "samsung", "macbook",
    "github", "gitlab", "bitbucket",
})

_LIFE_STAGES = frozenset({
    "graduated", "graduating", "engaged", "married", "pregnant",
    "retired", "retiring", "promoted", "fired", "laid off",
    "born", "died", "passed away", "moved", "moving",
    "enrolled", "accepted", "deployed", "discharged",
})

_COMPLETION_MARKERS = frozenset({
    "finally", "done", "finished", "completed", "passed", "failed",
    "got the results", "it's over", "it's done", "made it",
    "did it", "nailed it", "crushed it",
})

_WAITING_INDICATORS = frozenset({
    "waiting for", "expecting", "hoping", "fingers crossed",
    "should hear back", "should know", "any day now",
    "applied", "interviewing", "in review", "pending",
})

_OUTCOME_WORDS = frozenset({
    "got", "didn't get", "passed", "failed", "accepted", "rejected",
    "approved", "denied", "won", "lost", "made it", "didn't make it",
})

_ANNOUNCEMENT_FRAMES = frozenset({
    "guess what", "big news", "you won't believe", "i have something",
    "news flash", "breaking news", "update", "so get this",
    "omg so", "i need to tell you", "brace yourself",
})

_DISCOVERY_MARKERS = frozenset({
    "found out", "turns out", "discovered", "realized", "it's actually",
    "apparently", "come to find out", "would you believe",
})

_RESOLUTION_MARKERS = frozenset({
    "decided", "going with", "chose", "picked", "committed to",
    "settled on", "made up my mind", "final answer",
})

_SUDDENNESS_MARKERS = frozenset({
    "suddenly", "all of a sudden", "out of nowhere", "just like that",
    "overnight", "in an instant", "boom", "just now",
})

_GRADUAL_MARKERS = frozenset({
    "slowly", "gradually", "over time", "little by little",
    "coming around to", "warming up to", "starting to think",
})

_RELIEF_WORDS = frozenset({
    "relieved", "so relieved", "thank god", "finally", "phew",
    "weight off", "can breathe",
})

_DISAPPOINTMENT_WORDS = frozenset({
    "devastated", "crushed", "heartbroken", "gutted", "bummed",
    "disappointed", "let down",
})


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOPWORDS and len(w) > 1}


def _is_noise(text: str) -> bool:
    return text.lower().strip().rstrip("!?.… ") in _NOISE_EXACT or len(text.strip()) < 3


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


def _nearest_memory(content: str, memory_contents: list[str]) -> tuple[str, float]:
    if not memory_contents:
        return "", 0.0
    best_score = -1.0
    best_mem = ""
    for m in memory_contents:
        score = _word_overlap(content, m)
        if score > best_score:
            best_score = score
            best_mem = m
    return best_mem, best_score


def _nearest_memory_entity(content: str, memory_contents: list[str]) -> tuple[str, float]:
    if not memory_contents:
        return "", 0.0
    best_score = -1.0
    best_mem = ""
    for m in memory_contents:
        score = _entity_overlap(content, m)
        if score > best_score:
            best_score = score
            best_mem = m
    return best_mem, best_score


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


def _gz_len(text: str) -> int:
    return len(gzip.compress(text.encode("utf-8", errors="replace")))


def _sentiment_score(text: str) -> float:
    words = set(text.lower().split())
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def _has_hedge(text: str) -> bool:
    lower = text.lower()
    return any(h in lower for h in _HEDGE_WORDS)


def _has_certainty(text: str) -> bool:
    lower = text.lower()
    return any(c in lower for c in _CERTAINTY_WORDS)


def _extract_verb_stems(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    verbs = set()
    for w in words:
        if w.endswith("ed") and len(w) > 4:
            verbs.add(w[:-2])
            verbs.add(w[:-1])
        if w.endswith("ing") and len(w) > 5:
            verbs.add(w[:-3])
        verbs.add(w)
    return verbs


# ============================================================================
# CATEGORY 1: Contradiction Detection (001-020)
# ============================================================================


def variant_001(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Slot-value contradiction: entity+attribute+value triples, conflicting values."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    slot_re = re.compile(
        r"(?:i|she|he|we|they)\s+(?:am|is|are|was|were|'m|'s|'re)\s+(.+?)(?:\.|,|$|!|\?)",
        re.IGNORECASE,
    )
    verb_re = re.compile(
        r"(?:i|she|he|we|they)\s+((?:work|live|use|prefer|study|teach|drive|play|speak|eat|drink)s?\s+(?:at|in|with|on|for)?\s*.+?)(?:\.|,|$|!|\?)",
        re.IGNORECASE,
    )
    msg_slots = slot_re.findall(content) + verb_re.findall(content)
    if not msg_slots:
        return 0.0
    max_score = 0.0
    for mem in memory_contents:
        mem_slots = slot_re.findall(mem) + verb_re.findall(mem)
        if not mem_slots:
            continue
        overlap = _entity_overlap(content, mem)
        if overlap < 0.1:
            continue
        for ms in msg_slots:
            for ms2 in mem_slots:
                ms_words = _content_words(ms)
                ms2_words = _content_words(ms2)
                shared = ms_words & ms2_words
                diff = (ms_words ^ ms2_words) - _STOPWORDS
                if shared and diff:
                    score = overlap * min(1.0, len(diff) / max(len(shared), 1) * 0.5)
                    max_score = max(max_score, score)
    return _clamp(max_score * 2.0)


def variant_002(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Negation flip: message negates something stored."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    has_negation = any(neg in lower for neg in _NEGATION_PHRASES)
    if not has_negation:
        return 0.0
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp(overlap * 1.5 + 0.3)


def variant_003(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Verb-phrase antonym: antonym verbs about same entity."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_words = set(re.findall(r"[a-z]+", content.lower()))
    msg_antonyms = {_ALL_ANTONYMS[w] for w in msg_words if w in _ALL_ANTONYMS}
    if not msg_antonyms:
        return 0.0
    max_score = 0.0
    for mem in memory_contents:
        mem_words = set(re.findall(r"[a-z]+", mem.lower()))
        antonym_hits = msg_antonyms & mem_words
        if antonym_hits:
            overlap = _entity_overlap(content, mem)
            max_score = max(max_score, overlap + 0.3 * len(antonym_hits))
    return _clamp(max_score)


def variant_004(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Numeric contradiction: different numbers about same entity/topic."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_nums = _extract_numbers(content)
    if not msg_nums:
        return 0.0
    max_score = 0.0
    for mem in memory_contents:
        mem_nums = _extract_numbers(mem)
        if not mem_nums:
            continue
        overlap = _entity_overlap(content, mem)
        if overlap < 0.05:
            continue
        for mn in msg_nums:
            for en in mem_nums:
                try:
                    mn_val = float(mn.replace("$", "").replace("%", ""))
                    en_val = float(en.replace("$", "").replace("%", ""))
                    if mn_val != en_val and en_val != 0:
                        diff_ratio = abs(mn_val - en_val) / max(abs(en_val), 1)
                        max_score = max(max_score, overlap * min(1.0, diff_ratio))
                except ValueError:
                    if mn != en:
                        max_score = max(max_score, overlap * 0.5)
    return _clamp(max_score * 2.0)


def variant_005(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Location update: location mentions conflicting with stored locations."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_locs = _extract_locations(content)
    if not msg_locs:
        lower = content.lower()
        loc_verbs = ["moved to", "moving to", "relocated to", "lives in", "living in", "based in"]
        for lv in loc_verbs:
            if lv in lower:
                after = lower.split(lv, 1)[1].strip().split()[0:2]
                msg_locs.update(w for w in after if len(w) > 2)
    if not msg_locs:
        return 0.0
    for mem in memory_contents:
        mem_locs = _extract_locations(mem)
        lower_mem = mem.lower()
        for lv in ["lives in", "living in", "based in", "moved to", "from"]:
            if lv in lower_mem:
                after = lower_mem.split(lv, 1)[1].strip().split()[0:2]
                mem_locs.update(w for w in after if len(w) > 2)
        if mem_locs and msg_locs and not (msg_locs & mem_locs):
            overlap = _entity_overlap(content, mem)
            if overlap > 0.05 or any(lv in content.lower() for lv in ["moved", "moving", "relocated"]):
                return _clamp(0.6 + overlap)
    return 0.0


def variant_006(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Temporal contradiction: event placed at different time than stored."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    days = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
    months = {"january", "february", "march", "april", "may", "june",
              "july", "august", "september", "october", "november", "december"}
    msg_words = set(content.lower().split())
    msg_times = (msg_words & days) | (msg_words & months)
    msg_nums = set(_extract_numbers(content))
    if not msg_times and not msg_nums:
        return 0.0
    max_score = 0.0
    for mem in memory_contents:
        overlap = _entity_overlap(content, mem)
        if overlap < 0.1:
            continue
        mem_words = set(mem.lower().split())
        mem_times = (mem_words & days) | (mem_words & months)
        if msg_times and mem_times and msg_times != mem_times:
            max_score = max(max_score, 0.5 + overlap)
    return _clamp(max_score)


def variant_007(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Status change verbs: verbs implying state change × memory overlap."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    words = set(lower.split())
    change_hits = words & _STATE_CHANGE_VERBS
    if not change_hits:
        change_phrases = ["switched to", "changed to", "moved to", "converted to",
                          "upgraded to", "downgraded to", "replaced with", "traded for"]
        change_hits = {p for p in change_phrases if p in lower}
    if not change_hits:
        return 0.0
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp(0.3 + overlap * 0.7 + 0.1 * len(change_hits))


def variant_008(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Relationship status change: relationship words conflicting with stored state."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    msg_rel = {t for t in _RELATIONSHIP_TERMS if t in lower}
    if not msg_rel:
        return 0.0
    for mem in memory_contents:
        mem_lower = mem.lower()
        mem_rel = {t for t in _RELATIONSHIP_TERMS if t in mem_lower}
        if mem_rel and msg_rel != mem_rel:
            return _clamp(0.7 + _entity_overlap(content, mem) * 0.3)
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp(0.3 * overlap)


def variant_009(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Correction markers: explicit linguistic correction signals × memory overlap."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    correction_count = sum(1 for cm in _CORRECTION_MARKERS if cm in lower)
    not_x_but_y = bool(re.search(r"\bnot\s+\w+\s+but\s+\w+", lower))
    if correction_count == 0 and not not_x_but_y:
        return 0.0
    _, overlap = _nearest_memory(content, memory_contents)
    score = 0.4 + 0.2 * correction_count + 0.3 * overlap
    if not_x_but_y:
        score += 0.2
    return _clamp(score)


def variant_010(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Hedge→certainty shift: stored hedging, new message is certain."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    if not _has_certainty(content):
        return 0.0
    max_score = 0.0
    for mem in memory_contents:
        if _has_hedge(mem):
            overlap = _entity_overlap(content, mem)
            if overlap > 0.05:
                max_score = max(max_score, 0.5 + overlap * 0.5)
    return _clamp(max_score)


def variant_011(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Polarity reversal: opposite sentiment about a known topic."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_sent = _sentiment_score(content)
    if abs(msg_sent) < 0.3:
        return 0.0
    max_score = 0.0
    for mem in memory_contents:
        mem_sent = _sentiment_score(mem)
        if abs(mem_sent) < 0.3:
            continue
        if (msg_sent > 0) != (mem_sent > 0):
            overlap = _entity_overlap(content, mem)
            if overlap > 0.05:
                max_score = max(max_score, overlap * abs(msg_sent - mem_sent))
    return _clamp(max_score)


def variant_012(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Quantifier contradiction: all→none, always→never about same topic."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_words = set(content.lower().split())
    msg_quants = {w for w in msg_words if w in _ALL_QUANTIFIERS}
    if not msg_quants:
        return 0.0
    for mem in memory_contents:
        mem_words = set(mem.lower().split())
        for mq in msg_quants:
            opposite = _ALL_QUANTIFIERS.get(mq)
            if opposite and opposite in mem_words:
                overlap = _entity_overlap(content, mem)
                if overlap > 0.05:
                    return _clamp(0.6 + overlap * 0.4)
    return 0.0


def variant_013(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Role/title change: different role for a known person."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_words = set(content.lower().split())
    msg_roles = msg_words & _ROLE_TITLES
    if not msg_roles:
        return 0.0
    for mem in memory_contents:
        mem_words = set(mem.lower().split())
        mem_roles = mem_words & _ROLE_TITLES
        if mem_roles and msg_roles != mem_roles:
            overlap = _entity_overlap(content, mem)
            if overlap > 0.05:
                return _clamp(0.5 + overlap * 0.5)
    return 0.0


def variant_014(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Tool/technology swap: tech mentions replacing stored ones."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    msg_tech = {t for t in _TECH_TERMS if t in lower}
    if not msg_tech:
        return 0.0
    has_switch = any(sv in lower for sv in ["switched to", "moved to", "now using",
                                             "started using", "converted to", "migrated to"])
    for mem in memory_contents:
        mem_lower = mem.lower()
        mem_tech = {t for t in _TECH_TERMS if t in mem_lower}
        if mem_tech and msg_tech and not (msg_tech & mem_tech):
            overlap = _entity_overlap(content, mem)
            if overlap > 0.02 or has_switch:
                return _clamp(0.5 + 0.3 * has_switch + overlap * 0.3)
    return 0.0


def variant_015(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Frequency change: every day→stopped, weekly→daily, never→started."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    freq_words = {"daily", "weekly", "monthly", "yearly", "annually",
                  "every day", "every week", "twice a week", "once a week",
                  "never", "always", "often", "rarely", "sometimes", "regularly"}
    lower = content.lower()
    msg_freq = {f for f in freq_words if f in lower}
    if not msg_freq:
        return 0.0
    for mem in memory_contents:
        mem_lower = mem.lower()
        mem_freq = {f for f in freq_words if f in mem_lower}
        if mem_freq and msg_freq != mem_freq:
            overlap = _entity_overlap(content, mem)
            if overlap > 0.05:
                return _clamp(0.5 + overlap * 0.5)
    return 0.0


def variant_016(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Possession change: got/bought → gave away/sold for same item."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    acquire = {"got", "bought", "received", "adopted", "picked up"}
    dispose = {"sold", "gave away", "returned", "donated", "lost", "threw out", "got rid of"}
    lower = content.lower()
    msg_acquire = any(a in lower for a in acquire)
    msg_dispose = any(d in lower for d in dispose)
    if not msg_acquire and not msg_dispose:
        return 0.0
    for mem in memory_contents:
        mem_lower = mem.lower()
        mem_acquire = any(a in mem_lower for a in acquire)
        mem_dispose = any(d in mem_lower for d in dispose)
        if (msg_acquire and mem_dispose) or (msg_dispose and mem_acquire):
            overlap = _entity_overlap(content, mem)
            if overlap > 0.05:
                return _clamp(0.6 + overlap * 0.4)
    return 0.0


def variant_017(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Goal/plan contradiction: stored plan vs actual outcome."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    plan_words = {"planning", "plan to", "going to", "hoping to", "want to",
                  "aiming for", "goal is", "applying to", "trying for"}
    outcome_words = {"instead", "ended up", "actually went", "changed my mind",
                     "decided against", "not going", "cancelled", "backed out"}
    lower = content.lower()
    has_outcome_override = any(o in lower for o in outcome_words)
    if not has_outcome_override:
        return 0.0
    for mem in memory_contents:
        mem_lower = mem.lower()
        if any(p in mem_lower for p in plan_words):
            overlap = _entity_overlap(content, mem)
            if overlap > 0.05:
                return _clamp(0.6 + overlap * 0.4)
    return _clamp(0.2)


def variant_018(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Embedded correction: 'but'/'however' clause that contradicts memory."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    correction_conjunctions = [" but ", " however ", " although ", " though ",
                                " except ", " instead "]
    has_correction_conj = any(cc in lower for cc in correction_conjunctions)
    if not has_correction_conj:
        return 0.0
    for conj in correction_conjunctions:
        if conj in lower:
            parts = lower.split(conj, 1)
            if len(parts) == 2:
                after_conj = parts[1]
                for mem in memory_contents:
                    overlap = _word_overlap(after_conj, mem.lower())
                    if overlap > 0.1:
                        return _clamp(0.4 + overlap)
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp(0.2 * overlap)


def variant_019(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Implicit contradiction via timeline: events that can't coexist."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    location_re = re.compile(r"(?:i'm|i am|we're|we are)\s+(?:in|at)\s+(\w+)", re.IGNORECASE)
    msg_loc = location_re.search(content)
    if not msg_loc:
        return 0.0
    msg_place = msg_loc.group(1).lower()
    for mem in memory_contents:
        mem_loc = location_re.search(mem)
        if mem_loc:
            mem_place = mem_loc.group(1).lower()
            if msg_place != mem_place:
                return 0.7
    return 0.0


def variant_020(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Aggregate contradiction: weighted combination of top contradiction detectors."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    scores = [
        variant_001(content, memory_contents) * 0.25,
        variant_004(content, memory_contents) * 0.20,
        variant_007(content, memory_contents) * 0.20,
        variant_009(content, memory_contents) * 0.20,
        variant_005(content, memory_contents) * 0.15,
    ]
    return _clamp(sum(scores) * 1.5)


# ============================================================================
# CATEGORY 2: State-Change Detection (021-040)
# ============================================================================


def variant_021(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Update verb density × memory overlap."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    words = set(lower.split())
    all_change = words & (_STATE_CHANGE_VERBS | _UPDATE_VERBS)
    if not all_change:
        return 0.0
    density = len(all_change) / max(len(words), 1)
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp((density * 3.0 + 0.1) * (overlap + 0.3))


def variant_022(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Before→after frame: temporal transition markers × memory overlap."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    transition_patterns = [
        r"used to\b.*\bnow\b", r"before\b.*\bafter\b", r"was\b.*\bnow\b",
        r"went from\b.*\bto\b", r"no longer\b.*\binstead\b",
        r"used to be\b", r"not anymore\b",
    ]
    has_transition = any(re.search(p, lower) for p in transition_patterns)
    if not has_transition:
        return 0.0
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp(0.5 + overlap * 0.5)


def variant_023(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Life stage transition × memory overlap."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    stage_hits = sum(1 for ls in _LIFE_STAGES if ls in lower)
    if stage_hits == 0:
        return 0.0
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp(0.3 + 0.2 * stage_hits + overlap * 0.4)


def variant_024(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Decision announcement × uncertainty in memory."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    decision_markers = ["i decided", "we chose", "going with", "picked",
                        "committed to", "settled on", "made up my mind",
                        "final answer", "i'm going to", "we're doing"]
    has_decision = any(dm in lower for dm in decision_markers)
    if not has_decision:
        return 0.0
    for mem in memory_contents:
        if _has_hedge(mem):
            overlap = _entity_overlap(content, mem)
            if overlap > 0.05:
                return _clamp(0.6 + overlap * 0.4)
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp(0.3 + overlap * 0.3)


def variant_025(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Completion/resolution × open loops in memory."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    completion_hits = sum(1 for cm in _COMPLETION_MARKERS if cm in lower)
    if completion_hits == 0:
        return 0.0
    for mem in memory_contents:
        mem_lower = mem.lower()
        has_open = any(w in mem_lower for w in _WAITING_INDICATORS)
        if has_open:
            overlap = _entity_overlap(content, mem)
            if overlap > 0.05:
                return _clamp(0.5 + overlap * 0.5)
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp(0.2 + overlap * 0.3)


def variant_026(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Acquisition/loss × memory overlap."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    acq_loss = {"got", "bought", "received", "lost", "sold", "gave away",
                "threw out", "donated", "returned", "found", "inherited"}
    hits = sum(1 for al in acq_loss if al in lower)
    if hits == 0:
        return 0.0
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp(0.2 + 0.15 * hits + overlap * 0.5)


def variant_027(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Membership change × memory overlap."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    membership = {"joined", "left", "enrolled", "dropped out", "signed up",
                  "cancelled", "unsubscribed", "resigned", "transferred"}
    hits = sum(1 for m in membership if m in lower)
    if hits == 0:
        return 0.0
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp(0.3 + 0.2 * hits + overlap * 0.4)


def variant_028(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Emotional state shift: sentiment change about known topic."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_sent = _sentiment_score(content)
    if abs(msg_sent) < 0.1:
        return 0.0
    max_score = 0.0
    for mem in memory_contents:
        overlap = _entity_overlap(content, mem)
        if overlap < 0.05:
            continue
        mem_sent = _sentiment_score(mem)
        if abs(mem_sent) < 0.1:
            continue
        shift = abs(msg_sent - mem_sent)
        if shift > 0.5:
            max_score = max(max_score, overlap * shift)
    return _clamp(max_score * 1.5)


def variant_029(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Commitment escalation: progression about same entity."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    stages = [
        {"thinking about", "considering", "looking into", "exploring"},
        {"applied", "applied to", "signed up", "registered", "inquired"},
        {"accepted", "got in", "approved", "offered"},
        {"starting", "started", "joined", "enrolled", "moved"},
    ]
    lower = content.lower()
    msg_stage = -1
    for i, stage in enumerate(stages):
        if any(s in lower for s in stage):
            msg_stage = i
            break
    if msg_stage < 0:
        return 0.0
    for mem in memory_contents:
        mem_lower = mem.lower()
        mem_stage = -1
        for i, stage in enumerate(stages):
            if any(s in mem_lower for s in stage):
                mem_stage = i
                break
        if mem_stage >= 0 and msg_stage != mem_stage:
            overlap = _entity_overlap(content, mem)
            if overlap > 0.05:
                return _clamp(0.4 + abs(msg_stage - mem_stage) * 0.15 + overlap * 0.3)
    return 0.0


def variant_030(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Habit change × memory overlap."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    habit_patterns = [
        r"started\s+\w+ing", r"stopped\s+\w+ing", r"quit\s+\w+ing",
        r"gave up\s+\w+ing", r"switched from\b", r"switched to\b",
        r"no longer\s+\w+", r"now i\s+\w+",
    ]
    has_habit = any(re.search(p, lower) for p in habit_patterns)
    if not has_habit:
        return 0.0
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp(0.4 + overlap * 0.6)


def variant_031(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """New role/responsibility × memory overlap."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    role_patterns = ["i'm now", "they put me in charge", "i'm responsible for",
                     "i'm leading", "i'm the new", "i got promoted",
                     "i'm managing", "i'm heading up"]
    if not any(rp in lower for rp in role_patterns):
        return 0.0
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp(0.4 + overlap * 0.6)


def variant_032(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Health status change × memory overlap."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    health = {"diagnosed", "recovered", "surgery", "treatment", "test results",
              "in remission", "relapsed", "hospital", "doctor said",
              "feeling better", "feeling worse", "symptoms"}
    hits = sum(1 for h in health if h in lower)
    if hits == 0:
        return 0.0
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp(0.3 + 0.15 * hits + overlap * 0.4)


def variant_033(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Financial state change × memory overlap."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    financial = {"raise", "bonus", "debt", "paid off", "saved enough",
                 "can't afford", "promotion", "pay cut", "loan",
                 "salary", "got an offer", "new job"}
    hits = sum(1 for f in financial if f in lower)
    has_numbers = bool(_extract_numbers(content))
    if hits == 0 and not has_numbers:
        return 0.0
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp(0.2 + 0.15 * hits + overlap * 0.5)


def variant_034(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Environment change × memory overlap."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    env = {"new apartment", "new house", "remodeled", "moved desks",
           "new office", "working from home", "back in office",
           "new roommate", "living alone", "moved in", "moved out"}
    hits = sum(1 for e in env if e in lower)
    if hits == 0:
        return 0.0
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp(0.3 + 0.2 * hits + overlap * 0.4)


def variant_035(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Social circle change × memory overlap."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    social = {"met someone", "lost touch", "reconnected", "new roommate",
              "friend moved", "new coworker", "new neighbor", "stopped talking",
              "falling out", "made up", "new friend"}
    hits = sum(1 for s in social if s in lower)
    if hits == 0:
        return 0.0
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp(0.3 + 0.2 * hits + overlap * 0.4)


def variant_036(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Tense shift: different tense about known topic implies state change."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    present_re = re.compile(r"\b(is|are|works?|lives?|has|does|goes|plays?|runs?)\b", re.IGNORECASE)
    past_re = re.compile(r"\b(was|were|worked|lived|had|did|went|played|ran|used to)\b", re.IGNORECASE)
    msg_present = bool(present_re.search(content))
    msg_past = bool(past_re.search(content))
    if not msg_present and not msg_past:
        return 0.0
    max_score = 0.0
    for mem in memory_contents:
        overlap = _entity_overlap(content, mem)
        if overlap < 0.1:
            continue
        mem_present = bool(present_re.search(mem))
        mem_past = bool(past_re.search(mem))
        if (msg_past and mem_present) or (msg_present and mem_past):
            max_score = max(max_score, 0.4 + overlap * 0.5)
    return _clamp(max_score)


def variant_037(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Adverb intensifier × topic overlap: still/already/finally/just + known topic."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    intensifiers = {
        "still": 0.1,
        "already": 0.4,
        "finally": 0.6,
        "just": 0.3,
        "no longer": 0.7,
    }
    max_int_score = 0.0
    for word, score in intensifiers.items():
        if word in lower:
            max_int_score = max(max_int_score, score)
    if max_int_score == 0.0:
        return 0.0
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp(max_int_score * (overlap + 0.2))


def variant_038(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Modal verb shift: might→will, should→did about known topic."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    uncertain_modals = {"might", "may", "could", "should", "would"}
    certain_modals = {"will", "am", "did", "have", "is"}
    msg_words = set(content.lower().split())
    msg_certain = msg_words & certain_modals
    if not msg_certain:
        return 0.0
    for mem in memory_contents:
        mem_words = set(mem.lower().split())
        mem_uncertain = mem_words & uncertain_modals
        if mem_uncertain:
            overlap = _entity_overlap(content, mem)
            if overlap > 0.05:
                return _clamp(0.4 + overlap * 0.5)
    return 0.0


def variant_039(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Progressive→perfective: looking for→found, trying to→managed to."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    perfective = re.compile(
        r"\b(?:found|managed|succeeded|accomplished|achieved|got|made it|did it|finished)\b",
        re.IGNORECASE,
    )
    progressive = re.compile(
        r"\b(?:looking for|trying to|working on|searching for|hoping to|attempting)\b",
        re.IGNORECASE,
    )
    if not perfective.search(content):
        return 0.0
    for mem in memory_contents:
        if progressive.search(mem):
            overlap = _entity_overlap(content, mem)
            if overlap > 0.05:
                return _clamp(0.5 + overlap * 0.5)
    return 0.0


def variant_040(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """State-change keyword density × cosine sim to nearest memory."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    words = lower.split()
    all_change = _STATE_CHANGE_VERBS | _UPDATE_VERBS
    change_count = sum(1 for w in words if w in all_change)
    if change_count == 0:
        return 0.0
    density = change_count / max(len(words), 1)
    if memory_embeddings is not None and len(memory_embeddings) > 0:
        emb = _embed_one(content)
        sims = _cosine_sims(emb, memory_embeddings)
        max_sim = float(np.max(sims))
    else:
        _, max_sim = _nearest_memory(content, memory_contents)
    return _clamp(density * 3.0 * max_sim)


# ============================================================================
# CATEGORY 3: Uncertainty Resolution (041-060)
# ============================================================================


def variant_041(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Question→answer pair: message answers a question from stored memory."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    if content.strip().endswith("?"):
        return 0.0
    max_score = 0.0
    for mem in memory_contents:
        if "?" in mem:
            overlap = _entity_overlap(content, mem)
            if overlap > 0.1:
                max_score = max(max_score, 0.3 + overlap * 0.5)
    return _clamp(max_score)


def variant_042(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Outcome resolution: future language in memory + outcome in message."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    outcome_hits = sum(1 for o in _OUTCOME_WORDS if o in lower)
    if outcome_hits == 0:
        return 0.0
    future_words = {"interview", "application", "test", "trying", "applying",
                    "hoping", "waiting", "expecting", "audition", "exam"}
    for mem in memory_contents:
        mem_lower = mem.lower()
        if any(fw in mem_lower for fw in future_words):
            overlap = _entity_overlap(content, mem)
            if overlap > 0.05:
                return _clamp(0.5 + overlap * 0.5)
    return _clamp(0.1 * outcome_hits)


def variant_043(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Certainty injection: uncertainty in memory → certainty in message."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    if not _has_certainty(content):
        return 0.0
    for mem in memory_contents:
        if _has_hedge(mem):
            overlap = _entity_overlap(content, mem)
            if overlap > 0.05:
                return _clamp(0.5 + overlap * 0.5)
    return 0.0


def variant_044(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Waiting→resolved: waiting in memory + resolution in message."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    resolution = {"finally", "got it", "it came", "they said", "heard back",
                  "results are in", "answer is", "they decided"}
    has_resolution = any(r in lower for r in resolution)
    if not has_resolution:
        return 0.0
    for mem in memory_contents:
        mem_lower = mem.lower()
        if any(w in mem_lower for w in _WAITING_INDICATORS):
            overlap = _entity_overlap(content, mem)
            if overlap > 0.05:
                return _clamp(0.6 + overlap * 0.4)
    return _clamp(0.15)


def variant_045(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Open thread closure: entity overlap + completion markers."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    completion_hits = sum(1 for cm in _COMPLETION_MARKERS if cm in lower)
    if completion_hits == 0:
        return 0.0
    mem, overlap = _nearest_memory_entity(content, memory_contents)
    if overlap < 0.1:
        return 0.0
    mem_lower = mem.lower()
    open_ended = any(w in mem_lower for w in _WAITING_INDICATORS) or "?" in mem
    bonus = 0.3 if open_ended else 0.0
    return _clamp(0.3 + bonus + overlap * 0.4)


def variant_046(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Decision after deliberation: deliberation in memory, choice in message."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    has_decision = any(r in lower for r in _RESOLUTION_MARKERS)
    if not has_decision:
        return 0.0
    deliberation = {"pros and cons", "thinking about", "torn between",
                    "not sure if", "debating", "weighing", "considering"}
    for mem in memory_contents:
        mem_lower = mem.lower()
        if any(d in mem_lower for d in deliberation):
            overlap = _entity_overlap(content, mem)
            if overlap > 0.05:
                return _clamp(0.6 + overlap * 0.4)
    return _clamp(0.2)


def variant_047(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Result announcement × uncertainty in memory."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    result_patterns = ["i got in", "i passed", "we won", "it worked", "it broke",
                       "i made it", "i failed", "we lost", "it didn't work",
                       "i got the job", "i didn't get"]
    has_result = any(rp in lower for rp in result_patterns)
    if not has_result:
        return 0.0
    for mem in memory_contents:
        if _has_hedge(mem) or any(w in mem.lower() for w in _WAITING_INDICATORS):
            overlap = _entity_overlap(content, mem)
            if overlap > 0.05:
                return _clamp(0.6 + overlap * 0.4)
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp(0.3 + overlap * 0.3)


def variant_048(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Confirmation/denial of a previously stated possibility."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    confirm_deny = {"yes we are", "no we're not", "yes i did", "no i didn't",
                    "confirmed", "denied", "it's true", "it's not true",
                    "that's right", "that's wrong", "yep", "nope"}
    if not any(cd in lower for cd in confirm_deny):
        return 0.0
    for mem in memory_contents:
        if _has_hedge(mem) or "?" in mem:
            overlap = _entity_overlap(content, mem)
            if overlap > 0.05:
                return _clamp(0.4 + overlap * 0.5)
    return 0.0


def variant_049(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Timeline advancement: future in memory → past tense for same event."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    past_re = re.compile(r"\b(went|did|was|had|saw|heard|made|came|took|gave)\b", re.IGNORECASE)
    future_re = re.compile(r"\b(going to|will|gonna|planning to|about to)\b", re.IGNORECASE)
    if not past_re.search(content):
        return 0.0
    for mem in memory_contents:
        if future_re.search(mem):
            overlap = _entity_overlap(content, mem)
            if overlap > 0.1:
                return _clamp(0.4 + overlap * 0.5)
    return 0.0


def variant_050(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Scouting→commitment: looking→signed/committed."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    commitment = {"signed", "committed", "locked in", "put down a deposit",
                  "enrolled", "accepted the offer", "said yes", "booked"}
    has_commit = any(c in lower for c in commitment)
    if not has_commit:
        return 0.0
    scouting = {"looking at", "exploring", "checking out", "considering",
                "browsing", "shopping for", "researching"}
    for mem in memory_contents:
        mem_lower = mem.lower()
        if any(s in mem_lower for s in scouting):
            overlap = _entity_overlap(content, mem)
            if overlap > 0.05:
                return _clamp(0.6 + overlap * 0.4)
    return _clamp(0.2)


def variant_051(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Diagnosis/discovery × memory overlap."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    hits = sum(1 for d in _DISCOVERY_MARKERS if d in lower)
    if hits == 0:
        return 0.0
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp(0.3 + 0.15 * hits + overlap * 0.4)


def variant_052(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """External resolution × memory overlap."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    external = {"they approved", "the company decided", "my boss said",
                "the results came back", "the doctor said", "they offered",
                "they rejected", "she said yes", "he agreed", "they accepted"}
    hits = sum(1 for e in external if e in lower)
    if hits == 0:
        return 0.0
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp(0.3 + 0.2 * hits + overlap * 0.4)


def variant_053(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Implicit resolution via tense: uncertain topic now in definite past tense."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    past_definite = re.compile(
        r"\b(i\s+got|i\s+passed|i\s+went|it\s+was|we\s+had|i\s+made|i\s+did)\b",
        re.IGNORECASE,
    )
    if not past_definite.search(content):
        return 0.0
    for mem in memory_contents:
        if _has_hedge(mem) or any(w in mem.lower() for w in _WAITING_INDICATORS):
            overlap = _entity_overlap(content, mem)
            if overlap > 0.1:
                return _clamp(0.4 + overlap * 0.5)
    return 0.0


def variant_054(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Relief/disappointment × entity overlap with stored uncertainty."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    relief_hits = sum(1 for r in _RELIEF_WORDS if r in lower)
    disappoint_hits = sum(1 for d in _DISAPPOINTMENT_WORDS if d in lower)
    emotion_hits = relief_hits + disappoint_hits
    if emotion_hits == 0:
        return 0.0
    for mem in memory_contents:
        if _has_hedge(mem) or any(w in mem.lower() for w in _WAITING_INDICATORS):
            overlap = _entity_overlap(content, mem)
            if overlap > 0.05:
                return _clamp(0.5 + overlap * 0.4 + 0.1 * emotion_hits)
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp(0.15 * emotion_hits + overlap * 0.2)


def variant_055(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Announcement framing × memory overlap."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    hits = sum(1 for a in _ANNOUNCEMENT_FRAMES if a in lower)
    if hits == 0:
        return 0.0
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp(0.3 + 0.15 * hits + overlap * 0.4)


def variant_056(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Plan fulfillment: stored plan + past tense experience."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    past_exp = re.compile(
        r"\b(went to|visited|saw|attended|tried|experienced|it was)\b",
        re.IGNORECASE,
    )
    if not past_exp.search(content):
        return 0.0
    plan_markers = {"going to", "plan to", "want to visit", "tickets for",
                    "booked", "reservation at", "signed up for"}
    for mem in memory_contents:
        mem_lower = mem.lower()
        if any(p in mem_lower for p in plan_markers):
            overlap = _entity_overlap(content, mem)
            if overlap > 0.05:
                return _clamp(0.4 + overlap * 0.5)
    return 0.0


def variant_057(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Milestone reached × stored goal/aspiration."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    milestone = {"hit", "reached", "passed", "defended", "published",
                 "graduated", "certified", "completed", "earned"}
    has_milestone = any(m in lower for m in milestone)
    if not has_milestone:
        return 0.0
    goal_words = {"goal", "aiming", "working toward", "want to", "dream",
                  "aspiration", "trying to", "studying for", "training for"}
    for mem in memory_contents:
        mem_lower = mem.lower()
        if any(g in mem_lower for g in goal_words):
            overlap = _entity_overlap(content, mem)
            if overlap > 0.05:
                return _clamp(0.6 + overlap * 0.4)
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp(0.2 + overlap * 0.2)


def variant_058(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Bet/prediction resolved: stored prediction + outcome."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    outcome = {"happened", "came true", "was right", "was wrong",
               "called it", "nailed it", "jinxed it", "didn't happen"}
    has_outcome = any(o in lower for o in outcome)
    if not has_outcome:
        return 0.0
    predict = {"i think", "i bet", "probably", "my prediction",
               "i'm guessing", "mark my words", "watch"}
    for mem in memory_contents:
        mem_lower = mem.lower()
        if any(p in mem_lower for p in predict):
            overlap = _entity_overlap(content, mem)
            if overlap > 0.05:
                return _clamp(0.5 + overlap * 0.5)
    return 0.0


def variant_059(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Season/cycle resolution: semester→grades, sprint→shipped."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    endings = {"final grades", "shipped it", "annual review", "end of year",
               "semester over", "season ended", "race is done", "finished the"}
    has_ending = any(e in lower for e in endings)
    if not has_ending:
        return 0.0
    cycles = {"this semester", "this sprint", "this season", "this quarter",
              "this year", "this month", "training for"}
    for mem in memory_contents:
        mem_lower = mem.lower()
        if any(c in mem_lower for c in cycles):
            overlap = _entity_overlap(content, mem)
            if overlap > 0.05:
                return _clamp(0.5 + overlap * 0.5)
    return 0.0


def variant_060(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Aggregate uncertainty resolution: weighted combination of top 5."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    scores = [
        variant_042(content, memory_contents) * 0.25,
        variant_044(content, memory_contents) * 0.25,
        variant_047(content, memory_contents) * 0.20,
        variant_051(content, memory_contents) * 0.15,
        variant_045(content, memory_contents) * 0.15,
    ]
    return _clamp(sum(scores) * 1.5)


# ============================================================================
# CATEGORY 4: Information-Theoretic PE (061-080)
# ============================================================================


def variant_061(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Conditional entropy H(msg|memory) via char n-gram model."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    memory_text = " ".join(memory_contents)
    if len(memory_text) < 20:
        return 0.3
    trigrams: Counter = Counter()
    bigrams: Counter = Counter()
    mt = memory_text.lower()
    for i in range(len(mt) - 2):
        trigrams[mt[i:i+3]] += 1
        bigrams[mt[i:i+2]] += 1
    msg = content.lower()
    total_surprise = 0.0
    count = 0
    for i in range(len(msg) - 2):
        tri = msg[i:i+3]
        bi = msg[i:i+2]
        p_tri = (trigrams.get(tri, 0) + 1) / (bigrams.get(bi, 0) + 256)
        total_surprise += -math.log2(max(p_tri, 1e-10))
        count += 1
    if count == 0:
        return 0.0
    avg_surprise = total_surprise / count
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp((avg_surprise / 8.0) * (overlap + 0.2))


def variant_062(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Pointwise mutual information: PMI between message words and memory words."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_words = _content_words(content)
    if not msg_words:
        return 0.0
    mem_word_counts: Counter = Counter()
    total_mem_words = 0
    for m in memory_contents:
        mw = _content_words(m)
        mem_word_counts.update(mw)
        total_mem_words += len(mw)
    if total_mem_words == 0:
        return 0.0
    pmis = []
    for w in msg_words:
        p_w = (mem_word_counts.get(w, 0) + 1) / (total_mem_words + len(mem_word_counts))
        pmis.append(-math.log2(max(p_w, 1e-10)))
    avg_pmi = sum(pmis) / len(pmis)
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp((avg_pmi / 15.0) * (overlap + 0.3))


def variant_063(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Topic model surprise: NMF-style topic similarity inversion."""
    if _is_noise(content) or not memory_contents or len(memory_contents) < 3:
        return 0.0
    if memory_embeddings is None or len(memory_embeddings) < 3:
        return 0.0
    emb = _embed_one(content)
    sims = _cosine_sims(emb, memory_embeddings)
    max_sim = float(np.max(sims))
    mean_sim = float(np.mean(sims))
    topic_surprise = 1.0 - max_sim
    entity_focus = _entity_overlap(content, memory_contents[int(np.argmax(sims))])
    return _clamp(topic_surprise * entity_focus * 3.0)


def variant_064(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Entity-conditioned surprise: for known entities, how surprising are the predicates?"""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        return 0.0
    msg_words = _content_words(content) - msg_entities
    if not msg_words:
        return 0.0
    entity_word_sets: dict[str, set] = defaultdict(set)
    for mem in memory_contents:
        mem_entities = _extract_entities(mem)
        shared = msg_entities & mem_entities
        if shared:
            mem_words = _content_words(mem) - mem_entities
            for e in shared:
                entity_word_sets[e].update(mem_words)
    if not entity_word_sets:
        return 0.0
    max_surprise = 0.0
    for entity, known_words in entity_word_sets.items():
        new_words = msg_words - known_words
        if known_words:
            surprise = len(new_words) / max(len(msg_words), 1)
            max_surprise = max(max_surprise, surprise)
    return _clamp(max_surprise)


def variant_065(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Slot-filling surprise: how surprising is the new filler vs old filler?"""
    if _is_noise(content) or not memory_contents:
        return 0.0
    slot_patterns = [
        (r"(?:lives?|living)\s+(?:in|at)\s+(.+?)(?:\.|,|$|!)", "location"),
        (r"(?:works?|working)\s+(?:at|for)\s+(.+?)(?:\.|,|$|!)", "workplace"),
        (r"(?:uses?|using)\s+(.+?)(?:\.|,|$|!)", "tool"),
        (r"(?:studies|studying)\s+(?:at)?\s*(.+?)(?:\.|,|$|!)", "school"),
        (r"(?:salary|makes?|earns?|paid)\s+(.+?)(?:\.|,|$|!)", "compensation"),
    ]
    max_score = 0.0
    for pattern, slot_name in slot_patterns:
        msg_match = re.search(pattern, content, re.IGNORECASE)
        if not msg_match:
            continue
        msg_val = msg_match.group(1).strip().lower()
        for mem in memory_contents:
            mem_match = re.search(pattern, mem, re.IGNORECASE)
            if mem_match:
                mem_val = mem_match.group(1).strip().lower()
                if msg_val != mem_val:
                    if memory_embeddings is not None and len(memory_embeddings) > 0:
                        try:
                            e1, e2 = _embed([msg_val, mem_val])
                            val_sim = _cosine_sim(e1, e2)
                            max_score = max(max_score, 1.0 - val_sim)
                        except Exception:
                            max_score = max(max_score, 0.7)
                    else:
                        max_score = max(max_score, 0.7)
    return _clamp(max_score)


def variant_066(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Transition probability violation: unlikely entity→state transition."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    msg_verbs = _extract_verb_stems(content)
    if not msg_entities or not msg_verbs:
        return 0.0
    entity_verbs: dict[str, Counter] = defaultdict(Counter)
    for mem in memory_contents:
        mem_entities = _extract_entities(mem)
        mem_verbs = _extract_verb_stems(mem)
        for e in mem_entities:
            entity_verbs[e].update(mem_verbs)
    max_surprise = 0.0
    for e in msg_entities:
        if e in entity_verbs:
            known = entity_verbs[e]
            total = sum(known.values())
            for v in msg_verbs:
                p = (known.get(v, 0) + 0.1) / (total + len(known) * 0.1)
                surprise = -math.log2(max(p, 1e-10)) / 10.0
                max_surprise = max(max_surprise, surprise)
    return _clamp(max_surprise)


def variant_067(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Compression cost DELTA: how much does adding this message change memory compressibility?"""
    if _is_noise(content) or not memory_contents:
        return 0.0
    memory_text = "\n".join(memory_contents[-50:])
    before = _gz_len(memory_text)
    after = _gz_len(memory_text + "\n" + content)
    expected_addition = _gz_len(content)
    delta = after - before
    if expected_addition == 0:
        return 0.0
    ratio = delta / max(expected_addition, 1)
    _, overlap = _nearest_memory(content, memory_contents)
    if overlap < 0.05:
        return _clamp(ratio * 0.3)
    return _clamp(ratio * overlap * 2.0)


def variant_068(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Cross-entropy spike × entity overlap (pure new-topic perplexity is novelty, not PE)."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    memory_text = " ".join(memory_contents)
    if len(memory_text) < 30:
        return 0.0
    chars: Counter = Counter(memory_text.lower())
    total = sum(chars.values())
    msg = content.lower()
    ce = 0.0
    for c in msg:
        p = (chars.get(c, 0) + 1) / (total + 256)
        ce += -math.log2(p)
    if len(msg) == 0:
        return 0.0
    avg_ce = ce / len(msg)
    _, overlap = _nearest_memory(content, memory_contents)
    if overlap < 0.05:
        return 0.0
    return _clamp((avg_ce / 8.0 - 0.3) * overlap * 2.0)


def variant_069(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """KL divergence of entity attribute distributions."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        return 0.0
    msg_cw = _content_words(content)
    entity_attrs: dict[str, Counter] = defaultdict(Counter)
    for mem in memory_contents:
        mem_entities = _extract_entities(mem)
        shared = msg_entities & mem_entities
        if shared:
            mem_cw = _content_words(mem)
            for e in shared:
                entity_attrs[e].update(mem_cw)
    if not entity_attrs:
        return 0.0
    max_kl = 0.0
    for entity, prior_counts in entity_attrs.items():
        total_prior = sum(prior_counts.values())
        vocab = set(prior_counts.keys()) | msg_cw
        kl = 0.0
        for w in vocab:
            p = (prior_counts.get(w, 0) + 1) / (total_prior + len(vocab))
            q = (1 if w in msg_cw else 0.1) / max(len(msg_cw), 1)
            if q > 0:
                kl += p * abs(math.log2(max(p, 1e-10)) - math.log2(max(q, 1e-10)))
        max_kl = max(max_kl, kl)
    return _clamp(max_kl / 5.0)


def variant_070(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Mutual information gain via compression."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    mem, overlap = _nearest_memory(content, memory_contents)
    if overlap < 0.05:
        return 0.0
    c_msg = _gz_len(content)
    c_mem = _gz_len(mem)
    c_both = _gz_len(content + " " + mem)
    mi = (c_msg + c_mem - c_both) / max(min(c_msg, c_mem), 1)
    msg_words = _content_words(content)
    mem_words = _content_words(mem)
    diff_words = msg_words ^ mem_words
    shared = msg_words & mem_words
    if shared and diff_words:
        return _clamp(mi * len(diff_words) / max(len(shared), 1))
    return _clamp(mi * 0.3)


def variant_071(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Bayesian belief update magnitude: |prior - posterior| for entity states."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return 0.0
    emb = _embed_one(content)
    sims = _cosine_sims(emb, memory_embeddings)
    nearest_idx = int(np.argmax(sims))
    nearest_sim = float(sims[nearest_idx])
    if nearest_sim < 0.3:
        return 0.0
    nearest_mem = memory_contents[nearest_idx]
    msg_words = _content_words(content)
    mem_words = _content_words(nearest_mem)
    shared = msg_words & mem_words
    diff = (msg_words | mem_words) - shared
    if not diff:
        return 0.0
    update_magnitude = len(diff) / max(len(shared | diff), 1)
    return _clamp(nearest_sim * update_magnitude * 2.0)


def variant_072(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Vocabulary surprise per entity: new words associated with known entity."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    if not msg_entities:
        msg_cw = _content_words(content)
        if len(msg_cw) < 3:
            return 0.0
        for mem in memory_contents:
            overlap = _word_overlap(content, mem)
            if overlap > 0.15:
                mem_cw = _content_words(mem)
                new_words = msg_cw - mem_cw
                return _clamp(len(new_words) / max(len(msg_cw), 1) * overlap * 2.0)
        return 0.0
    entity_vocab: dict[str, set] = defaultdict(set)
    for mem in memory_contents:
        mem_entities = _extract_entities(mem)
        mem_cw = _content_words(mem)
        for e in mem_entities:
            entity_vocab[e].update(mem_cw)
    msg_cw = _content_words(content)
    max_surprise = 0.0
    for e in msg_entities:
        if e in entity_vocab:
            known = entity_vocab[e]
            new = msg_cw - known
            if known:
                surprise = len(new) / max(len(msg_cw), 1)
                max_surprise = max(max_surprise, surprise)
    return _clamp(max_surprise)


def variant_073(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Predictive coding residual: ||actual - predicted|| in embedding space."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    if memory_embeddings is None or len(memory_embeddings) < 2:
        return 0.0
    emb = _embed_one(content)
    sims = _cosine_sims(emb, memory_embeddings)
    top_k = min(3, len(sims))
    top_idx = np.argsort(sims)[::-1][:top_k]
    predicted = memory_embeddings[top_idx].mean(axis=0)
    residual = float(np.linalg.norm(emb - predicted))
    max_sim = float(sims[top_idx[0]])
    if max_sim < 0.3:
        return 0.0
    return _clamp(residual * max_sim)


def variant_074(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Schema violation: deviation from common patterns for this topic/entity."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_cw = _content_words(content)
    if len(msg_cw) < 2:
        return 0.0
    topic_words: Counter = Counter()
    matching_mems = 0
    for mem in memory_contents:
        overlap = _word_overlap(content, mem)
        if overlap > 0.1:
            topic_words.update(_content_words(mem))
            matching_mems += 1
    if matching_mems < 2:
        return 0.0
    common_words = {w for w, c in topic_words.items() if c >= matching_mems * 0.5}
    unusual = msg_cw - common_words
    if not common_words:
        return 0.0
    return _clamp(len(unusual) / max(len(msg_cw), 1) * 0.8)


def variant_075(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Temporal model violation: message at unexpected temporal position."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    time_words = {"morning", "afternoon", "evening", "night", "weekend",
                  "weekday", "early", "late", "midnight", "dawn", "dusk"}
    msg_words = set(content.lower().split())
    msg_time = msg_words & time_words
    if not msg_time:
        return 0.0
    for mem in memory_contents:
        overlap = _entity_overlap(content, mem)
        if overlap < 0.1:
            continue
        mem_words = set(mem.lower().split())
        mem_time = mem_words & time_words
        if mem_time and msg_time != mem_time:
            return _clamp(0.4 + overlap * 0.5)
    return 0.0


def variant_076(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Suffix tree surprise × entity overlap."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    memory_text = " ".join(memory_contents[-30:]).lower()
    msg = content.lower()
    if len(msg) < 4 or len(memory_text) < 20:
        return 0.0
    total_depth = 0
    count = 0
    for i in range(len(msg)):
        max_match = 0
        for length in range(1, min(len(msg) - i, 20) + 1):
            substr = msg[i:i+length]
            if substr in memory_text:
                max_match = length
            else:
                break
        total_depth += max_match
        count += 1
    if count == 0:
        return 0.0
    avg_depth = total_depth / count
    max_possible = min(len(msg), 20)
    match_ratio = avg_depth / max(max_possible, 1)
    _, overlap = _nearest_memory(content, memory_contents)
    if overlap < 0.05:
        return 0.0
    return _clamp((1.0 - match_ratio) * overlap * 1.5)


def variant_077(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Entity co-occurrence surprise: known entities appearing in new combination."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    if len(msg_entities) < 2:
        return 0.0
    cooccur: set[frozenset] = set()
    all_known: set[str] = set()
    for mem in memory_contents:
        mem_entities = _extract_entities(mem)
        all_known.update(mem_entities)
        if len(mem_entities) >= 2:
            for e1 in mem_entities:
                for e2 in mem_entities:
                    if e1 < e2:
                        cooccur.add(frozenset({e1, e2}))
    known_in_msg = msg_entities & all_known
    if len(known_in_msg) < 2:
        return 0.0
    new_pairs = 0
    total_pairs = 0
    for e1 in known_in_msg:
        for e2 in known_in_msg:
            if e1 < e2:
                total_pairs += 1
                if frozenset({e1, e2}) not in cooccur:
                    new_pairs += 1
    if total_pairs == 0:
        return 0.0
    return _clamp(new_pairs / total_pairs)


def variant_078(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Register shift: style/formality change for discussion about known entity."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    formal = {"moreover", "furthermore", "consequently", "therefore",
              "regarding", "concerning", "pursuant", "hereby"}
    informal = {"lol", "lmao", "omg", "dude", "bro", "yo", "haha",
                "nah", "yep", "gonna", "wanna", "gotta", "kinda"}
    msg_words = set(content.lower().split())
    msg_formal = len(msg_words & formal)
    msg_informal = len(msg_words & informal)
    if msg_formal == 0 and msg_informal == 0:
        return 0.0
    for mem in memory_contents:
        overlap = _entity_overlap(content, mem)
        if overlap < 0.1:
            continue
        mem_words = set(mem.lower().split())
        mem_formal = len(mem_words & formal)
        mem_informal = len(mem_words & informal)
        if (msg_formal > 0 and mem_informal > 0) or (msg_informal > 0 and mem_formal > 0):
            return _clamp(0.3 + overlap * 0.5)
    return 0.0


def variant_079(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Information gain on entity model: increase, decrease, or shift entropy."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_entities = _extract_entities(content)
    msg_cw = _content_words(content)
    if not msg_cw:
        return 0.0
    entity_attrs: dict[str, Counter] = defaultdict(Counter)
    for mem in memory_contents:
        mem_entities = _extract_entities(mem)
        shared = msg_entities & mem_entities if msg_entities else set()
        if shared:
            mem_cw = _content_words(mem)
            for e in shared:
                entity_attrs[e].update(mem_cw)
    if not entity_attrs:
        _, overlap = _nearest_memory(content, memory_contents)
        return _clamp(overlap * 0.2)
    max_shift = 0.0
    for entity, prior in entity_attrs.items():
        total = sum(prior.values())
        if total < 2:
            continue
        prior_entropy = 0.0
        for count in prior.values():
            p = count / total
            if p > 0:
                prior_entropy -= p * math.log2(p)
        posterior = prior.copy()
        posterior.update(msg_cw)
        total_post = sum(posterior.values())
        post_entropy = 0.0
        for count in posterior.values():
            p = count / total_post
            if p > 0:
                post_entropy -= p * math.log2(p)
        shift = abs(post_entropy - prior_entropy)
        max_shift = max(max_shift, shift)
    return _clamp(max_shift / 3.0)


def variant_080(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Aggregate info-theoretic PE: weighted combination of top 5."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    scores = [
        variant_065(content, memory_contents, memory_embeddings) * 0.25,
        variant_067(content, memory_contents) * 0.20,
        variant_071(content, memory_contents, memory_embeddings) * 0.20,
        variant_064(content, memory_contents) * 0.20,
        variant_073(content, memory_contents, memory_embeddings) * 0.15,
    ]
    return _clamp(sum(scores) * 1.5)


# ============================================================================
# CATEGORY 5: Hybrid and Cognitive-Inspired (081-100)
# ============================================================================


def _salience_proxy(content: str) -> float:
    lower = content.lower().strip()
    if lower in _NOISE_EXACT:
        return 0.02
    commitment_re = re.compile(
        r"\b(?:i\s+(?:got|did|made|found|started|quit|left|joined|"
        r"accepted|finished|signed|bought|sold|moved|passed|graduated|"
        r"earned|won|lost|broke|fixed)"
        r"|i'm\s+(?:pregnant|engaged|leaving|moving|starting)"
        r"|we're\s+(?:pregnant|engaged|moving|having)"
        r"|it's\s+(?:official|confirmed|done|over))\b",
        re.IGNORECASE,
    )
    if commitment_re.search(lower):
        return 0.8
    words = re.findall(r"[a-zA-Z]+", content)
    if len(words) >= 5:
        return 0.5
    return 0.25


def variant_081(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Contradiction × salience: correcting important beliefs has higher PE."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    contradiction = max(
        variant_001(content, memory_contents),
        variant_007(content, memory_contents),
        variant_009(content, memory_contents),
    )
    if contradiction < 0.1:
        return 0.0
    salience = _salience_proxy(content)
    return _clamp(contradiction * salience * 1.5)


def variant_082(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """State-change × memory age: changes to OLD memories are more surprising."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    words = set(lower.split())
    has_change = bool(words & (_STATE_CHANGE_VERBS | _UPDATE_VERBS))
    if not has_change:
        return 0.0
    best_score = 0.0
    for i, mem in enumerate(memory_contents):
        overlap = _entity_overlap(content, mem)
        if overlap < 0.05:
            continue
        age_factor = (i + 1) / len(memory_contents)
        recency_weight = 1.0 - age_factor
        score = overlap * (0.3 + 0.7 * recency_weight)
        best_score = max(best_score, score)
    return _clamp(best_score)


def variant_083(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """State-change × memory confidence: contradicting confident memories is higher PE."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    has_change = any(v in lower for v in _STATE_CHANGE_VERBS) or any(cm in lower for cm in _CORRECTION_MARKERS)
    if not has_change:
        return 0.0
    max_score = 0.0
    for mem in memory_contents:
        overlap = _entity_overlap(content, mem)
        if overlap < 0.05:
            continue
        confidence = 0.5
        if _has_certainty(mem):
            confidence = 0.9
        elif _has_hedge(mem):
            confidence = 0.2
        max_score = max(max_score, overlap * confidence)
    return _clamp(max_score * 1.5)


def variant_084(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Recency-weighted contradiction: contradicting recent memories is more surprising."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    max_score = 0.0
    n = len(memory_contents)
    for i, mem in enumerate(memory_contents):
        overlap = _entity_overlap(content, mem)
        if overlap < 0.1:
            continue
        msg_words = _content_words(content)
        mem_words = _content_words(mem)
        shared = msg_words & mem_words
        diff = (msg_words ^ mem_words) - _STOPWORDS
        if shared and diff:
            recency = (i + 1) / n
            score = overlap * recency * len(diff) / max(len(shared | diff), 1)
            max_score = max(max_score, score)
    return _clamp(max_score * 2.0)


def variant_085(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Multi-hop PE: contradicts something IMPLIED by stored memories."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    implications = {
        "vegan": {"meat", "steak", "burger", "bacon", "chicken", "fish", "bbq", "ribs"},
        "vegetarian": {"meat", "steak", "burger", "bacon", "chicken"},
        "sober": {"drunk", "drinking", "beer", "wine", "cocktail", "whiskey"},
        "single": {"girlfriend", "boyfriend", "wife", "husband", "partner", "dating"},
        "unemployed": {"coworker", "boss", "office", "meeting", "deadline", "project"},
        "retired": {"boss", "office", "meeting", "deadline", "commute"},
    }
    lower = content.lower()
    msg_words = set(lower.split())
    for mem in memory_contents:
        mem_lower = mem.lower()
        for identity, contradicts in implications.items():
            if identity in mem_lower:
                hits = msg_words & contradicts
                if hits:
                    return _clamp(0.5 + 0.1 * len(hits))
    return 0.0


def variant_086(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Hedged update: hedged updates score lower PE than definite ones."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    has_change = any(v in lower for v in _STATE_CHANGE_VERBS) or any(v in lower for v in _UPDATE_VERBS)
    if not has_change:
        return 0.0
    _, overlap = _nearest_memory(content, memory_contents)
    if overlap < 0.05:
        return 0.0
    base_score = 0.4 + overlap * 0.6
    if _has_hedge(content):
        base_score *= 0.5
    elif _has_certainty(content):
        base_score *= 1.2
    return _clamp(base_score)


def variant_087(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Source-sensitive PE: self-correction vs correcting others."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    self_correct = bool(re.search(r"\bi\s+(?:was wrong|actually|lied|misspoke|meant to say)\b", lower))
    if not self_correct:
        has_correction = any(cm in lower for cm in _CORRECTION_MARKERS)
        if not has_correction:
            return 0.0
    _, overlap = _nearest_memory(content, memory_contents)
    score = overlap * 0.7
    if self_correct:
        score += 0.3
    return _clamp(score)


def variant_088(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Repetition with variation: nearly identical message but one detail changed."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    if memory_embeddings is None or len(memory_embeddings) == 0:
        msg_cw = _content_words(content)
        if len(msg_cw) < 2:
            return 0.0
        max_score = 0.0
        for mem in memory_contents:
            mem_cw = _content_words(mem)
            shared = msg_cw & mem_cw
            diff = msg_cw ^ mem_cw
            total = len(shared | diff)
            if total > 0 and len(shared) > 0:
                overlap_ratio = len(shared) / total
                diff_ratio = len(diff) / total
                if 0.4 < overlap_ratio < 0.95 and diff_ratio > 0.05:
                    max_score = max(max_score, overlap_ratio * diff_ratio * 4.0)
        return _clamp(max_score)
    emb = _embed_one(content)
    sims = _cosine_sims(emb, memory_embeddings)
    max_sim = float(np.max(sims))
    if max_sim < 0.5 or max_sim > 0.98:
        return 0.0
    nearest_idx = int(np.argmax(sims))
    nearest_mem = memory_contents[nearest_idx]
    msg_cw = _content_words(content)
    mem_cw = _content_words(nearest_mem)
    diff = msg_cw ^ mem_cw
    shared = msg_cw & mem_cw
    if not diff or not shared:
        return 0.0
    return _clamp(max_sim * len(diff) / max(len(shared), 1))


def variant_089(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Sarcasm/exaggeration filter: hyperbole should not be treated as genuine PE."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    sarcasm_markers = {"best day ever", "worst day ever", "literally dying",
                       "so fun", "great just great", "love that for me",
                       "cool cool cool", "/s", "suuure", "riiiight",
                       "oh great", "just perfect", "fan-freaking-tastic"}
    exaggeration = {"literally", "100%", "never ever", "always always",
                    "worst ever", "best ever", "million times"}
    is_sarcastic = any(s in lower for s in sarcasm_markers)
    is_exaggerated = any(e in lower for e in exaggeration)
    if is_sarcastic or is_exaggerated:
        base_pe = variant_007(content, memory_contents)
        return _clamp(base_pe * 0.3)
    return variant_007(content, memory_contents)


def variant_090(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Gradual vs sudden shift: sudden changes score higher PE."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    is_sudden = any(s in lower for s in _SUDDENNESS_MARKERS)
    is_gradual = any(g in lower for g in _GRADUAL_MARKERS)
    has_change = any(v in lower for v in _STATE_CHANGE_VERBS) or any(v in lower for v in _UPDATE_VERBS)
    if not has_change and not is_sudden:
        return 0.0
    _, overlap = _nearest_memory(content, memory_contents)
    base = 0.3 + overlap * 0.5
    if is_sudden:
        base *= 1.4
    elif is_gradual:
        base *= 0.6
    return _clamp(base)


def variant_091(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Reconsolidation trigger: message retrieves stored memory AND modifies it."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    if memory_embeddings is not None and len(memory_embeddings) > 0:
        emb = _embed_one(content)
        sims = _cosine_sims(emb, memory_embeddings)
        max_sim = float(np.max(sims))
        if max_sim < 0.4:
            return 0.0
        nearest_idx = int(np.argmax(sims))
        nearest_mem = memory_contents[nearest_idx]
    else:
        nearest_mem, max_sim = _nearest_memory(content, memory_contents)
        if max_sim < 0.15:
            return 0.0
    msg_cw = _content_words(content)
    mem_cw = _content_words(nearest_mem)
    shared = msg_cw & mem_cw
    diff = msg_cw - mem_cw
    if not shared or not diff:
        return 0.0
    retrieval = len(shared) / max(len(msg_cw), 1)
    modification = len(diff) / max(len(msg_cw), 1)
    return _clamp(retrieval * modification * 4.0)


def variant_092(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Dopamine PE model: positive PE (better than expected) > negative PE > zero PE."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    positive_outcome = any(w in lower for w in {"got in", "passed", "won", "made it",
                                                  "approved", "accepted", "promoted",
                                                  "succeeded", "nailed"})
    negative_outcome = any(w in lower for w in {"rejected", "failed", "lost", "denied",
                                                  "fired", "didn't get", "didn't make"})
    if not positive_outcome and not negative_outcome:
        return 0.0
    for mem in memory_contents:
        mem_lower = mem.lower()
        was_uncertain = _has_hedge(mem) or any(w in mem_lower for w in _WAITING_INDICATORS)
        was_pessimistic = any(w in mem_lower for w in {"probably won't", "doubt", "unlikely",
                                                        "not sure if", "slim chance"})
        was_optimistic = any(w in mem_lower for w in {"definitely will", "sure to",
                                                       "guaranteed", "no doubt"})
        overlap = _entity_overlap(content, mem)
        if overlap < 0.05:
            continue
        if positive_outcome and was_pessimistic:
            return _clamp(0.9)
        elif negative_outcome and was_optimistic:
            return _clamp(0.7)
        elif (positive_outcome or negative_outcome) and was_uncertain:
            return _clamp(0.5 + overlap * 0.4)
    _, overlap = _nearest_memory(content, memory_contents)
    return _clamp(0.2 + overlap * 0.2)


def variant_093(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Memory specificity match: PE higher when contradicted memory is specific."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    has_change = any(v in lower for v in _STATE_CHANGE_VERBS) or any(cm in lower for cm in _CORRECTION_MARKERS)
    if not has_change:
        return 0.0
    max_score = 0.0
    for mem in memory_contents:
        overlap = _entity_overlap(content, mem)
        if overlap < 0.05:
            continue
        specificity = 0.3
        mem_nums = _extract_numbers(mem)
        mem_entities = _extract_entities(mem)
        mem_locs = _extract_locations(mem)
        specificity += 0.15 * min(len(mem_nums), 2)
        specificity += 0.15 * min(len(mem_entities), 3)
        specificity += 0.2 * min(len(mem_locs), 2)
        max_score = max(max_score, overlap * specificity)
    return _clamp(max_score * 2.0)


def variant_094(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Cascading PE: contradicting belief X that implies Y increases PE."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    base_pe = max(
        variant_001(content, memory_contents),
        variant_007(content, memory_contents),
        variant_005(content, memory_contents),
    )
    if base_pe < 0.2:
        return 0.0
    _, overlap = _nearest_memory(content, memory_contents)
    cascade = 0
    if overlap > 0.1:
        nearest_mem, _ = _nearest_memory(content, memory_contents)
        for other_mem in memory_contents:
            if other_mem == nearest_mem:
                continue
            if _word_overlap(nearest_mem, other_mem) > 0.15:
                cascade += 1
    cascade_bonus = min(0.3, cascade * 0.05)
    return _clamp(base_pe + cascade_bonus)


def variant_095(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Time-since-last-update: PE increases with time since entity was last updated."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    has_update = any(v in lower for v in _STATE_CHANGE_VERBS | _UPDATE_VERBS)
    if not has_update:
        return 0.0
    msg_entities = _extract_entities(content)
    msg_cw = _content_words(content)
    last_mention_idx = -1
    for i in range(len(memory_contents) - 1, -1, -1):
        mem = memory_contents[i]
        if msg_entities:
            if _extract_entities(mem) & msg_entities:
                last_mention_idx = i
                break
        elif _word_overlap(content, mem) > 0.15:
            last_mention_idx = i
            break
    if last_mention_idx < 0:
        return 0.0
    gap = len(memory_contents) - 1 - last_mention_idx
    gap_factor = min(1.0, gap / max(len(memory_contents), 1) * 3.0)
    return _clamp(0.3 + gap_factor * 0.7)


def variant_096(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Gist-change detector: compare gist of message vs gist of stored memories about entity."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    if memory_embeddings is None or len(memory_embeddings) == 0:
        return variant_091(content, memory_contents, None)
    emb = _embed_one(content)
    sims = _cosine_sims(emb, memory_embeddings)
    nearest_idx = int(np.argmax(sims))
    nearest_sim = float(sims[nearest_idx])
    if nearest_sim < 0.3:
        return 0.0
    nearest_mem = memory_contents[nearest_idx]
    msg_cw = sorted(_content_words(content))[:5]
    mem_cw = sorted(_content_words(nearest_mem))[:5]
    gist_overlap = len(set(msg_cw) & set(mem_cw))
    gist_diff = len(set(msg_cw) ^ set(mem_cw))
    if gist_overlap == 0:
        return 0.0
    return _clamp(nearest_sim * gist_diff / max(gist_overlap + gist_diff, 1) * 2.0)


def variant_097(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Fact density × contradiction overlap: multiple contradicting facts = higher PE."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    msg_nums = _extract_numbers(content)
    msg_entities = _extract_entities(content)
    msg_locs = _extract_locations(content)
    fact_count = len(msg_nums) + len(msg_entities) + len(msg_locs)
    if fact_count == 0:
        return 0.0
    contradiction_count = 0
    for mem in memory_contents:
        overlap = _entity_overlap(content, mem)
        if overlap < 0.1:
            continue
        mem_nums = _extract_numbers(mem)
        mem_locs = _extract_locations(mem)
        if msg_nums and mem_nums:
            for mn in msg_nums:
                for en in mem_nums:
                    if mn != en:
                        contradiction_count += 1
        if msg_locs and mem_locs and not (msg_locs & mem_locs):
            contradiction_count += 1
    if contradiction_count == 0:
        return 0.0
    density = min(1.0, fact_count / 5.0)
    contradiction_rate = min(1.0, contradiction_count / max(fact_count, 1))
    return _clamp(density * contradiction_rate * 1.5)


def variant_098(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Implicit vs explicit correction: both should score high via different paths."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    lower = content.lower()
    explicit_correction = bool(re.search(
        r"\b(?:correction|actually|i\s+was\s+wrong|not\s+\w+\s+but\s+\w+|i\s+meant)\b",
        lower,
    ))
    implicit_switch = bool(re.search(
        r"\b(?:now\s+i|switched\s+to|moved\s+to|started\s+using|changed\s+to)\b",
        lower,
    ))
    if not explicit_correction and not implicit_switch:
        return 0.0
    _, overlap = _nearest_memory(content, memory_contents)
    if explicit_correction:
        return _clamp(0.5 + overlap * 0.5)
    else:
        return _clamp(0.3 + overlap * 0.6)


def variant_099(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """AND-gate: entity_overlap × value_change. Neither alone triggers high PE."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    max_score = 0.0
    for mem in memory_contents:
        overlap = _entity_overlap(content, mem)
        if overlap < 0.1:
            continue
        msg_cw = _content_words(content)
        mem_cw = _content_words(mem)
        shared = msg_cw & mem_cw
        changed = (msg_cw | mem_cw) - shared
        if not shared or not changed:
            continue
        value_change = len(changed) / max(len(shared | changed), 1)
        score = overlap * value_change
        has_change_verb = any(v in content.lower() for v in _STATE_CHANGE_VERBS)
        if has_change_verb:
            score *= 1.5
        max_score = max(max_score, score)
    return _clamp(max_score * 2.5)


def variant_100(content: str, memory_contents: list[str], memory_embeddings: np.ndarray = None) -> float:
    """Kitchen sink: weighted ensemble of best from each category."""
    if _is_noise(content) or not memory_contents:
        return 0.0
    cat1 = max(
        variant_007(content, memory_contents),
        variant_009(content, memory_contents),
        variant_005(content, memory_contents),
    )
    cat2 = max(
        variant_021(content, memory_contents),
        variant_029(content, memory_contents),
        variant_040(content, memory_contents, memory_embeddings),
    )
    cat3 = max(
        variant_042(content, memory_contents),
        variant_044(content, memory_contents),
        variant_047(content, memory_contents),
    )
    cat4 = max(
        variant_065(content, memory_contents, memory_embeddings),
        variant_071(content, memory_contents, memory_embeddings),
    )
    cat5 = max(
        variant_091(content, memory_contents, memory_embeddings),
        variant_099(content, memory_contents),
    )
    return _clamp(
        cat1 * 0.30 + cat2 * 0.20 + cat3 * 0.20 + cat4 * 0.15 + cat5 * 0.15
    )


# ============================================================================
# Registry
# ============================================================================

ALL_VARIANTS: dict[str, Callable] = {}
for _name, _obj in list(globals().items()):
    if _name.startswith("variant_") and callable(_obj):
        ALL_VARIANTS[_name] = _obj

ALL_VARIANTS = dict(sorted(ALL_VARIANTS.items()))
