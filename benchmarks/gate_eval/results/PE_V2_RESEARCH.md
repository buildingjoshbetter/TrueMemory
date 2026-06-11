# PE v2 100-Variant Sweep — Research Notes

## Phase 1: Computational Feasibility

### Environment

| Package | Installed | Version |
|---------|-----------|---------|
| model2vec | Yes | 0.8.1 |
| numpy | Yes | 2.4.2 |
| scipy | Yes | 1.17.1 |
| sklearn | Yes | 1.8.0 |
| torch | Pending | CPU-only install attempted |
| transformers | Pending | Installing with torch |
| sentence-transformers | Pending | Installing with torch |
| spacy | No | Not needed if transformers available |
| nltk | No | Not needed (VADER reimplemented) |

### v1 Results Summary

- Best AUC: 0.626 (v070, MI gain via compression)
- Best context-sensitive: 0.54 (v005, location update)
- Best ablation gain: +0.021 AUC (v033, financial state change × memory overlap)
- N+S baseline: 0.792 AUC
- N+S+PE best: 0.813 AUC

### Critical v1 Failures

1. "We broke up" + "Dating Riley" → PE=0.0 across ALL 100 variants
2. "I GOT IT" + "Interviewing at Google" → PE=0.0 across ALL variants
3. Best variants not context-sensitive — same score regardless of memory state

---

## Paradigm Research

### 1. NLI Contradiction Classifiers

**Best model:** `cross-encoder/nli-MiniLM2-L6-H768` — 6-layer, 768-dim, ~80MB. Pre-trained on SNLI+MultiNLI (~1M sentence pairs). Returns [entailment, neutral, contradiction] logits.

**Key insight:** NLI models encode that "broke up" contradicts "dating" because SNLI/MNLI training data contains examples like:
- Premise: "They are a couple" → Hypothesis: "They broke up" → Contradiction

**Fallback (no torch):** Embedding-based NLI approximation:
1. Embed message and memory with model2vec
2. High cosine similarity = entailment proxy
3. Moderate similarity + entity overlap = potential contradiction
4. Construct "negated memory" by prepending "not" and compare distances
5. Won't match real NLI but captures some semantic signal

**Speed:** ~15ms/pair with torch, ~2ms with fallback

### 2. Embedding Direction Geometry

**No external dependencies needed.** Pure numpy vector math on model2vec embeddings.

**Key algorithms:**
- Cosine of difference vectors: O(d) where d=256 (model2vec dim)
- EMA velocity tracking: store per-entity running average, O(d) update
- PCA: numpy.linalg.svd on entity message matrix, O(n×d²)
- Semantic differential: pre-compute positive/negative anchor vectors once

**Entity extraction:** Reuse _PROPER_NOUN_RE and _extract_entities from v1.

**Speed:** All <5ms. This is the fastest paradigm.

### 3. Structured Triple Extraction

**No external dependencies needed.** Regex-based triple extraction.

**Key patterns (~50 templates):**
- `[Person] lives in [Place]` → (person, lives_in, place)
- `[Person] works at [Org]` → (person, works_at, org)
- `[Person] uses [Tool]` → (person, uses, tool)
- `[Person] is [Attribute]` → (person, is, attribute)
- `[Person]'s [Slot]` → (person, has_slot, value)
- `[Entity] costs/is $[Amount]` → (entity, costs, amount)

**Conflict detection:** Same subject + same relation + different value = PE.

**Fuzzy matching:** Jaccard similarity on lowercased tokens > 0.5 = "same value."

**Speed:** All <5ms. Pure regex.

### 4. Knowledge Graph Conflict Detection

**No external dependencies needed.** Dict-of-dicts structure.

**Key insight vs Paradigm 3:** KG maintains persistent state, detects CASCADE conflicts. "Alex quit Google" invalidates "Alex's Google coworker Bob."

**Functional dependencies to encode:**
- lives_in: 1 value at a time
- works_at: 1 value at a time  
- dating/married: 1 partner at a time
- uses (tool): multiple OK
- likes: multiple OK

**Graph structure:** `{entity: {relation: [(value, timestamp, msg_idx)]}}`

**TransE for v039:** Simple ||h + r - t|| scoring. Embed entities and relations with model2vec, learn offset vectors. This is approximate but captures geometric structure.

**Speed:** All <10ms. Dict lookups + BFS for cascade.

### 5. Cross-Encoder Pairwise Scoring

**Best model:** `cross-encoder/stsb-distilroberta-base` — trained on STS Benchmark, outputs similarity score 0-5.

**Key insight:** Cross-encoders see both texts simultaneously, enabling cross-attention. This is why "broke up" vs "dating" might register — the model attends to both tokens together.

**Fallback (no torch):** Embedding-based pairwise scoring:
1. Compute separate embeddings for message and memory
2. Similarity deficit: entity overlap implies high similarity expected, measure the gap
3. Negation injection: prepend "not" to memory, compare distances
4. Element-wise difference L2 norm as contradiction proxy

**Speed:** ~20ms/pair with model, ~3ms with fallback

### 6. Conversation Flow / Discourse Analysis

**No external dependencies needed.** State tracking + discourse markers + simple sentiment.

**Discourse markers for correction/contrast:**
- Correction: "actually", "turns out", "I realized", "wait", "no wait"
- Contrast: "but", "however", "instead", "on the other hand"
- Concession: "although", "despite", "even though"
- Update: "now", "anymore", "no longer", "finally"

**Sentiment without VADER:** Simple word-list approach:
- Positive: love, great, amazing, happy, excited, perfect, good...
- Negative: hate, terrible, awful, sad, angry, frustrated, devastated...
- Score = (positive_count - negative_count) / total_content_words

**Commitment tracking:** Detect "I will", "I'm going to", "I decided" — store as commitments. Compare later messages against commitments.

**Speed:** All <5ms. Pure Python state tracking.

### 7. Forensic Stylometry

**No external dependencies needed.** Pure regex and counting.

**Key features:**
- Epistemic markers: "actually", "turns out" etc — high precision for corrections
- Hedging: modal verbs (might, could, probably)
- Emphasis: ALL CAPS, !, intensifiers
- Temporal adverbs: "now", "anymore", "finally", "no longer"
- Pronoun shifts: we→I, I→we

**Speed:** All <2ms. Fastest paradigm after pre-computed constants.

### 8. Bioinformatic Sequence Alignment

**No external dependencies needed.** `difflib.SequenceMatcher` + custom implementations.

**Key algorithms:**
- Levenshtein via `difflib.SequenceMatcher.get_opcodes()` — classifies as equal, replace, insert, delete
- Smith-Waterman: local alignment, O(n×m), ~10-30 tokens = fast
- Needleman-Wunsch: global alignment with affine gap penalties
- Template alignment: replace entities with [ENTITY], numbers with [NUMBER], align templates

**Speed:** All <5ms on short sequences. O(n×m) where n,m ≤ 30 tokens.

### 9. Change-Point Detection

**Available via scipy/numpy.** No additional dependencies.

**Key algorithms:**
- CUSUM: cumulative sum of deviations from running mean. Pure numpy.
- BOCPD (Bayesian Online): O(n) per new observation. Implemented from scratch, ~50 lines.
- EMA deviation: exponential moving average, O(d) per update.
- k-means clustering: sklearn.cluster.KMeans for topic cluster shift.

**Cold start:** Need ≥3 messages about an entity. Return 0.0 for fewer.

**Speed:** All <10ms. Numpy vectorized operations.

### 10. Masked Slot-Filling Surprise

**Best model:** `distilbert-base-uncased` (~250MB) or `albert-base-v2` (~45MB).

**Key insight:** Mask the key slot in a memory ("Alice works at [MASK]"), let the MLM predict. If it predicts "Google" with high confidence but the message says "Anthropic," the surprise is high PE.

**Fallback (no torch):** Embedding-based slot comparison:
1. Extract templates from messages using regex (replace entities/numbers with [SLOT])
2. When templates match between message and memory, compare slot fillers
3. Use embedding distance between slot fillers as surprise
4. Weight by entity overlap (same entity + different filler = high PE)

This fallback is essentially a hybrid of Paradigm 3 (triple extraction) and Paradigm 8 (template alignment), applied specifically to slot-filling. It won't capture the MLM's learned world knowledge but will capture structural slot mismatches.

**Speed:** ~15ms with model, ~3ms with fallback

---

## Critical Feasibility Assessment

**Can all 10 paradigms run within 100ms per message?**

| Paradigm | With models | Fallback | Verdict |
|----------|------------|----------|---------|
| 1. NLI | ~15ms | ~2ms | ✓ |
| 2. Geometry | ~3ms | N/A | ✓ |
| 3. Triples | ~3ms | N/A | ✓ |
| 4. KG | ~5ms | N/A | ✓ |
| 5. Cross-Enc | ~20ms | ~3ms | ✓ |
| 6. Discourse | ~3ms | N/A | ✓ |
| 7. Stylometry | ~2ms | N/A | ✓ |
| 8. Alignment | ~3ms | N/A | ✓ |
| 9. Change-Pt | ~5ms | N/A | ✓ |
| 10. MLM | ~15ms | ~3ms | ✓ |

All paradigms fit within budget. Even with models, running all 10 variants from a single paradigm takes <200ms total.

**Total benchmark time estimate:**
- 2000 messages × 100 variants × ~5ms average = 1000 seconds (~17 minutes)
- With model-based paradigms: ~2000 messages × 30 model variants × 20ms = 1200 seconds (~20 minutes)
- Total: ~40 minutes for the full sweep

---

## Dependency Installation Plan

1. Try: `python3 -m pip install torch --cpu transformers sentence-transformers`
2. If successful: use real NLI, cross-encoder, and MLM models
3. If failed: use model2vec-based fallbacks for Paradigms 1, 5, 10
4. Either way: all 100 variants will run

---

## Key Design Decisions

1. **Entity extraction** shared across all paradigms: reuse v1's `_extract_entities`, `_extract_locations`, `_extract_numbers`
2. **Memory retrieval** shared: embedding similarity via model2vec  
3. **Per-entity state** for Paradigms 2, 4, 6, 7, 9: dict per conversation, reset between conversations
4. **Model loading** at script start: lazy initialization with try/except
5. **Scoring normalization**: all outputs clipped to [0, 1]
