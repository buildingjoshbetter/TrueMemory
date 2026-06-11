# Prediction Error Scoring Research — 100-Variant Sweep

## Research Sources
- 18+ ArXiv papers on contradiction detection, belief update, stance change, dialogue state tracking
- 6 cognitive neuroscience papers on prediction error in memory encoding
- Web searches on NLI-based approaches, fact-checking, knowledge graph updates
- Analysis of existing novelty sweep (120 variants) and salience sweep to identify what NOT to duplicate

## The Core Problem

Prediction Error (PE) is fundamentally different from novelty and salience:

| Signal | Question | Depends on memory? | Context-dependent? |
|--------|----------|--------------------|--------------------|
| Novelty | "Is this new?" | Yes (distance from stored) | No (new is new) |
| Salience | "Is this worth remembering?" | No (message alone) | No |
| Prediction Error | "Does this change what I believe?" | Yes (relationship to specific beliefs) | YES — same message, different PE based on memory state |

The critical property: PE is **context-dependent**. "I moved to Portland" is LOW PE with empty memory but HIGH PE when "Lives in Seattle" is stored. No novelty or salience variant has this property.

## Key Techniques by Area

### 1. Contradiction Detection (NLI)

**Key papers:**
- Bowman et al. 2015 (SNLI), Williams et al. 2018 (MultiNLI) — foundational NLI datasets
- Reimers & Gurevych 2019 — sentence-transformers cross-encoders for NLI
- Feng & Hunter 2024 — neurosymbolic NLI with logic decomposition

**Fast approaches (no LLM):**
- Cross-encoder NLI models: DeBERTa-v3-small, MiniLM — ~5ms per pair on CPU
- BUT: we can't add transformer model dependencies for the gate (must be <100ms, zero new deps)
- **Lightweight alternative:** Rule-based contradiction detection using slot-value extraction + negation detection + verb antonyms

**Key insight:** Direct contradiction detection is the purest PE signal but requires understanding entity-attribute-value triples in both message and memory.

### 2. Dialogue State Tracking (DST)

**Key papers:**
- Kim et al. 2020 (SOM-DST) — state operation prediction: {CARRYOVER, UPDATE, DELETE, DONTCARE}
- Guo et al. 2021 (Dual Slot Selector) — local reliability verification for slot changes
- Heck et al. 2020 (TripPy) — triple copy strategy, value-independent

**Key insight:** The SOM-DST decomposition maps directly to PE:
- CARRYOVER = 0 PE (no change)
- UPDATE = moderate PE (value changed)
- DELETE = high PE (belief removed)

For our use case, we can approximate this without a trained model by:
1. Extracting entity+attribute slots from both message and memory
2. Finding matching entities across message and memory
3. Detecting when a value changes for the same entity+attribute

### 3. Cognitive Neuroscience — PE in Memory Encoding

**Key papers:**
- Rescorla & Wagner 1972 — PE = α × β × (λ - ΣV). Learning proportional to mismatch
- Exton-McGuinness et al. 2015 — PE necessary for memory destabilization/reconsolidation
- Sinclair & Bhatt 2018 — PE magnitude determines reconsolidation threshold
- O'Neill et al. 2025 — D1R mediates PE-driven destabilization
- Song & Xin 2026 (D-MEM) — dopamine-gated PE routing for LLM agent memory

**Key insight:** Three PE regimes:
1. **Zero PE** — expected outcome, no learning (CARRYOVER)
2. **Moderate PE** — unexpected but compatible, update in place (UPDATE)
3. **High PE** — contradicts expectation, destabilize + reconsolidate (DELETE + re-store)

The biological model suggests PE should be multiplicative: `PE = surprise × relevance`. A surprising message about an unknown topic is novelty, not PE. A surprising message about a KNOWN topic is PE.

### 4. Change-Point Detection

**Key papers:**
- Adams & MacKay 2007 (BOCPD) — Bayesian online changepoint detection
- Xing et al. 2018 — online LDA + BOCPD for text streams

**Key insight:** Track entity-specific distributions. When a message shifts the distribution for a known entity, that's PE. This catches gradual drift that pairwise contradiction misses.

### 5. Stance Change Detection

**Key papers:**
- Mohammad et al. 2016 (SemEval-2016 Task 6)
- Tan et al. 2016 (ChangeMyView)

**Key insight:** Stance change collapses to NLI contradiction when applied to the same entity-topic pair over time.

### 6. Slot-Value Change Detection

**Key insight from DST:** For structured facts, maintain last known value per entity+attribute. When new value differs, PE = f(old_value, new_value). Simple embedding cosine between old and new values works.

## Architecture Decision

The 100 variants explore five categories of PE scoring:

1. **Contradiction Detection (001-020):** Direct detection of when message conflicts with memory
2. **State-Change Detection (021-040):** Detection of state transitions without explicit contradiction
3. **Uncertainty Resolution (041-060):** Detection of when message resolves open questions
4. **Information-Theoretic PE (061-080):** Formal information theory approaches
5. **Hybrid / Cognitive-Inspired (081-100):** Combinations and biologically-inspired

**CRITICAL CONSTRAINT:** All variants must be:
- Fast (<100ms per message)
- Local (no LLM calls, no API calls)
- Deterministic
- Memory-state-dependent (different scores for same message with different memory)
- Uncorrelated with shipped novelty (compression-based) and salience (speech-act) scorers

## What NOT to Build (avoid novelty/salience duplication)

- Compression cost against memory → novelty v025 (AUC 0.788)
- Cosine distance from memories → novelty v001-v020
- NCD/conditional compression → novelty v021-v040
- Speech act classification → salience v23/d (AUC 0.726)
- Message-in-isolation features → salience (length, caps, entity density)

**The litmus test:** If the variant gives the same score for "I moved to Portland" regardless of what's in memory, it's NOT measuring PE.
