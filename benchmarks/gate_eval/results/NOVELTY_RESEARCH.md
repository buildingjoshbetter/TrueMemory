# Novelty Scoring Research — 250-Variant Sweep

## Research Sources
- 18 ArXiv papers across novelty detection, anomaly detection, OOD detection, compression distance, information theory
- 5 PaperPlain/Semantic Scholar searches
- 7 web searches (academic blogs, GitHub implementations, surveys)
- 3 deep-think sessions on signal design, compression approaches, and cognitive neuroscience

## Top Techniques by Category

### Information-Theoretic
- **Normalized Compression Distance (NCD)**: gzip(x+y) vs gzip(x)+gzip(y). Zero deps, ~10 lines. (Cilibrasi & Vitanyi 2005, Jiang et al. ACL 2023)
- **Conditional compression ratio**: C(M+m) - C(M) / C(m). Directly measures "how much new information does this message add to stored memories?"
- **Bayesian surprise**: KL(posterior || prior) on topic distribution after observing message
- **Shannon surprise**: -log2(P(message | memory_model)) via character n-gram model
- **PPM prediction accuracy**: Prediction by Partial Matching model trained on memory corpus
- **Vendi Novelty Score**: Does this message increase the Vendi Score (diversity metric) of the memory buffer? (Pasarkar 2026)

### Embedding-Based
- **Mahalanobis distance**: Fit Gaussian to memory embeddings, score by statistical distance (NLP-ADBench validated)
- **Isolation Forest**: Anomaly score = path length to isolate point from memory embeddings
- **LOF (Local Outlier Factor)**: Local density vs neighbor density
- **Running centroid distance**: EMA of memory embeddings, cosine distance from centroid
- **Pattern completion failure (CA1)**: Distance from avg(top-k nearest memories) — NOT from individual memories

### Cognitive/Biological
- **Orienting Response**: AND-gate of unexpectedness * meaningfulness. The key insight: cosine distance alone fires on noise
- **Habituation curve**: exp(-k * exposure_count) for similar message types. Naturally solves "ok" problem
- **Von Restorff isolation**: Contextual distinctiveness within surrounding conversation window
- **Synaptic tagging**: Weak tags decay unless reinforced by related high-salience signal within time window
- **Levels of processing**: Depth markers (causal connectives, temporal sequencing, entity references)

### Text Features
- **Lexical density / TTR / Hapax ratio**: Vocabulary richness signals
- **Speech act classification**: Already proven in salience sweep (AUC 0.787)
- **Named entity density / numeric density / temporal markers**: Fact anchors
- **Discourse markers / hedging / backchannels**: Noise indicators

## Key Architectural Insight

The winning scorer will almost certainly be a **multiplicative gate** (AND-gate) between:
1. An **information content** signal (is this message saying something substantive?)
2. A **semantic distance** signal (is this different from what's already stored?)

Neither alone works:
- Cosine distance alone: "ok" scores high (distant from factual memories)
- Content alone: "I love pizza" scores high (has content) even if already stored

The brain does this via the orienting response: novelty AND significance must BOTH fire.

## Implementation Priority
1. Content-quality gates (speech act, entity density, compression) — solve the "ok" problem
2. Memory-comparative signals (NCD, conditional compression, Mahalanobis) — measure actual novelty
3. Hybrid AND-gates — combine content + distance
4. Biological approximations — habituation, Bayesian surprise, Von Restorff
