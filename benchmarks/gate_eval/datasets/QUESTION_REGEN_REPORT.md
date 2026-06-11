# GateLoCoMo Question Regeneration Report — v2

## Summary

Regenerated 200 benchmark questions from 194, fixing three problems:
1. **Difficulty rebalanced**: 63/30/6% → 40/40/20% (easy/medium/hard)
2. **Category diversified**: 72% fact → 8 categories with proper spread
3. **Count fixed**: 194 → 200 questions

## Distribution: Actual vs Targets

### By Difficulty

| Difficulty | Target | Actual | Status |
|-----------|--------|--------|--------|
| Easy | 80 (40%) | 80 | ✅ |
| Medium | 80 (40%) | 80 | ✅ |
| Hard | 40 (20%) | 40 | ✅ |

### By Category

| Category | Target | Actual | Status |
|---------|--------|--------|--------|
| fact | 50 (25%) | 50 | ✅ |
| preference | 25 (12.5%) | 25 | ✅ |
| decision | 25 (12.5%) | 25 | ✅ |
| temporal | 25 (12.5%) | 25 | ✅ |
| emotional | 25 (12.5%) | 25 | ✅ |
| life_event | 20 (10%) | 20 | ✅ |
| comparison | 15 (7.5%) | 15 | ✅ |
| recommendation | 15 (7.5%) | 15 | ✅ |

### By Conversation

| Conversation | Target | Actual | Status |
|-------------|--------|--------|--------|
| Conv 1 (Alex & Jordan) | 40 | 40 | ✅ |
| Conv 2 (Maria & Sam) | 40 | 40 | ✅ |
| Conv 3 (Dev & Casey) | 40 | 40 | ✅ |
| Conv 4 (Pat & Riley) | 40 | 40 | ✅ |
| Conv 5 (Taylor & Morgan) | 40 | 40 | ✅ |

## Validation Results

| Check | Result |
|-------|--------|
| Total: exactly 200 | ✅ 200 |
| Difficulty within ±2 of target | ✅ all exact |
| Category within ±3 of target | ✅ all exact |
| Per-conversation within ±2 | ✅ all exact at 40 |
| No duplicate question IDs | ✅ 0 duplicates |
| No near-duplicate questions | ✅ fixed 1 (Berlin life/neighborhood overlap) |
| All evidence_messages exist | ✅ 0 missing IDs |
| All evidence references signal/borderline | ✅ 0 noise references |
| Hard questions span 3+ sessions | ✅ all 40 hard questions verified |
| Medium questions use 2-3 messages | ✅ verified |
| Easy questions use exactly 1 message | ✅ all 80 verified |
| All 5 conversations in each category | ✅ verified |

## Category Spread per Conversation

| Category | Conv1 | Conv2 | Conv3 | Conv4 | Conv5 |
|---------|-------|-------|-------|-------|-------|
| fact | 9 | 8 | 10 | 12 | 11 |
| preference | 5 | 5 | 6 | 4 | 5 |
| decision | 5 | 5 | 5 | 6 | 4 |
| temporal | 5 | 6 | 4 | 5 | 5 |
| emotional | 5 | 6 | 5 | 4 | 5 |
| life_event | 4 | 4 | 5 | 3 | 4 |
| comparison | 4 | 3 | 2 | 3 | 3 |
| recommendation | 3 | 3 | 3 | 3 | 3 |

## Improvement Over v1

| Metric | v1 | v2 |
|--------|----|----|
| Total questions | 194 | 200 |
| Easy | 123 (63%) | 80 (40%) |
| Medium | 59 (30%) | 80 (40%) |
| Hard | 12 (6%) | 40 (20%) |
| fact category | 140 (72%) | 50 (25%) |
| Categories used | 6 | 8 |
| comparison questions | 0 | 15 |
| recommendation questions | 0 | 15 |

---

## Phase 5: Rustle-the-Feathers

### Perspective 1: The Gate Tester

**Can these questions distinguish a good gate from a bad gate?**

Yes, with caveats:

- **Perfect gate (keeps all signal, drops all noise)**: Would score ~100% on easy and medium questions. Hard questions require synthesis across sessions and inference, so even a perfect gate needs a competent retrieval+reasoning layer on top.
- **Signal-dropping gate (randomly drops 20% of signal)**: Easy questions (1 evidence message each) have a 20% chance of becoming unanswerable. Medium questions (2-3 evidence messages) have a 36-49% chance of losing at least one piece — and many become partially answerable but imprecise. Hard questions are more resilient because they draw from 3-8 sessions — losing one session out of 6 still leaves enough context.
- **Implication**: Easy questions are the most sensitive gate quality indicator. A system that scores well on easy questions but poorly on medium/hard likely has good gate quality but weak retrieval.

**Key concern**: Recommendation questions (15 total) test synthesis and inference more than gate quality. A system with perfect gate performance but poor reasoning could fail these. This is acceptable — we want to test the *memory system*, not just the gate — but the benchmark should weight recommendation/comparison questions separately when computing a gate-specific score.

### Perspective 2: The Naturalness Critic

**Random 20-question review**: All questions read as natural human speech. No "describe the trajectory of" or "what factual information was disclosed" patterns. Examples that work well:
- "How much is Alex making at the new gig?"
- "Has Casey figured out the career thing yet?" (reframed)
- "Would Sam prefer books or something artsy?"
- "What's the timeline on Pat's side project?"

**Potential issues found**:
- 2 questions start with "Tell me about..." which is natural but slightly interviewish. Acceptable.
- Hard questions like "Walk me through all the big things..." are natural for an AI assistant context but wouldn't be how you'd ask a friend. This is correct — the benchmark simulates someone asking their AI.

**Gold answer specificity**: Good. Answers cite specific numbers ($245k, $2,400/month), names (Derek, Priya, Marco), and dates. Recommendation answers give actual recommendations with reasoning, not vague gestures. Medium/hard answers explicitly show cross-session synthesis.

### Perspective 3: The Difficulty Skeptic

**Are the hard questions genuinely hard?**

Mostly yes, with a few that are "medium-plus":

**Genuinely hard (require synthesis/inference)**:
- "What would be a good birthday gift for Alex?" — requires inferring preferences from scattered mentions across sessions (food, music, climbing, AI safety)
- "How would you describe the dynamic of Alex and Jordan's friendship?" — requires tracking how each helps the other across the entire conversation arc
- "Who handles stress better — Maria or Sam?" — requires comparing coping styles from different situations across sessions
- "What's really going on with Riley emotionally beneath the marathon and promotion?" — requires reading between the lines of surface-level bravado

**Borderline hard (could be medium with more evidence)**:
- "What does 2026 look like for Dev and Casey?" — mostly just listing planned events from late sessions
- "How has Dev's cooking evolved?" — sequential tracking, not much inference needed

**Could a system with no gate (stores everything) ace the hard questions?**

Partially. A system that stores all 2,000 messages and has strong retrieval+reasoning would do well on temporal and life_event hard questions (which are mostly about finding and ordering information). But comparison and recommendation hard questions require actual reasoning about personality, preferences, and relationships — things that emerge from pattern matching across many messages, not from any single message. A no-gate system would have more raw material but also more noise to sift through, which could actually hurt on comparison questions where irrelevant social chatter might confuse the synthesis.

**Verdict**: The hard questions are meaningfully harder than medium. The main differentiator is that hard questions can't be answered by quoting 2-3 messages — they require the system to build a mental model of a person and apply it.
