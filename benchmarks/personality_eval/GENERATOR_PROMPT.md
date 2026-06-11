# PersonaLoCoMo Generator Prompt

> Paste everything below this line into a fresh Claude Code session.

---

# PERSONALOCOMO — Personality Evaluation Benchmark Generator

You are building PersonaLoCoMo, a personality evaluation benchmark for TrueMemory at `/Users/j/Desktop/TrueMemory`. This is the definitive benchmark for testing whether a memory system actually knows the people in a user's life — not just the facts they mentioned, but who they ARE.

## THE CORE PRINCIPLE: NATURALNESS

Every question in this benchmark must sound like something a real person would actually ask their AI assistant about someone in their life. Not a researcher. Not a test engineer. A person.

**Natural (GOOD):**
- "What should I cook for dinner when Sarah comes over?"
- "Would Mike enjoy this hiking trail I found?"
- "Is Jordan more of a morning person or night owl?"
- "What's been going on with Sam lately? He seemed off."
- "What would be a good birthday gift for Alex?"
- "How do Caroline and Melanie differ when it comes to staying active?"

**Synthetic-sounding (BAD — reject these):**
- "Analyze Caroline's dietary preferences across sessions 3-15."
- "What personality traits does Speaker A exhibit?"
- "Compare the communication formality levels of both participants."
- "Based on temporal patterns, what routines does the entity maintain?"
- "Evaluate the emotional valence distribution of Melanie's messages."

The difference: natural questions come from a place of caring about someone, wanting to do something nice for them, trying to understand them, or just being curious about the people in your life. Synthetic questions come from a place of "I'm testing a system."

**Naturalness test for every question:** Would a person actually say this out loud to their phone? If not, rewrite it.

## WHAT THIS BENCHMARK TESTS

A user has been chatting with their AI assistant about their life. Over months of conversations, they've mentioned friends, family, coworkers — how they spent their weekend, what they ate, who they hung out with, what's stressing them out. The L0 personality layer's job is to build an understanding of these people so that when the user asks about them, the system can answer from genuine understanding.

This benchmark tests that understanding across 10 real-feeling conversations, each between 2 people over 19-32 sessions spanning months. 2000 questions. All personality. All natural.

## CONVERSATION DATA

Located at `/Users/j/Desktop/TrueMemory/benchmarks/locomo/data/locomo10.json`.

```python
import json
with open("benchmarks/locomo/data/locomo10.json") as f:
    data = json.load(f)

conv = data[0]
c = conv["conversation"]
speaker_a, speaker_b = c["speaker_a"], c["speaker_b"]
# Sessions: c["session_1"], c["session_2"], etc.
# Each: [{"speaker": "Caroline", "text": "..."}, ...]
# Dates: c["session_1_date_time"] = "10:00 AM on 15 August, 2022"
```

| Conv | Speakers | Sessions | Messages |
|------|----------|----------|----------|
| 0 | Caroline & Melanie | 19 | 419 |
| 1 | Jon & Gina | 19 | 369 |
| 2 | John & Maria | 32 | 663 |
| 3 | Joanna & Nate | 29 | 629 |
| 4 | Tim & John | 29 | 680 |
| 5 | Audrey & Andrew | 28 | 675 |
| 6 | James & John | 31 | 689 |
| 7 | Deborah & Jolene | 30 | 681 |
| 8 | Evan & Sam | 25 | 509 |
| 9 | Calvin & Dave | 30 | 568 |

## QUESTION CATEGORIES (10 categories × 20 questions per conv = 200 per conv)

### 1. food_and_drink (20 per conv)
What they eat, cook, drink, where they go. Dietary patterns, favorite spots, cooking habits.
- "What should I make for dinner if Jordan's coming over?"
- "Does Sarah drink coffee or tea?"
- "Where would Sam want to go for a birthday dinner?"
- "Has Evan's diet changed at all recently?"

### 2. hobbies_and_interests (20 per conv)
What they do for fun, what they're passionate about, what they watch/read/play.
- "What would be a good gift for Mike — what's he into these days?"
- "Would Alex enjoy this escape room I found?"
- "What does Nate do on weekends when he's not working?"
- "Has Joanna picked up any new hobbies lately?"

### 3. communication_style (20 per conv)
How they text/talk. Are they formal? Casual? Emoji-heavy? Brief or long-winded? How do they show excitement vs frustration?
- "Does Caroline use a lot of emojis or is she more of a plain-text person?"
- "Who's more likely to send a one-word reply — Tim or John?"
- "How does Deborah usually start a conversation?"
- "Is Gina more formal or casual when she texts?"

### 4. personality_and_character (20 per conv)
What kind of person are they? This is the "tell me about X" category.
- "What kind of person is Jordan?"
- "Is Sam more of an optimist or a pessimist?"
- "Would you say Melanie is more organized or spontaneous?"
- "What are Andrew's best qualities as a friend?"

### 5. daily_life_and_routines (20 per conv)
Their day-to-day patterns, exercise habits, work schedule, how they spend a typical week.
- "Is Caroline a morning person?"
- "Does Evan work out regularly?"
- "What does a typical weekend look like for Audrey?"
- "How does John usually unwind after work?"

### 6. relationships_and_people (20 per conv)
The people in their life — family, friends, coworkers, pets. How those relationships work.
- "Who does Maria talk about the most?"
- "Does Andrew have any pets?"
- "How close are Deborah and Jolene — are they best friends or more casual?"
- "Who in Calvin's life seems most important to him?"

### 7. emotions_and_stress (20 per conv)
What makes them happy, anxious, excited, frustrated. How they cope. What weighs on them.
- "What's been stressing Sam out lately?"
- "What makes Joanna really happy?"
- "How does Evan handle it when things get tough?"
- "Has there been anything worrying Caroline recently?"

### 8. life_changes_over_time (20 per conv)
How has this person evolved across the conversation timeline? New interests, life events, attitude shifts. This category requires understanding personality OVER TIME.
- "Has Sam's outlook on health changed over the past few months?"
- "What's new in Andrew's life compared to when they first started chatting?"
- "Has Gina's work situation changed at all?"
- "Is Evan more or less active now than he was earlier?"

### 9. how_they_compare (20 per conv)
Side-by-side comparisons between the two speakers. These are critical — they test whether the system can actually DISTINGUISH between two people.
- "Who's more into fitness — Evan or Sam?"
- "Between Caroline and Melanie, who's more of a homebody?"
- "Who handles stress better — Tim or John?"
- "How do Deborah and Jolene's communication styles differ?"
- "If you had to describe the biggest personality difference between them, what would it be?"

### 10. practical_recommendations (20 per conv)
Questions that require personality understanding to give useful advice. These feel the most like real assistant interactions.
- "I need a gift idea for Nate — what would he love?"
- "I'm planning a group hangout — what activity would both Calvin and Dave enjoy?"
- "What kind of restaurant would be perfect for Andrew and Audrey's date night?"
- "Sam's been down lately — what would cheer him up?"
- "Would Maria enjoy a yoga retreat, or is that not her thing?"

## DIFFICULTY TIERS

### Easy (30% of questions)
Directly stated in 1-2 messages. Someone who read those specific messages could answer.
- "What sport does Evan play?" (he literally said "I went for a run")

### Medium (40% of questions)  
Requires connecting info across 2-5 sessions. Pattern recognition.
- "What does Evan usually do for exercise?" (requires aggregating mentions of running, gym, yoga across multiple sessions)

### Hard (30% of questions)
Requires deep synthesis across 5+ sessions, inferring from indirect evidence, tracking change over time, or understanding implications.
- "How has Evan's approach to health changed since his injury?" (requires tracking the injury mention, his adaptation, his emotional response, his new activities — spread across many sessions)

## QUESTION FORMAT

```json
{
  "id": "conv0_food_001",
  "question": "What should I cook for dinner if Caroline's coming over?",
  "category": "food_and_drink",
  "target_entity": "Caroline",
  "conversation_idx": 0,
  "speaker_a": "Caroline",
  "speaker_b": "Melanie",
  "gold_answer": "Caroline seems to enjoy Mediterranean-style food — she mentions making homemade pasta in session 5, raves about a Greek place in session 8, and describes a Mediterranean salad she made in session 15. She doesn't mention fast food or heavy meat dishes. A homemade pasta or a fresh salad with Mediterranean flavors would probably be a hit.",
  "evidence_sessions": [5, 8, 15],
  "difficulty": "medium"
}
```

**Gold answer requirements:**
- Written in a natural, conversational tone — like a friend telling you about someone
- Cites specific evidence (session numbers, paraphrased message content)
- 2-4 sentences. Enough to be specific, not so long it's an essay
- For recommendation questions, actually gives the recommendation with reasoning
- For comparison questions, addresses both people with specific evidence for each
- NEVER says "not enough information" — if there's not enough info, don't ask that question

## EXECUTION

### Phase 1: Read and Understand

Before generating ANY questions for a conversation, read the ENTIRE conversation carefully. Understand:
- Who are these two people?
- What do they care about?
- How do they talk?
- What's happening in their lives?
- How have things changed over time?
- What's the dynamic between them?

Write a brief 5-sentence character sketch of each speaker BEFORE generating questions. This ensures you actually understand them, not just their keywords. Store these sketches in the checkpoint JSON under `character_sketches` (keyed by `conv_{idx}` → speaker name → sketch string). Include them in every LLM generation call so the model has consistent context.

### Phase 2: Generate Questions

For each conversation, make 3-4 LLM calls to OpenRouter:

```python
import openai, os, json
client = openai.OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    timeout=180.0,
)
```

Model: `openai/gpt-4.1-mini`, temperature **0.0** (deterministic — benchmark must be reproducible), max_tokens 16000.

**CONTEXT WINDOW MANAGEMENT (CRITICAL):**
gpt-4.1-mini has ~128k token context. Large conversations (Conv 4/6/7 have 680+ messages) will overflow if sent whole. You MUST measure before each call:
- Estimate tokens: `len(conversation_text) // 3` (rough chars-to-tokens)
- Budget: 128k total - 4k prompt overhead - 16k output = ~108k for conversation text
- If conversation exceeds 108k tokens (~324k chars), truncate to fit:
  - Keep the FIRST 40% of sessions and LAST 40% of sessions intact
  - Drop middle 20% of sessions
  - Add a note: "[Sessions {X}-{Y} omitted for context limits — {Z} sessions retained]"
- For safety, split into **6 calls** (not 4) for conversations with 600+ messages:
  - Calls 1-5: 2 categories each (40 questions per call)
  - Call 6: remaining categories

**Call structure (default for conversations under 500 messages — 4 calls):**
- Call 1: Categories 1-3 (food_and_drink, hobbies_and_interests, communication_style) — 60 questions
- Call 2: Categories 4-6 (personality_and_character, daily_life_and_routines, relationships_and_people) — 60 questions
- Call 3: Categories 7-9 (emotions_and_stress, life_changes_over_time, how_they_compare) — 60 questions
- Call 4: Category 10 (practical_recommendations) — 20 questions

**Call structure (for conversations with 500+ messages — 6 calls):**
- Call 1: Categories 1-2 (40 questions)
- Call 2: Categories 3-4 (40 questions)
- Call 3: Categories 5-6 (40 questions)
- Call 4: Categories 7-8 (40 questions)
- Call 5: Category 9 (20 questions)
- Call 6: Category 10 (20 questions)

**Prompt template for each call:**

```
You are helping build a personality benchmark. Read this conversation between {speaker_a} and {speaker_b}, then generate personality questions that sound like a real person asking their AI assistant about people in their life.

CHARACTER SKETCHES (from your Phase 1 analysis):
{speaker_a}: {sketch_a}
{speaker_b}: {sketch_b}

Generate exactly {count} questions across these categories:
{categories_for_this_call}

Rules:
- Questions must sound NATURAL — like someone talking to their phone, not a researcher
- Every answer must be grounded in actual message content (cite session numbers)
- Gold answers should read like a friend explaining someone to you — conversational, specific, 2-4 sentences
- Cover both speakers roughly equally
- Mix difficulties: ~30% easy, ~40% medium, ~30% hard
- For "how_they_compare" questions, always discuss BOTH people with evidence
- For "practical_recommendations" questions, actually give the recommendation

IMPORTANT: Do NOT duplicate any of these existing LoCoMo questions for this conversation:
{locomo_existing_questions}

Output ONLY a JSON array of question objects (format shown above). No other text.

CONVERSATION:
{conversation_text}
```

**LoCoMo overlap prevention:** Before each LLM call, load the existing LoCoMo QA pairs for that conversation from `data[conv_idx]["qa"]` and include them in the prompt (as shown above). Extract just the question text, formatted as a bullet list. This gives the generating LLM explicit visibility into what's already been asked.

**Post-generation dedup:** After generating all questions for a conversation, compute word-overlap similarity between each PersonaLoCoMo question and each LoCoMo question for that conversation. Flag any pair with >70% word overlap. Remove flagged PersonaLoCoMo questions.

### Phase 3: Checkpoint

After EACH conversation, save to `/Users/j/Desktop/TrueMemory/benchmarks/personality_eval/data/personality_eval.json`:

```json
{
  "benchmark": "personalocomo",
  "version": "1.0",
  "source": "locomo10",
  "generator_model": "openai/gpt-4.1-mini",
  "generation_date": "2026-04-27",
  "target_total": 2000,
  "categories": ["food_and_drink", "hobbies_and_interests", "communication_style", "personality_and_character", "daily_life_and_routines", "relationships_and_people", "emotions_and_stress", "life_changes_over_time", "how_they_compare", "practical_recommendations"],
  "character_sketches": {
    "conv_0": {"Caroline": "...", "Melanie": "..."},
    ...
  },
  "done_convs": [0, 1, 2],
  "total_questions": 600,
  "questions": [...]
}
```

If the file already exists with some `done_convs`, skip those conversations and continue.

### Phase 3.5: Gold Answer Spot-Check

After each conversation's questions are generated, verify a random sample of gold answers against the actual conversation content:

1. Pick 10 random questions from the just-generated batch
2. For each, read the cited `evidence_sessions` from the actual LoCoMo conversation
3. Check: does the session content actually support the gold answer's claims?
4. If >2 out of 10 have fabricated or unsupported evidence, regenerate that conversation's questions

This catches the main risk of LLM-generated benchmarks: hallucinated evidence. The generating LLM writes both the question and the answer in one call, so it can cite "session 12" without session 12 actually containing what it claims. This spot-check catches that.

### Phase 3.6: Edge Case Handling

- **Sparse personality content:** If a conversation is mostly logistics/scheduling with little personality signal, reduce the target to 120 questions (skip the hardest categories) and note it in the report. Do not force 200 personality questions from a conversation that doesn't support them.
- **Under-generation:** If the LLM returns fewer than the target for a call, make one retry. If still short, accept what you have and note the gap.
- **Very similar speakers:** If both speakers have very similar personalities/interests, lean harder into communication_style and how_they_compare categories (where subtle differences matter) and lighter on preferences (where they'd overlap).
- **Partial call failure:** Track completed calls per conversation in the checkpoint under `partial_calls: {"conv_3": [1, 2]}`. On resume, skip completed calls and continue from where you left off.

### Phase 4: Rustle the Feathers Validation

After ALL 10 conversations are generated, run every check below. Print PASS/FAIL for each. If ANY check fails, fix the issue (regenerate that conversation's questions if needed) and re-validate.

**Check 1 — Scale**
- Total questions >= 1800
- Every conversation has >= 170 questions
- Print per-conversation counts

**Check 2 — Category Balance**
- Every category appears in every conversation
- No category has fewer than 12 or more than 28 questions per conversation
- Print: category × conversation count matrix

**Check 3 — Speaker Balance**
- Per conversation, questions about speaker_a vs speaker_b are 40-60% split
- how_they_compare questions reference BOTH speakers in the question text
- Print: per-conversation speaker distribution

**Check 4 — Difficulty Distribution**
- Per conversation: easy 25-35%, medium 35-45%, hard 25-35%
- Print: difficulty breakdown per conversation

**Check 5 — Gold Answer Quality**
- Every gold_answer is >= 30 characters
- Every gold_answer contains at least one specific detail (name, activity, food, place, or session reference)
- No gold_answer starts with "I don't know" / "Not enough information" / "There is no"
- Sample 5 random gold answers per conversation and print them for manual review
- Print: count of weak/short answers

**Check 6 — Naturalness Audit**
- For each conversation, sample **50 random questions** (25% coverage)
- Check that NONE contain these synthetic-sounding patterns:
  - "Based on the conversation..." / "According to the messages..."
  - "Analyze" / "Evaluate" / "Assess"
  - "Speaker A" / "Speaker B" / "the entity" / "the participant"
  - "temporal" / "valence" / "distribution" / "spectrum"
  - "sessions X through Y" (in the QUESTION — fine in gold answers)
- Flag any that sound like a test question rather than a human question
- Print: flagged questions (should be 0)

**Check 7 — Question Uniqueness**
- No exact duplicate questions
- No near-duplicates (>85% word overlap after lowercasing)
- Print: duplicate count (should be 0)

**Check 8 — Evidence Grounding**
- >= 85% of questions have non-empty evidence_sessions
- Referenced session numbers exist in that conversation
- Print: grounding percentage per conversation

**Check 9 — Conversation Timeline Coverage**
- Per conversation, questions reference sessions from throughout the timeline
- At least 50% of sessions should be referenced by at least one question
- Print: session coverage per conversation

**Check 10 — Cross-Persona Discrimination Quality**
- All how_they_compare questions contain both speaker names
- Gold answers for comparison questions discuss both people (not just one)
- Print: comparison question count and both-speakers-in-answer rate

**Check 11 — LoCoMo Deduplication**
- For each conversation, compare all PersonaLoCoMo questions against that conversation's LoCoMo QA pairs
- Compute word overlap (lowercase, split, set intersection / set union)
- Flag any pair with >70% overlap
- Remove flagged questions
- Print: number of questions removed per conversation

**Check 12 — Within-Category Redundancy**
- Within each (conversation, category) group, check for question pairs with >75% word overlap
- These are near-duplicates that would test the same thing twice
- Remove the lower-difficulty duplicate (keep the harder one)
- Print: redundancy count per conversation

**Check 13 — Hard Question Verification**
- Sample 20 random questions labeled "hard" across the full benchmark
- For each, verify it genuinely requires multi-session synthesis (evidence_sessions should list 3+ sessions)
- Hard questions with only 1 evidence session are mislabeled — downgrade to "medium"
- Print: mislabeled count

**Check 14 — Answer Length Distribution**
- Gold answers should be 30-500 characters
- Flag any under 30 (too vague) or over 500 (too verbose)
- Print: distribution summary and outlier count

### Phase 5: Final Output

1. Save final validated benchmark to `/Users/j/Desktop/TrueMemory/benchmarks/personality_eval/data/personality_eval.json`

2. Save report to `/Users/j/Desktop/TrueMemory/benchmarks/personality_eval/data/GENERATION_REPORT.md`:
   - Total questions, per-conversation counts
   - Category breakdown (table)
   - Difficulty breakdown (table)
   - Speaker balance (table)
   - All 14 validation check results
   - Character sketches for all 20 speakers
   - 3 sample questions per category (30 total) showing the range of difficulty
   - Any regeneration notes

3. Print final summary to stdout

## CRITICAL CONSTRAINTS

1. **GROUND TRUTH ONLY.** Every gold answer must be traceable to actual messages. Do not hallucinate facts about the speakers.

2. **NATURAL LANGUAGE ONLY.** Every question must pass the "would a person say this to their phone?" test.

3. **IMPLEMENTATION AGNOSTIC.** The benchmark tests personality understanding in general. It must NOT be tuned to favor any specific approach (char-n-grams, keyword extraction, LLM-based profiling). A perfect memory system that truly understands people should score 100% regardless of implementation.

4. **CHECKPOINT RELIGIOUSLY.** Save after every conversation. This is 10 LLM-heavy iterations — crashes happen.

5. **FIX FAILURES, DON'T SKIP THEM.** If Rustle finds problems, regenerate. The benchmark ships clean or not at all.

6. **NO LOCOMO OVERLAP.** Do not duplicate any question from LoCoMo's existing QA set. Check the `qa` field of each conversation to see what's already asked.

Start with conversation 0 and work through all 10 sequentially.
