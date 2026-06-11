# L4_probes — README

Dataset: `benchmarks/gate_eval/datasets/long_horizon_synthetic.json`.
L4 probes added under the `l4_probes` top-level key. Existing
`retrieval_queries` (91 baseline) are preserved but not counted in
L4 metrics.

## Counts (≥ spec targets)

| Probe type      | Count | Target |
|-----------------|-------|--------|
| generalization  | 51    | ≥ 50   |
| contradiction   | 31    | ≥ 30   |
| rehearsal       | 34    | ≥ 30   |
| near_duplicate  | 15    | ≥ 15   |

Total L4 probes: **131** on top of the existing 91 retrieval_queries.

## Authoring method (L4_COUPLING_CONTRACT §3)

- **Option 1: hand-authored.** Produced by
  `build_l4_probes.py` — deterministic templates that read the
  dataset's `planted_facts` metadata only. No LLM was invoked
  during probe authoring. No candidate summarization model was
  involved in probe generation.
- No probe author read session transcript text directly; probes
  were authored from the structured `planted_facts` table alone.
  This keeps candidates (especially abstractive-summary ones) from
  accidentally matching probe text by virtue of seeing similar
  source material.
- Reproducible: `python benchmarks/gate_eval/datasets/build_l4_probes.py`.
  Output checksum is recorded in `l4_probes.meta.checksum_sha256`.

## Probe-type semantics

### `generalization` (51 probes)

Tests whether retrieval can aggregate planted facts scattered
across multiple sessions. Gold answer is a multi-fact set; partial
credit scoring is delegated to the harness.

Subtypes:
- `technical_setup_summary` — language/framework/tools/editor per persona.
- `preferences_summary` — all preferences for a persona.
- `personal_life_summary` — family/home/health aggregated.
- `life_events_summary` — emotional events + role changes.
- `work_trajectory` — work history + current project.
- `lifestyle_summary` — preferences + personal lifestyle facts.
- `profile_overview` — 7-fact comprehensive overview.
- `cross_persona_*` — aggregation across multiple personas (harder).

Scoring: set-overlap of gold_facts vs retrieved content. Candidate
is "correct" if retrieval surfaces ≥ 50% of gold_facts (threshold
overridable in harness).

### `contradiction` (31 probes)

Tests whether retrieval returns the **post-supersession** claim,
not the superseded one, on real contradictions planted in the
dataset. Three real supersessions:
- Alex Park: lives in Boston → moved to NYC in week 3 (11 phrasings)
- Diego Ramos: moved team to GitLab → switched back to GitHub
  week 4 (10 phrasings)
- Priya Iyer: pregnancy in week 2 → miscarriage in week 3
  (8 phrasings; flagged `sensitivity: high`)
- 2 subtle cases (implicit markers, no "switched from X to Y").

Scoring:
- "correct" if retrieval surfaces the `new_fact` / `gold_answer`
  AND does NOT surface `superseded_fact_should_NOT_be_returned`
  as primary.
- Partial credit if the new fact appears in top-3 but alongside
  the old fact.
- "Incorrect" if only the superseded claim is returned.

### `rehearsal` (34 probes)

Stratified by N (rehearsal count) to measure correlation between
recall accuracy and repetition frequency (biomimetic prediction).

- **N=1 (20 probes):** facts asserted in a single session, no
  rehearsal list.
- **N=3 (14 probes):** facts rehearsed across 3 sessions
  (origin + rehearsals list).
- **N=10 (8 probes):** `aspirational_N=true`, `actual_N=3` — the
  synthetic dataset has max N=3. The harness should compute
  rehearsal_correlation treating aspirational probes at actual_N,
  OR drop them from the correlation and note "dataset ceiling
  N=3; true N=10 regime untested." REPORT should reflect this
  honestly.

This means the rehearsal-correlation metric (PREREGISTRATION §3)
is computed with N ∈ {1, 3}, not {1, 3, 10}, in this session.
That is a dataset limitation and will be noted in Phase 13
self-audit.

### `near_duplicate` (15 probes)

Tests whether consolidation can collapse paraphrases while
preserving recall. 15 paraphrase sets (4–5 variants each) for
editor preferences, project stacks, family facts, and allergies.

Scoring: "correct" if the underlying fact is recalled given any
one of the paraphrases; "bonus" if consolidation produces a
single dedup'd summary entry for the set.

## Checksum

Stored in `l4_probes.meta.checksum_sha256`. Recomputed on every
build by hashing the sorted-JSON of the `l4_probes` object. Phase 13
self-audit verifies this.

## Known limitations

1. **Rehearsal N=10 unavailable.** Dataset max N=3. Handled via
   aspirational labeling per above.
2. **Only 3 real supersession chains** — Alex, Diego, Priya. The
   31 contradiction probes exercise these three with many
   phrasings rather than many independent supersessions. This
   means contradiction-probe performance is somewhat vulnerable
   to overfitting a narrow entity set. REPORT should note.
3. **All personas are English-speaking technology workers.** No
   multilingual or non-tech corpus stress in this dataset.
4. **Synthetic dataset built by a different session** (encoding-gate
   MEMORIST Phase 7_2). Quality of planted facts constrains what
   L4 probes can measure.
