# MEMORIST Evaluation Datasets

Frozen evaluation datasets for the MEMORIST research session
(`_working/MEMORIST_SPEC.md`). Both files in this directory are immutable
once committed — re-running the build scripts must produce byte-identical
output.

---

## `short_horizon_200.json`

**Phase 7.1 deliverable.** A 200-question subset of LoCoMo
(`benchmarks/locomo/data/locomo10.json`), distributed proportionally
across the 4 numeric categories the TrueMemory paper reports on
(categories 1, 2, 3, 4 — category 5 = open_domain/adversarial is
excluded per the paper's 1,540-question convention).

Build: `python benchmarks/gate_eval/build_short_horizon_200.py` (deterministic, seed=42)

Shape:
- `qa`: 200 question objects with `{conv_id, conv_idx, qa_idx, question, answer, evidence, category}`.
- `convs`: 10 unique LoCoMo conversations the questions reference, inlined for self-containment.
- `meta`: seed, src, distribution targets vs actual, build timestamp.

Category distribution:

| Category | Total in 1,540-set | Sampled (n) | Proportion |
|---|---|---|---|
| 1 | 282 | 37 | 18.5% |
| 2 | 321 | 42 | 21.0% |
| 3 | 96  | 12 |  6.0% |
| 4 | 841 | 109 | 54.5% |
| **Total** | **1,540** | **200** | **100%** |

The numeric categories aren't labeled in the LoCoMo file header, but
based on the MEMORIST_SPEC.md note (~18% single_hop, ~21% multi_hop,
~6% temporal, ~55% open_domain) the mapping is approximately
`{1: single_hop, 2: temporal, 3: ?, 4: multi_hop}`. The harness uses
the numeric IDs directly so the human-readable mapping doesn't affect
scoring.

### Known biases

- **Selection bias:** 200 of 1,540 — confidence intervals will be wider
  than the paper's full-set numbers. Phase 11 Modal validation re-runs
  on the full 1,540 to triangulate.
- **Self-containment:** the inlined `convs` field duplicates the source
  data (~3 MB of JSON). Acceptable trade for ship-as-one-file portability.
- **Score fidelity:** the harness's local precision-at-k metric scores
  on whether the gold answer string appears in the top-k retrieved
  content — a coarser signal than the paper's gpt-4o-mini majority-of-3
  judge. Use the harness for relative-ranking-of-candidates only;
  Phase 11 runs the full judge protocol for absolute numbers.

---

## `long_horizon_synthetic.json`

**Phase 7.2 deliverable.** A purpose-built synthetic dataset that scores
selective ingestion across multi-week realistic Claude-Code-style
sessions — the instrument the field is missing per the TrueMemory paper
§6.4 / §8 (no public benchmark currently rewards a system that chooses
what to ingest).

Build: `env -u ANTHROPIC_API_KEY python benchmarks/gate_eval/build_synthetic_dataset.py --personas 6 --sessions-per-persona 8`

Cost: ~$3 of the $50 MEMORIST budget (uses `claude` CLI via OAuth).

Shape:
- `personas`: 6 hand-authored personas with structured attributes
  (role, location, project, stack, family, allergies, recent_events,
  pets). These are **the ground truth** — the LLM only generates the
  narrative surface, not the underlying facts.
- `planted_facts[persona_name]`: ~10-15 facts per persona, each with
  `{fact, session_id, category, should_remember, superseded_by, rehearsals, emotional_intensity}`.
  Adversarial probes (contradictions, updates, near-duplicates,
  emotional content, rehearsal repetition, trivial noise) are planted
  deterministically.
- `sessions`: 48 sessions (6 personas × 8 sessions) spanning 28 days.
  Each session has `{session_id, persona_name, day_offset, topic, mix_goal, transcript}`.
  Mix proportions per the spec: 40% task / 20% substantive / 30% pleasantry / 10% offtopic.
- `retrieval_queries`: ~90+ queries with `{query_id, asked_after_session, gold_from_session, time_gap_days, gold_answer, question, probe_type}`.
  `probe_type` ∈ `{baseline, contradiction, update, near_duplicate, emotional, rehearsal, noise}`.

### Adversarial probes

Each probe type tests a specific architectural property. The harness
breaks down accuracy by probe_type so candidates' tradeoffs are
visible.

| Probe type | What it tests | Example planted scenario |
|---|---|---|
| `baseline` | basic recall at given time gap | Alex says they prefer vim in session 1; queried in session 3 |
| `contradiction` | does the system surface the LATEST state? | Alex lives in Boston (session 1); Alex moves to NYC (session 5); query in session 7 — answer should be NYC |
| `update` | does the system handle evolving status? | Alex thinking about Meta offer (session 3); Alex accepted Meta (session 5); query in session 7 |
| `near_duplicate` | does the system avoid bloat from paraphrased restatements? | "Emma is 7" + "my daughter is 7" should NOT bloat storage |
| `emotional` | are emotional events preferentially recalled? | Priya's miscarriage (high-emotion) vs neutral preferences |
| `rehearsal` | does mention-frequency strengthen retention? | Alex's location mentioned in 3 sessions vs 1 |
| `noise` | does the system correctly NOT surface trivia? | "complaining about the weather" (session 4) — query "did Alex say anything about weather?" should ideally fail |

### Known biases

- **Persona coverage (n=6):** Far fewer than real production. The 6 personas span
  developer roles + life stages, but are biased toward English-speaking
  tech industry. Generalization to non-tech, non-English populations
  is not evaluated.
- **LLM-generated narrative:** Claude (Haiku 4.5) generates the surface
  text. This introduces stylistic bias that may not match real Claude
  Code session statistics. We rely on **planted facts** (not LLM
  narrative) for ground truth, which mitigates this for accuracy
  scoring but may distort the storage-cost numbers.
- **Mix-proportion enforcement:** the LLM is *prompted* to follow the
  40/20/30/10 mix_goal but compliance is approximate (typically within
  ±10% per category per session). Phase 9 should compute realized
  proportions from the produced data for traceability.
- **Time gap range:** queries fire 3-21 days after the source session
  (within the 4-week window). The paper's hypothesis tests at horizons
  of months-years; this dataset can only directly test ≤4-week
  retention. Long-horizon-projection (10×, 100×) per Phase 10 is by
  reasoning, not direct measurement.
- **Heuristic scoring:** the harness's substring-matching metric is
  loose. A candidate that paraphrases the planted fact (Mem0-style
  extraction) may score worse than a verbatim-substrate candidate
  even when its actual recall is comparable. The Phase 9 results
  must be read with this caveat.

---

## How to add a new evaluation dataset

1. Write a builder script: `benchmarks/gate_eval/build_<name>.py`.
2. Output to `benchmarks/gate_eval/datasets/<name>.json` with at minimum a `meta` block + a queryable structure.
3. Update `benchmarks/gate_eval/run_candidate.py` to add a `run_<name>(...)` function and route to it from `main()`.
4. Document the new dataset in this README with shape + biases + scoring caveats.
