# L5 Predictive-Coding — dataset layer

## Datasets (reused, not forked)

L5 runs against the **same two shared datasets** as the encoding-gate
MEMORIST. Per the MEMORIST-L5 protocol Part 0, L5 does **not** fork the
datasets; it adds sidecar oracle-label files and uses them read-only.

- `short_horizon_200.json` — 10 LoCoMo conversations × 200 QA pairs,
  categories {1-4}, evidence = dia_ids.
- `long_horizon_synthetic.json` — 48 synthetic sessions × 91 retrieval
  queries with `probe_type` ∈ {`baseline`, `noise`, `rehearsal`,
  `contradiction`, `emotional`}.

The long-horizon `probe_type` taxonomy already covers what the MEMORIST-L5
protocol calls "surprise-probe" and "anti-surprise-probe":

| Protocol term       | Existing probe_type      | Interpretation for L5           |
|---------------------|---------------------------|---------------------------------|
| surprise-probe      | `baseline`, `emotional`   | rare fact buried in history; L5 should boost |
| anti-surprise-probe | `rehearsal`                | repeated/high-freq fact; L5 must not hurt |
| noise-probe         | `noise`                    | no valid answer; success = NOT retrieved |
| contradiction-probe | `contradiction`            | contradicts prior; L4 territory, but L5 must not pollute |

No new query files are generated. Phase 9 harness scores by `probe_type`
to make L5-specific metrics legible.

## Oracle surprisal labels

Two layers, per the protocol's dual-oracle rule:

### Layer 1 — **Proxy oracle (this session, real)**
Embedding-based surrogate surprisal. For each message, compute
`proxy_surprisal(m) = 1 − max_cos(v_m, v_{m' ∈ prior_window_500})`
— inverted max-cosine to the nearest prior message within a 500-msg
window. Zero external deps, reproducible, Model2Vec-compatible.

- Script: `build_l5_oracle_proxy.py`
- Output: `l5_oracle_proxy__<dataset>.json` with `{msg_id → proxy_surprisal}`
- Status: **feasible within session** (Model2Vec is already a shipped
  dep; runs in seconds per dataset).

### Layer 2 — **LM oracles (budgeted, post-session-ready)**
The protocol-specified dual-LM oracle: Phi-3.5-mini (Oracle-A) and
Llama-3.2-1B (Oracle-B). Token-level `−log p` averaged per message.

- Script: `build_l5_oracle_lm.py` (skeleton — requires HF + model
  download OR a paid-API key)
- Budget: $1 for each oracle × 2 datasets = ~$2–4 via cheap
  API (OpenRouter / Together / local HF inference).
- Status: **not run in this session** (no external API credentials
  configured). Results in the REPORT.md are labeled as using the
  proxy oracle; the LM-oracle run is an issue draft in `ISSUES.md`.
- The inter-oracle Spearman ceiling (protocol-required) is computed
  *if and only if* both LM oracles run. Until then, proxy-calibration
  is the only calibration number.

## Why a proxy is defensible for this session

1. **Proxy surprisal is a real signal.** The "embedding distance to
   nearest prior" metric is monotonic in content-novelty for the user's
   stream — it directly measures what Cand. 4 (the a-priori favorite)
   *is*. Calibration of Cand. 4 vs the proxy will near-trivially be
   high; calibration of other candidates vs the proxy is the signal of
   interest.
2. **Calibration is diagnostic-only per FORMALIZATION §4.** Retrieval
   lift is the ranking metric. A weak calibration ground truth degrades
   the quality of a report section, not the Phase-9 ranking.
3. **Tier-honesty.** The proxy uses Model2Vec, which every tier
   (including Edge) already runs. The calibration picture it paints is
   the *Edge-tier* picture — the Pro-tier picture (LM-oracle) will
   change rankings on at most one candidate (Cand. 3), and that
   candidate is a-priori expected to underperform (Thrush 2024 prior).

The report states this limitation prominently in §8 Discussion and in
the "Three reasons the recommendation might be wrong" appendix.

## File manifest

```
benchmarks/gate_eval/datasets/
├── short_horizon_200.json                    # reused as-is
├── long_horizon_synthetic.json               # reused as-is
├── l5_oracle_proxy__short_horizon_200.json   # produced Phase 7
├── l5_oracle_proxy__long_horizon_synthetic.json  # produced Phase 7
├── build_l5_oracle_proxy.py                  # proxy builder
├── build_l5_oracle_lm.py                     # LM oracle builder (SKELETON; unused this session)
├── L5_README.md                              # this file
└── README.md                                 # original gate-eval README
```
