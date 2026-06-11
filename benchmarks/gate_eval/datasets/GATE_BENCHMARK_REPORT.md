# GateLoCoMo Benchmark Validation Report

Generated: 2026-04-29T03:39:58

## Summary

| Metric | Value |
|--------|-------|
| Total messages | 2000 |
| Total questions | 194 |
| Conversations | 5 |
| Overall noise | 958 (47.9%) |
| Overall signal | 655 (32.8%) |
| Overall borderline | 387 (19.4%) |

## Category Distribution

| Category | Count | Percentage |
|----------|-------|-----------|
| B1 | 152 | 7.6% |
| B2 | 154 | 7.7% |
| B3 | 81 | 4.0% |
| N1 | 500 | 25.0% |
| N2 | 115 | 5.8% |
| N3 | 169 | 8.5% |
| N4 | 114 | 5.7% |
| N5 | 60 | 3.0% |
| S1 | 377 | 18.9% |
| S2 | 77 | 3.9% |
| S3 | 58 | 2.9% |
| S4 | 24 | 1.2% |
| S5 | 119 | 5.9% |

## Per-Conversation Stats

| Conv | Messages | Noise% | Target |
|------|----------|--------|--------|
| conv1 | 400 | 50.0% | 50% |
| conv2 | 400 | 44.0% | 45% |
| conv3 | 400 | 52.2% | 55% |
| conv4 | 400 | 42.2% | 40% |
| conv5 | 400 | 51.0% | 50% |

## Questions Distribution

| Conv | Questions |
|------|-----------|
| conv1 | 39 |
| conv2 | 39 |
| conv3 | 39 |
| conv4 | 37 |
| conv5 | 40 |

## Validation Checks (25 passed, 0 failed)

✅ Total messages ~2000: PASS — 2000
✅   conv1 noise ≈50%: PASS — 50.0%
✅   conv2 noise ≈45%: PASS — 44.0%
✅   conv3 noise ≈55%: PASS — 52.2%
✅   conv4 noise ≈40%: PASS — 42.2%
✅   conv5 noise ≈50%: PASS — 51.0%
✅ All evidence_messages exist: PASS — 0 missing
✅ No duplicate message IDs: PASS — 0 duplicates
✅ No duplicate questions: PASS — 0 duplicates
✅ All 5 conversations in questions: PASS — found 5
✅   conv1 questions ~40: PASS — 39
✅   conv2 questions ~40: PASS — 39
✅   conv3 questions ~40: PASS — 39
✅   conv4 questions ~40: PASS — 37
✅   conv5 questions ~40: PASS — 40
✅ ≥80% evidence refs are signal: PASS — 99.1%
✅   conv1 speaker balance <65%: PASS — max=53.2%
✅   conv2 speaker balance <65%: PASS — max=51.2%
✅   conv3 speaker balance <65%: PASS — max=50.2%
✅   conv4 speaker balance <65%: PASS — max=50.2%
✅   conv5 speaker balance <65%: PASS — max=50.0%
✅ Timestamps chronological within sessions: PASS
✅ Session dates chronological: PASS
✅ No questions with empty evidence: PASS — 0 empty
✅ Naturalness spot check (20 random): PASS — 0 synthetic-sounding

## Sample Messages (naturalness check)

- [N3] Pat: "Sixteen miles?? That's like... a lot."
- [N2] Alex: "ok gotta run, team standup"
- [S1] Alex: "there's a couple interesting ones. Anthropic has a role that looks perfect actua"
- [N1] Pat: "Oh nice. Big family thing?"
- [N1] Maria: "OK you scared me. What about the applications?"
- [N1] Sam: "oh that's cool"
- [B1] Sam: "no I'm glad u told me. when are the follow up tests?"
- [B1] Jordan: "yeah it's fine. at least it's not Derek-level bad lol"
- [N2] Pat: "Lol. Congrats again, Senior Engineer."
- [S2] Alex: "she's great. already asked me what I actually want to work on in my first 1:1"
