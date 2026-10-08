# Published audit catalogue

[Execution tracker #771](https://github.com/buildingjoshbetter/TrueMemory/issues/771) | [Future Deep Search #770](https://github.com/buildingjoshbetter/TrueMemory/issues/770)

Publication date: 2026-10-08. All listed bodies/comments were read back and verified. No production fixes or release are claimed.

[Detailed dossier](https://github.com/buildingjoshbetter/TrueMemory/blob/4d1b80050a6b85229d6abc504461024fa91608c1/docs/audits/2026-10-08-v1/audit-dossier-v3.md)

| ID | GitHub | Action | Finding |
|---|---|---|---|
| R01 | [#735](https://github.com/buildingjoshbetter/TrueMemory/issues/735) | new | perf: model server discards batch controls before embedding and reranking |
| R03 | [#736](https://github.com/buildingjoshbetter/TrueMemory/issues/736) | new | fix: model requests whose deadlines expire in the queue still run inference |
| R04 | [#737](https://github.com/buildingjoshbetter/TrueMemory/issues/737) | new | perf: model server has no bound on accepted request threads or queued bytes |
| R05 | [#738](https://github.com/buildingjoshbetter/TrueMemory/issues/738) | new | fix: CPU OOM retries bypass inference ownership and overlap work on the same model |
| R06 | [#739](https://github.com/buildingjoshbetter/TrueMemory/issues/739) | new | fix: local embedding loads ignore TRUEMEMORY_DEVICE=cpu |
| R07 | [#740](https://github.com/buildingjoshbetter/TrueMemory/issues/740) | new | fix: local MPS fallback moves model devices while another inference is running |
| R08 | [#741](https://github.com/buildingjoshbetter/TrueMemory/issues/741) | new | perf: missing shared-server endpoint silently creates per-client local model copies |
| R09 | [#742](https://github.com/buildingjoshbetter/TrueMemory/issues/742) | new | perf: adaptive throttler sleeps 20 seconds inside a batch admission call |
| R10 | [#743](https://github.com/buildingjoshbetter/TrueMemory/issues/743) | new | fix: MPS resource budgets use physical RAM where PyTorch uses recommended working-set memory |
| R13 | [#297](https://github.com/buildingjoshbetter/TrueMemory/issues/297) | reopen-existing | perf: enforce a whole-process memory budget across CPU inference and recovery |
| A01 | [#744](https://github.com/buildingjoshbetter/TrueMemory/issues/744) | new | docs/architecture: distinguish verbatim benchmark ingestion from extracted-fact auto-capture |
| A02 | [#745](https://github.com/buildingjoshbetter/TrueMemory/issues/745) | new | fix: preserve source timestamps, session provenance, and gate signals on ingested facts |
| A03 | [#746](https://github.com/buildingjoshbetter/TrueMemory/issues/746) | new | fix: scheduled preference extraction reports success while doing no work |
| A04 | [#747](https://github.com/buildingjoshbetter/TrueMemory/issues/747) | new | fix: incremental L0 style vectors are not the advertised mean and depend on insert order |
| A05 | [#748](https://github.com/buildingjoshbetter/TrueMemory/issues/748) | new | fix: memory updates leave stale L0 and consolidated artifacts after source content changes |
| A06 | [#749](https://github.com/buildingjoshbetter/TrueMemory/issues/749) | new | perf: clustering holds the write lock during computation and can commit an empty cache after failure |
| A07 | [#750](https://github.com/buildingjoshbetter/TrueMemory/issues/750) | new | perf: surprise-index rebuild holds the writer lock and loses prior scores on caught failures |
| A08 | [#751](https://github.com/buildingjoshbetter/TrueMemory/issues/751) | new | perf: episode recomputation rewrites the full FTS index twice without content changes |
| A09 | [#752](https://github.com/buildingjoshbetter/TrueMemory/issues/752) | new | fix: separation OOM retries a partially inserted tier-rebuild batch without rollback |
| A10 | [#753](https://github.com/buildingjoshbetter/TrueMemory/issues/753) | new | perf: coalesce automatic consolidation by database and track freshness independently of clustering |
| A11 | [#754](https://github.com/buildingjoshbetter/TrueMemory/issues/754) | new | perf: bound clustered-search materialization without changing the retrieval pool |
| A12 | [#720](https://github.com/buildingjoshbetter/TrueMemory/issues/720) | update-existing | fix: report missing clustering capability honestly in background maintenance health |
| A14 | [#755](https://github.com/buildingjoshbetter/TrueMemory/issues/755) | new | fix: connect named landmark dates to temporal query resolution or narrow the advertised behavior |
| A17 | [#756](https://github.com/buildingjoshbetter/TrueMemory/issues/756) | new | perf: stream rebuild inputs instead of retaining the full message corpus before batching |
| C01 | [#757](https://github.com/buildingjoshbetter/TrueMemory/issues/757) | new | Fresh main installation permits MCP 2 although the server imports removed FastMCP |
| C02 | [#758](https://github.com/buildingjoshbetter/TrueMemory/issues/758) | new | Automatic extraction can send transcript text to cloud providers despite the local-only claim |
| C03 | [#759](https://github.com/buildingjoshbetter/TrueMemory/issues/759) | new | DeepSearch refinement sends retrieved memory excerpts to its LLM, contradicting query-only egress |
| C04 | [#760](https://github.com/buildingjoshbetter/TrueMemory/issues/760) | new | Telemetry described as anonymous includes configured email and stable device identifier |
| C05 | [#761](https://github.com/buildingjoshbetter/TrueMemory/issues/761) | new | Shared transcript allowlist rejects default native Codex and Gemini session paths |
| C06 | [#762](https://github.com/buildingjoshbetter/TrueMemory/issues/762) | new | Codex rollout messages parse as zero conversation turns |
| C07 | [#763](https://github.com/buildingjoshbetter/TrueMemory/issues/763) | new | Shared recall JSON does not match native Cursor, Gemini, or Codex hook output contracts |
| C08 | [#764](https://github.com/buildingjoshbetter/TrueMemory/issues/764) | new | OpenClaw per-prompt bridge sends the wrong key and discards recall output |
| C09 | [#765](https://github.com/buildingjoshbetter/TrueMemory/issues/765) | new | Hermes shell hooks are installed without translating native event payloads or context output |
| C10 | [#766](https://github.com/buildingjoshbetter/TrueMemory/issues/766) | new | Manual integration setup snippets still reproduce previously fixed adapter configuration failures |
| C13 | [#716](https://github.com/buildingjoshbetter/TrueMemory/issues/716) | update-existing | BEAM verification remains sensitive to the weak default judge identified in open #716 |
| C14 | [#722](https://github.com/buildingjoshbetter/TrueMemory/issues/722) | update-existing | Windows Claude Desktop capture still depends on an unverified SessionEnd dispatch path |
| C15 | [#767](https://github.com/buildingjoshbetter/TrueMemory/issues/767) | new | Installers report all models ready even after model downloads fail |
| P01 | [#768](https://github.com/buildingjoshbetter/TrueMemory/issues/768) | new | Ingestion marks sessions complete after failed extraction chunks or database writes |
| P02 | [#769](https://github.com/buildingjoshbetter/TrueMemory/issues/769) | new | Direct text ingestion leaves the recall cache stale after successful writes |
| F01 | [#770](https://github.com/buildingjoshbetter/TrueMemory/issues/770) | new | Deep Search across the complete accessible Claude and Codex conversation archive |
| O01 | [#732](https://github.com/buildingjoshbetter/TrueMemory/issues/732) | update-existing | Device-policy validation update |
| O02 | [#721](https://github.com/buildingjoshbetter/TrueMemory/issues/721) | update-existing | Clustering validation update |

## Publication totals

36 child issues created, five existing open issues updated, and #297 reopened. The separate execution tracker is #771. Eight held findings and the A13/P01 merge remain documented in the dossier. Ten reproduction command clarifications were applied with exact body read-back verification; original body receipts remain in the local audit ledger.
