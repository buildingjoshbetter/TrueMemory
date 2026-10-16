"""Synthetic control for dedup provenance on a fact restated many times.

Each ingestion of a growing session cites a restated fact from a new
extraction input, and a dedup SKIP appends that citation to the stored
memory. A citation is about 1.5 KB, so the 32 KiB provenance bound fills after
about 22 restatements. From then on every SKIP raised ProvenanceError, logged
"Dedup provenance write failed (ProvenanceError)" and became storage_failed:
nothing was written, but the session's ingestion could never be certified and
every retry paid for the extraction again. A SKIP adds no content, so a memory
whose citations are full now keeps them unchanged and the SKIP completes.

An UPDATE must never be dropped for a full list either: it keeps the origin
citation and the newest ones, elides the oldest in between, and records the
elision (count, first and last ids, one window covering them) in the same
write as the new content. A single citation too large for the bound on its
own is truncated behind a marker instead of failing.
"""
from __future__ import annotations

import json
import uuid

import pytest

from truememory.ingest import provenance


def test_a_fact_restated_by_a_growing_session_keeps_completing_once_its_citations_are_full(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TRUEMEMORY_GATE_ENABLED", "0")
    from truememory import Memory
    from truememory.ingest import extractor, pipeline
    from truememory.ingest.dedup import DedupAction, DedupDecision
    from truememory.ingest.hooks import _shared
    from truememory.ingest.models import LLMConfig
    monkeypatch.setattr(_shared, "RECALL_CACHE_PATH", home / "recall_cache.json")
    content = "Synthetic user keeps the release checklist in a paper notebook"
    monkeypatch.setattr(extractor, "extract_facts_outcome", lambda conversation, config, **kwargs: extractor.ExtractionOutcome(
        [extractor.ExtractedFact(content=content)], "complete", total_chunks=1, attempted_chunks=1, successful_chunks=1))
    memory = Memory(path=home / "memories.db")
    stored: list[int] = []

    def dedup(fact, memory, **kwargs):
        if not stored:
            return DedupDecision(action=DedupAction.ADD, fact=fact, reason="synthetic first statement")
        return DedupDecision(action=DedupAction.SKIP, fact=fact, existing_id=stored[0], reason="synthetic restatement")

    monkeypatch.setattr(pipeline, "check_duplicate", dedup)
    ingestion = pipeline.IngestionPipeline(memory=memory, llm_config=LLMConfig(provider="anthropic", model="synthetic",
                                                                               api_key="synthetic"), use_llm_dedup=False)
    session = str(uuid.uuid4())
    transcript = tmp_path / f"{session}.jsonl"
    lines: list[str] = []
    capped = None
    for revision in range(40):
        # The session grows between ingestions, as a long-running one does.
        lines.append(json.dumps({"type": "user", "message": {"role": "user", "content":
                                 f"synthetic note {revision}: the release checklist lives in the paper notebook"}}))
        lines.append(json.dumps({"type": "assistant", "message": {"role": "assistant", "content":
                                 [{"type": "text", "text": f"synthetic acknowledgement {revision}"}]}}))
        transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = ingestion.ingest_transcript(str(transcript), session_id=session, require_complete=True)
        assert [entry["action"] for entry in result.trace] == ["stored" if revision == 0 else "skipped_dedup"], revision
        if revision == 0:
            stored.append(memory.get_all(limit=1)[0]["id"])
        row = memory._engine.get(stored[0])
        assert row["content"] == content
        namespace = row["metadata"][provenance.NAMESPACE]
        size = len(json.dumps(namespace, sort_keys=True, separators=(",", ":")).encode())
        assert len(namespace["citations"]) <= provenance.MAX_CITATIONS and size <= provenance.MAX_NAMESPACE_BYTES
        if capped is None and result.facts_provenance_updated == 0 and revision > 0:
            capped = (revision, namespace)
        elif capped is not None:
            # Existing evidence is never dropped to make room.
            assert namespace == capped[1]
    assert capped is not None and 15 < capped[0] < 40


def test_only_a_full_citation_bound_counts_as_capacity():
    full = {provenance.NAMESPACE: {"version": 1, "citations": [{"id": f"{index:064x}"}
                                                               for index in range(provenance.MAX_CITATIONS)]}}
    incoming = {provenance.NAMESPACE: {"version": 1, "citations": [{"id": "f" * 64}], "last_ingested_at": "synthetic"}}
    with pytest.raises(provenance.ProvenanceCapacityError):
        provenance.merge_metadata(full, incoming)
    # An unsupported or malformed namespace stays a storage failure.
    for existing in ({provenance.NAMESPACE: {"version": 2}}, {provenance.NAMESPACE: {"version": 1, "citations": "x"}}):
        with pytest.raises(provenance.ProvenanceError) as raised:
            provenance.merge_metadata(existing, incoming)
        assert not getattr(raised.value, "at_capacity", False)


def _namespace_bytes(namespace: dict) -> int:
    return len(json.dumps(namespace, sort_keys=True, separators=(",", ":")).encode())


def test_store_and_update_round_trip_at_the_bound(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TRUEMEMORY_GATE_ENABLED", "0")
    from truememory import Memory
    from truememory.ingest import extractor, pipeline
    from truememory.ingest.dedup import DedupAction, DedupDecision
    from truememory.ingest.hooks import _shared
    from truememory.ingest.models import LLMConfig
    monkeypatch.setattr(_shared, "RECALL_CACHE_PATH", home / "recall_cache.json")
    revision = [0]

    def extract(conversation, config, **kwargs):
        text = f"Synthetic user keeps the release checklist in notebook number {revision[0]}"
        return extractor.ExtractionOutcome([extractor.ExtractedFact(content=text)], "complete",
                                           total_chunks=1, attempted_chunks=1, successful_chunks=1)

    monkeypatch.setattr(extractor, "extract_facts_outcome", extract)
    memory = Memory(path=home / "memories.db")
    stored: list[int] = []

    def dedup(fact, memory, **kwargs):
        if not stored:
            return DedupDecision(action=DedupAction.ADD, fact=fact, reason="synthetic first statement")
        return DedupDecision(action=DedupAction.UPDATE, fact=fact, existing_id=stored[0], reason="synthetic change")

    monkeypatch.setattr(pipeline, "check_duplicate", dedup)
    ingestion = pipeline.IngestionPipeline(memory=memory, llm_config=LLMConfig(provider="anthropic", model="synthetic",
                                                                               api_key="synthetic"), use_llm_dedup=False)
    session = str(uuid.uuid4())
    transcript = tmp_path / f"{session}.jsonl"
    lines: list[str] = []
    origin = None
    counts: list[int] = []
    first_ids: set[str] = set()
    for revision[0] in range(40):
        day = 1 + revision[0] % 28
        lines.append(json.dumps({"type": "user", "timestamp": f"2026-09-{day:02d}T10:00:00Z", "message": {
            "role": "user", "content": f"synthetic note {revision[0]}: the checklist moved to notebook {revision[0]}"}}))
        lines.append(json.dumps({"type": "assistant", "timestamp": f"2026-09-{day:02d}T10:01:00Z", "message": {
            "role": "assistant", "content": [{"type": "text", "text": f"synthetic acknowledgement {revision[0]}"}]}}))
        transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = ingestion.ingest_transcript(str(transcript), session_id=session, require_complete=True)
        assert [entry["action"] for entry in result.trace] == ["stored" if revision[0] == 0 else "updated"], revision[0]
        if revision[0] == 0:
            stored.append(memory.get_all(limit=1)[0]["id"])
        row = memory._engine.get(stored[0])
        namespace = row["metadata"][provenance.NAMESPACE]
        assert row["content"].endswith(f"notebook number {revision[0]}")
        assert _namespace_bytes(namespace) <= provenance.MAX_NAMESPACE_BYTES
        origin = origin or namespace["citations"][0]["id"]
        assert namespace["citations"][0]["id"] == origin
        elided = namespace.get("elided_citations")
        counts.append(elided["count"] if elided else 0)
        if elided:
            first_ids.add(elided["first_id"])
        assert row["timestamp"] == provenance.source_timestamp(row["metadata"])
    # Past the bound every update elides one more middle citation (the first
    # elision also makes room for its own record), and the first elided
    # citation stays the same across updates.
    full = next(index for index, count in enumerate(counts) if count)
    assert all(later - earlier == 1 for earlier, later in zip(counts[full:], counts[full + 1:]))
    assert len(first_ids) == 1
    assert counts[-1] == 40 - len(namespace["citations"])
    intervals, unknown = provenance.source_intervals(row["metadata"])
    assert not unknown and min(start for start, _ in intervals) == "2026-09-01T10:00:00+00:00"


def test_an_oversized_citation_is_truncated_not_failed():
    text = "User: synthetic source material for a durable preference"
    context = provenance.source_context(text, "synthetic-session", (provenance.SourceSegment(0, len(text), 0, "human",
                                                                                              "2026-10-01T10:00:00Z"),))
    from truememory.ingest.extractor import ExtractedFact
    oversized = {"status": "evaluated", "padding": "x" * (provenance.MAX_NAMESPACE_BYTES + 1)}
    metadata = provenance.fact_metadata(context, ExtractedFact(content="Prefers the paper notebook"), oversized,
                                        "2026-10-01T11:00:00+00:00")
    citation = metadata[provenance.NAMESPACE]["citations"][0]
    assert citation["truncated"] is True and "gate" not in citation
    assert _namespace_bytes(metadata[provenance.NAMESPACE]) <= provenance.MAX_NAMESPACE_BYTES
    assert provenance.source_intervals(metadata) == ([("2026-10-01T10:00:00+00:00", "2026-10-01T10:00:00+00:00")], False)

    # On an update the origin survives and an oversized newcomer is truncated.
    existing = provenance.fact_metadata(provenance.source_context(text, "earlier", ()),
                                        ExtractedFact(content="Prefers the paper notebook"), {}, "2026-09-01T00:00:00+00:00")
    newcomer = {key: value for key, value in citation.items() if key != "truncated"}
    incoming = {provenance.NAMESPACE: {**metadata[provenance.NAMESPACE], "citations": [{**newcomer, "gate": oversized}]}}
    merged, changed = provenance.merge_metadata(existing, incoming, elide=True)
    citations = merged[provenance.NAMESPACE]["citations"]
    assert changed and citations[0] == existing[provenance.NAMESPACE]["citations"][0]
    assert citations[1]["id"] == citation["id"] and citations[1]["truncated"] is True
    with pytest.raises(provenance.ProvenanceCapacityError):
        # A skip never rewrites citations to make room.
        provenance.merge_metadata(existing, incoming)
