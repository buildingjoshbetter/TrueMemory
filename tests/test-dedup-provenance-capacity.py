"""Synthetic control for dedup provenance on a fact restated many times.

Each ingestion of a growing session cites a restated fact from a new
extraction input, and a dedup SKIP appends that citation to the stored
memory. A citation is about 1.5 KB, so the 32 KiB provenance bound fills after
about 22 restatements. From then on every SKIP raised ProvenanceError, logged
"Dedup provenance write failed (ProvenanceError)" and became storage_failed:
nothing was written, but the session's ingestion could never be certified and
every retry paid for the extraction again. A SKIP adds no content, so a memory
whose citations are full now keeps them unchanged and the SKIP completes.
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
