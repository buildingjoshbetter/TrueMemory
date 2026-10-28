"""Synthetic, inert actual-source controls for source provenance #745."""

from __future__ import annotations

import ast
import contextlib
import datetime
import enum
import hashlib
import importlib.util
import json
import logging
import os
import sqlite3
import sys
import types
import unittest
from dataclasses import asdict, dataclass, field
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("source_provenance_745_controls", ROOT / "truememory/ingest/provenance.py")
assert SPEC is not None and SPEC.loader is not None
PROVENANCE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROVENANCE
SPEC.loader.exec_module(PROVENANCE)


def selected_source(relative: str, names: set[str], namespace: dict[str, object]) -> dict[str, object]:
    filename = ROOT / relative
    tree = ast.parse(filename.read_text(encoding="utf-8"))
    selected = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    selected.extend(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names)
    if {node.name for node in selected if isinstance(node, (ast.FunctionDef, ast.ClassDef))} != names:
        raise AssertionError("actual-source control selection is incomplete")
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(filename), "exec"), namespace)
    return namespace


EXTRACTOR = selected_source("truememory/ingest/extractor.py", {"ExtractedFact", "_chunk_transcript", "_dedupe_facts_by_content", "extract_facts_outcome"}, {
    "dataclass": dataclass, "field": field, "_CHUNK_CHAR_BUDGET": 100, "_DEFAULT_MAX_CHUNKS": 20,
    "log": logging.getLogger(__name__), "EXTRACTION_PROMPT": "{transcript}", "EXTRACTION_SYSTEM": "synthetic",
    "_neutralize_delimiters": lambda value: value, "LLMError": RuntimeError,
})
Fact = EXTRACTOR["ExtractedFact"]
STORAGE = selected_source("truememory/storage.py", {"_serialize_metadata", "_deserialize_metadata", "_row_to_dict", "insert_message", "update_message"}, {"json": json})
TRANSCRIPT = selected_source("truememory/ingest/transcript.py", {"format_for_extraction"}, {"strip_truememory_blocks": lambda value: value})
DEDUP = selected_source("truememory/ingest/dedup.py", {"DedupAction", "DedupDecision"}, {"Enum": enum.Enum, "dataclass": dataclass})


class SyntheticEngine:
    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY,content TEXT,sender TEXT,recipient TEXT,timestamp TEXT,category TEXT,modality TEXT,directive INTEGER,metadata TEXT)")
        self.writes: list[dict[str, object]] = []

    def add(self, **fields: object) -> dict[str, object]:
        identity = STORAGE["insert_message"](self.conn, fields)
        self.conn.commit()
        self.writes.append(dict(fields))
        return self.get(identity)

    def get(self, identity: int, *, _source_snapshot: bool = False) -> dict[str, object] | None:
        row = self.conn.execute("SELECT id,content,sender,recipient,timestamp,category,modality,directive,metadata FROM messages WHERE id=?", (identity,)).fetchone()
        if row is None:
            return None
        result = STORAGE["_row_to_dict"](row)
        if _source_snapshot:
            result["_source_snapshot_sha256"] = hashlib.sha256(
                json.dumps(list(row), ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        return result

    def update(self, identity: int, *, _expected_source: dict[str, object] | None = None,
               **fields: object) -> dict[str, object] | None:
        if _expected_source is not None and _expected_source != self.get(identity, _source_snapshot=True):
            raise sqlite3.OperationalError("synthetic source snapshot changed")
        if not STORAGE["update_message"](self.conn, identity, **fields):
            return None
        self.writes.append(dict(fields))
        return self.get(identity)

    def close(self) -> None:
        self.conn.close()


class SourceProvenanceControls(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SyntheticEngine()
        self.gate = types.SimpleNamespace(threshold=0.3, w_novelty=0.25, w_salience=0.2, w_prediction_error=0.3, salience_floor=0.1, _pe_available=True, reset_batch=lambda: None)
        self.decision = types.SimpleNamespace(should_encode=True, encoding_score=0.6, novelty=0.7, salience=0.8, prediction_error=0.3, reason="synthetic")
        self.gate.evaluate = lambda content, category: self.decision
        self.observation = PROVENANCE.gate_observation(self.gate, self.decision, True)
        self.now = "2026-10-12T01:00:00+00:00"
        self.invalidations: list[bool] = []
        self.runtime = types.ModuleType("truememory.tier_switch.runtime")
        self.runtime.fact_operation = lambda memory: contextlib.nullcontext()
        self.runtime.raise_if_serving_rejection = lambda error: None
        self.shared = types.ModuleType("truememory.ingest.hooks._shared")
        self.shared.invalidate_recall_cache = lambda: self.invalidations.append(True)
        namespace = {
            "contextlib": contextlib, "datetime": datetime, "sqlite3": sqlite3, "time": __import__("time"),
            "field": field, "dataclass": dataclass, "log": logging.getLogger(__name__), "_safe_log": str,
            "_dedup_store_lock": contextlib.nullcontext, "format_for_extraction": TRANSCRIPT["format_for_extraction"],
            "DedupAction": DEDUP["DedupAction"],
            "os": os, "_set_busy_timeout": lambda memory: None,
            "EncodingGate": lambda **kwargs: self.gate,
        }
        namespace.update({name: getattr(PROVENANCE, name) for name in ("ProvenanceError", "SourceSegment", "fact_metadata", "gate_observation", "merge_metadata", "source_context", "source_timestamp")})
        self.pipeline_namespace = selected_source("truememory/ingest/pipeline.py", {"IngestionPipeline", "IngestionResult", "_invalidate_recall_after_batch"}, namespace)
        self.pipeline = namespace["IngestionPipeline"].__new__(namespace["IngestionPipeline"])
        self.pipeline.memory = types.SimpleNamespace(_engine=self.engine)
        self.pipeline._record_source_provenance = True
        self.pipeline.user_id = "synthetic-account"
        self.pipeline.gate = self.gate
        self.pipeline.gate_enabled = True
        self.pipeline.llm_config = None
        self.pipeline.use_llm_dedup = False
        self.patch = patch.dict(sys.modules, {self.runtime.__name__: self.runtime, self.shared.__name__: self.shared})
        self.patch.start()

    def tearDown(self) -> None:
        self.patch.stop()
        self.engine.close()

    def metadata(self, session: str = "synthetic-session", times: tuple[object, ...] = ("2020-01-02T10:00:00Z",), fact: object | None = None) -> dict[str, object]:
        text = "User: synthetic source material for durable sample preference"
        segments = tuple(PROVENANCE.SourceSegment(0, len(text), index, "human", value) for index, value in enumerate(times))
        context = PROVENANCE.source_context(text, session, segments)
        return PROVENANCE.fact_metadata(context, fact or Fact("Prefers synthetic editor", "preference"), self.observation, self.now)

    def test_backfill_roundtrip_uses_source_time_and_separate_ingestion_time(self) -> None:
        metadata = self.metadata()
        self.pipeline._store_fact("Prefers synthetic editor", Fact("Prefers synthetic editor", "preference"), "synthetic-session", metadata=metadata)
        row = self.engine.get(1)
        self.assertEqual(row["timestamp"], "2020-01-02T10:00:00+00:00")
        self.assertEqual(row["metadata"][PROVENANCE.NAMESPACE]["ingested_at"], self.now)
        self.assertEqual(row["metadata"], metadata)
        self.assertEqual(row["sender"], "synthetic-account")
        self.assertEqual(row["content"], "[preference] Prefers synthetic editor")

    def test_unknown_invalid_naive_and_nonstring_times_are_not_ingestion_time(self) -> None:
        for value in ("", "not-a-date", "2020-01-02T10:00:00", None, 2020):
            with self.subTest(value=value):
                metadata = self.metadata(times=(value,))
                self.assertEqual(PROVENANCE.source_timestamp(metadata), "")
                self.assertEqual(PROVENANCE.source_intervals(metadata), ([], True))

    def test_mixed_coverage_is_explicit_and_scalar_stays_unknown(self) -> None:
        metadata = self.metadata(times=("2020-01-02T10:00:00Z", ""))
        ranges, unknown = PROVENANCE.source_intervals(metadata)
        self.assertEqual(ranges, [("2020-01-02T10:00:00+00:00", "2020-01-02T10:00:00+00:00")])
        self.assertTrue(unknown)
        self.assertEqual(PROVENANCE.source_timestamp(metadata), "")

    def test_synthesized_scope_has_no_exact_speaker_or_supporting_message_claim(self) -> None:
        citation = self.metadata(times=("2020-01-02T10:00:00Z", "2020-01-03T10:00:00Z"))[PROVENANCE.NAMESPACE]["citations"][0]
        self.assertEqual(citation["scope"]["kind"], "extraction_input_scope")
        self.assertEqual(citation["scope"]["message_count"], 2)
        self.assertNotIn("speaker", citation)
        self.assertNotIn("supporting_message_id", citation)
        self.assertEqual(citation["extractor_role_assertion"], "user")

    def test_merge_preserves_disjoint_windows_and_unrelated_metadata(self) -> None:
        first = {"external": {"preserve": True}, **self.metadata()}
        merged, changed = PROVENANCE.merge_metadata(first, self.metadata("synthetic-later", ("2026-01-02T10:00:00Z",)))
        self.assertTrue(changed)
        self.assertEqual(merged["external"], first["external"])
        self.assertEqual(len(PROVENANCE.source_intervals(merged)[0]), 2)
        self.assertEqual(first[PROVENANCE.NAMESPACE]["citations"], merged[PROVENANCE.NAMESPACE]["citations"][:1])

    def test_retry_preserves_original_gate_observation_and_clock_without_write(self) -> None:
        first = self.metadata()
        later = self.metadata()
        later[PROVENANCE.NAMESPACE]["ingested_at"] = "2030-01-01T00:00:00+00:00"
        later[PROVENANCE.NAMESPACE]["last_ingested_at"] = "2030-01-01T00:00:00+00:00"
        later[PROVENANCE.NAMESPACE]["citations"][0]["gate"] = {"status": "disabled"}
        merged, changed = PROVENANCE.merge_metadata(first, later)
        self.assertFalse(changed)
        self.assertEqual(merged, first)

    def test_disabled_gate_does_not_claim_synthetic_ones_are_evaluated_scores(self) -> None:
        disabled = PROVENANCE.gate_observation(self.gate, self.decision, False)
        self.assertEqual(disabled["status"], "disabled")
        self.assertIsNone(disabled["novelty"])
        self.assertIsNone(disabled["prediction_error"])
        self.assertNotIn("reason", disabled)
        self.assertTrue(all(len(value) == 64 for value in self.observation["source_sha256"].values() if value is not None))
        self.gate._pe_available = False
        self.assertEqual(PROVENANCE.gate_observation(self.gate, self.decision, True)["status"], "degraded_open")

    def test_update_merges_citations_atomically_with_content(self) -> None:
        self.engine.add(content="Original synthetic fact", sender="synthetic-account", metadata={"external": 7, **self.metadata()})
        self.pipeline._update_fact(1, "Revised synthetic fact", Fact("Revised synthetic fact"), "synthetic-later", metadata=self.metadata("synthetic-later"))
        row = self.engine.get(1)
        self.assertEqual(row["content"], "Revised synthetic fact")
        self.assertEqual(row["metadata"]["external"], 7)
        self.assertEqual(len(row["metadata"][PROVENANCE.NAMESPACE]["citations"]), 2)
        self.assertEqual(set(self.engine.writes[-1]), {"content", "metadata", "timestamp"})

    def test_skip_adds_only_metadata_and_retry_is_noop(self) -> None:
        self.engine.add(content="Original synthetic fact", metadata=self.metadata())
        incoming = self.metadata("synthetic-later")
        self.assertTrue(self.pipeline._merge_fact_provenance(1, incoming))
        self.assertNotIn("content", self.engine.writes[-1])
        writes = len(self.engine.writes)
        self.assertFalse(self.pipeline._merge_fact_provenance(1, incoming))
        self.assertEqual(len(self.engine.writes), writes)

    def test_citation_overflow_elides_middle_citations_in_the_update_write(self) -> None:
        # An update is never dropped because the citation list is full.
        metadata = self.metadata()
        scope = metadata[PROVENANCE.NAMESPACE]["citations"][0]["scope"]
        metadata[PROVENANCE.NAMESPACE]["citations"] = [{"id": f"{index:064x}", "scope": scope} for index in range(PROVENANCE.MAX_CITATIONS)]
        self.engine.add(content="Original synthetic fact", metadata=metadata)
        incoming = self.metadata("overflow")
        self.pipeline._update_fact(1, "Changed", Fact("Changed"), "overflow", metadata=incoming)
        row = self.engine.get(1)
        namespace = row["metadata"][PROVENANCE.NAMESPACE]
        ids = [citation["id"] for citation in namespace["citations"]]
        self.assertEqual(row["content"], "Changed")
        self.assertEqual(len(ids), PROVENANCE.MAX_CITATIONS)
        self.assertEqual(ids[0], f"{0:064x}")
        self.assertEqual(ids[-1], incoming[PROVENANCE.NAMESPACE]["citations"][0]["id"])
        self.assertEqual({key: namespace["elided_citations"][key] for key in ("count", "first_id", "last_id")},
                         {"count": 1, "first_id": f"{1:064x}", "last_id": f"{1:064x}"})
        # Content and the new citation land in one write.
        self.assertEqual(set(self.engine.writes[-1]), {"content", "metadata", "timestamp"})

    def test_byte_overflow_and_invalid_session_are_explicit_recoverable_errors(self) -> None:
        metadata = self.metadata()
        metadata[PROVENANCE.NAMESPACE]["padding"] = "x" * PROVENANCE.MAX_NAMESPACE_BYTES
        with self.assertRaises(sqlite3.OperationalError):
            PROVENANCE.merge_metadata({}, metadata)
        with self.assertRaises(sqlite3.OperationalError):
            PROVENANCE.source_context("synthetic", "x" * (PROVENANCE.MAX_SESSION_BYTES + 1))

    def test_unknown_future_namespace_never_uses_legacy_timestamp(self) -> None:
        self.assertIsNone(PROVENANCE.source_intervals({"external": 1}))
        self.assertEqual(PROVENANCE.source_intervals({PROVENANCE.NAMESPACE: {"version": 2}}), ([], True))
        with self.assertRaises(PROVENANCE.ProvenanceError):
            PROVENANCE.merge_metadata({PROVENANCE.NAMESPACE: {"version": 2}}, self.metadata())

    def test_unsupported_client_never_silently_stamps_now(self) -> None:
        writes: list[str] = []
        self.pipeline.memory = types.SimpleNamespace(add=lambda **fields: writes.append("called"))
        with self.assertRaises(PROVENANCE.ProvenanceError):
            self.pipeline._store_fact("synthetic fact", Fact("synthetic fact"), "session")
        self.assertEqual(writes, [])

    def test_storage_failure_never_invokes_client_fallback(self) -> None:
        def fail(**fields: object) -> None:
            raise sqlite3.OperationalError("synthetic locked")
        self.engine.add = fail
        calls: list[bool] = []
        self.pipeline.memory.add = lambda **fields: calls.append(True)
        with self.assertRaises(sqlite3.OperationalError):
            self.pipeline._store_fact("synthetic fact", Fact("synthetic fact"), "session")
        self.assertEqual(calls, [])

    def test_pipeline_skip_counter_and_invalidation_follow_actual_write(self) -> None:
        fact = Fact("Synthetic reusable preference")
        text = "User: synthetic durable preference with enough source context for extraction"
        context = PROVENANCE.source_context(text, "session")
        original = PROVENANCE.fact_metadata(context, fact, self.observation, self.now)
        self.engine.add(content=fact.content, metadata=original)
        self.pipeline_namespace["extract_facts_simple"] = lambda value: [fact]
        self.pipeline_namespace["check_duplicate"] = lambda *args, **kwargs: DEDUP["DedupDecision"](DEDUP["DedupAction"].SKIP, fact.content, existing_id=1)
        result = self.pipeline.ingest_text(text, "new-session")
        self.assertEqual(result.facts_skipped_dedup, 1)
        self.assertEqual(result.facts_provenance_updated, 1)
        self.assertEqual(result.facts_updated, 0)
        self.assertEqual(len(self.invalidations), 1)
        retry = self.pipeline.ingest_text(text, "new-session")
        self.assertEqual(retry.facts_skipped_dedup, 1)
        self.assertEqual(retry.facts_provenance_updated, 0)
        self.assertEqual(len(self.invalidations), 1)

    def test_pipeline_failed_skip_is_nonterminal(self) -> None:
        fact = Fact("Synthetic reusable preference")
        self.pipeline_namespace["extract_facts_simple"] = lambda value: [fact]
        self.pipeline_namespace["check_duplicate"] = lambda *args, **kwargs: DEDUP["DedupDecision"](DEDUP["DedupAction"].SKIP, fact.content, existing_id=999)
        result = self.pipeline.ingest_text("User: synthetic durable preference with enough source context for extraction", "session")
        self.assertEqual(result.facts_skipped_dedup, 0)
        self.assertEqual(result.facts_provenance_updated, 0)
        self.assertEqual(result.trace[0]["action"], "storage_failed")

    def test_confirmed_provenance_write_invalidates_after_later_abort(self) -> None:
        result = self.pipeline_namespace["IngestionResult"]()
        with self.assertRaises(RuntimeError):
            with self.pipeline_namespace["_invalidate_recall_after_batch"](result):
                result.facts_provenance_updated = 1
                raise RuntimeError("synthetic later failure")
        self.assertEqual(self.invalidations, [True])

    def test_extraction_dedup_unions_spans_preserving_first_fact_and_order(self) -> None:
        first, second, third = Fact("Same fact"), Fact("same FACT"), Fact("Different fact")
        first.source_spans, second.source_spans = ((0, 12),), ((14, 26),)
        selected = EXTRACTOR["_dedupe_facts_by_content"]([first, second, third])
        self.assertEqual(selected, [first, third])
        self.assertIs(selected[0], first)
        self.assertEqual(first.source_spans, ((0, 12), (14, 26)))

    def test_actual_chunk_extraction_cites_only_attempted_chunks(self) -> None:
        source = "User: " + "a" * 94 + "\n\nUser: " + "b" * 94 + "\n\nUser: " + "c" * 94
        calls: list[str] = []
        EXTRACTOR["complete"] = lambda config, prompt, system: calls.append(prompt) or "synthetic"
        EXTRACTOR["_parse_extraction_response_outcome"] = lambda response, cap: types.SimpleNamespace(facts=[Fact("Repeated synthetic fact")], status="complete", salvaged=False, invalid_items=0, omitted_items=0, omitted_facts=0, error_categories=())
        EXTRACTOR["ExtractionOutcome"] = lambda *args: types.SimpleNamespace(facts=args[0], status=args[1], deferred_chunks=args[7])
        outcome = EXTRACTOR["extract_facts_outcome"](source, object(), max_chunks=2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(outcome.status, "limited")
        self.assertEqual(outcome.facts[0].source_spans, ((0, 100), (102, 202)))
        self.assertEqual(outcome.deferred_chunks, 1)

    def test_actual_transcript_pipeline_keeps_formatted_text_and_source_window(self) -> None:
        messages = [types.SimpleNamespace(role="human", content="I prefer synthetic durable editor settings for all sample projects", timestamp="2020-01-02T10:00:00Z"), types.SimpleNamespace(role="assistant", content="Synthetic observation", timestamp="2020-01-03T10:00:00Z")]
        fact = Fact("Prefers synthetic editor", "preference")
        extracted: list[str] = []
        self.pipeline_namespace["parse_transcript"] = lambda location: messages
        self.pipeline_namespace["extract_facts_simple"] = lambda text: extracted.append(text) or [fact]
        self.pipeline_namespace["check_duplicate"] = lambda *args, **kwargs: DEDUP["DedupDecision"](DEDUP["DedupAction"].ADD, fact.content)
        result = self.pipeline.ingest_transcript("synthetic-no-file-read", "source-session")
        self.assertEqual(result.facts_stored, 1)
        self.assertEqual(extracted, [TRANSCRIPT["format_for_extraction"](messages)])
        self.assertEqual(self.engine.get(1)["timestamp"], "2020-01-03T10:00:00+00:00")
        citation = self.engine.get(1)["metadata"][PROVENANCE.NAMESPACE]["citations"][0]
        self.assertEqual(citation["scope"]["observed_roles"], ["assistant", "human"])
        self.assertEqual(citation["session_id"], "source-session")

    def test_legacy_contribution_remains_unknown_after_new_known_citation(self) -> None:
        self.engine.add(content="Legacy synthetic fact", timestamp="2025-01-01T00:00:00Z", metadata={"external": 7})
        self.assertTrue(self.pipeline._merge_fact_provenance(1, self.metadata()))
        row = self.engine.get(1)
        self.assertTrue(row["metadata"][PROVENANCE.NAMESPACE]["legacy_unknown"])
        self.assertEqual(row["timestamp"], "")
        self.assertEqual(row["metadata"]["external"], 7)
        self.assertTrue(PROVENANCE.source_intervals(row["metadata"])[1])
        self.assertEqual(len(PROVENANCE.source_intervals(row["metadata"])[0]), 1)

    def test_malformed_recognized_coverage_never_becomes_known(self) -> None:
        for field_name, value in (("version", True), ("legacy_unknown", "false"), ("timestamp_semantics", "fact_event_time")):
            with self.subTest(field_name=field_name):
                metadata = self.metadata()
                metadata[PROVENANCE.NAMESPACE][field_name] = value
                self.assertEqual(PROVENANCE.source_intervals(metadata), ([], True))
        for field_name, value in (("unknown_count", 1), ("unknown_count", False), ("status", "invented"), ("end", "2010-01-01T00:00:00Z")):
            with self.subTest(field_name=field_name):
                metadata = self.metadata()
                metadata[PROVENANCE.NAMESPACE]["citations"][0]["scope"]["source_time"][field_name] = value
                self.assertEqual(PROVENANCE.source_intervals(metadata), ([], True))
        metadata = self.metadata()
        metadata[PROVENANCE.NAMESPACE]["citations"][0]["scope"]["kind"] = "precise_support"
        self.assertEqual(PROVENANCE.source_intervals(metadata), ([], True))

    def test_missing_scorer_source_is_explicitly_unavailable(self) -> None:
        with patch.object(PROVENANCE, "_scorer_sources", return_value={"encoding_gate.py": None}):
            observation = PROVENANCE.gate_observation(self.gate, self.decision, True)
        self.assertEqual(observation["source_fingerprint_status"], "unavailable")
        self.assertIsNone(observation["source_sha256"]["encoding_gate.py"])

    def test_repeated_text_in_same_session_with_new_source_time_is_new_evidence(self) -> None:
        first = self.metadata()
        second = self.metadata(times=("2021-01-02T10:00:00Z",))
        merged, changed = PROVENANCE.merge_metadata(first, second)
        self.assertTrue(changed)
        self.assertEqual(len(merged[PROVENANCE.NAMESPACE]["citations"]), 2)
        self.assertEqual(len(PROVENANCE.source_intervals(merged)[0]), 2)

    def test_enormous_score_is_unavailable_without_changing_gate_decision(self) -> None:
        self.assertIsNone(PROVENANCE._number(10**1000))
        self.decision.novelty = 10**1000
        self.gate.threshold = 10**1000
        observed = PROVENANCE.gate_observation(self.gate, self.decision, True)
        self.assertIsNone(observed["novelty"])
        self.assertIsNone(observed["settings"]["threshold"])
        self.assertTrue(observed["passed"])
        self.assertTrue(self.decision.should_encode)

    def test_invalid_span_and_source_encoding_fail_recoverably(self) -> None:
        context = PROVENANCE.source_context("synthetic", "session")
        for span in ((-1, 1), (0, 100), (True, 1)):
            with self.subTest(span=span), self.assertRaises(sqlite3.OperationalError):
                fact = Fact("Synthetic")
                fact.source_spans = (span,)
                PROVENANCE.fact_metadata(context, fact, self.observation, self.now)
        with self.assertRaises(sqlite3.OperationalError):
            PROVENANCE.source_context("synthetic", "\ud800")

    def test_constructor_enables_provenance_unconditionally(self) -> None:
        initialized = self.pipeline_namespace["IngestionPipeline"](
            memory=types.SimpleNamespace(_engine=self.engine), llm_config=object(), user_id="synthetic")
        self.assertTrue(initialized._record_source_provenance)

    def test_private_spans_do_not_expand_public_dataclass_schema(self) -> None:
        fact = Fact("Synthetic fact")
        fact.source_spans = ((0, 10),)
        self.assertEqual(asdict(fact), {"content": "Synthetic fact", "category": "general",
                                     "confidence": "medium", "source_role": "user"})
        self.assertEqual(fact.source_spans, ((0, 10),))


if __name__ == "__main__":
    unittest.main()
