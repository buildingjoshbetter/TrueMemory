"""Run selected real control flow with synthetic collaborators; no ML imports."""
from __future__ import annotations

import ast
import contextlib
import dataclasses
import datetime
import enum
import json
import logging
import sqlite3
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
events: list[str] = []


def noop(*args: object, **kwargs: object) -> None:
    return None


def record(name: str):
    def callback(*args: object, **kwargs: object) -> None:
        events.append(name)
    return callback


def module(name: str, values: dict[str, object]) -> types.ModuleType:
    result = types.ModuleType(name)
    result.__dict__.update(values)
    sys.modules[name] = result
    return result


def extract(relative: str, names: set[str], namespace: dict[str, object]) -> None:
    parsed = ast.parse((SOURCE / relative).read_text())
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body.extend(node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names)
    assert len(body) == len(names) + 1
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), relative, "exec"), namespace)


class Action(enum.Enum):
    ADD = "add"
    UPDATE = "update"
    SKIP = "skip"


fact = types.SimpleNamespace(content="Synthetic fixture prefers a morning planning session.", category="preference", confidence=0.95)
decision = types.SimpleNamespace(should_encode=True, encoding_score=0.9, novelty=0.9, salience=0.9, prediction_error=0.9, reason="fixture")
dedup = types.SimpleNamespace(action=Action.ADD, fact=fact.content, existing_id=None, reason="fixture")
fixture_text = "This synthetic conversation contains enough text to exercise the actual ingestion pipeline."
logger = logging.getLogger("audit-fixture")
logger.addHandler(logging.NullHandler())
logger.propagate = False

module("truememory", {})
module("truememory.model_client", {"ensure_server_running": noop})
module("truememory.ingest", {})
module("truememory.ingest.hooks", {})
module("truememory.ingest.hooks._shared", {
    "invalidate_recall_cache": record("invalidate_recall_cache"),
    "mark_session_extracted": record("mark_session_extracted"),
    "clear_backlog_processing": record("clear_backlog_processing"),
})
scope = module("audit_pipeline_fixture", {
    "dataclass": dataclasses.dataclass, "field": dataclasses.field,
    "time": time, "datetime": datetime, "sqlite3": sqlite3, "log": logger,
    "parse_transcript": lambda path: [fixture_text],
    "format_for_extraction": lambda messages: fixture_text,
    "extract_facts_simple": lambda text: [fact],
    "check_duplicate": lambda *args, **kwargs: dedup,
    "_safe_log": lambda text: text,
    "_dedup_store_lock": contextlib.nullcontext,
    "DedupAction": Action,
}).__dict__
extract("truememory/ingest/pipeline.py", {"IngestionPipeline", "IngestionResult"}, scope)


def make_pipeline(fail: bool):
    def add(**kwargs: object) -> None:
        if fail:
            raise sqlite3.OperationalError("database is locked")
        events.append("stored")
    pipeline = scope["IngestionPipeline"].__new__(scope["IngestionPipeline"])
    pipeline.memory = types.SimpleNamespace(_engine=types.SimpleNamespace(add=add), db_path="synthetic")
    pipeline.user_id = "fixture"
    pipeline.llm_config = None
    pipeline.use_llm_dedup = False
    pipeline.gate_enabled = True
    pipeline.gate = types.SimpleNamespace(reset_batch=noop, evaluate=lambda *args: decision)
    return pipeline


failed = make_pipeline(True).ingest_transcript("synthetic", "fixture")
assert failed.facts_stored == 0 and failed.trace[0]["action"] == "storage_failed"
cli_scope = {"__name__": "audit_cli_fixture", "logging": logging, "sys": sys, "Path": Path,
             "_preflight_writable_target": lambda *args, **kwargs: True,
             "ingest": lambda **kwargs: failed, "_print_result": noop, "_cascade_next": noop}
extract("truememory/ingest/cli.py", {"_run_ingest"}, cli_scope)
events.clear()
args = types.SimpleNamespace(verbose=False, transcript=__file__, provider="auto", db="synthetic", trace=None, user="fixture", threshold=0.3, session="fixture")
cli_scope["_run_ingest"](args)
assert events == ["mark_session_extracted", "clear_backlog_processing"]
failure_evidence = {"stored": failed.facts_stored, "trace_action": failed.trace[0]["action"], "cli_events": list(events)}

events.clear()
text_result = make_pipeline(False).ingest_text(fixture_text)
text_events = list(events)
events.clear()
transcript_result = make_pipeline(False).ingest_transcript("synthetic")
transcript_events = list(events)
assert text_result.facts_stored == transcript_result.facts_stored == 1
assert "invalidate_recall_cache" not in text_events
assert "invalidate_recall_cache" in transcript_events

prompts: list[str] = []


def llm_fixture(prompt: str) -> str:
    prompts.append(prompt)
    return "synthetic query one\nsynthetic query two"


refinement_scope: dict[str, object] = {}
extract("truememory/agentic_search.py", {"generate_refined_queries"}, refinement_scope)
sentinel = "ARCHIVE_FIXTURE_PRIVATE_FACT"
refinement_scope["generate_refined_queries"]("Which projects matter?", [{"content": sentinel}], llm_fixture)
assert sentinel in prompts[0]
print(json.dumps({"source_revision": "063e5b8844af735a52fde886217a5d26a0f13064", "method": "AST-selected real functions, synthetic collaborators, no network or model imports", "P01": failure_evidence, "P02": {"text_events": text_events, "transcript_events": transcript_events}, "refinement_prompt": {"retrieved_memory_sentinel_in_llm_prompt": sentinel in prompts[0], "external_request_performed": False}}, indent=2))
