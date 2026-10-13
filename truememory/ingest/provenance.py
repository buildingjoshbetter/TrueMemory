"""Bounded input-scope citations, distinct from precise fact attribution."""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

NAMESPACE = "truememory_ingest"
MAX_CITATIONS = 64
MAX_NAMESPACE_BYTES = 32768
MAX_SESSION_BYTES = 256
_MAX_SCORER_BYTES = 1024 * 1024


class ProvenanceError(sqlite3.OperationalError):
    """A citation cannot be persisted without losing existing evidence."""


@dataclass(frozen=True)
class SourceSegment:
    start: int
    end: int
    message_index: int
    role: str
    timestamp: object


@dataclass(frozen=True)
class SourceContext:
    digest: str
    scope_digest: str
    length: int
    session_id: str
    segments: tuple[SourceSegment, ...] = ()


def source_context(text: str, session_id: str, segments: tuple[SourceSegment, ...] = ()) -> SourceContext:
    if not isinstance(session_id, str):
        raise ProvenanceError("source session identifier exceeds provenance bounds")
    try:
        encoded_session, encoded_text = session_id.encode("utf-8"), text.encode("utf-8")
    except UnicodeError as exc:
        raise ProvenanceError("source provenance encoding is invalid") from exc
    if len(encoded_session) > MAX_SESSION_BYTES:
        raise ProvenanceError("source session identifier exceeds provenance bounds")
    scope_identity = json.dumps([
        [segment.start, segment.end, segment.message_index, segment.role, _known_time(segment.timestamp)]
        for segment in segments
    ], separators=(",", ":"))
    return SourceContext(hashlib.sha256(encoded_text).hexdigest(), hashlib.sha256(scope_identity.encode("utf-8")).hexdigest(), len(text), session_id, segments)


def _known_time(value: object) -> str | None:
    if not isinstance(value, str) or not value or len(value) > 64:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(datetime.timezone.utc).isoformat()
    except (ValueError, OverflowError):
        return None


@lru_cache(maxsize=1)
def _scorer_sources() -> dict[str, str | None]:
    package = Path(__file__).parent
    sources: dict[str, str | None] = {}
    for relative in ("encoding_gate.py", "encoding_salience.py", "markers.py", "../salience.py"):
        try:
            with (package / relative).open("rb") as stream:
                data = stream.read(_MAX_SCORER_BYTES + 1)
            sources[relative] = hashlib.sha256(data).hexdigest() if len(data) <= _MAX_SCORER_BYTES else None
        except OSError:
            sources[relative] = None
    return sources


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return float(value) if math.isfinite(value) else None
        except (OverflowError, ValueError):
            return None
    return None


def gate_observation(gate: object, decision: object, enabled: bool) -> dict[str, object]:
    status = "disabled" if not enabled else (
        "degraded_open" if getattr(gate, "_pe_available", True) is False else "evaluated"
    )
    sources = _scorer_sources()
    return {
        "enabled": enabled,
        "status": status,
        "passed": getattr(decision, "should_encode", False) is True,
        "score": _number(getattr(decision, "encoding_score", None)) if enabled else None,
        "novelty": _number(getattr(decision, "novelty", None)) if enabled else None,
        "salience": _number(getattr(decision, "salience", None)) if enabled else None,
        "prediction_error": _number(getattr(decision, "prediction_error", None)) if enabled else None,
        "settings": {
            key: _number(getattr(gate, key, None))
            for key in ("threshold", "w_novelty", "w_salience", "w_prediction_error", "salience_floor")
        },
        "scorer": "truememory.ingest.encoding_gate.EncodingGate",
        "source_fingerprint_status": "observed_files" if all(sources.values()) else "unavailable",
        "source_sha256": dict(sources),
    }


def _scope(context: SourceContext, start: int, end: int) -> dict[str, object]:
    segments = [segment for segment in context.segments if segment.start < end and segment.end > start]
    known = [timestamp for segment in segments if (timestamp := _known_time(segment.timestamp)) is not None]
    unknown = len(segments) - len(known) if segments else 1
    status = "mixed" if known and unknown else "known" if known else "unknown"
    return {
        "kind": "extraction_input_scope",
        "formatted_start": start,
        "formatted_end": end,
        "message_start": segments[0].message_index if segments else None,
        "message_end": segments[-1].message_index if segments else None,
        "message_count": len(segments),
        "observed_roles": sorted({segment.role for segment in segments if segment.role in ("human", "assistant")}),
        "source_time": {
            "status": status,
            "start": min(known) if known else None,
            "end": max(known) if known else None,
            "unknown_count": unknown,
        },
    }


def _check_bounds(namespace: dict[str, object]) -> None:
    citations = namespace.get("citations")
    if not isinstance(citations, list) or len(citations) > MAX_CITATIONS:
        raise ProvenanceError("source citation count exceeds provenance bounds")
    try:
        encoded = json.dumps(namespace, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProvenanceError("source provenance is not valid bounded JSON") from exc
    if len(encoded) > MAX_NAMESPACE_BYTES:
        raise ProvenanceError("source provenance byte count exceeds provenance bounds")


def fact_metadata(context: SourceContext, fact: object, observation: dict[str, object], ingested_at: str) -> dict[str, object]:
    spans = getattr(fact, "source_spans", ()) or ((0, context.length),)
    content = getattr(fact, "content", "")
    if not isinstance(content, str):
        raise ProvenanceError("source fact content is invalid")
    fact_digest = hashlib.sha256(content.strip().lower().encode("utf-8")).hexdigest()
    citations: list[dict[str, object]] = []
    seen: set[str] = set()
    for span in spans:
        if (not isinstance(span, tuple) or len(span) != 2
                or any(type(value) is not int for value in span)
                or not 0 <= span[0] <= span[1] <= context.length):
            raise ProvenanceError("source extraction span is invalid")
        start, end = span
        identity = json.dumps([context.session_id, context.digest, context.scope_digest, fact_digest, start, end], separators=(",", ":"))
        citation_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        if citation_id in seen:
            continue
        seen.add(citation_id)
        citations.append({
            "id": citation_id,
            "session_id": context.session_id or None,
            "formatted_input_sha256": context.digest,
            "source_scope_sha256": context.scope_digest,
            "fact_sha256": fact_digest,
            "scope": _scope(context, start, end),
            "extractor_role_assertion": getattr(fact, "source_role", "") if getattr(fact, "source_role", "") in ("user", "inferred") else "unknown",
            "gate": observation,
        })
        if len(citations) > MAX_CITATIONS:
            raise ProvenanceError("source citation count exceeds provenance bounds")
    namespace: dict[str, object] = {
        "version": 1,
        "ingested_at": ingested_at,
        "last_ingested_at": ingested_at,
        "timestamp_semantics": "source_scope_end",
        "citations": citations,
    }
    _check_bounds(namespace)
    return {NAMESPACE: namespace}


def merge_metadata(existing: object, incoming: dict[str, object]) -> tuple[dict[str, object], bool]:
    if not isinstance(existing, dict):
        raise ProvenanceError("existing source metadata is not an object")
    new = incoming.get(NAMESPACE)
    if not isinstance(new, dict) or type(new.get("version")) is not int or new.get("version") != 1:
        raise ProvenanceError("incoming source provenance version is unsupported")
    _check_bounds(new)
    previous = existing.get(NAMESPACE)
    if previous is None:
        namespace = {**new, "legacy_unknown": True}
        _check_bounds(namespace)
        return {**existing, NAMESPACE: namespace}, True
    if not isinstance(previous, dict) or type(previous.get("version")) is not int or previous.get("version") != 1:
        raise ProvenanceError("existing source provenance version is unsupported")
    _check_bounds(previous)
    old_citations, new_citations = previous["citations"], new["citations"]
    if any(not isinstance(citation, dict) or not isinstance(citation.get("id"), str) for citation in old_citations + new_citations):
        raise ProvenanceError("source citation identity is invalid")
    seen = {citation["id"] for citation in old_citations}
    added = [citation for citation in new_citations if citation["id"] not in seen]
    if not added:
        return dict(existing), False
    namespace = {**previous, "citations": [*old_citations, *added], "last_ingested_at": new["last_ingested_at"]}
    _check_bounds(namespace)
    return {**existing, NAMESPACE: namespace}, True


def source_intervals(metadata: object) -> tuple[list[tuple[str, str]], bool] | None:
    """Individual observed scope windows and unknown coverage; None is legacy."""
    if not isinstance(metadata, dict) or NAMESPACE not in metadata:
        return None
    namespace = metadata[NAMESPACE]
    if (not isinstance(namespace, dict) or type(namespace.get("version")) is not int
            or namespace.get("version") != 1 or type(namespace.get("legacy_unknown", False)) is not bool
            or namespace.get("timestamp_semantics") != "source_scope_end"):
        return [], True
    try:
        _check_bounds(namespace)
    except ProvenanceError:
        return [], True
    citations = namespace.get("citations")
    if not isinstance(citations, list) or not citations or len(citations) > MAX_CITATIONS:
        return [], True
    intervals: list[tuple[str, str]] = []
    unknown = namespace.get("legacy_unknown", False)
    for citation in citations:
        scope = citation.get("scope") if isinstance(citation, dict) else None
        value = scope.get("source_time") if isinstance(scope, dict) else None
        if (not isinstance(scope, dict) or scope.get("kind") != "extraction_input_scope"
                or not isinstance(value, dict) or type(value.get("unknown_count")) is not int
                or value["unknown_count"] < 0):
            return [], True
        start, end = _known_time(value.get("start")), _known_time(value.get("end"))
        status = value.get("status")
        if ((status == "known" and value["unknown_count"] == 0)
                or (status == "mixed" and value["unknown_count"] > 0)) and start is not None and end is not None and start <= end:
            if (start, end) not in intervals:
                intervals.append((start, end))
            unknown |= status != "known"
        elif status == "unknown" and value["unknown_count"] > 0 and value.get("start") is None and value.get("end") is None:
            unknown = True
        else:
            return [], True
    return intervals, unknown


def source_timestamp(metadata: object) -> str:
    coverage = source_intervals(metadata)
    if coverage is None or coverage[1] or not coverage[0]:
        return ""
    return max(end for _, end in coverage[0])
