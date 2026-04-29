"""
Ingestion Pipeline — The "Sleep Consolidation" Analog
=====================================================

The full pipeline runs AFTER conversations end (cold path), not during
(hot path). This mirrors the brain's consolidation process during sleep:

1. Parse transcript into messages (perception)
2. Extract atomic facts via LLM (deep encoding)
3. Pass each fact through encoding gate (hippocampal filtering)
4. Deduplicate against existing memories (consolidation)
5. Store surviving facts (long-term potentiation)
6. Log the full encoding trace (introspection)

The hot path (during conversation) is handled by hooks — deterministic,
zero-overhead capture. The cold path here is where the deep processing
happens.

Design: Complementary Learning Systems (McClelland et al., 1995)
- Fast capture (hooks/hot path) → hippocampal-like rapid encoding
- Slow consolidation (this pipeline) → neocortical-like pattern extraction
"""

from __future__ import annotations

import contextlib
import datetime
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from truememory.ingest.transcript import parse_transcript, format_for_extraction
from truememory.ingest.extractor import extract_facts, ExtractedFact, extract_facts_simple
from truememory.ingest.encoding_gate import EncodingGate
from truememory.ingest.dedup import check_duplicate, DedupAction
from truememory.ingest.models import LLMConfig, auto_detect

log = logging.getLogger(__name__)


# Optional: fcntl for cross-process locking of the dedup-store critical
# section. On Windows fcntl is unavailable — we fall back to a no-op lock
# there (best-effort; sqlite's busy_timeout still protects writes).
try:
    import fcntl  # type: ignore[import-not-found]
    _HAS_FCNTL = True
except ImportError:  # pragma: no cover — Windows path
    _HAS_FCNTL = False

_LOCK_PATH = Path(os.environ.get(
    "TRUEMEMORY_INGEST_LOCK",
    str(Path.home() / ".truememory" / "ingest.lock"),
))


@contextlib.contextmanager
def _dedup_store_lock():
    """Serialize dedup-then-store across concurrent ingest processes.

    Two overlapping Stop hooks can race: process A searches for duplicates
    of fact X (none found), then spends 5s on an LLM extraction call. While
    A is waiting, process B writes a semantically identical fact Y. A
    finishes, finds no dup (it already searched), and writes a duplicate.
    SQLite's ``busy_timeout`` alone can't prevent this — it only serializes
    single writes, not the read-then-write sequence above.

    Holding this process-wide lock around the check_duplicate + store_fact
    pair makes the sequence atomic across concurrent hooks. On Windows
    (no fcntl) we skip locking and rely on ``busy_timeout`` + the embedding
    similarity check as best-effort protection.
    """
    if not _HAS_FCNTL:
        yield
        return

    try:
        _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Can't create the lock dir — degrade to no-lock rather than abort.
        yield
        return

    try:
        lock_fd = os.open(str(_LOCK_PATH), os.O_CREAT | os.O_RDWR, 0o600)
    except OSError as e:
        log.debug("Could not open ingest lock file %s: %s", _LOCK_PATH, e)
        yield
        return

    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        except OSError as e:
            log.debug("flock acquire failed: %s", e)
        try:
            yield
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        try:
            os.close(lock_fd)
        except OSError:
            pass


def _set_busy_timeout(memory, timeout_ms: int | None = None) -> None:
    """Best-effort: set ``PRAGMA busy_timeout`` on the underlying connection.

    Older truememory versions may not expose ``_engine`` / ``conn``.
    We swallow failures because the busy_timeout is a nice-to-have defence
    against concurrent Stop hooks — the outer :func:`_dedup_store_lock`
    already serializes the critical section, and even without busy_timeout
    sqlite will raise ``OperationalError`` which the caller already catches
    and records in the trace.

    Hunter F35: defaults come from ``storage.DEFAULT_BUSY_TIMEOUT_MS`` so
    this helper and :func:`truememory.storage.create_db` never drift apart.
    """
    if timeout_ms is None:
        from truememory.storage import DEFAULT_BUSY_TIMEOUT_MS
        timeout_ms = DEFAULT_BUSY_TIMEOUT_MS
    try:
        engine = getattr(memory, "_engine", None)
        if engine is None:
            return
        # Ensure the connection exists. Prefer the public API if present.
        ensure = getattr(engine, "_ensure_connection", None)
        if callable(ensure):
            try:
                ensure()
            except Exception:
                pass
        conn = getattr(engine, "conn", None)
        if conn is None:
            return
        conn.execute(f"PRAGMA busy_timeout={int(timeout_ms)}")
    except Exception as e:  # pragma: no cover — defensive
        log.debug("Could not set busy_timeout: %s", e)


@dataclass
class IngestionResult:
    """Complete result of a pipeline run."""
    facts_extracted: int = 0
    facts_encoded: int = 0       # Passed encoding gate
    facts_stored: int = 0        # Actually stored (after dedup)
    facts_updated: int = 0       # Updated existing memories
    facts_skipped_gate: int = 0  # Blocked by encoding gate
    facts_skipped_dedup: int = 0 # Blocked by dedup
    elapsed_seconds: float = 0.0
    trace: list[dict] = field(default_factory=list)  # Per-fact decision log


class IngestionPipeline:
    """
    Full ingestion pipeline with biomimetic encoding gate.

    This is the "sleep consolidation" process — it runs after conversations
    and processes the transcript through deep extraction and filtering.

    Args:
        memory: A truememory Memory instance.
        llm_config: LLM configuration for extraction and dedup.
                     If None, auto-detects the best available backend.
        user_id: User scope for memories.
        gate_threshold: Encoding gate threshold (0.0 - 1.0).
                        Lower = more permissive (stores more).
                        Higher = more selective (stores less).
        use_llm_dedup: Whether to use LLM for dedup decisions.
                       If False, uses heuristic-only dedup (faster, cheaper).
    """

    def __init__(
        self,
        memory=None,
        llm_config: LLMConfig | None = None,
        user_id: str = "",
        gate_threshold: float = 0.30,
        use_llm_dedup: bool = True,
        db_path: str | Path | None = None,
    ):
        # Lazy import to avoid circular deps and allow standalone use
        if memory is None:
            from truememory import Memory
            memory = Memory(path=db_path) if db_path else Memory()

        # Defensive: set PRAGMA busy_timeout so concurrent Stop hooks from
        # two Claude Code sessions don't hit ``database is locked`` on the
        # first write. 10 seconds is generous for embed + single-row
        # inserts; the process-level dedup/store lock below is the primary
        # serialization mechanism, busy_timeout is a fallback for cases
        # where the lock can't be acquired (e.g. Windows).
        _set_busy_timeout(memory)

        self.memory = memory
        self.user_id = user_id
        self.use_llm_dedup = use_llm_dedup

        # Auto-detect LLM if not provided
        if llm_config is None:
            try:
                self.llm_config = auto_detect()
            except RuntimeError:
                log.warning("No LLM backend found — will use heuristic extraction only")
                self.llm_config = None
        else:
            self.llm_config = llm_config

        # Initialize encoding gate
        self.gate_enabled = os.environ.get(
            "TRUEMEMORY_GATE_ENABLED", "1"
        ).lower() in ("1", "true", "yes")
        if not self.gate_enabled:
            log.info("Encoding gate disabled via TRUEMEMORY_GATE_ENABLED=0")
        self.gate = EncodingGate(
            memory=memory,
            threshold=gate_threshold,
            user_id=user_id,
        )

    def ingest_transcript(
        self,
        transcript_path: str | Path,
        session_id: str = "",
    ) -> IngestionResult:
        """
        Ingest a full conversation transcript.

        This is the main entry point — called by the Stop hook after
        a conversation ends.

        Args:
            transcript_path: Path to the transcript file.
            session_id: Optional session identifier for metadata.

        Returns:
            IngestionResult with full statistics and decision trace.
        """
        start = time.time()
        result = IngestionResult()
        # Clear batch-level state between transcripts so prediction error
        # scoring doesn't carry over facts from the previous conversation
        self.gate.reset_batch()

        # 1. Parse transcript
        log.info("Parsing transcript: %s", transcript_path)
        messages = parse_transcript(transcript_path)
        if not messages:
            log.info("Empty transcript, nothing to ingest")
            result.elapsed_seconds = time.time() - start
            return result

        # Filter to conversation messages only (skip tool calls)
        conversation = format_for_extraction(messages)
        if len(conversation.strip()) < 50:
            log.info("Transcript too short for extraction (%d chars)", len(conversation))
            result.elapsed_seconds = time.time() - start
            return result

        # 2. Extract facts
        log.info("Extracting facts from %d messages (%d chars)", len(messages), len(conversation))
        if self.llm_config:
            facts = extract_facts(conversation, self.llm_config)
        else:
            facts = extract_facts_simple(conversation)
        result.facts_extracted = len(facts)
        log.info("Extracted %d candidate facts", len(facts))

        if not facts:
            result.elapsed_seconds = time.time() - start
            return result

        # 3-5. Process each fact through gate and dedup
        for fact in facts:
            trace_entry = {
                "fact": fact.content,
                "category": fact.category,
                "confidence": fact.confidence,
            }

            # 3. Encoding gate
            if self.gate_enabled:
                decision = self.gate.evaluate(fact.content, fact.category)
            else:
                from truememory.ingest.encoding_gate import EncodingDecision
                decision = EncodingDecision(
                    should_encode=True, encoding_score=1.0,
                    novelty=1.0, salience=1.0, prediction_error=1.0,
                    reason="gate disabled",
                )
            trace_entry["gate"] = {
                "passed": decision.should_encode,
                "score": decision.encoding_score,
                "novelty": decision.novelty,
                "salience": decision.salience,
                "prediction_error": decision.prediction_error,
                "reason": decision.reason,
            }

            if not decision.should_encode:
                result.facts_skipped_gate += 1
                trace_entry["action"] = "skipped_gate"
                result.trace.append(trace_entry)
                log.debug("Gate blocked: %s — %s", fact.content[:50], decision.reason)
                continue

            result.facts_encoded += 1

            # 4-5. Deduplication + storage (atomic critical section).
            #
            # Holding the process-level lock around BOTH the dedup search
            # and the subsequent add/update prevents a TOCTOU race where
            # process A checks "no duplicate" for fact X, process B writes
            # fact Y (semantically equal), then A writes X as new. Without
            # this lock, concurrent Stop hooks from overlapping Claude Code
            # sessions accumulate near-duplicate memories.
            with _dedup_store_lock():
                dedup = check_duplicate(
                    fact.content,
                    self.memory,
                    user_id=self.user_id,
                    config=self.llm_config if self.use_llm_dedup else None,
                )
                trace_entry["dedup"] = {
                    "action": dedup.action.value,
                    "reason": dedup.reason,
                    "existing_id": dedup.existing_id,
                }

                # We catch sqlite3.OperationalError around the storage calls
                # so a transient DB lock or "unable to open database file"
                # condition doesn't abort the whole transcript with a stack
                # trace. The pipeline continues with the remaining facts;
                # each failed fact is flagged in the trace as
                # ``storage_failed`` so operators can diagnose what got
                # dropped. See Bug #2 in EDGE_CASE_REPORT.md.
                if dedup.action == DedupAction.ADD:
                    try:
                        self._store_fact(dedup.fact, fact, session_id)
                    except sqlite3.OperationalError as e:
                        log.error(
                            "Storage failed for fact (db=%s): %s — fact=%r",
                            getattr(self.memory, "db_path", "<unknown>"),
                            e,
                            dedup.fact[:120],
                        )
                        trace_entry["action"] = "storage_failed"
                        trace_entry["storage_error"] = {
                            "reason": "db locked" if "locked" in str(e).lower() else str(e),
                            "exception": type(e).__name__,
                        }
                    else:
                        result.facts_stored += 1
                        trace_entry["action"] = "stored"
                        log.info("Stored: %s", dedup.fact[:80])

                elif dedup.action == DedupAction.UPDATE:
                    try:
                        self._update_fact(dedup.existing_id, dedup.fact, fact, session_id)
                    except sqlite3.OperationalError as e:
                        log.error(
                            "Update failed for memory id=%s (db=%s): %s — fact=%r",
                            dedup.existing_id,
                            getattr(self.memory, "db_path", "<unknown>"),
                            e,
                            dedup.fact[:120],
                        )
                        trace_entry["action"] = "storage_failed"
                        trace_entry["storage_error"] = {
                            "reason": "db locked" if "locked" in str(e).lower() else str(e),
                            "exception": type(e).__name__,
                        }
                    else:
                        result.facts_updated += 1
                        trace_entry["action"] = "updated"
                        log.info("Updated [%s]: %s", dedup.existing_id, dedup.fact[:80])

                elif dedup.action == DedupAction.SKIP:
                    result.facts_skipped_dedup += 1
                    trace_entry["action"] = "skipped_dedup"
                    log.debug("Dedup skipped: %s — %s", fact.content[:50], dedup.reason)

            result.trace.append(trace_entry)

        result.elapsed_seconds = round(time.time() - start, 2)
        log.info(
            "Ingestion complete: %d extracted, %d stored, %d updated, %d skipped (gate=%d, dedup=%d) in %.1fs",
            result.facts_extracted, result.facts_stored, result.facts_updated,
            result.facts_skipped_gate + result.facts_skipped_dedup,
            result.facts_skipped_gate, result.facts_skipped_dedup,
            result.elapsed_seconds,
        )
        return result

    def ingest_text(self, text: str, session_id: str = "") -> IngestionResult:
        """
        Ingest a raw text string (not a file path).

        Useful for processing conversation text directly without a file.
        """
        # Write to a temp-like approach — just pass through as string
        start = time.time()
        result = IngestionResult()
        # Clear batch-level state between transcripts so prediction error
        # scoring doesn't carry over facts from the previous conversation
        self.gate.reset_batch()

        if len(text.strip()) < 50:
            result.elapsed_seconds = time.time() - start
            return result

        # Extract facts
        if self.llm_config:
            facts = extract_facts(text, self.llm_config)
        else:
            facts = extract_facts_simple(text)
        result.facts_extracted = len(facts)

        if not facts:
            result.elapsed_seconds = time.time() - start
            return result

        for fact in facts:
            if self.gate_enabled:
                decision = self.gate.evaluate(fact.content, fact.category)
            else:
                from truememory.ingest.encoding_gate import EncodingDecision
                decision = EncodingDecision(
                    should_encode=True, encoding_score=1.0,
                    novelty=1.0, salience=1.0, prediction_error=1.0,
                    reason="gate disabled",
                )
            if not decision.should_encode:
                result.facts_skipped_gate += 1
                continue
            result.facts_encoded += 1

            # Hold the process-level lock across the dedup-then-store pair
            # so concurrent ingest callers don't race and produce duplicates.
            with _dedup_store_lock():
                dedup = check_duplicate(
                    fact.content,
                    self.memory,
                    user_id=self.user_id,
                    config=self.llm_config if self.use_llm_dedup else None,
                )

                if dedup.action == DedupAction.ADD:
                    try:
                        self._store_fact(dedup.fact, fact, session_id)
                    except sqlite3.OperationalError as e:
                        log.error("Storage failed in ingest_text: %s — fact=%r", e, dedup.fact[:120])
                        continue
                    result.facts_stored += 1
                elif dedup.action == DedupAction.UPDATE:
                    try:
                        self._update_fact(dedup.existing_id, dedup.fact, fact, session_id)
                    except sqlite3.OperationalError as e:
                        log.error(
                            "Update failed in ingest_text for memory id=%s: %s — fact=%r",
                            dedup.existing_id, e, dedup.fact[:120],
                        )
                        continue
                    result.facts_updated += 1
                else:
                    result.facts_skipped_dedup += 1

        result.elapsed_seconds = round(time.time() - start, 2)
        return result

    def _store_fact(
        self,
        content: str,
        fact: ExtractedFact,
        session_id: str,
    ) -> None:
        """Store a new fact in truememory.

        Note: truememory's client.Memory.add() currently treats the `metadata`
        parameter as "reserved for future use" (client.py:59) and discards it.
        To preserve the category signal — which is important for retrieval
        weighting — we encode it as a prefix tag in the content itself and
        also pass it through the engine's `category` column when possible.
        """
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Encode category tag inline so it survives to storage
        # Format: "[category] content" — recognizable by humans and LLMs
        tagged_content = f"[{fact.category}] {content}" if fact.category and fact.category != "general" else content

        # Prefer engine-level add() which supports the `category` field.
        # If only the Memory client is available, fall back to that.
        #
        # Important: ``sqlite3.OperationalError`` must propagate upward so
        # the outer loop in ``ingest_transcript`` can record a
        # ``storage_failed`` trace entry. Swallowing it here and retrying
        # through the client would either mask a real failure (if the
        # client also hits the lock) or produce inconsistent "sometimes
        # succeeds, sometimes silently drops" behaviour (Bug #2).
        engine = getattr(self.memory, "_engine", None)
        if engine is not None and hasattr(engine, "add"):
            try:
                engine.add(
                    content=tagged_content,
                    sender=self.user_id or "",
                    timestamp=now,
                    category=fact.category or "",
                )
                return
            except sqlite3.OperationalError:
                # Real DB failure (locked, unwritable, etc.) — propagate so
                # the pipeline loop records it in the trace.
                raise
            except Exception as e:
                # Signature mismatch or other recoverable error — fall back
                # to the client-level API.
                log.warning("Engine-level add failed: %s, falling back to client", e)

        # Client-level fallback (category is lost but content is preserved)
        self.memory.add(
            content=tagged_content,
            user_id=self.user_id or None,
        )

    def _update_fact(
        self,
        existing_id: int | None,
        new_content: str,
        fact: ExtractedFact,
        session_id: str,
    ) -> None:
        """Update an existing memory with new content using the public API."""
        if existing_id is not None:
            # Use the public Memory.update() method (client.py:149)
            try:
                updater = getattr(self.memory, "update", None)
                if callable(updater):
                    result = updater(existing_id, new_content)
                    if result is not None:
                        log.debug("Updated memory %d with: %s", existing_id, new_content[:60])
                        return
                else:
                    log.warning("Memory.update() not available; storing as new")
            except sqlite3.OperationalError:
                # DB lock / "unable to open" — propagate so the pipeline
                # loop records a storage_failed entry (Bug #2).
                raise
            except Exception as e:
                log.warning("Failed to update memory %d: %s, storing as new", existing_id, e)

        # Fallback: store as new if update fails or isn't available
        self._store_fact(new_content, fact, session_id)


def save_trace(result: IngestionResult, output_path: str | Path) -> bool:
    """Save the full ingestion trace to a JSON file for debugging.

    Returns ``True`` on success, ``False`` if the trace could not be written.
    Trace is purely diagnostic — a write failure here must NOT propagate as
    an exception to callers, because that would contaminate the exit code of
    an otherwise-successful ingestion and cause retry loops that duplicate
    facts. See Bug #3 in EDGE_CASE_REPORT.md.
    """
    data = {
        "summary": {
            "facts_extracted": result.facts_extracted,
            "facts_encoded": result.facts_encoded,
            "facts_stored": result.facts_stored,
            "facts_updated": result.facts_updated,
            "facts_skipped_gate": result.facts_skipped_gate,
            "facts_skipped_dedup": result.facts_skipped_dedup,
            "elapsed_seconds": result.elapsed_seconds,
        },
        "trace": result.trace,
    }
    try:
        Path(output_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as e:
        log.warning(
            "Could not write trace file %s: %s (ingestion itself succeeded)",
            output_path, e,
        )
        return False
    log.info("Trace saved to %s", output_path)
    return True
