"""Base interface for MEMORIST candidate ingestion-layer implementations.

A candidate is any concrete subclass of `Candidate` that wires up:
  - `ingest(session_messages, db_path)` — push one session into the DB
  - `retrieve(query, db_path, k)` — return ranked rows for a query
  - `consolidate(db_path)` — optional cold-path pass

The harness (`run_candidate.py`) provides a uniform driver that loads a
dataset, calls each candidate method, computes per-candidate telemetry,
and emits a uniform result-JSON shape per spec Phase 8.

Per Phase 0 / Phase 6, the retrieval pipeline is held FIXED across all
candidates (Constraint C6). Subclasses implement `retrieve` by calling
into truememory's existing search stack; the only thing that varies is
what's stored and how.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class IngestTelemetry:
    """Per-session ingest telemetry returned by Candidate.ingest()."""
    n_messages_in: int = 0          # raw session messages
    n_facts_extracted: int = 0       # post-extraction (if applicable; else == messages)
    n_kept: int = 0                  # what actually entered storage
    n_dropped: int = 0               # what was filtered out
    n_updated: int = 0               # dedup UPDATE actions
    n_skipped_dedup: int = 0         # dedup SKIP actions
    drop_reasons: dict = field(default_factory=dict)  # reason → count
    extra: dict = field(default_factory=dict)  # candidate-specific fields


@dataclass
class RetrievalResult:
    """One retrieved item; uniform across candidates."""
    content: str
    score: float                # candidate's own ranking score
    source_session_id: int = -1
    metadata: dict = field(default_factory=dict)


class Candidate(ABC):
    """Abstract base for ingestion-layer candidates.

    Subclasses set `name: str` (used as the result-JSON key and for
    `--candidate <name>` CLI selection).
    """

    name: str = "abstract"

    def __init__(self, **kwargs):
        """Subclasses should accept and store any candidate-specific config
        kwargs (thresholds, weights, etc.) here."""
        self.config = dict(kwargs)

    @abstractmethod
    def ingest(self, session_messages: list[dict], db_path: str | Path,
               session_id: int = 0) -> IngestTelemetry:
        """Ingest one session of messages into the database.

        Args:
            session_messages: list of {"role": "user"|"assistant", "content": str}
            db_path: path to a SQLite database (created if missing)
            session_id: optional session tag for traceability

        Returns:
            IngestTelemetry with per-session counters.
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str, db_path: str | Path,
                 k: int = 10) -> list[RetrievalResult]:
        """Retrieve top-k items for a query. Held fixed across candidates
        unless the candidate explicitly modifies retrieval (most do not)."""
        raise NotImplementedError

    def consolidate(self, db_path: str | Path) -> dict:
        """Optional cold-path consolidation (sleep-replay analog, candidates
        #14 #15). Default: no-op."""
        return {"consolidated": False}

    def cleanup(self) -> None:
        """Optional teardown — close model handles, etc."""
        pass
