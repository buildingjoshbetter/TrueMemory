"""L4 Candidate protocol — MEMORIST-L4 Phase 8.

An L4 candidate varies what happens POST-ingestion. All candidates
share the same ingest path (standard TrueMemory) at SHA 9b7af17 per
L4_COUPLING_CONTRACT §1. Only consolidate() and retrieve_augmented()
vary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ConsolidateTelemetry:
    """Per-consolidation-pass telemetry returned by L4Candidate.consolidate()."""
    wall_clock_s: float = 0.0
    rows_written: dict = field(default_factory=dict)  # {table: count}
    bytes_added: int = 0
    notes: str = ""


@dataclass
class RetrievalResult:
    """One retrieved item; mirrors run_candidate.py shape."""
    content: str
    score: float
    source: str = "messages"  # messages | summaries | fact_timeline | ...
    metadata: dict = field(default_factory=dict)


class L4Candidate(ABC):
    """Base for L4 consolidation candidates."""

    name: str = "abstract"
    tier: str = "edge"  # edge | base | pro

    def __init__(self, **kwargs):
        self.config = dict(kwargs)

    @abstractmethod
    def consolidate(self, db_path: str | Path) -> ConsolidateTelemetry:
        """Run a consolidation pass on the DB. Measures its own wall-clock."""
        raise NotImplementedError

    @abstractmethod
    def retrieve_augmented(self, query: str, db_path: str | Path,
                           k: int = 10) -> list[RetrievalResult]:
        """Retrieve top-k items for a query, consulting consolidation artifacts."""
        raise NotImplementedError

    def cleanup(self) -> None:
        pass
