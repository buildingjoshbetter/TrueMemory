"""L0 Candidate protocol — MEMORIST-L0 Phase 8.

An L0 candidate defines how the per-user speaker profile is built
from a message history and how it is consumed at retrieval time.

All candidates share:
  * Input: list of messages, each {"text", "timestamp", "persona_id"}.
  * Stored artifact: an opaque dict the candidate may shape freely.
    The harness persists it (for auditability and for reproducibility),
    but the candidate owns the shape.
  * Retrieval: given a query + the candidate list + the profile,
    candidate returns a reranked candidate list OR a filtered list.

Per L0_COUPLING_CONTRACT §3 — the schema of entity_profiles is NOT
modified by candidates. Candidates that need extra storage
(e.g., C3 style_vector) declare a `profile_extra_bytes` attribute
so the harness can record it for the privacy block.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Profile:
    """A candidate's stored speaker profile for one persona.

    The `data` dict is opaque to the harness; the harness serializes it
    as JSON for the result file and for intra-persona consistency checks.

    For candidates that store a vector (C3, C4), put the vector in
    `data["style_vector"]` as a list[float]. The harness then computes
    dimension-agnostic consistency via cosine similarity.
    """
    persona_id: str
    data: dict[str, Any] = field(default_factory=dict)
    # Audit / privacy — reported in result JSON.
    bytes_estimate: int = 0
    readable_summary: str = ""  # <= 500 chars, for human audit


@dataclass
class RerankResult:
    """One scored-or-filtered candidate at retrieval time."""
    message_text: str
    source_persona_id: str  # the persona whose corpus produced this message
    score: float
    metadata: dict = field(default_factory=dict)


class L0Candidate(ABC):
    """Base for L0 speaker-profile candidates."""

    name: str = "abstract"
    tier: str = "edge"  # edge | base | pro
    # Flag: does this candidate consume the profile at retrieval time?
    # D0/D1 set this to False.
    consumes_at_retrieval: bool = True

    def __init__(self, **kwargs):
        self.config = dict(kwargs)

    @abstractmethod
    def build_profile(self, persona_id: str,
                      messages: list[dict]) -> Profile:
        """Construct the persona's profile from its full message history.

        `messages` is a list of {"text": str, "timestamp": str, ...}
        items for a SINGLE persona.  The candidate returns a Profile
        that the harness will serialize and index.

        Called once per (candidate, persona, fold) combination in the
        sweep. Candidates should be idempotent — two calls with the
        same inputs return equivalent profiles.
        """
        raise NotImplementedError

    @abstractmethod
    def score_for_personalization(self, query: str,
                                  active_profile: Profile,
                                  candidate_messages: list[dict]
                                  ) -> list[RerankResult]:
        """Given a query issued in the context of `active_profile`,
        score / rerank / filter `candidate_messages`.

        `candidate_messages` is a pool that may contain messages from
        multiple personas (intentionally, to enable leakage detection).
        Each item has {"text", "source_persona_id", "timestamp"}.

        Return a list of RerankResult sorted by descending score.
        Length must equal the input pool length (no dropping, so the
        harness can compare top-k decisions across candidates).
        """
        raise NotImplementedError

    def cleanup(self) -> None:
        """Override if the candidate allocates resources (model weights etc.)."""
        pass
