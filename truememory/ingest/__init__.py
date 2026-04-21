"""
TrueMemory Ingestion — Biomimetic Memory Encoding
================================================

Automatic memory ingestion for AI agents. The brain doesn't "decide"
to store memories — it stores automatically based on a cascade of
neurochemical filters. This package brings that pattern to TrueMemory.

Three integration layers:

1. **Hooks** (automatic, deterministic)
   Claude Code hooks capture conversations at lifecycle boundaries.
   No LLM cooperation required.

2. **MCP Tools** (on-demand, LLM-driven)
   The existing truememory_store/search tools stay for explicit use.

3. **Background Pipeline** (async, deep)
   After conversations end, the pipeline extracts facts through an
   LLM, filters them through a biomimetic encoding gate, deduplicates
   against existing memories, and stores the survivors.

Quick start::

    from truememory.ingest import ingest

    # Ingest a conversation transcript
    result = ingest("/path/to/transcript.json", user_id="alice")
    print(f"Stored {result.facts_stored} new memories")

    # Or ingest raw text
    from truememory.ingest import ingest_text
    result = ingest_text("User prefers dark mode and uses vim.", user_id="alice")
"""

__version__ = "0.4.0"

from truememory.ingest.pipeline import IngestionPipeline, IngestionResult, save_trace
from truememory.ingest.encoding_gate import EncodingGate, EncodingDecision
from truememory.ingest.extractor import extract_facts, ExtractedFact
from truememory.ingest.transcript import parse_transcript, format_for_extraction, Message
from truememory.ingest.dedup import check_duplicate, DedupAction, DedupDecision
from truememory.ingest.models import LLMConfig, auto_detect


def ingest(
    transcript_path: str,
    user_id: str = "",
    db_path: str | None = None,
    gate_threshold: float = 0.30,
    llm_config: LLMConfig | None = None,
    session_id: str = "",
) -> IngestionResult:
    """
    Ingest a conversation transcript into TrueMemory.

    This is the primary entry point. Call it after a conversation ends
    to extract and store memories automatically.

    Args:
        transcript_path: Path to the conversation transcript file.
        user_id: User identifier for scoping memories.
        db_path: Path to the truememory database. Defaults to ~/.truememory/memories.db.
        gate_threshold: Encoding gate sensitivity (0.0-1.0). Lower = stores more.
        llm_config: LLM configuration. Auto-detects if not provided.
        session_id: Optional session identifier used to tag stored facts
            and the decision trace (the Stop hook passes Claude Code's
            session UUID here).

    Returns:
        IngestionResult with statistics and per-fact decision trace.
    """
    pipeline = IngestionPipeline(
        user_id=user_id,
        db_path=db_path,
        gate_threshold=gate_threshold,
        llm_config=llm_config,
    )
    return pipeline.ingest_transcript(transcript_path, session_id=session_id)


def ingest_text(
    text: str,
    user_id: str = "",
    db_path: str | None = None,
    gate_threshold: float = 0.30,
    llm_config: LLMConfig | None = None,
    session_id: str = "",
) -> IngestionResult:
    """
    Ingest raw text into TrueMemory.

    Useful for processing conversation excerpts or direct fact input
    without a transcript file.
    """
    pipeline = IngestionPipeline(
        user_id=user_id,
        db_path=db_path,
        gate_threshold=gate_threshold,
        llm_config=llm_config,
    )
    return pipeline.ingest_text(text, session_id=session_id)


__all__ = [
    # Top-level functions
    "ingest",
    "ingest_text",
    # Pipeline
    "IngestionPipeline",
    "IngestionResult",
    "save_trace",
    # Encoding gate
    "EncodingGate",
    "EncodingDecision",
    # Extractor
    "extract_facts",
    "ExtractedFact",
    # Transcript
    "parse_transcript",
    "format_for_extraction",
    "Message",
    # Dedup
    "check_duplicate",
    "DedupAction",
    "DedupDecision",
    # Models
    "LLMConfig",
    "auto_detect",
]
