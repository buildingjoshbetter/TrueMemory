"""
TrueMemory - A 6-layer memory system for AI agents.

Quick start::

    from truememory import Memory

    m = Memory()
    m.add("Prefers dark mode", user_id="alex")
    results = m.search("preferences", user_id="alex")

Core modules:
    client         - Simple Memory API (Mem0-compatible interface)
    engine         - Full TrueMemoryEngine with 6-layer search pipeline
    storage        - SQLite + WAL database layer with schema and CRUD operations
    fts_search     - FTS5 full-text search with BM25 ranking and score normalization
    vector_search  - Semantic search via sqlite-vec (Edge: Model2Vec potion-base-8M @ 256d; Base/Pro: Qwen3-Embedding-0.6B @ 256d Matryoshka)
    hybrid         - Reciprocal Rank Fusion combining FTS5 + vector search
    temporal       - L2 temporal reasoning (date parsing, time-window filtering)
    salience       - L4 salience guard (noise filtering, entity disambiguation)
    personality    - L0 Personality Engram (entity profiles, preferences, communication style)
    consolidation  - L5 Consolidation (timelines, contradiction detection, summaries)
    predictive     - Predictive Coding Filter (surprise scoring, noise reduction)
    reranker       - Cross-encoder reranking (default: cross-encoder/ms-marco-MiniLM-L-6-v2; Base/Pro override to Alibaba-NLP/gte-reranker-modernbert-base)
    hyde           - HyDE hypothetical document embeddings for query enhancement
    clustering     - HDBSCAN scene clustering for episode-scoped retrieval
"""

__version__ = "0.4.0"

from truememory.client import Memory
from truememory.storage import (
    create_db, load_messages, load_messages_from_file,
    insert_message, delete_message, update_message,
    get_message, get_message_count,
)
from truememory.fts_search import search_fts, search_fts_by_sender, search_fts_in_range
from truememory.vector_search import init_vec_table, build_vectors, search_vector, build_separation_vectors, search_vector_separation, embed_single
from truememory.hybrid import search_hybrid, reciprocal_rank_fusion
from truememory.temporal import detect_temporal_intent, parse_date_reference, search_temporal, get_timeline, detect_episodes, get_episode_messages, expand_to_episodes, detect_landmark_events
from truememory.salience import apply_salience_guard, compute_message_salience, detect_entities
from truememory.personality import (
    build_entity_profiles, extract_preferences, search_personality,
    get_entity_profile, get_communication_pattern,
    resolve_entity, build_dunbar_hierarchy,
)
from truememory.consolidation import (
    build_entity_timelines, detect_contradictions, build_summaries,
    search_contradictions, search_consolidated,
    build_entity_summary_sheets, build_structured_facts,
)
from truememory.predictive import (
    compute_surprise_score, extract_facts, build_surprise_index,
    get_high_surprise_messages,
)
from truememory.query_classifier import classify_query, get_search_mode, QUERY_TYPES, DEFAULT_WEIGHTS
from truememory.reranker import rerank, rerank_with_fusion, get_reranker
from truememory.hyde import (
    hyde_search, hyde_multi_search,
    generate_hypothetical_doc, generate_multi_hypothetical_docs,
)
from truememory.clustering import cluster_messages, search_clustered, get_cluster_info
from truememory.engine import TrueMemoryEngine

__all__ = [
    "__version__",
    "Memory",
    "TrueMemoryEngine",
]


def __getattr__(name: str):
    """Lazy import for the ingest subpackage.

    The ingest module has heavyweight dependencies (LLM backends,
    encoding gate) that should not be loaded when importing truememory
    for core memory operations. This lazy accessor allows
    ``from truememory.ingest import ingest`` to work without eagerly
    importing the ingest module on ``import truememory``.
    """
    if name == "ingest":
        import importlib
        _ingest = importlib.import_module("truememory.ingest")
        return _ingest
    raise AttributeError(f"module 'truememory' has no attribute {name!r}")
