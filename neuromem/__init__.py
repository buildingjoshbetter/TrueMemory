"""
Neuromem - A 6-layer memory system for AI agents.

Quick start::

    from neuromem import Memory

    m = Memory()
    m.add("Prefers dark mode", user_id="alex")
    results = m.search("preferences", user_id="alex")

Core modules:
    client         - Simple Memory API (Mem0-compatible interface)
    engine         - Full NeuromemEngine with 6-layer search pipeline
    storage        - SQLite + WAL database layer with schema and CRUD operations
    fts_search     - FTS5 full-text search with BM25 ranking and score normalization
    vector_search  - Semantic search via sqlite-vec (Base: Model2Vec potion-base-8M; Pro: Qwen3-Embedding-0.6B)
    hybrid         - Reciprocal Rank Fusion combining FTS5 + vector search
    temporal       - L2 temporal reasoning (date parsing, time-window filtering)
    salience       - L4 salience guard (noise filtering, entity disambiguation)
    personality    - L0 Personality Engram (entity profiles, preferences, communication style)
    consolidation  - L5 Consolidation (timelines, contradiction detection, summaries)
    predictive     - Predictive Coding Filter (surprise scoring, noise reduction)
    reranker       - Cross-encoder reranking (default: mixedbread-ai/mxbai-rerank-large-v1)
    hyde           - HyDE hypothetical document embeddings for query enhancement
    clustering     - HDBSCAN scene clustering for episode-scoped retrieval
"""

__version__ = "0.2.1"

from neuromem.client import Memory
from neuromem.storage import (
    create_db, load_messages, load_messages_from_file,
    insert_message, delete_message, update_message,
    get_message, get_message_count,
)
from neuromem.fts_search import search_fts, search_fts_by_sender, search_fts_in_range
from neuromem.vector_search import init_vec_table, build_vectors, search_vector, build_separation_vectors, search_vector_separation, embed_single
from neuromem.hybrid import search_hybrid, reciprocal_rank_fusion
from neuromem.temporal import detect_temporal_intent, parse_date_reference, search_temporal, get_timeline, detect_episodes, get_episode_messages, expand_to_episodes, detect_landmark_events
from neuromem.salience import apply_salience_guard, compute_message_salience, detect_entities
from neuromem.personality import (
    build_entity_profiles, extract_preferences, search_personality,
    get_entity_profile, get_communication_pattern,
    resolve_entity, build_dunbar_hierarchy,
)
from neuromem.consolidation import (
    build_entity_timelines, detect_contradictions, build_summaries,
    search_contradictions, search_consolidated,
    build_entity_summary_sheets, build_structured_facts,
)
from neuromem.predictive import (
    compute_surprise_score, extract_facts, build_surprise_index,
    get_high_surprise_messages,
)
from neuromem.query_classifier import classify_query, get_search_mode, QUERY_TYPES, DEFAULT_WEIGHTS
from neuromem.reranker import rerank, rerank_with_fusion, get_reranker
from neuromem.hyde import (
    hyde_search, hyde_multi_search,
    generate_hypothetical_doc, generate_multi_hypothetical_docs,
)
from neuromem.clustering import cluster_messages, search_clustered, get_cluster_info
from neuromem.engine import NeuromemEngine

__all__ = [
    "__version__",
    "Memory",
    "NeuromemEngine",
]


def __getattr__(name: str):
    """Lazy import for the ingest subpackage.

    The ingest module has heavyweight dependencies (LLM backends,
    encoding gate) that should not be loaded when importing neuromem
    for core memory operations. This lazy accessor allows
    ``from neuromem.ingest import ingest`` to work without eagerly
    importing the ingest module on ``import neuromem``.
    """
    if name == "ingest":
        import importlib
        _ingest = importlib.import_module("neuromem.ingest")
        return _ingest
    raise AttributeError(f"module 'neuromem' has no attribute {name!r}")
