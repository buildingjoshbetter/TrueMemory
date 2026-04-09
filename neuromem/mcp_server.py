"""
Neuromem MCP Server
===================

Model Context Protocol server that exposes the Neuromem memory system
as tools for Claude and other MCP-compatible AI assistants.

Usage::

    # Direct
    python -m neuromem.mcp_server

    # Via entry point (after pip install)
    neuromem-mcp

Configuration via environment variables:
    NEUROMEM_DB    Path to .db file (default: ~/.neuromem/memories.db)
    ANTHROPIC_API_KEY   For agentic search via Anthropic (optional)
    OPENROUTER_API_KEY  For agentic search via OpenRouter (optional, fallback)
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ---------------------------------------------------------------------------
# HuggingFace offline mode — skip HTTP freshness checks when models are cached.
# Models are already downloaded on first install; subsequent loads should be
# pure disk reads (~170ms) instead of HTTP round-trips (~600ms+).
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from mcp.server.fastmcp import FastMCP

from neuromem.client import Memory

# ---------------------------------------------------------------------------
# Config management
# ---------------------------------------------------------------------------

_NEUROMEM_DIR = Path.home() / ".neuromem"
_CONFIG_PATH = _NEUROMEM_DIR / "config.json"


def _load_config() -> dict:
    """Load persistent config from ~/.neuromem/config.json."""
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_config(config: dict) -> None:
    """Save config to ~/.neuromem/config.json."""
    _NEUROMEM_DIR.mkdir(parents=True, exist_ok=True)
    _NEUROMEM_DIR.chmod(0o700)
    _CONFIG_PATH.write_text(json.dumps(config, indent=2))
    _CONFIG_PATH.chmod(0o600)


# Apply saved tier on startup (before any model loading)
_startup_config = _load_config()
if "tier" in _startup_config:
    os.environ["NEUROMEM_EMBED_MODEL"] = _startup_config["tier"]

# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "neuromem",
    instructions="""You have access to a persistent memory system (Neuromem). Use it proactively:

STORING MEMORIES (neuromem_store):
- When the user shares personal information, preferences, or facts about themselves, store them immediately without being asked.
- When important decisions are made during a conversation, store them.
- When the user corrects you or clarifies something, store the correction.
- Store each fact as a clear, atomic statement. Prefer "User prefers dark mode" over "The user mentioned something about dark mode."
- Include the user_id parameter when you know who the user is.

RECALLING MEMORIES (neuromem_search):
- At the START of a conversation, call neuromem_search with a broad query to load relevant context.
- You can search multiple topics at once using | separation: "user preferences | project context | recent decisions"
- Multiple queries run in parallel — no speed penalty for combining them.
- Before making recommendations, check if you already have relevant context from prior searches.
- When the user asks "do you remember" or references past conversations, search for the specific topic.

DEEP SEARCH (neuromem_search_deep):
- Use when neuromem_search doesn't find what you need, or for complex multi-part questions.
- Also supports | separated parallel queries.
- Retrieves 5x more candidates internally — best for questions requiring scattered evidence.

FIRST-TIME SETUP:
- When neuromem_stats shows "tier_configured": false, ask the user which tier they want:
  - Base (default): 88.2% accuracy on LoCoMo. Works on any machine. Fast and lightweight.
  - Pro: 91.5% accuracy on LoCoMo. Needs 4GB+ RAM and `pip install neuromem-core[gpu]`.
- Call neuromem_configure with their choice. This only needs to happen once.

You should store and recall memories as naturally as a good assistant who remembers past conversations. Do not ask permission to remember things — just do it.""",
)

_DB_PATH = os.path.expanduser(
    os.environ.get("NEUROMEM_DB", str(Path.home() / ".neuromem" / "memories.db"))
)
_memory: Memory | None = None
_memory_lock = threading.Lock()


def _get_memory() -> Memory:
    """Lazy-init the Memory instance (thread-safe for background preloading)."""
    global _memory
    if _memory is not None:
        return _memory  # Fast path, no lock
    with _memory_lock:
        if _memory is None:
            _memory = Memory(path=_DB_PATH)
        return _memory


# ---------------------------------------------------------------------------
# LLM backend for agentic search (HyDE, query refinement, reranking)
# ---------------------------------------------------------------------------

def _build_llm_fn():
    """Build an llm_fn from available API keys.

    Resolution order for each provider:
      1. Environment variable (ANTHROPIC_API_KEY / OPENROUTER_API_KEY)
      2. Persistent config (~/.neuromem/config.json, written by ``neuromem-ingest setup``)

    Provider priority: Anthropic direct → OpenRouter.
    """
    config = _load_config()

    # Try Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY") or config.get("anthropic_api_key")
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key, timeout=30.0)

            def _anthropic_llm(prompt: str) -> str:
                resp = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=300,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text

            return _anthropic_llm
        except Exception:
            pass

    # Try OpenRouter (OpenAI-compatible API)
    api_key = os.environ.get("OPENROUTER_API_KEY") or config.get("openrouter_api_key")
    if api_key:
        try:
            import httpx

            def _openrouter_llm(prompt: str) -> str:
                resp = httpx.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "anthropic/claude-haiku-4.5",
                        "max_tokens": 500,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]

            return _openrouter_llm
        except Exception:
            pass

    # Try OpenAI
    api_key = os.environ.get("OPENAI_API_KEY") or config.get("openai_api_key")
    if api_key:
        try:
            import httpx

            def _openai_llm(prompt: str) -> str:
                resp = httpx.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "max_tokens": 500,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]

            return _openai_llm
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# Cached LLM function (singleton — avoids rebuilding API client per search)
# ---------------------------------------------------------------------------

_cached_llm_fn = None
_cached_llm_fn_built = False


def _get_llm_fn():
    """Build and cache the LLM function. Only rebuilt on first call."""
    global _cached_llm_fn, _cached_llm_fn_built
    if not _cached_llm_fn_built:
        _cached_llm_fn = _build_llm_fn()
        _cached_llm_fn_built = True
    return _cached_llm_fn


# ---------------------------------------------------------------------------
# Parallel search helper
# ---------------------------------------------------------------------------

# Benchmark-proven internal retrieval limits.
# "Retrieve wide, rerank, present narrow."
_SEARCH_INTERNAL_LIMIT = 100   # Benchmark sweet spot
_DEEP_INTERNAL_LIMIT = 500     # Beyond benchmark — maximum recall

# Tiered rerankers: fast for standard search, heavy for deep search.
_SEARCH_RERANKER = "cross-encoder/ms-marco-MiniLM-L-12-v2"   # 33M, ~0.024s/query
_DEEP_RERANKER = "BAAI/bge-reranker-v2-m3"                   # 568M, ~0.77s/query, 91.5%+


def _set_reranker(model_name: str):
    """Set the active reranker model (lazy-loads on first use)."""
    try:
        from neuromem.reranker import get_reranker
        get_reranker(model_name=model_name)
    except Exception:
        pass  # Reranker unavailable — search degrades gracefully


def _parallel_search(queries, user_id, internal_limit, llm_fn, output_limit):
    """Run multiple agentic searches in parallel, merge and deduplicate."""
    db_path = _get_memory()._engine.db_path

    def _run_query(q):
        thread_m = Memory(path=db_path)
        try:
            return thread_m.search_deep(
                q, user_id=user_id, limit=internal_limit, llm_fn=llm_fn,
            )
        finally:
            thread_m.close()

    with ThreadPoolExecutor(max_workers=min(len(queries), 5)) as pool:
        futures = [pool.submit(_run_query, q) for q in queries]
        merged = []
        seen_ids = set()
        for f in futures:
            try:
                for r in f.result():
                    rid = r.get("id")
                    if rid not in seen_ids:
                        merged.append(r)
                        seen_ids.add(rid)
            except Exception:
                pass  # Individual query failure doesn't kill the batch

    merged.sort(key=lambda x: -x.get("score", 0))
    return merged[:output_limit]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def neuromem_store(
    content: str,
    user_id: str = "",
    metadata: str = "",
) -> str:
    """Store a memory. Call this proactively whenever the user shares preferences,
    personal facts, decisions, or corrections — do not wait to be asked.
    Store one clear fact per call (e.g. "Prefers Python over JavaScript").

    Args:
        content: The fact or preference to remember. Write as a clear, atomic statement.
        user_id: Owner of this memory (e.g. a person's name).
        metadata: Optional JSON string of metadata.
    """
    m = _get_memory()
    meta = json.loads(metadata) if metadata else None
    result = m.add(content=content, user_id=user_id or None, metadata=meta)
    return json.dumps(result, indent=2)


@mcp.tool()
def neuromem_search(
    query: str,
    user_id: str = "",
    limit: int = 10,
) -> str:
    """Search memories using the full agentic retrieval pipeline (HyDE query
    expansion, cross-encoder reranking, multi-round retrieval).

    Supports multiple queries separated by | for parallel execution.
    Example: "user preferences | project context | recent decisions"
    All queries run simultaneously and results are merged and deduplicated.

    Args:
        query: Natural language search query. Use | to separate multiple queries.
        user_id: Filter results to this user (optional).
        limit: Maximum number of results to return.
    """
    _set_reranker(_SEARCH_RERANKER)
    llm_fn = _get_llm_fn()
    uid = user_id or None
    queries = [q.strip() for q in query.split("|") if q.strip()]

    if len(queries) == 1:
        m = _get_memory()
        results = m.search_deep(
            queries[0], user_id=uid, limit=_SEARCH_INTERNAL_LIMIT, llm_fn=llm_fn,
        )
        return json.dumps(results[:limit], indent=2)

    results = _parallel_search(queries, uid, _SEARCH_INTERNAL_LIMIT, llm_fn, limit)
    return json.dumps(results, indent=2)


@mcp.tool()
def neuromem_search_deep(
    query: str,
    user_id: str = "",
    limit: int = 10,
) -> str:
    """Maximum-depth memory search (top_k=500, multi-round, full reranking).
    Uses the benchmark-grade reranker (91.5% LoCoMo accuracy).

    Use when neuromem_search doesn't find what you need, or for questions
    requiring evidence scattered across many memories. Supports multiple
    queries separated by | for parallel execution.

    Args:
        query: Natural language search query. Use | to separate multiple queries.
        user_id: Filter results to this user (optional).
        limit: Maximum number of results to return.
    """
    _set_reranker(_DEEP_RERANKER)
    llm_fn = _get_llm_fn()
    uid = user_id or None
    queries = [q.strip() for q in query.split("|") if q.strip()]

    if len(queries) == 1:
        m = _get_memory()
        results = m.search_deep(
            queries[0], user_id=uid, limit=_DEEP_INTERNAL_LIMIT, llm_fn=llm_fn,
        )
        return json.dumps(results[:limit], indent=2)

    results = _parallel_search(queries, uid, _DEEP_INTERNAL_LIMIT, llm_fn, limit)
    return json.dumps(results, indent=2)


@mcp.tool()
def neuromem_get(memory_id: int) -> str:
    """Get a specific memory by its ID.

    Args:
        memory_id: The integer ID of the memory to retrieve.
    """
    m = _get_memory()
    result = m.get(memory_id)
    if result is None:
        return json.dumps({"error": f"Memory {memory_id} not found"})
    return json.dumps(result, indent=2)


@mcp.tool()
def neuromem_forget(memory_id: int) -> str:
    """Delete a memory by its ID.

    Args:
        memory_id: The integer ID of the memory to delete.
    """
    m = _get_memory()
    deleted = m.delete(memory_id)
    return json.dumps({"deleted": deleted, "memory_id": memory_id})


@mcp.tool()
def neuromem_stats() -> str:
    """Get memory system statistics (message count, DB size, capabilities)."""
    m = _get_memory()
    stats = m.stats()
    config = _load_config()
    stats["tier"] = config.get("tier", "base")
    stats["tier_configured"] = "tier" in config
    return json.dumps(stats, indent=2, default=str)


@mcp.tool()
def neuromem_configure(tier: str) -> str:
    """Configure Neuromem's embedding tier. Call this once during first-time setup.

    Base: 88.2% accuracy on LoCoMo. Works on any machine. ~30MB download.
    Pro: 91.5% accuracy on LoCoMo. Needs 4GB+ RAM. ~1.5GB one-time download.

    Args:
        tier: "base" or "pro".
    """
    global _memory
    tier = tier.lower().strip()
    if tier not in ("base", "pro"):
        return json.dumps({"error": "tier must be 'base' or 'pro'"})

    # Check pro dependencies before committing
    if tier == "pro":
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return json.dumps({
                "error": "Pro tier requires an extra install. Run: pip install neuromem-core[gpu]",
                "current_tier": _load_config().get("tier", "base"),
            })

    # Save to persistent config
    config = _load_config()
    old_tier = config.get("tier", "base")
    config["tier"] = tier
    _save_config(config)

    # Apply model change — temporarily allow downloads for tier switch
    # (the new model may not be cached yet)
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ["NEUROMEM_EMBED_MODEL"] = tier
    from neuromem.vector_search import set_embedding_model
    set_embedding_model(tier)

    # If tier actually changed, re-embed any existing memories
    rebuilt = False
    if old_tier != tier:
        try:
            m = _get_memory()
            engine = m._engine
            engine._ensure_connection()
            conn = engine.conn
            count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            if count > 0:
                conn.execute("DROP TABLE IF EXISTS vec_messages")
                conn.execute("DROP TABLE IF EXISTS vec_messages_sep")
                conn.commit()
                from neuromem.vector_search import init_vec_table, build_vectors
                init_vec_table(conn)
                build_vectors(conn)
                rebuilt = True
        except Exception:
            pass
        _memory = None  # Force re-init with new model on next call

    # Restore offline mode now that the new model is downloaded/loaded
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    result = {
        "status": "configured",
        "tier": tier,
        "description": "Base: lightweight, works everywhere (~30MB)"
        if tier == "base"
        else "Pro: higher accuracy embeddings (~1.5GB)",
    }
    if rebuilt:
        result["note"] = "Existing memories have been re-embedded with the new model."

    return json.dumps(result, indent=2)


@mcp.tool()
def neuromem_entity_profile(entity: str) -> str:
    """Get the personality profile for an entity (person).

    Returns communication style, preferences, traits, and topics
    extracted from stored memories.

    Args:
        entity: Name of the person/entity to look up.
    """
    m = _get_memory()
    m._engine._ensure_connection()

    try:
        from neuromem.personality import get_entity_profile
        profile = get_entity_profile(m._engine.conn, entity)
        if profile:
            return json.dumps(profile, indent=2, default=str)
        return json.dumps({"error": f"No profile found for '{entity}'"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Background model preloading
# ---------------------------------------------------------------------------

def _preload_models():
    """Pre-load ML models in background threads so the first search is fast.

    Without preloading, the first search pays:
      - sentence_transformers import: ~2,300ms
      - CrossEncoder init: ~70ms
      - model2vec load: ~170ms
      Total: ~2,500ms+ on first query

    With preloading, these costs are absorbed during MCP handshake/init,
    so the first search sees the same latency as subsequent searches.
    """
    def _load_embedding_model_and_db():
        """Pre-load the embedding model and initialize the DB connection.

        Opening the DB + loading sqlite-vec + initializing the Memory singleton
        adds ~50-100ms on first access. Doing it here means _get_memory() is
        instant on the first tool call.
        """
        try:
            from neuromem.vector_search import get_model
            get_model()
        except Exception:
            pass  # Graceful degradation — model loads lazily on first search
        try:
            _get_memory()
        except Exception:
            pass

    def _load_reranker():
        """Pre-import sentence_transformers and load the default reranker.

        The sentence_transformers import alone is ~2.3s (torch, transformers,
        huggingface_hub). Loading it here means it's cached by the time
        the first search needs it.
        """
        try:
            from neuromem.reranker import get_reranker
            get_reranker(model_name=_SEARCH_RERANKER)
        except Exception:
            pass  # Graceful degradation — reranker loads lazily on first search

    # Fire both loads in parallel background threads.
    # daemon=True so they don't block server shutdown.
    t1 = threading.Thread(target=_load_embedding_model_and_db, daemon=True)
    t2 = threading.Thread(target=_load_reranker, daemon=True)
    t1.start()
    t2.start()
    # Don't join — let them finish in the background while the server
    # handles the MCP handshake. The singleton locks in each module
    # ensure thread safety if a search arrives before loading finishes.


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the MCP server (stdio transport)."""
    # Kick off model preloading before entering the event loop.
    # Models load in background threads (~2.5s) while the MCP handshake
    # completes (~1-3s), so by the time the first search arrives,
    # models are already warm.
    _preload_models()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
