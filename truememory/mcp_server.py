"""
TrueMemory MCP Server
===================

Model Context Protocol server that exposes the TrueMemory memory system
as tools for Claude and other MCP-compatible AI assistants.

Usage::

    # Direct
    python -m truememory.mcp_server

    # Via entry point (after pip install)
    truememory-mcp

Configuration via environment variables:
    TRUEMEMORY_DB    Path to .db file (default: ~/.truememory/memories.db)
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

from truememory import __version__
from truememory.client import Memory

# ---------------------------------------------------------------------------
# Config management
# ---------------------------------------------------------------------------

_TRUEMEMORY_DIR = Path.home() / ".truememory"
_CONFIG_PATH = _TRUEMEMORY_DIR / "config.json"


def _load_config() -> dict:
    """Load persistent config from ~/.truememory/config.json."""
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_config(config: dict) -> None:
    """Save config to ~/.truememory/config.json."""
    _TRUEMEMORY_DIR.mkdir(parents=True, exist_ok=True)
    _TRUEMEMORY_DIR.chmod(0o700)
    _CONFIG_PATH.write_text(json.dumps(config, indent=2))
    _CONFIG_PATH.chmod(0o600)


# Apply saved tier on startup (before any model loading)
_startup_config = _load_config()
if "tier" in _startup_config:
    os.environ["TRUEMEMORY_EMBED_MODEL"] = _startup_config["tier"]

# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "truememory",
    instructions="""You have access to a persistent memory system (TrueMemory). Use it proactively.

FIRST-TIME SETUP (do this FIRST, before anything else):
1. Call truememory_stats at the start of your first session.
2. If the response contains "setup_required": true, present the "welcome" message to the user EXACTLY as written — it contains the setup instructions.
3. Wait for the user to choose Edge, Base, or Pro.
4. If they choose Pro, ask for an API key for HyDE query expansion (Anthropic, OpenRouter, or OpenAI) — required for Pro, optional for Edge / Base.
5. Call truememory_configure with their choices (tier, and optionally api_key + api_provider).
6. Present the "next_steps" from the response to the user — it shows how to use TrueMemory.
7. Setup is done. Proceed normally.

STATUS CHECK:
- When the user asks "is TrueMemory running?", "what version?", or similar, call truememory_stats and tell them:
  "TrueMemory v{version} is running. Tier: {tier}. You have {message_count} memories stored."
- Include the tier (base/pro) and whether HyDE search is active.

AFTER SETUP — NORMAL USAGE:

Storing memories (truememory_store):
- When the user shares personal information, preferences, or facts about themselves, store them immediately without being asked.
- When important decisions are made during a conversation, store them.
- When the user corrects you or clarifies something, store the correction.
- Store each fact as a clear, atomic statement. Prefer "User prefers dark mode" over "The user mentioned something about dark mode."
- Include the user_id parameter when you know who the user is.

Recalling memories (truememory_search):
- At the START of each conversation, call truememory_search with a broad query to load relevant context.
- You can search multiple topics at once using | separation: "user preferences | project context | recent decisions"
- Multiple queries run in parallel — no speed penalty for combining them.
- Before making recommendations, check if you already have relevant context from prior searches.
- When the user asks "do you remember" or references past conversations, search for the specific topic.

Deep search (truememory_search_deep):
- Use when truememory_search doesn't find what you need, or for complex multi-part questions.
- Also supports | separated parallel queries.
- Retrieves 5x more candidates internally — best for questions requiring scattered evidence.

You should store and recall memories as naturally as a good assistant who remembers past conversations. Do not ask permission to remember things — just do it.""",
)

_DB_PATH = os.path.expanduser(
    os.environ.get("TRUEMEMORY_DB", str(Path.home() / ".truememory" / "memories.db"))
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
      2. Persistent config (~/.truememory/config.json, written by ``truememory-ingest setup``)

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
# Default is the paper §2.0 Edge reranker (ms-marco-MiniLM-L-6-v2, 22M params,
# CPU-friendly). Bench scripts for Base/Pro explicitly override to
# gte-reranker-modernbert-base via get_reranker(model_name=...).
_SEARCH_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"    # 22M, CPU-friendly
_DEEP_RERANKER = "BAAI/bge-reranker-v2-m3"                   # 568M, ~0.77s/query


def _set_reranker(model_name: str):
    """Set the active reranker model (lazy-loads on first use)."""
    try:
        from truememory.reranker import get_reranker
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
def truememory_store(
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
def truememory_search(
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
def truememory_search_deep(
    query: str,
    user_id: str = "",
    limit: int = 10,
) -> str:
    """Maximum-depth memory search (top_k=500, multi-round, full reranking).
    Uses the benchmark-grade reranker (91.5% LoCoMo accuracy).

    Use when truememory_search doesn't find what you need, or for questions
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
def truememory_get(memory_id: int) -> str:
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
def truememory_forget(memory_id: int) -> str:
    """Delete a memory by its ID.

    Args:
        memory_id: The integer ID of the memory to delete.
    """
    m = _get_memory()
    deleted = m.delete(memory_id)
    return json.dumps({"deleted": deleted, "memory_id": memory_id})


@mcp.tool()
def truememory_stats() -> str:
    """Get memory system statistics. On first run, returns a welcome message
    and setup instructions — present these to the user to walk them through
    choosing Edge, Base, or Pro tier."""
    m = _get_memory()
    m._engine._ensure_connection()
    stats = m.stats()
    config = _load_config()
    stats["version"] = __version__
    stats["tier"] = config.get("tier", "edge")
    stats["tier_configured"] = "tier" in config

    if not stats["tier_configured"]:
        stats["setup_required"] = True
        stats["welcome"] = (
            f"Welcome to TrueMemory v{__version__} — persistent memory for AI agents.\n"
            "\n"
            "Your memories are stored locally in a single SQLite file. "
            "Zero cloud, zero infrastructure cost.\n"
            "\n"
            "Choose your tier:\n"
            "\n"
            "  Edge — 90.1% accuracy on LoCoMo. CPU-only, works anywhere.\n"
            "          Lightweight (~30MB install). No API key needed.\n"
            "\n"
            "  Base — 91.5% accuracy on LoCoMo. GPU recommended.\n"
            "          Paper-aligned Qwen3 @ 256d + gte-reranker (~1.5GB install).\n"
            "          No API key needed — fully offline.\n"
            "\n"
            "  Pro  — 91.8% accuracy on LoCoMo. GPU recommended.\n"
            "          Same models as Base plus HyDE query expansion.\n"
            "          Requires an API key (Anthropic / OpenRouter / OpenAI) for the HyDE LLM call.\n"
            "\n"
            "Which would you like: Edge, Base, or Pro?"
        )
        stats["has_api_key"] = bool(
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or config.get("anthropic_api_key")
            or config.get("openrouter_api_key")
            or config.get("openai_api_key")
        )

    return json.dumps(stats, indent=2, default=str)


@mcp.tool()
def truememory_configure(
    tier: str,
    api_key: str = "",
    api_provider: str = "",
) -> str:
    """Configure TrueMemory. Call this once during first-time setup,
    or again to change tier or update API keys.

    Args:
        tier: "edge", "base", or "pro".
        api_key: API key for HyDE query expansion (required for Pro,
                 optional for Edge / Base).  Supported providers:
                 anthropic, openrouter, openai.
        api_provider: Required if api_key is provided. One of: "anthropic", "openrouter", "openai".
    """
    global _memory
    tier = tier.lower().strip()
    if tier not in ("edge", "base", "pro"):
        return json.dumps({"error": "tier must be 'edge', 'base', or 'pro'"})

    # Validate API key + provider pairing
    if api_key and not api_provider:
        return json.dumps({
            "error": "api_provider is required when api_key is provided. Use: anthropic, openrouter, or openai",
        })
    if api_provider:
        api_provider = api_provider.lower().strip()
        if api_provider not in ("anthropic", "openrouter", "openai"):
            return json.dumps({
                "error": "api_provider must be one of: anthropic, openrouter, openai",
            })

    # Check Base / Pro dependencies before committing (both need sentence-transformers
    # for the Qwen3 embedder + gte-reranker).
    if tier in ("base", "pro"):
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return json.dumps({
                "error": f"{tier.capitalize()} tier requires an extra install. Run: pip install truememory[gpu]",
                "current_tier": _load_config().get("tier", "edge"),
            })

    # Save to persistent config
    config = _load_config()
    old_tier = config.get("tier", "edge")
    config["tier"] = tier

    # Store API key if provided
    if api_key and api_provider:
        config[f"{api_provider}_api_key"] = api_key

    _save_config(config)

    # Invalidate cached LLM function so it picks up the new key
    if api_key:
        global _cached_llm_fn, _cached_llm_fn_built
        _cached_llm_fn = None
        _cached_llm_fn_built = False

    # Apply model change — temporarily allow downloads for tier switch
    # (the new model may not be cached yet)
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ["TRUEMEMORY_EMBED_MODEL"] = tier
    from truememory.vector_search import set_embedding_model
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
                from truememory.vector_search import init_vec_table, build_vectors
                init_vec_table(conn)
                build_vectors(conn)
                rebuilt = True
        except Exception:
            pass
        _memory = None  # Force re-init with new model on next call

    # Restore offline mode now that the new model is downloaded/loaded
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Build result with onboarding info
    result = {
        "status": "configured",
        "tier": tier,
        "description": "Base: lightweight, works everywhere (~30MB)"
        if tier == "base"
        else "Pro: higher accuracy embeddings (~1.5GB)",
    }
    if rebuilt:
        result["note"] = "Existing memories have been re-embedded with the new model."
    if api_key:
        result["api_key_saved"] = f"{api_provider} key stored"

    # Check if HyDE search is available
    has_key = bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or config.get("anthropic_api_key")
        or config.get("openrouter_api_key")
        or config.get("openai_api_key")
    )
    result["hyde_search"] = "enabled" if has_key else "disabled (no API key — search still works, just without query expansion)"

    # Onboarding: usage examples and next steps
    result["next_steps"] = (
        "TrueMemory is ready. Here's how it works:\n"
        "\n"
        "  Storing:  I'll automatically remember things you tell me — preferences,\n"
        "            facts, decisions, corrections. You don't need to ask.\n"
        "\n"
        "  Recalling: At the start of each session, I'll search your memories\n"
        "             for relevant context. You can also ask me directly:\n"
        "             \"Do you remember...?\" or \"What do you know about...?\"\n"
        "\n"
        "  Examples to try:\n"
        "    - \"I prefer dark mode and TypeScript\"\n"
        "    - \"Remember that our deploy freezes happen on Thursdays\"\n"
        "    - \"What are my preferences?\"\n"
        "    - \"Do you remember what we discussed about the auth rewrite?\"\n"
        "\n"
        "  Everything is stored locally at ~/.truememory/memories.db.\n"
        "  Go ahead — I'm ready."
    )

    return json.dumps(result, indent=2)


@mcp.tool()
def truememory_entity_profile(entity: str) -> str:
    """Get the personality profile for an entity (person).

    Returns communication style, preferences, traits, and topics
    extracted from stored memories.

    Args:
        entity: Name of the person/entity to look up.
    """
    m = _get_memory()
    m._engine._ensure_connection()

    try:
        from truememory.personality import get_entity_profile
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
            from truememory.vector_search import get_model
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
            from truememory.reranker import get_reranker
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
# Auto-setup for Claude Code and Claude Desktop
# ---------------------------------------------------------------------------

def _setup_claude():
    """Auto-configure TrueMemory as an MCP server in Claude Code and/or Claude Desktop.

    Detects which Claude clients are available and configures both.
    Uses sys.executable so the MCP server runs with the same Python
    that has truememory installed (works in venvs, homebrew, system python, etc.).

    Re-run behaviour:
      - If no ``truememory`` entry exists, creates one.
      - If an entry exists and its command points to a file that still exists
        on disk, the entry is preserved (so an active dev venv or an older
        working install is not clobbered).
      - If an entry exists but its command path no longer exists on disk
        (stale entry from a deleted venv, moved home dir, etc.), the entry
        is replaced with the current ``sys.executable``.
    """
    import shutil
    import subprocess
    import sys

    python_path = sys.executable
    mcp_args = ["-m", "truememory.mcp_server"]
    configured = []

    def _path_exists(p: str) -> bool:
        try:
            return bool(p) and Path(p).exists()
        except Exception:
            return False

    # --- Claude Code CLI ---
    claude_bin = shutil.which("claude")
    if claude_bin:
        add_cmd = [claude_bin, "mcp", "add", "truememory", "--",
                   python_path, *mcp_args]
        result = subprocess.run(add_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            configured.append("Claude Code")
        elif "already exists" in (result.stderr or "").lower():
            # Inspect the existing entry. `claude mcp list` output format:
            #   truememory: /path/to/python -m truememory.mcp_server - ✓ Connected
            list_result = subprocess.run(
                [claude_bin, "mcp", "list"], capture_output=True, text=True,
            )
            existing_cmd = ""
            if list_result.returncode == 0:
                for line in (list_result.stdout or "").splitlines():
                    stripped = line.strip()
                    if stripped.startswith("truememory:") or stripped.startswith("truememory "):
                        # Everything after the first colon, before the status marker.
                        rest = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
                        # Format: "<cmd> <args...> - <status>" — strip the trailing status.
                        if " - " in rest:
                            rest = rest.rsplit(" - ", 1)[0]
                        tokens = rest.split()
                        if tokens:
                            existing_cmd = tokens[0]
                        break

            if _path_exists(existing_cmd):
                # Working entry — preserve it (don't clobber a dev venv).
                configured.append("Claude Code (existing config preserved)")
            else:
                # Stale entry — remove and re-add.
                subprocess.run(
                    [claude_bin, "mcp", "remove", "truememory"],
                    capture_output=True, text=True,
                )
                retry = subprocess.run(add_cmd, capture_output=True, text=True)
                if retry.returncode == 0:
                    configured.append("Claude Code (stale entry replaced)")
                else:
                    print(f"  Claude Code: update failed — {retry.stderr.strip()}")
        else:
            print(f"  Claude Code: failed — {result.stderr.strip()}")

    # --- Claude Desktop ---
    desktop_config_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if desktop_config_path.parent.exists():
        try:
            if desktop_config_path.exists():
                config = json.loads(desktop_config_path.read_text())
            else:
                config = {}

            servers = config.setdefault("mcpServers", {})
            existing = servers.get("truememory")
            existing_cmd = (existing or {}).get("command", "") if isinstance(existing, dict) else ""

            if existing is None:
                # No entry — create one.
                servers["truememory"] = {"command": python_path, "args": list(mcp_args)}
                desktop_config_path.write_text(json.dumps(config, indent=2))
                configured.append("Claude Desktop")
            elif _path_exists(existing_cmd):
                # Working entry — preserve it.
                configured.append("Claude Desktop (existing config preserved)")
            else:
                # Stale entry — replace it.
                servers["truememory"] = {"command": python_path, "args": list(mcp_args)}
                desktop_config_path.write_text(json.dumps(config, indent=2))
                configured.append("Claude Desktop (stale entry replaced)")
        except Exception as e:
            print(f"  Claude Desktop: failed — {e}")

    # --- Report ---
    print()
    print(f"  TrueMemory v{__version__}")
    print()
    if configured:
        for c in configured:
            print(f"  + {c}")
        print()
        print("  Start a new Claude session — TrueMemory will walk you through setup.")
    else:
        if not claude_bin:
            print("  Claude Code CLI not found on PATH.")
        if not desktop_config_path.parent.exists():
            print("  Claude Desktop not detected.")
        print()
        print("  Manual setup:")
        print(f"    claude mcp add truememory -- {python_path} -m truememory.mcp_server")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the MCP server, or --setup to auto-configure Claude."""
    import sys
    if "--setup" in sys.argv:
        _setup_claude()
        return

    # Kick off model preloading before entering the event loop.
    # Models load in background threads (~2.5s) while the MCP handshake
    # completes (~1-3s), so by the time the first search arrives,
    # models are already warm.
    _preload_models()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
