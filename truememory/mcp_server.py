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
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from truememory import __version__
from truememory.client import Memory

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config management
# ---------------------------------------------------------------------------

_TRUEMEMORY_DIR = Path.home() / ".truememory"
_CONFIG_PATH = _TRUEMEMORY_DIR / "config.json"


def _load_config() -> dict:
    """Load persistent config from ~/.truememory/config.json.

    On JSON corruption, rename the file to ``config.json.corrupt.<unix-ts>``
    so the user can recover any API keys that were in it. On OSError, warn
    to stderr. Returns ``{}`` in both failure modes so callers never see a
    half-loaded config.
    """
    if not _CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(_CONFIG_PATH.read_text())
    except json.JSONDecodeError as e:
        # .with_suffix would replace ".json"; we want to APPEND to preserve
        # the origin filename in the backup so users can find it easily.
        backup = _CONFIG_PATH.parent / f"{_CONFIG_PATH.name}.corrupt.{int(time.time())}"
        try:
            _CONFIG_PATH.rename(backup)
            print(
                f"truememory: config.json is corrupt (JSON parse error at "
                f"line {e.lineno} col {e.colno}). Saved corrupt file to "
                f"{backup} — your API keys may be recoverable from there. "
                f"Run `truememory-mcp --setup` to recreate the config.",
                file=sys.stderr,
            )
        except OSError:
            print(
                "truememory: config.json is corrupt and could not be "
                "backed up. Run `truememory-mcp --setup` to recreate.",
                file=sys.stderr,
            )
        return {}
    except OSError as e:
        print(
            f"truememory: could not read config.json: {e}",
            file=sys.stderr,
        )
        return {}


def _save_config(config: dict) -> None:
    """Save config to ~/.truememory/config.json.

    Hunter F28: the POSIX `chmod` calls below are silent no-ops on Windows;
    the config file (which stores API keys in plaintext) inherits the
    parent-directory ACL — typically readable by all local users. When
    storing a key on Windows we warn to stderr and suggest the env-var
    route, which is the actually-private channel.
    """
    _TRUEMEMORY_DIR.mkdir(parents=True, exist_ok=True)
    _TRUEMEMORY_DIR.chmod(0o700)
    _CONFIG_PATH.write_text(json.dumps(config, indent=2))
    _CONFIG_PATH.chmod(0o600)
    if sys.platform == "win32" and any(k.endswith("_api_key") for k in config):
        print(
            "truememory: warning — on Windows, ~/.truememory/config.json "
            "permissions are inherited from the parent directory and may be "
            "readable by other local users. If this is a shared machine, set "
            "the API key via the ANTHROPIC_API_KEY / OPENROUTER_API_KEY / "
            "OPENAI_API_KEY environment variable instead.",
            file=sys.stderr,
        )


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

# Hunter F05: per-provider last-error state so truememory_stats.health can
# surface "Pro tier silently degraded to Base" instead of hiding the failure.
# Mutation happens from _build_llm_fn (MCP thread) and truememory_configure
# (MCP thread); readers are truememory_stats (MCP thread). A module-level
# lock keeps the dict consistent when F22 lands.
_llm_last_error: dict[str, str] = {}
_llm_error_lock = threading.Lock()
_current_llm_provider_name: str | None = None


def _record_llm_error(provider: str, err: Exception) -> None:
    with _llm_error_lock:
        _llm_last_error[provider] = f"{type(err).__name__}: {err}"
    log.warning("HyDE LLM init failed (%s): %s: %s",
                provider, type(err).__name__, err)


def _clear_llm_error(provider: str) -> None:
    with _llm_error_lock:
        _llm_last_error.pop(provider, None)


def _clear_all_llm_errors() -> None:
    with _llm_error_lock:
        _llm_last_error.clear()


def _build_anthropic_llm(api_key: str):
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


def _build_openrouter_llm(api_key: str):
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


def _build_openai_llm(api_key: str):
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


# Provider table — builders are looked up dynamically via module-level name
# so tests can monkeypatch ``_build_anthropic_llm`` etc. without having to
# reimport the module or mutate a frozen tuple of captured references.
_LLM_PROVIDERS = (
    ("anthropic", "ANTHROPIC_API_KEY", "anthropic_api_key", "_build_anthropic_llm"),
    ("openrouter", "OPENROUTER_API_KEY", "openrouter_api_key", "_build_openrouter_llm"),
    ("openai", "OPENAI_API_KEY", "openai_api_key", "_build_openai_llm"),
)


def _build_llm_fn():
    """Build an llm_fn from available API keys.

    Resolution order for each provider:
      1. Environment variable (ANTHROPIC_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY)
      2. Persistent config (~/.truememory/config.json, written by ``truememory-ingest setup``)

    Provider priority: Anthropic direct → OpenRouter → OpenAI. On init
    failure the error is logged at WARNING and stored in
    ``_llm_last_error[provider]`` so ``truememory_stats.health`` (F07) can
    surface the degradation instead of silently returning None.
    """
    global _current_llm_provider_name
    config = _load_config()
    for provider, env_var, config_key, builder_name in _LLM_PROVIDERS:
        api_key = os.environ.get(env_var) or config.get(config_key)
        if not api_key:
            continue
        builder = globals()[builder_name]
        try:
            fn = builder(api_key)
            _clear_llm_error(provider)
            _current_llm_provider_name = provider
            return fn
        except Exception as e:
            _record_llm_error(provider, e)
    _current_llm_provider_name = None
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

# Tiered rerankers: standard search resolves per-tier via _current_reranker()
# so Base / Pro get gte-reranker-modernbert-base (paper §2.0). The deep
# reranker is tier-independent by design — it's a "maximum recall" escape
# hatch used by truememory_search_deep regardless of tier.
_DEEP_RERANKER = "BAAI/bge-reranker-v2-m3"                   # 568M, ~0.77s/query


def _current_reranker() -> str:
    """Resolve the reranker HF model ID for the currently-configured tier.

    Thin wrapper around truememory.reranker.get_current_reranker_name().
    Reads the tier from persistent config (seeded lazily from env /
    ~/.truememory/config.json, updated explicitly via set_active_tier()
    when truememory_configure runs).
    """
    from truememory.reranker import get_current_reranker_name
    return get_current_reranker_name()


# Hunter F06: reranker init error state. ``_set_reranker`` is called on
# every truememory_search — the throttle flag prevents repeated log spam
# when the error is unchanged across calls, but the stored error string
# is always current so truememory_stats.health (F07) reports live state.
_reranker_last_error: str | None = None
_reranker_last_logged: str | None = None
_reranker_error_lock = threading.Lock()


def _record_reranker_error(msg: str) -> None:
    """Store ``msg`` and log once per distinct error value."""
    global _reranker_last_error, _reranker_last_logged
    with _reranker_error_lock:
        _reranker_last_error = msg
        should_log = (_reranker_last_logged != msg)
        if should_log:
            _reranker_last_logged = msg
    if should_log:
        log.warning("Reranker init failed: %s", msg)


def _clear_reranker_error() -> None:
    global _reranker_last_error, _reranker_last_logged
    with _reranker_error_lock:
        _reranker_last_error = None
        _reranker_last_logged = None


def _set_reranker(model_name: str):
    """Set the active reranker model (lazy-loads on first use).

    On failure: store the error in ``_reranker_last_error`` so F07's health
    payload can surface the degradation; log at WARNING once per distinct
    error to avoid spamming logs on every search call.
    """
    try:
        from truememory.reranker import get_reranker
        get_reranker(model_name=model_name)
        _clear_reranker_error()
    except ImportError as e:
        _record_reranker_error(
            f"ImportError: {e} — install truememory[gpu] for reranker support"
        )
    except Exception as e:
        _record_reranker_error(f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Health payload (Hunter F07) — surfaces per-subsystem degradation so MCP
# clients can diagnose "search quality is bad" without digging through logs.
# Reads state written by F05 (_llm_last_error), F06 (_reranker_last_error),
# and F08 (engine._vectors_load_error). Pure read — no mutation.
# ---------------------------------------------------------------------------


def _build_health_payload() -> dict:
    """Return the `stats['health']` dict exposed by `truememory_stats`.

    Each subsystem reports `{status, last_error, ...}`. ``status`` is one of
    ``"ok"`` or ``"degraded"`` — the latter means a writer (F05 / F06 / F08)
    stored a non-None error during the current process lifetime.
    """
    # Reranker — written by F06's _set_reranker.
    with _reranker_error_lock:
        reranker_err = _reranker_last_error

    # HyDE LLM — written by F05's _build_llm_fn.
    with _llm_error_lock:
        llm_errors = dict(_llm_last_error) if _llm_last_error else None
    active_provider = _current_llm_provider_name

    # sqlite-vec — written by F08 in engine.open().
    try:
        from truememory.engine import get_vectors_load_error
        vectors_err = get_vectors_load_error()
    except ImportError:
        vectors_err = None

    return {
        "reranker": {
            "status": "ok" if reranker_err is None else "degraded",
            "last_error": reranker_err,
        },
        "hyde_llm": {
            "status": "ok" if not llm_errors else "degraded",
            "active_provider": active_provider,
            "last_error_by_provider": llm_errors,
        },
        "vectors": {
            "status": "ok" if vectors_err is None else "degraded",
            "last_error": vectors_err,
        },
    }


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
    _set_reranker(_current_reranker())
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
    Uses a heavier cross-encoder (BAAI/bge-reranker-v2-m3, 568M params) than
    the standard tier-selected reranker — higher recall at higher latency.

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
    stats["health"] = _build_health_payload()

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
        # Clear stored LLM errors for the provider we just re-keyed
        _clear_llm_error(api_provider)

    # Apply model change — temporarily allow downloads for tier switch
    # (the new model may not be cached yet)
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ["TRUEMEMORY_EMBED_MODEL"] = tier
    from truememory.vector_search import set_embedding_model
    set_embedding_model(tier)

    # Tell the reranker module about the new tier so get_reranker(model_name=None)
    # calls (from direct Python-API users via rerank_with_modality_fusion etc.)
    # resolve to the tier-correct model. Then pre-load that reranker so the
    # first post-configure search doesn't pay a cold-start.
    from truememory.reranker import set_active_tier as _set_active_tier
    _set_active_tier(tier)
    _set_reranker(_current_reranker())

    # If tier actually changed, re-embed any existing memories.
    # Hunter F03: (1) rebuild BOTH vec_messages and vec_messages_sep (the
    # old code only rebuilt the completion table, leaving the separation
    # table silently empty); (2) surface exceptions as rebuild_error in
    # the response payload instead of swallowing into bare pass; (3) null
    # _memory in `finally` so the next call always gets a fresh instance
    # with the new model — even on failure.
    rebuilt = False
    rebuild_error: str | None = None
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
                from truememory.vector_search import (
                    build_separation_vectors,
                    build_vectors,
                    init_vec_table,
                )
                init_vec_table(conn)
                build_vectors(conn)
                build_separation_vectors(conn)
                rebuilt = True
        except Exception as e:
            rebuild_error = f"{type(e).__name__}: {e}"
            log.exception("truememory_configure re-embed failed")
        finally:
            _memory = None  # Always force re-init, even on failure

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
    if rebuild_error is not None:
        result["rebuild_error"] = rebuild_error
        result["warning"] = (
            "Tier switch succeeded but memory re-embedding failed. Re-run "
            "truememory_configure() to retry, or delete ~/.truememory/memories.db "
            "to start fresh."
        )
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
            get_reranker(model_name=_current_reranker())
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

def _claude_desktop_config_path() -> Path:
    """Return the per-platform Claude Desktop config file location.

    Hunter F29: previously hardcoded to macOS. Claude Desktop stores this
    file in a platform-specific app-data location:
      - macOS:   ``~/Library/Application Support/Claude/claude_desktop_config.json``
      - Windows: ``%APPDATA%/Claude/claude_desktop_config.json`` (fallback
                 ``~/AppData/Roaming/Claude/...`` when APPDATA is unset)
      - Linux:   ``~/.config/Claude/claude_desktop_config.json``

    The caller still gates on ``desktop_config_path.parent.exists()`` — if
    Claude Desktop isn't installed on this platform, ``_setup_claude`` will
    correctly report "not detected" instead of creating the directory.
    """
    if sys.platform == "darwin":
        return (Path.home() / "Library" / "Application Support"
                / "Claude" / "claude_desktop_config.json")
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return base / "Claude" / "claude_desktop_config.json"
    # Linux / BSD / other POSIX
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


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

    # Hunter F23: bound every `claude` CLI call. Without a timeout, a stalled
    # claude binary (auth prompt, blocked network, deadlock) wedges
    # `truememory-mcp --setup` forever. 30s is generous — claude CLI calls
    # should complete in well under a second.
    _CLAUDE_TIMEOUT = 30

    def _run_claude(cmd: list[str]) -> subprocess.CompletedProcess | None:
        """Run a `claude ...` command with a bounded wait.

        Returns None (and warns to stderr) if the command timed out so the
        caller can fall through gracefully.
        """
        try:
            return subprocess.run(
                cmd, capture_output=True, text=True, timeout=_CLAUDE_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            print(
                f"  Claude Code: `{' '.join(cmd[:4])}...` timed out after "
                f"{_CLAUDE_TIMEOUT}s. Skip or re-run later.",
                file=sys.stderr,
            )
            return None

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
        result = _run_claude(add_cmd)
        if result is None:
            pass  # Timeout already warned — fall through to Claude Desktop
        elif result.returncode == 0:
            configured.append("Claude Code")
        elif "already exists" in (result.stderr or "").lower():
            # Inspect the existing entry. `claude mcp list` output format:
            #   truememory: /path/to/python -m truememory.mcp_server - ✓ Connected
            list_result = _run_claude([claude_bin, "mcp", "list"])
            existing_cmd = ""
            if list_result is not None and list_result.returncode == 0:
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
                _run_claude([claude_bin, "mcp", "remove", "truememory"])
                retry = _run_claude(add_cmd)
                if retry is not None and retry.returncode == 0:
                    configured.append("Claude Code (stale entry replaced)")
                elif retry is not None:
                    print(f"  Claude Code: update failed — {retry.stderr.strip()}")
        else:
            print(f"  Claude Code: failed — {result.stderr.strip()}")

    # --- Claude Desktop ---
    # Hunter F29: pre-PR49, this path was hardcoded to the macOS
    # `~/Library/Application Support/Claude/...` location, so Linux and
    # Windows users silently got "Claude Desktop not detected" even with
    # Desktop installed. Resolve per-platform instead.
    desktop_config_path = _claude_desktop_config_path()
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

_HELP_TEXT = """Usage: truememory-mcp [--setup | --help | --version]

TrueMemory MCP server — persistent memory for AI agents.

Options:
  --setup       Auto-configure TrueMemory as an MCP server in Claude Code
                and/or Claude Desktop. Run this once after `pip install`.
  --help, -h    Show this help message and exit.
  --version, -V Show version and exit.

With no arguments, runs the MCP server on stdio. This is what Claude Code
and Claude Desktop invoke when they connect to the server — do not run it
directly in a terminal unless you're debugging the MCP stdio protocol.

See https://github.com/buildingjoshbetter/TrueMemory for documentation.
"""


def main():
    """Run the MCP server, or --setup / --help / --version.

    Returns a Unix-style exit code: 0 on success, 2 on unknown flags.
    """
    import sys
    argv = sys.argv[1:]

    # Informational flags — these must return immediately, before mcp.run()
    # blocks on stdin or _preload_models() starts background threads.
    #
    # Hunter F40: if both `--help` and `--setup` are passed, `--help` wins
    # (checked first, below). This is the conventional Unix posture —
    # docs-emitting flags short-circuit any side-effecting operation — and
    # is locked in by tests/test_cli_help.py. Flipping the precedence would
    # be a behaviour change worth a CHANGELOG entry.
    if "--help" in argv or "-h" in argv:
        print(_HELP_TEXT)
        return 0
    if "--version" in argv or "-V" in argv:
        print(f"truememory-mcp {__version__}")
        return 0
    if "--setup" in argv:
        # Unknown args alongside --setup are a user typo; surface it.
        extras = [a for a in argv if a != "--setup"]
        if extras:
            print(f"truememory-mcp: unexpected argument(s) after --setup: {' '.join(extras)}", file=sys.stderr)
            print(_HELP_TEXT, file=sys.stderr)
            return 2
        _setup_claude()
        return 0

    # Any flag we didn't recognize: exit 2 with usage rather than silently
    # entering mcp.run() and blocking on stdin. This is the #1 fresh-install
    # footgun — `truememory-mcp --halp` would hang forever.
    unknown = [a for a in argv if a.startswith("-")]
    if unknown:
        print(f"truememory-mcp: unknown argument(s): {' '.join(unknown)}", file=sys.stderr)
        print(_HELP_TEXT, file=sys.stderr)
        return 2

    # Any remaining positional args at this point are a user typo — e.g.,
    # `truememory-mcp help` (no dashes, not caught by the unknown-flag check
    # above). Reject rather than fall through to mcp.run() and hang on stdin.
    if argv:
        print(f"truememory-mcp: unexpected argument(s): {' '.join(argv)}", file=sys.stderr)
        print(_HELP_TEXT, file=sys.stderr)
        return 2

    # No args → this is the Claude-Code-invoked MCP server path.
    #
    # HuggingFace offline mode — skip HTTP freshness checks when models are
    # cached. Models are downloaded on first install; subsequent loads
    # should be pure disk reads (~170ms) instead of HTTP round-trips
    # (~600ms+). Set HERE (not at module import) so merely importing
    # truememory.mcp_server from a test or notebook doesn't poison the
    # environment for code that expects online HF access.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # Kick off model preloading before entering the event loop. Models
    # load in background threads (~2.5s) while the MCP handshake
    # completes (~1-3s), so the first search arrives with warm models.
    _preload_models()
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    main()
