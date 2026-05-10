# MCP Tool Reference

TrueMemory exposes 8 tools via the Model Context Protocol. These are called by Claude Code, Claude Desktop, and other MCP-compatible clients.

---

## truememory_store

Store a memory. Claude calls this proactively when the user shares preferences, facts, decisions, or corrections.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | `str` | required | The fact to remember. One clear, atomic statement. |
| `user_id` | `str` | `""` | Owner of this memory (e.g., a person's name). |
| `metadata` | `str` | `""` | Optional JSON string of metadata. |

**Content limit:** 50,000 characters max.

---

## truememory_search

Search memories using the full agentic retrieval pipeline (HyDE, cross-encoder reranking, multi-round retrieval).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Natural language search query. Use `\|` to separate multiple parallel queries. |
| `user_id` | `str` | `""` | Filter results to this user. |
| `limit` | `int` | `10` | Max results (clamped to 1-200). |

**Parallel queries:** `"user preferences | project context | recent decisions"` runs all three simultaneously and merges results.

**Query length:** Truncated to 2,000 characters.

---

## truememory_search_deep

Maximum-depth search with top_k=500 and a heavier cross-encoder (BAAI/bge-reranker-v2-m3, 568M params). Higher recall at higher latency.

Use when `truememory_search` doesn't find what you need, or for questions requiring evidence scattered across many memories.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Natural language search query. Supports `\|` parallel queries. |
| `user_id` | `str` | `""` | Filter to this user. |
| `limit` | `int` | `10` | Max results (clamped to 1-200). |

---

## truememory_get

Retrieve a specific memory by its integer ID.

| Parameter | Type | Description |
|-----------|------|-------------|
| `memory_id` | `int` | The ID of the memory to retrieve. |

---

## truememory_forget

Delete a memory by its integer ID.

| Parameter | Type | Description |
|-----------|------|-------------|
| `memory_id` | `int` | The ID of the memory to delete. |

---

## truememory_stats

Get memory system statistics. On first run, returns a welcome message and setup instructions.

No parameters.

**Returns:** version, tier, message count, health status, and capabilities.

---

## truememory_configure

Configure TrueMemory. Call during first-time setup or to change tier/API keys.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tier` | `str` | required | `"edge"`, `"base"`, or `"pro"`. |
| `api_key` | `str` | `""` | API key for HyDE query expansion (Pro tier). |
| `api_provider` | `str` | `""` | `"anthropic"`, `"openrouter"`, or `"openai"`. Required if api_key is provided. |
| `email` | `str` | `""` | User's email for updates and support (optional). |

**Tier switching:** Changes the embedding model and re-embeds all existing memories with the new model. All models are pre-installed — no downloads needed.

---

## truememory_entity_profile

Get the personality profile for an entity (person). Returns communication style, preferences, and behavioral traits derived from stored memories.

| Parameter | Type | Description |
|-----------|------|-------------|
| `entity` | `str` | Name of the person/entity to look up. |
