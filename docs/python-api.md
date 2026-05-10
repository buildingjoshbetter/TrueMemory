# Python API Reference

The `Memory` class is the primary interface for embedding TrueMemory in Python applications.

```python
from truememory import Memory
```

## Memory(path=None)

Create a Memory instance. The database is created automatically if it doesn't exist.

```python
m = Memory()                          # default: ~/.truememory/memories.db
m = Memory(path="/custom/path.db")    # custom location
```

**Context manager support:**

```python
with Memory() as m:
    m.add("User prefers dark mode", user_id="alice")
    results = m.search("preferences", user_id="alice")
# connection closed automatically
```

---

## m.add(content, user_id=None, metadata=None) → dict

Store a memory. Returns a dict with the new memory's `id`, `content`, and `timestamp`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | `str` | required | The fact or preference to store. Write as a clear, atomic statement. |
| `user_id` | `str \| None` | `None` | Scope this memory to a specific user. |
| `metadata` | `dict \| None` | `None` | Reserved for future use. |

```python
result = m.add("Prefers TypeScript over JavaScript", user_id="alice")
# {"id": 42, "content": "Prefers TypeScript over JavaScript", "timestamp": "2026-05-10T..."}
```

---

## m.search(query, user_id=None, limit=10) → list[dict]

Search memories using the full 6-layer retrieval pipeline with cross-encoder reranking.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Natural language search query. |
| `user_id` | `str \| None` | `None` | Filter results to this user only. |
| `limit` | `int` | `10` | Maximum number of results. |

```python
results = m.search("What programming language does Alice prefer?", user_id="alice")
for r in results:
    print(f"{r['content']} (score: {r['score']:.2f})")
```

**Result dict keys:** `id`, `content`, `sender`, `recipient`, `timestamp`, `category`, `modality`, `score`, `source`, `user_id`.

---

## m.search_deep(query, user_id=None, limit=10, llm_fn=None) → list[dict]

Agentic multi-round search with HyDE query expansion and heavier reranking. Slower but higher accuracy. Use when `search()` doesn't find what you need.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Natural language search query. |
| `user_id` | `str \| None` | `None` | Filter results to this user. |
| `limit` | `int` | `10` | Maximum number of results. |
| `llm_fn` | `callable \| None` | `None` | LLM function for HyDE. If None, HyDE is skipped. |

```python
results = m.search_deep("What are all of Alice's technical preferences?", user_id="alice")
```

---

## m.search_vectors(query, limit=5) → list[dict]

Pure vector cosine similarity search. No FTS, no RRF fusion, no reranking. Returns results with `score` as cosine similarity in [0, 1].

Used internally by the encoding gate for novelty detection. Rarely needed directly.

```python
results = m.search_vectors("dark mode preference", limit=3)
```

---

## m.get(memory_id) → dict | None

Retrieve a single memory by its integer ID. Returns `None` if not found.

```python
memory = m.get(42)
if memory:
    print(memory["content"])
```

---

## m.get_all(user_id=None, limit=100, offset=0) → list[dict]

List all memories with optional user filtering and pagination.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_id` | `str \| None` | `None` | Filter to this user's memories. |
| `limit` | `int` | `100` | Maximum results per page. |
| `offset` | `int` | `0` | Skip this many results (for pagination). |

```python
# Get all of Alice's memories
all_memories = m.get_all(user_id="alice")
print(f"Alice has {len(all_memories)} memories")

# Paginate through all memories
page1 = m.get_all(limit=50, offset=0)
page2 = m.get_all(limit=50, offset=50)
```

---

## m.delete(memory_id) → bool

Delete a single memory by ID. Returns `True` if deleted.

```python
m.delete(42)
```

---

## m.delete_all(user_id=None) → bool

Delete all memories. If `user_id` is provided, only deletes that user's memories. Returns `True` if any rows were deleted.

```python
m.delete_all(user_id="alice")   # delete Alice's memories only
m.delete_all()                   # delete ALL memories (use with caution)
```

---

## m.stats() → dict

Return memory system statistics including message count, capabilities, and tier info.

```python
stats = m.stats()
print(f"Memories stored: {stats['message_count']}")
print(f"Capabilities: {stats['capabilities']}")
```

---

## m.close()

Close the database connection. Called automatically when using the context manager.

```python
m = Memory()
# ... use it ...
m.close()
```
