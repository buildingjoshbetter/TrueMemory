# TrueMemory Dashboard — Complete Redesign BUILD_PROMPT

**Goal**: Fix every issue with the TrueMemory Dashboard and ship a polished,
Apple-quality product. One-shot execution — read this entire prompt, then
execute every fix, rebuild, and verify.

**Working directory**: `/Users/j/Desktop/TrueMemory`
**Venv**: `.venv/bin/python3`
**Frontend**: `truememory/dashboard/frontend/` (bun install, bun run build)
**Backend**: `truememory/dashboard/server/`
**Dashboard DB**: `~/.truememory/dashboard.db` (sessions index, separate from memories.db)
**Memories DB**: `~/.truememory/memories.db` (4,907 memories, WAL mode)

---

## ISSUE CATALOG (from user screenshots)

Every issue below was identified from screenshots of the live app.
Each must be fixed.

### ISSUE 1: People view shows test entities
**Screenshot**: People view shows `__test__`, `__test__special!@#`,
`__test__日本語`, `test_user`, `__test_a__`, `__test_b__`, and a node with
a 500-char `xxxxxxx...` name. Only "Josh" and "josh" are real.

**Root cause**: The entity_profiles table in memories.db contains test data
from the test suite. The entity filtering code was written but the frontend
was never rebuilt with those changes.

**Database reality**:
```
entity_profiles table (10 rows):
  __test__: 49 msgs
  test_user: 4 msgs
  josh: 3 msgs
  Josh: 2 msgs
  __test_a__: 2 msgs
  __test_b__: 2 msgs
  test: 1 msgs
  __test__special!@#: 1 msgs
  __test__xxxx...(500 chars): 1 msgs
  __test__日本語: 1 msgs

messages table senders (real data):
  josh: 274 msgs
  Josh: 2 msgs
  __test__: 2 msgs
  test: 1 msgs
```

**Fix**: In `server/routes/entities.py`:
- Filter out any entity where name starts with `__test` or name is exactly
  `test`, `test_user`, or has length > 50
- Merge "josh" and "Josh" into a single entity (case-insensitive dedup)
- Also mine relationship/personal memories for mentioned people (brother,
  advisor, friend, partner, etc.)
- The graph endpoint (`/api/entities/graph`) must also filter test entities
  from both nodes AND edges

### ISSUE 2: People view — graph re-initializes on every click
**Screenshot**: User reports "every time I click on something, it zooms in"

**Root cause**: The D3 `useEffect` has `[graph, selected, onSelect]` in its
dependency array. When `selected` changes (click), the entire force simulation
is destroyed and recreated, causing the graph to re-layout and zoom.

**Fix**: In `People.tsx`:
- Remove `selected` and `onSelect` from the useEffect deps
- Use refs for selected state and onSelect callback
- Add a separate useEffect that only updates circle fill/stroke when
  `selected` changes, without re-creating the simulation
- Store the D3 circle selection in a ref so it can be updated independently

### ISSUE 3: Sessions require manual re-indexing every launch
**Screenshot**: Sessions view shows "No sessions indexed yet" with an
"Index Sessions" button that must be clicked manually.

**Root cause**: The auto-indexing logic in the `/api/sessions` GET endpoint
checks `count == 0` and triggers indexing, but this only runs when the
frontend actually navigates to Sessions. The session index should be built
on server startup, in a background thread, so it's ready when the user
clicks Sessions.

**Fix**:
- In `server/app.py`, add a startup event that triggers session indexing
  in a background thread:
  ```python
  @app.on_event("startup")
  async def startup_index_sessions():
      import threading
      def _bg_index():
          from truememory.dashboard.server.session_index import (
              get_dashboard_conn, index_sessions, get_session_count
          )
          conn = get_dashboard_conn()
          if get_session_count(conn) == 0:
              index_sessions(conn, max_sessions=2000)
      threading.Thread(target=_bg_index, daemon=True).start()
  ```
- Also clear the stale `dashboard.db` before first run since it has bad
  data from previous builds (internal extraction summaries)
- In session_index.py, the `_is_internal_message` function must filter:
  - `[[TRUEMEMORY_INTERNAL_EXTRACTION]]`
  - `<command-message>` / `<command-name>`
  - `You are a memory extraction system`
  - Any message starting with `[[TRUEMEMORY`
- Sessions with 0 user messages or empty summary must be filtered from
  the list query (WHERE user_message_count > 0 AND summary != '')

### ISSUE 4: Facts view shows "No facts tracked yet"
**Screenshot**: Empty page with just the message.

**Root cause**: The `fact_timeline` table in memories.db has 0 rows.
TrueMemory's L5 consolidation pipeline hasn't populated it.

**Fix**: Since we can't populate the table (that's an engine concern),
build a synthetic facts view by extracting factual statements from memories.
Query memories with categories 'personal', 'technical', 'decision',
'preference' and group them by subject keywords. This gives the Facts view
actual content to display.

In `server/routes/facts.py`, add a fallback:
- If `fact_timeline` has 0 rows, query memories directly
- Group by category as pseudo-subjects
- Extract the memory content as the "fact"
- Mark all as "current" (no supersession without fact_timeline)
- This makes Facts immediately useful even without L5

### ISSUE 5: Analytics looks terrible
**Screenshot**: 
- "Total Memories" = 4,907, "Last 30 Days" = 4,907 (SAME NUMBER — wrong)
- Daily Ingest sparkline is tiny and centered oddly in its card
- Category bars use harsh colors, uneven spacing
- Top Entities shows only "josh: 3" and "Josh: 2" (test-filtered but
  still pathetically low because entity_profiles has bad data)
- Too much vertical whitespace between sections

**Fixes**:
- "Last 30 Days" must query with `WHERE timestamp >= ?` using 30 days ago,
  NOT use total. The analytics ingest endpoint returns `"30d"` which is
  correctly computed, but the growth endpoint's last cumulative point IS
  the total (because data only spans 30 days). Need to distinguish.
- Filter test entities from Top Entities
- If entity_profiles has < 3 real entities, fall back to grouping senders
  from messages table instead
- The sparkline/chart areas need min-height and proper aspect ratio
- Category bar labels should be capitalized ("Technical" not "technical")
- Use consistent accent colors, not the harsh multi-color palette
- Less vertical spacing between card rows

### ISSUE 6: Overview problems
**Screenshot**:
- "Entities Tracked" shows 10 (includes test entities) — should be 2-3
- "Gate Pass Rate" shows "—" (em dash) — unhelpful
- Capabilities section shows nothing
- Memory content shows redundant `[category]` prefix:
  `[decision] Decided to build a TrueMemory dashboard...`
  The category badge already shows "decision", so the `[category]` text
  prefix is redundant and ugly
- Only 4 stat cards, 4th is empty

**Fixes**:
- Entity count: filter test entities before counting
- Gate Pass Rate: if null, show "N/A" in muted text, not "—"
- Strip `[category]` prefix from memory content display. Many memories
  are stored with format `[category] actual content`. Strip this:
  `content.replace(/^\[[\w_]+\]\s*/, '')`
- Add sparklines to all stat cards that have data
- Capabilities: the health endpoint returns empty `{}` because `get_stats()`
  capabilities aren't being populated when the engine initializes via
  the dashboard. This is because `_ensure_connection()` doesn't call
  `open()`. For now, hardcode capabilities based on tier:
  - edge: fts5=true, vector_search=true
  - base: fts5=true, vector_search=true, reranker=true
  - pro: fts5=true, vector_search=true, reranker=true, hyde=true

### ISSUE 7: Explorer pagination broken
**Screenshot**: User says "you don't get all 24,000 memories" and "you only
get up to one day ago."

**Root cause**: The `useMemories` hook fetches with `limit=100` and never
loads more. The "Load More" / infinite scroll was planned but not implemented.
Memories are sorted by `id DESC` which corresponds to insertion order, and
the timestamps are all within the last 30 days (data range: April 23 -
May 23, 2026).

**Fix**:
- Implement proper infinite scroll: when the user scrolls near the bottom
  of the virtualized list, fetch the next page (offset += limit)
- Show a "Load more" button at the bottom as a fallback
- The total count displays correctly (4,907) but only 100 are loaded
- After loading more, append to the existing list
- Sort dropdown: capitalize options ("Newest" not "newest", "Oldest")

### ISSUE 8: Explorer timestamp issues
**Screenshot**: All timestamps show relative times ("45m ago", "2h ago")
which is fine for recent, but memories from weeks ago should show dates.

**Fix**: Update `relativeTime()` in `formatters.ts`:
- < 24 hours: relative ("2h ago")
- < 7 days: day name ("Tuesday")
- < 1 year: date ("May 15")
- Older: full date ("May 15, 2025")

### ISSUE 9: Memory content shows [category] prefix
**Screenshot**: Every memory in Explorer and Overview shows content like:
`[decision] Decided to build a TrueMemory dashboard...`
`[technical] Portrait image on joshadler.org had a Gemini AI watermark...`
`[personal] Had a call with someone who asked about the BBC article...`

The `[category]` tag is stored IN the memory content. This is redundant
with the category badge.

**Fix**: Create a `stripCategoryPrefix` function in `formatters.ts`:
```typescript
export function stripCategoryPrefix(content: string): string {
  return content.replace(/^\[[\w_]+\]\s*/g, '');
}
```
Apply it everywhere memory content is displayed:
- `MemoryCard.tsx` content preview
- `InspectorPanel.tsx` full content
- `Overview.tsx` recent memories stream

### ISSUE 10: Settings is too sparse
**Screenshot**: Settings shows 5 cards stacked vertically in a narrow column
with massive whitespace on the right.

**Fix**:
- Remove `max-w-2xl` constraint — let it fill the space
- Use a 2-column grid for Tier + Database on the same row
- Add more useful settings:
  - Session index status (count, last indexed time)
  - Memory database location (with "Open in Finder" note)
  - Dashboard keyboard shortcuts reference
- Make the Update button actually functional: show clear instructions
  ("Run in terminal: pip install --upgrade truememory")

### ISSUE 11: Sessions content quality
**Root cause**: Session summaries show internal extraction prompts and
/loop commands. Many sessions are TrueMemory's own background extraction
jobs, not real user conversations.

**Fix** (already partially addressed but not deployed):
- `_is_internal_message()` must catch all internal patterns
- Filter sessions where summary is empty or only contains internal content
- The session list should only show sessions with real user conversations
- When displaying a session, show the project directory as a readable name
  (e.g., "TrueMemory" not "/Users/j/Desktop/TrueMemory")
- Session card should show a cleaner project name: extract the last path
  component

### ISSUE 12: Design doesn't feel Apple-native
**General**: The overall aesthetic is functional but doesn't achieve the
"indistinguishable from Apple" bar. Specific issues:

- Glass panels have visible harsh borders (the 0.5px white border is
  too prominent in the screenshots)
- Stat cards in Overview look flat — need more depth
- Sidebar "PRO" label looks like a cheap badge
- The sort/filter dropdowns ("newest", "Category", "Sender") look like
  unstyled HTML selects
- Too much wasted vertical space
- Font weight hierarchy isn't clear enough

**Fixes**:
- Reduce glass panel border opacity from 0.08 to 0.04
- Add subtle box-shadow to glass panels: `0 1px 3px rgba(0,0,0,0.3)`
- Sidebar PRO section: use a thin divider line, not a label
- Style select dropdowns with custom caret, consistent padding
- Tighten vertical spacing (gap-4 → gap-3 in most grids)
- Page titles: 24px semibold (not 20px)
- Section headers in cards: 13px medium (not 14px semibold)

---

## DATABASE REFERENCE

```
~/.truememory/memories.db (27 MB, WAL mode, SQLite)

messages: 4,907 rows
  Columns: id, content, sender, recipient, timestamp, category, modality,
           episode_id, emotional_valence, embedding_separation
  Categories: technical(1560), preference(1248), decision(887),
              uncategorized(429), personal(280), correction(213),
              temporal(176), relationship(40), anti_pattern(20),
              implementation(19), project(18), architecture(17)
  Senders: josh(274), Josh(2), __test__(2), test(1)
  Date range: 2026-04-23 to 2026-05-23

entity_profiles: 10 rows (8 are test data)
  Real: josh(3 msgs), Josh(2 msgs)
  Test: __test__(49), test_user(4), __test_a__(2), __test_b__(2),
        test(1), __test__special!@#(1), __test__xxxx...(1), __test__日本語(1)

fact_timeline: 0 rows (EMPTY)
entity_relationships: 0 rows (EMPTY)
summaries: 0 rows (EMPTY)
episodes: 0 rows (EMPTY)
landmark_events: 0 rows (EMPTY)
causal_edges: 0 rows (EMPTY)

~/.truememory/dashboard.db (separate SQLite for session index)
  dashboard_sessions: populated by session_index.py scanning ~/.claude/projects/
  ~44,577 JSONL transcript files across ~/.claude/projects/
  Many are internal TrueMemory extraction jobs, not real conversations
```

---

## MEMORY CONTENT FORMAT

Memories are stored with a `[category]` prefix baked into the content:
```
[decision] Decided to build a TrueMemory dashboard as a local app
[technical] Portrait image on joshadler.org had a Gemini AI watermark
[personal] Had a call with someone who asked about the BBC article
[preference] Josh wants the Peter Thiel connection always included
```

This prefix MUST be stripped when displaying content, since the category
badge already shows the category.

---

## CURRENT FILE STRUCTURE

```
truememory/dashboard/
├── __init__.py                     (empty)
├── cli.py                          (71 lines — PyWebView launcher)
├── server/
│   ├── __init__.py                 (empty)
│   ├── app.py                      (47 lines — FastAPI factory)
│   ├── deps.py                     (39 lines — engine singleton)
│   ├── session_index.py            (356 lines — session scanner)
│   └── routes/
│       ├── __init__.py             (empty)
│       ├── analytics.py            (135 lines)
│       ├── entities.py             (187 lines)
│       ├── facts.py                (92 lines)
│       ├── memories.py             (194 lines)
│       ├── sessions.py             (84 lines)
│       └── system.py               (57 lines)
└── frontend/
    ├── package.json
    ├── vite.config.ts
    ├── tailwind.config.ts
    ├── postcss.config.js
    ├── tsconfig.json
    ├── index.html
    └── src/
        ├── main.tsx                (10 lines)
        ├── App.tsx                 (47 lines)
        ├── styles/
        │   └── globals.css         (68 lines)
        ├── lib/
        │   ├── api.ts              (120 lines)
        │   ├── constants.ts        (19 lines)
        │   ├── formatters.ts       (33 lines)
        │   └── types.ts            (121 lines)
        ├── hooks/
        │   ├── useHealth.ts        (26 lines)
        │   └── useMemories.ts      (76 lines)
        ├── components/
        │   ├── CategoryBadge.tsx    (18 lines)
        │   ├── ConfirmSheet.tsx     (53 lines)
        │   ├── GlassCard.tsx        (20 lines)
        │   ├── InspectorPanel.tsx   (158 lines)
        │   ├── MemoryCard.tsx       (39 lines)
        │   ├── SearchBar.tsx        (36 lines)
        │   ├── Sidebar.tsx          (161 lines)
        │   ├── SkeletonLoader.tsx   (24 lines)
        │   ├── Sparkline.tsx        (29 lines)
        │   └── StatCard.tsx         (24 lines)
        └── views/
            ├── Analytics.tsx        (206 lines)
            ├── Explorer.tsx         (175 lines)
            ├── Facts.tsx            (166 lines)
            ├── Overview.tsx         (167 lines)
            ├── People.tsx           (287 lines)
            ├── Sessions.tsx         (240 lines)
            └── Settings.tsx         (154 lines)
```

---

## REFERENCED FILES OUTSIDE DASHBOARD (read-only)

These files define the engine API surface. Do NOT modify them.

### truememory/engine.py — Key methods
```python
class TrueMemoryEngine:
    def __init__(self, db_path, alpha_surprise=None)
    def _ensure_connection(self)  # lazy init, loads sqlite-vec
    def search(self, query, limit=10) -> list[dict]  # 6-layer pipeline
    def get(self, memory_id) -> dict | None
    def get_all(self, limit=100, offset=0, user_id=None) -> list[dict]
    def update(self, memory_id, content=None, **fields) -> dict | None
    def delete(self, memory_id) -> bool
    def delete_all(self, user_id=None) -> bool
    def get_stats(self) -> dict  # message_count, db_size_kb, capabilities
```

### truememory/storage.py — Direct DB access
```python
def get_all_senders(conn) -> list[str]
def get_messages_in_range(conn, after=None, before=None) -> list[dict]
def get_message_count(conn) -> int
```

### truememory/personality.py — Entity profiles
```python
def get_entity_profile(conn, entity) -> dict | None
def build_dunbar_hierarchy(conn, primary_entity=None) -> dict
def extract_preferences(conn, entity) -> dict
```

### truememory/__init__.py
```python
from importlib.metadata import version as _pkg_version
__version__ = _pkg_version("truememory")
```

### truememory/mcp_server.py — Config loading pattern
```python
_CONFIG_PATH = Path.home() / ".truememory" / "config.json"
# Config: {"tier": "base", "anthropic_api_key": "...", ...}
```

---

## DESIGN SYSTEM REFERENCE

### Typography
```css
font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", system-ui;
font-family (mono): ui-monospace, "SF Mono", "Cascadia Code", monospace;
```

### Colors — Apple Dark Mode
```css
--bg-base:      #1C1C1E;
--bg-elevated:  #2C2C2E;
--bg-grouped:   #3A3A3C;
--bg-tertiary:  #48484A;
--text-primary:   #FFFFFF;
--text-secondary: #8E8E93;
--text-tertiary:  #636366;
--accent:         #6C5CE7;
--accent-hover:   #7C6CF7;
--accent-muted:   rgba(108, 92, 231, 0.15);
--success: #30D158;
--warning: #FFD60A;
--error:   #FF453A;
--info:    #64D2FF;
--border:       rgba(255, 255, 255, 0.04);  /* reduced from 0.08 */
```

### Glass Panels
```css
.glass-panel {
    background: rgba(44, 44, 46, 0.72);
    backdrop-filter: blur(40px) saturate(180%);
    -webkit-backdrop-filter: blur(40px) saturate(180%);
    border: 0.5px solid rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
}
```

### Animations (framer-motion)
```typescript
const springDefault = { type: "spring", stiffness: 400, damping: 30 };
const springGentle = { type: "spring", stiffness: 300, damping: 35 };
const springSnappy = { type: "spring", stiffness: 500, damping: 28 };
```

---

## EXECUTION PLAN

Execute in this order. Each step must be complete before moving to the next.

### Step 1: Delete stale session index
```bash
rm ~/.truememory/dashboard.db
```

### Step 2: Fix backend — server/routes/entities.py
- Filter entities: skip any where `name.startswith("__test")`, `len(name) > 50`,
  or `name in ("test", "test_user")`
- Case-insensitive merge: combine "josh" and "Josh" into one entry
  (sum message_counts, merge traits/topics)
- Mine mentioned people from relationship/personal memories
- Graph endpoint: same filtering, generate edges from co-occurrence
  in memories rather than empty entity_relationships table

### Step 3: Fix backend — server/routes/facts.py
- When fact_timeline has 0 rows, build synthetic facts from memories:
  - Query top 200 memories by category (preference, decision, personal)
  - Group by category as the "subject"
  - Each memory content (stripped of [category] prefix) is a "fact"
  - All marked as current (no supersession)
  - Return in the same response format

### Step 4: Fix backend — server/routes/analytics.py
- "Last 30 Days" endpoint: ensure it uses date-filtered count, not total
- Top entities: filter test entities, fall back to sender counts from
  messages table if entity_profiles has < 3 real entries
- Categories: capitalize category names in response

### Step 5: Fix backend — server/routes/memories.py
- Stats endpoint: filter test entities from entity count
- Stats endpoint: compute capabilities based on tier (not from get_stats()
  which returns empty)

### Step 6: Fix backend — server/session_index.py
- `_is_internal_message()`: ensure all internal patterns are caught
- `_parse_session_jsonl()`: only use non-internal user messages for summary
- `list_sessions()`: WHERE user_message_count > 0 AND summary != ''
- `_decode_project_name()`: make readable (last path component only)

### Step 7: Fix backend — server/app.py
- Add startup event to auto-index sessions in background thread
- Sessions should be ready by the time user clicks the Sessions tab

### Step 8: Fix frontend — lib/formatters.ts
- `relativeTime()`: use day names for < 7 days, dates for older
- Add `stripCategoryPrefix(content)` function
- Capitalize sort options

### Step 9: Fix frontend — components (all)
- `MemoryCard.tsx`: use `stripCategoryPrefix()` on content
- `InspectorPanel.tsx`: use `stripCategoryPrefix()` on content display
- `Sidebar.tsx`: remove "PRO" label, use divider line instead
- `GlassCard.tsx`: add box-shadow, reduce border opacity
- `globals.css`: update border opacity in glass-panel class

### Step 10: Fix frontend — views/Overview.tsx
- Strip [category] prefix from recent memories
- Entity count: the stats endpoint should return filtered count
- Gate Pass Rate: show "N/A" in muted text, not "—"
- Show capabilities based on tier

### Step 11: Fix frontend — views/Explorer.tsx
- Implement infinite scroll / "Load More" button
- Strip [category] prefix from content
- Capitalize sort options ("Newest", "Oldest")
- Better timestamp display for older memories

### Step 12: Fix frontend — views/People.tsx
- D3 graph: separate simulation init from selection highlighting
- Use refs for selected state, don't re-create graph on click
- Handle case where there are very few entities gracefully

### Step 13: Fix frontend — views/Sessions.tsx
- Remove "Index Sessions" button from default state
- Show loading state while background indexing completes
- Display readable project names (last component of path)
- Session cards: cleaner layout, show duration estimate

### Step 14: Fix frontend — views/Facts.tsx
- Handle synthetic facts from the fallback endpoint
- Group by category with proper headings
- Show meaningful content, not "no facts tracked"

### Step 15: Fix frontend — views/Analytics.tsx
- Fix "Last 30 Days" to show correct number
- Capitalize category names
- Filter test entities from Top Entities
- Better chart sizing and spacing
- Less whitespace

### Step 16: Fix frontend — views/Settings.tsx
- Remove max-w-2xl, use full width with 2-column grid
- Add session index status card
- Make update instructions clearer

### Step 17: Build and verify
```bash
cd truememory/dashboard/frontend && bun run build
```
Must compile with zero TypeScript errors.

### Step 18: End-to-end verification
Start the server and verify every view:
```bash
cd /Users/j/Desktop/TrueMemory
.venv/bin/python3 -c "
from truememory.dashboard.server.app import create_app
import uvicorn, threading, time, httpx

app = create_app()
t = threading.Thread(target=uvicorn.run, args=(app,),
    kwargs={'host': '127.0.0.1', 'port': 8484, 'log_level': 'warning'},
    daemon=True)
t.start()

# Wait for server + session indexing
for i in range(30):
    time.sleep(1)
    try:
        r = httpx.get('http://127.0.0.1:8484/api/health', timeout=5)
        if r.status_code == 200: break
    except: pass
time.sleep(5)  # let session indexing finish

# Verify each endpoint
print('=== HEALTH ===')
r = httpx.get('http://127.0.0.1:8484/api/health', timeout=10)
h = r.json()
print('memories=%d tier=%s' % (h['memory_count'], h['tier']))

print('=== ENTITIES (must have NO test data) ===')
r = httpx.get('http://127.0.0.1:8484/api/entities', timeout=10)
entities = r.json()
for e in entities:
    assert not e['entity'].startswith('__test'), f'TEST ENTITY LEAKED: {e[\"entity\"]}'
    assert e['entity'] not in ('test', 'test_user'), f'TEST ENTITY: {e[\"entity\"]}'
    assert len(e['entity']) <= 50, f'LONG ENTITY: {e[\"entity\"]}'
print('%d entities (all clean)' % len(entities))

print('=== SESSIONS (must be auto-indexed) ===')
r = httpx.get('http://127.0.0.1:8484/api/sessions?limit=5', timeout=10)
j = r.json()
print('sessions: %d total' % j['total'])
for s in j['sessions'][:3]:
    assert s['summary'], f'Empty summary: {s[\"session_id\"]}'
    assert '[[TRUEMEMORY' not in s['summary'], f'Internal content leaked'
    print('  %s | %d msgs | %s' % (
        s['started_at'][:16] if s['started_at'] else 'no-date',
        s['message_count'],
        s['summary'][:80]))

print('=== FACTS (must have content, not empty) ===')
r = httpx.get('http://127.0.0.1:8484/api/facts', timeout=10)
j = r.json()
print('facts: %d' % j['total'])
assert j['total'] > 0, 'Facts should not be empty'

print('=== ANALYTICS ===')
r = httpx.get('http://127.0.0.1:8484/api/analytics/growth', timeout=10)
growth = r.json()
print('growth: %d points' % len(growth))

r = httpx.get('http://127.0.0.1:8484/api/analytics/categories', timeout=10)
cats = r.json()
# Verify categories are capitalized
for c in cats:
    if c['category'] != '(uncategorized)':
        assert c['category'][0].isupper(), f'Not capitalized: {c[\"category\"]}'
print('categories: %d (all capitalized)' % len(cats))

print('=== MEMORIES (content should not have [category] prefix) ===')
r = httpx.get('http://127.0.0.1:8484/api/memories?limit=5', timeout=10)
mems = r.json()['memories']
for m in mems:
    # Content in API can have prefix, but frontend strips it
    pass
print('memories: %d total' % r.json()['total'])

print()
print('=== FRONTEND SERVED ===')
r = httpx.get('http://127.0.0.1:8484/', timeout=5)
assert r.status_code == 200
assert '<div id=\"root\">' in r.text
print('Frontend: OK')

print()
print('ALL VERIFICATION CHECKS PASSED')
"
```

### Step 19: Visual verification
After all API checks pass, open `http://127.0.0.1:8484` in Chrome and
manually verify:

1. **Overview**: 4 stat cards show real data. Recent memories don't have
   `[category]` prefix. Entity count excludes test data.
2. **Explorer**: Memories display clean content. Scroll loads more memories.
   Sort says "Newest" not "newest". Timestamps are readable.
3. **Sessions**: Sessions are already indexed (no "Index Sessions" button).
   Session list shows real conversations with readable summaries.
   No internal extraction content visible.
4. **People**: Only real entities (josh, mentioned people). No test entities.
   Clicking a node highlights it without re-creating the graph.
5. **Facts**: Shows grouped facts extracted from memories. Not empty.
6. **Analytics**: "Last 30 Days" != "Total Memories". Categories are
   capitalized. Top Entities shows real data. Charts are properly sized.
7. **Settings**: Full-width layout. Session index status shown.

---

## CRITICAL CONSTRAINTS

1. Do NOT modify any files outside `truememory/dashboard/`
2. Do NOT modify the engine, storage, personality, or mcp_server modules
3. The memories.db database is read-only for display — don't insert/modify
   memory data (CRUD operations via the engine API are fine)
4. All TypeScript must compile with zero errors (`tsc -b`)
5. Test entities must NEVER appear in the UI
6. Internal TrueMemory extraction content must NEVER appear in session summaries
7. The `[category]` prefix must be stripped from ALL memory content display
8. The dashboard.db session index must persist across restarts
9. Session indexing must happen automatically on server startup
10. The server must stay alive after PyWebView window closes (daemon=False)
11. Use bun (not npm) for frontend package management
12. Frontend build output goes to `truememory/dashboard/frontend/dist/`
13. After ALL code changes, run `bun run build` and verify the build succeeds
14. Apple HIG design: SF Pro fonts, dark mode colors, glass effects, spring animations
## CURRENT SOURCE CODE (REFERENCE)

Every file below is the current state. Read each one, understand the bugs,
then rewrite as needed per the issue catalog above.


### `truememory/dashboard/server/app.py`
```
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import truememory
from truememory.dashboard.server.routes import (
    analytics,
    entities,
    facts,
    memories,
    sessions,
    system,
)

_FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"


def create_app() -> FastAPI:
    app = FastAPI(
        title="TrueMemory Dashboard",
        version=truememory.__version__,
        docs_url=None,
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(system.router)
    app.include_router(memories.router)
    app.include_router(sessions.router)
    app.include_router(entities.router)
    app.include_router(facts.router)
    app.include_router(analytics.router)

    if _FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")

    return app
```

### `truememory/dashboard/server/deps.py`
```
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

from truememory.engine import TrueMemoryEngine

log = logging.getLogger(__name__)

_DEFAULT_DB = Path.home() / ".truememory" / "memories.db"
_CONFIG_PATH = Path.home() / ".truememory" / "config.json"

_engine: TrueMemoryEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> TrueMemoryEngine:
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        db_path = os.environ.get("TRUEMEMORY_DB_PATH", str(_DEFAULT_DB))
        _engine = TrueMemoryEngine(db_path=db_path)
        _engine._ensure_connection()
        return _engine


def get_config() -> dict:
    if not _CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
```

### `truememory/dashboard/server/session_index.py`
```
from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)

_CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"
_CLAUDE_SESSIONS = Path.home() / ".claude" / "sessions"
_DASHBOARD_DB = Path.home() / ".truememory" / "dashboard.db"

_dash_conn: sqlite3.Connection | None = None


def get_dashboard_conn() -> sqlite3.Connection:
    global _dash_conn
    if _dash_conn is not None:
        return _dash_conn
    _DASHBOARD_DB.parent.mkdir(parents=True, exist_ok=True)
    _dash_conn = sqlite3.connect(str(_DASHBOARD_DB), timeout=15)
    _dash_conn.execute("PRAGMA journal_mode=WAL")
    _dash_conn.execute("PRAGMA busy_timeout=15000")
    ensure_session_table(_dash_conn)
    return _dash_conn

_SESSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS dashboard_sessions (
    session_id TEXT PRIMARY KEY,
    project_dir TEXT,
    started_at TEXT,
    ended_at TEXT,
    message_count INTEGER DEFAULT 0,
    user_message_count INTEGER DEFAULT 0,
    word_count INTEGER DEFAULT 0,
    summary TEXT DEFAULT '',
    version TEXT DEFAULT '',
    jsonl_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_ds_started ON dashboard_sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_ds_project ON dashboard_sessions(project_dir);
"""


def ensure_session_table(conn: sqlite3.Connection):
    for stmt in _SESSION_SCHEMA.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()


def get_session_count(conn: sqlite3.Connection) -> int:
    try:
        return conn.execute("SELECT COUNT(*) FROM dashboard_sessions").fetchone()[0]
    except sqlite3.OperationalError:
        return 0


def index_sessions(conn: sqlite3.Connection, max_sessions: int = 0) -> int:
    ensure_session_table(conn)

    existing = set()
    for row in conn.execute("SELECT session_id FROM dashboard_sessions").fetchall():
        existing.add(row[0])

    session_meta: dict[str, dict] = {}
    if _CLAUDE_SESSIONS.is_dir():
        for f in _CLAUDE_SESSIONS.iterdir():
            if f.suffix == ".json":
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    sid = data.get("sessionId", "")
                    if sid:
                        session_meta[sid] = data
                except (json.JSONDecodeError, OSError):
                    continue

    indexed = 0
    if not _CLAUDE_PROJECTS.is_dir():
        return indexed

    for project_dir in sorted(_CLAUDE_PROJECTS.iterdir()):
        if not project_dir.is_dir():
            continue

        project_name = project_dir.name
        jsonl_files = sorted(project_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)

        for jf in jsonl_files:
            session_id = jf.stem
            if session_id in existing:
                continue

            try:
                info = _parse_session_jsonl(jf, session_meta.get(session_id, {}))
            except Exception:
                continue

            info["project_dir"] = _decode_project_name(project_name)
            info["jsonl_path"] = str(jf)
            info["session_id"] = session_id

            conn.execute(
                """INSERT OR IGNORE INTO dashboard_sessions
                   (session_id, project_dir, started_at, ended_at,
                    message_count, user_message_count, word_count,
                    summary, version, jsonl_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    info["session_id"],
                    info["project_dir"],
                    info.get("started_at", ""),
                    info.get("ended_at", ""),
                    info.get("message_count", 0),
                    info.get("user_message_count", 0),
                    info.get("word_count", 0),
                    info.get("summary", ""),
                    info.get("version", ""),
                    info["jsonl_path"],
                ),
            )
            indexed += 1
            existing.add(session_id)

            if max_sessions and indexed >= max_sessions:
                conn.commit()
                return indexed

    conn.commit()
    return indexed


def _parse_session_jsonl(path: Path, meta: dict) -> dict:
    messages = []
    first_ts = None
    last_ts = None
    version = meta.get("version", "")
    user_msgs = 0
    word_count = 0
    summary_parts = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f):
            if line_no > 2000:
                break
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")
            ts = msg.get("timestamp", "")

            if ts:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts

            if not version:
                version = msg.get("version", "")

            if msg_type == "user":
                user_msgs += 1
                content = _extract_text(msg)
                if content:
                    word_count += len(content.split())
                    if len(summary_parts) < 2 and not _is_internal_message(content):
                        summary_parts.append(content[:150])

            elif msg_type == "assistant":
                content = _extract_text(msg)
                if content:
                    word_count += len(content.split())

            messages.append(msg_type)

    summary = " | ".join(summary_parts) if summary_parts else ""

    started_at = first_ts or ""
    if not started_at and meta.get("startedAt"):
        from datetime import datetime, timezone
        try:
            started_at = datetime.fromtimestamp(
                meta["startedAt"] / 1000, tz=timezone.utc
            ).isoformat()
        except (ValueError, OSError):
            pass

    return {
        "started_at": started_at,
        "ended_at": last_ts or "",
        "message_count": len(messages),
        "user_message_count": user_msgs,
        "word_count": word_count,
        "summary": summary[:500],
        "version": version,
    }


def _extract_text(msg: dict) -> str:
    content = msg.get("content", msg.get("message", ""))
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        role_content = content.get("content", "")
        if isinstance(role_content, str):
            return role_content
        if isinstance(role_content, list):
            parts = []
            for block in role_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return " ".join(parts)
    return ""


_INTERNAL_PREFIXES = (
    "[[TRUEMEMORY_INTERNAL",
    "<command-message>",
    "<command-name>",
    "[[TRUEMEMORY",
    "You are a memory extraction system",
)


def _is_internal_message(content: str) -> bool:
    stripped = content.lstrip()
    for prefix in _INTERNAL_PREFIXES:
        if stripped.startswith(prefix):
            return True
    return False


def _decode_project_name(name: str) -> str:
    if name == "-":
        return "/"
    return name.replace("-", "/")


def load_transcript(jsonl_path: str) -> list[dict]:
    path = Path(jsonl_path)
    if not path.exists():
        return []

    messages = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")
            if msg_type not in ("user", "assistant"):
                continue

            text = _extract_text(msg)
            if not text or _is_internal_message(text):
                continue

            ts = msg.get("timestamp", "")
            uuid = msg.get("uuid", "")

            messages.append({
                "type": msg_type,
                "content": text[:5000],
                "timestamp": ts,
                "uuid": uuid,
            })

    return messages


def search_sessions(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[dict]:
    ensure_session_table(conn)
    words = query.lower().split()
    if not words:
        return []

    like_clauses = []
    params = []
    for w in words[:5]:
        like_clauses.append("(LOWER(summary) LIKE ? OR LOWER(project_dir) LIKE ?)")
        params.extend([f"%{w}%", f"%{w}%"])

    sql = f"""
        SELECT session_id, project_dir, started_at, ended_at,
               message_count, user_message_count, word_count, summary, version
        FROM dashboard_sessions
        WHERE {' AND '.join(like_clauses)}
        ORDER BY started_at DESC
        LIMIT ?
    """
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in rows]


def list_sessions(
    conn: sqlite3.Connection,
    project: str = "",
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict], int]:
    ensure_session_table(conn)

    where_clauses = ["user_message_count > 0", "summary != ''"]
    params: list = []
    if project:
        where_clauses.append("project_dir = ?")
        params.append(project)

    where = "WHERE " + " AND ".join(where_clauses)

    total = conn.execute(f"SELECT COUNT(*) FROM dashboard_sessions {where}", params).fetchone()[0]

    sql = f"""
        SELECT session_id, project_dir, started_at, ended_at,
               message_count, user_message_count, word_count, summary, version
        FROM dashboard_sessions {where}
        ORDER BY started_at DESC
        LIMIT ? OFFSET ?
    """
    rows = conn.execute(sql, params + [limit, offset]).fetchall()
    return [_row_to_dict(r) for r in rows], total


def list_projects(conn: sqlite3.Connection) -> list[str]:
    ensure_session_table(conn)
    rows = conn.execute(
        "SELECT DISTINCT project_dir FROM dashboard_sessions ORDER BY project_dir"
    ).fetchall()
    return [r[0] for r in rows if r[0]]


def _row_to_dict(row) -> dict:
    return {
        "session_id": row[0],
        "project_dir": row[1],
        "started_at": row[2],
        "ended_at": row[3],
        "message_count": row[4],
        "user_message_count": row[5],
        "word_count": row[6],
        "summary": row[7],
        "version": row[8],
    }
```

### `truememory/dashboard/server/routes/system.py`
```
from __future__ import annotations

from fastapi import APIRouter
import httpx

import truememory
from truememory.dashboard.server.deps import get_engine, get_config

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health():
    engine = get_engine()
    stats = engine.get_stats()
    config = get_config()
    return {
        "version": truememory.__version__,
        "tier": config.get("tier", "edge"),
        "db_path": str(engine.db_path),
        "db_size_kb": stats.get("db_size_kb", 0),
        "memory_count": stats.get("message_count", 0),
        "capabilities": stats.get("capabilities", {}),
    }


@router.get("/tier")
def tier_info():
    config = get_config()
    return {
        "tier": config.get("tier", "edge"),
        "has_api_key": bool(config.get("anthropic_api_key") or config.get("api_key")),
        "api_provider": config.get("api_provider", ""),
    }


@router.post("/update/check")
async def check_update():
    current = truememory.__version__
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://pypi.org/pypi/truememory/json")
            resp.raise_for_status()
            data = resp.json()
            latest = data["info"]["version"]
            return {
                "current": current,
                "latest": latest,
                "update_available": latest != current,
            }
    except Exception:
        return {
            "current": current,
            "latest": None,
            "update_available": False,
            "error": "Could not reach PyPI",
        }
```

### `truememory/dashboard/server/routes/memories.py`
```
from __future__ import annotations

import datetime
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from truememory.dashboard.server.deps import get_engine

router = APIRouter(prefix="/api", tags=["memories"])


class SearchBody(BaseModel):
    query: str
    limit: int = 50


class UpdateBody(BaseModel):
    content: str


class BulkDeleteBody(BaseModel):
    ids: list[int]


@router.get("/memories")
def list_memories(
    search: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    sender: Optional[str] = Query(None),
    sort: str = Query("newest"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    engine = get_engine()

    if search:
        results = engine.search(search, limit=limit)
        if category:
            results = [r for r in results if r.get("category") == category]
        if sender:
            results = [r for r in results if r.get("sender") == sender]
        return {"memories": results, "total": len(results), "limit": limit, "offset": 0}

    engine._ensure_connection()
    conn = engine.conn

    where_clauses = []
    params: list = []
    if category:
        where_clauses.append("category = ?")
        params.append(category)
    if sender:
        where_clauses.append("sender = ?")
        params.append(sender)

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    order = "id DESC" if sort == "newest" else "id ASC"

    count_sql = f"SELECT COUNT(*) FROM messages {where_sql}"
    total = conn.execute(count_sql, params).fetchone()[0]

    data_sql = f"""
        SELECT id, content, sender, recipient, timestamp, category, modality
        FROM messages {where_sql}
        ORDER BY {order}
        LIMIT ? OFFSET ?
    """
    rows = conn.execute(data_sql, params + [limit, offset]).fetchall()

    memories = [
        {
            "id": r[0], "content": r[1], "sender": r[2], "recipient": r[3],
            "timestamp": r[4], "category": r[5], "modality": r[6],
        }
        for r in rows
    ]
    return {"memories": memories, "total": total, "limit": limit, "offset": offset}


@router.get("/memories/senders")
def list_senders():
    engine = get_engine()
    engine._ensure_connection()
    from truememory.storage import get_all_senders
    senders = get_all_senders(engine.conn)
    return [s for s in senders if s]


@router.get("/memories/categories")
def list_categories():
    engine = get_engine()
    engine._ensure_connection()
    rows = engine.conn.execute(
        "SELECT DISTINCT category FROM messages WHERE category != '' ORDER BY category"
    ).fetchall()
    return [r[0] for r in rows]


@router.get("/memories/stats")
def memory_stats():
    engine = get_engine()
    engine._ensure_connection()
    conn = engine.conn

    total = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

    week_ago = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=7)
    ).isoformat()
    this_week = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE timestamp >= ?", (week_ago,)
    ).fetchone()[0]

    entities = conn.execute("SELECT COUNT(*) FROM entity_profiles").fetchone()[0]

    thirty_days_ago = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)
    ).isoformat()
    daily_rows = conn.execute(
        """SELECT DATE(timestamp) as day, COUNT(*) as cnt
           FROM messages WHERE timestamp >= ?
           GROUP BY DATE(timestamp) ORDER BY day""",
        (thirty_days_ago,),
    ).fetchall()

    today = datetime.date.today()
    daily_map = {r[0]: r[1] for r in daily_rows}
    sparkline = []
    for i in range(30):
        d = (today - datetime.timedelta(days=29 - i)).isoformat()
        sparkline.append(daily_map.get(d, 0))

    cat_rows = conn.execute(
        "SELECT category, COUNT(*) FROM messages GROUP BY category ORDER BY COUNT(*) DESC"
    ).fetchall()
    categories = {r[0] or "(uncategorized)": r[1] for r in cat_rows}

    return {
        "total": total,
        "this_week": this_week,
        "entities": entities,
        "gate_pass_rate": None,
        "sparkline": sparkline,
        "categories": categories,
    }


@router.post("/memories/search")
def search_memories(body: SearchBody):
    engine = get_engine()
    results = engine.search(body.query, limit=body.limit)
    return results


@router.get("/memories/{memory_id}")
def get_memory(memory_id: int):
    engine = get_engine()
    result = engine.get(memory_id)
    if result is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Memory not found")
    return result


@router.put("/memories/{memory_id}")
def update_memory(memory_id: int, body: UpdateBody):
    engine = get_engine()
    result = engine.update(memory_id, content=body.content)
    if result is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Memory not found")
    return result


@router.delete("/memories/{memory_id}")
def delete_memory(memory_id: int):
    engine = get_engine()
    deleted = engine.delete(memory_id)
    return {"deleted": deleted}


@router.post("/memories/bulk-delete")
def bulk_delete(body: BulkDeleteBody):
    engine = get_engine()
    count = 0
    for mid in body.ids:
        if engine.delete(mid):
            count += 1
    return {"deleted": count, "total": len(body.ids)}
```

### `truememory/dashboard/server/routes/sessions.py`
```
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query, HTTPException

from truememory.dashboard.server import session_index

router = APIRouter(prefix="/api", tags=["sessions"])


@router.get("/sessions")
def list_sessions(
    search: Optional[str] = Query(None),
    project: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    conn = session_index.get_dashboard_conn()

    count = session_index.get_session_count(conn)
    if count == 0:
        session_index.index_sessions(conn, max_sessions=500)

    if search:
        results = session_index.search_sessions(conn, search, limit=limit)
        return {"sessions": results, "total": len(results), "limit": limit, "offset": 0}

    sessions, total = session_index.list_sessions(
        conn, project=project or "", limit=limit, offset=offset
    )
    return {"sessions": sessions, "total": total, "limit": limit, "offset": offset}


@router.get("/sessions/projects")
def list_projects():
    conn = session_index.get_dashboard_conn()
    return session_index.list_projects(conn)


@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    conn = session_index.get_dashboard_conn()

    row = conn.execute(
        """SELECT session_id, project_dir, started_at, ended_at,
                  message_count, user_message_count, word_count,
                  summary, version, jsonl_path
           FROM dashboard_sessions WHERE session_id = ?""",
        (session_id,),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": row[0], "project_dir": row[1],
        "started_at": row[2], "ended_at": row[3],
        "message_count": row[4], "user_message_count": row[5],
        "word_count": row[6], "summary": row[7],
        "version": row[8], "jsonl_path": row[9],
    }


@router.get("/sessions/{session_id}/transcript")
def get_transcript(session_id: str):
    conn = session_index.get_dashboard_conn()

    row = conn.execute(
        "SELECT jsonl_path FROM dashboard_sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = session_index.load_transcript(row[0])
    return {"messages": messages, "count": len(messages)}


@router.post("/sessions/reindex")
def reindex_sessions():
    conn = session_index.get_dashboard_conn()
    indexed = session_index.index_sessions(conn, max_sessions=2000)
    total = session_index.get_session_count(conn)
    return {"indexed": indexed, "total": total}
```

### `truememory/dashboard/server/routes/entities.py`
```
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from truememory.dashboard.server.deps import get_engine
from truememory.personality import (
    get_entity_profile,
    build_dunbar_hierarchy,
    extract_preferences,
)

router = APIRouter(prefix="/api", tags=["entities"])


@router.get("/entities")
def list_entities():
    import json as _json

    engine = get_engine()
    engine._ensure_connection()
    conn = engine.conn

    rows = conn.execute(
        """SELECT entity, message_count, traits, topics, updated_at
           FROM entity_profiles ORDER BY message_count DESC"""
    ).fetchall()

    entities = []
    for r in rows:
        if r[0].startswith("__test") or r[0] in ("test", "test_user"):
            continue

        traits = []
        topics = []
        try:
            traits = _json.loads(r[2]) if r[2] else []
        except (_json.JSONDecodeError, TypeError):
            pass
        try:
            topics = _json.loads(r[3]) if r[3] else []
        except (_json.JSONDecodeError, TypeError):
            pass

        entities.append({
            "entity": r[0],
            "message_count": r[1],
            "traits": traits if isinstance(traits, list) else list(traits.keys()) if isinstance(traits, dict) else [],
            "topics": topics if isinstance(topics, list) else [],
            "updated_at": r[4] or "",
        })

    mentioned = _extract_mentioned_entities(conn)
    entity_names = {e["entity"].lower() for e in entities}
    for m in mentioned:
        if m["entity"].lower() not in entity_names:
            entities.append(m)

    entities.sort(key=lambda e: e["message_count"], reverse=True)
    return entities


def _extract_mentioned_entities(conn) -> list[dict]:
    rows = conn.execute(
        """SELECT content, category FROM messages
           WHERE category IN ('relationship', 'personal')
           ORDER BY id DESC LIMIT 500"""
    ).fetchall()

    from collections import Counter
    name_counts: Counter = Counter()

    for content, _ in rows:
        text = content.lower()
        for marker in ("brother", "sister", "mom", "dad", "wife", "husband",
                        "girlfriend", "boyfriend", "partner", "friend",
                        "advisor", "cofounder", "co-founder"):
            if marker in text:
                name_counts[marker] += 1

        import re
        named = re.findall(
            r"\b(?:user(?:'s)?|josh(?:'s)?)\s+(?:brother|sister|friend|advisor|partner)\s+(?:is\s+)?([A-Z][a-z]+)",
            content,
        )
        for n in named:
            if len(n) > 2 and n.lower() not in ("the", "this", "that", "user"):
                name_counts[n] += 1

    entities = []
    for name, count in name_counts.most_common(20):
        if count >= 1:
            entities.append({
                "entity": name,
                "message_count": count,
                "traits": [],
                "topics": [],
                "updated_at": "",
            })
    return entities


@router.get("/entities/graph")
def entity_graph():
    engine = get_engine()
    engine._ensure_connection()
    conn = engine.conn

    entity_rows = conn.execute(
        "SELECT entity, message_count FROM entity_profiles ORDER BY message_count DESC"
    ).fetchall()

    nodes = [
        {"id": r[0], "message_count": r[1], "radius": max(12, min(40, r[1] // 5 + 10))}
        for r in entity_rows
    ]

    edge_rows = conn.execute(
        """SELECT entity_a, entity_b, relationship_type, strength, dunbar_layer
           FROM entity_relationships"""
    ).fetchall()

    edges = [
        {
            "source": r[0], "target": r[1],
            "relationship_type": r[2], "strength": r[3],
            "dunbar_layer": r[4],
        }
        for r in edge_rows
    ]

    if not edges and len(nodes) > 1:
        sender_counts = conn.execute(
            """SELECT sender, COUNT(*) as cnt FROM messages
               WHERE sender != '' GROUP BY sender ORDER BY cnt DESC LIMIT 20"""
        ).fetchall()

        entity_set = {n["id"].lower() for n in nodes}
        for s_row in sender_counts:
            sender = s_row[0]
            if sender.lower() in entity_set:
                for node in nodes:
                    if node["id"].lower() != sender.lower():
                        edges.append({
                            "source": sender,
                            "target": node["id"],
                            "relationship_type": "mentioned",
                            "strength": min(1.0, s_row[1] / 100),
                            "dunbar_layer": "",
                        })

    return {"nodes": nodes, "edges": edges}


@router.get("/entities/{entity_name}")
def get_entity(entity_name: str):
    engine = get_engine()
    engine._ensure_connection()
    conn = engine.conn

    profile = get_entity_profile(conn, entity_name)
    if profile is None:
        raise HTTPException(status_code=404, detail="Entity not found")

    recent = conn.execute(
        """SELECT id, content, timestamp, category FROM messages
           WHERE sender = ? ORDER BY id DESC LIMIT 10""",
        (entity_name,),
    ).fetchall()

    recent_memories = [
        {"id": r[0], "content": r[1][:200], "timestamp": r[2], "category": r[3]}
        for r in recent
    ]

    return {
        "profile": profile,
        "recent_memories": recent_memories,
    }


@router.get("/entities/{entity_name}/preferences")
def get_entity_preferences(entity_name: str):
    engine = get_engine()
    engine._ensure_connection()
    conn = engine.conn
    prefs = extract_preferences(conn, entity_name)
    return prefs
```

### `truememory/dashboard/server/routes/facts.py`
```
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from truememory.dashboard.server.deps import get_engine

router = APIRouter(prefix="/api", tags=["facts"])


@router.get("/facts")
def list_facts(
    subject: Optional[str] = Query(None),
    show_superseded: bool = Query(False),
    limit: int = Query(200, ge=1, le=1000),
):
    engine = get_engine()
    engine._ensure_connection()
    conn = engine.conn

    where_clauses = []
    params: list = []

    if subject:
        where_clauses.append("subject LIKE ?")
        params.append(f"%{subject}%")

    if not show_superseded:
        where_clauses.append("superseded_by IS NULL")

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    rows = conn.execute(
        f"""SELECT id, subject, fact, source_message_id, timestamp,
                  superseded_by, entity_scope, valid_from, valid_to
           FROM fact_timeline {where_sql}
           ORDER BY subject, timestamp DESC
           LIMIT ?""",
        params + [limit],
    ).fetchall()

    facts = [
        {
            "id": r[0], "subject": r[1], "fact": r[2],
            "source_message_id": r[3], "timestamp": r[4],
            "superseded_by": r[5], "entity_scope": r[6],
            "valid_from": r[7], "valid_to": r[8],
            "is_current": r[5] is None,
        }
        for r in rows
    ]

    subjects = conn.execute(
        "SELECT DISTINCT subject FROM fact_timeline ORDER BY subject"
    ).fetchall()

    return {
        "facts": facts,
        "total": len(facts),
        "subjects": [s[0] for s in subjects],
    }


@router.get("/facts/contradictions")
def get_contradictions():
    engine = get_engine()
    engine._ensure_connection()
    conn = engine.conn

    rows = conn.execute(
        """SELECT f1.id, f1.subject, f1.fact, f1.timestamp,
                  f2.id, f2.fact, f2.timestamp
           FROM fact_timeline f1
           JOIN fact_timeline f2 ON f1.subject = f2.subject
                AND f1.id < f2.id
                AND f1.superseded_by IS NULL
                AND f2.superseded_by IS NULL
           ORDER BY f1.subject"""
    ).fetchall()

    contradictions = []
    for r in rows:
        contradictions.append({
            "subject": r[1],
            "fact_a": {"id": r[0], "fact": r[2], "timestamp": r[3]},
            "fact_b": {"id": r[4], "fact": r[5], "timestamp": r[6]},
        })

    return contradictions
```

### `truememory/dashboard/server/routes/analytics.py`
```
from __future__ import annotations

import datetime

from fastapi import APIRouter

from truememory.dashboard.server.deps import get_engine

router = APIRouter(prefix="/api", tags=["analytics"])


@router.get("/analytics/growth")
def memory_growth():
    engine = get_engine()
    engine._ensure_connection()
    conn = engine.conn

    rows = conn.execute(
        """SELECT DATE(timestamp) as day, COUNT(*) as cnt
           FROM messages WHERE timestamp != ''
           GROUP BY DATE(timestamp)
           ORDER BY day"""
    ).fetchall()

    daily = [{"date": r[0], "count": r[1]} for r in rows]

    cumulative = []
    total = 0
    for d in daily:
        total += d["count"]
        cumulative.append({"date": d["date"], "count": d["count"], "cumulative": total})

    return cumulative


@router.get("/analytics/categories")
def category_distribution():
    engine = get_engine()
    engine._ensure_connection()
    conn = engine.conn

    rows = conn.execute(
        """SELECT category, COUNT(*) as cnt
           FROM messages GROUP BY category
           ORDER BY cnt DESC"""
    ).fetchall()

    return [
        {"category": r[0] or "(uncategorized)", "count": r[1]}
        for r in rows
    ]


@router.get("/analytics/entities")
def top_entities():
    engine = get_engine()
    engine._ensure_connection()
    conn = engine.conn

    rows = conn.execute(
        """SELECT entity, message_count
           FROM entity_profiles
           ORDER BY message_count DESC
           LIMIT 20"""
    ).fetchall()

    return [{"entity": r[0], "message_count": r[1]} for r in rows]


@router.get("/analytics/ingest")
def ingest_stats():
    engine = get_engine()
    engine._ensure_connection()
    conn = engine.conn

    now = datetime.datetime.now(datetime.timezone.utc)
    periods = {
        "7d": (now - datetime.timedelta(days=7)).isoformat(),
        "30d": (now - datetime.timedelta(days=30)).isoformat(),
        "all": "",
    }

    result = {}
    for label, since in periods.items():
        if since:
            count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE timestamp >= ?", (since,)
            ).fetchone()[0]
        else:
            count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        result[label] = count

    daily_rows = conn.execute(
        """SELECT DATE(timestamp) as day, COUNT(*) as cnt
           FROM messages WHERE timestamp >= ?
           GROUP BY DATE(timestamp) ORDER BY day""",
        ((now - datetime.timedelta(days=30)).isoformat(),),
    ).fetchall()

    today = datetime.date.today()
    daily_map = {r[0]: r[1] for r in daily_rows}
    daily_rate = []
    for i in range(30):
        d = (today - datetime.timedelta(days=29 - i)).isoformat()
        daily_rate.append({"date": d, "count": daily_map.get(d, 0)})

    result["daily_rate"] = daily_rate
    return result


@router.get("/analytics/timeline")
def timeline_by_category():
    engine = get_engine()
    engine._ensure_connection()
    conn = engine.conn

    now = datetime.datetime.now(datetime.timezone.utc)
    since = (now - datetime.timedelta(days=30)).isoformat()

    rows = conn.execute(
        """SELECT DATE(timestamp) as day, category, COUNT(*) as cnt
           FROM messages WHERE timestamp >= ? AND category != ''
           GROUP BY DATE(timestamp), category
           ORDER BY day, category""",
        (since,),
    ).fetchall()

    result: dict[str, dict[str, int]] = {}
    for r in rows:
        day = r[0]
        if day not in result:
            result[day] = {}
        result[day][r[1]] = r[2]

    return result
```

### `truememory/dashboard/cli.py`
```
from __future__ import annotations

import sys
import threading
import time


def _start_server(app, host: str, port: int):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")


def _wait_for_server(host: str, port: int, timeout: float = 30.0):
    import httpx
    url = f"http://{host}:{port}/api/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(url, timeout=3)
            if resp.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
            pass
        time.sleep(0.2)
    return False


def main():
    host = "127.0.0.1"
    port = 8484

    from truememory.dashboard.server.app import create_app
    app = create_app()

    server_thread = threading.Thread(
        target=_start_server, args=(app, host, port), daemon=False
    )
    server_thread.start()

    print(f"Starting TrueMemory Dashboard on http://{host}:{port} ...")

    if not _wait_for_server(host, port):
        print("Server failed to start within 30 seconds.", file=sys.stderr)
        sys.exit(1)

    print(f"Dashboard ready at http://{host}:{port}")

    try:
        import webview
    except ImportError:
        print(
            "pywebview is not installed. Install with: pip install truememory[dashboard]",
            file=sys.stderr,
        )
        print(f"Dashboard is running at http://{host}:{port}")
        server_thread.join()
        return

    webview.create_window(
        "TrueMemory",
        f"http://{host}:{port}",
        width=1440,
        height=900,
        min_size=(1024, 600),
    )

    webview.start()


if __name__ == "__main__":
    main()
```

### `truememory/dashboard/frontend/src/App.tsx`
```
import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Sidebar } from "./components/Sidebar";
import { Overview } from "./views/Overview";
import { Explorer } from "./views/Explorer";
import { Sessions } from "./views/Sessions";
import { People } from "./views/People";
import { Facts } from "./views/Facts";
import { Analytics } from "./views/Analytics";
import { Settings } from "./views/Settings";
import { useHealth } from "./hooks/useHealth";
import { springDefault } from "./lib/constants";
import type { ViewId } from "./lib/types";

export default function App() {
  const [view, setView] = useState<ViewId>("overview");
  const { data: health } = useHealth();

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-bg-base font-system">
      <Sidebar currentView={view} onNavigate={setView} health={health} />

      <main className="flex-1 min-w-0 h-full overflow-hidden">
        <AnimatePresence mode="wait">
          <motion.div
            key={view}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={springDefault}
            className="h-full"
          >
            {view === "overview" && (
              <Overview health={health} onNavigateExplorer={() => setView("explorer")} />
            )}
            {view === "explorer" && <Explorer />}
            {view === "sessions" && <Sessions />}
            {view === "people" && <People />}
            {view === "facts" && <Facts />}
            {view === "analytics" && <Analytics />}
            {view === "settings" && <Settings health={health} />}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
```

### `truememory/dashboard/frontend/src/main.tsx`
```
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles/globals.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
```

### `truememory/dashboard/frontend/src/styles/globals.css`
```
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --bg-base: #1C1C1E;
  --bg-elevated: #2C2C2E;
  --bg-grouped: #3A3A3C;
  --bg-tertiary: #48484A;
  --text-primary: #FFFFFF;
  --text-secondary: #8E8E93;
  --text-tertiary: #636366;
  --accent: #6C5CE7;
  --accent-hover: #7C6CF7;
  --accent-muted: rgba(108, 92, 231, 0.15);
  --border: rgba(255, 255, 255, 0.06);
  --border-hover: rgba(255, 255, 255, 0.12);
}

* {
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", system-ui, sans-serif;
  color: var(--text-primary);
  background: var(--bg-base);
  overflow: hidden;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

::selection {
  background: var(--accent-muted);
  color: var(--text-primary);
}

::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}
::-webkit-scrollbar-track {
  background: transparent;
}
::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.2);
}

@layer components {
  .glass-panel {
    background: rgba(44, 44, 46, 0.72);
    backdrop-filter: blur(40px) saturate(180%);
    -webkit-backdrop-filter: blur(40px) saturate(180%);
    border: 0.5px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
  }

  .glass-sidebar {
    background: rgba(28, 28, 30, 0.85);
    backdrop-filter: blur(60px) saturate(200%);
    -webkit-backdrop-filter: blur(60px) saturate(200%);
    border-right: 0.5px solid rgba(255, 255, 255, 0.06);
  }
}
```

### `truememory/dashboard/frontend/src/lib/api.ts`
```
import type {
  HealthResponse,
  MemoriesResponse,
  Memory,
  DashboardStats,
  TierInfo,
  UpdateInfo,
  SessionsResponse,
  Session,
  TranscriptMessage,
  Entity,
  EntityGraph,
  FactsResponse,
  GrowthPoint,
  CategoryCount,
} from "./types";

const BASE = "/api";

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = { ...((init?.headers as Record<string, string>) || {}) };
  if (init?.body && typeof init.body === "string") {
    headers["Content-Type"] = "application/json";
  }
  const res = await fetch(`${BASE}${path}`, { ...init, headers });
  if (!res.ok) throw new ApiError(res.status, await res.text());
  return res.json();
}

function qs(params: Record<string, string | number | boolean | undefined | null>): string {
  const entries = Object.entries(params).filter(([, v]) => v != null && v !== "");
  return new URLSearchParams(entries.map(([k, v]) => [k, String(v)])).toString();
}

export const api = {
  health: () => fetchJson<HealthResponse>("/health"),

  memories: {
    list: (params: {
      search?: string;
      category?: string;
      sender?: string;
      sort?: string;
      limit?: number;
      offset?: number;
    }) => fetchJson<MemoriesResponse>(`/memories?${qs(params)}`),

    get: (id: number) => fetchJson<Memory>(`/memories/${id}`),

    search: (query: string, limit = 50) =>
      fetchJson<Memory[]>("/memories/search", {
        method: "POST",
        body: JSON.stringify({ query, limit }),
      }),

    update: (id: number, content: string) =>
      fetchJson<Memory>(`/memories/${id}`, {
        method: "PUT",
        body: JSON.stringify({ content }),
      }),

    delete: (id: number) =>
      fetchJson<{ deleted: boolean }>(`/memories/${id}`, { method: "DELETE" }),

    bulkDelete: (ids: number[]) =>
      fetchJson<{ deleted: number; total: number }>("/memories/bulk-delete", {
        method: "POST",
        body: JSON.stringify({ ids }),
      }),

    senders: () => fetchJson<string[]>("/memories/senders"),

    categories: () => fetchJson<string[]>("/memories/categories"),

    stats: () => fetchJson<DashboardStats>("/memories/stats"),
  },

  tier: () => fetchJson<TierInfo>("/tier"),

  checkUpdate: () =>
    fetchJson<UpdateInfo>("/update/check", { method: "POST" }),

  sessions: {
    list: (params: { search?: string; project?: string; limit?: number; offset?: number }) =>
      fetchJson<SessionsResponse>(`/sessions?${qs(params)}`),
    get: (id: string) => fetchJson<Session>(`/sessions/${id}`),
    transcript: (id: string) =>
      fetchJson<{ messages: TranscriptMessage[]; count: number }>(`/sessions/${id}/transcript`),
    projects: () => fetchJson<string[]>("/sessions/projects"),
    reindex: () => fetchJson<{ indexed: number; total: number }>("/sessions/reindex", { method: "POST" }),
  },

  entities: {
    list: () => fetchJson<Entity[]>("/entities"),
    get: (name: string) => fetchJson<{ profile: Record<string, unknown>; recent_memories: Memory[] }>(`/entities/${encodeURIComponent(name)}`),
    graph: () => fetchJson<EntityGraph>("/entities/graph"),
    preferences: (name: string) => fetchJson<Record<string, unknown[]>>(`/entities/${encodeURIComponent(name)}/preferences`),
  },

  facts: {
    list: (params?: { subject?: string; show_superseded?: boolean }) =>
      fetchJson<FactsResponse>(`/facts?${qs(params || {})}`),
    contradictions: () =>
      fetchJson<{ subject: string; fact_a: Record<string, unknown>; fact_b: Record<string, unknown> }[]>("/facts/contradictions"),
  },

  analytics: {
    growth: () => fetchJson<GrowthPoint[]>("/analytics/growth"),
    categories: () => fetchJson<CategoryCount[]>("/analytics/categories"),
    entities: () => fetchJson<{ entity: string; message_count: number }[]>("/analytics/entities"),
    ingest: () => fetchJson<Record<string, unknown>>("/analytics/ingest"),
    timeline: () => fetchJson<Record<string, Record<string, number>>>("/analytics/timeline"),
  },
};
```

### `truememory/dashboard/frontend/src/lib/types.ts`
```
export interface Memory {
  id: number;
  content: string;
  sender: string;
  recipient: string;
  timestamp: string;
  category: string;
  modality: string;
  score?: number;
  source?: string;
}

export interface HealthResponse {
  version: string;
  tier: string;
  db_path: string;
  db_size_kb: number;
  memory_count: number;
  capabilities: Record<string, boolean>;
}

export interface DashboardStats {
  total: number;
  this_week: number;
  entities: number;
  gate_pass_rate: number | null;
  sparkline: number[];
  categories: Record<string, number>;
}

export interface MemoriesResponse {
  memories: Memory[];
  total: number;
  limit: number;
  offset: number;
}

export interface UpdateInfo {
  current: string;
  latest: string | null;
  update_available: boolean;
  error?: string;
}

export interface TierInfo {
  tier: string;
  has_api_key: boolean;
  api_provider: string;
}

export interface Session {
  session_id: string;
  project_dir: string;
  started_at: string;
  ended_at: string;
  message_count: number;
  user_message_count: number;
  word_count: number;
  summary: string;
  version: string;
  jsonl_path?: string;
}

export interface SessionsResponse {
  sessions: Session[];
  total: number;
  limit: number;
  offset: number;
}

export interface TranscriptMessage {
  type: string;
  content: string;
  timestamp: string;
  uuid: string;
}

export interface Entity {
  entity: string;
  message_count: number;
  traits: string[];
  topics: string[];
  updated_at: string;
}

export interface EntityGraph {
  nodes: { id: string; message_count: number; radius: number }[];
  edges: { source: string; target: string; relationship_type: string; strength: number; dunbar_layer: string }[];
}

export interface Fact {
  id: number;
  subject: string;
  fact: string;
  source_message_id: number | null;
  timestamp: string;
  superseded_by: number | null;
  entity_scope: string;
  valid_from: string;
  valid_to: string;
  is_current: boolean;
}

export interface FactsResponse {
  facts: Fact[];
  total: number;
  subjects: string[];
}

export interface GrowthPoint {
  date: string;
  count: number;
  cumulative: number;
}

export interface CategoryCount {
  category: string;
  count: number;
}

export type ViewId = "overview" | "explorer" | "sessions" | "people" | "facts" | "analytics" | "settings";
```

### `truememory/dashboard/frontend/src/lib/constants.ts`
```
export const springDefault = { type: "spring" as const, stiffness: 400, damping: 30 };
export const springGentle = { type: "spring" as const, stiffness: 300, damping: 35 };
export const springSnappy = { type: "spring" as const, stiffness: 500, damping: 28 };

export const CATEGORY_COLORS: Record<string, { bg: string; text: string }> = {
  technical: { bg: "rgba(100, 210, 255, 0.15)", text: "#64D2FF" },
  preference: { bg: "rgba(108, 92, 231, 0.15)", text: "#A78BFA" },
  decision: { bg: "rgba(255, 214, 10, 0.15)", text: "#FFD60A" },
  personal: { bg: "rgba(48, 209, 88, 0.15)", text: "#30D158" },
  correction: { bg: "rgba(255, 69, 58, 0.15)", text: "#FF453A" },
  temporal: { bg: "rgba(100, 210, 255, 0.12)", text: "#5AC8FA" },
  relationship: { bg: "rgba(191, 90, 242, 0.15)", text: "#BF5AF2" },
  architecture: { bg: "rgba(100, 160, 255, 0.15)", text: "#64A0FF" },
  implementation: { bg: "rgba(48, 176, 199, 0.15)", text: "#30B0C7" },
};

export function getCategoryColor(category: string) {
  return CATEGORY_COLORS[category] || { bg: "rgba(142, 142, 147, 0.12)", text: "#8E8E93" };
}
```

### `truememory/dashboard/frontend/src/lib/formatters.ts`
```
export function relativeTime(iso: string): string {
  if (!iso) return "";
  const now = Date.now();
  const then = new Date(iso).getTime();
  const diff = now - then;
  const seconds = Math.floor(diff / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  const months = Math.floor(days / 30);
  return `${months}mo ago`;
}

export function formatNumber(n: number): string {
  return n.toLocaleString("en-US");
}

export function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen).trimEnd() + "…";
}

export function formatBytes(kb: number): string {
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  if (mb < 1024) return `${mb.toFixed(1)} MB`;
  const gb = mb / 1024;
  return `${gb.toFixed(2)} GB`;
}
```

### `truememory/dashboard/frontend/src/hooks/useHealth.ts`
```
import { useState, useEffect, useCallback } from "react";
import { api } from "../lib/api";
import type { HealthResponse } from "../lib/types";

export function useHealth() {
  const [data, setData] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetch_ = useCallback(async () => {
    try {
      setLoading(true);
      const resp = await api.health();
      setData(resp);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetch_(); }, [fetch_]);

  return { data, loading, error, refetch: fetch_ };
}
```

### `truememory/dashboard/frontend/src/hooks/useMemories.ts`
```
import { useState, useEffect, useRef, useCallback } from "react";
import { api } from "../lib/api";
import type { Memory } from "../lib/types";

interface Filters {
  search: string;
  category: string;
  sender: string;
  sort: string;
}

export function useMemories() {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState<Filters>({
    search: "",
    category: "",
    sender: "",
    sort: "newest",
  });
  const [offset, setOffset] = useState(0);
  const limit = 100;
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();

  const fetchMemories = useCallback(async (f: Filters, off: number) => {
    try {
      setLoading(true);
      if (f.search) {
        const results = await api.memories.search(f.search, limit);
        let filtered = results;
        if (f.category) filtered = filtered.filter((m) => m.category === f.category);
        if (f.sender) filtered = filtered.filter((m) => m.sender === f.sender);
        setMemories(filtered);
        setTotal(filtered.length);
      } else {
        const resp = await api.memories.list({
          category: f.category || undefined,
          sender: f.sender || undefined,
          sort: f.sort,
          limit,
          offset: off,
        });
        setMemories(resp.memories);
        setTotal(resp.total);
      }
    } catch {
      // keep existing data on error
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      fetchMemories(filters, offset);
    }, filters.search ? 300 : 0);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [filters, offset, fetchMemories]);

  const updateFilter = useCallback((key: keyof Filters, value: string) => {
    setOffset(0);
    setFilters((prev) => ({ ...prev, [key]: value }));
  }, []);

  const loadMore = useCallback(() => {
    setOffset((prev) => prev + limit);
  }, []);

  const refetch = useCallback(() => {
    fetchMemories(filters, offset);
  }, [filters, offset, fetchMemories]);

  return { memories, total, loading, filters, updateFilter, loadMore, refetch, offset, limit };
}
```

### `truememory/dashboard/frontend/src/components/GlassCard.tsx`
```
import { motion } from "framer-motion";
import { springDefault } from "../lib/constants";

interface Props {
  children: React.ReactNode;
  className?: string;
  padding?: boolean;
}

export function GlassCard({ children, className = "", padding = true }: Props) {
  return (
    <motion.div
      layout
      transition={springDefault}
      className={`glass-panel ${padding ? "p-5" : ""} ${className}`}
    >
      {children}
    </motion.div>
  );
}
```

### `truememory/dashboard/frontend/src/components/Sparkline.tsx`
```
interface Props {
  data: number[];
  color?: string;
  width?: number;
  height?: number;
}

export function Sparkline({ data, color = "#6C5CE7", width = 100, height = 32 }: Props) {
  if (!data.length) return null;
  const max = Math.max(...data, 1);
  const points = data
    .map((v, i) => `${(i / Math.max(data.length - 1, 1)) * width},${height - (v / max) * (height - 2) - 1}`)
    .join(" ");
  const areaPoints = `0,${height} ${points} ${width},${height}`;
  const id = `sparkline-${color.replace("#", "")}`;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ height }}>
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.3} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      <polygon points={areaPoints} fill={`url(#${id})`} />
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
```

### `truememory/dashboard/frontend/src/components/StatCard.tsx`
```
import { GlassCard } from "./GlassCard";
import { Sparkline } from "./Sparkline";
import { formatNumber } from "../lib/formatters";

interface Props {
  value: number | string;
  label: string;
  sparklineData?: number[];
  color?: string;
}

export function StatCard({ value, label, sparklineData, color = "#6C5CE7" }: Props) {
  return (
    <GlassCard className="flex flex-col gap-2 min-w-0">
      <span className="text-[28px] font-bold leading-none tracking-tight">
        {typeof value === "number" ? formatNumber(value) : value}
      </span>
      <span className="text-xs text-text-secondary">{label}</span>
      {sparklineData && sparklineData.length > 0 && (
        <Sparkline data={sparklineData} color={color} height={28} />
      )}
    </GlassCard>
  );
}
```

### `truememory/dashboard/frontend/src/components/CategoryBadge.tsx`
```
import { getCategoryColor } from "../lib/constants";

interface Props {
  category: string;
}

export function CategoryBadge({ category }: Props) {
  if (!category) return null;
  const { bg, text } = getCategoryColor(category);
  return (
    <span
      className="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-medium whitespace-nowrap"
      style={{ backgroundColor: bg, color: text }}
    >
      {category}
    </span>
  );
}
```

### `truememory/dashboard/frontend/src/components/SearchBar.tsx`
```
interface Props {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

export function SearchBar({ value, onChange, placeholder = "Search memories…" }: Props) {
  return (
    <div className="relative">
      <svg
        className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-tertiary"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <circle cx="11" cy="11" r="8" strokeWidth="2" />
        <path d="m21 21-4.35-4.35" strokeWidth="2" strokeLinecap="round" />
      </svg>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full pl-10 pr-4 py-2.5 bg-bg-elevated border border-[rgba(255,255,255,0.06)] rounded-[10px] text-sm text-text-primary placeholder:text-text-tertiary outline-none focus:border-accent/40 transition-colors"
      />
      {value && (
        <button
          onClick={() => onChange("")}
          className="absolute right-3 top-1/2 -translate-y-1/2 text-text-tertiary hover:text-text-secondary text-xs"
        >
          ✕
        </button>
      )}
    </div>
  );
}
```

### `truememory/dashboard/frontend/src/components/MemoryCard.tsx`
```
import React from "react";
import { CategoryBadge } from "./CategoryBadge";
import { relativeTime, truncate } from "../lib/formatters";
import type { Memory } from "../lib/types";

interface Props {
  memory: Memory;
  selected: boolean;
  onClick: () => void;
}

export const MemoryCard = React.memo(function MemoryCard({ memory, selected, onClick }: Props) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-4 py-3 border-l-2 transition-colors ${
        selected
          ? "bg-accent-muted border-l-accent"
          : "border-l-transparent hover:bg-bg-elevated"
      }`}
    >
      <div className="flex items-center gap-2 mb-1.5">
        <CategoryBadge category={memory.category} />
        {memory.sender && (
          <span className="text-[11px] text-text-tertiary">{memory.sender}</span>
        )}
        <span className="text-[11px] text-text-tertiary ml-auto whitespace-nowrap">
          {relativeTime(memory.timestamp)}
        </span>
        {memory.score != null && (
          <span className="text-[10px] text-accent font-mono">{memory.score.toFixed(2)}</span>
        )}
      </div>
      <p className="text-[13px] text-text-primary/90 leading-snug line-clamp-2">
        {truncate(memory.content, 200)}
      </p>
    </button>
  );
});
```

### `truememory/dashboard/frontend/src/components/InspectorPanel.tsx`
```
import { useState } from "react";
import { motion } from "framer-motion";
import { springDefault } from "../lib/constants";
import { CategoryBadge } from "./CategoryBadge";
import { ConfirmSheet } from "./ConfirmSheet";
import { relativeTime } from "../lib/formatters";
import { api } from "../lib/api";
import type { Memory } from "../lib/types";

interface Props {
  memory: Memory;
  onClose: () => void;
  onDeleted: () => void;
  onUpdated: (m: Memory) => void;
}

export function InspectorPanel({ memory, onClose, onDeleted, onUpdated }: Props) {
  const [editing, setEditing] = useState(false);
  const [editContent, setEditContent] = useState(memory.content);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    try {
      const updated = await api.memories.update(memory.id, editContent);
      onUpdated(updated);
      setEditing(false);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    await api.memories.delete(memory.id);
    setConfirmDelete(false);
    onDeleted();
  };

  return (
    <>
      <motion.div
        initial={{ x: 380, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        exit={{ x: 380, opacity: 0 }}
        transition={springDefault}
        className="w-[380px] flex-shrink-0 border-l border-[rgba(255,255,255,0.06)] bg-bg-elevated h-full overflow-y-auto"
      >
        <div className="p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-text-secondary">Memory #{memory.id}</h3>
            <button
              onClick={onClose}
              className="text-text-tertiary hover:text-text-secondary text-lg leading-none"
            >
              ✕
            </button>
          </div>

          {editing ? (
            <div className="mb-4">
              <textarea
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                className="w-full h-40 bg-bg-base border border-[rgba(255,255,255,0.08)] rounded-lg p-3 text-[13px] font-mono text-text-primary resize-none outline-none focus:border-accent/40"
              />
              <div className="flex gap-2 mt-2">
                <button
                  onClick={handleSave}
                  disabled={saving}
                  className="px-3 py-1.5 text-xs rounded-lg bg-accent/20 text-accent hover:bg-accent/30 transition-colors font-medium"
                >
                  {saving ? "Saving…" : "Save"}
                </button>
                <button
                  onClick={() => { setEditing(false); setEditContent(memory.content); }}
                  className="px-3 py-1.5 text-xs rounded-lg bg-bg-grouped text-text-secondary hover:bg-bg-tertiary transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <div className="mb-4 p-3 bg-bg-base rounded-lg">
              <p className="text-[13px] font-mono text-text-primary/90 leading-relaxed whitespace-pre-wrap break-words">
                {memory.content}
              </p>
            </div>
          )}

          <div className="space-y-3 mb-6">
            <MetaRow label="Category">
              <CategoryBadge category={memory.category} />
            </MetaRow>
            <MetaRow label="Sender">
              <span className="text-sm">{memory.sender || "—"}</span>
            </MetaRow>
            <MetaRow label="Timestamp">
              <span className="text-sm">{memory.timestamp ? `${relativeTime(memory.timestamp)}` : "—"}</span>
            </MetaRow>
            {memory.timestamp && (
              <MetaRow label="Date">
                <span className="text-xs text-text-tertiary font-mono">
                  {new Date(memory.timestamp).toLocaleString()}
                </span>
              </MetaRow>
            )}
            {memory.score != null && (
              <MetaRow label="Score">
                <span className="text-sm font-mono text-accent">{memory.score.toFixed(4)}</span>
              </MetaRow>
            )}
            {memory.source && (
              <MetaRow label="Source">
                <span className="text-xs font-mono text-text-tertiary">{memory.source}</span>
              </MetaRow>
            )}
          </div>

          <div className="flex gap-2">
            {!editing && (
              <button
                onClick={() => { setEditing(true); setEditContent(memory.content); }}
                className="px-3 py-1.5 text-xs rounded-lg bg-bg-grouped text-text-primary hover:bg-bg-tertiary transition-colors"
              >
                Edit
              </button>
            )}
            <button
              onClick={() => setConfirmDelete(true)}
              className="px-3 py-1.5 text-xs rounded-lg bg-error/10 text-error hover:bg-error/20 transition-colors"
            >
              Delete
            </button>
          </div>
        </div>
      </motion.div>

      <ConfirmSheet
        open={confirmDelete}
        title="Delete Memory"
        message="This action cannot be undone. This memory will be permanently removed."
        confirmLabel="Delete"
        onConfirm={handleDelete}
        onCancel={() => setConfirmDelete(false)}
      />
    </>
  );
}

function MetaRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between">
      <span className="text-xs text-text-tertiary">{label}</span>
      <div className="text-right">{children}</div>
    </div>
  );
}
```

### `truememory/dashboard/frontend/src/components/ConfirmSheet.tsx`
```
import { motion, AnimatePresence } from "framer-motion";
import { springGentle } from "../lib/constants";

interface Props {
  open: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmSheet({ open, title, message, confirmLabel = "Delete", onConfirm, onCancel }: Props) {
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
          onClick={onCancel}
        >
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            transition={springGentle}
            className="glass-panel p-6 max-w-sm w-full mx-4"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-lg font-semibold mb-2">{title}</h3>
            <p className="text-sm text-text-secondary mb-6">{message}</p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={onCancel}
                className="px-4 py-2 text-sm rounded-lg bg-bg-grouped text-text-primary hover:bg-bg-tertiary transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={onConfirm}
                className="px-4 py-2 text-sm rounded-lg bg-error/20 text-error hover:bg-error/30 transition-colors font-medium"
              >
                {confirmLabel}
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
```

### `truememory/dashboard/frontend/src/components/SkeletonLoader.tsx`
```
export function SkeletonCard() {
  return (
    <div className="glass-panel p-4 animate-pulse">
      <div className="flex items-center gap-2 mb-3">
        <div className="h-5 w-16 rounded-full bg-bg-tertiary" />
        <div className="h-3 w-12 rounded bg-bg-tertiary ml-auto" />
      </div>
      <div className="space-y-2">
        <div className="h-3 w-full rounded bg-bg-tertiary" />
        <div className="h-3 w-3/4 rounded bg-bg-tertiary" />
      </div>
    </div>
  );
}

export function SkeletonStat() {
  return (
    <div className="glass-panel p-5 animate-pulse">
      <div className="h-8 w-20 rounded bg-bg-tertiary mb-2" />
      <div className="h-3 w-24 rounded bg-bg-tertiary mb-3" />
      <div className="h-7 w-full rounded bg-bg-tertiary" />
    </div>
  );
}
```

### `truememory/dashboard/frontend/src/components/Sidebar.tsx`
```
import type { ViewId, HealthResponse } from "../lib/types";

interface Props {
  currentView: ViewId;
  onNavigate: (view: ViewId) => void;
  health: HealthResponse | null;
}

const navItems: { id: ViewId; label: string; icon: React.ReactNode }[] = [
  {
    id: "overview",
    label: "Overview",
    icon: (
      <svg className="w-[18px] h-[18px]" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.8}>
        <rect x="3" y="3" width="7" height="7" rx="1.5" />
        <rect x="14" y="3" width="7" height="7" rx="1.5" />
        <rect x="3" y="14" width="7" height="7" rx="1.5" />
        <rect x="14" y="14" width="7" height="7" rx="1.5" />
      </svg>
    ),
  },
  {
    id: "explorer",
    label: "Explorer",
    icon: (
      <svg className="w-[18px] h-[18px]" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.8}>
        <circle cx="11" cy="11" r="8" />
        <path d="m21 21-4.35-4.35" strokeLinecap="round" />
      </svg>
    ),
  },
];

const proItems: { id: ViewId; label: string; icon: React.ReactNode }[] = [
  {
    id: "sessions",
    label: "Sessions",
    icon: (
      <svg className="w-[18px] h-[18px]" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.8}>
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
      </svg>
    ),
  },
  {
    id: "people",
    label: "People",
    icon: (
      <svg className="w-[18px] h-[18px]" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.8}>
        <circle cx="9" cy="7" r="4" />
        <path d="M3 21v-2a4 4 0 0 1 4-4h4a4 4 0 0 1 4 4v2" />
        <circle cx="17" cy="7" r="3" />
        <path d="M21 21v-2a3 3 0 0 0-2-2.83" />
      </svg>
    ),
  },
  {
    id: "facts",
    label: "Facts",
    icon: (
      <svg className="w-[18px] h-[18px]" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.8}>
        <path d="M12 2L2 7l10 5 10-5-10-5z" />
        <path d="M2 17l10 5 10-5" />
        <path d="M2 12l10 5 10-5" />
      </svg>
    ),
  },
  {
    id: "analytics",
    label: "Analytics",
    icon: (
      <svg className="w-[18px] h-[18px]" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.8}>
        <path d="M3 3v18h18" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M7 16l4-8 4 4 4-8" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
];

const settingsItem: { id: ViewId; label: string; icon: React.ReactNode } = {
  id: "settings",
  label: "Settings",
  icon: (
    <svg className="w-[18px] h-[18px]" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.8}>
      <circle cx="12" cy="12" r="3" />
      <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" strokeLinecap="round" />
    </svg>
  ),
};

export function Sidebar({ currentView, onNavigate, health }: Props) {
  const tierColors: Record<string, string> = {
    edge: "#30D158",
    base: "#64D2FF",
    pro: "#BF5AF2",
  };
  const tier = health?.tier || "edge";

  const renderNavButton = (item: { id: ViewId; label: string; icon: React.ReactNode }) => (
    <button
      key={item.id}
      onClick={() => onNavigate(item.id)}
      className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-[13px] transition-colors ${
        currentView === item.id
          ? "bg-accent-muted text-text-primary"
          : "text-text-secondary hover:bg-bg-elevated hover:text-text-primary"
      }`}
    >
      <span className={currentView === item.id ? "text-accent" : ""}>{item.icon}</span>
      {item.label}
    </button>
  );

  return (
    <div className="w-[220px] flex-shrink-0 glass-sidebar flex flex-col h-full select-none">
      <div className="px-5 pt-6 pb-4">
        <h1 className="text-[15px] font-semibold tracking-tight">TrueMemory</h1>
      </div>

      <nav className="flex-1 px-3">
        <div className="space-y-0.5">
          {navItems.map(renderNavButton)}
        </div>

        <div className="mt-6 pt-4 border-t border-[rgba(255,255,255,0.04)]">
          <p className="px-3 text-[10px] font-medium text-text-tertiary uppercase tracking-wider mb-2">
            Pro
          </p>
          <div className="space-y-0.5">
            {proItems.map(renderNavButton)}
          </div>
        </div>

        <div className="mt-6 pt-4 border-t border-[rgba(255,255,255,0.04)]">
          <div className="space-y-0.5">
            {renderNavButton(settingsItem)}
          </div>
        </div>
      </nav>

      <div className="px-4 py-4 border-t border-[rgba(255,255,255,0.04)]">
        <div className="flex items-center gap-2 mb-1">
          <span
            className="text-[10px] font-bold uppercase px-1.5 py-0.5 rounded"
            style={{
              backgroundColor: `${tierColors[tier] || "#8E8E93"}20`,
              color: tierColors[tier] || "#8E8E93",
            }}
          >
            {tier}
          </span>
          <span className="text-[11px] text-text-tertiary">
            v{health?.version || "…"}
          </span>
        </div>
        <p className="text-[11px] text-text-tertiary">
          {health ? `${health.memory_count.toLocaleString()} memories` : "Loading…"}
        </p>
      </div>
    </div>
  );
}
```

### `truememory/dashboard/frontend/src/views/Overview.tsx`
```
import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { StatCard } from "../components/StatCard";
import { GlassCard } from "../components/GlassCard";
import { CategoryBadge } from "../components/CategoryBadge";
import { SkeletonStat } from "../components/SkeletonLoader";
import { api } from "../lib/api";
import { relativeTime, truncate, formatBytes } from "../lib/formatters";
import type { DashboardStats, Memory, HealthResponse } from "../lib/types";
import { springDefault } from "../lib/constants";

interface Props {
  health: HealthResponse | null;
  onNavigateExplorer: () => void;
}

export function Overview({ health, onNavigateExplorer }: Props) {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [recent, setRecent] = useState<Memory[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.memories.stats(),
      api.memories.list({ sort: "newest", limit: 20 }),
    ]).then(([s, r]) => {
      setStats(s);
      setRecent(r.memories);
      setLoading(false);
    });
  }, []);

  return (
    <div className="h-full overflow-y-auto p-6 space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={springDefault}
      >
        <h2 className="text-xl font-semibold mb-5">Overview</h2>
      </motion.div>

      <div className="grid grid-cols-4 gap-4">
        {loading || !stats ? (
          <>
            <SkeletonStat />
            <SkeletonStat />
            <SkeletonStat />
            <SkeletonStat />
          </>
        ) : (
          <>
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0, ...springDefault }}>
              <StatCard value={stats.total} label="Total Memories" sparklineData={stats.sparkline} color="#6C5CE7" />
            </motion.div>
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05, ...springDefault }}>
              <StatCard value={stats.this_week} label="This Week" sparklineData={stats.sparkline.slice(-7)} color="#64D2FF" />
            </motion.div>
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1, ...springDefault }}>
              <StatCard value={stats.entities} label="Entities Tracked" color="#30D158" />
            </motion.div>
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15, ...springDefault }}>
              <StatCard
                value={stats.gate_pass_rate != null ? `${Math.round(stats.gate_pass_rate * 100)}%` : "—"}
                label="Gate Pass Rate"
                color="#FFD60A"
              />
            </motion.div>
          </>
        )}
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2">
          <GlassCard padding={false}>
            <div className="flex items-center justify-between px-5 pt-4 pb-3">
              <h3 className="text-sm font-semibold">Recent Memories</h3>
              <button
                onClick={onNavigateExplorer}
                className="text-xs text-accent hover:text-accent-hover transition-colors"
              >
                View all →
              </button>
            </div>
            <div className="max-h-[400px] overflow-y-auto">
              {recent.map((m) => (
                <div
                  key={m.id}
                  className="px-5 py-3 border-t border-[rgba(255,255,255,0.04)] hover:bg-bg-elevated/50 transition-colors cursor-pointer"
                  onClick={onNavigateExplorer}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <CategoryBadge category={m.category} />
                    <span className="text-[11px] text-text-tertiary ml-auto">
                      {relativeTime(m.timestamp)}
                    </span>
                  </div>
                  <p className="text-[13px] text-text-primary/80 leading-snug">
                    {truncate(m.content, 120)}
                  </p>
                </div>
              ))}
              {recent.length === 0 && !loading && (
                <p className="px-5 py-8 text-sm text-text-tertiary text-center">
                  No memories yet
                </p>
              )}
            </div>
          </GlassCard>
        </div>

        <GlassCard>
          <h3 className="text-sm font-semibold mb-4">System Health</h3>
          <div className="space-y-3 text-[13px]">
            <HealthRow label="Tier" value={health?.tier?.toUpperCase() || "—"} />
            <HealthRow label="DB Size" value={health ? formatBytes(health.db_size_kb) : "—"} />
            <HealthRow label="Version" value={health?.version || "—"} />
            {health?.capabilities && (
              <>
                <div className="border-t border-[rgba(255,255,255,0.04)] pt-3 mt-3">
                  <p className="text-[11px] text-text-tertiary mb-2">Capabilities</p>
                  <div className="flex flex-wrap gap-1.5">
                    {Object.entries(health.capabilities)
                      .filter(([, v]) => v)
                      .map(([k]) => (
                        <span
                          key={k}
                          className="text-[10px] px-1.5 py-0.5 rounded bg-success/10 text-success"
                        >
                          {k}
                        </span>
                      ))}
                  </div>
                </div>
              </>
            )}
          </div>

          {stats?.categories && (
            <div className="mt-5 pt-4 border-t border-[rgba(255,255,255,0.04)]">
              <p className="text-[11px] text-text-tertiary mb-2">Categories</p>
              <div className="space-y-1.5">
                {Object.entries(stats.categories)
                  .slice(0, 6)
                  .map(([cat, count]) => (
                    <div key={cat} className="flex items-center justify-between">
                      <CategoryBadge category={cat} />
                      <span className="text-[11px] text-text-tertiary font-mono">{count}</span>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </GlassCard>
      </div>
    </div>
  );
}

function HealthRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-text-tertiary">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  );
}
```

### `truememory/dashboard/frontend/src/views/Explorer.tsx`
```
import { useState, useEffect, useRef } from "react";
import { AnimatePresence } from "framer-motion";
import { useVirtualizer } from "@tanstack/react-virtual";
import { SearchBar } from "../components/SearchBar";
import { MemoryCard } from "../components/MemoryCard";
import { InspectorPanel } from "../components/InspectorPanel";
import { SkeletonCard } from "../components/SkeletonLoader";
import { useMemories } from "../hooks/useMemories";
import { api } from "../lib/api";
import { formatNumber } from "../lib/formatters";
import type { Memory } from "../lib/types";

export function Explorer() {
  const { memories, total, loading, filters, updateFilter, refetch } = useMemories();
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [senders, setSenders] = useState<string[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const parentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.memories.senders().then(setSenders);
    api.memories.categories().then(setCategories);
  }, []);

  const selectedMemory = memories.find((m) => m.id === selectedId) || null;

  const virtualizer = useVirtualizer({
    count: memories.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 76,
    overscan: 10,
  });

  const handleUpdated = (updated: Memory) => {
    refetch();
    setSelectedId(updated.id);
  };

  const handleDeleted = () => {
    setSelectedId(null);
    refetch();
  };

  return (
    <div className="h-full flex flex-col">
      <div className="flex-shrink-0 px-5 pt-5 pb-3 space-y-3">
        <div className="flex items-center gap-3">
          <div className="flex-1">
            <SearchBar
              value={filters.search}
              onChange={(v) => updateFilter("search", v)}
            />
          </div>
          <FilterSelect
            value={filters.category}
            onChange={(v) => updateFilter("category", v)}
            options={categories}
            placeholder="Category"
          />
          <FilterSelect
            value={filters.sender}
            onChange={(v) => updateFilter("sender", v)}
            options={senders}
            placeholder="Sender"
          />
          {!filters.search && (
            <FilterSelect
              value={filters.sort}
              onChange={(v) => updateFilter("sort", v)}
              options={["newest", "oldest"]}
              placeholder="Sort"
            />
          )}
        </div>
      </div>

      <div className="flex-1 flex min-h-0">
        <div className="flex-1 flex flex-col min-w-0">
          {loading && memories.length === 0 ? (
            <div className="p-4 space-y-2">
              <SkeletonCard />
              <SkeletonCard />
              <SkeletonCard />
              <SkeletonCard />
              <SkeletonCard />
            </div>
          ) : memories.length === 0 ? (
            <div className="flex-1 flex items-center justify-center">
              <p className="text-text-tertiary text-sm">
                {filters.search ? "No results found" : "No memories"}
              </p>
            </div>
          ) : (
            <div ref={parentRef} className="flex-1 overflow-y-auto">
              <div
                style={{
                  height: `${virtualizer.getTotalSize()}px`,
                  width: "100%",
                  position: "relative",
                }}
              >
                {virtualizer.getVirtualItems().map((virtualItem) => {
                  const memory = memories[virtualItem.index];
                  return (
                    <div
                      key={memory.id}
                      style={{
                        position: "absolute",
                        top: 0,
                        left: 0,
                        width: "100%",
                        height: `${virtualItem.size}px`,
                        transform: `translateY(${virtualItem.start}px)`,
                      }}
                    >
                      <MemoryCard
                        memory={memory}
                        selected={selectedId === memory.id}
                        onClick={() => setSelectedId(selectedId === memory.id ? null : memory.id)}
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <div className="flex-shrink-0 px-5 py-2.5 border-t border-[rgba(255,255,255,0.04)] text-[11px] text-text-tertiary">
            {formatNumber(total)} memories
            {selectedId != null && " · 1 selected"}
          </div>
        </div>

        <AnimatePresence>
          {selectedMemory && (
            <InspectorPanel
              key={selectedMemory.id}
              memory={selectedMemory}
              onClose={() => setSelectedId(null)}
              onDeleted={handleDeleted}
              onUpdated={handleUpdated}
            />
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

function FilterSelect({
  value,
  onChange,
  options,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  options: string[];
  placeholder: string;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="bg-bg-elevated border border-[rgba(255,255,255,0.06)] rounded-lg px-3 py-2.5 text-[13px] text-text-primary outline-none appearance-none cursor-pointer min-w-[100px]"
    >
      <option value="">{placeholder}</option>
      {options.map((opt) => (
        <option key={opt} value={opt}>
          {opt}
        </option>
      ))}
    </select>
  );
}
```

### `truememory/dashboard/frontend/src/views/Sessions.tsx`
```
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { SearchBar } from "../components/SearchBar";
import { GlassCard } from "../components/GlassCard";
import { SkeletonCard } from "../components/SkeletonLoader";
import { api } from "../lib/api";
import { relativeTime, formatNumber } from "../lib/formatters";
import { springDefault } from "../lib/constants";
import type { Session, TranscriptMessage } from "../lib/types";

export function Sessions() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [project, setProject] = useState("");
  const [projects, setProjects] = useState<string[]>([]);
  const [selectedSession, setSelectedSession] = useState<Session | null>(null);
  const [transcript, setTranscript] = useState<TranscriptMessage[]>([]);
  const [loadingTranscript, setLoadingTranscript] = useState(false);
  const [indexing, setIndexing] = useState(false);

  useEffect(() => {
    api.sessions.projects().then(setProjects).catch(() => {});
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      setLoading(true);
      api.sessions.list({ search: search || undefined, project: project || undefined, limit: 100 })
        .then((resp) => { setSessions(resp.sessions); setTotal(resp.total); })
        .finally(() => setLoading(false));
    }, search ? 300 : 0);
    return () => clearTimeout(timer);
  }, [search, project]);

  const handleSelectSession = async (s: Session) => {
    setSelectedSession(s);
    setLoadingTranscript(true);
    try {
      const resp = await api.sessions.transcript(s.session_id);
      setTranscript(resp.messages);
    } catch {
      setTranscript([]);
    } finally {
      setLoadingTranscript(false);
    }
  };

  const handleReindex = async () => {
    setIndexing(true);
    try {
      const result = await api.sessions.reindex();
      setTotal(result.total);
      const resp = await api.sessions.list({ limit: 100 });
      setSessions(resp.sessions);
      setTotal(resp.total);
    } catch (err) {
      console.error("Reindex failed:", err);
    } finally {
      setIndexing(false);
    }
  };

  if (selectedSession) {
    return (
      <TranscriptView
        session={selectedSession}
        messages={transcript}
        loading={loadingTranscript}
        onBack={() => { setSelectedSession(null); setTranscript([]); }}
      />
    );
  }

  return (
    <div className="h-full flex flex-col">
      <div className="flex-shrink-0 px-5 pt-5 pb-3 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold">Sessions</h2>
          <button
            onClick={handleReindex}
            disabled={indexing}
            className="px-3 py-1.5 text-xs rounded-lg bg-accent/15 text-accent hover:bg-accent/25 transition-colors font-medium"
          >
            {indexing ? "Indexing…" : "Reindex"}
          </button>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex-1">
            <SearchBar value={search} onChange={setSearch} placeholder="Search sessions…" />
          </div>
          <select
            value={project}
            onChange={(e) => setProject(e.target.value)}
            className="bg-bg-elevated border border-[rgba(255,255,255,0.06)] rounded-lg px-3 py-2.5 text-[13px] text-text-primary outline-none appearance-none cursor-pointer max-w-[200px]"
          >
            <option value="">All projects</option>
            {projects.map((p) => (
              <option key={p} value={p}>{p.split("/").pop() || p}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-5 pb-5 space-y-2">
        {loading ? (
          Array.from({ length: 5 }).map((_, i) => <SkeletonCard key={i} />)
        ) : sessions.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 text-text-tertiary">
            <p className="text-sm mb-3">No sessions indexed yet</p>
            <button
              onClick={handleReindex}
              className="px-4 py-2 text-sm rounded-lg bg-accent/15 text-accent hover:bg-accent/25"
            >
              Index Sessions
            </button>
          </div>
        ) : (
          sessions.map((s, i) => (
            <motion.div
              key={s.session_id}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: Math.min(i * 0.02, 0.3), ...springDefault }}
            >
              <button
                onClick={() => handleSelectSession(s)}
                className="w-full text-left glass-panel p-4 hover:bg-bg-elevated/80 transition-colors"
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[13px] text-text-primary">
                    {s.started_at ? new Date(s.started_at).toLocaleDateString("en-US", {
                      month: "short", day: "numeric", year: "numeric",
                    }) : "Unknown date"}
                    {s.started_at && (
                      <span className="text-text-tertiary ml-2">
                        {new Date(s.started_at).toLocaleTimeString("en-US", {
                          hour: "numeric", minute: "2-digit",
                        })}
                      </span>
                    )}
                  </span>
                  <span className="text-[11px] text-text-tertiary">
                    {formatNumber(s.message_count)} msgs
                  </span>
                </div>
                <p className="text-xs text-text-tertiary font-mono mb-1.5 truncate">
                  {s.project_dir}
                </p>
                {s.summary && (
                  <p className="text-[13px] text-text-secondary leading-snug line-clamp-2">
                    {s.summary}
                  </p>
                )}
              </button>
            </motion.div>
          ))
        )}
      </div>

      <div className="flex-shrink-0 px-5 py-2.5 border-t border-[rgba(255,255,255,0.04)] text-[11px] text-text-tertiary">
        {formatNumber(total)} sessions indexed
      </div>
    </div>
  );
}

function TranscriptView({
  session,
  messages,
  loading,
  onBack,
}: {
  session: Session;
  messages: TranscriptMessage[];
  loading: boolean;
  onBack: () => void;
}) {
  return (
    <div className="h-full flex flex-col">
      <div className="flex-shrink-0 px-5 pt-5 pb-3 border-b border-[rgba(255,255,255,0.04)]">
        <div className="flex items-center gap-3 mb-2">
          <button onClick={onBack} className="text-accent hover:text-accent-hover text-sm">
            ← Back
          </button>
          <span className="text-[13px] text-text-secondary">
            {session.started_at && new Date(session.started_at).toLocaleDateString("en-US", {
              weekday: "short", month: "short", day: "numeric", year: "numeric",
            })}
          </span>
          <span className="text-[11px] text-text-tertiary ml-auto">
            {formatNumber(session.message_count)} messages
          </span>
        </div>
        <p className="text-xs text-text-tertiary font-mono">{session.project_dir}</p>
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
        {loading ? (
          Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} />)
        ) : messages.length === 0 ? (
          <p className="text-text-tertiary text-sm text-center py-10">No transcript data available</p>
        ) : (
          messages.map((msg, i) => (
            <motion.div
              key={msg.uuid || i}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: Math.min(i * 0.01, 0.5), ...springDefault }}
              className={`${msg.type === "user" ? "ml-0 mr-12" : "ml-6 mr-0"}`}
            >
              <div className="flex items-center gap-2 mb-1">
                <span className={`text-[11px] font-semibold uppercase tracking-wider ${
                  msg.type === "user" ? "text-info" : "text-accent"
                }`}>
                  {msg.type === "user" ? "You" : "Claude"}
                </span>
                {msg.timestamp && (
                  <span className="text-[10px] text-text-tertiary">
                    {new Date(msg.timestamp).toLocaleTimeString("en-US", {
                      hour: "numeric", minute: "2-digit",
                    })}
                  </span>
                )}
              </div>
              <div className={`rounded-xl px-4 py-3 text-[13px] leading-relaxed ${
                msg.type === "user"
                  ? "bg-info/10 text-text-primary"
                  : "bg-bg-elevated text-text-primary/90"
              }`}>
                <p className="whitespace-pre-wrap break-words">{msg.content || "(empty)"}</p>
              </div>
            </motion.div>
          ))
        )}
      </div>
    </div>
  );
}
```

### `truememory/dashboard/frontend/src/views/People.tsx`
```
import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import * as d3 from "d3";
import { GlassCard } from "../components/GlassCard";
import { CategoryBadge } from "../components/CategoryBadge";
import { SkeletonCard } from "../components/SkeletonLoader";
import { api } from "../lib/api";
import { springDefault } from "../lib/constants";
import type { Entity, EntityGraph } from "../lib/types";

export function People() {
  const [entities, setEntities] = useState<Entity[]>([]);
  const [graph, setGraph] = useState<EntityGraph | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [profile, setProfile] = useState<Record<string, unknown> | null>(null);
  const [recentMemories, setRecentMemories] = useState<{ id: number; content: string; timestamp: string; category: string }[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const e = await api.entities.list();
        setEntities(e);
      } catch (err) {
        console.error("Failed to load entities:", err);
      }
      try {
        const g = await api.entities.graph();
        setGraph(g);
      } catch (err) {
        console.error("Failed to load graph:", err);
      }
      setLoading(false);
    };
    load();
  }, []);

  const handleSelect = async (name: string) => {
    setSelected(name);
    try {
      const data = await api.entities.get(name);
      setProfile(data.profile);
      setRecentMemories(data.recent_memories as typeof recentMemories);
    } catch {
      setProfile(null);
      setRecentMemories([]);
    }
  };

  if (loading) {
    return (
      <div className="h-full p-6 space-y-4">
        <h2 className="text-xl font-semibold">People</h2>
        <SkeletonCard />
        <SkeletonCard />
      </div>
    );
  }

  if (entities.length === 0) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center text-text-tertiary">
          <p className="text-lg mb-2">No entity profiles yet</p>
          <p className="text-sm">Entity profiles are built as TrueMemory processes conversations.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex">
      <div className="flex-1 flex flex-col min-w-0">
        <div className="flex-shrink-0 px-5 pt-5 pb-3">
          <h2 className="text-xl font-semibold">People</h2>
        </div>

        <div className="flex-1 min-h-0 relative">
          {graph && <ForceGraph graph={graph} selected={selected} onSelect={handleSelect} />}
        </div>

        <div className="flex-shrink-0 px-5 py-2.5 border-t border-[rgba(255,255,255,0.04)] text-[11px] text-text-tertiary">
          {entities.length} entities · {graph?.edges.length || 0} relationships
        </div>
      </div>

      <AnimatePresence>
        {selected && profile && (
          <motion.div
            initial={{ x: 380, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 380, opacity: 0 }}
            transition={springDefault}
            className="w-[380px] flex-shrink-0 border-l border-[rgba(255,255,255,0.06)] bg-bg-elevated h-full overflow-y-auto"
          >
            <div className="p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">{selected}</h3>
                <button onClick={() => setSelected(null)} className="text-text-tertiary hover:text-text-secondary text-lg">✕</button>
              </div>

              <ProfileSection profile={profile} />

              {recentMemories.length > 0 && (
                <div className="mt-5 pt-4 border-t border-[rgba(255,255,255,0.04)]">
                  <h4 className="text-xs text-text-tertiary mb-3">Recent Memories</h4>
                  <div className="space-y-2">
                    {recentMemories.map((m) => (
                      <div key={m.id} className="p-2.5 bg-bg-base rounded-lg">
                        <div className="flex items-center gap-2 mb-1">
                          <CategoryBadge category={m.category} />
                        </div>
                        <p className="text-[12px] text-text-primary/80 line-clamp-2">{m.content}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function ProfileSection({ profile }: { profile: Record<string, unknown> }) {
  const traits = profile.traits as string[] | Record<string, unknown> | undefined;
  const topics = profile.topics as string[] | undefined;
  const commStyle = profile.communication_style as Record<string, unknown> | undefined;
  const msgCount = profile.message_count as number | undefined;

  const traitList = Array.isArray(traits) ? traits : traits && typeof traits === "object" ? Object.keys(traits) : [];
  const topicList = Array.isArray(topics) ? topics : [];

  return (
    <div className="space-y-4">
      {msgCount != null && (
        <div className="text-sm text-text-secondary">{msgCount} memories</div>
      )}

      {traitList.length > 0 && (
        <div>
          <h4 className="text-xs text-text-tertiary mb-2">Traits</h4>
          <div className="flex flex-wrap gap-1.5">
            {traitList.map((t, i) => (
              <span key={i} className="text-[11px] px-2 py-0.5 rounded-full bg-accent/10 text-accent">
                {String(t)}
              </span>
            ))}
          </div>
        </div>
      )}

      {topicList.length > 0 && (
        <div>
          <h4 className="text-xs text-text-tertiary mb-2">Topics</h4>
          <div className="flex flex-wrap gap-1.5">
            {topicList.map((t, i) => (
              <span key={i} className="text-[11px] px-2 py-0.5 rounded-full bg-info/10 text-info">
                {t}
              </span>
            ))}
          </div>
        </div>
      )}

      {commStyle && (
        <div>
          <h4 className="text-xs text-text-tertiary mb-2">Communication Style</h4>
          <div className="space-y-1.5 text-[12px]">
            {Object.entries(commStyle).map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span className="text-text-tertiary">{k.replace(/_/g, " ")}</span>
                <span className="text-text-primary">{String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ForceGraph({
  graph,
  selected,
  onSelect,
}: {
  graph: EntityGraph;
  selected: string | null;
  onSelect: (name: string) => void;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const selectedRef = useRef(selected);
  const onSelectRef = useRef(onSelect);
  selectedRef.current = selected;
  onSelectRef.current = onSelect;
  const circlesRef = useRef<d3.Selection<SVGCircleElement, any, SVGGElement, unknown> | null>(null);

  useEffect(() => {
    if (circlesRef.current) {
      circlesRef.current
        .attr("fill", (d: any) => d.id === selected ? "#6C5CE7" : "rgba(108, 92, 231, 0.25)")
        .attr("stroke", (d: any) => d.id === selected ? "#7C6CF7" : "rgba(255,255,255,0.12)")
        .attr("stroke-width", (d: any) => d.id === selected ? 2.5 : 1);
    }
  }, [selected]);

  useEffect(() => {
    if (!svgRef.current || !graph.nodes.length) return;

    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;

    svg.selectAll("*").remove();

    const g = svg.append("g");

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 3])
      .on("zoom", (event) => g.attr("transform", event.transform));
    svg.call(zoom);

    const nodeMap = new Map(graph.nodes.map((n) => [n.id, n]));
    const validEdges = graph.edges.filter(
      (e) => nodeMap.has(e.source as string) && nodeMap.has(e.target as string)
    );

    const simulation = d3.forceSimulation(graph.nodes as d3.SimulationNodeDatum[])
      .force("link", d3.forceLink(validEdges).id((d: any) => d.id).distance(140).strength(0.4))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius((d: any) => d.radius + 8));

    g.append("g")
      .selectAll("line")
      .data(validEdges)
      .join("line")
      .attr("stroke", "rgba(255,255,255,0.06)")
      .attr("stroke-width", (d: any) => Math.max(0.5, d.strength * 2));

    const node = g.append("g")
      .selectAll("g")
      .data(graph.nodes)
      .join("g")
      .style("cursor", "pointer")
      .on("click", (_, d: any) => onSelectRef.current(d.id))
      .call(
        d3.drag<SVGGElement, any>()
          .on("start", (event, d) => { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
          .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
          .on("end", (event, d) => { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }) as any
      );

    circlesRef.current = node.append("circle")
      .attr("r", (d: any) => d.radius)
      .attr("fill", "rgba(108, 92, 231, 0.25)")
      .attr("stroke", "rgba(255,255,255,0.12)")
      .attr("stroke-width", 1);

    node.append("text")
      .text((d: any) => d.id)
      .attr("text-anchor", "middle")
      .attr("dy", (d: any) => d.radius + 16)
      .attr("fill", "#EBEBF5")
      .attr("font-size", "12px")
      .attr("font-weight", "500")
      .attr("font-family", "-apple-system, BlinkMacSystemFont, system-ui");

    simulation.on("tick", () => {
      g.selectAll("line")
        .attr("x1", (d: any) => d.source.x)
        .attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x)
        .attr("y2", (d: any) => d.target.y);
      node.attr("transform", (d: any) => `translate(${d.x},${d.y})`);
    });

    return () => { simulation.stop(); };
  }, [graph]);

  return (
    <svg ref={svgRef} className="w-full h-full" style={{ background: "transparent" }} />
  );
}
```

### `truememory/dashboard/frontend/src/views/Facts.tsx`
```
import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { SearchBar } from "../components/SearchBar";
import { GlassCard } from "../components/GlassCard";
import { SkeletonCard } from "../components/SkeletonLoader";
import { api } from "../lib/api";
import { relativeTime } from "../lib/formatters";
import { springDefault } from "../lib/constants";
import type { Fact } from "../lib/types";

export function Facts() {
  const [facts, setFacts] = useState<Fact[]>([]);
  const [subjects, setSubjects] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [subjectFilter, setSubjectFilter] = useState("");
  const [showSuperseded, setShowSuperseded] = useState(false);
  const [contradictions, setContradictions] = useState<
    { subject: string; fact_a: Record<string, unknown>; fact_b: Record<string, unknown> }[]
  >([]);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      api.facts.list({ subject: subjectFilter || undefined, show_superseded: showSuperseded }),
      api.facts.contradictions(),
    ])
      .then(([resp, contras]) => {
        setFacts(resp.facts);
        setSubjects(resp.subjects);
        setContradictions(contras);
      })
      .finally(() => setLoading(false));
  }, [subjectFilter, showSuperseded]);

  const filteredFacts = search
    ? facts.filter((f) => f.fact.toLowerCase().includes(search.toLowerCase()) || f.subject.toLowerCase().includes(search.toLowerCase()))
    : facts;

  const grouped = new Map<string, Fact[]>();
  for (const f of filteredFacts) {
    const list = grouped.get(f.subject) || [];
    list.push(f);
    grouped.set(f.subject, list);
  }

  if (loading) {
    return (
      <div className="h-full p-6 space-y-4">
        <h2 className="text-xl font-semibold">Facts & Contradictions</h2>
        <SkeletonCard />
        <SkeletonCard />
        <SkeletonCard />
      </div>
    );
  }

  if (facts.length === 0) {
    return (
      <div className="h-full flex flex-col">
        <div className="flex-shrink-0 px-5 pt-5 pb-3">
          <h2 className="text-xl font-semibold">Facts & Contradictions</h2>
        </div>
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center text-text-tertiary max-w-md">
            <p className="text-lg mb-3">No facts tracked yet</p>
            <p className="text-sm leading-relaxed">
              The fact timeline tracks evolving truths — like when someone moves cities or
              changes jobs. As TrueMemory's L5 consolidation processes more conversations,
              facts and contradictions will appear here.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <div className="flex-shrink-0 px-5 pt-5 pb-3 space-y-3">
        <h2 className="text-xl font-semibold">Facts & Contradictions</h2>
        <div className="flex items-center gap-3">
          <div className="flex-1">
            <SearchBar value={search} onChange={setSearch} placeholder="Search facts…" />
          </div>
          <select
            value={subjectFilter}
            onChange={(e) => setSubjectFilter(e.target.value)}
            className="bg-bg-elevated border border-[rgba(255,255,255,0.06)] rounded-lg px-3 py-2.5 text-[13px] text-text-primary outline-none appearance-none cursor-pointer"
          >
            <option value="">All subjects</option>
            {subjects.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          <label className="flex items-center gap-2 text-[12px] text-text-secondary cursor-pointer">
            <input
              type="checkbox"
              checked={showSuperseded}
              onChange={(e) => setShowSuperseded(e.target.checked)}
              className="accent-accent"
            />
            Show superseded
          </label>
        </div>
      </div>

      {contradictions.length > 0 && (
        <div className="px-5 pb-3">
          <GlassCard className="border-warning/20">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-warning text-sm">⚠</span>
              <h3 className="text-sm font-semibold text-warning">Contradictions Detected</h3>
            </div>
            <div className="space-y-2">
              {contradictions.map((c, i) => (
                <div key={i} className="text-[12px] text-text-secondary">
                  <span className="text-text-tertiary">{c.subject}:</span>{" "}
                  "{String(c.fact_a.fact)}" vs "{String(c.fact_b.fact)}"
                </div>
              ))}
            </div>
          </GlassCard>
        </div>
      )}

      <div className="flex-1 overflow-y-auto px-5 pb-5 space-y-5">
        {[...grouped.entries()].map(([subject, factList], gi) => (
          <motion.div
            key={subject}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: Math.min(gi * 0.05, 0.3), ...springDefault }}
          >
            <h3 className="text-sm font-semibold text-text-primary mb-2">{subject}</h3>
            <div className="space-y-1.5 ml-2">
              {factList.map((f) => (
                <div
                  key={f.id}
                  className={`flex items-start gap-2.5 ${f.is_current ? "" : "opacity-40"}`}
                >
                  <span className={`mt-1.5 w-2 h-2 rounded-full flex-shrink-0 ${
                    f.is_current ? "bg-accent" : "bg-text-tertiary"
                  }`} />
                  <div className="flex-1 min-w-0">
                    <p className={`text-[13px] ${f.is_current ? "text-text-primary" : "text-text-tertiary line-through"}`}>
                      {f.fact}
                    </p>
                    <div className="flex items-center gap-3 mt-0.5">
                      {f.timestamp && (
                        <span className="text-[10px] text-text-tertiary">{relativeTime(f.timestamp)}</span>
                      )}
                      <span className={`text-[10px] ${f.is_current ? "text-success" : "text-text-tertiary"}`}>
                        {f.is_current ? "current" : "superseded"}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
```

### `truememory/dashboard/frontend/src/views/Analytics.tsx`
```
import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { GlassCard } from "../components/GlassCard";
import { Sparkline } from "../components/Sparkline";
import { SkeletonStat } from "../components/SkeletonLoader";
import { api } from "../lib/api";
import { formatNumber } from "../lib/formatters";
import { springDefault, getCategoryColor } from "../lib/constants";
import type { GrowthPoint, CategoryCount } from "../lib/types";

export function Analytics() {
  const [growth, setGrowth] = useState<GrowthPoint[]>([]);
  const [categories, setCategories] = useState<CategoryCount[]>([]);
  const [topEntities, setTopEntities] = useState<{ entity: string; message_count: number }[]>([]);
  const [ingest, setIngest] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try { setGrowth(await api.analytics.growth()); } catch {}
      try { setCategories(await api.analytics.categories()); } catch {}
      try { setTopEntities(await api.analytics.entities()); } catch {}
      try { setIngest(await api.analytics.ingest()); } catch {}
      setLoading(false);
    };
    load();
  }, []);

  if (loading) {
    return (
      <div className="h-full p-6 space-y-5">
        <h2 className="text-xl font-semibold">Analytics</h2>
        <div className="grid grid-cols-3 gap-4"><SkeletonStat /><SkeletonStat /><SkeletonStat /></div>
      </div>
    );
  }

  const total = growth.length > 0 ? growth[growth.length - 1].cumulative : 0;
  const last7 = ingest?.["7d"] as number ?? 0;
  const last30 = ingest?.["30d"] as number ?? 0;
  const dailyRate = (ingest?.daily_rate as { date: string; count: number }[]) || [];
  const realCategories = categories.filter((c) => c.category !== "(uncategorized)");
  const realEntities = topEntities.filter((e) => !e.entity.startsWith("__test") && e.entity !== "test" && e.entity !== "test_user");

  return (
    <div className="h-full overflow-y-auto p-6 space-y-6">
      <motion.h2
        className="text-xl font-semibold"
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={springDefault}
      >
        Analytics
      </motion.h2>

      {/* Key metrics */}
      <div className="grid grid-cols-3 gap-4">
        <Metric value={formatNumber(total)} label="Total Memories" delay={0} />
        <Metric value={formatNumber(last30)} label="Last 30 Days" delay={0.04} />
        <Metric value={formatNumber(last7)} label="Last 7 Days" delay={0.08} />
      </div>

      {/* Growth + Daily Rate */}
      <div className="grid grid-cols-2 gap-4">
        <FadeIn delay={0.12}>
          <GlassCard>
            <h3 className="text-sm font-semibold mb-1">Memory Growth</h3>
            <p className="text-[11px] text-text-tertiary mb-3">Cumulative memories over time</p>
            <div className="h-36">
              <GrowthChart data={growth} />
            </div>
          </GlassCard>
        </FadeIn>
        <FadeIn delay={0.16}>
          <GlassCard>
            <h3 className="text-sm font-semibold mb-1">Daily Ingest</h3>
            <p className="text-[11px] text-text-tertiary mb-3">Memories created per day (30 days)</p>
            <div className="h-36">
              <Sparkline data={dailyRate.map((d) => d.count)} color="#64D2FF" height={144} />
            </div>
          </GlassCard>
        </FadeIn>
      </div>

      {/* Categories + Entities */}
      <div className="grid grid-cols-2 gap-4">
        <FadeIn delay={0.2}>
          <GlassCard>
            <h3 className="text-sm font-semibold mb-4">Category Distribution</h3>
            <div className="space-y-2.5">
              {realCategories.slice(0, 8).map((c) => {
                const maxCount = realCategories[0]?.count || 1;
                const pct = (c.count / maxCount) * 100;
                const { text: color } = getCategoryColor(c.category);
                return (
                  <div key={c.category} className="group">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[12px] text-text-secondary">{c.category}</span>
                      <span className="text-[11px] text-text-tertiary font-mono">{formatNumber(c.count)}</span>
                    </div>
                    <div className="h-1.5 bg-bg-base rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${pct}%` }}
                        transition={{ delay: 0.3, duration: 0.5, ease: "easeOut" }}
                        className="h-full rounded-full"
                        style={{ backgroundColor: color }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </GlassCard>
        </FadeIn>

        <FadeIn delay={0.24}>
          <GlassCard>
            <h3 className="text-sm font-semibold mb-4">Top Entities</h3>
            {realEntities.length === 0 ? (
              <p className="text-sm text-text-tertiary">No entities tracked</p>
            ) : (
              <div className="space-y-2.5">
                {realEntities.slice(0, 8).map((e) => {
                  const maxCount = realEntities[0]?.message_count || 1;
                  const pct = (e.message_count / maxCount) * 100;
                  return (
                    <div key={e.entity} className="group">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-[12px] text-text-secondary">{e.entity}</span>
                        <span className="text-[11px] text-text-tertiary font-mono">{formatNumber(e.message_count)}</span>
                      </div>
                      <div className="h-1.5 bg-bg-base rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${pct}%` }}
                          transition={{ delay: 0.35, duration: 0.5, ease: "easeOut" }}
                          className="h-full rounded-full bg-accent/60"
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </GlassCard>
        </FadeIn>
      </div>
    </div>
  );
}

function Metric({ value, label, delay }: { value: string; label: string; delay: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, ...springDefault }}
    >
      <GlassCard className="text-center">
        <span className="text-[32px] font-bold tracking-tight block">{value}</span>
        <span className="text-xs text-text-secondary mt-1 block">{label}</span>
      </GlassCard>
    </motion.div>
  );
}

function FadeIn({ children, delay }: { children: React.ReactNode; delay: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, ...springDefault }}
    >
      {children}
    </motion.div>
  );
}

function GrowthChart({ data }: { data: GrowthPoint[] }) {
  if (!data.length) return <p className="text-text-tertiary text-sm">No data</p>;

  const max = Math.max(...data.map((d) => d.cumulative));
  const w = 400;
  const h = 144;
  const pad = 4;

  const points = data.map((d, i) => {
    const x = (i / Math.max(data.length - 1, 1)) * w;
    const y = h - pad - ((d.cumulative / max) * (h - pad * 2));
    return `${x},${y}`;
  }).join(" ");

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-full" preserveAspectRatio="none">
      <defs>
        <linearGradient id="growth-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#6C5CE7" stopOpacity={0.2} />
          <stop offset="100%" stopColor="#6C5CE7" stopOpacity={0} />
        </linearGradient>
      </defs>
      <polygon points={`0,${h} ${points} ${w},${h}`} fill="url(#growth-fill)" />
      <polyline points={points} fill="none" stroke="#6C5CE7" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
```

### `truememory/dashboard/frontend/src/views/Settings.tsx`
```
import { useState } from "react";
import { GlassCard } from "../components/GlassCard";
import { ConfirmSheet } from "../components/ConfirmSheet";
import { api } from "../lib/api";
import { formatBytes } from "../lib/formatters";
import type { HealthResponse, UpdateInfo } from "../lib/types";

interface Props {
  health: HealthResponse | null;
}

export function Settings({ health }: Props) {
  const [updateInfo, setUpdateInfo] = useState<UpdateInfo | null>(null);
  const [checking, setChecking] = useState(false);
  const [confirmDeleteAll, setConfirmDeleteAll] = useState(false);

  const handleCheckUpdate = async () => {
    setChecking(true);
    try {
      const info = await api.checkUpdate();
      setUpdateInfo(info);
    } finally {
      setChecking(false);
    }
  };

  const handleDeleteAll = async () => {
    await fetch("/api/memories", { method: "DELETE" }).catch(() => {});
    setConfirmDeleteAll(false);
    window.location.reload();
  };

  const tierColors: Record<string, string> = {
    edge: "#30D158",
    base: "#64D2FF",
    pro: "#BF5AF2",
  };
  const tier = health?.tier || "edge";

  return (
    <div className="h-full overflow-y-auto p-6 space-y-5 max-w-2xl">
      <h2 className="text-xl font-semibold mb-5">Settings</h2>

      <GlassCard>
        <h3 className="text-sm font-semibold mb-4">Tier</h3>
        <div className="flex items-center gap-3">
          <span
            className="text-sm font-bold uppercase px-2.5 py-1 rounded-lg"
            style={{
              backgroundColor: `${tierColors[tier]}20`,
              color: tierColors[tier],
            }}
          >
            {tier}
          </span>
          <span className="text-sm text-text-secondary">
            {tier === "edge" && "Lightweight local model (Model2Vec 8M)"}
            {tier === "base" && "Full semantic search (Qwen3 600M)"}
            {tier === "pro" && "Full semantic + HyDE query expansion"}
          </span>
        </div>
      </GlassCard>

      <GlassCard>
        <h3 className="text-sm font-semibold mb-4">Database</h3>
        <div className="space-y-2.5 text-[13px]">
          <Row label="Path">
            <span className="font-mono text-text-tertiary text-xs break-all">
              {health?.db_path || "—"}
            </span>
          </Row>
          <Row label="Size">
            <span>{health ? formatBytes(health.db_size_kb) : "—"}</span>
          </Row>
          <Row label="Memories">
            <span>{health?.memory_count?.toLocaleString() || "—"}</span>
          </Row>
        </div>
      </GlassCard>

      <GlassCard>
        <h3 className="text-sm font-semibold mb-4">Version</h3>
        <div className="flex items-center gap-4">
          <span className="text-[13px] font-mono">{health?.version || "—"}</span>
          <button
            onClick={handleCheckUpdate}
            disabled={checking}
            className="px-3 py-1.5 text-xs rounded-lg bg-accent/15 text-accent hover:bg-accent/25 transition-colors font-medium"
          >
            {checking ? "Checking…" : "Check for Updates"}
          </button>
        </div>
        {updateInfo && (
          <div className="mt-3 text-[13px]">
            {updateInfo.update_available ? (
              <p className="text-success">
                Update available: <span className="font-mono font-bold">{updateInfo.latest}</span>
                <br />
                <span className="text-text-tertiary text-xs">
                  Run: pip install --upgrade truememory
                </span>
              </p>
            ) : updateInfo.error ? (
              <p className="text-text-tertiary">{updateInfo.error}</p>
            ) : (
              <p className="text-text-secondary">You're on the latest version.</p>
            )}
          </div>
        )}
      </GlassCard>

      <GlassCard>
        <h3 className="text-sm font-semibold mb-2">About</h3>
        <p className="text-[13px] text-text-secondary">
          TrueMemory — SOTA agent memory at zero infrastructure cost.
        </p>
        <p className="text-[12px] text-text-tertiary mt-1">
          Built by Josh Adler
        </p>
      </GlassCard>

      <GlassCard className="border-error/20">
        <h3 className="text-sm font-semibold text-error mb-3">Danger Zone</h3>
        <p className="text-[13px] text-text-secondary mb-3">
          Permanently delete all memories from the database. This cannot be undone.
        </p>
        <button
          onClick={() => setConfirmDeleteAll(true)}
          className="px-4 py-2 text-xs rounded-lg bg-error/10 text-error hover:bg-error/20 transition-colors font-medium"
        >
          Delete All Memories
        </button>
      </GlassCard>

      <ConfirmSheet
        open={confirmDeleteAll}
        title="Delete All Memories"
        message="This will permanently delete every memory in your database. This action cannot be undone."
        confirmLabel="Delete Everything"
        onConfirm={handleDeleteAll}
        onCancel={() => setConfirmDeleteAll(false)}
      />
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4">
      <span className="text-text-tertiary flex-shrink-0">{label}</span>
      <div className="text-right">{children}</div>
    </div>
  );
}
```

### `truememory/dashboard/frontend/tailwind.config.ts`
```
import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: {
          base: "#1C1C1E",
          elevated: "#2C2C2E",
          grouped: "#3A3A3C",
          tertiary: "#48484A",
        },
        text: {
          primary: "#FFFFFF",
          secondary: "#8E8E93",
          tertiary: "#636366",
        },
        accent: {
          DEFAULT: "#6C5CE7",
          hover: "#7C6CF7",
          muted: "rgba(108, 92, 231, 0.15)",
        },
        success: "#30D158",
        warning: "#FFD60A",
        error: "#FF453A",
        info: "#64D2FF",
      },
      fontFamily: {
        system: [
          "-apple-system",
          "BlinkMacSystemFont",
          "SF Pro Display",
          "system-ui",
          "sans-serif",
        ],
        mono: ["ui-monospace", "SF Mono", "Cascadia Code", "monospace"],
      },
      borderRadius: {
        apple: "12px",
      },
    },
  },
  plugins: [],
};

export default config;
```

### `truememory/dashboard/frontend/vite.config.ts`
```
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8484",
    },
  },
});
```

### `truememory/dashboard/frontend/package.json`
```
{
  "name": "truememory-dashboard",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "@tanstack/react-virtual": "^3.11.0",
    "d3": "^7.9.0",
    "framer-motion": "^11.15.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@types/d3": "^7.4.3",
    "@types/react": "^18.3.12",
    "@types/react-dom": "^18.3.1",
    "@vitejs/plugin-react": "^4.3.4",
    "autoprefixer": "^10.4.20",
    "postcss": "^8.4.49",
    "tailwindcss": "^3.4.16",
    "typescript": "^5.6.3",
    "vite": "^5.4.11"
  }
}
```

### `truememory/dashboard/frontend/index.html`
```
<!doctype html>
<html lang="en" class="dark">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>TrueMemory</title>
    <style>
      body { background: #1C1C1E; margin: 0; }
    </style>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```
