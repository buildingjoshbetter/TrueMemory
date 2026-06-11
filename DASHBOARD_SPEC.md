# TrueMemory Dashboard — Product & Technical Specification

**Version:** 0.1 (Draft)
**Date:** 2026-05-23
**Author:** Josh Adler

---

## Vision

A local-first, stunningly beautiful macOS application for managing, exploring,
and understanding your TrueMemory corpus. It looks and feels like Apple built
it — native glassmorphism, SF Pro typography, spring animations, sidebar
navigation. Under the hood it's a lightweight Python server + React frontend
running in a PyWebView native window. Zero Electron bloat, zero signing
ceremony, zero cloud dependency.

The dashboard is the first visual product surface for TrueMemory. It turns an
invisible background process into something users can see, touch, understand,
and pay for.

---

## Architecture

```
truememory-dashboard (CLI entry point)
│
├── Opens PyWebView native macOS window
│   └── Uses system WebKit (not bundled Chromium)
│   └── Native vibrancy, dock icon, Cmd+Tab
│   └── ~3MB overhead on top of existing install
│
├── FastAPI server on localhost:8484
│   └── Imports truememory.Memory + TrueMemoryEngine directly
│   └── No subprocess bridging, no MCP calls, no serialization
│   └── Same Python process owns the engine singleton
│   └── SQLite WAL mode — safe concurrent reads while MCP server writes
│
├── React 18 + Vite + Tailwind frontend
│   └── Built at install time, served as static files by FastAPI
│   └── framer-motion for spring animations
│   └── D3.js for entity graph + timeline visualizations
│   └── Recharts for line charts + analytics
│
└── ~/.truememory/memories.db (single SQLite file, already exists)
```

### Why This Stack

| Rejected          | Why                                                    |
|-------------------|--------------------------------------------------------|
| Electron          | 150MB+ bundled Chromium. Absurd for a local tool.      |
| Tauri             | Rust shell requires Python bridging for every query.   |
| SwiftUI native    | Xcode + signing + notarization. Same bridging problem. |
| Streamlit         | Can't achieve Apple-quality design. No glassmorphism.  |
| iOS / Mac Catalyst| App Store review for a developer tool. No.             |

PyWebView on macOS uses the system WebKit engine. It supports native vibrancy
(`vibrancy=True`), creates a real dock icon, supports Cmd+Tab, and adds
essentially nothing to the install size. The frontend has full access to WebGL,
CSS backdrop-filter, canvas, SVG — everything Chrome can do, because it's the
same rendering engine.

### Entry Point

```python
# truememory/dashboard/cli.py
import webview
import threading
from truememory.dashboard.server import create_app

def main():
    app = create_app()
    threading.Thread(target=uvicorn.run, args=(app,),
                     kwargs={"host": "127.0.0.1", "port": 8484},
                     daemon=True).start()
    webview.create_window(
        "TrueMemory",
        "http://127.0.0.1:8484",
        width=1440, height=900,
        min_size=(1024, 600),
        vibrancy=True,
    )
    webview.start()
```

Ships as `truememory-dashboard` CLI command via pyproject.toml `[project.scripts]`.

---

## Design System

### Principles

This app must feel like it was designed by Apple. Not "inspired by" — 
indistinguishable from. Every pixel, every animation, every interaction should
feel like it ships with macOS.

### Typography

```css
--font-system: -apple-system, BlinkMacSystemFont, "SF Pro Display", system-ui, sans-serif;
--font-mono: ui-monospace, "SF Mono", "Cascadia Code", monospace;
```

No Inter. No custom fonts. The system font stack gives us SF Pro on Mac for
free — the same font in every native Apple app. This is the single most
important signal that the app "belongs."

| Context            | Weight | Size   |
|--------------------|--------|--------|
| Page titles        | 700    | 28px   |
| Section headers    | 600    | 20px   |
| Body text          | 400    | 14px   |
| Secondary / labels | 400    | 12px   |
| Memory content     | 400    | 13px, monospace |
| Badges / pills     | 500    | 11px   |

### Color System — Apple Dark Mode

Layered grays with depth, not flat charcoal.

```css
/* Backgrounds — layered with increasing elevation */
--bg-base:      #1C1C1E;   /* Window background */
--bg-elevated:  #2C2C2E;   /* Cards, panels */
--bg-grouped:   #3A3A3C;   /* Grouped content, hover states */
--bg-tertiary:  #48484A;   /* Active states, selected items */

/* Text */
--text-primary:   #FFFFFF;
--text-secondary: #8E8E93;  /* Apple systemGray */
--text-tertiary:  #636366;  /* Apple systemGray2 */

/* Accent — TrueMemory signature */
--accent:         #6C5CE7;  /* Warm indigo — distinctive but Apple-native */
--accent-hover:   #7C6CF7;
--accent-muted:   rgba(108, 92, 231, 0.15);

/* Semantic */
--success:  #30D158;  /* Apple systemGreen */
--warning:  #FFD60A;  /* Apple systemYellow */
--error:    #FF453A;  /* Apple systemRed */
--info:     #64D2FF;  /* Apple systemCyan */

/* Borders */
--border:       rgba(255, 255, 255, 0.06);
--border-hover: rgba(255, 255, 255, 0.12);

/* Shadows */
--shadow-card: 0 2px 8px rgba(0, 0, 0, 0.3);
--shadow-elevated: 0 8px 32px rgba(0, 0, 0, 0.5);
```

### Glassmorphism — Apple Vibrancy

```css
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
}
```

Key: Apple uses 40-80px blur, not 10px. `saturate(180%)` is the trick that
makes vibrancy feel vibrant instead of muddy.

### Animations — Spring Physics

Everything moves with mass. No linear easing. No `transition: all 0.3s`.

```tsx
// Standard transition — cards, panels, list items
const springDefault = { type: "spring", stiffness: 400, damping: 30 };

// Gentle — modals, sheets, overlays
const springGentle = { type: "spring", stiffness: 300, damping: 35 };

// Snappy — toggles, pills, small elements
const springSnappy = { type: "spring", stiffness: 500, damping: 28 };
```

All views use framer-motion `layout` prop for automatic layout animations.
List items animate in with staggered `initial/animate` variants.

### Interaction Patterns

| Pattern                 | Implementation                                 |
|-------------------------|-------------------------------------------------|
| Sidebar navigation      | Like Finder — icon + label, highlight on hover |
| View switching           | Segmented controls, not tabs                   |
| Search                  | Cmd+K command palette (global), plus per-view  |
| Context menus           | Right-click on any memory or entity             |
| Selection               | Click to select, Shift+click range, Cmd+click multi |
| Drag                    | Drag memories to categories (future)            |
| Scroll                  | Momentum scroll, no pagination buttons          |
| Empty states            | Illustration + helpful text, never blank        |
| Loading                 | Skeleton screens, never spinners                |
| Destructive actions     | Red text, confirmation sheet with Apple styling  |

### Window Chrome

```
┌─────────────────────────────────────────────────────────────┐
│  ● ● ●                    TrueMemory                        │
├──────────┬──────────────────────────────────────────────────┤
│          │                                                   │
│ Sidebar  │                 Main Content                      │
│          │                                                   │
│          │                                                   │
│          │                                                   │
│          │                                                   │
│          │                                                   │
│          │                                                   │
├──────────┤                                                   │
│ Status   │                                                   │
│ Bar      │                                                   │
└──────────┴──────────────────────────────────────────────────┘
```

Sidebar width: 220px, collapsible to 60px (icon-only). Status bar at sidebar
bottom shows: tier badge, memory count, update indicator.

---

## Views

### 1. Overview (Home)

The first thing you see. Dense but not cluttered. Everything at a glance.

**Layout:**
- Top row: 4 stat cards (total memories, memories this week, entities tracked,
  encoding gate pass rate) — each with sparkline showing 30-day trend
- Middle: "Recent Memories" stream — last 20 memories as compact cards,
  live-updating
- Right column: System Health panel (tier, reranker status, model server,
  DB size, backlog depth)
- Bottom: Quick search bar that routes to Explorer

**Stat Cards:**
```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  4,906           │  │  127             │  │  10              │  │  78%             │
│  Total Memories  │  │  This Week       │  │  Entities        │  │  Gate Pass Rate  │
│  ▁▂▃▄▅▆▇█▇▆     │  │  ▁▁▂▃▅▇█▅▃▂     │  │  ▁▁▁▂▂▃▃▃▃▃     │  │  ▃▅▆▇▇▇▆▇▇▇     │
└──────────────────┘  └──────────────────┘  └──────────────────┘  └──────────────────┘
```

Each card: glass panel, large number (28px, bold), label (12px, secondary),
inline SVG sparkline (32px tall, accent color with gradient fill).

**Recent Memories Stream:**
Compact cards — one line of content (truncated with ellipsis), category badge,
timestamp (relative: "2h ago"), sender pill if present. Hover expands to show
full content. Click opens in Explorer with that memory selected.

### 2. Memory Explorer

The workhorse. Where you spend most of your time.

**Layout: Three-Column**
```
┌──────────────────────────────────────────────────────────────────┐
│  [Search bar]  [Category ▾]  [Sender ▾]  [Date Range]  [Sort ▾] │
├──────────────────────────────────┬───────────────────────────────┤
│                                  │                               │
│  Memory List                     │  Inspector Panel              │
│  (scrollable, virtualized)       │  (selected memory detail)     │
│                                  │                               │
│  ┌────────────────────────────┐  │  Content (full text)          │
│  │ [preference] 2h ago       │  │  ─────────────────────        │
│  │ User prefers dark mode... │  │  Metadata:                    │
│  │ josh · score: 1.42        │  │    Category: preference       │
│  └────────────────────────────┘  │    Sender: josh               │
│  ┌────────────────────────────┐  │    Timestamp: 2026-05-22...   │
│  │ [technical] 5h ago        │  │    Score: 1.42                │
│  │ TrueMemory uses sqlite... │  │    Source: fts+entity_boost   │
│  │ · score: 1.21             │  │    Episode: #47               │
│  └────────────────────────────┘  │                               │
│  ┌────────────────────────────┐  │  Actions:                     │
│  │ ...                       │  │    [Edit] [Delete] [Source]    │
│  └────────────────────────────┘  │                               │
│                                  │  Related Memories:            │
│                                  │    • "User prefers line..."   │
│                                  │    • "Dashboard design..."    │
│                                  │                               │
├──────────────────────────────────┴───────────────────────────────┤
│  Showing 4,906 memories · Selected: 1 · Bulk: [Delete] [Export] │
└──────────────────────────────────────────────────────────────────┘
```

**Search:**
- Types into search bar → debounced 300ms → hits full 6-layer retrieval pipeline
- Results show relevance score + source layer badge (L0/L2/L3/L5/HyDE)
- When search is empty, shows all memories sorted by timestamp (most recent first)
- Cmd+K opens global command palette from any view

**Filters:**
- Category: dropdown multi-select (technical, preference, decision, personal,
  correction, temporal, relationship)
- Sender: dropdown (auto-populated from DB)
- Date range: calendar picker (Apple-style, not a generic datepicker)
- Sort: Relevance (when searching), Newest, Oldest, Category

**Memory Cards:**
- Category badge (colored pill: technical=blue, preference=purple,
  decision=amber, personal=green, correction=red, temporal=cyan)
- Content preview (2 lines, truncated)
- Timestamp (relative), sender (if present), relevance score (when searching)
- Hover: subtle background shift to --bg-grouped
- Click: selects, opens Inspector panel on right
- Right-click: context menu (Edit, Delete, View Source Session, Copy)

**Inspector Panel:**
- Slides in from right with spring animation (width: 380px)
- Full memory content rendered with monospace font
- Metadata table: category, sender, timestamp, score, source layer, episode ID
- "Edit" — inline content editing with save/cancel
- "Delete" — confirmation sheet (Apple red, "This action cannot be undone")
- "View Source Session" — jumps to Sessions view filtered to this memory's
  originating session (if traceable via extracted/ data)
- "Related Memories" — quick vector search showing 5 most similar memories

**Bulk Operations:**
- Shift+click / Cmd+click to multi-select
- Bottom bar appears with count + actions: Delete Selected, Export as JSON
- Select All (with current filters applied)

**Virtualization:**
- 4,906+ memories can't render as DOM nodes. Use react-window or @tanstack/virtual
- Smooth scrolling with overscan, no jank at any corpus size

### 3. Sessions (Paid)

The killer feature. Full archive of every Claude Code conversation, searchable.

**Data Source:**
Claude Code stores session transcripts as JSONL files in
`~/.claude/projects/*/[session-id].jsonl`. Each line has `type` (user,
assistant, attachment, tool_call, tool_result), `timestamp`, `sessionId`,
and `content`. Session metadata lives in `~/.claude/sessions/[pid].json`
with `sessionId`, `cwd`, `startedAt`, `version`, `status`.

TrueMemory's ingest pipeline already sees these transcripts at session end.
Currently it extracts atomic facts and discards the transcript. The dashboard
indexes these sessions for browsing and search.

**Session Index Build:**
On first launch (and periodically), the dashboard scans `~/.claude/projects/`
and `~/.claude/sessions/` to build a session index. Stored in a new
`dashboard_sessions` table in `memories.db`:

```sql
CREATE TABLE IF NOT EXISTS dashboard_sessions (
    session_id TEXT PRIMARY KEY,
    project_dir TEXT,
    started_at TEXT,
    ended_at TEXT,
    message_count INTEGER,
    word_count INTEGER,
    summary TEXT,
    version TEXT
);
```

Summary is auto-generated from the first user message + assistant response
(or via LLM summarization on first index).

**Layout:**
```
┌──────────────────────────────────────────────────────────────────┐
│  [Search sessions...]           [Project ▾]  [Date Range]       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  May 22, 2026 · 10:34 PM                       42 msgs    │  │
│  │  /Users/j/Desktop/TrueMemory                              │  │
│  │  "PyPI push for v0.7.1, then dashboard design discussion" │  │
│  │  ▸ 12 memories extracted                                  │  │
│  └────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  May 22, 2026 · 3:15 PM                        18 msgs    │  │
│  │  /Users/j/Desktop/TrueMemory                              │  │
│  │  "CI test suite fixes, Windows portability patches"       │  │
│  │  ▸ 8 memories extracted                                   │  │
│  └────────────────────────────────────────────────────────────┘  │
│  ...                                                             │
└──────────────────────────────────────────────────────────────────┘
```

**Click into a session → Transcript View:**
```
┌──────────────────────────────────────────────────────────────────┐
│  ← Back to Sessions    May 22, 2026 · TrueMemory · 42 messages │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  YOU  10:34 PM                                                   │
│  Can we go ahead and do a PyPI push on GitHub real quick?        │
│                                                                  │
│  CLAUDE  10:34 PM                                                │
│  Let me check the current state of the TrueMemory repo...       │
│                                                                  │
│  ┌─ Tool: Bash ─────────────────────────────────────────────┐   │
│  │ git status && git remote -v && git log --oneline -10     │   │
│  │ > On branch main                                          │   │
│  │ > Your branch is up to date with 'origin/main'.          │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
│  CLAUDE  10:35 PM                                                │
│  v0.7.1 is already live on PyPI — published earlier today.       │
│                                                                  │
│  ───── Memories Extracted from This Session ─────                │
│  • [decision] Dashboard design: FastAPI + React + PyWebView      │
│  • [preference] Apple HIG native design direction                │
│  • [preference] SF Pro system fonts, not Inter                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Transcript Rendering:**
- User messages: left-aligned, slightly lighter background
- Assistant messages: full-width, standard background
- Tool calls: collapsible blocks with syntax highlighting (collapsed by default)
- Tool results: collapsible, truncated to 20 lines with "Show more"
- Code blocks: syntax highlighted (highlight.js or Shiki)
- Extracted memories: linked section at bottom — click any to jump to Explorer

**Search:**
- Semantic search across all session transcripts using TrueMemory's retrieval
  pipeline (sessions indexed as searchable content)
- Search results highlight matching passages within the transcript
- Filter by project directory, date range

### 4. People (Paid)

Entity profiles and relationship graph.

**Layout: Split View**
- Left: Force-directed D3 graph of entity relationships
- Right: Selected entity profile detail

**Graph:**
- Nodes = entities from `entity_profiles` table
- Node size = message_count (logarithmic scale)
- Edges = relationships from `entity_relationships` table
- Edge thickness = strength value
- Concentric rings = Dunbar layers (intimate → active → extended)
- Click node → select entity, show profile on right
- Hover node → highlight connected edges, dim unrelated nodes
- Zoom/pan with mouse wheel + drag
- Physics simulation settles naturally, then holds position

**Entity Profile Panel:**
```
┌───────────────────────────────────┐
│  Josh                             │
│  274 memories · Primary user      │
│                                   │
│  Traits                           │
│  ───────                          │
│  entrepreneur, founder, technical │
│  detail-oriented, high design bar │
│                                   │
│  Communication Style              │
│  ────────────────────             │
│  casual, lowercase, abbreviations │
│  direct, prefers terse responses  │
│                                   │
│  Topics                           │
│  ──────                           │
│  TrueMemory, AI memory, startups  │
│  infrastructure, personal health  │
│                                   │
│  Recent Memories ──────────────── │
│  • "Prefers line charts over..."  │
│  • "Dashboard design: Apple HIG"  │
│  • "Wants maximalist info..."     │
│                                   │
│  [View All Memories by Josh →]    │
└───────────────────────────────────┘
```

Note: `entity_relationships` table is currently empty in production. The graph
starts sparse and grows as TrueMemory's L0 pipeline populates it. Empty state:
show entities as standalone nodes without edges, with a note explaining
relationships will appear as more conversations are processed.

### 5. Facts & Contradictions (Paid)

Fact timeline with supersession tracking and contradiction detection.

**Layout:**
```
┌──────────────────────────────────────────────────────────────────┐
│  [Search facts...]              [Subject ▾]  [Show superseded]  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Location                                                        │
│  ─────────                                                       │
│  ● Josh lives in Austin, TX              May 2026    [current]  │
│  ○ Josh lives in San Francisco           Jan 2026    [superseded]│
│                                                                  │
│  Employment                                                      │
│  ──────────                                                      │
│  ● Josh is founder of Sauron Inc.        Apr 2026    [current]  │
│                                                                  │
│  Technical Preferences                                           │
│  ─────────────────────                                           │
│  ● Prefers bun over npm                  May 2026    [current]  │
│  ● Prefers FastAPI + React stack         May 2026    [current]  │
│  ○ Uses Create React App                 Mar 2026    [superseded]│
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

- Current facts: full opacity, accent-colored bullet
- Superseded facts: dimmed (40% opacity), strikethrough, gray bullet
- Toggle "Show superseded" to hide/show old facts
- Click any fact → opens source memory in Explorer
- Contradiction highlight: when two current (non-superseded) facts conflict,
  show amber warning badge

Note: `fact_timeline` is currently empty. This view activates as L5
consolidation populates it. Until then, show an empty state with
explanation of what the fact timeline will track.

### 6. Analytics (Paid)

Usage patterns and corpus health over time.

**Panels:**
- **Memory Growth** — Line chart (filled area, accent color) showing cumulative
  memory count over time. X-axis: days. Y-axis: count. Must be a line chart,
  not bar chart.
- **Category Breakdown** — Donut chart showing distribution
  (technical: 32%, preference: 25%, decision: 18%, etc.)
- **Daily Ingest Rate** — Line chart of memories created per day
- **Encoding Gate Funnel** — Visual funnel: "Raw extractions → Gate pass →
  Dedup pass → Stored" with counts and percentages at each stage
- **Top Entities** — Horizontal bar chart of most-referenced entities
- **Search Performance** — Average retrieval latency, cache hit rate
- **Tier Comparison** — If user has run benchmarks, show LoCoMo scores by tier

**Data Source:**
Analytics are computed from aggregate queries on the `messages` table
(GROUP BY date, category, sender) and from ingest logs in
`~/.truememory/extracted/` and `~/.truememory/logs/`.

### 7. Settings

**Sections:**

**Tier Management:**
- Visual tier selector: three cards (Edge, Base, Pro) with current highlighted
- Feature comparison grid (what each tier includes)
- Switch tier → shows rebuild progress inline (reads `rebuild_status` table)
- API key input for Pro tier (masked, with test button)

**Update:**
- Current version display (reads from `truememory.__version__`)
- "Check for Updates" button → queries PyPI JSON API
- If update available: version diff, changelog preview, "Update Now" button
- Update runs `pip install --upgrade truememory` via subprocess
- Progress indicator → restart prompt on completion

**Database:**
- DB path display (clickable to reveal in Finder)
- DB size (human-readable)
- Memory count
- "Export All" → dumps all memories as JSON
- "Import" → loads memories from JSON export
- Danger zone: "Delete All Memories" (double confirmation)

**About:**
- Version, tier, license status
- Links: GitHub, docs, support email
- "Built by Josh Adler" credit

---

## Free vs. Paid

### Philosophy

The free tier must be genuinely useful — not a crippled demo. Users need to be
able to see and manage their memories without paying. The paid tier unlocks the
features that justify $10/month: session intelligence, visual analytics,
entity graphs, and the encoding gate inspector.

### Split

| Feature                          | Free | Pro ($10/mo) |
|----------------------------------|------|--------------|
| Overview dashboard               | ✓    | ✓            |
| Memory Explorer (search + CRUD)  | ✓    | ✓            |
| Basic stats (count, tier, size)  | ✓    | ✓            |
| Settings + Update button         | ✓    | ✓            |
| Bulk delete                      | ✓    | ✓            |
| Session Archive + Search         |      | ✓            |
| Session Transcript Viewer        |      | ✓            |
| People (Entity Graph + Profiles) |      | ✓            |
| Facts & Contradictions           |      | ✓            |
| Analytics Dashboard              |      | ✓            |
| Encoding Gate Inspector          |      | ✓            |
| Export to JSON                   |      | ✓            |
| Cmd+K Command Palette            |      | ✓            |
| Related Memories in Inspector    |      | ✓            |

### What Free Looks Like

The free user gets: Overview, Explorer, Settings. Three sidebar items. The
Explorer has full semantic search (because that's TrueMemory's core value
proposition — you can't demo the product with FTS-only search). They can
browse, search, edit, delete memories. They see their tier, DB size, health.
They can update TrueMemory from the dashboard.

Paid views show in the sidebar with a subtle lock icon. Clicking them opens a
tasteful upgrade prompt — not a paywall modal, just a glass card explaining
what the feature does with a "Start Free Trial" button.

### Licensing

- **Lemon Squeezy** for payment processing (simpler than Stripe for indie
  products, handles tax/VAT globally)
- License key stored in `~/.truememory/config.json` alongside tier config
- Validation: single API call on launch to verify key is active
- Grace period: works offline for 7 days without validation
- No DRM, no heavy enforcement — trust the user, make it easy to pay
- 14-day free trial, no credit card required (just enter email)

### Revenue Model

$10/month is positioned as:
- Less than a coffee habit
- All compute is local (you're not paying for our servers)
- You're paying for the dashboard product, not the memory engine
- The memory engine (Edge/Base/Pro tiers) remains free and open source
- Dashboard Pro = the visual layer on top

This is distinct from the planned Corpus Sync pricing (monthly fee based on
corpus size for cloud storage). Corpus Sync is a separate product concern.
Dashboard Pro is about local visualization and management.

---

## API Routes

```
GET  /api/health                    → { version, tier, db_size, memory_count, ... }

# Memories
GET  /api/memories                  → paginated list, accepts ?search, ?category, ?sender, ?from, ?to, ?sort, ?limit, ?offset
GET  /api/memories/:id              → single memory with full metadata
PUT  /api/memories/:id              → update content
DEL  /api/memories/:id              → delete single
POST /api/memories/search           → { query, limit } → full pipeline search results
POST /api/memories/bulk-delete      → { ids: [...] }
POST /api/memories/export           → JSON download of all memories

# Sessions (Pro)
GET  /api/sessions                  → paginated session index, accepts ?search, ?project, ?from, ?to
GET  /api/sessions/:id              → session metadata
GET  /api/sessions/:id/transcript   → full JSONL transcript parsed into structured messages
POST /api/sessions/reindex          → trigger session index rebuild

# Entities (Pro)
GET  /api/entities                  → list all entities with message counts
GET  /api/entities/:name            → entity profile (traits, style, topics, relationships)
GET  /api/entities/graph            → nodes + edges for D3 visualization

# Facts (Pro)
GET  /api/facts                     → fact timeline, accepts ?subject, ?show_superseded
GET  /api/facts/contradictions      → conflicting current facts

# Analytics (Pro)
GET  /api/analytics/growth          → memory count by day
GET  /api/analytics/categories      → category distribution
GET  /api/analytics/ingest          → encoding gate stats from logs
GET  /api/analytics/entities        → top entities by memory count

# System
GET  /api/tier                      → current tier + capabilities
POST /api/tier/switch               → { tier, api_key? } → triggers tier switch
GET  /api/tier/rebuild-status       → rebuild progress if switching
POST /api/update/check              → query PyPI for latest version
POST /api/update/install            → run pip upgrade + restart
GET  /api/license                   → license status (Pro features unlocked?)
POST /api/license/activate          → { key } → validate and store
```

All Pro endpoints return `{ "requires_pro": true, "feature": "..." }` with
HTTP 402 when accessed without an active license.

---

## File Structure

```
truememory/
├── dashboard/
│   ├── __init__.py
│   ├── cli.py              # Entry point: truememory-dashboard
│   ├── server/
│   │   ├── __init__.py
│   │   ├── app.py          # FastAPI app factory
│   │   ├── deps.py         # TrueMemoryEngine singleton + DB connection
│   │   ├── license.py      # License validation logic
│   │   ├── routes/
│   │   │   ├── memories.py
│   │   │   ├── sessions.py
│   │   │   ├── entities.py
│   │   │   ├── facts.py
│   │   │   ├── analytics.py
│   │   │   └── system.py   # health, tier, update, license
│   │   └── session_index.py  # Claude Code session scanner + indexer
│   └── frontend/           # React 18 + Vite + Tailwind
│       ├── package.json
│       ├── vite.config.ts
│       ├── tailwind.config.ts
│       ├── index.html
│       ├── src/
│       │   ├── App.tsx
│       │   ├── main.tsx
│       │   ├── styles/
│       │   │   └── globals.css   # Design system tokens
│       │   ├── hooks/
│       │   │   ├── useMemories.ts
│       │   │   ├── useSessions.ts
│       │   │   ├── useEntities.ts
│       │   │   └── useHealth.ts
│       │   ├── views/
│       │   │   ├── Overview.tsx
│       │   │   ├── Explorer.tsx
│       │   │   ├── Sessions.tsx
│       │   │   ├── People.tsx
│       │   │   ├── Facts.tsx
│       │   │   ├── Analytics.tsx
│       │   │   └── Settings.tsx
│       │   ├── components/
│       │   │   ├── Sidebar.tsx
│       │   │   ├── CommandPalette.tsx  # Cmd+K
│       │   │   ├── MemoryCard.tsx
│       │   │   ├── InspectorPanel.tsx
│       │   │   ├── SessionCard.tsx
│       │   │   ├── TranscriptViewer.tsx
│       │   │   ├── EntityGraph.tsx     # D3 force-directed
│       │   │   ├── Timeline.tsx
│       │   │   ├── GlassCard.tsx
│       │   │   ├── StatCard.tsx
│       │   │   ├── CategoryBadge.tsx
│       │   │   ├── SearchBar.tsx
│       │   │   ├── UpgradePrompt.tsx   # Tasteful Pro upsell
│       │   │   └── ConfirmSheet.tsx    # Apple-style destructive confirmation
│       │   └── lib/
│       │       ├── api.ts            # Typed fetch wrapper
│       │       ├── constants.ts      # Colors, springs, breakpoints
│       │       └── formatters.ts     # Dates, numbers, truncation
│       └── dist/                     # Built output, served by FastAPI
```

---

## Build & Distribution

### PyPI Package

Dashboard ships as an optional dependency:

```toml
[project.optional-dependencies]
dashboard = [
    "fastapi>=0.100,<1.0",
    "uvicorn[standard]>=0.20,<1.0",
    "pywebview>=5.0,<6.0",
]

[project.scripts]
truememory-dashboard = "truememory.dashboard.cli:main"
```

Install: `pip install truememory[dashboard]`
Run: `truememory-dashboard`

The React frontend is pre-built (`npm run build` in CI) and the `dist/`
directory is included in the wheel. No Node.js runtime needed at install time.

### Frontend Build (CI)

GitHub Actions workflow builds the frontend and commits the `dist/` directory
before publishing to PyPI. Users never need Node.js installed.

---

## Implementation Priority

### Phase 1 — Foundation (Week 1-2)
- [ ] FastAPI server skeleton with health endpoint
- [ ] PyWebView launcher (macOS vibrancy, dock icon)
- [ ] React app with Tailwind + design system tokens
- [ ] Sidebar navigation shell
- [ ] Overview view with stat cards
- [ ] Memory Explorer: list, search, filter, paginate
- [ ] Inspector panel with memory detail + edit + delete

### Phase 2 — Sessions (Week 3-4)
- [ ] Session indexer (scan ~/.claude/projects/ and ~/.claude/sessions/)
- [ ] Sessions list view with search
- [ ] Transcript viewer with syntax highlighting
- [ ] Link memories back to source sessions

### Phase 3 — Visualization (Week 5-6)
- [ ] Entity graph (D3 force-directed)
- [ ] Entity profile panel
- [ ] Analytics dashboard (growth, categories, ingest funnel)
- [ ] Fact timeline view

### Phase 4 — Polish & Monetization (Week 7-8)
- [ ] Lemon Squeezy integration (license keys, trial)
- [ ] Free/Pro gate on views
- [ ] Upgrade prompts (tasteful, not aggressive)
- [ ] Self-update mechanism
- [ ] Cmd+K command palette
- [ ] Bulk operations
- [ ] CI pipeline: frontend build + PyPI publish
- [ ] Performance testing with large corpora (10K+ memories)

---

## Open Questions

1. **Session transcript storage:** Should we copy JSONL files into
   `memories.db` (single file, portable) or read them in-place from
   `~/.claude/projects/`? In-place is simpler but couples us to Claude Code's
   file layout. Copying adds a one-time index step but decouples.

2. **Auto-launch:** Should `truememory-dashboard` start automatically on
   macOS login? Menu bar agent that shows a dot icon + "Open Dashboard" menu
   item? Or keep it explicit / CLI-only?

3. **Accent color:** Warm indigo (#6C5CE7) proposed above. Should it match
   TrueMemory's brand color instead? Does TrueMemory have an established
   brand color?

4. **Encoding Gate Inspector:** How deep should this go? Full three-signal
   breakdown with threshold tuning? Or read-only view of what passed/failed?

5. **Mobile companion?** Not in scope now, but the FastAPI backend could
   serve a mobile-optimized view if accessed from a phone on the same network.
   Worth keeping the API clean enough to support this later.
