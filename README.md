<p align="center">
  <img src="https://raw.githubusercontent.com/buildingjoshbetter/TrueMemory/main/assets/charts/hero-banner.png?v=2" alt="TrueMemory" />
</p>

<p align="center">
  One SQLite file. Zero cloud. One command to set up.
</p>

<p align="center">
  <a href="https://pypi.org/project/truememory/"><img src="https://img.shields.io/pypi/v/truememory?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/truememory/"><img src="https://img.shields.io/pypi/pyversions/truememory?color=blue" alt="Python"></a>
  <a href="https://github.com/buildingjoshbetter/TrueMemory/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="License"></a>
  <img src="https://img.shields.io/badge/Edge-90.1%25_LoCoMo-brightgreen" alt="Edge Score">
  <img src="https://img.shields.io/badge/Base-91.5%25_LoCoMo-blue" alt="Base Score">
  <img src="https://img.shields.io/badge/Pro-91.8%25_LoCoMo-blueviolet" alt="Pro Score">
</p>

<p align="center">
  <strong>🏆 91.8% on LoCoMo (Pro) · 📦 One SQLite File · ☁️ Zero Cloud · 💰 Zero Infrastructure Cost</strong>
</p>

<p align="center">
  <a href="#-benchmark">Benchmark</a> · <a href="#-research-highlights">Highlights</a> · <a href="#%EF%B8%8F-edge--base--pro">Edge / Base / Pro</a> · <a href="#-quickstart">Install</a> · <a href="#-what-happens-on-first-run">First Run</a> · <a href="#-api">API</a>
</p>

---

## 🔬 Benchmark

Tested on [LoCoMo](https://github.com/snap-research/locomo), the standard benchmark for conversational memory. 1,540 questions across 10 conversations. All 8 systems share the same answer model, judge, scoring, top-k, and byte-identical answer prompt — only retrieval differs.

> **Note on charts:** the hero banner and the three charts below still show the v0.3.0 single-Pro-tier layout (91.5%). Chart regeneration is tracked for a later release; all numerical claims in the README text and tables reflect the v0.4.0 three-tier scores (90.1 / 91.5 / 91.8%).

<p align="center">
  <img src="https://raw.githubusercontent.com/buildingjoshbetter/TrueMemory/main/assets/charts/leaderboard-bar.png?v=2" alt="LoCoMo 8-System Comparison" />
</p>

TrueMemory achieves **state-of-the-art accuracy for fully-local memory systems** at zero ongoing infrastructure cost. Edge and Base run entirely offline with no API keys. Pro adds one small LLM call per query for HyDE query expansion.

<p align="center">
  <img src="https://raw.githubusercontent.com/buildingjoshbetter/TrueMemory/main/assets/charts/accuracy-vs-cost.png?v=2" alt="Accuracy vs Infrastructure Cost" />
</p>

All scores use the same evaluation pipeline: GPT-4.1-mini answer generation, GPT-4o-mini judge (3x majority vote), temperature=0. Zero errors across 12,320 total answers. Scores use a lenient semantic-match judge; rankings are valid across all systems but absolute values are higher than published LoCoMo baselines using strict exact-match. [Full methodology](https://github.com/buildingjoshbetter/TrueMemory/blob/main/benchmarks/locomo/BENCHMARK_RESULTS.md) and reproduction scripts in [`benchmarks/`](https://github.com/buildingjoshbetter/TrueMemory/blob/main/benchmarks/locomo/).

---

## ⚡ Research Highlights

- **30+ percentage points more accurate than Mem0** on LoCoMo (91.8% Pro vs 61.4%)
- **2x more cost-efficient** per correct answer than Mem0
- **Runs offline** on any device with Python 3.10+ and 512MB RAM (Edge tier)
- **One SQLite file, zero API keys** for Edge and Base tiers. The entire 6-layer system runs offline.
- **Within 2.7pp of EverMemOS**, the only higher-scoring system — and EverMemOS uses pre-computed retrieval rather than live search at query time.

<p align="center">
  <img src="https://raw.githubusercontent.com/buildingjoshbetter/TrueMemory/main/assets/charts/category-radar.png?v=2" alt="Category Breakdown" />
</p>

TrueMemory Pro nearly matches EverMemOS across all 4 question categories. Mem0 collapses on multi-hop reasoning (37.7% vs 90.7%).

---

## 🏗️ Edge / Base / Pro

Same features, same 6-layer pipeline. Three tiers trade off install size, hardware, and LoCoMo accuracy. All three use the same retrieval architecture (FTS5 + dense + RRF + cross-encoder reranker); the differences are the embedder, the reranker, and whether HyDE query expansion is used. Base and Pro share an embedder and reranker — only HyDE differs.

| | Edge | Base | Pro |
|---|------|------|-----|
| **LoCoMo** | 90.1% | 91.5% | 91.8% |
| **Embedder** | Model2Vec potion-base-8M (8M params, 256d) | Qwen3-Embedding-0.6B @ 256d Matryoshka (600M params) | Qwen3-Embedding-0.6B @ 256d Matryoshka (600M params) |
| **Reranker** | ms-marco-MiniLM-L-6-v2 (22M) | gte-reranker-modernbert-base (149M) | gte-reranker-modernbert-base (149M) |
| **HyDE** | off | off | on (requires an LLM API key) |
| **Runs on** | Any machine, CPU only | 4GB+ RAM, CPU or GPU | 4GB+ RAM, CPU or GPU + LLM API key |
| **First install** | ~30MB | ~1.5GB one-time download | ~1.5GB one-time download |
| **Speed** | Ultra-fast | Fast | Fast + 1 LLM call/query |

**Edge** works everywhere. **Base** is the strongest fully-offline tier. **Pro** adds HyDE for the highest LoCoMo score.

---

## 🚀 Quickstart

### Claude Code / Claude Desktop

One command. Works on any Mac or Linux box, **even if your system Python is old or missing entirely.**

**Step 1.** Open Terminal:
- **Mac:** press `Cmd + Space`, type `Terminal`, press `Enter`
- **Linux:** press `Ctrl + Alt + T` (or open your distro's terminal app)

**Step 2.** Paste this one line and press `Enter`:

```bash
curl -LsSf https://raw.githubusercontent.com/buildingjoshbetter/TrueMemory/main/install.sh | sh
```

**Step 3.** Wait ~1-2 minutes while it downloads and installs. You'll see progress messages scroll by — that's normal.

**Step 4.** If Claude Desktop was already open, **quit it with `Cmd+Q` and reopen it** (a new chat window is not enough — the config is only read at launch). Then start a new Claude session and TrueMemory walks you through choosing **Edge**, **Base**, or **Pro** on first run.

> **What this actually does:** installs [uv](https://docs.astral.sh/uv/) (Astral's Python tool manager) if needed, fetches a managed Python 3.12 into `~/.local/share/uv/`, installs TrueMemory into an isolated tool environment, and auto-configures Claude Code and Claude Desktop. **Your system Python is never touched.** No sudo, no venvs, no pip struggle. Uninstall cleanly with `uv tool uninstall truememory`.

> **Want to audit the script first?** It's ~140 lines of shell, no sudo, stays entirely under `$HOME`. Read the source at [`install.sh`](https://github.com/buildingjoshbetter/TrueMemory/blob/main/install.sh), or download and inspect locally: `curl -LsSf https://raw.githubusercontent.com/buildingjoshbetter/TrueMemory/main/install.sh -o install.sh && less install.sh && sh install.sh`.

> **Want Base or Pro (adds Qwen3 embeddings + gte-reranker + sentence-transformers, ~1.5-2.5GB depending on OS)?**
> ```bash
> curl -LsSf https://raw.githubusercontent.com/buildingjoshbetter/TrueMemory/main/install.sh | TRUEMEMORY_EXTRAS="gpu,mcp" sh
> ```
> The default install is **Edge** (~30MB, the CPU-only tier). If you pick Base or Pro during first-run setup, TrueMemory will prompt you to install the extra models. Pro additionally requires an LLM API key at runtime for HyDE.
> *(Linux CPU-only boxes will pull PyTorch's default CUDA wheel, which is larger — ~2.5GB total. Mac installs are closer to ~1.5GB.)*

### Python library (for developers)

If you're embedding TrueMemory in your own Python project (requires Python 3.10+):

```bash
pip install truememory
```

```python
from truememory import Memory

m = Memory()
m.add("Prefers dark mode and TypeScript", user_id="alex")
m.add("Allergic to peanuts", user_id="alex")

results = m.search("What are Alex's preferences?", user_id="alex")
print(results[0]["content"])
# → "Prefers dark mode and TypeScript"
```

The database is created automatically at `~/.truememory/memories.db`.

---

## 🤖 What happens on first run?

Claude forgets you between sessions. TrueMemory fixes that.

On your first session after installing, TrueMemory will:

1. **Welcome you** and show your current version
2. **Ask Edge / Base / Pro** — you choose your accuracy tier
3. **Optionally accept an API key** — required for Pro (HyDE query expansion via Anthropic, OpenRouter, or OpenAI); optional for Edge and Base
4. **Show you how it works** — with example prompts to try

After setup, TrueMemory runs automatically. It stores what you tell it and recalls it in future sessions — no manual work needed.

### Make it automatic (optional)

Copy [`CLAUDE.md.example`](https://github.com/buildingjoshbetter/TrueMemory/blob/main/CLAUDE.md.example) to your home directory as `CLAUDE.md`. This tells Claude to store your preferences and recall them without being asked:

```bash
cp CLAUDE.md.example ~/CLAUDE.md
```

### Manual setup

If auto-setup doesn't detect your Claude installation, you can configure manually.

**Claude Code** (if you used the installer above):
```bash
claude mcp add truememory -- truememory-mcp
```

**Claude Desktop:** add to `claude_desktop_config.json` (Settings > Developer > Edit Config):

```json
{
  "mcpServers": {
    "truememory": {
      "command": "/Users/YOU/.local/bin/truememory-mcp"
    }
  }
}
```

> Use the **absolute path** to `truememory-mcp` — run `which truememory-mcp` to find it. Claude Desktop (and most non-Claude-Code MCP clients) don't inherit your shell's PATH, so relative commands will silently fail.

**No-install alternative (uvx):** skip installing TrueMemory entirely and let Claude run it ephemerally. Requires [uv](https://docs.astral.sh/uv/) to be installed.

```json
{
  "mcpServers": {
    "truememory": {
      "command": "/Users/YOU/.local/bin/uvx",
      "args": ["--python", "3.12", "--from", "truememory[mcp]", "truememory-mcp"]
    }
  }
}
```

uvx creates a cached environment on first run; subsequent spawns are fast. Good if you want TrueMemory to always be latest-on-PyPI without managing an install.

---

## Configuration

- `TRUEMEMORY_ENTITY_SHEETS=1` — re-enable the legacy L4 entity-profile writer (disabled by default in v0.6.0+). **Set before first `open()`** to preserve existing legacy rows on upgrade. Accepts `1`, `true`, `yes`, `on` (case-insensitive).

---

## 📖 API

| Method | What it does |
|--------|-------------|
| `m.add(content, user_id=None)` | Store a memory |
| `m.search(query, user_id=None, limit=10)` | Search memories |
| `m.search_deep(query, user_id=None, limit=10)` | Agentic multi-round search (higher latency + LLM cost; best for ambiguous queries) |
| `m.get(memory_id)` | Get one memory |
| `m.get_all(user_id=None, limit=100)` | List all memories |
| `m.update(memory_id, content)` | Update a memory |
| `m.delete(memory_id)` | Delete a memory |
| `m.delete_all(user_id=None)` | Delete all |

---

## 📊 Full Benchmark Details

Every benchmark script is self-contained and runs on [Modal](https://modal.com).

- **[Leaderboard & Reproduction](https://github.com/buildingjoshbetter/TrueMemory/blob/main/benchmarks/locomo/README.md)**: run any system yourself
- **[Full Technical Report](https://github.com/buildingjoshbetter/TrueMemory/blob/main/benchmarks/locomo/BENCHMARK_RESULTS.md)**: per-category breakdowns, latency, cost, hardware
- **[Evaluation Config](https://github.com/buildingjoshbetter/TrueMemory/blob/main/benchmarks/locomo/EVAL_CONFIG.md)**: exact models, prompts, parameters

---

## 📝 Citation

```bibtex
@software{truememory2026,
  title = {TrueMemory: State-of-the-Art Local-First Agent Memory},
  author = {@Building\_Josh},
  organization = {Sauron},
  year = {2026},
  url = {https://github.com/buildingjoshbetter/TrueMemory},
  version = {0.4.0}
}
```

---

## ⚖️ License

Licensed under [AGPL-3.0](https://github.com/buildingjoshbetter/TrueMemory/blob/main/LICENSE). Free for personal and research use. Commercial use requires a separate license — contact buildingjoshbetter@gmail.com.
