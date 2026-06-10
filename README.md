<p align="center">
  <img src="assets/charts/hero-banner.png" alt="TrueMemory" width="800">
</p>

<p align="center">
  <a href="https://github.com/buildingjoshbetter/TrueMemory"><img src="https://img.shields.io/github/stars/buildingjoshbetter/TrueMemory?style=social" alt="Stars"></a>
  <a href="https://pypi.org/project/truememory/"><img src="https://img.shields.io/badge/installs-7.3k+-brightgreen" alt="Installs"></a>
  <a href="./i18n/README.zh-CN.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
  <a href="./i18n/README.ru.md"><img alt="Русский" src="https://img.shields.io/badge/Русский-d9d9d9"></a>
  <a href="https://arxiv.org/abs/2605.04897"><img src="https://img.shields.io/badge/arXiv-2605.04897-b31b1b" alt="arXiv"></a>
  <a href="https://discord.gg/ZJ74JB2gVW"><img src="https://img.shields.io/badge/Discord-Join%20us-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/LoCoMo-93.0%25-blueviolet" alt="LoCoMo">
  <img src="https://img.shields.io/badge/LongMemEval-92.0%25-blue" alt="LongMemEval">
  <img src="https://img.shields.io/badge/BEAM--1M-76.6%25_(SOTA)-orange" alt="BEAM-1M">
</p>

<p align="center">
  <a href="#the-problem">Why</a> · <a href="#quick-start">Quick Start</a> · <a href="#how-truememory-compares">Compare</a> · <a href="#tiers">Tiers</a> · <a href="#benchmarks">Benchmarks</a> · <a href="#python-api">API</a> · <a href="#docs">Docs</a> · <a href="#faq">FAQ</a> · <a href="https://discord.gg/ZJ74JB2gVW">Discord</a>
</p>

<p align="center">
  <img src="assets/terminal-demo.svg" alt="TrueMemory terminal demo" width="750">
</p>

---

## Why TrueMemory

**It finds signal in the noise.** Your AI sees thousands of messages. TrueMemory figures out which ones actually matter and throws away the rest. No manual tagging, no prompt engineering. It just knows.

**It gets sharper over time.** It's not a static database. It's a living memory that grows with you. It resolves contradictions when you change your mind, updates stale facts, and consolidates what it knows. The longer you use it, the better it gets.

**It works without you thinking about it.** TrueMemory automatically captures memories from your conversations and automatically injects the right ones into your next session. You never have to store or search for anything manually. It just happens.

**It's 100% local.** One SQLite file on your machine. Your memories never leave your device. No cloud, no API keys needed. Your data is yours. (Anonymous usage telemetry is the only exception — never memory content — and one env var turns it off. [Details](#faq).)

> **Without TrueMemory:** "What framework are we using?" Asked for the 12th time this week. Your agent starts every session with amnesia. It doesn't know your name, your stack, or anything you told it yesterday.
>
> **With TrueMemory:** Your agent already knows you use FastAPI, prefer Pydantic v2, and that your auth middleware lives in `src/auth/`. It remembers your corrections, your preferences, and your decisions. Across every session, forever.

<p align="center"><a href="https://discord.gg/ZJ74JB2gVW"><strong>Join the Discord</strong></a> to see what others are building with TrueMemory.</p>

---

## How TrueMemory Compares

| System | LoCoMo | LongMemEval | Local-first | Auto-capture | License |
|--------|--------|-------------|:-----------:|:------------:|---------|
| **TrueMemory Pro** | **93.0%** | **92.0%** | ✅ | ✅ | AGPL-3.0 |
| **TrueMemory Base** | **92.0%** | **84.1%** | ✅ | ✅ | AGPL-3.0 |
| Mem0 | 61.4% | — | Partial | ❌ | Apache-2.0 |
| Supermemory | 65.4% | — | ❌ | ❌ | Cloud API |
| MemOS | 75.8% | — | ✅ | ❌ | Apache-2.0 |
| ReadAgent | 79.5% | — | ❌ | ❌ | Research |

> All benchmarks independently reproducible. Scripts included in [`benchmarks/`](benchmarks/).

---

## Quick Start

### For Claude Code / Claude CLI / Cursor / Codex CLI / Gemini CLI

```bash
curl -LsSf https://raw.githubusercontent.com/buildingjoshbetter/TrueMemory/main/install.sh | sh
```

> Installs everything in an isolated environment. Downloads ~1.5GB of AI models. No data leaves your machine. No sudo required.

<details><summary>New to the terminal? Click here for step-by-step instructions.</summary>

1. **Open a terminal:** Mac: `Cmd + Space`, type `Terminal`. Linux: `Ctrl + Alt + T`. Windows: open PowerShell.
2. **Paste the command above** and press Enter.
3. **Wait 3-5 minutes** for models to download.
4. **Quit your AI tool completely** and reopen it (Mac: `Cmd+Q`).
5. **Type "Set up TrueMemory"** and pick a tier.

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/buildingjoshbetter/TrueMemory/main/install.ps1 | iex
```

</details>

That's it. TrueMemory remembers your conversations automatically from here. Need help? [Join our Discord](https://discord.gg/ZJ74JB2gVW).

> If TrueMemory saves you time, [a ⭐ helps other devs find it](https://github.com/buildingjoshbetter/TrueMemory).

### For developers (Python library)

```bash
pip install truememory
```

```python
from truememory import Memory

m = Memory()
m.add("Prefers dark mode and TypeScript", user_id="alex")
print(m.search("preferences", user_id="alex"))
```

---

## Tiers

Same architecture, three tiers. All included in a single install. Switch anytime by saying "switch to Pro" or "switch to Base."

| | Edge | Base | Pro |
|---|------|------|-----|
| **LoCoMo** | 89.6% | 92.0% | **93.0%** |
| **LongMemEval** | — | — | **92.0%** |
| **BEAM-1M** | — | — | **76.6% (SOTA)** |
| **Embedding model** | 8 MB lightweight | 600 MB high-accuracy | 600 MB high-accuracy |
| **Reranker** | 22M params | 149M params | 149M params |
| **HyDE search** | — | — | ✅ (requires LLM API key) |
| **Runs on** | Any machine, CPU only | 4 GB+ RAM | 4 GB+ RAM + API key |

**Edge** works everywhere. **Base** is the strongest fully-offline tier. **Pro** adds AI-powered query expansion for the highest scores.

---

## Benchmarks

<p align="center">
  <img src="assets/charts/leaderboard-bar.png" alt="Benchmark Leaderboard" width="700" />
</p>

Tested across three major benchmarks with all systems sharing the same answer model (GPT-4.1-mini), judge (GPT-4o-mini, 3x majority vote), and scoring pipeline.

| Benchmark | What it tests | TrueMemory Pro |
|-----------|--------------|:--------------:|
| [LoCoMo](https://github.com/snap-research/locomo) | 1,540 questions across 10 conversations | **93.0%** |
| [LongMemEval](https://github.com/xiaowu0162/LongMemEval) | 500 multi-session questions | **92.0%** |
| [BEAM-1M](https://github.com/mohammadtavakoli78/BEAM) | 700 questions at 1M+ tokens | **76.6% (SOTA)** |
| BEAM-10M | 200 questions at 10M tokens | **65.0%** |

### Reproduce any result yourself

Every benchmark script is self-contained and runs on [Modal](https://modal.com).

- **[LoCoMo Scripts](benchmarks/locomo/scripts/)** — 8 systems (TrueMemory, Mem0, Zep, Engram, etc.)
- **[LoCoMo Results](benchmarks/locomo/BENCHMARK_RESULTS.md)** — per-category breakdowns, latency, cost
- **[LoCoMo Eval Config](benchmarks/locomo/EVAL_CONFIG.md)** — exact models, prompts, parameters
- **[LongMemEval Scripts](benchmarks/longmemeval/)** — oracle + strict variants
- **[LongMemEval Results](benchmarks/longmemeval/results/)** — 6 TM Pro runs + 5 competitor results
- **[BEAM-1M Script](benchmarks/beam/bench_truememory_pro_beam1m.py)** — 35 conversations at 1M+ tokens
- **[BEAM-10M Script](benchmarks/beam/bench_truememory_pro_beam10m.py)** — 10 conversations at 10M tokens
- **[BEAM Results](benchmarks/beam/)** — 3 runs (1M) + 1 run (10M)

All benchmarks use the same eval pipeline. Nothing is hidden. Full details: [LoCoMo](benchmarks/locomo/EVAL_CONFIG.md) | [LongMemEval](benchmarks/longmemeval/README.md) | [BEAM](benchmarks/beam/README.md)

---

## Works With

<p align="center">
  <strong>Claude Code</strong> · <strong>Claude CLI</strong> · <strong>Cursor</strong> · <strong>Codex CLI</strong> · <strong>Gemini CLI</strong> · <strong>Claude Desktop</strong>
</p>

Lifecycle hooks capture conversations automatically. No manual work needed. Your memories stay local in a single SQLite file.

---

## Python API

```python
from truememory import Memory

m = Memory()

m.add("Prefers dark mode and TypeScript", user_id="alex")
m.add("Works at Anthropic as a senior engineer", user_id="alex")
m.add("Always sign commits with GPG", directive=True)  # directives auto-load every session

results = m.search("What are Alex's preferences?", user_id="alex")
results = m.search_deep("career history?", user_id="alex")  # multi-round, higher accuracy
```

| Method | Description |
|--------|-------------|
| `m.add(content, user_id)` | Store a memory (`directive=True` for standing instructions that auto-load at the start of every session) |
| `m.search(query, user_id)` | Search (6-layer pipeline + reranker) |
| `m.search_deep(query, user_id)` | Multi-round agentic search |
| `m.get(id)` / `m.get_all(user_id)` | Retrieve memories |
| `m.update(id, content)` / `m.delete(id)` | Modify or remove |
| `m.stats()` | System statistics |

[Full API reference →](docs/python-api.md)

---

## Docs

| | |
|---|---|
| [Getting Started](docs/guides/getting-started.md) | Install to first memory |
| [Python API Reference](docs/python-api.md) | Full `Memory` class reference |
| [MCP Tool Reference](docs/mcp-tools.md) | All 8 MCP tools |
| [CLI Reference](docs/cli.md) | `truememory-mcp` and `truememory-ingest` |
| [Environment Variables](docs/env-vars.md) | All `TRUEMEMORY_*` config options |
| [Architecture Deep Dive](docs/architecture.md) | 6-layer retrieval pipeline, encoding gate |
| [Tier Selection Guide](docs/guides/tier-selection.md) | Edge vs Base vs Pro |
| [Debugging](docs/guides/debugging.md) | Logs, traces, common issues |

---

## FAQ

<details><summary><strong>Where is my data stored? Is anything sent to the cloud?</strong></summary>

Everything lives locally in `~/.truememory/memories.db`. Edge and Base tiers make zero external calls. Pro sends only your search query text to an LLM for query expansion. Your memories are never transmitted.
</details>

<details><summary><strong>Do I need Python installed?</strong></summary>

No. The installer uses [uv](https://docs.astral.sh/uv/) to manage a sandboxed Python 3.12. Your system Python is never touched.
</details>

<details><summary><strong>Why not just use a bigger context window?</strong></summary>

Context windows are expensive, slow, and empty at the start of every session. TrueMemory gives instant recall for zero tokens of context, in under 200ms.
</details>

<details><summary><strong>Does TrueMemory collect telemetry?</strong></summary>

Anonymous usage telemetry (tool calls, session counts, platform info) is on by default. We **never** track memory content, queries, file paths, or API keys. Opt out: `export TRUEMEMORY_TELEMETRY=off`
</details>

---

## Get Started in 60 Seconds

For Claude Code, Claude CLI, Cursor, Codex CLI, or Gemini CLI:

```bash
curl -LsSf https://raw.githubusercontent.com/buildingjoshbetter/TrueMemory/main/install.sh | sh
```

Using it as a Python library instead? `pip install truememory`

Questions? [Join our Discord](https://discord.gg/ZJ74JB2gVW) or [open a Discussion](https://github.com/buildingjoshbetter/TrueMemory/discussions). If TrueMemory saves you time, [give us a ⭐](https://github.com/buildingjoshbetter/TrueMemory)

---

## Thanks to Our Contributors

<table>
  <tr>
    <td align="center"><a href="https://github.com/buildingjoshbetter"><img src="https://avatars.githubusercontent.com/u/236675935?v=4" width="80" alt="buildingjoshbetter"/><br><sub>buildingjoshbetter</sub></a></td>
    <td align="center"><a href="https://github.com/SoilChang"><img src="https://avatars.githubusercontent.com/u/12407044?v=4" width="80" alt="SoilChang"/><br><sub>SoilChang</sub></a></td>
    <td align="center"><a href="https://github.com/Huntehhh"><img src="https://avatars.githubusercontent.com/u/66277108?v=4" width="80" alt="Huntehhh"/><br><sub>Huntehhh</sub></a></td>
    <td align="center"><a href="https://github.com/mseep-ai"><img src="https://avatars.githubusercontent.com/u/207640208?v=4" width="80" alt="mseep-ai"/><br><sub>mseep-ai</sub></a></td>
    <td align="center"><a href="https://github.com/adityajha2005"><img src="https://avatars.githubusercontent.com/u/147301021?v=4" width="80" alt="adityajha2005"/><br><sub>adityajha2005</sub></a></td>
    <td align="center"><a href="https://github.com/shivamverma1999"><img src="https://avatars.githubusercontent.com/u/33162868?v=4" width="80" alt="shivamverma1999"/><br><sub>shivamverma1999</sub></a></td>
    <td align="center"><a href="https://github.com/Adarsh-031"><img src="https://avatars.githubusercontent.com/u/185399508?v=4" width="80" alt="Adarsh-031"/><br><sub>Adarsh-031</sub></a></td>
    <td align="center"><a href="https://github.com/nareshkarthigeyan"><img src="https://avatars.githubusercontent.com/u/130460278?v=4" width="80" alt="nareshkarthigeyan"/><br><sub>nareshkarthigeyan</sub></a></td>
  </tr>
</table>

---

## Research

TrueMemory is backed by a research paper (arXiv preprint) on retrieval-centered agent memory.

**[Storage Is Not Memory: A Retrieval-Centered Architecture for Agent Recall](https://arxiv.org/abs/2605.04897)** (arXiv 2605.04897)

```bibtex
@article{sauronlabs2026storage,
  title   = {Storage Is Not Memory: A Retrieval-Centered Architecture for Agent Recall},
  author  = {Sauron Labs},
  journal = {arXiv preprint arXiv:2605.04897},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.04897}
}
```

---

## Community

- [Join our Discord](https://discord.gg/ZJ74JB2gVW) for help, feedback, and updates
- Follow [@Building_Josh](https://x.com/Building_Josh) on X for updates
- Follow [@Sauron_Labs](https://x.com/Sauron_Labs) for company news
- [Open a Discussion](https://github.com/buildingjoshbetter/TrueMemory/discussions) for questions or ideas
- Read the paper on [arXiv](https://arxiv.org/abs/2605.04897)
- Visit [truememory.net](https://truememory.net) · [sauronlabs.ai](https://sauronlabs.ai)

If TrueMemory saves you time, [give us a ⭐](https://github.com/buildingjoshbetter/TrueMemory)

---

## Translations

Read this in: [简体中文](./i18n/README.zh-CN.md) · [हिन्दी](./i18n/README.hi.md) · [Español](./i18n/README.es.md) · [Français](./i18n/README.fr.md) · [العربية](./i18n/README.ar.md) · [বাংলা](./i18n/README.bn.md) · [Português](./i18n/README.pt-BR.md) · [Русский](./i18n/README.ru.md) · [日本語](./i18n/README.ja.md) · [한국어](./i18n/README.ko.md) · [Deutsch](./i18n/README.de.md) · [Bahasa Indonesia](./i18n/README.id.md) · [Tiếng Việt](./i18n/README.vi.md) · [Türkçe](./i18n/README.tr.md) · [Italiano](./i18n/README.it.md) · [ไทย](./i18n/README.th.md) · [Polski](./i18n/README.pl.md) · [Українська](./i18n/README.uk.md) · [Nederlands](./i18n/README.nl.md)

---

## License

[AGPL-3.0](LICENSE). Free for personal and research use. Commercial use requires a separate license. Contact josh@sauronlabs.ai.

---

<p align="center">
  <em>TrueMemory, a <a href="https://sauronlabs.ai"><strong>sauron</strong></a> company</em>
</p>
