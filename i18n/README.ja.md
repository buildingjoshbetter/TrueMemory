<p align="center">
  <img src="../assets/hero-banner-dark.svg" alt="TrueMemory" width="800">
</p>

<p align="center">
  <em>あなたのAIが最初から持っているべきだった記憶。</em>
</p>

<p align="center">
  <a href="../README.md"><img alt="English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="./README.zh-CN.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
  <a href="./README.hi.md"><img alt="हिन्दी" src="https://img.shields.io/badge/हिन्दी-d9d9d9"></a>
  <a href="./README.es.md"><img alt="Español" src="https://img.shields.io/badge/Español-d9d9d9"></a>
  <a href="./README.fr.md"><img alt="Français" src="https://img.shields.io/badge/Français-d9d9d9"></a>
  <a href="./README.ar.md"><img alt="العربية" src="https://img.shields.io/badge/العربية-d9d9d9"></a>
  <a href="./README.bn.md"><img alt="বাংলা" src="https://img.shields.io/badge/বাংলা-d9d9d9"></a>
  <a href="./README.pt-BR.md"><img alt="Português" src="https://img.shields.io/badge/Português-d9d9d9"></a>
  <a href="./README.ru.md"><img alt="Русский" src="https://img.shields.io/badge/Русский-d9d9d9"></a>
  <a href="./README.ja.md"><img alt="日本語" src="https://img.shields.io/badge/日本語-d9d9d9"></a>
  <a href="./README.ko.md"><img alt="한국어" src="https://img.shields.io/badge/한국어-d9d9d9"></a>
  <a href="./README.de.md"><img alt="Deutsch" src="https://img.shields.io/badge/Deutsch-d9d9d9"></a>
  <a href="./README.id.md"><img alt="Bahasa Indonesia" src="https://img.shields.io/badge/Indonesia-d9d9d9"></a>
  <a href="./README.vi.md"><img alt="Tiếng Việt" src="https://img.shields.io/badge/Tiếng Việt-d9d9d9"></a>
  <a href="./README.tr.md"><img alt="Türkçe" src="https://img.shields.io/badge/Türkçe-d9d9d9"></a>
  <a href="./README.it.md"><img alt="Italiano" src="https://img.shields.io/badge/Italiano-d9d9d9"></a>
  <a href="./README.th.md"><img alt="ไทย" src="https://img.shields.io/badge/ไทย-d9d9d9"></a>
  <a href="./README.pl.md"><img alt="Polski" src="https://img.shields.io/badge/Polski-d9d9d9"></a>
  <a href="./README.uk.md"><img alt="Українська" src="https://img.shields.io/badge/Українська-d9d9d9"></a>
  <a href="./README.nl.md"><img alt="Nederlands" src="https://img.shields.io/badge/Nederlands-d9d9d9"></a>
</p>

<p align="center">
  <a href="https://github.com/buildingjoshbetter/TrueMemory"><img src="https://img.shields.io/github/stars/buildingjoshbetter/TrueMemory?style=social" alt="Stars"></a>
  <a href="https://pypi.org/project/truememory/"><img src="https://img.shields.io/badge/installs-7.3k+-brightgreen" alt="Installs"></a>
  <a href="https://arxiv.org/abs/2605.04897"><img src="https://img.shields.io/badge/arXiv-2605.04897-b31b1b" alt="arXiv"></a>
  <a href="https://discord.gg/ZJ74JB2gVW"><img src="https://img.shields.io/badge/Discord-Join%20us-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/LoCoMo-93.0%25-blueviolet" alt="LoCoMo">
  <img src="https://img.shields.io/badge/LongMemEval-92.0%25-blue" alt="LongMemEval">
  <img src="https://img.shields.io/badge/BEAM--1M-76.6%25_(SOTA)-orange" alt="BEAM-1M">
</p>

<p align="center">
  <a href="#なぜ-truememory-なのか">なぜ</a> · <a href="#クイックスタート">クイックスタート</a> · <a href="#truememory-の比較">比較</a> · <a href="#ティア">ティア</a> · <a href="#ベンチマーク">ベンチマーク</a> · <a href="#python-api">API</a> · <a href="#ドキュメント">ドキュメント</a> · <a href="#よくある質問">FAQ</a>
</p>

<p align="center">
  <img src="../assets/terminal-demo.svg" alt="TrueMemory terminal demo" width="750">
</p>

---

## なぜ TrueMemory なのか

**ノイズの中からシグナルを見つけ出します。** あなたのAIは何千ものメッセージを目にします。TrueMemoryはどれが本当に重要かを判断し、残りを捨てます。手動のタグ付けもプロンプトエンジニアリングも不要。ただ分かるのです。

**時間とともに精度が上がります。** 静的なデータベースではありません。あなたと共に成長する生きた記憶です。考えが変わったら矛盾を解消し、古くなった事実を更新し、知識を統合します。使えば使うほど、より良くなります。

**意識せずに動きます。** TrueMemoryは会話から自動的に記憶をキャプチャし、次のセッションに適切な記憶を自動で注入します。手動で何かを保存したり検索したりする必要はありません。すべて自動で行われます。

**100%ローカルです。** マシン上の1つのSQLiteファイル。デバイスから何も外に出ません。クラウドなし、APIキー不要。データはあなただけのものです。

> **TrueMemoryなしの場合:** 「どのフレームワークを使っていますか？」今週12回目の質問。エージェントは毎回記憶喪失の状態でセッションを開始します。あなたの名前も、スタックも、昨日言ったことも何も知りません。
>
> **TrueMemoryありの場合:** エージェントはあなたがFastAPIを使い、Pydantic v2を好み、authミドルウェアが`src/auth/`にあることを既に知っています。あなたの修正、好み、決定をすべてのセッションを通じて永遠に記憶しています。

---

## TrueMemory の比較

| システム | LoCoMo | LongMemEval | ローカル優先 | 自動キャプチャ | ライセンス |
|--------|--------|-------------|:-----------:|:------------:|---------|
| **TrueMemory Pro** | **93.0%** | **92.0%** | ✅ | ✅ | AGPL-3.0 |
| **TrueMemory Base** | **87.7%** | **84.1%** | ✅ | ✅ | AGPL-3.0 |
| Mem0 | 61.4% | — | 部分的 | ❌ | Apache-2.0 |
| Supermemory | 65.4% | — | ❌ | ❌ | Cloud API |
| MemOS | 75.8% | — | ✅ | ❌ | Apache-2.0 |
| ReadAgent | 79.5% | — | ❌ | ❌ | Research |

> すべてのベンチマークは独立して再現可能です。スクリプトは[`benchmarks/`](../benchmarks/)に含まれています。

---

## クイックスタート

### Claude Code / Claude CLI / Cursor / Codex CLI / Gemini CLI 向け

```bash
curl -LsSf https://raw.githubusercontent.com/buildingjoshbetter/TrueMemory/main/install.sh | sh
```

> 隔離された環境にすべてをインストールします。約1.5GBのAIモデルをダウンロードします。データはマシンから一切外部に送信されません。sudoは不要です。

<details><summary>ターミナルが初めての方はこちらをクリックしてください。</summary>

1. **ターミナルを開く：** Mac: `Cmd + Space`で`Terminal`と入力。Linux: `Ctrl + Alt + T`。Windows: PowerShellを開く。
2. **上記のコマンドを貼り付けて** Enterを押す。
3. **モデルのダウンロードに3〜5分待つ。**
4. **AIツールを完全に終了して** 再度開く（Mac: `Cmd+Q`）。
5. **「Set up TrueMemory」と入力して** ティアを選択。

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/buildingjoshbetter/TrueMemory/main/install.ps1 | iex
```

</details>

以上です。これ以降、TrueMemoryは会話を自動的に記憶します。

### 開発者向け（Pythonライブラリ）

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

## ティア

同じアーキテクチャ、3つのティア。すべて1回のインストールに含まれています。「switch to Pro」や「switch to Base」と言えばいつでも切り替えられます。

| | Edge | Base | Pro |
|---|------|------|-----|
| **LoCoMo** | 89.6% | 92.0% | **93.0%** |
| **LongMemEval** | — | — | **92.0%** |
| **BEAM-1M** | — | — | **76.6% (SOTA)** |
| **エンベディングモデル** | 8 MB 軽量 | 600 MB 高精度 | 600 MB 高精度 |
| **リランカー** | 22Mパラメータ | 149Mパラメータ | 149Mパラメータ |
| **HyDE検索** | — | — | ✅（LLM APIキーが必要） |
| **動作環境** | あらゆるマシン、CPUのみ | 4 GB以上のRAM | 4 GB以上のRAM + APIキー |

**Edge**はどこでも動きます。**Base**は完全オフラインで最も強力なティアです。**Pro**はAIによるクエリ拡張を追加し、最高スコアを実現します。

---

## ベンチマーク

<p align="center">
  <img src="../assets/charts/leaderboard-bar.png" alt="Benchmark Leaderboard" width="700" />
</p>

3つの主要ベンチマークでテスト済み。すべてのシステムが同じ回答モデル（GPT-4.1-mini）、ジャッジ（GPT-4o-mini、3回多数決）、スコアリングパイプラインを共有しています。

| ベンチマーク | テスト内容 | TrueMemory Pro |
|-----------|--------------|:--------------:|
| [LoCoMo](https://github.com/snap-research/locomo) | 10の会話にわたる1,540問 | **93.0%** |
| [LongMemEval](https://github.com/xiaowu0162/LongMemEval) | 500のマルチセッション問題 | **92.0%** |
| [BEAM-1M](https://github.com/mohammadtavakoli78/BEAM) | 1M+トークンでの700問 | **76.6% (SOTA)** |
| BEAM-10M | 10Mトークンでの200問 | **65.0%** |

### どの結果も自分で再現できます

すべてのベンチマークスクリプトは自己完結型で、[Modal](https://modal.com)上で実行できます。

- **[LoCoMoスクリプト](../benchmarks/locomo/scripts/)** — 8システム（TrueMemory、Mem0、Zep、Engramなど）
- **[LoCoMo結果](../benchmarks/locomo/BENCHMARK_RESULTS.md)** — カテゴリ別の内訳、レイテンシ、コスト
- **[LoCoMo評価設定](../benchmarks/locomo/EVAL_CONFIG.md)** — 正確なモデル、プロンプト、パラメータ
- **[LongMemEvalスクリプト](../benchmarks/longmemeval/)** — oracle + strictバリアント
- **[LongMemEval結果](../benchmarks/longmemeval/results/)** — TM Pro 6回 + 競合5システムの結果
- **[BEAM-1Mスクリプト](../benchmarks/beam/bench_truememory_pro_beam1m.py)** — 1M+トークンで35の会話
- **[BEAM-10Mスクリプト](../benchmarks/beam/bench_truememory_pro_beam10m.py)** — 10Mトークンで10の会話
- **[BEAM結果](../benchmarks/beam/)** — 3回の実行（1M）+ 1回の実行（10M）

すべてのベンチマークは同じ評価パイプラインを使用しています。何も隠されていません。詳細: [LoCoMo](../benchmarks/locomo/EVAL_CONFIG.md) | [LongMemEval](../benchmarks/longmemeval/README.md) | [BEAM](../benchmarks/beam/README.md)

---

## 対応ツール

<p align="center">
  <strong>Claude Code</strong> · <strong>Claude CLI</strong> · <strong>Cursor</strong> · <strong>Codex CLI</strong> · <strong>Gemini CLI</strong> · <strong>Claude Desktop</strong>
</p>

ライフサイクルフックが会話を自動的にキャプチャします。手動作業は不要です。メモリは1つのSQLiteファイルにローカルで保持されます。

---

## Python API

```python
from truememory import Memory

m = Memory()

m.add("Prefers dark mode and TypeScript", user_id="alex")
m.add("Works at Anthropic as a senior engineer", user_id="alex")

results = m.search("What are Alex's preferences?", user_id="alex")
results = m.search_deep("career history?", user_id="alex")  # multi-round, higher accuracy
```

| メソッド | 説明 |
|--------|-------------|
| `m.add(content, user_id)` | メモリを保存する |
| `m.search(query, user_id)` | 検索（6層パイプライン + リランカー） |
| `m.search_deep(query, user_id)` | マルチラウンドのエージェント型検索 |
| `m.get(id)` / `m.get_all(user_id)` | メモリを取得する |
| `m.update(id, content)` / `m.delete(id)` | 変更または削除 |
| `m.stats()` | システム統計 |

[完全なAPIリファレンス →](../docs/python-api.md)

---

## ドキュメント

| | |
|---|---|
| [はじめに](../docs/guides/getting-started.md) | インストールから最初のメモリまで |
| [Python APIリファレンス](../docs/python-api.md) | `Memory`クラスの完全なリファレンス |
| [MCPツールリファレンス](../docs/mcp-tools.md) | 全8つのMCPツール |
| [CLIリファレンス](../docs/cli.md) | `truememory-mcp`と`truememory-ingest` |
| [環境変数](../docs/env-vars.md) | すべての`TRUEMEMORY_*`設定オプション |
| [アーキテクチャ詳解](../docs/architecture.md) | 6層検索パイプライン、エンコーディングゲート |
| [ティア選択ガイド](../docs/guides/tier-selection.md) | Edge vs Base vs Pro |
| [デバッグ](../docs/guides/debugging.md) | ログ、トレース、よくある問題 |

---

## よくある質問

<details><summary><strong>データはどこに保存されますか？クラウドに送信されますか？</strong></summary>

すべてローカルの`~/.truememory/memories.db`に保存されます。EdgeとBaseティアは外部への通信を一切行いません。Proはクエリ拡張のためにLLMに検索クエリテキストのみを送信します。メモリの内容が送信されることはありません。
</details>

<details><summary><strong>Pythonのインストールは必要ですか？</strong></summary>

いいえ。インストーラーは[uv](https://docs.astral.sh/uv/)を使用してサンドボックス化されたPython 3.12を管理します。システムのPythonには一切触れません。
</details>

<details><summary><strong>より大きなコンテキストウィンドウを使えばよいのでは？</strong></summary>

コンテキストウィンドウは高コストで、低速で、毎セッションの開始時に空です。TrueMemoryはゼロトークンのコンテキストで200ミリ秒未満の即時リコールを提供します。
</details>

<details><summary><strong>TrueMemoryはテレメトリを収集しますか？</strong></summary>

匿名の使用状況テレメトリ（ツール呼び出し、セッション数、プラットフォーム情報）がデフォルトで有効です。メモリの内容、クエリ、ファイルパス、APIキーは**一切**追跡しません。オプトアウト: `export TRUEMEMORY_TELEMETRY=off`
</details>

---

## 60秒で始めましょう

```bash
pip install truememory
```

質問がありますか？[ディスカッションを開く](https://github.com/buildingjoshbetter/TrueMemory/discussions)。TrueMemoryが時間の節約になったら、[スターをお願いします ⭐](https://github.com/buildingjoshbetter/TrueMemory)

---

## コントリビューターの皆さんに感謝

<table>
  <tr>
    <td align="center"><a href="https://github.com/buildingjoshbetter"><img src="https://avatars.githubusercontent.com/u/236675935?v=4" width="80" alt="buildingjoshbetter"/><br><sub>buildingjoshbetter</sub></a></td>
    <td align="center"><a href="https://github.com/SoilChang"><img src="https://avatars.githubusercontent.com/u/12407044?v=4" width="80" alt="SoilChang"/><br><sub>SoilChang</sub></a></td>
    <td align="center"><a href="https://github.com/Huntehhh"><img src="https://avatars.githubusercontent.com/u/66277108?v=4" width="80" alt="Huntehhh"/><br><sub>Huntehhh</sub></a></td>
    <td align="center"><a href="https://github.com/mseep-ai"><img src="https://avatars.githubusercontent.com/u/207640208?v=4" width="80" alt="mseep-ai"/><br><sub>mseep-ai</sub></a></td>
    <td align="center"><a href="https://github.com/adityajha2005"><img src="https://avatars.githubusercontent.com/u/147301021?v=4" width="80" alt="adityajha2005"/><br><sub>adityajha2005</sub></a></td>
    <td align="center"><a href="https://github.com/shivamverma1999"><img src="https://avatars.githubusercontent.com/u/33162868?v=4" width="80" alt="shivamverma1999"/><br><sub>shivamverma1999</sub></a></td>
  </tr>
</table>

---

## 研究

TrueMemoryは、検索中心のエージェントメモリに関する査読済みの研究論文に裏付けられています。

**[Storage Is Not Memory: A Retrieval-Centered Architecture for Agent Recall](https://arxiv.org/abs/2605.04897)** — arXiv 2605.04897

```bibtex
@article{adler2025storage,
  title   = {Storage Is Not Memory: A Retrieval-Centered Architecture for Agent Recall},
  author  = {Sauron Labs},
  journal = {arXiv preprint arXiv:2605.04897},
  year    = {2025},
  url     = {https://arxiv.org/abs/2605.04897}
}
```

---

## コミュニティ

- [Discordに参加する](https://discord.gg/ZJ74JB2gVW) ヘルプ、フィードバック、最新情報はこちら
- 更新情報はXで[@Building_Josh](https://x.com/Building_Josh)をフォロー
- 企業ニュースは[@Sauron_Labs](https://x.com/Sauron_Labs)をフォロー
- 質問やアイデアは[ディスカッションを開く](https://github.com/buildingjoshbetter/TrueMemory/discussions)
- 論文を読む: [arXiv](https://arxiv.org/abs/2605.04897) · [Google Scholar](https://scholar.google.com/citations?user=YOUR_ID) · [Semantic Scholar](https://www.semanticscholar.org/)
- [truememory.net](https://truememory.net) · [sauronlabs.ai](https://sauronlabs.ai)

TrueMemoryが時間の節約になったら、[スターをお願いします ⭐](https://github.com/buildingjoshbetter/TrueMemory)

---

## ライセンス

[AGPL-3.0](../LICENSE)。個人利用および研究利用は無料です。商用利用には別途ライセンスが必要です。josh@sauronlabs.aiまでご連絡ください。

---

<p align="center">
  <em>TrueMemory、<a href="https://sauronlabs.ai"><strong>sauron</strong></a>のプロダクト</em>
</p>
