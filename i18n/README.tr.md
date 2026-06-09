<p align="center">
  <img src="../assets/hero-banner-dark.svg" alt="TrueMemory" width="800">
</p>

<p align="center">
  <em>Yapay zekanizin en basindan beri sahip olmasi gereken hafiza.</em>
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
  <a href="#neden-truememory">Neden</a> · <a href="#hizli-baslangic">Hizli Baslangic</a> · <a href="#truememory-nasil-karsilastirilir">Karsilastirma</a> · <a href="#katmanlar">Katmanlar</a> · <a href="#benchmark-sonuclari">Benchmarklar</a> · <a href="#python-api">API</a> · <a href="#belgeler">Belgeler</a> · <a href="#faq">FAQ</a>
</p>

<p align="center">
  <img src="../assets/terminal-demo.svg" alt="TrueMemory terminal demosu" width="750">
</p>

---

## Neden TrueMemory

**Gurultudeki sinyali bulur.** Yapay zekaniz binlerce mesaj gorur. TrueMemory hangilerinin gercekten onemli oldugunu belirler, ve gerisini atar. Manuel etiketleme yok, prompt muhendisligi yok. O sadece bilir.

**Zamanla daha keskin hale gelir.** Statik bir veritabani degildir. Sizinle birlikte buyuyen canli bir hafizadir. Fikrinizi degistirdiginizde celiskileri cozer, eski bilgileri gunceller ve bildiklerini pekistirir. Ne kadar uzun kullanirsiniz, o kadar iyi olur.

**Dusunmenize gerek kalmadan calisir.** TrueMemory konusmalarinizdan otomatik olarak anilari yakalar ve bir sonraki oturumunuza dogru olanlari otomatik olarak enjekte eder. Hicbir seyi manuel olarak kaydetmeniz veya aramaniz gerekmez. Her sey kendilginden olur.

**%100 yereldir.** Makinenizdeki tek bir SQLite dosyasi. Hicbir sey cihazinizdan cikmaz. Bulut yok, API anahtari gerekmez. Verileriniz sizindir.

> **TrueMemory olmadan:** "Hangi framework'u kullaniyoruz?" Bu hafta 12. kez soruluyor. Ajaniniz her oturuma amnezi ile baslar. Adinizi, stack'inizi veya dun soylediklerinizi bilmez.
>
> **TrueMemory ile:** Ajaniniz zaten FastAPI kullandiginizi, Pydantic v2 tercih ettiginizi ve auth middleware'inizin `src/auth/` icinde oldugunu bilir. Duzeltmelerinizi, tercihlerinizi ve kararlarinizi hatirlar. Her oturumda, sonsuza kadar.

---

## TrueMemory Nasil Karsilastirilir

| Sistem | LoCoMo | LongMemEval | Yerel-oncelikli | Otomatik yakalama | Lisans |
|--------|--------|-------------|:-----------:|:------------:|---------|
| **TrueMemory Pro** | **93.0%** | **92.0%** | ✅ | ✅ | AGPL-3.0 |
| **TrueMemory Base** | **87.7%** | **84.1%** | ✅ | ✅ | AGPL-3.0 |
| Mem0 | 61.4% | — | Kismi | ❌ | Apache-2.0 |
| Supermemory | 65.4% | — | ❌ | ❌ | Cloud API |
| MemOS | 75.8% | — | ✅ | ❌ | Apache-2.0 |
| ReadAgent | 79.5% | — | ❌ | ❌ | Research |

> Tum benchmarklar bagimsiz olarak tekrarlanabilir. Scriptler [`benchmarks/`](../benchmarks/) klasorunde yer almaktadir.

---

## Hizli Baslangic

### Claude Code / Claude CLI / Cursor / Codex CLI / Gemini CLI icin

```bash
curl -LsSf https://raw.githubusercontent.com/buildingjoshbetter/TrueMemory/main/install.sh | sh
```

> Her seyi izole bir ortamda kurar. ~1,5 GB AI modeli indirir. Hicbir veri makinenizden cikmaz. sudo gerekmez.

<details><summary>Terminal'e yeni misiniz? Adim adim talimatlar icin buraya tiklayin.</summary>

1. **Terminal acin:** Mac: `Cmd + Space`, `Terminal` yazin. Linux: `Ctrl + Alt + T`. Windows: PowerShell acin.
2. **Yukaridaki komutu yapistirin** ve Enter'a basin.
3. **Modellerin indirilmesi icin 3-5 dakika bekleyin.**
4. **AI aracinizi tamamen kapatin** ve yeniden acin (Mac: `Cmd+Q`).
5. **"Set up TrueMemory" yazin** ve bir katman secin.

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/buildingjoshbetter/TrueMemory/main/install.ps1 | iex
```

</details>

Hepsi bu. TrueMemory bundan sonra konusmalarinizi otomatik olarak hatirlar.

### Gelistiriciler icin (Python kutuphanesi)

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

## Katmanlar

Ayni mimari, uc katman. Tumu tek bir kuruluma dahildir. Istediginiz zaman "switch to Pro" veya "switch to Base" diyerek gecis yapin.

| | Edge | Base | Pro |
|---|------|------|-----|
| **LoCoMo** | 89.6% | 92.0% | **93.0%** |
| **LongMemEval** | — | — | **92.0%** |
| **BEAM-1M** | — | — | **76.6% (SOTA)** |
| **Embedding modeli** | 8 MB hafif | 600 MB yuksek dogruluk | 600 MB yuksek dogruluk |
| **Reranker** | 22M parametre | 149M parametre | 149M parametre |
| **HyDE arama** | — | — | ✅ (LLM API anahtari gerektirir) |
| **Calistigi ortam** | Her makine, yalnizca CPU | 4 GB+ RAM | 4 GB+ RAM + API anahtari |

**Edge** her yerde calisir. **Base** tamamen cevrimdisi calisan en guclu katmandir. **Pro** en yuksek puanlar icin AI destekli sorgu genisletme ekler.

---

## Benchmark Sonuclari

<p align="center">
  <img src="../assets/charts/leaderboard-bar.png" alt="Benchmark Siralamasi" width="700" />
</p>

Tum sistemlerin ayni yanit modelini (GPT-4.1-mini), yargici (GPT-4o-mini, 3x cogunluk oyu) ve puanlama hattini paylastigi uc buyuk benchmark uzerinde test edilmistir.

| Benchmark | Ne test eder | TrueMemory Pro |
|-----------|-------------|:--------------:|
| [LoCoMo](https://github.com/snap-research/locomo) | 10 konusmadan 1.540 soru | **93.0%** |
| [LongMemEval](https://github.com/xiaowu0162/LongMemEval) | 500 cok oturumlu soru | **92.0%** |
| [BEAM-1M](https://github.com/mohammadtavakoli78/BEAM) | 1M+ token'da 700 soru | **76.6% (SOTA)** |
| BEAM-10M | 10M token'da 200 soru | **65.0%** |

### Herhangi bir sonucu kendiniz yeniden uretin

Her benchmark scripti baglimsizdir ve [Modal](https://modal.com) uzerinde calisir.

- **[LoCoMo Scriptleri](../benchmarks/locomo/scripts/)** — 8 sistem (TrueMemory, Mem0, Zep, Engram, vb.)
- **[LoCoMo Sonuclari](../benchmarks/locomo/BENCHMARK_RESULTS.md)** — kategori bazinda dokumler, gecikme, maliyet
- **[LoCoMo Eval Yapilandirmasi](../benchmarks/locomo/EVAL_CONFIG.md)** — tam modeller, promptlar, parametreler
- **[LongMemEval Scriptleri](../benchmarks/longmemeval/)** — oracle + strict varyantlari
- **[LongMemEval Sonuclari](../benchmarks/longmemeval/results/)** — 6 TM Pro calismasi + 5 rakip sonucu
- **[BEAM-1M Scripti](../benchmarks/beam/bench_truememory_pro_beam1m.py)** — 1M+ token'da 35 konusma
- **[BEAM-10M Scripti](../benchmarks/beam/bench_truememory_pro_beam10m.py)** — 10M token'da 10 konusma
- **[BEAM Sonuclari](../benchmarks/beam/)** — 3 calisma (1M) + 1 calisma (10M)

Tum benchmarklar ayni degerlendirme hattini kullanir. Hicbir sey gizli degildir. Tam detaylar: [LoCoMo](../benchmarks/locomo/EVAL_CONFIG.md) | [LongMemEval](../benchmarks/longmemeval/README.md) | [BEAM](../benchmarks/beam/README.md)

---

## Uyumlu Olduklari

<p align="center">
  <strong>Claude Code</strong> · <strong>Claude CLI</strong> · <strong>Cursor</strong> · <strong>Codex CLI</strong> · <strong>Gemini CLI</strong> · <strong>Claude Desktop</strong>
</p>

Lifecycle hook'lari konusmalari otomatik olarak yakalar. Manuel islem gerekmez. Anilariniz tek bir SQLite dosyasinda yerel olarak kalir.

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

| Metot | Aciklama |
|-------|----------|
| `m.add(content, user_id)` | Bir ani kaydet |
| `m.search(query, user_id)` | Arama (6 katmanli hat + reranker) |
| `m.search_deep(query, user_id)` | Cok turlu ajantik arama |
| `m.get(id)` / `m.get_all(user_id)` | Anilari getir |
| `m.update(id, content)` / `m.delete(id)` | Degistir veya sil |
| `m.stats()` | Sistem istatistikleri |

[Tam API referansi →](../docs/python-api.md)

---

## Belgeler

| | |
|---|---|
| [Baslarken](../docs/guides/getting-started.md) | Kurulumdan ilk aniya |
| [Python API Referansi](../docs/python-api.md) | Tam `Memory` sinif referansi |
| [MCP Arac Referansi](../docs/mcp-tools.md) | 8 MCP araci |
| [CLI Referansi](../docs/cli.md) | `truememory-mcp` ve `truememory-ingest` |
| [Ortam Degiskenleri](../docs/env-vars.md) | Tum `TRUEMEMORY_*` yapilandirma secenekleri |
| [Mimari Derinlemesine](../docs/architecture.md) | 6 katmanli erisim hatti, encoding gate |
| [Katman Secim Kilavuzu](../docs/guides/tier-selection.md) | Edge vs Base vs Pro |
| [Hata Ayiklama](../docs/guides/debugging.md) | Loglar, izler, yaygin sorunlar |

---

## FAQ

<details><summary><strong>Verilerim nerede saklanir? Buluta bir sey gonderiliyor mu?</strong></summary>

Her sey yerel olarak `~/.truememory/memories.db` dosyasinda yer alir. Edge ve Base katmanlari hicbir harici cagri yapmaz. Pro yalnizca arama sorgu metninizi sorgu genisletme icin bir LLM'ye gonderir. Anilariniz asla iletilmez.
</details>

<details><summary><strong>Python kurulu olmasi gerekiyor mu?</strong></summary>

Hayir. Yukleyici, korumalanmis bir Python 3.12 ortamini yonetmek icin [uv](https://docs.astral.sh/uv/) kullanir. Sistem Python'unuz dokunulmaz.
</details>

<details><summary><strong>Neden sadece daha buyuk bir context window kullanmiyorsunuz?</strong></summary>

Context window'lar pahali, yavas ve her oturumun basinda bostur. TrueMemory sifir token baglam ile 200 ms'nin altinda aninda geri cagirma saglar.
</details>

<details><summary><strong>TrueMemory telemetri topluyor mu?</strong></summary>

Anonim kullanim telemetrisi (arac cagrilari, oturum sayilari, platform bilgisi) varsayilan olarak aciktir. Ani iceriklerini, sorguları, dosya yollarini veya API anahtarlarini **asla** takip etmeyiz. Devre disi birakin: `export TRUEMEMORY_TELEMETRY=off`
</details>

---

## 60 Saniyede Baslayin

```bash
pip install truememory
```

Sorulariniz mi var? [Bir Tartisma Acin](https://github.com/buildingjoshbetter/TrueMemory/discussions). TrueMemory size zaman kazandiriyorsa, [bize bir yildiz verin](https://github.com/buildingjoshbetter/TrueMemory)

---

## Katkilarcilarimiza Tesekkurler

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

## Arastirma

TrueMemory, erisim merkezli ajan hafizasi uzerine hakem degerlenirmesi gecmis bir arastirma makalesiyle desteklenmektedir.

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

## Topluluk

- [Discord'umuza katilın](https://discord.gg/ZJ74JB2gVW) yardim, geri bildirim ve guncellemeler icin
- Guncellemeler icin X'te [@Building_Josh](https://x.com/Building_Josh) takip edin
- Sirket haberleri icin [@Sauron_Labs](https://x.com/Sauron_Labs) takip edin
- Sorular veya fikirler icin [Bir Tartisma Acin](https://github.com/buildingjoshbetter/TrueMemory/discussions)
- Makaleyi okuyun: [arXiv](https://arxiv.org/abs/2605.04897) · [Google Scholar](https://scholar.google.com/citations?user=YOUR_ID) · [Semantic Scholar](https://www.semanticscholar.org/)
- Ziyaret edin: [truememory.net](https://truememory.net) · [sauronlabs.ai](https://sauronlabs.ai)

TrueMemory size zaman kazandiriyorsa, [bize bir yildiz verin](https://github.com/buildingjoshbetter/TrueMemory)

---

## Lisans

[AGPL-3.0](../LICENSE). Kisisel ve arastirma kullanimi icin ucretsiz. Ticari kullanim ayri bir lisans gerektirir. josh@sauronlabs.ai ile iletisime gecin.

---

<p align="center">
  <em>TrueMemory, bir <a href="https://sauronlabs.ai"><strong>sauron</strong></a> sirketi</em>
</p>
