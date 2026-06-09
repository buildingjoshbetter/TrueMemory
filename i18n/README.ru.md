<p align="center">
  <img src="../assets/hero-banner-dark.svg" alt="TrueMemory" width="800">
</p>

<p align="center">
  <em>Память, которая должна была быть у вашего ИИ с самого начала.</em>
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
  <a href="#почему-truememory">Почему</a> · <a href="#быстрый-старт">Быстрый старт</a> · <a href="#сравнение-truememory">Сравнение</a> · <a href="#уровни">Уровни</a> · <a href="#бенчмарки">Бенчмарки</a> · <a href="#python-api">API</a> · <a href="#документация">Документация</a> · <a href="#часто-задаваемые-вопросы">FAQ</a>
</p>

<p align="center">
  <img src="../assets/terminal-demo.svg" alt="TrueMemory terminal demo" width="750">
</p>

---

## Почему TrueMemory

**Находит сигнал в шуме.** Ваш ИИ видит тысячи сообщений. TrueMemory определяет, какие из них действительно важны, и отбрасывает остальное. Без ручной разметки, без prompt-инженерии. Он просто знает.

**Становится точнее со временем.** Это не статическая база данных. Это живая память, которая растёт вместе с вами: разрешает противоречия, когда вы меняете мнение, обновляет устаревшие факты и консолидирует знания. Чем дольше вы используете его, тем лучше он работает.

**Работает без вашего участия.** TrueMemory автоматически фиксирует воспоминания из ваших разговоров и автоматически вставляет нужные в следующую сессию. Вам не нужно ничего сохранять или искать вручную. Всё происходит само.

**100% локально.** Один файл SQLite на вашей машине. Ничего не покидает ваше устройство. Без облака, без API-ключей. Ваши данные только ваши.

> **Без TrueMemory:** «Какой фреймворк мы используем?» Спрошено в 12-й раз за неделю. Ваш агент начинает каждую сессию с амнезии. Он не знает вашего имени, вашего стека и ничего из того, что вы сказали вчера.
>
> **С TrueMemory:** Ваш агент уже знает, что вы используете FastAPI, предпочитаете Pydantic v2, и что ваш auth middleware находится в `src/auth/`. Он помнит ваши исправления, ваши предпочтения и ваши решения. Через все сессии, навсегда.

---

## Сравнение TrueMemory

| Система | LoCoMo | LongMemEval | Локально | Автозахват | Лицензия |
|--------|--------|-------------|:-----------:|:------------:|---------|
| **TrueMemory Pro** | **93.0%** | **92.0%** | ✅ | ✅ | AGPL-3.0 |
| **TrueMemory Base** | **87.7%** | **84.1%** | ✅ | ✅ | AGPL-3.0 |
| Mem0 | 61.4% | — | Частично | ❌ | Apache-2.0 |
| Supermemory | 65.4% | — | ❌ | ❌ | Cloud API |
| MemOS | 75.8% | — | ✅ | ❌ | Apache-2.0 |
| ReadAgent | 79.5% | — | ❌ | ❌ | Research |

> Все бенчмарки воспроизводимы независимо. Скрипты включены в [`benchmarks/`](../benchmarks/).

---

## Быстрый старт

### Для Claude Code / Claude CLI / Cursor / Codex CLI / Gemini CLI

```bash
curl -LsSf https://raw.githubusercontent.com/buildingjoshbetter/TrueMemory/main/install.sh | sh
```

> Устанавливает всё в изолированном окружении. Скачивает ~1.5 ГБ моделей ИИ. Никакие данные не покидают вашу машину. sudo не требуется.

<details><summary>Впервые в терминале? Нажмите для пошаговой инструкции.</summary>

1. **Откройте терминал:** Mac: `Cmd + Space`, введите `Terminal`. Linux: `Ctrl + Alt + T`. Windows: откройте PowerShell.
2. **Вставьте команду выше** и нажмите Enter.
3. **Подождите 3-5 минут**, пока скачиваются модели.
4. **Полностью закройте ваш ИИ-инструмент** и откройте заново (Mac: `Cmd+Q`).
5. **Введите «Set up TrueMemory»** и выберите уровень.

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/buildingjoshbetter/TrueMemory/main/install.ps1 | iex
```

</details>

Готово. TrueMemory автоматически запоминает ваши разговоры с этого момента.

### Для разработчиков (библиотека Python)

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

## Уровни

Одна архитектура, три уровня. Все включены в одну установку. Переключайтесь в любой момент, сказав «switch to Pro» или «switch to Base».

| | Edge | Base | Pro |
|---|------|------|-----|
| **LoCoMo** | 89.6% | 92.0% | **93.0%** |
| **LongMemEval** | — | — | **92.0%** |
| **BEAM-1M** | — | — | **76.6% (SOTA)** |
| **Модель эмбеддинга** | 8 МБ облегчённая | 600 МБ высокоточная | 600 МБ высокоточная |
| **Реранкер** | 22M параметров | 149M параметров | 149M параметров |
| **Поиск HyDE** | — | — | ✅ (требуется API-ключ LLM) |
| **Работает на** | Любая машина, только CPU | 4 ГБ+ RAM | 4 ГБ+ RAM + API-ключ |

**Edge** работает везде. **Base** самый мощный полностью офлайн-уровень. **Pro** добавляет расширение запросов с помощью ИИ для наивысших результатов.

---

## Бенчмарки

<p align="center">
  <img src="../assets/charts/leaderboard-bar.png" alt="Benchmark Leaderboard" width="700" />
</p>

Протестировано на трёх крупных бенчмарках, где все системы используют одну и ту же модель ответов (GPT-4.1-mini), модель-судью (GPT-4o-mini, голосование большинством 3x) и пайплайн оценки.

| Бенчмарк | Что тестирует | TrueMemory Pro |
|-----------|--------------|:--------------:|
| [LoCoMo](https://github.com/snap-research/locomo) | 1 540 вопросов по 10 диалогам | **93.0%** |
| [LongMemEval](https://github.com/xiaowu0162/LongMemEval) | 500 мультисессионных вопросов | **92.0%** |
| [BEAM-1M](https://github.com/mohammadtavakoli78/BEAM) | 700 вопросов на 1M+ токенов | **76.6% (SOTA)** |
| BEAM-10M | 200 вопросов на 10M токенов | **65.0%** |

### Воспроизведите любой результат самостоятельно

Каждый скрипт бенчмарка самодостаточен и запускается на [Modal](https://modal.com).

- **[Скрипты LoCoMo](../benchmarks/locomo/scripts/)** — 8 систем (TrueMemory, Mem0, Zep, Engram и др.)
- **[Результаты LoCoMo](../benchmarks/locomo/BENCHMARK_RESULTS.md)** — разбивка по категориям, задержка, стоимость
- **[Конфиг оценки LoCoMo](../benchmarks/locomo/EVAL_CONFIG.md)** — точные модели, промпты, параметры
- **[Скрипты LongMemEval](../benchmarks/longmemeval/)** — варианты oracle + strict
- **[Результаты LongMemEval](../benchmarks/longmemeval/results/)** — 6 запусков TM Pro + 5 результатов конкурентов
- **[Скрипт BEAM-1M](../benchmarks/beam/bench_truememory_pro_beam1m.py)** — 35 диалогов на 1M+ токенов
- **[Скрипт BEAM-10M](../benchmarks/beam/bench_truememory_pro_beam10m.py)** — 10 диалогов на 10M токенов
- **[Результаты BEAM](../benchmarks/beam/)** — 3 запуска (1M) + 1 запуск (10M)

Все бенчмарки используют один и тот же пайплайн оценки. Ничего не скрыто. Полные детали: [LoCoMo](../benchmarks/locomo/EVAL_CONFIG.md) | [LongMemEval](../benchmarks/longmemeval/README.md) | [BEAM](../benchmarks/beam/README.md)

---

## Совместимость

<p align="center">
  <strong>Claude Code</strong> · <strong>Claude CLI</strong> · <strong>Cursor</strong> · <strong>Codex CLI</strong> · <strong>Gemini CLI</strong> · <strong>Claude Desktop</strong>
</p>

Хуки жизненного цикла автоматически фиксируют разговоры. Никакой ручной работы. Ваши воспоминания хранятся локально в одном файле SQLite.

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

| Метод | Описание |
|--------|-------------|
| `m.add(content, user_id)` | Сохранить воспоминание |
| `m.search(query, user_id)` | Поиск (6-уровневый пайплайн + реранкер) |
| `m.search_deep(query, user_id)` | Многораундовый агентный поиск |
| `m.get(id)` / `m.get_all(user_id)` | Получить воспоминания |
| `m.update(id, content)` / `m.delete(id)` | Изменить или удалить |
| `m.stats()` | Системная статистика |

[Полный справочник API →](../docs/python-api.md)

---

## Документация

| | |
|---|---|
| [Начало работы](../docs/guides/getting-started.md) | От установки до первого воспоминания |
| [Справочник Python API](../docs/python-api.md) | Полный справочник класса `Memory` |
| [Справочник инструментов MCP](../docs/mcp-tools.md) | Все 8 инструментов MCP |
| [Справочник CLI](../docs/cli.md) | `truememory-mcp` и `truememory-ingest` |
| [Переменные окружения](../docs/env-vars.md) | Все параметры конфигурации `TRUEMEMORY_*` |
| [Глубокий разбор архитектуры](../docs/architecture.md) | 6-уровневый пайплайн извлечения, gate кодирования |
| [Руководство по выбору уровня](../docs/guides/tier-selection.md) | Edge vs Base vs Pro |
| [Отладка](../docs/guides/debugging.md) | Логи, трассировки, типичные проблемы |

---

## Часто задаваемые вопросы

<details><summary><strong>Где хранятся мои данные? Что-нибудь отправляется в облако?</strong></summary>

Всё хранится локально в `~/.truememory/memories.db`. Уровни Edge и Base не делают никаких внешних вызовов. Pro отправляет только текст вашего поискового запроса в LLM для расширения запроса. Ваши воспоминания никогда не передаются.
</details>

<details><summary><strong>Нужен ли мне установленный Python?</strong></summary>

Нет. Установщик использует [uv](https://docs.astral.sh/uv/) для управления изолированным Python 3.12. Системный Python не затрагивается.
</details>

<details><summary><strong>Почему бы не использовать контекстное окно большего размера?</strong></summary>

Контекстные окна дорогие, медленные и пустые в начале каждой сессии. TrueMemory обеспечивает мгновенное извлечение при нулевых токенах контекста менее чем за 200 мс.
</details>

<details><summary><strong>Собирает ли TrueMemory телеметрию?</strong></summary>

Анонимная телеметрия использования (вызовы инструментов, количество сессий, информация о платформе) включена по умолчанию. Мы **никогда** не отслеживаем содержимое памяти, запросы, пути к файлам или API-ключи. Отключить: `export TRUEMEMORY_TELEMETRY=off`
</details>

---

## Начните за 60 секунд

```bash
pip install truememory
```

Есть вопросы? [Откройте обсуждение](https://github.com/buildingjoshbetter/TrueMemory/discussions). Если TrueMemory экономит ваше время, [поставьте нам ⭐](https://github.com/buildingjoshbetter/TrueMemory)

---

## Спасибо нашим контрибьюторам

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

## Исследование

TrueMemory подкреплён рецензированной научной статьёй о памяти агентов, основанной на поиске.

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

## Сообщество

- [Присоединяйтесь к нашему Discord](https://discord.gg/ZJ74JB2gVW) для помощи, обратной связи и обновлений
- Следите за обновлениями [@Building_Josh](https://x.com/Building_Josh) в X
- Новости компании [@Sauron_Labs](https://x.com/Sauron_Labs)
- [Откройте обсуждение](https://github.com/buildingjoshbetter/TrueMemory/discussions) для вопросов или идей
- Читайте статью на [arXiv](https://arxiv.org/abs/2605.04897) · [Google Scholar](https://scholar.google.com/citations?user=YOUR_ID) · [Semantic Scholar](https://www.semanticscholar.org/)
- Посетите [truememory.net](https://truememory.net) · [sauronlabs.ai](https://sauronlabs.ai)

Если TrueMemory экономит ваше время, [поставьте нам ⭐](https://github.com/buildingjoshbetter/TrueMemory)

---

## Лицензия

[AGPL-3.0](../LICENSE). Бесплатно для личного и исследовательского использования. Коммерческое использование требует отдельной лицензии. Пишите на josh@sauronlabs.ai.

---

<p align="center">
  <em>TrueMemory, компания <a href="https://sauronlabs.ai"><strong>sauron</strong></a></em>
</p>
