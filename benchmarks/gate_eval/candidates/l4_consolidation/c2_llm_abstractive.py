"""C2 — LLM-abstractive summaries (Pro).

Replaces the extractive top-N summary selection in build_summaries
with a single `gpt-4.1-mini` call per calendar-month bucket producing
3–5 bullet-point abstractive summary. Uses OpenAI's synchronous API
by default (Batch API would be 50% cheaper but 24hr async — not
suited to iterative sweep).

Falls back to C1 extractive if OPENAI_API_KEY unset.

Retains C1's contradiction detection for apples-to-apples comparison
(isolates the summary-generation change).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.gate_eval.candidates.l4_consolidation._base import (
    ConsolidateTelemetry, L4Candidate, RetrievalResult,
)


SUMMARY_MODEL = "gpt-4.1-mini"
SUMMARY_TEMPERATURE = 0.7  # per L4_COUPLING_CONTRACT §4 variance source
SUMMARY_MAX_TOKENS = 300


SUMMARY_PROMPT = """You will receive up to 30 messages from a conversation bucket.
Produce a compact factual summary in 3-5 bullet points that preserves:
- Entity names (people, companies, products)
- Dates, numbers, amounts
- Decisions, changes, events
- Preferences / stated facts

Do NOT invent facts not in the source. Do NOT add commentary.

Messages:
{messages}

Summary (3-5 bullets):"""


def _bytes(p):
    try:
        return Path(p).stat().st_size
    except OSError:
        return 0


def _extract_month(ts: str) -> str:
    if not ts or len(ts) < 7:
        return "unknown"
    return ts[:7]


class C2LLMAbstractive(L4Candidate):
    name = "c2_llm_abstractive"
    tier = "pro"

    def _openai_client(self):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            return None
        try:
            from openai import OpenAI
            seed = self.config.get("seed", 42)
            return OpenAI(api_key=key), seed
        except ImportError:
            return None

    def _summarize(self, client_seed_tuple, messages: list[dict]) -> str:
        client, seed = client_seed_tuple
        txt = "\n".join(
            f"[{m.get('sender', '?')}] {m['content'][:300]}"
            for m in messages[:30]
        )
        try:
            resp = client.chat.completions.create(
                model=SUMMARY_MODEL,
                temperature=SUMMARY_TEMPERATURE,
                max_tokens=SUMMARY_MAX_TOKENS,
                seed=seed,
                messages=[{"role": "user",
                           "content": SUMMARY_PROMPT.format(messages=txt)}],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            return f"[SUMMARY_ERROR: {exc}]"

    def consolidate(self, db_path: str | Path) -> ConsolidateTelemetry:
        from truememory.consolidation import (
            detect_contradictions,
            build_summaries,
            build_entity_summary_sheets,
            build_structured_facts,
        )

        before = _bytes(db_path)
        conn = sqlite3.connect(str(db_path))
        ok = self._openai_client()

        try:
            t0 = time.time()

            if ok is None:
                # Fallback: run baseline C1 extractive
                build_summaries(conn)
                detect_contradictions(conn)
                try:
                    build_entity_summary_sheets(conn)
                    build_structured_facts(conn)
                except Exception:
                    pass
                wall = time.time() - t0
                rows = {
                    "summaries": conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0],
                    "fact_timeline": conn.execute("SELECT COUNT(*) FROM fact_timeline").fetchone()[0],
                }
                return ConsolidateTelemetry(
                    wall_clock_s=wall,
                    rows_written=rows,
                    bytes_added=_bytes(db_path) - before,
                    notes="C2 FALLBACK — OPENAI_API_KEY unset; ran C1 extractive pipeline",
                )

            # Primary path: LLM-abstractive per calendar month.
            conn.execute("DELETE FROM summaries")
            now = datetime.now(timezone.utc).isoformat()

            rows = conn.execute(
                "SELECT id, content, sender, recipient, timestamp "
                "FROM messages ORDER BY timestamp"
            ).fetchall()
            msgs = [
                {"id": r[0], "content": r[1], "sender": r[2],
                 "recipient": r[3], "timestamp": r[4]}
                for r in rows
            ]

            by_month: dict[str, list[dict]] = defaultdict(list)
            for m in msgs:
                by_month[_extract_month(m["timestamp"])].append(m)

            llm_calls = 0
            for month, bucket in sorted(by_month.items()):
                if month == "unknown" or not bucket:
                    continue
                summary_text = self._summarize(ok, bucket)
                llm_calls += 1
                ts = [m["timestamp"] for m in bucket if m["timestamp"]]
                conn.execute(
                    "INSERT INTO summaries (period, start_date, end_date, entity, "
                    "summary, key_facts, message_ids, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    ("llm_monthly", min(ts) if ts else "", max(ts) if ts else "",
                     "", summary_text, "[]",
                     json.dumps([m["id"] for m in bucket]), now),
                )
            conn.commit()

            # Contradictions + entity + structured facts unchanged (apples-to-apples)
            detect_contradictions(conn)
            try:
                build_entity_summary_sheets(conn)
                build_structured_facts(conn)
            except Exception:
                pass

            wall = time.time() - t0
            rows_out = {
                "summaries": conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0],
                "fact_timeline": conn.execute("SELECT COUNT(*) FROM fact_timeline").fetchone()[0],
                "llm_calls": llm_calls,
            }
        finally:
            conn.close()

        return ConsolidateTelemetry(
            wall_clock_s=wall,
            rows_written=rows_out,
            bytes_added=_bytes(db_path) - before,
            notes=f"C2 — LLM-abstractive ({SUMMARY_MODEL} T={SUMMARY_TEMPERATURE}); "
                  f"{llm_calls} LLM calls",
        )

    def retrieve_augmented(self, query: str, db_path: str | Path,
                           k: int = 10) -> list[RetrievalResult]:
        from truememory.consolidation import search_consolidated
        from truememory.fts_search import search_fts

        conn = sqlite3.connect(str(db_path))
        results: list[RetrievalResult] = []
        try:
            for r in search_consolidated(conn, query, limit=k):
                results.append(RetrievalResult(
                    content=r.get("content", ""),
                    score=float(r.get("score", 0.0)),
                    source=r.get("source", "summary"),
                    metadata={"period": r.get("period"), "entity": r.get("entity", "")},
                ))
            need = max(0, k - len(results))
            if need > 0:
                for r in search_fts(conn, query, limit=need):
                    results.append(RetrievalResult(
                        content=r.get("content", ""),
                        score=float(r.get("score", 0.0)),
                        source="messages",
                        metadata={"sender": r.get("sender", "")},
                    ))
        finally:
            conn.close()
        return results[:k]
