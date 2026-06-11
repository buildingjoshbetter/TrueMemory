"""C4 — HDBSCAN clusters + LLM-abstractive per cluster (Pro).

Combines C3 (HDBSCAN episode bucketing) and C2 (LLM-abstractive
summaries). For each cluster produced by truememory/clustering.py,
call gpt-4.1-mini to produce a 3–5 bullet factual summary.

This is the closest in-tree analog to EverMemOS's scene-clustering
+ per-scene-summary architecture (Phase 3 Card 8).

Falls back to C3 extractive if OPENAI_API_KEY unset.
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
SUMMARY_TEMPERATURE = 0.7
SUMMARY_MAX_TOKENS = 300

SUMMARY_PROMPT = """You will receive up to 30 messages from one topical conversation cluster.
Produce a compact factual summary in 3-5 bullet points that preserves:
- Entity names (people, companies, products)
- Dates, numbers, amounts
- Decisions, changes, events
- Preferences / stated facts

Do NOT invent facts not in the source. Do NOT add commentary.

Messages (same topic cluster):
{messages}

Cluster summary (3-5 bullets):"""


def _bytes(p):
    try:
        return Path(p).stat().st_size
    except OSError:
        return 0


class C4HDBSCANAbstractive(L4Candidate):
    name = "c4_hdbscan_abstractive"
    tier = "pro"

    def _openai(self):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            return None
        try:
            from openai import OpenAI
            return OpenAI(api_key=key), self.config.get("seed", 42)
        except ImportError:
            return None

    def _summarize(self, client_seed, messages):
        client, seed = client_seed
        txt = "\n".join(
            f"[{m.get('sender', '?')}] {m['content'][:300]}" for m in messages[:30]
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
        from truememory.clustering import cluster_messages
        from truememory.consolidation import (
            detect_contradictions,
            build_entity_summary_sheets,
            build_structured_facts,
        )
        from benchmarks.gate_eval.candidates.l4_consolidation.c3_hdbscan_extractive import (
            C3HDBSCANExtractive,
        )

        before = _bytes(db_path)
        client_seed = self._openai()

        if client_seed is None:
            # Fall back entirely to C3 extractive
            tel = C3HDBSCANExtractive().consolidate(db_path)
            tel.notes = "C4 FALLBACK — OPENAI_API_KEY unset; ran C3 extractive path"
            return tel

        conn = sqlite3.connect(str(db_path))
        # Load sqlite-vec so cluster_messages can access vec_messages
        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception:
            pass
        try:
            t0 = time.time()
            try:
                cluster_messages(conn)
                hdbscan_ok = True
            except Exception as exc:
                hdbscan_ok = False
                notes_suffix = f"; HDBSCAN_FAIL: {exc}"

            conn.execute("DELETE FROM summaries")
            now = datetime.now(timezone.utc).isoformat()
            llm_calls = 0

            if hdbscan_ok:
                rows = conn.execute(
                    "SELECT message_id, cluster_id FROM message_clusters"
                ).fetchall()
                by_cluster: dict[int, list[int]] = defaultdict(list)
                for mid, cid in rows:
                    if cid is not None and cid >= 0:
                        by_cluster[cid].append(mid)

                for cid, mids in by_cluster.items():
                    msg_rows = conn.execute(
                        "SELECT id, content, sender, recipient, timestamp "
                        "FROM messages WHERE id IN (" +
                        ",".join("?" * len(mids)) + ") ORDER BY timestamp",
                        mids,
                    ).fetchall()
                    msgs = [
                        {"id": r[0], "content": r[1], "sender": r[2],
                         "recipient": r[3], "timestamp": r[4]}
                        for r in msg_rows
                    ]
                    if len(msgs) < 3:
                        # Tiny cluster — skip LLM call, fall back to raw concat
                        summary_text = "\n".join(
                            f"[{m['sender']}] {m['content'][:200]}" for m in msgs
                        )
                    else:
                        summary_text = self._summarize(client_seed, msgs)
                        llm_calls += 1
                    ts = [m["timestamp"] for m in msgs if m["timestamp"]]
                    conn.execute(
                        "INSERT INTO summaries (period, start_date, end_date, entity, "
                        "summary, key_facts, message_ids, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        ("cluster_abstractive",
                         min(ts) if ts else "", max(ts) if ts else "",
                         "", summary_text, "[]",
                         json.dumps([m["id"] for m in msgs]), now),
                    )
                conn.commit()

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
            notes = (f"C4 — HDBSCAN + LLM-abstractive per cluster "
                     f"({SUMMARY_MODEL} T={SUMMARY_TEMPERATURE}); {llm_calls} LLM calls")
            if not hdbscan_ok:
                notes += notes_suffix
        finally:
            conn.close()

        return ConsolidateTelemetry(
            wall_clock_s=wall,
            rows_written=rows_out,
            bytes_added=_bytes(db_path) - before,
            notes=notes,
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
