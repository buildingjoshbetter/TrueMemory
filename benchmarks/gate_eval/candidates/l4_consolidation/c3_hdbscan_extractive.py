"""C3 — HDBSCAN clustering + extractive summarization.

Uses existing truememory/clustering.py HDBSCAN over Model2Vec embeddings
(already writes message_clusters table) as the episode-boundary primitive,
then runs an extractive top-N summary per cluster instead of per
calendar month. Also retains contradiction detection from C1 for
apples-to-apples contradiction-probe comparison.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.gate_eval.candidates.l4_consolidation._base import (
    ConsolidateTelemetry, L4Candidate, RetrievalResult,
)


def _bytes(p):
    try:
        return Path(p).stat().st_size
    except OSError:
        return 0


class C3HDBSCANExtractive(L4Candidate):
    name = "c3_hdbscan_extractive"
    tier = "base"  # also runs on edge/pro

    def consolidate(self, db_path: str | Path) -> ConsolidateTelemetry:
        from truememory.clustering import cluster_messages
        from truememory.consolidation import (
            detect_contradictions,
            build_entity_summary_sheets,
            build_structured_facts,
            _message_salience,
            _extract_sentences,
            _score_sentence,
            _extract_numbers,
        )

        before = _bytes(db_path)
        conn = sqlite3.connect(str(db_path))
        # Load sqlite-vec so cluster_messages can read from vec_messages
        # virtual table (populated by the ingest phase).
        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception:
            pass
        try:
            t0 = time.time()

            # 1. HDBSCAN clustering over message embeddings.
            min_cluster_size = self.config.get("min_cluster_size", 10)
            min_samples = self.config.get("min_samples", 5)
            try:
                cluster_messages(conn, min_cluster_size=min_cluster_size,
                                 min_samples=min_samples)
                hdbscan_ok = True
            except Exception as exc:
                # Fall back gracefully — no message_clusters table / hdbscan not installed
                hdbscan_ok = False
                cluster_error = str(exc)[:100]

            # 2. Build extractive summaries per cluster (instead of per calendar month).
            conn.execute("DELETE FROM summaries")
            now = datetime.now(timezone.utc).isoformat()

            if hdbscan_ok:
                cluster_rows = conn.execute(
                    "SELECT message_id, cluster_id FROM message_clusters"
                ).fetchall()
                by_cluster: dict[int, list[int]] = defaultdict(list)
                for msg_id, cid in cluster_rows:
                    if cid is not None and cid >= 0:  # skip HDBSCAN noise (-1)
                        by_cluster[cid].append(msg_id)

            summary_count = 0

            if hdbscan_ok and by_cluster:
                for cid, msg_ids in by_cluster.items():
                    rows = conn.execute(
                        "SELECT id, content, sender, recipient, timestamp, category, modality "
                        "FROM messages WHERE id IN (" +
                        ",".join("?" * len(msg_ids)) + ") ORDER BY timestamp",
                        msg_ids,
                    ).fetchall()
                    if not rows:
                        continue

                    messages = [
                        {"id": r[0], "content": r[1], "sender": r[2],
                         "recipient": r[3], "timestamp": r[4],
                         "category": r[5], "modality": r[6]}
                        for r in rows
                    ]

                    # Salience-based extractive selection (reusing C1's scorer)
                    scored = [(m, _message_salience(m)) for m in messages]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    top_count = max(5, len(messages) // 4)
                    top = scored[:top_count]
                    top.sort(key=lambda x: x[0]["timestamp"])

                    # Sentence-level refinement
                    sentences = []
                    for m, sal in top:
                        for sent in _extract_sentences(m["content"]):
                            sscore = _score_sentence(sent) + sal * 0.3
                            sentences.append((f"[{m['sender']}] {sent}", sscore, m))
                    sentences.sort(key=lambda x: x[1], reverse=True)
                    sentences = sentences[:max(15, len(sentences) // 3)]
                    sentences.sort(key=lambda x: x[2]["timestamp"])
                    summary_text = "\n".join(s[0] for s in sentences)

                    key_facts = []
                    for m, _s in top:
                        nums = _extract_numbers(m["content"])
                        if nums:
                            key_facts.extend(nums[:3])

                    ts = [m["timestamp"] for m, _s in top if m["timestamp"]]
                    start_date = min(ts) if ts else ""
                    end_date = max(ts) if ts else ""

                    conn.execute(
                        "INSERT INTO summaries (period, start_date, end_date, entity, "
                        "summary, key_facts, message_ids, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        ("cluster_extractive", start_date, end_date, "",
                         summary_text, json.dumps(key_facts),
                         json.dumps([m["id"] for m, _s in top]), now),
                    )
                    summary_count += 1

            conn.commit()

            # 3. Keep contradiction + entity + structured-fact pipelines (apples-to-apples)
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
                "message_clusters": conn.execute(
                    "SELECT COUNT(*) FROM message_clusters"
                ).fetchone()[0] if hdbscan_ok else 0,
            }
            notes = "C3 — HDBSCAN clusters + extractive per cluster"
            if not hdbscan_ok:
                notes += f"; FALLBACK_NO_HDBSCAN: {cluster_error}"
        finally:
            conn.close()

        return ConsolidateTelemetry(
            wall_clock_s=wall,
            rows_written=rows,
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
