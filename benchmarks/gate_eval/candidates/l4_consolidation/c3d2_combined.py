"""C3D2 — HDBSCAN-tight clustering + disable contradiction detection.

Phase 10 auxiliary candidate: combines the Pareto-dominant move from
Phase 9 (D2: disable regex contradiction detection) with the
granular-cluster bucketing of C3-tight. Tests whether the two
improvements compose.

Retention:
  - HDBSCAN min_cluster_size=3, min_samples=2 (C3-tight params)
  - extractive summary per cluster
  - NO regex contradiction detection (per D2)
  - KEEP build_entity_summary_sheets + build_structured_facts
    (these are separate L4 outputs, not part of the retired contradiction path)
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


class C3D2Combined(L4Candidate):
    name = "c3d2_combined"
    tier = "base"

    def consolidate(self, db_path: str | Path) -> ConsolidateTelemetry:
        from truememory.clustering import cluster_messages
        from truememory.consolidation import (
            build_entity_summary_sheets,
            build_structured_facts,
            _message_salience,
            _extract_sentences,
            _score_sentence,
            _extract_numbers,
        )

        before = _bytes(db_path)
        conn = sqlite3.connect(str(db_path))
        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception:
            pass

        try:
            t0 = time.time()

            # 1. DELETE fact_timeline — D2's signature move.
            try:
                conn.execute("DELETE FROM fact_timeline")
            except sqlite3.Error:
                pass
            conn.commit()

            # 2. HDBSCAN with tight params
            try:
                cluster_messages(conn, min_cluster_size=3, min_samples=2)
                hdbscan_ok = True
            except Exception as exc:
                hdbscan_ok = False
                notes_suffix = f"; HDBSCAN_FAIL: {exc}"

            conn.execute("DELETE FROM summaries")
            now = datetime.now(timezone.utc).isoformat()

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
                        "SELECT id, content, sender, recipient, timestamp, category, modality "
                        "FROM messages WHERE id IN (" +
                        ",".join("?" * len(mids)) + ") ORDER BY timestamp",
                        mids,
                    ).fetchall()
                    if not msg_rows:
                        continue
                    msgs = [
                        {"id": r[0], "content": r[1], "sender": r[2],
                         "recipient": r[3], "timestamp": r[4],
                         "category": r[5], "modality": r[6]}
                        for r in msg_rows
                    ]

                    scored = [(m, _message_salience(m)) for m in msgs]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    top = scored[:max(5, len(msgs) // 4)]
                    top.sort(key=lambda x: x[0]["timestamp"])

                    sents = []
                    for m, sal in top:
                        for sent in _extract_sentences(m["content"]):
                            sents.append((f"[{m['sender']}] {sent}",
                                          _score_sentence(sent) + sal * 0.3, m))
                    sents.sort(key=lambda x: x[1], reverse=True)
                    sents = sents[:max(15, len(sents) // 3)]
                    sents.sort(key=lambda x: x[2]["timestamp"])
                    summary_text = "\n".join(s[0] for s in sents)

                    key_facts = []
                    for m, _ in top:
                        nums = _extract_numbers(m["content"])
                        if nums:
                            key_facts.extend(nums[:3])

                    ts = [m["timestamp"] for m, _ in top if m["timestamp"]]
                    conn.execute(
                        "INSERT INTO summaries (period, start_date, end_date, "
                        "entity, summary, key_facts, message_ids, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        ("cluster_extractive_d2",
                         min(ts) if ts else "", max(ts) if ts else "",
                         "", summary_text, json.dumps(key_facts),
                         json.dumps([m["id"] for m, _ in top]), now),
                    )
            conn.commit()

            # 3. NO detect_contradictions
            # 4. Keep entity + structured-fact outputs
            try:
                build_entity_summary_sheets(conn)
                build_structured_facts(conn)
            except Exception:
                pass

            wall = time.time() - t0
            rows_out = {
                "summaries": conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0],
                "fact_timeline": conn.execute("SELECT COUNT(*) FROM fact_timeline").fetchone()[0],
                "message_clusters": conn.execute(
                    "SELECT COUNT(*) FROM message_clusters"
                ).fetchone()[0] if hdbscan_ok else 0,
            }
            notes = "C3D2 — HDBSCAN tight (3/2) cluster-extractive summaries + NO contradiction detection"
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
                if r.get("source") == "fact_timeline":
                    continue  # D2 signature: suppress fact_timeline
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
