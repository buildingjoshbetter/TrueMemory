"""C5 — NLI-based contradiction detection.

Replaces the 5 _CHANGE_PATTERNS regex in truememory/consolidation.py
with DeBERTa-v3-base zero-shot NLI over same-entity message pairs
within a 30-day window, filtered by keyword-overlap ≥ 2 to keep
wall-clock under Base tier's 2-min/1k-msgs budget.

Model: MoritzLaurer/deberta-v3-base-zeroshot-v2.0
  - 184M params
  - ~100ms/pair on CPU (rough; measured per-pair in Phase 9)

Falls back to C1 regex if transformers or model download unavailable.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from benchmarks.gate_eval.candidates.l4_consolidation._base import (
    ConsolidateTelemetry, L4Candidate, RetrievalResult,
)


def _bytes(p):
    try:
        return Path(p).stat().st_size
    except OSError:
        return 0


def _parse_ts(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _tokenize(s: str) -> set[str]:
    return {w.lower().strip(".,!?\"'")
            for w in s.split()
            if len(w) > 3}


class C5NLIContradictions(L4Candidate):
    name = "c5_nli_contradictions"
    tier = "base"

    _nli_classifier = None  # lazily loaded, shared across consolidate() calls

    def _load_nli(self):
        """Lazy-load DeBERTa-base zeroshot NLI. Returns classifier or None."""
        if C5NLIContradictions._nli_classifier is not None:
            return C5NLIContradictions._nli_classifier
        try:
            from transformers import pipeline
            # zero-shot-classification pipeline supports the NLI checkpoint directly
            classifier = pipeline(
                "text-classification",
                model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
                device=-1,  # CPU
                top_k=None,
            )
            C5NLIContradictions._nli_classifier = classifier
            return classifier
        except Exception as exc:
            print(f"  [C5] NLI model load failed, fallback to regex: {exc}")
            return None

    def consolidate(self, db_path: str | Path) -> ConsolidateTelemetry:
        from truememory.consolidation import (
            detect_contradictions,
            build_summaries,
            build_entity_summary_sheets,
            build_structured_facts,
        )

        before = _bytes(db_path)
        conn = sqlite3.connect(str(db_path))

        # NLI first (new mechanism); fall back to regex if model unavailable.
        nli = self._load_nli()
        nli_pairs_checked = 0
        nli_contradictions = 0
        keep_regex_fallback = nli is None

        try:
            t0 = time.time()
            conn.execute("DELETE FROM fact_timeline")
            conn.commit()

            if nli is not None:
                rows = conn.execute(
                    "SELECT id, content, sender, recipient, timestamp "
                    "FROM messages ORDER BY timestamp"
                ).fetchall()
                msgs = [
                    {"id": r[0], "content": r[1], "sender": r[2],
                     "recipient": r[3], "timestamp": r[4]}
                    for r in rows
                ]

                # Build per-sender buckets for pair enumeration
                by_sender: dict[str, list[dict]] = defaultdict(list)
                for m in msgs:
                    s = (m["sender"] or "").lower()
                    if s:
                        by_sender[s].append(m)

                window = timedelta(days=30)
                pairs_to_check: list[tuple[dict, dict]] = []
                for sender, ms in by_sender.items():
                    for i, m1 in enumerate(ms):
                        t1 = _parse_ts(m1["timestamp"])
                        for m2 in ms[i + 1:]:
                            t2 = _parse_ts(m2["timestamp"])
                            if t1 and t2 and (t2 - t1) > window:
                                break
                            # Cheap keyword-overlap filter to cap pair count
                            tok1 = _tokenize(m1["content"])
                            tok2 = _tokenize(m2["content"])
                            overlap = tok1 & tok2
                            if len(overlap) >= 2:
                                pairs_to_check.append((m1, m2))

                # Check each pair with NLI
                for m1, m2 in pairs_to_check:
                    nli_pairs_checked += 1
                    # Prompt: premise = m1, hypothesis = m2; look for
                    # contradiction label on DeBERTa-v3-zeroshot-v2.0 format
                    text_in = f"{m1['content']} [SEP] {m2['content']}"
                    try:
                        out = nli(text_in, truncation=True, max_length=512)
                    except Exception:
                        continue
                    # Parse output — pipeline returns list of {label, score} dicts
                    contradiction_score = 0.0
                    if isinstance(out, list) and out:
                        flat = out[0] if isinstance(out[0], list) else out
                        for item in flat:
                            if "contradiction" in item.get("label", "").lower() \
                                    or "entail" in item.get("label", "").lower() == "no":
                                contradiction_score = max(contradiction_score,
                                                           item.get("score", 0.0))
                    if contradiction_score > 0.7:
                        nli_contradictions += 1
                        subject = f"nli_{m1['sender']}_{hash(m1['content']) & 0xffff}"
                        conn.execute(
                            "INSERT INTO fact_timeline (subject, fact, "
                            "source_message_id, timestamp, entity_scope, valid_from) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (subject, m1["content"][:200], m1["id"],
                             m1["timestamp"], m1["sender"] or "", m1["timestamp"]),
                        )
                        old_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                        conn.execute(
                            "INSERT INTO fact_timeline (subject, fact, "
                            "source_message_id, timestamp, entity_scope, valid_from) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (subject, m2["content"][:200], m2["id"],
                             m2["timestamp"], m2["sender"] or "", m2["timestamp"]),
                        )
                        new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                        conn.execute(
                            "UPDATE fact_timeline SET superseded_by = ?, valid_to = ? "
                            "WHERE id = ?",
                            (new_id, m2["timestamp"], old_id),
                        )

                conn.commit()

            if keep_regex_fallback:
                detect_contradictions(conn)

            # Summaries + entity + structured-fact pipelines unchanged (shared with C1)
            build_summaries(conn)
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
            notes = (f"C5 — NLI contradictions; {nli_pairs_checked} pairs checked, "
                     f"{nli_contradictions} contradictions found")
            if keep_regex_fallback:
                notes += " [FALLBACK_REGEX: NLI model unavailable]"
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
        # Retrieval shape identical to C1 — consolidation writes feed the
        # same search_consolidated + FTS backfill path.
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
