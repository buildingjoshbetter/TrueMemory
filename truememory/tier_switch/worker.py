"""RebuildWorker — the batch embedding loop for tier-switch re-embedding.

Processes messages in throttled batches, building both completion and
separation vector tables for the target tier group. Integrates with
DynamicThrottler for hardware-adaptive batch sizing and
VectorCacheRegistry for progress tracking.
"""

import logging
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable

from truememory.tier_switch.cache import VectorCacheRegistry
from truememory.tier_switch.throttler import DynamicThrottler

log = logging.getLogger(__name__)

_HARD_TIMEOUT = 9000  # 2.5 hours

StatusCallback = Callable[[int, int, dict], None]


@contextmanager
def _rebuild_transaction(conn: sqlite3.Connection) -> Iterator[None]:
    owned = not conn.in_transaction
    conn.execute("BEGIN" if owned else "SAVEPOINT truememory_rebuild_batch")
    try:
        yield
        conn.execute("COMMIT" if owned else "RELEASE SAVEPOINT truememory_rebuild_batch")
    except BaseException:
        if owned:
            conn.rollback()
        else:
            # Do not release if rollback fails: that could publish partial work.
            conn.execute("ROLLBACK TO SAVEPOINT truememory_rebuild_batch")
            conn.execute("RELEASE SAVEPOINT truememory_rebuild_batch")
        raise


class RebuildWorker:
    """Batch embedding loop with throttling and progress tracking."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        target_tier: str,
        target_group: str,
        throttler: DynamicThrottler,
        status_id: int = 0,
        status_callback: StatusCallback | None = None,
    ):
        self.conn = conn
        self.target_tier = target_tier
        self.target_group = target_group
        self.throttler = throttler
        self.status_id = status_id
        self.status_callback = status_callback
        self._cancelled = False

    def cancel(self):
        """Signal the worker to stop after the current batch."""
        self._cancelled = True

    def run(
        self,
        messages: list[dict],
        is_full_rebuild: bool,
    ) -> tuple[bool, int]:
        """Execute the batch embedding loop.

        Args:
            messages: List of message dicts to embed (id, content, sender,
                      recipient, timestamp).
            is_full_rebuild: If True, clear target tables before inserting.

        Returns:
            (success, total_processed) tuple.
        """
        from truememory.vector_search import (
            _build_sep_text,
            get_model,
            init_vec_table,
            serialize_f32,
            set_embedding_model,
        )

        if not messages:
            log.info("No messages to embed — nothing to do")
            return True, 0

        # Manager workers use dedicated connections. Initialization and status
        # publication commit independently, so caller-owned work cannot join run().
        if self.conn.in_transaction:
            raise RuntimeError("RebuildWorker.run requires a connection with no active transaction")

        total = len(messages)
        log.info(
            "RebuildWorker starting: %d messages, group=%s, full=%s",
            total, self.target_group, is_full_rebuild,
        )

        set_embedding_model(self.target_tier)
        model = get_model()

        vec_table = f"vec_messages_{self.target_group}"
        sep_table = f"vec_messages_sep_{self.target_group}"
        init_vec_table(self.conn, tier_group=self.target_group)

        if is_full_rebuild:
            try:
                with _rebuild_transaction(self.conn):
                    self.conn.execute(f"DELETE FROM {vec_table}")
                    self.conn.execute(f"DELETE FROM {sep_table}")
                    VectorCacheRegistry.update_progress(
                        self.conn, self.target_group, 0, 0, commit=False,
                    )
            except Exception as exc:
                self._update_status("failed", 0, total, str(exc))
                return False, 0

        try:
            import torch
            no_grad = torch.no_grad()
        except ImportError:
            from contextlib import nullcontext
            no_grad = nullcontext()

        processed = 0
        offset = 0
        start_time = time.time()

        with no_grad:
            while offset < total:
                if self._cancelled:
                    log.info("RebuildWorker cancelled at %d/%d", processed, total)
                    self._update_status("cancelled", processed, total)
                    return False, processed

                if time.time() - start_time > _HARD_TIMEOUT:
                    log.warning(
                        "Re-embedding timed out at %d/%d (%.0f%%) after %.0f minutes",
                        processed, total, processed / total * 100,
                        (time.time() - start_time) / 60,
                    )
                    self._update_status("timeout", processed, total)
                    return False, processed

                batch_size, metrics = self.throttler.before_batch()
                batch = messages[offset : offset + batch_size]
                if not batch:
                    break

                batch_start = time.time()

                try:
                    success = self._process_batch(
                        batch, model, vec_table, sep_table, serialize_f32,
                        _build_sep_text,
                        record_progress=lambda: VectorCacheRegistry.update_progress(
                            self.conn, self.target_group, batch[-1]["id"],
                            self._count_vectors(vec_table), commit=False,
                        ),
                    )
                except Exception as exc:
                    if self._is_oom_error(exc):
                        log.warning(
                            "OOM at batch_size=%d, triggering BACKOFF",
                            batch_size,
                        )
                        self.throttler.on_oom()
                        DynamicThrottler.flush_gpu_cache()
                        continue
                    log.error(
                        "RebuildWorker error at %d/%d: %s",
                        processed, total, exc,
                    )
                    self._update_status("failed", processed, total, str(exc))
                    return False, processed

                if not success:
                    self._update_status("failed", processed, total)
                    return False, processed

                batch_time = time.time() - batch_start
                batch_count = len(batch)
                processed += batch_count
                offset += batch_count

                self.throttler.after_batch(batch_count, batch_time)
                if self.throttler.should_flush_cache():
                    DynamicThrottler.flush_gpu_cache()

                self._update_status("running", processed, total)

        self._update_status("complete", processed, total)
        log.info(
            "RebuildWorker complete: %d messages, group=%s",
            processed, self.target_group,
        )
        return True, processed

    def _process_batch(
        self,
        batch: list[dict],
        model,
        vec_table: str,
        sep_table: str,
        serialize_f32,
        build_sep_text,
        record_progress: Callable[[], None] | None = None,
    ) -> bool:
        """Prepare both embeddings, then publish their pair and checkpoint atomically."""
        from truememory.mps_utils import encode_with_model_ownership

        texts = [m["content"] for m in batch]
        ids = [m["id"] for m in batch]

        embeddings = encode_with_model_ownership(model, texts, show_progress_bar=False)
        if len(embeddings) != len(ids):
            raise ValueError("Completion embedding count does not match the rebuild batch")
        completion_rows = [(mid, serialize_f32(emb)) for mid, emb in zip(ids, embeddings)]
        del embeddings

        sep_texts = [
            build_sep_text(
                m.get("sender", "?"),
                m.get("recipient", "?"),
                m.get("timestamp", "?"),
                m["content"],
            )
            for m in batch
        ]
        sep_embeddings = encode_with_model_ownership(model, sep_texts, show_progress_bar=False)
        if len(sep_embeddings) != len(ids):
            raise ValueError("Separation embedding count does not match the rebuild batch")
        separation_rows = [(mid, serialize_f32(emb)) for mid, emb in zip(ids, sep_embeddings)]
        del sep_embeddings

        with _rebuild_transaction(self.conn):
            self.conn.executemany(
                f"INSERT INTO {vec_table}(rowid, embedding) VALUES (?, ?)", completion_rows,
            )
            self.conn.executemany(
                f"INSERT INTO {sep_table}(rowid, embedding) VALUES (?, ?)", separation_rows,
            )
            if record_progress is not None:
                record_progress()
        return True

    def _count_vectors(self, vec_table: str) -> int:
        """Count rows in the vector table."""
        row = self.conn.execute(f"SELECT COUNT(*) FROM {vec_table}").fetchone()
        return row[0] if row else 0

    def _update_status(
        self,
        status: str,
        processed: int,
        total: int,
        error: str | None = None,
    ):
        """Update rebuild_status table and call the status callback."""
        remaining = total - processed
        eta = self.throttler.get_eta_seconds(remaining)
        throughput = self.throttler.get_throughput()
        ram_pct = 0.0
        try:
            import psutil
            ram_pct = psutil.virtual_memory().percent
        except Exception:
            pass

        pct = (processed / total * 100) if total > 0 else 0

        if self.status_id:
            now = time.time()
            completed_at = now if status in ("complete", "failed", "cancelled") else None
            try:
                with _rebuild_transaction(self.conn):
                    self.conn.execute(
                        "UPDATE rebuild_status SET "
                        "status=?, processed_messages=?, progress_pct=?, "
                        "eta_seconds=?, batch_size=?, throughput_ips=?, "
                        "ram_pct=?, last_heartbeat=?, completed_at=?, error=? "
                        "WHERE id=?",
                        (
                            status, processed, pct,
                            eta, self.throttler.batch_size, throughput,
                            ram_pct, now, completed_at, error,
                            self.status_id,
                        ),
                    )
            except sqlite3.OperationalError:
                pass

        if self.status_callback:
            metrics = {
                "batch_size": self.throttler.batch_size,
                "eta_seconds": eta,
                "throughput_ips": throughput,
                "ram_pct": ram_pct,
                "progress_pct": pct,
            }
            try:
                self.status_callback(processed, total, metrics)
            except Exception:
                pass

    @staticmethod
    def _is_oom_error(exc: Exception) -> bool:
        """Check if an exception is an out-of-memory error."""
        msg = str(exc).lower()
        if "out of memory" in msg or "oom" in msg:
            return True
        try:
            import torch
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                return True
        except (ImportError, AttributeError):
            pass
        return False
