"""RebuildWorker — the batch embedding loop for tier-switch re-embedding.

Processes messages in throttled batches, building both completion and
separation vector tables for the target tier group. Integrates with
DynamicThrottler for hardware-adaptive batch sizing and
VectorCacheRegistry for progress tracking.
"""

import logging
import math
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, NamedTuple

from truememory.tier_switch.cache import VectorCacheRegistry
from truememory.tier_switch.throttler import DynamicThrottler

if TYPE_CHECKING:
    from truememory.embedding_target import EmbeddingTarget
    from truememory.tier_switch.source import TierSourcePlan
    from truememory.vector_search import _PreparedEmbeddingLease

log = logging.getLogger(__name__)

_HARD_TIMEOUT = 9000  # 2.5 hours

StatusCallback = Callable[[int, int, dict], None]


class SourceBuildResult(NamedTuple):
    status: str
    plan: "TierSourcePlan"
    processed: int
    captured_range_complete: bool = False
    current_source_complete: bool = False
    activated: bool = False


class _SourceStopped(Exception):
    pass


def _encode_source_prefix(lease: "_PreparedEmbeddingLease", rows: tuple, count: int,
                          admit: Callable[[], float]) -> tuple[list[bytes], list[bytes]]:
    from truememory.vector_search import _build_sep_text, serialize_f32

    # Keep arrays inside this frame so a failed attempt releases its traceback
    # before the outer loop performs allocator cleanup or revalidates SQL.
    texts = [row[1] for row in rows[:count]]
    embeddings = lease.encode(texts, timeout=admit(), batch_size=32)
    admit()
    if len(embeddings) != count:
        raise ValueError("Completion embedding count does not match the rebuild prefix")
    completion = [serialize_f32(vector) for vector in embeddings]
    del embeddings, texts
    sep_texts = [_build_sep_text(row[2], row[3], row[4], row[1]) for row in rows[:count]]
    embeddings = lease.encode(sep_texts, timeout=admit(), batch_size=32)
    admit()
    if len(embeddings) != count:
        raise ValueError("Separation embedding count does not match the rebuild prefix")
    separation = [serialize_f32(vector) for vector in embeddings]
    del embeddings, sep_texts
    admit()
    return completion, separation


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

    def run_source(
        self, target: "EmbeddingTarget", plan: "TierSourcePlan", *,
        page_size: int = 64, timeout: float = _HARD_TIMEOUT,
    ) -> SourceBuildResult:
        """Build an initialized inactive pair without activating or clearing it.

        The caller owns maintenance, the clean connection and exclusive pair
        writer, and has validated target schema and database identity. At most
        one source page plus encoded prefix buffers is retained. Cancellation is
        cooperative between calls: it cannot interrupt native/SQLite calls or
        their internal waits. Deadlines also flow into the prepared lease.
        """
        from truememory.embedding_target import EmbeddingTarget
        from truememory.tier_switch.source import (
            TierSourcePlan, TierSourceUntrusted, finish_tier_source, plan_tier_source,
            publish_tier_prefix, read_tier_page,
        )
        from truememory.vector_search import prepare_embedding_target

        if not isinstance(target, EmbeddingTarget) or not isinstance(plan, TierSourcePlan):
            raise TypeError("An explicit frozen target and initialized source plan are required")
        if (plan.connection is not self.conn or plan.action not in ("resume", "complete")
                or plan.manifest.model != target.model_id or plan.manifest.dimension != target.dimension
                or plan.manifest.targets != target.tables or self.target_tier != target.tier
                or self.target_group != target.tier_group):
            raise ValueError("Worker, initialized source and prepared target identities must agree")
        if type(page_size) is not int or page_size < 1:
            raise ValueError("Source page size must be a positive integer")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Source timeout must be finite and positive")
        if self.conn.in_transaction:
            raise RuntimeError("RebuildWorker.run_source requires a clean owned connection")
        expires = time.monotonic() + min(timeout, _HARD_TIMEOUT)
        initial_consumed = plan.manifest.consumed
        phase_ceiling = effective_ceiling = None
        phase_failures = 0
        captured_complete = current_complete = False
        publication_uncertain = False

        def admit() -> float:
            if self.conn.in_transaction:
                raise RuntimeError("Source worker connection did not return to a clean boundary")
            if self._cancelled:
                raise _SourceStopped("cancelled")
            remaining = expires - time.monotonic()
            if remaining <= 0:
                raise _SourceStopped("timeout")
            return remaining

        def result(status: str) -> SourceBuildResult:
            if publication_uncertain:
                raise TierSourceUntrusted("Failed publication requires durable checkpoint revalidation")
            return SourceBuildResult(status, plan, plan.manifest.consumed - initial_consumed,
                                     captured_complete, current_complete)

        try:
            admit()
            # A stale plan must fail before preparation, even for an empty range.
            pending = read_tier_page(self.conn, plan, size=page_size)
            admit()
            try:
                import torch
                no_grad = torch.no_grad()
            except ImportError:
                from contextlib import nullcontext
                no_grad = nullcontext()
            with no_grad, prepare_embedding_target(target, timeout=admit()) as lease:
                admit()
                while pending.rows:
                    batch_size, metrics = self.throttler.before_batch()
                    admit()
                    if type(batch_size) is not int or batch_size < 1:
                        raise ValueError("Throttle admission must grant a positive integer prefix")
                    count = min(batch_size, len(pending.rows))
                    if effective_ceiling is not None:
                        count = min(count, effective_ceiling)
                    started = time.monotonic()
                    oom = False
                    publishing = False
                    completion = separation = None
                    try:
                        completion, separation = _encode_source_prefix(lease, pending.rows, count, admit)
                        admit()
                        publishing = True
                        plan, pending = publish_tier_prefix(
                            self.conn, plan, pending, completion=completion, separation=separation,
                        )
                    except _SourceStopped:
                        raise
                    except Exception as error:
                        # No DB queries, status updates or retries after uncertain
                        # rollback. The caller must discard this connection.
                        if (self.conn.in_transaction or (publishing and
                                (error.__context__ is not None or error.__cause__ is not None))):
                            raise
                        if not self._is_oom_error(error):
                            raise
                        publication_uncertain = publishing
                        oom = True
                    finally:
                        completion = separation = None

                    if oom:
                        if phase_ceiling is None:
                            phase_ceiling = effective_ceiling = count
                            phase_failures = 1
                        else:
                            effective_ceiling = min(effective_ceiling, count)
                            if count <= phase_ceiling // 2:
                                phase_ceiling = count
                                phase_failures = 1
                            else:
                                phase_failures += 1
                        exhausted = phase_ceiling == 1 and phase_failures == 2
                        if phase_failures == 2 and not exhausted:
                            phase_ceiling //= 2
                            effective_ceiling = min(effective_ceiling, phase_ceiling)
                            phase_failures = 0
                        admit()
                        self.throttler.on_oom()
                        admit()
                        DynamicThrottler.flush_gpu_cache()
                        admit()
                        if exhausted and not publication_uncertain:
                            return result("oom_exhausted")
                        if self.conn.total_changes != plan.changes:
                            # Rolled-back SQL contributes to total_changes. One
                            # audit per failed attempt, bounded by OOM credits.
                            resumed = plan_tier_source(
                                self.conn, model_id=target.model_id, dimension=target.dimension,
                                targets=target.tables,
                            )
                            admit()
                            if resumed.action != "resume" or resumed.manifest != plan.manifest:
                                raise TierSourceUntrusted("Failed publication no longer has the same source checkpoint")
                            plan = resumed
                        checked = read_tier_page(self.conn, plan, size=len(pending.rows))
                        admit()
                        # The adapter checks exact SQLite types again under the
                        # writer. Retain the original immutable pending suffix.
                        from truememory.tier_switch.source import _same_rows
                        if checked.generation != pending.generation or checked.cursor != pending.cursor or not _same_rows(checked.rows, pending.rows):
                            raise TierSourceUntrusted("Failed publication no longer has the same pending source rows")
                        del checked
                        publication_uncertain = False
                        if exhausted:
                            return result("oom_exhausted")
                        continue

                    # Capture the committed checkpoint before observing a
                    # cancellation that arrived during SQLite's writer wait.
                    phase_ceiling = effective_ceiling = None
                    phase_failures = 0
                    admit()
                    self.throttler.after_batch(count, time.monotonic() - started)
                    admit()
                    flush = self.throttler.should_flush_cache()
                    admit()
                    if flush:
                        DynamicThrottler.flush_gpu_cache()
                        admit()
                    if self.status_callback is not None:
                        live = dict(metrics, batch_size=count, activated=False)
                        try:
                            self.status_callback(plan.manifest.consumed, plan.manifest.total, live)
                        except Exception:
                            pass
                        admit()
                    if not pending.rows:
                        pending = read_tier_page(self.conn, plan, size=page_size)
                        admit()
                admit()
                finished = finish_tier_source(self.conn, plan)
                plan = finished.plan
                captured_complete = finished.captured_range_complete
                current_complete = finished.current_source_complete
                admit()
                return SourceBuildResult(
                    "complete", plan, plan.manifest.consumed - initial_consumed,
                    finished.captured_range_complete, finished.current_source_complete,
                )
        except _SourceStopped as stopped:
            return result(str(stopped))

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
                if self.conn.in_transaction:
                    raise
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
        phase_ceiling: int | None = None
        effective_ceiling: int | None = None
        phase_failures = 0

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
                if effective_ceiling is not None:
                    # Admission itself can consume the remaining recovery time.
                    if self._cancelled or time.time() - start_time > _HARD_TIMEOUT:
                        continue
                    batch_size = min(batch_size, effective_ceiling)
                batch = messages[offset : offset + batch_size]
                if not batch:
                    break

                batch_start = time.time()

                oom = False
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
                    # run() owns a clean connection. An active transaction here
                    # means rollback did not restore that boundary.
                    if self.conn.in_transaction:
                        raise
                    if self._is_oom_error(exc):
                        oom = True
                    else:
                        log.error(
                            "RebuildWorker error at %d/%d: %s",
                            processed, total, exc,
                        )
                        self._update_status("failed", processed, total, str(exc))
                        return False, processed

                if oom:
                    # Leave the exception scope before cleanup releases native
                    # workspace retained by the failed forward's traceback.
                    failed_count = len(batch)
                    if phase_ceiling is None:
                        phase_ceiling = effective_ceiling = failed_count
                        phase_failures = 1
                    else:
                        effective_ceiling = min(effective_ceiling, failed_count)
                        if failed_count <= phase_ceiling // 2:
                            phase_ceiling = failed_count
                            phase_failures = 1
                        else:
                            phase_failures += 1
                    exhausted = phase_ceiling == 1 and phase_failures == 2
                    if phase_failures == 2 and not exhausted:
                        phase_ceiling //= 2
                        effective_ceiling = min(effective_ceiling, phase_ceiling)
                        phase_failures = 0
                    log.warning("OOM at effective batch_size=%d, triggering BACKOFF", failed_count)
                    self.throttler.on_oom()
                    DynamicThrottler.flush_gpu_cache()
                    if self._cancelled or time.time() - start_time > _HARD_TIMEOUT:
                        continue
                    if exhausted:
                        self._update_status(
                            "failed", processed, total,
                            "Rebuild ran out of memory twice at batch size 1; retry after freeing memory.",
                        )
                        return False, processed
                    continue

                if not success:
                    self._update_status("failed", processed, total)
                    return False, processed

                batch_time = time.time() - batch_start
                batch_count = len(batch)
                processed += batch_count
                offset += batch_count
                phase_ceiling = effective_ceiling = None
                phase_failures = 0

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
            owned_status = not self.conn.in_transaction
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
                if owned_status and self.conn.in_transaction:
                    raise
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
