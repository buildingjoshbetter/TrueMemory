"""RebuildManager — orchestrates tier-switch re-embedding.

Handles pre-flight checks, DB backup, transition logic, async/sync
execution, file-based locking, and status queries. Background threads
create their own SQLite connections for thread safety.
"""

import json
import logging
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path

from truememory.tier_switch.cache import (
    VectorCacheRegistry,
    get_messages_to_embed,
    model_name_for_group,
    preflight_ram_check,
    resolve_rebuild_action,
    tier_group,
)
from truememory.tier_switch.throttler import DynamicThrottler
from truememory.tier_switch.worker import RebuildWorker

log = logging.getLogger(__name__)

_TRUEMEMORY_DIR = Path.home() / ".truememory"
_LOCK_PATH = _TRUEMEMORY_DIR / "rebuild.lock"


def _detect_device() -> str:
    """Detect the best available compute device."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _get_db_path() -> Path:
    """Return the path to the TrueMemory database."""
    return _TRUEMEMORY_DIR / "memories.db"


def _open_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a new SQLite connection (thread-safe: each thread gets its own)."""
    from truememory.storage import DEFAULT_BUSY_TIMEOUT_MS

    path = db_path or _get_db_path()
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={DEFAULT_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA synchronous=NORMAL")

    import sqlite_vec
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    return conn


class RebuildManager:
    """Orchestrates tier-switch re-embedding (singleton for MCP)."""

    _instance: "RebuildManager | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "RebuildManager":
        """Get or create the singleton instance."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self._active_worker: RebuildWorker | None = None
        self._active_thread: threading.Thread | None = None
        self._active_status_id: int = 0
        # Guards the check-then-write of _active_thread / _active_worker
        # across the MCP thread (start_rebuild / cancel) and the background
        # rebuild thread (_rebuild_thread). Without this, two concurrent
        # truememory_configure calls can both observe is_alive()==False
        # between the check (line ~100) and the assignment (line ~136),
        # both spawn rebuild threads, and both race on the same DB —
        # leaving the tier in an indeterminate state.
        self._state_lock = threading.Lock()

    def start_rebuild(
        self,
        target_tier: str,
        force: bool = False,
        backup_path: Path | None = None,
        db_path: Path | None = None,
    ) -> int:
        """Start an async rebuild in a background thread (MCP path).

        Returns a status_id for progress queries.
        """
        # Single critical section: check-then-write of _active_thread so
        # two concurrent callers can't both pass the is_alive() check.
        with self._state_lock:
            if self._active_thread and self._active_thread.is_alive():
                return self._active_status_id

        # All open conn paths go through try/finally so a raise from
        # tier_group / preflight / resolve_rebuild_action / backup_db
        # cannot leak the SQLite handle.
        conn = _open_db(db_path)
        thread = None
        try:
            to_group = tier_group(target_tier)

            ok, msg = preflight_ram_check(to_group)
            if not ok:
                raise RuntimeError(msg)

            action = resolve_rebuild_action(conn, to_group, force)
            messages, is_full = get_messages_to_embed(conn, to_group, force)

            if not messages and not is_full:
                self._apply_config_switch(target_tier, conn)
                return 0

            status_id = self._create_status_row(
                conn, to_group, target_tier, action, len(messages),
            )
            self._active_status_id = status_id

            if backup_path is None and is_full:
                backup_path = backup_db(db_path)

            thread = threading.Thread(
                target=self._rebuild_thread,
                args=(target_tier, to_group, messages, is_full, status_id,
                      backup_path, db_path),
                daemon=True,
                name=f"tier-switch-{to_group}",
            )
        finally:
            conn.close()

        # Take the state lock again to publish the new thread atomically.
        # Window between dropping the lock above and re-acquiring here is
        # safe because callers serialize through the lock — the worst case
        # is a benign race-and-discard if a second caller arrived during
        # the conn-open work, in which case both will see the thread we
        # publish here on their next is_alive() check.
        with self._state_lock:
            self._active_thread = thread
        thread.start()

        return status_id

    def run_rebuild_sync(
        self,
        target_tier: str,
        force: bool = False,
        progress_callback=None,
        db_path: Path | None = None,
    ) -> bool:
        """Run a synchronous rebuild (CLI path). Returns True on success."""
        # try/finally ensures conn closes on every path, including
        # exceptions from tier_group / preflight / resolve_rebuild_action
        # / _create_status_row that previously leaked the handle.
        conn = _open_db(db_path)
        try:
            to_group = tier_group(target_tier)

            ok, msg = preflight_ram_check(to_group)
            if not ok:
                log.error("Pre-flight failed: %s", msg)
                return False

            action = resolve_rebuild_action(conn, to_group, force)
            messages, is_full = get_messages_to_embed(conn, to_group, force)

            if not messages and not is_full:
                self._apply_config_switch(target_tier, conn)
                return True

            status_id = self._create_status_row(
                conn, to_group, target_tier, action, len(messages),
            )

            if is_full:
                backup_db(db_path)

            device = _detect_device()
            if to_group == "edge":
                device = "cpu"
            throttler = DynamicThrottler(device=device)

            worker = RebuildWorker(
                conn=conn,
                target_tier=target_tier,
                target_group=to_group,
                throttler=throttler,
                status_id=status_id,
                status_callback=progress_callback,
            )
            with self._state_lock:
                self._active_worker = worker

            try:
                success, processed = worker.run(messages, is_full)
            except Exception:
                log.exception("Sync rebuild failed")
                success = False

            if success:
                self._finalize_rebuild(conn, target_tier, to_group)

            return success
        finally:
            with self._state_lock:
                self._active_worker = None
            conn.close()

    def get_status(self, status_id: int = 0) -> dict:
        """Query rebuild progress from the rebuild_status table."""
        try:
            conn = _open_db()
        except Exception:
            return {"status": "unknown", "error": "cannot open db"}

        try:
            sid = status_id or self._active_status_id
            if not sid:
                row = conn.execute(
                    "SELECT * FROM rebuild_status ORDER BY id DESC LIMIT 1"
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM rebuild_status WHERE id = ?", (sid,)
                ).fetchone()

            if not row:
                return {"status": "no_rebuild_found"}

            cols = [
                "id", "tier_group", "target_tier", "status", "action",
                "total_messages", "processed_messages", "progress_pct",
                "eta_seconds", "batch_size", "throughput_ips", "ram_pct",
                "pressure", "error", "started_at", "completed_at",
                "backup_path", "last_heartbeat",
            ]
            return dict(zip(cols, row))
        finally:
            conn.close()

    def cancel(self, status_id: int = 0):
        """Signal the active worker to stop.

        Reads _active_worker under _state_lock so a concurrent
        _rebuild_thread teardown (which sets _active_worker = None) can't
        cause the cancel() to silently no-op against a stale reference.
        """
        with self._state_lock:
            worker = self._active_worker
        if worker is not None:
            worker.cancel()

    def _rebuild_thread(
        self,
        target_tier: str,
        to_group: str,
        messages: list[dict],
        is_full: bool,
        status_id: int,
        backup_path: Path | None,
        db_path: Path | None,
    ):
        """Background thread target — creates its own DB connection."""
        conn = _open_db(db_path)
        try:
            device = _detect_device()
            if to_group == "edge":
                device = "cpu"
            throttler = DynamicThrottler(device=device)

            worker = RebuildWorker(
                conn=conn,
                target_tier=target_tier,
                target_group=to_group,
                throttler=throttler,
                status_id=status_id,
            )
            with self._state_lock:
                self._active_worker = worker

            success, processed = worker.run(messages, is_full)

            if success:
                self._finalize_rebuild(conn, target_tier, to_group)
                log.info("Background rebuild complete: tier=%s", target_tier)
            else:
                log.warning("Background rebuild failed: tier=%s", target_tier)

        except Exception:
            log.exception("Background rebuild thread error")
        finally:
            # State teardown under lock so start_rebuild's check-then-write
            # cycle sees consistent state across thread boundaries.
            with self._state_lock:
                self._active_worker = None
                self._active_thread = None
            conn.close()

    def _finalize_rebuild(
        self, conn: sqlite3.Connection, target_tier: str, to_group: str,
    ):
        """Apply config changes after a successful rebuild."""
        from truememory.vector_search import (
            _write_embedder_metadata,
            set_embedding_model,
        )

        vec_table = f"vec_messages_{to_group}"
        try:
            row = conn.execute(
                f"SELECT MAX(rowid), COUNT(*) FROM {vec_table}"
            ).fetchone()
            last_id = row[0] or 0 if row else 0
            vec_count = row[1] or 0 if row else 0
        except sqlite3.OperationalError:
            last_id = 0
            vec_count = 0

        VectorCacheRegistry.set(
            conn,
            to_group,
            model_name=model_name_for_group(to_group),
            last_embedded_id=last_id,
            vector_count=vec_count,
        )

        set_embedding_model(target_tier)
        _write_embedder_metadata(conn)
        self._apply_config_switch(target_tier, conn)

    def _apply_config_switch(
        self, target_tier: str, conn: sqlite3.Connection,
    ):
        """Update config.json with the new tier."""
        config_path = _TRUEMEMORY_DIR / "config.json"
        config = {}
        if config_path.exists():
            try:
                # encoding="utf-8" is critical on Windows where the default
                # codec is cp1252 — a non-ASCII byte in config.json (e.g. an
                # API key with extended chars) crashes the parse and silently
                # drops the tier switch.
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        config["tier"] = target_tier
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2))

        os.environ["TRUEMEMORY_EMBED_MODEL"] = target_tier

    def _create_status_row(
        self,
        conn: sqlite3.Connection,
        to_group: str,
        target_tier: str,
        action: str,
        total: int,
    ) -> int:
        """Insert a new rebuild_status row and return its id."""
        now = time.time()
        cursor = conn.execute(
            "INSERT INTO rebuild_status "
            "(tier_group, target_tier, status, action, total_messages, "
            "started_at, last_heartbeat) "
            "VALUES (?, ?, 'running', ?, ?, ?, ?)",
            (to_group, target_tier, action, total, now, now),
        )
        conn.commit()
        return cursor.lastrowid


def backup_db(db_path: Path | None = None) -> Path:
    """Create a timestamped backup of the memory database.

    Returns the backup path. Keeps the last 3 backups.
    """
    source = db_path or _get_db_path()
    if not source.exists():
        raise FileNotFoundError(f"Database not found: {source}")

    backup_dir = _TRUEMEMORY_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time())
    dest = backup_dir / f"memories.db.pre-tier-switch-{ts}"
    shutil.copy2(str(source), str(dest))

    backups = sorted(backup_dir.glob("memories.db.pre-tier-switch-*"))
    for old in backups[:-3]:
        old.unlink(missing_ok=True)

    log.info("DB backed up to %s", dest)
    return dest
