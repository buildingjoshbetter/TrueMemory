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


class TierSwitchUnsupportedError(RuntimeError):
    """Raised when a tier switch cannot proceed because the sqlite-vec
    extension cannot be loaded on this platform/Python (M-23)."""


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
    from truememory.storage import DEFAULT_BUSY_TIMEOUT_MS, _validate_db_path

    path = db_path or _get_db_path()
    conn = sqlite3.connect(_validate_db_path(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={DEFAULT_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")

    # M-23: guard the sqlite-vec load. On macOS system Python (and any build
    # compiled without SQLITE_ENABLE_LOAD_EXTENSION) ``enable_load_extension``
    # is absent → bare access raised AttributeError, crashing the tier switch
    # (reached from truememory_configure, the CLI upgrade-tier path, and a
    # background thread that swallowed it). A tier switch is meaningless
    # without vec support, so fail with actionable platform guidance instead.
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except (AttributeError, ImportError, OSError, sqlite3.OperationalError) as exc:
        conn.close()
        raise TierSwitchUnsupportedError(
            "Tier switching requires the sqlite-vec extension, but this "
            "Python's sqlite3 cannot load it "
            f"({type(exc).__name__}: {exc}). This commonly happens with the "
            "macOS system Python, which is built without "
            "SQLITE_ENABLE_LOAD_EXTENSION. Install a Python with extension "
            "support (e.g. python.org, Homebrew, or pyenv) and retry."
        ) from exc

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
        # M-51: True while start_rebuild() is mid-preflight (before the worker
        # Thread exists) so a concurrent call can't double-start, yet a
        # pre-thread failure clears it instead of bricking the manager.
        self._claimed: bool = False
        self._active_status_id: int = 0
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
        # M-51: reserve the slot with a claim flag rather than parking the
        # long-lived MCP caller thread in ``_active_thread``. Previously a raise
        # in any pre-thread step below (e.g. M-23 in _open_db, preflight) left
        # ``_active_thread = current_thread()`` set; since that thread stays
        # alive for the process lifetime, ``is_alive()`` was permanently True
        # and every subsequent start_rebuild no-op'd — bricking tier switching.
        # Only the real worker Thread should occupy the slot; release the claim
        # on any pre-thread failure.
        with self._state_lock:
            if self._active_thread and self._active_thread.is_alive():
                return self._active_status_id
            if self._claimed:
                return self._active_status_id
            self._claimed = True

        try:
            conn = _open_db(db_path)
            to_group = tier_group(target_tier)

            ok, msg = preflight_ram_check(to_group)
            if not ok:
                conn.close()
                raise RuntimeError(msg)

            action = resolve_rebuild_action(conn, to_group, force)
            messages, is_full = get_messages_to_embed(conn, to_group, force)

            if not messages and not is_full:
                self._apply_config_switch(target_tier, conn)
                conn.close()
                with self._state_lock:
                    self._claimed = False
                return 0

            status_id = self._create_status_row(
                conn, to_group, target_tier, action, len(messages),
            )
            self._active_status_id = status_id

            if backup_path is None and is_full:
                backup_path = backup_db(db_path)

            conn.close()

            thread = threading.Thread(
                target=self._rebuild_thread,
                args=(target_tier, to_group, messages, is_full, status_id,
                      backup_path, db_path),
                daemon=True,
                name=f"tier-switch-{to_group}",
            )
            with self._state_lock:
                self._active_thread = thread
                self._claimed = False
            thread.start()
        except BaseException:
            # Pre-thread failure: release the claim so the next call can retry.
            # The worker Thread never started, so it never occupied the slot.
            with self._state_lock:
                self._claimed = False
            raise

        return status_id

    def run_rebuild_sync(
        self,
        target_tier: str,
        force: bool = False,
        progress_callback=None,
        db_path: Path | None = None,
    ) -> bool:
        """Run a synchronous rebuild (CLI path). Returns True on success."""
        _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = open(_LOCK_PATH, "w")
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            log.error("Another rebuild is already running (lock held)")
            lock_fd.close()
            return False
        try:
            return self._run_rebuild_sync_inner(
                target_tier, force, progress_callback, db_path,
            )
        finally:
            if os.name == "nt":
                import msvcrt
                try:
                    msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    def _run_rebuild_sync_inner(
        self,
        target_tier: str,
        force: bool = False,
        progress_callback=None,
        db_path: Path | None = None,
    ) -> bool:
        conn = _open_db(db_path)
        to_group = tier_group(target_tier)

        ok, msg = preflight_ram_check(to_group)
        if not ok:
            conn.close()
            log.error("Pre-flight failed: %s", msg)
            return False

        action = resolve_rebuild_action(conn, to_group, force)
        messages, is_full = get_messages_to_embed(conn, to_group, force)

        if not messages and not is_full:
            self._apply_config_switch(target_tier, conn)
            conn.close()
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
        self._active_worker = worker

        try:
            success, processed = worker.run(messages, is_full)
        except Exception:
            log.exception("Sync rebuild failed")
            success = False

        if success:
            self._finalize_rebuild(conn, target_tier, to_group)

        self._active_worker = None
        conn.close()
        return success

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
        """Signal the active worker to stop."""
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
        """Update config.json with the new tier (atomic write).

        M-58 (#641): the read-modify-replace below previously ran with NO
        cross-process lock, so a concurrent writer (telemetry user_id, CLI
        setup) could land its update in the window between our read and our
        ``os.replace`` and have it silently clobbered (e.g. a freshly-stored API
        key dropped). Hold the shared ``_config_file_lock`` (same fcntl/msvcrt
        primitive used by ``run_rebuild_sync`` and ``_save_config``) across the
        WHOLE read-modify-write so the tier write composes with — instead of
        racing — other config writers.
        """
        import tempfile

        from truememory.mcp_server import _config_file_lock

        config_path = _TRUEMEMORY_DIR / "config.json"
        with _config_file_lock():
            config = {}
            if config_path.exists():
                try:
                    # utf-8-sig tolerates a BOM; isinstance guard discards a
                    # valid-JSON non-object config so the tier write starts from
                    # a clean dict instead of crashing on config["tier"] (#640).
                    loaded = json.loads(config_path.read_text(encoding="utf-8-sig"))
                    if isinstance(loaded, dict):
                        config = loaded
                except (json.JSONDecodeError, OSError):
                    pass

            config["tier"] = target_tier
            config_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                prefix=".config.tmp.", suffix=".json",
                dir=str(_TRUEMEMORY_DIR),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2)
                os.replace(tmp_path, str(config_path))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

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
