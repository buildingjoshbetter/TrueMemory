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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from truememory.tier_switch.activation import ActivationIntent, ActivationState, ConfigGuard, LegacyTierPolicy, TierSelection
    from truememory.tier_switch.job import TierJob
    from truememory.tier_switch.source import TierSourcePlan
    from truememory.tier_switch.worker import StatusCallback

from truememory.tier_switch.cache import (
    preflight_ram_check,
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
    return Path(os.environ.get("TRUEMEMORY_DB_PATH") or os.environ.get("TRUEMEMORY_DB") or _TRUEMEMORY_DIR / "memories.db")


def _open_db(db_path: Path | None = None, *, initialize: bool = False) -> sqlite3.Connection:
    """Validate existing journal storage before any connection or schema writes."""
    from truememory.storage import DEFAULT_BUSY_TIMEOUT_MS, _validate_db_path
    from truememory.tier_switch.activation import read_activation_state

    path = Path(_validate_db_path(db_path or _get_db_path()))
    conn = (sqlite3.connect(path, check_same_thread=False) if initialize else
            sqlite3.connect(path.absolute().as_uri() + "?mode=rw", uri=True, check_same_thread=False))
    try:
        objects = conn.execute(
            "SELECT 1 FROM main.sqlite_master WHERE name NOT LIKE 'sqlite_%' LIMIT 1",
        ).fetchone()
        if objects is None:
            if not initialize:
                raise TierSwitchUnsupportedError("Database has not been initialized")
            conn.close()
            from truememory.storage import create_db
            conn = create_db(path)
        else:
            read_activation_state(conn)
            if conn.execute("SELECT 1 FROM main.sqlite_master WHERE type='table' AND name='messages'").fetchone() is None:
                raise TierSwitchUnsupportedError("Existing source schema cannot be initialized by a tier switch")
        conn.execute(f"PRAGMA busy_timeout={DEFAULT_BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        _load_tier_extension(conn)
        return conn
    except BaseException:
        conn.close()
        raise


def _load_tier_extension(conn: sqlite3.Connection) -> None:
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except (AttributeError, ImportError, OSError, sqlite3.OperationalError) as exc:
        raise TierSwitchUnsupportedError(
            "Tier switching requires the sqlite-vec extension, but this "
            "Python's sqlite3 cannot load it "
            f"({type(exc).__name__}: {exc}). This commonly happens with the "
            "macOS system Python, which is built without "
            "SQLITE_ENABLE_LOAD_EXTENSION. Install a Python with extension "
            "support (e.g. python.org, Homebrew, or pyenv) and retry."
        ) from exc


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
        self._active_job_id: str | None = None
        self._state_lock = threading.Lock()
        self._cancel_requested = threading.Event()
        self._active_db_path: Path | None = None
        self._last_outcome: str | None = None
        self._requested_tier: str | None = None
        self._last_action: str | None = None
        self._served_tier: str | None = None
        self._config_path: Path = _TRUEMEMORY_DIR / "config.json"
        self._config_lock_path: Path | None = None

    def start_rebuild(
        self, target_tier: str, force: bool = False,
        backup_path: Path | None = None, db_path: Path | None = None, *,
        config_path: Path | None = None, config_lock_path: Path | None = None,
    ) -> int:
        """Stage an inactive job, then hand only its immutable identity to a thread."""
        from truememory.maintenance import maintenance_owner

        with self._state_lock:
            if self._claimed or self._active_thread and self._active_thread.is_alive():
                if self._requested_tier != target_tier:
                    raise TierSwitchUnsupportedError("Another requested tier is already being prepared")
                return self._active_status_id
            self._requested_tier = target_tier
            self._active_status_id = 0
            self._active_job_id = None
            self._last_action = None
            self._last_outcome = "preparing"
            self._served_tier = None
            self._claimed = True
            self._cancel_requested.clear()
            self._config_path = config_path or _TRUEMEMORY_DIR / "config.json"
            self._config_lock_path = config_lock_path
        path = Path(db_path) if db_path is not None else _get_db_path()
        conn = None
        thread = None
        self._active_db_path = path
        try:
            if force:
                raise TierSwitchUnsupportedError("Active force/reset requires an alternate inactive pair and is not supported")
            if self._try_legacy_noop(path, target_tier):
                return 0
            with maintenance_owner(path):
                if self._bootstrap_empty_configuration(path, target_tier, force):
                    return 0
                conn = _open_db(path, initialize=True)
                prepared = self._prepare_job(conn, target_tier, force, backup_path=backup_path)
                conn.close()
                conn = None
            if prepared is None:
                return 0
            job, intent, status_id = prepared
            thread = threading.Thread(target=self._rebuild_thread, args=(job, intent, status_id),
                                      daemon=True, name=f"tier-switch-{job.target.tier_group}")
            with self._state_lock:
                self._active_status_id = status_id
                self._active_job_id = job.job_id
                self._active_db_path = job.database_path
                self._active_thread = thread
            thread.start()
            return status_id
        except BaseException:
            if conn is not None:
                conn.close()
                conn = None
            if self._last_outcome == "activation_pending":
                self._refresh_committed_serving(path)
            else:
                self._last_outcome = "interrupted"
            if thread is not None and not thread.is_alive():
                with self._state_lock:
                    self._active_thread = None
            raise
        finally:
            if conn is not None:
                conn.close()
            with self._state_lock:
                self._claimed = False

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
        self, target_tier: str, force: bool = False, progress_callback=None,
        db_path: Path | None = None,
    ) -> bool:
        from truememory.maintenance import maintenance_owner

        if force:
            raise TierSwitchUnsupportedError("Active force/reset requires an alternate inactive pair and is not supported")
        path = Path(db_path) if db_path is not None else _get_db_path()
        self._cancel_requested.clear()
        self._active_db_path = path
        if self._try_legacy_noop(path, target_tier):
            return True
        with maintenance_owner(path):
            if self._bootstrap_empty_configuration(path, target_tier, force):
                return self._last_outcome == "complete"
            conn = _open_db(path, initialize=True)
            try:
                prepared = self._prepare_job(conn, target_tier, force)
                if prepared is None:
                    return True
                job, intent, status_id = prepared
                self._active_status_id, self._active_db_path = status_id, job.database_path
                self._active_job_id = job.job_id
                return self._run_job(conn, job, intent, status_id, progress_callback)
            except BaseException:
                conn.close()
                conn = None
                if self._last_outcome == "activation_pending":
                    self._refresh_committed_serving(path)
                else:
                    self._last_outcome = "interrupted"
                raise
            finally:
                self._active_worker = None
                if conn is not None:
                    conn.close()

    def _refresh_committed_serving(self, path: Path) -> None:
        """Read back on a fresh validated connection after discarding ambiguity."""
        from truememory.tier_switch.activation import read_activation_state
        self._served_tier = None
        try:
            conn = _open_db(path)
            try:
                state = read_activation_state(conn)
                result = state.selection or state.legacy_policy
                if result is not None:
                    self._served_tier = result.target.tier
            finally:
                conn.close()
        except Exception:
            log.warning("Committed tier readback remains unavailable", exc_info=True)

    @staticmethod
    def _apply_bootstrap_identity(target: object, reranker_id: str) -> None:
        """Configure only proven-unloaded slots; caller holds exclusive serving."""
        from truememory import vector_search, reranker
        from truememory.tier_config import get_tier_config
        if vector_search._model is not None or reranker._model is not None:
            raise TierSwitchUnsupportedError("First-run identity projection requires unloaded model slots")
        if reranker_id != get_tier_config(target.tier)["reranker"]:
            raise TierSwitchUnsupportedError("First-run reranker identity must match its tier")
        # Onboarding has no database policy to authorize policy-tagged slots.
        vector_search.set_embedding_model(target.tier)
        reranker.set_active_tier(target.tier)
        os.environ["TRUEMEMORY_EMBED_MODEL"] = target.tier

    def _try_legacy_noop(self, path: Path, target_tier: str) -> bool:
        """Recognize an unchanged legacy tier without competing for maintenance."""
        try:
            path.stat()
        except FileNotFoundError:
            return False
        from truememory import vector_search, reranker
        from truememory.embedding_target import EmbeddingTarget
        from truememory.tier_config import get_tier_config
        from truememory.tier_switch.activation import _guard_storage, _state
        from truememory.tier_switch.job import read_selected_job_marker
        from truememory.tier_switch.projection import CONFIG_WRITE_LOCK, ConfigFileLock, _guard, _read_config
        from truememory.tier_switch.runtime import _legacy_key, serving_operation
        from truememory.tier_switch.serving import operation_lease

        target = EmbeddingTarget.capture(target_tier)
        if (vector_search._frozen_embedding_target is not None
                or getattr(vector_search, "_runtime_policy_tier", None) is not None):
            return False
        lock_path = self._config_lock_path or self._config_path.with_name(self._config_path.name + ".lock")
        # The shared lease permits ordinary maintenance and pins the process
        # identity. All database reads use one owned read-only snapshot.
        with operation_lease(_legacy_key(), cancelled=self._cancel_requested):
            conn = sqlite3.connect(path.absolute().as_uri() + "?mode=ro", uri=True)
            try:
                conn.execute("BEGIN")
                if conn.execute("SELECT 1 FROM main.sqlite_master WHERE name NOT LIKE 'sqlite_%' LIMIT 1").fetchone() is None:
                    return False
                _guard_storage(conn)
                state = _state(conn)
                if state.selection is not None or state.legacy_policy is not None or state.intent is not None:
                    return False
                if read_selected_job_marker(conn) is not None:
                    return False
                with ConfigFileLock(lock_path, strict=True), CONFIG_WRITE_LOCK:
                    guard = _guard(_read_config(self._config_path))
                    if guard.generation is not None or guard.tier != target.tier:
                        return False
                    old_target, _ = self._serving_identity(conn, state, guard)
                    if old_target != target:
                        return False
                    with serving_operation(conn, cancelled=self._cancel_requested) as operation:
                        if (operation.tier != target.tier
                                or operation.reranker_id != get_tier_config(target.tier)["reranker"]
                                or reranker._frozen_reranker_id is not None):
                            raise TierSwitchUnsupportedError("Legacy runtime identity must be coherent before a tier no-op")
                        self._served_tier = target.tier
                        self._last_action = "noop"
                        self._last_outcome = "complete"
                        return True
            finally:
                try:
                    if conn.in_transaction:
                        conn.rollback()
                finally:
                    conn.close()

    def _bootstrap_empty_configuration(self, path: Path, target_tier: str, force: bool) -> bool:
        """Configure a proven uninitialized store without inventing vector state."""
        from truememory.embedding_target import EmbeddingTarget
        from truememory.tier_switch.projection import CONFIG_WRITE_LOCK, ConfigFileLock, _read_config, _write_config
        from truememory.tier_switch.serving import exclusive_activation

        if force:
            return False
        target = EmbeddingTarget.capture(target_tier)
        with exclusive_activation(cancelled=self._cancel_requested):
            try:
                path.stat()
                present = True
            except FileNotFoundError:
                present = False
            if present:
                conn = sqlite3.connect(path.absolute().as_uri() + "?mode=ro", uri=True)
                try:
                    if conn.execute("SELECT 1 FROM main.sqlite_master WHERE name NOT LIKE 'sqlite_%' LIMIT 1").fetchone():
                        return False
                finally:
                    conn.close()
            from truememory import vector_search, reranker
            from truememory.tier_config import get_tier_config
            if (getattr(vector_search, "_frozen_embedding_target", None) is not None
                    or getattr(vector_search, "_runtime_policy_tier", None) is not None
                    or getattr(reranker, "_frozen_reranker_id", None) is not None):
                raise TierSwitchUnsupportedError("First-run configuration cannot inherit another database's controlled runtime")
            served_tier = vector_search.resolve_tier()
            observed = EmbeddingTarget.capture(served_tier)
            observed_reranker = reranker.get_current_reranker_name()
            coherent = ((vector_search.EMBEDDING_MODEL, vector_search._embedding_dim) == (observed.model_id, observed.dimension)
                        and observed_reranker == get_tier_config(observed.tier)["reranker"])
            self._served_tier = served_tier if coherent else None
            ready = coherent and served_tier == target.tier
            if not ready and (vector_search._model is not None or reranker._model is not None):
                raise TierSwitchUnsupportedError("First-run tier configuration requires unloaded model slots; restart before changing tier")
            lock_path = self._config_lock_path or self._config_path.with_name(self._config_path.name + ".lock")
            with ConfigFileLock(lock_path, strict=True), CONFIG_WRITE_LOCK:
                config = _read_config(self._config_path)
                if "tier_activation_generation" in config:
                    raise TierSwitchUnsupportedError("An activation-tagged config cannot bootstrap a missing store")
                self._last_action = "onboarding"
                old_env = os.environ.get("TRUEMEMORY_EMBED_MODEL")
                if not ready and not coherent:
                    raise TierSwitchUnsupportedError("First-run legacy identity must be coherent before lazy projection")
                requested_config = dict(config, tier=target_tier)
                try:
                    if not ready:
                        self._last_outcome = "activation_pending"
                        self._apply_bootstrap_identity(target, get_tier_config(target.tier)["reranker"])
                        if ((vector_search.EMBEDDING_MODEL, vector_search._embedding_dim) != (target.model_id, target.dimension)
                                or vector_search.resolve_tier() != target.tier
                                or reranker.get_current_reranker_name() != get_tier_config(target.tier)["reranker"]):
                            raise TierSwitchUnsupportedError("First-run lazy model identity did not reconcile")
                        self._served_tier = target.tier
                    _write_config(self._config_path, requested_config)
                except BaseException:
                    # Replace can fail before or after publication. Restore only
                    # when a locked reread proves the original file remains.
                    if not ready and _read_config(self._config_path) == config:
                        self._apply_bootstrap_identity(observed, observed_reranker)
                        if old_env is None:
                            os.environ.pop("TRUEMEMORY_EMBED_MODEL", None)
                        else:
                            os.environ["TRUEMEMORY_EMBED_MODEL"] = old_env
                        self._served_tier = observed.tier
                    raise
            self._last_outcome = "complete"
            return True

    def _prepare_job(self, conn: sqlite3.Connection, target_tier: str, force: bool, *, backup_path: Path | None = None) -> "tuple[TierJob, ActivationIntent, int] | None":
        from truememory.embedding_target import EmbeddingTarget
        from truememory.rebuild_source import ensure_rebuild_tracking
        from truememory.tier_config import get_tier_config
        from truememory.tier_switch.activation import (
            commit_config_only_transition, read_activation_state, stage_activation_intent,
        )
        from truememory.tier_switch.job import read_selected_job_marker, select_tier_job
        from truememory.tier_switch.projection import capture_config_guard
        from truememory.tier_switch.serving import exclusive_activation
        from truememory.vector_search import init_prepared_target_tables

        self._last_outcome = "preparing"
        self._last_action = None
        if force:
            raise TierSwitchUnsupportedError("Active force/reset requires an alternate inactive pair and is not supported")
        target = EmbeddingTarget.capture(target_tier)
        reranker_id = get_tier_config(target_tier)["reranker"]
        with exclusive_activation(cancelled=self._cancel_requested):
            state = read_activation_state(conn)
            guard = capture_config_guard(config_path=self._config_path, lock_path=self._config_lock_path)
            previous = state.selection or state.legacy_policy
            if previous is not None:
                self._served_tier = previous.target.tier
            if previous is not None and not previous.config_acknowledged:
                self._last_action = "config_only" if type(state.intent).__name__ == "PolicyIntent" else "reconcile"
                if previous.target != target:
                    raise TierSwitchUnsupportedError("The committed tier must be reconciled before another switch")
                self._project_result(conn, previous, state.intent, policy=state.legacy_policy is not None or type(state.intent).__name__ == "PolicyIntent")
                return None
            old_target, old_tables = self._serving_identity(conn, state, guard)
            self._served_tier = old_target.tier
            if old_target == target:
                self._last_action = "noop"
                if previous is not None:
                    from truememory.tier_switch.runtime import apply_runtime_policy
                    self._last_outcome = "activation_pending"
                    apply_runtime_policy(previous)
                self._last_outcome = "complete"
                return None
            if old_target.identity == target.identity and {old_target.tier, target.tier} == {"base", "pro"}:
                self._last_action = "config_only"
                self._last_outcome = "activation_pending"
                self._served_tier = None
                result = commit_config_only_transition(
                    conn, expected_selection_generation=state.selection.generation if state.selection else None,
                    expected_intent_id=state.intent.intent_id if state.intent else None, target=target,
                    reranker_id=reranker_id, expected_config=guard, legacy_pair=old_tables,
                )
                self._project_result(conn, result, read_activation_state(conn).intent, policy=True)
                return None
            if set(old_tables).intersection(target.tables) or old_target.tier_group == target.tier_group:
                raise TierSwitchUnsupportedError("Rebuilding the serving pair requires an alternate inactive slot")
            ok, message = preflight_ram_check(target.tier_group)
            if not ok:
                raise RuntimeError(message)
            marker = read_selected_job_marker(conn)
            if marker is not None and marker.state == "selected":
                raise TierSwitchUnsupportedError("A selected job still owns the inactive pair; cancel or reconcile it first")
            self._check_inactive_target(conn, target)
            self._guard_status(conn)
            self._last_action = "rebuild"
            ensure_rebuild_tracking(conn)
            init_prepared_target_tables(conn, target)
            status_id = self._create_status_row(conn, target.tier_group, target.tier, "streamed", 0,
                                                backup_path=backup_path)
            self._active_status_id = status_id
            job = select_tier_job(conn, target, previous_job_id=marker.job_id if marker else None)
            self._active_job_id = job.job_id
            intent = stage_activation_intent(conn, job,
                expected_generation=state.selection.generation if state.selection else None,
                reranker_id=reranker_id, expected_config=guard, status_id=status_id)
            return job, intent, status_id

    @staticmethod
    def _check_inactive_target(conn: sqlite3.Connection, target: object) -> None:
        """Refuse incompatible or unreceipted rows before tracking/table writes."""
        from truememory.tier_switch.activation import _guard_pair
        from truememory.tier_switch.source import read_tier_progress

        existing = []
        nonempty = False
        for name in target.tables:
            if conn.execute("SELECT 1 FROM main.sqlite_master WHERE name=? COLLATE NOCASE", (name,)).fetchone():
                _guard_pair(conn, target, tables=(name,))
                existing.append(name)
                nonempty |= conn.execute(f'SELECT 1 FROM main."{name}" LIMIT 1').fetchone() is not None
        if nonempty:
            if len(existing) != 2 or read_tier_progress(conn, model_id=target.model_id,
                    dimension=target.dimension, targets=target.tables) is None:
                raise TierSwitchUnsupportedError("Nonempty inactive vectors require their exact paired receipt")

    @staticmethod
    def _serving_identity(conn: sqlite3.Connection, state: "ActivationState", guard: "ConfigGuard") -> tuple:
        from truememory.embedding_target import EmbeddingTarget
        from truememory import vector_search

        previous = state.selection or state.legacy_policy
        if previous is not None:
            if guard.tier != previous.target.tier or guard.generation != previous.generation:
                raise TierSwitchUnsupportedError("Config and authoritative serving generation disagree")
            return previous.target, previous.tables
        target = EmbeddingTarget.capture(guard.tier)
        if vector_search.EMBEDDING_MODEL != target.model_id or vector_search._embedding_dim != target.dimension:
            raise TierSwitchUnsupportedError("Legacy model configuration must be coherent before tier switching")
        tables = (vector_search._active_vec_table(conn), vector_search._active_sep_table(conn))
        if tables not in (target.tables, ("vec_messages", "vec_messages_sep")):
            raise TierSwitchUnsupportedError("Legacy serving table pair is ambiguous")
        require_identity = target.tier in {"base", "pro", "custom"}
        if require_identity or conn.execute("SELECT 1 FROM main.messages LIMIT 1").fetchone() is not None:
            for key, expected in (("embed_model", target.model_id), ("embed_dim", str(target.dimension))):
                row = conn.execute("SELECT typeof(value)='text' AND value=? FROM main.metadata WHERE key=?", (expected, key)).fetchone()
                if row is None or row[0] != 1:
                    raise TierSwitchUnsupportedError("Legacy source has no coherent serving embedding identity")
        return target, tables

    def _run_job(self, conn: sqlite3.Connection, job: "TierJob", intent: "ActivationIntent", status_id: int,
                 progress_callback: "StatusCallback | None" = None) -> bool:
        from truememory.tier_switch.activation import read_activation_state
        from truememory.tier_switch.job import check_selected_tier_job, end_selected_tier_job
        from truememory.tier_switch.source import initialize_tier_source, plan_tier_source
        from truememory.tier_switch.serving import exclusive_activation

        check_selected_tier_job(conn, job)
        if read_activation_state(conn).intent != intent:
            raise TierSwitchUnsupportedError("Staged activation changed before worker admission")
        _load_tier_extension(conn)
        device = "cpu" if job.target.tier_group == "edge" else _detect_device()
        worker = RebuildWorker(conn, job.target.tier, job.target.tier_group,
                               DynamicThrottler(device=device), status_id, progress_callback)
        with self._state_lock:
            self._active_worker = worker
            if self._cancel_requested.is_set():
                worker.cancel()
        expires = time.monotonic() + 9000
        plan = plan_tier_source(conn, model_id=job.target.model_id, dimension=job.target.dimension, targets=job.target.tables)
        if plan.action != "resume" and plan.action != "complete":
            plan = initialize_tier_source(conn, plan)
        result = worker.run_source(job.target, plan, timeout=max(0.001, expires - time.monotonic()))
        if (result.status != "complete" or not result.captured_range_complete
                or not result.current_source_complete or self._cancel_requested.is_set()):
            outcome = "cancelled" if self._cancel_requested.is_set() or result.status == "cancelled" else "failed"
            end_selected_tier_job(conn, job, outcome=outcome)
            self._terminal_status(conn, status_id, outcome, result.plan.manifest.consumed, result.plan.manifest.total)
            self._last_outcome = outcome
            return False
        with exclusive_activation(deadline=expires, cancelled=self._cancel_requested):
            self._finalize_rebuild(conn, job.target.tier, job.target.tier_group, intent=intent, job=job, plan=result.plan)
        self._last_outcome = "activation_pending"
        self._terminal_status(conn, status_id, "complete", result.plan.manifest.consumed, result.plan.manifest.total)
        self._last_outcome = "complete"
        return True

    def _project_result(self, conn: sqlite3.Connection, result: "TierSelection | LegacyTierPolicy",
                        intent: object, *, policy: bool = False) -> None:
        from truememory.tier_switch.activation import acknowledge_config, acknowledge_policy_config, LegacyTierPolicy
        from truememory.tier_switch.projection import mirror_activation
        from truememory.tier_switch.runtime import apply_frozen_selection, apply_runtime_policy

        self._last_outcome = "activation_pending"
        self._served_tier = result.target.tier
        mirror_activation(result, expected_config=intent.expected_config, config_path=self._config_path, lock_path=self._config_lock_path)
        if policy:
            apply_runtime_policy(result)
        else:
            apply_frozen_selection(result)
        if type(result) is LegacyTierPolicy:
            acknowledge_policy_config(conn, generation=result.generation)
        else:
            acknowledge_config(conn, generation=result.generation)
        self._served_tier = result.target.tier
        self._last_outcome = "complete"

    @staticmethod
    def _terminal_status(conn: sqlite3.Connection, status_id: int, status: str, processed: int, total: int) -> None:
        from truememory.tier_switch.activation import _transaction
        with _transaction(conn, write=True):
            RebuildManager._guard_status(conn)
            conn.execute("UPDATE rebuild_status SET status=?, processed_messages=?, total_messages=?, progress_pct=?, completed_at=? WHERE id=?",
                         (status, processed, total, 100.0 if status == "complete" else 100.0 * processed / max(1,total), time.time(), status_id))

    def get_status(self, status_id: int = 0) -> dict:
        """Read one journal/marker/manifest snapshot, without corpus audits."""
        from truememory.tier_switch.activation import ActivationIntent, _state, _transaction
        from truememory.tier_switch.job import read_selected_job_marker
        from truememory.tier_switch.source import read_tier_progress

        try:
            conn = _open_db(self._active_db_path)
        except Exception:
            return {"status": "unknown", "error": "cannot open db"}
        try:
            with _transaction(conn, write=False):
                self._guard_status(conn)
                sid = status_id or self._active_status_id
                columns = ("id", "tier_group", "target_tier", "status", "action", "total_messages", "processed_messages",
                           "progress_pct", "eta_seconds", "batch_size", "throughput_ips", "ram_pct", "pressure", "error",
                           "started_at", "completed_at", "backup_path", "last_heartbeat")
                text_columns = {"tier_group", "target_tier", "status", "action", "error", "backup_path"}
                fields = ", ".join(
                    f"CASE WHEN typeof({name})='text' AND length(CAST({name} AS BLOB))<=4096 THEN {name} ELSE NULL END"
                    if name in text_columns else
                    f"CASE WHEN typeof({name}) IN ('integer','real') THEN {name} ELSE NULL END"
                    for name in columns
                )
                row = conn.execute(f"SELECT {fields} FROM main.rebuild_status WHERE id=?", (sid,)).fetchone() if sid else conn.execute(
                    f"SELECT {fields} FROM main.rebuild_status ORDER BY id DESC LIMIT 1").fetchone()
                if not row:
                    return {"status": "no_rebuild_found"}
                result = dict(zip(columns, row))
                state = _state(conn)
                intent = state.intent
                if type(intent) is ActivationIntent and intent.status_id == result["id"]:
                    if (intent.target.tier, intent.target.tier_group) != (result["target_tier"], result["tier_group"]):
                        raise TierSwitchUnsupportedError("Status row and frozen intent disagree")
                    if intent.state != "staged":
                        result["status"] = "complete" if state.selection.config_acknowledged else "activation_pending"
                        result["activated"] = state.selection.config_acknowledged
                    else:
                        marker = read_selected_job_marker(conn)
                        if marker is None or (marker.job_id, marker.target, marker.tracker, marker.source_epoch, marker.source_schema_signature) != (
                            intent.job_id, intent.target, intent.tracker, intent.source_epoch, intent.source_schema_signature):
                            raise TierSwitchUnsupportedError("Status job no longer owns its source progress")
                        progress = read_tier_progress(conn, model_id=intent.target.model_id,
                                                     dimension=intent.target.dimension, targets=intent.target.tables)
                        if progress is not None:
                            if (progress.tracker, progress.source_epoch, progress.source_schema_signature) != (
                                intent.tracker, intent.source_epoch, intent.source_schema_signature):
                                raise TierSwitchUnsupportedError("Progress source belongs to another job")
                            result.update(processed_messages=progress.processed, total_messages=progress.total,
                                          progress_pct=100.0 * progress.processed / max(1,progress.total))
                        result["activated"] = False
                        result["job_id"] = marker.job_id
                        result["recovery"] = (
                            f"If interrupted, run truememory-ingest cancel-rebuild --status-id {result['id']} "
                            f"--job-id {marker.job_id}, then retry the original tier request explicitly. "
                            "Use the same database path. A live maintenance owner prevents cancellation."
                        )
                        if marker.state != "selected":
                            result["status"] = marker.state
                        elif self._last_outcome == "interrupted" and result["id"] == self._active_status_id:
                            result["status"] = "interrupted"
                elif result["status"] in {"pending", "running"}:
                    marker = read_selected_job_marker(conn)
                    result["status"] = "superseded" if intent is not None else "interrupted"
                    result["activated"] = False
                    if marker is not None and marker.state == "selected":
                        result["selected_job_id"] = marker.job_id
                        result["recovery"] = "This status does not prove selected marker ownership. Do not use its ID with cancel-rebuild; a matching journal/status binding is required."
                return result
        except Exception as error:
            return {"status": "unknown", "error": type(error).__name__}
        finally:
            conn.close()

    def cancel(self, status_id: int = 0, *, expected_job_id: str | None = None,
               db_path: Path | None = None) -> dict:
        """Signal local work, or explicitly cancel one durable interrupted marker."""
        from truememory.maintenance import maintenance_owner
        from truememory.tier_switch.activation import ActivationIntent, _transaction, read_activation_state
        from truememory.tier_switch.job import _cancel_policy_job_in_writer, read_selected_job_marker

        with self._state_lock:
            active = self._claimed or self._active_thread is not None or self._active_worker is not None
            if active:
                if expected_job_id is not None and expected_job_id != self._active_job_id:
                    raise TierSwitchUnsupportedError("Cancellation job does not identify the active job")
                if status_id and status_id != self._active_status_id:
                    raise TierSwitchUnsupportedError("Cancellation status does not identify the active job")
                self._cancel_requested.set()
                if self._active_worker is not None:
                    self._active_worker.cancel()
                return {"status": "cancellation_requested", "status_id": self._active_status_id}
        path = db_path or self._active_db_path or _get_db_path()
        with maintenance_owner(path):
            conn = _open_db(path)
            try:
                state = read_activation_state(conn)
                marker = read_selected_job_marker(conn)
                if marker is None or marker.state != "selected":
                    return {"status": "no_selected_job"}
                intent = state.intent
                bound = (type(intent) is ActivationIntent and intent.state == "staged"
                         and (intent.job_id, intent.target, intent.tracker, intent.source_epoch, intent.source_schema_signature)
                         == (marker.job_id, marker.target, marker.tracker, marker.source_epoch, marker.source_schema_signature))
                if expected_job_id != marker.job_id or (status_id and (not bound or intent.status_id != status_id)):
                    raise TierSwitchUnsupportedError("Interrupted cancellation requires the exact current job_id and matching status")
                with _transaction(conn, write=True):
                    _cancel_policy_job_in_writer(conn, job_id=marker.job_id, target=marker.target,
                        tracker=marker.tracker, source_epoch=marker.source_epoch,
                        source_schema_signature=marker.source_schema_signature)
                return {"status": "cancelled", "job_id": marker.job_id}
            finally:
                conn.close()

    def _rebuild_thread(self, job: "TierJob", intent: "ActivationIntent", status_id: int) -> None:
        """Reopen the exact accepted job under worker-owned maintenance."""
        from truememory.tier_switch.job import open_selected_tier_job

        try:
            with open_selected_tier_job(job) as conn:
                self._run_job(conn, job, intent, status_id)
        except Exception:
            # Do not issue status SQL on a possibly uncertain connection.
            if self._last_outcome == "activation_pending":
                self._refresh_committed_serving(job.database_path)
            else:
                self._last_outcome = "interrupted"
            log.exception("Background tier activation did not complete")
        finally:
            with self._state_lock:
                self._active_worker = None
                self._active_thread = None

    def _finalize_rebuild(self, conn: sqlite3.Connection, target_tier: str, to_group: str, *,
                          intent: "ActivationIntent | None" = None, job: "TierJob | None" = None,
                          plan: "TierSourcePlan | None" = None) -> None:
        """A paired current-source certificate is the only activation authority."""
        from truememory.tier_switch.activation import commit_certified_selection

        if intent is None or job is None or plan is None or (target_tier, to_group) != (job.target.tier, job.target.tier_group):
            raise TierSwitchUnsupportedError("Tier activation requires an exact paired completion certificate")
        self._last_outcome = "activation_pending"
        self._served_tier = None
        result = commit_certified_selection(conn, intent, job, plan)
        self._project_result(conn, result, intent)

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
        if conn is not None:
            raise TierSwitchUnsupportedError("Legacy config-only helper cannot activate a database tier")
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
        *, backup_path: Path | None = None,
    ) -> int:
        """Insert a new rebuild_status row and return its id."""
        from truememory.tier_switch.activation import _transaction
        now = time.time()
        with _transaction(conn, write=True):
            self._guard_status(conn)
            cursor = conn.execute(
                "INSERT INTO rebuild_status "
                "(tier_group, target_tier, status, action, total_messages, started_at, last_heartbeat, backup_path) "
                "VALUES (?, ?, 'running', ?, ?, ?, ?, ?)",
                (to_group, target_tier, action, total, now, now, str(backup_path) if backup_path is not None else None),
            )
            status_id = cursor.lastrowid
        return status_id

    @staticmethod
    def _guard_status(conn: sqlite3.Connection) -> None:
        from truememory.tier_switch.activation import _normalized, _schema_sql
        if conn.execute("SELECT 1 FROM temp.sqlite_master WHERE name='rebuild_status' COLLATE NOCASE LIMIT 1").fetchone():
            raise TierSwitchUnsupportedError("TEMP status shadows are unsupported")
        expected = """CREATE TABLE rebuild_status (
            id INTEGER PRIMARY KEY AUTOINCREMENT, tier_group TEXT NOT NULL, target_tier TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', action TEXT, total_messages INTEGER DEFAULT 0,
            processed_messages INTEGER DEFAULT 0, progress_pct REAL DEFAULT 0, eta_seconds REAL DEFAULT 0,
            batch_size INTEGER DEFAULT 0, throughput_ips REAL DEFAULT 0, ram_pct REAL DEFAULT 0,
            pressure REAL, error TEXT, started_at REAL, completed_at REAL, backup_path TEXT, last_heartbeat REAL
        )"""
        if _normalized(_schema_sql(conn, "rebuild_status")) != _normalized(expected):
            raise TierSwitchUnsupportedError("Unsupported rebuild status schema")
        if conn.execute("SELECT 1 FROM main.sqlite_master WHERE type='trigger' AND tbl_name='rebuild_status' COLLATE NOCASE LIMIT 1").fetchone():
            raise TierSwitchUnsupportedError("Status triggers cannot participate in tier publication")


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
