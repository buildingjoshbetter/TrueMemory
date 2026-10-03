"""Opt-in encrypted cloud backup and restore for TrueMemory (issue #199, Phase 1).

One-way, local to S3-compatible object storage:

    LOCAL DATABASE
    -> consistent SQLite snapshot (Online Backup API)
    -> integrity validation
    -> AES-256-GCM encryption
    -> upload of the encrypted artifact only

    S3 OBJECT -> download -> decrypt -> SQLite validation
    -> pre-restore safety snapshot -> safe replacement -> reopen validation

This is deliberately NOT sync: no bidirectional sync, conflict resolution,
multi-device coordination, categorization, sharing, or team features. The
feature is inert unless cloud backup is explicitly enabled and configured; every
local-only code path is unchanged.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from truememory.backup_crypto import (
    decrypt_file,
    encrypt_file,
    generate_key,
    load_key,
)
from truememory.backup_errors import (
    BackupConfigError,
    BackupNotConfiguredError,
    BackupRestoreError,
    BackupSnapshotError,
    BackupValidationError,
)
from truememory.backup_store import S3ObjectStore
from truememory.storage import _backup_database, snapshot_sqlite

log = logging.getLogger(__name__)

_DEFAULT_DB = Path.home() / ".truememory" / "memories.db"
_BACKUP_SUFFIX = ".tmbackup"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_REQUIRED_TABLES = frozenset({"messages"})

__all__ = [
    "BackupConfig",
    "BackupResult",
    "RestoreResult",
    "backup_database",
    "generate_key",
    "resolve_db_path",
    "restore_database",
    "validate_sqlite_database",
]


@dataclass(frozen=True)
class BackupConfig:
    """Opt-in cloud backup settings.

    Built from ``TRUEMEMORY_BACKUP_*`` environment variables via
    :meth:`from_env`, or constructed explicitly by Python callers. Secret
    fields are excluded from ``repr()`` so a logged config cannot leak them.
    """

    enabled: bool = False
    bucket: str = ""
    endpoint_url: str | None = None
    region: str | None = None
    prefix: str = "truememory/"
    object_key: str | None = None
    access_key_id: str | None = field(default=None, repr=False)
    secret_access_key: str | None = field(default=None, repr=False)
    encryption_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> BackupConfig:
        """Load settings from ``TRUEMEMORY_BACKUP_*`` environment variables."""
        env = os.environ if environ is None else environ
        return cls(
            enabled=_env_bool(env.get("TRUEMEMORY_BACKUP_ENABLED")),
            bucket=(env.get("TRUEMEMORY_BACKUP_BUCKET") or "").strip(),
            endpoint_url=_clean(env.get("TRUEMEMORY_BACKUP_ENDPOINT")),
            region=_clean(env.get("TRUEMEMORY_BACKUP_REGION")),
            prefix=env.get("TRUEMEMORY_BACKUP_PREFIX") or "truememory/",
            object_key=_clean(env.get("TRUEMEMORY_BACKUP_OBJECT")),
            access_key_id=_clean(env.get("TRUEMEMORY_BACKUP_ACCESS_KEY_ID")),
            secret_access_key=_clean(env.get("TRUEMEMORY_BACKUP_SECRET_ACCESS_KEY")),
            encryption_key=_clean(env.get("TRUEMEMORY_BACKUP_KEY")),
        )

    def require_enabled(self) -> None:
        """Raise unless cloud backup is explicitly enabled."""
        if not self.enabled:
            raise BackupNotConfiguredError(
                "cloud backup is not enabled. Set TRUEMEMORY_BACKUP_ENABLED=1 "
                "and configure TRUEMEMORY_BACKUP_BUCKET plus "
                "TRUEMEMORY_BACKUP_KEY to opt in. See docs/cloud-backup.md. "
                "Without this, TrueMemory stays fully local."
            )

    def validate(self) -> None:
        """Validate the opt-in settings needed before any network call."""
        self.require_enabled()
        if not self.bucket:
            raise BackupConfigError(
                "TRUEMEMORY_BACKUP_BUCKET is required when cloud backup is enabled"
            )
        if bool(self.access_key_id) != bool(self.secret_access_key):
            raise BackupConfigError(
                "TRUEMEMORY_BACKUP_ACCESS_KEY_ID and "
                "TRUEMEMORY_BACKUP_SECRET_ACCESS_KEY must be set together "
                "(or both omitted to use the provider credential chain)"
            )

    def resolve_object_key(self, db_path: str | Path) -> str:
        """Return the object key for *db_path* (explicit key wins)."""
        if self.object_key:
            return self.object_key
        prefix = self.prefix or ""
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        return f"{prefix}{Path(db_path).name}{_BACKUP_SUFFIX}"


@dataclass(frozen=True)
class BackupResult:
    """Non-sensitive metadata about a completed backup."""

    bucket: str
    key: str
    encrypted_bytes: int
    format_version: int
    db_path: str


@dataclass(frozen=True)
class RestoreResult:
    """Non-sensitive metadata about a completed restore."""

    bucket: str
    key: str
    encrypted_bytes: int
    format_version: int
    db_path: str
    safety_backup: str | None


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _env_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _TRUE_VALUES


def resolve_db_path(environ: dict[str, str] | None = None) -> Path:
    """Resolve the local database path the same way the MCP server does.

    ``TRUEMEMORY_DB_PATH`` wins, ``TRUEMEMORY_DB`` is the legacy alias, and the
    fallback is ``~/.truememory/memories.db``.
    """
    env = os.environ if environ is None else environ
    raw = env.get("TRUEMEMORY_DB_PATH") or env.get("TRUEMEMORY_DB")
    if raw:
        return Path(os.path.expanduser(raw))
    return _DEFAULT_DB


def validate_sqlite_database(path: str | Path) -> None:
    """Reject anything that is not a structurally usable TrueMemory database.

    Runs SQLite's ``quick_check`` (the same cheap check ``create_db`` performs
    at open) and requires the core ``messages`` table to exist. Intended to run
    on a snapshot or downloaded copy BEFORE it is ever allowed to replace the
    live database.
    """
    conn = None
    try:
        conn = sqlite3.connect(str(path))
        row = conn.execute("PRAGMA quick_check(1)").fetchone()
        if row is None or row[0] != "ok":
            detail = row[0] if row else "no result"
            raise BackupValidationError(f"SQLite integrity check failed: {detail}")
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        missing = _REQUIRED_TABLES - tables
        if missing:
            raise BackupValidationError(
                "backup is not a TrueMemory database (missing table(s): "
                + ", ".join(sorted(missing))
                + ")"
            )
    except sqlite3.DatabaseError as exc:
        raise BackupValidationError(f"not a valid SQLite database: {exc}") from exc
    finally:
        if conn is not None:
            with contextlib.suppress(sqlite3.Error):
                conn.close()


def _build_store(config: BackupConfig, store: Any | None) -> Any:
    if store is not None:
        return store
    return S3ObjectStore(
        config.bucket,
        endpoint_url=config.endpoint_url,
        region=config.region,
        access_key_id=config.access_key_id,
        secret_access_key=config.secret_access_key,
    )


def backup_database(
    config: BackupConfig | None = None,
    *,
    db_path: str | Path | None = None,
    key: str | bytes | None = None,
    store: Any | None = None,
) -> BackupResult:
    """Snapshot, validate, encrypt, and upload the local database.

    Uploads only the encrypted artifact; the plaintext snapshot is deleted
    before the upload starts and the entire staging directory is removed on
    exit, success or failure. ``store`` is injectable for tests and advanced
    callers (any object with ``upload_file``/``download_file``).
    """
    cfg = config if config is not None else BackupConfig.from_env()
    cfg.validate()
    db = Path(db_path) if db_path is not None else resolve_db_path()
    if str(db) == ":memory:":
        raise BackupSnapshotError("cannot back up an in-memory database")
    if not db.is_file():
        raise BackupSnapshotError(f"database not found: {db}")
    enc_key = load_key(key if key is not None else cfg.encryption_key)
    object_key = cfg.resolve_object_key(db)
    obj_store = _build_store(cfg, store)

    tmp_dir = Path(tempfile.mkdtemp(prefix="truememory-backup-"))
    try:
        snapshot = tmp_dir / "snapshot.db"
        try:
            snapshot_sqlite(db, snapshot)
        except (sqlite3.Error, OSError) as exc:
            raise BackupSnapshotError(
                f"could not create a consistent snapshot of {db}: {exc}"
            ) from exc
        validate_sqlite_database(snapshot)

        encrypted = tmp_dir / "snapshot.tmbak"
        version = encrypt_file(snapshot, encrypted, enc_key)
        # The plaintext snapshot MUST be deleted before anything is uploaded.
        # If deletion fails, abort: uploading while plaintext lingers on disk
        # would be a silent security regression, and carrying on would report
        # a successful backup while a plaintext copy remains behind.
        try:
            snapshot.unlink()
        except OSError as exc:
            raise BackupSnapshotError(
                f"could not delete the plaintext snapshot at {snapshot}; the "
                "upload was aborted (nothing was sent to cloud storage). "
                "Remove the file manually."
            ) from exc
        size = encrypted.stat().st_size

        obj_store.upload_file(encrypted, object_key)
        log.info(
            "Uploaded encrypted backup to s3://%s/%s (%d bytes)",
            cfg.bucket,
            object_key,
            size,
        )
        return BackupResult(
            bucket=cfg.bucket,
            key=object_key,
            encrypted_bytes=size,
            format_version=version,
            db_path=str(db),
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def restore_database(
    config: BackupConfig | None = None,
    *,
    db_path: str | Path | None = None,
    key: str | bytes | None = None,
    store: Any | None = None,
) -> RestoreResult:
    """Download, decrypt, validate, and safely restore a cloud backup.

    Every step that can fail (download, decryption, SQLite validation, safety
    snapshot) happens BEFORE the live database is touched. The current database
    is then protected by a consistent pre-restore snapshot, stale SQLite
    sidecar files are removed, and the restored file atomically replaces the
    live database in the same directory. A failure before replacement leaves
    the current database untouched; a failure during or after replacement
    triggers automatic rollback to the pre-restore snapshot.
    """
    cfg = config if config is not None else BackupConfig.from_env()
    cfg.validate()
    db = Path(db_path) if db_path is not None else resolve_db_path()
    if str(db) == ":memory:":
        raise BackupRestoreError("cannot restore into an in-memory database")
    enc_key = load_key(key if key is not None else cfg.encryption_key)
    object_key = cfg.resolve_object_key(db)
    obj_store = _build_store(cfg, store)

    tmp_dir = Path(tempfile.mkdtemp(prefix="truememory-restore-"))
    staged_base: str | None = None
    safety_path: str | None = None
    try:
        downloaded = tmp_dir / "backup.tmbak"
        obj_store.download_file(object_key, downloaded)
        size = downloaded.stat().st_size

        db.parent.mkdir(parents=True, exist_ok=True)
        fd, staged_name = tempfile.mkstemp(
            prefix=f".{db.name}.restore-", suffix=".tmp", dir=str(db.parent)
        )
        os.close(fd)
        staged_base = staged_name
        staged = Path(staged_base)

        version = decrypt_file(downloaded, staged, enc_key)
        validate_sqlite_database(staged)

        if db.exists():
            safety = _backup_database(db, reason="pre-restore")
            if safety is None:
                raise BackupRestoreError(
                    "could not create a safety snapshot of the current "
                    "database; restore aborted and the current database is "
                    "unchanged"
                )
            safety_path = str(safety)

        _remove_sidecars(db, strict=True)
        try:
            _atomic_replace(staged, db)
        except OSError as exc:
            _recover_safety_backup(safety_path, db)
            detail = (
                f"The pre-restore safety backup is at {safety_path}."
                if safety_path
                else "No local database existed before this restore."
            )
            raise BackupRestoreError(
                f"could not replace {db} with the restored database "
                f"({type(exc).__name__}: {exc}). {detail} Close all TrueMemory "
                "processes and retry."
            ) from exc

        with contextlib.suppress(OSError):
            os.chmod(db, 0o600)
        try:
            validate_sqlite_database(db)
        except BackupValidationError as exc:
            # The replacement already happened, so a plain validation error
            # here must not leave the user with a silently swapped database:
            # roll back to the pre-restore snapshot when one exists.
            _recover_safety_backup(safety_path, db)
            detail = (
                f"Recovery was attempted using the pre-restore safety backup "
                f"at {safety_path} (see logs)."
                if safety_path
                else "No pre-existing database existed to recover; inspect "
                "the target file manually."
            )
            raise BackupRestoreError(
                f"the restored database at {db} failed post-replacement "
                f"validation: {exc}. {detail}"
            ) from exc
        # Reopen-validation may create fresh -wal/-shm sidecars; a restored
        # database starts as a single clean file.
        _remove_sidecars(db, strict=False)

        log.info(
            "Restored database from s3://%s/%s (%d bytes)",
            cfg.bucket,
            object_key,
            size,
        )
        return RestoreResult(
            bucket=cfg.bucket,
            key=object_key,
            encrypted_bytes=size,
            format_version=version,
            db_path=str(db),
            safety_backup=safety_path,
        )
    finally:
        if staged_base is not None:
            # Sweep the staging file plus any SQLite sidecars staged validation
            # may have created. After a successful replace the base name no
            # longer exists (it became *db*); only true leftovers are removed.
            for suffix in ("", "-wal", "-shm", "-journal"):
                with contextlib.suppress(OSError):
                    Path(f"{staged_base}{suffix}").unlink(missing_ok=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _remove_sidecars(db: Path, *, strict: bool) -> None:
    """Remove stale SQLite ``-wal``/``-shm``/``-journal`` files for *db*.

    A WAL or journal belonging to the OLD database must never survive next to
    the restored file. Pre-replacement this is strict: if a sidecar cannot be
    removed (for example a live process holds it on Windows) the caller aborts
    with the original database still intact. Post-replacement it is
    best-effort.
    """
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(f"{db}{suffix}")
        try:
            sidecar.unlink(missing_ok=True)
        except OSError as exc:
            if strict:
                raise BackupRestoreError(
                    f"could not remove stale SQLite sidecar file {sidecar} "
                    f"({exc}). Close all TrueMemory processes and retry; the "
                    "current database was not replaced."
                ) from exc
            log.warning("Could not remove leftover SQLite sidecar %s: %s", sidecar, exc)


def _atomic_replace(src: Path, dst: Path, *, attempts: int = 10, delay: float = 0.03) -> None:
    """``os.replace`` with a short retry for Windows sharing violations.

    POSIX replaces over an open destination immediately. Windows raises
    ``PermissionError`` while any handle lacks ``FILE_SHARE_DELETE``; transient
    readers (antivirus, another short-lived query) clear within milliseconds.
    """
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


def _recover_safety_backup(safety_path: str | None, db: Path) -> None:
    """Best-effort copy of the pre-restore snapshot back over *db*.

    Used when the restore failed after the live database had already been
    (partially) mutated: replacement failure, or post-replacement validation
    failure. ``shutil.copyfile`` is used (not ``os.replace``) so this recovery
    path is independent of the replacement that just failed.
    """
    if not safety_path:
        return
    try:
        shutil.copyfile(safety_path, db)
    except OSError:
        log.error(
            "Restore failed and the pre-restore safety backup could not be "
            "copied back automatically; recover manually from %s",
            safety_path,
        )
    else:
        log.warning(
            "Restore could not complete; the current database was recovered "
            "from the pre-restore safety backup at %s",
            safety_path,
        )
