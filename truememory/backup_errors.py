"""Exception hierarchy for the opt-in cloud backup feature (issue #199).

Every failure path in the backup/restore flow raises a ``BackupError`` subclass
with an actionable message. Messages never include encryption keys, cloud
credentials, or database content.
"""

from __future__ import annotations


class BackupError(Exception):
    """Base class for every cloud backup/restore failure."""


class BackupNotConfiguredError(BackupError):
    """Cloud backup was requested but is disabled or unconfigured."""


class BackupConfigError(BackupError):
    """Cloud backup configuration is invalid (missing bucket, half credentials)."""


class BackupKeyError(BackupError, ValueError):
    """The user-supplied encryption key is missing or malformed.

    Subclasses ``ValueError`` so callers validating raw input can catch it as a
    value error, while still being a ``BackupError`` for the backup API.
    """


class BackupEncryptionError(BackupError):
    """Encrypting the snapshot failed."""


class BackupDecryptionError(BackupError):
    """Decrypting the backup failed (wrong key or corrupted/tampered data)."""


class BackupSnapshotError(BackupError):
    """A consistent SQLite snapshot could not be produced."""


class BackupValidationError(BackupError):
    """A snapshot or downloaded backup is not a usable TrueMemory database."""


class BackupStoreError(BackupError):
    """Upload/download against the S3-compatible object store failed."""


class BackupNotFoundError(BackupStoreError):
    """The requested backup object does not exist in the bucket."""


class BackupDependencyError(BackupStoreError):
    """An optional dependency required for cloud backup is not installed."""


class BackupRestoreError(BackupError):
    """Restore failed; the pre-restore database (if any) is left recoverable."""
