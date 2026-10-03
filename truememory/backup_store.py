"""Minimal S3-compatible object storage for TrueMemory cloud backups.

Phase 1 only needs upload and download. The adapter is provider-agnostic: any
endpoint that speaks the S3 API works (AWS S3, Cloudflare R2, MinIO, ...).
``boto3`` is an optional dependency (``pip install truememory[backup]``) and is
imported lazily so installations that never enable cloud backup do not need it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from truememory.backup_errors import (
    BackupDependencyError,
    BackupNotFoundError,
    BackupStoreError,
)

_DEFAULT_REGION = "us-east-1"
_NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


class S3ObjectStore:
    """Upload/download backup artifacts to an S3-compatible bucket.

    Args:
        bucket: Destination bucket name.
        endpoint_url: Custom endpoint for non-AWS providers (R2, MinIO). None
            uses the AWS S3 endpoint.
        region: Region name. S3-compatible providers usually ignore it, but
            boto3 requires one, so it defaults to ``us-east-1``.
        access_key_id / secret_access_key: Optional explicit credentials. When
            omitted, boto3's standard provider chain is used (environment
            variables, shared credentials file, instance role).
        client: Pre-built boto3-compatible client. Intended for tests and
            advanced callers; when None the client is created lazily.
    """

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        region: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not bucket:
            raise BackupStoreError("no backup bucket configured")
        self.bucket = bucket
        self.endpoint_url = endpoint_url or None
        self.region = region or _DEFAULT_REGION
        self.access_key_id = access_key_id or None
        self.secret_access_key = secret_access_key or None
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import boto3
        except ImportError as exc:
            raise BackupDependencyError(
                "cloud backup requires boto3, which is not installed. "
                "Install the optional extra with: pip install 'truememory[backup]'"
            ) from exc
        kwargs: dict[str, Any] = {"region_name": self.region}
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        if self.access_key_id:
            kwargs["aws_access_key_id"] = self.access_key_id
        if self.secret_access_key:
            kwargs["aws_secret_access_key"] = self.secret_access_key
        self._client = boto3.client("s3", **kwargs)
        return self._client

    def upload_file(self, local_path: str | Path, key: str) -> None:
        """Upload *local_path* to *key* in the bucket."""
        client = self._get_client()
        try:
            client.upload_file(str(local_path), self.bucket, key)
        except Exception as exc:
            raise BackupStoreError(
                f"upload to s3://{self.bucket}/{key} failed "
                f"({type(exc).__name__}): {exc}"
            ) from exc

    def download_file(self, key: str, local_path: str | Path) -> None:
        """Download *key* from the bucket to *local_path*.

        Raises :class:`BackupNotFoundError` when the object does not exist and
        :class:`BackupStoreError` for any other provider failure.
        """
        client = self._get_client()
        try:
            client.download_file(self.bucket, key, str(local_path))
        except Exception as exc:
            if _is_not_found(exc):
                raise BackupNotFoundError(
                    f"no backup object at s3://{self.bucket}/{key}"
                ) from exc
            raise BackupStoreError(
                f"download from s3://{self.bucket}/{key} failed "
                f"({type(exc).__name__}): {exc}"
            ) from exc


def _is_not_found(exc: BaseException) -> bool:
    """True when a provider exception means "object does not exist"."""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = str(response.get("Error", {}).get("Code", ""))
        if code in _NOT_FOUND_CODES:
            return True
    return str(getattr(exc, "code", "")) in _NOT_FOUND_CODES
