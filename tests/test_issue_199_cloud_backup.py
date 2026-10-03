"""Issue #199 Phase 1: opt-in encrypted cloud backup + safe restore.

Covers snapshot consistency (WAL), opt-in enforcement, configuration, the S3
storage adapter (fake/stubbed, never real cloud), backup orchestration, and
the restore safety contract: a bad download, wrong key, corrupted ciphertext,
invalid SQLite file, or failed replacement must never destroy the current
database. No test requires real AWS/R2 credentials or network access.
"""

from __future__ import annotations

import base64
import hashlib
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

from truememory.backup_crypto import encrypt_file, load_key
from truememory.backup_errors import (
    BackupConfigError,
    BackupDecryptionError,
    BackupDependencyError,
    BackupKeyError,
    BackupNotConfiguredError,
    BackupNotFoundError,
    BackupRestoreError,
    BackupSnapshotError,
    BackupStoreError,
    BackupValidationError,
)
from truememory.backup_store import S3ObjectStore
from truememory.cloud_backup import (
    BackupConfig,
    backup_database,
    resolve_db_path,
    restore_database,
    validate_sqlite_database,
)
from truememory.storage import _MAX_PRE_MIGRATION_BACKUPS, create_db, insert_message

KEY_A = base64.b64encode(b"A" * 32).decode("ascii")
KEY_B = base64.b64encode(b"B" * 32).decode("ascii")
CANARY = "canary-secret-memory-content-199"
DEFAULT_KEY_NAME = "truememory/memories.db.tmbackup"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeStore:
    """In-memory stand-in for the S3 object store. Records every call."""

    def __init__(self, fail_upload=None, fail_download=None):
        self.objects: dict[str, bytes] = {}
        self.upload_calls: list[tuple[str, str]] = []
        self.download_calls: list[str] = []
        self.fail_upload = fail_upload
        self.fail_download = fail_download

    def upload_file(self, local_path, key):
        self.upload_calls.append((str(local_path), key))
        if self.fail_upload is not None:
            raise self.fail_upload
        self.objects[key] = Path(local_path).read_bytes()

    def download_file(self, key, local_path):
        self.download_calls.append(key)
        if self.fail_download is not None:
            raise self.fail_download
        if key not in self.objects:
            raise BackupNotFoundError(f"no backup object at {key}")
        Path(local_path).write_bytes(self.objects[key])


def _make_db(path: Path, contents: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = create_db(path)
    for text in contents:
        insert_message(conn, {"content": text, "sender": "alice"})
    conn.commit()
    conn.close()
    return path


def _enabled_config(key: str = KEY_A, **overrides) -> BackupConfig:
    values = {
        "enabled": True,
        "bucket": "test-bucket",
        "encryption_key": key,
    }
    values.update(overrides)
    return BackupConfig(**values)


def _env_backup(monkeypatch, **overrides):
    """Set a minimal enabled backup env, applying *overrides* (None deletes)."""
    base = {
        "TRUEMEMORY_BACKUP_ENABLED": "1",
        "TRUEMEMORY_BACKUP_BUCKET": "test-bucket",
        "TRUEMEMORY_BACKUP_KEY": KEY_A,
    }
    base.update(overrides)
    for name, value in base.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    for name in list(os.environ):
        if name.startswith("TRUEMEMORY_BACKUP_") and name not in base:
            monkeypatch.delenv(name, raising=False)


def _track_tmpdirs(monkeypatch) -> list[Path]:
    """Track tempfile.mkdtemp directories created during the test."""
    created: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def tracking(prefix=None, suffix=None, dir=None):
        path = real_mkdtemp(prefix=prefix, suffix=suffix, dir=dir)
        created.append(Path(path))
        return path

    monkeypatch.setattr(tempfile, "mkdtemp", tracking)
    return created


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contents(path: Path) -> set[str]:
    conn = sqlite3.connect(str(path))
    try:
        return {row[0] for row in conn.execute("SELECT content FROM messages")}
    finally:
        conn.close()


def _decrypt_blob_to(blob: bytes, key: str, dest: Path) -> None:
    from truememory.backup_crypto import decrypt_file

    enc = dest.with_suffix(".enc")
    enc.write_bytes(blob)
    decrypt_file(enc, dest, load_key(key))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestBackupConfig:
    def test_defaults_are_disabled_and_local_only(self):
        cfg = BackupConfig.from_env({})
        assert cfg.enabled is False
        assert cfg.bucket == ""
        assert cfg.encryption_key is None

    def test_from_env_reads_all_settings(self, monkeypatch):
        monkeypatch.setenv("TRUEMEMORY_BACKUP_ENABLED", "true")
        monkeypatch.setenv("TRUEMEMORY_BACKUP_BUCKET", "my-bucket")
        monkeypatch.setenv("TRUEMEMORY_BACKUP_ENDPOINT", "https://fake.example")
        monkeypatch.setenv("TRUEMEMORY_BACKUP_REGION", "auto")
        monkeypatch.setenv("TRUEMEMORY_BACKUP_PREFIX", "backups/")
        monkeypatch.setenv("TRUEMEMORY_BACKUP_OBJECT", "explicit/key.tmbak")
        monkeypatch.setenv("TRUEMEMORY_BACKUP_ACCESS_KEY_ID", "AKIAFAKE")
        monkeypatch.setenv("TRUEMEMORY_BACKUP_SECRET_ACCESS_KEY", "secretfake")
        monkeypatch.setenv("TRUEMEMORY_BACKUP_KEY", KEY_A)
        cfg = BackupConfig.from_env()
        assert cfg.enabled is True
        assert cfg.bucket == "my-bucket"
        assert cfg.endpoint_url == "https://fake.example"
        assert cfg.region == "auto"
        assert cfg.prefix == "backups/"
        assert cfg.object_key == "explicit/key.tmbak"
        assert cfg.access_key_id == "AKIAFAKE"
        assert cfg.secret_access_key == "secretfake"
        assert cfg.encryption_key == KEY_A

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "Yes"])
    def test_enabled_accepts_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("TRUEMEMORY_BACKUP_ENABLED", value)
        assert BackupConfig.from_env().enabled is True

    @pytest.mark.parametrize("value", ["0", "false", "off", "no", "", "bogus"])
    def test_enabled_rejects_everything_else(self, monkeypatch, value):
        monkeypatch.setenv("TRUEMEMORY_BACKUP_ENABLED", value)
        assert BackupConfig.from_env().enabled is False

    def test_validate_requires_enabled(self):
        with pytest.raises(BackupNotConfiguredError, match="not enabled"):
            BackupConfig(bucket="b", encryption_key=KEY_A).validate()

    def test_validate_requires_bucket(self):
        with pytest.raises(BackupConfigError, match="BUCKET"):
            BackupConfig(enabled=True, encryption_key=KEY_A).validate()

    def test_validate_rejects_access_key_without_secret(self):
        cfg = BackupConfig(
            enabled=True, bucket="b", encryption_key=KEY_A, access_key_id="AKIAFAKE"
        )
        with pytest.raises(BackupConfigError, match="together"):
            cfg.validate()

    def test_validate_rejects_secret_without_access_key(self):
        cfg = BackupConfig(
            enabled=True, bucket="b", encryption_key=KEY_A, secret_access_key="x"
        )
        with pytest.raises(BackupConfigError, match="together"):
            cfg.validate()

    def test_repr_redacts_secrets(self):
        cfg = BackupConfig(
            enabled=True,
            bucket="my-bucket",
            access_key_id="AKIAFAKE000",
            secret_access_key="SECRETFAKE000",
            encryption_key="KEYFAKE000",
        )
        rendered = repr(cfg)
        assert "AKIAFAKE000" not in rendered
        assert "SECRETFAKE000" not in rendered
        assert "KEYFAKE000" not in rendered
        assert "my-bucket" in rendered

    def test_resolve_object_key_uses_db_name_and_prefix(self):
        cfg = BackupConfig(enabled=True, bucket="b", prefix="corp/")
        assert cfg.resolve_object_key("/home/u/memories.db") == (
            "corp/memories.db.tmbackup"
        )

    def test_resolve_object_key_normalizes_missing_slash(self):
        cfg = BackupConfig(enabled=True, bucket="b", prefix="corp")
        assert cfg.resolve_object_key("/x/memories.db") == "corp/memories.db.tmbackup"

    def test_resolve_object_key_explicit_wins(self):
        cfg = BackupConfig(enabled=True, bucket="b", object_key="fixed/key.tmbak")
        assert cfg.resolve_object_key("/x/memories.db") == "fixed/key.tmbak"


class TestResolveDbPath:
    def test_env_var_wins(self, monkeypatch):
        monkeypatch.setenv("TRUEMEMORY_DB_PATH", "/custom/path.db")
        assert resolve_db_path() == Path("/custom/path.db")

    def test_legacy_alias(self, monkeypatch):
        monkeypatch.delenv("TRUEMEMORY_DB_PATH", raising=False)
        monkeypatch.setenv("TRUEMEMORY_DB", "/legacy/path.db")
        assert resolve_db_path() == Path("/legacy/path.db")

    def test_default_falls_back_to_module_constant(self, monkeypatch):
        from truememory import cloud_backup

        monkeypatch.delenv("TRUEMEMORY_DB_PATH", raising=False)
        monkeypatch.delenv("TRUEMEMORY_DB", raising=False)
        assert resolve_db_path() == cloud_backup._DEFAULT_DB


# ---------------------------------------------------------------------------
# Opt-in enforcement
# ---------------------------------------------------------------------------


class TestOptIn:
    def test_backup_disabled_raises_before_any_network(self, tmp_path, monkeypatch):
        _env_backup(monkeypatch, TRUEMEMORY_BACKUP_ENABLED=None)
        db = _make_db(tmp_path / "memories.db", ["fact"])
        store = FakeStore()
        with pytest.raises(BackupNotConfiguredError):
            backup_database(db_path=db, store=store)
        assert store.upload_calls == []
        assert store.objects == {}

    def test_restore_disabled_raises_before_any_network(self, tmp_path, monkeypatch):
        _env_backup(monkeypatch, TRUEMEMORY_BACKUP_ENABLED=None)
        db = _make_db(tmp_path / "memories.db", ["fact"])
        store = FakeStore()
        with pytest.raises(BackupNotConfiguredError):
            restore_database(db_path=db, store=store)
        assert store.download_calls == []

    def test_unconfigured_bucket_raises(self, tmp_path, monkeypatch):
        _env_backup(monkeypatch, TRUEMEMORY_BACKUP_BUCKET=None)
        db = _make_db(tmp_path / "memories.db", ["fact"])
        with pytest.raises(BackupConfigError):
            backup_database(db_path=db, store=FakeStore())

    def test_missing_key_raises_before_snapshot_or_network(self, tmp_path, monkeypatch):
        _env_backup(monkeypatch, TRUEMEMORY_BACKUP_KEY=None)
        db = _make_db(tmp_path / "memories.db", ["fact"])
        store = FakeStore()
        with pytest.raises(BackupKeyError):
            backup_database(db_path=db, store=store)
        assert store.upload_calls == []

    def test_invalid_key_raises(self, tmp_path):
        db = _make_db(tmp_path / "memories.db", ["fact"])
        with pytest.raises(BackupKeyError):
            backup_database(_enabled_config(), db_path=db, key="!!!not-base64!!!")

    def test_default_install_never_imports_backup_modules(self):
        """Core package must stay import-clean of the backup feature."""
        import truememory

        init_src = Path(truememory.__file__).read_text(encoding="utf-8")
        assert "cloud_backup" not in init_src
        mcp_src = (
            Path(truememory.__file__).parent / "mcp_server.py"
        ).read_text(encoding="utf-8")
        assert "cloud_backup" not in mcp_src


# ---------------------------------------------------------------------------
# SQLite validation
# ---------------------------------------------------------------------------


class TestValidateSqlite:
    def test_valid_db_passes(self, tmp_path):
        db = _make_db(tmp_path / "m.db", ["a", "b"])
        validate_sqlite_database(db)

    def test_garbage_file_rejected(self, tmp_path):
        bad = tmp_path / "bad.db"
        bad.write_text("definitely not a sqlite file", encoding="utf-8")
        with pytest.raises(BackupValidationError, match="not a valid SQLite database"):
            validate_sqlite_database(bad)

    def test_empty_sqlite_missing_tables_rejected(self, tmp_path):
        empty = tmp_path / "empty.db"
        conn = sqlite3.connect(str(empty))
        conn.execute("CREATE TABLE other (x)")
        conn.commit()
        conn.close()
        with pytest.raises(BackupValidationError, match="not a TrueMemory database"):
            validate_sqlite_database(empty)

    def test_corrupt_db_rejected(self, tmp_path):
        db = _make_db(tmp_path / "m.db", ["a"])
        data = bytearray(db.read_bytes())
        for i in range(100, min(len(data), 4000)):
            data[i] ^= 0xFF
        db.write_bytes(bytes(data))
        with pytest.raises(BackupValidationError):
            validate_sqlite_database(db)


# ---------------------------------------------------------------------------
# S3 storage adapter
# ---------------------------------------------------------------------------


class _StubClient:
    """boto3-shaped client stub: records calls, raises on demand."""

    class _NotFound(Exception):
        def __init__(self, code="NoSuchKey"):
            self.response = {"Error": {"Code": code, "Message": "not found"}}

    def __init__(self, fail_download_code=None, fail_download_error=None):
        self.uploaded: list[tuple] = []
        self.downloaded: list[tuple] = []
        self.fail_download_code = fail_download_code
        self.fail_download_error = fail_download_error

    def upload_file(self, local, bucket, key):
        self.uploaded.append((bucket, key, local))

    def download_file(self, bucket, key, dest):
        self.downloaded.append((bucket, key, dest))
        if self.fail_download_code is not None:
            raise self._NotFound(self.fail_download_code)
        if self.fail_download_error is not None:
            raise self.fail_download_error


class TestS3ObjectStore:
    def test_requires_bucket(self):
        with pytest.raises(BackupStoreError, match="bucket"):
            S3ObjectStore("")

    def test_upload_uses_bucket_and_key(self, tmp_path):
        stub = _StubClient()
        store = S3ObjectStore("my-bucket", client=stub)
        store.upload_file(tmp_path / "f.bin", "some/key")
        assert stub.uploaded == [("my-bucket", "some/key", str(tmp_path / "f.bin"))]

    def test_download_uses_bucket_and_key(self, tmp_path):
        stub = _StubClient()
        store = S3ObjectStore("my-bucket", client=stub)
        store.download_file("some/key", tmp_path / "out.bin")
        assert stub.downloaded[0][:2] == ("my-bucket", "some/key")

    def test_download_missing_object_raises_not_found(self, tmp_path):
        stub = _StubClient(fail_download_code="NoSuchKey")
        store = S3ObjectStore("my-bucket", client=stub)
        with pytest.raises(BackupNotFoundError):
            store.download_file("missing/key", tmp_path / "out.bin")

    def test_download_404_code_raises_not_found(self, tmp_path):
        stub = _StubClient(fail_download_code="404")
        store = S3ObjectStore("my-bucket", client=stub)
        with pytest.raises(BackupNotFoundError):
            store.download_file("k", tmp_path / "out.bin")

    def test_download_other_errors_wrapped(self, tmp_path):
        stub = _StubClient(fail_download_error=RuntimeError("socket exploded"))
        store = S3ObjectStore("my-bucket", client=stub)
        with pytest.raises(BackupStoreError, match="download from s3://"):
            store.download_file("k", tmp_path / "out.bin")

    def test_upload_errors_wrapped(self, tmp_path):
        class _Boom(Exception):
            pass

        class _FailUploadClient(_StubClient):
            def upload_file(self, local, bucket, key):
                raise _Boom("permission denied")

        store = S3ObjectStore("my-bucket", client=_FailUploadClient())
        with pytest.raises(BackupStoreError, match="upload to s3://"):
            store.upload_file(tmp_path / "f.bin", "k")

    def test_missing_boto3_gives_actionable_error(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "boto3", None)
        store = S3ObjectStore("my-bucket")
        with pytest.raises(BackupDependencyError, match=r"truememory\[backup\]"):
            store.upload_file(tmp_path / "f.bin", "k")
        with pytest.raises(BackupDependencyError, match=r"truememory\[backup\]"):
            store.download_file("k", tmp_path / "out.bin")

    def test_boto3_client_built_with_config(self, monkeypatch):
        boto3 = pytest.importorskip("boto3")
        captured: dict = {}
        stub = _StubClient()

        def fake_client(service, **kwargs):
            captured["service"] = service
            captured.update(kwargs)
            return stub

        monkeypatch.setattr(boto3, "client", fake_client)
        store = S3ObjectStore(
            "my-bucket",
            endpoint_url="https://fake.example",
            region="auto",
            access_key_id="AKIAFAKE",
            secret_access_key="secretfake",
        )
        store.upload_file("unused", "k")
        assert captured["service"] == "s3"
        assert captured["endpoint_url"] == "https://fake.example"
        assert captured["region_name"] == "auto"
        assert captured["aws_access_key_id"] == "AKIAFAKE"
        assert captured["aws_secret_access_key"] == "secretfake"

    def test_boto3_client_defaults_region(self, monkeypatch):
        boto3 = pytest.importorskip("boto3")
        captured: dict = {}

        def fake_client(service, **kwargs):
            captured.update(kwargs)
            return _StubClient()

        monkeypatch.setattr(boto3, "client", fake_client)
        store = S3ObjectStore("my-bucket")
        store.download_file("k", "unused-dest")
        assert captured["region_name"] == "us-east-1"
        assert "endpoint_url" not in captured
        assert "aws_access_key_id" not in captured


# ---------------------------------------------------------------------------
# Backup operation
# ---------------------------------------------------------------------------


class TestBackupOperation:
    def test_live_wal_db_snapshot_is_consistent(self, tmp_path):
        """Data still living in the -wal must be folded into the snapshot."""
        db = tmp_path / "memories.db"
        conn = create_db(db)
        for i in range(10):
            insert_message(conn, {"content": f"fact {i}", "sender": "alice"})
        conn.commit()
        try:
            assert (tmp_path / "memories.db-wal").exists(), "expected WAL content"

            store = FakeStore()
            result = backup_database(_enabled_config(), db_path=db, store=store)
            assert result.encrypted_bytes > 0
        finally:
            conn.close()

        blob = store.objects[result.key]
        decrypted = tmp_path / "decrypted.db"
        _decrypt_blob_to(blob, KEY_A, decrypted)
        assert _contents(decrypted) == {f"fact {i}" for i in range(10)}

    def test_snapshot_opens_standalone_without_sidecars(self, tmp_path):
        db = _make_db(tmp_path / "memories.db", ["alpha"])
        store = FakeStore()
        backup_database(_enabled_config(), db_path=db, store=store)
        blob = store.objects[DEFAULT_KEY_NAME]
        decrypted = tmp_path / "decrypted.db"
        _decrypt_blob_to(blob, KEY_A, decrypted)
        assert not Path(f"{decrypted}-wal").exists()
        assert not Path(f"{decrypted}-shm").exists()
        validate_sqlite_database(decrypted)

    def test_upload_is_encrypted_and_leaks_no_plaintext(self, tmp_path):
        db = _make_db(tmp_path / "memories.db", [CANARY])
        store = FakeStore()
        backup_database(_enabled_config(), db_path=db, store=store)
        blob = store.objects[DEFAULT_KEY_NAME]
        assert blob[:5] == b"TMBAK"
        assert CANARY.encode() not in blob
        assert b"SQLite format 3" not in blob

    def test_result_metadata(self, tmp_path):
        db = _make_db(tmp_path / "memories.db", ["fact"])
        store = FakeStore()
        result = backup_database(_enabled_config(), db_path=db, store=store)
        assert result.bucket == "test-bucket"
        assert result.key == DEFAULT_KEY_NAME
        assert result.format_version == 1
        assert result.db_path == str(db)
        assert result.encrypted_bytes > 0
        assert len(store.upload_calls) == 1
        assert store.upload_calls[0][1] == result.key

    def test_explicit_object_key_used(self, tmp_path):
        db = _make_db(tmp_path / "memories.db", ["fact"])
        store = FakeStore()
        result = backup_database(
            _enabled_config(object_key="custom/object.tmbak"),
            db_path=db,
            store=store,
        )
        assert result.key == "custom/object.tmbak"
        assert "custom/object.tmbak" in store.objects

    def test_explicit_key_overrides_config_key(self, tmp_path):
        db = _make_db(tmp_path / "memories.db", ["fact"])
        store = FakeStore()
        backup_database(_enabled_config(key=KEY_A), db_path=db, key=KEY_B, store=store)
        blob = store.objects[DEFAULT_KEY_NAME]
        decrypted = tmp_path / "decrypted.db"
        _decrypt_blob_to(blob, KEY_B, decrypted)
        assert _contents(decrypted) == {"fact"}
        with pytest.raises(BackupDecryptionError):
            _decrypt_blob_to(blob, KEY_A, tmp_path / "should_fail.db")

    def test_missing_source_fails_without_upload(self, tmp_path):
        store = FakeStore()
        with pytest.raises(BackupSnapshotError, match="not found"):
            backup_database(
                _enabled_config(), db_path=tmp_path / "missing.db", store=store
            )
        assert store.upload_calls == []

    def test_in_memory_db_rejected(self):
        with pytest.raises(BackupSnapshotError, match="in-memory"):
            backup_database(_enabled_config(), db_path=":memory:", store=FakeStore())

    def test_invalid_source_db_fails(self, tmp_path):
        bad = tmp_path / "bad.db"
        bad.write_text("this is not sqlite", encoding="utf-8")
        store = FakeStore()
        with pytest.raises(BackupSnapshotError):
            backup_database(_enabled_config(), db_path=bad, store=store)
        assert store.upload_calls == []

    def test_no_plaintext_left_next_to_db(self, tmp_path):
        db = _make_db(tmp_path / "memories.db", [CANARY])
        before = {p.name for p in tmp_path.iterdir()}
        backup_database(_enabled_config(), db_path=db, store=FakeStore())
        after = {p.name for p in tmp_path.iterdir()}
        assert after == before, "backup left files next to the database"
        for p in tmp_path.iterdir():
            if p.is_file() and p != db:
                assert CANARY.encode() not in p.read_bytes()

    def test_temp_dirs_cleaned_on_success(self, tmp_path, monkeypatch):
        db = _make_db(tmp_path / "memories.db", ["fact"])
        created = _track_tmpdirs(monkeypatch)
        backup_database(_enabled_config(), db_path=db, store=FakeStore())
        assert created, "expected at least one staging dir"
        assert all(not p.exists() for p in created)

    def test_temp_dirs_cleaned_on_upload_failure(self, tmp_path, monkeypatch):
        db = _make_db(tmp_path / "memories.db", ["fact"])
        created = _track_tmpdirs(monkeypatch)
        store = FakeStore(fail_upload=BackupStoreError("network down"))
        with pytest.raises(BackupStoreError):
            backup_database(_enabled_config(), db_path=db, store=store)
        assert created
        assert all(not p.exists() for p in created)

    def test_temp_dirs_cleaned_on_snapshot_failure(self, tmp_path, monkeypatch):
        created = _track_tmpdirs(monkeypatch)
        bad = tmp_path / "bad.db"
        bad.write_text("not sqlite", encoding="utf-8")
        with pytest.raises(BackupSnapshotError):
            backup_database(_enabled_config(), db_path=bad, store=FakeStore())
        assert created
        assert all(not p.exists() for p in created)

    def test_backup_aborts_when_plaintext_snapshot_deletion_fails(
        self, tmp_path, monkeypatch
    ):
        """If the plaintext snapshot cannot be deleted, nothing is uploaded."""
        db = _make_db(tmp_path / "memories.db", ["fact"])
        store = FakeStore()
        real_unlink = os.unlink

        def guarded_unlink(path, *args, **kwargs):
            if str(path).endswith("snapshot.db"):
                raise OSError("simulated deletion failure")
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(os, "unlink", guarded_unlink)
        with pytest.raises(BackupSnapshotError, match="plaintext snapshot"):
            backup_database(_enabled_config(), db_path=db, store=store)
        assert store.upload_calls == []
        assert store.objects == {}


# ---------------------------------------------------------------------------
# Restore operation
# ---------------------------------------------------------------------------


class TestRestoreOperation:
    @pytest.fixture
    def source_backup(self, tmp_path):
        """A populated source DB backed up into a FakeStore, plus a target DB."""
        source = _make_db(
            tmp_path / "src" / "memories.db", ["new-fact-1", "new-fact-2"]
        )
        target = _make_db(tmp_path / "dst" / "memories.db", ["old-fact-1"])
        store = FakeStore()
        backup_database(_enabled_config(), db_path=source, store=store)
        assert DEFAULT_KEY_NAME in store.objects
        return store, target

    def test_round_trip_restores_data(self, source_backup):
        store, target = source_backup
        result = restore_database(_enabled_config(), db_path=target, store=store)
        assert _contents(target) == {"new-fact-1", "new-fact-2"}
        assert result.db_path == str(target)

    def test_restored_db_opens_with_create_db(self, source_backup):
        store, target = source_backup
        restore_database(_enabled_config(), db_path=target, store=store)
        conn = create_db(target)  # full schema migration path must accept it
        try:
            rows = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        finally:
            conn.close()
        assert rows == 2

    def test_safety_backup_created_with_old_data(self, source_backup):
        store, target = source_backup
        result = restore_database(_enabled_config(), db_path=target, store=store)
        assert result.safety_backup is not None
        safety = Path(result.safety_backup)
        assert safety.exists()
        assert safety.name.startswith("memories.db.backup-pre-restore-")
        assert _contents(safety) == {"old-fact-1"}

    def test_safety_backups_rotate(self, source_backup):
        store, target = source_backup
        for _ in range(_MAX_PRE_MIGRATION_BACKUPS + 3):
            restore_database(_enabled_config(), db_path=target, store=store)
        backups = [
            p
            for p in target.parent.glob("memories.db.backup-pre-restore-*")
            if not p.name.endswith(("-wal", "-shm"))
        ]
        assert len(backups) <= _MAX_PRE_MIGRATION_BACKUPS

    def test_wrong_key_preserves_current_db(self, source_backup):
        store, target = source_backup
        digest_before = _sha256(target)
        with pytest.raises(BackupDecryptionError):
            restore_database(_enabled_config(key=KEY_B), db_path=target, store=store)
        assert _sha256(target) == digest_before
        assert _contents(target) == {"old-fact-1"}
        assert not list(target.parent.glob("*.backup-pre-restore-*"))
        assert not list(target.parent.glob(".*.restore-*.tmp"))

    def test_corrupted_backup_preserves_current_db(self, source_backup):
        store, target = source_backup
        blob = bytearray(store.objects[DEFAULT_KEY_NAME])
        blob[len(blob) // 2] ^= 0xFF
        store.objects[DEFAULT_KEY_NAME] = bytes(blob)
        digest_before = _sha256(target)
        with pytest.raises((BackupDecryptionError, BackupValidationError)):
            restore_database(_enabled_config(), db_path=target, store=store)
        assert _sha256(target) == digest_before
        assert _contents(target) == {"old-fact-1"}
        assert not list(target.parent.glob("*.backup-pre-restore-*"))

    def test_invalid_sqlite_preserves_current_db(self, tmp_path):
        target = _make_db(tmp_path / "dst" / "memories.db", ["old-fact-1"])
        garbage = tmp_path / "garbage.bin"
        garbage.write_bytes(b"just some bytes, not a database")
        encrypted_garbage = tmp_path / "garbage.tmbak"
        encrypt_file(garbage, encrypted_garbage, load_key(KEY_A))
        store = FakeStore()
        store.objects[DEFAULT_KEY_NAME] = encrypted_garbage.read_bytes()
        digest_before = _sha256(target)
        with pytest.raises(BackupValidationError):
            restore_database(_enabled_config(), db_path=target, store=store)
        assert _sha256(target) == digest_before
        assert _contents(target) == {"old-fact-1"}

    def test_missing_object_preserves_current_db(self, tmp_path):
        target = _make_db(tmp_path / "dst" / "memories.db", ["old-fact-1"])
        digest_before = _sha256(target)
        with pytest.raises(BackupNotFoundError):
            restore_database(_enabled_config(), db_path=target, store=FakeStore())
        assert _sha256(target) == digest_before
        assert not list(target.parent.glob("*.backup-pre-restore-*"))

    def test_aborts_when_safety_backup_fails(self, source_backup, monkeypatch):
        store, target = source_backup
        digest_before = _sha256(target)
        monkeypatch.setattr(
            "truememory.cloud_backup._backup_database", lambda db, reason: None
        )
        with pytest.raises(BackupRestoreError, match="safety"):
            restore_database(_enabled_config(), db_path=target, store=store)
        assert _sha256(target) == digest_before
        assert _contents(target) == {"old-fact-1"}

    def test_stale_sidecars_removed(self, source_backup):
        store, target = source_backup
        # Simulate leftover WAL/SHM/journal artifacts from the old database.
        Path(f"{target}-wal").write_bytes(b"")
        Path(f"{target}-shm").write_bytes(b"")
        Path(f"{target}-journal").write_bytes(b"")
        restore_database(_enabled_config(), db_path=target, store=store)
        assert _contents(target) == {"new-fact-1", "new-fact-2"}
        assert not Path(f"{target}-wal").exists()
        assert not Path(f"{target}-shm").exists()
        assert not Path(f"{target}-journal").exists()

    def test_sidecar_removal_failure_aborts_before_replacing(
        self, source_backup, monkeypatch
    ):
        store, target = source_backup
        Path(f"{target}-wal").write_bytes(b"")
        real_unlink = os.unlink

        def guarded_unlink(path, *args, **kwargs):
            if str(path).endswith("-wal"):
                raise PermissionError(13, "simulated sharing violation")
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(os, "unlink", guarded_unlink)
        digest_before = _sha256(target)
        with pytest.raises(BackupRestoreError, match="sidecar"):
            restore_database(_enabled_config(), db_path=target, store=store)
        assert _sha256(target) == digest_before
        assert _contents(target) == {"old-fact-1"}

    def test_replace_failure_recovers_from_safety_backup(
        self, source_backup, monkeypatch
    ):
        store, target = source_backup

        def failing_replace(src, dst, **kwargs):
            raise PermissionError(13, "destination locked by another process")

        monkeypatch.setattr("truememory.cloud_backup._atomic_replace", failing_replace)
        with pytest.raises(BackupRestoreError) as exc_info:
            restore_database(_enabled_config(), db_path=target, store=store)
        message = str(exc_info.value)
        assert "could not replace" in message
        # The pre-restore snapshot is copied back, so no data is lost.
        assert _contents(target) == {"old-fact-1"}
        safety_files = [
            p
            for p in target.parent.glob("memories.db.backup-pre-restore-*")
            if not p.name.endswith(("-wal", "-shm"))
        ]
        assert safety_files
        assert _contents(safety_files[0]) == {"old-fact-1"}
        # ... and the recovered database is structurally healthy, not torn.
        conn = sqlite3.connect(str(target))
        try:
            assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        finally:
            conn.close()

    def test_restore_to_missing_db_creates_it(self, source_backup, tmp_path):
        store, _target = source_backup
        fresh = tmp_path / "brand-new" / "memories.db"
        result = restore_database(_enabled_config(), db_path=fresh, store=store)
        assert fresh.is_file()
        assert _contents(fresh) == {"new-fact-1", "new-fact-2"}
        assert result.safety_backup is None
        validate_sqlite_database(fresh)

    def test_restore_in_memory_target_rejected(self):
        with pytest.raises(BackupRestoreError, match="in-memory"):
            restore_database(_enabled_config(), db_path=":memory:", store=FakeStore())

    def test_post_replacement_validation_failure_rolls_back(
        self, source_backup, monkeypatch
    ):
        """If the DB fails validation AFTER os.replace succeeded, the
        pre-restore snapshot must be copied back: a failed restore must leave
        the original database in place, not a silently swapped one."""
        from truememory import cloud_backup

        store, target = source_backup
        real_validate = cloud_backup.validate_sqlite_database

        def fail_only_post_replacement(path):
            if Path(path) == target:
                raise BackupValidationError("simulated post-replacement failure")
            return real_validate(path)

        monkeypatch.setattr(
            cloud_backup, "validate_sqlite_database", fail_only_post_replacement
        )
        with pytest.raises(BackupRestoreError, match="post-replacement validation"):
            restore_database(_enabled_config(), db_path=target, store=store)
        # Rolled back to the pre-restore snapshot: original data is live again
        assert _contents(target) == {"old-fact-1"}
        conn = sqlite3.connect(str(target))
        try:
            assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        finally:
            conn.close()

    def test_temp_files_cleaned_after_success(self, source_backup, monkeypatch):
        store, target = source_backup
        created = _track_tmpdirs(monkeypatch)
        restore_database(_enabled_config(), db_path=target, store=store)
        assert created
        assert all(not p.exists() for p in created)
        # No staging file or staged-sidecar leftovers of any kind.
        assert not list(target.parent.glob(".*.restore-*"))

    def test_temp_files_cleaned_after_decrypt_failure(self, source_backup, monkeypatch):
        store, target = source_backup
        created = _track_tmpdirs(monkeypatch)
        with pytest.raises(BackupDecryptionError):
            restore_database(_enabled_config(key=KEY_B), db_path=target, store=store)
        assert created
        assert all(not p.exists() for p in created)
        assert not list(target.parent.glob(".*.restore-*"))

    def test_error_messages_do_not_leak_key(self, source_backup):
        store, target = source_backup
        with pytest.raises(BackupDecryptionError) as exc_info:
            restore_database(_enabled_config(key=KEY_B), db_path=target, store=store)
        message = str(exc_info.value)
        assert KEY_B not in message
        assert base64.b64decode(KEY_B).decode("latin-1") not in message


# ---------------------------------------------------------------------------
# CLI (in-process, mirroring tests/ingest/test_hook_schema.py conventions)
# ---------------------------------------------------------------------------


def _clear_backup_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("TRUEMEMORY_"):
            monkeypatch.delenv(name, raising=False)


def _run_cli_main(monkeypatch, argv):
    from truememory.ingest import cli as ingest_cli

    monkeypatch.setattr(sys, "argv", ["truememory-ingest"] + argv)
    ingest_cli.main()


class TestCloudBackupCli:
    def test_help_lists_backup_and_restore(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc_info:
            _run_cli_main(monkeypatch, ["--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "Create an encrypted cloud backup" in out
        assert "restore" in out

    def test_backup_generate_key_prints_valid_key(self, monkeypatch, capsys):
        _clear_backup_env(monkeypatch)
        _run_cli_main(monkeypatch, ["backup", "--generate-key"])
        out = capsys.readouterr()
        key_text = out.out.strip().splitlines()[0]
        assert len(base64.b64decode(key_text, validate=True)) == 32
        assert "TRUEMEMORY_BACKUP_KEY" in out.err
        assert "lose" in out.err.lower()

    def test_restore_requires_yes_when_not_a_tty(self, monkeypatch, capsys, tmp_path):
        # With a VALID backup configuration but no TTY, restore must refuse to
        # proceed without --yes rather than silently replacing the database.
        _env_backup(monkeypatch)
        with pytest.raises(SystemExit) as exc_info:
            _run_cli_main(monkeypatch, ["restore", "--db", str(tmp_path / "m.db")])
        assert exc_info.value.code == 2
        assert "--yes" in capsys.readouterr().err

    def test_backup_unconfigured_exits_with_clear_error(self, monkeypatch, capsys, tmp_path):
        _clear_backup_env(monkeypatch)
        with pytest.raises(SystemExit) as exc_info:
            _run_cli_main(monkeypatch, ["backup", "--db", str(tmp_path / "m.db")])
        assert exc_info.value.code == 1
        assert "not enabled" in capsys.readouterr().err

    def test_restore_unconfigured_exits_with_clear_error(self, monkeypatch, capsys, tmp_path):
        _clear_backup_env(monkeypatch)
        with pytest.raises(SystemExit) as exc_info:
            _run_cli_main(
                monkeypatch, ["restore", "--db", str(tmp_path / "m.db"), "--yes"]
            )
        assert exc_info.value.code == 1
        assert "not enabled" in capsys.readouterr().err

    def test_backup_missing_key_file_exits(self, monkeypatch, capsys, tmp_path):
        _clear_backup_env(monkeypatch)
        with pytest.raises(SystemExit) as exc_info:
            _run_cli_main(
                monkeypatch,
                [
                    "backup",
                    "--db", str(tmp_path / "m.db"),
                    "--key-file", str(tmp_path / "no-such-keyfile"),
                ],
            )
        assert exc_info.value.code == 2
        assert "key file not found" in capsys.readouterr().err

    def test_backup_dispatches_to_backup_database(self, monkeypatch, capsys, tmp_path):
        """The CLI wires args through to the API (no network in this test)."""
        _env_backup(monkeypatch)
        from truememory.cloud_backup import BackupResult

        captured: dict = {}

        def fake_backup(config, **kwargs):
            captured.update(kwargs)
            return BackupResult(
                bucket=config.bucket,
                key="k",
                encrypted_bytes=7,
                format_version=1,
                db_path=str(kwargs["db_path"]),
            )

        monkeypatch.setattr("truememory.cloud_backup.backup_database", fake_backup)
        _run_cli_main(monkeypatch, ["backup", "--db", str(tmp_path / "m.db")])
        assert captured["db_path"] == str(tmp_path / "m.db")
        assert "Encrypted backup uploaded" in capsys.readouterr().out

    def test_restore_yes_dispatches_to_restore_database(
        self, monkeypatch, capsys, tmp_path
    ):
        _env_backup(monkeypatch)
        from truememory.cloud_backup import RestoreResult

        captured: dict = {}

        def fake_restore(config, **kwargs):
            captured.update(kwargs)
            return RestoreResult(
                bucket=config.bucket,
                key="k",
                encrypted_bytes=7,
                format_version=1,
                db_path=str(kwargs["db_path"]),
                safety_backup=None,
            )

        monkeypatch.setattr("truememory.cloud_backup.restore_database", fake_restore)
        _run_cli_main(
            monkeypatch, ["restore", "--db", str(tmp_path / "m.db"), "--yes"]
        )
        assert captured["db_path"] == Path(tmp_path / "m.db")
        assert "Restored database" in capsys.readouterr().out

    def test_restore_interactive_decline_aborts_without_restoring(
        self, monkeypatch, capsys, tmp_path
    ):
        _env_backup(monkeypatch)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda *a, **k: "no")
        restore_called = {"called": False}

        def fake_restore(*args, **kwargs):
            restore_called["called"] = True
            raise AssertionError("restore_database must not run after a decline")

        monkeypatch.setattr("truememory.cloud_backup.restore_database", fake_restore)
        with pytest.raises(SystemExit) as exc_info:
            _run_cli_main(monkeypatch, ["restore", "--db", str(tmp_path / "m.db")])
        assert exc_info.value.code == 1
        assert restore_called["called"] is False
        assert "Aborted" in capsys.readouterr().out

    def test_restore_interactive_eof_aborts_cleanly(
        self, monkeypatch, capsys, tmp_path
    ):
        """Ctrl-D / closed stdin on a TTY must abort, not traceback."""
        _env_backup(monkeypatch)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

        def _raise_eof(*a, **k):
            raise EOFError

        monkeypatch.setattr("builtins.input", _raise_eof)
        with pytest.raises(SystemExit) as exc_info:
            _run_cli_main(monkeypatch, ["restore", "--db", str(tmp_path / "m.db")])
        assert exc_info.value.code == 1
        assert "Aborted" in capsys.readouterr().out
