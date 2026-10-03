"""Unit tests for the encrypted backup format (issue #199, Phase 1).

Covers: key loading/generation, AES-256-GCM round trips, wrong keys, tampered
ciphertext/tag/header, truncation, format-version rejection, and cleanup of
partial output on failure. No network, no real cloud provider.
"""

from __future__ import annotations

import base64
import sqlite3
from pathlib import Path

import pytest

from truememory.backup_crypto import (
    ALGORITHM_AES_256_GCM,
    FORMAT_VERSION,
    decrypt_file,
    encrypt_file,
    generate_key,
    load_key,
    read_format_version,
)
from truememory.backup_errors import (
    BackupDecryptionError,
    BackupEncryptionError,
    BackupKeyError,
)

RAW_KEY = b"0123456789abcdef0123456789abcdef"
OTHER_KEY = b"fedcba9876543210fedcba9876543210"
B64_KEY = base64.b64encode(RAW_KEY).decode("ascii")


@pytest.fixture
def plaintext(tmp_path: Path) -> Path:
    """A small standalone SQLite file as the plaintext payload."""
    path = tmp_path / "payload.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE t (x TEXT)")
    conn.executemany("INSERT INTO t VALUES (?)", [(f"row-{i}",) for i in range(50)])
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def encrypted(tmp_path: Path, plaintext: Path) -> Path:
    path = tmp_path / "payload.tmbak"
    version = encrypt_file(plaintext, path, RAW_KEY)
    assert version == FORMAT_VERSION
    return path


# ---------------------------------------------------------------------------
# Key handling
# ---------------------------------------------------------------------------


def test_generate_key_is_32_byte_base64():
    key_text = generate_key()
    assert isinstance(key_text, str)
    decoded = base64.b64decode(key_text, validate=True)
    assert len(decoded) == 32


def test_generated_keys_are_unique():
    assert generate_key() != generate_key()


def test_load_key_accepts_base64_text():
    assert load_key(B64_KEY) == RAW_KEY


def test_load_key_accepts_unpadded_base64():
    unpadded = B64_KEY.rstrip("=")
    assert load_key(unpadded) == RAW_KEY


def test_load_key_accepts_raw_bytes():
    assert load_key(RAW_KEY) == RAW_KEY


def test_load_key_rejects_none():
    with pytest.raises(BackupKeyError, match="no backup encryption key configured"):
        load_key(None)


def test_load_key_rejects_empty_string():
    with pytest.raises(BackupKeyError, match="empty"):
        load_key("   ")


def test_load_key_rejects_invalid_base64():
    with pytest.raises(BackupKeyError, match="not valid base64"):
        load_key("not-base64!!!")


def test_load_key_rejects_wrong_length():
    short = base64.b64encode(b"tooshort").decode("ascii")
    with pytest.raises(BackupKeyError, match="must decode to 32 bytes"):
        load_key(short)


def test_load_key_rejects_wrong_type():
    with pytest.raises(BackupKeyError, match="must be str or bytes"):
        load_key(12345)  # type: ignore[arg-type]


def test_key_errors_never_echo_the_key():
    with pytest.raises(BackupKeyError) as exc_info:
        load_key(base64.b64encode(b"x" * 40).decode())
    assert base64.b64encode(b"x" * 40).decode() not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_round_trip_preserves_content(tmp_path: Path, plaintext: Path):
    enc = tmp_path / "out.tmbak"
    dec = tmp_path / "out.db"
    encrypt_file(plaintext, enc, RAW_KEY)
    assert decrypt_file(enc, dec, RAW_KEY) == FORMAT_VERSION
    assert dec.read_bytes() == plaintext.read_bytes()


def test_encrypt_output_is_not_the_plaintext(plaintext: Path, encrypted: Path):
    assert encrypted.read_bytes() != plaintext.read_bytes()
    data = plaintext.read_bytes()
    assert data not in encrypted.read_bytes()


def test_header_has_magic_version_algorithm_nonce(encrypted: Path):
    header = encrypted.read_bytes()[:19]
    assert header[:5] == b"TMBAK"
    assert header[5] == FORMAT_VERSION
    assert header[6] == ALGORITHM_AES_256_GCM
    assert len(header[7:]) == 12


def test_read_format_version_without_decrypting(encrypted: Path):
    assert read_format_version(encrypted) == FORMAT_VERSION


def test_nonces_differ_between_encryptions(tmp_path: Path, plaintext: Path):
    a = tmp_path / "a.tmbak"
    b = tmp_path / "b.tmbak"
    encrypt_file(plaintext, a, RAW_KEY)
    encrypt_file(plaintext, b, RAW_KEY)
    nonce_a = a.read_bytes()[7:19]
    nonce_b = b.read_bytes()[7:19]
    assert nonce_a != nonce_b


def test_wrong_key_fails(tmp_path: Path, encrypted: Path):
    dec = tmp_path / "out.db"
    with pytest.raises(BackupDecryptionError, match="wrong key or corrupted"):
        decrypt_file(encrypted, dec, OTHER_KEY)
    assert not dec.exists()


def test_corrupted_ciphertext_fails(tmp_path: Path, encrypted: Path):
    data = bytearray(encrypted.read_bytes())
    # Flip a byte near the middle of the ciphertext.
    middle = len(data) // 2
    data[middle] ^= 0xFF
    corrupted = tmp_path / "corrupt.tmbak"
    corrupted.write_bytes(bytes(data))
    dec = tmp_path / "out.db"
    with pytest.raises(BackupDecryptionError, match="wrong key or corrupted"):
        decrypt_file(corrupted, dec, RAW_KEY)
    assert not dec.exists()


def test_modified_tag_fails(tmp_path: Path, encrypted: Path):
    data = bytearray(encrypted.read_bytes())
    data[-1] ^= 0x01
    corrupted = tmp_path / "corrupt.tmbak"
    corrupted.write_bytes(bytes(data))
    with pytest.raises(BackupDecryptionError, match="wrong key or corrupted"):
        decrypt_file(corrupted, tmp_path / "out.db", RAW_KEY)


def test_modified_nonce_fails(tmp_path: Path, encrypted: Path):
    """The header is authenticated (AAD), so a nonce flip must be detected."""
    data = bytearray(encrypted.read_bytes())
    data[7] ^= 0xFF
    corrupted = tmp_path / "corrupt.tmbak"
    corrupted.write_bytes(bytes(data))
    with pytest.raises(BackupDecryptionError):
        decrypt_file(corrupted, tmp_path / "out.db", RAW_KEY)


def test_modified_version_fails(tmp_path: Path, encrypted: Path):
    data = bytearray(encrypted.read_bytes())
    data[5] ^= 0x01
    corrupted = tmp_path / "corrupt.tmbak"
    corrupted.write_bytes(bytes(data))
    with pytest.raises(BackupDecryptionError, match="unsupported backup format"):
        decrypt_file(corrupted, tmp_path / "out.db", RAW_KEY)


def test_bad_magic_fails(tmp_path: Path, plaintext: Path):
    enc = tmp_path / "out.tmbak"
    encrypt_file(plaintext, enc, RAW_KEY)
    data = bytearray(enc.read_bytes())
    data[0] = ord("X")
    bogus = tmp_path / "bogus.tmbak"
    bogus.write_bytes(bytes(data))
    with pytest.raises(BackupDecryptionError, match="not a TrueMemory encrypted backup"):
        decrypt_file(bogus, tmp_path / "out.db", RAW_KEY)


def test_truncated_file_fails(tmp_path: Path, encrypted: Path):
    data = encrypted.read_bytes()
    for cut in (5, 10, len(data) - 8):
        truncated = tmp_path / "trunc.tmbak"
        truncated.write_bytes(data[:cut])
        with pytest.raises(BackupDecryptionError):
            decrypt_file(truncated, tmp_path / "out.db", RAW_KEY)
        assert not (tmp_path / "out.db").exists()


def test_appended_trailing_bytes_fail(tmp_path: Path, encrypted: Path):
    """Bytes appended after the tag shift the tag slot and must be rejected."""
    tampered = tmp_path / "appended.tmbak"
    tampered.write_bytes(encrypted.read_bytes() + b"EVIL-TRAILING-GARBAGE")
    with pytest.raises(BackupDecryptionError, match="wrong key or corrupted"):
        decrypt_file(tampered, tmp_path / "out.db", RAW_KEY)
    assert not (tmp_path / "out.db").exists()


def test_encrypt_rejects_wrong_length_key(tmp_path: Path, plaintext: Path):
    with pytest.raises(BackupKeyError):
        encrypt_file(plaintext, tmp_path / "out.tmbak", b"short")


def test_encrypt_missing_source_cleans_nothing(tmp_path: Path):
    dest = tmp_path / "out.tmbak"
    with pytest.raises(BackupEncryptionError):
        encrypt_file(tmp_path / "missing.db", dest, RAW_KEY)
    assert not dest.exists()


def test_decrypt_missing_source_raises(tmp_path: Path, encrypted: Path):
    with pytest.raises(FileNotFoundError):
        decrypt_file(tmp_path / "missing.tmbak", tmp_path / "out.db", RAW_KEY)


def test_large_file_round_trip(tmp_path: Path):
    """Chunked streaming (1 MiB chunks) must handle multi-chunk payloads."""
    src = tmp_path / "big.bin"
    src.write_bytes(b"\x00\xff" * (1024 * 1024 * 3 // 2))  # ~3 MiB
    enc = tmp_path / "big.tmbak"
    dec = tmp_path / "big.out"
    encrypt_file(src, enc, RAW_KEY)
    decrypt_file(enc, dec, RAW_KEY)
    assert dec.read_bytes() == src.read_bytes()


def test_decrypted_plaintext_is_valid_sqlite(tmp_path: Path, encrypted: Path):
    dec = tmp_path / "out.db"
    decrypt_file(encrypted, dec, RAW_KEY)
    conn = sqlite3.connect(str(dec))
    count = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    conn.close()
    assert count == 50
