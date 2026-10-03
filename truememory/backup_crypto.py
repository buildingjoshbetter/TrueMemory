"""Authenticated encryption for TrueMemory cloud backups (issue #199, Phase 1).

Format (versioned, little-endian not applicable, raw bytes)::

    offset  size  field
    0       5     magic, b"TMBAK"
    5       1     format version (currently 1)
    6       1     algorithm id (1 = AES-256-GCM)
    7       12    random nonce
    19      N     ciphertext
    19+N    16    GCM authentication tag

The entire header is passed to AES-GCM as additional authenticated data, so any
change to the version, algorithm, or nonce makes decryption fail with an
authentication error. The key is user-controlled, exactly 32 bytes, and is
transported as standard base64. Encryption streams in chunks so large databases
do not need to fit in memory.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import os
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from truememory.backup_errors import (
    BackupDecryptionError,
    BackupEncryptionError,
    BackupKeyError,
)

_MAGIC = b"TMBAK"
FORMAT_VERSION = 1
ALGORITHM_AES_256_GCM = 1
_KEY_SIZE = 32
_NONCE_SIZE = 12
_TAG_SIZE = 16
_HEADER_SIZE = len(_MAGIC) + 2 + _NONCE_SIZE
_CHUNK_SIZE = 1024 * 1024


def generate_key() -> str:
    """Return a fresh random 256-bit key as standard base64 text."""
    return base64.b64encode(os.urandom(_KEY_SIZE)).decode("ascii")


def load_key(raw: str | bytes | None) -> bytes:
    """Decode and validate a user-supplied backup encryption key.

    Accepts raw 32-byte key material or standard base64 text (padded or
    unpadded). Raises :class:`BackupKeyError` on anything else; error messages
    include only lengths, never the key itself.
    """
    if raw is None:
        raise BackupKeyError(
            "no backup encryption key configured. Set TRUEMEMORY_BACKUP_KEY to "
            "the base64 key (generate one with "
            "`truememory-ingest backup --generate-key`)."
        )
    if isinstance(raw, bytes):
        key = raw
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise BackupKeyError("backup encryption key is empty")
        padded = text + "=" * (-len(text) % 4)
        try:
            key = base64.b64decode(padded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise BackupKeyError(
                "backup encryption key is not valid base64; expected the "
                "standard base64 encoding of a 32-byte key"
            ) from exc
    else:
        raise BackupKeyError(
            f"backup encryption key must be str or bytes, got {type(raw).__name__}"
        )
    if len(key) != _KEY_SIZE:
        raise BackupKeyError(
            f"backup encryption key must decode to {_KEY_SIZE} bytes, got {len(key)}"
        )
    return key


def _require_raw_key(key: bytes) -> bytes:
    if not isinstance(key, bytes) or len(key) != _KEY_SIZE:
        raise BackupKeyError(
            f"backup encryption key must be raw {_KEY_SIZE}-byte material; "
            "use load_key() to decode a configured key"
        )
    return key


def _build_header(nonce: bytes) -> bytes:
    return _MAGIC + bytes((FORMAT_VERSION, ALGORITHM_AES_256_GCM)) + nonce


def _parse_header(header: bytes) -> bytes:
    """Validate *header* and return its nonce."""
    if len(header) < _HEADER_SIZE:
        raise BackupDecryptionError("backup file is truncated or not a TrueMemory backup")
    if header[: len(_MAGIC)] != _MAGIC:
        raise BackupDecryptionError("backup file is not a TrueMemory encrypted backup")
    version = header[len(_MAGIC)]
    if version != FORMAT_VERSION:
        raise BackupDecryptionError(
            f"unsupported backup format version {version} "
            f"(this build supports version {FORMAT_VERSION})"
        )
    algorithm = header[len(_MAGIC) + 1]
    if algorithm != ALGORITHM_AES_256_GCM:
        raise BackupDecryptionError(
            f"unsupported backup encryption algorithm id {algorithm}"
        )
    return header[len(_MAGIC) + 2 :]


def read_format_version(path: str | Path) -> int:
    """Return the format version stored in an encrypted backup header.

    Raises :class:`BackupDecryptionError` if the header is not a TrueMemory
    backup. Does not decrypt or authenticate the payload.
    """
    with open(path, "rb") as f:
        header = f.read(_HEADER_SIZE)
    _parse_header(header)
    return header[len(_MAGIC)]


def encrypt_file(src_path: str | Path, dest_path: str | Path, key: bytes) -> int:
    """Encrypt *src_path* into *dest_path*, returning the format version.

    The destination is removed if encryption fails partway through, so a
    partial ciphertext can never be uploaded.
    """
    key = _require_raw_key(key)
    src = Path(src_path)
    dest = Path(dest_path)
    nonce = os.urandom(_NONCE_SIZE)
    header = _build_header(nonce)
    encryptor = Cipher(algorithms.AES(key), modes.GCM(nonce)).encryptor()
    encryptor.authenticate_additional_data(header)
    completed = False
    try:
        with open(src, "rb") as f_in, open(dest, "wb") as f_out:
            f_out.write(header)
            while True:
                chunk = f_in.read(_CHUNK_SIZE)
                if not chunk:
                    break
                f_out.write(encryptor.update(chunk))
            f_out.write(encryptor.finalize())
            f_out.write(encryptor.tag)
        completed = True
    except OSError as exc:
        raise BackupEncryptionError(f"could not encrypt snapshot: {exc}") from exc
    finally:
        if not completed:
            with contextlib.suppress(OSError):
                dest.unlink(missing_ok=True)
    return FORMAT_VERSION


def decrypt_file(src_path: str | Path, dest_path: str | Path, key: bytes) -> int:
    """Decrypt *src_path* into *dest_path*, returning the format version.

    Raises :class:`BackupDecryptionError` for a wrong key, tampered ciphertext,
    modified header, unsupported version, or truncated file. The destination is
    removed when decryption fails so a partial plaintext file is never left
    behind.
    """
    key = _require_raw_key(key)
    src = Path(src_path)
    dest = Path(dest_path)
    size = src.stat().st_size
    if size < _HEADER_SIZE + _TAG_SIZE:
        raise BackupDecryptionError("backup file is truncated")
    completed = False
    try:
        with open(src, "rb") as f_in:
            header = f_in.read(_HEADER_SIZE)
            nonce = _parse_header(header)
            f_in.seek(size - _TAG_SIZE)
            tag = f_in.read(_TAG_SIZE)
            if len(tag) != _TAG_SIZE:
                raise BackupDecryptionError("backup file is truncated")
            f_in.seek(_HEADER_SIZE)
            decryptor = Cipher(algorithms.AES(key), modes.GCM(nonce, tag)).decryptor()
            decryptor.authenticate_additional_data(header)
            remaining = size - _HEADER_SIZE - _TAG_SIZE
            with open(dest, "wb") as f_out:
                while remaining > 0:
                    chunk = f_in.read(min(_CHUNK_SIZE, remaining))
                    if not chunk:
                        raise BackupDecryptionError("backup file is truncated")
                    remaining -= len(chunk)
                    f_out.write(decryptor.update(chunk))
                decryptor.finalize()
        completed = True
    except InvalidTag as exc:
        raise BackupDecryptionError(
            "backup could not be decrypted: wrong key or corrupted/tampered backup"
        ) from exc
    finally:
        if not completed:
            with contextlib.suppress(OSError):
                dest.unlink(missing_ok=True)
    return FORMAT_VERSION
