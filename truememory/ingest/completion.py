"""Bounded, retained completion evidence for observed transcript bytes.

Receipts certify one immutable capture, never a whole native turn/session.
Legacy size/PID markers are advisory and do not participate in this protocol.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

log = logging.getLogger(__name__)
MAX_CAPTURE_BYTES = 16 * 1024 * 1024
MAX_ACCOUNT_BYTES = 64 * 1024 * 1024
MAX_CAPTURES = 1024
MAX_STATE_BYTES = 8 * 1024 * 1024
MAX_ATTEMPTS = 5
MAX_RECEIPT_BYTES = 32768
_KEY = re.compile(r"[0-9a-f]{64}")
_COUNTS = ("facts_extracted", "facts_stored", "facts_updated", "facts_skipped_gate", "facts_skipped_dedup")
# Admission refusals that clear once retained captures settle and are released.
BACKPRESSURE = ("capture_account_quota", "capture_count_limit", "retained_state_limit")


class CompletionError(OSError):
    """Retained nonterminal failure; never permission to discard evidence."""


def _json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _text(value: object, limit: int = 4096) -> str:
    if type(value) is not str or not value or "\0" in value:
        raise CompletionError("invalid_identity")
    try:
        if len(value.encode("utf-8")) > limit:
            raise CompletionError("identity_limit")
    except UnicodeEncodeError as error:
        raise CompletionError("invalid_identity") from error
    return value


def normalized_database(value: str | None) -> str:
    return str(Path(value or Path.home() / ".truememory" / "memories.db").expanduser().resolve())


def account_root(user_id: str, db_path: str | None, root: Path | None = None) -> Path:
    if type(user_id) is not str or "\0" in user_id:
        raise CompletionError("invalid_account")
    if user_id:
        _text(user_id)
    return _base(root) / _digest([user_id, normalized_database(db_path)])


def _base(root: Path | None = None) -> Path:
    return root or Path(os.environ.get("TRUEMEMORY_BACKLOG_DIR", str(Path.home() / ".truememory" / "backlog"))).parent / "ingest-completion-v1"


def _private_directory(directory: Path) -> None:
    # Do not follow an existing link or change permissions of foreign paths.
    if directory.is_symlink():
        raise CompletionError("unsafe_directory")
    if os.name == "nt":
        from truememory._windows_private import create_private_directory
        if not directory.parent.exists():
            # A missing drive or share root is its own parent; stop instead of recursing.
            if directory.parent.parent == directory.parent:
                raise CompletionError("unsafe_directory")
            _private_directory(directory.parent)
        try:
            create_private_directory(directory)
        except FileExistsError:
            pass
    else:
        directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    for value in (directory.parent, directory):
        info = value.lstat()
        if not stat.S_ISDIR(info.st_mode) or value.is_symlink():
            raise CompletionError("unsafe_directory")
        if os.name != "nt" and hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise CompletionError("foreign_directory")
        if value == directory and os.name != "nt" and info.st_mode & 0o077:
            raise CompletionError("nonprivate_directory")
    if os.name == "nt":
        _windows_private_directory(directory)


def _account_directory(directory: Path) -> None:
    _private_directory(directory.parent)
    _private_directory(directory)


def _open_private(path: Path, flags: int) -> int:
    if path.is_symlink():
        raise CompletionError("unsafe_file")
    if os.name == "nt" and flags & os.O_CREAT:
        from truememory._windows_private import create_private_file
        try:
            fd = create_private_file(path, flags)
        except FileExistsError:
            if flags & os.O_EXCL:
                raise
            fd = os.open(path, (flags & ~(os.O_CREAT | os.O_EXCL)) | getattr(os, "O_NOFOLLOW", 0), 0o600)
    else:
        fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise CompletionError("unsafe_file")
        if os.name != "nt" and hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise CompletionError("foreign_file")
        if os.name != "nt" and info.st_mode & 0o077:
            raise CompletionError("nonprivate_file")
        if os.name == "nt":
            try:
                import msvcrt
            except ImportError as error:
                raise CompletionError("private_ownership_unavailable") from error
            if not _windows_security(msvcrt.get_osfhandle(fd), private=True):
                raise CompletionError("nonprivate_file")
    except OSError:
        os.close(fd)
        raise
    return fd


@contextlib.contextmanager
def _lock(path: Path, *, blocking: bool = True) -> Iterator[bool]:
    fd = _open_private(path, os.O_CREAT | os.O_RDWR)
    acquired = False
    try:
        if os.name == "nt":
            try:
                import msvcrt
            except ImportError as error:
                raise CompletionError("os_ownership_unavailable") from error
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"0")
            os.lseek(fd, 0, os.SEEK_SET)
            try:
                msvcrt.locking(fd, msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
                acquired = True
            except OSError:
                if blocking:
                    raise
        else:
            try:
                import fcntl
            except ImportError as error:
                raise CompletionError("os_ownership_unavailable") from error
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
                acquired = True
            except BlockingIOError:
                pass
        yield acquired
    finally:
        if acquired:
            if os.name == "nt":
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _read_private(path: Path, limit: int) -> bytes:
    fd = _open_private(path, os.O_RDONLY)
    with os.fdopen(fd, "rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise CompletionError("retained_state_limit")
    return data


def _state(directory: Path) -> dict:
    target = directory / "state.json"
    if target.is_symlink():
        raise CompletionError("unsafe_state")
    if not target.exists():
        return {"version": 1, "captures": {}}
    try:
        value = json.loads(_read_private(target, MAX_STATE_BYTES))
    except (ValueError, UnicodeDecodeError) as error:
        raise CompletionError("invalid_retained_state") from error
    if (type(value) is not dict or type(value.get("version")) is not int
            or value["version"] != 1 or type(value.get("captures")) is not dict
            or len(value["captures"]) > MAX_CAPTURES):
        raise CompletionError("invalid_retained_state")
    return value


def _save_state(directory: Path, value: dict) -> None:
    payload = _json(value)
    retained = sum(path.lstat().st_size for path in directory.glob("state-*.pending"))
    if len(payload) + retained > MAX_STATE_BYTES:
        raise CompletionError("retained_state_limit")
    if (directory / "state.json").is_symlink():
        raise CompletionError("unsafe_state")
    # Unique failed temporaries are retained rather than replacing evidence.
    temporary = directory / ("state-" + uuid.uuid4().hex + ".pending")
    fd = _open_private(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, directory / "state.json")
    _sync_directory(directory)


def _sync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _immutable(path: Path, data: bytes) -> None:
    try:
        fd = _open_private(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if _read_private(path, max(len(data), MAX_STATE_BYTES)) != data:
            raise CompletionError("immutable_evidence_conflict")
        return
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    _sync_directory(path.parent)


def _file_identity(info: os.stat_result) -> list[int]:
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _windows_security(handle: int, *, private: bool = False) -> bool:
    """Compare documented file-owner/token SIDs; unsupported ACLs fail closed."""
    import ctypes
    from ctypes import wintypes
    try:
        security = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        pointer = ctypes.c_void_p
        security.GetSecurityInfo.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(pointer), pointer, ctypes.POINTER(pointer), pointer, ctypes.POINTER(pointer)]
        security.GetSecurityInfo.restype = wintypes.DWORD
        security.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
        security.OpenProcessToken.restype = wintypes.BOOL
        security.GetTokenInformation.argtypes = [wintypes.HANDLE, wintypes.DWORD, pointer, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        security.GetTokenInformation.restype = wintypes.BOOL
        security.EqualSid.argtypes = [pointer, pointer]
        security.EqualSid.restype = wintypes.BOOL
        security.IsWellKnownSid.argtypes = [pointer, wintypes.DWORD]
        security.IsWellKnownSid.restype = wintypes.BOOL
        security.GetAclInformation.argtypes = [pointer, pointer, wintypes.DWORD, wintypes.DWORD]
        security.GetAclInformation.restype = wintypes.BOOL
        security.GetAce.argtypes = [pointer, wintypes.DWORD, ctypes.POINTER(pointer)]
        security.GetAce.restype = wintypes.BOOL
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        kernel.LocalFree.argtypes = [pointer]
        kernel.LocalFree.restype = pointer
        owner, descriptor, dacl = pointer(), pointer(), pointer()
        token = wintypes.HANDLE()
        try:
            if security.GetSecurityInfo(handle, 1, 1 | (4 if private else 0), ctypes.byref(owner), None,
                                        ctypes.byref(dacl) if private else None, None, ctypes.byref(descriptor)):
                return False
            if not owner or not security.OpenProcessToken(kernel.GetCurrentProcess(), 8, ctypes.byref(token)):
                return False
            needed = wintypes.DWORD()
            ctypes.set_last_error(0)
            queried = security.GetTokenInformation(token, 1, None, 0, ctypes.byref(needed))
            if queried or ctypes.get_last_error() != 122 or not 1 <= needed.value <= 65536:
                return False
            buffer = ctypes.create_string_buffer(needed.value)
            if not security.GetTokenInformation(token, 1, buffer, needed.value, ctypes.byref(needed)):
                return False
            user_sid = pointer.from_buffer(buffer)
            if not security.EqualSid(owner, user_sid):
                return False
            if not private:
                return True
            if not dacl:
                return False
            class AclSize(ctypes.Structure):
                _fields_ = [("count", wintypes.DWORD), ("used", wintypes.DWORD), ("free", wintypes.DWORD)]
            size = AclSize()
            if not security.GetAclInformation(dacl, ctypes.byref(size), ctypes.sizeof(size), 2) or size.count > 1024:
                return False
            for index in range(size.count):
                ace = pointer()
                if not security.GetAce(dacl, index, ctypes.byref(ace)):
                    return False
                header = ctypes.string_at(ace, 4)
                if header[0] == 1 or header[1] & 8:
                    continue
                if header[0] != 0 or int.from_bytes(header[2:4], "little") < 16:
                    return False
                sid = pointer(ace.value + 8)
                if not (security.EqualSid(sid, user_sid) or security.IsWellKnownSid(sid, 22) or security.IsWellKnownSid(sid, 26)):
                    return False
            return True
        finally:
            if token:
                kernel.CloseHandle(token)
            if descriptor:
                kernel.LocalFree(descriptor)
    except (OSError, AttributeError, ImportError, ValueError):
        return False


def _windows_private_directory(path: Path) -> None:
    import ctypes
    from ctypes import wintypes
    try:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.CreateFileW(str(path), 0x20080, 7, None, 3, 0x02200000, None)
        if handle in (None, ctypes.c_void_p(-1).value):
            raise CompletionError("private_ownership_unavailable")
        try:
            if not _windows_security(handle, private=True):
                raise CompletionError("private_ownership_unavailable")
        finally:
            kernel.CloseHandle(handle)
    except (OSError, AttributeError, ImportError, ValueError) as error:
        raise CompletionError("private_ownership_unavailable") from error


def _source_owned(info: os.stat_result, fd: int) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise CompletionError("source_not_regular")
    if os.name == "nt":
        try:
            import msvcrt
        except ImportError as error:
            raise CompletionError("source_ownership_unavailable") from error
        if not _windows_security(msvcrt.get_osfhandle(fd)):
            raise CompletionError("source_ownership_unavailable")
    elif hasattr(os, "getuid"):
        if info.st_uid != os.getuid():
            raise CompletionError("source_not_owned")
    else:
        raise CompletionError("source_ownership_unavailable")


def _capture_inventory(directory: Path) -> tuple[int, int]:
    count = total = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            count += 1
            if count > MAX_CAPTURES:
                raise CompletionError("capture_count_limit")
            fd = _open_private(Path(entry.path), os.O_RDONLY)
            try:
                total += os.fstat(fd).st_size
            finally:
                os.close(fd)
    return count, total


@dataclass(frozen=True)
class Capture:
    directory: Path
    key: str
    path: Path
    metadata: dict


def _capture_from(directory: Path, key: str, entry: dict) -> Capture:
    if type(key) is not str or _KEY.fullmatch(key) is None or type(entry) is not dict:
        raise CompletionError("invalid_capture")
    metadata = entry.get("metadata")
    if type(metadata) is not dict or _digest(metadata) != key:
        raise CompletionError("invalid_capture")
    if (type(metadata.get("version")) is not int or metadata["version"] != 1
            or metadata.get("host") not in ("claude", "codex", "gemini", "cursor")
            or type(metadata.get("byte_count")) is not int or not 0 <= metadata["byte_count"] <= MAX_CAPTURE_BYTES
            or type(metadata.get("sha256")) is not str or _KEY.fullmatch(metadata["sha256"]) is None
            or type(metadata.get("file_identity")) is not list or len(metadata["file_identity"]) != 5
            or any(type(value) is not int for value in metadata["file_identity"])
            or type(entry.get("attempts")) is not int or not 0 <= entry["attempts"] <= MAX_ATTEMPTS
            or entry.get("status") not in ("queued", "pending", "running", "retry", "exhausted", "complete", "superseded")
            or type(entry.get("outcomes")) is not list or len(entry["outcomes"]) > MAX_ATTEMPTS):
        raise CompletionError("invalid_capture")
    _text(metadata.get("session_id"), 256)
    _text(metadata.get("source"))
    _text(metadata.get("database"))
    if type(metadata.get("user_id")) is not str or "\0" in metadata["user_id"]:
        raise CompletionError("invalid_capture")
    suffix = ".jsonl" if metadata.get("source_suffix") == ".jsonl" else ".json"
    return Capture(directory, key, directory / "captures" / (key + suffix), metadata)


def _claim_free(directory: Path, key: str) -> bool:
    # Workers hold the per-capture lock for their whole attempt; an unusable
    # lock file counts as held, so collection defers instead of failing.
    path = directory / (key + ".lock")
    if not path.exists():
        return True
    try:
        with _lock(path, blocking=False) as free:
            return free
    except OSError:
        return False


def _activity(entry: dict) -> float:
    times = [0, entry.get("admitted_at", 0)] + [item.get("observed_at", 0) for item in entry.get("outcomes", ()) if type(item) is dict]
    return max(value for value in times if type(value) in (int, float))


def _collect(directory: Path, state: dict) -> None:
    """Release bytes no attempt can still use; receipts and outcomes stay retained.

    Runs under state.lock before admission. State is saved before any unlink,
    so a crash leaves orphan bytes for a later pass, never a live entry without
    its bytes. The oldest released entries are forgotten first, which keeps room
    for one more capture and bounds the retained state.
    """
    from truememory._platform import pid_is_alive
    captures = state["captures"]
    changed = False
    for key, entry in captures.items():
        try:
            capture = _capture_from(directory, key, entry)
        except CompletionError:
            continue
        if entry.get("released") is True or not _claim_free(directory, key):
            continue
        pid = entry.get("worker_pid", 0)
        if (entry["status"] in ("pending", "running") and entry["attempts"] >= MAX_ATTEMPTS
                and not (type(pid) is int and pid > 0 and pid_is_alive(pid))):
            # The final attempt's worker exited without recording an outcome.
            _outcome(entry, "worker_lost")
            changed = True
        if entry["status"] in ("exhausted", "superseded") or has_receipt(capture):
            entry["released"] = changed = True
    released = sorted((key for key, entry in captures.items() if type(entry) is dict and entry.get("released") is True),
                      key=lambda key: _activity(captures[key]))
    size = len(_json(state))
    evicted = []
    for key in released:
        if len(captures) < MAX_CAPTURES and size <= MAX_STATE_BYTES // 2:
            break
        size -= len(_json({key: captures.pop(key)}))
        evicted.append(key)
    if changed or evicted:
        _save_state(directory, state)
    _discard(directory, captures, evicted)


def _discard(directory: Path, captures: dict, forgotten: list[str] | tuple[str, ...] = ()) -> None:
    # Only bytes of released or unknown keys go; a held claim defers to a later pass.
    live = {key for key, entry in captures.items() if not (type(entry) is dict and entry.get("released") is True)}
    with os.scandir(directory / "captures") as entries:
        names = [entry.name for entry in islice(entries, 2 * MAX_CAPTURES)]
    for name in names:
        key = name.split(".", 1)[0]
        if (_KEY.fullmatch(key) and name[len(key):] in (".json", ".jsonl") and key not in live
                and _claim_free(directory, key)):
            with contextlib.suppress(OSError):
                (directory / "captures" / name).unlink()
    for key in forgotten:
        for path in (directory / "logs").glob(key + "-*.log"):
            with contextlib.suppress(OSError):
                path.unlink()
        with contextlib.suppress(OSError):
            (directory / (key + ".lock")).unlink()


def _replaced(directory: Path, captures: dict, incoming: dict, data: bytes) -> list[str]:
    """Queued or retrying captures of the same native source that ``data`` begins with."""
    keys = []
    for key, entry in captures.items():
        try:
            meta = _capture_from(directory, key, entry).metadata
        except CompletionError:
            continue
        if (entry["status"] in ("queued", "retry") and entry.get("released") is not True
                and meta["byte_count"] <= len(data)
                and all(meta[name] == incoming[name] for name in ("host", "session_id", "source"))
                and hashlib.sha256(data[:meta["byte_count"]]).hexdigest() == meta["sha256"]
                and _claim_free(directory, key)):
            keys.append(key)
    return keys


def admit_capture(source: str, session_id: str, user_id: str = "", db_path: str | None = None,
                  host: str = "claude", *, root: Path | None = None) -> Capture:
    from truememory.ingest.hooks._shared import is_allowed_transcript, normalize_transcript_host
    host = normalize_transcript_host(host)
    _text(session_id, 256)
    if session_id == "unknown" or not host or not is_allowed_transcript(source, host=host):
        raise CompletionError("source_not_admitted")
    locator = str(Path(_text(source)).expanduser().resolve())
    original = Path(source).expanduser()
    if original.is_symlink():
        raise CompletionError("source_link")
    try:
        fd = os.open(original, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                     | getattr(os, "O_NONBLOCK", 0))
        try:
            before_stat = os.fstat(fd)
            _source_owned(before_stat, fd)
            handle = os.fdopen(fd, "rb")
            fd = None
            with handle:
                before = _file_identity(before_stat)
                if before[2] > MAX_CAPTURE_BYTES:
                    raise CompletionError("capture_limit")
                data = handle.read(MAX_CAPTURE_BYTES + 1)
                after_stat = os.fstat(handle.fileno())
                _source_owned(after_stat, handle.fileno())
                after = _file_identity(after_stat)
        finally:
            if fd is not None:
                os.close(fd)
        fd = os.open(original, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                     | getattr(os, "O_NONBLOCK", 0))
        try:
            path_stat = os.fstat(fd)
            _source_owned(path_stat, fd)
            handle = os.fdopen(fd, "rb")
            fd = None
            with handle:
                path_after = _file_identity(path_stat)
        finally:
            if fd is not None:
                os.close(fd)
    except CompletionError:
        raise
    except OSError as error:
        raise CompletionError("source_capture_failed") from error
    if len(data) > MAX_CAPTURE_BYTES:
        raise CompletionError("capture_limit")
    if before != after or after != path_after or len(data) != after[2]:
        raise CompletionError("source_changed")
    metadata = {"version": 1, "user_id": user_id, "database": normalized_database(db_path),
                "host": host, "session_id": session_id, "source": locator,
                "source_suffix": original.suffix.lower(), "file_identity": before,
                "byte_count": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    key = _digest(metadata)
    directory = account_root(user_id, db_path, root)
    _account_directory(directory)
    for child in ("captures", "receipts", "logs"):
        _private_directory(directory / child)
    with _lock(directory / "state.lock"):
        state = _state(directory)
        if key in state["captures"]:
            capture = _capture_from(directory, key, state["captures"][key])
            if state["captures"][key].get("released") is not True:
                verify_capture(capture)
            return capture
        suffix = ".jsonl" if metadata["source_suffix"] == ".jsonl" else ".json"
        capture = Capture(directory, key, directory / "captures" / (key + suffix), metadata)
        if has_receipt(capture):
            # Forgotten state entry; the retained receipt still certifies these bytes.
            return capture
        _collect(directory, state)
        replaced = _replaced(directory, state["captures"], metadata, data)
        freed = sum(state["captures"][name]["metadata"]["byte_count"] for name in replaced)
        retained_count, retained_bytes = _capture_inventory(directory / "captures")
        already_retained = (directory / "captures" / (key + suffix)).exists()
        if (retained_count - len(replaced) + int(not already_retained) > MAX_CAPTURES
                or len(state["captures"]) - len(replaced) >= MAX_CAPTURES):
            raise CompletionError("capture_count_limit")
        if retained_bytes - freed + (0 if already_retained else len(data)) > MAX_ACCOUNT_BYTES:
            raise CompletionError("capture_account_quota")
        forgotten = []
        for name in replaced:
            # These exact bytes begin the incoming capture, which covers them.
            state["captures"][name].update(status="superseded", superseded_by=key, released=True)
            if len(state["captures"]) >= MAX_CAPTURES:
                del state["captures"][name]
                forgotten.append(name)
        entry = {"metadata": metadata, "attempts": 0, "status": "queued",
                 "next_attempt_at": 0, "worker_pid": 0, "outcomes": [], "admitted_at": time.time()}
        state["captures"][key] = entry
        if len(_json(state)) > MAX_STATE_BYTES:
            raise CompletionError("retained_state_limit")
        capture = _capture_from(directory, key, entry)
        _immutable(capture.path, data)
        _save_state(directory, state)
        with contextlib.suppress(OSError):
            # Admission is durable; an incomplete cleanup is finished by a later pass.
            _discard(directory, state["captures"], forgotten)
        return capture


def load_capture(key: str, user_id: str, db_path: str | None, host: str,
                 *, root: Path | None = None) -> Capture:
    directory = account_root(user_id, db_path, root)
    _account_directory(directory)
    with _lock(directory / "state.lock"):
        state = _state(directory)
        capture = _capture_from(directory, key, state["captures"].get(key))
    if (capture.metadata["user_id"] != user_id or capture.metadata["database"] != normalized_database(db_path)
            or capture.metadata["host"] != host):
        raise CompletionError("capture_scope_mismatch")
    verify_capture(capture)
    return capture


def verify_capture(capture: Capture) -> None:
    data = _read_private(capture.path, MAX_CAPTURE_BYTES)
    if (len(data) != capture.metadata["byte_count"]
            or hashlib.sha256(data).hexdigest() != capture.metadata["sha256"]):
        raise CompletionError("capture_version_mismatch")


def strict_complete(result: object, capture: Capture | None = None) -> bool:
    coverage = getattr(result, "coverage", None)
    if (type(coverage) is not dict or coverage.get("parser_status") != "complete"
            or coverage.get("extractor_status") != "complete"
            or type(coverage.get("extractor_calls")) is not int or coverage["extractor_calls"] != 1
            or coverage.get("stable") is not True or coverage.get("provenance_observed") is not True):
        return False
    if any(type(coverage.get(name)) is not int or coverage[name] != 0
           for name in ("malformed_records", "unrecognized_records")):
        return False
    if any(type(coverage.get(name)) is not list or coverage[name]
           for name in ("parser_errors", "extractor_errors")):
        return False
    version = coverage.get("source_version")
    if (type(version) is not dict or type(version.get("sha256")) is not str
            or _KEY.fullmatch(version["sha256"]) is None):
        return False
    if type(version.get("byte_count")) is not int or version["byte_count"] < 0:
        return False
    if capture is not None and any(version.get(name) != capture.metadata[name] for name in ("sha256", "byte_count")):
        return False
    values = [getattr(result, name, None) for name in _COUNTS]
    if any(type(value) is not int or value < 0 for value in values) or values[0] != sum(values[1:]):
        return False
    trace = getattr(result, "trace", None)
    if type(trace) is not list or len(trace) != values[0]:
        return False
    actions = ("stored", "updated", "skipped_gate", "skipped_dedup")
    if any(type(item) is not dict or item.get("action") not in actions or "storage_error" in item for item in trace):
        return False
    return all(sum(item["action"] == action for item in trace) == count
               for action, count in zip(actions, values[1:]))


def has_receipt(capture: Capture) -> bool:
    path = capture.directory / "receipts" / (capture.key + ".json")
    if not path.exists():
        return False
    try:
        value = json.loads(_read_private(path, MAX_RECEIPT_BYTES))
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    if not (type(value) is dict and type(value.get("version")) is int and value["version"] == 1
            and value.get("capture_key") == capture.key
            and value.get("scope") == capture.metadata and value.get("coverage") == "observed_bytes"
            and value.get("parser_status") == "complete" and value.get("extractor_status") == "complete"):
        return False
    proof = value.get("proof")
    if (type(proof) is not dict or type(proof.get("counts")) is not dict
            or set(proof["counts"]) != set(_COUNTS) or type(proof.get("actions")) is not list):
        return False
    reconstructed = SimpleNamespace(**proof["counts"], coverage=proof.get("coverage"),
                                    trace=[{"action": action} for action in proof["actions"]])
    return strict_complete(reconstructed, capture)


def _outcome(entry: dict, category: str, details: dict | None = None, *, terminal: bool = False) -> None:
    attempt = entry.get("attempts", 0)
    if type(attempt) is not int or not 0 <= attempt <= MAX_ATTEMPTS:
        raise CompletionError("invalid_attempt_state")
    entry["status"] = "exhausted" if terminal or attempt >= MAX_ATTEMPTS else "retry"
    entry["worker_pid"] = 0
    entry["next_attempt_at"] = time.time() + min(300, 2 ** min(attempt, 9))
    outcome = {"attempt": attempt, "category": category, "observed_at": time.time()}
    if details is not None:
        if len(_json(details)) > 8192:
            raise CompletionError("outcome_limit")
        outcome["details"] = details
    entry["outcomes"].append(outcome)


def record_failure(capture: Capture, category: str, details: dict | None = None, *,
                   terminal: bool = False) -> None:
    with _lock(capture.directory / "state.lock"):
        state = _state(capture.directory)
        entry = state["captures"][capture.key]
        _outcome(entry, category, details, terminal=terminal)
        _save_state(capture.directory, state)


def record_receipt(capture: Capture, result: object) -> bool:
    if not strict_complete(result, capture):
        coverage = getattr(result, "coverage", None)
        details = {"coverage": coverage if type(coverage) is dict else None,
                   "counts": {name: getattr(result, name, None) for name in (
                       "facts_extracted", "facts_stored", "facts_updated", "facts_skipped_gate", "facts_skipped_dedup")},
                   "actions": [entry.get("action") for entry in getattr(result, "trace", []) if type(entry) is dict]}
        record_failure(capture, "coverage_unproven", details)
        return False
    verify_capture(capture)
    receipt = {"version": 1, "capture_key": capture.key, "scope": capture.metadata,
               "coverage": "observed_bytes", "parser_status": "complete", "extractor_status": "complete",
               "proof": {"coverage": result.coverage,
                         "counts": {name: getattr(result, name) for name in _COUNTS},
                         "actions": [entry["action"] for entry in result.trace]}}
    with _lock(capture.directory / "state.lock"):
        state = _state(capture.directory)
        entry = state["captures"][capture.key]
        if has_receipt(capture):
            return True
        if entry["status"] != "running" or entry["worker_pid"] != os.getpid() or not 1 <= entry["attempts"] <= MAX_ATTEMPTS:
            raise CompletionError("receipt_without_owned_attempt")
        payload = _json(receipt)
        if len(payload) > MAX_RECEIPT_BYTES:
            raise CompletionError("receipt_limit")
        _immutable(capture.directory / "receipts" / (capture.key + ".json"), payload)
        state["captures"][capture.key]["status"] = "complete"
        state["captures"][capture.key]["worker_pid"] = 0
        _save_state(capture.directory, state)
    return True


@contextlib.contextmanager
def worker_claim(capture: Capture, attempt: int) -> Iterator[bool]:
    # The OS releases ownership on exit/crash. TTL and PID are not lock owners.
    with _lock(capture.directory / (capture.key + ".lock"), blocking=False) as allowed:
        if not allowed or has_receipt(capture):
            yield False
            return
        with _lock(capture.directory / "state.lock"):
            state = _state(capture.directory)
            entry = state["captures"][capture.key]
            if (type(attempt) is not int or attempt != entry["attempts"]
                    or not 1 <= attempt <= MAX_ATTEMPTS or entry["status"] != "pending"):
                yield False
                return
            entry["status"] = "running"
            entry["worker_pid"] = os.getpid()
            _save_state(capture.directory, state)
        try:
            verify_capture(capture)
            yield True
        finally:
            with _lock(capture.directory / "state.lock"):
                state = _state(capture.directory)
                entry = state["captures"][capture.key]
                if entry["attempts"] == attempt and entry["status"] == "running":
                    _outcome(entry, "worker_unproven")
                    _save_state(capture.directory, state)


def _settled(entry: dict) -> bool:
    return entry["status"] in ("exhausted", "superseded") or entry.get("released") is True


def reserve_inline_attempt(capture: Capture) -> int:
    from truememory._platform import pid_is_alive
    with _lock(capture.directory / (capture.key + ".lock"), blocking=False) as allowed:
        if not allowed or has_receipt(capture):
            return 0
        with _lock(capture.directory / "state.lock"):
            state = _state(capture.directory)
            entry = state["captures"][capture.key]
            if (_settled(entry) or entry["attempts"] >= MAX_ATTEMPTS
                    or entry.get("next_attempt_at", 0) > time.time()):
                return 0
            pid = entry.get("worker_pid", 0)
            if entry["status"] == "pending" and type(pid) is int and pid > 0 and pid_is_alive(pid):
                return 0
            entry["attempts"] += 1
            entry["status"] = "pending"
            entry["worker_pid"] = os.getpid()
            _save_state(capture.directory, state)
            return entry["attempts"]


def dispatch(capture: Capture, threshold: float = 0.30) -> int:
    """One durable attempt before launch; cap/budget denials retain queued work."""
    from truememory.ingest.hooks._shared import check_extraction_budget, refund_extraction_budget
    from truememory.hooks.core import spawn_gate, register_spawned_pid
    from truememory._platform import pid_is_alive, spawn_kwargs
    from truememory.ingest.hooks.stop import MIN_MESSAGES, _has_enough_messages
    directory = capture.directory
    with _lock(directory / "state.lock"):
        entry = _state(directory)["captures"].get(capture.key)
        if entry is None or _settled(entry) or has_receipt(capture):
            return 0
    verify_capture(capture)
    enough = _has_enough_messages(str(capture.path), MIN_MESSAGES)
    with _lock(directory / "state.lock"):
        state = _state(directory)
        entry = state["captures"][capture.key]
        if _settled(entry) or has_receipt(capture) or entry["attempts"] >= MAX_ATTEMPTS:
            return 0
        if not enough:
            # Retained bytes never gain messages; a longer revision is a new capture.
            if entry["status"] in ("queued", "retry"):
                _outcome(entry, "below_minimum_messages", terminal=True)
                _save_state(directory, state)
            return 0
        if entry.get("next_attempt_at", 0) > time.time():
            return 0
        with _lock(directory / (capture.key + ".lock"), blocking=False) as free:
            if not free:
                return 0
        pid = entry.get("worker_pid", 0)
        if entry["status"] == "pending" and type(pid) is int and pid > 0 and pid_is_alive(pid):
            return 0
        if not check_extraction_budget():
            return 0
        with spawn_gate() as allowed:
            if not allowed:
                refund_extraction_budget()
                return 0
            entry["attempts"] += 1
            entry["status"] = "pending"
            entry["worker_pid"] = 0
            _save_state(directory, state)
            attempt = entry["attempts"]
            meta = capture.metadata
            command = [sys.executable, "-m", "truememory.ingest.cli", "ingest", str(capture.path),
                       "--automatic", "--completion-key", capture.key, "--completion-attempt", str(attempt),
                       "--user", meta["user_id"], "--db", meta["database"], "--session", meta["session_id"],
                       "--transcript-host", meta["host"], "--threshold", str(threshold)]
            try:
                log_path = directory / "logs" / (capture.key + "-" + str(attempt) + ".log")
                fd = _open_private(log_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "wb") as output:
                    process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                               stdin=subprocess.DEVNULL, **spawn_kwargs())
                entry["worker_pid"] = process.pid
                _save_state(directory, state)
                register_spawned_pid(process.pid)
                return process.pid
            except OSError as error:
                refund_extraction_budget()
                _outcome(entry, "spawn_failed_" + type(error).__name__)
                _save_state(directory, state)
                log.warning("Completion worker launch failed (%s)", type(error).__name__)
                return 0


def drain(user_id: str = "", db_path: str | None = None, host: str = "claude") -> None:
    directory = account_root(user_id, db_path)
    if not directory.exists():
        return
    _account_directory(directory)
    with _lock(directory / "state.lock"):
        state = _state(directory)
        captures = [_capture_from(directory, key, entry) for key, entry in state["captures"].items()
                    if entry.get("metadata", {}).get("host") == host]
    for capture in captures:
        try:
            if dispatch(capture):
                return
        except OSError as error:
            log.warning("Completion drain retained nonterminal work (%s)", type(error).__name__)


def adopt_legacy_pointer(pointer: Path, scope: tuple[str, str | None, str] | None = None) -> Capture | None:
    """Admit one legacy pointer's current source bytes, then retire the pointer.

    Completion owns retries once the bytes are admitted, so a pointer is never
    replayed. ``scope`` limits adoption to one account and host; without it the
    pointer's own validated scope is used. None retains the pointer (another
    scope, a live claim or an unstable read); ValueError marks it invalid.
    """
    if pointer.suffix not in (".json", ".processing"):
        return None
    from truememory.ingest.hooks._shared import backlog_transcript_source
    from truememory._platform import pid_is_alive
    base = _base()
    _private_directory(base)
    # MCP drainers and SessionStart maintenance adopt one pointer at a time.
    with _lock(base / "legacy.lock"):
        try:
            fd = _open_private(pointer, os.O_RDONLY)
        except FileNotFoundError:
            return None
        with os.fdopen(fd, "rb") as handle:
            before_stat = os.fstat(handle.fileno())
            _source_owned(before_stat, handle.fileno())
            before = _file_identity(before_stat)
            raw = handle.read(16385)
            after_stat = os.fstat(handle.fileno())
            _source_owned(after_stat, handle.fileno())
            after = _file_identity(after_stat)
        fd = _open_private(pointer, os.O_RDONLY)
        with os.fdopen(fd, "rb") as handle:
            path_stat = os.fstat(handle.fileno())
            _source_owned(path_stat, handle.fileno())
            path_after = _file_identity(path_stat)
        if before != after or after != path_after:
            return None
        if len(raw) > 16384 or len(raw) != after[2]:
            raise ValueError("legacy pointer exceeds its bound")
        value = json.loads(raw)
        if type(value) is not dict:
            raise ValueError("legacy pointer is not an object")
        if scope is None:
            from truememory.ingest.hooks._shared import normalize_transcript_host
            user_id, db_path = value.get("user_id"), value.get("db_path")
            host = normalize_transcript_host(value.get("transcript_host", "claude"))
            if type(user_id) is not str or type(db_path) is not str or not host:
                raise ValueError("legacy pointer scope is invalid")
        else:
            user_id, db_path, host = scope
            if (type(value.get("user_id")) is not str or value["user_id"] != user_id
                    or type(value.get("db_path")) is not str
                    or normalized_database(value["db_path"]) != normalized_database(db_path)
                    or value.get("transcript_host") != host):
                return None
        source = backlog_transcript_source(value, host=host)
        if source is None or value.get("session_id") != source[1]:
            raise ValueError("legacy pointer source is not admitted")
        pid = value.get("claimed_pid", 0)
        if type(pid) is not int or pid < 0:
            raise ValueError("legacy pointer claim is invalid")
        if pointer.suffix == ".processing" and (not pid or pid_is_alive(pid)):
            return None
        transcript, session = source
        # A removed transcript can never be admitted; retire its pointer too.
        capture = admit_capture(transcript, session, user_id, db_path, host) if Path(transcript).is_file() else None
        with contextlib.suppress(FileNotFoundError):
            pointer.unlink()
        return capture


def convert_legacy_pointers(user_id: str = "", db_path: str | None = None, host: str = "claude") -> None:
    """Adopt this account's legacy pointers; each retires once its bytes are admitted."""
    directory = Path(os.environ.get("TRUEMEMORY_BACKLOG_DIR", str(Path.home() / ".truememory" / "backlog")))
    if not directory.exists() or directory.is_symlink():
        return
    with os.scandir(directory) as entries:
        pointers = [Path(entry.path) for entry in islice(entries, MAX_CAPTURES)]
    for pointer in pointers:
        try:
            capture = adopt_legacy_pointer(pointer, (user_id, db_path, host))
            if capture is not None and dispatch(capture):
                return
        except (OSError, ValueError, TypeError) as error:
            log.warning("Legacy pointer retained without completion (%s)", type(error).__name__)
