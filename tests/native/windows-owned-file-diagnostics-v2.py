"""Observe default ownership and owner-only assignment on one retained fixture."""
from __future__ import annotations

import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import tempfile
import uuid

MAX_BUFFER = 65536
MAX_OUTPUT = 65536
MAX_CALLS = 256
SOURCE_BYTES = b'{"type":"user","message":{"content":"synthetic diagnostic"}}\n'


class DiagnosticError(ValueError):
    pass


def identity(value: os.stat_result) -> list[int]:
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def main() -> None:
    if os.name != "nt":
        print(json.dumps({"platform": os.name, "executed": False}))
        return
    import msvcrt
    from ctypes import wintypes
    from truememory.ingest import completion

    directory = Path(tempfile.gettempdir()).resolve() / ("tm-windows-owned-diagnostic-v2-" + uuid.uuid4().hex)
    directory.mkdir(mode=0o700)
    source = directory / "synthetic-source.jsonl"
    fd = os.open(source, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_BINARY, 0o600)
    calls: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    snapshots: list[dict[str, object]] = []
    original = ctypes.WinDLL
    phase = "initial"
    trace_bytes = 0
    trace_refusals: list[dict[str, str]] = []
    owner_handle: int | None = None
    token = wintypes.HANDLE()
    descriptor = ctypes.c_void_p()
    kernel: object | None = None
    buffers: list[object] = []
    outcome = "observed"
    failure: str | None = None

    class FunctionTrace:
        def __init__(self, name: str, function: object) -> None:
            object.__setattr__(self, "name", name)
            object.__setattr__(self, "function", function)

        def __getattr__(self, name: str) -> object:
            return getattr(self.function, name)

        def __setattr__(self, name: str, value: object) -> None:
            setattr(self.function, name, value)

        def __call__(self, *args: object) -> object:
            nonlocal trace_bytes
            if (len(calls) >= MAX_CALLS - 8 or trace_bytes > 45056) and self.name not in {"CloseHandle", "LocalFree"}:
                if len(trace_refusals) < 8:
                    trace_refusals.append({"api": self.name, "phase": phase})
                raise DiagnosticError("trace_bound_before_native_call")
            result = self.function(*args)
            error = ctypes.get_last_error()
            if self.name in {"GetSecurityInfo", "SetSecurityInfo"}:
                displayed = int(result)
                success = result == 0
                semantics = "DWORD_return_is_authoritative_last_error_unspecified"
            elif self.name in {"CreateFileW", "GetCurrentProcess"}:
                success = result is not None and result != ctypes.c_void_p(-1).value
                if self.name == "GetCurrentProcess":
                    success = result is not None
                displayed = bool(success)
                semantics = "stale_or_unspecified_on_success" if success else "last_error_on_failure"
            elif self.name == "LocalFree":
                success = not result
                displayed = bool(success)
                semantics = "stale_or_unspecified_on_success" if success else "last_error_on_failure"
            else:
                displayed = bool(result)
                success = bool(result)
                semantics = "stale_or_unspecified_on_success" if success else "last_error_on_failure"
                if self.name in {"EqualSid", "IsWellKnownSid"}:
                    semantics = "last_error_not_authoritative_for_boolean_match_result"
            row: dict[str, object] = {"api": self.name, "phase": phase, "result": displayed,
                                      "last_error": error, "last_error_semantics": semantics,
                                      "last_error_may_be_stale": success or self.name in {
                                          "GetSecurityInfo", "SetSecurityInfo", "EqualSid", "IsWellKnownSid"}}
            if self.name == "GetTokenInformation":
                row.update(information_class=int(args[1]), buffer_bytes=int(args[3]),
                           needed_bytes=ctypes.cast(args[4], ctypes.POINTER(wintypes.DWORD)).contents.value)
            elif self.name in {"GetSecurityInfo", "SetSecurityInfo"}:
                row["security_information"] = int(args[2])
            elif self.name == "IsWellKnownSid":
                row["well_known_type"] = int(args[1])
            elif self.name == "CreateFileW":
                row.update(desired_access=int(args[1]), share_mode=int(args[2]),
                           creation_disposition=int(args[4]), flags=int(args[5]))
            size = len(json.dumps(row, sort_keys=True).encode("utf-8"))
            if size > 1024:
                raise DiagnosticError("trace_bound")
            trace_bytes += size
            calls.append(row)
            return result

    class LibraryTrace:
        def __init__(self, library: object) -> None:
            self.library = library
            self.functions: dict[str, FunctionTrace] = {}

        def __getattr__(self, name: str) -> FunctionTrace:
            if name not in self.functions:
                self.functions[name] = FunctionTrace(name, getattr(self.library, name))
            return self.functions[name]

    def traced_library(*args: object, **kwargs: object) -> LibraryTrace:
        return LibraryTrace(original(*args, **kwargs))

    def snapshot(label: str) -> dict[str, object]:
        if source.parent != directory or source.name != "synthetic-source.jsonl":
            raise DiagnosticError("fixture_path_changed")
        if directory.is_symlink() or source.is_symlink() or source.resolve() != source:
            raise DiagnosticError("fixture_link")
        info, path_info = os.fstat(fd), source.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or identity(info) != identity(path_info):
            raise DiagnosticError("fixture_identity_changed")
        if getattr(info, "st_file_attributes", 0) & 0x400 or getattr(path_info, "st_file_attributes", 0) & 0x400:
            raise DiagnosticError("fixture_reparse_point")
        os.lseek(fd, 0, os.SEEK_SET)
        data = os.read(fd, MAX_BUFFER + 1)
        if data != SOURCE_BYTES or info.st_size != len(SOURCE_BYTES):
            raise DiagnosticError("fixture_bytes_changed")
        row = {"phase": label, "fstat": identity(info), "lstat": identity(path_info),
               "byte_count": len(data), "bytes_hex": data.hex(), "sha256": hashlib.sha256(data).hexdigest()}
        snapshots.append(row)
        return row

    def completion_cases(label: str) -> None:
        nonlocal phase
        for private, flags in ((False, os.O_RDONLY), (True, os.O_RDWR)):
            phase = label + ("_private" if private else "_public")
            opened = os.open(source, flags | os.O_BINARY)
            try:
                if identity(os.fstat(opened)) != identity(os.fstat(fd)):
                    raise DiagnosticError("completion_descriptor_identity")
                start = len(calls)
                ctypes.WinDLL = traced_library
                try:
                    accepted = completion._windows_security(msvcrt.get_osfhandle(opened), private=private)
                finally:
                    ctypes.WinDLL = original
                cases.append({"phase": label, "private": private, "accepted": accepted,
                              "call_start": start, "call_end": len(calls),
                              "fstat": identity(os.fstat(opened)), "lstat": identity(source.lstat())})
            finally:
                os.close(opened)

    class FileTime(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

    class FileInformation(ctypes.Structure):
        _fields_ = [("attributes", wintypes.DWORD), ("creation", FileTime), ("access", FileTime),
                    ("write", FileTime), ("volume", wintypes.DWORD), ("size_high", wintypes.DWORD),
                    ("size_low", wintypes.DWORD), ("links", wintypes.DWORD),
                    ("index_high", wintypes.DWORD), ("index_low", wintypes.DWORD)]

    def native_file_identity(handle: int) -> tuple[int, int, int]:
        info = FileInformation()
        if not kernel.GetFileInformationByHandle(handle, ctypes.byref(info)):
            raise DiagnosticError("native_file_identity_unavailable")
        if info.attributes & 0x400 or info.attributes & 0x10 or info.links != 1:
            raise DiagnosticError("native_file_not_single_regular_no_reparse")
        return (info.volume, info.index_high, info.index_low)

    try:
        remaining = SOURCE_BYTES
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise DiagnosticError("fixture_write_no_progress")
            remaining = remaining[written:]
        os.fsync(fd)
        snapshot("before")
        completion_cases("before_owner_assignment")
        phase = "owner_diagnostic"
        security = traced_library("advapi32", use_last_error=True)
        kernel = traced_library("kernel32", use_last_error=True)
        pointer = ctypes.c_void_p
        security.GetSecurityInfo.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                            ctypes.POINTER(pointer), pointer, pointer, pointer, ctypes.POINTER(pointer)]
        security.GetSecurityInfo.restype = wintypes.DWORD
        security.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
        security.OpenProcessToken.restype = wintypes.BOOL
        security.GetTokenInformation.argtypes = [wintypes.HANDLE, wintypes.DWORD, pointer,
                                               wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        security.GetTokenInformation.restype = wintypes.BOOL
        security.EqualSid.argtypes = [pointer, pointer]
        security.EqualSid.restype = wintypes.BOOL
        security.IsWellKnownSid.argtypes = [pointer, wintypes.DWORD]
        security.IsWellKnownSid.restype = wintypes.BOOL
        security.IsValidSid.argtypes = [pointer]
        security.IsValidSid.restype = wintypes.BOOL
        security.SetSecurityInfo.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                            pointer, pointer, pointer, pointer]
        security.SetSecurityInfo.restype = wintypes.DWORD
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        kernel.LocalFree.argtypes = [pointer]
        kernel.LocalFree.restype = pointer
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, pointer,
                                      wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.GetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.POINTER(FileInformation)]
        kernel.GetFileInformationByHandle.restype = wintypes.BOOL
        if not security.OpenProcessToken(kernel.GetCurrentProcess(), 8, ctypes.byref(token)):
            raise DiagnosticError("token_query_open_failed")

        def token_sid(information_class: int) -> ctypes.c_void_p:
            needed = wintypes.DWORD()
            ctypes.set_last_error(0)
            queried = security.GetTokenInformation(token, information_class, None, 0, ctypes.byref(needed))
            if queried or ctypes.get_last_error() != 122 or not ctypes.sizeof(pointer) <= needed.value <= MAX_BUFFER:
                raise DiagnosticError("token_size_query_failed_or_bound")
            buffer = ctypes.create_string_buffer(needed.value)
            if not security.GetTokenInformation(token, information_class, buffer, len(buffer), ctypes.byref(needed)):
                raise DiagnosticError("token_information_failed")
            if needed.value > len(buffer):
                raise DiagnosticError("token_information_buffer_changed")
            sid = pointer.from_buffer(buffer)
            if not sid or not ctypes.addressof(buffer) <= sid.value < ctypes.addressof(buffer) + len(buffer):
                raise DiagnosticError("token_sid_outside_owned_buffer")
            if not security.IsValidSid(sid):
                raise DiagnosticError("token_sid_invalid")
            buffers.append(buffer)
            return sid

        user_sid, token_owner_sid = token_sid(1), token_sid(4)

        def owner_comparison(handle: int, label: str) -> None:
            nonlocal descriptor
            owner = pointer()
            descriptor = pointer()
            try:
                if security.GetSecurityInfo(handle, 1, 1, ctypes.byref(owner), None, None, None, ctypes.byref(descriptor)):
                    raise DiagnosticError("owner_query_failed")
                if not owner or not security.IsValidSid(owner):
                    raise DiagnosticError("owner_sid_invalid")
                comparisons.append({"phase": label,
                                    "owner_is_builtin_administrators": bool(security.IsWellKnownSid(owner, 26)),
                                    "owner_is_local_system": bool(security.IsWellKnownSid(owner, 22)),
                                    "owner_equals_token_user": bool(security.EqualSid(owner, user_sid)),
                                    "owner_equals_token_owner": bool(security.EqualSid(owner, token_owner_sid)),
                                    "token_owner_equals_token_user": bool(security.EqualSid(token_owner_sid, user_sid))})
            finally:
                if descriptor:
                    if kernel.LocalFree(descriptor):
                        raise DiagnosticError("security_descriptor_free_failed")
                    descriptor = pointer()

        owner_comparison(msvcrt.get_osfhandle(fd), "before_owner_assignment")
        snapshot("before_CreateFileW")
        phase = "owner_assignment"
        owner_handle = kernel.CreateFileW(str(source), 0x00020000 | 0x00080000, 1 | 2,
                                         None, 3, 0x80 | 0x00200000, None)
        if owner_handle is None or owner_handle == ctypes.c_void_p(-1).value:
            owner_handle = None
            raise DiagnosticError("owner_assignment_handle_denied")
        if native_file_identity(owner_handle) != native_file_identity(msvcrt.get_osfhandle(fd)):
            raise DiagnosticError("owner_assignment_wrong_file")
        snapshot("before_SetSecurityInfo")
        if security.SetSecurityInfo(owner_handle, 1, 1, user_sid, None, None, None):
            outcome = "owner_assignment_returned_error"
        snapshot("after_SetSecurityInfo")
        owner_comparison(owner_handle, "after_owner_assignment")
        completion_cases("after_owner_assignment")
        snapshot("after")
    except (OSError, ValueError, AttributeError) as error:
        outcome = "diagnostic_incomplete"
        failure = type(error).__name__ + (":" + str(error) if isinstance(error, DiagnosticError) else "")
    finally:
        ctypes.WinDLL = original
        phase = "cleanup"
        if kernel is not None:
            if descriptor:
                kernel.LocalFree(descriptor)
            if owner_handle is not None:
                kernel.CloseHandle(owner_handle)
            if token:
                kernel.CloseHandle(token)
        os.close(fd)
    output = {"schema": "truememory.windows.owned_file.diagnostic.v2", "executed": True,
              "synthetic_only": True, "retained_directory_basename": directory.name,
              "retained_file_basename": source.name, "source_bytes_hex": SOURCE_BYTES.hex(),
              "source_sha256": hashlib.sha256(inspect.getsource(completion._windows_security).encode()).hexdigest(),
              "outcome": outcome, "failure": failure, "cases": cases, "comparisons": comparisons,
              "snapshots": snapshots, "calls": calls, "trace_refusals": trace_refusals,
              "acceptance_claimed": False,
              "token_or_privilege_or_DACL_or_group_changes": False,
              "only_requested_security_information": "OWNER_SECURITY_INFORMATION"}
    data = json.dumps(output, sort_keys=True, allow_nan=False)
    if len(data.encode("utf-8")) > MAX_OUTPUT:
        raise DiagnosticError("output_bound")
    print(data)


if __name__ == "__main__":
    main()
