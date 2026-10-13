"""Observe explicit owner and protected DACL at fresh synthetic Windows creation."""
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
MAX_DIAGNOSTIC_ACES = 32
SOURCE_BYTES = b'{"type":"user","message":{"content":"synthetic diagnostic"}}\n'


class DiagnosticError(ValueError):
    pass


def identity(value: os.stat_result) -> list[int]:
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def full9(value: os.stat_result) -> list[int]:
    return [value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_nlink]


def main() -> None:
    if os.name != "nt":
        print(json.dumps({"platform": os.name, "executed": False}))
        return
    import msvcrt
    from ctypes import wintypes
    from truememory.ingest import completion

    parent = Path(tempfile.gettempdir()).resolve()
    directory = parent / ("tm-windows-owned-diagnostic-v4-" + uuid.uuid4().hex)
    source = directory / "synthetic-source.jsonl"
    fd: int | None = None
    file_handle: int | None = None
    directory_handle: int | None = None
    parent_handle: int | None = None
    calls: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    snapshots: list[dict[str, object]] = []
    dacl_observations: list[dict[str, object]] = []
    creations: list[dict[str, object]] = []
    cleanup: list[dict[str, object]] = []
    handle_observations: list[dict[str, object]] = []
    original = ctypes.WinDLL
    phase = "initial"
    trace_bytes = 0
    trace_refusals: list[dict[str, str]] = []
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
            if self.name in {"GetSecurityInfo"}:
                displayed = int(result)
                success = result == 0
                semantics = "DWORD_return_is_authoritative_last_error_unspecified"
            elif self.name in {"GetSecurityDescriptorLength", "GetLengthSid"}:
                displayed = int(result)
                success = result > 0
                semantics = "DWORD_length_return_last_error_unspecified"
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
                                          "GetSecurityInfo", "EqualSid", "IsWellKnownSid"}}
            if self.name == "GetTokenInformation":
                row.update(information_class=int(args[1]), buffer_bytes=int(args[3]),
                           needed_bytes=ctypes.cast(args[4], ctypes.POINTER(wintypes.DWORD)).contents.value)
            elif self.name in {"GetSecurityInfo"}:
                row["security_information"] = int(args[2])
            elif self.name == "IsWellKnownSid":
                row["well_known_type"] = int(args[1])
            elif self.name == "GetAce":
                row["ace_index"] = int(args[1])
            elif self.name == "GetAclInformation":
                row.update(buffer_bytes=int(args[2]), information_class=int(args[3]))
            elif self.name == "CreateFileW":
                row.update(desired_access=int(args[1]), share_mode=int(args[2]),
                           creation_disposition=int(args[4]), flags=int(args[5]))
            if self.name == "CreateDirectoryW":
                row["explicit_security_attributes"] = args[1] is not None
            elif self.name == "InitializeAcl":
                row.update(buffer_bytes=int(args[1]), revision=int(args[2]))
            elif self.name == "AddAccessAllowedAceEx":
                row.update(revision=int(args[1]), flags=int(args[2]), mask=int(args[3]))
            elif self.name == "SetSecurityDescriptorControl":
                row.update(control_mask=int(args[1]), control_set=int(args[2]))
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

    def fresh_observation(opened: int, label: str) -> dict[str, object]:
        if source.parent != directory or source.name != "synthetic-source.jsonl":
            raise DiagnosticError("fixture_path_changed")
        if directory.is_symlink() or source.is_symlink() or source.resolve() != source:
            raise DiagnosticError("fixture_link")
        if file_handle is None:
            raise DiagnosticError("birth_file_handle_not_held")
        fresh_before, path_before, held = os.fstat(opened), source.lstat(), os.fstat(fd)
        for info in (fresh_before, path_before, held):
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or getattr(info, "st_file_attributes", 0) & 0x400:
                raise DiagnosticError("fixture_not_single_regular_no_reparse")
        native_before = native_file_identity(msvcrt.get_osfhandle(opened))
        if native_before != native_file_identity(msvcrt.get_osfhandle(fd)) or native_before != native_file_identity(file_handle):
            raise DiagnosticError("birth_native_identity_changed")
        os.lseek(opened, 0, os.SEEK_SET)
        data = os.read(opened, MAX_BUFFER + 1)
        fresh_after, path_after = os.fstat(opened), source.lstat()
        if full9(fresh_before) != full9(fresh_after) or full9(path_before) != full9(path_after):
            raise DiagnosticError("fresh_read_identity_changed")
        for info in (fresh_after, path_after):
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or getattr(info, "st_file_attributes", 0) & 0x400:
                raise DiagnosticError("fixture_not_single_regular_no_reparse")
        if native_before != native_file_identity(msvcrt.get_osfhandle(opened)) or native_before != native_file_identity(msvcrt.get_osfhandle(fd)) or native_before != native_file_identity(file_handle):
            raise DiagnosticError("birth_native_identity_changed")
        if data != SOURCE_BYTES or any(info.st_size != len(SOURCE_BYTES) for info in (fresh_before, fresh_after, path_before, path_after, held)):
            raise DiagnosticError("fixture_bytes_changed")
        held_fields, path_fields = identity(held), identity(path_after)
        row: dict[str, object] = {
            "phase": label, "identity_policy": "native_volume_index_same_birth_handle_and_fresh; full_fresh_fields_stable_during_read",
            "held_fstat": held_fields,
            "held_fstat_full9": full9(held),
            "fresh_fstat_full9_before_read": full9(fresh_before),
            "fresh_fstat_full9_after_read": full9(fresh_after),
            "fresh_lstat_full9_before_read": full9(path_before),
            "fresh_lstat_full9_after_read": full9(path_after), "fresh_fstat_before_read": identity(fresh_before),
            "fresh_fstat_after_read": identity(fresh_after), "fresh_lstat_before_read": identity(path_before),
            "fresh_lstat_after_read": path_fields, "native_volume_index": list(native_before),
            "native_identity_all_handles_equal": True,
            "held_fstat_minus_fresh_lstat": [left - right for left, right in zip(held_fields, path_fields)],
            "held_fstat_equals_fresh_lstat": held_fields == path_fields,
            "byte_count": len(data), "bytes_hex": data.hex(), "sha256": hashlib.sha256(data).hexdigest()}
        snapshots.append(row)
        return row

    class AbsoluteSecurityDescriptor(ctypes.Structure):
        _fields_ = [("revision", wintypes.BYTE), ("sbz1", wintypes.BYTE),
                    ("control", wintypes.WORD), ("owner", ctypes.c_void_p),
                    ("group", ctypes.c_void_p), ("sacl", ctypes.c_void_p),
                    ("dacl", ctypes.c_void_p)]

    class SecurityAttributes(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("descriptor", ctypes.c_void_p),
                    ("inherit", wintypes.BOOL)]

    def observe_handle(handle: int, label: str) -> None:
        flags = wintypes.DWORD()
        if not kernel.GetHandleInformation(handle, ctypes.byref(flags)):
            raise DiagnosticError("handle_information_failed")
        handle_observations.append({"kind": label, "flags": int(flags.value),
                                    "inheritable": bool(flags.value & 1)})
        if flags.value & 1:
            raise DiagnosticError("created_or_held_handle_inheritable")

    def native_directory_identity(handle: int) -> tuple[int, int, int]:
        info = FileInformation()
        if not kernel.GetFileInformationByHandle(handle, ctypes.byref(info)):
            raise DiagnosticError("native_directory_identity_unavailable")
        if info.attributes & 0x400 or not info.attributes & 0x10:
            raise DiagnosticError("native_directory_not_plain_no_reparse")
        return (info.volume, info.index_high, info.index_low)

    def directory_observation(label: str) -> dict[str, object]:
        if directory.parent != parent or directory.is_symlink() or directory.resolve() != directory:
            raise DiagnosticError("directory_path_or_link_changed")
        before = directory.lstat()
        opened = kernel.CreateFileW(str(directory), 0x00020000 | 0x80, 1 | 2,
                                    None, 3, 0x02000000 | 0x00200000, None)
        if opened is None or opened == ctypes.c_void_p(-1).value:
            raise DiagnosticError("fresh_directory_open_failed")
        try:
            native = native_directory_identity(opened)
            if native != native_directory_identity(directory_handle):
                raise DiagnosticError("directory_native_identity_changed")
            after = directory.lstat()
            if full9(before) != full9(after) or not stat.S_ISDIR(after.st_mode) or getattr(after, "st_file_attributes", 0) & 0x400:
                raise DiagnosticError("directory_fresh_window_changed")
            row = {"phase": label, "kind": "directory", "native_volume_index": list(native),
                   "fresh_lstat_full9_before": full9(before), "fresh_lstat_full9_after": full9(after),
                   "create_then_open_is_atomic": False}
            snapshots.append(row)
            return row
        finally:
            if not kernel.CloseHandle(opened):
                raise DiagnosticError("fresh_directory_close_failed")

    def completion_cases(label: str, *, is_directory: bool) -> None:
        nonlocal phase
        for private in (False, True):
            phase = label + ("_private" if private else "_public")
            opened = None if is_directory else os.open(source, (os.O_RDWR if private else os.O_RDONLY) | os.O_BINARY)
            try:
                before = directory_observation(phase + "_before_security") if is_directory else fresh_observation(opened, phase + "_before_security")
                handle = directory_handle if is_directory else msvcrt.get_osfhandle(opened)
                start = len(calls)
                ctypes.WinDLL = traced_library
                try:
                    accepted = completion._windows_security(handle, private=private)
                finally:
                    ctypes.WinDLL = original
                after = directory_observation(phase + "_after_security") if is_directory else fresh_observation(opened, phase + "_after_security")
                if is_directory:
                    stable = before["fresh_lstat_full9_after"] == after["fresh_lstat_full9_before"]
                else:
                    stable = (before["fresh_fstat_full9_after_read"] == after["fresh_fstat_full9_before_read"]
                              and before["fresh_lstat_full9_after_read"] == after["fresh_lstat_full9_before_read"])
                if not stable:
                    raise DiagnosticError("completion_fresh_window_changed")
                cases.append({"phase": label, "kind": "directory" if is_directory else "file",
                              "private": private, "accepted": accepted, "call_start": start,
                              "call_end": len(calls), "stable_full9_security_window": True,
                              "acceptance_claimed": False})
            finally:
                if opened is not None:
                    os.close(opened)

    try:
        phase = "native_initialization"
        security = traced_library("advapi32", use_last_error=True)
        kernel = traced_library("kernel32", use_last_error=True)
        pointer = ctypes.c_void_p
        security.GetSecurityInfo.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                            ctypes.POINTER(pointer), pointer, ctypes.POINTER(pointer), pointer, ctypes.POINTER(pointer)]
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
        security.GetLengthSid.argtypes = [pointer]
        security.GetLengthSid.restype = wintypes.DWORD
        security.GetAclInformation.argtypes = [pointer, pointer, wintypes.DWORD, wintypes.DWORD]
        security.GetAclInformation.restype = wintypes.BOOL
        security.GetAce.argtypes = [pointer, wintypes.DWORD, ctypes.POINTER(pointer)]
        security.GetAce.restype = wintypes.BOOL
        security.GetSecurityDescriptorLength.argtypes = [pointer]
        security.GetSecurityDescriptorLength.restype = wintypes.DWORD
        security.GetSecurityDescriptorControl.argtypes = [pointer, ctypes.POINTER(wintypes.WORD), ctypes.POINTER(wintypes.DWORD)]
        security.GetSecurityDescriptorControl.restype = wintypes.BOOL
        security.CreateWellKnownSid.argtypes = [wintypes.DWORD, pointer, pointer, ctypes.POINTER(wintypes.DWORD)]
        security.CreateWellKnownSid.restype = wintypes.BOOL
        security.InitializeSecurityDescriptor.argtypes = [pointer, wintypes.DWORD]
        security.InitializeSecurityDescriptor.restype = wintypes.BOOL
        security.SetSecurityDescriptorOwner.argtypes = [pointer, pointer, wintypes.BOOL]
        security.SetSecurityDescriptorOwner.restype = wintypes.BOOL
        security.InitializeAcl.argtypes = [pointer, wintypes.DWORD, wintypes.DWORD]
        security.InitializeAcl.restype = wintypes.BOOL
        security.AddAccessAllowedAceEx.argtypes = [pointer, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, pointer]
        security.AddAccessAllowedAceEx.restype = wintypes.BOOL
        security.SetSecurityDescriptorDacl.argtypes = [pointer, wintypes.BOOL, pointer, wintypes.BOOL]
        security.SetSecurityDescriptorDacl.restype = wintypes.BOOL
        security.SetSecurityDescriptorControl.argtypes = [pointer, wintypes.WORD, wintypes.WORD]
        security.SetSecurityDescriptorControl.restype = wintypes.BOOL
        security.IsValidSecurityDescriptor.argtypes = [pointer]
        security.IsValidSecurityDescriptor.restype = wintypes.BOOL
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.GetHandleInformation.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetHandleInformation.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        kernel.LocalFree.argtypes = [pointer]
        kernel.LocalFree.restype = pointer
        kernel.CreateDirectoryW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(SecurityAttributes)]
        kernel.CreateDirectoryW.restype = wintypes.BOOL
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

        def observe_dacl(handle: int, label: str) -> None:
            nonlocal descriptor, phase
            phase = label + "_dacl_observation"
            dacl = pointer()
            descriptor = pointer()
            class AclSize(ctypes.Structure):
                _fields_ = [("count", wintypes.DWORD), ("used", wintypes.DWORD), ("free", wintypes.DWORD)]
            try:
                if security.GetSecurityInfo(handle, 1, 4, None, None, ctypes.byref(dacl), None, ctypes.byref(descriptor)):
                    raise DiagnosticError("dacl_query_failed")
                if not descriptor or not dacl:
                    raise DiagnosticError("dacl_missing")
                control, revision = wintypes.WORD(), wintypes.DWORD()
                if not security.GetSecurityDescriptorControl(descriptor, ctypes.byref(control), ctypes.byref(revision)):
                    raise DiagnosticError("dacl_control_query_failed")
                length = security.GetSecurityDescriptorLength(descriptor)
                if not 20 <= length <= MAX_BUFFER or not descriptor.value <= dacl.value <= descriptor.value + length - 8:
                    raise DiagnosticError("dacl_descriptor_bound")
                size = AclSize()
                if not security.GetAclInformation(dacl, ctypes.byref(size), ctypes.sizeof(size), 2):
                    raise DiagnosticError("dacl_size_query_failed")
                if size.count > MAX_DIAGNOSTIC_ACES or not 8 <= size.used <= MAX_BUFFER or dacl.value + size.used > descriptor.value + length:
                    raise DiagnosticError("dacl_ace_or_bytes_bound")
                rows: list[dict[str, object]] = []
                for index in range(size.count):
                    ace = pointer()
                    if not security.GetAce(dacl, index, ctypes.byref(ace)) or not ace or not dacl.value + 8 <= ace.value <= dacl.value + size.used - 4:
                        raise DiagnosticError("dacl_ace_pointer_bound")
                    header = ctypes.string_at(ace, 4)
                    ace_bytes = int.from_bytes(header[2:4], "little")
                    if ace_bytes < 4 or ace.value + ace_bytes > dacl.value + size.used:
                        raise DiagnosticError("dacl_ace_size_bound")
                    row: dict[str, object] = {"index": index, "type": header[0], "flags": header[1], "ace_bytes": ace_bytes,
                                              "inherit_only": bool(header[1] & 8), "allow": header[0] == 0,
                                              "mask": int.from_bytes(ctypes.string_at(ace.value + 4, 4), "little") if ace_bytes >= 8 else None,
                                              "simple_allow_or_deny_sid_observed": header[0] in (0, 1),
                                              "sid_equals_token_user": None, "sid_is_local_system": None,
                                              "sid_is_builtin_administrators": None, "sid_is_owner_rights": None}
                    if header[0] in (0, 1):
                        if ace_bytes < 16:
                            raise DiagnosticError("dacl_simple_ace_short")
                        sid = pointer(ace.value + 8)
                        sid_header = ctypes.string_at(sid, 8)
                        sid_bytes = 8 + 4 * sid_header[1]
                        if sid_header[1] > 15 or 8 + sid_bytes > ace_bytes or not security.IsValidSid(sid):
                            raise DiagnosticError("dacl_simple_sid_invalid_or_bound")
                        row.update(sid_equals_token_user=bool(security.EqualSid(sid, user_sid)),
                                   sid_is_local_system=bool(security.IsWellKnownSid(sid, 22)),
                                   sid_is_builtin_administrators=bool(security.IsWellKnownSid(sid, 26)),
                                   sid_is_owner_rights=bool(security.IsWellKnownSid(sid, 71)))
                    rows.append(row)
                dacl_observations.append({"phase": label, "descriptor_bytes": int(length),
                                          "descriptor_revision": int(revision.value),
                                          "descriptor_control": int(control.value),
                                          "dacl_protected": bool(control.value & 0x1000),
                                          "dacl_present": bool(control.value & 4), "acl_bytes_used": int(size.used),
                                          "ace_count": int(size.count), "aces": rows, "acceptance_claimed": False})
            finally:
                if descriptor:
                    if kernel.LocalFree(descriptor):
                        raise DiagnosticError("security_descriptor_free_failed")
                    descriptor = pointer()

        def well_known_sid(kind: int) -> ctypes.c_void_p:
            buffer = ctypes.create_string_buffer(68)
            needed = wintypes.DWORD(len(buffer))
            if not security.CreateWellKnownSid(kind, None, buffer, ctypes.byref(needed)):
                raise DiagnosticError("well_known_sid_creation_failed")
            sid = ctypes.cast(buffer, pointer)
            if not 8 <= needed.value <= len(buffer) or not security.IsValidSid(sid):
                raise DiagnosticError("well_known_sid_bound")
            buffers.append(buffer)
            return sid

        system_sid, admin_sid = well_known_sid(22), well_known_sid(26)

        def birth_attributes(ace_flags: int) -> SecurityAttributes:
            phase_sd = AbsoluteSecurityDescriptor()
            sid_lengths = [int(security.GetLengthSid(sid)) for sid in (user_sid, system_sid, admin_sid)]
            if any(length < 8 or length > 68 for length in sid_lengths):
                raise DiagnosticError("birth_sid_length_bound")
            acl_bytes = 8 + sum(8 + length for length in sid_lengths)
            if not 32 <= acl_bytes <= 236 or acl_bytes % 4:
                raise DiagnosticError("birth_acl_size_bound")
            acl = ctypes.create_string_buffer(acl_bytes)
            if not security.InitializeSecurityDescriptor(ctypes.byref(phase_sd), 1):
                raise DiagnosticError("birth_descriptor_init_failed")
            if not security.SetSecurityDescriptorOwner(ctypes.byref(phase_sd), user_sid, False):
                raise DiagnosticError("birth_descriptor_owner_failed")
            if not security.InitializeAcl(acl, len(acl), 2):
                raise DiagnosticError("birth_acl_init_failed")
            for sid in (user_sid, system_sid, admin_sid):
                if not security.AddAccessAllowedAceEx(acl, 2, ace_flags, 0x001F01FF, sid):
                    raise DiagnosticError("birth_allow_ace_failed")
            if not security.SetSecurityDescriptorDacl(ctypes.byref(phase_sd), True, acl, False):
                raise DiagnosticError("birth_descriptor_dacl_failed")
            if not security.SetSecurityDescriptorControl(ctypes.byref(phase_sd), 0x1000, 0x1000):
                raise DiagnosticError("birth_descriptor_protect_failed")
            if not security.IsValidSecurityDescriptor(ctypes.byref(phase_sd)):
                raise DiagnosticError("birth_descriptor_invalid")
            attributes = SecurityAttributes(ctypes.sizeof(SecurityAttributes), ctypes.addressof(phase_sd), False)
            buffers.extend((acl, phase_sd, attributes))
            return attributes

        phase = "parent_open"
        if parent.is_symlink() or parent.resolve() != parent:
            raise DiagnosticError("parent_link")
        parent_handle = kernel.CreateFileW(str(parent), 0x00020000 | 0x80, 1 | 2, None, 3,
                                          0x02000000 | 0x00200000, None)
        if parent_handle is None or parent_handle == ctypes.c_void_p(-1).value:
            parent_handle = None
            raise DiagnosticError("parent_handle_failed")
        observe_handle(parent_handle, "parent")
        parent_native = native_directory_identity(parent_handle)
        parent_before = full9(parent.lstat())
        phase = "directory_birth_descriptor"
        directory_attributes = birth_attributes(3)
        phase = "directory_CREATE_NEW"
        if not kernel.CreateDirectoryW(str(directory), ctypes.byref(directory_attributes)):
            error = ctypes.get_last_error()
            creations.append({"kind": "directory", "created": False, "last_error": error,
                              "classification": "already_exists" if error in (80, 183) else "native_failure"})
            raise DiagnosticError("directory_creation_refused")
        creations.append({"kind": "directory", "created": True, "descriptor_at_creation": True,
                          "requested_inherit_handle": False, "requested_protected_nonnull_dacl": True,
                          "requested_principals": ["TokenUser", "LocalSystem22", "BuiltinAdministrators26"],
                          "requested_ace_flags": 3, "creation_then_open_atomic": False})
        directory_handle = kernel.CreateFileW(str(directory), 0x00020000 | 0x80, 1 | 2, None, 3,
                                             0x02000000 | 0x00200000, None)
        if directory_handle is None or directory_handle == ctypes.c_void_p(-1).value:
            directory_handle = None
            raise DiagnosticError("new_directory_open_failed")
        observe_handle(directory_handle, "new_directory")
        directory_native = native_directory_identity(directory_handle)
        directory_observation("directory_birth")
        owner_comparison(directory_handle, "directory_birth")
        observe_dacl(directory_handle, "directory_birth")
        completion_cases("directory_birth", is_directory=True)
        phase = "file_birth_descriptor"
        file_attributes = birth_attributes(0)
        phase = "file_CREATE_NEW"
        file_handle = kernel.CreateFileW(str(source), 0x80000000 | 0x40000000 | 0x00020000, 1 | 2,
                                        ctypes.byref(file_attributes), 1, 0x80 | 0x00200000, None)
        if file_handle is None or file_handle == ctypes.c_void_p(-1).value:
            file_handle = None
            error = ctypes.get_last_error()
            creations.append({"kind": "file", "created": False, "last_error": error,
                              "classification": "already_exists" if error in (80, 183) else "native_failure"})
            raise DiagnosticError("file_creation_refused")
        observe_handle(file_handle, "new_file")
        birth_native = native_file_identity(file_handle)
        creations.append({"kind": "file", "created": True, "descriptor_at_creation": True,
                          "requested_inherit_handle": False, "creation_disposition": 1, "requested_ace_flags": 0,
                          "share_delete": False, "native_birth_volume_index": list(birth_native)})
        fd = msvcrt.open_osfhandle(file_handle, os.O_RDWR | os.O_BINARY)
        if fd < 0:
            fd = None
            raise DiagnosticError("birth_handle_fd_transfer_failed")
        remaining = SOURCE_BYTES
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise DiagnosticError("fixture_write_no_progress")
            remaining = remaining[written:]
        os.fsync(fd)
        opened = os.open(source, os.O_RDONLY | os.O_BINARY)
        try:
            fresh_observation(opened, "file_birth")
        finally:
            os.close(opened)
        owner_comparison(file_handle, "file_birth")
        observe_dacl(file_handle, "file_birth")
        completion_cases("file_birth", is_directory=False)
        if native_directory_identity(parent_handle) != parent_native or native_directory_identity(directory_handle) != directory_native:
            raise DiagnosticError("held_directory_native_identity_changed")
        creations.append({"kind": "parent", "native_volume_index": list(parent_native),
                          "native_identity_stable": True, "full9_before": parent_before,
                          "full9_after": full9(parent.lstat()), "timestamps_unchanged_claimed": False})
    except (OSError, ValueError, AttributeError) as error:
        outcome = "diagnostic_incomplete"
        failure = type(error).__name__ + (":" + str(error) if isinstance(error, DiagnosticError) else "")
    finally:
        ctypes.WinDLL = original
        phase = "cleanup"
        if kernel is not None:
            if descriptor:
                cleanup.append({"kind": "descriptor", "closed": not bool(kernel.LocalFree(descriptor))})
            if fd is not None:
                try:
                    os.close(fd)
                    cleanup.append({"kind": "transferred_birth_fd_and_handle", "closed": True})
                except OSError:
                    cleanup.append({"kind": "transferred_birth_fd_and_handle", "closed": False})
                    outcome = "diagnostic_incomplete"
                    failure = failure or "birth_fd_close_failed"
                file_handle = None
            if file_handle is not None:
                cleanup.append({"kind": "untransferred_birth_handle", "closed": bool(kernel.CloseHandle(file_handle))})
            for kind, handle in (("new_directory_handle", directory_handle), ("parent_handle", parent_handle), ("token", token)):
                if handle:
                    cleanup.append({"kind": kind, "closed": bool(kernel.CloseHandle(handle))})
            if any(not item["closed"] for item in cleanup):
                outcome = "diagnostic_incomplete"
                failure = failure or "native_cleanup_failed"
    output = {"schema": "truememory.windows.owned_file.diagnostic.v4", "executed": True,
              "synthetic_only": True, "retained_directory_basename": directory.name,
              "retained_file_basename": source.name, "source_bytes_hex": SOURCE_BYTES.hex(),
              "payload_sha256": hashlib.sha256(SOURCE_BYTES).hexdigest(),
              "source_sha256": hashlib.sha256(inspect.getsource(completion._windows_security).encode()).hexdigest(),
              "outcome": outcome, "failure": failure, "cases": cases, "comparisons": comparisons,
              "snapshots": snapshots, "dacl_observations": dacl_observations, "calls": calls,
              "trace_refusals": trace_refusals, "creations": creations, "cleanup": cleanup,
              "handle_observations": handle_observations,
              "acceptance_claimed": False, "existing_owner_or_DACL_repair": False,
              "token_or_privilege_or_global_changes": False,
              "new_object_security_only": True, "directory_create_then_open_atomic": False,
              "in_memory_layout_bytes": {"pointer": ctypes.sizeof(ctypes.c_void_p),
                                         "absolute_descriptor": ctypes.sizeof(AbsoluteSecurityDescriptor),
                                         "security_attributes": ctypes.sizeof(SecurityAttributes)},
              "original_guard_unchanged": True, "files_deleted": False}
    data = json.dumps(output, sort_keys=True, allow_nan=False)
    if len(data.encode("utf-8")) > MAX_OUTPUT:
        raise DiagnosticError("output_bound")
    print(data)


if __name__ == "__main__":
    main()
