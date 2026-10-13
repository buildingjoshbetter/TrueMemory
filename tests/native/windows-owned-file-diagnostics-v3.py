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
MAX_DIAGNOSTIC_ACES = 32
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

    directory = Path(tempfile.gettempdir()).resolve() / ("tm-windows-owned-diagnostic-v3-" + uuid.uuid4().hex)
    directory.mkdir(mode=0o700)
    source = directory / "synthetic-source.jsonl"
    fd = os.open(source, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_BINARY, 0o600)
    calls: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    snapshots: list[dict[str, object]] = []
    dacl_observations: list[dict[str, object]] = []
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
            elif self.name == "GetSecurityDescriptorLength":
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
                                          "GetSecurityInfo", "SetSecurityInfo", "EqualSid", "IsWellKnownSid"}}
            if self.name == "GetTokenInformation":
                row.update(information_class=int(args[1]), buffer_bytes=int(args[3]),
                           needed_bytes=ctypes.cast(args[4], ctypes.POINTER(wintypes.DWORD)).contents.value)
            elif self.name in {"GetSecurityInfo", "SetSecurityInfo"}:
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

    def snapshot(label: str, *, after_mutation: bool = False) -> dict[str, object]:
        if after_mutation:
            opened = os.open(source, os.O_RDONLY | os.O_BINARY)
            try:
                return fresh_observation(opened, label)
            finally:
                os.close(opened)
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

    def completion_cases(label: str, *, after_mutation: bool = False) -> None:
        nonlocal phase
        for private, flags in ((False, os.O_RDONLY), (True, os.O_RDWR)):
            phase = label + ("_private" if private else "_public")
            opened = os.open(source, flags | os.O_BINARY)
            try:
                if after_mutation:
                    before = fresh_observation(opened, phase + "_before_security")
                elif identity(os.fstat(opened)) != identity(os.fstat(fd)):
                    raise DiagnosticError("completion_descriptor_identity")
                start = len(calls)
                ctypes.WinDLL = traced_library
                try:
                    accepted = completion._windows_security(msvcrt.get_osfhandle(opened), private=private)
                finally:
                    ctypes.WinDLL = original
                after = None
                if after_mutation:
                    after = fresh_observation(opened, phase + "_after_security")
                    if before["fresh_fstat_after_read"] != after["fresh_fstat_before_read"] or before["fresh_lstat_after_read"] != after["fresh_lstat_before_read"]:
                        raise DiagnosticError("completion_fresh_descriptor_changed")
                cases.append({"phase": label, "private": private, "accepted": accepted,
                              "fresh_observation_before": before if after_mutation else None,
                              "fresh_observation_after": after,
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

    def fresh_observation(opened: int, label: str) -> dict[str, object]:
        if source.parent != directory or source.name != "synthetic-source.jsonl":
            raise DiagnosticError("fixture_path_changed")
        if directory.is_symlink() or source.is_symlink() or source.resolve() != source:
            raise DiagnosticError("fixture_link")
        if owner_handle is None:
            raise DiagnosticError("mutation_owner_handle_not_held")
        fresh_before, path_before, held = os.fstat(opened), source.lstat(), os.fstat(fd)
        for info in (fresh_before, path_before, held):
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or getattr(info, "st_file_attributes", 0) & 0x400:
                raise DiagnosticError("fixture_not_single_regular_no_reparse")
        native_before = native_file_identity(msvcrt.get_osfhandle(opened))
        if native_before != native_file_identity(msvcrt.get_osfhandle(fd)) or native_before != native_file_identity(owner_handle):
            raise DiagnosticError("mutation_native_identity_changed")
        os.lseek(opened, 0, os.SEEK_SET)
        data = os.read(opened, MAX_BUFFER + 1)
        fresh_after, path_after = os.fstat(opened), source.lstat()
        if identity(fresh_before) != identity(fresh_after) or identity(path_before) != identity(path_after):
            raise DiagnosticError("fresh_read_identity_changed")
        for info in (fresh_after, path_after):
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or getattr(info, "st_file_attributes", 0) & 0x400:
                raise DiagnosticError("fixture_not_single_regular_no_reparse")
        if native_before != native_file_identity(msvcrt.get_osfhandle(opened)) or native_before != native_file_identity(msvcrt.get_osfhandle(fd)) or native_before != native_file_identity(owner_handle):
            raise DiagnosticError("mutation_native_identity_changed")
        if data != SOURCE_BYTES or any(info.st_size != len(SOURCE_BYTES) for info in (fresh_before, fresh_after, path_before, path_after, held)):
            raise DiagnosticError("fixture_bytes_changed")
        held_fields, path_fields = identity(held), identity(path_after)
        row: dict[str, object] = {
            "phase": label, "identity_policy": "native_volume_index_same_held_owner_and_fresh; full_fresh_fields_stable_during_read",
            "held_fstat": held_fields, "fresh_fstat_before_read": identity(fresh_before),
            "fresh_fstat_after_read": identity(fresh_after), "fresh_lstat_before_read": identity(path_before),
            "fresh_lstat_after_read": path_fields, "native_volume_index": list(native_before),
            "native_identity_all_handles_equal": True,
            "held_fstat_minus_fresh_lstat": [left - right for left, right in zip(held_fields, path_fields)],
            "held_fstat_equals_fresh_lstat": held_fields == path_fields,
            "byte_count": len(data), "bytes_hex": data.hex(), "sha256": hashlib.sha256(data).hexdigest()}
        snapshots.append(row)
        return row

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
        security.GetAclInformation.argtypes = [pointer, pointer, wintypes.DWORD, wintypes.DWORD]
        security.GetAclInformation.restype = wintypes.BOOL
        security.GetAce.argtypes = [pointer, wintypes.DWORD, ctypes.POINTER(pointer)]
        security.GetAce.restype = wintypes.BOOL
        security.GetSecurityDescriptorLength.argtypes = [pointer]
        security.GetSecurityDescriptorLength.restype = wintypes.DWORD
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
                dacl_observations.append({"phase": label, "descriptor_bytes": int(length), "acl_bytes_used": int(size.used),
                                          "ace_count": int(size.count), "aces": rows, "acceptance_claimed": False})
            finally:
                if descriptor:
                    if kernel.LocalFree(descriptor):
                        raise DiagnosticError("security_descriptor_free_failed")
                    descriptor = pointer()

        owner_comparison(msvcrt.get_osfhandle(fd), "before_owner_assignment")
        observe_dacl(msvcrt.get_osfhandle(fd), "before_owner_assignment")
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
        snapshot("after_SetSecurityInfo", after_mutation=True)
        owner_comparison(owner_handle, "after_owner_assignment")
        observe_dacl(owner_handle, "after_owner_assignment")
        completion_cases("after_owner_assignment", after_mutation=True)
        snapshot("after", after_mutation=True)
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
    output = {"schema": "truememory.windows.owned_file.diagnostic.v3", "executed": True,
              "synthetic_only": True, "retained_directory_basename": directory.name,
              "retained_file_basename": source.name, "source_bytes_hex": SOURCE_BYTES.hex(),
              "source_sha256": hashlib.sha256(inspect.getsource(completion._windows_security).encode()).hexdigest(),
              "outcome": outcome, "failure": failure, "cases": cases, "comparisons": comparisons,
              "snapshots": snapshots, "dacl_observations": dacl_observations, "calls": calls, "trace_refusals": trace_refusals,
              "acceptance_claimed": False,
              "token_or_privilege_or_DACL_or_group_changes": False,
              "only_requested_security_information": "OWNER_SECURITY_INFORMATION"}
    data = json.dumps(output, sort_keys=True, allow_nan=False)
    if len(data.encode("utf-8")) > MAX_OUTPUT:
        raise DiagnosticError("output_bound")
    print(data)


if __name__ == "__main__":
    main()
