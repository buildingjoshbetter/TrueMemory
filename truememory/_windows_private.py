"""Fresh Windows objects with an explicit current-user owner and private DACL.

Existing objects are never adopted, repaired or given a new security descriptor.
CreateDirectoryW followed by opening the new directory is not an atomic birth
identity proof; callers must retain their ordinary post-creation ownership checks.
"""
from __future__ import annotations

import contextlib
import ctypes
import errno
import os
from pathlib import Path
from typing import Iterator


class _SecurityDescriptor(ctypes.Structure):
    _fields_ = [("revision", ctypes.c_uint8), ("sbz1", ctypes.c_uint8),
                ("control", ctypes.c_uint16), ("owner", ctypes.c_void_p),
                ("group", ctypes.c_void_p), ("sacl", ctypes.c_void_p),
                ("dacl", ctypes.c_void_p)]


class _SecurityAttributes(ctypes.Structure):
    _fields_ = [("length", ctypes.c_uint32), ("descriptor", ctypes.c_void_p),
                ("inherit", ctypes.c_int32)]


class _FileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]


class _FileInformation(ctypes.Structure):
    _fields_ = [("attributes", ctypes.c_uint32), ("creation", _FileTime),
                ("access", _FileTime), ("write", _FileTime),
                ("volume", ctypes.c_uint32), ("size_high", ctypes.c_uint32),
                ("size_low", ctypes.c_uint32), ("links", ctypes.c_uint32),
                ("index_high", ctypes.c_uint32), ("index_low", ctypes.c_uint32)]


def _libraries() -> tuple[object, object]:
    if os.name != "nt":
        raise OSError(errno.ENOSYS, "Windows private creation is unavailable")
    security = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    pointer, dword, boolean, handle = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32, ctypes.c_void_p
    bindings = (
        (security, "OpenProcessToken", [handle, dword, ctypes.POINTER(handle)], boolean),
        (security, "GetTokenInformation", [handle, dword, pointer, dword, ctypes.POINTER(dword)], boolean),
        (security, "IsValidSid", [pointer], boolean),
        (security, "GetLengthSid", [pointer], dword),
        (security, "CreateWellKnownSid", [dword, pointer, pointer, ctypes.POINTER(dword)], boolean),
        (security, "InitializeSecurityDescriptor", [pointer, dword], boolean),
        (security, "SetSecurityDescriptorOwner", [pointer, pointer, boolean], boolean),
        (security, "InitializeAcl", [pointer, dword, dword], boolean),
        (security, "AddAccessAllowedAceEx", [pointer, dword, dword, dword, pointer], boolean),
        (security, "SetSecurityDescriptorDacl", [pointer, boolean, pointer, boolean], boolean),
        (security, "SetSecurityDescriptorControl", [pointer, ctypes.c_uint16, ctypes.c_uint16], boolean),
        (security, "IsValidSecurityDescriptor", [pointer], boolean),
        (kernel, "GetCurrentProcess", [], handle),
        (kernel, "CloseHandle", [handle], boolean),
        (kernel, "GetHandleInformation", [handle, ctypes.POINTER(dword)], boolean),
        (kernel, "GetFileType", [handle], dword),
        (kernel, "GetFileInformationByHandle", [handle, ctypes.POINTER(_FileInformation)], boolean),
        (kernel, "CreateDirectoryW", [ctypes.c_wchar_p, pointer], boolean),
        (kernel, "CreateFileW", [ctypes.c_wchar_p, dword, dword, pointer, dword, dword, handle], handle),
    )
    for library, name, arguments, result in bindings:
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, result
    return security, kernel


def _error(reason: str, path: Path | None = None) -> OSError:
    code = ctypes.get_last_error()
    if code in (80, 183):
        return FileExistsError(errno.EEXIST, reason, str(path) if path is not None else None)
    return ctypes.WinError(code)


def _close(kernel: object, handle: int) -> None:
    if not kernel.CloseHandle(handle):
        raise _error("private_creation_close_failed")


def _path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if (not path.is_absolute() or ".." in path.parts or "\0" in str(path)
            or path == path.parent or len(str(path)) > 4096 or len(path.parents) > 64):
        raise ValueError("Private birth requires an absolute child path without traversal")
    return path


def _information(kernel: object, handle: int, *, directory: bool) -> tuple[int, int, int]:
    if not directory and kernel.GetFileType(handle) != 1:
        raise OSError(errno.EPERM, "Private creation requires a native disk file")
    flags = ctypes.c_uint32()
    if not kernel.GetHandleInformation(handle, ctypes.byref(flags)):
        raise _error("private_creation_handle_information_failed")
    if flags.value & 1:
        raise OSError(errno.EPERM, "Private creation handle is inheritable")
    info = _FileInformation()
    if not kernel.GetFileInformationByHandle(handle, ctypes.byref(info)):
        raise _error("private_creation_identity_failed")
    if info.attributes & 0x400 or bool(info.attributes & 0x10) != directory:
        raise OSError(errno.EPERM, "Private creation requires a plain no-reparse object")
    if not directory and info.links != 1:
        raise OSError(errno.EPERM, "Private creation requires one regular file link")
    return info.volume, info.index_high, info.index_low


@contextlib.contextmanager
def _parents(kernel: object, path: Path) -> Iterator[None]:
    held: list[int] = []
    try:
        for parent in reversed(path.parents):
            handle = kernel.CreateFileW(str(parent), 0x20080, 3, None, 3, 0x02200000, None)
            if handle in (None, ctypes.c_void_p(-1).value):
                raise _error("private_creation_parent_unavailable", parent)
            held.append(handle)
            _information(kernel, handle, directory=True)
        yield
    finally:
        failure: OSError | None = None
        for handle in reversed(held):
            try:
                _close(kernel, handle)
            except OSError as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure


@contextlib.contextmanager
def _birth_security(security: object, kernel: object, *, directory: bool) -> Iterator[_SecurityAttributes]:
    token = ctypes.c_void_p()
    if not security.OpenProcessToken(kernel.GetCurrentProcess(), 8, ctypes.byref(token)):
        raise _error("private_creation_token_unavailable")
    try:
        needed = ctypes.c_uint32()
        ctypes.set_last_error(0)
        queried = security.GetTokenInformation(token, 1, None, 0, ctypes.byref(needed))
        if queried or ctypes.get_last_error() != 122 or not ctypes.sizeof(ctypes.c_void_p) + 4 <= needed.value <= 65536:
            raise OSError(errno.EPERM, "Private creation token information is unbounded")
        buffer = ctypes.create_string_buffer(needed.value)
        capacity = len(buffer)
        if not security.GetTokenInformation(token, 1, buffer, capacity, ctypes.byref(needed)):
            raise _error("private_creation_token_information_failed")
        if not 1 <= needed.value <= capacity:
            raise OSError(errno.EPERM, "Private creation token information changed size")
        user = ctypes.c_void_p.from_buffer(buffer)
        start = ctypes.addressof(buffer)
        if user.value is None or not start <= user.value <= start + needed.value - 8:
            raise OSError(errno.EPERM, "Private creation token SID is outside its buffer")
        header = ctypes.string_at(user, 2)
        sid_bytes = 8 + 4 * header[1]
        if header[0] != 1 or header[1] > 15 or user.value + sid_bytes > start + needed.value:
            raise OSError(errno.EPERM, "Private creation token SID is malformed")
        if not security.IsValidSid(user) or security.GetLengthSid(user) != sid_bytes:
            raise OSError(errno.EPERM, "Private creation token SID is invalid")
        well_known_buffers: list[ctypes.Array] = []
        sids = [user]
        sizes = [sid_bytes]
        for kind in (22, 26):
            sid = ctypes.create_string_buffer(68)
            size = ctypes.c_uint32(68)
            if not security.CreateWellKnownSid(kind, None, sid, ctypes.byref(size)):
                raise _error("private_creation_well_known_sid_failed")
            if not 8 <= size.value <= 68 or not security.IsValidSid(sid) or security.GetLengthSid(sid) != size.value:
                raise OSError(errno.EPERM, "Private creation well-known SID is invalid")
            well_known_buffers.append(sid)
            sids.append(ctypes.cast(sid, ctypes.c_void_p))
            sizes.append(size.value)
        acl = ctypes.create_string_buffer(8 + sum(8 + size for size in sizes))
        descriptor = _SecurityDescriptor()
        if not security.InitializeSecurityDescriptor(ctypes.byref(descriptor), 1):
            raise _error("private_creation_descriptor_failed")
        if not security.SetSecurityDescriptorOwner(ctypes.byref(descriptor), user, False):
            raise _error("private_creation_owner_failed")
        if not security.InitializeAcl(acl, len(acl), 2):
            raise _error("private_creation_acl_failed")
        for sid in sids:
            if not security.AddAccessAllowedAceEx(acl, 2, 3 if directory else 0, 0x001F01FF, sid):
                raise _error("private_creation_ace_failed")
        if not security.SetSecurityDescriptorDacl(ctypes.byref(descriptor), True, acl, False):
            raise _error("private_creation_dacl_failed")
        if not security.SetSecurityDescriptorControl(ctypes.byref(descriptor), 0x1000, 0x1000):
            raise _error("private_creation_dacl_protection_failed")
        if not security.IsValidSecurityDescriptor(ctypes.byref(descriptor)):
            raise OSError(errno.EPERM, "Private creation descriptor is invalid")
        attributes = _SecurityAttributes(ctypes.sizeof(_SecurityAttributes), ctypes.addressof(descriptor), False)
        # Token SID, well-known SID, ACL and absolute descriptor storage stay live through the native birth.
        yield attributes
    finally:
        if token:
            _close(kernel, token)


def create_private_directory(value: str | os.PathLike[str]) -> None:
    """Create one new directory; an existing name raises FileExistsError."""
    path = _path(value)
    security, kernel = _libraries()
    with _parents(kernel, path), _birth_security(security, kernel, directory=True) as attributes:
        if not kernel.CreateDirectoryW(str(path), ctypes.byref(attributes)):
            raise _error("private_directory_create_failed", path)
        # This separate open cannot establish an atomic CreateDirectoryW birth identity.
        handle = kernel.CreateFileW(str(path), 0x20080, 3, None, 3, 0x02200000, None)
        if handle in (None, ctypes.c_void_p(-1).value):
            raise _error("private_directory_open_failed", path)
        try:
            _information(kernel, handle, directory=True)
        finally:
            _close(kernel, handle)


def create_private_file(value: str | os.PathLike[str], flags: int) -> int:
    """Create a new file and transfer its no-share-delete handle into one CRT fd."""
    if type(flags) is not int or flags not in (os.O_CREAT | os.O_RDWR, os.O_CREAT | os.O_EXCL | os.O_WRONLY):
        raise ValueError("Private birth requires one of the two completion creation flag combinations")
    path = _path(value)
    security, kernel = _libraries()
    import msvcrt
    handle = None
    fd = None
    try:
        with _parents(kernel, path), _birth_security(security, kernel, directory=False) as attributes:
            access = 0x20080
            if flags & os.O_RDWR:
                access |= 0x80000000 | 0x40000000
            elif flags & os.O_WRONLY:
                access |= 0x40000000
            else:
                access |= 0x80000000
            handle = kernel.CreateFileW(str(path), access, 3, ctypes.byref(attributes), 1, 0x00200080, None)
            if handle in (None, ctypes.c_void_p(-1).value):
                handle = None
                raise _error("private_file_create_failed", path)
            identity = _information(kernel, handle, directory=False)
            from truememory.ingest.completion import _windows_security
            if not _windows_security(handle, private=True):
                raise OSError(errno.EPERM, "Private creation failed the original private ownership guard")
            transfer = flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND)
            transfer |= os.O_BINARY | os.O_NOINHERIT
            fd = msvcrt.open_osfhandle(int(handle), transfer)
            handle = None
            if _information(kernel, msvcrt.get_osfhandle(fd), directory=False) != identity:
                raise OSError(errno.EPERM, "Private file identity changed during fd transfer")
        return fd
    except BaseException:
        if fd is not None:
            os.close(fd)
        raise
    finally:
        if handle is not None:
            _close(kernel, handle)
