"""Trace actual documented Windows ownership calls on retained synthetic files."""
from __future__ import annotations

import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import uuid


def main() -> None:
    if os.name != "nt":
        print(json.dumps({"platform": os.name, "executed": False}))
        return
    import msvcrt
    from truememory.ingest import completion

    directory = Path(tempfile.gettempdir()) / ("tm-windows-owned-diagnostic-" + uuid.uuid4().hex)
    directory.mkdir(mode=0o700)
    source = directory / "synthetic-source.jsonl"
    source.write_text('{"type":"user","message":{"content":"synthetic diagnostic"}}\n', encoding="utf-8")
    calls: list[dict[str, object]] = []
    original = ctypes.WinDLL

    class FunctionTrace:
        def __init__(self, name: str, function: object) -> None:
            object.__setattr__(self, "name", name)
            object.__setattr__(self, "function", function)

        def __getattr__(self, name: str) -> object:
            return getattr(self.function, name)

        def __setattr__(self, name: str, value: object) -> None:
            setattr(self.function, name, value)

        def __call__(self, *args: object) -> object:
            result = self.function(*args)
            row: dict[str, object] = {"api": self.name, "result": result if isinstance(result, (int, bool)) else "pointer", "last_error": ctypes.get_last_error()}
            if self.name == "GetTokenInformation":
                row["information_class"] = args[1]
                row["buffer_bytes"] = args[3]
                row["needed_bytes"] = ctypes.cast(args[4], ctypes.POINTER(ctypes.c_ulong)).contents.value
            if self.name == "GetSecurityInfo":
                row["security_information"] = args[2]
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

    cases: list[dict[str, object]] = []
    for private, flags in ((False, os.O_RDONLY), (True, os.O_RDWR)):
        fd = os.open(source, flags)
        try:
            before = len(calls)
            ctypes.WinDLL = traced_library
            try:
                accepted = completion._windows_security(msvcrt.get_osfhandle(fd), private=private)
            finally:
                ctypes.WinDLL = original
            info = os.fstat(fd)
            path_info = source.lstat()
            cases.append({"private": private, "accepted": accepted, "calls": calls[before:], "fstat": [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns], "lstat": [path_info.st_dev, path_info.st_ino, path_info.st_size, path_info.st_mtime_ns, path_info.st_ctime_ns]})
        finally:
            os.close(fd)
    print(json.dumps({"executed": True, "synthetic_only": True, "source_sha256": hashlib.sha256(inspect.getsource(completion._windows_security).encode()).hexdigest(), "cases": cases}, sort_keys=True))


if __name__ == "__main__":
    main()
