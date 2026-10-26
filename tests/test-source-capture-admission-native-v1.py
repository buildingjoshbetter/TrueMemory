"""Tiny retained synthetic GPU controls for actual source capture admission.

Linux only. This file does not provide Mac, installed-host or model proof.
"""
from __future__ import annotations

import contextlib
import errno
import io
import os
import signal
import stat
import sys
import types
import unittest
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from truememory.ingest.completion import Capture

ROOT = Path(__file__).resolve().parents[1]
_CASE_NUMBER = 0
_ANNOUNCEMENT_BYTES = 0
MAX_ANNOUNCEMENT_BYTES = 65536
MAX_REGULAR_FIXTURE_BYTES_PER_CASE = 65536
EVIDENCE_ROOT = (
    Path("/results/source-capture-admission-native-v1")
    if Path("/results").is_dir()
    else Path(os.environ.get("TEMP", "/tmp")) / f"truememory-source-capture-admission-native-v1-{os.getpid()}"
)


def announce(path: Path, purpose: str) -> None:
    global _ANNOUNCEMENT_BYTES
    line = f"Create {path}: {purpose}\n"
    _ANNOUNCEMENT_BYTES += len(line.encode("utf-8"))
    if _ANNOUNCEMENT_BYTES > MAX_ANNOUNCEMENT_BYTES:
        raise AssertionError("Synthetic fixture announcements exceeded 65536 bytes")
    print(line, end="", flush=True)


def directory(path: Path) -> None:
    announce(path, "Retained synthetic GPU source-capture fixture directory")
    path.mkdir(mode=0o700)


@contextlib.contextmanager
def actual_completion(home: Path) -> Iterator[types.ModuleType]:
    previous = {name: value for name, value in sys.modules.items()
                if name == "truememory" or name.startswith("truememory.")}
    for name in previous:
        del sys.modules[name]
    try:
        with patch.dict(os.environ, {"HOME": str(home)}, clear=False):
            for name, relative in (("truememory", "truememory"),
                                   ("truememory.ingest", "truememory/ingest"),
                                   ("truememory.ingest.hooks", "truememory/ingest/hooks")):
                module = types.ModuleType(name)
                module.__path__ = [str(ROOT / relative)]
                sys.modules[name] = module
            for name, relative in (("truememory._platform", "truememory/_platform.py"),
                                   ("truememory.ingest.hooks._shared", "truememory/ingest/hooks/_shared.py"),
                                   ("truememory.ingest.completion", "truememory/ingest/completion.py")):
                module = types.ModuleType(name)
                module.__file__ = str(ROOT / relative)
                module.__package__ = name.rpartition(".")[0]
                sys.modules[name] = module
                exec(compile((ROOT / relative).read_bytes(), str(ROOT / relative), "exec", dont_inherit=True), module.__dict__)
            yield sys.modules["truememory.ingest.completion"]
    finally:
        for name in tuple(sys.modules):
            if name == "truememory" or name.startswith("truememory."):
                del sys.modules[name]
        sys.modules.update(previous)


@contextlib.contextmanager
def bounded_call() -> Iterator[None]:
    previous = signal.getsignal(signal.SIGALRM)
    timer = signal.getitimer(signal.ITIMER_REAL)

    def expired(signum: int, frame: object) -> None:
        raise TimeoutError("Synthetic GPU capture call blocked")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 1)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        signal.setitimer(signal.ITIMER_REAL, *timer)


@unittest.skipUnless(sys.platform.startswith("linux"), "Bounded synthetic GPU descriptor controls are Linux only")
class SourceCaptureAdmission(unittest.TestCase):
    def setUp(self) -> None:
        global _CASE_NUMBER
        _CASE_NUMBER += 1
        if not EVIDENCE_ROOT.exists():
            directory(EVIDENCE_ROOT)
        self.base = EVIDENCE_ROOT / f"{os.getpid()}-{_CASE_NUMBER}"
        directory(self.base)
        self.home = self.base / "home"
        directory(self.home)
        directory(self.home / ".claude")
        directory(self.home / ".claude/projects")
        self.source = self.home / ".claude/projects/source-v1.jsonl"
        self.raw = b'{"type":"user","timestamp":"2020-01-01T00:00:00Z","message":{"role":"user","content":"Synthetic scope only."}}\n'
        self.context = actual_completion(self.home)
        self.module = self.context.__enter__()

    def tearDown(self) -> None:
        self.context.__exit__(None, None, None)
        regular_bytes = sum(info.st_size for path in self.base.rglob("*")
                            if stat.S_ISREG((info := path.lstat()).st_mode))
        self.assertLessEqual(regular_bytes, MAX_REGULAR_FIXTURE_BYTES_PER_CASE)
        # All evidence remains in the announced private synthetic directory.

    def regular_source(self) -> None:
        announce(self.source, "Immutable tiny synthetic source version")
        fd = os.open(self.source, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, self.raw)
        finally:
            os.close(fd)

    def capture(self) -> Capture:
        original_open, original_mkdir, original_replace = os.open, os.mkdir, os.replace

        def opened(path: str | Path, flags: int, *args: object, **kwargs: object) -> int:
            if flags & os.O_CREAT:
                announce(Path(path), "Retained synthetic GPU production capture artifact")
            return original_open(path, flags, *args, **kwargs)

        def made(path: str | Path, *args: object, **kwargs: object) -> None:
            announce(Path(path), "Private synthetic GPU production capture directory")
            return original_mkdir(path, *args, **kwargs)

        def replaced(source: str | Path, destination: str | Path, *args: object, **kwargs: object) -> None:
            announce(Path(destination), "Publish synthetic GPU state from its retained temporary")
            return original_replace(source, destination, *args, **kwargs)

        with bounded_call(), patch.object(self.module.os, "open", opened), patch.object(self.module.os, "mkdir", made), patch.object(self.module.os, "replace", replaced):
            return self.module.admit_capture(str(self.source), "synthetic-session-v1", "synthetic-account-v1",
                                            str(self.base / "locator-only-v1.db"), root=self.base / "completion-v1")

    @contextlib.contextmanager
    def source_descriptors(self, before_open: Callable[[int], None] | None = None) -> Iterator[None]:
        original = os.open
        self.opened: list[tuple[int, int, int]] = []
        self.flags: list[int] = []

        def observed(path: str | Path, flags: int, *args: object, **kwargs: object) -> int:
            if Path(path) == self.source:
                if before_open is not None:
                    before_open(len(self.flags))
                self.flags.append(flags)
                fd = original(path, flags, *args, **kwargs)
                info = os.fstat(fd)
                self.opened.append((fd, info.st_dev, info.st_ino))
                return fd
            return original(path, flags, *args, **kwargs)

        with patch.object(self.module.os, "open", observed):
            yield
        for fd, device, inode in self.opened:
            try:
                info = os.fstat(fd)
            except OSError as error:
                self.assertEqual(error.errno, errno.EBADF)
            else:
                self.assertNotEqual((info.st_dev, info.st_ino), (device, inode), "Source descriptor leaked")

    def test_regular_capture_bytes_identity_and_idempotence(self) -> None:
        self.regular_source()
        with self.source_descriptors():
            first = self.capture()
            second = self.capture()
        self.assertEqual(first.key, second.key)
        self.assertEqual(first.path.read_bytes(), self.raw)
        self.assertEqual(first.metadata["byte_count"], len(self.raw))
        self.module.verify_capture(first)
        self.assertEqual(len(self.flags), 4)
        self.assertTrue(all(flags & os.O_NONBLOCK and flags & os.O_NOFOLLOW for flags in self.flags))
        self.assertFalse((self.base / "locator-only-v1.db").exists())

    def test_first_fifo_refused_before_wrapper(self) -> None:
        announce(self.source, "Synthetic GPU FIFO negative; never a Mac fixture")
        os.mkfifo(self.source, 0o600)
        with self.source_descriptors(), patch.object(self.module.os, "fdopen", side_effect=AssertionError("FIFO reached wrapper")):
            with self.assertRaisesRegex(self.module.CompletionError, "source_not_regular"):
                self.capture()
        self.assertTrue(self.flags[0] & os.O_NONBLOCK)

    def replace_source(self, kind: str) -> None:
        backup = self.source.with_name("preserved-source-v1.jsonl")
        announce(backup, "Preserved original synthetic source before pathname replacement")
        os.rename(self.source, backup)
        announce(self.source, "Synthetic GPU replacement " + kind)
        if kind == "FIFO":
            os.mkfifo(self.source, 0o600)
        else:
            self.source.mkdir(mode=0o700)

    def test_second_fifo_refused_without_blocking(self) -> None:
        self.regular_source()

        def replace_on_second(ordinal: int) -> None:
            if ordinal == 1:
                self.replace_source("FIFO")

        with self.source_descriptors(replace_on_second):
            with self.assertRaisesRegex(self.module.CompletionError, "source_not_regular"):
                self.capture()
        self.assertEqual(len(self.flags), 2)
        self.assertTrue(all(flags & os.O_NONBLOCK for flags in self.flags))

    def test_first_directory_refused_before_wrapper(self) -> None:
        directory(self.source)
        with self.source_descriptors(), patch.object(self.module.os, "fdopen", side_effect=AssertionError("Directory reached wrapper")):
            with self.assertRaisesRegex(self.module.CompletionError, "source_not_regular"):
                self.capture()

    def test_second_directory_refused_before_wrapper(self) -> None:
        self.regular_source()

        def replace_on_second(ordinal: int) -> None:
            if ordinal == 1:
                self.replace_source("directory")

        with self.source_descriptors(replace_on_second):
            with self.assertRaisesRegex(self.module.CompletionError, "source_not_regular"):
                self.capture()

    def test_first_and_second_wrapper_failure_closes_raw_descriptor(self) -> None:
        self.regular_source()
        original = os.fdopen
        for target in (1, 2):
            with self.subTest(source_open=target):
                count = 0

                source = os.stat(self.source)

                def failed(fd: int, mode: str) -> io.IOBase:
                    # Count only wrappers of the source itself: a state save
                    # (also os.fdopen) may run first, depending on timing.
                    nonlocal count
                    opened = os.fstat(fd)
                    if (opened.st_dev, opened.st_ino) == (source.st_dev, source.st_ino):
                        count += 1
                        if count == target:
                            raise OSError(errno.EIO, "Synthetic wrapper failure")
                    return original(fd, mode)

                with self.source_descriptors(), patch.object(self.module.os, "fdopen", failed):
                    with self.assertRaisesRegex(self.module.CompletionError, "source_capture_failed"):
                        self.capture()

    def test_valueerror_wrapper_failure_also_closes_raw_descriptor(self) -> None:
        self.regular_source()
        with self.source_descriptors(), patch.object(self.module.os, "fdopen", side_effect=ValueError("Synthetic wrapper failure")):
            with self.assertRaisesRegex(ValueError, "Synthetic wrapper failure"):
                self.capture()

    def test_first_and_second_fstat_failure_closes_raw_descriptor(self) -> None:
        self.regular_source()
        original = os.fstat
        original_open = os.open
        for target in (1, 3):
            with self.subTest(fstat_ordinal=target):
                calls = 0
                fds: list[int] = []

                def observed(path: str | Path, flags: int, *args: object, **kwargs: object) -> int:
                    fd = original_open(path, flags, *args, **kwargs)
                    if Path(path) == self.source:
                        fds.append(fd)
                    return fd

                def failed(fd: int) -> os.stat_result:
                    nonlocal calls
                    if fd in fds:
                        calls += 1
                        if calls == target:
                            raise OSError(errno.EIO, "Synthetic fstat failure")
                    return original(fd)

                with patch.object(self.module.os, "open", observed), patch.object(self.module.os, "fstat", failed):
                    with self.assertRaisesRegex(self.module.CompletionError, "source_capture_failed"):
                        self.capture()
                for fd in fds:
                    with self.assertRaises(OSError) as error:
                        original(fd)
                    self.assertEqual(error.exception.errno, errno.EBADF)


if __name__ == "__main__":
    unittest.main()
