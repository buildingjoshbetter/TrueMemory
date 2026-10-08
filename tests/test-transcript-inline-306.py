"""Legacy transcript path handling must not expose or ingest source names."""
from __future__ import annotations

import errno
import json
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SENTINEL = "synthetic-private-content-sentinel"
PATH_SENTINEL = "synthetic-private-path-sentinel"


def load_parser() -> types.ModuleType:
    path = Path(__file__).resolve().parents[1] / "truememory/ingest/transcript.py"
    module = types.ModuleType("inline_boundary_parser")
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


class InlineBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parser = load_parser()

    def assert_private_failure(self, source: str | Path) -> None:
        with self.assertLogs(self.parser.log, level="ERROR") as captured:
            self.assertEqual(self.parser.parse_transcript(source), [])
        self.assertEqual(len(captured.records), 1)
        record = captured.records[0]
        self.assertEqual(getattr(record, "error_category", None), "source_unreadable")
        self.assertEqual(record.args, ())
        self.assertIsNone(record.exc_info)
        self.assertNotIn(SENTINEL, str(record.__dict__))
        self.assertNotIn(PATH_SENTINEL, str(record.__dict__))

    def test_real_long_inline_input(self) -> None:
        content = SENTINEL + "x" * 600
        for text in (
            json.dumps([{"role": "user", "content": content}]),
            json.dumps({"role": "user", "content": content}),
            "Human: " + content,
        ):
            with self.subTest(format=text[0]):
                messages = self.parser.parse_transcript(text)
                self.assertEqual(len(messages), 1)
                self.assertEqual(messages[0].content, content)
                self.assertEqual(messages[0].role, "human")

    def test_missing_or_invalid_path_probe_parses_inline(self) -> None:
        text = json.dumps([{"role": "user", "content": SENTINEL}])
        invalid_windows_name = OSError(errno.EINVAL, SENTINEL, text)
        invalid_windows_name.winerror = 123
        for error in (OSError(errno.ENAMETOOLONG, SENTINEL, text),
                      FileNotFoundError(errno.ENOENT, SENTINEL, text),
                      NotADirectoryError(errno.ENOTDIR, SENTINEL, text),
                      invalid_windows_name, ValueError(SENTINEL)):
            with self.subTest(error=type(error).__name__, errno=getattr(error, "errno", None)):
                with patch.object(Path, "stat", side_effect=error):
                    messages = self.parser.parse_transcript(text)
                    detailed = self.parser.parse_transcript_outcome(text)
            self.assertEqual([(m.role, m.content) for m in messages], [("human", SENTINEL)])
            self.assertEqual(detailed.messages, messages)
            self.assertTrue(detailed.complete)

    def test_denied_or_broken_stat_does_not_ingest_path(self) -> None:
        for code in (errno.EACCES, errno.EIO, errno.ELOOP, errno.EINVAL):
            with self.subTest(errno=code):
                with patch.object(Path, "stat", side_effect=OSError(code, SENTINEL, PATH_SENTINEL)):
                    self.assert_private_failure(PATH_SENTINEL)
                    detailed = self.parser.parse_transcript_outcome(PATH_SENTINEL)
                self.assertEqual((detailed.status, detailed.messages), ("unreadable", []))
                self.assertEqual(detailed.error_categories, ("source_unreadable",))
                self.assertNotIn(SENTINEL, repr(detailed))
                self.assertNotIn(PATH_SENTINEL, repr(detailed))

    def test_python314_suppressed_error_probes_are_not_used(self) -> None:
        with patch.object(Path, "exists", return_value=False) as exists:
            with patch.object(Path, "is_file", return_value=False) as is_file:
                with patch.object(Path, "stat", side_effect=PermissionError(errno.EACCES, SENTINEL, PATH_SENTINEL)):
                    self.assert_private_failure(PATH_SENTINEL)
                    detailed = self.parser.parse_transcript_outcome(PATH_SENTINEL)
        self.assertEqual((detailed.status, detailed.messages), ("unreadable", []))
        exists.assert_not_called()
        is_file.assert_not_called()

    def test_post_probe_read_failure_never_falls_back(self) -> None:
        for error in (PermissionError(errno.EACCES, SENTINEL, PATH_SENTINEL),
                      OSError(errno.EIO, SENTINEL, PATH_SENTINEL),
                      OSError(errno.ENAMETOOLONG, SENTINEL, PATH_SENTINEL),
                      ValueError(SENTINEL)):
            with self.subTest(error=type(error).__name__):
                with patch.object(Path, "stat", return_value=types.SimpleNamespace(st_mode=stat.S_IFREG)):
                    with patch.object(Path, "read_text", side_effect=error):
                        self.assert_private_failure(PATH_SENTINEL)

    def test_explicit_unreadable_path_never_falls_back(self) -> None:
        for error in (PermissionError(errno.EACCES, SENTINEL, PATH_SENTINEL),
                      OSError(errno.EIO, SENTINEL, PATH_SENTINEL),
                      OSError(errno.ENAMETOOLONG, SENTINEL, PATH_SENTINEL),
                      ValueError(SENTINEL)):
            with self.subTest(error=type(error).__name__):
                with patch.object(Path, "read_text", side_effect=error), patch.object(Path, "stat") as probe:
                    self.assert_private_failure(Path(PATH_SENTINEL))
                probe.assert_not_called()

    def test_explicit_missing_path_preserves_empty_result(self) -> None:
        with patch.object(Path, "stat", return_value=types.SimpleNamespace(st_mode=stat.S_IFREG)):
            with patch.object(Path, "read_text", side_effect=FileNotFoundError(PATH_SENTINEL)):
                with patch.object(self.parser.log, "error") as error:
                    for source in (Path(PATH_SENTINEL), PATH_SENTINEL):
                        self.assertEqual(self.parser.parse_transcript(source), [])
        error.assert_not_called()

    def test_existing_file_is_read_once_after_probe(self) -> None:
        text = json.dumps([{"role": "assistant", "content": SENTINEL}])
        with patch.object(Path, "stat", return_value=types.SimpleNamespace(st_mode=stat.S_IFREG)) as probe:
            with patch.object(Path, "read_text", return_value=text) as read:
                messages = self.parser.parse_transcript(PATH_SENTINEL)
        probe.assert_called_once_with()
        read.assert_called_once_with(encoding="utf-8", errors="replace")
        self.assertEqual([(m.role, m.content) for m in messages], [("assistant", SENTINEL)])

    def test_plain_non_path_and_empty_inputs_preserve_behavior(self) -> None:
        with patch.object(Path, "stat", side_effect=FileNotFoundError(errno.ENOENT, "synthetic missing")):
            self.assertEqual(self.parser.parse_transcript(""), [])
            messages = self.parser.parse_transcript("Human: synthetic ordinary message")
        self.assertEqual([(m.role, m.content) for m in messages], [("human", "synthetic ordinary message")])

    def test_nonregular_string_keeps_inline_compatibility(self) -> None:
        with patch.object(Path, "stat", return_value=types.SimpleNamespace(st_mode=stat.S_IFDIR)):
            messages = self.parser.parse_transcript("Human: synthetic inline text")
        self.assertEqual([(m.role, m.content) for m in messages], [("human", "synthetic inline text")])


class InlineBoundaryFileTests(unittest.TestCase):
    def test_real_file_string_and_explicit_path(self) -> None:
        parser = load_parser()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic-transcript.json"
            path.write_text(json.dumps([{"role": "user", "content": SENTINEL}]), encoding="utf-8")
            for source in (path, str(path)):
                with self.subTest(explicit=isinstance(source, Path)):
                    messages = parser.parse_transcript(source)
                    self.assertEqual([(m.role, m.content) for m in messages], [("human", SENTINEL)])
            self.assertEqual(parser.parse_transcript(Path(folder) / "missing.json"), [])


if __name__ == "__main__":
    unittest.main()
