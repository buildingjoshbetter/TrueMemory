"""Synthetic actual-source controls for guarded source and field-only SEP writes."""
from __future__ import annotations

import ast
import contextlib
import copy
import json
import logging
import sqlite3
import sys
import threading
import types
import unittest
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def functions(relative: str, names: set[str], namespace: dict[str, object]) -> None:
    path = ROOT / relative
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes.extend(node for node in ast.parse(path.read_text(encoding="utf-8")).body
                 if isinstance(node, ast.FunctionDef) and node.name in names)
    if {node.name for node in nodes if isinstance(node, ast.FunctionDef)} != names:
        raise AssertionError("Missing actual source functions")
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)


class SourceUpdateFenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY,content TEXT,sender TEXT,recipient TEXT,timestamp TEXT,category TEXT,modality TEXT,directive INTEGER,metadata TEXT)")
        self.conn.execute("INSERT INTO messages VALUES(1,'synthetic content','synthetic user','synthetic recipient','2020-01-01','general','text',0,'{\"citations\":[\"original\"]}')")
        for table in ("vec_selected", "sep_selected", "sep_other"):
            self.conn.execute(f"CREATE TABLE {table}(rowid INTEGER PRIMARY KEY,embedding BLOB)")
            self.conn.execute(f"INSERT INTO {table} VALUES(1,?)", (b"original",))
        self.conn.commit()
        storage = types.ModuleType("truememory.storage")
        storage.__dict__.update({"json": json, "sqlite3": sqlite3,
            "_OPTIONAL_SELECT_DEFAULTS": {"sender": "''", "recipient": "''", "timestamp": "''",
                "category": "''", "modality": "''", "directive": "0", "metadata": "'{}'"}})
        functions("truememory/storage.py", {"_serialize_metadata", "_deserialize_metadata", "_row_to_dict",
            "_message_columns", "select_message_cols", "get_message", "update_message"}, storage.__dict__)
        writer = types.ModuleType("truememory.tier_switch.writer")
        writer.capture_writer_selection = lambda conn: types.SimpleNamespace(selection=None)

        @contextlib.contextmanager
        def transaction(conn: sqlite3.Connection, selection: object, **kwargs: object) -> Iterator[None]:
            nested = conn.in_transaction
            conn.execute("SAVEPOINT synthetic_update" if nested else "BEGIN IMMEDIATE")
            try:
                yield
                conn.execute("RELEASE SAVEPOINT synthetic_update") if nested else conn.commit()
            except Exception:
                if nested:
                    conn.execute("ROLLBACK TO SAVEPOINT synthetic_update")
                    conn.execute("RELEASE SAVEPOINT synthetic_update")
                else:
                    conn.rollback()
                raise

        writer.writer_transaction = transaction
        vector = types.ModuleType("truememory.vector_search")

        class PublicationChanged(RuntimeError):
            pass

        vector.VectorPublicationChanged = PublicationChanged
        vector._capture_rebuild_model = lambda conn: (object(), (7, "synthetic-model", 3))
        self.encoded: list[str] = []
        self.encoding_error = False
        self.publication_error = False
        self.after_encode = None

        def encode(model: object, texts: list[str]) -> list[bytes]:
            self.encoded.extend(texts)
            if self.encoding_error:
                raise ValueError("synthetic encode failure")
            if self.after_encode is not None:
                self.after_encode()
            return [text.encode() for text in texts]

        vector._encode_with_mps_fallback = encode
        functions("truememory/vector_search.py", {"_build_sep_text"}, vector.__dict__)
        vector.serialize_f32 = lambda value: value
        vector._write_foreground_embedder_metadata_no_commit = lambda *args, **kwargs: None

        @contextlib.contextmanager
        def publication(conn: sqlite3.Connection, identity: object, memory_id: int,
                        *, selection: object, retry_busy: bool = False) -> Iterator[tuple[str, str]]:
            if self.publication_error:
                raise PublicationChanged("synthetic publication failure")
            with transaction(conn, selection):
                yield "vec_selected", "sep_selected"

        vector._foreground_vector_publication = publication
        runtime = types.ModuleType("truememory.tier_switch.runtime")
        runtime.raise_if_serving_rejection = lambda exc: None
        package = types.ModuleType("truememory")
        package.__path__ = []
        self.modules = patch.dict(sys.modules, {"truememory": package, storage.__name__: storage,
            writer.__name__: writer, vector.__name__: vector, runtime.__name__: runtime})
        self.modules.start()
        path = ROOT / "truememory/engine.py"
        original = next(node for node in ast.parse(path.read_text(encoding="utf-8")).body
                        if isinstance(node, ast.ClassDef) and node.name == "TrueMemoryEngine")
        methods = [copy.deepcopy(node) for node in original.body
                   if isinstance(node, ast.FunctionDef) and node.name in {"update", "get"}]
        for method in methods:
            method.decorator_list = []
        selected = ast.ClassDef(name="SelectedEngine", bases=[], keywords=[], body=methods, decorator_list=[])
        namespace = {"sqlite3": sqlite3, "logger": logging.getLogger("source-update-control"),
                     "update_message": storage.update_message, "get_message": storage.get_message}
        nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), selected]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
        self.engine = namespace["SelectedEngine"]()
        self.engine.conn = self.conn
        self.engine._has_vectors = False
        self.engine._ensure_connection = lambda: None
        self.engine._write_lock = threading.RLock()
        self.consolidations: list[bool] = []
        self.engine._maybe_auto_consolidate = lambda: self.consolidations.append(True)
        self.changed_error = PublicationChanged

    def tearDown(self) -> None:
        self.modules.stop()
        self.conn.close()

    def source(self) -> tuple:
        return self.conn.execute("SELECT * FROM messages WHERE id=1").fetchone()

    def vector(self, table: str) -> bytes:
        return self.conn.execute(f"SELECT embedding FROM {table} WHERE rowid=1").fetchone()[0]

    def test_matched_snapshot_and_default_shape(self) -> None:
        self.assertNotIn("_source_snapshot_sha256", self.engine.get(1))
        expected = self.engine.get(1, _source_snapshot=True)
        result = self.engine.update(1, metadata={"citations": ["original", "new"]}, _expected_source=expected)
        self.assertEqual(result["metadata"], {"citations": ["original", "new"]})
        self.assertNotIn("_source_snapshot_sha256", result)
        self.assertEqual(self.encoded, [])
        self.assertFalse(self.conn.in_transaction)

    def test_each_concurrent_relevant_field_rejects_before_mutation(self) -> None:
        for column, value in {"content": "new content", "sender": "other user", "recipient": "other recipient",
                "timestamp": "2025-01-01", "category": "other", "modality": "image", "directive": 1,
                "metadata": '{"citations":["external"]}'}.items():
            with self.subTest(column=column):
                expected = self.engine.get(1, _source_snapshot=True)
                self.conn.execute(f"UPDATE messages SET {column}=? WHERE id=1", (value,))
                self.conn.commit()
                before = self.source()
                with self.assertRaises(sqlite3.OperationalError):
                    self.engine.update(1, metadata={"citations": ["lost"]}, _expected_source=expected)
                self.assertEqual(self.source(), before)
                self.assertEqual(self.vector("sep_selected"), b"original")
                self.assertFalse(self.conn.in_transaction)

    def test_raw_malformed_metadata_cannot_alias_deserialized_empty(self) -> None:
        self.conn.execute("UPDATE messages SET metadata='invalid-first' WHERE id=1")
        self.conn.commit()
        expected = self.engine.get(1, _source_snapshot=True)
        self.conn.execute("UPDATE messages SET metadata='invalid-second' WHERE id=1")
        self.conn.commit()
        self.assertEqual(expected["metadata"], self.engine.get(1)["metadata"])
        with self.assertRaises(sqlite3.OperationalError):
            self.engine.update(1, metadata={}, _expected_source=expected)
        self.assertEqual(self.source()[-1], "invalid-second")

    def test_missing_and_typed_tampered_snapshot_reject(self) -> None:
        expected = self.engine.get(1, _source_snapshot=True)
        for tampered in ({}, {**expected, "directive": 0}, {**expected, "_source_snapshot_sha256": "bad"}):
            with self.subTest(tampered=tampered):
                with self.assertRaises(sqlite3.OperationalError):
                    self.engine.update(1, category="changed", _expected_source=tampered)
                self.assertEqual(self.source()[5], "general")
        self.conn.execute("DELETE FROM messages WHERE id=1")
        self.conn.commit()
        with self.assertRaises(sqlite3.OperationalError):
            self.engine.update(1, category="changed", _expected_source=expected)

    def test_unconditional_update_compatibility(self) -> None:
        self.assertEqual(self.engine.update(1, category="legacy")["category"], "legacy")
        self.assertIsNone(self.engine.update(999, category="missing"))

    def test_unsupported_raw_snapshot_is_controlled_and_preserved(self) -> None:
        self.conn.execute("UPDATE messages SET metadata=? WHERE id=1", (b"synthetic opaque metadata",))
        self.conn.commit()
        before = self.source()
        self.assertEqual(self.engine.get(1)["metadata"], {})
        with self.assertRaises(sqlite3.OperationalError):
            self.engine.get(1, _source_snapshot=True)
        self.assertEqual(self.source(), before)

    def test_timestamp_only_replaces_selected_SEP_and_keeps_primary(self) -> None:
        self.engine._has_vectors = True
        expected = self.engine.get(1, _source_snapshot=True)
        result = self.engine.update(1, timestamp="2025-01-01", _expected_source=expected)
        self.assertEqual(result["timestamp"], "2025-01-01")
        self.assertEqual(len(self.encoded), 1)
        self.assertIn("2025-01-01", self.encoded[0])
        self.assertIn("synthetic content", self.encoded[0])
        self.assertEqual(self.vector("sep_selected"), self.encoded[0].encode())
        self.assertEqual(self.vector("vec_selected"), b"original")
        self.assertEqual(self.vector("sep_other"), b"original")
        self.assertFalse(self.conn.in_transaction)

    def test_sender_recipient_changes_refresh_SEP(self) -> None:
        self.engine._has_vectors = True
        self.engine.update(1, sender="next user", recipient="next recipient")
        self.assertIn("next user", self.encoded[0])
        self.assertIn("next recipient", self.encoded[0])
        self.assertEqual(self.vector("vec_selected"), b"original")

    def test_metadata_or_identical_timestamp_does_not_encode(self) -> None:
        self.engine._has_vectors = True
        self.engine.update(1, metadata={"extra": True})
        self.engine.update(1, timestamp="2020-01-01")
        self.assertEqual(self.encoded, [])
        self.assertEqual(self.vector("sep_selected"), b"original")

    def test_field_only_encode_failure_preserves_source_and_vectors(self) -> None:
        self.engine._has_vectors = self.encoding_error = True
        before = self.source()
        with self.assertRaises(sqlite3.OperationalError):
            self.engine.update(1, timestamp="2025-01-01")
        self.assertEqual(self.source(), before)
        self.assertEqual(self.vector("sep_selected"), b"original")
        self.assertEqual(self.consolidations, [])

    def test_field_only_publication_failure_rolls_back_source(self) -> None:
        self.engine._has_vectors = self.publication_error = True
        before = self.source()
        with self.assertRaises(self.changed_error):
            self.engine.update(1, timestamp="2025-01-01", metadata={"changed": True})
        self.assertEqual(self.source(), before)
        self.assertEqual(self.vector("sep_selected"), b"original")
        self.assertEqual(self.vector("vec_selected"), b"original")
        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(self.consolidations, [])

    def test_source_change_during_field_only_encoding_is_rejected(self) -> None:
        self.engine._has_vectors = True
        def mutate() -> None:
            self.conn.execute("UPDATE messages SET content='external newer content' WHERE id=1")
            self.conn.commit()
        self.after_encode = mutate
        with self.assertRaises(sqlite3.OperationalError):
            self.engine.update(1, timestamp="2025-01-01")
        self.assertEqual(self.source()[1], "external newer content")
        self.assertEqual(self.source()[4], "2020-01-01")
        self.assertEqual(self.vector("sep_selected"), b"original")

    def test_content_publication_failure_keeps_historical_committed_source(self) -> None:
        self.engine._has_vectors = self.publication_error = True
        with self.assertRaisesRegex(self.changed_error, "Source update committed"):
            self.engine.update(1, content="replacement synthetic content")
        self.assertEqual(self.source()[1], "replacement synthetic content")
        self.assertEqual(self.vector("vec_selected"), b"original")
        self.assertEqual(self.consolidations, [True])


if __name__ == "__main__":
    unittest.main()
