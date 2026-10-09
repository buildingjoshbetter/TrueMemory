"""Foreground metadata publication under actual serving and writer guards."""
from __future__ import annotations

import datetime
from pathlib import Path
import runpy
import sqlite3
import unittest
from unittest.mock import Mock

BOUNDARIES = runpy.run_path(str(Path(__file__).with_name("test-tier-public-boundaries-795.py")))


class TestForegroundMetadata(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = BOUNDARIES["TestPublicBoundaries"]()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.conn, self.runtime = self.fixture.conn, self.fixture.api
        self.writer = self.fixture.load_application("tier_switch.writer")
        self.vector = self.fixture.fixture.vector
        self.source = BOUNDARIES["definitions"](
            "vector_search.py",
            {"VectorPublicationChanged", "_write_foreground_embedder_metadata_no_commit",
             "_write_embedder_metadata_no_commit", "_write_embedder_metadata"},
            self.fixture.namespace(datetime=datetime, _model_generation=7,
                                   EMBEDDING_MODEL=self.vector.EMBEDDING_MODEL,
                                   _embedding_dim=self.vector._embedding_dim),
        )
        self.identity = (7, self.source.EMBEDDING_MODEL, self.source._embedding_dim)
        self.conn.execute("DELETE FROM main.metadata WHERE key IN ('embed_model','embed_dim')")
        self.conn.commit()
        self.owner = Mock(side_effect=RuntimeError("Synthetic maintenance owner busy"))
        self.fixture.modules["maintenance"].maintenance_owner = self.owner

    def publish(self, selection: object) -> None:
        self.source._write_foreground_embedder_metadata_no_commit(
            self.conn, self.identity, selection=selection)

    def rows(self) -> list:
        return self.conn.execute(
            "SELECT key,value FROM main.metadata WHERE key IN ('embed_model','embed_dim') ORDER BY key"
        ).fetchall()

    def test_initial_and_matching_publication_do_not_acquire_maintenance(self) -> None:
        with self.runtime.serving_operation(self.conn):
            selection = self.writer.capture_writer_selection(self.conn)
            for _ in range(2):
                with self.writer.writer_transaction(self.conn, selection):
                    self.publish(selection)
                    self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.rows(), [("embed_dim", "256"), ("embed_model", "model2vec")])
        self.owner.assert_not_called()

    def test_unbound_and_clean_calls_refuse_without_writes(self) -> None:
        selection = self.writer.capture_writer_selection(self.conn)
        with self.assertRaises(self.source.VectorPublicationChanged):
            self.publish(selection)
        with self.runtime.serving_operation(self.conn):
            with self.assertRaises(self.writer.WriterSelectionChanged):
                self.publish(selection)
        self.assertEqual(self.rows(), [])
        self.assertFalse(self.conn.in_transaction)

    def test_changed_model_generation_or_identity_refuses_without_writes(self) -> None:
        with self.runtime.serving_operation(self.conn):
            selection = self.writer.capture_writer_selection(self.conn)
            for name, value in (("_model_generation", 8), ("EMBEDDING_MODEL", "synthetic/changed"),
                                ("_embedding_dim", 512)):
                with self.subTest(field=name):
                    old = getattr(self.source, name)
                    setattr(self.source, name, value)
                    with self.writer.writer_transaction(self.conn, selection):
                        with self.assertRaises(self.source.VectorPublicationChanged):
                            self.publish(selection)
                        self.assertEqual(self.rows(), [])
                    setattr(self.source, name, old)

    def test_partial_mismatched_and_nontext_metadata_refuse_atomically(self) -> None:
        for rows in (
            [("embed_model", "model2vec")], [("embed_dim", "256")],
            [("embed_model", "synthetic/wrong"), ("embed_dim", "256")],
            [("embed_model", "model2vec"), ("embed_dim", "512")],
            [("embed_model", b"model2vec"), ("embed_dim", "256")],
        ):
            with self.subTest(rows=rows):
                self.conn.execute("DELETE FROM main.metadata WHERE key IN ('embed_model','embed_dim')")
                self.conn.executemany("INSERT INTO main.metadata(key,value) VALUES(?,?)", rows)
                self.conn.commit()
                before = self.rows()
                with self.runtime.serving_operation(self.conn):
                    selection = self.writer.capture_writer_selection(self.conn)
                    with self.writer.writer_transaction(self.conn, selection):
                        changes = self.conn.total_changes
                        with self.assertRaises(self.source.VectorPublicationChanged):
                            self.publish(selection)
                        self.assertEqual(self.conn.total_changes, changes)
                self.assertEqual(self.rows(), before)

    def test_temp_metadata_neither_satisfies_nor_receives_main_publication(self) -> None:
        self.conn.execute("CREATE TEMP TABLE metadata(key TEXT PRIMARY KEY,value TEXT,updated_at TEXT)")
        self.conn.execute("INSERT INTO temp.metadata VALUES('embed_model','synthetic/temp',NULL)")
        self.conn.commit()
        with self.runtime.serving_operation(self.conn):
            selection = self.writer.capture_writer_selection(self.conn)
            with self.writer.writer_transaction(self.conn, selection):
                self.publish(selection)
        self.assertEqual(self.rows(), [("embed_dim", "256"), ("embed_model", "model2vec")])
        self.assertEqual(self.conn.execute("SELECT value FROM temp.metadata").fetchall(), [("synthetic/temp",)])

    def test_borrowed_failure_rolls_back_our_metadata_and_retains_outer_work(self) -> None:
        self.conn.execute("INSERT INTO messages(id,content) VALUES(701,'synthetic outer work')")
        with self.runtime.serving_operation(self.conn):
            selection = self.writer.capture_writer_selection(self.conn)
            before = self.conn.commits, self.conn.rollbacks
            with self.assertRaisesRegex(RuntimeError, "synthetic downstream"):
                with self.writer.writer_transaction(self.conn, selection):
                    self.publish(selection)
                    raise RuntimeError("synthetic downstream failure")
            self.assertTrue(self.conn.in_transaction)
            self.assertEqual((self.conn.commits, self.conn.rollbacks), before)
            self.assertEqual(self.rows(), [])
            self.assertEqual(self.conn.execute("SELECT content FROM messages WHERE id=701").fetchone(),
                             ("synthetic outer work",))

    def test_owned_failure_rolls_back_metadata(self) -> None:
        with self.runtime.serving_operation(self.conn):
            selection = self.writer.capture_writer_selection(self.conn)
            with self.assertRaisesRegex(RuntimeError, "synthetic downstream"):
                with self.writer.writer_transaction(self.conn, selection):
                    self.publish(selection)
                    raise RuntimeError("synthetic downstream failure")
        self.assertEqual(self.rows(), [])
        self.assertFalse(self.conn.in_transaction)

    def test_selected_publication_validates_but_does_not_rewrite_metadata(self) -> None:
        selected = self.fixture.fixture.select()
        self.source.EMBEDDING_MODEL = selected.target.model_id
        self.source._embedding_dim = selected.target.dimension
        self.identity = (7, self.source.EMBEDDING_MODEL, self.source._embedding_dim)
        with self.runtime.serving_operation(self.conn):
            selection = self.writer.capture_writer_selection(self.conn)
            with self.writer.writer_transaction(self.conn, selection):
                changes = self.conn.total_changes
                self.publish(selection)
                self.assertEqual(self.conn.total_changes, changes)
        self.owner.assert_not_called()

    def test_general_metadata_mutators_still_require_maintenance_ownership(self) -> None:
        for name in ("_write_embedder_metadata_no_commit", "_write_embedder_metadata"):
            with self.subTest(name=name), self.runtime.serving_operation(self.conn):
                selection = self.writer.capture_writer_selection(self.conn)
                with self.writer.writer_transaction(self.conn, selection):
                    with self.assertRaisesRegex(RuntimeError, "Synthetic maintenance owner busy"):
                        getattr(self.source, name)(self.conn)
        self.assertEqual(self.owner.call_count, 2)
        self.assertEqual(self.rows(), [])

    def test_matching_metadata_supports_sqlite_row_factory(self) -> None:
        self.conn.execute("INSERT INTO main.metadata(key,value) VALUES('embed_model','model2vec')")
        self.conn.execute("INSERT INTO main.metadata(key,value) VALUES('embed_dim','256')")
        self.conn.commit()
        self.conn.row_factory = sqlite3.Row
        try:
            with self.runtime.serving_operation(self.conn):
                selection = self.writer.capture_writer_selection(self.conn)
                with self.writer.writer_transaction(self.conn, selection):
                    self.publish(selection)
            self.assertEqual([tuple(row) for row in self.rows()],
                             [("embed_dim", "256"), ("embed_model", "model2vec")])
        finally:
            self.conn.row_factory = None


if __name__ == "__main__":
    unittest.main()
