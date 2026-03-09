"""Embedded-mode counterpart of graph_stores/.../tests/test_graph_stores_arcadedb.py.

Tests the legacy ArcadeDBGraphStore (triplet store) against an in-process
embedded ArcadeDB database.  No Docker or running server required.

Skipped automatically if ``arcadedb_embedded`` is not installed.
"""

import os
import shutil
import tempfile
import unittest

import pytest

from llama_index.graph_stores.arcadedb import ArcadeDBGraphStore, EMBEDDED_AVAILABLE
from llama_index.core.graph_stores.types import GraphStore

pytestmark = pytest.mark.skipif(
    not EMBEDDED_AVAILABLE,
    reason="arcadedb_embedded is not installed",
)


class TestEmbeddedArcadeDBGraphStore(unittest.TestCase):
    """Embedded counterpart of TestArcadeDBGraphStore."""

    _tmp_dir: str
    _db_path: str

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp_dir = tempfile.mkdtemp(prefix="arcadedb_emb_graph_")
        cls._db_path = os.path.join(cls._tmp_dir, "test_graph")
        cls.graph_store = ArcadeDBGraphStore(
            mode="embedded",
            db_path=cls._db_path,
            embedded_server=False,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        import gc, time
        gc.collect()
        if os.path.exists(cls._tmp_dir):
            try:
                shutil.rmtree(cls._tmp_dir)
            except PermissionError:
                time.sleep(0.5)
                shutil.rmtree(cls._tmp_dir, ignore_errors=True)

    def setUp(self) -> None:
        try:
            self.graph_store._db.query("sql", "DELETE FROM Entity", is_command=True)
        except Exception:
            pass
        # Delete all edge types created by tests
        for et in ["KNOWS", "WORKS_FOR", "CREATED", "LIVES_IN"]:
            try:
                self.graph_store._db.query("sql", f"DELETE FROM {et}", is_command=True)
            except Exception:
                pass

    def test_graph_store_interface(self) -> None:
        self.assertIsInstance(self.graph_store, GraphStore)

    def test_upsert_triplet(self) -> None:
        self.graph_store.upsert_triplet("Alice", "KNOWS", "Bob")
        alice_triplets = self.graph_store.get("Alice")
        self.assertGreaterEqual(len(alice_triplets), 1)
        found = any(
            t[0] == "Alice" and t[1] == "KNOWS" and t[2] == "Bob"
            for t in alice_triplets
        )
        self.assertTrue(found, f"Expected triplet not found in: {alice_triplets}")

    def test_get_empty_subject(self) -> None:
        triplets = self.graph_store.get("NonExistentSubject")
        self.assertIsInstance(triplets, list)
        self.assertEqual(len(triplets), 0)

    def test_multiple_triplets_same_subject(self) -> None:
        self.graph_store.upsert_triplet("John", "WORKS_FOR", "Alfresco")
        self.graph_store.upsert_triplet("John", "CREATED", "CMIS")
        self.graph_store.upsert_triplet("John", "LIVES_IN", "Boston")

        john_triplets = self.graph_store.get("John")
        self.assertGreaterEqual(len(john_triplets), 3)
        for triplet in john_triplets:
            self.assertEqual(triplet[0], "John")

        relations = [t[1] for t in john_triplets]
        self.assertIn("WORKS_FOR", relations)
        self.assertIn("CREATED", relations)
        self.assertIn("LIVES_IN", relations)


if __name__ == "__main__":
    unittest.main(verbosity=2)
