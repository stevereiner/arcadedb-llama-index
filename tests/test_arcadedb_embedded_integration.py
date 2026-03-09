"""Integration tests for ArcadeDBPropertyGraphStore in *embedded* mode.

The same test logic from ``test_arcadedb_integration.py`` is re-used via a
shared base class and run under two configurations:

* ``TestEmbeddedDirect``  — ``mode='embedded'``, no HTTP server, direct file
  access via ``DatabaseFactory`` (fastest, no port required).
* ``TestEmbeddedServer``  — ``mode='embedded'``, embedded HTTP server on
  port **2482** so it does not conflict with:
  - 2480 — your local ArcadeDB docker instance
  - 2481 — the integration-test docker container in
            ``test_arcadedb_integration.py``

JPype constraint
----------------
JPype only allows one JVM per process and it cannot be restarted after
``shutdownJVM()`` is called.  When pytest collects and runs both test classes
in the same process we must ensure:

1. The JVM is started *at most once* — handled automatically by arcadedb_embedded
   (it only calls ``startJVM`` if the JVM is not already running).
2. The JVM is never shut down between test classes — we do *not* call
   ``db.close()`` in a way that kills the JVM, and ``server.stop()`` only
   stops the Java server process, not the JVM.
3. Both classes share the same underlying JVM process. Their ``setUpClass``
   each open a *different* database directory, so there is no file-lock
   conflict between them.

When running the *entire* pytest suite (``pytest`` from the project root),
pytest discovers and runs this file *after* ``test_pg_stores_arcadedb.py``
(remote-only, no JVM involved) and *before* ``test_arcadedb_integration.py``
(Docker remote, also no JVM).  The JVM therefore starts cleanly on the first
embedded ``setUpClass`` and stays alive for the second.

Skip policy
-----------
The whole module is skipped at collection time if ``arcadedb_embedded`` is not
installed.  Individual tests in the ``TestEmbeddedServer`` class are also
skipped when the embedded server URL is not available (e.g. port conflict on
CI).
"""

import os
import shutil
import tempfile
import unittest

import pytest

from llama_index.graph_stores.arcadedb import ArcadeDBPropertyGraphStore, EMBEDDED_AVAILABLE
from llama_index.core.graph_stores.types import EntityNode, ChunkNode, Relation
from llama_index.core.vector_stores.types import VectorStoreQuery

pytestmark = pytest.mark.skipif(
    not EMBEDDED_AVAILABLE,
    reason="arcadedb_embedded is not installed — skipping embedded integration tests",
)

# Port for the embedded HTTP server test.  Chosen to avoid collisions with
# the user's docker (2480) and the remote integration-test container (2481).
EMBEDDED_SERVER_PORT = 2482


# ---------------------------------------------------------------------------
# Module-level shared state
# ---------------------------------------------------------------------------
# Both test classes register their tempdir here so teardown_module() can
# remove them even if individual tearDownClass calls are skipped.
_tmp_dirs: list[str] = []


def teardown_module(_module=None) -> None:
    """Remove all temp directories created by the embedded test classes.

    Uses gc.collect() + retry to handle Windows file locks held by the JVM.
    """
    import gc
    import time

    gc.collect()
    for d in _tmp_dirs:
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
            except PermissionError:
                time.sleep(0.5)
                try:
                    shutil.rmtree(d)
                except PermissionError:
                    pass  # OS will clean up temp dir eventually
    _tmp_dirs.clear()


# ---------------------------------------------------------------------------
# Shared test logic
# ---------------------------------------------------------------------------

class _EmbeddedGraphStoreTests:
    """Mixin with all test methods.  Concrete subclasses provide ``pg_store``."""

    pg_store: ArcadeDBPropertyGraphStore  # set by setUpClass

    # --- per-test cleanup ---------------------------------------------------

    def setUp(self) -> None:  # noqa: N802
        """Wipe all data before each test for isolation."""
        vertex_types = [
            "Entity", "TextChunk",
            "PERSON", "ORGANIZATION", "LOCATION", "PLACE", "PROJECT", "TECHNOLOGY",
        ]
        edge_types = ["MENTIONS", "WORKS_FOR", "CREATED", "STANDARDIZED_BY", "MEMBER_OF"]
        for vt in vertex_types:
            try:
                self.pg_store.structured_query(f"DELETE FROM {vt}")
            except Exception:
                pass
        for et in edge_types:
            try:
                self.pg_store.structured_query(f"DELETE FROM {et}")
            except Exception:
                pass

    # --- tests (mirrors test_arcadedb_integration.py) -----------------------

    def test_upsert_nodes_and_get(self) -> None:
        entity = EntityNode(label="PERSON", name="John Newton")
        chunk = ChunkNode(text="John Newton was Chairman and CTO of Alfresco.")
        self.pg_store.upsert_nodes([entity, chunk])

        retrieved_entities = self.pg_store.get(ids=[entity.id])
        self.assertEqual(len(retrieved_entities), 1)
        self.assertIsInstance(retrieved_entities[0], EntityNode)
        self.assertEqual(retrieved_entities[0].name, "John Newton")

        retrieved_chunks = self.pg_store.get(ids=[chunk.id])
        self.assertEqual(len(retrieved_chunks), 1)
        self.assertIsInstance(retrieved_chunks[0], ChunkNode)
        self.assertIn("John Newton", retrieved_chunks[0].text)

    def test_upsert_relations(self) -> None:
        person = EntityNode(label="PERSON", name="John Newton")
        org = EntityNode(label="ORGANIZATION", name="Alfresco")
        self.pg_store.upsert_nodes([person, org])

        relation = Relation(
            label="WORKS_FOR",
            source_id=person.id,
            target_id=org.id,
            properties={"role": "Chairman and CTO"},
        )
        self.pg_store.upsert_relations([relation])

        triplets = self.pg_store.get_triplets()
        self.assertGreater(len(triplets), 0)
        found = any(
            t[0] == "John Newton" and t[1] == "WORKS_FOR" and t[2] == "Alfresco"
            for t in triplets
        )
        self.assertTrue(found)

    def test_get_nodes_by_properties(self) -> None:
        person1 = EntityNode(label="PERSON", name="John Newton")
        person2 = EntityNode(label="PERSON", name="Alice Smith")
        org = EntityNode(label="ORGANIZATION", name="Alfresco")
        self.pg_store.upsert_nodes([person1, person2, org])

        all_nodes = self.pg_store.get()
        persons = [n for n in all_nodes if hasattr(n, "label") and n.label == "PERSON"]
        self.assertEqual(len(persons), 2)
        self.assertIn("John Newton", [p.name for p in persons])
        self.assertIn("Alice Smith", [p.name for p in persons])

        orgs = self.pg_store.get(properties={"name": "Alfresco"})
        self.assertEqual(len(orgs), 1)
        self.assertEqual(orgs[0].name, "Alfresco")

    def test_vector_query_with_embeddings(self) -> None:
        entity1 = EntityNode(label="PERSON", name="John Newton", embedding=[0.1, 0.2, 0.3, 0.4])
        entity2 = EntityNode(label="PERSON", name="Alice Smith", embedding=[0.2, 0.3, 0.4, 0.5])
        entity3 = EntityNode(label="ORGANIZATION", name="Alfresco", embedding=[0.9, 0.8, 0.7, 0.6])
        self.pg_store.upsert_nodes([entity1, entity2, entity3])

        query = VectorStoreQuery(query_embedding=[0.15, 0.25, 0.35, 0.45], similarity_top_k=2)
        nodes, scores = self.pg_store.vector_query(query)

        self.assertLessEqual(len(nodes), 2)
        self.assertEqual(len(nodes), len(scores))

        from llama_index.core.graph_stores.types import LabelledNode
        for node in nodes:
            self.assertIsInstance(node, LabelledNode)

        node_names = [n.name for n in nodes if hasattr(n, "name")]
        self.assertTrue(
            any(name in node_names for name in ["John Newton", "Alice Smith"]),
            f"Expected a PERSON node in top-2 results, got: {node_names}",
        )

    def test_structured_query(self) -> None:
        person = EntityNode(label="PERSON", name="John Newton")
        org = EntityNode(label="ORGANIZATION", name="Alfresco")
        self.pg_store.upsert_nodes([person, org])

        results = self.pg_store.structured_query("SELECT name FROM PERSON")
        self.assertGreaterEqual(len(results), 1)
        names = [r.get("name") for r in results if "name" in r]
        self.assertIn("John Newton", names)

    def test_get_schema(self) -> None:
        schema = self.pg_store.get_schema()
        self.assertIsNotNone(schema)
        if isinstance(schema, dict):
            self.assertTrue("vertex_types" in schema or "edge_types" in schema)
        else:
            self.assertIsInstance(schema, str)
            self.assertGreater(len(schema), 0)

    def test_chunk_node_operations(self) -> None:
        chunk = ChunkNode(
            text="The CMIS specification is backed by multiple organizations.",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        self.pg_store.upsert_nodes([chunk])

        retrieved = self.pg_store.get(ids=[chunk.id])
        self.assertEqual(len(retrieved), 1)
        self.assertIsInstance(retrieved[0], ChunkNode)
        self.assertIn("CMIS specification", retrieved[0].text)

        entity = EntityNode(label="ORGANIZATION", name="Alfresco")
        self.pg_store.upsert_nodes([entity])
        self.pg_store.upsert_relations([
            Relation(label="MENTIONS", source_id=chunk.id, target_id=entity.id)
        ])

        triplets = self.pg_store.get_triplets()
        self.assertTrue(
            any(t[1] == "MENTIONS" and t[2] == "Alfresco" for t in triplets)
        )

    def test_entity_types_and_schema(self) -> None:
        nodes = [
            EntityNode(label="PERSON", name="John Newton"),
            EntityNode(label="ORGANIZATION", name="Alfresco"),
            EntityNode(label="LOCATION", name="Boston"),
            EntityNode(label="PLACE", name="MIT"),
        ]
        self.pg_store.upsert_nodes(nodes)

        all_entities = self.pg_store.get()
        self.assertGreaterEqual(len(all_entities), 4)

        labels = [e.label for e in all_entities if hasattr(e, "label")]
        for expected in ("PERSON", "ORGANIZATION", "LOCATION", "PLACE"):
            self.assertIn(expected, labels)

    def test_complex_relationships(self) -> None:
        john = EntityNode(label="PERSON", name="John Newton")
        alfresco = EntityNode(label="ORGANIZATION", name="Alfresco")
        cmis = EntityNode(label="PROJECT", name="CMIS")
        oasis = EntityNode(label="ORGANIZATION", name="OASIS")
        self.pg_store.upsert_nodes([john, alfresco, cmis, oasis])

        relations = [
            Relation(label="WORKS_FOR", source_id=john.id, target_id=alfresco.id),
            Relation(label="CREATED", source_id=john.id, target_id=cmis.id),
            Relation(label="STANDARDIZED_BY", source_id=cmis.id, target_id=oasis.id),
            Relation(label="MEMBER_OF", source_id=alfresco.id, target_id=oasis.id),
        ]
        self.pg_store.upsert_relations(relations)

        triplets = self.pg_store.get_triplets()
        self.assertGreaterEqual(len(triplets), 4)
        rel_labels = [t[1] for t in triplets]
        for expected in ("WORKS_FOR", "CREATED", "STANDARDIZED_BY", "MEMBER_OF"):
            self.assertIn(expected, rel_labels)

    def test_dynamic_schema_creation(self) -> None:
        project = EntityNode(label="PROJECT", name="CMIS Specification")
        tech = EntityNode(label="TECHNOLOGY", name="Content Management")
        self.pg_store.upsert_nodes([project, tech])

        retrieved = self.pg_store.get(properties={"name": "CMIS Specification"})
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0].label, "PROJECT")

        retrieved_tech = self.pg_store.get(properties={"name": "Content Management"})
        self.assertEqual(len(retrieved_tech), 1)
        self.assertEqual(retrieved_tech[0].label, "TECHNOLOGY")

    def test_delete_by_entity_names(self) -> None:
        person = EntityNode(label="PERSON", name="DeleteMe Person")
        org = EntityNode(label="ORGANIZATION", name="DeleteMe Org")
        self.pg_store.upsert_nodes([person, org])

        self.assertEqual(len(self.pg_store.get(properties={"name": "DeleteMe Person"})), 1)
        self.assertEqual(len(self.pg_store.get(properties={"name": "DeleteMe Org"})), 1)

        self.pg_store.delete(entity_names=["DeleteMe Person", "DeleteMe Org"])

        self.assertEqual(len(self.pg_store.get(properties={"name": "DeleteMe Person"})), 0)
        self.assertEqual(len(self.pg_store.get(properties={"name": "DeleteMe Org"})), 0)

    def test_delete_by_ids(self) -> None:
        person = EntityNode(label="PERSON", name="DeleteById Person")
        chunk = ChunkNode(text="DeleteById chunk text")
        self.pg_store.upsert_nodes([person, chunk])

        self.assertEqual(len(self.pg_store.get(ids=[person.id])), 1)
        self.assertEqual(len(self.pg_store.get(ids=[chunk.id])), 1)

        self.pg_store.delete(ids=[person.id, chunk.id])

        self.assertEqual(len(self.pg_store.get(ids=[person.id])), 0)
        self.assertEqual(len(self.pg_store.get(ids=[chunk.id])), 0)

    def test_delete_by_properties_ref_doc_id(self) -> None:
        doc_id = "test-doc-abc123"
        chunk1 = ChunkNode(text="Chunk one", properties={"ref_doc_id": doc_id})
        chunk2 = ChunkNode(text="Chunk two", properties={"ref_doc_id": doc_id})
        other = ChunkNode(text="Unrelated chunk", properties={"ref_doc_id": "other-doc-xyz"})
        self.pg_store.upsert_nodes([chunk1, chunk2, other])

        stored = self.pg_store.structured_query(
            f"SELECT id, ref_doc_id FROM TextChunk WHERE ref_doc_id = '{doc_id}'"
        )
        self.assertEqual(len(stored), 2)

        self.pg_store.delete(properties={"ref_doc_id": doc_id})

        remaining = self.pg_store.structured_query(
            f"SELECT id FROM TextChunk WHERE ref_doc_id = '{doc_id}'"
        )
        self.assertEqual(len(remaining), 0)
        self.assertEqual(len(self.pg_store.get(ids=[other.id])), 1)

    def test_delete_by_properties_across_vertex_types(self) -> None:
        tag = "delete_prop_test"
        person = EntityNode(label="PERSON", name="PropDelete Person", properties={"test_tag": tag})
        org = EntityNode(label="ORGANIZATION", name="PropDelete Org", properties={"test_tag": tag})
        self.pg_store.upsert_nodes([person, org])

        self.assertEqual(len(self.pg_store.get(properties={"name": "PropDelete Person"})), 1)
        self.assertEqual(len(self.pg_store.get(properties={"name": "PropDelete Org"})), 1)

        self.pg_store.delete(properties={"test_tag": tag})

        self.assertEqual(len(self.pg_store.get(properties={"name": "PropDelete Person"})), 0)
        self.assertEqual(len(self.pg_store.get(properties={"name": "PropDelete Org"})), 0)

    def test_delete_relation_names(self) -> None:
        person = EntityNode(label="PERSON", name="RelDelete Person")
        org = EntityNode(label="ORGANIZATION", name="RelDelete Org")
        self.pg_store.upsert_nodes([person, org])
        self.pg_store.upsert_relations([
            Relation(label="WORKS_FOR", source_id=person.id, target_id=org.id)
        ])

        triplets = self.pg_store.get_triplets()
        self.assertTrue(any(t[1] == "WORKS_FOR" for t in triplets))

        self.pg_store.delete(relation_names=["WORKS_FOR"])

        triplets_after = self.pg_store.get_triplets()
        self.assertFalse(any(t[1] == "WORKS_FOR" for t in triplets_after))


# ---------------------------------------------------------------------------
# Concrete test classes
# ---------------------------------------------------------------------------

class TestEmbeddedDirect(_EmbeddedGraphStoreTests, unittest.TestCase):
    """Run the full suite against an in-process embedded database (no HTTP server).

    The JVM is started here (on first use) and remains alive for the rest of
    the pytest process — including TestEmbeddedServer below.  We intentionally
    do NOT call db.close() or shutdownJVM() in tearDownClass because:
    - db.close() alone is fine, but keeping the database open avoids any
      file-lock edge-cases between teardown and the next class's setUpClass.
    - shutdownJVM() would make it impossible for TestEmbeddedServer to start
      the JVM again (JPype limitation).
    Temp directory cleanup is deferred to teardown_module().
    """

    _db_dir: str

    @classmethod
    def setUpClass(cls) -> None:  # noqa: N802
        cls._db_dir = tempfile.mkdtemp(prefix="arcadedb_embedded_direct_")
        _tmp_dirs.append(cls._db_dir)
        db_path = os.path.join(cls._db_dir, "test_graph")
        cls.pg_store = ArcadeDBPropertyGraphStore(
            mode="embedded",
            db_path=db_path,
            embedded_server=False,
            include_basic_schema=True,
            embedding_dimension=4,
        )

    @classmethod
    def tearDownClass(cls) -> None:  # noqa: N802
        # Do NOT close the db or shut down the JVM — TestEmbeddedServer needs
        # the JVM alive.  Temp dir is removed by teardown_module().
        pass


class TestEmbeddedServer(_EmbeddedGraphStoreTests, unittest.TestCase):
    """Run the full suite with the embedded HTTP server active on port 2482.

    Also exercises ``embedded_server_url`` so you can verify the Studio UI is
    reachable at that address during a manual run.

    The JVM is already running (started by TestEmbeddedDirect), so this class
    only needs to start an ArcadeDBServer and open a fresh database under it.
    """

    _db_dir: str

    @classmethod
    def setUpClass(cls) -> None:  # noqa: N802
        cls._db_dir = tempfile.mkdtemp(prefix="arcadedb_embedded_server_")
        _tmp_dirs.append(cls._db_dir)
        db_path = os.path.join(cls._db_dir, "test_graph")
        cls.pg_store = ArcadeDBPropertyGraphStore(
            mode="embedded",
            db_path=db_path,
            embedded_server=True,
            embedded_server_port=EMBEDDED_SERVER_PORT,
            embedded_server_password="playwithdata",
            include_basic_schema=True,
            embedding_dimension=4,
        )

    @classmethod
    def tearDownClass(cls) -> None:  # noqa: N802
        # Stop the HTTP server (releases the port and Java server thread) but
        # do NOT shut down the JVM.  Temp dir is removed by teardown_module().
        try:
            cls.pg_store._db.stop_embedded_server()
        except Exception:
            pass

    def test_embedded_server_url_is_set(self) -> None:
        """Verify that the embedded server URL is populated and uses the configured port."""
        url = self.pg_store.embedded_server_url
        self.assertIsNotNone(url, "embedded_server_url should be set when embedded_server=True")
        self.assertIn(str(EMBEDDED_SERVER_PORT), url,
                      f"URL should contain port {EMBEDDED_SERVER_PORT}, got: {url}")


if __name__ == "__main__":
    unittest.main()
