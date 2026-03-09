"""Embedded-mode counterpart of graph_stores/.../tests/test_pg_stores_arcadedb.py.

Runs the same PropertyGraphStore test scenarios against an in-process embedded
ArcadeDB database instead of a remote HTTP server.  No Docker or running server
required — the JVM (and optionally the embedded HTTP server) starts inside this
process.

Skipped automatically if ``arcadedb_embedded`` is not installed.
"""

import os
import shutil
import tempfile

import pytest

from llama_index.graph_stores.arcadedb import ArcadeDBPropertyGraphStore, EMBEDDED_AVAILABLE
from llama_index.core.graph_stores.types import EntityNode, ChunkNode, Relation
from llama_index.core.vector_stores.types import VectorStoreQuery

pytestmark = pytest.mark.skipif(
    not EMBEDDED_AVAILABLE,
    reason="arcadedb_embedded is not installed",
)

# ---------------------------------------------------------------------------
# Module-level shared database directory and store (one JVM, one db, one store)
# ---------------------------------------------------------------------------
_tmp_dir: str = ""
_db_path: str = ""
_store: "ArcadeDBPropertyGraphStore | None" = None


def setup_module(_module=None) -> None:
    global _tmp_dir, _db_path, _store
    _tmp_dir = tempfile.mkdtemp(prefix="arcadedb_emb_pg_")
    _db_path = os.path.join(_tmp_dir, "test_pg")
    _store = ArcadeDBPropertyGraphStore(
        mode="embedded",
        db_path=_db_path,
        embedded_server=False,
        include_basic_schema=True,
        embedding_dimension=4,
    )


def teardown_module(_module=None) -> None:
    import gc, time
    gc.collect()
    if os.path.exists(_tmp_dir):
        try:
            shutil.rmtree(_tmp_dir)
        except PermissionError:
            time.sleep(0.5)
            try:
                shutil.rmtree(_tmp_dir)
            except PermissionError:
                pass


@pytest.fixture()
def arcadedb_store() -> ArcadeDBPropertyGraphStore:
    """Shared embedded store — data wiped before each test."""
    for vt in ["Entity", "TextChunk", "PERSON", "ORGANIZATION", "LOCATION",
               "PLACE", "PROJECT", "TECHNOLOGY", "CUSTOM_TYPE"]:
        try:
            _store.structured_query(f"DELETE FROM {vt}")
        except Exception:
            pass
    for et in ["MENTIONS", "WORKS_FOR", "CREATED", "STANDARDIZED_BY",
               "MEMBER_OF", "KNOWS", "LIVES_IN"]:
        try:
            _store.structured_query(f"DELETE FROM {et}")
        except Exception:
            pass
    return _store


# ---------------------------------------------------------------------------
# Tests (mirrors test_pg_stores_arcadedb.py)
# ---------------------------------------------------------------------------

def test_upsert_nodes_and_get(arcadedb_store: ArcadeDBPropertyGraphStore):
    entity = EntityNode(label="PERSON", name="John Newton")
    chunk = ChunkNode(text="John Newton was Chairman and CTO of Alfresco.")
    arcadedb_store.upsert_nodes([entity, chunk])

    retrieved_entities = arcadedb_store.get(ids=[entity.id])
    assert len(retrieved_entities) == 1
    assert isinstance(retrieved_entities[0], EntityNode)
    assert retrieved_entities[0].name == "John Newton"

    retrieved_chunks = arcadedb_store.get(ids=[chunk.id])
    assert len(retrieved_chunks) == 1
    assert isinstance(retrieved_chunks[0], ChunkNode)
    assert "John Newton" in retrieved_chunks[0].text


def test_upsert_relations(arcadedb_store: ArcadeDBPropertyGraphStore):
    person = EntityNode(label="PERSON", name="John Newton")
    org = EntityNode(label="ORGANIZATION", name="Alfresco")
    arcadedb_store.upsert_nodes([person, org])

    relation = Relation(
        label="WORKS_FOR",
        source_id=person.id,
        target_id=org.id,
        properties={"role": "Chairman and CTO"},
    )
    arcadedb_store.upsert_relations([relation])

    triplets = arcadedb_store.get_triplets()
    assert len(triplets) > 0
    found = any(t[0] == "John Newton" and t[1] == "WORKS_FOR" and t[2] == "Alfresco"
                for t in triplets)
    assert found


def test_get_nodes_by_properties(arcadedb_store: ArcadeDBPropertyGraphStore):
    person1 = EntityNode(label="PERSON", name="John Newton")
    person2 = EntityNode(label="PERSON", name="Alice Smith")
    org = EntityNode(label="ORGANIZATION", name="Alfresco")
    arcadedb_store.upsert_nodes([person1, person2, org])

    all_nodes = arcadedb_store.get()
    persons = [n for n in all_nodes if hasattr(n, "label") and n.label == "PERSON"]
    assert len(persons) == 2
    assert "John Newton" in [p.name for p in persons]
    assert "Alice Smith" in [p.name for p in persons]

    orgs = arcadedb_store.get(properties={"name": "Alfresco"})
    assert len(orgs) == 1
    assert orgs[0].name == "Alfresco"


def test_vector_query_with_embeddings(arcadedb_store: ArcadeDBPropertyGraphStore):
    entity1 = EntityNode(label="PERSON", name="John Newton", embedding=[0.1, 0.2, 0.3, 0.4])
    entity2 = EntityNode(label="PERSON", name="Alice Smith", embedding=[0.2, 0.3, 0.4, 0.5])
    entity3 = EntityNode(label="ORGANIZATION", name="Alfresco", embedding=[0.9, 0.8, 0.7, 0.6])
    arcadedb_store.upsert_nodes([entity1, entity2, entity3])

    query = VectorStoreQuery(query_embedding=[0.15, 0.25, 0.35, 0.45], similarity_top_k=2)
    nodes, scores = arcadedb_store.vector_query(query)

    assert len(nodes) <= 2
    assert len(scores) == len(nodes)
    for node in nodes:
        assert isinstance(node, EntityNode)
    for score in scores:
        assert isinstance(score, (int, float))


def test_structured_query(arcadedb_store: ArcadeDBPropertyGraphStore):
    person = EntityNode(label="PERSON", name="John Newton")
    org = EntityNode(label="ORGANIZATION", name="Alfresco")
    arcadedb_store.upsert_nodes([person, org])

    results = arcadedb_store.structured_query("SELECT name FROM PERSON")
    assert len(results) >= 1
    names = [r.get("name") for r in results if "name" in r]
    assert "John Newton" in names


def test_get_schema(arcadedb_store: ArcadeDBPropertyGraphStore):
    schema = arcadedb_store.get_schema()
    assert schema is not None
    if isinstance(schema, dict):
        assert "vertex_types" in schema or "edge_types" in schema
    else:
        assert isinstance(schema, str)
        assert len(schema) > 0


def test_chunk_node_operations(arcadedb_store: ArcadeDBPropertyGraphStore):
    chunk = ChunkNode(
        text="The CMIS specification is backed by multiple organizations.",
        embedding=[0.1, 0.2, 0.3, 0.4],
    )
    arcadedb_store.upsert_nodes([chunk])

    retrieved = arcadedb_store.get(ids=[chunk.id])
    assert len(retrieved) == 1
    assert isinstance(retrieved[0], ChunkNode)
    assert "CMIS specification" in retrieved[0].text

    entity = EntityNode(label="ORGANIZATION", name="Alfresco")
    arcadedb_store.upsert_nodes([entity])
    arcadedb_store.upsert_relations([
        Relation(label="MENTIONS", source_id=chunk.id, target_id=entity.id)
    ])

    triplets = arcadedb_store.get_triplets()
    assert any(t[1] == "MENTIONS" and t[2] == "Alfresco" for t in triplets)


def test_entity_types_and_schema(arcadedb_store: ArcadeDBPropertyGraphStore):
    nodes = [
        EntityNode(label="PERSON", name="John Newton"),
        EntityNode(label="ORGANIZATION", name="Alfresco"),
        EntityNode(label="LOCATION", name="Boston"),
        EntityNode(label="PLACE", name="MIT"),
    ]
    arcadedb_store.upsert_nodes(nodes)

    all_entities = arcadedb_store.get()
    assert len(all_entities) >= 4
    labels = [e.label for e in all_entities if hasattr(e, "label")]
    for expected in ("PERSON", "ORGANIZATION", "LOCATION", "PLACE"):
        assert expected in labels


def test_complex_relationships(arcadedb_store: ArcadeDBPropertyGraphStore):
    john = EntityNode(label="PERSON", name="John Newton")
    alfresco = EntityNode(label="ORGANIZATION", name="Alfresco")
    cmis = EntityNode(label="PROJECT", name="CMIS")
    oasis = EntityNode(label="ORGANIZATION", name="OASIS")
    arcadedb_store.upsert_nodes([john, alfresco, cmis, oasis])

    relations = [
        Relation(label="WORKS_FOR", source_id=john.id, target_id=alfresco.id),
        Relation(label="CREATED", source_id=john.id, target_id=cmis.id),
        Relation(label="STANDARDIZED_BY", source_id=cmis.id, target_id=oasis.id),
        Relation(label="MEMBER_OF", source_id=alfresco.id, target_id=oasis.id),
    ]
    arcadedb_store.upsert_relations(relations)

    triplets = arcadedb_store.get_triplets()
    assert len(triplets) >= 4
    rel_labels = [t[1] for t in triplets]
    for expected in ("WORKS_FOR", "CREATED", "STANDARDIZED_BY", "MEMBER_OF"):
        assert expected in rel_labels


def test_dynamic_schema_creation(arcadedb_store: ArcadeDBPropertyGraphStore):
    project = EntityNode(label="PROJECT", name="CMIS Specification")
    tech = EntityNode(label="TECHNOLOGY", name="Content Management")
    arcadedb_store.upsert_nodes([project, tech])

    retrieved = arcadedb_store.get(properties={"name": "CMIS Specification"})
    assert len(retrieved) == 1
    assert retrieved[0].label == "PROJECT"

    retrieved_tech = arcadedb_store.get(properties={"name": "Content Management"})
    assert len(retrieved_tech) == 1
    assert retrieved_tech[0].label == "TECHNOLOGY"


def test_include_basic_schema_false():
    """Test behavior when include_basic_schema=False (uses a sub-dir of shared tmp)."""
    minimal_store = ArcadeDBPropertyGraphStore(
        mode="embedded",
        db_path=os.path.join(_tmp_dir, "minimal"),
        embedded_server=False,
        include_basic_schema=False,
        embedding_dimension=4,
    )
    entity = EntityNode(label="CUSTOM_TYPE", name="Test Entity")
    minimal_store.upsert_nodes([entity])

    retrieved = minimal_store.get(ids=[entity.id])
    assert len(retrieved) == 1
    assert retrieved[0].name == "Test Entity"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
