import os
import pytest
import docker
import time

from llama_index.graph_stores.arcadedb import ArcadeDBPropertyGraphStore
from llama_index.core.graph_stores.types import EntityNode, ChunkNode, Relation
from llama_index.core.vector_stores.types import VectorStoreQuery

arcadedb_host = os.environ.get("ARCADEDB_HOST", "localhost")
arcadedb_port = int(os.environ.get("ARCADEDB_PORT", "2480"))
arcadedb_username = os.environ.get("ARCADEDB_USERNAME", "root")
arcadedb_password = os.environ.get("ARCADEDB_PASSWORD", "playwithdata")
arcadedb_database = os.environ.get("ARCADEDB_DATABASE", "graph")

# Test against existing ArcadeDB instance (Docker management disabled for now)
def setup_module():
    """Verify ArcadeDB instance is available."""
    # Try different credential combinations
    credentials_to_try = [
        ("root", ""),  # No password
        ("root", "arcadedb"),  # Default password
        ("root", "playwithdata"),  # Our custom password
    ]
    
    for username, password in credentials_to_try:
        try:
            from arcadedb_python import SyncClient
            test_client = SyncClient(host="localhost", port=2480, username=username, password=password)
            print(f"✅ Connected to ArcadeDB with credentials: {username}/{password or '(no password)'}")
            # Update global variables for tests
            global arcadedb_username, arcadedb_password
            arcadedb_username = username
            arcadedb_password = password
            return
        except Exception as e:
            print(f"❌ Failed with {username}/{password or '(no password)'}: {e}")
            continue
    
    raise Exception("❌ Could not connect to ArcadeDB with any credential combination")

def teardown_module():
    """No cleanup needed for existing instance."""
    pass


@pytest.fixture()
def arcadedb_store() -> ArcadeDBPropertyGraphStore:
    """Provides a fresh ArcadeDBPropertyGraphStore for each test."""
    arcadedb_store = ArcadeDBPropertyGraphStore(
        host=arcadedb_host,
        port=arcadedb_port,
        username=arcadedb_username,
        password=arcadedb_password,
        database=arcadedb_database,
        include_basic_schema=True,
        embedding_dimension=1536
    )
    
    # Clear the database before each test
    try:
        arcadedb_store.structured_query("DELETE FROM V")
        arcadedb_store.structured_query("DELETE FROM E")
    except Exception:
        pass  # Ignore errors if tables don't exist
    
    return arcadedb_store


def test_upsert_nodes_and_get(arcadedb_store: ArcadeDBPropertyGraphStore):
    """Test inserting entity and chunk nodes, then retrieving them."""
    entity = EntityNode(label="PERSON", name="John Newton")
    chunk = ChunkNode(text="John Newton was Chairman and CTO of Alfresco.")
    arcadedb_store.upsert_nodes([entity, chunk])

    # Get by ID
    retrieved_entities = arcadedb_store.get(ids=[entity.id])
    assert len(retrieved_entities) == 1
    assert isinstance(retrieved_entities[0], EntityNode)
    assert retrieved_entities[0].name == "John Newton"

    retrieved_chunks = arcadedb_store.get(ids=[chunk.id])
    assert len(retrieved_chunks) == 1
    assert isinstance(retrieved_chunks[0], ChunkNode)
    assert "John Newton" in retrieved_chunks[0].text


def test_upsert_relations(arcadedb_store: ArcadeDBPropertyGraphStore):
    """Test inserting relations between nodes."""
    # Create entities
    person = EntityNode(label="PERSON", name="John Newton")
    org = EntityNode(label="ORGANIZATION", name="Alfresco")
    arcadedb_store.upsert_nodes([person, org])

    # Create relation
    relation = Relation(
        label="WORKS_FOR",
        source_id=person.id,
        target_id=org.id,
        properties={"role": "Chairman and CTO"}
    )
    arcadedb_store.upsert_relations([relation])

    # Verify relation exists
    triplets = arcadedb_store.get_triplets()
    assert len(triplets) > 0
    
    # Find our relation
    found_relation = False
    for triplet in triplets:
        if (triplet[0] == "John Newton" and 
            triplet[1] == "WORKS_FOR" and 
            triplet[2] == "Alfresco"):
            found_relation = True
            break
    assert found_relation


def test_get_nodes_by_properties(arcadedb_store: ArcadeDBPropertyGraphStore):
    """Test retrieving nodes by their properties."""
    # Create entities with different labels
    person1 = EntityNode(label="PERSON", name="John Newton")
    person2 = EntityNode(label="PERSON", name="Alice Smith")
    org = EntityNode(label="ORGANIZATION", name="Alfresco")
    arcadedb_store.upsert_nodes([person1, person2, org])

    # Get all PERSON entities
    persons = arcadedb_store.get(properties={"label": "PERSON"})
    assert len(persons) == 2
    person_names = [p.name for p in persons]
    assert "John Newton" in person_names
    assert "Alice Smith" in person_names

    # Get specific organization
    orgs = arcadedb_store.get(properties={"name": "Alfresco"})
    assert len(orgs) == 1
    assert orgs[0].name == "Alfresco"


def test_vector_query_with_embeddings(arcadedb_store: ArcadeDBPropertyGraphStore):
    """Test vector similarity search with embeddings."""
    # Create entities with embeddings
    entity1 = EntityNode(
        label="PERSON", 
        name="John Newton",
        embedding=[0.1, 0.2, 0.3, 0.4]
    )
    entity2 = EntityNode(
        label="PERSON", 
        name="Alice Smith",
        embedding=[0.2, 0.3, 0.4, 0.5]
    )
    entity3 = EntityNode(
        label="ORGANIZATION", 
        name="Alfresco",
        embedding=[0.9, 0.8, 0.7, 0.6]
    )
    arcadedb_store.upsert_nodes([entity1, entity2, entity3])

    # Perform vector query
    query_embedding = [0.15, 0.25, 0.35, 0.45]  # Close to entity1 and entity2
    query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=2)
    
    results = arcadedb_store.vector_query(query)
    assert len(results) <= 2  # Should return top 2 similar entities
    
    # Results should be EntityNodes with similarity scores
    for result in results:
        assert hasattr(result, 'node')
        assert hasattr(result, 'score')
        assert isinstance(result.node, EntityNode)


def test_structured_query(arcadedb_store: ArcadeDBPropertyGraphStore):
    """Test executing structured SQL queries."""
    # Create test data
    person = EntityNode(label="PERSON", name="John Newton")
    org = EntityNode(label="ORGANIZATION", name="Alfresco")
    arcadedb_store.upsert_nodes([person, org])

    # Execute structured query
    results = arcadedb_store.structured_query("SELECT name FROM PERSON")
    assert len(results) >= 1
    
    # Should find John Newton
    names = [r.get('name') for r in results if 'name' in r]
    assert "John Newton" in names


def test_get_schema(arcadedb_store: ArcadeDBPropertyGraphStore):
    """Test retrieving database schema information."""
    schema = arcadedb_store.get_schema()
    assert schema is not None
    
    # Schema should be a string or dict with schema information
    if isinstance(schema, dict):
        # Should contain vertex and edge type information
        assert 'vertex_types' in schema or 'edge_types' in schema
    else:
        # Should be a string description
        assert isinstance(schema, str)
        assert len(schema) > 0


def test_chunk_node_operations(arcadedb_store: ArcadeDBPropertyGraphStore):
    """Test operations specific to chunk nodes."""
    # Create chunk with embedding
    chunk = ChunkNode(
        text="The CMIS specification is backed by multiple organizations including Alfresco, EMC, IBM, Microsoft, OpenText, Oracle and SAP.",
        embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    arcadedb_store.upsert_nodes([chunk])

    # Retrieve chunk
    retrieved = arcadedb_store.get(ids=[chunk.id])
    assert len(retrieved) == 1
    assert isinstance(retrieved[0], ChunkNode)
    assert "CMIS specification" in retrieved[0].text

    # Test chunk with entity relationships (MENTIONS)
    entity = EntityNode(label="ORGANIZATION", name="Alfresco")
    arcadedb_store.upsert_nodes([entity])
    
    # Create MENTIONS relationship
    mentions_relation = Relation(
        label="MENTIONS",
        source_id=chunk.id,
        target_id=entity.id
    )
    arcadedb_store.upsert_relations([mentions_relation])

    # Verify MENTIONS relationship
    triplets = arcadedb_store.get_triplets()
    mentions_found = False
    for triplet in triplets:
        if triplet[1] == "MENTIONS" and triplet[2] == "Alfresco":
            mentions_found = True
            break
    assert mentions_found


def test_entity_types_and_schema(arcadedb_store: ArcadeDBPropertyGraphStore):
    """Test that basic entity types are created and work correctly."""
    # Create entities of different types
    person = EntityNode(label="PERSON", name="John Newton")
    org = EntityNode(label="ORGANIZATION", name="Alfresco")
    location = EntityNode(label="LOCATION", name="Boston")
    place = EntityNode(label="PLACE", name="MIT")
    
    arcadedb_store.upsert_nodes([person, org, location, place])

    # Verify all entities were created
    all_entities = arcadedb_store.get()
    assert len(all_entities) >= 4

    # Check that different entity types exist
    entity_labels = [e.label for e in all_entities if hasattr(e, 'label')]
    assert "PERSON" in entity_labels
    assert "ORGANIZATION" in entity_labels
    assert "LOCATION" in entity_labels
    assert "PLACE" in entity_labels


def test_complex_relationships(arcadedb_store: ArcadeDBPropertyGraphStore):
    """Test complex relationship scenarios."""
    # Create a network of entities
    john = EntityNode(label="PERSON", name="John Newton")
    alfresco = EntityNode(label="ORGANIZATION", name="Alfresco")
    cmis = EntityNode(label="PROJECT", name="CMIS")  # Dynamic type
    oasis = EntityNode(label="ORGANIZATION", name="OASIS")
    
    arcadedb_store.upsert_nodes([john, alfresco, cmis, oasis])

    # Create multiple relationships
    relations = [
        Relation(label="WORKS_FOR", source_id=john.id, target_id=alfresco.id),
        Relation(label="CREATED", source_id=john.id, target_id=cmis.id),
        Relation(label="STANDARDIZED_BY", source_id=cmis.id, target_id=oasis.id),
        Relation(label="MEMBER_OF", source_id=alfresco.id, target_id=oasis.id)
    ]
    arcadedb_store.upsert_relations(relations)

    # Verify all relationships exist
    triplets = arcadedb_store.get_triplets()
    assert len(triplets) >= 4

    # Check specific relationships
    relation_labels = [t[1] for t in triplets]
    assert "WORKS_FOR" in relation_labels
    assert "CREATED" in relation_labels
    assert "STANDARDIZED_BY" in relation_labels
    assert "MEMBER_OF" in relation_labels


def test_dynamic_schema_creation(arcadedb_store: ArcadeDBPropertyGraphStore):
    """Test that dynamic entity types are created automatically."""
    # Create entity with new type not in basic schema
    project = EntityNode(label="PROJECT", name="CMIS Specification")
    technology = EntityNode(label="TECHNOLOGY", name="Content Management")
    
    arcadedb_store.upsert_nodes([project, technology])

    # Verify entities were created successfully
    retrieved = arcadedb_store.get(properties={"name": "CMIS Specification"})
    assert len(retrieved) == 1
    assert retrieved[0].label == "PROJECT"

    retrieved_tech = arcadedb_store.get(properties={"name": "Content Management"})
    assert len(retrieved_tech) == 1
    assert retrieved_tech[0].label == "TECHNOLOGY"


def test_include_basic_schema_false(arcadedb_store: ArcadeDBPropertyGraphStore):
    """Test behavior when include_basic_schema=False."""
    # Create new store with basic schema disabled
    minimal_store = ArcadeDBPropertyGraphStore(
        host=arcadedb_host,
        port=arcadedb_port,
        username=arcadedb_username,
        password=arcadedb_password,
        database=arcadedb_database + "_minimal",
        include_basic_schema=False,
        embedding_dimension=1536
    )
    
    # Clear database
    try:
        minimal_store.structured_query("DELETE FROM V")
        minimal_store.structured_query("DELETE FROM E")
    except Exception:
        pass

    # Should still be able to create entities (dynamic schema)
    entity = EntityNode(label="CUSTOM_TYPE", name="Test Entity")
    minimal_store.upsert_nodes([entity])

    # Verify entity was created
    retrieved = minimal_store.get(ids=[entity.id])
    assert len(retrieved) == 1
    assert retrieved[0].name == "Test Entity"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
