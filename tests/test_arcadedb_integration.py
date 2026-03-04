import time
import unittest
import docker
import pytest
from llama_index.graph_stores.arcadedb import ArcadeDBPropertyGraphStore
from llama_index.core.graph_stores.types import EntityNode, ChunkNode, Relation
from llama_index.core.vector_stores.types import VectorStoreQuery

# Set up Docker client
docker_client = docker.from_env()


class TestArcadeDBPropertyGraphStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup method called once for the entire test class."""
        # Clean up any existing test containers
        try:
            existing = docker_client.containers.get("arcadedb_test_instance_pg")
            existing.stop()
            existing.remove()
        except:
            pass  # Container doesn't exist
        
        # Start ArcadeDB container using HTTP API port 2480
        try:
            cls.container = docker_client.containers.run(
                "arcadedata/arcadedb:latest",
                detach=True,
                name="arcadedb_test_instance_pg",
                ports={"2480/tcp": 2481, "2424/tcp": 2425},  # Map to offset ports to avoid conflicts
                environment={
                    "JAVA_OPTS": "-Darcadedb.server.rootPassword=playwithdata"
                }
            )
            time.sleep(15)  # Allow time for ArcadeDB to initialize
        except Exception as e:
            print(f"Error starting ArcadeDB container: {e}")
            raise

        # Set up the property graph store using HTTP API
        # embedding_dimension=4 matches the 4-dim test vectors used in test_vector_query_with_embeddings
        cls.pg_store = ArcadeDBPropertyGraphStore(
            host="localhost",
            port=2481,  # HTTP API port (mapped from container's 2480)
            username="root",
            password="playwithdata",
            database="test_graph",  # Use separate test database
            include_basic_schema=True,  # Enable basic entity types (PERSON, ORGANIZATION, etc.)
            embedding_dimension=4
        )

    @classmethod
    def tearDownClass(cls):
        """Teardown method called once after all tests are done."""
        try:
            cls.container.stop()
            cls.container.remove()
        except Exception as e:
            print(f"Error stopping/removing container: {e}")

    def setUp(self):
        """Clear database before each test."""
        # Clear all vertices and edges for fresh test using actual type names
        try:
            # Delete from all vertex types
            vertex_types = ["Entity", "TextChunk", "PERSON", "ORGANIZATION", "LOCATION", "PLACE", "PROJECT", "TECHNOLOGY"]
            for vertex_type in vertex_types:
                try:
                    self.pg_store.structured_query(f"DELETE FROM {vertex_type}")
                except Exception:
                    pass  # Ignore if type doesn't exist
            
            # Delete from all edge types  
            edge_types = ["MENTIONS", "WORKS_FOR", "CREATED", "STANDARDIZED_BY", "MEMBER_OF"]
            for edge_type in edge_types:
                try:
                    self.pg_store.structured_query(f"DELETE FROM {edge_type}")
                except Exception:
                    pass  # Ignore if type doesn't exist
                    
        except Exception:
            pass  # Ignore any cleanup errors

    def test_upsert_nodes_and_get(self):
        """Test inserting entity and chunk nodes, then retrieving them."""
        entity = EntityNode(label="PERSON", name="John Newton")
        chunk = ChunkNode(text="John Newton was Chairman and CTO of Alfresco.")
        self.pg_store.upsert_nodes([entity, chunk])

        # Get by ID
        retrieved_entities = self.pg_store.get(ids=[entity.id])
        self.assertEqual(len(retrieved_entities), 1)
        self.assertIsInstance(retrieved_entities[0], EntityNode)
        self.assertEqual(retrieved_entities[0].name, "John Newton")

        retrieved_chunks = self.pg_store.get(ids=[chunk.id])
        self.assertEqual(len(retrieved_chunks), 1)
        self.assertIsInstance(retrieved_chunks[0], ChunkNode)
        self.assertIn("John Newton", retrieved_chunks[0].text)

    def test_upsert_relations(self):
        """Test inserting relations between nodes."""
        # Create entities
        person = EntityNode(label="PERSON", name="John Newton")
        org = EntityNode(label="ORGANIZATION", name="Alfresco")
        self.pg_store.upsert_nodes([person, org])

        # Create relation
        relation = Relation(
            label="WORKS_FOR",
            source_id=person.id,
            target_id=org.id,
            properties={"role": "Chairman and CTO"}
        )
        self.pg_store.upsert_relations([relation])

        # Verify relation exists
        triplets = self.pg_store.get_triplets()
        self.assertGreater(len(triplets), 0)
        
        # Find our relation
        found_relation = False
        for triplet in triplets:
            if (triplet[0] == "John Newton" and 
                triplet[1] == "WORKS_FOR" and 
                triplet[2] == "Alfresco"):
                found_relation = True
                break
        self.assertTrue(found_relation)

    def test_get_nodes_by_properties(self):
        """Test retrieving nodes by their properties."""
        # Create entities with different labels
        person1 = EntityNode(label="PERSON", name="John Newton")
        person2 = EntityNode(label="PERSON", name="Alice Smith")
        org = EntityNode(label="ORGANIZATION", name="Alfresco")
        self.pg_store.upsert_nodes([person1, person2, org])

        # Get all nodes and filter by type (since ArcadeDB stores entities in separate tables by type)
        all_nodes = self.pg_store.get()
        persons = [node for node in all_nodes if hasattr(node, 'label') and node.label == "PERSON"]
        self.assertEqual(len(persons), 2)
        person_names = [p.name for p in persons]
        self.assertIn("John Newton", person_names)
        self.assertIn("Alice Smith", person_names)

        # Get specific organization by name property
        orgs = self.pg_store.get(properties={"name": "Alfresco"})
        self.assertEqual(len(orgs), 1)
        self.assertEqual(orgs[0].name, "Alfresco")

    def test_vector_query_with_embeddings(self):
        """Test vector similarity search with embeddings via ArcadeDB LSM_VECTOR index."""
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
        self.pg_store.upsert_nodes([entity1, entity2, entity3])

        # Query embedding close to entity1 and entity2
        query_embedding = [0.15, 0.25, 0.35, 0.45]
        query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=2)

        nodes, scores = self.pg_store.vector_query(query)

        # Should return at most top_k results
        self.assertLessEqual(len(nodes), 2)
        self.assertEqual(len(nodes), len(scores))

        # Every result must be a LabelledNode (EntityNode or ChunkNode)
        from llama_index.core.graph_stores.types import LabelledNode
        for node in nodes:
            self.assertIsInstance(node, LabelledNode)

        # The two PERSON nodes should rank above the distant ORGANIZATION node
        node_names = [n.name for n in nodes if hasattr(n, 'name')]
        self.assertTrue(
            any(name in node_names for name in ["John Newton", "Alice Smith"]),
            f"Expected a PERSON node in results, got: {node_names}"
        )

    def test_structured_query(self):
        """Test executing structured SQL queries."""
        # Create test data
        person = EntityNode(label="PERSON", name="John Newton")
        org = EntityNode(label="ORGANIZATION", name="Alfresco")
        self.pg_store.upsert_nodes([person, org])

        # Execute structured query
        results = self.pg_store.structured_query("SELECT name FROM PERSON")
        self.assertGreaterEqual(len(results), 1)
        
        # Should find John Newton
        names = [r.get('name') for r in results if 'name' in r]
        self.assertIn("John Newton", names)

    def test_get_schema(self):
        """Test retrieving database schema information."""
        schema = self.pg_store.get_schema()
        self.assertIsNotNone(schema)
        
        # Schema should be a string or dict with schema information
        if isinstance(schema, dict):
            # Should contain vertex and edge type information
            self.assertTrue('vertex_types' in schema or 'edge_types' in schema)
        else:
            # Should be a string description
            self.assertIsInstance(schema, str)
            self.assertGreater(len(schema), 0)

    def test_chunk_node_operations(self):
        """Test operations specific to chunk nodes."""
        # Create chunk with embedding
        chunk = ChunkNode(
            text="The CMIS specification is backed by multiple organizations including Alfresco, EMC, IBM, Microsoft, OpenText, Oracle and SAP.",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        self.pg_store.upsert_nodes([chunk])

        # Retrieve chunk
        retrieved = self.pg_store.get(ids=[chunk.id])
        self.assertEqual(len(retrieved), 1)
        self.assertIsInstance(retrieved[0], ChunkNode)
        self.assertIn("CMIS specification", retrieved[0].text)

        # Test chunk with entity relationships (MENTIONS)
        entity = EntityNode(label="ORGANIZATION", name="Alfresco")
        self.pg_store.upsert_nodes([entity])
        
        # Create MENTIONS relationship
        mentions_relation = Relation(
            label="MENTIONS",
            source_id=chunk.id,
            target_id=entity.id
        )
        self.pg_store.upsert_relations([mentions_relation])

        # Verify MENTIONS relationship
        triplets = self.pg_store.get_triplets()
        mentions_found = False
        for triplet in triplets:
            if triplet[1] == "MENTIONS" and triplet[2] == "Alfresco":
                mentions_found = True
                break
        self.assertTrue(mentions_found)

    def test_entity_types_and_schema(self):
        """Test that basic entity types are created and work correctly."""
        # Create entities of different types
        person = EntityNode(label="PERSON", name="John Newton")
        org = EntityNode(label="ORGANIZATION", name="Alfresco")
        location = EntityNode(label="LOCATION", name="Boston")
        place = EntityNode(label="PLACE", name="MIT")
        
        self.pg_store.upsert_nodes([person, org, location, place])

        # Verify all entities were created
        all_entities = self.pg_store.get()
        self.assertGreaterEqual(len(all_entities), 4)

        # Check that different entity types exist
        entity_labels = [e.label for e in all_entities if hasattr(e, 'label')]
        self.assertIn("PERSON", entity_labels)
        self.assertIn("ORGANIZATION", entity_labels)
        self.assertIn("LOCATION", entity_labels)
        self.assertIn("PLACE", entity_labels)

    def test_complex_relationships(self):
        """Test complex relationship scenarios."""
        # Create a network of entities
        john = EntityNode(label="PERSON", name="John Newton")
        alfresco = EntityNode(label="ORGANIZATION", name="Alfresco")
        cmis = EntityNode(label="PROJECT", name="CMIS")  # Dynamic type
        oasis = EntityNode(label="ORGANIZATION", name="OASIS")
        
        self.pg_store.upsert_nodes([john, alfresco, cmis, oasis])

        # Create multiple relationships
        relations = [
            Relation(label="WORKS_FOR", source_id=john.id, target_id=alfresco.id),
            Relation(label="CREATED", source_id=john.id, target_id=cmis.id),
            Relation(label="STANDARDIZED_BY", source_id=cmis.id, target_id=oasis.id),
            Relation(label="MEMBER_OF", source_id=alfresco.id, target_id=oasis.id)
        ]
        self.pg_store.upsert_relations(relations)

        # Verify all relationships exist
        triplets = self.pg_store.get_triplets()
        self.assertGreaterEqual(len(triplets), 4)

        # Check specific relationships
        relation_labels = [t[1] for t in triplets]
        self.assertIn("WORKS_FOR", relation_labels)
        self.assertIn("CREATED", relation_labels)
        self.assertIn("STANDARDIZED_BY", relation_labels)
        self.assertIn("MEMBER_OF", relation_labels)

    def test_dynamic_schema_creation(self):
        """Test that dynamic entity types are created automatically."""
        # Create entity with new type not in basic schema
        project = EntityNode(label="PROJECT", name="CMIS Specification")
        technology = EntityNode(label="TECHNOLOGY", name="Content Management")
        
        self.pg_store.upsert_nodes([project, technology])

        # Verify entities were created successfully
        retrieved = self.pg_store.get(properties={"name": "CMIS Specification"})
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0].label, "PROJECT")

        retrieved_tech = self.pg_store.get(properties={"name": "Content Management"})
        self.assertEqual(len(retrieved_tech), 1)
        self.assertEqual(retrieved_tech[0].label, "TECHNOLOGY")

    # ------------------------------------------------------------------
    # delete() tests
    # ------------------------------------------------------------------

    def test_delete_by_entity_names(self):
        """delete(entity_names=[...]) removes nodes from ALL vertex types."""
        # Create nodes in two different vertex types
        person = EntityNode(label="PERSON", name="DeleteMe Person")
        org = EntityNode(label="ORGANIZATION", name="DeleteMe Org")
        self.pg_store.upsert_nodes([person, org])

        # Confirm they exist
        self.assertEqual(len(self.pg_store.get(properties={"name": "DeleteMe Person"})), 1)
        self.assertEqual(len(self.pg_store.get(properties={"name": "DeleteMe Org"})), 1)

        # Delete both by name in one call
        self.pg_store.delete(entity_names=["DeleteMe Person", "DeleteMe Org"])

        # Both must be gone
        self.assertEqual(len(self.pg_store.get(properties={"name": "DeleteMe Person"})), 0,
                         "PERSON node should have been deleted")
        self.assertEqual(len(self.pg_store.get(properties={"name": "DeleteMe Org"})), 0,
                         "ORGANIZATION node should have been deleted")

    def test_delete_by_ids(self):
        """delete(ids=[...]) removes nodes looked up by logical id/name, not @rid."""
        # LlamaIndex logical IDs: EntityNode.id is derived from name, ChunkNode.id is a UUID
        person = EntityNode(label="PERSON", name="DeleteById Person")
        chunk = ChunkNode(text="DeleteById chunk text")
        self.pg_store.upsert_nodes([person, chunk])

        # Confirm they exist
        self.assertEqual(len(self.pg_store.get(ids=[person.id])), 1,
                         "Person should exist before delete")
        self.assertEqual(len(self.pg_store.get(ids=[chunk.id])), 1,
                         "Chunk should exist before delete")

        # Delete by logical IDs
        self.pg_store.delete(ids=[person.id, chunk.id])

        # Both must be gone
        self.assertEqual(len(self.pg_store.get(ids=[person.id])), 0,
                         "Person node should have been deleted by logical id")
        self.assertEqual(len(self.pg_store.get(ids=[chunk.id])), 0,
                         "Chunk node should have been deleted by logical id")

    def test_delete_by_properties_ref_doc_id(self):
        """delete(properties={'ref_doc_id': ...}) removes TextChunk nodes by ref_doc_id.

        This is the path used by flexible-graphrag's _delete_from_graph_helper.
        ref_doc_id must be persisted on TextChunk by _upsert_chunk_node.
        """
        doc_id = "test-doc-abc123"
        chunk1 = ChunkNode(text="Chunk one for ref_doc_id test",
                           properties={"ref_doc_id": doc_id})
        chunk2 = ChunkNode(text="Chunk two for ref_doc_id test",
                           properties={"ref_doc_id": doc_id})
        other_chunk = ChunkNode(text="Unrelated chunk",
                                properties={"ref_doc_id": "other-doc-xyz"})
        self.pg_store.upsert_nodes([chunk1, chunk2, other_chunk])

        # Verify ref_doc_id was stored
        stored = self.pg_store.structured_query(
            f"SELECT id, ref_doc_id FROM TextChunk WHERE ref_doc_id = '{doc_id}'"
        )
        self.assertEqual(len(stored), 2,
                         f"Expected 2 chunks with ref_doc_id='{doc_id}', got {len(stored)}")

        # Delete by ref_doc_id property
        self.pg_store.delete(properties={"ref_doc_id": doc_id})

        # The two target chunks must be gone; the unrelated one must survive
        remaining = self.pg_store.structured_query(
            f"SELECT id FROM TextChunk WHERE ref_doc_id = '{doc_id}'"
        )
        self.assertEqual(len(remaining), 0,
                         "All chunks with the given ref_doc_id should be deleted")

        unrelated = self.pg_store.get(ids=[other_chunk.id])
        self.assertEqual(len(unrelated), 1,
                         "Chunk with a different ref_doc_id should NOT be deleted")

    def test_delete_by_properties_across_vertex_types(self):
        """delete(properties={...}) removes matching nodes from ALL vertex types."""
        tag = "delete_prop_test"
        person = EntityNode(label="PERSON", name="PropDelete Person",
                            properties={"test_tag": tag})
        org = EntityNode(label="ORGANIZATION", name="PropDelete Org",
                         properties={"test_tag": tag})
        self.pg_store.upsert_nodes([person, org])

        # Confirm both exist
        self.assertEqual(len(self.pg_store.get(properties={"name": "PropDelete Person"})), 1)
        self.assertEqual(len(self.pg_store.get(properties={"name": "PropDelete Org"})), 1)

        # Delete by the shared custom property
        self.pg_store.delete(properties={"test_tag": tag})

        # Both must be gone
        self.assertEqual(len(self.pg_store.get(properties={"name": "PropDelete Person"})), 0,
                         "PERSON node should be deleted by property match")
        self.assertEqual(len(self.pg_store.get(properties={"name": "PropDelete Org"})), 0,
                         "ORGANIZATION node should be deleted by property match")

    def test_delete_relation_names(self):
        """delete(relation_names=[...]) removes all edges of the given type."""
        person = EntityNode(label="PERSON", name="RelDelete Person")
        org = EntityNode(label="ORGANIZATION", name="RelDelete Org")
        self.pg_store.upsert_nodes([person, org])
        self.pg_store.upsert_relations([
            Relation(label="WORKS_FOR", source_id=person.id, target_id=org.id)
        ])

        # Confirm relation exists
        triplets = self.pg_store.get_triplets()
        self.assertTrue(any(t[1] == "WORKS_FOR" for t in triplets),
                        "WORKS_FOR relation should exist before delete")

        # Delete the relation type
        self.pg_store.delete(relation_names=["WORKS_FOR"])

        # No WORKS_FOR edges should remain
        triplets_after = self.pg_store.get_triplets()
        self.assertFalse(any(t[1] == "WORKS_FOR" for t in triplets_after),
                         "WORKS_FOR relations should all be deleted")


if __name__ == "__main__":
    unittest.main()