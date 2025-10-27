import time
import unittest
from llama_index.core.graph_stores.types import GraphStore
from llama_index.graph_stores.arcadedb import ArcadeDBGraphStore


class TestArcadeDBGraphStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup method called once for the entire test class."""
        # Reuse existing ArcadeDB server from docker-compose but use different database
        # This avoids port conflicts and is more efficient
        
        # Set up the graph store using existing server with unique database name
        cls.graph_store = ArcadeDBGraphStore(
            host="localhost",
            port=2480,  # Use existing docker-compose port
            username="root",
            password="playwithdata",
            database=f"graph_test_{int(time.time())}"  # Unique database name
        )

    @classmethod
    def tearDownClass(cls):
        """Teardown method called once after all tests are done."""
        # Clean up the test database
        try:
            cls.graph_store.clear_database()
        except Exception as e:
            print(f"Error cleaning up test database: {e}")

    def setUp(self):
        """Clear database before each test."""
        try:
            # Use TRUNCATE instead of DELETE for better performance and idempotency
            self.graph_store._db.query("sql", "TRUNCATE TYPE V UNSAFE")
            self.graph_store._db.query("sql", "TRUNCATE TYPE E UNSAFE")
        except Exception:
            pass  # Ignore errors if tables don't exist

    def test_graph_store_interface(self):
        """Test that ArcadeDBGraphStore implements GraphStore interface."""
        self.assertIsInstance(self.graph_store, GraphStore)

    def test_upsert_triplet(self):
        """Test basic triplet upsert functionality."""
        # Test basic triplet
        self.graph_store.upsert_triplet("Alice", "KNOWS", "Bob")
        
        # Verify the triplet was stored using get method
        alice_triplets = self.graph_store.get("Alice")
        self.assertGreaterEqual(len(alice_triplets), 1)
        
        # Check if our triplet is in the results
        triplet_found = False
        for triplet in alice_triplets:
            if (triplet[0] == "Alice" and 
                triplet[1] == "KNOWS" and 
                triplet[2] == "Bob"):
                triplet_found = True
                break
        self.assertTrue(triplet_found, f"Expected triplet not found in: {alice_triplets}")

    def test_get_empty_subject(self):
        """Test getting triplets for non-existent subject."""
        triplets = self.graph_store.get("NonExistentSubject")
        self.assertIsInstance(triplets, list)
        self.assertEqual(len(triplets), 0)

    def test_multiple_triplets_same_subject(self):
        """Test multiple triplets with same subject."""
        # Add multiple triplets for John
        self.graph_store.upsert_triplet("John", "WORKS_FOR", "Alfresco")
        self.graph_store.upsert_triplet("John", "CREATED", "CMIS")
        self.graph_store.upsert_triplet("John", "LIVES_IN", "Boston")
        
        # Get all John's triplets
        john_triplets = self.graph_store.get("John")
        self.assertGreaterEqual(len(john_triplets), 3)
        
        # Verify all triplets are for John
        for triplet in john_triplets:
            self.assertEqual(triplet[0], "John")
        
        # Check specific relationships exist
        relations = [triplet[1] for triplet in john_triplets]
        self.assertIn("WORKS_FOR", relations)
        self.assertIn("CREATED", relations)
        self.assertIn("LIVES_IN", relations)


if __name__ == "__main__":
    unittest.main(verbosity=2)