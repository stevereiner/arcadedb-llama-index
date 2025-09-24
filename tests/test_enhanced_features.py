"""
Test suite for enhanced ArcadeDB Python driver features.

This test suite covers the new functionality added to address
the critical issues identified in the LlamaIndex integration.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock

from arcadedb_python import (
    DatabaseDao, 
    SyncClient,
    ArcadeDBException,
    QueryParsingException,
    TransactionException,
    BulkOperationException,
    VectorOperationException,
    ValidationException
)


class TestEnhancedExceptions:
    """Test the new typed exception system."""
    
    def test_arcadedb_exception_basic(self):
        """Test basic ArcadeDBException functionality."""
        exc = ArcadeDBException("Test error", java_error_code="TestException", detail="Test detail")
        assert str(exc) == "Test error: Test detail"
        assert exc.java_error_code == "TestException"
        assert exc.detail == "Test detail"
    
    def test_query_parsing_exception(self):
        """Test QueryParsingException with query context."""
        query = "SELECT * FROM (SELECT FROM Entity UNION SELECT FROM TextChunk)"
        exc = QueryParsingException("Parsing failed", query=query)
        assert query in str(exc)
        assert exc.query == query
    
    def test_transaction_exception_idempotent(self):
        """Test TransactionException with idempotent error flag."""
        exc = TransactionException("Not idempotent", is_idempotent_error=True)
        assert exc.is_idempotent_error is True
    
    def test_bulk_operation_exception_with_counts(self):
        """Test BulkOperationException with failure counts."""
        exc = BulkOperationException("Bulk failed", failed_records=5, total_records=100)
        assert "5/100" in str(exc)
        assert exc.failed_records == 5
        assert exc.total_records == 100


class TestBulkOperations:
    """Test bulk operation functionality."""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock DatabaseDao for testing."""
        client = Mock(spec=SyncClient)
        db = DatabaseDao(client, "test_db")
        db.query = Mock()
        return db
    
    def test_bulk_insert_success(self, mock_db):
        """Test successful bulk insert operation."""
        records = [
            {"name": "John", "age": 30},
            {"name": "Jane", "age": 25}
        ]
        
        mock_db.query.return_value = {"count": 2}
        
        result = mock_db.bulk_insert("Person", records)
        
        assert result == 2
        mock_db.query.assert_called()
    
    def test_bulk_upsert_with_key(self, mock_db):
        """Test bulk upsert operation with key field."""
        records = [
            {"id": "1", "name": "John", "age": 30},
            {"id": "2", "name": "Jane", "age": 25}
        ]
        
        mock_db.query.return_value = {"count": 2}
        
        result = mock_db.bulk_upsert("Person", records, "id")
        
        assert result == 2
        mock_db.query.assert_called()
    
    def test_bulk_delete_safe_mode(self, mock_db):
        """Test bulk delete with safe mode enabled."""
        with pytest.raises(ValidationException, match="safe mode"):
            mock_db.bulk_delete("Person", [], safe_mode=True)
    
    def test_execute_batch_queries(self, mock_db):
        """Test batch query execution."""
        queries = [
            "INSERT INTO Person SET name = 'John'",
            "INSERT INTO Person SET name = 'Jane'"
        ]
        
        mock_db.query.side_effect = [{"count": 1}, {"count": 1}]
        
        results = mock_db.execute_batch(queries)
        
        assert len(results) == 2
        assert mock_db.query.call_count == 2
    
    def test_execute_transaction_success(self, mock_db):
        """Test successful transaction execution."""
        queries = ["INSERT INTO Person SET name = 'John'"]
        
        mock_db.begin_transaction = Mock(return_value="session123")
        mock_db.commit_transaction = Mock()
        mock_db.execute_batch = Mock(return_value=[{"count": 1}])
        
        results = mock_db.execute_transaction(queries)
        
        assert len(results) == 1
        mock_db.begin_transaction.assert_called_once()
        mock_db.commit_transaction.assert_called_once_with("session123")
    
    def test_execute_transaction_rollback_on_error(self, mock_db):
        """Test transaction rollback on error."""
        queries = ["INVALID QUERY"]
        
        mock_db.begin_transaction = Mock(return_value="session123")
        mock_db.rollback_transaction = Mock()
        mock_db.execute_batch = Mock(side_effect=Exception("Query failed"))
        
        with pytest.raises(TransactionException):
            mock_db.execute_transaction(queries)
        
        mock_db.rollback_transaction.assert_called_once_with("session123")


class TestVectorOperations:
    """Test vector operation functionality."""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock DatabaseDao for testing."""
        client = Mock(spec=SyncClient)
        db = DatabaseDao(client, "test_db")
        db.query = Mock()
        return db
    
    def test_vector_search_success(self, mock_db):
        """Test successful vector similarity search."""
        query_embedding = [0.1, 0.2, 0.3]
        mock_results = [
            {"id": "1", "name": "doc1", "similarity_score": 0.95},
            {"id": "2", "name": "doc2", "similarity_score": 0.87}
        ]
        
        mock_db.query.return_value = mock_results
        
        results = mock_db.vector_search("Document", "embedding", query_embedding, top_k=2)
        
        assert len(results) == 2
        assert results[0]["similarity_score"] == 0.95
        mock_db.query.assert_called_once()
    
    def test_vector_search_invalid_embedding(self, mock_db):
        """Test vector search with invalid embedding."""
        with pytest.raises(ValidationException, match="non-empty list"):
            mock_db.vector_search("Document", "embedding", [], top_k=10)
        
        with pytest.raises(ValidationException, match="numeric values"):
            mock_db.vector_search("Document", "embedding", ["not", "numeric"], top_k=10)
    
    def test_create_vector_index_success(self, mock_db):
        """Test successful vector index creation."""
        mock_db.query.return_value = {"result": "ok"}
        
        result = mock_db.create_vector_index("Document", "embedding", 768)
        
        assert result is True
        mock_db.query.assert_called()
    
    def test_get_vector_similarity(self, mock_db):
        """Test vector similarity calculation."""
        query_embedding = [0.1, 0.2, 0.3]
        mock_db.query.return_value = [{"similarity": 0.92}]
        
        similarity = mock_db.get_vector_similarity("Document", "embedding", "#1:1", query_embedding)
        
        assert similarity == 0.92
        mock_db.query.assert_called_once()
    
    def test_batch_vector_search(self, mock_db):
        """Test batch vector search operations."""
        searches = [
            {
                "type_name": "Document",
                "embedding_field": "embedding",
                "query_embedding": [0.1, 0.2, 0.3],
                "top_k": 5
            },
            {
                "type_name": "Image", 
                "embedding_field": "features",
                "query_embedding": [0.4, 0.5, 0.6],
                "top_k": 3
            }
        ]
        
        # Mock the vector_search method
        mock_db.vector_search = Mock(side_effect=[
            [{"id": "doc1", "score": 0.9}],
            [{"id": "img1", "score": 0.8}]
        ])
        
        results = mock_db.batch_vector_search(searches)
        
        assert len(results) == 2
        assert len(results[0]) == 1
        assert len(results[1]) == 1
        assert mock_db.vector_search.call_count == 2


class TestQueryParsing:
    """Test enhanced query parsing functionality."""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock DatabaseDao for testing."""
        client = Mock(spec=SyncClient)
        db = DatabaseDao(client, "test_db")
        db.query = Mock()
        return db
    
    def test_get_records_single_type(self, mock_db):
        """Test getting records from a single type."""
        mock_results = [{"id": "1", "name": "John"}]
        mock_db.query.return_value = mock_results
        
        results = mock_db.get_records("Person", where_clause="age > 18", limit=10)
        
        assert results == mock_results
        mock_db.query.assert_called_once()
        
        # Check the generated query
        call_args = mock_db.query.call_args[0]
        assert "SELECT * FROM Person" in call_args[1]
        assert "WHERE age > 18" in call_args[1]
        assert "LIMIT 10" in call_args[1]
    
    def test_get_records_multiple_types_union_success(self, mock_db):
        """Test getting records from multiple types with successful UNION."""
        mock_results = [{"id": "1", "name": "John"}, {"id": "2", "name": "Jane"}]
        mock_db.query.return_value = mock_results
        
        results = mock_db.get_records(["Person", "Employee"], limit=10)
        
        assert results == mock_results
        mock_db.query.assert_called_once()
        
        # Check the generated UNION query
        call_args = mock_db.query.call_args[0]
        assert "UNION" in call_args[1]
    
    def test_get_records_multiple_types_union_fallback(self, mock_db):
        """Test fallback to individual queries when UNION fails."""
        # First call (UNION) fails, subsequent calls (individual queries) succeed
        mock_db.query.side_effect = [
            QueryParsingException("UNION failed"),  # UNION query fails
            [{"id": "1", "name": "John"}],          # Person query succeeds
            [{"id": "2", "name": "Jane"}]           # Employee query succeeds
        ]
        
        results = mock_db.get_records(["Person", "Employee"])
        
        assert len(results) == 2
        assert mock_db.query.call_count == 3  # 1 failed UNION + 2 individual queries
    
    def test_get_triplets_match_success(self, mock_db):
        """Test getting triplets with successful MATCH query."""
        mock_results = [
            {
                "subject": {"id": "1", "name": "John"},
                "relation": {"type": "KNOWS"},
                "object": {"id": "2", "name": "Jane"}
            }
        ]
        mock_db.query.return_value = mock_results
        
        results = mock_db.get_triplets(subject_types=["Person"], relation_types=["KNOWS"])
        
        assert results == mock_results
        mock_db.query.assert_called_once()
        
        # Check the generated MATCH query
        call_args = mock_db.query.call_args[0]
        assert "MATCH" in call_args[1]
        assert "RETURN subject, relation, object" in call_args[1]
    
    def test_get_triplets_match_fallback(self, mock_db):
        """Test fallback to edge traversal when MATCH fails."""
        # First call (MATCH) fails, second call (edge traversal) succeeds
        mock_db.query.side_effect = [
            QueryParsingException("MATCH failed"),  # MATCH query fails
            [{"in": "#1:1", "out": "#2:1", "type": "KNOWS"}]  # Edge query succeeds
        ]
        
        results = mock_db.get_triplets()
        
        assert len(results) == 1
        assert mock_db.query.call_count == 2


class TestTransactionManagement:
    """Test enhanced transaction management."""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock DatabaseDao for testing."""
        client = Mock(spec=SyncClient)
        db = DatabaseDao(client, "test_db")
        return db
    
    @patch('arcadedb_python.dao.database.logging')
    def test_query_retry_on_idempotent_error(self, mock_logging, mock_db):
        """Test automatic retry of non-idempotent queries as commands."""
        # Mock the client.post method to simulate the retry behavior
        mock_db.client.post = Mock()
        
        # First call raises TransactionException with idempotent error
        idempotent_error = TransactionException("Not idempotent", is_idempotent_error=True)
        # Second call succeeds
        success_result = {"result": "success"}
        
        mock_db.client.post.side_effect = [idempotent_error, success_result]
        
        result = mock_db.query("sql", "DELETE FROM Person WHERE id = 1", retry_on_idempotent_error=True)
        
        assert result == success_result
        assert mock_db.client.post.call_count == 2
        mock_logging.info.assert_called()
    
    def test_safe_delete_all_truncate_success(self, mock_db):
        """Test safe delete all with successful TRUNCATE."""
        mock_db.query = Mock(return_value={"result": "ok"})
        
        result = mock_db.safe_delete_all("Person")
        
        assert result == 0  # TRUNCATE doesn't return count
        mock_db.query.assert_called_once()
        
        # Check that TRUNCATE was attempted
        call_args = mock_db.query.call_args[0]
        assert "TRUNCATE TYPE Person UNSAFE" in call_args[1]
    
    def test_safe_delete_all_batch_fallback(self, mock_db):
        """Test safe delete all with batch fallback when TRUNCATE fails."""
        # First call (TRUNCATE) fails, subsequent calls (batch delete) succeed
        mock_db.query = Mock(side_effect=[
            Exception("TRUNCATE failed"),           # TRUNCATE fails
            [{"@rid": "#1:1"}, {"@rid": "#1:2"}],  # First batch of records
            {"result": "ok"},                       # Delete batch succeeds
            []                                      # No more records
        ])
        
        result = mock_db.safe_delete_all("Person", batch_size=2)
        
        assert result == 2  # 2 records deleted
        assert mock_db.query.call_count == 4
    
    def test_safe_bulk_operation_retry_success(self, mock_db):
        """Test safe bulk operation with successful retry."""
        # Mock operation function that fails once then succeeds
        operation_func = Mock(side_effect=[
            BulkOperationException("Partial failure", failed_records=1, total_records=10),
            5  # Success on retry
        ])
        
        result = mock_db.safe_bulk_operation(operation_func, "arg1", max_retries=2)
        
        assert result == 5
        assert operation_func.call_count == 2
    
    def test_get_transaction_status(self, mock_db):
        """Test getting transaction status."""
        mock_status = {"session_id": "session123", "status": "active", "isolation": "READ_COMMITTED"}
        mock_db.query = Mock(return_value=[mock_status])
        
        status = mock_db.get_transaction_status("session123")
        
        assert status == mock_status
        mock_db.query.assert_called_once()


class TestValidation:
    """Test input validation functionality."""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock DatabaseDao for testing."""
        client = Mock(spec=SyncClient)
        db = DatabaseDao(client, "test_db")
        return db
    
    def test_query_language_validation(self, mock_db):
        """Test query language validation."""
        with pytest.raises(ValidationException, match="not supported"):
            mock_db.query("invalid_language", "SELECT 1")
    
    def test_query_limit_validation(self, mock_db):
        """Test query limit validation."""
        with pytest.raises(ValidationException, match="integer"):
            mock_db.query("sql", "SELECT 1", limit="not_an_integer")
    
    def test_query_serializer_validation(self, mock_db):
        """Test query serializer validation."""
        with pytest.raises(ValidationException, match="graph"):
            mock_db.query("sql", "SELECT 1", serializer="invalid")


if __name__ == "__main__":
    pytest.main([__file__])
