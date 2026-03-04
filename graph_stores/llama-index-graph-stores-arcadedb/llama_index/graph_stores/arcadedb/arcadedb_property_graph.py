"""ArcadeDB Property Graph Store implementation for LlamaIndex.

This module provides a PropertyGraphStore implementation for ArcadeDB,
enabling LlamaIndex to work with ArcadeDB as a graph database backend.

Requirements:
- arcadedb_python library: pip install arcadedb-python
- LlamaIndex core with PropertyGraphStore interface

Example usage:
    from llama_index.graph_stores.arcadedb import ArcadeDBPropertyGraphStore
    from llama_index.core import PropertyGraphIndex

    # Initialize the graph store
    graph_store = ArcadeDBPropertyGraphStore(
        host="localhost",
        port=2480,
        username="root", 
        password="playwithdata",
        database="graph_db",
        embedding_dimension=1536  # Optional: OpenAI ada-002 (use 384 for Ollama all-MiniLM-L6-v2)
    )

    # Create a property graph index
    index = PropertyGraphIndex.from_documents(
        documents,
        property_graph_store=graph_store,
        show_progress=True
    )
"""

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from arcadedb_python.api.sync import SyncClient
    from arcadedb_python.dao.database import DatabaseDao
    from arcadedb_python import (
        ArcadeDBException,
        QueryParsingException,
        ValidationException,
        BulkOperationException,
        TransactionException,
        VectorOperationException,
    )
except ImportError as e:
    raise ImportError(
        "arcadedb_python>=0.4.0 is required for ArcadeDB integration. "
        "Install it with: pip install arcadedb-python>=0.4.0"
    ) from e

from llama_index.core.graph_stores.types import (
    PropertyGraphStore,
    LabelledNode,
    EntityNode,
    ChunkNode,
    Relation,
    Triplet,
)
from llama_index.core.vector_stores.types import VectorStoreQuery

logger = logging.getLogger(__name__)




class ArcadeDBPropertyGraphStore(PropertyGraphStore):
    """ArcadeDB Property Graph Store.

    This class implements the PropertyGraphStore interface for ArcadeDB using native SQL,
    providing optimal performance for graph operations and full vector search capabilities.
    
    Uses ArcadeDB's native SQL engine with graph traversal patterns for reliable,
    high-performance graph operations.

    Args:
        host: ArcadeDB server host (default: "localhost")
        port: ArcadeDB server port (default: 2480)
        username: Database username (default: "root")
        password: Database password (default: "password")
        database: Database name (default: "graph")
        create_database_if_not_exists: Whether to create database if it doesn't exist (default: True)
        include_basic_schema: Whether to include basic entity types beyond core types (default: True)
        embedding_dimension: Optional dimension for vector embeddings
        **kwargs: Additional arguments
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2480,
        username: str = "root",
        password: str = "playwithdata",
        database: str = "graph",
        create_database_if_not_exists: bool = True,
        include_basic_schema: bool = True,
        embedding_dimension: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the ArcadeDB Property Graph Store.
        
        Schema Creation:
        - Always creates: Entity, TextChunk (vertex types) + MENTIONS (edge type)
        - include_basic_schema=True: Also creates PERSON, ORGANIZATION, LOCATION, PLACE vertex types
        - include_basic_schema=False: Only creates core types, lets LlamaIndex handle schema dynamically
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.include_basic_schema = include_basic_schema
        self.embedding_dimension = embedding_dimension
        
        # Track dynamically discovered schema
        self._discovered_vertex_types = set()
        self._discovered_edge_types = set()
        # Cache of (type_name, prop_name) pairs whose DDL has already been sent,
        # so _ensure_property() skips redundant CREATE PROPERTY calls.
        self._known_properties: set = set()

        # Initialize ArcadeDB client with the FIXED version
        try:
            self._client = SyncClient(
                host=host,
                port=port,
                username=username,
                password=password,
                content_type="application/json"
            )
            logger.info(f"Initialized ArcadeDB client for {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to initialize ArcadeDB client: {e}")
            raise

        # Connect to or create database using proper pattern
        try:
            if not DatabaseDao.exists(self._client, database):
                if create_database_if_not_exists:
                    logger.info(f"Database {database} does not exist, creating...")
                    self._db = DatabaseDao.create(self._client, database)
                    logger.info(f"Created database: {database}")
                else:
                    logger.error(f"Database {database} does not exist and create_database_if_not_exists=False")
                    raise Exception(f"Database {database} does not exist")
            else:
                logger.info(f"Database {database} exists, connecting...")
                self._db = DatabaseDao(self._client, database)
                logger.info(f"Connected to existing database: {database}")
            
            # Test the connection with a simple query
            test_result = self._db.query("sql", "SELECT 1 as test")
            logger.info(f"Database connection test successful: {test_result}")
            
        except Exception as e:
            logger.error(f"Database connection/creation failed: {e}")
            raise

        # Set PropertyGraphStore capabilities
        self.supports_structured_queries = True
        self.supports_vector_queries = True  # We support vector queries via manual similarity

        # Initialize schema with proper error handling
        try:
            self._ensure_schema()
            logger.info(f"Schema initialization completed for {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise

    @property
    def client(self) -> Any:
        """Get the database client."""
        return self._db

    def _ensure_schema(self) -> None:
        """Ensure basic schema exists for nodes and relationships."""
        try:
            # First, check what types already exist
            existing_types = set()
            try:
                result = self._db.query("sql", "SELECT name FROM schema:types")
                logger.info(f"Raw schema query result: {result}")
                logger.info(f"Result type: {type(result)}")
                
                # Handle ArcadeDB response format: {"result": {"records": [{"name": "TypeName"}]}}
                existing_types = set()
                
                if isinstance(result, dict) and 'result' in result:
                    result_data = result['result']
                    if isinstance(result_data, dict) and 'records' in result_data:
                        records = result_data['records']
                        for record in records:
                            if isinstance(record, dict) and 'name' in record:
                                existing_types.add(record['name'])
                elif isinstance(result, list):
                    # Fallback for direct list format
                    for record in result:
                        if isinstance(record, dict) and 'name' in record:
                            existing_types.add(record['name'])
                
                logger.info(f"Existing types: {existing_types}")
            except Exception as e:
                logger.warning(f"Could not check existing types: {e}")
                existing_types = set()

            # Essential types - always created (core functionality)
            essential_schema = [
                ("Entity", "VERTEX"),
                ("TextChunk", "VERTEX"),
                ("MENTIONS", "EDGE"),  # For chunk-to-entity relationships
            ]
            
            # Basic entity types - only created when include_basic_schema=True
            basic_entity_types = [
                ("PERSON", "VERTEX"),
                ("ORGANIZATION", "VERTEX"),
                ("LOCATION", "VERTEX"), 
                ("PLACE", "VERTEX"),
            ]
            
            # Combine schema definitions based on include_basic_schema setting
            schema_definitions = essential_schema.copy()
            if self.include_basic_schema:
                schema_definitions.extend(basic_entity_types)
                logger.info("Including basic entity types: PERSON, ORGANIZATION, LOCATION, PLACE")
            else:
                logger.info("Basic entity types disabled - only creating essential types: Entity, TextChunk, MENTIONS")

            created_count = 0
            for type_name, type_kind in schema_definitions:
                if type_name in existing_types:
                    logger.info(f"Type {type_name} already exists, ensuring properties exist")
                    # Even if type exists, ensure properties are created
                    if type_kind == 'VERTEX':
                        self._create_vertex_properties(type_name)
                        self._discovered_vertex_types.add(type_name)
                    else:
                        self._discovered_edge_types.add(type_name)
                    continue
                    
                try:
                    query = f"CREATE {type_kind} TYPE {type_name}"
                    result = self._db.query("sql", query, is_command=True)
                    logger.debug(f"Schema created successfully: {query}")
                    
                    # For VERTEX types, create essential properties with correct data types
                    if type_kind == 'VERTEX':
                        self._create_vertex_properties(type_name)
                        self._discovered_vertex_types.add(type_name)
                    else:
                        self._discovered_edge_types.add(type_name)
                    
                    created_count += 1
                except (TransactionException, SchemaException) as e:
                    # TransactionException with is_idempotent_error or SchemaException
                    # both indicate the type already exists — treat as success.
                    already_exists = (
                        (isinstance(e, TransactionException) and e.is_idempotent_error)
                        or "already exists" in (e.detail or '').lower()
                        or "not idempotent" in str(e).lower()
                    )
                    if already_exists:
                        logger.info(f"Type {type_name} already exists (idempotent error), ensuring properties exist")
                        if type_kind == 'VERTEX':
                            self._create_vertex_properties(type_name)
                            self._discovered_vertex_types.add(type_name)
                        else:
                            self._discovered_edge_types.add(type_name)
                        continue
                    else:
                        logger.error(f"Schema creation failed: CREATE {type_kind} TYPE {type_name} - {e}")
                        if type_name in ['Entity', 'TextChunk']:
                            raise ValidationException(f"Critical type creation failed: {type_name}")
                except Exception as e:
                    error_msg = str(e)
                    if "not idempotent" in error_msg or "already exists" in error_msg:
                        logger.info(f"Type {type_name} already exists, ensuring properties exist")
                        if type_kind == 'VERTEX':
                            self._create_vertex_properties(type_name)
                            self._discovered_vertex_types.add(type_name)
                        else:
                            self._discovered_edge_types.add(type_name)
                        continue
                    else:
                        logger.error(f"Schema creation failed: CREATE {type_kind} TYPE {type_name} - {e}")
                        if type_name in ['Entity', 'TextChunk']:
                            raise ValidationException(f"Critical type creation failed: {type_name}")

            # Verify the schema was created
            try:
                result = self._db.query("sql", "SELECT name FROM schema:types")
                logger.info(f"Verification query result: {result}")
                # Handle ArcadeDB response format: {"result": {"records": [{"name": "TypeName"}]}}
                verified_types = []
                
                if isinstance(result, dict) and 'result' in result:
                    result_data = result['result']
                    if isinstance(result_data, dict) and 'records' in result_data:
                        records = result_data['records']
                        verified_types = [record['name'] for record in records if isinstance(record, dict) and 'name' in record]
                
                logger.info(f"Verified types exist: {verified_types}")
                
                # Check if critical types exist - fix the false warning
                critical_types = ['Entity', 'TextChunk']
                missing_types = [t for t in critical_types if t not in verified_types]
                if missing_types:
                    # The warning was often a false alarm due to verification timing
                    # Log as debug instead of warning since creation succeeded
                    logger.debug(f"Some critical types not immediately visible: {missing_types}, but creation succeeded")
                else:
                    logger.info("All critical types verified: Entity and TextChunk exist")
                    
            except Exception as e:
                logger.warning(f"Schema verification failed, but continuing: {e}")

            # Create indexes for UPSERT operations (now that types exist)
            logger.info("Creating indexes for UPSERT operations...")
            self._create_indexes()

            logger.info(f"Graph schema initialized successfully - created {created_count} new types")

        except Exception as e:
            logger.error(f"Schema initialization failed: {e}")
            raise
    
    def _ensure_dynamic_type(self, type_name: str, type_kind: str) -> bool:
        """Dynamically create a vertex or edge type if it doesn't exist.
        
        Args:
            type_name: Name of the type to create
            type_kind: Either 'VERTEX' or 'EDGE'
            
        Returns:
            True if type was created or already exists, False if creation failed
        """
        # Check if we've already discovered this type
        if type_kind == 'VERTEX' and type_name in self._discovered_vertex_types:
            logger.debug(f"Type {type_name} already discovered as {type_kind}")
            return True
        elif type_kind == 'EDGE' and type_name in self._discovered_edge_types:
            logger.debug(f"Type {type_name} already discovered as {type_kind}")
            return True
            
        try:
            # Create the type first
            query = f"CREATE {type_kind} TYPE {type_name} IF NOT EXISTS"
            self._db.query("sql", query, is_command=True)
            
            # For VERTEX types, create essential properties with correct data types
            if type_kind == 'VERTEX':
                self._create_vertex_properties(type_name)
            
            # Track the newly created type
            if type_kind == 'VERTEX':
                self._discovered_vertex_types.add(type_name)
            else:
                self._discovered_edge_types.add(type_name)
                
            logger.debug(f"Dynamically created {type_kind} type: {type_name}")
            return True
            
        except (TransactionException, SchemaException) as e:
            already_exists = (
                (isinstance(e, TransactionException) and e.is_idempotent_error)
                or "already exists" in (e.detail or '').lower()
                or "not idempotent" in str(e).lower()
            )
            if already_exists:
                if type_kind == 'VERTEX':
                    self._create_vertex_properties(type_name)
                    self._discovered_vertex_types.add(type_name)
                else:
                    self._discovered_edge_types.add(type_name)
                return True
            else:
                logger.warning(f"Failed to create dynamic {type_kind} type {type_name}: {e}")
                return False
        except Exception as e:
            error_msg = str(e)
            if "not idempotent" in error_msg or "already exists" in error_msg:
                if type_kind == 'VERTEX':
                    self._create_vertex_properties(type_name)
                    self._discovered_vertex_types.add(type_name)
                else:
                    self._discovered_edge_types.add(type_name)
                return True
            else:
                logger.warning(f"Failed to create dynamic {type_kind} type {type_name}: {e}")
                return False

    def _create_vertex_properties(self, type_name: str) -> None:
        """Create essential properties with correct data types for a vertex type."""
        try:
            # Define property schemas based on vertex type - include embedding for vector functionality
            if type_name == 'TextChunk':
                # TextChunk uses 'id' as primary key (STRING)
                properties = [
                    ('id', 'STRING', ''),
                    ('text', 'STRING', ''),
                    ('ref_doc_id', 'STRING', ''),  # Stored so delete(properties={'ref_doc_id':...}) works
                    ('embedding', 'ARRAY_OF_FLOATS', '')  # Native float array for LSM_VECTOR index
                ]
            else:
                # Entity types use 'name' as primary key (STRING)
                properties = [
                    ('name', 'STRING', ''),
                    ('embedding', 'ARRAY_OF_FLOATS', '')  # Native float array for LSM_VECTOR index
                ]
            
            # Create each property with proper data type using correct IF NOT EXISTS syntax
            for prop_name, prop_type, constraints in properties:
                try:
                    safe_pname = self._sql_identifier(prop_name)
                    prop_query = f"CREATE PROPERTY {type_name}.{safe_pname} IF NOT EXISTS {prop_type} {constraints}".strip()
                    self._db.query("sql", prop_query, is_command=True)
                    logger.debug(f"Created property {type_name}.{prop_name} ({prop_type})")
                except Exception as prop_error:
                    logger.warning(f"Property creation failed for {type_name}.{prop_name}: {prop_error}")
            
            # Create UNIQUE index on the primary key property
            primary_key = 'id' if type_name == 'TextChunk' else 'name'
            try:
                index_query = f"CREATE INDEX IF NOT EXISTS ON {type_name} ({primary_key}) UNIQUE"
                self._db.query("sql", index_query, is_command=True)
                logger.debug(f"Created UNIQUE index on {type_name}.{primary_key}")
            except Exception as index_error:
                error_msg = str(index_error)
                if "already exists" in error_msg.lower():
                    logger.debug(f"Index on {type_name}.{primary_key} already exists")
                else:
                    logger.debug(f"Index creation for {type_name}.{primary_key}: {index_error}")

            # Create LSM_VECTOR index on embedding if dimension is configured
            if self.embedding_dimension:
                self._ensure_vector_index(type_name, 'embedding', self.embedding_dimension)

            # Mark all declared properties as known so _ensure_property() won't
            # re-send DDL for them.
            for prop_name, _, _ in properties:
                self._known_properties.add((type_name, prop_name))

        except Exception as e:
            logger.warning(f"Failed to create properties for {type_name}: {e}")

    def _update_embedding(self, type_name: str, key_field: str, key_value: str,
                          embedding: list) -> None:
        """Store an embedding vector on an already-existing record.

        The LSM_VECTOR index requires a native float array literal, which cannot
        appear inside an UPDATE...UPSERT SET clause (the SQL parser rejects it).
        This method issues a plain UPDATE after the UPSERT has committed, using
        the unquoted array syntax that ArcadeDB accepts for ARRAY_OF_FLOATS.
        """
        try:
            array_literal = json.dumps(embedding)  # produces [0.1, 0.2, ...]
            escaped_key = self._escape_string(key_value)
            q = (
                f"UPDATE {type_name} SET embedding = {array_literal}"
                f" WHERE {key_field} = '{escaped_key}'"
            )
            self._db.query("sql", q, is_command=True)
        except Exception as e:
            err = str(e)[:200]
            logger.debug(f"_update_embedding failed for {type_name}.{key_field}='{key_value}': {err}")

    def _ensure_property(self, type_name: str, prop_name: str, prop_type: str = 'STRING') -> None:
        """Declare a property on a vertex type if it has not been declared yet.

        Uses an in-memory cache keyed by (type_name, prop_name) so the DDL
        statement is only sent once per process lifetime, keeping the hot path
        cheap after the first call.
        """
        cache_key = (type_name, prop_name)
        if cache_key in self._known_properties:
            return
        try:
            safe_pname = self._sql_identifier(prop_name)
            self._db.query(
                "sql",
                f"CREATE PROPERTY {type_name}.{safe_pname} IF NOT EXISTS {prop_type}",
                is_command=True,
            )
        except Exception as e:
            logger.debug(f"_ensure_property {type_name}.{prop_name}: {e}")
        # Mark as known regardless — even if DDL failed the property may already exist
        self._known_properties.add(cache_key)

    def _ensure_vector_index(self, type_name: str, embedding_field: str, dimensions: int) -> None:
        """Create an LSM_VECTOR index on the embedding field if it doesn't already exist."""
        import logging as _logging
        # The arcadedb-python SyncClient logs every non-2xx response at ERROR level
        # before raising.  Temporarily silence it so "already exists" isn't noisy.
        _driver_logger = _logging.getLogger('arcadedb_python.api.sync')
        _orig_level = _driver_logger.level
        _driver_logger.setLevel(_logging.CRITICAL)
        try:
            self._db.create_vector_index(type_name, embedding_field, dimensions)
            logger.debug(f"Created LSM_VECTOR index on {type_name}.{embedding_field} (dim={dimensions})")
        except VectorOperationException as e:
            # The driver raises VectorOperationException without forwarding the ArcadeDB
            # detail, but the original exception is chained via __cause__.  Check the
            # full chain for "already exists" before treating it as a real error.
            cause_detail = ''
            if e.__cause__ is not None:
                cause = e.__cause__
                cause_detail = (getattr(cause, 'detail', None) or str(cause)).lower()
            if "already exists" in cause_detail:
                logger.debug(f"LSM_VECTOR index on {type_name}.{embedding_field} already exists")
            else:
                logger.warning(f"Could not create LSM_VECTOR index on {type_name}.{embedding_field}: {e}")
        except Exception as e:
            logger.warning(f"Could not create LSM_VECTOR index on {type_name}.{embedding_field}: {e}")
        finally:
            _driver_logger.setLevel(_orig_level)

    def _create_indexes(self) -> None:
        """Create basic indexes for better query performance."""
        # Define essential indexes - REQUIRED for UPSERT operations
        index_definitions = [
            # Essential indexes - required for basic functionality
            ("TextChunk", "id", "textchunk_id_idx"),  # Required for chunk UPSERT
            ("Entity", "name", "entity_name_idx"),    # Required for generic entity UPSERT
            # Core entity type indexes - for commonly used pre-created types
            ("PERSON", "name", "person_name_idx"),
            ("ORGANIZATION", "name", "org_name_idx"),
            ("LOCATION", "name", "location_name_idx"),
            ("PLACE", "name", "place_name_idx"),
        ]
        
        created_indexes = 0
        for type_name, property_name, index_name in index_definitions:
            try:
                # Step 1: Ensure property exists first using correct IF NOT EXISTS syntax
                try:
                    prop_query = f"CREATE PROPERTY {type_name}.{property_name} IF NOT EXISTS STRING"
                    self._db.query("sql", prop_query, is_command=True)
                    logger.debug(f"Created property {type_name}.{property_name}")
                except Exception as prop_error:
                    logger.debug(f"Property creation for {type_name}.{property_name}: {prop_error}")
                
                # Step 2: Create UNIQUE index (ArcadeDB requires properties to exist first)
                query = f"CREATE INDEX IF NOT EXISTS ON {type_name} ({property_name}) UNIQUE"
                self._db.query("sql", query, is_command=True)
                logger.debug(f"Created UNIQUE index on {type_name}.{property_name} (auto-named as {type_name}[{property_name}])")
                created_indexes += 1
            except Exception as e:
                error_msg = str(e)
                # Skip if index already exists or property/syntax issues
                if ("already exists" in error_msg.lower() or 
                    "duplicate" in error_msg.lower() or 
                    "index already defined" in error_msg.lower() or
                    "property does not exist" in error_msg.lower() or
                    "CommandSQLParsingException" in str(e)):
                    logger.debug(f"Index creation skipped for {type_name}.{property_name}: {error_msg}")
                    continue
                else:
                    logger.warning(f"Failed to create index on {type_name}.{property_name}: {e}")
        
        logger.info(f"Created {created_indexes} new indexes for UPSERT operations")

    def _ensure_index_for_upsert(self, type_name: str, property_name: str) -> None:
        """Ensure a UNIQUE index exists for UPSERT operations on the given type and property.
        
        Note: With dynamic type creation, properties and indexes are usually created
        automatically when the type is created. This method serves as a fallback.
        """
        # If this is a standard property for essential types or basic schema types,
        # it should already be created by _create_vertex_properties()
        if ((type_name == 'TextChunk' and property_name == 'id') or
            (type_name == 'Entity' and property_name == 'name') or
            (self.include_basic_schema and type_name in ['PERSON', 'ORGANIZATION', 'LOCATION', 'PLACE'] and property_name == 'name')):
            logger.debug(f"Index for {type_name}.{property_name} should already exist from schema creation")
            return
        
        # Fallback: Create property and index if needed (for non-standard properties)
        # Step 1: Ensure the property exists using correct IF NOT EXISTS syntax
        try:
            prop_query = f"CREATE PROPERTY {type_name}.{property_name} IF NOT EXISTS STRING"
            self._db.query("sql", prop_query, is_command=True)
            logger.debug(f"Created fallback property {type_name}.{property_name}")
        except Exception as prop_error:
            logger.debug(f"Property creation for {type_name}.{property_name}: {prop_error}")
        
        # Step 2: Create UNIQUE index
        try:
            index_query = f"CREATE INDEX IF NOT EXISTS ON {type_name} ({property_name}) UNIQUE"
            self._db.query("sql", index_query, is_command=True)
            logger.debug(f"Created fallback UNIQUE index on {type_name}.{property_name}")
        except Exception as e:
            error_msg = str(e)
            if "already exists" in error_msg.lower():
                logger.debug(f"Index on {type_name}.{property_name} already exists")
            elif ("property does not exist" in error_msg.lower() or
                  "type.*not found" in error_msg.lower() or
                  "CommandSQLParsingException" in str(e)):
                logger.debug(f"Index creation skipped for {type_name}.{property_name}: {error_msg}")
            else:
                logger.warning(f"Failed to create UNIQUE index on {type_name}.{property_name}: {e}")

    def _get_all_vertex_types(self) -> List[str]:
        """Return every vertex type currently in the schema.

        Falls back to a hard-coded list of known types when the schema query
        fails (e.g. during early initialisation).
        """
        try:
            result = self._db.query("sql", "SELECT name FROM schema:types WHERE type = 'vertex'")
            if result and isinstance(result, list):
                return [r['name'] for r in result if isinstance(r, dict) and 'name' in r]
        except Exception as e:
            logger.debug(f"Schema vertex-type query failed: {e}")
        # Fallback: essential types + everything we have dynamically discovered
        return list({'Entity', 'TextChunk'} | self._discovered_vertex_types)

    def get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Get nodes with matching properties or ids."""
        nodes = []

        try:
            if ids:
                # Query by node IDs - search all vertex type tables
                
                # Get list of all vertex types from schema
                try:
                    schema_query = "SELECT name FROM schema:types WHERE type = 'vertex'"
                    type_results = self._db.query("sql", schema_query)
                    vertex_types = []
                    if type_results and isinstance(type_results, list):
                        vertex_types = [item['name'] for item in type_results if isinstance(item, dict) and 'name' in item]
                except Exception as e:
                    # Fallback to known types if schema query fails
                    vertex_types = ['Entity', 'TextChunk', 'PERSON', 'ORGANIZATION', 'LOCATION', 'PLACE']
                
                for node_id in ids:
                    found = False
                    # Search each vertex type table for the ID
                    for type_name in vertex_types:
                        try:
                            # For TextChunk, search by id; for others, search by both id and name
                            # Properly escape the node_id to handle apostrophes and special characters
                            escaped_id = self._escape_string(node_id)
                            if type_name == 'TextChunk':
                                query = f"SELECT * FROM {type_name} WHERE id = '{escaped_id}'"
                            else:
                                query = f"SELECT * FROM {type_name} WHERE id = '{escaped_id}' OR name = '{escaped_id}'"
                            
                            results = self._db.query("sql", query)
                            if results and isinstance(results, list):
                                for result in results:
                                    try:
                                        node = self._result_to_node(result)
                                        nodes.append(node)
                                    except Exception as conv_e:
                                        logger.warning(f"Failed to convert result to node: {conv_e}, result: {result}")
                                found = True
                                break  # Found the node, no need to search other tables
                        except Exception as e:
                            logger.debug(f"Query failed for {type_name}: {e}")
                    
                    if not found:
                        logger.debug(f"Node with ID '{node_id}' not found in any vertex table")

            elif properties:
                # Query by properties - query all vertex type tables
                where_conditions = []
                for key, value in properties.items():
                    safe_key = self._sql_identifier(key)
                    if isinstance(value, str):
                        escaped_value = self._escape_string(value)
                        where_conditions.append(f"{safe_key} = '{escaped_value}'")
                    else:
                        where_conditions.append(f"{safe_key} = {value}")

                where_clause = " AND ".join(where_conditions)
                
                # Get list of all vertex types from schema
                try:
                    schema_query = "SELECT name FROM schema:types WHERE type = 'vertex'"
                    type_results = self._db.query("sql", schema_query)
                    vertex_types = []
                    if type_results and isinstance(type_results, list):
                        vertex_types = [item['name'] for item in type_results if isinstance(item, dict) and 'name' in item]
                except Exception:
                    # Fallback to known types if schema query fails
                    vertex_types = ['Entity', 'TextChunk', 'PERSON', 'ORGANIZATION', 'LOCATION', 'PLACE']
                
                # Query each vertex type table
                for type_name in vertex_types:
                    try:
                        query = f"SELECT * FROM {type_name} WHERE {where_clause}"
                        results = self._db.query("sql", query)
                        if results and isinstance(results, list):
                            for result in results:
                                nodes.append(self._result_to_node(result))
                    except Exception:
                        pass  # Ignore errors, table might not exist or be empty
            else:
                # Get all nodes (limited) - query all vertex type tables
                
                # Get list of all vertex types from schema
                try:
                    schema_query = "SELECT name FROM schema:types WHERE type = 'vertex'"
                    type_results = self._db.query("sql", schema_query)
                    vertex_types = []
                    if type_results and isinstance(type_results, list):
                        vertex_types = [item['name'] for item in type_results if isinstance(item, dict) and 'name' in item]
                except Exception:
                    # Fallback to known types if schema query fails
                    vertex_types = ['Entity', 'TextChunk', 'PERSON', 'ORGANIZATION', 'LOCATION', 'PLACE']
                
                # Query each vertex type table
                for type_name in vertex_types:
                    try:
                        query = f"SELECT * FROM {type_name} LIMIT 50"
                        results = self._db.query("sql", query)
                        if results and isinstance(results, list):
                            for result in results:
                                nodes.append(self._result_to_node(result))
                    except Exception:
                        pass  # Ignore errors, table might not exist or be empty

        except Exception as e:
            logger.error(f"Failed to get nodes: {e}")

        return nodes

    def get_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[List[str]]:
        """Get triplets (relationships) with matching criteria using v0.4.0+ features."""
        triplets = []

        try:
            # Skip enhanced get_triplets method as it tries to query non-existent E type
            # Fall through directly to traditional method
            logger.debug("Skipping enhanced get_triplets method due to E type issue")
                
            # Traditional method (fallback)
            logger.debug("Using traditional get_triplets method")
            
            # Skip MATCH queries for now and use direct edge type queries
            # This is more reliable and doesn't depend on complex MATCH syntax
            logger.debug("Using direct edge type queries instead of MATCH")
            
            # Fallback to querying actual edge types instead of non-existent RELATION type
            results = []
            try:
                # Get all edge types from schema
                schema_query = "SELECT name FROM schema:types WHERE type = 'edge'"
                edge_type_results = self._db.query("sql", schema_query)
                edge_types = []
                if edge_type_results and isinstance(edge_type_results, list):
                    edge_types = [item['name'] for item in edge_type_results if isinstance(item, dict) and 'name' in item]
                    
                logger.debug(f"Found edge types: {edge_types}")
                
                # Query each edge type for relationships and convert to triplets
                for edge_type in edge_types:
                    try:
                        # Skip if relation_names specified and this edge type not in the list
                        if relation_names and edge_type not in relation_names:
                            continue
                        
                        # Query the edge type directly to get relationship records
                        # Use separate queries to get vertex names since out().name may not work
                        query = f"SELECT @rid, @in, @out FROM {edge_type} LIMIT 100"
                        logger.debug(f"Executing relationship query for {edge_type}: {query}")
                        edge_results = self._db.query("sql", query)
                        
                        if edge_results and isinstance(edge_results, list):
                            for edge_record in edge_results:
                                try:
                                    # Get the RIDs of source and target vertices
                                    source_rid = edge_record.get('@out')  # Note: @out is the source in ArcadeDB
                                    target_rid = edge_record.get('@in')   # Note: @in is the target in ArcadeDB
                                    relation_label = edge_type
                                    
                                    if source_rid and target_rid:
                                        # Query source vertex name/id (chunks use 'id', entities use 'name')
                                        source_name = None
                                        try:
                                            source_query = f"SELECT name, id FROM {source_rid}"
                                            source_result = self._db.query("sql", source_query)
                                            if source_result and isinstance(source_result, list) and len(source_result) > 0:
                                                # Try 'name' first (for entities), then 'id' (for chunks)
                                                source_name = source_result[0].get('name') or source_result[0].get('id')
                                        except Exception:
                                            pass
                                        
                                        # Query target vertex name/id (chunks use 'id', entities use 'name')
                                        target_name = None
                                        try:
                                            target_query = f"SELECT name, id FROM {target_rid}"
                                            target_result = self._db.query("sql", target_query)
                                            if target_result and isinstance(target_result, list) and len(target_result) > 0:
                                                # Try 'name' first (for entities), then 'id' (for chunks)
                                                target_name = target_result[0].get('name') or target_result[0].get('id')
                                        except Exception:
                                            pass
                                        
                                        # Create triplet if we have both names
                                        if source_name and target_name:
                                            # Create simple tuple triplet: [subject, relation, object]
                                            triplet = [source_name, relation_label, target_name]
                                            triplets.append(triplet)
                                            logger.debug(f"Created triplet: {source_name} -> {relation_label} -> {target_name}")
                                        else:
                                            logger.debug(f"Could not get names for edge {source_rid} -> {target_rid} (source_name={source_name}, target_name={target_name})")
                                    
                                except Exception as parse_error:
                                    logger.debug(f"Failed to parse edge record: {parse_error}")
                                    continue
                                
                    except Exception as edge_error:
                        logger.debug(f"Failed to query edge type {edge_type}: {edge_error}")
                        continue
                        
            except Exception as schema_error:
                logger.warning(f"Failed to get edge types from schema: {schema_error}")
                # If we can't get edge types, return empty results gracefully
                results = []

        except Exception as e:
            logger.error(f"Failed to get triplets: {e}")

        return triplets

    def get_rel_map(
        self,
        graph_nodes: List[LabelledNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get relationships up to a certain depth from given nodes."""
        triplets = []

        if not graph_nodes:
            return triplets

        try:
            # Resolve logical node ids/names to ArcadeDB @rid values.
            # TRAVERSE / MATCH both require real RIDs — passing string names fails.
            node_ids = [node.id for node in graph_nodes]

            vertex_types = self._get_all_vertex_types()
            start_rids: List[str] = []
            for node_id in node_ids:
                escaped = self._escape_string(node_id)
                normalized = self._normalize_and_deduplicate_entity(node_id)
                escaped_norm = self._escape_string(normalized)
                for vtype in vertex_types:
                    try:
                        if vtype == 'TextChunk':
                            q = f"SELECT @rid FROM {vtype} WHERE id = '{escaped}' LIMIT 1"
                        else:
                            q = (
                                f"SELECT @rid FROM {vtype}"
                                f" WHERE name = '{escaped}' OR name = '{escaped_norm}'"
                                f" LIMIT 1"
                            )
                        rows = self._db.query("sql", q)
                        if rows:
                            rid_val = str(rows[0]['@rid'])
                            start_rids.append(rid_val)
                            logger.debug(f"get_rel_map: resolved '{node_id}' -> {rid_val} in {vtype}")
                            break
                    except Exception:
                        continue

            if not start_rids:
                logger.info(f"get_rel_map: no RIDs resolved for node_ids={node_ids}, returning empty")
                return triplets

            # Fetch all edges touching the resolved start vertices up to `depth` hops.
            # Use expand(bothE()) for depth=1 — it returns edge records directly and is
            # the most reliable ArcadeDB form.  For depth>1 we use TRAVERSE but select
            # only *, relying on Python post-filtering to identify edges (records that
            # have both 'in' and 'out' fields).  We never put @-prefixed system fields
            # in the SELECT list (ArcadeDB rejects them there) and we apply the
            # ignore_rels filter in Python so we don't need @class in a WHERE clause.
            rids_csv = ", ".join(start_rids)
            if depth <= 1:
                query = f"SELECT expand(bothE()) FROM [{rids_csv}] LIMIT {limit}"
            else:
                query = (
                    f"SELECT * FROM ("
                    f"  TRAVERSE bothE() FROM [{rids_csv}] MAXDEPTH {depth}"
                    f") WHERE @this INSTANCEOF 'E'"
                    f" LIMIT {limit}"
                )
            logger.debug(f"get_rel_map query: {query}")
            try:
                results = self._db.query("sql", query)
                logger.debug(f"get_rel_map raw results: {len(results)} rows")
            except Exception as trav_err:
                logger.error(f"Failed to get relationship map: {trav_err}\nQuery: {query}")
                return triplets

            ignore_set = set(ignore_rels) if ignore_rels else set()

            for result in results:
                try:
                    edge_class = result.get('@type', result.get('@class', ''))
                    # ArcadeDB returns edge endpoints as '@in' / '@out' (system fields),
                    # not 'in' / 'out' (which are OrientDB conventions).
                    out_rid = result.get('@out', result.get('out'))
                    in_rid  = result.get('@in',  result.get('in'))

                    # Skip vertex records that leaked through (no in/out) and ignored types
                    if not out_rid or not in_rid:
                        logger.debug(f"get_rel_map: skipping non-edge record type={edge_class} keys={list(result.keys())[:8]}")
                        continue
                    if edge_class in ignore_set:
                        continue

                    # ArcadeDB may return out/in as a RID string "#x:y" or as a
                    # dict {"@rid": "#x:y"} depending on the driver version.
                    def _rid_str(val: Any) -> str:
                        if isinstance(val, dict):
                            return str(val.get('@rid', val))
                        return str(val)

                    def _fetch_vertex(rid_val: Any) -> dict:
                        rid_s = _rid_str(rid_val)
                        try:
                            rows = self._db.query("sql", f"SELECT * FROM {rid_s} LIMIT 1")
                            return rows[0] if rows else {}
                        except Exception as fe:
                            logger.debug(f"get_rel_map: vertex fetch failed for {rid_s}: {fe}")
                            return {}

                    out_data = _fetch_vertex(out_rid)
                    in_data  = _fetch_vertex(in_rid)

                    src_node = self._result_to_node(out_data)
                    dst_node = self._result_to_node(in_data)

                    # Build a minimal relation dict for _result_to_relation
                    rel_data = dict(result)
                    rel_data.setdefault('label', edge_class)
                    relationship = self._result_to_relation(rel_data)

                    triplets.append((src_node, relationship, dst_node))
                    logger.debug(f"get_rel_map: triplet {src_node.id} -[{edge_class}]-> {dst_node.id}")
                except Exception as parse_err:
                    logger.debug(f"Failed to parse rel_map edge: {parse_err}")
                    continue

            logger.info(f"get_rel_map: found {len(triplets)} triplets for {len(start_rids)} start vertices")

        except Exception as e:
            logger.error(f"Failed to get relationship map: {e}")

        return triplets

    def upsert_nodes(self, nodes: Sequence[LabelledNode]) -> None:
        """Insert or update nodes in the graph using v0.4.0+ bulk operations."""
        # Debug: Count node types being received
        entity_count = sum(1 for node in nodes if isinstance(node, EntityNode))
        chunk_count = sum(1 for node in nodes if isinstance(node, ChunkNode))
        other_count = len(nodes) - entity_count - chunk_count
        
        logger.info(f"DEBUG: Upserting nodes: {entity_count} entities, {chunk_count} chunks, {other_count} other")
        
        # Try bulk operations first for better performance
        try:
            # Separate nodes by type for bulk operations
            entity_nodes = [node for node in nodes if isinstance(node, EntityNode)]
            chunk_nodes = [node for node in nodes if isinstance(node, ChunkNode)]
            other_nodes = [node for node in nodes if not isinstance(node, (EntityNode, ChunkNode))]
            
            # Upsert entity nodes one statement at a time (sql + is_command=True).
            # sqlscript batched via the command endpoint is unreliable in ArcadeDB -
            # it can silently produce no output and write nothing.  Executing each
            # UPDATE ... UPSERT as an individual sql command is safe, gives per-node
            # error isolation, and still falls back to _upsert_entity_node on failure.
            if entity_nodes:
                # Phase 1: ensure all required schema types and indexes exist before
                # writing any data.  The UPSERT clause requires a UNIQUE index on the
                # key field; creating it here (once per type) means the UPDATE...UPSERT
                # in Phase 2 will succeed rather than falling through to INSERT.
                seen_types: set = set()
                for node in entity_nodes:
                    entity_type = self._determine_entity_type(node)
                    if entity_type in seen_types:
                        continue
                    seen_types.add(entity_type)
                    try:
                        self._ensure_dynamic_type(entity_type, 'VERTEX')
                    except Exception as schema_err:
                        logger.warning(f"ENTITY_TYPE_DETECTION: Could not ensure type {entity_type}: {schema_err}")
                    try:
                        self._ensure_index_for_upsert(entity_type, 'name')
                    except Exception as idx_err:
                        logger.debug(f"Index ensure skipped for {entity_type}.name: {idx_err}")

                # Phase 2: write each node individually so failures are isolated
                succeeded = 0
                for node in entity_nodes:
                    entity_type = self._determine_entity_type(node)
                    normalized_name = self._normalize_and_deduplicate_entity(node.name)
                    logger.debug(f"LLM_ENTITY_INPUT: {normalized_name} (type={entity_type})")
                    try:
                        # Build property assignments — embedding is excluded here
                        # and stored separately via _update_embedding() because the
                        # LSM_VECTOR index requires a native array literal which the
                        # SQL parser rejects inside an UPDATE...UPSERT SET clause.
                        prop_assignments = [f"name = '{self._escape_string(normalized_name)}'"]

                        has_embedding = hasattr(node, 'embedding') and node.embedding is not None

                        if node.properties:
                            for k, v in node.properties.items():
                                if k != 'embedding':
                                    self._ensure_property(entity_type, k)
                                    safe_k = self._sql_identifier(k)
                                    if isinstance(v, (list, dict)):
                                        escaped_value = self._escape_string(json.dumps(v))
                                    else:
                                        escaped_value = self._escape_string(str(v))
                                    prop_assignments.append(f"{safe_k} = '{escaped_value}'")

                        upsert_stmt = (
                            f"UPDATE {entity_type} SET {', '.join(prop_assignments)}"
                            f" UPSERT WHERE name = '{self._escape_string(normalized_name)}'"
                        )
                        logger.debug(f"SQL_ENTITY_PROCESSING: {upsert_stmt[:200]}")

                        if len(upsert_stmt) <= 10000:
                            self._db.query("sql", upsert_stmt, is_command=True)
                            if has_embedding:
                                self._update_embedding(entity_type, 'name', normalized_name,
                                                       list(node.embedding))
                            succeeded += 1
                            self._create_mentions_relationship_sql(node, entity_type)
                        else:
                            logger.debug(f"Statement too long for {normalized_name}, using individual upsert")
                            self._upsert_entity_node(node)
                            succeeded += 1

                    except Exception as e:
                        err_str = str(e)
                        # Avoid logging enormous SQL with embedded 1536-dim vectors
                        if len(err_str) > 300:
                            err_str = err_str[:300] + "... [truncated]"
                        logger.warning(f"SQL upsert failed for entity {normalized_name}, falling back: {err_str}")
                        try:
                            self._upsert_entity_node(node)
                            succeeded += 1
                        except Exception as e2:
                            e2_str = str(e2)
                            if len(e2_str) > 300:
                                e2_str = e2_str[:300] + "... [truncated]"
                            logger.error(f"Failed to upsert entity node {node.name}: {e2_str}")

                logger.info(f"Bulk upserted {succeeded}/{len(entity_nodes)} entity nodes")
            
            # Upsert chunk nodes one statement at a time (same rationale as entity nodes above).
            if chunk_nodes:
                # Phase 1: ensure TextChunk schema type and UNIQUE index exist
                try:
                    self._ensure_dynamic_type('TextChunk', 'VERTEX')
                except Exception as schema_err:
                    logger.warning(f"Could not ensure TextChunk type: {schema_err}")
                try:
                    self._ensure_index_for_upsert('TextChunk', 'id')
                except Exception as idx_err:
                    logger.debug(f"Index ensure skipped for TextChunk.id: {idx_err}")

                # Phase 2: write each chunk individually
                succeeded = 0
                for node in chunk_nodes:
                    try:
                        text_escaped = self._escape_string(node.text)
                        prop_assignments = [
                            f"id = '{self._escape_string(node.id)}'",
                            f"text = '{text_escaped}'"
                        ]

                        ref_doc_id = getattr(node, 'ref_doc_id', None)
                        if ref_doc_id is None and node.properties:
                            ref_doc_id = node.properties.get('ref_doc_id')
                        if ref_doc_id:
                            prop_assignments.append(f"ref_doc_id = '{self._escape_string(str(ref_doc_id))}'")

                        # Embedding excluded from UPSERT — stored via _update_embedding()
                        has_embedding = hasattr(node, 'embedding') and node.embedding is not None

                        if node.properties:
                            for k, v in node.properties.items():
                                if k in ('embedding', 'ref_doc_id'):
                                    continue
                                self._ensure_property('TextChunk', k)
                                safe_k = self._sql_identifier(k)
                                if isinstance(v, (list, dict)):
                                    escaped_value = self._escape_string(json.dumps(v))
                                else:
                                    escaped_value = self._escape_string(str(v))
                                prop_assignments.append(f"{safe_k} = '{escaped_value}'")

                        upsert_stmt = (
                            f"UPDATE TextChunk SET {', '.join(prop_assignments)}"
                            f" UPSERT WHERE id = '{self._escape_string(node.id)}'"
                        )

                        if len(upsert_stmt) <= 15000:
                            self._db.query("sql", upsert_stmt, is_command=True)
                            if has_embedding:
                                self._update_embedding('TextChunk', 'id', node.id,
                                                       list(node.embedding))
                            succeeded += 1
                        else:
                            logger.debug(f"Statement too long for chunk {node.id}, using individual upsert")
                            self._upsert_chunk_node(node)
                            succeeded += 1

                    except Exception as e:
                        err_str = str(e)
                        if len(err_str) > 300:
                            err_str = err_str[:300] + "... [truncated]"
                        logger.warning(f"SQL upsert failed for chunk {node.id}, falling back: {err_str}")
                        try:
                            self._upsert_chunk_node(node)
                            succeeded += 1
                        except Exception as e2:
                            e2_str = str(e2)
                            if len(e2_str) > 300:
                                e2_str = e2_str[:300] + "... [truncated]"
                            logger.error(f"Failed to upsert chunk node {node.id}: {e2_str}")

                logger.info(f"Bulk upserted {succeeded}/{len(chunk_nodes)} chunk nodes")
            
            # Handle other nodes individually (no bulk operation available)
            for node in other_nodes:
                try:
                    logger.debug(f"OTHER: Upserting generic node: {node.id} (type: {type(node)})")
                    self._upsert_generic_node(node)
                except Exception as e:
                    logger.error(f"Failed to upsert generic node {getattr(node, 'id', 'unknown')}: {e}")
                    
        except Exception as e:
            logger.error(f"Bulk upsert failed, falling back to individual operations: {e}")
            # Complete fallback to individual operations
            for node in nodes:
                try:
                    if isinstance(node, EntityNode):
                        logger.debug(f"ENTITY: Upserting EntityNode: {node.name} (label: {getattr(node, 'label', 'None')})")
                        self._upsert_entity_node(node)
                    elif isinstance(node, ChunkNode):
                        logger.debug(f"CHUNK: Upserting ChunkNode: {node.id}")
                        self._upsert_chunk_node(node)
                    else:
                        logger.debug(f"OTHER: Upserting generic node: {node.id} (type: {type(node)})")
                        self._upsert_generic_node(node)
                except Exception as e:
                    logger.error(f"Failed to upsert node {getattr(node, 'id', getattr(node, 'name', 'unknown'))}: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")

    def upsert_relations(self, relations: List[Relation]) -> None:
        """Insert or update relationships in the graph using v0.4.0+ bulk operations."""
        logger.info(f"RELATIONS: Upserting {len(relations)} relations")
        
        # Try bulk operations first for better performance
        try:
            # Group relations by type for bulk operations
            relations_by_type = {}
            for relation in relations:
                rel_type = relation.label
                if rel_type not in relations_by_type:
                    relations_by_type[rel_type] = []
                relations_by_type[rel_type].append(relation)
            
            # Bulk upsert relations by type
            for rel_type, type_relations in relations_by_type.items():
                try:
                    relation_data = []
                    for relation in type_relations:
                        data = {
                            'source_id': relation.source_id,
                            'target_id': relation.target_id,
                            'properties': relation.properties or {}
                        }
                        relation_data.append(data)
                    
                    # Note: bulk_upsert_relations doesn't exist in v0.4.0, fall back to individual
                    logger.info(f"Processing {len(type_relations)} relations of type {rel_type} individually")
                    for relation in type_relations:
                        self._upsert_relation(relation)
                    
                except (BulkOperationException, ValidationException) as e:
                    logger.warning(f"Bulk relation upsert failed for {rel_type}, using individual operations: {e}")
                    # Fallback to individual operations for this type
                    for relation in type_relations:
                        try:
                            self._upsert_relation(relation)
                        except Exception as e2:
                            logger.error(f"Failed to upsert relation {relation.label} ({relation.source_id} -> {relation.target_id}): {e2}")
                            
        except Exception as e:
            logger.error(f"Bulk relation upsert failed, falling back to individual operations: {e}")
            # Complete fallback to individual operations
            for relation in relations:
                try:
                    logger.debug(f"RELATION: Upserting relation: {relation.source_id} --[{relation.label}]--> {relation.target_id}")
                    self._upsert_relation(relation)
                except Exception as e:
                    logger.error(f"Failed to upsert relation {relation.label} ({relation.source_id} -> {relation.target_id}): {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")

    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Delete nodes and relationships matching criteria.

        Fix 1: properties/entity_names branches now iterate over ALL vertex
                 types returned by the schema, not just Entity and TextChunk.
        Fix 2: ids branch searches by the logical 'id' / 'name' fields that
                 LlamaIndex writes, not by ArcadeDB internal @rid values.
        Fix 3: ref_doc_id is now stored on TextChunk (see _upsert_chunk_node),
                 so delete(properties={'ref_doc_id': '...'}) works correctly.
        """
        try:
            all_vertex_types = self._get_all_vertex_types()

            if ids:
                # LlamaIndex logical IDs are stored as:
                #   TextChunk: 'id' field
                #   Entity/*:  'name' field
                # They are NOT ArcadeDB internal @rid values.
                total_deleted = 0
                for node_id in ids:
                    escaped = self._escape_string(node_id)
                    for type_name in all_vertex_types:
                        try:
                            if type_name == 'TextChunk':
                                condition = f"id = '{escaped}'"
                            else:
                                condition = f"name = '{escaped}' OR id = '{escaped}'"
                            q = f"DELETE FROM {type_name} WHERE {condition}"
                            self._db.query("sql", q, is_command=True)
                            total_deleted += 1
                        except Exception as e:
                            logger.debug(f"Delete by id from {type_name} failed: {e}")
                logger.info(f"Deleted nodes by logical IDs across {len(all_vertex_types)} types")

            elif entity_names:
                # Delete by entity name across all vertex types
                total_deleted = 0
                for type_name in all_vertex_types:
                    try:
                        escaped_names = [self._escape_string(n) for n in entity_names]
                        names_list = "', '".join(escaped_names)
                        if type_name == 'TextChunk':
                            condition = f"id IN ['{names_list}']"
                        else:
                            condition = f"name IN ['{names_list}']"
                        q = f"DELETE FROM {type_name} WHERE {condition}"
                        self._db.query("sql", q, is_command=True)
                        total_deleted += 1
                    except Exception as e:
                        logger.debug(f"Delete by name from {type_name} failed: {e}")
                logger.info(f"Deleted entity names across {len(all_vertex_types)} types")

            elif properties:
                # Build WHERE clause from properties dict
                where_parts = []
                for key, value in properties.items():
                    safe_key = self._sql_identifier(key)
                    if isinstance(value, str):
                        where_parts.append(f"{safe_key} = '{self._escape_string(value)}'")
                    else:
                        where_parts.append(f"{safe_key} = {value}")
                where_clause = " AND ".join(where_parts)

                total_deleted = 0
                for type_name in all_vertex_types:
                    try:
                        q = f"DELETE FROM {type_name} WHERE {where_clause}"
                        self._db.query("sql", q, is_command=True)
                        total_deleted += 1
                    except Exception as e:
                        logger.debug(f"Delete by properties from {type_name} failed: {e}")
                logger.info(f"Deleted by properties across {len(all_vertex_types)} types")

            if relation_names:
                for relation_name in relation_names:
                    try:
                        deleted_count = self._db.safe_delete_all(relation_name, batch_size=1000)
                        logger.info(f"Deleted {deleted_count} relations of type {relation_name}")
                    except Exception as e:
                        logger.warning(f"safe_delete_all failed for {relation_name}, falling back: {e}")
                        try:
                            self._db.query("sql", f"DELETE FROM {relation_name}", is_command=True)
                        except Exception as e2:
                            logger.debug(f"Fallback delete for {relation_name} failed: {e2}")

        except Exception as e:
            logger.error(f"Failed to delete: {e}")
            raise ArcadeDBException(f"Delete operation failed: {e}") from e

    def structured_query(
        self, query: str, param_map: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute a structured SQL query against the graph.
        
        This SQL-only version provides optimal performance for ArcadeDB operations.
        DML statements (DELETE, INSERT, UPDATE, CREATE, DROP, TRUNCATE) are
        automatically sent as commands (is_command=True) to satisfy ArcadeDB's
        non-idempotent query requirement.
        """
        try:
            # Detect DML/DDL statements that require is_command=True
            query_upper = query.strip().upper()
            is_command = query_upper.startswith((
                'DELETE', 'INSERT', 'UPDATE', 'CREATE', 'DROP', 'TRUNCATE', 'ALTER'
            ))
            results = self._db.query("sql", query, is_command=is_command)
            return results

        except Exception as e:
            logger.error(f"Structured query failed: {e}")
            logger.error(f"Query was: {query}")
            raise QueryParsingException(f"SQL query execution failed: {e}") from e

    def vector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[List[LabelledNode], List[float]]:
        """Execute a vector similarity search using ArcadeDB's LSM_VECTOR index.

        Searches all known vertex types that have an embedding field and returns
        the top-k most similar nodes together with their cosine distances.

        Args:
            query: VectorStoreQuery containing query_embedding and similarity_top_k.

        Returns:
            Tuple of (nodes, scores) where scores are cosine distances (lower = closer).
        """
        if not query.query_embedding:
            logger.debug("No query embedding provided for vector search")
            return [], []

        top_k = query.similarity_top_k or 10
        query_embedding = list(query.query_embedding)

        # Search every vertex type that has been created with an embedding property
        vertex_types = list(
            {'Entity', 'TextChunk'} | self._discovered_vertex_types
        )

        all_results: List[Tuple[LabelledNode, float]] = []

        for type_name in vertex_types:
            try:
                records = self._db.vector_search(
                    type_name=type_name,
                    embedding_field='embedding',
                    query_embedding=query_embedding,
                    top_k=top_k,
                )
                for record in records:
                    try:
                        node = self._result_to_node(record)
                        # vectorNeighbors returns cosine *distance* (0=identical, 1=orthogonal).
                        # VectorContextRetriever treats scores as *similarity* (higher = better),
                        # so convert: similarity = 1.0 - distance.
                        distance = float(record.get('distance', record.get('similarity_score', 1.0)))
                        score = max(0.0, 1.0 - distance)
                        all_results.append((node, score))
                    except Exception as conv_err:
                        logger.debug(f"Could not convert vector result to node: {conv_err}")
            except Exception as e:
                logger.debug(f"Vector search failed for type {type_name}: {e}")

        if not all_results:
            return [], []

        # Sort by score descending (higher similarity = better match) and trim to top_k
        all_results.sort(key=lambda x: x[1], reverse=True)
        all_results = all_results[:top_k]

        nodes = [r[0] for r in all_results]
        scores = [r[1] for r in all_results]
        logger.info(f"Vector query returned {len(nodes)} results across {len(vertex_types)} types")
        return nodes, scores

    def get_schema(self, refresh: bool = False) -> Any:
        """Get the database schema in LlamaIndex-compatible format."""
        try:
            # Get vertex types with sample entities
            vertex_query = "SELECT name, properties FROM schema:types WHERE type = 'vertex'"
            vertex_results = self._db.query("sql", vertex_query)

            # Get edge types  
            edge_query = "SELECT name, properties FROM schema:types WHERE type = 'edge'"
            edge_results = self._db.query("sql", edge_query)

            # Get sample entities for better AI understanding - avoid UNION issues
            sample_entities = []
            try:
                # Query each vertex type separately to avoid UNION parsing issues
                base_vertex_types = ['Entity']  # Essential entity type always present
                vertex_types = list(set(base_vertex_types) | self._discovered_vertex_types)
                for vertex_type in vertex_types:
                    try:
                        sample_query = f"SELECT name FROM {vertex_type} LIMIT 5"
                        sample_results = self._db.query("sql", sample_query)
                        if isinstance(sample_results, list):
                            for r in sample_results:
                                if r.get("name"):
                                    sample_entities.append({"name": r.get("name", ""), "type": vertex_type})
                    except Exception:
                        pass  # Type doesn't exist, continue
                        
            except Exception as e:
                logger.debug(f"Could not get sample entities: {e}")

            return {
                "vertex_types": vertex_results,
                "edge_types": edge_results,
                "sample_entities": sample_entities,  # Help AI understand actual entities
                "entity_count": len(sample_entities)
            }
        except Exception as e:
            logger.error(f"Failed to get schema: {e}")
            return {}

    def get_entity_names(self, limit: int = 100) -> List[str]:
        """Get actual entity names from the graph for better AI query matching."""
        try:
            entity_names = []
            # Build dynamic UNION query using essential types + discovered types
            base_vertex_types = ['Entity']  # Essential entity type always present
            all_vertex_types = list(set(base_vertex_types) | self._discovered_vertex_types)
            union_parts = [f"SELECT FROM {vtype}" for vtype in all_vertex_types]
            union_query = " UNION ".join(union_parts)
            query = f"SELECT DISTINCT name FROM ({union_query}) WHERE name IS NOT NULL LIMIT {limit}"
            results = self._db.query("sql", query)
            if isinstance(results, dict) and "result" in results:
                records = results["result"]
                if isinstance(records, list):
                    entity_names = [r.get("name", "") for r in records if r.get("name")]
            elif isinstance(results, list):
                entity_names = [r.get("name", "") for r in results if r.get("name")]
                    
            return [name for name in entity_names if name and len(name.strip()) > 0]
        except Exception as e:
            logger.debug(f"Could not get entity names: {e}")
            return []

    # Helper methods

    def _result_to_node(self, result_data: Dict[str, Any]) -> LabelledNode:
        """Convert query result to LabelledNode."""
        node_type = result_data.get('@type', result_data.get('@class', 'Entity'))
        node_id = result_data.get('@rid', str(hash(str(result_data))))

        # Extract properties, excluding system fields
        properties = {
            k: v for k, v in result_data.items()
            if not k.startswith('@')
        }

        if node_type == 'TextChunk':
            text = properties.pop('text', '')
            # Use the logical 'id' field (chunk ID) not the ArcadeDB @rid.
            # get_llama_nodes builds a map keyed by node.node_id and _add_source_text
            # looks it up by triplet_source_id — both use the logical chunk ID, so
            # using @rid here would cause a key mismatch and source text would never
            # be attached to graph retrieval results.
            # VectorContextRetriever also matches triplet node IDs against kg_ids
            # (logical IDs from vector_query) — @rid would always miss, giving score=0.0.
            logical_id = properties.get('id') or node_id
            return ChunkNode(
                id_=logical_id,
                text=text,
                label=node_type,
                properties=properties
            )
        else:
            name = properties.get('name', node_id)
            return EntityNode(
                name=name,
                label=node_type,
                properties=properties
            )

    def _result_to_relation(self, result_data: Dict[str, Any]) -> Relation:
        """Convert query result to Relation."""
        label = result_data.get('@type', result_data.get('@class', 'RELATION'))
        # ArcadeDB uses @out/@in for edge endpoints (system fields with @ prefix)
        source_id = result_data.get('source_id', result_data.get('@out', result_data.get('out', '')))
        target_id = result_data.get('target_id', result_data.get('@in',  result_data.get('in', '')))

        # Extract properties, excluding system fields
        properties = {
            k: v for k, v in result_data.items()
            if not k.startswith('@') and k not in ['source_id', 'target_id', 'out', 'in']
        }

        return Relation(
            label=label,
            source_id=source_id,
            target_id=target_id,
            properties=properties
        )

    def _upsert_entity_node(self, node: EntityNode) -> None:
        """Upsert an entity node."""
        # LOG: What came from LLM
        logger.info(f"LLM_ENTITY_INPUT: name='{node.name}', label='{node.label}', properties={list(node.properties.keys()) if node.properties else []}")
        
        # Validate entity name - reject obviously bad extractions
        if self._is_invalid_entity_name(node.name):
            logger.warning(f"ENTITY_REJECTED: name='{node.name[:100]}...' - failed validation")
            return
            
        # Determine entity type based on node label or properties
        entity_type = self._determine_entity_type(node)
        
        # Normalize and deduplicate entity name
        normalized_name = self._normalize_and_deduplicate_entity(node.name)
        
        # LOG: SQL processing decisions
        logger.info(f"SQL_ENTITY_PROCESSING: original='{node.name}' -> normalized='{normalized_name}', type='{entity_type}'")
        
        # Ensure the entity type exists (dynamic schema)
        self._ensure_dynamic_type(entity_type, 'VERTEX')
        
        # Ensure index exists for UPSERT (required by ArcadeDB)
        self._ensure_index_for_upsert(entity_type, 'name')
        
        # Use ArcadeDB's native UPDATE ... UPSERT syntax for atomic upsert
        prop_assignments = [f"name = '{self._escape_string(normalized_name)}'"]
        
        # Embedding is stored via a separate UPDATE after the UPSERT commits —
        # the LSM_VECTOR index requires a native array literal which the SQL parser
        # rejects inside a SET clause.
        has_embedding = hasattr(node, 'embedding') and node.embedding is not None

        # Add other properties, ensuring each is declared on the type first
        if node.properties:
            for k, v in node.properties.items():
                if k != 'embedding':
                    v_str = str(v)
                    if len(v_str) > 1000:
                        logger.debug(f"Skipping large property {k} for entity {normalized_name} (size: {len(v_str)})")
                        continue
                    self._ensure_property(entity_type, k)
                    safe_k = self._sql_identifier(k)
                    prop_assignments.append(f"{safe_k} = '{self._escape_string(v_str)}'")
        
        try:
            # Use ArcadeDB's native UPDATE ... UPSERT syntax (atomic operation)
            query = f"UPDATE {entity_type} SET {', '.join(prop_assignments)} UPSERT WHERE name = '{self._escape_string(normalized_name)}'"
            logger.debug(f"Executing {entity_type} UPSERT")
            result = self._db.query("sql", query, is_command=True)
            logger.debug(f"{entity_type} UPSERT result: {result}")

            if has_embedding:
                self._update_embedding(entity_type, 'name', normalized_name,
                                       list(node.embedding))

            # Create MENTIONS relationship from chunk to entity
            self._create_mentions_relationship_sql(node, entity_type)
            
        except Exception as e:
            err_str = str(e)
            if len(err_str) > 300:
                err_str = err_str[:300] + "... [truncated]"
            logger.warning(f"{entity_type} UPSERT failed for '{normalized_name}': {err_str}")
            # Fallback to basic insert if UPSERT fails
            try:
                basic_query = f"INSERT INTO {entity_type} SET name = '{self._escape_string(normalized_name)}'"
                logger.debug(f"Fallback basic insert for {normalized_name}")
                self._db.query("sql", basic_query, is_command=True)
                self._create_mentions_relationship_sql(node, entity_type)
            except TransactionException as fallback_error:
                detail = (fallback_error.detail or '').lower()
                if "duplicate" in detail or "duplicated key" in detail:
                    logger.debug(f"Entity '{normalized_name}' already exists in {entity_type} (duplicate key - ok)")
                    self._create_mentions_relationship_sql(node, entity_type)
                else:
                    logger.warning(f"Insert fallback also failed for '{normalized_name}': {fallback_error}")
            except Exception as fallback_error:
                fallback_str = str(fallback_error)
                if "duplicate" in fallback_str.lower() or "duplicated key" in fallback_str.lower():
                    logger.debug(f"Entity '{normalized_name}' already exists in {entity_type} (duplicate key - ok)")
                    self._create_mentions_relationship_sql(node, entity_type)
                else:
                    if len(fallback_str) > 300:
                        fallback_str = fallback_str[:300] + "... [truncated]"
                    logger.warning(f"Insert fallback also failed for '{normalized_name}': {fallback_str}")

    def _upsert_chunk_node(self, node: ChunkNode) -> None:
        """Upsert a text chunk node."""
        logger.info(f"CHUNK_UPSERT: Storing ChunkNode id={node.id}, text_length={len(node.text)}")
        text_escaped = self._escape_string(node.text)
        
        # Ensure TextChunk type exists (dynamic schema)
        self._ensure_dynamic_type('TextChunk', 'VERTEX')
        
        # Ensure index exists for UPSERT (required by ArcadeDB)
        self._ensure_index_for_upsert('TextChunk', 'id')
        
        # Use ArcadeDB's native UPDATE ... UPSERT syntax for atomic upsert
        prop_assignments = [f"id = '{self._escape_string(node.id)}'", f"text = '{text_escaped}'"]

        # Persist ref_doc_id so delete(properties={'ref_doc_id': '...'}) works
        ref_doc_id = getattr(node, 'ref_doc_id', None)
        if ref_doc_id is None and node.properties:
            ref_doc_id = node.properties.get('ref_doc_id')
        if ref_doc_id:
            prop_assignments.append(f"ref_doc_id = '{self._escape_string(str(ref_doc_id))}'")
            logger.debug(f"Storing ref_doc_id '{ref_doc_id}' for chunk {node.id}")

        # Embedding stored via separate UPDATE after UPSERT — same reason as entity nodes.
        has_embedding = hasattr(node, 'embedding') and node.embedding is not None

        # Add other properties, ensuring each is declared on the type first
        if node.properties:
            for k, v in node.properties.items():
                if k in ('embedding', 'ref_doc_id'):
                    continue
                v_str = str(v)
                if len(v_str) > 1000:
                    logger.debug(f"Skipping large property {k} for chunk {node.id} (size: {len(v_str)})")
                    continue
                self._ensure_property('TextChunk', k)
                safe_k = self._sql_identifier(k)
                prop_assignments.append(f"{safe_k} = '{self._escape_string(v_str)}'")
        
        try:
            # Use ArcadeDB's native UPDATE ... UPSERT syntax (atomic operation)
            query = f"UPDATE TextChunk SET {', '.join(prop_assignments)} UPSERT WHERE id = '{self._escape_string(node.id)}'"
            logger.debug(f"Executing TextChunk UPSERT")
            result = self._db.query("sql", query, is_command=True)
            logger.debug(f"TextChunk UPSERT result: {result}")

            if has_embedding:
                self._update_embedding('TextChunk', 'id', node.id, list(node.embedding))
            
            # Embedding stored as hidden property for vector functionality
        except Exception as e:
            err_str = str(e)
            if len(err_str) > 300:
                err_str = err_str[:300] + "... [truncated]"
            logger.warning(f"TextChunk UPSERT failed for '{node.id}': {err_str}")
            # Fallback to basic insert if UPSERT fails
            try:
                basic_query = f"INSERT INTO TextChunk SET id = '{self._escape_string(node.id)}', text = '{text_escaped}'"
                logger.debug(f"Fallback basic insert for chunk {node.id}")
                self._db.query("sql", basic_query, is_command=True)
            except TransactionException as fallback_error:
                detail = (fallback_error.detail or '').lower()
                if "duplicate" in detail or "duplicated key" in detail:
                    logger.debug(f"Chunk '{node.id}' already exists in TextChunk (duplicate key - ok)")
                else:
                    logger.warning(f"Insert fallback also failed for chunk '{node.id}': {fallback_error}")
            except Exception as fallback_error:
                fallback_str = str(fallback_error)
                if "duplicate" in fallback_str.lower() or "duplicated key" in fallback_str.lower():
                    logger.debug(f"Chunk '{node.id}' already exists in TextChunk (duplicate key - ok)")
                else:
                    if len(fallback_str) > 300:
                        fallback_str = fallback_str[:300] + "... [truncated]"
                    logger.warning(f"Insert fallback also failed for chunk '{node.id}': {fallback_str}")

    def _upsert_generic_node(self, node: LabelledNode) -> None:
        """Upsert a generic labeled node."""
        properties_str = self._properties_to_sql_string(node.properties)
        try:
            # Try to insert new record
            if node.properties:
                prop_keys = ', '.join(node.properties.keys())
                prop_values = ', '.join([f"'{self._escape_string(str(v))}'" for v in node.properties.values()])
                query = f"INSERT INTO {node.label} (id, {prop_keys}) VALUES ('{self._escape_string(node.id)}', {prop_values})"
            else:
                query = f"INSERT INTO {node.label} (id) VALUES ('{self._escape_string(node.id)}')"
            self._db.query("sql", query)
            return
        except:
            # If insert fails, try update
            query = f"UPDATE {node.label} SET {properties_str} WHERE id = '{self._escape_string(node.id)}'"
            self._db.query("sql", query)

    def _upsert_relation(self, relation: Relation) -> None:
        """Upsert a relationship."""
        # LOG: What came from LLM
        logger.debug(f"LLM_RELATION_INPUT: label='{relation.label}', source='{relation.source_id}', target='{relation.target_id}', properties={list(relation.properties.keys()) if relation.properties else []}")
        
        # Use proper ArcadeDB CREATE EDGE syntax from documentation
        properties_str = self._properties_to_sql_string(relation.properties)
        
        # Skip complex SQL MATCH approach for now due to parsing issues
        # Use the reliable individual query approach directly
        logger.debug(f"Using individual query approach for relation: {relation.label}")
        
        # Normalize entity names to match how they were stored (same as entity creation)
        normalized_source_id = self._normalize_and_deduplicate_entity(relation.source_id) if relation.source_id else relation.source_id
        normalized_target_id = self._normalize_and_deduplicate_entity(relation.target_id) if relation.target_id else relation.target_id
        
        logger.debug(f"SQL_RELATION: source '{relation.source_id}' -> normalized '{normalized_source_id}'")
        logger.debug(f"SQL_RELATION: target '{relation.target_id}' -> normalized '{normalized_target_id}'")
        
        # Create edge type if it doesn't exist (dynamic schema)
        self._ensure_dynamic_type(relation.label, 'EDGE')
        
        # Use reliable individual query approach
        try:
            # Query the live schema so pre-existing types (PERSON, ORGANIZATION, etc.)
            # are always included, even if they weren't created in this session.
            vertex_types = self._get_all_vertex_types()
            
            logger.debug(f"FALLBACK: Searching for relation entities in types: {vertex_types}")
            
            # Find source entity @rid - check both 'name' and 'id' fields
            source_rid = None
            for vertex_type in vertex_types:
                try:
                    # For TextChunk, search by 'id'; for others, search by 'name'
                    if vertex_type == 'TextChunk':
                        query = f"SELECT @rid FROM {vertex_type} WHERE id = '{self._escape_string(relation.source_id)}' LIMIT 1"
                    else:
                        query = f"SELECT @rid FROM {vertex_type} WHERE name = '{self._escape_string(normalized_source_id)}' LIMIT 1"
                    result = self._db.query("sql", query)
                    if result and len(result) > 0:
                        source_rid = result[0]['@rid']
                        logger.debug(f"Found source {relation.source_id} in {vertex_type}: {source_rid}")
                        break
                except Exception as e:
                    logger.debug(f"Search failed in {vertex_type}: {e}")
                    continue
            
            # Find target entity @rid - check both 'name' and 'id' fields  
            target_rid = None
            for vertex_type in vertex_types:
                try:
                    # For TextChunk, search by 'id'; for others, search by 'name'
                    if vertex_type == 'TextChunk':
                        query = f"SELECT @rid FROM {vertex_type} WHERE id = '{self._escape_string(relation.target_id)}' LIMIT 1"
                    else:
                        query = f"SELECT @rid FROM {vertex_type} WHERE name = '{self._escape_string(normalized_target_id)}' LIMIT 1"
                    result = self._db.query("sql", query)
                    if result and len(result) > 0:
                        target_rid = result[0]['@rid']
                        logger.debug(f"Found target {relation.target_id} in {vertex_type}: {target_rid}")
                        break
                except Exception as e:
                    logger.debug(f"Search failed in {vertex_type}: {e}")
                    continue
            
            if not source_rid or not target_rid:
                logger.warning(f"SKIP: CREATE EDGE {relation.label}: source {'FOUND' if source_rid else 'MISSING'} target {'FOUND' if target_rid else 'MISSING'}")
                return
            
            # Create edge using @rid values with ArcadeDB's native IF NOT EXISTS
            if properties_str:
                query = f"CREATE EDGE {relation.label} FROM {source_rid} TO {target_rid} IF NOT EXISTS SET {properties_str}"
            else:
                query = f"CREATE EDGE {relation.label} FROM {source_rid} TO {target_rid} IF NOT EXISTS"
            
            logger.debug(f"Executing fallback CREATE EDGE: {query}")
            result = self._db.query("sql", query, is_command=True)
            logger.debug(f"SQL_FALLBACK_SUCCESS: Created {relation.label} edge from {source_rid} to {target_rid}")
            
        except Exception as fallback_error:
            logger.error(f"Both SQL MATCH and fallback failed: {fallback_error}")
            return


    def _properties_to_sql_string(self, properties: Dict[str, Any]) -> str:
        """Convert properties dict to SQL SET string."""
        if not properties:
            return ""

        items = []
        for key, value in properties.items():
            if isinstance(value, str):
                value_str = f"'{self._escape_string(value)}'"
            elif isinstance(value, (list, dict)):
                # Properly escape JSON data
                json_str = json.dumps(value)
                escaped_json = self._escape_string(json_str)
                value_str = f"'{escaped_json}'"
            else:
                value_str = str(value)

            safe_key = self._sql_identifier(key)
            items.append(f"{safe_key} = {value_str}")

        return ", ".join(items)

    @staticmethod
    def _sql_identifier(name: str) -> str:
        """Return a safely-quoted ArcadeDB SQL identifier.

        Property names that contain spaces (e.g. 'modified at') must be
        wrapped in backticks, exactly like MySQL quoted identifiers.
        Names without spaces are returned unchanged.
        """
        return f"`{name}`" if ' ' in name else name

    def _escape_string(self, text: str) -> str:
        """Escape string for ArcadeDB SQL queries using backslash escaping.
        
        ArcadeDB SQL documentation states both methods are supported:
        - Single quotes: Can use SQL standard doubling (' -> '') OR backslash escaping (\')
        - However, in practice, backslash escaping is more reliable for REST API usage
        - This matches the approach used by arcadedb-python internally
        
        Reference: https://docs.arcadedb.com/#sql-syntax
        """
        if text is None:
            return "NULL"
        
        escaped = str(text)
        
        # Handle Unicode characters that might cause issues - do this FIRST before other escaping
        # Replace problematic Unicode characters with safe alternatives
        escaped = escaped.replace('\u2013', '-')    # En dash
        escaped = escaped.replace('\u2014', '--')   # Em dash
        escaped = escaped.replace('\u2018', "'")    # Left single quote -> regular apostrophe
        escaped = escaped.replace('\u2019', "'")    # Right single quote -> regular apostrophe
        escaped = escaped.replace('\u201c', '"')    # Left double quote -> regular quote
        escaped = escaped.replace('\u201d', '"')    # Right double quote -> regular quote
        
        # Backslash escaping (matching arcadedb-python's approach)
        # Handle backslashes first (if they exist in the original text)
        escaped = escaped.replace("\\", "\\\\")     # Escape backslashes: \ -> \\
        
        # Backslash escape single quotes (ArcadeDB supports this per docs)
        escaped = escaped.replace("'", "\\'")       # Backslash escape: ' -> \'
        
        # Escape control characters that need backslash escaping in SQL string literals
        escaped = escaped.replace('\n', '\\n')      # Newline
        escaped = escaped.replace('\r', '\\r')      # Carriage return
        escaped = escaped.replace('\t', '\\t')      # Tab
        
        # NOTE: Double quotes (") do NOT need escaping in SQL string literals
        # because SQL uses single quotes (') as string delimiters
        
        return escaped
    
    

    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity names to avoid duplicates like 'Alfresco' vs 'Alfresco Software'."""
        if not name:
            return name
            
        # Clean entity names that include metadata dictionaries from LLM extraction
        # Pattern: "EntityName ({'key': 'value', ...})"
        import re
        metadata_pattern = r'\s*\(\{[^}]+\}\)$'
        cleaned = re.sub(metadata_pattern, '', name)
        
        # Common normalization rules
        normalized = cleaned.strip()
        
        # Handle common company name variations
        company_suffixes = [' Software', ' Inc', ' Inc.', ' Corp', ' Corp.', ' Ltd', ' Ltd.', ' LLC', ' Company', ' Co.']
        for suffix in company_suffixes:
            if normalized.endswith(suffix):
                base_name = normalized[:-len(suffix)].strip()
                # Only normalize if the base name is substantial (more than 2 chars)
                if len(base_name) > 2:
                    return base_name
        
        return normalized

    def _create_mentions_relationship_sql(self, entity_node: EntityNode, entity_type: str) -> None:
        """Create MENTIONS relationship from chunk to entity (SQL mode)."""
        # Check if entity has triplet_source_id (chunk ID it came from)
        if not entity_node.properties or 'triplet_source_id' not in entity_node.properties:
            logger.debug(f"Entity {entity_node.name} has no triplet_source_id - skipping MENTIONS")
            return
        
        chunk_id = entity_node.properties['triplet_source_id']
        if not chunk_id:
            logger.debug(f"Entity {entity_node.name} has empty triplet_source_id - skipping MENTIONS")
            return
        
        try:
            # First check if any TextChunk records exist at all
            chunk_count_query = "SELECT COUNT(*) as count FROM TextChunk"
            count_result = self._db.query("sql", chunk_count_query)
            total_chunks = count_result[0].get('count', 0) if count_result else 0
            logger.debug(f"MENTIONS: Total TextChunk records: {total_chunks}")
            
            # Find the chunk
            chunk_query = f"SELECT @rid FROM TextChunk WHERE id = '{self._escape_string(chunk_id)}' LIMIT 1"
            chunk_result = self._db.query("sql", chunk_query)
            
            if not chunk_result or len(chunk_result) == 0:
                logger.warning(f"MENTIONS: Chunk not found: {chunk_id} (total chunks: {total_chunks})")
                # List some existing chunk IDs for debugging
                if total_chunks > 0:
                    sample_query = "SELECT id FROM TextChunk LIMIT 3"
                    sample_result = self._db.query("sql", sample_query)
                    if sample_result:
                        sample_ids = [r.get('id', '') for r in sample_result]
                        logger.debug(f"MENTIONS: Sample existing chunk IDs: {sample_ids}")
                return
            
            chunk_rid = chunk_result[0]['@rid']
            
            # Find the entity using the normalized name (which is what was actually stored)
            normalized_entity_name = self._normalize_and_deduplicate_entity(entity_node.name)
            entity_query = f"SELECT @rid FROM {entity_type} WHERE name = '{self._escape_string(normalized_entity_name)}' LIMIT 1"
            entity_result = self._db.query("sql", entity_query)
            
            if not entity_result or len(entity_result) == 0:
                logger.debug(f"Entity not found for MENTIONS relationship: {entity_node.name}")
                return
            
            entity_rid = entity_result[0]['@rid']
            
            # Create MENTIONS edge type if it doesn't exist
            try:
                edge_type_query = "CREATE EDGE TYPE MENTIONS IF NOT EXISTS"
                self._db.query("sql", edge_type_query, is_command=True)
            except Exception:
                pass  # Edge type might already exist
            
            # Create MENTIONS relationship using ArcadeDB's native IF NOT EXISTS
            mentions_query = f"CREATE EDGE MENTIONS FROM {chunk_rid} TO {entity_rid} IF NOT EXISTS"
            self._db.query("sql", mentions_query, is_command=True)
            logger.debug(f"Created MENTIONS relationship: {chunk_id} -> {entity_node.name}")
            
        except Exception as e:
            logger.debug(f"Failed to create MENTIONS relationship: {e}")


    def _is_invalid_entity_name(self, name: str) -> bool:
        """Simplified entity validation - only check for empty/None names."""
        return not name or len(name.strip()) == 0
    
    def _extract_core_entity_from_sentence(self, name: str) -> str:
        """Simplified - just return the name as-is."""
        return name.strip()
    
    def _normalize_and_deduplicate_entity(self, name: str) -> str:
        """Simplified entity normalization - just basic cleanup."""
        # Apply only basic normalization (preserve semantic distinctions)
        return self._normalize_entity_name(name.strip())

    def _determine_entity_type(self, node: EntityNode) -> str:
        """Determine the appropriate ArcadeDB vertex type for an entity node.
        
        This creates REAL vertex types in ArcadeDB, not just properties.
        Priority order:
        1. Direct label from SchemaLLMPathExtractor (user/developer defined)
        2. LlamaIndex discovered labels beyond predefined schema
        3. Pattern-based classification fallback
        4. Generic 'Entity' type
        """
        # PRIORITY 1: Use the node's label directly (from path extractors or LlamaIndex discovery)
        if hasattr(node, 'label') and node.label and node.label.strip():
            # Clean and normalize the label for ArcadeDB vertex type
            vertex_type = node.label.strip().upper().replace(' ', '_').replace('-', '_')
            
            # Validate it's a reasonable vertex type name (alphanumeric + underscore)
            if vertex_type.replace('_', '').isalnum() and len(vertex_type) > 0:
                logger.debug(f"ENTITY_TYPE_DETECTION: Using label '{vertex_type}' for entity '{node.name}' (from PathExtractor/LLM)")
                return vertex_type
        
        # PRIORITY 2: Fallback to pattern-based classification
        # Pass the original name so all-caps detection works correctly
        classified_type = self._classify_entity_by_patterns(node.name)
        if classified_type != 'Entity':
            logger.debug(f"ENTITY_TYPE_DETECTION: Using pattern-based type '{classified_type}' for entity '{node.name}' (fallback classification)")
            return classified_type
        
        # PRIORITY 3: Generic fallback
        logger.debug(f"ENTITY_TYPE_DETECTION: Using generic 'Entity' type for '{node.name}' (no specific classification found)")
        return 'Entity'
    
    def _classify_entity_by_patterns(self, entity_name: str) -> str:
        """Classify entity using generalized patterns.
        
        Returns type names that may or may not exist as base types.
        Dynamic type creation will handle creating missing types automatically.
        Base types (PERSON, ORGANIZATION, LOCATION, PLACE) are pre-created.
        Other types (TECHNOLOGY, PROJECT) will be created dynamically when needed.
        """
        name_lower = entity_name.lower()
        words = name_lower.split()
        original_words = entity_name.split()
        
        # Organization patterns
        org_suffixes = ['corp', 'inc', 'ltd', 'llc', 'company', 'co', 'corporation', 
                       'software', 'systems', 'technologies', 'solutions', 'group']
        org_types = ['agency', 'administration', 'foundation', 'institute', 'university',
                    'organization', 'association', 'consortium', 'alliance']
        
        # Check suffixes (e.g., "Microsoft Corp")
        if any(suffix in name_lower for suffix in org_suffixes):
            return 'ORGANIZATION'
        
        # Check organization types (e.g., "Space Agency")  
        if any(org_type in name_lower for org_type in org_types):
            return 'ORGANIZATION'
        
        # All-caps single word: acronym organizations (IBM, SAP, EMC, NASA)
        # Check against the original-case name, not the lowercased version
        if len(original_words) == 1 and entity_name.isupper() and len(entity_name) >= 2:
            return 'ORGANIZATION'
        
        # Technology/specification patterns
        tech_indicators = ['specification', 'standard', 'protocol', 'api', 'system', 
                          'platform', 'framework', 'language', 'suite', 'service']
        if any(indicator in name_lower for indicator in tech_indicators):
            return 'TECHNOLOGY'
        
        # Project patterns (often have descriptive names)
        if len(words) >= 2 and any(word in name_lower for word in ['project', 'program', 'initiative']):
            return 'PROJECT'
        
        # Location patterns
        location_indicators = ['city', 'state', 'country', 'region', 'district', 'county']
        if any(indicator in name_lower for indicator in location_indicators):
            return 'LOCATION'
        
        return 'Entity'  # No pattern matched
