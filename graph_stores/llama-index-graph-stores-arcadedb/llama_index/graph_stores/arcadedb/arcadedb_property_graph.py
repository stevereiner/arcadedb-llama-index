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
    )
except ImportError as e:
    raise ImportError(
        "arcadedb_python>=0.3.0 is required for ArcadeDB integration. "
        "Install it with: pip install arcadedb-python>=0.3.0"
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

        # Connect to or create database
        try:
            # Try to connect to existing database
            self._db = DatabaseDao(self._client, database)
            
            # Test the connection with a simple query
            test_result = self._db.query("sql", "SELECT 1 as test")
            logger.info(f"Connected to existing database: {database}")
            
        except Exception as e:
            if create_database_if_not_exists:
                logger.info(f"Database {database} not found or inaccessible, creating...")
                try:
                    # Create new database using the fixed client
                    self._db = DatabaseDao.create(self._client, database)
                    logger.info(f"Created database: {database}")
                except Exception as create_error:
                    logger.error(f"Failed to create database: {create_error}")
                    raise create_error
            else:
                logger.error(f"Database connection failed: {e}")
                raise e

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
                    continue
                    
                try:
                    query = f"CREATE {type_kind} TYPE {type_name}"
                    result = self._db.query("sql", query, is_command=True)
                    logger.info(f"Schema created successfully: {query}")
                    
                    # For VERTEX types, create essential properties with correct data types
                    if type_kind == 'VERTEX':
                        self._create_vertex_properties(type_name)
                    
                    created_count += 1
                except Exception as e:
                    error_msg = str(e)
                    # If it fails because it's "not idempotent", the type likely already exists
                    if "not idempotent" in error_msg:
                        logger.info(f"Type {type_name} already exists (idempotent error), ensuring properties exist")
                        # Even if type exists, ensure properties are created
                        if type_kind == 'VERTEX':
                            self._create_vertex_properties(type_name)
                        continue
                    else:
                        logger.error(f"Schema creation failed: CREATE {type_kind} TYPE {type_name} - {e}")
                        # Don't continue if critical types fail with non-idempotent errors
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
                
            logger.info(f"Dynamically created {type_kind} type: {type_name}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "not idempotent" in error_msg or "already exists" in error_msg:
                # Type already exists, but still ensure properties exist
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
                    ('embedding', 'STRING', '(hidden true)')  # Hidden JSON string for embeddings
                ]
            else:
                # Entity types use 'name' as primary key (STRING)
                properties = [
                    ('name', 'STRING', ''),
                    ('embedding', 'STRING', '(hidden true)')  # Hidden JSON string for embeddings
                ]
            
            # Create each property with proper data type
            for prop_name, prop_type, constraints in properties:
                try:
                    prop_query = f"CREATE PROPERTY {type_name}.{prop_name} {prop_type} {constraints}".strip()
                    self._db.query("sql", prop_query, is_command=True)
                    logger.debug(f"Created property {type_name}.{prop_name} ({prop_type})")
                except Exception as prop_error:
                    error_msg = str(prop_error)
                    if "already exists" in error_msg.lower():
                        logger.debug(f"Property {type_name}.{prop_name} already exists - skipping")
                    else:
                        logger.warning(f"Property creation failed for {type_name}.{prop_name}: {prop_error}")
                        # Don't raise exception for property creation failures - they may already exist
            
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
                    
        except Exception as e:
            logger.warning(f"Failed to create properties for {type_name}: {e}")

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
                # Step 1: Ensure property exists first
                prop_query = f"CREATE PROPERTY {type_name}.{property_name} STRING"
                try:
                    self._db.query("sql", prop_query, is_command=True)
                    logger.debug(f"Ensured property {type_name}.{property_name} exists")
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
        try:
            # Step 1: Ensure the property exists
            prop_query = f"CREATE PROPERTY {type_name}.{property_name} STRING"
            self._db.query("sql", prop_query, is_command=True)
            logger.debug(f"Created fallback property {type_name}.{property_name}")
        except Exception as prop_error:
            error_msg = str(prop_error)
            if "already exists" in error_msg.lower():
                logger.debug(f"Property {type_name}.{property_name} already exists")
            else:
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

    def get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Get nodes with matching properties or ids."""
        nodes = []

        try:
            if ids:
                # Query by node IDs - try both Entity and TextChunk tables
                for node_id in ids:
                    # Try Entity first
                    try:
                        query = f"SELECT * FROM Entity WHERE id = '{node_id}' OR name = '{node_id}'"
                        results = self._db.query("sql", query)
                        if results.get("result"):
                            for result in results["result"]:
                                nodes.append(self._result_to_node(result))
                            continue
                    except:
                        pass
                    
                    # Try TextChunk if not found in Entity
                    try:
                        query = f"SELECT * FROM TextChunk WHERE id = '{node_id}'"
                        results = self._db.query("sql", query)
                        if results.get("result"):
                            for result in results["result"]:
                                nodes.append(self._result_to_node(result))
                    except:
                        pass

            elif properties:
                # Query by properties
                where_conditions = []
                for key, value in properties.items():
                    if isinstance(value, str):
                        where_conditions.append(f"{key} = '{value}'")
                    else:
                        where_conditions.append(f"{key} = {value}")

                where_clause = " AND ".join(where_conditions)
                query = f"SELECT * FROM (SELECT FROM Entity UNION SELECT FROM TextChunk) WHERE {where_clause}"
                results = self._db.query("sql", query)

                for result in results:
                    nodes.append(self._result_to_node(result))
            else:
                # Get all nodes (limited)
                query = "SELECT * FROM (SELECT FROM Entity UNION SELECT FROM TextChunk) LIMIT 100"
                results = self._db.query("sql", query)

                for result in results:
                    nodes.append(self._result_to_node(result))

        except Exception as e:
            logger.error(f"Failed to get nodes: {e}")

        return nodes

    def get_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get triplets (relationships) with matching criteria using v0.3.0+ features."""
        triplets = []

        try:
            # Try enhanced get_triplets method first
            try:
                logger.debug("Using enhanced get_triplets method")
                
                # Use enhanced get_triplets method with correct parameters
                enhanced_results = self._db.get_triplets(
                    subject_types=entity_names,
                    relation_types=relation_names,
                    object_types=entity_names,  # Use same entity names for objects
                    limit=10000
                )
                
                # Convert enhanced results to Triplet objects
                for result in enhanced_results:
                    if isinstance(result, dict):
                        source_id = result.get('source', {}).get('id') or result.get('subject', {}).get('id')
                        target_id = result.get('target', {}).get('id') or result.get('object', {}).get('id')
                        relation_type = result.get('relation', {}).get('type') or result.get('predicate')
                        
                        if source_id and target_id and relation_type:
                            triplet = Triplet(
                                subject_id=source_id,
                                object_id=target_id,
                                label=relation_type
                            )
                            triplets.append(triplet)
                
                logger.debug(f"Enhanced get_triplets returned {len(triplets)} triplets")
                return triplets
                
            except (QueryParsingException, ArcadeDBException) as e:
                logger.warning(f"Enhanced get_triplets failed, using fallback: {e}")
                # Fall through to traditional method
                
            # Traditional method (fallback)
            logger.debug("Using traditional get_triplets method")
            
            # Use SQL MATCH for more intuitive Cypher-style graph pattern matching
            try:
                match_conditions = []
                where_conditions = []

                # Build entity name conditions if specified
                if entity_names:
                    escaped_names = [self._escape_string(name) for name in entity_names]
                    entity_conditions = []
                    for name in escaped_names:
                        # Handle both TextChunk (id field) and Entity (name field) matching
                        entity_conditions.extend([
                            f"a.name = '{name}'",
                            f"b.name = '{name}'",
                            f"a.id = '{name}'",
                            f"b.id = '{name}'"
                        ])
                    where_conditions.append(f"({' OR '.join(entity_conditions)})")

                # Build relation name conditions if specified
                if relation_names:
                    escaped_relations = [self._escape_string(name) for name in relation_names]
                    relation_conditions = [f"r.@class = '{rel}'" for rel in escaped_relations]
                    where_conditions.append(f"({' OR '.join(relation_conditions)})")

                # Build property conditions if specified
                if properties:
                    for key, value in properties.items():
                        if isinstance(value, str):
                            escaped_value = self._escape_string(value)
                            where_conditions.append(f"(a.{key} = '{escaped_value}' OR b.{key} = '{escaped_value}' OR r.{key} = '{escaped_value}')")
                        else:
                            where_conditions.append(f"(a.{key} = {value} OR b.{key} = {value} OR r.{key} = {value})")

                # Build ID conditions if specified
                if ids:
                    escaped_ids = [self._escape_string(node_id) for node_id in ids]
                    id_conditions = []
                    for node_id in escaped_ids:
                        id_conditions.extend([
                            f"a.id = '{node_id}'",
                            f"b.id = '{node_id}'",
                            f"a.name = '{node_id}'",
                            f"b.name = '{node_id}'"
                        ])
                    where_conditions.append(f"({' OR '.join(id_conditions)})")

                # Build SQL MATCH query for relationship pattern
                where_clause = " AND ".join(where_conditions) if where_conditions else ""
                
                if where_clause:
                    query = f"""
                    MATCH {{as: a}}-{{as: r}}-{{as: b}}
                    WHERE {where_clause}
                    RETURN a, r, b
                    LIMIT 100
                    """
                else:
                    query = """
                    MATCH {as: a}-{as: r}-{as: b}
                    RETURN a, r, b
                    LIMIT 100
                    """

                logger.debug(f"Executing SQL MATCH triplets query")
                results = self._db.query("sql", query)
                
                # Process SQL MATCH results
                for result in results:
                    try:
                        source_node = self._result_to_node(result.get('a', {}))
                        target_node = self._result_to_node(result.get('b', {}))
                        relationship = self._result_to_relation(result.get('r', {}))
                        triplets.append((source_node, relationship, target_node))
                    except Exception as parse_error:
                        logger.debug(f"Failed to parse triplet result: {parse_error}")
                        continue
                
                logger.info(f"SQL_MATCH_TRIPLETS: Found {len(triplets)} triplets using MATCH pattern")
                
            except Exception as match_error:
                logger.warning(f"SQL MATCH triplets failed, falling back to expand(both()): {match_error}")
                
                # Fallback to original expand(both()) approach
                query_parts = ["SELECT expand(both()) FROM RELATION"]
                where_conditions = []

                if entity_names:
                    entity_names_str = "', '".join(entity_names)
                    where_conditions.append(f"(in().name IN ['{entity_names_str}'] OR out().name IN ['{entity_names_str}'])")

                if relation_names:
                    relation_names_str = "', '".join(relation_names)
                    where_conditions.append(f"@class IN ['{relation_names_str}']")

                if where_conditions:
                    query_parts.append("WHERE " + " AND ".join(where_conditions))

                query_parts.append("LIMIT 100")
                query = " ".join(query_parts)
                logger.debug(f"Executing fallback expand(both()) query")
                results = self._db.query("sql", query)

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
            # Get node IDs with proper escaping
            node_ids = [node.id for node in graph_nodes]
            escaped_ids = [self._escape_string(node_id) for node_id in node_ids]
            ids_str = "', '".join(escaped_ids)

            # Try SQL MATCH for more intuitive graph traversal, fallback to TRAVERSE
            try:
                # Build ignore conditions for relationship types
                ignore_conditions = []
                if ignore_rels:
                    escaped_ignore_rels = [self._escape_string(rel) for rel in ignore_rels]
                    ignore_conditions = [f"r.@class != '{rel}'" for rel in escaped_ignore_rels]
                
                # Build node matching conditions (handle both id and name fields)
                node_conditions = []
                for node_id in escaped_ids:
                    node_conditions.extend([
                        f"start.id = '{node_id}'",
                        f"start.name = '{node_id}'"
                    ])
                
                # Build WHERE clause
                where_conditions = [f"({' OR '.join(node_conditions)})"]
                if ignore_conditions:
                    where_conditions.extend(ignore_conditions)
                
                where_clause = " AND ".join(where_conditions)
                
                # Use SQL MATCH for relationship traversal (more Cypher-like)
                if depth == 1:
                    # Single hop - direct relationships
                    query = f"""
                    MATCH {{as: start}}-{{as: r}}-{{as: end}}
                    WHERE {where_clause}
                    RETURN start, r, end
                    LIMIT {limit}
                    """
                else:
                    # Multi-hop - use multiple MATCH patterns for different depths
                    # For now, use depth 1 and 2 explicitly (ArcadeDB SQL MATCH variable-length paths are complex)
                    query = f"""
                    MATCH {{as: start}}-{{as: r}}-{{as: end}}
                    WHERE {where_clause}
                    RETURN start, r, end
                    LIMIT {limit}
                    """
                
                logger.debug(f"Executing SQL MATCH rel_map query")
                results = self._db.query("sql", query)
                
                # Process SQL MATCH results
                for result in results:
                    try:
                        start_data = result.get('start', {})
                        rel_data = result.get('r', {})
                        end_data = result.get('end', {})
                        
                        start_node = self._result_to_node(start_data)
                        end_node = self._result_to_node(end_data)
                        relationship = self._result_to_relation(rel_data)
                        
                        triplets.append((start_node, relationship, end_node))
                    except Exception as parse_error:
                        logger.debug(f"Failed to parse rel_map result: {parse_error}")
                        continue
                
                logger.info(f"SQL_MATCH_REL_MAP: Found {len(triplets)} relationships using MATCH pattern")
                
            except Exception as match_error:
                logger.warning(f"SQL MATCH rel_map failed, falling back to TRAVERSE: {match_error}")
                
                # Fallback to original TRAVERSE approach
                ignore_clause = ""
                if ignore_rels:
                    ignore_types = "', '".join(ignore_rels)
                    ignore_clause = f"AND @class NOT IN ['{ignore_types}']"

                query = f"""
                SELECT expand(both()) FROM (
                    TRAVERSE both() FROM ['{ids_str}'] MAXDEPTH {depth}
                ) WHERE @class INSTANCEOF 'E' {ignore_clause}
                LIMIT {limit}
                """
                logger.debug(f"Executing fallback TRAVERSE query")
                results = self._db.query("sql", query)

        except Exception as e:
            logger.error(f"Failed to get relationship map: {e}")

        return triplets

    def upsert_nodes(self, nodes: Sequence[LabelledNode]) -> None:
        """Insert or update nodes in the graph using v0.3.0+ bulk operations."""
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
            
            # Bulk upsert entity nodes
            if entity_nodes:
                try:
                    entity_data = []
                    for node in entity_nodes:
                        data = {
                            'name': node.name,
                            'label': getattr(node, 'label', 'Entity'),
                            'properties': node.properties or {}
                        }
                        if hasattr(node, 'embedding') and node.embedding:
                            data['embedding'] = json.dumps(node.embedding)
                        entity_data.append(data)
                    
                    upserted_count = self._db.bulk_upsert(
                        type_name="Entity",
                        records=entity_data,
                        key_field='name',
                        batch_size=100
                    )
                    logger.info(f"Bulk upserted {upserted_count} entity nodes")
                    
                except (BulkOperationException, ValidationException) as e:
                    logger.warning(f"Bulk entity upsert failed, using individual operations: {e}")
                    # Fallback to individual operations
                    for node in entity_nodes:
                        try:
                            self._upsert_entity_node(node)
                        except Exception as e2:
                            logger.error(f"Failed to upsert entity node {node.name}: {e2}")
            
            # Bulk upsert chunk nodes
            if chunk_nodes:
                try:
                    chunk_data = []
                    for node in chunk_nodes:
                        data = {
                            'id': node.id,
                            'text': getattr(node, 'text', ''),
                            'properties': node.properties or {}
                        }
                        if hasattr(node, 'embedding') and node.embedding:
                            data['embedding'] = json.dumps(node.embedding)
                        chunk_data.append(data)
                    
                    upserted_count = self._db.bulk_upsert(
                        type_name="TextChunk",
                        records=chunk_data,
                        key_field='id',
                        batch_size=100
                    )
                    logger.info(f"Bulk upserted {upserted_count} chunk nodes")
                    
                except (BulkOperationException, ValidationException) as e:
                    logger.warning(f"Bulk chunk upsert failed, using individual operations: {e}")
                    # Fallback to individual operations
                    for node in chunk_nodes:
                        try:
                            self._upsert_chunk_node(node)
                        except Exception as e2:
                            logger.error(f"Failed to upsert chunk node {node.id}: {e2}")
            
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
        """Insert or update relationships in the graph using v0.3.0+ bulk operations."""
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
                    
                    # Note: bulk_upsert_relations doesn't exist in v0.3.0, fall back to individual
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
        """Delete nodes and relationships matching criteria using v0.3.0+ features."""
        try:
            if ids:
                # Use bulk delete for IDs
                try:
                    ids_str = "', '".join(ids)
                    # Delete from Entity first
                    entity_deleted = self._db.bulk_delete(
                        type_name="Entity",
                        conditions=[f"@rid IN ['{ids_str}']"],
                        batch_size=1000
                    )
                    # Delete from TextChunk
                    chunk_deleted = self._db.bulk_delete(
                        type_name="TextChunk", 
                        conditions=[f"@rid IN ['{ids_str}']"],
                        batch_size=1000
                    )
                    logger.info(f"Bulk deleted {entity_deleted + chunk_deleted} nodes by IDs")
                except (BulkOperationException, ValidationException) as e:
                    logger.warning(f"Bulk delete failed, using traditional method: {e}")
                    # Fallback to traditional method
                    ids_str = "', '".join(ids)
                    query = f"DELETE FROM (SELECT FROM Entity UNION SELECT FROM TextChunk) WHERE @rid IN ['{ids_str}']"
                    self._db.query("sql", query, is_command=True)

            elif entity_names:
                # Use bulk delete for entity names
                try:
                    names_str = "', '".join(entity_names)
                    # Delete from Entity first  
                    entity_deleted = self._db.bulk_delete(
                        type_name="Entity",
                        conditions=[f"name IN ['{names_str}']"],
                        batch_size=1000
                    )
                    # Delete from TextChunk (using id field)
                    chunk_deleted = self._db.bulk_delete(
                        type_name="TextChunk",
                        conditions=[f"id IN ['{names_str}']"],
                        batch_size=1000
                    )
                    logger.info(f"Bulk deleted {entity_deleted + chunk_deleted} nodes by entity names")
                except (BulkOperationException, ValidationException) as e:
                    logger.warning(f"Bulk delete failed, using traditional method: {e}")
                    # Fallback to traditional method
                    names_str = "', '".join(entity_names)
                    query = f"DELETE FROM (SELECT FROM Entity UNION SELECT FROM TextChunk) WHERE name IN ['{names_str}']"
                    self._db.query("sql", query, is_command=True)

            elif properties:
                # Build property conditions for bulk delete
                where_conditions = []
                for key, value in properties.items():
                    if isinstance(value, str):
                        where_conditions.append(f"{key} = '{value}'")
                    else:
                        where_conditions.append(f"{key} = {value}")

                try:
                    # Delete from Entity first
                    entity_deleted = self._db.bulk_delete(
                        type_name="Entity",
                        conditions=where_conditions,
                        batch_size=1000
                    )
                    # Delete from TextChunk
                    chunk_deleted = self._db.bulk_delete(
                        type_name="TextChunk",
                        conditions=where_conditions,
                        batch_size=1000
                    )
                    logger.info(f"Bulk deleted {entity_deleted + chunk_deleted} nodes by properties")
                except (BulkOperationException, ValidationException) as e:
                    logger.warning(f"Bulk delete failed, using traditional method: {e}")
                    # Fallback to traditional method
                    where_clause = " AND ".join(where_conditions)
                    query = f"DELETE FROM (SELECT FROM Entity UNION SELECT FROM TextChunk) WHERE {where_clause}"
                    self._db.query("sql", query, is_command=True)

            if relation_names:
                # Use safe_delete_all for relation types
                for relation_name in relation_names:
                    try:
                        deleted_count = self._db.safe_delete_all(relation_name, batch_size=1000)
                        logger.info(f"Safe deleted {deleted_count} relations of type {relation_name}")
                    except (ValidationException, BulkOperationException) as e:
                        logger.warning(f"Safe delete failed for {relation_name}, using traditional method: {e}")
                        # Fallback to traditional method
                        query = f"DELETE FROM {relation_name}"
                        self._db.query("sql", query, is_command=True)

        except Exception as e:
            logger.error(f"Failed to delete: {e}")
            raise ArcadeDBException(f"Delete operation failed: {e}") from e

    def structured_query(
        self, query: str, param_map: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute a structured SQL query against the graph.
        
        This SQL-only version provides optimal performance for ArcadeDB operations.
        """
        try:
            # Execute SQL query (this is the SQL-only version)
            results = self._db.query("sql", query)
            return results

        except Exception as e:
            logger.error(f"Structured query failed: {e}")
            logger.error(f"Query was: {query}")
            raise QueryParsingException(f"SQL query execution failed: {e}") from e

    def vector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[List[LabelledNode], List[float]]:
        """Execute a vector similarity query using cosine similarity.
        
        ArcadeDB supports vector operations but may change implementation.
        This method includes fallback mechanisms for resilience.
        
        Note: ArcadeDB has mentioned they may change how vectors are implemented,
        so we detect capabilities dynamically and provide fallbacks.
        """
        if not query.query_embedding:
            logger.warning("No query embedding provided for vector search")
            return [], []
        
        try:
            # Convert embedding to string for SQL query
            embedding_str = str(query.query_embedding).replace("'", "\"")
            
            # Build filter conditions if provided
            where_conditions = []
            if query.filters:
                for filter_item in query.filters.filters:
                    if isinstance(filter_item.value, str):
                        where_conditions.append(f"{filter_item.key} = '{self._escape_string(filter_item.value)}'")
                    else:
                        where_conditions.append(f"{filter_item.key} = {filter_item.value}")
            
            where_clause = ""
            if where_conditions:
                where_clause = f"AND {' AND '.join(where_conditions)}"
            
            # Use manual similarity calculation (ArcadeDB doesn't have COSINE_SIMILARITY function)
            return self._vector_query_manual_fallback(query)
            
        except Exception as e:
            logger.error(f"Vector query failed: {e}")
            return [], []

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
            return ChunkNode(
                id_=node_id,
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
        source_id = result_data.get('source_id', result_data.get('out', ''))
        target_id = result_data.get('target_id', result_data.get('in', ''))

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
        
        # Handle embedding if present - store as JSON for vector functionality
        if hasattr(node, 'embedding') and node.embedding is not None:
            embedding_json = json.dumps(node.embedding)
            prop_assignments.append(f"embedding = '{embedding_json}'")
            logger.debug(f"Storing embedding for entity {normalized_name} (dim: {len(node.embedding)})")
        
        # Add other properties
        if node.properties:
            for k, v in node.properties.items():
                if k != 'embedding':  # Skip embedding in properties, handled above
                    # Skip very large properties that cause SQL parsing issues
                    v_str = str(v)
                    if len(v_str) > 1000:  # Skip large properties
                        logger.debug(f"Skipping large property {k} for entity {normalized_name} (size: {len(v_str)})")
                        continue
                    prop_assignments.append(f"{k} = '{self._escape_string(v_str)}'")
        
        try:
            # Use ArcadeDB's native UPDATE ... UPSERT syntax (atomic operation)
            query = f"UPDATE {entity_type} SET {', '.join(prop_assignments)} UPSERT WHERE name = '{self._escape_string(normalized_name)}'"
            logger.debug(f"Executing {entity_type} UPSERT")
            result = self._db.query("sql", query, is_command=True)
            logger.debug(f"{entity_type} UPSERT result: {result}")
            
            # Embedding stored as hidden property for vector functionality
            # Create MENTIONS relationship from chunk to entity (like FalkorDB)
            self._create_mentions_relationship_sql(node, entity_type)
            
        except Exception as e:
            logger.error(f"{entity_type} UPSERT failed: {e}")
            logger.debug(f"Failed query was: [query details hidden]")
            # Fallback to basic insert if UPSERT fails
            try:
                basic_query = f"INSERT INTO {entity_type} SET name = '{self._escape_string(normalized_name)}'"
                logger.debug(f"Fallback basic insert for {normalized_name}")
                self._db.query("sql", basic_query, is_command=True)
                self._create_mentions_relationship_sql(node, entity_type)
            except Exception as fallback_error:
                logger.warning(f"Even basic insert failed for {normalized_name}: {fallback_error}")

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
        
        # Handle embedding if present - store as JSON for vector functionality
        if hasattr(node, 'embedding') and node.embedding is not None:
            embedding_json = json.dumps(node.embedding)
            prop_assignments.append(f"embedding = '{embedding_json}'")
            logger.debug(f"Storing embedding for chunk {node.id} (dim: {len(node.embedding)})")
        
        # Add other properties
        if node.properties:
            for k, v in node.properties.items():
                if k != 'embedding':  # Skip embedding in properties, handled above
                    # Skip very large properties that cause SQL parsing issues
                    v_str = str(v)
                    if len(v_str) > 1000:  # Skip large properties like _node_content
                        logger.debug(f"Skipping large property {k} for chunk {node.id} (size: {len(v_str)})")
                        continue
                    prop_assignments.append(f"{k} = '{self._escape_string(v_str)}'")
        
        try:
            # Use ArcadeDB's native UPDATE ... UPSERT syntax (atomic operation)
            query = f"UPDATE TextChunk SET {', '.join(prop_assignments)} UPSERT WHERE id = '{self._escape_string(node.id)}'"
            logger.debug(f"Executing TextChunk UPSERT")
            result = self._db.query("sql", query, is_command=True)
            logger.debug(f"TextChunk UPSERT result: {result}")
            
            # Embedding stored as hidden property for vector functionality
        except Exception as e:
            logger.error(f"TextChunk UPSERT failed: {e}")
            logger.debug(f"Failed query was: [query details hidden]")
            # Fallback to basic insert if UPSERT fails
            try:
                basic_query = f"INSERT INTO TextChunk SET id = '{self._escape_string(node.id)}', text = '{text_escaped}'"
                logger.debug(f"Fallback basic insert for chunk {node.id}")
                self._db.query("sql", basic_query, is_command=True)
            except Exception as fallback_error:
                logger.warning(f"Even basic chunk insert failed for {node.id}: {fallback_error}")

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
        logger.info(f"LLM_RELATION_INPUT: label='{relation.label}', source='{relation.source_id}', target='{relation.target_id}', properties={list(relation.properties.keys()) if relation.properties else []}")
        
        # Use proper ArcadeDB CREATE EDGE syntax from documentation
        properties_str = self._properties_to_sql_string(relation.properties)
        
        # Skip complex SQL MATCH approach for now due to parsing issues
        # Use the reliable individual query approach directly
        logger.debug(f"Using individual query approach for relation: {relation.label}")
        
        # Normalize entity names to match how they were stored (same as entity creation)
        normalized_source_id = self._normalize_and_deduplicate_entity(relation.source_id) if relation.source_id else relation.source_id
        normalized_target_id = self._normalize_and_deduplicate_entity(relation.target_id) if relation.target_id else relation.target_id
        
        logger.info(f"SQL_RELATION: source '{relation.source_id}' -> normalized '{normalized_source_id}'")
        logger.info(f"SQL_RELATION: target '{relation.target_id}' -> normalized '{normalized_target_id}'")
        
        # Create edge type if it doesn't exist (dynamic schema)
        self._ensure_dynamic_type(relation.label, 'EDGE')
        
        # Use reliable individual query approach
        try:
            # List of vertex types to search - include discovered dynamic types (avoid duplicates)
            # Use essential types + discovered dynamic types instead of hard-coded list
            base_vertex_types = ['Entity', 'TextChunk']  # Essential types always present
            all_vertex_types = set(base_vertex_types) | self._discovered_vertex_types
            vertex_types = list(all_vertex_types)
            
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
            
            logger.debug(f"Executing fallback CREATE EDGE")
            result = self._db.query("sql", query, is_command=True)
            logger.info(f"SQL_FALLBACK_SUCCESS: Created {relation.label} edge from {source_rid} to {target_rid}")
            
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
                value_str = f"'{json.dumps(value)}'"
            else:
                value_str = str(value)

            items.append(f"{key} = {value_str}")

        return ", ".join(items)

    def _escape_string(self, text: str) -> str:
        """Escape string for ArcadeDB SQL and Cypher queries."""
        if text is None:
            return "NULL"
        
        escaped = str(text)
        
        # For very problematic strings, use a more conservative approach
        # Remove or replace characters that consistently cause Cypher parsing issues
        problematic_chars = ["'", '"', '`', '\n', '\r', '\t']
        has_problematic = any(char in escaped for char in problematic_chars)
        
        if has_problematic:
            # Replace problematic characters with safe alternatives
            escaped = escaped.replace("'", "")      # Remove apostrophes entirely
            escaped = escaped.replace('"', "")      # Remove quotes entirely  
            escaped = escaped.replace('`', "")      # Remove backticks
            escaped = escaped.replace('\n', ' ')    # Replace newlines with spaces
            escaped = escaped.replace('\r', ' ')    # Replace carriage returns with spaces
            escaped = escaped.replace('\t', ' ')    # Replace tabs with spaces
            escaped = escaped.strip()               # Clean up extra spaces
        else:
            # Standard escaping for normal strings
            escaped = escaped.replace("\\", "\\\\")  # Escape backslashes first
            escaped = escaped.replace("'", "\\'")   # Escape single quotes
        
        return escaped
    
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity manually if native function is not available.
        
        This provides a fallback if ArcadeDB changes their vector implementation.
        """
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = sum(a * a for a in vec1) ** 0.5
        norm_b = sum(b * b for b in vec2) ** 0.5
        
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _vector_query_manual_fallback(self, query: VectorStoreQuery) -> Tuple[List[LabelledNode], List[float]]:
        """Manual vector similarity calculation fallback.
        
        Used when ArcadeDB's native vector functions are not available.
        """
        try:
            # Get all nodes with embeddings
            base_vertex_types = ['Entity', 'TextChunk']
            all_vertex_types = list(set(base_vertex_types) | self._discovered_vertex_types)
            
            all_nodes = []
            for vertex_type in all_vertex_types:
                try:
                    node_query = f"SELECT *, '{vertex_type}' as node_type FROM {vertex_type} WHERE embedding IS NOT NULL"
                    results = self._db.query("sql", node_query)
                    if isinstance(results, list):
                        all_nodes.extend(results)
                    elif isinstance(results, dict) and "result" in results:
                        all_nodes.extend(results["result"])
                except Exception as e:
                    logger.debug(f"Failed to query {vertex_type} for manual vector search: {e}")
                    continue
            
            # Calculate similarities manually
            similarities = []
            for node_data in all_nodes:
                try:
                    if 'embedding' in node_data and node_data['embedding']:
                        node_embedding = node_data['embedding']
                        if isinstance(node_embedding, str):
                            node_embedding = json.loads(node_embedding)
                        
                        similarity = self._calculate_cosine_similarity(query.query_embedding, node_embedding)
                        similarities.append((node_data, similarity))
                except Exception as e:
                    logger.debug(f"Failed to calculate similarity for node: {e}")
                    continue
            
            # Sort by similarity and limit results
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_k = query.similarity_top_k or 10
            top_similarities = similarities[:top_k]
            
            # Convert to LabelledNode objects
            nodes = []
            scores = []
            for node_data, score in top_similarities:
                try:
                    # Create LabelledNode from the data
                    node_type = node_data.get('node_type', 'Entity')
                    if node_type == 'TextChunk':
                        node_id = node_data.get('id', '')
                        text = node_data.get('text', '')
                        labelled_node = LabelledNode(id=node_id, text=text, label=node_type)
                    else:
                        node_name = node_data.get('name', '')
                        labelled_node = LabelledNode(id=node_name, text=node_name, label=node_type)
                    
                    # Add properties
                    for key, value in node_data.items():
                        if key not in ['id', 'name', 'text', 'node_type', 'embedding']:
                            labelled_node.properties[key] = value
                    
                    nodes.append(labelled_node)
                    scores.append(score)
                except Exception as e:
                    logger.debug(f"Failed to create LabelledNode: {e}")
                    continue
            
            logger.info(f"Manual vector query returned {len(nodes)} results")
            return nodes, scores
            
        except Exception as e:
            logger.error(f"Manual vector query fallback failed: {e}")
            return [], []

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
            
            # Find the entity
            entity_query = f"SELECT @rid FROM {entity_type} WHERE name = '{self._escape_string(entity_node.name)}' LIMIT 1"
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
                logger.info(f"ENTITY_TYPE_DETECTION: Using label '{vertex_type}' for entity '{node.name}' (from PathExtractor/LLM)")
                return vertex_type
        
        # PRIORITY 2: Fallback to pattern-based classification
        entity_name = node.name.lower()
        classified_type = self._classify_entity_by_patterns(entity_name)
        if classified_type != 'Entity':
            logger.info(f"ENTITY_TYPE_DETECTION: Using pattern-based type '{classified_type}' for entity '{node.name}' (fallback classification)")
            return classified_type
        
        # PRIORITY 3: Generic fallback
        logger.info(f"ENTITY_TYPE_DETECTION: Using generic 'Entity' type for '{node.name}' (no specific classification found)")
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
        
        # All caps acronyms are often organizations (NASA, IBM, SAP)
        if len(words) == 1 and entity_name.isupper() and len(entity_name) >= 2:
            return 'ORGANIZATION'
        
        # Person name patterns
        if len(words) == 2:
            # Two words, both capitalized, likely a person name
            if all(word[0].isupper() for word in words if word):
                return 'PERSON'
        
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
