"""ArcadeDB Property Graph Store implementation for LlamaIndex - Clean Version.

This module provides a clean, focused PropertyGraphStore implementation for ArcadeDB,
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
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from arcadedb_python.api.sync import SyncClient
    from arcadedb_python.dao.database import DatabaseDao
except ImportError as e:
    raise ImportError(
        "arcadedb_python is required for ArcadeDB integration. "
        "Install it with: pip install arcadedb-python"
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


class ArcadeDBPropertyGraphStoreClean(PropertyGraphStore):
    """ArcadeDB Property Graph Store - Clean Version.

    This class implements the PropertyGraphStore interface for ArcadeDB using native SQL,
    providing optimal performance for graph operations with a clean, focused codebase.

    Args:
        host: ArcadeDB server host (default: "localhost")
        port: ArcadeDB server port (default: 2480)
        username: Database username (default: "root")
        password: Database password (default: "playwithdata")
        database: Database name (default: "graph")
        create_database_if_not_exists: Whether to create database if it doesn't exist (default: True)
        enable_dynamic_schema: Whether to create types dynamically (default: True)
        embedding_dimension: Optional dimension for vector embeddings
        **kwargs: Additional arguments
    """

    supports_vector_queries: bool = True

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2480,
        username: str = "root",
        password: str = "playwithdata",
        database: str = "graph",
        create_database_if_not_exists: bool = True,
        enable_dynamic_schema: bool = True,
        embedding_dimension: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the ArcadeDB Property Graph Store."""
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.enable_dynamic_schema = enable_dynamic_schema
        self.embedding_dimension = embedding_dimension
        
        # Initialize ArcadeDB client
        self._sync_client = SyncClient(
            host=host,
            port=str(port),
            username=username,
            password=password,
            content_type="application/json"
        )
        
        # Ensure database exists first
        if create_database_if_not_exists:
            self._ensure_database()
        
        # Initialize database DAO (after ensuring database exists)
        self._client = DatabaseDao(self._sync_client, database)
        
        # Initialize schema
        self._initialize_schema()
        
        logger.info(f"Initialized ArcadeDB PropertyGraphStore for database '{database}'")

    @property
    def client(self) -> Any:
        """Get the database client."""
        return self._client

    def _ensure_database(self) -> None:
        """Ensure the database exists."""
        try:
            from arcadedb_python.dao.database import DatabaseDao
            databases = DatabaseDao.list_databases(self._sync_client)
            if self.database not in [db.get("name") for db in databases]:
                DatabaseDao.create(self._sync_client, self.database)
                logger.info(f"Created database: {self.database}")
        except Exception as e:
            logger.warning(f"Database check/creation failed: {e}")

    def _initialize_schema(self) -> None:
        """Initialize essential schema types."""
        try:
            # Essential base types
            base_types = [
                "Entity",
                "TextChunk", 
                "PERSON",
                "ORGANIZATION", 
                "LOCATION",
                "PLACE"
            ]
            
            for vertex_type in base_types:
                self._ensure_vertex_type(vertex_type)
            
            # Essential edge types
            self._ensure_edge_type("MENTIONS")
            
            logger.debug("Schema initialization completed")
            
        except Exception as e:
            logger.warning(f"Schema initialization failed: {e}")

    def _ensure_vertex_type(self, type_name: str) -> None:
        """Ensure a vertex type exists."""
        try:
            create_sql = f"CREATE VERTEX TYPE IF NOT EXISTS {type_name}"
            self._client.query(
                language="sql",
                command=create_sql,
                is_command=True
            )
        except Exception as e:
            logger.debug(f"Failed to create vertex type {type_name}: {e}")

    def _ensure_edge_type(self, type_name: str) -> None:
        """Ensure an edge type exists."""
        try:
            create_sql = f"CREATE EDGE TYPE IF NOT EXISTS {type_name}"
            self._client.query(
                language="sql",
                command=create_sql,
                is_command=True
            )
        except Exception as e:
            logger.debug(f"Failed to create edge type {type_name}: {e}")

    def _ensure_dynamic_type(self, type_name: str, is_edge: bool = False) -> None:
        """Dynamically create vertex or edge type if enabled."""
        if not self.enable_dynamic_schema:
            return
            
        try:
            if is_edge:
                self._ensure_edge_type(type_name)
            else:
                self._ensure_vertex_type(type_name)
        except Exception as e:
            logger.debug(f"Dynamic type creation failed for {type_name}: {e}")

    def upsert_nodes(self, nodes: Sequence[LabelledNode]) -> None:
        """Add or update nodes in the graph."""
        for node in nodes:
            try:
                if isinstance(node, EntityNode):
                    self._upsert_entity_node(node)
                elif isinstance(node, ChunkNode):
                    self._upsert_chunk_node(node)
                else:
                    logger.warning(f"Unknown node type: {type(node)}")
            except Exception as e:
                logger.error(f"Failed to upsert node {getattr(node, 'id', getattr(node, 'name', 'unknown'))}: {e}")

    def _upsert_entity_node(self, node: EntityNode) -> None:
        """Upsert an entity node."""
        # Determine entity type
        entity_type = self._determine_entity_type(node)
        self._ensure_dynamic_type(entity_type)
        
        # Prepare properties
        properties = {
            "name": node.name,
            "label": node.label or node.name,
            **node.properties
        }
        
        # Add embedding if present
        if hasattr(node, 'embedding') and node.embedding:
            properties["embedding"] = json.dumps(node.embedding)
        
        # Build upsert query
        property_assignments = []
        parameters = []
        
        for key, value in properties.items():
            if key != 'embedding':  # Handle embedding separately to avoid escaping issues
                property_assignments.append(f"{key} = ?")
                parameters.append(str(value))
        
        # Add embedding separately if present
        if "embedding" in properties:
            property_assignments.append("embedding = ?")
            parameters.append(properties["embedding"])
        
        upsert_sql = f"""
            UPDATE {entity_type} SET {', '.join(property_assignments)} 
            UPSERT WHERE name = ? RETURN @this
        """
        parameters.append(node.name)
        
        try:
            self._client.query(
                language="sql",
                command=upsert_sql,
                params=parameters,
                is_command=True
            )
            logger.debug(f"Upserted entity node: {node.name} ({entity_type})")
        except Exception as e:
            logger.error(f"Failed to upsert entity node {node.name}: {e}")

    def _upsert_chunk_node(self, node: ChunkNode) -> None:
        """Upsert a chunk node."""
        self._ensure_dynamic_type("TextChunk")
        
        # Prepare properties
        properties = {
            "id": node.id,
            "label": node.label or f"Chunk {node.id}",
            "text": getattr(node, 'text', ''),
            **node.properties
        }
        
        # Add embedding if present
        if hasattr(node, 'embedding') and node.embedding:
            properties["embedding"] = json.dumps(node.embedding)
        
        # Build upsert query
        property_assignments = []
        parameters = []
        
        for key, value in properties.items():
            if key != 'embedding':  # Handle embedding separately
                property_assignments.append(f"{key} = ?")
                parameters.append(str(value))
        
        # Add embedding separately if present
        if "embedding" in properties:
            property_assignments.append("embedding = ?")
            parameters.append(properties["embedding"])
        
        upsert_sql = f"""
            UPDATE TextChunk SET {', '.join(property_assignments)} 
            UPSERT WHERE id = ? RETURN @this
        """
        parameters.append(node.id)
        
        try:
            self._client.query(
                language="sql",
                command=upsert_sql,
                params=parameters,
                is_command=True
            )
            logger.debug(f"Upserted chunk node: {node.id}")
        except Exception as e:
            logger.error(f"Failed to upsert chunk node {node.id}: {e}")

    def _determine_entity_type(self, node: EntityNode) -> str:
        """Determine the appropriate type for an entity node."""
        # Check if type is specified in properties
        if 'type' in node.properties:
            return node.properties['type'].upper()
        
        # Check label for common patterns
        label_lower = (node.label or node.name).lower()
        
        if any(word in label_lower for word in ['person', 'people', 'individual', 'human']):
            return 'PERSON'
        elif any(word in label_lower for word in ['company', 'organization', 'org', 'corporation', 'business']):
            return 'ORGANIZATION'
        elif any(word in label_lower for word in ['place', 'location', 'city', 'country', 'region']):
            return 'LOCATION'
        elif any(word in label_lower for word in ['technology', 'tech', 'software', 'system']):
            return 'TECHNOLOGY'
        elif any(word in label_lower for word in ['project', 'initiative', 'program']):
            return 'PROJECT'
        
        # Default to generic Entity type
        return 'Entity'

    def upsert_relations(self, relations: Sequence[Relation]) -> None:
        """Add or update relations in the graph."""
        for relation in relations:
            try:
                self._upsert_relation(relation)
            except Exception as e:
                logger.error(f"Failed to upsert relation {relation.label}: {e}")

    def _upsert_relation(self, relation: Relation) -> None:
        """Upsert a single relation."""
        # Normalize relation type
        relation_type = relation.label.replace(" ", "_").replace("-", "_").upper()
        self._ensure_dynamic_type(relation_type, is_edge=True)
        
        # Find source and target vertices
        vertex_types = ["Entity", "PERSON", "ORGANIZATION", "LOCATION", "TECHNOLOGY", "PROJECT", "TextChunk"]
        
        try:
            # Find source vertex
            source_rid = None
            for vertex_type in vertex_types:
                try:
                    if vertex_type == "TextChunk":
                        query = f"SELECT @rid FROM {vertex_type} WHERE id = ? LIMIT 1"
                    else:
                        query = f"SELECT @rid FROM {vertex_type} WHERE name = ? LIMIT 1"
                    
                    result = self._client.query(
                        language="sql",
                        command=query,
                        params=[relation.source_id]
                    )
                    
                    if result and len(result) > 0:
                        source_rid = result[0]['@rid']
                        break
                except Exception:
                    continue
            
            # Find target vertex
            target_rid = None
            for vertex_type in vertex_types:
                try:
                    if vertex_type == "TextChunk":
                        query = f"SELECT @rid FROM {vertex_type} WHERE id = ? LIMIT 1"
                    else:
                        query = f"SELECT @rid FROM {vertex_type} WHERE name = ? LIMIT 1"
                    
                    result = self._client.query(
                        language="sql",
                        command=query,
                        params=[relation.target_id]
                    )
                    
                    if result and len(result) > 0:
                        target_rid = result[0]['@rid']
                        break
                except Exception:
                    continue
            
            if not source_rid or not target_rid:
                logger.warning(f"Could not find vertices for relation: {relation.source_id} -> {relation.target_id}")
                return
            
            # Create edge
            properties_str = ""
            if relation.properties:
                prop_items = []
                for key, value in relation.properties.items():
                    if isinstance(value, str):
                        prop_items.append(f"{key} = '{value}'")
                    else:
                        prop_items.append(f"{key} = {value}")
                properties_str = f"SET {', '.join(prop_items)}"
            
            create_edge_sql = f"CREATE EDGE {relation_type} FROM {source_rid} TO {target_rid} IF NOT EXISTS {properties_str}"
            
            self._client.query(
                language="sql",
                command=create_edge_sql,
                is_command=True
            )
            
            logger.debug(f"Upserted relation: {relation.source_id} -{relation.label}-> {relation.target_id}")
            
        except Exception as e:
            logger.error(f"Failed to create edge {relation.label}: {e}")

    def get(self, properties: Optional[dict] = None, ids: Optional[List[str]] = None) -> List[LabelledNode]:
        """Get nodes by properties or IDs."""
        nodes = []
        
        try:
            if ids:
                # Get nodes by IDs
                for node_id in ids:
                    node = self._get_node_by_id(node_id)
                    if node:
                        nodes.append(node)
            elif properties:
                # Get nodes by properties
                nodes = self._get_nodes_by_properties(properties)
            else:
                # Get all nodes (limited)
                nodes = self._get_all_nodes(limit=100)
                
        except Exception as e:
            logger.error(f"Failed to get nodes: {e}")
            
        return nodes

    def _get_node_by_id(self, node_id: str) -> Optional[LabelledNode]:
        """Get a single node by ID."""
        try:
            # Search across all vertex types
            vertex_types = ["Entity", "TextChunk", "PERSON", "ORGANIZATION", "LOCATION", "TECHNOLOGY", "PROJECT"]
            
            for vertex_type in vertex_types:
                if vertex_type == "TextChunk":
                    query = f"SELECT * FROM {vertex_type} WHERE id = ? LIMIT 1"
                else:
                    query = f"SELECT * FROM {vertex_type} WHERE name = ? LIMIT 1"
                
                result = self._client.query(
                    language="sql",
                    command=query,
                    params=[node_id]
                )
                
                if result and len(result) > 0:
                    return self._record_to_node(result[0], vertex_type)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get node {node_id}: {e}")
            return None

    def _get_nodes_by_properties(self, properties: dict) -> List[LabelledNode]:
        """Get nodes by properties."""
        nodes = []
        try:
            vertex_types = ["Entity", "TextChunk", "PERSON", "ORGANIZATION", "LOCATION", "TECHNOLOGY", "PROJECT"]
            
            for vertex_type in vertex_types:
                conditions = []
                parameters = []
                
                for key, value in properties.items():
                    conditions.append(f"{key} = ?")
                    parameters.append(value)
                
                if conditions:
                    query = f"SELECT * FROM {vertex_type} WHERE {' AND '.join(conditions)} LIMIT 50"
                    result = self._client.query(
                        language="sql",
                        command=query,
                        params=parameters
                    )
                    
                    for record in result:
                        node = self._record_to_node(record, vertex_type)
                        if node:
                            nodes.append(node)
            
        except Exception as e:
            logger.error(f"Failed to get nodes by properties: {e}")
            
        return nodes

    def _get_all_nodes(self, limit: int = 100) -> List[LabelledNode]:
        """Get all nodes with a limit."""
        nodes = []
        try:
            vertex_types = ["Entity", "TextChunk", "PERSON", "ORGANIZATION", "LOCATION", "TECHNOLOGY", "PROJECT"]
            
            for vertex_type in vertex_types:
                query = f"SELECT * FROM {vertex_type} LIMIT {limit // len(vertex_types)}"
                result = self._client.query(
                    language="sql",
                    command=query
                )
                
                for record in result:
                    node = self._record_to_node(record, vertex_type)
                    if node:
                        nodes.append(node)
            
        except Exception as e:
            logger.error(f"Failed to get all nodes: {e}")
            
        return nodes

    def _record_to_node(self, record: dict, vertex_type: str) -> Optional[LabelledNode]:
        """Convert a database record to a LabelledNode."""
        try:
            if vertex_type == "TextChunk":
                node_id = record.get("id")
                if not node_id:
                    return None
                
                text = record.get("text", "")
                label = record.get("label", node_id)
                properties = {k: v for k, v in record.items() if k not in ["id", "label", "embedding", "text"]}
                
                node = ChunkNode(id_=node_id, label=label, properties=properties, text=text)
            else:
                node_name = record.get("name")
                if not node_name:
                    return None
                
                label = record.get("label", node_name)
                properties = {k: v for k, v in record.items() if k not in ["name", "label", "embedding"]}
                
                node = EntityNode(name=node_name, label=label, properties=properties)
            
            # Handle embedding
            if "embedding" in record and record["embedding"]:
                try:
                    embedding = json.loads(record["embedding"])
                    node.embedding = embedding
                except (json.JSONDecodeError, TypeError):
                    pass
            
            return node
            
        except Exception as e:
            logger.error(f"Failed to convert record to node: {e}")
            return None

    def get_triplets(self, entity_names: Optional[List[str]] = None) -> List[Triplet]:
        """Get triplets from the graph."""
        triplets = []
        
        try:
            if entity_names:
                # Get triplets for specific entities
                entity_list = "', '".join(entity_names)
                query = f"""
                    SELECT 
                        in().name as subject,
                        @type as relation,
                        out().name as object
                    FROM E
                    WHERE in().name IN ['{entity_list}'] OR out().name IN ['{entity_list}']
                    LIMIT 1000
                """
            else:
                # Get all triplets (limited)
                query = """
                    SELECT 
                        in().name as subject,
                        @type as relation,
                        out().name as object
                    FROM E
                    LIMIT 1000
                """
            
            result = self._client.query(
                language="sql",
                command=query
            )
            
            for record in result:
                subject = record.get("subject")
                relation = record.get("relation")
                obj = record.get("object")
                
                if subject and relation and obj:
                    triplets.append(Triplet(subject, relation, obj))
            
        except Exception as e:
            logger.error(f"Failed to get triplets: {e}")
            
        return triplets

    def get_rel_map(
        self, 
        subjs: Optional[List[str]] = None, 
        depth: int = 2, 
        limit: int = 30
    ) -> Dict[str, List[List[str]]]:
        """Get relationship map for given subjects."""
        rel_map: Dict[str, List[List[str]]] = {}
        
        if not subjs:
            return rel_map
        
        try:
            # Build multi-hop relationship query
            subj_list = "', '".join(subjs)
            query = f"""
                MATCH (n)-[r*1..{depth}]->(m)
                WHERE n.name IN ['{subj_list}'] OR n.id IN ['{subj_list}']
                RETURN n.name as subject, r as relations, m.name as target
                LIMIT {limit}
            """
            
            result = self._client.query(
                language="sql",
                command=query
            )
            
            for record in result:
                subject = record.get("subject")
                relations = record.get("relations", [])
                target = record.get("target")
                
                if subject and relations and target:
                    if subject not in rel_map:
                        rel_map[subject] = []
                    
                    # Flatten relationship path
                    path = []
                    for rel in relations:
                        path.append(rel.get("@type", "UNKNOWN"))
                        path.append(target)
                    
                    rel_map[subject].append(path)
            
        except Exception as e:
            logger.error(f"Failed to get relationship map: {e}")
            
        return rel_map

    def vector_query(self, query: VectorStoreQuery, **kwargs: Any) -> Tuple[List[LabelledNode], List[float]]:
        """Perform vector similarity search."""
        if not query.query_embedding:
            return [], []
        
        try:
            # Get all nodes with embeddings
            vertex_types = ["Entity", "TextChunk", "PERSON", "ORGANIZATION", "LOCATION", "TECHNOLOGY", "PROJECT"]
            
            all_results = []
            
            for vertex_type in vertex_types:
                search_query = f"""
                    SELECT *, '{vertex_type}' as node_type
                    FROM {vertex_type} 
                    WHERE embedding IS NOT NULL
                    LIMIT 100
                """
                
                result = self._client.query(
                    language="sql",
                    command=search_query
                )
                
                for record in result:
                    try:
                        embedding_str = record.get("embedding")
                        if embedding_str:
                            embedding = json.loads(embedding_str)
                            similarity = self._cosine_similarity(query.query_embedding, embedding)
                            
                            node = self._record_to_node(record, vertex_type)
                            if node:
                                all_results.append((node, similarity))
                    except Exception as e:
                        logger.debug(f"Failed to process embedding for record: {e}")
            
            # Sort by similarity and take top results
            all_results.sort(key=lambda x: x[1], reverse=True)
            top_k = query.similarity_top_k or 10
            top_results = all_results[:top_k]
            
            nodes = [result[0] for result in top_results]
            similarities = [result[1] for result in top_results]
            
            return nodes, similarities
            
        except Exception as e:
            logger.error(f"Vector query failed: {e}")
            return [], []

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            if len(vec1) != len(vec2):
                return 0.0
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
            
        except Exception:
            return 0.0

    def delete(self, entity_names: List[str], relation_names: List[str]) -> None:
        """Delete entities and relations."""
        try:
            # Delete relations first
            for relation_name in relation_names:
                relation_type = relation_name.replace(" ", "_").replace("-", "_").upper()
                delete_sql = f"DELETE FROM {relation_type}"
                self._client.query(
                    language="sql",
                    command=delete_sql,
                    is_command=True
                )
            
            # Delete entities
            vertex_types = ["Entity", "TextChunk", "PERSON", "ORGANIZATION", "LOCATION", "TECHNOLOGY", "PROJECT"]
            
            for entity_name in entity_names:
                for vertex_type in vertex_types:
                    if vertex_type == "TextChunk":
                        delete_sql = f"DELETE FROM {vertex_type} WHERE id = ?"
                    else:
                        delete_sql = f"DELETE FROM {vertex_type} WHERE name = ?"
                    
                    self._client.query(
                        language="sql",
                        command=delete_sql,
                        params=[entity_name],
                        is_command=True
                    )
            
            logger.debug(f"Deleted entities: {entity_names}, relations: {relation_names}")
            
        except Exception as e:
            logger.error(f"Failed to delete entities/relations: {e}")

    def get_schema(self, refresh: bool = False) -> str:
        """Get the schema of the property graph store."""
        try:
            # Get vertex types
            vertex_query = "SELECT name FROM schema:types WHERE type = 'Vertex'"
            vertex_result = self._client.query(
                language="sql",
                command=vertex_query
            )
            
            # Get edge types  
            edge_query = "SELECT name FROM schema:types WHERE type = 'Edge'"
            edge_result = self._client.query(
                language="sql",
                command=edge_query
            )
            
            vertex_types = [r.get("name") for r in vertex_result]
            edge_types = [r.get("name") for r in edge_result]
            
            schema = f"""
            Vertex Types: {vertex_types}
            Edge Types: {edge_types}
            Vector Support: {self.supports_vector_queries}
            Embedding Dimension: {self.embedding_dimension or 'Not specified'}
            """
            
            return schema
            
        except Exception as e:
            logger.warning(f"Schema retrieval failed: {e}")
            return "Schema information unavailable"

    def structured_query(self, query: str, param_map: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a structured SQL query against the graph."""
        try:
            parameters = list(param_map.values()) if param_map else None
            results = self._client.query(
                language="sql",
                command=query,
                params=parameters
            )
            return results
        except Exception as e:
            logger.error(f"Structured query failed: {e}")
            return []
