"""ArcadeDB integration for LlamaIndex Graph Stores.

This package provides both basic graph store and property graph store implementations
for ArcadeDB, enabling powerful GraphRAG applications with vector search capabilities.

Classes:
    ArcadeDBGraphStore: Basic graph store for simple triplet operations
    ArcadeDBPropertyGraphStore: Advanced property graph store with vector support

Example usage:
    # Property Graph Store (recommended for most use cases)
    from llama_index.graph_stores.arcadedb import ArcadeDBPropertyGraphStore
    
    graph_store = ArcadeDBPropertyGraphStore(
        host="localhost",
        port=2480,
        username="root",
        password="playwithdata",
        database="knowledge_graph",
        embedding_dimension=1536  # OpenAI ada-002 (use 384 for Ollama all-MiniLM-L6-v2)
    )
    
    # Basic Graph Store (for simple triplet operations)
    from llama_index.graph_stores.arcadedb import ArcadeDBGraphStore
    
    graph_store = ArcadeDBGraphStore(
        host="localhost",
        port=2480,
        username="root",
        password="playwithdata",
        database="knowledge_graph"
    )
"""

from llama_index.graph_stores.arcadedb.base import ArcadeDBGraphStore
from llama_index.graph_stores.arcadedb.arcadedb_property_graph import ArcadeDBPropertyGraphStore

__all__ = [
    "ArcadeDBGraphStore",
    "ArcadeDBPropertyGraphStore",
]

__version__ = "0.3.1"
