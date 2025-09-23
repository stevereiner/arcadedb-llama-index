# LlamaIndex Graph_Stores Integration: ArcadeDB

ArcadeDB is a Multi-Model DBMS that supports Graph, Document, Key-Value, Vector, and Time-Series models in a single engine. It's designed to be fast, scalable, and easy to use, making it an excellent choice for GraphRAG applications.

This integration provides both basic graph store and property graph store implementations for ArcadeDB, enabling LlamaIndex to work with ArcadeDB as a graph database backend with full vector search capabilities.

## Features

- **Multi-Model Support**: Graph, Document, Key-Value, Vector, and Time-Series in one database
- **High Performance**: Native SQL with graph traversal capabilities
- **Vector Search**: Built-in vector similarity search capabilities
- **Schema Flexibility**: Dynamic schema creation and management
- **Production Ready**: ACID transactions, clustering, and enterprise features

## Installation

```shell
pip install llama-index-graph-stores-arcadedb
```

## Usage

### Property Graph Store (Recommended)

The property graph store is the recommended approach for most GraphRAG applications:

```python
from llama_index.graph_stores.arcadedb import ArcadeDBPropertyGraphStore
from llama_index.core import PropertyGraphIndex

# For OpenAI embeddings (ada-002)
graph_store = ArcadeDBPropertyGraphStore(
    host="localhost",
    port=2480,
    username="root",
    password="playwithdata",
    database="knowledge_graph",
    embedding_dimension=1536  # OpenAI text-embedding-ada-002
)

# For Ollama embeddings (all-MiniLM-L6-v2 - common in flexible-graphrag)
graph_store = ArcadeDBPropertyGraphStore(
    host="localhost",
    port=2480,
    username="root",
    password="playwithdata",
    database="knowledge_graph",
    embedding_dimension=384   # Ollama all-MiniLM-L6-v2
)

# Or omit embedding_dimension to disable vector operations
graph_store = ArcadeDBPropertyGraphStore(
    host="localhost",
    port=2480,
    username="root",
    password="playwithdata",
    database="knowledge_graph"
    # No embedding_dimension = no vector search
)

# Create a property graph index
index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
    show_progress=True
)

# Query the graph
response = index.query("What are the main topics discussed?")
```

### Basic Graph Store

For simpler use cases, you can use the basic graph store:

```python
from llama_index.graph_stores.arcadedb import ArcadeDBGraphStore
from llama_index.core import KnowledgeGraphIndex

# Initialize the graph store
graph_store = ArcadeDBGraphStore(
    host="localhost",
    port=2480,
    username="root",
    password="playwithdata",
    database="knowledge_graph"
)

# Create a knowledge graph index
index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=StorageContext.from_defaults(graph_store=graph_store)
)
```

## Configuration

### Connection Parameters

- `host`: ArcadeDB server hostname (default: "localhost")
- `port`: ArcadeDB server port (default: 2480)
- `username`: Database username (default: "root")
- `password`: Database password
- `database`: Database name
- `embedding_dimension`: Vector dimension for embeddings (optional)

### Query Engine

The property graph store uses **native ArcadeDB SQL** for optimal performance and reliability. ArcadeDB's SQL engine provides excellent graph traversal capabilities with MATCH patterns and is the recommended approach for production use.

### Embedding Dimensions

Choose the correct `embedding_dimension` based on your embedding model:

| **Model** | **Dimension** | **Example** | **Usage** |
|-----------|---------------|-------------|-----------|
| OpenAI text-embedding-ada-002 | 1536 | `embedding_dimension=1536` | Production OpenAI |
| Ollama all-MiniLM-L6-v2 | 384 | `embedding_dimension=384` | **flexible-graphrag default** |
| Ollama nomic-embed-text | 768 | `embedding_dimension=768` | Alternative Ollama |
| Ollama mxbai-embed-large | 1024 | `embedding_dimension=1024` | High-quality Ollama |
| No vector search | None | Omit parameter entirely | Graph-only mode |

## Requirements

- ArcadeDB server (version 23.10+)
- Python 3.9+
- LlamaIndex core

## Getting Started

1. Start ArcadeDB server:
```bash
docker run -d --name arcadedb -p 2480:2480 -p 2424:2424 \
  -e JAVA_OPTS="-Darcadedb.server.rootPassword=playwithdata" \
  arcadedata/arcadedb:latest
```

2. Install the package:
```bash
pip install llama-index-graph-stores-arcadedb
```

3. Run your GraphRAG application!

## Examples

Check out the [examples directory](examples/) for complete working examples including:
- Basic usage with document ingestion
- Advanced GraphRAG workflows
- Vector similarity search
- Migration from other graph databases

## License

Apache License 2.0
