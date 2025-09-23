# LlamaIndex ArcadeDB Graph Store Integration

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![ArcadeDB](https://img.shields.io/badge/ArcadeDB-24.4.1+-green.svg)](https://arcadedb.com/)

A PropertyGraphStore implementation for [LlamaIndex](https://developers.llamaindex.ai/python/framework/) that integrates with [ArcadeDB](https://arcadedb.com/), enabling GraphRAG applications with multi-model database capabilities.

## Features

- **PropertyGraphStore Interface**: Full implementation of LlamaIndex's PropertyGraphStore interface
- **Multi-Model Database**: Graph, Document, Key-Value, and Vector data models in one database
- **Native SQL Support**: Uses ArcadeDB's high-performance SQL engine for graph operations
- **Vector Search**: Built-in vector similarity search capabilities
- **Dynamic Schema**: Automatic schema creation and management
- **Production Ready**: Docker support, comprehensive testing, and error handling

## Requirements

- **Python**: 3.10 or higher
- **ArcadeDB Server**: 24.4.1 or higher
- **arcadedb-python**: Python client for ArcadeDB (automatically installed as dependency)
- **LlamaIndex**: Core components for PropertyGraph functionality

## Installation

### 1. Install ArcadeDB Server

**Docker (Recommended):**
```bash
docker run --rm -p 2480:2480 -p 2424:2424 \
  -e JAVA_OPTS="-Darcadedb.server.rootPassword=playwithdata" \
  arcadedata/arcadedb:latest
```

**Manual Installation:**
```bash
# Download and extract ArcadeDB
wget https://github.com/ArcadeData/arcadedb/releases/latest/download/arcadedb-latest.tar.gz
tar -xf arcadedb-latest.tar.gz
cd arcadedb-*

# Start server
bin/server.sh -Darcadedb.server.rootPassword=playwithdata
```

### 2. Install Python Dependencies

```bash
# Install from PyPI (coming soon)
pip install llama-index-graph-stores-arcadedb

# Or install from source
pip install -e ./graph_stores/llama-index-graph-stores-arcadedb

# Install LlamaIndex components
pip install llama-index-core llama-index-embeddings-openai llama-index-llms-openai
```

## Usage

### Basic Example

```python
import os
from llama_index.core import PropertyGraphIndex, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.graph_stores.arcadedb import ArcadeDBPropertyGraphStore

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Initialize ArcadeDB graph store
graph_store = ArcadeDBPropertyGraphStore(
    host="localhost",
    port=2480,
    username="root",
    password="playwithdata",
    database="knowledge_graph",
    embedding_dimension=1536  # OpenAI text-embedding-ada-002
)

# Create documents
documents = [
    Document(text="Apple Inc. was founded by Steve Jobs in 1976."),
    Document(text="Steve Jobs was the CEO of Apple until 2011."),
    Document(text="Tim Cook became Apple's CEO after Steve Jobs."),
]

# Create PropertyGraphIndex
index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
    embed_model=OpenAIEmbedding(model_name="text-embedding-ada-002"),
    show_progress=True,
)

# Query the knowledge graph
query_engine = index.as_query_engine()
response = query_engine.query("Who founded Apple?")
print(response)
```

### Class Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | str | "localhost" | ArcadeDB server host |
| `port` | int | 2480 | ArcadeDB server port |
| `username` | str | "root" | Database username |
| `password` | str | "playwithdata" | Database password |
| `database` | str | "graph" | Database name |
| `create_database_if_not_exists` | bool | True | Create database if it doesn't exist |
| `include_basic_schema` | bool | True | Include basic entity types (PERSON, ORGANIZATION, LOCATION, PLACE) |
| `embedding_dimension` | int | None | Vector dimension for embeddings (optional) |

### Schema Configuration

The PropertyGraphStore creates schema types from multiple sources:

**Always Created (Core Types):**
- `Entity`, `TextChunk` (vertex types)
- `MENTIONS` (edge type for chunk-to-entity relationships)

**Basic Schema Types (`include_basic_schema=True`):**
- `PERSON`, `ORGANIZATION`, `LOCATION`, `PLACE` (common entity types)

**Dynamic Types from LlamaIndex/LLMs:**
- Additional entity and relationship types discovered during processing
- Custom types from SchemaLLMPathExtractor configurations
- Types inferred by LLMs during knowledge graph extraction

```python
# Default configuration - includes basic entity types
graph_store = ArcadeDBPropertyGraphStore(
    host="localhost",
    port=2480,
    username="root",
    password="playwithdata",
    database="knowledge_graph",
    include_basic_schema=True  # Pre-creates common entity types
)

# Minimal configuration - only core types, let LlamaIndex handle schema
graph_store = ArcadeDBPropertyGraphStore(
    host="localhost",
    port=2480,
    username="root",
    password="playwithdata",
    database="knowledge_graph",
    include_basic_schema=False  # Only creates Entity, TextChunk, MENTIONS
)
```

### Embedding Dimensions

Choose the correct `embedding_dimension` based on your embedding model:

| Model | Dimension | Usage |
|-------|-----------|-------|
| OpenAI text-embedding-ada-002 | 1536 | Production OpenAI |
| Ollama all-MiniLM-L6-v2 | 384 | Local embeddings |
| OpenAI text-embedding-3-small | 1536 | Cost-effective OpenAI |
| OpenAI text-embedding-3-large | 3072 | High-performance OpenAI |

## Advanced Usage

### Direct Graph Operations

```python
# Get nodes by properties
apple_nodes = graph_store.get(properties={"name": "Apple"})

# Get relationships
triplets = graph_store.get_triplets(entity_names=["Apple"])

# Execute custom SQL queries
results = graph_store.structured_query("""
    SELECT person.name as founder, company.name as company
    FROM Person person, Company company, FOUNDED rel
    WHERE rel.out = person.@rid AND rel.in = company.@rid
""")
```


## Docker Deployment

### Docker Compose

```yaml
version: '3.8'
services:
  arcadedb:
    image: arcadedata/arcadedb:latest
    ports:
      - "2480:2480"
      - "2424:2424"
    environment:
      - JAVA_OPTS=-Darcadedb.server.rootPassword=playwithdata
    volumes:
      - arcadedb_data:/home/arcadedb/databases

  app:
    build: .
    depends_on:
      - arcadedb
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ARCADEDB_HOST=arcadedb
      - ARCADEDB_PORT=2480

volumes:
  arcadedb_data:
```

## Testing

### Run Tests

```bash
# Install test dependencies
uv pip install pytest pytest-asyncio

# Run tests
pytest tests/ -v
```

### Integration Tests (Requires Running ArcadeDB)

```bash
# Start ArcadeDB server first
docker run --rm -p 2480:2480 -p 2424:2424 \
  -e JAVA_OPTS="-Darcadedb.server.rootPassword=playwithdata" \
  arcadedata/arcadedb:latest

# Run integration tests
pytest tests/test_final_integration.py -v
```

## Examples

See the `examples/` directory for complete working examples:

- `basic_usage.py` - Simple PropertyGraphStore usage
- `advanced_usage.py` - Custom schema and vector search
- `migration_from_neo4j.py` - Migration guide from Neo4j

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## References

- [ArcadeDB](https://arcadedb.com/) - Multi-model database
- [LlamaIndex Framework](https://developers.llamaindex.ai/python/framework/) - Framework for building LLM applications
- [arcadedb-python](https://github.com/your-org/arcadedb-python) - Python client for ArcadeDB (available on PyPI and GitHub)

## Support

- **Documentation**: [ArcadeDB Docs](https://docs.arcadedb.com/)
- **Issues**: [GitHub Issues](https://github.com/your-org/llama-index-graph-stores-arcadedb/issues)