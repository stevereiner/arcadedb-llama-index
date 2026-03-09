# LlamaIndex ArcadeDB Graph Store Integration

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![ArcadeDB](https://img.shields.io/badge/ArcadeDB-26.2.1+-green.svg)](https://arcadedb.com/)

A PropertyGraphStore implementation for [LlamaIndex](https://developers.llamaindex.ai/python/framework/) that integrates with [ArcadeDB](https://arcadedb.com/), enabling GraphRAG applications with multi-model database capabilities.

## Features

- **PropertyGraphStore Interface**: Full implementation of LlamaIndex's PropertyGraphStore interface
- **Two Connection Modes**: Remote HTTP/REST (Docker or standalone server) and embedded in-process (no server required)
- **Multi-Model Database**: Graph, Document, Key-Value, and Vector data models in one database
- **Native SQL Support**: Uses ArcadeDB's high-performance SQL engine for graph operations
- **Vector Search**: Built-in vector similarity search with native `ARRAY_OF_FLOATS` storage
- **Dynamic Schema**: Automatic schema creation and management
- **Embedded Studio UI**: Optional built-in HTTP server for ArcadeDB Studio access when using embedded mode

## Connection Modes

### Remote mode (default)

Connects to a running ArcadeDB server over HTTP/REST. This is the standard production setup — run ArcadeDB via Docker or as a standalone server, and connect from your application.

### Embedded mode

Runs ArcadeDB directly inside your Python process using the [`arcadedb-embedded`](https://pypi.org/project/arcadedb-embedded/) package, which bundles the ArcadeDB JARs and a JVM via [JPype](https://jpype.readthedocs.io/). No separate server process is needed.

Embedded mode has two sub-options:

- **Direct** (`embedded_server=False`, default): fastest, file-system access only. No HTTP port is opened.
- **With embedded server** (`embedded_server=True`): starts ArcadeDB's built-in HTTP server inside the process. The Studio web UI becomes available at `http://localhost:<embedded_server_port>/` for the lifetime of your application — useful for development and inspection without stopping anything.

> **Note**: The embedded JVM starts once per process and cannot be restarted. Keep this in mind if you instantiate multiple stores in the same process.

## Requirements

### Remote mode

- **Python**: 3.10 or higher
- **ArcadeDB Server**: 26.2.1 or higher
- **arcadedb-python**: Python HTTP client for ArcadeDB (installed automatically)
- **LlamaIndex**: Core components for PropertyGraph functionality

### Embedded mode (additional)

- **arcadedb-embedded**: Python/JPype bindings that bundle the ArcadeDB JARs and a JVM
  - Optional dependency — install separately (see below)
  - ~95 MB installed (bundled JRE ~60 MB + ArcadeDB JARs ~32 MB)
  - The package version equals the ArcadeDB version it embeds (e.g. `arcadedb-embedded==26.2.1` bundles ArcadeDB 26.2.1)

## Installation

### Remote mode

**1. Start ArcadeDB Server**

```bash
# Docker (recommended)
docker run --rm -p 2480:2480 -p 2424:2424 \
  -e JAVA_OPTS="-Darcadedb.server.rootPassword=playwithdata" \
  arcadedata/arcadedb:latest
```

```bash
# Or download and run manually
wget https://github.com/ArcadeData/arcadedb/releases/latest/download/arcadedb-latest.tar.gz
tar -xf arcadedb-latest.tar.gz && cd arcadedb-*
bin/server.sh -Darcadedb.server.rootPassword=playwithdata
```

**2. Install the package**

```bash
pip install llama-index-graph-stores-arcadedb
pip install llama-index-core llama-index-embeddings-openai llama-index-llms-openai
```

### Embedded mode

No ArcadeDB server installation needed. Install the optional `arcadedb-embedded` package:

```bash
pip install llama-index-graph-stores-arcadedb
pip install arcadedb-embedded>=26.2.1   # ~95 MB download
pip install llama-index-core llama-index-embeddings-openai llama-index-llms-openai
```

Or using the optional extras shorthand:

```bash
pip install "llama-index-graph-stores-arcadedb[embedded]"
```

## Usage

### Remote mode

```python
import os
from llama_index.core import PropertyGraphIndex, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.graph_stores.arcadedb import ArcadeDBPropertyGraphStore

os.environ["OPENAI_API_KEY"] = "your-api-key-here"

graph_store = ArcadeDBPropertyGraphStore(
    host="localhost",
    port=2480,
    username="root",
    password="playwithdata",
    database="knowledge_graph",
    embedding_dimension=1536,   # match your embedding model
)

documents = [
    Document(text="Apple Inc. was founded by Steve Jobs in 1976."),
    Document(text="Steve Jobs was the CEO of Apple until 2011."),
    Document(text="Tim Cook became Apple's CEO after Steve Jobs."),
]

index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
    embed_model=OpenAIEmbedding(model_name="text-embedding-ada-002"),
    show_progress=True,
)

query_engine = index.as_query_engine()
response = query_engine.query("Who founded Apple?")
print(response)
```

### Embedded mode — direct (no server)

The database is stored on disk at `db_path/database`. No HTTP port is opened.

```python
from llama_index.graph_stores.arcadedb import ArcadeDBPropertyGraphStore

graph_store = ArcadeDBPropertyGraphStore(
    mode="embedded",
    db_path="./arcadedb_data",      # root directory for database files
    database="knowledge_graph",     # sub-directory name
    embedding_dimension=1536,
)
```

### Embedded mode — with built-in HTTP server and Studio

Starts ArcadeDB's HTTP server inside the process. Studio is available at
`http://localhost:2482/` for the entire lifetime of your application.

```python
from llama_index.graph_stores.arcadedb import ArcadeDBPropertyGraphStore

graph_store = ArcadeDBPropertyGraphStore(
    mode="embedded",
    db_path="./arcadedb_data",
    database="knowledge_graph",
    embedding_dimension=1536,
    embedded_server=True,
    embedded_server_port=2482,      # avoid 2480 (default ArcadeDB) and 2481 (test container)
)

# Check the Studio URL
print(graph_store.embedded_server_url)   # http://localhost:2482/
```

Use port 2482 (not 2480/2481) to avoid conflicts with a running ArcadeDB Docker container or integration test containers.

### Inspect an existing embedded database in Studio

If your application already created a database at `./arcadedb_data/knowledge_graph` in embedded direct mode, you can open Studio against it afterwards using the helper script:

```bash
python examples/embedded_server.py \
    --db-path ./arcadedb_data \
    --database knowledge_graph \
    --port 2482 \
    --embedding-dim 1536
```

The script holds the server open until you press Ctrl+C.

### Class Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | `"remote"` | Connection mode: `"remote"` or `"embedded"` |
| `host` | str | `"localhost"` | *(remote)* ArcadeDB server host |
| `port` | int | `2480` | *(remote)* ArcadeDB server port |
| `username` | str | `"root"` | *(remote)* Database username |
| `password` | str | `"playwithdata"` | *(remote)* Database password |
| `database` | str | `"graph"` | Database name |
| `db_path` | str | `None` | *(embedded)* Root directory for database files |
| `embedded_server` | bool | `False` | *(embedded)* Start built-in HTTP server |
| `embedded_server_port` | int | `2480` | *(embedded)* HTTP server port |
| `embedded_server_password` | str | `None` | *(embedded)* Root password for the HTTP server |
| `create_database_if_not_exists` | bool | `True` | Create the database if it doesn't exist |
| `include_basic_schema` | bool | `True` | Pre-create common entity types (PERSON, ORGANIZATION, LOCATION, PLACE) |
| `embedding_dimension` | int | `None` | Vector dimension — required to enable vector search |

### Schema Configuration

The store creates schema types from multiple sources:

**Always created (core types):**
- `Entity`, `TextChunk` (vertex types)
- `MENTIONS` (edge type for chunk-to-entity relationships)

**Basic schema types (`include_basic_schema=True`):**
- `PERSON`, `ORGANIZATION`, `LOCATION`, `PLACE`

**Dynamic types from LlamaIndex/LLMs:**
- Additional entity and relationship types discovered during processing
- Custom types from `SchemaLLMPathExtractor` configurations

### Embedding Dimensions

| Model | Dimension |
|-------|-----------|
| OpenAI `text-embedding-ada-002` | 1536 |
| OpenAI `text-embedding-3-small` | 1536 |
| OpenAI `text-embedding-3-large` | 3072 |
| Ollama `nomic-embed-text` | 768 |
| Ollama `all-MiniLM-L6-v2` | 384 |

## Advanced Usage

### Direct Graph Operations

```python
# Get nodes by properties
nodes = graph_store.get(properties={"name": "Apple"})

# Get relationships
triplets = graph_store.get_triplets(entity_names=["Apple"])

# Execute custom SQL
results = graph_store.structured_query("""
    SELECT person.name AS founder, company.name AS company
    FROM PERSON person, ORGANIZATION company, FOUNDED rel
    WHERE rel.out = person.@rid AND rel.in = company.@rid
""")
```

## Docker Deployment (Remote Mode)

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

```bash
# Install test dependencies
uv pip install pytest pytest-asyncio

# All tests (remote integration tests require a running ArcadeDB on port 2481)
pytest tests/ -v

# Embedded tests only (no server required)
pytest tests/test_arcadedb_embedded_integration.py \
       tests/test_pg_stores_arcadedb_embedded.py \
       tests/test_graph_stores_arcadedb_embedded.py \
       tests/test_final_integration_embedded.py -v

# Remote integration tests (requires Docker)
docker run --rm -p 2481:2480 -p 2425:2424 \
  -e JAVA_OPTS="-Darcadedb.server.rootPassword=playwithdata" \
  arcadedata/arcadedb:latest

pytest tests/test_final_integration.py -v
```

## Examples

See the `examples/` directory:

- `basic_usage.py` — Simple PropertyGraphStore usage (remote mode)
- `advanced_usage.py` — Custom schema and vector search (remote mode)
- `embedded_server.py` — Start a Studio-accessible embedded server against an existing database

## License

Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

## References

- [ArcadeDB](https://arcadedb.com/) — Multi-model database
- [LlamaIndex Framework](https://developers.llamaindex.ai/python/framework/) — Framework for building LLM applications
- [arcadedb-python](https://pypi.org/project/arcadedb-python/) — Python HTTP client for ArcadeDB
- [arcadedb-embedded](https://pypi.org/project/arcadedb-embedded/) — Python/JPype bindings for embedded ArcadeDB

## Support

- **Documentation**: [ArcadeDB Docs](https://docs.arcadedb.com/)
- **Issues**: [GitHub Issues](https://github.com/your-org/llama-index-graph-stores-arcadedb/issues)
