"""ArcadeDB integration for LlamaIndex Graph Stores.

This package provides both basic graph store and property graph store implementations
for ArcadeDB, enabling powerful GraphRAG applications with vector search capabilities.

Two connection modes are supported by ``ArcadeDBPropertyGraphStore``:

* ``mode="remote"`` (default) — connects to a running ArcadeDB server via HTTP/REST.
* ``mode="embedded"`` — runs ArcadeDB in-process (no server required).
  Requires ``pip install arcadedb-embedded`` and a ``db_path`` argument.

Classes:
    ArcadeDBGraphStore: Basic graph store for simple triplet operations
    ArcadeDBPropertyGraphStore: Advanced property graph store with vector support

Attributes:
    EMBEDDED_AVAILABLE: True if the ``arcadedb-embedded`` package is installed.

Example usage (remote — property graph):
    from llama_index.graph_stores.arcadedb import ArcadeDBPropertyGraphStore

    graph_store = ArcadeDBPropertyGraphStore(
        host="localhost",
        port=2480,
        username="root",
        password="playwithdata",
        database="knowledge_graph",
        embedding_dimension=1536,
    )

Example usage (embedded — no server needed):
    graph_store = ArcadeDBPropertyGraphStore(
        mode="embedded",
        db_path="./my_graph_data",
        database="knowledge_graph",
        embedding_dimension=384,
    )

Example usage (embedded + embedded HTTP server for interactive inspection):
    graph_store = ArcadeDBPropertyGraphStore(
        mode="embedded",
        db_path="./my_graph_data",
        database="knowledge_graph",
        embedding_dimension=384,
        embedded_server=True,              # start HTTP server
        embedded_server_port=2480,         # http://localhost:2480/
        embedded_server_password="secret", # root password for login
    )
    print(graph_store.embedded_server_url)  # "http://localhost:2480/"
    # Studio lets you pick any database under db_path from the DB picker

Example usage (basic graph store, remote):
    from llama_index.graph_stores.arcadedb import ArcadeDBGraphStore

    graph_store = ArcadeDBGraphStore(
        host="localhost",
        port=2480,
        username="root",
        password="playwithdata",
        database="knowledge_graph",
    )
"""

from llama_index.graph_stores.arcadedb.base import ArcadeDBGraphStore
from llama_index.graph_stores.arcadedb.arcadedb_property_graph import (
    ArcadeDBPropertyGraphStore,
    _EMBEDDED_AVAILABLE as EMBEDDED_AVAILABLE,
)

__all__ = [
    "ArcadeDBGraphStore",
    "ArcadeDBPropertyGraphStore",
    "EMBEDDED_AVAILABLE",
]

__version__ = "0.4.1"
