"""Standalone script to start an embedded ArcadeDB HTTP server on a persistent
database directory and keep it running until you press Ctrl+C.

Usage:
    python examples/embedded_server.py
    python examples/embedded_server.py --db-path ./my_db --port 2482 --password mypassword
    python examples/embedded_server.py --db-path ./my_db --embedding-dim 768  (default: 1536)

Studio will be available at http://localhost:<port> in your browser.
Select the database in the Studio database picker.

This is useful for:
- Browsing a database created by flexible-graphrag or other embedded usage
- Running Studio alongside a long-running process that holds the same db
  (use embedded_server=True in ArcadeDBPropertyGraphStore instead)
- Manual inspection / debugging of graph data
"""

import argparse
import os
import signal
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start an embedded ArcadeDB HTTP server for Studio access."
    )
    parser.add_argument(
        "--db-path",
        default=os.path.join(os.path.dirname(__file__), "..", "embedded_db", "graph"),
        help="Path to the ArcadeDB database directory (default: ./embedded_db/graph)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2482,
        help="HTTP port for Studio / REST API (default: 2482, avoids collision with Docker on 2480)",
    )
    parser.add_argument(
        "--password",
        default="playwithdata",
        help="Server root password (default: playwithdata)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=1536,
        help="Embedding dimension matching the database (default: 1536 for text-embedding-ada-002, 768 for many open-source models)",
    )
    args = parser.parse_args()

    db_path = os.path.abspath(args.db_path)

    try:
        from llama_index.graph_stores.arcadedb import ArcadeDBPropertyGraphStore, EMBEDDED_AVAILABLE
    except ImportError:
        print("ERROR: llama-index-graph-stores-arcadedb is not installed.")
        sys.exit(1)

    if not EMBEDDED_AVAILABLE:
        print("ERROR: arcadedb_embedded is not installed.")
        print("Install it with:  uv pip install arcadedb-embedded>=26.2.1")
        sys.exit(1)

    print(f"Opening embedded database at: {db_path}")
    print(f"Starting embedded HTTP server on port {args.port} ...")

    store = ArcadeDBPropertyGraphStore(
        mode="embedded",
        db_path=db_path,
        embedded_server=True,
        embedded_server_port=args.port,
        embedded_server_password=args.password,
        include_basic_schema=True,
        embedding_dimension=args.embedding_dim,
    )

    url = store.embedded_server_url
    if url:
        print(f"\nStudio is ready at: {url}")
        print("Open that URL in your browser and select the database in the picker.")
    else:
        print(f"\nServer started. Try: http://localhost:{args.port}")

    print("\nPress Ctrl+C to stop.\n")

    stop = False

    def _handle_signal(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while not stop:
            time.sleep(1)
    finally:
        print("\nShutting down embedded server...")
        try:
            store._db.stop_embedded_server()
        except Exception:
            pass
        print("Done.")


if __name__ == "__main__":
    main()
