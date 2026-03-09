"""Database adapter layer for ArcadeDB property graph store.

Provides a unified interface over two backends:

* ``RemoteAdapter``  — wraps ``arcadedb_python.dao.database.DatabaseDao``
  (HTTP/REST, existing behaviour).
* ``EmbeddedAdapter`` — wraps an ``arcadedb_embedded.Database`` instance
  (in-process JVM via ``arcadedb-embedded``).

Both adapters expose the same surface used by
``ArcadeDBPropertyGraphStore``:

    query(language, sql, *, is_command=False) -> list[dict]
    vector_search(type_name, embedding_field, query_embedding, top_k) -> list[dict]
    create_vector_index(type_name, field, dimensions) -> None
    safe_delete_all(type_name, batch_size=1000) -> int

``EmbeddedAdapter`` also exposes:

    embedded_server_url -> Optional[str]   # None when no embedded server is running
    stop_embedded_server() -> None         # gracefully stop the embedded HTTP server

Embedded server vs. direct open
--------------------------------
``arcadedb-embedded`` supports two ways to open a database:

1. **Direct** (``arcadedb_embedded.create_database`` / ``open_database``) —
   ``DatabaseFactory`` acquires an exclusive file lock.
2. **Via server** (``ArcadeDBServer.create_database`` / ``get_database``) —
   the server process holds the file lock and all database access goes
   through it.

**These two paths must never be used simultaneously on the same database
directory** — the second open will fail or cause data corruption.

``build_embedded_adapter()`` enforces this rule:

* ``embedded_server=False`` → direct open, no server started.
* ``embedded_server=True``  → server starts first and claims the lock;
  the ``Database`` handle is then obtained *back from the server* so
  there is only one Java handle on the files.  The running server also
  exposes the HTTP REST API and the Studio web UI (browse all databases
  under ``db_path`` by selecting them in the Studio database picker).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Remote adapter (wraps arcadedb_python DatabaseDao)
# ---------------------------------------------------------------------------

class RemoteAdapter:
    """Thin wrapper around ``arcadedb_python.dao.database.DatabaseDao``
    that matches the adapter interface."""

    def __init__(self, dao: Any) -> None:
        self._dao = dao

    # ------------------------------------------------------------------
    # Core query / command
    # ------------------------------------------------------------------

    def query(
        self,
        language: str,
        sql: str,
        *,
        is_command: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute a SQL query/command and return a plain ``list[dict]``."""
        result = self._dao.query(language, sql, is_command=is_command)
        # DatabaseDao already returns list[dict] — return as-is
        if result is None:
            return []
        if isinstance(result, list):
            return result
        # Unexpected shape — return empty to be safe
        return []

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def vector_search(
        self,
        type_name: str,
        embedding_field: str,
        query_embedding: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Delegate to the DAO's built-in ``vector_search`` method."""
        return self._dao.vector_search(
            type_name=type_name,
            embedding_field=embedding_field,
            query_embedding=query_embedding,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # DDL helpers
    # ------------------------------------------------------------------

    def create_vector_index(
        self,
        type_name: str,
        embedding_field: str,
        dimensions: int,
    ) -> None:
        """Create an LSM_VECTOR index via the DAO helper."""
        self._dao.create_vector_index(type_name, embedding_field, dimensions)

    def safe_delete_all(self, type_name: str, batch_size: int = 1000) -> int:
        """Delegate to the DAO's bulk-delete helper."""
        return self._dao.safe_delete_all(type_name, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Embedded adapter (wraps arcadedb_embedded.Database)
# ---------------------------------------------------------------------------

def _result_to_dict(rec: Any) -> Dict[str, Any]:
    """Convert an ``arcadedb_embedded.results.Result`` to a plain Python dict.

    ``convert_java_to_python()`` in the embedded wrapper handles strings,
    numbers, dates and collections, but has no case for
    ``com.arcadedb.database.RID``.  Those fall through as raw Java objects.
    The system fields that carry RID values are:

    * ``@rid``  — record's own identity  (also exposed via ``get_rid()``)
    * ``@in``   — edge target vertex RID
    * ``@out``  — edge source vertex RID

    We call ``get_rid()`` for ``@rid`` (always returns a clean Python str) and
    call ``str()`` on any remaining value whose Java class name ends in "RID",
    so ``@in``/``@out`` and any other RID-typed property also become strings.

    **Key coercion**: ``getPropertyNames()`` returns a Java collection whose
    elements are Java ``String`` objects, not Python ``str``.  Using them
    directly as dict keys causes downstream failures (``k.startswith('@')``
    raises ``AttributeError`` because Java String has no such method).  We
    call ``str()`` on every key to guarantee pure Python strings.
    """
    d: Dict[str, Any] = {}
    for name in rec.property_names:
        key = str(name)  # Java String → Python str
        val = rec.get(name)  # convert_java_to_python() applied; RIDs still Java objects
        # Coerce any remaining Java RID objects to plain strings
        if val is not None and hasattr(val, "getClass"):
            class_name = val.getClass().getSimpleName()
            if "RID" in class_name:
                val = str(val)
        d[key] = val

    # get_rid() always returns a clean Python str (or None); overwrite the
    # Java RID object that property_names/@rid may have produced above.
    rid = rec.get_rid()
    if rid is not None:
        d["@rid"] = str(rid)

    # Attach the record's type name as @type so _result_to_node can distinguish
    # TextChunk from entity types without relying on @class (OrientDB convention).
    try:
        type_name = rec.get_type_name()
        if type_name:
            d["@type"] = str(type_name)
    except Exception:
        pass

    # For edge records, extract @out/@in RIDs via the Edge wrapper.
    # property_names only returns user-defined properties; @out/@in are system
    # fields on the Java edge object, not exposed through getPropertyNames().
    if "@out" not in d or "@in" not in d:
        try:
            from arcadedb_embedded.graph import Edge as _ArcadeEdge
            element = rec.get_element()
            if element is not None and isinstance(element, _ArcadeEdge):
                d["@out"] = str(element.get_out().get_rid())
                d["@in"]  = str(element.get_in().get_rid())
        except Exception:
            pass

    # Fallback: try reading @out/@in directly from the Java result's getProperty.
    # ArcadeDB exposes these as "@out" / "@in" even though getPropertyNames() omits them.
    if "@out" not in d:
        try:
            from arcadedb_embedded.type_conversion import convert_java_to_python
            val = convert_java_to_python(rec._java_result.getProperty("@out"))
            if val is not None:
                if hasattr(val, "getClass") and "RID" in val.getClass().getSimpleName():
                    val = str(val)
                d["@out"] = str(val)
        except Exception:
            pass
    if "@in" not in d:
        try:
            from arcadedb_embedded.type_conversion import convert_java_to_python
            val = convert_java_to_python(rec._java_result.getProperty("@in"))
            if val is not None:
                if hasattr(val, "getClass") and "RID" in val.getClass().getSimpleName():
                    val = str(val)
                d["@in"] = str(val)
        except Exception:
            pass

    return d


def _vertex_to_dict(vertex: Any) -> Dict[str, Any]:
    """Convert an ``arcadedb_embedded.graph.Vertex`` (from vector search) to dict."""
    d: Dict[str, Any] = {}
    try:
        from arcadedb_embedded.type_conversion import convert_java_to_python
        for name in vertex._java_document.getPropertyNames():
            val = convert_java_to_python(vertex._java_document.get(name))
            # Coerce any remaining Java RID values to str
            if val is not None and hasattr(val, "getClass"):
                class_name = val.getClass().getSimpleName()
                if "RID" in class_name:
                    val = str(val)
            d[str(name)] = val  # str() key: Java String → Python str
        rid = vertex.get_rid()
        if rid:
            d["@rid"] = str(rid)
        type_name = vertex.get_type_name()
        if type_name:
            d["@type"] = str(type_name)
    except Exception as exc:
        logger.debug("_vertex_to_dict failed: %s", exc)
    return d


class EmbeddedAdapter:
    """Adapter over ``arcadedb_embedded.Database`` matching the adapter interface.

    Key behavioural differences from the remote DAO that this class handles:

    * **Writes** must be executed via ``db.command()`` inside a
      ``with db.transaction():`` block.  The ``query()`` method here
      auto-wraps DML/DDL in a transaction when ``is_command=True``.
    * **Results** come back as ``ResultSet`` objects containing ``Result``
      items rather than plain dicts — we normalise them here.
    * **Vector search** uses ``db.create_vector_index(...)`` which returns a
      ``VectorIndex``, and then ``index.find_nearest(vec, k)`` which returns
      ``[(Vertex, score)]`` tuples.  We cache ``VectorIndex`` objects so
      repeated searches don't recreate them.
    * **Embedded server** — when the adapter is created via
      ``build_embedded_adapter(embedded_server=True)`` an ``ArcadeDBServer``
      is the *sole owner* of the data files.  The ``Database`` handle is
      obtained from the server so there is only one Java file lock.  The
      running server also exposes the HTTP REST API and Studio UI — all
      databases under ``db_path`` are visible and selectable in the Studio
      database picker at ``http://localhost:<port>/``.
    """

    def __init__(
        self,
        db: Any,
        *,
        server: Optional[Any] = None,
    ) -> None:
        """
        Args:
            db: An open ``arcadedb_embedded.Database`` instance.  When
                ``server`` is also provided this handle must have been obtained
                from that server (``server.get_database()`` /
                ``server.create_database()``), *not* opened independently via
                ``DatabaseFactory`` — that would create a second file lock on
                the same directory.
            server: Optional running ``ArcadeDBServer`` instance.  Stored so
                callers can read ``embedded_server_url`` and call
                ``stop_embedded_server()``.
        """
        self._db = db
        self._server: Optional[Any] = server
        # Cache of {"type_name:embedding_field": VectorIndex}
        self._vector_indexes: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Embedded server accessors
    # ------------------------------------------------------------------

    @property
    def embedded_server_url(self) -> Optional[str]:
        """Base URL of the embedded HTTP server, or ``None`` if not running.

        When not ``None`` the server exposes:

        * **Studio UI** at ``<url>`` — visual DB explorer, pick any database
          that lives under ``db_path`` from the Studio database picker.
        * **HTTP REST API** at ``<url>api/v1/`` — query/command endpoints
          compatible with the standard ArcadeDB HTTP API.
        """
        if self._server is not None:
            try:
                return self._server.get_studio_url()
            except Exception:
                pass
        return None

    def stop_embedded_server(self) -> None:
        """Gracefully stop the embedded HTTP server, if one was started.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        if self._server is not None:
            try:
                self._server.stop()
                logger.info("EmbeddedAdapter: embedded server stopped")
            except Exception as exc:
                logger.debug("EmbeddedAdapter: stop_embedded_server error: %s", exc)
            finally:
                self._server = None

    # ------------------------------------------------------------------
    # Core query / command
    # ------------------------------------------------------------------

    def query(
        self,
        language: str,
        sql: str,
        *,
        is_command: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute SQL and return a plain ``list[dict]``.

        Reads (``is_command=False``) go directly to ``db.query()``.
        Writes (``is_command=True``: INSERT, UPDATE, DELETE, CREATE …) are
        automatically wrapped in ``with db.transaction(): db.command()``.
        """
        try:
            if is_command:
                with self._db.transaction():
                    result_set = self._db.command(language, sql)
            else:
                result_set = self._db.query(language, sql)
        except Exception:
            raise

        if result_set is None:
            return []

        rows: List[Dict[str, Any]] = []
        try:
            for rec in result_set:
                rows.append(_result_to_dict(rec))
        except Exception as exc:
            logger.debug("EmbeddedAdapter: result iteration error: %s", exc)

        return rows

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def vector_search(
        self,
        type_name: str,
        embedding_field: str,
        query_embedding: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Search using the LSMVectorIndex for the given type/field.

        **Server mode** (``embedded_server=True``): Uses an ArcadeDB SQL query
        with the ``vectorNeighbors()`` function.  This goes through the server's
        query engine which correctly uses the server-managed LSMVectorIndex.
        Calling ``VectorIndex.find_nearest()`` or ``build_graph_now()`` in server
        mode causes page-I/O conflicts with the server ("Invalid position" /
        "FALLBACK: database closing") and must be avoided.

        **Direct mode** (``embedded_server=False``): Refreshes the cached
        ``VectorIndex`` from schema, flushes WAL, rebuilds the HNSW graph via
        ``build_graph_now()``, then calls ``find_nearest()``.
        """
        cache_key = f"{type_name}:{embedding_field}"

        if self.is_server_mode:
            return self._vector_search_sql(type_name, embedding_field, query_embedding, top_k)

        # Direct mode: re-fetch index from schema and rebuild before searching.
        self._cache_existing_vector_index(type_name, embedding_field, cache_key)

        idx = self._vector_indexes.get(cache_key)
        if idx is None:
            logger.debug(
                "EmbeddedAdapter.vector_search: no cached VectorIndex for %s — "
                "skipping (call create_vector_index first)",
                cache_key,
            )
            return []

        self.flush_wal()
        try:
            idx.build_graph_now()
        except Exception as build_exc:
            logger.debug(
                "EmbeddedAdapter.vector_search: build_graph_now for %s: %s",
                cache_key,
                build_exc,
            )

        try:
            results = idx.find_nearest(query_embedding, k=top_k)
        except Exception as exc:
            logger.debug("EmbeddedAdapter.vector_search error: %s", exc)
            return []

        rows: List[Dict[str, Any]] = []
        for vertex, score in results:
            d = _vertex_to_dict(vertex)
            d["distance"] = score
            rows.append(d)
        return rows

    def _vector_search_sql(
        self,
        type_name: str,
        embedding_field: str,
        query_embedding: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """SQL-based vector search for embedded server mode.

        Uses the ArcadeDB embedded Python bindings' documented pattern
        (https://docs.humem.ai/arcadedb/latest/examples/03_vector_search/):

            result_set = db.query("sql",
                "SELECT vectorNeighbors('TypeName[prop]', <vec>, k) as res")
            rows = result_set.to_list()   # → [{"res": [{"record": <JavaVertex>, "distance": float}, ...]}]
            for hit in rows[0]["res"]:
                vertex_java = hit["record"]   # raw Java Vertex object
                distance    = hit["distance"] # float

        ``convert_java_to_python`` converts JavaList/JavaMap recursively but
        leaves Java Vertex objects as-is (no Python case for them).  We wrap
        each one with ``Document.wrap()`` then call ``.to_dict()`` /
        ``.get_rid()`` / ``.get_type_name()`` to produce a plain Python dict.

        This goes through the server's SQL engine — no direct
        LSMVectorIndex page-file access — avoiding the
        "FALLBACK: database closing" / "Invalid position" errors.
        """
        from arcadedb_embedded.graph import Document as ArcadeDocument

        index_name = f"{type_name}[{embedding_field}]"
        vec_literal = "[" + ", ".join(str(float(x)) for x in query_embedding) + "]"
        sql = f"SELECT vectorNeighbors('{index_name}', {vec_literal}, {top_k}) as res"
        logger.debug(
            "EmbeddedAdapter._vector_search_sql: querying index %s top_k=%d",
            index_name,
            top_k,
        )
        try:
            # db.query() returns a ResultSet; .to_list() calls result.to_dict()
            # on each Result row, which runs convert_java_to_python on each value.
            # For the "res" list, convert_java_to_python converts JavaList→list and
            # JavaMap→dict recursively, but leaves the Java Vertex "record" as-is.
            result_set = self._db.query("sql", sql)
            if result_set is None:
                logger.debug(
                    "EmbeddedAdapter._vector_search_sql: %s — result_set is None",
                    index_name,
                )
                return []

            rows = result_set.to_list()   # List[Dict]  (1 row with "res" key)
            if not rows:
                logger.debug(
                    "EmbeddedAdapter._vector_search_sql: %s — no rows returned",
                    index_name,
                )
                return []

            neighbors = rows[0].get("res") or []
            # Coerce Java collection to Python list if needed
            if not isinstance(neighbors, list):
                try:
                    neighbors = list(neighbors)
                except Exception:
                    neighbors = []

            logger.debug(
                "EmbeddedAdapter._vector_search_sql: %s returned %d neighbors",
                index_name,
                len(neighbors),
            )

            result: List[Dict[str, Any]] = []
            for hit in neighbors:
                if not isinstance(hit, dict):
                    logger.debug(
                        "EmbeddedAdapter._vector_search_sql: hit is not dict, skipping (type=%s)",
                        type(hit).__name__,
                    )
                    continue
                java_vertex = hit.get("record")
                distance = hit.get("distance", 1.0)

                if java_vertex is None:
                    continue

                # Wrap the raw Java Vertex in the Python bindings wrapper, then
                # convert to a plain dict using the bindings' own to_dict().
                try:
                    wrapped = ArcadeDocument.wrap(java_vertex)
                    # wrapped.to_dict() and get_type_name()/get_rid() return Java String
                    # objects — coerce everything to Python str so Pydantic validation passes.
                    flat = {str(k): v for k, v in wrapped.to_dict().items()}
                    flat["@rid"] = str(wrapped.get_rid())
                    flat["@type"] = str(wrapped.get_type_name())
                except Exception as wrap_exc:
                    import traceback
                    logger.debug(
                        "EmbeddedAdapter._vector_search_sql: wrap failed: %s\n%s",
                        wrap_exc, traceback.format_exc(),
                    )
                    continue

                flat["distance"] = float(distance) if distance is not None else 1.0
                result.append(flat)

            return result

        except Exception as exc:
            logger.debug(
                "EmbeddedAdapter._vector_search_sql failed for %s: %s",
                type_name,
                exc,
            )
            return []

    @property
    def is_server_mode(self) -> bool:
        """True when the DB is owned by an embedded HTTP server.

        In server mode the server process manages all page I/O and WAL flushing
        internally.  Callers must NOT call ``build_graph_now()`` or manipulate
        WAL flush settings on a server-managed database — doing so causes the
        ``LSMVectorIndex`` page reader to fight with the server's I/O layer,
        producing "Invalid position" / "FALLBACK: database closing" errors.
        """
        return self._server is not None

    def flush_wal(self) -> None:
        """Best-effort WAL flush to commit embedding data to base page files.

        **Server mode** (``embedded_server=True``): This is a no-op.  The server
        owns all page I/O and flushes the WAL on its own schedule.  Calling
        ``set_wal_flush`` or forcing commits on a server-managed database
        interferes with the server's I/O layer.

        **Direct mode** (``embedded_server=False``): Flush via
        ``set_wal_flush('yes_nometadata')`` + a commit cycle, then restore
        ``'no'``.  Falls back to an explicit begin/commit if the first strategy
        fails.
        """
        if self.is_server_mode:
            # Server manages WAL flushing — nothing to do here.
            return

        try:
            self._db.set_wal_flush('yes_nometadata')
            # A commit cycle triggers WAL flushing under the new mode.
            if not self._db.is_transaction_active():
                self._db.begin()
            self._db.commit()
            self._db.set_wal_flush('no')
            logger.debug("EmbeddedAdapter: WAL flushed via set_wal_flush")
            return
        except Exception as exc1:
            logger.debug("EmbeddedAdapter: set_wal_flush strategy failed: %s", exc1)

        # Fallback: explicit commit flushes pending WAL entries in ArcadeDB.
        try:
            if not self._db.is_transaction_active():
                self._db.begin()
            self._db.commit()
            logger.debug("EmbeddedAdapter: WAL flushed via explicit commit")
        except Exception as exc2:
            logger.debug("EmbeddedAdapter: WAL flush fallback also failed: %s", exc2)

    # ------------------------------------------------------------------
    # DDL helpers
    # ------------------------------------------------------------------

    def create_vector_index(
        self,
        type_name: str,
        embedding_field: str,
        dimensions: int,
    ) -> None:
        """Create (or re-use) a vector index and cache the ``VectorIndex`` object.

        ``build_graph_now=False`` is used here so that the LSMVectorIndex does
        **not** try to eagerly build its HNSW graph immediately after creation.
        Building on an empty (just-created) database causes the JVM to flush
        zero-entry pages, which produces:

        * ``FALLBACK: Could not read vectors from pages (database closing),
          using existing vectorIndex with 0 entries``
        * ``LSMVectorIndexPageParser: Error parsing page N: Invalid position``

        The graph will be rebuilt lazily on the first ``find_nearest`` call (or
        explicitly via ``build_graph_now()`` after data has been written).
        """
        cache_key = f"{type_name}:{embedding_field}"
        try:
            idx = self._db.create_vector_index(
                type_name,
                embedding_field,
                dimensions=dimensions,
                build_graph_now=False,  # defer graph build until data exists
            )
            self._vector_indexes[cache_key] = idx
            logger.debug(
                "EmbeddedAdapter: created/opened vector index %s (graph build deferred)",
                cache_key,
            )
        except Exception as exc:
            err = str(exc).lower()
            if "already exists" in err:
                # Index exists — retrieve it from schema so searches still work.
                logger.debug(
                    "EmbeddedAdapter: vector index %s already exists, "
                    "attempting schema lookup",
                    cache_key,
                )
                self._cache_existing_vector_index(type_name, embedding_field, cache_key)
            else:
                logger.warning(
                    "EmbeddedAdapter: could not create vector index %s: %s",
                    cache_key,
                    exc,
                )

    def _cache_existing_vector_index(
        self,
        type_name: str,
        embedding_field: str,
        cache_key: str,
    ) -> None:
        """Look up an existing LSM_VECTOR index from the schema and cache it."""
        try:
            schema = self._db._java_db.getSchema()
            idx = schema.getIndexByName(f"{type_name}[{embedding_field}]")
            from arcadedb_embedded.vector import VectorIndex
            self._vector_indexes[cache_key] = VectorIndex(idx, self._db)
            logger.debug(
                "EmbeddedAdapter: cached vector index %s (size=%s)",
                cache_key,
                self._vector_indexes[cache_key].get_size(),
            )
        except Exception as exc:
            logger.warning(
                "EmbeddedAdapter: could not retrieve existing vector index %s: %s",
                cache_key,
                exc,
            )

    def safe_delete_all(self, type_name: str, batch_size: int = 1000) -> int:
        """Delete all records of *type_name* in batches.

        The embedded engine has no ``safe_delete_all`` equivalent, so we
        implement it here with batched ``DELETE FROM … LIMIT N`` commands.
        """
        total = 0
        try:
            while True:
                rows = self.query("sql", f"SELECT @rid FROM {type_name} LIMIT {batch_size}")
                if not rows:
                    break
                with self._db.transaction():
                    self._db.command("sql", f"DELETE FROM {type_name} LIMIT {batch_size}")
                total += len(rows)
                if len(rows) < batch_size:
                    break
        except Exception as exc:
            logger.debug(
                "EmbeddedAdapter.safe_delete_all(%s): %s", type_name, exc
            )
        return total


# ---------------------------------------------------------------------------
# Helper — silence ArcadeDB's verbose java.util.logging INFO output
# ---------------------------------------------------------------------------

def _suppress_java_info_logging() -> None:
    """Set the java.util.logging root logger to WARNING.

    ArcadeDB (and JVector) emit many INFO-level messages to stdout when
    rebuilding vector-index graphs (``LSMVectorIndex``, ``LSMVectorIndexGraphFile``
    etc.).  These go through java.util.logging — Python's ``logging`` module
    cannot suppress them.  Raising the JUL root level to WARNING silences the
    chatter while still surfacing genuine problems.

    This is a no-op if the JVM has not been started yet (the call site in
    ``build_embedded_adapter`` is always after JVM startup), and it is safe
    to call multiple times.
    """
    try:
        from jpype import JClass
        Level = JClass("java.util.logging.Level")
        LogManager = JClass("java.util.logging.LogManager")
        mgr = LogManager.getLogManager()
        mgr.getLogger("").setLevel(Level.WARNING)
        arcade_log = mgr.getLogger("com.arcadedb")
        if arcade_log is not None:
            arcade_log.setLevel(Level.WARNING)
    except Exception:
        pass  # Best-effort; never break the caller


# ---------------------------------------------------------------------------
# Factory — builds an EmbeddedAdapter, enforcing the single-owner lock rule
# ---------------------------------------------------------------------------

def build_embedded_adapter(
    db_path: str,
    database: str,
    create_if_not_exists: bool,
    *,
    embedded_server: bool = False,
    embedded_server_port: int = 2480,
    embedded_server_password: Optional[str] = None,
) -> "EmbeddedAdapter":
    """Open (or create) an embedded ArcadeDB database and return an adapter.

    **File-lock safety**: ArcadeDB uses exclusive file locks per database
    directory.  Only one Java handle may hold a lock at a time.  This
    factory ensures the two possible owners — ``DatabaseFactory`` (direct
    open) and ``ArcadeDBServer`` — are never mixed on the same path:

    * ``embedded_server=False`` (default) — database opened directly via
      ``DatabaseFactory``; no HTTP server is started.
    * ``embedded_server=True`` — an ``ArcadeDBServer`` is started first so it
      acquires the lock, then the ``Database`` handle is obtained *from the
      server* via ``server.get_database()`` / ``server.create_database()``.
      This gives a single Java file handle.  The server also exposes:

      - **HTTP REST API**: ``http://localhost:<port>/api/v1/``
      - **Studio web UI**: ``http://localhost:<port>/``  — all databases
        under ``db_path`` appear in the Studio database picker, so you can
        switch between them interactively.

    Args:
        db_path: Root directory for database files.  Individual databases
            live at ``<db_path>/<database>/``.
        database: Name of the specific database to open/create.
        create_if_not_exists: Create the database when it doesn't exist.
        embedded_server: Start the embedded HTTP server (default: False).
        embedded_server_port: HTTP port (default: 2480).
        embedded_server_password: Root password for the server.  When
            ``None`` the ArcadeDB default (insecure) credential is used.
    """
    import os
    import arcadedb_embedded as _emb

    full_path = os.path.join(db_path, database)

    if not embedded_server:
        # Direct open — no server, no lock contention.
        if _emb.database_exists(full_path):
            logger.info("Opening existing embedded database at %s", full_path)
            raw_db = _emb.open_database(full_path)
        elif create_if_not_exists:
            logger.info("Creating embedded database at %s", full_path)
            raw_db = _emb.create_database(full_path)
        else:
            raise FileNotFoundError(
                f"Embedded database does not exist at {full_path}"
            )
        # DB is fully open — all ArcadeDB loggers are now registered.
        # Raise JUL root and com.arcadedb levels to WARNING to suppress
        # verbose INFO output (LSMVectorIndex graph-build progress, etc.).
        _suppress_java_info_logging()
        return EmbeddedAdapter(raw_db)

    # Server path — server claims the file lock; we get the DB handle from it.
    logger.info(
        "Starting embedded server (root_path=%s, port=%s) as sole DB owner",
        db_path,
        embedded_server_port,
    )
    server = _emb.create_server(
        root_path=db_path,
        root_password=embedded_server_password,
        config={"http_port": embedded_server_port},
    )
    try:
        server.start()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to start embedded server on port {embedded_server_port}: {exc}"
        ) from exc

    # JVM is now started — raise JUL root level to suppress verbose INFO messages
    _suppress_java_info_logging()

    # Open/create the database through the server — never via DatabaseFactory.
    try:
        raw_db = server.get_database(database)
        logger.info(
            "Opened existing database '%s' via server  (url: %s)",
            database,
            server.get_studio_url(),
        )
    except Exception:
        if not create_if_not_exists:
            server.stop()
            raise FileNotFoundError(
                f"Embedded database '{database}' does not exist under {db_path}"
            )
        try:
            raw_db = server.create_database(database)
            logger.info(
                "Created database '%s' via server  (url: %s)",
                database,
                server.get_studio_url(),
            )
        except Exception as exc:
            server.stop()
            raise RuntimeError(
                f"Failed to create database '{database}' via server: {exc}"
            ) from exc

    logger.info(
        "Embedded server running — Studio/HTTP at %s  (all DBs under %s visible)",
        server.get_studio_url(),
        db_path,
    )
    return EmbeddedAdapter(raw_db, server=server)
