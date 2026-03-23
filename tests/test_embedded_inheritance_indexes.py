"""
Embedded ArcadeDB: vertex inheritance + indexes (diagnostic).

Uses **only** ``import arcadedb_embedded`` — no LlamaIndex graph store or
``EmbeddedAdapter``.  Commands go through ``Database.command`` / ``query`` and
``transaction()`` as in the upstream docs.  After open, applies the same
``java.util.logging`` quieting as ``_db_adapter._suppress_java_info_logging``
(plus WARNING on Python loggers ``arcadedb_embedded`` / ``jpype``).

Tests cover SQL ``vectorNeighbors``, ``Schema.get_vector_index`` →
``VectorIndex.find_nearest``, and ``Database.create_vector_index`` → ``find_nearest``.

**Schema / Studio (vector-related)** after DDL in ``setup_module``:

- **SQL ``LSM_VECTOR`` declared on a base:** only **Base1**.
  **Base2**, **ApiBase**, and **BixBase** have no SQL vector index at DDL time; **BixSub** none either.
- **SQL ``LSM_VECTOR`` on a subtype:** **Sub2** only (none on **Base2**).
- **Subtype-only data + base SQL index (Base1):** SQL ``vectorNeighbors('Base1[embedding]', …)``
  stays **0**; ``vectorNeighbors('Sub1[embedding]', …)`` hits.  **Embedded-only (Java path):**
  ``schema.get_vector_index('Base1', 'embedding').find_nearest`` returns hits.
- **Java** ``create_vector_index`` tests use **ApiBase**/**ApiSub** and **BixBase**/**BixSub**
  (no prior SQL vector DDL; indexes created only by the Java API call).

Embedded modes (context):
- **Direct open** (this file): SQL and Java-backed Python APIs
  (``Database.create_vector_index``, ``VectorIndex``, etc.) against one
  on-disk database directory.
- **Embedded HTTP server** (``ArcadeDBServer`` / ``create_server``): serves the
  Studio web UI and manages one or more databases under a root path (select DB by
  directory / name in Studio).

Run:
  .\\venv-3.13\\Scripts\\python.exe -m pytest tests/test_embedded_inheritance_indexes.py -v
  .\\venv-3.13\\Scripts\\python.exe tests/test_embedded_inheritance_indexes.py

  The script entrypoint starts the embedded HTTP server on port **2482** (Studio at
  http://localhost:2482/), runs the diagnostics, then waits for Enter before closing
  the DB and stopping the server (skipped when stdin is not a TTY).  The server root
  password defaults to ``playwithdata`` (same as Flexible GraphRAG’s
  ``embedded_server_password``; set ``ARCADEDB_EMB_CLI_ROOT_PASSWORD`` to override)
  so first-time startup stays non-interactive.

  **Flexible GraphRAG** (and ``ArcadeDBPropertyGraphStore``) use the same knobs under
  different names: ``db_path`` → server root (databases live under
  ``<db_path>/<database>/``), ``database`` → folder name, ``embedded_server=True``,
  ``embedded_server_port=2482``, ``embedded_server_password=...``.  This script’s
  ``main()`` is standalone ``arcadedb_embedded`` but mirrors that server layout.

References:
  https://docs.humem.ai/arcadedb/latest/
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List

import pytest

try:
    import arcadedb_embedded as arcadedb
except ImportError:
    arcadedb = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    arcadedb is None,
    reason="arcadedb_embedded is not installed",
)


def _suppress_java_info_logging() -> None:
    """Raise ``java.util.logging`` root + ``com.arcadedb`` to WARNING.

    Mirrors ``llama_index.graph_stores.arcadedb._db_adapter._suppress_java_info_logging``
    so JVector / LSMVectorIndex INFO lines do not flood stderr. Call after the
    JVM is up and the database is open.
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
        pass


def _quiet_python_loggers() -> None:
    for name in ("arcadedb_embedded", "jpype"):
        logging.getLogger(name).setLevel(logging.WARNING)


# Type names for embedded DB directory.
_BASE = "Base1"
_SUB = "Sub1"
_CV_BASE = "Base2"
_CV_SUB = "Sub2"
# No SQL LSM_VECTOR at DDL — vector index appears only via ``create_vector_index``.
_API_BASE = "ApiBase"
_API_SUB = "ApiSub"
_BIX_BASE = "BixBase"
_BIX_SUB = "BixSub"
_DIM = 4
_DB_DIRNAME = "inherit_emb"
# Studio port for ``main()`` only; avoids docker (2480) and test container (2481).
_CLI_EMBEDDED_HTTP_PORT = 2482
# Non-interactive first server start (avoids JVM stdin password prompts).
# Default matches Flexible GraphRAG ``embedded_server_password`` (e.g. playwithdata).
# Override: set ARCADEDB_EMB_CLI_ROOT_PASSWORD=...
_CLI_ROOT_PASSWORD = os.environ.get("ARCADEDB_EMB_CLI_ROOT_PASSWORD", "playwithdata")

_tmp_dir: str = ""
_db_path: str = ""
_db: Any = None


def setup_module(_module=None) -> None:
    global _tmp_dir, _db_path, _db
    _tmp_dir = tempfile.mkdtemp(prefix="arcadedb_emb_inherit_")
    root = os.path.join(_tmp_dir, "root")
    os.makedirs(root, exist_ok=True)
    _db_path = os.path.join(root, _DB_DIRNAME)
    if arcadedb.database_exists(_db_path):
        _db = arcadedb.open_database(_db_path)
    else:
        _db = arcadedb.create_database(_db_path)
    _suppress_java_info_logging()
    _quiet_python_loggers()
    _ensure_main_schema(_db)
    _ensure_control_schema(_db)
    _ensure_java_api_vertex_types(_db)


def teardown_module(_module=None) -> None:
    global _tmp_dir, _db_path, _db
    if _db is not None:
        try:
            _db.close()
        except Exception:
            pass
        _db = None
    gc.collect()
    if _tmp_dir and os.path.exists(_tmp_dir):
        try:
            shutil.rmtree(_tmp_dir)
        except PermissionError:
            time.sleep(0.5)
            try:
                shutil.rmtree(_tmp_dir)
            except PermissionError:
                pass
    _tmp_dir = ""
    _db_path = ""


def _normalize_row(row: Dict[Any, Any]) -> Dict[str, Any]:
    return {str(k): v for k, v in row.items()}


def _cmd(db, sql: str) -> None:
    with db.transaction():
        db.command("sql", sql)


def _select(db, sql: str) -> List[Dict[str, Any]]:
    rs = db.query("sql", sql)
    rows = rs.to_list()
    return [_normalize_row(r) for r in rows]


def _neighbor_count(row: Dict[str, Any] | None) -> int:
    if not row:
        return 0
    for key in ("neighbors", "res"):
        n = row.get(key)
        if isinstance(n, list):
            return len(n)
    return 0


def _ensure_main_schema(db) -> None:
    _cmd(db, f"CREATE VERTEX TYPE {_BASE} IF NOT EXISTS")
    _cmd(db, f"CREATE VERTEX TYPE {_SUB} IF NOT EXISTS EXTENDS {_BASE}")
    _cmd(db, f"CREATE PROPERTY {_BASE}.name IF NOT EXISTS STRING")
    _cmd(db, f"CREATE PROPERTY {_BASE}.embedding IF NOT EXISTS ARRAY_OF_FLOATS")
    try:
        _cmd(db, f"CREATE INDEX IF NOT EXISTS ON {_BASE} (name) UNIQUE")
    except Exception:
        _cmd(db, f"CREATE INDEX IF NOT EXISTS ON {_BASE} (name)")
    _cmd(
        db,
        f"CREATE INDEX IF NOT EXISTS ON {_BASE} (embedding) LSM_VECTOR METADATA "
        f"{{ dimensions: {_DIM}, similarity: 'COSINE' }}",
    )


def _ensure_control_schema(db) -> None:
    _cmd(db, f"CREATE VERTEX TYPE {_CV_BASE} IF NOT EXISTS")
    _cmd(db, f"CREATE VERTEX TYPE {_CV_SUB} IF NOT EXISTS EXTENDS {_CV_BASE}")
    _cmd(db, f"CREATE PROPERTY {_CV_BASE}.name IF NOT EXISTS STRING")
    _cmd(db, f"CREATE PROPERTY {_CV_BASE}.embedding IF NOT EXISTS ARRAY_OF_FLOATS")
    _cmd(
        db,
        f"CREATE INDEX IF NOT EXISTS ON {_CV_SUB} (embedding) LSM_VECTOR METADATA "
        f"{{ dimensions: {_DIM}, similarity: 'COSINE' }}",
    )


def _ensure_java_api_vertex_types(db) -> None:
    """Vertex types for Java vector API tests (no SQL ``LSM_VECTOR`` on these)."""
    for base, sub in ((_API_BASE, _API_SUB), (_BIX_BASE, _BIX_SUB)):
        _cmd(db, f"CREATE VERTEX TYPE {base} IF NOT EXISTS")
        _cmd(db, f"CREATE VERTEX TYPE {sub} IF NOT EXISTS EXTENDS {base}")
        _cmd(db, f"CREATE PROPERTY {base}.name IF NOT EXISTS STRING")
        _cmd(db, f"CREATE PROPERTY {base}.embedding IF NOT EXISTS ARRAY_OF_FLOATS")


def _wipe_api_pair(db) -> None:
    _cmd(db, f"DELETE FROM {_API_SUB}")
    _cmd(db, f"DELETE FROM {_API_BASE}")


def _wipe_bix_pair(db) -> None:
    _cmd(db, f"DELETE FROM {_BIX_SUB}")
    _cmd(db, f"DELETE FROM {_BIX_BASE}")


def _wipe_main(db) -> None:
    _cmd(db, f"DELETE FROM {_SUB}")
    _cmd(db, f"DELETE FROM {_BASE}")


def _wipe_control(db) -> None:
    _cmd(db, f"DELETE FROM {_CV_SUB}")
    _cmd(db, f"DELETE FROM {_CV_BASE}")


@pytest.fixture(scope="module")
def embedded_db():
    assert _db is not None
    return _db


def test_embedded_polymorphic_and_subtype_select_find_same_row(embedded_db):
    """Subtype-only rows visible from base SELECT and subtype WHERE."""
    db = embedded_db
    _wipe_main(db)
    name = f"poly-{uuid.uuid4().hex[:12]}"
    emb = [1.0, 0.0, 0.0, 0.0]
    payload = json.dumps({"name": name, "embedding": emb})
    _cmd(db, f"INSERT INTO {_SUB} CONTENT {payload}")

    from_sub = _select(db, f"SELECT FROM {_SUB} WHERE name = '{name}'")
    from_base = _select(db, f"SELECT FROM {_BASE} WHERE name = '{name}'")

    assert isinstance(from_sub, list) and len(from_sub) == 1, from_sub
    assert isinstance(from_base, list) and len(from_base) == 1, (
        f"Expected polymorphic SELECT FROM {_BASE} to return the subtype row; got {from_base!r}"
    )


def test_embedded_vector_neighbors_base_vs_subtype_index_names(embedded_db):
    """SQL vectorNeighbors: base index id does not return subtype-only rows."""
    db = embedded_db
    _wipe_main(db)
    name = f"vec-{uuid.uuid4().hex[:12]}"
    emb = [0.0, 1.0, 0.0, 0.0]
    payload = json.dumps({"name": name, "embedding": emb})
    _cmd(db, f"INSERT INTO {_SUB} CONTENT {payload}")

    q_base = (
        f"SELECT vectorNeighbors('{_BASE}[embedding]', {json.dumps(emb)}, 5) "
        f"AS neighbors FROM {_BASE} LIMIT 1"
    )
    raw_base = _select(db, q_base)
    n_base = _neighbor_count(raw_base[0] if raw_base else None)

    q_sub_from = (
        f"SELECT vectorNeighbors('{_BASE}[embedding]', {json.dumps(emb)}, 5) "
        f"AS neighbors FROM {_SUB} LIMIT 1"
    )
    raw_cross = _select(db, q_sub_from)
    n_cross = _neighbor_count(raw_cross[0] if raw_cross else None)

    q_sub_idx = (
        f"SELECT vectorNeighbors('{_SUB}[embedding]', {json.dumps(emb)}, 5) "
        f"AS neighbors FROM {_SUB} LIMIT 1"
    )
    raw_sub_idx = _select(db, q_sub_idx)
    n_sub_idx = _neighbor_count(raw_sub_idx[0] if raw_sub_idx else None)

    assert n_base == 0 and n_cross == 0, (
        f"Expected no neighbors via {_BASE}[embedding] for subtype-only rows; "
        f"got FROM {_BASE}: {n_base}, FROM {_SUB}: {n_cross}"
    )
    assert n_sub_idx >= 1, (
        f"Expected vectorNeighbors('{_SUB}[embedding]', ...) to find the row; got {n_sub_idx}"
    )


def test_embedded_vector_neighbors_subtype_index_control(embedded_db):
    """Control: explicit LSM_VECTOR on subtype; neighbors via SQL only."""
    db = embedded_db
    _wipe_control(db)
    name = f"ctl-{uuid.uuid4().hex[:12]}"
    emb = [0.0, 0.0, 1.0, 0.0]
    payload = json.dumps({"name": name, "embedding": emb})
    _cmd(db, f"INSERT INTO {_CV_SUB} CONTENT {payload}")

    q = (
        f"SELECT vectorNeighbors('{_CV_SUB}[embedding]', {json.dumps(emb)}, 5) "
        f"AS neighbors FROM {_CV_SUB} LIMIT 1"
    )
    raw = _select(db, q)
    assert _neighbor_count(raw[0] if raw else None) >= 1, raw


def test_embedded_schema_get_vector_index_find_nearest_base_and_sub(embedded_db):
    """Embedded Java API: ``get_vector_index(Base1).find_nearest`` hits subtype-only rows.

    With SQL ``LSM_VECTOR`` only on **Base1** (not a separate DDL index on Sub1):
    ``vectorNeighbors('Base1[embedding]', ...)`` is still **0** for subtype-only data,
    but here ``schema.get_vector_index('Base1', 'embedding').find_nearest`` often
    returns **≥ 1** — a behavior you only get on this embedded/Java path.
    ``get_vector_index('Sub1', 'embedding').find_nearest`` also hits (concrete-type index).
    """
    db = embedded_db
    _wipe_main(db)
    name = f"jwrap-{uuid.uuid4().hex[:12]}"
    emb = [0.25, 0.25, 0.25, 0.25]
    payload = json.dumps({"name": name, "embedding": emb})
    _cmd(db, f"INSERT INTO {_SUB} CONTENT {payload}")

    v_sub = db.schema.get_vector_index(_SUB, "embedding")
    assert v_sub is not None, "Expected subtype vector index for inherited embedding"
    pairs_sub = v_sub.find_nearest(emb, k=5)
    assert isinstance(pairs_sub, list) and len(pairs_sub) >= 1, pairs_sub

    v_base = db.schema.get_vector_index(_BASE, "embedding")
    if v_base is not None:
        pairs_base = v_base.find_nearest(emb, k=5)
        assert isinstance(pairs_base, list) and len(pairs_base) >= 1, pairs_base


def test_embedded_create_vector_index_subtype_find_nearest(embedded_db):
    """``Database.create_vector_index`` on subtype only (no prior SQL vector DDL)."""
    db = embedded_db
    _wipe_api_pair(db)
    name = f"api-sub-{uuid.uuid4().hex[:12]}"
    emb = [0.1, 0.2, 0.3, 0.4]
    payload = json.dumps({"name": name, "embedding": emb})
    _cmd(db, f"INSERT INTO {_API_SUB} CONTENT {payload}")

    idx = db.create_vector_index(
        _API_SUB,
        "embedding",
        _DIM,
        distance_function="cosine",
        build_graph_now=True,
    )
    assert idx is not None
    pairs = idx.find_nearest(emb, k=5)
    assert isinstance(pairs, list) and len(pairs) >= 1, pairs


def test_embedded_create_vector_index_base_find_nearest_subtype_only_row(embedded_db):
    """``create_vector_index(BixBase, …)`` with rows only on **BixSub** (no SQL LSM_VECTOR).

    Unlike **Base1** / **HierBase**, **BixBase** has no SQL vector DDL until this call;
    the Java API attaches the graph to **BixBase** (not **BixSub**); Studio may list it
    after this runs, separately from the SQL-declared indexes on Base1 / HierBase.
    """
    db = embedded_db
    _wipe_bix_pair(db)
    name = f"bix-{uuid.uuid4().hex[:12]}"
    emb = [0.0, 0.0, 0.0, 1.0]
    payload = json.dumps({"name": name, "embedding": emb})
    _cmd(db, f"INSERT INTO {_BIX_SUB} CONTENT {payload}")

    idx = db.create_vector_index(
        _BIX_BASE,
        "embedding",
        _DIM,
        distance_function="cosine",
        build_graph_now=True,
    )
    pairs = idx.find_nearest(emb, k=5)
    assert isinstance(pairs, list), pairs
    # May be 0 or >=1 depending on how JVector buckets subtype rows for a base TypeIndex.
    assert len(pairs) >= 1, (
        f"find_nearest on base-created index returned no hits for subtype-only row: {pairs!r}"
    )


def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def _pause_before_embedded_shutdown(
    studio_url: str, db_folder_name: str, root_password: str
) -> None:
    print(
        f"\nEmbedded HTTP server is running.\n"
        f"  Studio: {studio_url}\n"
        f"  Root user password (this run): {root_password!r}\n"
        f"  Pick database: {db_folder_name!r} (under server root / Studio picker)\n"
    )
    if not sys.stdin.isatty():
        print("(stdin is not a TTY — not waiting; shutting down.)")
        return
    try:
        input("Press Enter to stop the server, close the DB, and delete temp files... ")
    except EOFError:
        pass


def main() -> None:
    if arcadedb is None:
        raise SystemExit("Install arcadedb_embedded (see project venv / pyproject).")
    tmp = tempfile.mkdtemp(prefix="arcadedb_emb_inherit_cli_")
    root = os.path.join(tmp, "root")
    os.makedirs(root, exist_ok=True)
    db_folder = _DB_DIRNAME + "_cli"
    server = None
    db = None
    try:
        server = arcadedb.create_server(
            root_path=root,
            root_password=_CLI_ROOT_PASSWORD,
            config={"http_port": _CLI_EMBEDDED_HTTP_PORT},
        )
        server.start()
        _suppress_java_info_logging()
        _quiet_python_loggers()
        db = server.create_database(db_folder)
        _ensure_main_schema(db)
        _ensure_control_schema(db)
        _ensure_java_api_vertex_types(db)

        _print_section("1) Polymorphic / inherited property lookup")
        _wipe_main(db)
        name = f"cli-{uuid.uuid4().hex[:12]}"
        emb = [1.0, 0.0, 0.0, 0.0]
        payload = json.dumps({"name": name, "embedding": emb})
        _cmd(db, f"INSERT INTO {_SUB} CONTENT {payload}")
        print(f"INSERT INTO {_SUB} name={name!r}")
        from_sub = _select(db, f"SELECT FROM {_SUB} WHERE name = '{name}'")
        from_base = _select(db, f"SELECT FROM {_BASE} WHERE name = '{name}'")
        print(f"SELECT FROM {_SUB} WHERE name = ... -> {len(from_sub)} row(s)")
        print(f"SELECT FROM {_BASE} WHERE name = ... -> {len(from_base)} row(s) (polymorphic)")

        _print_section("2) Vector: LSM_VECTOR on base only, row on subtype (SQL)")
        q_base = (
            f"SELECT vectorNeighbors('{_BASE}[embedding]', {json.dumps(emb)}, 5) "
            f"AS neighbors FROM {_BASE} LIMIT 1"
        )
        raw_base = _select(db, q_base)
        print(
            f"vectorNeighbors('{_BASE}[embedding]', q) FROM {_BASE}: "
            f"{_neighbor_count(raw_base[0] if raw_base else None)} neighbors"
        )
        q_cross = (
            f"SELECT vectorNeighbors('{_BASE}[embedding]', {json.dumps(emb)}, 5) "
            f"AS neighbors FROM {_SUB} LIMIT 1"
        )
        raw_cross = _select(db, q_cross)
        print(
            f"vectorNeighbors('{_BASE}[embedding]', q) FROM {_SUB}: "
            f"{_neighbor_count(raw_cross[0] if raw_cross else None)} neighbors"
        )
        q_sub_idx = (
            f"SELECT vectorNeighbors('{_SUB}[embedding]', {json.dumps(emb)}, 5) "
            f"AS neighbors FROM {_SUB} LIMIT 1"
        )
        raw_sub = _select(db, q_sub_idx)
        print(
            f"vectorNeighbors('{_SUB}[embedding]', q) FROM {_SUB}: "
            f"{_neighbor_count(raw_sub[0] if raw_sub else None)} neighbors"
        )

        _print_section("3) Control: LSM_VECTOR on subtype only (SQL)")
        _wipe_control(db)
        emb2 = [0.0, 0.0, 1.0, 0.0]
        payload2 = json.dumps({"name": f"cli-ctl-{uuid.uuid4().hex[:8]}", "embedding": emb2})
        _cmd(db, f"INSERT INTO {_CV_SUB} CONTENT {payload2}")
        q_ctl = (
            f"SELECT vectorNeighbors('{_CV_SUB}[embedding]', {json.dumps(emb2)}, 5) "
            f"AS neighbors FROM {_CV_SUB} LIMIT 1"
        )
        raw_ctl = _select(db, q_ctl)
        print(
            f"vectorNeighbors('{_CV_SUB}[embedding]', q) FROM {_CV_SUB}: "
            f"{_neighbor_count(raw_ctl[0] if raw_ctl else None)} neighbors"
        )

        _print_section("4) Java API: get_vector_index (base + sub) + create_vector_index")
        _wipe_main(db)
        emb_api = [0.15, 0.15, 0.15, 0.55]
        payload_api = json.dumps({"name": f"cli-jwrap-{uuid.uuid4().hex[:8]}", "embedding": emb_api})
        _cmd(db, f"INSERT INTO {_SUB} CONTENT {payload_api}")
        vb = db.schema.get_vector_index(_BASE, "embedding")
        vs = db.schema.get_vector_index(_SUB, "embedding")
        if vb is not None:
            hb = vb.find_nearest(emb_api, k=3)
            print(f"schema.get_vector_index({_BASE!r}, 'embedding').find_nearest -> {len(hb)} hit(s) ")
        else:
            print(f"schema.get_vector_index({_BASE!r}, 'embedding') -> None")
        if vs is not None:
            hs = vs.find_nearest(emb_api, k=3)
            print(f"schema.get_vector_index({_SUB!r}, 'embedding').find_nearest -> {len(hs)} hit(s)")
        else:
            print(f"schema.get_vector_index({_SUB!r}, 'embedding') -> None")

        _wipe_api_pair(db)
        emb_c = [0.11, 0.22, 0.33, 0.44]
        _cmd(
            db,
            f"INSERT INTO {_API_SUB} CONTENT {json.dumps({'name': 'cli-api', 'embedding': emb_c})}",
        )
        idx_c = db.create_vector_index(
            _API_SUB, "embedding", _DIM, distance_function="cosine", build_graph_now=True
        )
        pc = idx_c.find_nearest(emb_c, k=3)
        print(f"create_vector_index({_API_SUB!r}, ...).find_nearest -> {len(pc)} hit(s)")

        _print_section("Done")
        _pause_before_embedded_shutdown(
            server.get_studio_url(), db_folder, _CLI_ROOT_PASSWORD
        )
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass
        if server is not None:
            try:
                server.stop()
            except Exception:
                pass
        gc.collect()
        try:
            shutil.rmtree(tmp)
        except PermissionError:
            time.sleep(0.5)
            try:
                shutil.rmtree(tmp)
            except PermissionError:
                pass


if __name__ == "__main__":
    main()
