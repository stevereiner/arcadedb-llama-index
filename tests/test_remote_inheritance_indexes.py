"""
Remote ArcadeDB: vertex inheritance + indexes (diagnostic).

Checks whether:
1) Rows stored only on a subtype are visible via polymorphic SELECT on the base
   and via WHERE on an inherited property (regular index on the base type).
2) With that base-only ``LSM_VECTOR``, subtype-only rows are **not** found via the
   base index name in SQL or via ``DatabaseDao``: ``vector_search('Base1', ...) -> 0``,
   while ``vector_search('Sub1', ...) -> 1``.

Requires ArcadeDB HTTP on port 2481 (pytest conftest Docker session or manual).

Run:
  .\\venv-3.13\\Scripts\\python.exe -m pytest tests/test_remote_inheritance_indexes.py -v
  .\\venv-3.13\\Scripts\\python.exe tests/test_remote_inheritance_indexes.py

References:
  https://github.com/stevereiner/arcadedb-python
  https://docs.arcadedb.com/
"""

from __future__ import annotations

import json
import os
import uuid

import pytest
import requests

try:
    from arcadedb_python import DatabaseDao, SyncClient
except ImportError:
    DatabaseDao = None  # type: ignore[misc, assignment]
    SyncClient = None  # type: ignore[misc, assignment]

_TEST_PORT = int(os.environ.get("ARCADEDB_TEST_PORT", "2481"))
_DB_NAME = "inherit_idx_remote"

# Main scenario: index on base only
_BASE = "Base1"
_SUB = "Sub1"

# Control: vector index on subtype only (proves vectorNeighbors works for this DB)
_CV_BASE = "Base2"
_CV_SUB = "Sub2"

_DIM = 4


def _server_available() -> bool:
    try:
        requests.get(f"http://localhost:{_TEST_PORT}", timeout=2)
        return True
    except Exception:
        return False


def _connect():
    assert SyncClient is not None and DatabaseDao is not None
    client = SyncClient(
        "localhost",
        _TEST_PORT,
        username="root",
        password="playwithdata",
    )
    if not DatabaseDao.exists(client, _DB_NAME):
        return DatabaseDao.create(client, _DB_NAME)
    return DatabaseDao(client, _DB_NAME)


def _cmd(db, sql: str):
    return db.query("sql", sql, is_command=True)


def _select(db, sql: str):
    return db.query("sql", sql, is_command=False)


def _neighbor_count(row) -> int:
    if not row:
        return 0
    n = row.get("neighbors")
    if not isinstance(n, list):
        return 0
    return len(n)


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


def _wipe_main(db) -> None:
    _cmd(db, f"DELETE FROM {_SUB}")
    _cmd(db, f"DELETE FROM {_BASE}")


def _wipe_control(db) -> None:
    _cmd(db, f"DELETE FROM {_CV_SUB}")
    _cmd(db, f"DELETE FROM {_CV_BASE}")


@pytest.fixture(scope="module")
def remote_db():
    if DatabaseDao is None or SyncClient is None:
        pytest.skip("arcadedb-python not installed")
    if not _server_available():
        pytest.skip(
            f"ArcadeDB not reachable on localhost:{_TEST_PORT} "
            "(start Docker test container or set ARCADEDB_TEST_PORT)"
        )
    db = _connect()
    _ensure_main_schema(db)
    _ensure_control_schema(db)
    yield db


def test_remote_polymorphic_and_subtype_select_find_same_row(remote_db):
    """Subtype-only rows must be visible from base SELECT and subtype WHERE."""
    db = remote_db
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


def test_remote_vector_neighbors_base_vs_subtype_index_names(remote_db):
    """Base-only ``LSM_VECTOR`` + subtype-only rows: SQL and ``DatabaseDao`` use concrete type.

    Observed (remote, ArcadeDB 26.x): ``vectorNeighbors('Base1[embedding]', …)`` → 0;
    ``vectorNeighbors('Sub1[embedding]', …)`` → hits.  Python client:
    ``DatabaseDao.vector_search('Base1', ...) -> 0``;
    ``vector_search('Sub1', ...) -> 1``.  Same pattern as ``ArcadeDBPropertyGraphStore``
    using per-subtype vector indexes for search.
    """
    db = remote_db
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
    n_base = _neighbor_count(raw_base[0]) if raw_base else 0

    q_sub_from = (
        f"SELECT vectorNeighbors('{_BASE}[embedding]', {json.dumps(emb)}, 5) "
        f"AS neighbors FROM {_SUB} LIMIT 1"
    )
    raw_cross = _select(db, q_sub_from)
    n_cross = _neighbor_count(raw_cross[0]) if raw_cross else 0

    q_sub_idx = (
        f"SELECT vectorNeighbors('{_SUB}[embedding]', {json.dumps(emb)}, 5) "
        f"AS neighbors FROM {_SUB} LIMIT 1"
    )
    raw_sub_idx = _select(db, q_sub_idx)
    n_sub_idx = _neighbor_count(raw_sub_idx[0]) if raw_sub_idx else 0

    dao_base_hits = len(db.vector_search(_BASE, "embedding", emb, top_k=5))
    dao_sub_hits = len(db.vector_search(_SUB, "embedding", emb, top_k=5))

    assert n_base == 0 and n_cross == 0, (
        f"Expected no neighbors via {_BASE}[embedding] for subtype-only rows; "
        f"got FROM {_BASE}: {n_base}, FROM {_SUB}: {n_cross}"
    )
    assert dao_base_hits == 0, (
        f"Expected DatabaseDao.vector_search({_BASE!r}) to return 0; got {dao_base_hits}"
    )
    assert n_sub_idx >= 1, (
        f"Expected vectorNeighbors('{_SUB}[embedding]', ...) to find the row; got {n_sub_idx}"
    )
    assert dao_sub_hits >= 1, (
        f"Expected DatabaseDao.vector_search({_SUB!r}) to find the row; got {dao_sub_hits}"
    )


def test_remote_vector_neighbors_subtype_index_control(remote_db):
    """Control: vector index on the concrete subtype must find inserted rows."""
    db = remote_db
    _wipe_control(db)
    name = f"ctl-{uuid.uuid4().hex[:12]}"
    emb = [0.0, 0.0, 1.0, 0.0]
    payload = json.dumps({"name": name, "embedding": emb})
    _cmd(db, f"INSERT INTO {_CV_SUB} CONTENT {payload}")

    hits = db.vector_search(_CV_SUB, "embedding", emb, top_k=5)
    assert len(hits) >= 1, hits


def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> None:
    if DatabaseDao is None or SyncClient is None:
        raise SystemExit("Install arcadedb-python (project venv).")
    if not _server_available():
        raise SystemExit(
            f"ArcadeDB not reachable on localhost:{_TEST_PORT}. "
            "Start the test container or export ARCADEDB_TEST_PORT."
        )
    db = _connect()
    _ensure_main_schema(db)
    _ensure_control_schema(db)

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

    _print_section("2) Vector: index only on base type, data only on subtype")
    q_base = (
        f"SELECT vectorNeighbors('{_BASE}[embedding]', {json.dumps(emb)}, 5) "
        f"AS neighbors FROM {_BASE} LIMIT 1"
    )
    raw_base = _select(db, q_base)
    print(f"vectorNeighbors('{_BASE}[embedding]', q) FROM {_BASE}: {_neighbor_count(raw_base[0]) if raw_base else 0} neighbors")
    q_cross = (
        f"SELECT vectorNeighbors('{_BASE}[embedding]', {json.dumps(emb)}, 5) "
        f"AS neighbors FROM {_SUB} LIMIT 1"
    )
    raw_cross = _select(db, q_cross)
    print(
        f"vectorNeighbors('{_BASE}[embedding]', q) FROM {_SUB}: "
        f"{_neighbor_count(raw_cross[0]) if raw_cross else 0} neighbors"
    )
    q_sub_idx = (
        f"SELECT vectorNeighbors('{_SUB}[embedding]', {json.dumps(emb)}, 5) "
        f"AS neighbors FROM {_SUB} LIMIT 1"
    )
    raw_sub = _select(db, q_sub_idx)
    print(
        f"vectorNeighbors('{_SUB}[embedding]', q) FROM {_SUB} (no sub index): "
        f"{_neighbor_count(raw_sub[0]) if raw_sub else 0} neighbors"
    )
    try:
        dao_base = db.vector_search(_BASE, "embedding", emb, top_k=5)
        dao_sub = db.vector_search(_SUB, "embedding", emb, top_k=5)
        print(
            f"DatabaseDao.vector_search({_BASE!r}, ...) -> {len(dao_base)}; "
            f"vector_search({_SUB!r}, ...) -> {len(dao_sub)}"
        )
    except Exception as e:
        print(f"DatabaseDao.vector_search failed: {e}")

    _print_section("3) Control: LSM_VECTOR index on subtype only")
    _wipe_control(db)
    emb2 = [0.0, 0.0, 1.0, 0.0]
    payload2 = json.dumps({"name": f"cli-ctl-{uuid.uuid4().hex[:8]}", "embedding": emb2})
    _cmd(db, f"INSERT INTO {_CV_SUB} CONTENT {payload2}")
    ctl = db.vector_search(_CV_SUB, "embedding", emb2, top_k=5)
    print(f"vector_search({_CV_SUB!r}, ...) -> {len(ctl)} record(s)")

    _print_section("Done")


if __name__ == "__main__":
    main()
