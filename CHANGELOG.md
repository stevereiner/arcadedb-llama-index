# Changelog

All notable changes to this project will be documented in this file.

## [2026-05-11] - 0.4.4 Release: Cypher query routing and vertex/edge label collision guard

### Fixed

- **`arcadedb_property_graph.py` — `structured_query()`: OpenCypher detection**: the method previously routed all queries through ArcadeDB's SQL engine, causing `SQL syntax error` failures when LlamaIndex issued OpenCypher queries (`MATCH`, `MERGE`, `OPTIONAL MATCH`, `CREATE (`, `WITH`, `CALL`) during graph traversal and retrieval. The method now detects these keywords and routes them to ArcadeDB's OpenCypher engine (`self._db.query("opencypher", query)`), while all other queries (DDL/DML) continue through the SQL engine with `is_command=True` where appropriate.

- **`arcadedb_property_graph.py` — `upsert_relations()`: vertex/edge label collision guard**: ArcadeDB shares a single namespace for vertex types and edge types. When the LLM extracts a label used for both a node type and a relation type (e.g. `DATE`), creating an edge of that type raised `"Type 'DATE' is not an Edge type"`. `upsert_relations()` now checks each relation label against `self._discovered_vertex_types` before upserting; any colliding label has `_REL` appended (e.g. `DATE` → `DATE_REL`), matching the equivalent fix already present in the LangChain ArcadeDB adapter.

### Changed

- `pyproject.toml` version `0.4.3` → `0.4.4`

---

## [2026-05-04] - 0.4.3 Release: Guard against empty/None type names and relation labels

### Fixed

- **`arcadedb_property_graph.py` — `_ensure_dynamic_type`**: added an early-return guard when `type_name` is `None` or blank. Without this, an empty string was interpolated directly into the DDL, producing malformed SQL such as `CREATE EDGE TYPE  IF NOT EXISTS`, which raises a `SemanticError` in ArcadeDB. Empty names are now logged at WARNING and the method returns `False` immediately.
- **`arcadedb_property_graph.py` — `_add_relation`**: added a guard before edge-type creation and edge insertion when `relation.label` is `None` or blank. Relations with an empty label are logged at WARNING and the method returns early; the `_ensure_dynamic_type` call and subsequent INSERT that follow are never reached.

### Changed

- `pyproject.toml` version `0.4.2` → `0.4.3`

---

## [2026-03-22] - 0.4.2 Release: Entity base-type inheritance, real vector search scores, TextChunk vector index, logging suppression

### Added

- **Inheritance index tests** — `tests/test_remote_inheritance_indexes.py` and `tests/test_embedded_inheritance_indexes.py` verify that a `UNIQUE` property index and an `LSM_VECTOR` index on a base type are correctly inherited and usable from subtypes, in both remote and embedded modes.

### Changed

- **Vector search scores** — `vector_query()` now returns real differentiated cosine similarity scores per entity type by searching each concrete subtype individually (e.g. `vectorNeighbors('PERSON[embedding]', q) FROM PERSON`). Previously all results could share a uniform score; now the merged top-k reflects actual semantic relevance, improving graph retrieval ranking in downstream consumers such as flexible-graphrag.
- **Entity inheritance model** — entity subtypes (`PERSON`, `ORGANIZATION`, `LOCATION`, `PLACE`, and dynamically created types) now extend a common `Entity` base vertex type. The `UNIQUE(name)` index is created per subtype; the `LSM_VECTOR(embedding)` index is created once on `Entity` and inherited by all subtypes. `TextChunk` remains separate from this hierarchy with its own `UNIQUE(id)` and `LSM_VECTOR(embedding)` indexes.
- **`DuplicatedKeyException` handling** — concurrent upserts of the same entity name no longer log spurious warnings; a fallback `UPDATE ... WHERE name = ?` is issued automatically and duplicate-key exceptions in the bulk upsert path are demoted to DEBUG.
- **Java logging suppression** (`_db_adapter.py`) — a JUL `Filter` (via `JProxy`) is installed on the root handler at startup to block all `com.arcadedb.index.*` log records. This covers `LSMVectorIndexPageParser` "Invalid position" / "FALLBACK" warnings across all threads including embedded server worker threads. `_suppress_page_parser_once()` additionally silences any newly registered index loggers on the first vector search.
- **Python driver logging suppression** — `arcadedb_python.api.sync` logger (which logs every non-2xx HTTP response at ERROR) is suppressed to CRITICAL around remote vector searches and remote index creation.

### Fixed

- `TextChunk.embedding` property now explicitly created before the `LSM_VECTOR` index is applied, preventing schema errors on fresh databases.
- `_cache_existing_vector_index` no longer emits a WARNING when the Java schema API cannot find a subtype index by name (expected behaviour when the index lives on the `Entity` base); demoted to DEBUG.

### Dependencies

- Package version `0.4.1` → `0.4.2`
- Tested with Python 3.13 and 3.14, `arcadedb-python>=0.4.0`, optional `arcadedb-embedded>=26.3.2`

---

## [2026-03-07] - 0.4.1 Release: embedded ArcadeDB in-process mode

### Added
- **`_db_adapter.py`** — adapter layer (`RemoteAdapter`, `EmbeddedAdapter`) unifying remote HTTP and embedded in-process access; handles Java→Python result conversion and JUL log suppression
- **`arcadedb_property_graph.py`** — `mode="embedded"` support with `db_path`, `embedded_server`, `embedded_server_port`, `embedded_server_password` params and `embedded_server_url` property; `faulthandler` management around JPype import
- **`__init__.py`** — exports `EMBEDDED_AVAILABLE`
- **`examples/embedded_server.py`** — script to start Studio UI against an existing embedded database
- **`base.py` (`ArcadeDBGraphStore`)** — embedded mode support added (`mode`, `db_path`, `embedded_server` params) via the same adapter pattern as `ArcadeDBPropertyGraphStore`
- **4 new test files** mirroring the remote integration suite for embedded mode: `test_arcadedb_embedded_integration.py`, `test_pg_stores_arcadedb_embedded.py`, `test_graph_stores_arcadedb_embedded.py`, `test_final_integration_embedded.py`; embedded tests share one store/JVM per module (no reconnect per test)
- **`conftest.py`** — session-scoped Docker container startup/teardown on port 2481 (independent of any user container on 2480); `pytest_unconfigure` calls `os._exit(0)` to prevent JPype JVM thread hang after tests
- **`pytest.ini`** — `addopts = -p no:faulthandler` to suppress JPype signal-handler console dumps

### Fixed
- **Embedded vector search** (`_db_adapter.py`) — use SQL `vectorNeighbors()` instead of direct Java index access; eliminates `FALLBACK: Could not read vectors from pages` and `Invalid position` errors
- **Java String Pydantic errors** (`_db_adapter.py`, `arcadedb_property_graph.py`) — coerce all Java `String` values to Python `str` before passing to LlamaIndex node constructors
- **Graph triplets missing in embedded mode** (`arcadedb_property_graph.py`) — `get_rel_map` switched to `SELECT expand(bothE())` for reliable edge traversal
- **`SchemaException` not imported** (`arcadedb_property_graph.py`) — was caught but never imported; added to `arcadedb_python` import block


### Changed
- **`requirements.txt`** / **`pyproject.toml`** — `arcadedb-embedded>=26.2.1` added as commented-out / optional (`[embedded]`) dependency
- **`README.md`** — rewritten with remote vs embedded modes, installation, usage examples, and full parameter table
- **Trace-level log lines demoted from INFO to DEBUG** (`arcadedb_property_graph.py`, `_db_adapter.py`)
### Dependencies

- Package version `0.4.0` → `0.4.1`
- `arcadedb-embedded>=26.2.1` optional (package version = bundled ArcadeDB version; ~95 MB installed)

---

## [2026-03-02] - 0.4.0 Release: native vector support, fixed delete nodes, added tests

### Added
- `vector_query()` implemented using ArcadeDB's native LSM_VECTOR index
- Two-phase embedding write (`_update_embedding()`) 
- `_ensure_property()` — dynamic schema property declaration with cache
- `_sql_identifier()` — backtick-quotes property names containing spaces (e.g. `modified at`)
- `_get_all_vertex_types()` — live schema query helper

### Changed
- Fixed delete nodes so flexible-graphrag auto incremental update could use
- `get_rel_map()` rewritten: resolves node names to `@rid` values before traversal, reads correct `@in`/`@out` edge fields, replaces broken OrientDB MATCH syntax
- `upsert_nodes()` restructured; 
- Exception handling updated to use typed exceptions (`VectorOperationException`, `TransactionException`, `SchemaException`) from arcadedb-python
- "Already exists" on LSM_VECTOR index creation demoted from ERROR to DEBUG on reconnect
- Entity classification fixes for acronyms and multi-word company names
- 4 test files updated; 6 new delete/vector tests added to `test_arcadedb_integration.py`

### Dependencies
- Package version `0.3.1` → `0.4.0`; requires `arcadedb-python>=0.4.0`

---

## [2025-10-26] - 0.3.1 release: use arcadedb-python 0.3.1, updates to pass all tests, and not get some query errors

### Changed

**In `graph_stores/llama-index-graph-stores-arcadedb/base.py` (ArcadeDBGraphStore):**
- use DatabseDao.exists to check if need to create database
- in clear_database delete from all vertex types instead of non-existent E type,
- skip trying _db.get_triplets since needs updating to not use E type, and just use other basic method

**In `graph_stores/llama-index-graph-stores-arcadedb/arcade_property_graph.py` (ArcadeDBPropertyGraphStore):**
- use DatabseDao.exists to check if need to create database, when creating properties add check "if not exists"
- in get(), query by node id search all vertex type tables not just Entity and TextChunk tables for both by id and by properties
- in get_triplets, skip trying to use db.get_triplets oy MATCH, and use basic methods
- in upsert_nodes, for bulk upsert use a sqlscript
- in delete() 3 selects were change to have their missing "*"
- in vector_query() changed to return empty results since not supported in database or arcadedb-python yet and doing a vector operation manually was removed
- in `_properties_to_sql_string()` change to properly escape json data
- in `_escape_string` removed escaded.replace code that was causing problems (that removed apostrophes, quotes, backticks, replaced '\n' '\r' '\t' with ' ') with code that first replaced unicode characters, the escaded.replace with extra \\

### Removed
- removed `_vector_query_manual_fallback`, `_calculate_cosine_similarity`

### Tests
- standard llama-index tests updated in `graph_stores/llama-index-graph-stores-arcadedb/tests` tests updated:`test_graph_stores_arcadedb.py`, `test_pg_stores_arcadedb.py`
- extra `/tests` updated `test_auth.py`, `test_arcadedb_integration.py`, and `test_final_integration.py`

### Dependencies
- In `graph_stores/llama-index-graph-stores-arcadedb/pyproject.py` the package version was updated to 0.3.1  version and to depend on arcadedb-python 0.3.1
- In `graph_stores/llama-index-graph-stores-arcadedb/__init__.py` the version was updated to 0.3.1
- In `requirements.txt`, dependency changed from to arcadedb-python 0.3.1

### Infrastructure
- In `docker-compose.yml`, the arcadedb docker image was switched from the latest to 25.9.1 release

---

## [2025-09-24] - 0.3.0 Release: use arcadedb-python 0.3.0, other improvements

### Added
- Integrated bulk operations: Added bulk_upsert(), bulk_delete(), and safe_delete_all() method calls with fallback to individual operations when bulk methods fail or are unavailable

### Changed
- Enhanced query methods: Updated get_triplets() to use v0.3.0 API with subject_types, relation_types, object_types parameters and improved error handling with typed exceptions
- Updated both ArcadeDBGraphStore and ArcadeDBPropertyGraphStore to leverage enhanced get_triplets() method, typed exception handling (ArcadeDBException, QueryParsingException, ValidationException),
- Enhanced Performance: Integrated arcadedb-python v0.3.0+ with bulk operations (bulk_upsert, bulk_delete, safe_delete_all)

### Dependencies
- Updated dependencies: Changed requirement from `arcadedb-python>=0.2.0` to `arcadedb-python>=0.3.0` in `pyproject.toml`

### Tests
- Updated test files

---

## [2025-09-22] - Initial Commit