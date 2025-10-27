# Changelog

All notable changes to this project will be documented in this file.

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