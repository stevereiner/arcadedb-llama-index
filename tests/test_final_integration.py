#!/usr/bin/env python3
"""
Final integration test of ArcadeDBPropertyGraphStore.

Requires:
- ArcadeDB container on port 2481 (started automatically by conftest.py)
- OPENAI_API_KEY environment variable
"""

import os
import tempfile
import pytest
import requests

try:
    from llama_index.graph_stores.arcadedb import ArcadeDBPropertyGraphStore
    from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    DEPENDENCIES_AVAILABLE = False

_TEST_PORT = 2481


def _server_available() -> bool:
    try:
        requests.get(f"http://localhost:{_TEST_PORT}", timeout=2)
        return True
    except Exception:
        return False


def test_final_integration():
    if not DEPENDENCIES_AVAILABLE:
        pytest.skip("Required dependencies not available (OpenAI, LlamaIndex components)")
    if not _server_available():
        pytest.skip(f"ArcadeDB test container not available on port {_TEST_PORT}")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    store = ArcadeDBPropertyGraphStore(
        host="localhost",
        port=_TEST_PORT,
        username="root",
        password="playwithdata",
        database="final_test",
        create_database_if_not_exists=True,
        include_basic_schema=True,
        embedding_dimension=1536,
    )

    test_content = """
    John Newton was Chairman and CTO of Alfresco at the time of CMIS development.
    CMIS (Content Management Interoperability Services) is a content management interoperability services standard.
    The CMIS draft specification is backed by multiple major organizations including Alfresco, EMC, IBM, Microsoft, OpenText, Oracle and SAP.
    The CMIS standard enables interoperability between different content management systems and was developed by OASIS.
    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_file = f.name

    try:
        documents = SimpleDirectoryReader(input_files=[temp_file]).load_data()
        llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
        embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

        print("Creating PropertyGraphIndex...")
        PropertyGraphIndex.from_documents(
            documents,
            llm=llm,
            embed_model=embed_model,
            property_graph_store=store,
            show_progress=True,
        )
        print("SUCCESS: PropertyGraphIndex created successfully!")

        all_nodes = store.get()
        all_relations = store.get_triplets()
        print(f"Total nodes: {len(all_nodes)}, relationships: {len(all_relations)}")
        assert len(all_nodes) > 0, "Expected at least one node"

    except Exception as e:
        import traceback
        traceback.print_exc()
        assert False, f"Integration test failed: {e}"

    finally:
        try:
            os.unlink(temp_file)
        except Exception:
            pass


if __name__ == "__main__":
    test_final_integration()
