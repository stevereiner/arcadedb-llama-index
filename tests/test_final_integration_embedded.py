"""Embedded-mode counterpart of tests/test_final_integration.py.

Runs a realistic end-to-end PropertyGraphIndex ingest against an in-process
embedded ArcadeDB database.  No Docker, no running server, no OpenAI key
required — a simple mock LLM and embedding model are used so the test is
fully self-contained.

Skipped automatically if ``arcadedb_embedded`` is not installed.
"""

import os
import shutil
import tempfile

import pytest

from llama_index.graph_stores.arcadedb import ArcadeDBPropertyGraphStore, EMBEDDED_AVAILABLE

pytestmark = pytest.mark.skipif(
    not EMBEDDED_AVAILABLE,
    reason="arcadedb_embedded is not installed",
)


def test_final_integration_embedded() -> None:
    """End-to-end ingest + query using embedded ArcadeDB and mock LLM/embeddings."""
    try:
        from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex
        from llama_index.core.llms import MockLLM
        from llama_index.core.embeddings import MockEmbedding
    except ImportError as e:
        pytest.skip(f"llama-index-core components not available: {e}")

    tmp_dir = tempfile.mkdtemp(prefix="arcadedb_emb_final_")
    db_path = os.path.join(tmp_dir, "final_test")
    txt_file = os.path.join(tmp_dir, "test_content.txt")

    try:
        with open(txt_file, "w") as f:
            f.write(
                "John Newton was Chairman and CTO of Alfresco at the time of CMIS development.\n"
                "CMIS (Content Management Interoperability Services) is a standard backed by "
                "Alfresco, EMC, IBM, Microsoft, OpenText, Oracle and SAP.\n"
                "The CMIS standard was developed by OASIS.\n"
            )

        store = ArcadeDBPropertyGraphStore(
            mode="embedded",
            db_path=db_path,
            embedded_server=False,
            include_basic_schema=True,
            embedding_dimension=4,
        )

        documents = SimpleDirectoryReader(input_files=[txt_file]).load_data()

        # Mock LLM and embedding — no API key needed
        llm = MockLLM(max_tokens=256)
        embed_model = MockEmbedding(embed_dim=4)

        index = PropertyGraphIndex.from_documents(
            documents,
            llm=llm,
            embed_model=embed_model,
            property_graph_store=store,
            show_progress=False,
        )

        assert index is not None

        all_nodes = store.get()
        all_relations = store.get_triplets()

        # MockLLM may not extract real triplets, but the store must be queryable
        assert isinstance(all_nodes, list)
        assert isinstance(all_relations, list)

    finally:
        import gc, time
        gc.collect()
        try:
            shutil.rmtree(tmp_dir)
        except PermissionError:
            time.sleep(0.5)
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
