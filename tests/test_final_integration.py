#!/usr/bin/env python3
"""
Final test of ArcadeDB PropertyGraphStore with corrected CREATE PROPERTY and CREATE INDEX syntax.
"""

import sys
import tempfile
sys.path.append('arcadedb-integration')

from llama_index.graph_stores.arcadedb import ArcadeDBPropertyGraphStore
from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

def test_final_integration():
    print("🚀 Testing final ArcadeDB integration with corrected syntax...")
    
    # Initialize with corrected syntax
    store = ArcadeDBPropertyGraphStore(
        host="localhost",
        port=2480,
        username="root",
        password="playwithdata",
        database="final_test",
        create_database_if_not_exists=True,
        include_basic_schema=True,  # Include common entity types
        embedding_dimension=1536
    )
    
    # Test content - accurate information about John Newton and CMIS (circa 2008)
    test_content = """
    John Newton was Chairman and CTO of Alfresco at the time of CMIS development. He was instrumental in the creation of CMIS technology.
    CMIS (Content Management Interoperability Services) is a content management interoperability services standard.
    The CMIS draft specification is backed by multiple major organizations including Alfresco, EMC, IBM, Microsoft, OpenText, Oracle and SAP.
    The CMIS standard enables interoperability between different content management systems and was developed by OASIS.
    """
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_file = f.name
    
    try:
        # Load documents
        documents = SimpleDirectoryReader(input_files=[temp_file]).load_data()
        
        # Initialize LLM and embeddings
        llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
        embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
        
        print("📊 Creating PropertyGraphIndex...")
        index = PropertyGraphIndex.from_documents(
            documents,
            llm=llm,
            embed_model=embed_model,
            property_graph_store=store,
            show_progress=True,
        )
        
        print("✅ SUCCESS: PropertyGraphIndex created successfully!")
        
        # Test queries
        all_nodes = store.get()
        all_relations = store.get_triplets()
        
        print(f"📈 Results:")
        print(f"  - Total nodes: {len(all_nodes)}")
        print(f"  - Total relationships: {len(all_relations)}")
        print(f"  - Database: final_test")
        print(f"  - Mode: SQL with native UPSERT")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        import os
        try:
            os.unlink(temp_file)
        except:
            pass

if __name__ == "__main__":
    success = test_final_integration()
    if success:
        print("🎉 FINAL INTEGRATION TEST PASSED!")
    else:
        print("💥 FINAL INTEGRATION TEST FAILED!")
