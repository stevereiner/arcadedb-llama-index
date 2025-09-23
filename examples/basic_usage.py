#!/usr/bin/env python3
"""
Basic usage example for ArcadeDB PropertyGraphStore integration.

This example demonstrates:
1. Setting up ArcadeDB PropertyGraphStore
2. Creating a PropertyGraphIndex
3. Extracting knowledge from documents
4. Querying the knowledge graph

Requirements:
- ArcadeDB server running on localhost:2480
- ArcadeDB Python client: pip install arcadedb-python
- LlamaIndex ArcadeDB integration: pip install llama-index-graph-stores-arcadedb
- LlamaIndex: uv pip install llama-index-core llama-index-embeddings-openai llama-index-llms-openai
"""

import os
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LlamaIndex imports
from llama_index.core import PropertyGraphIndex, Document
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Import our ArcadeDB implementation
from llama_index.graph_stores.arcadedb import ArcadeDBPropertyGraphStore


def main():
    """Main example function demonstrating ArcadeDB integration."""
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set your OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return

    print("🚀 Initializing ArcadeDB PropertyGraphStore...")
    
    # Step 1: Initialize ArcadeDB PropertyGraphStore
    graph_store = ArcadeDBPropertyGraphStore(
        host="localhost",
        port=2480,
        username="root",
        password="playwithdata",  # Default ArcadeDB password
        database="knowledge_graph",
        create_database_if_not_exists=True,
        include_basic_schema=True,  # Include common entity types
        embedding_dimension=1536  # OpenAI text-embedding-3-small
    )
    
    print("✅ Connected to ArcadeDB successfully!")
    
    # Step 2: Create sample documents
    print("\n📚 Preparing sample documents...")
    
    documents = [
        Document(
            text="John Newton was Chairman and CTO of Alfresco during the development of CMIS in 2008. "
                 "He played a key role in creating the Content Management Interoperability Services standard."
        ),
        Document(
            text="CMIS (Content Management Interoperability Services) is a standard that enables "
                 "interoperability between different content management systems. It was developed by OASIS."
        ),
        Document(
            text="The CMIS draft specification is backed by multiple major organizations including "
                 "Alfresco, EMC, IBM, Microsoft, OpenText, Oracle and SAP."
        ),
        Document(
            text="The CMIS standard allows applications to work with content repositories from different "
                 "vendors through a common set of services and protocols."
        ),
    ]
    
    print(f"📄 Created {len(documents)} documents for processing")
    
    # Step 3: Set up knowledge extraction
    print("\n🧠 Setting up knowledge extraction...")
    
    kg_extractor = SchemaLLMPathExtractor(
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.0),
        # You can customize entity and relation types here
        strict=False,  # Allow flexible extraction
    )
    
    # Step 4: Create PropertyGraphIndex
    print("\n🏗️  Building PropertyGraphIndex...")
    
    index = PropertyGraphIndex.from_documents(
        documents,
        property_graph_store=graph_store,
        embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
        kg_extractors=[kg_extractor],
        show_progress=True,
    )
    
    print("✅ PropertyGraphIndex created successfully!")
    
    # Step 5: Query the knowledge graph
    print("\n🔍 Querying the knowledge graph...")
    
    query_engine = index.as_query_engine(
        include_text=True,
        similarity_top_k=5
    )
    
    # Ask questions
    questions = [
        "Who was John Newton and what was his role at Alfresco?",
        "What is CMIS and what does it do?", 
        "Which organizations backed the CMIS draft specification?",
        "What organization developed the CMIS standard?",
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        response = query_engine.query(question)
        print(f"💡 Answer: {response}")
    
    # Step 6: Direct graph operations
    print("\n🕸️  Performing direct graph operations...")
    
    # Get all entities
    all_nodes = graph_store.get()
    print(f"📊 Found {len(all_nodes)} total nodes in the graph")
    
    # Show first few entities
    entity_nodes = [node for node in all_nodes if hasattr(node, 'name')][:5]
    print("👥 Sample entities:")
    for node in entity_nodes:
        print(f"  - {node.name} ({node.label})")
    
    # Step 7: Execute custom queries
    print("\n🔧 Executing custom queries...")
    
    # SQL query (faster)
    sql_results = graph_store.structured_query(
        "SELECT name, @class FROM Entity LIMIT 5"
    )
    print("🏢 Entities (SQL query):")
    for result in sql_results:
        print(f"  - {result.get('name', 'Unknown')} ({result.get('@class', 'Unknown')})")
    
    # Get relationships
    triplets = graph_store.get_triplets()
    print(f"🔗 Found {len(triplets)} relationships in the graph")
    if triplets:
        print("Sample relationships:")
        for triplet in triplets[:3]:
            print(f"  - {triplet.subject} -> {triplet.predicate} -> {triplet.object}")
    
    # Step 8: Schema information
    print("\n📋 Graph schema information...")
    
    schema = graph_store.get_schema()
    print("📈 Database schema:")
    print(f"  Vertex types: {len(schema.get('vertex_types', []))}")
    print(f"  Edge types: {len(schema.get('edge_types', []))}")
    
    print("\n🎉 ArcadeDB PropertyGraphStore example completed successfully!")
    print("\n💡 Your GraphRAG application is now powered by ArcadeDB!")
    
    return index, graph_store


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure ArcadeDB server is running:")
        print("   docker run --rm -p 2480:2480 -p 2424:2424 \\")
        print("     -e JAVA_OPTS='-Darcadedb.server.rootPassword=playwithdata' \\")
        print("     arcadedata/arcadedb:latest")
        print("2. Install arcadedb-python: pip install arcadedb-python")
        print("3. Set OPENAI_API_KEY environment variable")
        print("4. Check your connection parameters")
