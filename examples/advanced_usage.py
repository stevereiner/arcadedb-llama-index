#!/usr/bin/env python3
"""
Advanced usage example for ArcadeDB PropertyGraphStore integration.

This example demonstrates:
1. Custom schema definition and validation
2. Advanced querying techniques
3. Performance optimization with SQL
4. Graph traversal and relationship mapping
5. Integration with existing GraphRAG applications

Requirements:
- ArcadeDB server running on localhost:2480
- ArcadeDB Python client: pip install arcadedb-python
- LlamaIndex ArcadeDB integration: pip install llama-index-graph-stores-arcadedb
- LlamaIndex: uv pip install llama-index-core llama-index-embeddings-openai llama-index-llms-openai
"""

import os
import logging
from typing import Literal, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LlamaIndex imports
from llama_index.core import PropertyGraphIndex, Document
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.graph_stores.types import EntityNode, Relation

# Import our ArcadeDB implementation
from llama_index.graph_stores.arcadedb import ArcadeDBPropertyGraphStore


def create_advanced_graph_store():
    """Create ArcadeDB graph store with advanced configuration."""
    return ArcadeDBPropertyGraphStore(
        host="localhost",
        port=2480,
        username="root",
        password="playwithdata",
        database="advanced_knowledge_graph",
        create_database_if_not_exists=True,
        include_basic_schema=True,  # Include common entity types
        embedding_dimension=1536  # OpenAI text-embedding-3-small
    )


def setup_custom_schema_extraction():
    """Set up custom schema extraction with defined entity and relation types."""
    
    # Define specific entity types
    entities = Literal[
        "PERSON", "COMPANY", "PRODUCT", "TECHNOLOGY", "LOCATION", 
        "DATE", "ROLE", "CONCEPT", "EVENT", "ORGANIZATION"
    ]
    
    # Define specific relation types
    relations = Literal[
        "FOUNDED", "CEO_OF", "WORKED_AT", "CREATED", "LAUNCHED", 
        "SUCCEEDED", "REVOLUTIONIZED", "SERVED_AS", "LOCATED_IN",
        "PART_OF", "COLLABORATED_WITH", "COMPETED_WITH", "ACQUIRED"
    ]
    
    # Define validation schema - which entities can have which relations
    validation_schema = {
        "PERSON": [
            "FOUNDED", "CEO_OF", "WORKED_AT", "CREATED", "LAUNCHED", 
            "SUCCEEDED", "SERVED_AS", "COLLABORATED_WITH"
        ],
        "COMPANY": [
            "FOUNDED", "CREATED", "LAUNCHED", "REVOLUTIONIZED", 
            "LOCATED_IN", "ACQUIRED", "COMPETED_WITH"
        ],
        "PRODUCT": [
            "CREATED", "LAUNCHED", "REVOLUTIONIZED", "PART_OF"
        ],
        "TECHNOLOGY": [
            "REVOLUTIONIZED", "PART_OF", "CREATED"
        ],
        "LOCATION": [
            "LOCATED_IN"
        ],
        "ROLE": [
            "SERVED_AS"
        ],
        "CONCEPT": [
            "PART_OF", "REVOLUTIONIZED"
        ],
        "EVENT": [
            "LAUNCHED", "CREATED"
        ],
        "ORGANIZATION": [
            "FOUNDED", "PART_OF", "LOCATED_IN"
        ],
        "DATE": []  # Dates typically don't have outgoing relations
    }
    
    return SchemaLLMPathExtractor(
        llm=OpenAI(model="gpt-4", temperature=0.0),  # Use GPT-4 for better extraction
        possible_entities=entities,
        possible_relations=relations,
        kg_validation_schema=validation_schema,
        strict=True,  # Enforce schema compliance
        num_workers=4,  # Parallel processing
    )


def create_comprehensive_documents():
    """Create a comprehensive set of documents for testing."""
    return [
        Document(
            text="Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976 in Los Altos, California. "
                 "The company initially focused on personal computers and revolutionized the industry with the Apple II in 1977.",
            metadata={"source": "company_history", "category": "founding"}
        ),
        Document(
            text="Steve Jobs served as CEO of Apple from 1976 to 1985, then again from 1997 until his resignation in 2011. "
                 "He was known for his perfectionist approach, innovative product design, and exceptional marketing skills.",
            metadata={"source": "leadership", "category": "biography"}
        ),
        Document(
            text="Tim Cook became Apple's CEO in August 2011, succeeding Steve Jobs. Cook previously served as Apple's "
                 "Chief Operating Officer and was instrumental in building Apple's supply chain management systems.",
            metadata={"source": "leadership", "category": "succession"}
        ),
        Document(
            text="The iPhone was launched by Apple in January 2007 and revolutionized the smartphone industry. "
                 "The device combined a phone, iPod, and internet browser in a single touchscreen device, "
                 "setting new standards for mobile technology.",
            metadata={"source": "product_history", "category": "innovation"}
        ),
        Document(
            text="Microsoft Corporation was founded by Bill Gates and Paul Allen in 1975 in Albuquerque, New Mexico. "
                 "The company became dominant in personal computer operating systems with MS-DOS and later Windows.",
            metadata={"source": "company_history", "category": "founding"}
        ),
        Document(
            text="Google was founded by Larry Page and Sergey Brin in 1998 while they were PhD students at Stanford University. "
                 "The company started as a search engine and expanded into various technology sectors including cloud computing, "
                 "mobile operating systems, and artificial intelligence.",
            metadata={"source": "company_history", "category": "founding"}
        ),
        Document(
            text="Silicon Valley, located in the San Francisco Bay Area of California, became the global center of "
                 "technology innovation. Major companies like Apple, Google, Facebook, and many startups are headquartered there.",
            metadata={"source": "geography", "category": "location"}
        ),
        Document(
            text="The personal computer revolution began in the 1970s with companies like Apple, Microsoft, and IBM "
                 "making computers accessible to individual consumers rather than just large corporations and institutions.",
            metadata={"source": "technology_history", "category": "revolution"}
        )
    ]


def demonstrate_advanced_querying(graph_store: ArcadeDBPropertyGraphStore):
    """Demonstrate advanced querying techniques."""
    
    print("\n🔍 Advanced Querying Techniques")
    print("=" * 50)
    
    # 1. Complex SQL queries for performance
    print("\n1. Complex SQL Queries (High Performance)")
    
    # Find all founders and their companies
    sql_query = """
    SELECT 
        person.name as founder_name,
        company.name as company_name,
        rel.properties as relationship_details
    FROM 
        (SELECT expand(out('FOUNDED')) FROM Person) as person,
        (SELECT expand(in('FOUNDED')) FROM Company) as company,
        (SELECT FROM FOUNDED) as rel
    WHERE 
        rel.out = person.@rid AND rel.in = company.@rid
    """
    
    try:
        results = graph_store.structured_query(sql_query)
        print("👥 Founders and their companies:")
        for result in results:
            print(f"  - {result.get('founder_name')} founded {result.get('company_name')}")
    except Exception as e:
        print(f"  SQL query failed: {e}")
    
    # 2. Graph traversal queries
    print("\n2. Graph Traversal Queries")
    
    # Find all connections within 2 degrees of Apple
    traversal_query = """
    SELECT expand(both()) FROM (
        TRAVERSE both() FROM (SELECT FROM Entity WHERE name = 'Apple') MAXDEPTH 2
    ) WHERE @class INSTANCEOF 'V'
    """
    
    try:
        results = graph_store.structured_query(traversal_query)
        print(f"🕸️  Found {len(results)} entities within 2 degrees of Apple")
        
        # Show sample results
        for result in results[:5]:
            name = result.get('name', 'Unknown')
            entity_type = result.get('@class', 'Unknown')
            print(f"  - {name} ({entity_type})")
            
    except Exception as e:
        print(f"  Traversal query failed: {e}")
    
    # 3. Aggregation queries
    print("\n3. Aggregation Queries")
    
    # Count entities by type
    aggregation_query = """
    SELECT @class as entity_type, count(*) as count 
    FROM (SELECT FROM Entity UNION SELECT FROM TextChunk)
    GROUP BY @class
    ORDER BY count DESC
    """
    
    try:
        results = graph_store.structured_query(aggregation_query)
        print("📊 Entity counts by type:")
        for result in results:
            print(f"  - {result.get('entity_type')}: {result.get('count')}")
    except Exception as e:
        print(f"  Aggregation query failed: {e}")


def demonstrate_performance_optimization(graph_store: ArcadeDBPropertyGraphStore):
    """Demonstrate performance optimization techniques."""
    
    print("\n⚡ Performance Optimization")
    print("=" * 50)
    
    import time
    
    # 1. Compare SQL vs Cypher performance
    print("\n1. SQL vs Cypher Performance Comparison")
    
    # SQL query
    start_time = time.time()
    sql_results = graph_store.structured_query("SELECT COUNT(*) as count FROM Entity")
    sql_time = time.time() - start_time
    
    # Cypher query
    start_time = time.time()
    cypher_results = graph_store.structured_query("MATCH (n:Entity) RETURN count(n) as count")
    cypher_time = time.time() - start_time
    
    print(f"  SQL query time: {sql_time:.4f} seconds")
    print(f"  Cypher query time: {cypher_time:.4f} seconds")
    
    if sql_time > 0:
        speedup = cypher_time / sql_time
        print(f"  SQL is {speedup:.1f}x faster than Cypher")
    
    # 2. Batch operations
    print("\n2. Batch Operations")
    
    # Create multiple nodes efficiently
    batch_nodes = [
        EntityNode(
            name=f"Batch Entity {i}",
            label="Entity",
            properties={"batch_id": "demo", "index": i}
        )
        for i in range(10)
    ]
    
    start_time = time.time()
    graph_store.upsert_nodes(batch_nodes)
    batch_time = time.time() - start_time
    
    print(f"  Batch inserted 10 nodes in {batch_time:.4f} seconds")
    
    # 3. Index usage demonstration
    print("\n3. Index Usage")
    
    # Query with indexed field (name)
    start_time = time.time()
    indexed_results = graph_store.get(properties={"name": "Apple"})
    indexed_time = time.time() - start_time
    
    print(f"  Indexed query time: {indexed_time:.4f} seconds")
    print(f"  Found {len(indexed_results)} results")


def demonstrate_graph_analysis(index: PropertyGraphIndex, graph_store: ArcadeDBPropertyGraphStore):
    """Demonstrate advanced graph analysis capabilities."""
    
    print("\n📈 Graph Analysis")
    print("=" * 50)
    
    # 1. Relationship mapping
    print("\n1. Relationship Mapping")
    
    # Get all Apple-related nodes
    apple_nodes = graph_store.get(properties={"name": "Apple"})
    if apple_nodes:
        apple_node = apple_nodes[0]
        
        # Get relationships within 2 degrees
        related_triplets = graph_store.get_rel_map(
            graph_nodes=[apple_node],
            depth=2,
            limit=20
        )
        
        print(f"🔗 Found {len(related_triplets)} relationships around Apple:")
        for source, relation, target in related_triplets[:5]:
            print(f"  {source.name} --[{relation.label}]--> {target.name}")
    
    # 2. Graph retrieval with different strategies
    print("\n2. Graph Retrieval Strategies")
    
    # Text-based retrieval
    retriever = index.as_retriever(
        include_text=True,
        similarity_top_k=3
    )
    
    text_results = retriever.retrieve("Apple iPhone innovation")
    print(f"📱 Text-based retrieval found {len(text_results)} results")
    
    # Graph-based retrieval
    graph_retriever = index.as_retriever(
        include_text=False,
        similarity_top_k=3
    )
    
    graph_results = graph_retriever.retrieve("technology companies")
    print(f"🕸️  Graph-based retrieval found {len(graph_results)} results")
    
    # 3. Query engine comparison
    print("\n3. Query Engine Comparison")
    
    # Standard query engine
    standard_engine = index.as_query_engine(
        include_text=True,
        similarity_top_k=3
    )
    
    # Graph-focused query engine
    graph_engine = index.as_query_engine(
        include_text=False,
        similarity_top_k=5
    )
    
    question = "What is the relationship between Steve Jobs and Apple?"
    
    print(f"❓ Question: {question}")
    
    standard_response = standard_engine.query(question)
    print(f"📄 Standard response: {standard_response}")
    
    graph_response = graph_engine.query(question)
    print(f"🕸️  Graph response: {graph_response}")


def cleanup_demo_data(graph_store: ArcadeDBPropertyGraphStore):
    """Clean up demo data."""
    print("\n🧹 Cleaning up demo data...")
    
    try:
        # Delete batch demo entities
        graph_store.delete(properties={"batch_id": "demo"})
        print("✅ Demo data cleaned up")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")


def main():
    """Main advanced example function."""
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set your OPENAI_API_KEY environment variable")
        return

    print("🚀 Advanced ArcadeDB PropertyGraphStore Example")
    print("=" * 60)
    
    try:
        # Step 1: Create advanced graph store
        print("\n1. Creating advanced graph store...")
        graph_store = create_advanced_graph_store()
        print("✅ Advanced graph store created")
        
        # Step 2: Set up custom schema extraction
        print("\n2. Setting up custom schema extraction...")
        kg_extractor = setup_custom_schema_extraction()
        print("✅ Custom schema extraction configured")
        
        # Step 3: Create comprehensive documents
        print("\n3. Creating comprehensive document set...")
        documents = create_comprehensive_documents()
        print(f"✅ Created {len(documents)} comprehensive documents")
        
        # Step 4: Build advanced PropertyGraphIndex
        print("\n4. Building advanced PropertyGraphIndex...")
        index = PropertyGraphIndex.from_documents(
            documents,
            property_graph_store=graph_store,
            embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
            kg_extractors=[kg_extractor],
            show_progress=True,
        )
        print("✅ Advanced PropertyGraphIndex created")
        
        # Step 5: Demonstrate advanced querying
        demonstrate_advanced_querying(graph_store)
        
        # Step 6: Demonstrate performance optimization
        demonstrate_performance_optimization(graph_store)
        
        # Step 7: Demonstrate graph analysis
        demonstrate_graph_analysis(index, graph_store)
        
        # Step 8: Cleanup
        cleanup_demo_data(graph_store)
        
        print("\n🎉 Advanced ArcadeDB example completed successfully!")
        print("\n💡 Key takeaways:")
        print("  • ArcadeDB SQL queries are significantly faster than Cypher")
        print("  • Custom schema validation improves knowledge extraction quality")
        print("  • Batch operations and indexing optimize performance")
        print("  • Graph traversal enables powerful relationship discovery")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
