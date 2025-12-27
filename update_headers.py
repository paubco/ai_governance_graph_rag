# -*- coding: utf-8 -*-
"""
Script to update Python file headers to standardized format.

This script reads Python files, understands their purpose, and writes proper
headers with descriptions, examples, and references following the agreed format.
"""

import sys
from pathlib import Path

def update_header(filepath: str, new_header: str):
    """Update a file's header, preserving everything after the docstring."""
    path = Path(filepath)
    content = path.read_text()

    # Find where the first docstring ends
    lines = content.split('\n')

    # Skip encoding line and find docstring end
    in_docstring = False
    docstring_end_line = 0

    for i, line in enumerate(lines):
        if i == 0 and line.startswith('# -*- coding:'):
            continue
        if '"""' in line:
            if not in_docstring:
                in_docstring = True
            else:
                docstring_end_line = i + 1
                break

    # Keep everything after the old header
    rest_of_file = '\n'.join(lines[docstring_end_line:])

    # Write new header + rest
    new_content = new_header.rstrip() + '\n' + rest_of_file
    path.write_text(new_content)
    print(f"✓ Updated {filepath}")


# Headers for each file
HEADERS = {}

# src/retrieval/chunk_retriever.py
HEADERS['src/retrieval/chunk_retriever.py'] = '''# -*- coding: utf-8 -*-
"""
Chunk retrieval for GraphRAG pipeline with dual-track retrieval.

Implements two parallel retrieval strategies: (1) semantic retrieval using FAISS
vector similarity over the full chunk corpus, and (2) graph-aware retrieval that
fetches chunks connected to expanded entities via provenance relations. Combines
results from both tracks for comprehensive coverage.

The semantic track performs top-K similarity search against chunk embeddings. The
graph track retrieves chunks that mention entities in the PCST subgraph, split by
provenance type (relation-mentioned vs entity-mentioned). All retrieved chunks
include metadata (doc_id, doc_type, jurisdiction, score, source_path) for ranking.

Examples:
    # Initialize with FAISS indices
    from src.retrieval.chunk_retriever import ChunkRetriever
    retriever = ChunkRetriever(
        faiss_chunk_index_path="data/processed/faiss/chunk_embeddings.index",
        chunk_ids_path="data/processed/faiss/chunk_id_map.json",
        embedding_model=embedder
    )

    # Semantic-only retrieval
    chunks = retriever.retrieve_semantic(query_embedding, top_k=15)

    # Graph-aware retrieval
    graph_chunks = retriever.retrieve_from_subgraph(entity_ids, neo4j_driver)

References:
    FAISS: Facebook AI Similarity Search for vector retrieval
    Neo4j: Graph database for chunk-entity provenance tracking
'''

# src/retrieval/entity_resolver.py
HEADERS['src/retrieval/entity_resolver.py'] = '''# -*- coding: utf-8 -*-
"""
Entity resolution for mapping query mentions to knowledge graph entities.

Resolves extracted query entities to canonical entity IDs in the knowledge graph using
FAISS similarity search over entity embeddings. Supports fuzzy matching with cosine
similarity thresholds and returns top-K candidates per mention with confidence scores.

The resolver embeds query entity mentions in "name (type)" format matching the entity
embedding format from disambiguation, then performs FAISS search against the entity
index. Results include entity IDs, similarity scores, and metadata for downstream
graph expansion.

Examples:
    # Initialize with entity FAISS index
    from src.retrieval.entity_resolver import EntityResolver
    resolver = EntityResolver(
        faiss_entity_index_path="data/processed/faiss/entity_embeddings.index",
        entity_ids_path="data/processed/faiss/entity_id_map.json",
        embedding_model=embedder
    )

    # Resolve query entities
    query_entities = ["EU AI Act", "GDPR"]
    resolved = resolver.resolve(query_entities, top_k=5, threshold=0.75)

    for mention, candidates in resolved.items():
        print(f"{mention} -> {candidates[0].entity_id} (score: {candidates[0].score:.3f})")

References:
    FAISS: HNSW index for fast approximate nearest neighbor search
    BGE-M3: Embedding model for entity representations (1024 dimensions)
'''

# src/retrieval/graph_expander.py
HEADERS['src/retrieval/graph_expander.py'] = '''# -*- coding: utf-8 -*-
"""
Graph expansion using Prize-Collecting Steiner Tree (PCST) algorithm.

Expands resolved query entities into a connected subgraph by finding a minimum-cost
tree that connects seed entities while collecting high-prize intermediate entities.
Uses Neo4j's PCST implementation with configurable prize-cost balance (delta parameter)
to control expansion aggressiveness.

The expander retrieves K-nearest semantic neighbors for each seed entity via FAISS,
assigns prizes and costs based on configuration strategy (uniform, frequency, similarity),
then executes PCST traversal in Neo4j to extract the optimal subgraph. Returns entity
IDs and relations for downstream chunk retrieval and answer generation.

Examples:
    # Initialize with Neo4j and FAISS
    from src.retrieval.graph_expander import GraphExpander
    expander = GraphExpander(
        neo4j_driver=driver,
        faiss_entity_index=faiss_index,
        entity_id_map=entity_map,
        embedding_model=embedder
    )

    # Expand query entities into subgraph
    seed_entity_ids = ["entity_001", "entity_042"]
    subgraph = expander.expand(seed_entity_ids, delta=0.5, max_entities=50)

    print(f"Expanded to {len(subgraph.entity_ids)} entities")
    print(f"Found {len(subgraph.relations)} relations")

References:
    PCST: Prize-Collecting Steiner Tree algorithm for graph expansion
    Neo4j GDS: Graph Data Science library PCST implementation
    FAISS: For semantic neighbor candidate retrieval
'''

# src/retrieval/query_parser.py
HEADERS['src/retrieval/query_parser.py'] = '''# -*- coding: utf-8 -*-
"""
Query understanding for extracting entities, jurisdictions, and document types.

Parses natural language queries to extract structured hints for retrieval: (1) entity
mentions for graph traversal, (2) jurisdiction codes for source filtering (EU, US, etc.),
and (3) document type preferences (regulation vs academic paper). Uses LLM-based entity
extraction with regex fallbacks for jurisdiction/type detection.

The parser calls Mistral-7B to extract entity mentions with types, then applies regex
patterns to detect jurisdiction mentions ("EU", "California") and document type signals
("regulations", "research papers"). Results guide entity resolution, PCST expansion,
and chunk ranking.

Examples:
    # Initialize parser
    from src.retrieval.query_parser import QueryParser
    parser = QueryParser(api_key=together_api_key)

    # Parse complex comparative query
    result = parser.parse("Compare EU and US AI regulations with academic research")

    print(f"Entities: {result.extracted_entities}")  # ["EU AI Act", "AI regulations"]
    print(f"Jurisdictions: {result.jurisdictions}")  # ["EU", "US"]
    print(f"Doc types: {result.doc_types}")  # ["regulation", "academic"]

    # Simple factual query
    result = parser.parse("What is GDPR?")
    print(f"Entities: {result.extracted_entities}")  # ["GDPR"]

References:
    Mistral-7B: mistralai/Mistral-7B-Instruct-v0.3 for entity extraction
    Jurisdiction patterns: config.retrieval_config.JURISDICTION_PATTERNS
    Doc type patterns: config.retrieval_config.DOC_TYPE_PATTERNS
'''

# src/retrieval/result_ranker.py
HEADERS['src/retrieval/result_ranker.py'] = '''# -*- coding: utf-8 -*-
"""
Result ranking with entity coverage scoring for GraphRAG chunks.

Ranks retrieved chunks using entity coverage analysis and provenance bonuses. Graph-
aware chunks receive coverage scores based on what fraction of resolved query entities
they mention, plus bonuses for appearing in PCST relations. Semantic chunks use raw
FAISS similarity. Final ranking merges both tracks and applies jurisdiction/doc-type
hint penalties.

The ranker implements multiplicative scoring where all components are bounded [0,1].
Entity coverage = (entities_in_chunk / total_resolved_entities), provenance bonus
adds a fixed multiplier for relation-mentioned chunks, and hint mismatches apply
penalty multipliers. Final top-K selection balances semantic relevance with entity
context.

Examples:
    # Initialize ranker
    from src.retrieval.result_ranker import ResultRanker
    ranker = ResultRanker(final_top_k=20)

    # Rank chunks with entity coverage
    ranked = ranker.rank(
        semantic_chunks=semantic_results,
        graph_chunks=graph_results,
        resolved_entities=query_entity_ids,
        jurisdiction_hints=["EU"],
        doc_type_hints=["regulation"]
    )

    # Inspect top chunk
    top = ranked[0]
    print(f"Score: {top.score:.3f}, Method: {top.source_path}")
    print(f"Entity coverage: {top.entity_coverage:.2f}")

References:
    Entity coverage scoring: Section 3.3.2c in PHASE_3_DESIGN.md
    Multiplicative system: All scores bounded [0,1] for fair comparison
'''

# src/retrieval/retrieval_processor.py
HEADERS['src/retrieval/retrieval_processor.py'] = '''# -*- coding: utf-8 -*-
"""
Main retrieval orchestration pipeline for GraphRAG system (Phase 3).

Coordinates the full question-answering workflow: query parsing, entity resolution,
graph expansion via PCST, dual-track chunk retrieval (semantic + graph), and result
ranking with entity coverage. Supports three retrieval modes for ablation studies:
SEMANTIC (baseline), GRAPH (entity-centric), and DUAL (combined).

The processor initializes all components (parser, resolver, expander, chunk retriever,
ranker), then executes the pipeline sequentially. Query -> entities -> resolve ->
expand PCST -> retrieve chunks (both tracks) -> rank -> return RetrievalResult with
chunks, subgraph, and metadata for answer generation.

Examples:
    # Initialize full pipeline
    from src.retrieval.retrieval_processor import RetrievalProcessor
    from config.retrieval_config import RetrievalMode

    processor = RetrievalProcessor(
        embedding_model=embedder,
        faiss_entity_index_path="data/processed/faiss/entity_embeddings.index",
        faiss_chunk_index_path="data/processed/faiss/chunk_embeddings.index",
        entity_ids_path="data/processed/faiss/entity_id_map.json",
        chunk_ids_path="data/processed/faiss/chunk_id_map.json",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password"
    )

    # Run dual retrieval
    result = processor.retrieve(
        query="What are high-risk AI systems under the EU AI Act?",
        mode=RetrievalMode.DUAL
    )

    print(f"Retrieved {len(result.chunks)} chunks")
    print(f"Subgraph: {len(result.subgraph.entity_ids)} entities, {len(result.subgraph.relations)} relations")

References:
    PHASE_3_DESIGN.md: Complete pipeline architecture and evaluation
    RetrievalMode enum: config.retrieval_config.RetrievalMode
'''

if __name__ == '__main__':
    for filepath, header in HEADERS.items():
        try:
            update_header(filepath, header)
        except Exception as e:
            print(f"✗ Failed {filepath}: {e}")
