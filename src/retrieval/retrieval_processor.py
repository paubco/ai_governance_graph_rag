# -*- coding: utf-8 -*-
"""
Module: retrieval_processor.py
Package: src.retrieval
Purpose: Orchestrate full Phase 3 retrieval pipeline (3.3.1 + 3.3.2)

MODIFIED: Added entity metadata passthrough to RetrievalResult for evaluation

Author: Pau Barba i Colomer
Created: 2025-12-07
Modified: 2025-12-14 (evaluation extension)

References:
    - PHASE_3_DESIGN.md § 4-5 (Query Understanding + Context Retrieval)
    - ARCHITECTURE.md § 3.3 (Phase 3 pipeline overview)
    - PHASE_3.3.2_IMPLEMENTATION.md (PCST-based expansion)

Pipeline:
    1. Query Understanding (3.3.1)
        - Parse query (LLM entity extraction + filters)
        - Resolve entities (FAISS matching)
    
    2. Context Retrieval (3.3.2)
        - Graph expansion (PCST optimization)
        - Dual-path retrieval (GraphRAG + Naive RAG)
        - Ranking (provenance bonus)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .config import ParsedQuery, ResolvedEntity, RetrievalResult
from .query_parser import QueryParser
from .entity_resolver import EntityResolver
from .graph_expander import GraphExpander
from .chunk_retriever import ChunkRetriever
from .result_ranker import ResultRanker


@dataclass
class QueryUnderstanding:
    """
    Complete query understanding result (Phase 3.3.1 output).
    
    Combines parsed query structure with resolved entities.
    """
    parsed_query: ParsedQuery
    resolved_entities: List[ResolvedEntity]


class RetrievalProcessor:
    """
    Orchestrate full Phase 3 retrieval pipeline.
    
    Coordinates:
    - Phase 3.3.1: Query understanding (parse + resolve entities)
    - Phase 3.3.2: Context retrieval (PCST expansion + dual-path + ranking)
    """
    
    def __init__(
        self,
        embedding_model,
        # Phase 3.3.1 paths
        faiss_entity_index_path: Path,
        entity_ids_path: Path,
        normalized_entities_path: Path,
        # Phase 3.3.2 paths
        faiss_chunk_index_path: Path,
        chunk_ids_path: Path,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        # Config
        fuzzy_threshold: float = 0.75,
        entity_top_k: int = 10,
    ):
        """
        Initialize retrieval processor.
        
        Args:
            embedding_model: BGE-M3 embedding model with embed_single() method
            faiss_entity_index_path: Path to FAISS entity index
            entity_ids_path: Path to entity ID mapping JSON
            normalized_entities_path: Path to normalized entities JSON
            faiss_chunk_index_path: Path to FAISS chunk index
            chunk_ids_path: Path to chunk ID mapping JSON
            neo4j_uri: Neo4j connection string
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            fuzzy_threshold: Entity fuzzy matching threshold
            entity_top_k: Candidates per entity for fuzzy matching
        """
        # Phase 3.3.1: Query Understanding
        self.query_parser = QueryParser(embedding_model)
        
        self.entity_resolver = EntityResolver(
            faiss_index_path=faiss_entity_index_path,
            entity_ids_path=entity_ids_path,
            normalized_entities_path=normalized_entities_path,
            embedding_model=embedding_model,
            threshold=fuzzy_threshold,
            top_k=entity_top_k,
        )
        
        # Phase 3.3.2: Context Retrieval
        self.graph_expander = GraphExpander(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            entity_index_path=str(faiss_entity_index_path),
            entity_id_map_path=str(entity_ids_path),
        )
        
        self.chunk_retriever = ChunkRetriever(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            chunk_index_path=str(faiss_chunk_index_path),
            chunk_id_map_path=str(chunk_ids_path),
        )
        
        self.result_ranker = ResultRanker()
    
    def understand_query(self, query: str) -> QueryUnderstanding:
        """
        Phase 3.3.1: Query Understanding only.
        
        Steps:
        1. Parse query → structured form (entities, filters, embedding)
        2. Resolve entities → canonical entity IDs
        
        Args:
            query: Natural language query string
            
        Returns:
            QueryUnderstanding with parsed query and resolved entities
        """
        # Step 1: Parse query
        parsed_query = self.query_parser.parse(query)
        
        # Step 2: Resolve entity mentions
        resolved_entities = self.entity_resolver.resolve(
            parsed_query.extracted_entities
        )
        
        return QueryUnderstanding(
            parsed_query=parsed_query,
            resolved_entities=resolved_entities,
        )
    
    def retrieve(self, query: str, mode: str = "dual") -> RetrievalResult:
        """
        Full Phase 3 pipeline: Understanding + Context Retrieval.
        
        Steps:
        1. Query Understanding (3.3.1)
            - Parse query (LLM entity extraction)
            - Resolve entities (FAISS matching)
        
        2. Graph Expansion (3.3.2a)
            - PCST optimization to find minimal connecting subgraph
        
        3. Dual-Path Retrieval (3.3.2b)
            - Path A: Corpus retrospective (entity → chunks)
            - Path B: Semantic FAISS search
        
        4. Ranking (3.3.2c)
            - Merge + deduplicate
            - Score with provenance bonus
            - Top-K selection
        
        Args:
            query: Natural language query string
            mode: Retrieval mode (for testing) - "naive", "graphrag", or "dual"
            
        Returns:
            RetrievalResult with ranked chunks, subgraph, and entity metadata
        """
        # Phase 3.3.1: Query Understanding
        understanding = self.understand_query(query)
        
        if not understanding.resolved_entities:
            # No entities found - fall back to Path B only
            print("⚠️  No entities resolved, using semantic search only")
            path_a_chunks = []
            path_b_chunks = self.chunk_retriever._retrieve_path_b(
                understanding.parsed_query.query_embedding
            )
            
            # Create empty subgraph
            from .config import Subgraph
            subgraph = Subgraph(entities=[], relations=[])
            
        else:
            # Phase 3.3.2a: Graph Expansion (PCST)
            subgraph = self.graph_expander.expand(understanding.resolved_entities)
            
            # Phase 3.3.2b: Dual-Path Retrieval
            path_a_chunks, path_b_chunks = self.chunk_retriever.retrieve_dual(
                subgraph=subgraph,
                query_embedding=understanding.parsed_query.query_embedding
            )
        
        # Phase 3.3.2c: Ranking
        result = self.result_ranker.rank(
            path_a_chunks=path_a_chunks,
            path_b_chunks=path_b_chunks,
            subgraph=subgraph,
            filters=understanding.parsed_query.filters,
            query=query
        )
        
        # MODIFIED: Extend result with entity metadata for evaluation/testing
        # This enables ablation studies to track entity resolution quality
        result.extracted_entities = understanding.parsed_query.extracted_entities
        result.resolved_entities = understanding.resolved_entities
        
        return result
    
    def batch_retrieve(self, queries: List[str]) -> List[RetrievalResult]:
        """
        Process multiple queries through full pipeline.
        
        Args:
            queries: List of query strings
            
        Returns:
            List of RetrievalResult objects
        """
        return [self.retrieve(q) for q in queries]
    
    def close(self):
        """Close all connections."""
        self.graph_expander.close()
        self.chunk_retriever.close()