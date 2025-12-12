# -*- coding: utf-8 -*-
"""
Module: retrieval_processor.py
Package: src.retrieval
Purpose: Orchestrate full Phase 3 retrieval pipeline (3.3.1 + 3.3.2)

Author: Pau Barba i Colomer
Created: 2025-12-07
Modified: 2025-12-12

References:
    - PHASE_3_DESIGN.md § 4-5 (Query Understanding + Context Retrieval)
    - ARCHITECTURE.md § 3.3 (Phase 3 pipeline overview)
    - PHASE_3.3.2_IMPLEMENTATION.md (PCST-based expansion)
    - PHASE_3_DESIGN.md § 6 (Evaluation - ablation studies)

Pipeline:
    1. Query Understanding (3.3.1)
        - Parse query (LLM entity extraction + filters)
        - Resolve entities (FAISS matching)
    
    2. Context Retrieval (3.3.2)
        - Graph expansion (PCST optimization)
        - Dual-path retrieval (GraphRAG + Naive RAG)
        - Ranking (provenance bonus)

Modes (for ablation studies):
    - NAIVE: Path B only (semantic FAISS search)
    - GRAPHRAG: Path A only (entity-centric + PCST)
    - DUAL: Both paths merged (default)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .config import ParsedQuery, ResolvedEntity, RetrievalResult, RetrievalMode
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
            embedding_model: BGE-M3 embedding model with embed_single() method.
            faiss_entity_index_path: Path to FAISS entity index.
            entity_ids_path: Path to entity ID mapping JSON.
            normalized_entities_path: Path to normalized entities JSON.
            faiss_chunk_index_path: Path to FAISS chunk index.
            chunk_ids_path: Path to chunk ID mapping JSON.
            neo4j_uri: Neo4j connection string.
            neo4j_user: Neo4j username.
            neo4j_password: Neo4j password.
            fuzzy_threshold: Entity fuzzy matching threshold.
            entity_top_k: Candidates per entity for fuzzy matching.
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
            query: Natural language query string.
            
        Returns:
            QueryUnderstanding with parsed query and resolved entities.
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
    
    def retrieve(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.DUAL
    ) -> RetrievalResult:
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
            query: Natural language query string.
            mode: Retrieval mode (NAIVE, GRAPHRAG, or DUAL).
            
        Returns:
            RetrievalResult with ranked chunks and subgraph.
        """
        # Import here to avoid circular dependency
        from .config import GraphSubgraph
        
        # Mode-specific execution
        if mode == RetrievalMode.NAIVE:
            # Path B only: semantic search, no entity resolution
            print("🔍 Mode: NAIVE (semantic search only)")
            
            # Parse query for embedding only (skip entity extraction)
            parsed_query = self.query_parser.parse(query)
            
            # Path B: semantic search
            path_b_chunks = self.chunk_retriever._retrieve_path_b(
                parsed_query.query_embedding
            )
            
            # No Path A chunks
            path_a_chunks = []
            
            # Empty subgraph
            subgraph = GraphSubgraph(entities=[], relations=[])
            
            # Empty resolved entities for result
            resolved_entity_names = []
        
        elif mode == RetrievalMode.GRAPHRAG:
            # Path A only: entity resolution + PCST, no semantic search
            print("🔍 Mode: GRAPHRAG (entity-centric only)")
            
            # Phase 3.3.1: Query Understanding
            understanding = self.understand_query(query)
            
            if not understanding.resolved_entities:
                # No entities found - cannot do GraphRAG
                print("⚠️  No entities resolved, returning empty result")
                subgraph = GraphSubgraph(entities=[], relations=[])
                path_a_chunks = []
                path_b_chunks = []
                resolved_entity_names = []
            else:
                # Phase 3.3.2a: Graph Expansion (PCST)
                subgraph = self.graph_expander.expand(understanding.resolved_entities)
                
                # Phase 3.3.2b: Path A only (corpus retrospective)
                path_a_chunks = self.chunk_retriever._retrieve_path_a(subgraph)
                
                # No Path B chunks
                path_b_chunks = []
                
                # Extract entity names for result
                resolved_entity_names = [e.name for e in understanding.resolved_entities]
        
        else:  # RetrievalMode.DUAL (default)
            # Both paths: full pipeline
            print("🔍 Mode: DUAL (GraphRAG + semantic)")
            
            # Phase 3.3.1: Query Understanding
            understanding = self.understand_query(query)
            
            if not understanding.resolved_entities:
                # No entities found - fall back to Path B only
                print("⚠️  No entities resolved, using semantic search only")
                path_a_chunks = []
                path_b_chunks = self.chunk_retriever._retrieve_path_b(
                    understanding.parsed_query.query_embedding
                )
                subgraph = GraphSubgraph(entities=[], relations=[])
                resolved_entity_names = []
            else:
                # Phase 3.3.2a: Graph Expansion (PCST)
                subgraph = self.graph_expander.expand(understanding.resolved_entities)
                
                # Phase 3.3.2b: Dual-Path Retrieval
                path_a_chunks, path_b_chunks = self.chunk_retriever.retrieve_dual(
                    subgraph=subgraph,
                    query_embedding=understanding.parsed_query.query_embedding
                )
                
                # Extract entity names for result
                resolved_entity_names = [e.name for e in understanding.resolved_entities]
        
        # Phase 3.3.2c: Ranking (common for all modes)
        # For NAIVE mode, use parsed_query.filters if available
        if mode == RetrievalMode.NAIVE:
            filters = parsed_query.filters
        else:
            filters = understanding.parsed_query.filters
        
        result = self.result_ranker.rank(
            path_a_chunks=path_a_chunks,
            path_b_chunks=path_b_chunks,
            subgraph=subgraph,
            filters=filters,
            query=query
        )
        
        return result
    
    def batch_retrieve(
        self,
        queries: List[str],
        mode: RetrievalMode = RetrievalMode.DUAL
    ) -> List[RetrievalResult]:
        """
        Process multiple queries through full pipeline.
        
        Args:
            queries: List of query strings.
            mode: Retrieval mode to use for all queries.
            
        Returns:
            List of RetrievalResult objects.
        """
        return [self.retrieve(q, mode=mode) for q in queries]
    
    def close(self):
        """Close all connections."""
        self.graph_expander.close()
        self.chunk_retriever.close()