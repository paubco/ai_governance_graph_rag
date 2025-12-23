# -*- coding: utf-8 -*-
"""
Retrieval processor for AI governance GraphRAG pipeline.

Orchestrates full retrieval pipeline combining query understanding and context
retrieval.
"""

# Standard library
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Config imports (direct)
from config.retrieval_config import RetrievalMode

# Dataclass imports (direct)
from src.utils.dataclasses import (
    ParsedQuery,
    ResolvedEntity,
    RetrievalResult,
    Subgraph,
)

# Local module imports
from src.retrieval.query_parser import QueryParser
from src.retrieval.entity_resolver import EntityResolver
from src.retrieval.graph_expander import GraphExpander
from src.retrieval.chunk_retriever import ChunkRetriever
from src.retrieval.result_ranker import ResultRanker


@dataclass
class QueryUnderstanding:
    """Complete query understanding result (Phase 3.3.1 output)."""
    parsed_query: ParsedQuery
    resolved_entities: List[ResolvedEntity]


class RetrievalProcessor:
    """
    Orchestrate full Phase 3 retrieval pipeline.
    
    Coordinates:
    - Phase 3.3.1: Query understanding (parse + resolve entities)
    - Phase 3.3.2: Context retrieval (PCST expansion + dual-channel + ranking)
    """
    
    def __init__(
        self,
        embedding_model,
        # Phase 3.3.1 paths
        faiss_entity_index_path: Path,
        entity_ids_path: Path,
        normalized_entities_path: Path,
        aliases_path: Path = None,
        # Phase 3.3.2 paths
        faiss_chunk_index_path: Path = None,
        chunk_ids_path: Path = None,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
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
            aliases_path: Path to aliases.json for alias resolution.
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
            aliases_path=aliases_path,
            threshold=fuzzy_threshold,
            top_k=entity_top_k,
        )
        
        # Phase 3.3.2: Context Retrieval (optional - for full pipeline)
        self.graph_expander = None
        self.chunk_retriever = None
        self.result_ranker = None
        
        if neo4j_uri and faiss_chunk_index_path:
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
        
        Args:
            query: Natural language query string.
            
        Returns:
            QueryUnderstanding with parsed query and resolved entities.
        """
        parsed_query = self.query_parser.parse(query)
        resolved_entities = self.entity_resolver.resolve(parsed_query.extracted_entities)
        
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
        
        Args:
            query: Natural language query string.
            mode: Retrieval mode (SEMANTIC, GRAPH, or DUAL).
            
        Returns:
            RetrievalResult with ranked chunks, subgraph, and entity metadata.
        """
        if not self.graph_expander or not self.chunk_retriever:
            raise RuntimeError(
                "Full pipeline not initialized. "
                "Provide neo4j_uri and faiss_chunk_index_path."
            )
        
        # Phase 3.3.1: Query Understanding
        understanding = self.understand_query(query)
        
        if not understanding.resolved_entities:
            print("Warning: No entities resolved")
            subgraph = Subgraph(entities=[], relations=[])
            graph_chunks = []
            
            if mode in [RetrievalMode.SEMANTIC, RetrievalMode.DUAL]:
                semantic_chunks = self.chunk_retriever._retrieve_semantic(
                    understanding.parsed_query.embedding
                )
            else:
                semantic_chunks = []
        else:
            # Phase 3.3.2a: Graph Expansion
            subgraph = self.graph_expander.expand(understanding.resolved_entities)
            
            # Phase 3.3.2b: Mode-Aware Retrieval
            if mode == RetrievalMode.SEMANTIC:
                graph_chunks = []
                semantic_chunks = self.chunk_retriever._retrieve_semantic(
                    understanding.parsed_query.embedding
                )
            elif mode == RetrievalMode.GRAPH:
                graph_chunks = self.chunk_retriever._retrieve_graph(subgraph)
                semantic_chunks = []
            else:  # DUAL
                graph_chunks, semantic_chunks = self.chunk_retriever.retrieve_dual(
                    subgraph=subgraph,
                    query_embedding=understanding.parsed_query.embedding
                )
        
        # Phase 3.3.2c: Ranking
        result = self.result_ranker.rank(
            graph_chunks=graph_chunks,
            semantic_chunks=semantic_chunks,
            subgraph=subgraph,
            filters=understanding.parsed_query.filters,
            query=query
        )
        
        # Attach metadata for evaluation
        result.parsed_query = understanding.parsed_query
        
        return result
    
    def batch_retrieve(
        self, 
        queries: List[str],
        mode: RetrievalMode = RetrievalMode.DUAL
    ) -> List[RetrievalResult]:
        """Process multiple queries through full pipeline."""
        return [self.retrieve(q, mode) for q in queries]
    
    def close(self):
        """Close all connections."""
        if self.graph_expander:
            self.graph_expander.close()
        if self.chunk_retriever:
            self.chunk_retriever.close()