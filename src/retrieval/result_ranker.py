# -*- coding: utf-8 -*-
"""
Result ranker for AI governance GraphRAG pipeline.

Merges, deduplicates, and ranks chunks from dual retrieval channels.
Implements multiplicative scoring with entity coverage and provenance tracking.
"""

# Standard library
from typing import List, Set, Dict, Optional
from dataclasses import dataclass

# Config imports (direct)
from config.retrieval_config import RANKING_CONFIG

# Dataclass imports (direct)
from src.utils.dataclasses import (
    Chunk,
    RankedChunk,
    Subgraph,
    QueryFilters,
    RetrievalResult,
)


@dataclass
class ScoringDebugInfo:
    """Debug information for scoring transparency."""
    chunk_id: str
    method: str
    base_score: float
    entity_coverage: Optional[float]
    coverage_bonus: float
    provenance_bonus: float
    final_score: float
    rank: int


class ResultRanker:
    """
    Merge and rank chunks from dual retrieval channels.
    
    Scoring strategy:
    1. Graph chunks: Base score + entity coverage bonus + provenance bonus
    2. Semantic chunks: FAISS similarity (baseline)
    3. Deduplication: Keep highest score per chunk
    """
    
    def __init__(self):
        """Initialize result ranker with config."""
        self.config = RANKING_CONFIG
        self.debug_info: Optional[List[ScoringDebugInfo]] = None
    
    def rank(
        self,
        graph_chunks: List[Chunk],
        semantic_chunks: List[Chunk],
        subgraph: Subgraph,
        filters: QueryFilters,
        query: str,
        debug: bool = False
    ) -> RetrievalResult:
        """
        Merge, deduplicate, and rank chunks with entity coverage.
        
        Args:
            graph_chunks: Chunks from graph retrieval channel.
            semantic_chunks: Chunks from semantic search.
            subgraph: PCST subgraph.
            filters: Query filters.
            query: Original query string.
            debug: If True, collect detailed scoring information.
        
        Returns:
            RetrievalResult with top-K ranked chunks.
        """
        self.debug_info = [] if debug else None
        
        relation_chunk_ids = self._get_relation_chunk_ids(subgraph)
        total_entities = len(subgraph.entities) if subgraph.entities else 0
        
        scored_chunks = {}
        
        # Score Graph chunks
        for chunk in graph_chunks:
            score, source_path, debug_data = self._score_graph_chunk(
                chunk, relation_chunk_ids, total_entities, filters
            )
            
            chunk_id = chunk.chunk_id
            if chunk_id not in scored_chunks or score > scored_chunks[chunk_id]['score']:
                scored_chunks[chunk_id] = {
                    'chunk': chunk,
                    'score': score,
                    'source_path': source_path,
                }
        
        # Score Semantic chunks
        for chunk in semantic_chunks:
            score, source_path, debug_data = self._score_semantic_chunk(chunk, filters)
            
            chunk_id = chunk.chunk_id
            if chunk_id not in scored_chunks or score > scored_chunks[chunk_id]['score']:
                scored_chunks[chunk_id] = {
                    'chunk': chunk,
                    'score': score,
                    'source_path': source_path,
                }
        
        # Convert to RankedChunk objects
        ranked_chunks = []
        for chunk_data in scored_chunks.values():
            chunk = chunk_data['chunk']
            ranked_chunk = RankedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=chunk_data['score'],
                source_path=chunk_data['source_path'],
                doc_id=chunk.document_id,
                doc_type=chunk.metadata.get('doc_type', 'unknown'),
                jurisdiction=chunk.metadata.get('jurisdiction'),
                matching_entities=chunk.metadata.get('entities', []),
            )
            ranked_chunks.append(ranked_chunk)
        
        # Sort by score and take top-K
        ranked_chunks.sort(key=lambda c: c.score, reverse=True)
        top_k = ranked_chunks[:self.config['final_top_k']]
        
        return RetrievalResult(
            query=query,
            chunks=top_k,
            subgraph=subgraph,
        )
    
    def _get_relation_chunk_ids(self, subgraph: Subgraph) -> Set[str]:
        """Extract chunk IDs from relation provenance."""
        chunk_ids = set()
        for relation in subgraph.relations:
            chunk_ids.update(relation.chunk_ids)
        return chunk_ids
    
    def _score_graph_chunk(
        self,
        chunk: Chunk,
        relation_chunk_ids: Set[str],
        total_entities: int,
        filters: QueryFilters
    ) -> tuple[float, str, Dict]:
        """
        Score chunk from graph retrieval using MULTIPLICATIVE system.
        
        Formula: Final = BS × GB × SB
        """
        # BASE SCORE: Entity coverage
        num_entities = len(chunk.metadata.get('entities', []))
        entity_coverage = num_entities / total_entities if total_entities > 0 else 0.0
        base_score = max(entity_coverage, chunk.metadata.get('score', 0.5))
        
        # GRAPH BONUS
        is_provenance = chunk.metadata.get('is_relation_provenance', False)
        if is_provenance:
            graph_multiplier = self.config['graph_provenance_multiplier']
            source_path = "graph_provenance"
        else:
            graph_multiplier = self.config['graph_entity_multiplier']
            source_path = "graph_entity"
        
        # STANDARD BONUS (penalties)
        standard_multiplier = 1.0
        jurisdiction = chunk.metadata.get('jurisdiction')
        doc_type = chunk.metadata.get('doc_type')
        
        if filters.jurisdiction_hints and jurisdiction not in filters.jurisdiction_hints:
            standard_multiplier *= self.config['jurisdiction_penalty']
        
        if filters.doc_type_hints and doc_type not in filters.doc_type_hints:
            standard_multiplier *= self.config['doc_type_penalty']
        
        final_score = base_score * graph_multiplier * standard_multiplier
        
        debug_data = {
            'base_score': base_score,
            'entity_coverage': entity_coverage,
            'graph_multiplier': graph_multiplier,
            'standard_multiplier': standard_multiplier,
        }
        
        return final_score, source_path, debug_data
    
    def _score_semantic_chunk(
        self,
        chunk: Chunk,
        filters: QueryFilters
    ) -> tuple[float, str, Dict]:
        """
        Score chunk from semantic retrieval using MULTIPLICATIVE system.
        
        Formula: Final = BS × SB
        """
        base_score = chunk.metadata.get('score', 0.5)
        
        standard_multiplier = 1.0
        jurisdiction = chunk.metadata.get('jurisdiction')
        doc_type = chunk.metadata.get('doc_type')
        
        if filters.jurisdiction_hints and jurisdiction not in filters.jurisdiction_hints:
            standard_multiplier *= self.config['jurisdiction_penalty']
        
        if filters.doc_type_hints and doc_type not in filters.doc_type_hints:
            standard_multiplier *= self.config['doc_type_penalty']
        
        final_score = base_score * standard_multiplier
        source_path = "semantic"
        
        debug_data = {
            'base_score': base_score,
            'standard_multiplier': standard_multiplier,
        }
        
        return final_score, source_path, debug_data