# -*- coding: utf-8 -*-
"""
Module: result_ranker.py
Package: src.retrieval
Purpose: Merge, deduplicate, and rank chunks from dual channels

CRITICAL BUG FIX: Now properly sets source_path when creating RankedChunk objects
MODIFIED: Updated to graph/semantic nomenclature (removed Path A/B naming)

Author: Pau Barba i Colomer
Created: 2025-12-08
Modified: 2025-12-15 (bug fix + nomenclature update)

References:
    - PHASE_3_DESIGN.md § 5.3 (Ranking strategy)
    - RAGulating (Agarwal et al., 2025) - Provenance bonus concept
"""

from typing import List, Set, Dict, Optional
from dataclasses import dataclass, field

from .config import (
    Chunk,
    RankedChunk,
    Subgraph,
    QueryFilters,
    RetrievalResult,
    RANKING_CONFIG,
)


# ============================================================================
# SCORING DEBUG INFO (for ablation studies)
# ============================================================================

@dataclass
class ScoringDebugInfo:
    """Debug information for scoring transparency."""
    chunk_id: str
    method: str  # 'graph' or 'semantic'
    base_score: float
    entity_coverage: Optional[float]
    coverage_bonus: float
    provenance_bonus: float
    final_score: float
    rank: int


# ============================================================================
# RESULT RANKER
# ============================================================================

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
            subgraph: PCST subgraph (for relation provenance and entity count).
            filters: Query filters (reserved for future use).
            query: Original query string.
            debug: If True, collect detailed scoring information.
        
        Returns:
            RetrievalResult with top-K ranked chunks.
        """
        self.debug_info = [] if debug else None
        
        # Get relation provenance chunk IDs
        relation_chunk_ids = self._get_relation_chunk_ids(subgraph)
        
        # Total resolved entities (for coverage calculation)
        total_entities = len(subgraph.entities)
        
        # Score Graph chunks
        scored_chunks = {}
        for chunk in graph_chunks:
            score, retrieval_method, source_path, debug_data = self._score_graph_chunk(
                chunk, 
                relation_chunk_ids,
                total_entities,
                filters
            )
            
            if debug:
                self.debug_info.append(ScoringDebugInfo(
                    chunk_id=chunk.chunk_id,
                    method='graph',
                    base_score=debug_data['base_score'],
                    entity_coverage=debug_data['entity_coverage'],
                    coverage_bonus=debug_data['coverage_bonus'],
                    provenance_bonus=debug_data['provenance_bonus'],
                    final_score=score,
                    rank=0  # Will be updated after sorting
                ))
            
            # If chunk already seen, keep higher score
            if chunk.chunk_id in scored_chunks:
                if score > scored_chunks[chunk.chunk_id]['score']:
                    scored_chunks[chunk.chunk_id] = {
                        'chunk': chunk,
                        'score': score,
                        'retrieval_method': retrieval_method,
                        'source_path': source_path  # BUG FIX: Store source_path
                    }
            else:
                scored_chunks[chunk.chunk_id] = {
                    'chunk': chunk,
                    'score': score,
                    'retrieval_method': retrieval_method,
                    'source_path': source_path  # BUG FIX: Store source_path
                }
        
        # Score Semantic chunks
        for chunk in semantic_chunks:
            score, retrieval_method, source_path, debug_data = self._score_semantic_chunk(chunk, filters)
            
            if debug:
                self.debug_info.append(ScoringDebugInfo(
                    chunk_id=chunk.chunk_id,
                    method='semantic',
                    base_score=debug_data['base_score'],
                    entity_coverage=None,  # Semantic doesn't use coverage
                    coverage_bonus=0.0,
                    provenance_bonus=0.0,
                    final_score=score,
                    rank=0
                ))
            
            # If chunk already seen from Graph, only update if Semantic score is higher
            if chunk.chunk_id in scored_chunks:
                if score > scored_chunks[chunk.chunk_id]['score']:
                    scored_chunks[chunk.chunk_id]['score'] = score
                    scored_chunks[chunk.chunk_id]['retrieval_method'] = retrieval_method
                    scored_chunks[chunk.chunk_id]['source_path'] = source_path  # BUG FIX
            else:
                scored_chunks[chunk.chunk_id] = {
                    'chunk': chunk,
                    'score': score,
                    'retrieval_method': retrieval_method,
                    'source_path': source_path  # BUG FIX: Store source_path
                }
        
        # Convert to RankedChunk objects
        ranked_chunks = []
        for chunk_data in scored_chunks.values():
            chunk = chunk_data['chunk']
            ranked_chunk = RankedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=chunk_data['score'],
                source_path=chunk_data['source_path'],  # BUG FIX: Actually set source_path!
                retrieval_method=chunk_data['retrieval_method'],
                doc_id=chunk.doc_id,
                doc_type=chunk.doc_type,
                jurisdiction=chunk.jurisdiction,
                entities=chunk.metadata.get('entities', []),
                metadata=chunk.metadata
            )
            ranked_chunks.append(ranked_chunk)
        
        # Sort by score and take top-K
        ranked_chunks.sort(key=lambda c: c.score, reverse=True)
        top_k = ranked_chunks[:self.config['final_top_k']]
        
        # Update ranks in debug info
        if debug:
            chunk_ranks = {c.chunk_id: i+1 for i, c in enumerate(ranked_chunks)}
            for info in self.debug_info:
                info.rank = chunk_ranks.get(info.chunk_id, 999)
        
        # Extract resolved entity names for result
        resolved_entity_names = list(subgraph.entities) if subgraph.entities else []
        
        return RetrievalResult(
            query=query,
            resolved_entities=resolved_entity_names,
            subgraph=subgraph,
            chunks=top_k
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
    ) -> tuple[float, str, str, Dict]:
        """
        Score chunk from graph retrieval.
        
        Returns:
            (final_score, retrieval_method, source_path, debug_data)
        """
        # Base score from chunk (entity match score from retriever)
        base_score = chunk.score
        
        # Entity coverage bonus
        num_entities = len(chunk.metadata.get('entities', []))
        entity_coverage = num_entities / total_entities if total_entities > 0 else 0.0
        coverage_bonus = entity_coverage * self.config['entity_coverage_bonus']
        
        # Provenance bonus (highest priority)
        is_provenance = chunk.metadata.get('is_relation_provenance', False)
        provenance_bonus = self.config['provenance_bonus'] if is_provenance else 0.0
        
        # Graph retrieval bonus (for non-provenance chunks)
        graph_bonus = 0.0 if is_provenance else self.config['graph_bonus']
        
        # Jurisdiction boost (soft hint)
        jurisdiction_boost = 0.0
        if filters.jurisdiction_hints and chunk.jurisdiction in filters.jurisdiction_hints:
            jurisdiction_boost = self.config['jurisdiction_boost']
        
        # Doc type boost
        doc_type_boost = 0.0
        if filters.doc_type_hints and chunk.doc_type in filters.doc_type_hints:
            doc_type_boost = self.config['doc_type_boost']
        
        # Final score
        final_score = (
            base_score +
            coverage_bonus +
            provenance_bonus +
            graph_bonus +
            jurisdiction_boost +
            doc_type_boost
        )
        
        # Determine source path and retrieval method
        if is_provenance:
            source_path = "graph_relation"
            retrieval_method = "graph_provenance"
        else:
            source_path = "graph_entity"
            retrieval_method = "graph_entity"
        
        debug_data = {
            'base_score': base_score,
            'entity_coverage': entity_coverage,
            'coverage_bonus': coverage_bonus,
            'provenance_bonus': provenance_bonus,
            'graph_bonus': graph_bonus,
            'jurisdiction_boost': jurisdiction_boost,
            'doc_type_boost': doc_type_boost,
        }
        
        return final_score, retrieval_method, source_path, debug_data
    
    def _score_semantic_chunk(
        self,
        chunk: Chunk,
        filters: QueryFilters
    ) -> tuple[float, str, str, Dict]:
        """
        Score chunk from semantic retrieval.
        
        Returns:
            (final_score, retrieval_method, source_path, debug_data)
        """
        # Base score is FAISS similarity
        base_score = chunk.score
        
        # Semantic baseline (no bonus for being from semantic path)
        semantic_baseline = self.config['semantic_baseline']
        
        # Jurisdiction boost
        jurisdiction_boost = 0.0
        if filters.jurisdiction_hints and chunk.jurisdiction in filters.jurisdiction_hints:
            jurisdiction_boost = self.config['jurisdiction_boost']
        
        # Doc type boost
        doc_type_boost = 0.0
        if filters.doc_type_hints and chunk.doc_type in filters.doc_type_hints:
            doc_type_boost = self.config['doc_type_boost']
        
        final_score = (
            base_score +
            semantic_baseline +
            jurisdiction_boost +
            doc_type_boost
        )
        
        source_path = "semantic"
        retrieval_method = "semantic"
        
        debug_data = {
            'base_score': base_score,
            'semantic_baseline': semantic_baseline,
            'jurisdiction_boost': jurisdiction_boost,
            'doc_type_boost': doc_type_boost,
        }
        
        return final_score, retrieval_method, source_path, debug_data