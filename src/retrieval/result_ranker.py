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
                    coverage_bonus=debug_data['graph_multiplier'],  # Store multiplier as bonus
                    provenance_bonus=debug_data['standard_multiplier'],  # Store standard multiplier
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
                    entity_coverage=None,  # Semantic doesn't use entity coverage
                    coverage_bonus=debug_data['standard_multiplier'],  # Store standard multiplier
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
        Score chunk from graph retrieval using MULTIPLICATIVE system.
        
        Formula: Final = BS × GB × SB
        - BS (Base Score): entity_coverage = entities_in_chunk / total_entities
        - GB (Graph Bonus): 1.0 if provenance, 0.85 if entity expansion
        - SB (Standard Bonus): penalty multipliers for missing filters
        
        Returns:
            (final_score, retrieval_method, source_path, debug_data)
        """
        # BASE SCORE: Entity coverage (0-1)
        num_entities = len(chunk.metadata.get('entities', []))
        entity_coverage = num_entities / total_entities if total_entities > 0 else 0.0
        base_score = entity_coverage
        
        # GRAPH BONUS: Provenance vs entity expansion
        is_provenance = chunk.metadata.get('is_relation_provenance', False)
        if is_provenance:
            graph_multiplier = self.config['graph_provenance_multiplier']  # 1.0
            source_path = "graph_relation"
            retrieval_method = "graph_provenance"
        else:
            graph_multiplier = self.config['graph_entity_multiplier']  # 0.85
            source_path = "graph_entity"
            retrieval_method = "graph_entity"
        
        # STANDARD BONUS: Penalty multipliers for missing filters
        standard_multiplier = 1.0
        
        # Jurisdiction penalty (only if filter provided AND chunk doesn't match)
        if filters.jurisdiction_hints and chunk.jurisdiction not in filters.jurisdiction_hints:
            standard_multiplier *= self.config['jurisdiction_penalty']  # 0.9
        
        # Doc type penalty (only if filter provided AND chunk doesn't match)
        if filters.doc_type_hints and chunk.doc_type not in filters.doc_type_hints:
            standard_multiplier *= self.config['doc_type_penalty']  # 0.85
        
        # FINAL SCORE: BS × GB × SB (bounded to [0, 1])
        final_score = base_score * graph_multiplier * standard_multiplier
        
        debug_data = {
            'base_score': base_score,
            'entity_coverage': entity_coverage,
            'num_entities': num_entities,
            'total_entities': total_entities,
            'graph_multiplier': graph_multiplier,
            'standard_multiplier': standard_multiplier,
            'is_provenance': is_provenance,
        }
        
        return final_score, retrieval_method, source_path, debug_data
    
    def _score_semantic_chunk(
        self,
        chunk: Chunk,
        filters: QueryFilters
    ) -> tuple[float, str, str, Dict]:
        """
        Score chunk from semantic retrieval using MULTIPLICATIVE system.
        
        Formula: Final = BS × SB
        - BS (Base Score): FAISS cosine similarity (0-1)
        - SB (Standard Bonus): penalty multipliers for missing filters
        
        Returns:
            (final_score, retrieval_method, source_path, debug_data)
        """
        # BASE SCORE: FAISS similarity (already 0-1)
        base_score = chunk.score
        
        # STANDARD BONUS: Penalty multipliers for missing filters
        standard_multiplier = 1.0
        
        # Jurisdiction penalty (only if filter provided AND chunk doesn't match)
        if filters.jurisdiction_hints and chunk.jurisdiction not in filters.jurisdiction_hints:
            standard_multiplier *= self.config['jurisdiction_penalty']  # 0.9
        
        # Doc type penalty (only if filter provided AND chunk doesn't match)
        if filters.doc_type_hints and chunk.doc_type not in filters.doc_type_hints:
            standard_multiplier *= self.config['doc_type_penalty']  # 0.85
        
        # FINAL SCORE: BS × SB (bounded to [0, 1])
        final_score = base_score * standard_multiplier
        
        source_path = "semantic"
        retrieval_method = "semantic"
        
        debug_data = {
            'base_score': base_score,
            'standard_multiplier': standard_multiplier,
        }
        
        return final_score, retrieval_method, source_path, debug_data