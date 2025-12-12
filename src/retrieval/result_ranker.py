# -*- coding: utf-8 -*-
"""
Module: result_ranker.py
Package: src.retrieval
Purpose: Merge and rank chunks from dual-path retrieval

Author: Pau Barba i Colomer
Created: 2025-12-07
Modified: 2025-12-12

References:
    - RAGulating (Agarwal et al., 2025) - Provenance tracking
    - RAKG (Zhang et al., 2025) - Entity Coverage metric
    - PHASE_3_DESIGN.md § 5.3 (Ranking strategy)

Ranking Strategy (Entity Coverage-Based):
    
    GraphRAG scoring:
        score = base_score + (entity_coverage * max_coverage_bonus) + provenance_bonus
        
        where entity_coverage = entities_in_chunk / total_resolved_entities
    
    Naive scoring:
        score = faiss_similarity  # NO bonuses (pure semantic)
    
    Examples (4 entities resolved):
    
    Scenario 1: Full context GraphRAG with provenance
      Base: 0.40
      Coverage: 4/4 = 1.0 → +0.40
      Provenance: +0.15
      = 0.95 (strong but not unbeatable)
    
    Scenario 2: Perfect semantic match (naive)
      FAISS: 0.98
      = 0.98 (beats partial GraphRAG)
    
    Scenario 3: Partial context GraphRAG
      Base: 0.40
      Coverage: 1/4 = 0.25 → +0.10
      = 0.50 (loses to good naive)
    
    Scenario 4: Mediocre naive
      FAISS: 0.45
      = 0.45 (loses to everything)
"""

import numpy as np
from typing import List, Set, Dict, Optional
from collections import defaultdict
from dataclasses import dataclass

from .config import (
    Chunk,
    RankedChunk,
    GraphSubgraph,
    QueryFilters,
    RetrievalResult,
    RANKING_CONFIG,
)


# ============================================================================
# DEBUG INFO
# ============================================================================

@dataclass
class ScoringDebugInfo:
    """Debug information for scoring decisions."""
    chunk_id: str
    method: str  # 'graphrag' or 'naive'
    base_score: float
    entity_coverage: Optional[float]  # Only for graphrag
    coverage_bonus: float
    provenance_bonus: float
    final_score: float
    rank: int


# ============================================================================
# RESULT RANKER
# ============================================================================

class ResultRanker:
    """
    Merge and rank chunks using entity coverage.
    
    Key principle: GraphRAG chunks rewarded for discussing MORE resolved entities.
    This penalizes chunks mentioning random entities without full context.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize ranker.
        
        Args:
            config: Ranking configuration (uses RANKING_CONFIG if None).
        """
        self.config = config or RANKING_CONFIG
        self.debug_info: List[ScoringDebugInfo] = []
    
    def rank(
        self,
        graphrag_chunks: List[Chunk],
        naive_chunks: List[Chunk],
        subgraph: GraphSubgraph,
        filters: QueryFilters,
        query: str,
        debug: bool = False
    ) -> RetrievalResult:
        """
        Merge, deduplicate, and rank chunks with entity coverage.
        
        Args:
            graphrag_chunks: Chunks from entity-centric GraphRAG path.
            naive_chunks: Chunks from semantic search.
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
        
        # Score GraphRAG chunks
        scored_chunks = {}
        for chunk in graphrag_chunks:
            score, retrieval_method, debug_data = self._score_graphrag_chunk(
                chunk, 
                relation_chunk_ids,
                total_entities,
                filters
            )
            
            if debug:
                self.debug_info.append(ScoringDebugInfo(
                    chunk_id=chunk.chunk_id,
                    method='graphrag',
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
                        'retrieval_method': retrieval_method
                    }
            else:
                scored_chunks[chunk.chunk_id] = {
                    'chunk': chunk,
                    'score': score,
                    'retrieval_method': retrieval_method
                }
        
        # Score Naive chunks
        for chunk in naive_chunks:
            score, retrieval_method, debug_data = self._score_naive_chunk(chunk, filters)
            
            if debug:
                self.debug_info.append(ScoringDebugInfo(
                    chunk_id=chunk.chunk_id,
                    method='naive',
                    base_score=debug_data['base_score'],
                    entity_coverage=None,  # Naive doesn't use coverage
                    coverage_bonus=0.0,
                    provenance_bonus=0.0,
                    final_score=score,
                    rank=0
                ))
            
            # If chunk already seen from GraphRAG, only update if Naive score is higher
            if chunk.chunk_id in scored_chunks:
                if score > scored_chunks[chunk.chunk_id]['score']:
                    scored_chunks[chunk.chunk_id]['score'] = score
                    scored_chunks[chunk.chunk_id]['retrieval_method'] = 'naive'
            else:
                scored_chunks[chunk.chunk_id] = {
                    'chunk': chunk,
                    'score': score,
                    'retrieval_method': retrieval_method
                }
        
        # Convert to RankedChunk objects
        ranked_chunks = []
        for chunk_data in scored_chunks.values():
            chunk = chunk_data['chunk']
            ranked_chunk = RankedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=chunk_data['score'],
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
    
    def _get_relation_chunk_ids(self, subgraph: GraphSubgraph) -> Set[str]:
        """Extract chunk IDs from PCST relations (provenance)."""
        chunk_ids = set()
        for rel in subgraph.relations:
            chunk_ids.update(rel.chunk_ids)
        return chunk_ids
    
    def _score_graphrag_chunk(
        self,
        chunk: Chunk,
        relation_chunk_ids: Set[str],
        total_entities: int,
        filters: QueryFilters
    ) -> tuple[float, str, Dict]:
        """
        Score GraphRAG chunk using entity coverage.
        
        Formula:
            score = base_score + (entity_coverage * max_coverage_bonus) + provenance_bonus
        
        Args:
            chunk: Chunk to score
            relation_chunk_ids: Chunks containing PCST relations
            total_entities: Total resolved entities (for coverage calculation)
            filters: Query filters (reserved for future)
        
        Returns:
            (final_score, retrieval_method, debug_data)
        """
        debug_data = {}
        
        # Base score from chunk (entity match quality)
        base_score = chunk.score
        debug_data['base_score'] = base_score
        
        # Entity coverage bonus
        entities_in_chunk = chunk.metadata.get('entities', [])
        entity_coverage = len(entities_in_chunk) / max(total_entities, 1)  # Avoid div by 0
        coverage_bonus = entity_coverage * self.config['entity_coverage_bonus']
        
        debug_data['entity_coverage'] = entity_coverage
        debug_data['coverage_bonus'] = coverage_bonus
        
        # Provenance bonus (flat)
        provenance_bonus = 0.0
        if chunk.chunk_id in relation_chunk_ids:
            provenance_bonus = self.config['provenance_bonus']
        
        debug_data['provenance_bonus'] = provenance_bonus
        
        # Final score
        final_score = base_score + coverage_bonus + provenance_bonus
        
        return final_score, 'graphrag', debug_data
    
    def _score_naive_chunk(
        self,
        chunk: Chunk,
        filters: QueryFilters
    ) -> tuple[float, str, Dict]:
        """
        Score Naive chunk (pure FAISS similarity, NO bonuses).
        
        Args:
            chunk: Chunk to score
            filters: Query filters (reserved for future)
        
        Returns:
            (final_score, retrieval_method, debug_data)
        """
        debug_data = {}
        
        # Naive = pure semantic similarity
        score = chunk.score
        debug_data['base_score'] = score
        
        return score, 'naive', debug_data
    
    def get_debug_info(self) -> Optional[List[ScoringDebugInfo]]:
        """Return collected debug information."""
        return self.debug_info
    
    def format_for_prompt(self, result: RetrievalResult) -> dict:
        """
        Format retrieval result for LLM prompt.
        
        Returns structured dict with:
        - graph_structure: Relations from PCST
        - entities: Key entities with context
        - sources: Numbered chunks with citations
        """
        # Format relations
        relations_text = []
        for rel in result.subgraph.relations:
            relations_text.append(
                f"  • {rel.source_name} --{rel.predicate}--> {rel.target_name}"
            )
        
        # Format sources with citation numbers
        sources_text = []
        for i, chunk in enumerate(result.chunks, 1):
            source_label = f"[{i}]"
            doc_info = f"{chunk.doc_type}"
            if chunk.jurisdiction:
                doc_info += f" ({chunk.jurisdiction})"
            
            sources_text.append(
                f"{source_label} {chunk.text}\n    Source: {doc_info}"
            )
        
        return {
            'graph_structure': '\n'.join(relations_text),
            'sources': '\n\n'.join(sources_text),
            'query': result.query
        }