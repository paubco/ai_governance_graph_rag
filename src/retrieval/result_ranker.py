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
    - PHASE_3_DESIGN.md § 5.3 (Ranking strategy)

Ranking Strategy:
    GraphRAG chunks: Entity-match base score (0.5-1.0) + bonuses
    Naive chunks: FAISS similarity (0-1.0) + bonuses
    
    Bonuses:
    - Provenance bonus (+0.3): Chunk contains PCST relation
    - GraphRAG bonus (+0.2): Chunk from entity expansion
    - Jurisdiction boost (+0.1): Matches query jurisdiction
    
    Examples of final scores:
    
    Scenario 1: High-quality GraphRAG chunk with provenance
      Base: 0.8 (4 entity matches)
      + Provenance: 0.3
      = 1.1 (beats most naive chunks)
    
    Scenario 2: Perfect semantic match (naive)
      Base: 0.95 (FAISS similarity)
      + Jurisdiction: 0.1
      = 1.05 (beats low-quality graphrag)
    
    Scenario 3: Mediocre GraphRAG chunk
      Base: 0.6 (2 entity matches)
      + GraphRAG bonus: 0.2
      = 0.8 (loses to strong naive chunks)
    
    Scenario 4: Off-topic naive chunk
      Base: 0.3 (low FAISS similarity)
      = 0.3 (loses to everything)
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
    original_score: float
    method: str  # 'graphrag' or 'naive'
    bonuses_applied: Dict[str, float]
    final_score: float
    rank: int


# ============================================================================
# RESULT RANKER
# ============================================================================

class ResultRanker:
    """
    Merge and rank chunks from dual-path retrieval.
    
    Handles:
    - Deduplication (same chunk from both paths)
    - Scoring with provenance bonus
    - Top-K selection
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
        Merge, deduplicate, and rank chunks.
        
        Args:
            graphrag_chunks: Chunks from entity-centric GraphRAG path.
            naive_chunks: Chunks from semantic search.
            subgraph: PCST subgraph (for relation provenance).
            filters: Query filters (for jurisdiction boosting).
            query: Original query string.
            debug: If True, collect detailed scoring information.
        
        Returns:
            RetrievalResult with top-K ranked chunks.
        """
        self.debug_info = [] if debug else None
        
        # Get relation provenance chunk IDs
        relation_chunk_ids = self._get_relation_chunk_ids(subgraph)
        
        # Score GraphRAG chunks
        scored_chunks = {}
        for chunk in graphrag_chunks:
            score, retrieval_method, bonuses = self._score_graphrag_chunk(
                chunk, 
                relation_chunk_ids,
                filters
            )
            
            if debug:
                self.debug_info.append(ScoringDebugInfo(
                    chunk_id=chunk.chunk_id,
                    original_score=chunk.score,
                    method='graphrag',
                    bonuses_applied=bonuses,
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
            score, retrieval_method, bonuses = self._score_naive_chunk(chunk, filters)
            
            if debug:
                self.debug_info.append(ScoringDebugInfo(
                    chunk_id=chunk.chunk_id,
                    original_score=chunk.score,
                    method='naive',
                    bonuses_applied=bonuses,
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
        filters: QueryFilters
    ) -> tuple[float, str, Dict[str, float]]:
        """
        Score chunk from GraphRAG path.
        
        Preserves original chunk score and adds bonuses.
        
        Returns:
            (final_score, retrieval_method, bonuses_dict)
        """
        bonuses = {}
        
        # Start with chunk's original score (from entity resolution)
        score = chunk.score
        
        # Add provenance bonus if chunk contains PCST relation
        if chunk.chunk_id in relation_chunk_ids:
            bonuses['provenance'] = self.config['provenance_bonus']
            score += bonuses['provenance']
            retrieval_method = 'graphrag'
        else:
            bonuses['graphrag_baseline'] = self.config['path_a_bonus']
            score += bonuses['graphrag_baseline']
            retrieval_method = 'graphrag'
        
        # Jurisdiction boost (soft filter)
        if filters.jurisdiction_hints and chunk.jurisdiction:
            if chunk.jurisdiction in filters.jurisdiction_hints:
                bonuses['jurisdiction'] = self.config['jurisdiction_boost']
                score += bonuses['jurisdiction']
        
        return score, retrieval_method, bonuses
    
    def _score_naive_chunk(
        self,
        chunk: Chunk,
        filters: QueryFilters
    ) -> tuple[float, str, Dict[str, float]]:
        """
        Score chunk from Naive RAG path.
        
        Preserves original FAISS similarity score.
        
        Returns:
            (final_score, retrieval_method, bonuses_dict)
        """
        bonuses = {}
        
        # Start with chunk's original FAISS score
        score = chunk.score
        
        # Jurisdiction boost
        if filters.jurisdiction_hints and chunk.jurisdiction:
            if chunk.jurisdiction in filters.jurisdiction_hints:
                bonuses['jurisdiction'] = self.config['jurisdiction_boost']
                score += bonuses['jurisdiction']
        
        return score, 'naive', bonuses
    
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