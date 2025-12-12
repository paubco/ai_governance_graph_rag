# -*- coding: utf-8 -*-
"""
Module: result_ranker.py
Package: src.retrieval
Purpose: Merge and rank chunks from dual-path retrieval

Author: Pau Barba i Colomer
Created: 2025-12-07
Modified: 2025-12-07

References:
    - RAGulating (Agarwal et al., 2025) - Provenance tracking
    - PHASE_3_DESIGN.md § 5.3 (Ranking strategy)

Ranking Strategy:
    Base score: Semantic similarity (from FAISS or entity resolution)
    
    Bonuses:
    - Provenance bonus (+0.3): Chunk contains PCST relation (highest priority)
    - Path A bonus (+0.2): Chunk from entity expansion (medium priority)
    - Jurisdiction boost (+0.1): Chunk matches jurisdiction hint (soft filter)
    
    Final: Top-K chunks by score
"""

import numpy as np
from typing import List, Set
from collections import defaultdict

from .config import (
    Chunk,
    RankedChunk,
    GraphSubgraph,
    QueryFilters,
    RetrievalResult,
    RANKING_CONFIG,
)


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
            config: Ranking configuration (uses RANKING_CONFIG if None)
        """
        self.config = config or RANKING_CONFIG
    
    def rank(
        self,
        path_a_chunks: List[Chunk],
        path_b_chunks: List[Chunk],
        subgraph: GraphSubgraph,
        filters: QueryFilters,
        query: str
    ) -> RetrievalResult:
        """
        Merge, deduplicate, and rank chunks.
        
        Args:
            path_a_chunks: Chunks from GraphRAG path
            path_b_chunks: Chunks from semantic search
            subgraph: PCST subgraph (for relation provenance)
            filters: Query filters (for jurisdiction boosting)
            query: Original query string
        
        Returns:
            RetrievalResult with top-K ranked chunks
        """
        # Get relation provenance chunk IDs
        relation_chunk_ids = self._get_relation_chunk_ids(subgraph)
        
        # Score Path A chunks
        scored_chunks = {}
        for chunk in path_a_chunks:
            score, source_path = self._score_chunk_path_a(
                chunk, 
                relation_chunk_ids,
                filters
            )
            
            # If chunk already seen (from Path A), keep higher score
            if chunk.chunk_id in scored_chunks:
                if score > scored_chunks[chunk.chunk_id]['score']:
                    scored_chunks[chunk.chunk_id] = {
                        'chunk': chunk,
                        'score': score,
                        'source_path': source_path
                    }
            else:
                scored_chunks[chunk.chunk_id] = {
                    'chunk': chunk,
                    'score': score,
                    'source_path': source_path
                }
        
        # Score Path B chunks
        for chunk in path_b_chunks:
            score, source_path = self._score_chunk_path_b(chunk, filters)
            
            # If chunk already seen from Path A, only update if Path B score is higher
            if chunk.chunk_id in scored_chunks:
                if score > scored_chunks[chunk.chunk_id]['score']:
                    scored_chunks[chunk.chunk_id]['score'] = score
                    scored_chunks[chunk.chunk_id]['source_path'] = 'naive'
            else:
                scored_chunks[chunk.chunk_id] = {
                    'chunk': chunk,
                    'score': score,
                    'source_path': source_path
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
                entities=chunk.metadata.get('entities', []),
                doc_type=chunk.doc_type,
                jurisdiction=chunk.jurisdiction,
                metadata=chunk.metadata
            )
            ranked_chunks.append(ranked_chunk)
        
        # Sort by score and take top-K
        ranked_chunks.sort(key=lambda c: c.score, reverse=True)
        top_k = ranked_chunks[:self.config['final_top_k']]
        
        return RetrievalResult(
            chunks=top_k,
            subgraph=subgraph,
            query=query
        )
    
    def _get_relation_chunk_ids(self, subgraph: GraphSubgraph) -> Set[str]:
        """Extract chunk IDs from PCST relations (provenance)."""
        chunk_ids = set()
        for rel in subgraph.relations:
            chunk_ids.update(rel.chunk_ids)
        return chunk_ids
    
    def _score_chunk_path_a(
        self,
        chunk: Chunk,
        relation_chunk_ids: Set[str],
        filters: QueryFilters
    ) -> tuple[float, str]:
        """
        Score chunk from Path A.
        
        Returns:
            (score, source_path)
        """
        # Base score: 0.5 (entity expansion baseline)
        score = 0.5
        
        # Check if chunk contains PCST relation (highest priority)
        if chunk.chunk_id in relation_chunk_ids:
            score += self.config['provenance_bonus']
            source_path = 'graphrag_relation'
        else:
            score += self.config['path_a_bonus']
            source_path = 'graphrag_entity'
        
        # Jurisdiction boost (soft filter)
        if filters.jurisdiction_hints and chunk.jurisdiction:
            if chunk.jurisdiction in filters.jurisdiction_hints:
                score += self.config['jurisdiction_boost']
        
        return score, source_path
    
    def _score_chunk_path_b(
        self,
        chunk: Chunk,
        filters: QueryFilters
    ) -> tuple[float, str]:
        """
        Score chunk from Path B.
        
        Uses FAISS rank as base score (normalized).
        
        Returns:
            (score, source_path)
        """
        # Base score from FAISS rank (inverse rank, normalized to 0-0.5 range)
        # This ensures even rank 0 (best) starts at 0.5, below provenance bonus
        faiss_rank = chunk.metadata.get('faiss_rank', 0)
        max_rank = self.config['final_top_k']
        base_score = 0.5 * (1.0 - (faiss_rank / max_rank))
        
        score = base_score + self.config['path_b_baseline']
        
        # Jurisdiction boost
        if filters.jurisdiction_hints and chunk.jurisdiction:
            if chunk.jurisdiction in filters.jurisdiction_hints:
                score += self.config['jurisdiction_boost']
        
        return score, 'naive'
    
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