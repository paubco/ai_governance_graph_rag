# -*- coding: utf-8 -*-
"""
Result ranking with mode-specific strategies for ablation studies.

Merges and ranks chunks from dual retrieval channels using three distinct strategies:
(1) SEMANTIC mode ranks by pure FAISS similarity for baseline vector retrieval,
(2) GRAPH mode uses entity coverage scoring prioritizing chunks with query entities,
and (3) DUAL mode applies two-stage ranking with graph recall followed by semantic
reranking for precision.

Entity coverage scoring calculates terminal_coverage (query entities / total terminals)
weighted at 1.0 and path_coverage (subgraph entities / path entities) weighted at 0.5,
plus 0.1 provenance bonus for relation-source chunks. Filter penalties apply multiplicatively
for jurisdiction/doc-type mismatches. DUAL mode merges both channels, deduplicates, then
reranks using FAISS similarity while preserving entity context.

Examples:
    # Initialize ranker with FAISS index for DUAL mode
    from src.retrieval.result_ranker import ResultRanker
    import faiss
    
    chunk_index = faiss.read_index("data/processed/faiss/chunk_embeddings.index")
    chunk_id_map = {...}  # chunk_id -> index mapping
    
    ranker = ResultRanker(chunk_index=chunk_index, chunk_id_map=chunk_id_map)

    # Rank in DUAL mode (graph + semantic)
    result = ranker.rank(
        graph_chunks=graph_chunks,
        semantic_chunks=semantic_chunks,
        subgraph=subgraph,
        filters=query_filters,
        query=query_text,
        query_embedding=query_emb,
        terminal_entity_ids=terminal_ids,
        mode="dual"
    )

    # Inspect top chunks
    for chunk in result.chunks[:5]:
        print(f"[{chunk.score:.3f}] {chunk.source_path}: {chunk.text[:100]}...")

References:
    Entity coverage scoring: PHASE_3_DESIGN.md § 3.3.2c
    config.retrieval_config.RANKING_CONFIG: penalties and final_top_k
    Multiplicative scoring: All components bounded [0,1] for fair comparison
"""
# Standard library
from typing import List, Set, Dict, Optional
from dataclasses import dataclass
import numpy as np

# Third-party
import faiss

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


class ResultRanker:
    """
    Merge and rank chunks from dual retrieval channels.
    
    Strategies by mode:
    - SEMANTIC: Rank by FAISS similarity (already scored)
    - GRAPH: Rank by entity coverage (terminal > path > other)
    - DUAL: Two-stage (graph recall + semantic rerank)
    """
    
    def __init__(self, chunk_index: faiss.Index = None, chunk_id_map: Dict = None):
        """
        Initialize result ranker.
        
        Args:
            chunk_index: FAISS index for semantic reranking (needed for DUAL mode).
            chunk_id_map: Mapping of chunk_id -> FAISS index position.
        """
        self.config = RANKING_CONFIG
        self.chunk_index = chunk_index
        self.chunk_id_map = chunk_id_map or {}
        self.index_to_chunk = {v: k for k, v in self.chunk_id_map.items()}
    
    def rank(
        self,
        graph_chunks: List[Chunk],
        semantic_chunks: List[Chunk],
        subgraph: Subgraph,
        filters: QueryFilters,
        query: str,
        query_embedding: np.ndarray = None,
        terminal_entity_ids: Set[str] = None,
        mode: str = "dual"
    ) -> RetrievalResult:
        """
        Merge, deduplicate, and rank chunks.
        
        Args:
            graph_chunks: Chunks from graph retrieval.
            semantic_chunks: Chunks from semantic search.
            subgraph: PCST subgraph.
            filters: Query filters (jurisdiction, doc_type).
            query: Original query string.
            query_embedding: Query embedding for DUAL reranking.
            terminal_entity_ids: Original query entity IDs (for graph scoring).
            mode: "semantic", "graph", or "dual".
        
        Returns:
            RetrievalResult with ranked chunks.
        """
        terminal_ids = terminal_entity_ids or set()
        
        if mode == "semantic":
            ranked = self._rank_semantic(semantic_chunks, filters)
        elif mode == "graph":
            ranked = self._rank_graph(graph_chunks, subgraph, terminal_ids, filters)
        else:  # dual
            ranked = self._rank_dual_twostage(
                graph_chunks, semantic_chunks, subgraph, 
                terminal_ids, filters, query_embedding
            )
        
        # Take top-K
        top_k = ranked[:self.config['final_top_k']]
        
        return RetrievalResult(
            query=query,
            chunks=top_k,
            subgraph=subgraph,
        )
    
    def _rank_semantic(
        self, 
        chunks: List[Chunk], 
        filters: QueryFilters
    ) -> List[RankedChunk]:
        """
        SEMANTIC mode: Pure FAISS similarity with filter penalties.
        
        Score = similarity × jurisdiction_penalty × doc_type_penalty
        """
        ranked = []
        for chunk in chunks:
            base_score = chunk.metadata.get('score', 0.5)
            penalty = self._compute_filter_penalty(chunk, filters)
            final_score = base_score * penalty
            
            ranked.append(RankedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=final_score,
                source_path="semantic",
                doc_id=chunk.document_id,
                doc_type=chunk.metadata.get('doc_type', 'unknown'),
                jurisdiction=chunk.metadata.get('jurisdiction'),
                matching_entities=[],
            ))
        
        ranked.sort(key=lambda c: c.score, reverse=True)
        return ranked
    
    def _rank_graph(
        self,
        chunks: List[Chunk],
        subgraph: Subgraph,
        terminal_ids: Set[str],
        filters: QueryFilters
    ) -> List[RankedChunk]:
        """
        GRAPH mode: Entity coverage scoring.
        
        Score = terminal_coverage × 1.0 + path_coverage × 0.5 + provenance_bonus
        
        Where:
        - terminal_coverage = # terminal entities in chunk / # total terminals
        - path_coverage = # path entities in chunk / # path entities
        - provenance_bonus = 0.1 if chunk is relation source
        """
        # Separate terminal vs path entities
        path_ids = set(subgraph.entity_ids) - terminal_ids if subgraph.entity_ids else set()
        
        # Get relation provenance chunk IDs
        relation_chunk_ids = set()
        for rel in subgraph.relations:
            relation_chunk_ids.update(rel.chunk_ids)
        
        ranked = []
        for chunk in chunks:
            chunk_entities = set(chunk.metadata.get('entities', []))
            
            # Terminal coverage (most important)
            terminal_overlap = len(chunk_entities & terminal_ids)
            terminal_coverage = terminal_overlap / len(terminal_ids) if terminal_ids else 0
            
            # Path coverage (secondary)
            path_overlap = len(chunk_entities & path_ids)
            path_coverage = path_overlap / len(path_ids) if path_ids else 0
            
            # Provenance bonus
            is_provenance = chunk.metadata.get('is_relation_provenance', False)
            provenance_bonus = 0.1 if is_provenance else 0
            
            # Combined score
            base_score = (terminal_coverage * 1.0) + (path_coverage * 0.5) + provenance_bonus
            
            # Normalize to [0, 1] range (max possible ~1.6)
            base_score = min(base_score / 1.6, 1.0)
            
            # Apply filter penalties
            penalty = self._compute_filter_penalty(chunk, filters)
            final_score = base_score * penalty
            
            source_path = "graph_provenance" if is_provenance else "graph_entity"
            
            ranked.append(RankedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=final_score,
                source_path=source_path,
                doc_id=chunk.document_id,
                doc_type=chunk.metadata.get('doc_type', 'unknown'),
                jurisdiction=chunk.metadata.get('jurisdiction'),
                matching_entities=list(chunk_entities),
            ))
        
        ranked.sort(key=lambda c: c.score, reverse=True)
        return ranked
    
    def _rank_dual_twostage(
        self,
        graph_chunks: List[Chunk],
        semantic_chunks: List[Chunk],
        subgraph: Subgraph,
        terminal_ids: Set[str],
        filters: QueryFilters,
        query_embedding: np.ndarray
    ) -> List[RankedChunk]:
        """
        DUAL mode: Two-stage retrieval.
        
        Stage 1: Graph provides RECALL (find chunks you wouldn't find semantically)
        Stage 2: Semantic provides PRECISION (rerank by query similarity)
        
        This leverages the strength of both:
        - Graph finds structurally-related content
        - Semantic ensures relevance to query
        """
        # Collect all unique chunks
        all_chunks = {}
        chunk_sources = {}  # Track where each chunk came from
        
        # Get relation provenance chunk IDs
        relation_chunk_ids = set()
        for rel in subgraph.relations:
            relation_chunk_ids.update(rel.chunk_ids)
        
        # Add graph chunks (with source tracking)
        for chunk in graph_chunks:
            cid = chunk.chunk_id
            if cid not in all_chunks:
                all_chunks[cid] = chunk
                is_prov = chunk.metadata.get('is_relation_provenance', False)
                chunk_sources[cid] = "graph_provenance" if is_prov else "graph_entity"
        
        # Add semantic chunks (with source tracking)
        for chunk in semantic_chunks:
            cid = chunk.chunk_id
            if cid not in all_chunks:
                all_chunks[cid] = chunk
                chunk_sources[cid] = "semantic"
            elif chunk_sources[cid].startswith("graph"):
                # Chunk found by both - mark as dual
                chunk_sources[cid] = "dual"
        
        # Stage 2: Rerank ALL by semantic similarity
        if query_embedding is not None and self.chunk_index is not None:
            # Get similarities for all chunks at once
            chunk_similarities = self._batch_similarity(
                list(all_chunks.keys()), 
                query_embedding
            )
        else:
            # Fallback: use existing scores
            chunk_similarities = {
                cid: chunk.metadata.get('score', 0.5) 
                for cid, chunk in all_chunks.items()
            }
        
        # Build ranked results
        ranked = []
        for cid, chunk in all_chunks.items():
            base_score = chunk_similarities.get(cid, 0.5)
            
            # Apply filter penalties
            penalty = self._compute_filter_penalty(chunk, filters)
            final_score = base_score * penalty
            
            # Small bonus for graph-found chunks (they're structurally relevant)
            source = chunk_sources[cid]
            if source == "dual":
                # Found by both - slight boost
                final_score *= 1.05
            elif source in ["graph_provenance", "graph_entity"]:
                # Found only by graph - keep score but track source
                pass
            
            chunk_entities = chunk.metadata.get('entities', [])
            
            ranked.append(RankedChunk(
                chunk_id=cid,
                text=chunk.text,
                score=final_score,
                source_path=source,
                doc_id=chunk.document_id,
                doc_type=chunk.metadata.get('doc_type', 'unknown'),
                jurisdiction=chunk.metadata.get('jurisdiction'),
                matching_entities=chunk_entities if isinstance(chunk_entities, list) else [],
            ))
        
        ranked.sort(key=lambda c: c.score, reverse=True)
        return ranked
    
    def _batch_similarity(
        self, 
        chunk_ids: List[str], 
        query_embedding: np.ndarray
    ) -> Dict[str, float]:
        """
        Get semantic similarity for multiple chunks at once.
        
        Uses FAISS index to compute query-chunk similarities.
        """
        similarities = {}
        
        # Get FAISS indices for chunks we have
        faiss_indices = []
        chunk_id_order = []
        
        for cid in chunk_ids:
            if cid in self.chunk_id_map:
                faiss_indices.append(self.chunk_id_map[cid])
                chunk_id_order.append(cid)
        
        if not faiss_indices or self.chunk_index is None:
            # Fallback: return default scores
            return {cid: 0.5 for cid in chunk_ids}
        
        # Reconstruct chunk embeddings from FAISS
        try:
            chunk_embeddings = np.zeros((len(faiss_indices), self.chunk_index.d), dtype='float32')
            for i, idx in enumerate(faiss_indices):
                self.chunk_index.reconstruct(idx, chunk_embeddings[i])
            
            # Compute similarities
            query_vec = query_embedding.reshape(1, -1).astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_vec)
            faiss.normalize_L2(chunk_embeddings)
            
            # Dot product = cosine similarity (after normalization)
            sims = np.dot(chunk_embeddings, query_vec.T).flatten()
            
            for cid, sim in zip(chunk_id_order, sims):
                similarities[cid] = float(sim)
                
        except Exception as e:
            # Fallback on any FAISS error
            print(f"Warning: FAISS similarity failed: {e}")
            return {cid: 0.5 for cid in chunk_ids}
        
        # Fill in missing chunks with default
        for cid in chunk_ids:
            if cid not in similarities:
                similarities[cid] = 0.5
        
        return similarities
    
    def _compute_filter_penalty(self, chunk: Chunk, filters: QueryFilters) -> float:
        """
        Compute multiplicative penalty for filter mismatches.
        
        Only applies if filters were actually detected in query.
        """
        penalty = 1.0
        
        jurisdiction = chunk.metadata.get('jurisdiction')
        doc_type = chunk.metadata.get('doc_type')
        
        # Only penalize if user query had jurisdiction hints
        if filters.jurisdiction_hints and jurisdiction:
            if jurisdiction not in filters.jurisdiction_hints:
                penalty *= self.config['jurisdiction_penalty']
        
        # Only penalize if user query had doc_type hints
        if filters.doc_type_hints and doc_type:
            if doc_type not in filters.doc_type_hints:
                penalty *= self.config['doc_type_penalty']
        
        return penalty