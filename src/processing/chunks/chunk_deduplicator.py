# -*- coding: utf-8 -*-
"""
Chunk deduplication using embedding cosine similarity

Deduplicates chunks using cosine similarity on BGE-M3 embeddings with configurable
threshold. Processes chunks in batches to avoid memory issues with large matrices,
marks higher-index chunks as duplicates, and tracks duplicate pairs for reporting
and analysis.

References:
ARCHITECTURE.md: Section 3.1.1 for deduplication design

"""
# Standard library
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Foundation
from src.utils.dataclasses import EmbeddedChunk

logger = logging.getLogger(__name__)


class ChunkDeduplicator:
    """
    Deduplicate chunks using cosine similarity on BGE-M3 embeddings.
    
    Strategy:
        - Compute pairwise cosine similarity for all chunks
        - If similarity >= threshold, mark higher-index chunk as duplicate
        - Keep first occurrence, track duplicate pairs for reporting
    
    Args:
        threshold: Cosine similarity threshold (default 0.95)
    """
    
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        logger.info(f"ChunkDeduplicator initialized: threshold={threshold}")
    
    def deduplicate(
        self,
        chunks: List[EmbeddedChunk]
    ) -> Tuple[List[EmbeddedChunk], Dict]:
        """
        Remove duplicate chunks based on embedding similarity.
        
        Args:
            chunks: List of EmbeddedChunk with embeddings
            
        Returns:
            Tuple of (filtered_chunks, stats_dict)
        """
        if not chunks:
            return [], {'duplicates_removed': 0, 'duplicate_pairs': []}
        
        logger.info(f"Deduplicating {len(chunks)} chunks at threshold={self.threshold}")
        
        # Stack embeddings into matrix
        embeddings = np.array([c.embedding for c in chunks])
        
        # Find duplicates using batched similarity computation
        duplicate_indices = set()
        duplicate_pairs = []
        doc_duplicate_counts = defaultdict(int)
        
        # Process in batches to avoid memory issues with large matrices
        batch_size = 1000
        n_chunks = len(chunks)
        
        for i in tqdm(range(0, n_chunks, batch_size), desc="Computing similarities"):
            batch_end = min(i + batch_size, n_chunks)
            batch_embeddings = embeddings[i:batch_end]
            
            # Compare batch against all subsequent chunks
            if batch_end < n_chunks:
                remaining_embeddings = embeddings[batch_end:]
                similarities = cosine_similarity(batch_embeddings, remaining_embeddings)
                
                # Find pairs above threshold
                for batch_idx in range(similarities.shape[0]):
                    global_idx = i + batch_idx
                    for remaining_idx in range(similarities.shape[1]):
                        other_global_idx = batch_end + remaining_idx
                        
                        if similarities[batch_idx, remaining_idx] >= self.threshold:
                            # Mark higher index as duplicate
                            duplicate_indices.add(other_global_idx)
                            duplicate_pairs.append({
                                'kept': chunks[global_idx].chunk_id,
                                'removed': chunks[other_global_idx].chunk_id,
                                'similarity': float(similarities[batch_idx, remaining_idx])
                            })
                            # Track which doc had duplicates
                            doc_duplicate_counts[chunks[other_global_idx].document_id] += 1
            
            # Also check within batch
            if batch_end - i > 1:
                within_batch_sim = cosine_similarity(batch_embeddings)
                for j in range(within_batch_sim.shape[0]):
                    for k in range(j + 1, within_batch_sim.shape[1]):
                        if within_batch_sim[j, k] >= self.threshold:
                            global_k = i + k
                            if global_k not in duplicate_indices:
                                duplicate_indices.add(global_k)
                                duplicate_pairs.append({
                                    'kept': chunks[i + j].chunk_id,
                                    'removed': chunks[global_k].chunk_id,
                                    'similarity': float(within_batch_sim[j, k])
                                })
                                doc_duplicate_counts[chunks[global_k].document_id] += 1
        
        # Filter out duplicates
        filtered_chunks = [
            chunk for idx, chunk in enumerate(chunks)
            if idx not in duplicate_indices
        ]
        
        # Build stats
        top_duplicate_docs = sorted(
            doc_duplicate_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        stats = {
            'duplicates_removed': len(duplicate_indices),
            'original_count': len(chunks),
            'final_count': len(filtered_chunks),
            'dedup_ratio': len(filtered_chunks) / len(chunks) if chunks else 1.0,
            'duplicate_pairs_sample': duplicate_pairs[:20],  # First 20 for inspection
            'top_duplicate_docs': [
                {'doc_id': doc_id, 'duplicates': count}
                for doc_id, count in top_duplicate_docs
            ]
        }
        
        logger.info(
            f"Deduplication complete: {len(chunks)} -> {len(filtered_chunks)} "
            f"({len(duplicate_indices)} duplicates at threshold={self.threshold})"
        )
        
        return filtered_chunks, stats