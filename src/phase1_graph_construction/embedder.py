"""
Chunk Embedder - Core embedding logic for RAKG pipeline
Uses BGE-M3 (1024-dim multilingual embeddings)

Author: Pau Barba i Colomer
Phase: 1A-2 Chunk Embedding (Week 2)
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class ChunkEmbedder:
    """
    BGE-M3 embedder for semantic chunks.
    
    RAKG methodology: Embed chunks for corpus retrospective retrieval
    during entity relation extraction (Phase 1).
    
    Model: BAAI/bge-m3 (1024 dimensions, multilingual)
    """
    
    def __init__(self, model_name: str = 'BAAI/bge-m3', device: str = None):
        """
        Initialize embedder with BGE-M3 model.
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cpu' or 'cuda'. If None, auto-detects
        """
        logger.info(f"Loading embedding model: {model_name}")
        
        # Load model (sentence-transformers handles device automatically)
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = 1024  # BGE-M3 standard
        
        logger.info(f"Model loaded on device: {self.model.device}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single text chunk.
        
        Args:
            text: Chunk text to embed
            
        Returns:
            1024-dim embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 32, 
                    show_progress: bool = True) -> np.ndarray:
        """
        Embed multiple chunks in batches (memory efficient).
        
        Args:
            texts: List of chunk texts
            batch_size: Number of chunks per batch (32 is optimal for CPU)
            show_progress: Show progress bar
            
        Returns:
            Array of shape (n_chunks, 1024)
        """
        logger.info(f"Embedding {len(texts)} chunks with batch_size={batch_size}")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Validate dimensions
        assert embeddings.shape[1] == self.embedding_dim, \
            f"Expected {self.embedding_dim}-dim, got {embeddings.shape[1]}-dim"
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Return embedding dimensionality."""
        return self.embedding_dim