"""
Universal BGE-M3 Embedder for RAKG Pipeline
Supports both chunk and entity embedding

Author: Pau Barba i Colomer
Usage: Phase 1A-2 (chunks), Phase 1C-1 (entities)
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class BGEEmbedder:
    """
    Universal BGE-M3 embedder for RAKG pipeline.
    
    Used across multiple phases:
    - Phase 1A-2: Chunk embedding (full text for corpus retrieval)
    - Phase 1C-1: Entity embedding ("name [type]" for VecJudge)
    
    Model: BAAI/bge-m3 (1024 dimensions, multilingual)
    
    Example:
        embedder = BGEEmbedder(device='cuda')
        
        # For chunks
        chunk_texts = ["Article 1: ...", "Section 2: ..."]
        embeddings = embedder.embed_batch(chunk_texts)
        
        # For entities  
        entity_texts = ["GDPR [Regulation]", "EU [Organization]"]
        embeddings = embedder.embed_batch(entity_texts)
    """
    
    def __init__(self, model_name: str = 'BAAI/bge-m3', device: str = None):
        """
        Initialize BGE-M3 embedder.
        
        Args:
            model_name: HuggingFace model identifier (default: BAAI/bge-m3)
            device: 'cpu', 'cuda', or None for auto-detect
        """
        logger.info(f"Loading embedding model: {model_name}")
        
        # Load model (sentence-transformers handles device automatically)
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = 1024  # BGE-M3 standard
        
        logger.info(f"Model loaded on device: {self.model.device}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single text (chunk or entity).
        
        Args:
            text: Text to embed (chunk content or "name [type]" format)
            
        Returns:
            1024-dim embedding vector as numpy array
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 32, 
                    show_progress: bool = True) -> np.ndarray:
        """
        Embed multiple texts in batches (memory efficient).
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch
                - CPU: 32 is optimal
                - GPU: 64-128 depending on VRAM
            show_progress: Show progress bar
            
        Returns:
            Array of shape (n_texts, 1024)
        """
        logger.info(f"Embedding {len(texts)} texts with batch_size={batch_size}")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Validate dimensions
        assert embeddings.shape[1] == self.embedding_dim, \
            f"Expected {self.embedding_dim}-dim, got {embeddings.shape[1]}-dim"
        
        logger.info(f"✓ Embedded {len(texts)} texts successfully")
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Return embedding dimensionality (1024 for BGE-M3)."""
        return self.embedding_dim
