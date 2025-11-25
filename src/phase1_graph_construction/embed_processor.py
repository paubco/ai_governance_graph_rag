"""
Embed Processor - Pipeline orchestration for chunk embedding
Handles batch processing, checkpoints, and progress tracking

Author: Pau Barba i Colomer
Phase: 1A-2 Chunk Embedding (Week 2)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import numpy as np
from tqdm import tqdm

from .embedder import ChunkEmbedder

logger = logging.getLogger(__name__)


class EmbedProcessor:
    """
    Orchestrates chunk embedding pipeline with checkpoints.
    
    Flow:
    1. Load chunks from JSON
    2. Batch embed with progress tracking
    3. Append embeddings to chunk dicts
    4. Save checkpoints every N chunks
    5. Save final enriched chunks
    """
    
    def __init__(self, embedder: ChunkEmbedder, checkpoint_freq: int = 1000):
        """
        Initialize processor.
        
        Args:
            embedder: ChunkEmbedder instance
            checkpoint_freq: Save progress every N chunks
        """
        self.embedder = embedder
        self.checkpoint_freq = checkpoint_freq
    
    def load_chunks(self, filepath: Path) -> Dict:
        """
        Load chunks from JSON file.
        
        Args:
            filepath: Path to chunks_text.json
            
        Returns:
            Dictionary of chunk_id -> chunk_data
        """
        logger.info(f"Loading chunks from: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        logger.info(f"Loaded {len(chunks)} chunks")
        return chunks
    
    def save_chunks(self, chunks: Dict, filepath: Path):
        """
        Save enriched chunks to JSON.
        
        Args:
            chunks: Dictionary with embeddings added
            filepath: Output path
        """
        logger.info(f"Saving {len(chunks)} enriched chunks to: {filepath}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        logger.info("Save complete")
    
    def save_checkpoint(self, chunks: Dict, checkpoint_dir: Path, 
                       chunk_count: int):
        """
        Save intermediate checkpoint.
        
        Args:
            chunks: Current chunk dictionary
            checkpoint_dir: Directory for checkpoints
            chunk_count: Number of chunks processed
        """
        checkpoint_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = checkpoint_dir / f"checkpoint_{chunk_count}_{timestamp}.json"
        
        logger.info(f"Saving checkpoint: {chunk_count} chunks processed")
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    def process_chunks(self, chunks: Dict, batch_size: int = 32,
                      checkpoint_dir: Path = None) -> Dict:
        """
        Embed all chunks with batch processing and checkpoints.
        
        Args:
            chunks: Dictionary of chunk_id -> chunk_data
            batch_size: Embedding batch size
            checkpoint_dir: Directory for saving checkpoints (optional)
            
        Returns:
            Chunks dictionary with 'embedding' field added
        """
        logger.info(f"Starting chunk embedding with batch_size={batch_size}")
        logger.info(f"Checkpoint frequency: every {self.checkpoint_freq} chunks")
        
        # Convert dict to list for batch processing
        chunk_ids = list(chunks.keys())
        chunk_texts = [chunks[cid]['text'] for cid in chunk_ids]
        
        # Track progress
        processed_count = 0
        
        # Process in batches
        logger.info("Embedding chunks...")
        embeddings = self.embedder.embed_batch(
            chunk_texts, 
            batch_size=batch_size,
            show_progress=True
        )
        
        # Add embeddings to chunk dictionaries
        logger.info("Adding embeddings to chunks...")
        for chunk_id, embedding in tqdm(zip(chunk_ids, embeddings), 
                                         total=len(chunk_ids),
                                         desc="Enriching chunks"):
            # Convert numpy array to list for JSON serialization
            chunks[chunk_id]['embedding'] = embedding.tolist()
            
            processed_count += 1
            
            # Save checkpoint if needed
            if checkpoint_dir and processed_count % self.checkpoint_freq == 0:
                self.save_checkpoint(chunks, checkpoint_dir, processed_count)
        
        logger.info(f"✓ Embedded {processed_count} chunks")
        return chunks
    
    def verify_embeddings(self, chunks: Dict) -> Dict:
        """
        Verify all chunks have valid embeddings.
        
        Returns:
            Dictionary with verification statistics
        """
        logger.info("Verifying embeddings...")
        
        total = len(chunks)
        with_embeddings = 0
        correct_dim = 0
        
        for chunk_id, chunk_data in chunks.items():
            if 'embedding' in chunk_data:
                with_embeddings += 1
                if len(chunk_data['embedding']) == self.embedder.get_embedding_dim():
                    correct_dim += 1
        
        stats = {
            'total_chunks': total,
            'chunks_with_embeddings': with_embeddings,
            'chunks_correct_dim': correct_dim,
            'success_rate': (correct_dim / total) * 100 if total > 0 else 0
        }
        
        logger.info(f"Verification results:")
        logger.info(f"  Total chunks: {stats['total_chunks']}")
        logger.info(f"  With embeddings: {stats['chunks_with_embeddings']}")
        logger.info(f"  Correct dimensions: {stats['chunks_correct_dim']}")
        logger.info(f"  Success rate: {stats['success_rate']:.2f}%")
        
        return stats