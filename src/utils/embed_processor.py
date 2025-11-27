"""
Universal Embed Processor - Pipeline orchestration for embedding
Handles batch processing, checkpoints, and progress tracking
Works for both chunks and entities

Author: Pau Barba i Colomer
Usage: Phase 1A-2 (chunks), Phase 1C-1 (entities)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import numpy as np
from tqdm import tqdm

from .embedder import BGEEmbedder

logger = logging.getLogger(__name__)


class EmbedProcessor:
    """
    Universal orchestrator for embedding pipeline with checkpoints.
    
    Works for both:
    - Chunks (Phase 1A-2): dict[chunk_id] -> {text, metadata, embedding}
    - Entities (Phase 1C-1): dict[entity_id] -> {name, type, embedding}
    
    Flow:
    1. Load items from JSON (chunks or entities)
    2. Batch embed with progress tracking
    3. Append embeddings to item dicts
    4. Save checkpoints every N items
    5. Save final enriched items
    """
    
    def __init__(self, embedder: BGEEmbedder, checkpoint_freq: int = 1000):
        """
        Initialize processor.
        
        Args:
            embedder: BGEEmbedder instance
            checkpoint_freq: Save progress every N items
        """
        self.embedder = embedder
        self.checkpoint_freq = checkpoint_freq
    
    def load_items(self, filepath: Path) -> Dict:
        """
        Load items from JSON file (generic for chunks or entities).
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Dictionary of item_id -> item_data
        """
        logger.info(f"Loading items from: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            items = json.load(f)
        
        # Handle both dict and list formats
        if isinstance(items, list):
            items = {i: item for i, item in enumerate(items)}
        
        logger.info(f"Loaded {len(items)} items")
        return items
    
    def save_items(self, items: Dict, filepath: Path):
        """
        Save enriched items to JSON.
        
        Args:
            items: Dictionary with embeddings added
            filepath: Output path
        """
        logger.info(f"Saving {len(items)} enriched items to: {filepath}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        
        logger.info("Save complete")
    
    def save_checkpoint(self, items: Dict, checkpoint_dir: Path, 
                       item_count: int):
        """
        Save intermediate checkpoint.
        
        Args:
            items: Current item dictionary
            checkpoint_dir: Directory for checkpoints
            item_count: Number of items processed
        """
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = checkpoint_dir / f"checkpoint_{item_count}_{timestamp}.json"
        
        logger.info(f"Saving checkpoint: {item_count} items processed")
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    
    def process_items(self, items: Dict, text_key: str = 'text',
                     batch_size: int = 32,
                     checkpoint_dir: Path = None) -> Dict:
        """
        Embed all items with batch processing and checkpoints.
        
        Args:
            items: Dictionary of item_id -> item_data
            text_key: Key to extract text from item dict
                - For chunks: 'text' (full chunk content)
                - For entities: 'formatted_text' (should be "name [type]")
            batch_size: Embedding batch size
            checkpoint_dir: Directory for saving checkpoints (optional)
            
        Returns:
            Items dictionary with 'embedding' field added
        """
        logger.info(f"Starting embedding with batch_size={batch_size}")
        logger.info(f"Checkpoint frequency: every {self.checkpoint_freq} items")
        
        # Convert dict to list for batch processing
        item_ids = list(items.keys())
        item_texts = [items[item_id][text_key] for item_id in item_ids]
        
        # Track progress
        processed_count = 0
        
        # Process in batches
        logger.info("Embedding items...")
        embeddings = self.embedder.embed_batch(
            item_texts, 
            batch_size=batch_size,
            show_progress=True
        )
        
        # Add embeddings to item dictionaries
        logger.info("Adding embeddings to items...")
        for item_id, embedding in tqdm(zip(item_ids, embeddings), 
                                       total=len(item_ids),
                                       desc="Enriching items"):
            # Convert numpy array to list for JSON serialization
            items[item_id]['embedding'] = embedding.tolist()
            
            processed_count += 1
            
            # Save checkpoint if needed
            if checkpoint_dir and processed_count % self.checkpoint_freq == 0:
                self.save_checkpoint(items, checkpoint_dir, processed_count)
        
        logger.info(f"✓ Embedded {processed_count} items")
        return items
    
    def verify_embeddings(self, items: Dict) -> Dict:
        """
        Verify all items have valid embeddings.
        
        Returns:
            Dictionary with verification statistics
        """
        logger.info("Verifying embeddings...")
        
        total = len(items)
        with_embeddings = 0
        correct_dim = 0
        
        for item_id, item_data in items.items():
            if 'embedding' in item_data:
                with_embeddings += 1
                if len(item_data['embedding']) == self.embedder.get_embedding_dim():
                    correct_dim += 1
        
        stats = {
            'total_items': total,
            'items_with_embeddings': with_embeddings,
            'items_correct_dim': correct_dim,
            'success_rate': (correct_dim / total) * 100 if total > 0 else 0
        }
        
        logger.info(f"Verification results:")
        logger.info(f"  Total items: {stats['total_items']}")
        logger.info(f"  With embeddings: {stats['items_with_embeddings']}")
        logger.info(f"  Correct dimensions: {stats['items_correct_dim']}")
        logger.info(f"  Success rate: {stats['success_rate']:.2f}%")
        
        return stats
