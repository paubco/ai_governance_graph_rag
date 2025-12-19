# -*- coding: utf-8 -*-
"""
Universal embedding processor with batch processing and checkpoints.

Orchestrates batch embedding pipeline with progress tracking and checkpointing.
Works for both text chunks (Phase 1A-2) and entities (Phase 1C-1) using BGE-M3
embedder. Features rolling checkpoint cleanup, optimized append phase without
checkpoints to prevent O(nÂ²) slowdown, and numpy array serialization for JSON
compatibility.
"""

# Standard library
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
import numpy as np
from tqdm import tqdm

# Local
from src.utils.embedder import BGEEmbedder

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
    3. Append embeddings to item dicts (fast, no checkpoints needed)
    4. Save final enriched items
    """
    
    def __init__(self, embedder: BGEEmbedder, checkpoint_freq: int = 1000):
        """
        Initialize processor.
        
        Args:
            embedder: BGEEmbedder instance
            checkpoint_freq: Save progress every N items (for future use)
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
        
        # Convert numpy arrays to lists for JSON serialization
        items_serializable = {}
        for key, item in items.items():
            item_copy = item.copy()
            if 'embedding' in item_copy and isinstance(item_copy['embedding'], np.ndarray):
                item_copy['embedding'] = item_copy['embedding'].tolist()
            items_serializable[key] = item_copy
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(items_serializable, f, ensure_ascii=False, indent=2)
        
        logger.info("Save complete")
    
    def save_checkpoint(self, items: Dict, checkpoint_dir: Path, 
                       item_count: int, max_checkpoints: int = 2):
        """
        Save intermediate checkpoint with rolling cleanup.
        Keeps only the N most recent checkpoints to save disk space.
        
        NOTE: Currently unused during append phase to avoid O(nÂ²) behavior.
        Could be used for chunked embedding in future optimization.
        
        Args:
            items: Current item dictionary
            checkpoint_dir: Directory for checkpoints
            item_count: Number of items processed
            max_checkpoints: Maximum number of checkpoints to keep (default: 2)
        """
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = checkpoint_dir / f"checkpoint_{item_count}_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        items_serializable = {}
        for key, item in items.items():
            item_copy = item.copy()
            if 'embedding' in item_copy and isinstance(item_copy['embedding'], np.ndarray):
                item_copy['embedding'] = item_copy['embedding'].tolist()
            items_serializable[key] = item_copy
        
        # Save new checkpoint
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(items_serializable, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Checkpoint saved: {item_count} items processed")
        
        # Rolling cleanup: Keep only N most recent checkpoints
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.json"))
        if len(checkpoints) > max_checkpoints:
            for old_checkpoint in checkpoints[:-max_checkpoints]:
                old_checkpoint.unlink()
                logger.debug(f"Deleted old checkpoint: {old_checkpoint.name}")
    
    def process_items(self, items: Dict, text_key: str = 'text',
                     batch_size: int = 32,
                     checkpoint_dir: Path = None) -> Dict:
        """
        Embed all items with batch processing.
        
        Args:
            items: Dictionary of item_id -> item_data
            text_key: Key to extract text from item dict
                - For chunks: 'text' (full chunk content)
                - For entities: 'formatted_text' (should be "name [type]")
            batch_size: Embedding batch size
            checkpoint_dir: Directory for checkpoints (currently unused)
            
        Returns:
            Items dictionary with 'embedding' field added
        """
        logger.info(f"Starting embedding with batch_size={batch_size}")
        
        # Convert dict to list for batch processing
        item_ids = list(items.keys())
        item_texts = [items[item_id][text_key] for item_id in item_ids]
        
        # Embed all items at once (BGEEmbedder handles batching internally)
        logger.info(f"Embedding {len(item_texts)} items...")
        embeddings = self.embedder.embed_batch(
            item_texts, 
            batch_size=batch_size,
            show_progress=True
        )
        
        # Add embeddings to item dictionaries
        # This is fast (just dict updates), no checkpoints needed
        logger.info("Adding embeddings to items...")
        for item_id, embedding in tqdm(zip(item_ids, embeddings), 
                                       total=len(item_ids),
                                       desc="Enriching items"):
            # Store as numpy array in memory
            items[item_id]['embedding'] = embedding
        
        logger.info(f"Embedded {len(items)} items successfully")
        return items
    
    def embed_with_checkpoints(
        self,
        items: Dict,
        text_key: str = 'text',
        batch_size: int = 8,
        chunk_size: int = 500,
        checkpoint_dir: Path = None,
        min_batch_size: int = 1
    ) -> Dict:
        """
        Embed items with OOM recovery and checkpointing.
        
        Processes items in chunks, saving progress after each chunk.
        On CUDA OOM, reduces batch size and retries. Resumes from
        checkpoint if items already have embeddings.
        
        Args:
            items: Dictionary of item_id -> item_data
            text_key: Key to extract text from item dict
            batch_size: Initial embedding batch size (will reduce on OOM)
            chunk_size: Number of items to process before checkpointing
            checkpoint_dir: Directory for checkpoints (required)
            min_batch_size: Minimum batch size before giving up
            
        Returns:
            Items dictionary with 'embedding' field added
        """
        import torch
        
        if checkpoint_dir is None:
            checkpoint_dir = Path('checkpoints/embeddings')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter to items without embeddings (for resume)
        pending_ids = [
            item_id for item_id, data in items.items()
            if 'embedding' not in data or data['embedding'] is None
        ]
        
        if not pending_ids:
            logger.info("All items already have embeddings, skipping")
            return items
        
        already_done = len(items) - len(pending_ids)
        if already_done > 0:
            logger.info(f"Resuming: {already_done} already embedded, {len(pending_ids)} remaining")
        else:
            logger.info(f"Embedding {len(pending_ids)} items with chunk_size={chunk_size}")
        
        current_batch_size = batch_size
        processed = 0
        
        # Process in chunks
        for chunk_start in range(0, len(pending_ids), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(pending_ids))
            chunk_ids = pending_ids[chunk_start:chunk_end]
            chunk_texts = [items[item_id][text_key] for item_id in chunk_ids]
            
            # Embed with OOM recovery
            embeddings = None
            while embeddings is None and current_batch_size >= min_batch_size:
                try:
                    # Clear CUDA cache before batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    embeddings = self.embedder.embed_batch(
                        chunk_texts,
                        batch_size=current_batch_size,
                        show_progress=True
                    )
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        old_size = current_batch_size
                        current_batch_size = max(current_batch_size // 2, min_batch_size)
                        logger.warning(
                            f"CUDA OOM with batch_size={old_size}, "
                            f"reducing to {current_batch_size}"
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        if current_batch_size < min_batch_size:
                            raise RuntimeError(
                                f"Cannot embed with batch_size={min_batch_size}. "
                                "Try reducing chunk length or using CPU."
                            )
                    else:
                        raise
            
            # Add embeddings to items
            for item_id, embedding in zip(chunk_ids, embeddings):
                items[item_id]['embedding'] = embedding
            
            processed += len(chunk_ids)
            
            # Save checkpoint
            self.save_checkpoint(items, checkpoint_dir, processed + already_done)
            logger.info(
                f"Progress: {processed + already_done}/{len(items)} "
                f"({(processed + already_done)/len(items)*100:.1f}%)"
            )
        
        logger.info(f"Embedded {len(items)} items (final batch_size={current_batch_size})")
        return items
    
    def load_checkpoint(self, checkpoint_dir: Path) -> Dict:
        """
        Load most recent checkpoint if available.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            
        Returns:
            Dictionary of items with partial embeddings, or empty dict
        """
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return {}
        
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.json"))
        if not checkpoints:
            return {}
        
        # Try loading from newest to oldest until one works
        for checkpoint in reversed(checkpoints):
            try:
                logger.info(f"Loading checkpoint: {checkpoint.name}")
                with open(checkpoint, 'r', encoding='utf-8') as f:
                    items = json.load(f)
                
                # Convert embedding lists back to numpy arrays
                for item_id, data in items.items():
                    if 'embedding' in data and isinstance(data['embedding'], list):
                        items[item_id]['embedding'] = np.array(data['embedding'])
                
                logger.info(f"Loaded {len(items)} items from checkpoint")
                return items
                
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Corrupted checkpoint {checkpoint.name}: {e}")
                # Delete corrupted checkpoint
                checkpoint.unlink()
                logger.info(f"Deleted corrupted checkpoint: {checkpoint.name}")
                continue
        
        return {}

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
                embedding = item_data['embedding']
                # Handle both numpy arrays and lists
                if isinstance(embedding, np.ndarray):
                    dim = len(embedding)
                else:
                    dim = len(embedding)
                
                if dim == self.embedder.get_embedding_dim():
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