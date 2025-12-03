"""
Phase 1D-0: Entity Co-occurrence Matrix Construction

Pre-computes which normalized entities appear in each chunk.
This enables O(1) entity lookup during relation extraction and
supports entity-aware diversity in chunk selection.

Input:
  - normalized_entities.json (~30-50k entities)
  - chunks_embedded.json (25,131 chunks)

Output:
  - entity_cooccurrence.json (~5-10MB)
    Format: {chunk_id: [entity_ids]}

Runtime: ~30-60 minutes (one-time cost)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_normalized_entities(entities_file: str) -> List[Dict]:
    """Load normalized entities from JSON file"""
    logger.info(f"Loading normalized entities from {entities_file}")
    
    with open(entities_file, 'r', encoding='utf-8') as f:
        entities = json.load(f)
    
    logger.info(f"✓ Loaded {len(entities)} normalized entities")
    return entities


def load_chunks(chunks_file: str) -> List[Dict]:
    """Load chunks from JSON file"""
    logger.info(f"Loading chunks from {chunks_file}")
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle dict or list format
    if isinstance(data, dict):
        chunks = list(data.values())
    else:
        chunks = data
    
    logger.info(f"✓ Loaded {len(chunks)} chunks")
    return chunks


def build_cooccurrence_matrix(
    chunks: List[Dict],
    entities: List[Dict],
    checkpoint_file: str = "data/interim/entities/cooccurrence_checkpoint.json",
    checkpoint_interval: int = 5000
) -> Dict[str, List[str]]:
    """
    Build entity co-occurrence matrix using existing chunk_ids
    
    Uses chunk_ids already stored in entities (from Phase 1B extraction)
    instead of scanning all chunks. This is ~10,000x faster.
    
    Args:
        chunks: List of chunk dicts (only used for validation)
        entities: List of normalized entity dicts with chunk_ids
        checkpoint_file: Path to save progress checkpoints
        checkpoint_interval: Save every N entities
    
    Returns:
        Dict mapping chunk_id to list of entity names
    """
    import time
    from collections import defaultdict
    
    logger.info(f"Building co-occurrence matrix from {len(entities)} entities")
    logger.info(f"  Using existing chunk_ids (no scanning needed)")
    
    # Load checkpoint if exists
    start_idx = 0
    cooccurrence = defaultdict(set)
    
    if Path(checkpoint_file).exists():
        logger.info(f"Loading checkpoint from {checkpoint_file}")
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
            # Convert lists back to sets
            cooccurrence_lists = checkpoint_data.get('cooccurrence', {})
            cooccurrence = defaultdict(set, {k: set(v) for k, v in cooccurrence_lists.items()})
            start_idx = checkpoint_data.get('last_index', 0) + 1
        logger.info(f"  Resuming from entity {start_idx}/{len(entities)}")
    
    # Build co-occurrence from entity chunk_ids
    start_time = time.time()
    last_checkpoint_time = start_time
    
    for i in range(start_idx, len(entities)):
        entity = entities[i]
        entity_name = entity['name']
        chunk_ids = entity.get('chunk_ids', [])
        
        # Add this entity to all its chunks
        for chunk_id in chunk_ids:
            cooccurrence[chunk_id].add(entity_name)
        
        # Progress logging every 1000 entities
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            entities_done = i + 1 - start_idx
            rate = entities_done / elapsed if elapsed > 0 else 0
            remaining = len(entities) - (i + 1)
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_minutes = eta_seconds / 60
            
            logger.info(f"  Progress: {i+1}/{len(entities)} entities ({(i+1)/len(entities)*100:.1f}%) | "
                       f"Rate: {rate:.1f} entities/s | ETA: {eta_minutes:.1f} min")
        
        # Checkpoint every N entities
        if (i + 1) % checkpoint_interval == 0:
            checkpoint_elapsed = time.time() - last_checkpoint_time
            logger.info(f"  Saving checkpoint at {i+1}/{len(entities)} ({checkpoint_elapsed:.1f}s since last)")
            
            # Convert sets to lists for JSON
            cooccurrence_lists = {k: list(v) for k, v in cooccurrence.items()}
            
            checkpoint_data = {
                'cooccurrence': cooccurrence_lists,
                'last_index': i,
                'timestamp': time.time()
            }
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            last_checkpoint_time = time.time()
    
    # Convert sets to lists for final output
    cooccurrence_final = {k: list(v) for k, v in cooccurrence.items()}
    
    # Remove checkpoint file on completion
    if Path(checkpoint_file).exists():
        logger.info(f"Removing checkpoint file (processing complete)")
        Path(checkpoint_file).unlink()
    
    # Statistics
    total_entries = sum(len(names) for names in cooccurrence_final.values())
    chunks_with_entities = len(cooccurrence_final)
    avg_entities_per_chunk = total_entries / chunks_with_entities if chunks_with_entities > 0 else 0
    
    total_time = time.time() - start_time
    logger.info(f"✓ Co-occurrence matrix built in {total_time/60:.1f} minutes:")
    logger.info(f"  Chunks with entities: {chunks_with_entities}/{len(chunks)}")
    logger.info(f"  Total entity mentions: {total_entries}")
    logger.info(f"  Avg entities/chunk: {avg_entities_per_chunk:.1f}")
    
    return cooccurrence_final


def main():
    """Build entity co-occurrence matrix"""
    
    # File paths
    entities_file = "data/interim/entities/normalized_entities.json"
    chunks_file = "data/interim/chunks/chunks_embedded.json"
    output_file = "data/interim/entities/entity_cooccurrence.json"
    
    # Create output directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("PHASE 1D-0: ENTITY CO-OCCURRENCE MATRIX CONSTRUCTION")
    logger.info("=" * 80)
    
    # Load data
    entities = load_normalized_entities(entities_file)
    chunks = load_chunks(chunks_file)
    
    # Build co-occurrence matrix
    cooccurrence = build_cooccurrence_matrix(chunks, entities)
    
    # Save output
    logger.info(f"Saving co-occurrence matrix to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cooccurrence, f, indent=2, ensure_ascii=False)
    
    # File size
    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    logger.info(f"✓ Saved co-occurrence matrix ({file_size_mb:.1f} MB)")
    
    logger.info("=" * 80)
    logger.info("PHASE 1D-0 COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()