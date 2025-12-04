# -*- coding: utf-8 -*-
"""
Phase 1D-0: Entity Co-occurrence Matrix Construction (Typed Matrices)

Pre-computes which normalized entities appear in each chunk.
Creates 3 matrices with different type filters:
  - semantic: Excludes academic entities (for Track 1 OPENIE)
  - concept: Only concept-type entities (for Track 2 objects)
  - full: All entities (backup/debugging)

This enables O(1) entity lookup during relation extraction and
supports entity-aware diversity in chunk selection.

Input:
  - normalized_entities.json (~18-21k entities)
  - chunks_embedded.json (25,131 chunks)

Output:
  - cooccurrence_semantic.json (~8-10MB)
  - cooccurrence_concept.json (~3-5MB)
  - cooccurrence_full.json (~10-12MB)

Runtime: ~30-60 minutes (one-time cost)
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.entity_type_classification import (
    is_semantic, is_concept, is_skip, is_academic,
    get_extraction_strategy, print_classification_stats
)

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


def build_typed_cooccurrence_matrices(
    chunks: List[Dict],
    entities: List[Dict],
    checkpoint_file: str = "data/interim/entities/cooccurrence_checkpoint_typed.json",
    checkpoint_interval: int = 5000
) -> Dict[str, Dict[str, List[str]]]:
    """
    Build 3 typed co-occurrence matrices using existing chunk_ids
    
    Strategy: Define what's NOT semantic, everything else defaults to semantic
    
    Matrices:
      - semantic: All entities except academic/skip types (for Track 1 OPENIE)
      - concept: Only concept-type entities (for Track 2 objects)
      - full: All entities except skip types (backup/debugging)
    
    Uses chunk_ids already stored in entities (from Phase 1B extraction)
    instead of scanning all chunks. This is ~10,000x faster.
    
    Args:
        chunks: List of chunk dicts (only used for validation)
        entities: List of normalized entity dicts with chunk_ids
        checkpoint_file: Path to save progress checkpoints
        checkpoint_interval: Save every N entities
    
    Returns:
        Dict with 'semantic', 'concept', 'full' matrices
        Each matrix maps chunk_id to list of entity names
    """
    logger.info(f"Building typed co-occurrence matrices from {len(entities)} entities")
    logger.info(f"  Using existing chunk_ids (no scanning needed)")
    
    # Load checkpoint if exists
    start_idx = 0
    semantic_cooccur = defaultdict(set)
    concept_cooccur = defaultdict(set)
    full_cooccur = defaultdict(set)
    
    if Path(checkpoint_file).exists():
        logger.info(f"Loading checkpoint from {checkpoint_file}")
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
            
            # Convert lists back to sets
            semantic_lists = checkpoint_data.get('semantic', {})
            concept_lists = checkpoint_data.get('concept', {})
            full_lists = checkpoint_data.get('full', {})
            
            semantic_cooccur = defaultdict(set, {k: set(v) for k, v in semantic_lists.items()})
            concept_cooccur = defaultdict(set, {k: set(v) for k, v in concept_lists.items()})
            full_cooccur = defaultdict(set, {k: set(v) for k, v in full_lists.items()})
            
            start_idx = checkpoint_data.get('last_index', 0) + 1
        logger.info(f"  Resuming from entity {start_idx}/{len(entities)}")
    
    # Build co-occurrence matrices from entity chunk_ids
    start_time = time.time()
    last_checkpoint_time = start_time
    
    # Track statistics
    semantic_count = 0
    concept_count = 0
    full_count = 0
    skip_count = 0
    
    for i in range(start_idx, len(entities)):
        entity = entities[i]
        entity_name = entity['name']
        entity_type = entity['type']
        chunk_ids = entity.get('chunk_ids', [])
        
        # Check if should skip
        if is_skip(entity_type, entity_name):
            skip_count += 1
            continue
        
        # Add entity to appropriate matrices
        for chunk_id in chunk_ids:
            # Full matrix (everything except skips)
            full_cooccur[chunk_id].add(entity_name)
            full_count += 1
            
            # Semantic matrix (exclude academic + skip types)
            if is_semantic(entity_type, entity_name):
                semantic_cooccur[chunk_id].add(entity_name)
                semantic_count += 1
            
            # Concept matrix (only concept types)
            if is_concept(entity_type):
                concept_cooccur[chunk_id].add(entity_name)
                concept_count += 1
        
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
            checkpoint_data = {
                'semantic': {k: list(v) for k, v in semantic_cooccur.items()},
                'concept': {k: list(v) for k, v in concept_cooccur.items()},
                'full': {k: list(v) for k, v in full_cooccur.items()},
                'last_index': i,
                'timestamp': time.time()
            }
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            last_checkpoint_time = time.time()
    
    # Convert sets to lists for final output
    semantic_final = {k: list(v) for k, v in semantic_cooccur.items()}
    concept_final = {k: list(v) for k, v in concept_cooccur.items()}
    full_final = {k: list(v) for k, v in full_cooccur.items()}
    
    # Remove checkpoint file on completion
    if Path(checkpoint_file).exists():
        logger.info(f"Removing checkpoint file (processing complete)")
        Path(checkpoint_file).unlink()
    
    # Statistics
    total_time = time.time() - start_time
    
    logger.info(f"✓ Typed co-occurrence matrices built in {total_time/60:.1f} minutes:")
    logger.info(f"")
    logger.info(f"  SEMANTIC MATRIX:")
    logger.info(f"    Chunks with entities: {len(semantic_final):,}/{len(chunks):,}")
    logger.info(f"    Total entity mentions: {semantic_count:,}")
    logger.info(f"    Avg entities/chunk: {semantic_count/len(semantic_final) if semantic_final else 0:.1f}")
    logger.info(f"")
    logger.info(f"  CONCEPT MATRIX:")
    logger.info(f"    Chunks with entities: {len(concept_final):,}/{len(chunks):,}")
    logger.info(f"    Total entity mentions: {concept_count:,}")
    logger.info(f"    Avg entities/chunk: {concept_count/len(concept_final) if concept_final else 0:.1f}")
    logger.info(f"")
    logger.info(f"  FULL MATRIX:")
    logger.info(f"    Chunks with entities: {len(full_final):,}/{len(chunks):,}")
    logger.info(f"    Total entity mentions: {full_count:,}")
    logger.info(f"    Avg entities/chunk: {full_count/len(full_final) if full_final else 0:.1f}")
    logger.info(f"")
    logger.info(f"  Skipped entities: {skip_count:,}")
    
    return {
        'semantic': semantic_final,
        'concept': concept_final,
        'full': full_final
    }


def main():
    """Build typed entity co-occurrence matrices"""
    
    # File paths
    entities_file = "data/interim/entities/normalized_entities.json"
    chunks_file = "data/interim/chunks/chunks_embedded.json"
    
    output_files = {
        'semantic': "data/interim/entities/cooccurrence_semantic.json",
        'concept': "data/interim/entities/cooccurrence_concept.json",
        'full': "data/interim/entities/cooccurrence_full.json"
    }
    
    # Create output directory
    for output_file in output_files.values():
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("PHASE 1D-0: TYPED ENTITY CO-OCCURRENCE MATRIX CONSTRUCTION")
    logger.info("=" * 80)
    
    # Load data
    entities = load_normalized_entities(entities_file)
    chunks = load_chunks(chunks_file)
    
    # Print classification statistics
    logger.info("")
    print_classification_stats(entities)
    logger.info("")
    
    # Build typed co-occurrence matrices
    matrices = build_typed_cooccurrence_matrices(chunks, entities)
    
    # Save outputs
    logger.info("")
    logger.info("Saving typed co-occurrence matrices...")
    
    for matrix_type, matrix_data in matrices.items():
        output_file = output_files[matrix_type]
        
        logger.info(f"  Saving {matrix_type} matrix to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(matrix_data, f, indent=2, ensure_ascii=False)
        
        # File size
        file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
        logger.info(f"    ✓ Saved ({file_size_mb:.1f} MB)")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 1D-0 COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Verify file sizes match expectations:")
    logger.info("     - cooccurrence_semantic.json: ~8-10 MB")
    logger.info("     - cooccurrence_concept.json:  ~3-5 MB")
    logger.info("     - cooccurrence_full.json:     ~10-12 MB")
    logger.info("  2. Test on sample entities before full Phase 1D")
    logger.info("  3. Proceed to Phase 1D main loop (relation extraction)")


if __name__ == "__main__":
    main()