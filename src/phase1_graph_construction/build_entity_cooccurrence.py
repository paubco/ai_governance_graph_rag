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
import re
import logging
from pathlib import Path
from typing import List, Dict, Set
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
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, list):
        entities = data
    elif isinstance(data, dict) and 'entities' in data:
        entities = data['entities']
    elif isinstance(data, dict):
        entities = list(data.values())
    else:
        raise ValueError(f"Unexpected format in {entities_file}")
    
    logger.info(f"✓ Loaded {len(entities)} normalized entities")
    return entities


def load_chunks(chunks_file: str) -> List[Dict]:
    """Load chunks from JSON file"""
    logger.info(f"Loading chunks from {chunks_file}")
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, dict):
        chunks = list(data.values())
    elif isinstance(data, list):
        chunks = data
    else:
        raise ValueError(f"Unexpected format in {chunks_file}")
    
    logger.info(f"✓ Loaded {len(chunks)} chunks")
    return chunks


def entity_appears_in_text(entity_name: str, text_lower: str) -> bool:
    """
    Check if entity name appears in text with word boundaries
    
    Args:
        entity_name: Entity name to search for
        text_lower: Lowercased text to search in
    
    Returns:
        True if entity appears in text
    """
    name_lower = entity_name.lower()
    # Word boundary matching
    pattern = r'\b' + re.escape(name_lower) + r'\b'
    return bool(re.search(pattern, text_lower))


def build_cooccurrence_matrix(
    chunks: List[Dict],
    entities: List[Dict]
) -> Dict[str, List[str]]:
    """
    Build entity co-occurrence matrix
    
    For each chunk, detect which entities appear in the text.
    
    Args:
        chunks: List of chunk dicts with 'text' field
        entities: List of normalized entity dicts
    
    Returns:
        Dict mapping chunk_id to list of entity_ids
    """
    logger.info(f"Building co-occurrence matrix for {len(chunks)} chunks × {len(entities)} entities")
    
    cooccurrence = {}
    
    # Build entity lookup
    entity_names = [(e['id'], e['name']) for e in entities]
    
    for i, chunk in enumerate(chunks):
        if i % 1000 == 0 and i > 0:
            logger.info(f"  Processed {i}/{len(chunks)} chunks...")
        
        chunk_id = chunk.get('chunk_id', chunk.get('id', f'chunk_{i}'))
        text = chunk.get('text', '')
        text_lower = text.lower()
        
        # Detect entities in this chunk
        detected_entity_ids = []
        for entity_id, entity_name in entity_names:
            if entity_appears_in_text(entity_name, text_lower):
                detected_entity_ids.append(entity_id)
        
        if detected_entity_ids:
            cooccurrence[chunk_id] = detected_entity_ids
    
    # Statistics
    total_entries = sum(len(ids) for ids in cooccurrence.values())
    chunks_with_entities = len(cooccurrence)
    avg_entities_per_chunk = total_entries / chunks_with_entities if chunks_with_entities > 0 else 0
    
    logger.info(f"✓ Co-occurrence matrix built:")
    logger.info(f"  Chunks with entities: {chunks_with_entities}/{len(chunks)}")
    logger.info(f"  Total entity mentions: {total_entries}")
    logger.info(f"  Avg entities/chunk: {avg_entities_per_chunk:.1f}")
    
    return cooccurrence


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
