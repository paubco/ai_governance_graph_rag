# -*- coding: utf-8 -*-
"""
Entity co-occurrence matrix construction with entity_id output (v2.0).

Pre-computes which normalized entities appear in each chunk using three
type-filtered matrices for optimized relation extraction lookups.

v2.0 Changes:
- Output format: {chunk_id: [entity_ids]} instead of {chunk_id: [names]}
- Uses entity_id from Phase 1C disambiguation output
- Enables direct ID-based relation extraction without name→ID lookup

Matrix types:
    semantic: Excludes academic entities (Track 1 OpenIE)
    concept: Only concept-type entities (Track 2 objects)
    full: All entities except skip types (backup/debugging)

Input files:
    entities_semantic.jsonl (~21K entities with entity_id)
    entities_metadata.jsonl (~17K entities with entity_id)
    chunks_embedded.json (25,131 chunks)

Output files:
    cooccurrence_semantic.json {chunk_id: [entity_ids]}
    cooccurrence_concept.json {chunk_id: [entity_ids]}
    cooccurrence_full.json {chunk_id: [entity_ids]}

Runtime: ~2-5 minutes (uses entity chunk_ids, no scanning)

Example:
    python src/processing/relations/build_entity_cooccurrence.py
"""

# Standard library
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Set

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local
from src.utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


# ============================================================================
# ENTITY TYPE CLASSIFICATION
# ============================================================================

SEMANTIC_TYPES = {
    'RegulatoryConcept', 'TechnicalConcept', 'PoliticalConcept', 'EconomicConcept',
    'Regulation', 'Technology', 'Organization', 'Location', 'Risk'
}

CONCEPT_TYPES = {
    'RegulatoryConcept', 'TechnicalConcept', 'PoliticalConcept', 'EconomicConcept'
}

ACADEMIC_TYPES = {
    'Citation', 'Author', 'Journal', 'Affiliation'
}

SKIP_TYPES = {
    'Document', 'DocumentSection'  # These have PART_OF relations, not semantic
}


def is_semantic(entity_type: str) -> bool:
    """Check if entity type is semantic (Track 1 extraction)."""
    return entity_type in SEMANTIC_TYPES


def is_concept(entity_type: str) -> bool:
    """Check if entity type is concept (valid Track 2 object)."""
    return entity_type in CONCEPT_TYPES


def is_skip(entity_type: str) -> bool:
    """Check if entity type should be skipped."""
    return entity_type in SKIP_TYPES


def get_extraction_strategy(entity_type: str) -> str:
    """
    Get extraction strategy for entity type.
    
    Returns:
        'semantic': Full OpenIE extraction (Track 1)
        'academic': Subject-constrained discusses (Track 2)
        'skip': No extraction needed
    """
    if entity_type in SKIP_TYPES:
        return 'skip'
    elif entity_type in ACADEMIC_TYPES:
        return 'academic'
    elif entity_type in SEMANTIC_TYPES:
        return 'semantic'
    else:
        return 'skip'


# ============================================================================
# DATA LOADING
# ============================================================================

def load_entities_jsonl(entities_file: Path) -> List[Dict]:
    """Load entities from JSONL file."""
    logger.info(f"Loading entities from {entities_file}")
    
    entities = []
    with open(entities_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entities.append(json.loads(line))
    
    logger.info(f"✓ Loaded {len(entities):,} entities")
    return entities


def load_chunks(chunks_file: Path) -> List[Dict]:
    """Load chunks from JSON file."""
    logger.info(f"Loading chunks from {chunks_file}")
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle dict or list format
    if isinstance(data, dict):
        chunks = list(data.values())
    else:
        chunks = data
    
    logger.info(f"✓ Loaded {len(chunks):,} chunks")
    return chunks


# ============================================================================
# CO-OCCURRENCE MATRIX CONSTRUCTION
# ============================================================================

def build_typed_cooccurrence_matrices(
    semantic_entities: List[Dict],
    metadata_entities: List[Dict],
    chunks: List[Dict]
) -> Dict[str, Dict[str, List[str]]]:
    """
    Build 3 typed co-occurrence matrices using entity_ids.
    
    v2.0: Output format is {chunk_id: [entity_ids]} not {chunk_id: [names]}
    
    Uses chunk_ids already stored in entities (from Phase 1C) - no scanning needed.
    
    Args:
        semantic_entities: List of semantic entity dicts with entity_id, chunk_ids
        metadata_entities: List of metadata entity dicts with entity_id, chunk_ids
        chunks: List of chunk dicts (only used for validation)
    
    Returns:
        Dict with 'semantic', 'concept', 'full' matrices
        Each matrix maps chunk_id to list of entity_ids
    """
    all_entities = semantic_entities + metadata_entities
    logger.info(f"Building co-occurrence matrices from {len(all_entities):,} entities")
    
    # Initialize matrices
    semantic_cooccur = defaultdict(set)
    concept_cooccur = defaultdict(set)
    full_cooccur = defaultdict(set)
    
    # Track statistics
    stats = {
        'semantic_entities': 0,
        'concept_entities': 0,
        'academic_entities': 0,
        'skip_entities': 0,
    }
    
    start_time = time.time()
    
    for entity in all_entities:
        entity_id = entity.get('entity_id')
        entity_type = entity.get('type', '')
        chunk_ids = entity.get('chunk_ids', [])
        
        if not entity_id:
            logger.warning(f"Entity missing entity_id: {entity.get('name', 'Unknown')}")
            continue
        
        # Determine strategy
        strategy = get_extraction_strategy(entity_type)
        
        if strategy == 'skip':
            stats['skip_entities'] += 1
            continue
        
        # Add to appropriate matrices
        for chunk_id in chunk_ids:
            # Full matrix (everything except skip)
            full_cooccur[chunk_id].add(entity_id)
            
            # Semantic matrix (semantic types only)
            if strategy == 'semantic':
                semantic_cooccur[chunk_id].add(entity_id)
                stats['semantic_entities'] += 1
                
                # Concept matrix (subset of semantic - concepts only)
                if is_concept(entity_type):
                    concept_cooccur[chunk_id].add(entity_id)
                    stats['concept_entities'] += 1
            
            elif strategy == 'academic':
                stats['academic_entities'] += 1
                # Academic entities not added to semantic/concept matrices
                # They use Track 2 extraction
    
    # Convert sets to sorted lists for deterministic output
    semantic_final = {k: sorted(list(v)) for k, v in semantic_cooccur.items()}
    concept_final = {k: sorted(list(v)) for k, v in concept_cooccur.items()}
    full_final = {k: sorted(list(v)) for k, v in full_cooccur.items()}
    
    elapsed = time.time() - start_time
    
    # Log statistics
    logger.info(f"✓ Co-occurrence matrices built in {elapsed:.1f}s")
    logger.info(f"")
    logger.info(f"  Entity classification:")
    logger.info(f"    Semantic: {stats['semantic_entities']:,} mentions")
    logger.info(f"    Concept:  {stats['concept_entities']:,} mentions")
    logger.info(f"    Academic: {stats['academic_entities']:,} mentions")
    logger.info(f"    Skipped:  {stats['skip_entities']:,} entities")
    logger.info(f"")
    logger.info(f"  Matrix sizes:")
    logger.info(f"    Semantic: {len(semantic_final):,} chunks")
    logger.info(f"    Concept:  {len(concept_final):,} chunks")
    logger.info(f"    Full:     {len(full_final):,} chunks")
    
    # Compute average entities per chunk
    if semantic_final:
        avg_semantic = sum(len(v) for v in semantic_final.values()) / len(semantic_final)
        logger.info(f"    Avg entities/chunk (semantic): {avg_semantic:.1f}")
    if concept_final:
        avg_concept = sum(len(v) for v in concept_final.values()) / len(concept_final)
        logger.info(f"    Avg entities/chunk (concept): {avg_concept:.1f}")
    
    return {
        'semantic': semantic_final,
        'concept': concept_final,
        'full': full_final
    }


def build_entity_lookup(
    semantic_entities: List[Dict],
    metadata_entities: List[Dict]
) -> Dict[str, Dict]:
    """
    Build entity_id → entity lookup for relation extraction.
    
    Args:
        semantic_entities: List of semantic entity dicts
        metadata_entities: List of metadata entity dicts
    
    Returns:
        Dict mapping entity_id to full entity dict
    """
    lookup = {}
    
    for entity in semantic_entities + metadata_entities:
        entity_id = entity.get('entity_id')
        if entity_id:
            lookup[entity_id] = entity
    
    logger.info(f"✓ Built entity lookup with {len(lookup):,} entries")
    return lookup


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Build typed entity co-occurrence matrices with entity_id output."""
    
    # File paths
    semantic_file = PROJECT_ROOT / "data/processed/entities/entities_semantic.jsonl"
    metadata_file = PROJECT_ROOT / "data/processed/entities/entities_metadata.jsonl"
    chunks_file = PROJECT_ROOT / "data/processed/chunks/chunks_embedded.json"
    
    output_dir = PROJECT_ROOT / "data/interim/entities"
    output_files = {
        'semantic': output_dir / "cooccurrence_semantic.json",
        'concept': output_dir / "cooccurrence_concept.json",
        'full': output_dir / "cooccurrence_full.json",
        'entity_lookup': output_dir / "entity_id_lookup.json",
    }
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("PHASE 1D-0: ENTITY CO-OCCURRENCE MATRIX CONSTRUCTION (v2.0)")
    logger.info("=" * 80)
    logger.info(f"Output format: {{chunk_id: [entity_ids]}}")
    logger.info("")
    
    # Check input files exist
    missing = []
    if not semantic_file.exists():
        missing.append(f"entities_semantic.jsonl: {semantic_file}")
    if not metadata_file.exists():
        missing.append(f"entities_metadata.jsonl: {metadata_file}")
    if not chunks_file.exists():
        missing.append(f"chunks_embedded.json: {chunks_file}")
    
    if missing:
        logger.error("Missing input files:")
        for m in missing:
            logger.error(f"  - {m}")
        logger.error("Run Phase 1C first to generate entity files.")
        return 1
    
    # Load data
    semantic_entities = load_entities_jsonl(semantic_file)
    metadata_entities = load_entities_jsonl(metadata_file)
    chunks = load_chunks(chunks_file)
    
    # Build co-occurrence matrices
    matrices = build_typed_cooccurrence_matrices(
        semantic_entities, metadata_entities, chunks
    )
    
    # Build entity lookup
    entity_lookup = build_entity_lookup(semantic_entities, metadata_entities)
    
    # Save outputs
    logger.info("")
    logger.info("Saving outputs...")
    
    for matrix_type, matrix_data in matrices.items():
        output_file = output_files[matrix_type]
        logger.info(f"  Saving {matrix_type} matrix to {output_file.name}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(matrix_data, f, indent=2, ensure_ascii=False)
        
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"    ✓ Saved ({file_size_mb:.1f} MB)")
    
    # Save entity lookup
    logger.info(f"  Saving entity lookup to {output_files['entity_lookup'].name}")
    with open(output_files['entity_lookup'], 'w', encoding='utf-8') as f:
        json.dump(entity_lookup, f, indent=2, ensure_ascii=False)
    file_size_mb = output_files['entity_lookup'].stat().st_size / (1024 * 1024)
    logger.info(f"    ✓ Saved ({file_size_mb:.1f} MB)")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 1D-0 COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Output files:")
    for name, path in output_files.items():
        logger.info(f"  - {path}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Run preflight test: python -m src.processing.relations.tests.test_relation_preflight")
    logger.info("  2. Run sample extraction: python run_relation_extraction.py --entities 100")
    logger.info("  3. Run full extraction: python run_relation_extraction.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())