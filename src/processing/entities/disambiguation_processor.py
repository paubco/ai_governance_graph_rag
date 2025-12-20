# -*- coding: utf-8 -*-
"""
Phase 1C Entity Disambiguation - Main Orchestrator.

Two-path architecture:
    - Semantic path: FAISS blocking + tiered thresholds + LLM SameJudge
    - Academic path: No merge, PART_OF relations only

Usage:
    # Full run
    python -m src.processing.entities.disambiguation_processor
    
    # Sample for threshold tuning
    python -m src.processing.entities.disambiguation_processor --sample 500 --seed 42
    
    # Resume from checkpoint
    python -m src.processing.entities.disambiguation_processor --resume

Outputs:
    - data/processed/entities/entities_semantic_raw.jsonl (without embeddings)
    - data/processed/entities/entities_academic.jsonl
    - data/processed/entities/aliases.json
    - data/processed/relations/part_of_relations.jsonl
    - data/interim/entities/filter_report.json
    - data/interim/entities/hallucinations.jsonl
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.processing.entities.pre_entity_filter import (
    PreEntityFilter, load_chunks_as_dict
)
from src.processing.entities.semantic_disambiguator import (
    ExactDeduplicator, FAISSBlocker, TieredThresholdFilter, SameJudge,
    apply_merges, route_by_type, build_entity_map, ACADEMIC_TYPES
)
from src.processing.entities.academic_disambiguator import (
    AcademicDisambiguator, build_chunk_entity_map
)
from src.utils.io import load_jsonl, save_jsonl, save_json, load_json
from src.utils.id_generator import generate_entity_id
from src.utils.checkpoint_manager import CheckpointManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PATHS
# =============================================================================

DATA_DIR = PROJECT_ROOT / 'data'

# Inputs
PRE_ENTITIES_FILE = DATA_DIR / 'interim' / 'entities' / 'pre_entities.jsonl'
CHUNKS_FILE = DATA_DIR / 'processed' / 'chunks' / 'chunks_embedded.jsonl'

# Outputs
SEMANTIC_OUTPUT_RAW = DATA_DIR / 'processed' / 'entities' / 'entities_semantic_raw.jsonl'
SEMANTIC_OUTPUT = DATA_DIR / 'processed' / 'entities' / 'entities_semantic.jsonl'
ACADEMIC_OUTPUT = DATA_DIR / 'processed' / 'entities' / 'entities_academic.jsonl'
ALIASES_FILE = DATA_DIR / 'processed' / 'entities' / 'aliases.json'
PART_OF_FILE = DATA_DIR / 'processed' / 'relations' / 'part_of_relations.jsonl'

# Interim
FILTER_REPORT = DATA_DIR / 'interim' / 'entities' / 'filter_report.json'
HALLUCINATIONS_FILE = DATA_DIR / 'interim' / 'entities' / 'hallucinations.jsonl'
CHECKPOINT_DIR = DATA_DIR / 'interim' / 'entities' / 'checkpoints'


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Filtering
    'verify_provenance': True,
    
    # FAISS blocking
    'faiss_k': 50,
    'faiss_threshold': 0.70,
    'faiss_M': 32,
    'faiss_ef_construction': 200,
    'faiss_ef_search': 64,
    
    # Tiered thresholds
    'auto_merge_threshold': 0.95,
    'auto_reject_threshold': 0.85,
    
    # LLM SameJudge
    'llm_model': 'mistralai/Mistral-7B-Instruct-v0.3',
    'max_llm_pairs': 1000,  # Limit LLM calls for cost control
}


# =============================================================================
# MAIN PROCESSOR
# =============================================================================

class DisambiguationProcessor:
    """
    Main orchestrator for Phase 1C entity disambiguation.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize processor with configuration."""
        self.config = config or CONFIG
        self.stats = {}
        self.aliases = {}
    
    def run(self, 
            sample_size: int = None, 
            seed: int = None,
            skip_llm: bool = False,
            resume: bool = False) -> Dict:
        """
        Run full disambiguation pipeline.
        
        Args:
            sample_size: If set, only process this many entities (for testing)
            seed: Random seed for sampling
            skip_llm: Skip LLM verification stage
            resume: Resume from checkpoint
            
        Returns:
            Stats dict with all metrics
        """
        logger.info("=" * 70)
        logger.info("Phase 1C: Entity Disambiguation")
        logger.info("=" * 70)
        
        # Ensure output directories exist
        for path in [SEMANTIC_OUTPUT_RAW, ACADEMIC_OUTPUT, ALIASES_FILE, 
                     PART_OF_FILE, FILTER_REPORT, HALLUCINATIONS_FILE]:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        # =================================================================
        # STEP 1: Load inputs
        # =================================================================
        logger.info("\n[1/7] Loading inputs...")
        
        pre_entities = self._load_pre_entities(sample_size, seed)
        chunks = load_chunks_as_dict(str(CHUNKS_FILE))
        
        self.stats['input_entities'] = len(pre_entities)
        self.stats['input_chunks'] = len(chunks)
        
        # =================================================================
        # STEP 2: Garbage filter + provenance
        # =================================================================
        logger.info("\n[2/7] Filtering garbage + verifying provenance...")
        
        pre_entity_filter = PreEntityFilter(
            chunks=chunks,
            verify_provenance=self.config['verify_provenance']
        )
        clean_entities, filter_stats = pre_entity_filter.filter(pre_entities)
        
        # Save filter outputs
        save_json(filter_stats, str(FILTER_REPORT))
        if pre_entity_filter.hallucinations:
            save_jsonl(pre_entity_filter.hallucinations, str(HALLUCINATIONS_FILE))
        
        self.stats['after_filter'] = len(clean_entities)
        self.stats['filter_stats'] = filter_stats
        
        # =================================================================
        # STEP 3: Route by type
        # =================================================================
        logger.info("\n[3/7] Routing by entity type...")
        
        semantic_raw, academic_raw = route_by_type(clean_entities)
        
        self.stats['semantic_raw'] = len(semantic_raw)
        self.stats['academic_raw'] = len(academic_raw)
        
        # =================================================================
        # STEP 4: Semantic disambiguation
        # =================================================================
        logger.info("\n[4/7] Disambiguating semantic entities...")
        
        semantic_entities, self.aliases = self._disambiguate_semantic(
            semantic_raw, 
            skip_llm=skip_llm
        )
        
        # Generate entity IDs
        for entity in semantic_entities:
            if 'entity_id' not in entity:
                entity['entity_id'] = generate_entity_id(
                    entity['name'], 
                    entity['type']
                )
        
        self.stats['semantic_after_dedup'] = len(semantic_entities)
        self.stats['aliases_count'] = len(self.aliases)
        
        # =================================================================
        # STEP 5: Academic linking
        # =================================================================
        logger.info("\n[5/7] Processing academic entities + PART_OF relations...")
        
        # Build chunk→entity map for PART_OF lookup
        chunk_entity_map = build_chunk_entity_map(semantic_entities)
        
        disambiguator = AcademicDisambiguator(
            semantic_entities=semantic_entities,
            chunk_entity_map=chunk_entity_map
        )
        academic_entities, part_of_relations = disambiguator.process(academic_raw)
        
        self.stats['academic_output'] = len(academic_entities)
        self.stats['part_of_relations'] = len(part_of_relations)
        
        # =================================================================
        # STEP 6: Save outputs (without embeddings)
        # =================================================================
        logger.info("\n[6/7] Saving outputs (raw, without embeddings)...")
        
        # Save semantic entities (raw - will embed separately)
        save_jsonl(semantic_entities, str(SEMANTIC_OUTPUT_RAW))
        logger.info(f"  Saved {len(semantic_entities)} semantic entities → {SEMANTIC_OUTPUT_RAW}")
        
        # Save academic entities
        save_jsonl(academic_entities, str(ACADEMIC_OUTPUT))
        logger.info(f"  Saved {len(academic_entities)} academic entities → {ACADEMIC_OUTPUT}")
        
        # Save aliases
        save_json(self.aliases, str(ALIASES_FILE))
        logger.info(f"  Saved {len(self.aliases)} alias mappings → {ALIASES_FILE}")
        
        # Save PART_OF relations
        save_jsonl(part_of_relations, str(PART_OF_FILE))
        logger.info(f"  Saved {len(part_of_relations)} PART_OF relations → {PART_OF_FILE}")
        
        # =================================================================
        # STEP 7: Summary
        # =================================================================
        logger.info("\n[7/7] Summary")
        self._print_summary()
        
        logger.info("\n" + "=" * 70)
        logger.info("NEXT STEP: Run embedding phase separately:")
        logger.info(f"  python -m src.processing.entities.embed_entities")
        logger.info("=" * 70)
        
        return self.stats
    
    def _load_pre_entities(self, sample_size: int = None, seed: int = None) -> List[Dict]:
        """Load pre-entities (nested format: chunk → entities[])."""
        logger.info(f"Loading from {PRE_ENTITIES_FILE}...")
        
        entities = []
        with open(PRE_ENTITIES_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chunk_data = json.loads(line)
                    # Flatten: extract entities from each chunk
                    for entity in chunk_data.get('entities', []):
                        entities.append(entity)
        
        logger.info(f"Loaded {len(entities):,} pre-entities")
        
        if sample_size and sample_size < len(entities):
            import random
            if seed:
                random.seed(seed)
            entities = random.sample(entities, sample_size)
            logger.info(f"Sampled {len(entities):,} entities (seed={seed})")
        
        return entities
    
    def _disambiguate_semantic(self, 
                               entities: List[Dict],
                               skip_llm: bool = False) -> Tuple[List[Dict], Dict]:
        """
        Run semantic disambiguation pipeline.
        
        Stages:
            1. ExactDeduplicator (hash-based)
            2. FAISSBlocker (embedding similarity)
            3. TieredThresholdFilter (auto-merge/reject)
            4. SameJudge (LLM verification for uncertain pairs)
        """
        # Stage 1: Exact deduplication
        deduplicator = ExactDeduplicator()
        deduped, aliases = deduplicator.deduplicate(entities)
        
        self.stats['stage1_dedup'] = deduplicator.stats
        
        # Check if entities have embeddings
        has_embeddings = any('embedding' in e for e in deduped)
        
        if not has_embeddings:
            logger.warning("No embeddings found - skipping FAISS stages")
            logger.warning("Run embedding first, or entities will only be hash-deduplicated")
            return deduped, aliases
        
        # Stage 2: FAISS blocking
        blocker = FAISSBlocker(
            embedding_dim=1024,
            M=self.config['faiss_M']
        )
        blocker.build_index(deduped, ef_construction=self.config['faiss_ef_construction'])
        
        pairs = blocker.find_candidates(
            deduped,
            k=self.config['faiss_k'],
            threshold=self.config['faiss_threshold'],
            ef_search=self.config['faiss_ef_search']
        )
        
        self.stats['stage2_faiss'] = blocker.stats
        
        if not pairs:
            logger.info("No candidate pairs found")
            return deduped, aliases
        
        # Stage 3: Tiered threshold filtering
        threshold_filter = TieredThresholdFilter(
            auto_merge_threshold=self.config['auto_merge_threshold'],
            auto_reject_threshold=self.config['auto_reject_threshold']
        )
        filtered = threshold_filter.filter_pairs(pairs)
        
        self.stats['stage3_threshold'] = threshold_filter.stats
        
        # Stage 4: LLM verification for uncertain pairs
        llm_approved = []
        
        if not skip_llm and filtered['uncertain']:
            uncertain = filtered['uncertain']
            
            # Limit LLM calls for cost control
            if len(uncertain) > self.config['max_llm_pairs']:
                logger.warning(f"Limiting LLM pairs: {len(uncertain)} → {self.config['max_llm_pairs']}")
                uncertain = sorted(uncertain, key=lambda x: -x['similarity'])[:self.config['max_llm_pairs']]
            
            judge = SameJudge(model=self.config['llm_model'])
            entity_map = build_entity_map(deduped)
            llm_approved = judge.judge_pairs(uncertain, entity_map)
            
            self.stats['stage4_llm'] = judge.stats
        
        # Apply all merges
        all_merges = filtered['merged'] + llm_approved
        
        if all_merges:
            final_entities, aliases = apply_merges(deduped, all_merges, aliases)
        else:
            final_entities = deduped
        
        self.stats['total_merges'] = len(all_merges)
        
        return final_entities, aliases
    
    def _print_summary(self):
        """Print pipeline summary."""
        logger.info("-" * 50)
        logger.info("PIPELINE SUMMARY")
        logger.info("-" * 50)
        logger.info(f"Input entities:     {self.stats.get('input_entities', 0):,}")
        logger.info(f"After filter:       {self.stats.get('after_filter', 0):,}")
        logger.info(f"  Semantic raw:     {self.stats.get('semantic_raw', 0):,}")
        logger.info(f"  Academic raw:     {self.stats.get('academic_raw', 0):,}")
        logger.info(f"Semantic output:    {self.stats.get('semantic_after_dedup', 0):,}")
        logger.info(f"Academic output:    {self.stats.get('academic_output', 0):,}")
        logger.info(f"Aliases tracked:    {self.stats.get('aliases_count', 0):,}")
        logger.info(f"PART_OF relations:  {self.stats.get('part_of_relations', 0):,}")
        
        if 'filter_stats' in self.stats:
            fs = self.stats['filter_stats']
            logger.info(f"\nFilter breakdown:")
            logger.info(f"  Blacklist removed:   {fs.get('blacklist_removed', 0):,}")
            logger.info(f"  Provenance failed:   {fs.get('provenance_failed', 0):,}")


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Phase 1C Entity Disambiguation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--sample', type=int, default=None,
        help='Sample size for testing (default: full run)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for sampling (default: 42)'
    )
    parser.add_argument(
        '--skip-llm', action='store_true',
        help='Skip LLM verification stage'
    )
    parser.add_argument(
        '--skip-provenance', action='store_true',
        help='Skip provenance verification'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from checkpoint'
    )
    
    args = parser.parse_args()
    
    # Update config
    config = CONFIG.copy()
    if args.skip_provenance:
        config['verify_provenance'] = False
    
    # Run pipeline
    processor = DisambiguationProcessor(config)
    stats = processor.run(
        sample_size=args.sample,
        seed=args.seed,
        skip_llm=args.skip_llm,
        resume=args.resume
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())