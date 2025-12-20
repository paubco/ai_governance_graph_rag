# -*- coding: utf-8 -*-
"""
Phase 1C Entity Disambiguation - Main Orchestrator (v2.0).

Two-path architecture:
    - Semantic path: FAISS blocking + tiered thresholds + LLM SameJudge
    - Metadata path: Document/DocumentSection relations, no merge

v2.0 Changes:
    - "Academic" → "Metadata" terminology
    - 6 metadata types: Citation, Author, Journal, Affiliation, Document, DocumentSection
    - NEW: SAME_AS relations (Document ↔ Regulation)
    - PART_OF now links DocumentSection → Document (not Citation → Regulation)

Usage:
    # Full run
    python -m src.processing.entities.disambiguation_processor
    
    # Sample for threshold tuning
    python -m src.processing.entities.disambiguation_processor --sample 500 --seed 42
    
    # Resume from checkpoint
    python -m src.processing.entities.disambiguation_processor --resume

Outputs:
    - data/processed/entities/entities_semantic_raw.jsonl (without embeddings)
    - data/processed/entities/entities_metadata.jsonl
    - data/processed/entities/aliases.json
    - data/processed/relations/part_of_relations.jsonl (DocumentSection → Document)
    - data/processed/relations/same_as_relations.jsonl (Document ↔ Regulation)
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
    apply_merges, route_by_type, build_entity_map, METADATA_TYPES
)
from src.processing.entities.metadata_disambiguator import (
    MetadataDisambiguator, build_chunk_entity_map
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

# Outputs (v2.0)
SEMANTIC_OUTPUT = DATA_DIR / 'processed' / 'entities' / 'entities_semantic.jsonl'
METADATA_OUTPUT = DATA_DIR / 'processed' / 'entities' / 'entities_metadata.jsonl'
ALIASES_FILE = DATA_DIR / 'processed' / 'entities' / 'aliases.json'
PART_OF_FILE = DATA_DIR / 'processed' / 'relations' / 'part_of_relations.jsonl'
SAME_AS_FILE = DATA_DIR / 'processed' / 'relations' / 'same_as_relations.jsonl'

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
    
    # Tiered thresholds (tuned from manual review)
    'auto_merge_threshold': 0.98,   # 0.96-0.98 has false positives
    'auto_reject_threshold': 0.88,  # 0.85-0.88 is ~95% DIFF
    
    # LLM SameJudge
    'llm_model': 'mistralai/Mistral-7B-Instruct-v0.3',
    'max_llm_pairs': 25000,  # ~21K expected, buffer for safety
    'max_workers': 8,        # Parallel workers for LLM calls
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
            phase: str = 'faiss',
            resume: bool = False) -> Dict:
        """
        Run disambiguation pipeline up to specified phase.
        
        Args:
            sample_size: If set, only process this many entities (for testing)
            seed: Random seed for sampling
            phase: Pipeline phase to run up to:
                - 'filter': Pre-entity filtering only
                - 'dedup': Filter + exact deduplication  
                - 'embed': Filter + dedup + embedding
                - 'faiss': Filter + dedup + embed + FAISS blocking
                - 'full': All above + LLM verification
            resume: Resume from checkpoint
            resume: Resume from checkpoint
            
        Returns:
            Stats dict with all metrics
        """
        logger.info("=" * 70)
        logger.info("Phase 1C: Entity Disambiguation")
        logger.info("=" * 70)
        
        # Ensure output directories exist
        for path in [SEMANTIC_OUTPUT, METADATA_OUTPUT, ALIASES_FILE, 
                     PART_OF_FILE, SAME_AS_FILE, FILTER_REPORT, HALLUCINATIONS_FILE]:
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
        
        if phase == 'filter':
            logger.info("\nPhase 'filter' complete - stopping before routing")
            self._print_summary()
            return self.stats
        
        # =================================================================
        # STEP 3: Route by type
        # =================================================================
        logger.info("\n[3/7] Routing by entity type...")
        
        semantic_raw, metadata_raw = route_by_type(clean_entities)
        
        self.stats['semantic_raw'] = len(semantic_raw)
        self.stats['metadata_raw'] = len(metadata_raw)
        
        # =================================================================
        # STEP 4: Semantic disambiguation (phases: dedup, embed, faiss, full)
        # =================================================================
        logger.info("\n[4/7] Disambiguating semantic entities...")
        
        semantic_entities, self.aliases = self._disambiguate_semantic(
            semantic_raw, 
            phase=phase
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
        # STEP 5: Metadata processing (Document ↔ Regulation, DocumentSection → Document)
        # =================================================================
        logger.info("\n[5/7] Processing metadata entities + relations...")
        
        # MetadataDisambiguator needs semantic entities for SAME_AS matching
        disambiguator = MetadataDisambiguator(semantic_entities=semantic_entities)
        metadata_entities, part_of_relations, same_as_relations = disambiguator.process(metadata_raw)
        
        self.stats['metadata_output'] = len(metadata_entities)
        self.stats['part_of_relations'] = len(part_of_relations)
        self.stats['same_as_relations'] = len(same_as_relations)
        
        # =================================================================
        # STEP 6: Save outputs (without embeddings)
        # =================================================================
        logger.info("\n[6/7] Saving outputs (raw, without embeddings)...")
        
        # Save semantic entities (raw - will embed separately)
        save_jsonl(semantic_entities, str(SEMANTIC_OUTPUT))
        logger.info(f"  Saved {len(semantic_entities)} semantic entities → {SEMANTIC_OUTPUT}")
        
        # Save metadata entities
        save_jsonl(metadata_entities, str(METADATA_OUTPUT))
        logger.info(f"  Saved {len(metadata_entities)} metadata entities → {METADATA_OUTPUT}")
        
        # Save aliases
        save_json(self.aliases, str(ALIASES_FILE))
        logger.info(f"  Saved {len(self.aliases)} alias mappings → {ALIASES_FILE}")
        
        # Save PART_OF relations (DocumentSection → Document)
        save_jsonl(part_of_relations, str(PART_OF_FILE))
        logger.info(f"  Saved {len(part_of_relations)} PART_OF relations → {PART_OF_FILE}")
        
        # Save SAME_AS relations (Document ↔ Regulation)
        save_jsonl(same_as_relations, str(SAME_AS_FILE))
        logger.info(f"  Saved {len(same_as_relations)} SAME_AS relations → {SAME_AS_FILE}")
        
        # =================================================================
        # STEP 7: Summary
        # =================================================================
        logger.info("\n[7/7] Summary")
        self._print_summary()
        
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
                               phase: str = 'faiss') -> Tuple[List[Dict], Dict]:
        """
        Run semantic disambiguation pipeline up to specified phase.
        
        Phases:
            'dedup': ExactDeduplicator only (hash-based)
            'embed': Dedup + entity embedding
            'faiss': Dedup + embed + FAISS blocking (outputs pairs)
            'full':  All above + LLM verification
        """
        # Stage 1: Exact deduplication (always runs)
        deduplicator = ExactDeduplicator()
        deduped, aliases = deduplicator.deduplicate(entities)
        
        self.stats['stage1_dedup'] = deduplicator.stats
        
        if phase == 'dedup':
            logger.info("Phase 'dedup' complete - stopping before embedding")
            return deduped, aliases
        
        # Stage 1.5: Embed entities
        has_embeddings = any('embedding' in e for e in deduped)
        
        if not has_embeddings:
            logger.info("Embedding entities with BGE-M3...")
            deduped = self._embed_entities(deduped)
        else:
            logger.info("Entities already have embeddings - skipping embed")
        
        if phase == 'embed':
            logger.info("Phase 'embed' complete - stopping before FAISS")
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
        
        if phase == 'faiss':
            # Save candidate pairs for threshold tuning
            pairs_file = DATA_DIR / 'interim' / 'entities' / 'candidate_pairs.jsonl'
            pairs_file.parent.mkdir(parents=True, exist_ok=True)
            save_jsonl(pairs, str(pairs_file))
            logger.info(f"Saved {len(pairs)} candidate pairs → {pairs_file}")
            logger.info(f"\nPhase 'faiss' complete - run threshold tuning next:")
            logger.info(f"  python -m src.processing.entities.tests.test_threshold_refinement")
            
            # Apply only auto-merges (high confidence), no LLM
            if filtered['merged']:
                final_entities, aliases = apply_merges(deduped, filtered['merged'], aliases)
                self.stats['auto_merges'] = len(filtered['merged'])
            else:
                final_entities = deduped
            
            return final_entities, aliases
        
        # Stage 4: LLM verification for uncertain pairs (phase == 'full')
        llm_approved = []
        
        if filtered['uncertain']:
            uncertain = filtered['uncertain']
            
            # Limit LLM calls for cost control
            if len(uncertain) > self.config['max_llm_pairs']:
                logger.warning(f"Limiting LLM pairs: {len(uncertain)} → {self.config['max_llm_pairs']}")
                uncertain = sorted(uncertain, key=lambda x: -x['similarity'])[:self.config['max_llm_pairs']]
            
            judge = SameJudge(model=self.config['llm_model'])
            entity_map = build_entity_map(deduped)
            
            checkpoint_dir = str(DATA_DIR / 'interim' / 'entities')
            llm_approved = judge.judge_pairs(
                uncertain, 
                entity_map,
                checkpoint_dir=checkpoint_dir,
                checkpoint_freq=1000,
                max_workers=self.config['max_workers']
            )
            
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
        logger.info("PIPELINE SUMMARY (v2.0)")
        logger.info("-" * 50)
        logger.info(f"Input entities:     {self.stats.get('input_entities', 0):,}")
        logger.info(f"After filter:       {self.stats.get('after_filter', 0):,}")
        logger.info(f"  Semantic raw:     {self.stats.get('semantic_raw', 0):,}")
        logger.info(f"  Metadata raw:     {self.stats.get('metadata_raw', 0):,}")
        logger.info(f"Semantic output:    {self.stats.get('semantic_after_dedup', 0):,}")
        logger.info(f"Metadata output:    {self.stats.get('metadata_output', 0):,}")
        logger.info(f"Aliases tracked:    {self.stats.get('aliases_count', 0):,}")
        logger.info(f"PART_OF relations:  {self.stats.get('part_of_relations', 0):,}")
        logger.info(f"SAME_AS relations:  {self.stats.get('same_as_relations', 0):,}")
        
        if 'filter_stats' in self.stats:
            fs = self.stats['filter_stats']
            logger.info(f"\nFilter breakdown:")
            logger.info(f"  Blacklist removed:   {fs.get('blacklist_removed', 0):,}")
            logger.info(f"  Provenance failed:   {fs.get('provenance_failed', 0):,}")
    
    def _embed_entities(self, entities: List[Dict], batch_size: int = 64) -> List[Dict]:
        """
        Embed entities using existing EmbedProcessor + BGEEmbedder.
        
        Format: "{name}({type})" per extraction_config.py
        
        Args:
            entities: List of entity dicts with 'name' and 'type'
            batch_size: Embedding batch size (64 for GPU)
            
        Returns:
            Entities with 'embedding' field added
        """
        from src.utils.embedder import BGEEmbedder
        from src.utils.embed_processor import EmbedProcessor
        
        # Get format from config
        entity_format = self.config.get('semantic_format', '{name}({type})')
        
        logger.info(f"Embedding {len(entities)} entities...")
        logger.info(f"Format: {entity_format}")
        
        # Convert to dict format expected by EmbedProcessor
        # Add formatted_text field for embedding
        items = {}
        for i, e in enumerate(entities):
            entity_id = e.get('entity_id', f'ent_{i:05d}')
            e['formatted_text'] = entity_format.format(
                name=e.get('name', ''), 
                type=e.get('type', '')
            )
            items[entity_id] = e
        
        logger.info(f"Sample: '{entities[0].get('formatted_text', '')}'")
        
        # Use existing EmbedProcessor
        embedder = BGEEmbedder(device='cuda')
        processor = EmbedProcessor(embedder)
        
        items = processor.process_items(
            items, 
            text_key='formatted_text',
            batch_size=batch_size
        )
        
        # Convert back to list
        entities = list(items.values())
        
        logger.info(f"✓ Embedded {len(entities)} entities")
        self.stats['entities_embedded'] = len(entities)
        
        return entities


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Phase 1C Entity Disambiguation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases (cumulative):
  filter  - Pre-entity filtering only
  dedup   - Filter + exact deduplication
  embed   - Filter + dedup + embedding
  faiss   - Filter + dedup + embed + FAISS blocking (for threshold tuning)
  full    - All above + LLM verification

Typical workflow:
  1. python -m ... --phase faiss    # Run up to FAISS, outputs pairs
  2. python -m ... tests.test_threshold_refinement  # Review pairs, tune thresholds
  3. Update thresholds in extraction_config.py
  4. python -m ... --phase full     # Run with LLM on uncertain band
"""
    )
    
    parser.add_argument(
        '--phase', type=str, default='faiss',
        choices=['filter', 'dedup', 'embed', 'faiss', 'full'],
        help='Pipeline phase to run up to (default: faiss for threshold tuning)'
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
        phase=args.phase,
        resume=args.resume
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())