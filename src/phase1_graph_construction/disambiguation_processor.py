"""
Module: disambiguation_processor.py
Phase: 1C - Entity Disambiguation
Purpose: Orchestrate 4-stage entity disambiguation pipeline
Author: Pau Barba i Colomer
Created: 2025-11-29
Last Modified: 2025-11-29
"""

# Standard library
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Local imports
from src.phase1_graph_construction.entity_disambiguator import (
    ExactDeduplicator,
    FAISSBlocker,
    TieredThresholdFilter,
    SameJudge
)

# Import YOUR existing code
from src.utils.embedder import BGEEmbedder
from src.utils.embed_processor import EmbedProcessor

# Load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use environment variables directly

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DisambiguationProcessor:
    """
    Orchestrates full 4-stage entity disambiguation pipeline (CPU version)
    
    Pipeline:
        Stage 1: Exact deduplication (hash-based)
        Stage 1.5: Embed deduplicated entities (YOUR BGEEmbedder)
        Stage 2: FAISS HNSW blocking (CPU)
        Stage 3: Tiered threshold filtering
        Stage 4: LLM verification (SameJudge, single-threaded)
    
    For GPU-optimized version with multithreading, use:
        server_scripts/disambiguation_server.py
    """
    
    def __init__(self, 
                 together_api_key: str = None,
                 data_dir: str = "data/interim/entities"):
        """
        Initialize processor with all stages
        
        Args:
            together_api_key: API key for Stage 4 LLM verification
            data_dir: Directory for intermediate files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all stages (CPU versions)
        self.stage1 = ExactDeduplicator()
        self.embedder = BGEEmbedder()  # YOUR existing universal embedder
        self.embed_processor = EmbedProcessor(
            embedder=self.embedder,
            checkpoint_freq=1000
        )  # YOUR existing batch processor
        self.stage2 = FAISSBlocker()
        self.stage3 = TieredThresholdFilter()
        
        # Stage 4: Initialize only when needed (lazy loading)
        self.stage4 = None
        self.together_api_key = together_api_key
        
        self.stats = {}
        
        logger.info("DisambiguationProcessor initialized (CPU version)")
        logger.info("For GPU-optimized pipeline, use server_scripts/disambiguation_server.py")
    
    def load_json(self, filepath: str) -> List[Dict]:
        """Load JSON file (handles nested chunk+entities structure from Phase 1A)"""
        logger.info(f"Loading {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle Phase 1A format: {"metadata": {...}, "entities": [chunks]}
        if isinstance(data, dict):
            if 'entities' in data:
                logger.info("Extracting entities from chunks...")
                chunks = data['entities']
                
                # Flatten: each chunk has nested 'entities' array
                entities = []
                for chunk in chunks:
                    if isinstance(chunk, dict) and 'entities' in chunk:
                        # This is a chunk with nested entities
                        entities.extend(chunk['entities'])
                    else:
                        # Direct entity
                        entities.append(chunk)
                
                data = entities
                logger.info(f"Flattened {len(chunks)} chunks into {len(entities)} entities")
            else:
                # Fallback: entity IDs as keys
                logger.info("Converting dict format to list...")
                data = list(data.values())
        
        logger.info(f"Loaded {len(data)} entities")
        return data
    
    def save_json(self, data: List[Dict], filepath: str):
        """Save JSON file"""
        logger.info(f"Saving to {filepath}...")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(data)} entities")
    
    def run_stage1(self, input_file: str) -> List[Dict]:
        """
        Run Stage 1: Exact deduplication
        
        Args:
            input_file: Path to pre_entities.json
            
        Returns:
            Deduplicated entities
        """
        logger.info("="*60)
        logger.info("STAGE 1: EXACT DEDUPLICATION")
        logger.info("="*60)
        
        entities = self.load_json(input_file)
        entities = self.stage1.deduplicate(entities)
        
        # Save intermediate result
        output_file = self.data_dir / "stage1_deduplicated.json"
        self.save_json(entities, str(output_file))
        
        self.stats['stage1'] = self.stage1.stats
        
        return entities
    
    def run_stage1_5(self, entities: List[Dict]) -> List[Dict]:
        """
        Run Stage 1.5: Embed deduplicated entities using YOUR BGEEmbedder
        
        Args:
            entities: Deduplicated entities (without embeddings)
            
        Returns:
            Entities with embeddings
        """
        logger.info("="*60)
        logger.info("STAGE 1.5: EMBEDDING DEDUPLICATED ENTITIES")
        logger.info("="*60)
        logger.info("Using YOUR existing BGEEmbedder (universal embedder)")
        
        # Format entities as "name [type]" for RAKG embedding (Eq. 19)
        # This is the RAKG standard format for entity embeddings
        logger.info("Formatting entities as 'name [type]' for embedding...")
        
        # Add formatted text to each entity for embed_processor
        for entity in entities:
            entity['formatted_text'] = f"{entity['name']} [{entity['type']}]"
        
        # Use YOUR existing EmbedProcessor for batch processing with checkpoints
        entities_dict = {i: entity for i, entity in enumerate(entities)}
        
        embedded_dict = self.embed_processor.process_items(
            items=entities_dict,
            text_key='formatted_text',  # Use our formatted text
            batch_size=32,
            checkpoint_dir=self.data_dir / "embed_checkpoints"
        )
        
        # Convert back to list
        entities = list(embedded_dict.values())
        
        # Remove temporary formatted_text field
        for entity in entities:
            entity.pop('formatted_text', None)
        
        # Save intermediate result
        output_file = self.data_dir / "stage1_deduplicated_embedded.json"
        self.save_json(entities, str(output_file))
        
        # Verify embeddings
        stats = self.embed_processor.verify_embeddings(embedded_dict)
        self.stats['stage1_5'] = stats
        
        logger.info(f"✓ Stage 1.5 complete: {len(entities)} entities embedded")
        
        return entities
    
    def run_stage2(self, entities: List[Dict]) -> List[tuple]:
        """
        Run Stage 2: FAISS HNSW blocking
        
        Args:
            entities: Entities with embeddings
            
        Returns:
            Candidate pairs
        """
        logger.info("="*60)
        logger.info("STAGE 2: FAISS HNSW BLOCKING")
        logger.info("="*60)
        
        # Build index and search
        self.stage2.build_index(entities)
        pairs = self.stage2.find_candidates(entities, k=50)
        
        # Save intermediate result with entity info for traceability
        output_file = self.data_dir / "stage2_candidate_pairs.json"
        pairs_enriched = []
        for i, j, sim in pairs:
            pairs_enriched.append({
                'entity1_idx': int(i),
                'entity1_name': entities[i]['name'],
                'entity1_type': entities[i]['type'],
                'entity2_idx': int(j),
                'entity2_name': entities[j]['name'],
                'entity2_type': entities[j]['type'],
                'similarity': float(sim)
            })
        
        with open(output_file, 'w') as f:
            json.dump(pairs_enriched, f, indent=2)
        logger.info(f"Saved {len(pairs)} candidate pairs with entity names for traceability")
        
        self.stats['stage2'] = self.stage2.stats
        
        return pairs
    
    def run_stage3(self, 
                   pairs: List[tuple], 
                   entities: List[Dict]) -> Dict:
        """
        Run Stage 3: Tiered threshold filtering
        
        Args:
            pairs: Candidate pairs from Stage 2
            entities: Entities
            
        Returns:
            Filtered pairs dict
        """
        logger.info("="*60)
        logger.info("STAGE 3: TIERED THRESHOLD FILTERING")
        logger.info("="*60)
        
        filtered = self.stage3.filter_pairs(pairs, entities)
        
        # Apply auto-merges and get index mapping
        logger.info("Applying auto-merge decisions...")
        entities, index_mapping = self.stage3.apply_merges(entities, filtered['merged'])
        
        # Update uncertain pairs with new indices
        updated_uncertain = []
        for i, j, sim in filtered['uncertain']:
            new_i = index_mapping.get(i, i)
            new_j = index_mapping.get(j, j)
            
            # Skip if both merged into same entity
            if new_i == new_j:
                continue
            
            # Ensure i < j for consistency
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            
            updated_uncertain.append((new_i, new_j, sim))
        
        logger.info(f"Updated uncertain pairs: {len(filtered['uncertain'])} → {len(updated_uncertain)}")
        filtered['uncertain'] = updated_uncertain
        
        # Save intermediate results with entity info for traceability
        output_file = self.data_dir / "stage3_filtered_pairs.json"
        filtered_json = {
            'merged': [
                {
                    'entity1_idx': int(i),
                    'entity1_name': entities[i]['name'],
                    'entity1_type': entities[i]['type'],
                    'entity2_idx': int(j),
                    'entity2_name': entities[j]['name'],
                    'entity2_type': entities[j]['type']
                }
                for i, j in filtered['merged']
            ],
            'rejected': [
                {
                    'entity1_idx': int(i),
                    'entity1_name': entities[i]['name'],
                    'entity1_type': entities[i]['type'],
                    'entity2_idx': int(j),
                    'entity2_name': entities[j]['name'],
                    'entity2_type': entities[j]['type']
                }
                for i, j in filtered['rejected']
            ],
            'uncertain': [
                {
                    'entity1_idx': int(i),
                    'entity1_name': entities[i]['name'],
                    'entity1_type': entities[i]['type'],
                    'entity2_idx': int(j),
                    'entity2_name': entities[j]['name'],
                    'entity2_type': entities[j]['type'],
                    'similarity': float(sim)
                }
                for i, j, sim in filtered['uncertain']
            ]
        }
        with open(output_file, 'w') as f:
            json.dump(filtered_json, f, indent=2)
        
        # Save entities after auto-merges
        output_file = self.data_dir / "stage3_entities_after_automerge.json"
        self.save_json(entities, str(output_file))
        
        self.stats['stage3'] = self.stage3.stats
        
        return filtered, entities
    
    
    def run_stage4(self, 
                   uncertain_pairs: List[tuple],
                   entities: List[Dict]) -> List[Dict]:
        """
        Run Stage 4: LLM verification (CPU version - single-threaded)
        
        Args:
            uncertain_pairs: Uncertain pairs from Stage 3
            entities: Entities after Stage 3 auto-merges
            
        Returns:
            Final canonical entities
        """
        logger.info("="*60)
        logger.info("STAGE 4: LLM VERIFICATION (SAMEJUDGE - CPU)")
        logger.info("="*60)
        logger.info("Single-threaded processing")
        logger.info("For GPU + 8 workers, use server_scripts/disambiguation_server.py")
        
        # Initialize Stage 4 now (lazy loading)
        if self.stage4 is None:
            if not self.together_api_key:
                raise ValueError(
                    "Stage 4 requires Together API key!\n"
                    "Set via: export TOGETHER_API_KEY='your-key'\n"
                    "Or pass: --api-key your-key"
                )
            logger.info("Initializing Stage 4 (SameJudge)...")
            self.stage4 = SameJudge(api_key=self.together_api_key)
        
        # Verify uncertain pairs with LLM
        llm_matches = self.stage4.verify_batch(uncertain_pairs, entities)
        
        # Apply LLM match decisions
        logger.info("Applying LLM match decisions...")
        entities, _ = self.stage3.apply_merges(entities, llm_matches)  # Discard mapping (final stage)
        
        self.stats['stage4'] = self.stage4.stats
        
        return entities
    
    def run_pipeline(self, 
                     input_file: str = None,
                     output_file: str = None,
                     start_from_stage: int = 1,
                     stop_at_stage: int = None) -> Dict:
        """
        Execute stage-by-stage pipeline (CPU version)
        
        Args:
            input_file: Path to input entities
            output_file: Path to save final normalized entities
            start_from_stage: 
                1 = Start from exact dedup (need to embed after)
                2 = Start from FAISS (entities already deduplicated + embedded)
            stop_at_stage:
                1 = Stop after dedup (no embedding)
                2 = Stop after FAISS blocking
                3 = Stop after threshold filtering
                4 = Complete pipeline (default)
                
        Returns:
            Pipeline statistics
        """
        logger.info("="*60)
        logger.info("ENTITY DISAMBIGUATION PIPELINE (CPU)")
        logger.info("="*60)
        logger.info(f"Start from stage: {start_from_stage}")
        if stop_at_stage:
            logger.info(f"Stop at stage: {stop_at_stage}")
        logger.info("For GPU-optimized version, use server_scripts/disambiguation_server.py")
        
        if start_from_stage == 1:
            # Normal flow: dedup → embed → FAISS → thresholds → LLM
            if not input_file:
                raise ValueError("input_file required when starting from Stage 1")
            
            # Stage 1: Exact dedup
            entities = self.run_stage1(input_file)
            
            if stop_at_stage == 1:
                logger.info("✓ Stopping after Stage 1 (exact dedup)")
                return self.stats
            
            # Stage 1.5: Embed deduplicated entities (YOUR BGEEmbedder)
            entities = self.run_stage1_5(entities)
            
            if stop_at_stage == 2:
                logger.info("✓ Stopping after Stage 1.5 (embedding complete)")
                return self.stats
            
        elif start_from_stage == 2:
            # Skip to Stage 2 (entities already deduplicated + embedded)
            embedded_file = self.data_dir / "stage1_deduplicated_embedded.json"
            if not embedded_file.exists():
                raise ValueError(f"File not found: {embedded_file}. Run Stages 1-1.5 first!")
            
            entities = self.load_json(str(embedded_file))
            logger.info("Loaded deduplicated+embedded entities (skipping Stages 1-1.5)")
        
        else:
            raise ValueError(f"Invalid start_from_stage: {start_from_stage}")
        
        # Stage 2: FAISS blocking
        pairs = self.run_stage2(entities)
        
        if stop_at_stage == 2:
            logger.info("✓ Stopping after Stage 2 (FAISS blocking)")
            return self.stats
        
        # Stage 3: Tiered thresholds
        filtered, entities = self.run_stage3(pairs, entities)
        
        if stop_at_stage == 3:
            logger.info("✓ Stopping after Stage 3 (threshold filtering)")
            return self.stats
        
        # Stage 4: LLM verification (CPU - single-threaded)
        entities = self.run_stage4(filtered['uncertain'], entities)
        
        # Save final output
        if output_file:
            self.save_json(entities, output_file)
        
        # Compile stats
        self.stats['final_entity_count'] = len(entities)
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETE!")
        logger.info("="*60)
        logger.info(f"Final entity count: {len(entities)}")
        logger.info("")
        logger.info("Statistics:")
        for stage, stats in self.stats.items():
            logger.info(f"  {stage}: {stats}")
        
        return self.stats


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="Entity Disambiguation Pipeline (CPU - All 4 Stages)")
    parser.add_argument(
        '--input',
        type=str,
        help='Input file (pre_entities.json) - required if --start-from-stage 1'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/interim/entities/normalized_entities.json',
        help='Output file for normalized entities'
    )
    parser.add_argument(
        '--start-from-stage',
        type=int,
        choices=[1, 2],
        default=1,
        help='Start from Stage 1 (dedup) or Stage 2 (FAISS)'
    )
    parser.add_argument(
        '--stop-at-stage',
        type=int,
        choices=[1, 2, 3, 4],
        help='Stop after this stage (default: run all stages)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='Together.ai API key for Stage 4 (or set TOGETHER_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    import os
    api_key = args.api_key or os.getenv('TOGETHER_API_KEY')
    if not api_key and args.start_from_stage == 1:
        logger.info("No API key provided - Stage 4 (LLM verification) will be skipped")
        logger.info("To run Stage 4: export TOGETHER_API_KEY='your-key' or use --api-key")
    elif not api_key:
        logger.warning("No API key provided - pipeline will fail at Stage 4!")
    
    logger.info("=" * 70)
    logger.info("Phase 1C: Entity Disambiguation (CPU - All 4 Stages)")
    logger.info("=" * 70)
    if args.input:
        logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Start from stage: {args.start_from_stage}")
    logger.info("For GPU-optimized version, use server_scripts/disambiguation_server.py")
    
    # Run pipeline (all 4 stages)
    processor = DisambiguationProcessor(together_api_key=api_key)
    
    try:
        stats = processor.run_pipeline(
            input_file=args.input,
            output_file=args.output,
            start_from_stage=args.start_from_stage,
            stop_at_stage=args.stop_at_stage
        )
        
        logger.info("Pipeline succeeded!")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()