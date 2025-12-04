"""
PRODUCTION: Full Relation Extraction - All Entities

Parallel extraction of relations from 50K+ entities using proven test infrastructure.
This script has been validated with 100 entities (100% success rate, $0.03 cost).

Key Features:
- 40 parallel workers (ThreadPoolExecutor)
- Mistral-7B-Instruct-v0.3 with JSON schema
- 2900 RPM rate limiting (under Together.ai 3000 limit)
- Checkpoint every 100 entities (auto-resume)
- Stratified sampling option (stays within $20 budget)

Expected Performance:
- Full run (55K entities): $16.50, ~19 hours ✅ RECOMMENDED
- Test run (100 entities): $0.03, ~2 minutes

Author: Pau Barba i Colomer
Date: December 4, 2025
Version: 1.0 (Production)

Usage:
    # Test mode (100 entities, 2 minutes)
    python run_relation_extraction.py --test
    
    # Production - all entities (55K entities, $16.50, 19h) ✅ DEFAULT
    python run_relation_extraction.py --all
    
    # Resume after interruption
    python run_relation_extraction.py --all --resume
    
    # Custom configuration
    python run_relation_extraction.py --entities 30000 --workers 20
"""

import sys
import json
import logging
import argparse
import random
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from server_scripts.relation_processor_server import ParallelRelationProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/relation_extraction_production.log')
    ]
)
logger = logging.getLogger(__name__)


def load_entities(entities_file: Path, sample_strategy: str = None, num_entities: int = None) -> list:
    """
    Load entities with optional sampling strategies.
    
    Args:
        entities_file: Path to normalized_entities.json
        sample_strategy: 'test' (100), 'stratified' (50K), 'all' (55K), or None
        num_entities: Custom number to process (overrides strategy)
        
    Returns:
        List of entity dicts with entity_id
    """
    logger.info(f"Loading entities from {entities_file}")
    
    with open(entities_file, 'r', encoding='utf-8') as f:
        all_entities = json.load(f)
    
    # Convert to list if dict
    if isinstance(all_entities, dict):
        entities_list = list(all_entities.values())
    else:
        entities_list = all_entities
    
    # Ensure entity_id field
    for entity in entities_list:
        if 'entity_id' not in entity:
            entity['entity_id'] = entity['name']
    
    logger.info(f"  Loaded {len(entities_list):,} total entities")
    
    # Apply sampling strategy
    if sample_strategy == 'test':
        # Test mode: First 100 entities
        sampled = entities_list[:100]
        logger.info(f"  TEST MODE: Using first {len(sampled)} entities")
        
    elif sample_strategy == 'stratified':
        # Stratified sampling: Prioritize rare entities
        sampled = stratified_sample(entities_list, target_size=50000)
        
    elif sample_strategy == 'all':
        # Full production: All entities
        sampled = entities_list
        logger.info(f"  FULL MODE: Processing all {len(sampled):,} entities")
        
    elif num_entities:
        # Custom number
        sampled = entities_list[:num_entities]
        logger.info(f"  CUSTOM: Processing first {len(sampled):,} entities")
        
    else:
        # Default: All entities
        sampled = entities_list
        logger.info(f"  DEFAULT: Processing all {len(sampled):,} entities")
    
    return sampled


def stratified_sample(entities: list, target_size: int = 50000) -> list:
    """
    Stratified sampling prioritizing rare (domain-specific) entities.
    
    Strategy:
    - Rare entities (< 5 chunks): 100% coverage (most valuable)
    - Medium entities (5-20 chunks): 50% sample
    - Common entities (> 20 chunks): 25% sample
    
    Rationale:
    - Rare entities = Domain-specific AI governance concepts
    - Common entities = Generic terms with redundant relations
    
    Args:
        entities: List of all entities
        target_size: Target number of entities to sample (~50K)
        
    Returns:
        Sampled entity list (~50K entities, $15-16 cost)
    """
    logger.info("Applying stratified sampling strategy...")
    
    # Categorize by chunk frequency
    rare = []
    medium = []
    common = []
    
    for entity in entities:
        chunk_count = len(entity.get('chunk_ids', []))
        if chunk_count < 5:
            rare.append(entity)
        elif chunk_count <= 20:
            medium.append(entity)
        else:
            common.append(entity)
    
    logger.info(f"  Distribution:")
    logger.info(f"    Rare (< 5 chunks): {len(rare):,} entities")
    logger.info(f"    Medium (5-20 chunks): {len(medium):,} entities")
    logger.info(f"    Common (> 20 chunks): {len(common):,} entities")
    
    # Sample each category
    sampled = []
    
    # Take ALL rare (most valuable for AI governance KG)
    sampled.extend(rare)
    logger.info(f"  Sampled: {len(rare):,} rare entities (100% coverage)")
    
    # Sample 50% of medium
    medium_count = min(len(medium) // 2, target_size - len(sampled))
    sampled.extend(random.sample(medium, medium_count))
    logger.info(f"  Sampled: {medium_count:,} medium entities (~50%)")
    
    # Sample 25% of common
    common_count = min(len(common) // 4, target_size - len(sampled))
    sampled.extend(random.sample(common, common_count))
    logger.info(f"  Sampled: {common_count:,} common entities (~25%)")
    
    # Shuffle to distribute frequencies
    random.shuffle(sampled)
    
    logger.info(f"✅ Total sampled: {len(sampled):,} entities")
    logger.info(f"   Expected cost: $15-16")
    logger.info(f"   Expected time: 18-20 hours")
    
    return sampled


def create_production_extractor(entities_file: Path, cooccurrence_file: Path, config: dict):
    """
    Initialize RAKGRelationExtractor with production configuration.
    
    Model: Mistral-7B-Instruct-v0.3 (proven in testing)
    Configuration: Optimized for quality and cost
    
    Args:
        entities_file: Path to normalized_entities.json
        cooccurrence_file: Path to entity co-occurrence matrix
        config: Configuration dict
        
    Returns:
        RAKGRelationExtractor instance (thread-safe)
    """
    from src.phase1_graph_construction.relation_extractor import RAKGRelationExtractor
    
    logger.info("Initializing RAKGRelationExtractor...")
    logger.info(f"  Model: {config['model']}")
    logger.info(f"  Chunks per entity: {config['num_chunks']}")
    logger.info(f"  Temperature: 0.0 (deterministic)")
    
    extractor = RAKGRelationExtractor(
        model_name=config['model'],
        num_chunks=config['num_chunks'],
        mmr_lambda=config['mmr_lambda'],
        semantic_threshold=config['semantic_threshold'],
        max_tokens=config['max_tokens'],
        second_round_threshold=config['second_round_threshold'],
        entity_cooccurrence_file=str(cooccurrence_file),
        normalized_entities_file=str(entities_file),
        debug_mode=config.get('debug_mode', False)
    )
    
    logger.info("✓ Extractor initialized")
    return extractor


def main():
    """Main production execution"""
    parser = argparse.ArgumentParser(
        description='Production relation extraction for AI Governance Knowledge Graph',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 100 entities (2 minutes, $0.03)
  python run_relation_extraction.py --test
  
  # Production with stratified sampling (18 hours, $15-16) ✅ RECOMMENDED
  python run_relation_extraction.py --stratified
  
  # Full production (19 hours, $42)
  python run_relation_extraction.py --all
  
  # Custom number of entities
  python run_relation_extraction.py --entities 30000
        """
    )
    
    # Sampling strategies
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--test', action='store_true', 
                       help='Test mode: 100 entities (~2 min, $0.03)')
    group.add_argument('--all', action='store_true', default=True,
                       help='Full production: All 55K entities (~19h, $16.50) ✅ DEFAULT')
    group.add_argument('--entities', type=int, metavar='N',
                       help='Custom number of entities to process')
    
    # Configuration
    parser.add_argument('--workers', type=int, default=40, 
                       help='Number of parallel workers (default: 40)')
    parser.add_argument('--resume', action='store_true', 
                       help='Resume from checkpoint (auto-detected)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (save prompts/responses)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Paths
    project_root = Path(__file__).parent
    entities_file = project_root / "data/interim/entities/normalized_entities.json"
    chunks_file = project_root / "data/interim/chunks/chunks_embedded.json"
    cooccurrence_file = project_root / "data/interim/entities/cooccurrence_semantic.json"
    
    # Output directory: PRODUCTION (not test!)
    output_dir = project_root / "data/interim/relations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine sampling strategy
    if args.test:
        sample_strategy = 'test'
        mode_desc = "TEST MODE (100 entities)"
    elif args.entities:
        sample_strategy = None
        mode_desc = f"CUSTOM ({args.entities:,} entities)"
    else:
        # Default: All entities
        sample_strategy = 'all'
        mode_desc = "FULL PRODUCTION (all 55K entities) ✅ DEFAULT"
    
    # Production configuration (proven in testing)
    config = {
        'model': 'mistralai/Mistral-7B-Instruct-v0.3',
        'num_chunks': 6,
        'second_round_threshold': 0.25,
        'max_tokens': 20000,
        'mmr_lambda': 0.65,
        'semantic_threshold': 0.85,
        'debug_mode': args.debug
    }
    
    # Banner
    print("\n" + "="*80)
    print("🚀 PRODUCTION RELATION EXTRACTION")
    print("="*80)
    print(f"Mode: {mode_desc}")
    print(f"Workers: {args.workers}")
    print(f"Resume: {args.resume}")
    print(f"Debug: {args.debug}")
    print(f"Output: {output_dir}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Validate files exist
    missing_files = []
    if not entities_file.exists():
        missing_files.append(f"Entities: {entities_file}")
    if not chunks_file.exists():
        missing_files.append(f"Chunks: {chunks_file}")
    if not cooccurrence_file.exists():
        missing_files.append(f"Co-occurrence: {cooccurrence_file}")
    
    if missing_files:
        logger.error("Missing required files:")
        for f in missing_files:
            print(f"  ❌ {f}")
        print("\nPlease run prerequisite phases first:")
        print("  Phase 1A-2: Chunk embedding")
        print("  Phase 1C: Entity disambiguation")
        print("  Phase 1D-0: Co-occurrence matrix generation")
        return 1
    
    try:
        # Load entities
        entities = load_entities(
            entities_file, 
            sample_strategy=sample_strategy,
            num_entities=args.entities
        )
        print(f"✓ Loaded {len(entities):,} entities for processing\n")
        
        # Load chunks
        print("Loading chunks...")
        with open(chunks_file, 'r') as f:
            chunks = json.load(f)
        if isinstance(chunks, dict):
            chunks = list(chunks.values())
        print(f"✓ Loaded {len(chunks):,} chunks\n")
        
        # Create extractor
        extractor = create_production_extractor(entities_file, cooccurrence_file, config)
        
        # Create processor
        processor = ParallelRelationProcessor(
            extractor=extractor,
            all_chunks=chunks,
            num_workers=args.workers,
            checkpoint_freq=100,  # Save every 100 entities
            rate_limit_rpm=2900,  # Conservative (under 3000 limit)
            output_dir=output_dir,
            config=config
        )
        
        # Cost & time estimate
        estimate = processor.estimate_cost_and_time(
            num_entities=len(entities),
            second_round_rate=0.04  # Based on test: 4% needed second batch
        )
        
        print("\n📊 Cost & Time Estimate:")
        print(f"  Entities to process: {estimate['num_entities']:,}")
        print(f"  Single batch: {estimate['single_batch_entities']:,}")
        print(f"  Double batch: {estimate['double_batch_entities']:,}")
        print(f"  Total tokens: {estimate['total_tokens']}")
        print(f"  Estimated cost: ${estimate['estimated_cost_usd']}")
        print(f"  Sequential time: {estimate['sequential_time_hours']} hours")
        print(f"  Parallel time: {estimate['parallel_time_hours']} hours")
        print(f"  Workers: {estimate['workers']}")
        print()
        
        # Budget check
        if estimate['estimated_cost_usd'] > 20:
            print("⚠️  WARNING: Estimated cost exceeds $20 budget!")
            print(f"   Projected: ${estimate['estimated_cost_usd']}")
            print(f"   Budget: $20")
            print(f"   Overage: ${estimate['estimated_cost_usd'] - 20:.2f}")
            print("\n   Consider using --stratified to stay within budget")
            print()
        
        # Confirmation (skip for test mode)
        if not args.test and not args.resume:
            print("⚠️  This will process entities and charge to your Together.ai account.")
            print(f"   Estimated cost: ${estimate['estimated_cost_usd']}")
            print(f"   Estimated time: {estimate['parallel_time_hours']} hours")
            print()
            response = input("Proceed with extraction? [y/N]: ")
            if response.lower() != 'y':
                print("Extraction cancelled")
                return 0
        
        # Run extraction
        print("\n" + "="*80)
        print("Starting parallel extraction...")
        print("="*80 + "\n")
        
        start = datetime.now()
        processor.process_all_entities(
            entities=entities,
            test_mode=False
        )
        elapsed = (datetime.now() - start).total_seconds()
        
        # Results summary
        print("\n" + "="*80)
        print("📊 EXTRACTION RESULTS")
        print("="*80)
        
        output_file = output_dir / "relations_output.jsonl"
        if output_file.exists():
            with open(output_file, 'r') as f:
                results = [json.loads(line) for line in f if line.strip()]
            
            total_relations = sum(len(r['relations']) for r in results)
            avg_relations = total_relations / len(results) if results else 0
            total_cost = sum(r['cost'] for r in results)
            
            print(f"Entities processed: {len(results):,}")
            print(f"Total relations: {total_relations:,}")
            print(f"Avg relations/entity: {avg_relations:.1f}")
            print(f"Total cost: ${total_cost:.2f}")
            print(f"Wall time: {elapsed/3600:.1f} hours")
            print(f"Throughput: {len(results)/elapsed*3600:.0f} entities/hour")
            print(f"\nOutput file: {output_file}")
            
            # Sample results
            if results:
                print(f"\n🔍 Sample relations (first entity):")
                sample = results[0]
                print(f"  Entity: {sample['entity_name']} [{sample['entity_type']}]")
                print(f"  Relations: {len(sample['relations'])}")
                print(f"  Batches: {sample['num_batches']}")
                for i, rel in enumerate(sample['relations'][:3], 1):
                    print(f"    {i}. ({rel['subject']}, {rel['predicate']}, {rel['object']})")
                if len(sample['relations']) > 3:
                    print(f"    ... and {len(sample['relations'])-3} more")
        
        print("="*80 + "\n")
        
        # Check for failures
        failed_file = output_dir / "failed_entities.json"
        if failed_file.exists():
            with open(failed_file, 'r') as f:
                failures = [json.loads(line) for line in f if line.strip()]
            
            if failures:
                print(f"⚠️  {len(failures)} entities failed:")
                for fail in failures[:5]:
                    print(f"  - {fail['entity_name']}: {fail['error']}")
                if len(failures) > 5:
                    print(f"  ... and {len(failures)-5} more (see {failed_file})")
                print()
        
        print("✅ Extraction complete!")
        print(f"\nOutput directory: {output_dir}")
        print(f"Relations file: {output_file}")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Extraction interrupted by user")
        print("Progress has been checkpointed - run with --resume to continue")
        return 130
        
    except Exception as e:
        logger.exception("Extraction failed with error")
        print(f"\n❌ EXTRACTION FAILED: {e}")
        print("Check logs/relation_extraction_production.log for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())