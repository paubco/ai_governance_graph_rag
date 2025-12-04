"""
Phase 1D: Parallel Relation Extraction

Production runner for extracting semantic relations from 76K normalized entities using 
parallel processing with Mistral-7B-Instruct-v0.3.

Features:
- 40 parallel workers with ThreadPoolExecutor
- 2900 RPM rate limiting for Together.ai compliance
- JSONL append-only output for resilience
- Automatic checkpoint and resume capability
- Real-time progress tracking and cost monitoring

Architecture:
- Subject-Predicate-Object triplet extraction (OpenIE-style)
- Two-stage MMR chunk selection (semantic diversity + entity co-occurrence)
- JSON schema enforcement with Pydantic for reliability
- Entity-aware relation extraction using normalized entity list

Author: Pau Barba i Colomer
Date: Dec 4, 2025
Thesis: MSc Data Science, UOC

Usage:
    # Full extraction with 40 workers
    python run_phase1d_extraction.py
    
    # Custom worker count
    python run_phase1d_extraction.py --workers 20
    
    # Resume from checkpoint
    python run_phase1d_extraction.py --resume
    
    # Extract subset for validation
    python run_phase1d_extraction.py --entities 1000
    
    # Enable debug mode (saves prompts/responses)
    python run_phase1d_extraction.py --debug
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from server_scripts.relation_processor_server import ParallelRelationProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('phase1d_extraction.log')
    ]
)
logger = logging.getLogger(__name__)


def load_entities_for_extraction(
    entities_file: Path,
    max_entities: int = None
) -> list:
    """
    Load normalized entities for relation extraction.
    
    Args:
        entities_file: Path to normalized_entities.json
        max_entities: Optional limit (for validation runs)
        
    Returns:
        List of entity dicts with entity_id, name, type, chunk_ids
    """
    logger.info(f"Loading entities from {entities_file}")
    
    with open(entities_file, 'r', encoding='utf-8') as f:
        all_entities = json.load(f)
    
    # Convert to list if dict
    if isinstance(all_entities, dict):
        entities_list = list(all_entities.values())
    else:
        entities_list = all_entities
    
    # Add entity_id to each entity (use name as unique ID)
    for entity in entities_list:
        if 'entity_id' not in entity:
            entity['entity_id'] = entity['name']
    
    # Apply limit if specified (for validation runs)
    if max_entities and len(entities_list) > max_entities:
        step = len(entities_list) // max_entities
        sample = entities_list[::step][:max_entities]
        logger.info(f"Sampled {len(sample)} entities (every {step}th entity) for validation")
    else:
        sample = entities_list[:max_entities] if max_entities else entities_list
        logger.info(f"Loaded {len(sample)} entities for extraction")
    
    return sample


def initialize_extractor(entities_file: Path, cooccurrence_file: Path, config: dict):
    """
    Initialize RAKGRelationExtractor with production configuration.
    
    Args:
        entities_file: Path to normalized_entities.json
        cooccurrence_file: Path to entity co-occurrence matrix
        config: Configuration dict with model parameters
        
    Returns:
        Configured RAKGRelationExtractor instance
    """
    from src.phase1_graph_construction.relation_extractor import RAKGRelationExtractor
    
    logger.info("Initializing RAKGRelationExtractor...")
    logger.info(f"Model: {config['model']}")
    logger.info(f"Chunks per entity: {config['num_chunks']}")
    logger.info(f"Second round threshold: {config['second_round_threshold']}")
    logger.info(f"MMR lambda: {config['mmr_lambda']}")
    logger.info(f"Semantic threshold: {config['semantic_threshold']}")
    
    extractor = RAKGRelationExtractor(
        model_name=config.get('model', 'mistralai/Mistral-7B-Instruct-v0.3'),
        num_chunks=config.get('num_chunks', 6),
        mmr_lambda=config.get('mmr_lambda', 0.65),
        semantic_threshold=config.get('semantic_threshold', 0.85),
        max_tokens=config.get('max_tokens', 20000),
        second_round_threshold=config.get('second_round_threshold', 0.25),
        entity_cooccurrence_file=str(cooccurrence_file),
        normalized_entities_file=str(entities_file),
        debug_mode=config.get('debug_mode', False)
    )
    
    logger.info("✓ Extractor initialized successfully")
    return extractor


def validate_prerequisites(entities_file: Path, chunks_file: Path, cooccurrence_file: Path) -> bool:
    """
    Validate that all required input files exist.
    
    Args:
        entities_file: Path to normalized_entities.json (Phase 1C output)
        chunks_file: Path to chunks_embedded.json (Phase 1A-2 output)
        cooccurrence_file: Path to cooccurrence matrix (Phase 1D-0 output)
        
    Returns:
        True if all files exist, False otherwise
    """
    missing_files = []
    
    if not entities_file.exists():
        missing_files.append(('normalized_entities.json', 'Phase 1C', str(entities_file)))
    
    if not chunks_file.exists():
        missing_files.append(('chunks_embedded.json', 'Phase 1A-2', str(chunks_file)))
    
    if not cooccurrence_file.exists():
        missing_files.append(('cooccurrence_semantic.json', 'Phase 1D-0', str(cooccurrence_file)))
    
    if missing_files:
        print("\n" + "="*80)
        print("❌ MISSING PREREQUISITES")
        print("="*80)
        for filename, phase, path in missing_files:
            print(f"\n  File: {filename}")
            print(f"  Required from: {phase}")
            print(f"  Expected path: {path}")
        print("\n" + "="*80)
        print("Please complete the required phases before running Phase 1D.\n")
        return False
    
    return True


def main():
    """Main extraction execution"""
    parser = argparse.ArgumentParser(
        description='Phase 1D: Parallel relation extraction from normalized entities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full extraction (default: 40 workers)
  python run_phase1d_extraction.py
  
  # Validation run with subset
  python run_phase1d_extraction.py --entities 1000
  
  # Resume from checkpoint
  python run_phase1d_extraction.py --resume
  
  # Custom configuration
  python run_phase1d_extraction.py --workers 20 --debug
        """
    )
    
    parser.add_argument(
        '--workers', 
        type=int, 
        default=40,
        help='Number of parallel workers (default: 40, recommended for 2900 RPM limit)'
    )
    parser.add_argument(
        '--resume', 
        action='store_true',
        help='Resume from checkpoint (skip already completed entities)'
    )
    parser.add_argument(
        '--entities', 
        type=int, 
        default=None,
        help='Limit number of entities to process (for validation runs)'
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode (saves prompts and LLM responses to logs/)'
    )
    
    args = parser.parse_args()
    
    # Define paths
    project_root = Path(__file__).parent.parent
    entities_file = project_root / "data/interim/entities/normalized_entities.json"
    chunks_file = project_root / "data/interim/chunks/chunks_embedded.json"
    cooccurrence_file = project_root / "data/interim/entities/cooccurrence_semantic.json"
    output_dir = project_root / "data/interim/relations"
    
    # Configuration (optimized for Mistral-7B)
    config = {
        'model': 'mistralai/Mistral-7B-Instruct-v0.3',
        'num_chunks': 6,
        'second_round_threshold': 0.25,
        'max_tokens': 20000,
        'mmr_lambda': 0.65,
        'semantic_threshold': 0.85,
        'debug_mode': args.debug
    }
    
    # Header
    print("\n" + "="*80)
    print("PHASE 1D: PARALLEL RELATION EXTRACTION")
    print("="*80)
    print(f"Workers: {args.workers}")
    print(f"Entity limit: {args.entities if args.entities else 'None (full extraction)'}")
    print(f"Resume mode: {'Enabled' if args.resume else 'Disabled'}")
    print(f"Debug mode: {'Enabled' if args.debug else 'Disabled'}")
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"\nOutput directory: {output_dir}")
    print("="*80 + "\n")
    
    # Validate prerequisites
    if not validate_prerequisites(entities_file, chunks_file, cooccurrence_file):
        return 1
    
    try:
        # Load entities
        entities = load_entities_for_extraction(entities_file, max_entities=args.entities)
        print(f"✓ Loaded {len(entities)} entities for extraction\n")
        
        # Load chunks
        print("Loading chunks...")
        with open(chunks_file, 'r') as f:
            chunks = json.load(f)
        if isinstance(chunks, dict):
            chunks = list(chunks.values())
        print(f"✓ Loaded {len(chunks)} chunks\n")
        
        # Initialize extractor
        extractor = initialize_extractor(entities_file, cooccurrence_file, config)
        print()
        
        # Create parallel processor
        processor = ParallelRelationProcessor(
            extractor=extractor,
            all_chunks=chunks,
            num_workers=args.workers,
            checkpoint_freq=100,  # Save progress every 100 entities
            rate_limit_rpm=2900,  # Conservative (3000 RPM limit)
            output_dir=output_dir,
            config=config
        )
        
        # Cost and time estimate
        estimate = processor.estimate_cost_and_time(
            num_entities=len(entities),
            second_round_rate=0.35  # ~35% of entities trigger second batch
        )
        
        print("\n📊 COST & TIME ESTIMATE")
        print("="*80)
        for key, value in estimate.items():
            print(f"  {key}: {value}")
        print("="*80 + "\n")
        
        # Confirmation prompt (skip if resume mode)
        if not args.resume:
            response = input("Proceed with extraction? [y/N]: ")
            if response.lower() != 'y':
                print("Extraction cancelled")
                return 0
        else:
            print("Resume mode enabled - proceeding automatically\n")
        
        # Run extraction
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
            second_batch_count = sum(1 for r in results if r.get('num_batches', 1) > 1)
            
            print(f"Entities processed: {len(results)}")
            print(f"Total relations extracted: {total_relations}")
            print(f"Average relations/entity: {avg_relations:.1f}")
            print(f"Second batches triggered: {second_batch_count} ({100*second_batch_count/len(results):.1f}%)")
            print(f"Total cost: ${total_cost:.4f}")
            print(f"Wall time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
            print(f"Throughput: {len(results)/elapsed*3600:.0f} entities/hour")
            print(f"\nOutput file: {output_file}")
            
            # Sample relations
            if results:
                print(f"\n🔍 Sample relations (first entity):")
                sample = results[0]
                print(f"  Entity: {sample['entity_name']} [{sample['entity_type']}]")
                print(f"  Relations: {len(sample['relations'])}")
                print(f"  Batches: {sample['num_batches']}")
                print(f"  Chunks used: {sample['chunks_used']}")
                for i, rel in enumerate(sample['relations'][:5], 1):
                    print(f"    {i}. ({rel['subject']}, {rel['predicate']}, {rel['object']})")
                if len(sample['relations']) > 5:
                    print(f"    ... and {len(sample['relations'])-5} more")
        
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
        
        # Completion message
        print("✅ Phase 1D relation extraction complete!")
        print(f"\nOutput location: {output_dir}")
        print(f"Main output: relations_output.jsonl")
        print(f"Progress checkpoint: extraction_progress.json")
        print(f"Failed entities: failed_entities.json")
        
        print("\nNext steps:")
        print("  1. Validate relation quality with sample inspection")
        print("  2. Run relation statistics analysis")
        print("  3. Proceed to Phase 2A (entity clustering) if budget allows")
        print("  4. Update week_3.md with final statistics\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Extraction interrupted by user")
        print("Progress has been saved. Use --resume flag to continue.\n")
        return 130
        
    except Exception as e:
        logger.exception("Extraction failed with error")
        print(f"\n❌ EXTRACTION FAILED: {e}")
        print(f"See phase1d_extraction.log for details\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())