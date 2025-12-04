"""
Test Script: 100-Entity Parallel Extraction

Validates parallel processing system before full 55K entity run.
Tests: threading, checkpointing, rate limiting, resume capability.

Author: Pau Barba i Colomer
Date: Dec 4, 2025

Usage:
    python test_parallel_100.py [--workers N] [--resume]
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
        logging.FileHandler('test_parallel_100.log')
    ]
)
logger = logging.getLogger(__name__)


def load_entities_for_test(
    entities_file: Path,
    num_entities: int = 100
) -> list:
    """
    Load first N entities for testing.
    
    Args:
        entities_file: Path to normalized_entities.json
        num_entities: Number to load (default 100)
        
    Returns:
        List of entity dicts
    """
    logger.info(f"Loading {num_entities} entities from {entities_file}")
    
    with open(entities_file, 'r', encoding='utf-8') as f:
        all_entities = json.load(f)
    
    # Convert dict to list if needed
    if isinstance(all_entities, dict):
        entities_list = [
            {
                'entity_id': k,
                'name': v['name'],
                'type': v['type'],
                **v  # Include all other fields
            }
            for k, v in all_entities.items()
        ]
    else:
        entities_list = all_entities
    
    # Get diverse sample (not just first 100)
    # Mix of high-mention and low-mention entities
    if len(entities_list) > num_entities:
        step = len(entities_list) // num_entities
        sample = entities_list[::step][:num_entities]
        logger.info(f"Sampled {len(sample)} entities (every {step}th entity)")
    else:
        sample = entities_list[:num_entities]
        logger.info(f"Using first {len(sample)} entities")
    
    return sample


def create_test_extractor(chunks_file: Path, config: dict):
    """
    Create RAKGRelationExtractor instance for testing.
    
    Args:
        chunks_file: Path to chunks_embedded.json
        config: Configuration dict
        
    Returns:
        RAKGRelationExtractor instance
    """
    from src.phase1_graph_construction.relation_extractor import RAKGRelationExtractor
    
    logger.info("Initializing RAKGRelationExtractor...")
    
    extractor = RAKGRelationExtractor(
        chunks_file=chunks_file,
        model=config.get('model', 'mistralai/Mistral-7B-Instruct-v0.3'),
        num_chunks=config.get('num_chunks', 6),
        second_round_threshold=config.get('second_round_threshold', 0.30),
        mmr_lambda=config.get('mmr_lambda', 0.65)
    )
    
    logger.info("✓ Extractor initialized")
    return extractor


def main():
    """Main test execution"""
    parser = argparse.ArgumentParser(description='Test parallel extraction with 100 entities')
    parser.add_argument('--workers', type=int, default=40, help='Number of parallel workers')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--entities', type=int, default=100, help='Number of entities to test')
    args = parser.parse_args()
    
    # Paths (adjust to your project structure)
    project_root = Path(__file__).parent
    entities_file = "data/interim/entities/normalized_entities.json"
    chunks_file = "data/interim/chunks/chunks_embedded.json"
    output_dir = "data/interim/relations_test"
    
    # Configuration
    config = {
        'model': 'mistralai/Mistral-7B-Instruct-v0.3',
        'num_chunks': 6,  # Chunks per batch
        'second_round_threshold': 0.30,  # Centroid distance for second batch
        'mmr_lambda': 0.65,
        'semantic_threshold': 0.85
    }
    
    print("\n" + "="*80)
    print("🧪 PARALLEL EXTRACTION TEST - 100 ENTITIES")
    print("="*80)
    print(f"Workers: {args.workers}")
    print(f"Entities: {args.entities}")
    print(f"Resume: {args.resume}")
    print(f"Config: {config}")
    print(f"Output: {output_dir}")
    print("="*80 + "\n")
    
    # Validate files exist
    if not entities_file.exists():
        logger.error(f"Entities file not found: {entities_file}")
        print(f"\n❌ ERROR: {entities_file} not found")
        print("Please run Phase 1C first to generate normalized_entities.json")
        return 1
    
    if not chunks_file.exists():
        logger.error(f"Chunks file not found: {chunks_file}")
        print(f"\n❌ ERROR: {chunks_file} not found")
        print("Please run Phase 1A-2 first to generate chunks_embedded.json")
        return 1
    
    try:
        # Load entities
        entities = load_entities_for_test(entities_file, num_entities=args.entities)
        print(f"✓ Loaded {len(entities)} entities for testing\n")
        
        # Create extractor
        extractor = create_test_extractor(chunks_file, config)
        
        # Create processor
        processor = ParallelRelationProcessor(
            extractor=extractor,
            num_workers=args.workers,
            checkpoint_freq=10,  # More frequent for testing
            rate_limit_rpm=2900,
            output_dir=output_dir,
            config=config
        )
        
        # Cost estimate
        estimate = processor.estimate_cost_and_time(
            num_entities=len(entities),
            second_round_rate=0.30
        )
        print("\n📊 Cost & Time Estimate:")
        for key, value in estimate.items():
            print(f"  {key}: {value}")
        print()
        
        # Confirmation
        if not args.resume:
            response = input("Proceed with test? [y/N]: ")
            if response.lower() != 'y':
                print("Test cancelled")
                return 0
        
        # Run extraction
        start = datetime.now()
        processor.process_all_entities(
            entities=entities,
            test_mode=False
        )
        elapsed = (datetime.now() - start).total_seconds()
        
        # Results summary
        print("\n" + "="*80)
        print("📊 TEST RESULTS")
        print("="*80)
        
        output_file = output_dir / "relations_output.jsonl"
        if output_file.exists():
            with open(output_file, 'r') as f:
                results = [json.loads(line) for line in f if line.strip()]
            
            total_relations = sum(len(r['relations']) for r in results)
            avg_relations = total_relations / len(results) if results else 0
            total_cost = sum(r['cost'] for r in results)
            
            print(f"Entities processed: {len(results)}")
            print(f"Total relations: {total_relations}")
            print(f"Avg relations/entity: {avg_relations:.1f}")
            print(f"Total cost: ${total_cost:.4f}")
            print(f"Wall time: {elapsed/60:.1f} minutes")
            print(f"Throughput: {len(results)/elapsed*3600:.0f} entities/hour")
            print(f"\nOutput file: {output_file}")
            
            # Sample results
            print(f"\n📝 Sample relations (first entity):")
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
        
        print("✅ Test complete! Check output in:", output_dir)
        print("\nNext steps:")
        print("  1. Review sample relations for quality")
        print("  2. Check failed_entities.json for any errors")
        print("  3. Adjust config if needed (threshold, num_chunks)")
        print("  4. If satisfied, run full extraction on 55K entities\n")
        
        return 0
        
    except Exception as e:
        logger.exception("Test failed with error")
        print(f"\n❌ TEST FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
