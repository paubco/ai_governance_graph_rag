"""
PRODUCTION: Full Relation Extraction

Extracts relations from all 55K entities using proven test infrastructure.

Cost: $16.50 (within $20 budget)
Time: 18-20 hours
Model: Mistral-7B-Instruct-v0.3

Usage:
    # Test first (2 minutes, $0.03)
    python run_relation_extraction.py --test
    
    # Then run full production (19 hours, $16.50)
    python run_relation_extraction.py
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from server_scripts.relation_processor_server import ParallelRelationProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/relation_extraction_production.log')
    ]
)
logger = logging.getLogger(__name__)


def load_entities(entities_file: Path, num_entities: int = None) -> list:
    """Load entities from JSON"""
    logger.info(f"Loading entities from {entities_file}")
    
    with open(entities_file, 'r', encoding='utf-8') as f:
        all_entities = json.load(f)
    
    if isinstance(all_entities, dict):
        entities_list = list(all_entities.values())
    else:
        entities_list = all_entities
    
    # Add entity_id
    for entity in entities_list:
        if 'entity_id' not in entity:
            entity['entity_id'] = entity['name']
    
    logger.info(f"  Loaded {len(entities_list):,} total entities")
    
    # Sample if specified
    if num_entities:
        entities_list = entities_list[:num_entities]
        logger.info(f"  Using first {len(entities_list):,} entities")
    
    return entities_list


def create_extractor(entities_file: Path, cooccurrence_file: Path, config: dict):
    """Initialize RAKGRelationExtractor"""
    from src.phase1_graph_construction.relation_extractor import RAKGRelationExtractor
    
    logger.info("Initializing RAKGRelationExtractor...")
    
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
    parser = argparse.ArgumentParser(description='Relation extraction for AI Governance KG')
    parser.add_argument('--test', action='store_true', help='Test mode: 100 entities (2 min, $0.03)')
    parser.add_argument('--entities', type=int, help='Number of entities to process (default: all 55K)')
    parser.add_argument('--workers', type=int, default=40, help='Number of parallel workers')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Paths
    project_root = Path(__file__).parent.parent
    entities_file = project_root / "data/interim/entities/normalized_entities.json"
    chunks_file = project_root / "data/interim/chunks/chunks_embedded.json"
    cooccurrence_file = project_root / "data/interim/entities/cooccurrence_semantic.json"
    output_dir = project_root / "data/interim/relations_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = {
        'model': 'mistralai/Mistral-7B-Instruct-v0.3',
        'num_chunks': 6,
        'second_round_threshold': 0.25,
        'max_tokens': 20000,
        'mmr_lambda': 0.65,
        'semantic_threshold': 0.85,
        'debug_mode': args.debug
    }
    
    # Determine mode
    if args.test:
        num_entities = 100
        mode = "TEST MODE"
    elif args.entities:
        num_entities = args.entities
        mode = f"CUSTOM ({num_entities:,} entities)"
    else:
        num_entities = None
        mode = "FULL PRODUCTION (all 55K entities)"
    
    # Banner
    print("\n" + "="*80)
    print("🚀 RELATION EXTRACTION")
    print("="*80)
    print(f"Mode: {mode}")
    print(f"Workers: {args.workers}")
    print(f"Output: {output_dir}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Validate files
    for path, name in [(entities_file, "Entities"), (chunks_file, "Chunks"), (cooccurrence_file, "Co-occurrence")]:
        if not path.exists():
            print(f"❌ ERROR: {name} file not found: {path}")
            return 1
    
    try:
        # Load data
        entities = load_entities(entities_file, num_entities=num_entities)
        print(f"✓ Loaded {len(entities):,} entities\n")
        
        with open(chunks_file, 'r') as f:
            chunks = json.load(f)
        if isinstance(chunks, dict):
            chunks = list(chunks.values())
        print(f"✓ Loaded {len(chunks):,} chunks\n")
        
        # Create extractor and processor
        extractor = create_extractor(entities_file, cooccurrence_file, config)
        
        processor = ParallelRelationProcessor(
            extractor=extractor,
            all_chunks=chunks,
            num_workers=args.workers,
            checkpoint_freq=100,
            rate_limit_rpm=2900,
            output_dir=output_dir,
            config=config
        )
        
        # Estimate
        estimate = processor.estimate_cost_and_time(
            num_entities=len(entities),
            second_round_rate=0.04
        )
        
        print("\n📊 Estimate:")
        print(f"  Entities: {estimate['num_entities']:,}")
        print(f"  Cost: ${estimate['estimated_cost_usd']}")
        print(f"  Time: {estimate['parallel_time_hours']} hours")
        print()
        
        # Confirm
        if not args.test and not args.resume:
            if estimate['estimated_cost_usd'] > 20:
                print(f"⚠️  WARNING: Cost ${estimate['estimated_cost_usd']} exceeds $20 budget!")
            response = input("Proceed? [y/N]: ")
            if response.lower() != 'y':
                print("Cancelled")
                return 0
        
        # Run
        print("\n" + "="*80)
        print("Starting extraction...")
        print("="*80 + "\n")
        
        start = datetime.now()
        processor.process_all_entities(entities=entities, test_mode=False)
        elapsed = (datetime.now() - start).total_seconds()
        
        # Results
        print("\n" + "="*80)
        print("📊 RESULTS")
        print("="*80)
        
        output_file = output_dir / "relations_output.jsonl"
        if output_file.exists():
            with open(output_file, 'r') as f:
                results = [json.loads(line) for line in f if line.strip()]
            
            total_relations = sum(len(r['relations']) for r in results)
            total_cost = sum(r['cost'] for r in results)
            
            print(f"Entities: {len(results):,}")
            print(f"Relations: {total_relations:,}")
            print(f"Avg/entity: {total_relations/len(results):.1f}")
            print(f"Cost: ${total_cost:.2f}")
            print(f"Time: {elapsed/3600:.1f} hours")
            print(f"Output: {output_file}")
        
        print("="*80 + "\n")
        print("✅ Complete!\n")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted - progress saved, use --resume to continue")
        return 130
    except Exception as e:
        logger.exception("Failed")
        print(f"\n❌ ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())