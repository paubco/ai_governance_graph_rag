"""
Phase 1D Relation Extraction - Consolidated Test Script

Merges best of:
- Original test approach (PREFERRED_ENTITIES, build_entity_map)
- Enhanced extractor with bug fixes (prompt validation, error handling)
- Parameter tuning capabilities

Usage:
    # Basic test (8 entities)
    python scripts/test_phase1d_consolidated.py
    
    # Parameter tuning (2 entities, multiple configs)
    python scripts/test_phase1d_consolidated.py --tune-params
    
    # Custom parameters
    python scripts/test_phase1d_consolidated.py --threshold 0.80 --lambda 0.60 --chunks 15
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase1_graph_construction.relation_extractor import RAKGRelationExtractor
from src.phase1_graph_construction.entity_disambiguator import build_entity_map
from dotenv import load_dotenv
import os

# Load environment
load_dotenv(PROJECT_ROOT / '.env')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/phase1d_test_consolidated.log')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# TEST ENTITIES (From actual normalized_entities.json)
# ============================================================================

PREFERRED_ENTITIES = [
    ("AI System", "Technology"),           # 4469 chunks
    ("European Union", "Organization"),    # 1063 chunks
    ("Chat-GPT", "AI System"),             # 840 chunks
    ("EU AI Act", "Regulation"),           # 474 chunks
    ("transparency", "Concept"),           # 338 chunks
    ("US", "Country"),                     # 304 chunks
    ("human rights", "Legal Concept"),     # 300 chunks
    ("accountability", "Concept"),         # 192 chunks
]


def load_data(entities_file: str, chunks_file: str):
    """Load entities and chunks.
    
    Expected formats:
    - Entities: {entity_id: entity_obj, ...} or {"entities": [...]}
    - Chunks: {chunk_id: chunk_obj, ...}
    """
    logger.info(f"Loading entities from {entities_file}")
    with open(entities_file, 'r', encoding='utf-8') as f:
        entities_data = json.load(f)
    
    # Handle dict with 'entities' key or dict with entity_ids as keys
    if isinstance(entities_data, dict) and 'entities' in entities_data:
        entities = entities_data['entities']
    elif isinstance(entities_data, dict):
        entities = list(entities_data.values())
    elif isinstance(entities_data, list):
        entities = entities_data
    else:
        raise ValueError(f"Unexpected entities format: {type(entities_data).__name__}")
    
    logger.info(f"✓ Loaded {len(entities)} entities")
    
    logger.info(f"Loading chunks from {chunks_file}")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    # Convert dict of {chunk_id: chunk_obj} to list
    if isinstance(chunks_data, dict):
        chunks = list(chunks_data.values())
    elif isinstance(chunks_data, list):
        chunks = chunks_data
    else:
        raise ValueError(f"Expected dict or list for chunks, got {type(chunks_data).__name__}")
    
    logger.info(f"✓ Loaded {len(chunks)} chunks")
    
    return entities, chunks


def get_test_entities(entities: List[Dict], n: int = 8) -> List[Dict]:
    """
    Get test entities using PREFERRED_ENTITIES with fallback
    
    Strategy:
    1. Try preferred entities from list
    2. Fallback to top by chunk count
    """
    entity_map = build_entity_map(entities)
    
    test_entities = []
    for key in PREFERRED_ENTITIES:
        if key in entity_map:
            entity = entity_map[key]
            test_entities.append(entity)
            if len(test_entities) >= n:
                break
    
    # Fallback to top entities
    if len(test_entities) < n:
        logger.warning(f"Only found {len(test_entities)} preferred entities, using fallback")
        sorted_entities = sorted(
            entities,
            key=lambda e: len(e.get('chunk_ids', [])),
            reverse=True
        )
        
        for entity in sorted_entities:
            key = (entity['name'], entity['type'])
            if key not in [(e['name'], e['type']) for e in test_entities]:
                chunk_count = len(entity.get('chunk_ids', []))
                if chunk_count >= 100:
                    test_entities.append(entity)
                    if len(test_entities) >= n:
                        break
    
    logger.info(f"\n✓ Selected {len(test_entities)} test entities:")
    for i, entity in enumerate(test_entities, 1):
        chunk_count = len(entity.get('chunk_ids', []))
        logger.info(f"  {i}. {entity['name']} [{entity['type']}] - {chunk_count} chunks")
    
    return test_entities


def test_basic(entities: List[Dict], 
               chunks: List[Dict],
               threshold: float = 0.85,
               mmr_lambda: float = 0.55,
               num_chunks: int = 20,
               save_prompts: bool = True):
    """
    Basic test: Extract relations for test entities
    """
    api_key = os.getenv('TOGETHER_API_KEY')
    if not api_key:
        logger.error("TOGETHER_API_KEY not set!")
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("BASIC TEST: Relation Extraction with Enhanced Extractor")
    logger.info("=" * 80)
    logger.info(f"Parameters: threshold={threshold}, lambda={mmr_lambda}, chunks={num_chunks}")
    logger.info(f"Prompt logging: {'ENABLED' if save_prompts else 'DISABLED'}")
    
    # Initialize extractor with entity co-occurrence
    extractor = RAKGRelationExtractor(
        model_name="Qwen/Qwen2.5-7B-Instruct-Turbo",
        api_key=api_key,
        semantic_threshold=threshold,
        mmr_lambda=mmr_lambda,
        num_chunks=num_chunks,
        entity_cooccurrence_file="data/interim/entities/entity_cooccurrence.json",
        normalized_entities_file="data/interim/entities/normalized_entities.json"
    )
    
    # Get test entities
    test_entities = get_test_entities(entities, n=8)
    
    all_relations = []
    stats = {
        'total': len(test_entities),
        'success': 0,
        'failed': 0,
        'total_relations': 0
    }
    
    logger.info("\n" + "-" * 80)
    logger.info("Starting extraction...")
    logger.info("-" * 80)
    
    for i, entity in enumerate(test_entities, 1):
        logger.info(f"\nEntity {i}/{len(test_entities)}: {entity['name']} [{entity['type']}]")
        logger.info(f"  Chunks available: {len(entity.get('chunk_ids', []))}")
        
        try:
            # Extract with enhanced extractor (includes all bug fixes)
            relations = extractor.extract_relations_for_entity(
                entity,
                chunks,
                save_prompt=save_prompts
            )
            
            logger.info(f"  ✓ Extracted {len(relations)} relations")
            all_relations.extend(relations)
            stats['success'] += 1
            stats['total_relations'] += len(relations)
            
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            stats['failed'] += 1
    
    # Save results
    output_file = Path('data/interim/relations/test_relations_consolidated.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'test_type': 'basic',
                'parameters': {
                    'threshold': threshold,
                    'mmr_lambda': mmr_lambda,
                    'num_chunks': num_chunks
                }
            },
            'stats': stats,
            'relations': all_relations
        }, f, indent=2, ensure_ascii=False)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ TEST COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Success: {stats['success']}/{stats['total']}")
    logger.info(f"Total relations: {stats['total_relations']}")
    logger.info(f"Avg per entity: {stats['total_relations']/stats['success']:.1f}" if stats['success'] > 0 else "N/A")
    logger.info(f"Output: {output_file}")
    if save_prompts:
        logger.info(f"Prompts: logs/phase1d_prompts/")
    logger.info("=" * 80)


def test_parameter_tuning(entities: List[Dict], chunks: List[Dict]):
    """
    Parameter tuning: Test multiple configurations on 2 entities
    """
    api_key = os.getenv('TOGETHER_API_KEY')
    if not api_key:
        logger.error("TOGETHER_API_KEY not set!")
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("PARAMETER TUNING: Multiple Configurations")
    logger.info("=" * 80)
    
    # Get 2 test entities
    test_entities = get_test_entities(entities, n=2)
    
    # Parameter combinations
    param_sets = [
        {'threshold': 0.85, 'mmr_lambda': 0.55, 'num_chunks': 20},
        {'threshold': 0.80, 'mmr_lambda': 0.55, 'num_chunks': 20},
        {'threshold': 0.85, 'mmr_lambda': 0.50, 'num_chunks': 20},
        {'threshold': 0.85, 'mmr_lambda': 0.60, 'num_chunks': 20},
        {'threshold': 0.85, 'mmr_lambda': 0.55, 'num_chunks': 15},
        {'threshold': 0.85, 'mmr_lambda': 0.55, 'num_chunks': 25},
    ]
    
    results = []
    
    for i, params in enumerate(param_sets, 1):
        logger.info(f"\n--- Configuration {i}/6 ---")
        logger.info(f"threshold={params['threshold']}, lambda={params['mmr_lambda']}, chunks={params['num_chunks']}")
        
        extractor = RAKGRelationExtractor(
            model_name="Qwen/Qwen2.5-7B-Instruct-Turbo",
            api_key=api_key,
            semantic_threshold=params['threshold'],
            mmr_lambda=params['mmr_lambda'],
            num_chunks=params['num_chunks'],
            entity_cooccurrence_file="data/interim/entities/entity_cooccurrence.json",
            normalized_entities_file="data/interim/entities/normalized_entities.json"
        )
        
        config_relations = 0
        for entity in test_entities:
            try:
                relations = extractor.extract_relations_for_entity(
                    entity,
                    chunks,
                    save_prompt=False
                )
                config_relations += len(relations)
                logger.info(f"  {entity['name']}: {len(relations)} relations")
            except Exception as e:
                logger.error(f"  {entity['name']}: Failed - {e}")
        
        results.append({
            'config': params,
            'relations': config_relations
        })
        logger.info(f"  Total: {config_relations} relations")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PARAMETER TUNING RESULTS")
    logger.info("=" * 80)
    for i, result in enumerate(results, 1):
        logger.info(f"{i}. {result['config']} → {result['relations']} relations")
    
    best = max(results, key=lambda x: x['relations'])
    logger.info(f"\n✓ Best config: {best['config']} with {best['relations']} relations")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Consolidated Phase 1D test script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--entities', default='data/interim/entities/normalized_entities.json')
    parser.add_argument('--chunks-file', dest='chunks', default='data/interim/chunks/chunks_embedded.json')
    parser.add_argument('--tune-params', action='store_true', help='Run parameter tuning')
    parser.add_argument('--threshold', type=float, default=0.85)
    parser.add_argument('--lambda', dest='mmr_lambda', type=float, default=0.55)
    parser.add_argument('--num-chunks', dest='num_chunks', type=int, default=20, help='Number of chunks for MMR selection')
    parser.add_argument('--no-save-prompts', action='store_true', help='Disable prompt logging')
    
    args = parser.parse_args()
    
    # Validate files
    if not Path(args.entities).exists():
        logger.error(f"Entities file not found: {args.entities}")
        return 1
    if not Path(args.chunks).exists():
        logger.error(f"Chunks file not found: {args.chunks}")
        return 1
    
    # Load data
    entities, chunks = load_data(args.entities, args.chunks)
    
    # Run test
    if args.tune_params:
        test_parameter_tuning(entities, chunks)
    else:
        test_basic(
            entities,
            chunks,
            threshold=args.threshold,
            mmr_lambda=args.mmr_lambda,
            num_chunks=args.num_chunks,
            save_prompts=not args.no_save_prompts
        )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())