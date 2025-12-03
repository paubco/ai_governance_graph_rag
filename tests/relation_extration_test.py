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
               threshold: float = 0.90,
               mmr_lambda: float = 0.70,
               num_chunks: int = 6,
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
        entity_cooccurrence_file="data/interim/entities/cooccurrence_semantic.json",
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
    Parameter tuning: Test multiple configurations with comprehensive metrics
    
    Tests combinations of:
    - lambda: MMR diversity parameter
    - threshold: Second-round trigger threshold  
    - num_chunks: Chunk count (optional)
    
    Metrics tracked:
    - num_relations: Total relations extracted
    - unique_predicates: Diversity of predicates
    - unique_objects: Entity coverage
    - second_round_triggered: Whether second round happened
    - second_round_distance: Centroid distance value
    """
    api_key = os.getenv('TOGETHER_API_KEY')
    if not api_key:
        logger.error("TOGETHER_API_KEY not set!")
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("PARAMETER TUNING: Comprehensive Metrics")
    logger.info("=" * 80)
    
    # Get 8 test entities (mix of semantic and academic)
    test_entities = get_test_entities(entities, n=8)
    
    # Classify entities
    semantic_entities = []
    academic_entities = []
    for entity in test_entities:
        # Simple classification based on type
        entity_type = entity.get('type', '')
        if entity_type in {'Citation', 'Author', 'Editor', 'Journal', 'Publication', 
                          'Book', 'Paper', 'Article', 'Report', 'Conference'}:
            academic_entities.append(entity)
        else:
            semantic_entities.append(entity)
    
    logger.info(f"Test set: {len(semantic_entities)} semantic, {len(academic_entities)} academic")
    
    # Parameter grid (focus on semantic entities since academic is fixed)
    param_sets = [
        # Baseline
        {'lambda': 0.55, 'threshold': 0.15, 'num_chunks': 10, 'label': 'Baseline'},
        
        # Lambda variations (diversity control)
        {'lambda': 0.3, 'threshold': 0.15, 'num_chunks': 10, 'label': 'Low diversity'},
        {'lambda': 0.7, 'threshold': 0.15, 'num_chunks': 10, 'label': 'High diversity'},
        
        # Threshold variations (second-round trigger)
        {'lambda': 0.55, 'threshold': 0.10, 'num_chunks': 10, 'label': 'Loose threshold'},
        {'lambda': 0.55, 'threshold': 0.20, 'num_chunks': 10, 'label': 'Strict threshold'},
        
        # Chunk count variations
        {'lambda': 0.55, 'threshold': 0.15, 'num_chunks': 15, 'label': 'More chunks'},
        {'lambda': 0.55, 'threshold': 0.15, 'num_chunks': 20, 'label': 'Many chunks'},
    ]
    
    all_results = []
    
    for i, params in enumerate(param_sets, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Configuration {i}/{len(param_sets)}: {params['label']}")
        logger.info(f"  λ={params['lambda']}, threshold={params['threshold']}, k={params['num_chunks']}")
        logger.info(f"{'='*80}")
        
        extractor = RAKGRelationExtractor(
            model_name="Qwen/Qwen2.5-7B-Instruct-Turbo",
            api_key=api_key,
            semantic_threshold=0.85,  # Keep fixed
            mmr_lambda=params['lambda'],
            num_chunks=params['num_chunks'],
            entity_cooccurrence_file="data/interim/entities/cooccurrence_semantic.json",
            normalized_entities_file="data/interim/entities/normalized_entities.json"
        )
        
        config_results = {
            'config': params,
            'semantic_entities': [],
            'academic_entities': [],
        }
        
        # Test on all entities
        for entity in test_entities:
            entity_type = entity.get('type', '')
            is_academic = entity_type in {'Citation', 'Author', 'Editor', 'Journal', 
                                         'Publication', 'Book', 'Paper', 'Article', 
                                         'Report', 'Conference'}
            
            try:
                relations = extractor.extract_relations_for_entity(
                    entity,
                    chunks,
                    save_prompt=False
                )
                
                # Compute metrics
                predicates = list(set(r.get('predicate') for r in relations))
                objects = list(set(r.get('object') for r in relations))
                
                entity_result = {
                    'name': entity['name'],
                    'type': entity_type,
                    'num_relations': len(relations),
                    'unique_predicates': len(predicates),
                    'unique_objects': len(objects),
                    'predicates': predicates[:5],  # Sample
                }
                
                if is_academic:
                    config_results['academic_entities'].append(entity_result)
                else:
                    config_results['semantic_entities'].append(entity_result)
                    
                logger.info(f"  {entity['name']} [{entity_type}]: {len(relations)} relations, {len(predicates)} predicates")
                
            except Exception as e:
                logger.error(f"  {entity['name']}: Failed - {e}")
        
        # Aggregate metrics
        semantic_rels = [e['num_relations'] for e in config_results['semantic_entities']]
        academic_rels = [e['num_relations'] for e in config_results['academic_entities']]
        
        config_results['summary'] = {
            'total_relations': sum(semantic_rels) + sum(academic_rels),
            'semantic_avg': sum(semantic_rels) / len(semantic_rels) if semantic_rels else 0,
            'academic_avg': sum(academic_rels) / len(academic_rels) if academic_rels else 0,
        }
        
        all_results.append(config_results)
        
        logger.info(f"\n  Summary:")
        logger.info(f"    Total relations: {config_results['summary']['total_relations']}")
        logger.info(f"    Semantic avg: {config_results['summary']['semantic_avg']:.1f}")
        logger.info(f"    Academic avg: {config_results['summary']['academic_avg']:.1f}")
    
    # Final comparison
    logger.info("\n" + "=" * 80)
    logger.info("PARAMETER TUNING RESULTS")
    logger.info("=" * 80)
    
    # Table header
    logger.info(f"\n{'Config':<20s} {'λ':>6s} {'θ':>6s} {'k':>4s} {'Total':>7s} {'Sem':>6s} {'Acad':>6s}")
    logger.info("-" * 80)
    
    # Table rows
    for result in all_results:
        cfg = result['config']
        summ = result['summary']
        logger.info(
            f"{cfg['label']:<20s} "
            f"{cfg['lambda']:>6.2f} "
            f"{cfg['threshold']:>6.2f} "
            f"{cfg['num_chunks']:>4d} "
            f"{summ['total_relations']:>7d} "
            f"{summ['semantic_avg']:>6.1f} "
            f"{summ['academic_avg']:>6.1f}"
        )
    
    # Best config
    best = max(all_results, key=lambda x: x['summary']['total_relations'])
    logger.info(f"\n✓ Best config: {best['config']['label']}")
    logger.info(f"  λ={best['config']['lambda']}, θ={best['config']['threshold']}, k={best['config']['num_chunks']}")
    logger.info(f"  Total relations: {best['summary']['total_relations']}")
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
    parser.add_argument('--num-chunks', dest='num_chunks', type=int, default=10, help='Number of chunks per stage for MMR selection (max 20 total with second round)')
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