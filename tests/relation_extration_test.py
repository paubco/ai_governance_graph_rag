"""
Phase 1D Real Integration Test
Tests extraction with real LLM + optional parameter tuning

Usage:
    # Quick test (2 entities, default params)
    python tests/test_phase_1d_real.py

    # Parameter tuning mode
    python tests/test_phase_1d_real.py --tune-params
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.WARNING,  # Quieter
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS - Entity Lookup
# ============================================================================

def find_entity_by_name(entities: List[Dict], target_name: str, 
                        partial_match: bool = True) -> Optional[Dict]:
    """
    Find entity by name (case-insensitive, supports partial matching)
    
    Args:
        entities: List of entity dictionaries
        target_name: Name to search for
        partial_match: If True, match if target is substring of name or vice versa
    
    Returns:
        First matching entity or None if not found
    
    Example:
        >>> entity = find_entity_by_name(entities, "artificial intelligence")
        >>> entity['name']
        'artificial intelligence'
    """
    target = target_name.lower().strip()
    
    for entity in entities:
        name = entity.get('name', '').lower().strip()
        
        if partial_match:
            # Match if either is substring of the other
            if target in name or name in target:
                return entity
        else:
            # Exact match only
            if name == target:
                return entity
    
    return None


def validate_entity_quality(entity: Dict) -> tuple[bool, str]:
    """
    Check if entity has required fields for relation extraction
    
    Args:
        entity: Entity dictionary
    
    Returns:
        (is_valid, reason) tuple
    
    Example:
        >>> valid, reason = validate_entity_quality(entity)
        >>> if not valid:
        ...     print(f"Invalid: {reason}")
    """
    required_fields = ['name', 'type', 'description']
    
    # Check required fields exist
    for field in required_fields:
        if not entity.get(field):
            return False, f"Missing required field: {field}"
    
    # Check has chunk_ids (optional but recommended)
    chunk_ids = entity.get('chunk_ids', [])
    if not chunk_ids:
        logger.warning(f"Entity '{entity['name']}' has no chunk_ids - may have poor retrieval")
    
    return True, "OK"


def find_good_test_entities(entities: List[Dict], 
                            preferred_names: List[str],
                            min_chunks: int = 5,
                            max_results: int = 3) -> List[Dict]:
    """
    Find high-quality test entities by name with quality filtering
    
    Args:
        entities: List of all entities
        preferred_names: Names to look for (in order of preference)
        min_chunks: Minimum chunk_ids required
        max_results: Maximum entities to return
    
    Returns:
        List of good test entities (up to max_results)
    
    Example:
        >>> test_entities = find_good_test_entities(
        ...     entities, 
        ...     ["artificial intelligence", "GDPR", "EU"],
        ...     min_chunks=5,
        ...     max_results=2
        ... )
    """
    results = []
    
    for name in preferred_names:
        if len(results) >= max_results:
            break
        
        entity = find_entity_by_name(entities, name)
        
        if entity:
            # Validate quality
            valid, reason = validate_entity_quality(entity)
            if not valid:
                logger.warning(f"Skipping '{name}': {reason}")
                continue
            
            # Check chunk count
            chunk_count = len(entity.get('chunk_ids', []))
            if chunk_count < min_chunks:
                logger.warning(f"Skipping '{name}': only {chunk_count} chunks (need {min_chunks}+)")
                continue
            
            results.append(entity)
            logger.info(f"Found good test entity: '{entity['name']}' ({chunk_count} chunks)")
    
    return results


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load real entities and chunks"""
    
    # Load entities
    entities_paths = [
        PROJECT_ROOT / 'data/processed/normalized_entities.json',
        PROJECT_ROOT / 'data/processed/entities/normalized_entities.json',
        PROJECT_ROOT / 'data/interim/entities/normalized_entities.json',
    ]
    
    entities_path = None
    for path in entities_paths:
        if path.exists():
            entities_path = path
            break
    
    if not entities_path:
        raise FileNotFoundError("normalized_entities.json not found")
    
    print(f"Loading entities from: {entities_path}")
    with open(entities_path, 'r', encoding='utf-8') as f:
        entities_data = json.load(f)
    
    if isinstance(entities_data, dict) and 'entities' in entities_data:
        entities = entities_data['entities']
    elif isinstance(entities_data, list):
        entities = entities_data
    else:
        raise ValueError("Unknown entity format")
    
    print(f"  Loaded: {len(entities)} entities\n")
    
    # Load chunks
    chunks_path = PROJECT_ROOT / 'data/interim/chunks/chunks_embedded.json'
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks_embedded.json not found at {chunks_path}")
    
    print(f"Loading chunks from: {chunks_path}")
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    if isinstance(chunks_data, dict):
        chunks = list(chunks_data.values()) if 'chunks' not in chunks_data else chunks_data['chunks']
    elif isinstance(chunks_data, list):
        chunks = chunks_data
    else:
        raise ValueError("Unknown chunk format")
    
    print(f"  Loaded: {len(chunks)} chunks\n")
    
    return entities, chunks


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_basic(entities, chunks):
    """Basic test - 2 good entities with default params"""
    print("=" * 80)
    print("PHASE 1D - BASIC INTEGRATION TEST")
    print("=" * 80)
    print()
    
    from src.phase1_graph_construction.relation_extractor import RAKGRelationExtractor
    
    # Find good test entities by name
    preferred_entities = [
        "artificial intelligence",
        "GDPR",
        "EU",
        "machine learning",
        "neural network"
    ]
    
    print("Finding good test entities...")
    test_entities = find_good_test_entities(
        entities, 
        preferred_entities, 
        min_chunks=5, 
        max_results=2
    )
    
    if not test_entities:
        print("\n❌ ERROR: No good test entities found!")
        print(f"\nSearched for: {preferred_entities}")
        print(f"\nAvailable entities (first 20):")
        for i, e in enumerate(entities[:20], 1):
            chunk_count = len(e.get('chunk_ids', []))
            print(f"  {i}. {e.get('name', 'Unknown')} ({e.get('type', 'Unknown')}) - {chunk_count} chunks")
        print("\nTip: Update 'preferred_entities' list with entity names from above")
        return
    
    print(f"✓ Found {len(test_entities)} test entities\n")
    
    # Initialize extractor
    print("Initializing extractor (default params)...")
    extractor = RAKGRelationExtractor(
        model_name="Qwen/Qwen2.5-7B-Instruct-Turbo",  # Serverless model
        semantic_threshold=0.85,
        mmr_lambda=0.55,
        num_chunks=20
    )
    print(f"  ✓ Ready (threshold=0.85, lambda=0.55)\n")
    
    # Test extraction on each entity
    print(f"Testing extraction on {len(test_entities)} entities:\n")
    for i, entity in enumerate(test_entities, 1):
        print(f"Entity {i}: {entity.get('name', 'Unknown')} ({entity.get('type', 'Unknown')})")
        print(f"  Description: {entity.get('description', 'N/A')[:100]}...")
        print(f"  Available chunks: {len(entity.get('chunk_ids', []))}")
        
        try:
            print(f"  Extracting relations...")
            relations = extractor.extract_relations_for_entity(entity, chunks)
            
            print(f"  ✓ Extracted {len(relations)} relations")
            
            if relations:
                print(f"\n  Sample relations:")
                for j, rel in enumerate(relations[:3], 1):
                    print(f"    {j}. ({rel.get('subject')}) --[{rel.get('predicate')}]--> ({rel.get('object')})")
            else:
                print(f"  ℹ️  No relations found")
                
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  ⚠️  Failed: {e}")
            print(f"  → Skipping to next entity")
            continue
        
        print()


def test_parameters(entities, chunks):
    """Parameter tuning test - 1 entity, multiple param combinations"""
    print("=" * 80)
    print("PHASE 1D - PARAMETER TUNING TEST")
    print("=" * 80)
    print()
    
    from src.phase1_graph_construction.relation_extractor import RAKGRelationExtractor
    
    # Preferred test entities (in order of preference)
    preferred_names = [
        "artificial intelligence",
        "GDPR",
        "EU",
        "machine learning",
        "ChatGPT"
    ]
    
    print("Finding optimal test entity...")
    print(f"Searching for: {', '.join(preferred_names)}\n")
    
    # Try to find best entity
    test_entity = None
    for name in preferred_names:
        entity = find_entity_by_name(entities, name)
        if entity:
            valid, reason = validate_entity_quality(entity)
            if valid and len(entity.get('chunk_ids', [])) >= 10:
                test_entity = entity
                print(f"✓ Selected: '{entity['name']}'\n")
                break
    
    # Fallback: use first entity with enough chunks
    if not test_entity:
        print("⚠️  Preferred entities not found, using fallback...\n")
        for entity in entities:
            if len(entity.get('chunk_ids', [])) >= 10:
                valid, reason = validate_entity_quality(entity)
                if valid:
                    test_entity = entity
                    print(f"✓ Using fallback: '{entity['name']}'\n")
                    break
    
    if not test_entity:
        print("\n❌ ERROR: No suitable test entity found!")
        print(f"\nNeeds: 10+ chunks, valid name/type/description")
        print(f"\nAvailable entities (first 20):")
        for i, e in enumerate(entities[:20], 1):
            chunk_count = len(e.get('chunk_ids', []))
            print(f"  {i}. {e.get('name', 'Unknown')} ({e.get('type', 'Unknown')}) - {chunk_count} chunks")
        return
    
    # Display selected entity info
    print(f"Test entity: {test_entity.get('name', 'Unknown')}")
    print(f"Type: {test_entity.get('type', 'Unknown')}")
    print(f"Description: {test_entity.get('description', 'N/A')[:100]}...")
    print(f"Chunks available: {len(test_entity.get('chunk_ids', []))}\n")
    
    # Parameter combinations
    params_list = [
        {'semantic_threshold': 0.80, 'mmr_lambda': 0.55, 'label': 'threshold=0.80'},
        {'semantic_threshold': 0.85, 'mmr_lambda': 0.55, 'label': 'threshold=0.85 (default)'},
        {'semantic_threshold': 0.90, 'mmr_lambda': 0.55, 'label': 'threshold=0.90'},
        {'semantic_threshold': 0.85, 'mmr_lambda': 0.45, 'label': 'lambda=0.45 (more diversity)'},
        {'semantic_threshold': 0.85, 'mmr_lambda': 0.55, 'label': 'lambda=0.55 (default)'},
        {'semantic_threshold': 0.85, 'mmr_lambda': 0.65, 'label': 'lambda=0.65 (more relevance)'},
    ]
    
    print(f"Testing {len(params_list)} parameter combinations...")
    print(f"Cost: ~$0.10, Time: 3-5 minutes\n")
    print("Starting tests...\n")
    
    results = []
    
    for i, params in enumerate(params_list, 1):
        print(f"Test {i}/{len(params_list)}: {params['label']}")
        
        try:
            extractor = RAKGRelationExtractor(
                model_name="Qwen/Qwen2.5-7B-Instruct-Turbo",  # Serverless model
                semantic_threshold=params['semantic_threshold'],
                mmr_lambda=params['mmr_lambda'],
                num_chunks=20
            )
            
            # Get selected chunks (extractor uses instance variables)
            selected_chunks = extractor.gather_candidate_chunks(test_entity, chunks)
            selected_chunks = extractor.mmr_select_chunks(test_entity, selected_chunks)
            
            relations = extractor.extract_relations_for_entity(test_entity, chunks)
            
            result = {
                'threshold': params['semantic_threshold'],
                'lambda': params['mmr_lambda'],
                'num_relations': len(relations),
                'unique_predicates': len(set(r.get('predicate') for r in relations)),
                'unique_objects': len(set(r.get('object') for r in relations)),
                'relations': relations,
                'selected_chunks': selected_chunks[:5]  # First 5 chunks for display
            }
            results.append(result)
            
            print(f"  ✓ {len(relations)} relations, {result['unique_predicates']} unique predicates")
            print(f"  Selected chunks: {len(selected_chunks)}")
            print(f"\n  Sample chunks (first 3):")
            for j, chunk in enumerate(selected_chunks[:3], 1):
                chunk_text = chunk.get('text', '')[:150]
                print(f"    {j}. [{chunk.get('chunk_id', 'unknown')}] {chunk_text}...")
            print()
            
        except Exception as e:
            print(f"  ✗ Failed: {e}\n")
            results.append({'threshold': params['semantic_threshold'], 'lambda': params['mmr_lambda'], 'error': str(e)})
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Threshold':<12} {'Lambda':<8} {'Relations':<12} {'Predicates':<12} {'Objects':<10}")
    print("-" * 80)
    
    for r in results:
        if 'error' in r:
            print(f"{r['threshold']:<12} {r['lambda']:<8} ERROR")
        else:
            print(f"{r['threshold']:<12} {r['lambda']:<8} {r['num_relations']:<12} "
                  f"{r['unique_predicates']:<12} {r['unique_objects']:<10}")
    
    # Recommendations
    valid = [r for r in results if 'error' not in r and r['num_relations'] > 0]
    if valid:
        best_count = max(valid, key=lambda x: x['num_relations'])
        best_diversity = max(valid, key=lambda x: x['unique_predicates'])
        
        print(f"\nRecommendations:")
        print(f"  Most relations: threshold={best_count['threshold']}, lambda={best_count['lambda']} ({best_count['num_relations']} relations)")
        print(f"  Most diverse: threshold={best_diversity['threshold']}, lambda={best_diversity['lambda']} ({best_diversity['unique_predicates']} predicates)")
        
        # Detailed chunk comparison
        print("\n" + "=" * 80)
        print("CHUNK COMPARISON - Visual Inspection")
        print("=" * 80)
        print("\nCompare retrieved chunks across parameter combinations:")
        print("(First 5 chunks shown for each)\n")
        
        for r in valid[:3]:  # Show top 3 configurations
            if 'selected_chunks' in r:
                print(f"\n{'='*80}")
                print(f"Configuration: threshold={r['threshold']}, lambda={r['lambda']}")
                print(f"Relations: {r['num_relations']}, Unique predicates: {r['unique_predicates']}")
                print(f"{'='*80}\n")
                
                for i, chunk in enumerate(r['selected_chunks'], 1):
                    chunk_id = chunk.get('chunk_id', 'unknown')
                    chunk_text = chunk.get('text', '')
                    
                    print(f"Chunk {i}: [{chunk_id}]")
                    print(f"{chunk_text}")
                    print(f"{'-'*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Phase 1D Integration Test')
    parser.add_argument('--tune-params', action='store_true', help='Run parameter tuning test')
    args = parser.parse_args()
    
    try:
        entities, chunks = load_data()
        
        if args.tune_params:
            test_parameters(entities, chunks)
        else:
            test_basic(entities, chunks)
        
        print("\n" + "=" * 80)
        print("✅ TEST COMPLETE")
        print("=" * 80)
        print()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())