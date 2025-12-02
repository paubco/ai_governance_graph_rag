"""
Phase 1D - Relation Extraction Testing
Updated: Uses exact entity keys from normalized_entities.json

Test Strategy:
1. Try preferred high-quality entities (from your actual data)
2. Fallback to top entities by chunk_count
3. No fragile name search!

Usage:
    # Basic test (2 entities, default params)
    python tests/relation_extraction_test_updated.py
    
    # Parameter tuning (1 entity, 6 param combinations)
    python tests/relation_extraction_test_updated.py --tune-params
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phase1_graph_construction.entity_disambiguator import build_entity_map


# ============================================================================
# GOOD TEST ENTITIES (From your actual normalized_entities.json)
# ============================================================================

# Top entities by chunk count (high quality, well-represented)
PREFERRED_ENTITIES = [
    ("AI System", "Technology"),           # 4469 chunks
    ("European Union", "Organization"),    # 1063 chunks
    ("Chat-GPT", "AI System"),             # 840 chunks
    ("EU AI Act", "Regulation"),           # 474 chunks
    ("AI governance", "Concept"),          # 370 chunks
    ("AI Act", "Legislation"),             # 349 chunks
    ("transparency", "Concept"),           # 338 chunks
    ("stakeholders", "Stakeholder Group"), # 314 chunks
    ("US", "Country"),                     # 304 chunks
    ("human rights", "Legal Concept"),     # 300 chunks
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_entities(entities_file: str) -> List[Dict]:
    """Load normalized entities"""
    print(f"Loading entities from: {entities_file}")
    with open(entities_file, 'r', encoding='utf-8') as f:
        entities = json.load(f)
    print(f"✓ Loaded {len(entities)} entities")
    return entities


def load_chunks(chunks_file: str) -> Dict[str, str]:
    """Load chunks for retrieval"""
    print(f"Loading chunks from: {chunks_file}")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    # Convert to dict: chunk_id -> text
    chunks = {}
    if isinstance(chunks_data, list):
        for chunk in chunks_data:
            chunks[chunk['chunk_id']] = chunk['text']
    elif isinstance(chunks_data, dict):
        chunks = {k: v['text'] if isinstance(v, dict) else v 
                 for k, v in chunks_data.items()}
    
    print(f"✓ Loaded {len(chunks)} chunks")
    return chunks


def get_test_entities(entities: List[Dict], 
                      n: int = 2,
                      min_chunks: int = 100) -> List[Dict]:
    """
    Get good test entities with robust fallback
    
    Strategy:
    1. Try preferred entities from PREFERRED_ENTITIES list
    2. Fallback to top entities by chunk_count if needed
    
    Args:
        entities: List of all entities
        n: Number of entities needed
        min_chunks: Minimum chunk count for quality
        
    Returns:
        List of test entities
    """
    entity_map = build_entity_map(entities)
    
    # Try preferred entities first
    test_entities = []
    for key in PREFERRED_ENTITIES:
        if key in entity_map:
            entity = entity_map[key]
            if len(entity.get('chunk_ids', [])) >= min_chunks:
                test_entities.append(entity)
                if len(test_entities) >= n:
                    break
    
    # Fallback: top entities by chunk_count
    if len(test_entities) < n:
        print(f"⚠️ Only found {len(test_entities)} preferred entities")
        print(f"   Using top entities by chunk count as fallback...")
        
        # Sort by chunk count
        sorted_entities = sorted(
            entities, 
            key=lambda e: len(e.get('chunk_ids', [])), 
            reverse=True
        )
        
        # Add until we have n entities
        for entity in sorted_entities:
            if len(entity.get('chunk_ids', [])) >= min_chunks:
                # Check if already added
                key = (entity['name'], entity['type'])
                if key not in [(e['name'], e['type']) for e in test_entities]:
                    test_entities.append(entity)
                    if len(test_entities) >= n:
                        break
    
    if len(test_entities) < n:
        print(f"❌ Could not find {n} entities with {min_chunks}+ chunks")
        print(f"   Found: {len(test_entities)}")
        return test_entities
    
    # Report selection
    print(f"\n✓ Selected {len(test_entities)} test entities:")
    for i, entity in enumerate(test_entities, 1):
        print(f"  {i}. {entity['name']} [{entity['type']}] - {len(entity.get('chunk_ids', []))} chunks")
    
    return test_entities


def validate_entity(entity: Dict) -> Tuple[bool, str]:
    """
    Validate entity has required fields for relation extraction
    
    Returns:
        (is_valid, reason)
    """
    # Check required fields
    if not entity.get('name'):
        return False, "missing name"
    if not entity.get('type'):
        return False, "missing type"
    if not entity.get('chunk_ids'):
        return False, "no chunk_ids"
    
    # Check chunk count
    chunk_ids = entity.get('chunk_ids', [])
    if len(chunk_ids) < 10:
        return False, f"only {len(chunk_ids)} chunks"
    
    return True, "OK"


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_basic(entities_file: str, 
               chunks_file: str,
               k: int = 5,
               max_depth: int = 2,
               min_similarity: float = 0.7):
    """
    Basic test: Extract relations for 2 high-quality entities
    
    Args:
        entities_file: Path to normalized_entities.json
        chunks_file: Path to chunks file
        k: Top-k chunks to retrieve
        max_depth: BFS expansion depth
        min_similarity: Similarity threshold
    """
    print("=" * 80)
    print("BASIC TEST: 2 Entities with Default Parameters")
    print("=" * 80)
    print()
    
    # Load data
    entities = load_entities(entities_file)
    chunks = load_chunks(chunks_file)
    
    # Get test entities
    test_entities = get_test_entities(entities, n=2, min_chunks=100)
    
    if len(test_entities) < 2:
        print("❌ Test failed: Could not find 2 good entities")
        return
    
    print()
    print(f"Parameters: k={k}, max_depth={max_depth}, min_similarity={min_similarity}")
    print()
    
    # TODO: Call relation extraction here
    print("🚧 Relation extraction not implemented yet")
    print("   Next step: Integrate with your relation extractor")
    
    # Placeholder for now
    for entity in test_entities:
        print(f"\nEntity: {entity['name']} [{entity['type']}]")
        print(f"  Chunks: {len(entity['chunk_ids'])}")
        print(f"  Sample chunk_ids: {entity['chunk_ids'][:3]}")


def test_parameter_tuning(entities_file: str,
                          chunks_file: str):
    """
    Parameter tuning: Test 6 parameter combinations on 1 entity
    
    Tests:
    1. k=3, depth=1, sim=0.7
    2. k=5, depth=1, sim=0.7
    3. k=3, depth=2, sim=0.7
    4. k=5, depth=2, sim=0.7
    5. k=5, depth=2, sim=0.6
    6. k=5, depth=2, sim=0.8
    """
    print("=" * 80)
    print("PARAMETER TUNING: 1 Entity with 6 Configurations")
    print("=" * 80)
    print()
    
    # Load data
    entities = load_entities(entities_file)
    chunks = load_chunks(chunks_file)
    
    # Get ONE high-quality entity
    test_entities = get_test_entities(entities, n=1, min_chunks=200)
    
    if not test_entities:
        print("❌ Test failed: Could not find a good entity")
        return
    
    entity = test_entities[0]
    print(f"\nTest entity: {entity['name']} [{entity['type']}]")
    print(f"  Chunks: {len(entity['chunk_ids'])}")
    print()
    
    # Parameter combinations
    param_sets = [
        {"k": 3, "max_depth": 1, "min_similarity": 0.7},
        {"k": 5, "max_depth": 1, "min_similarity": 0.7},
        {"k": 3, "max_depth": 2, "min_similarity": 0.7},
        {"k": 5, "max_depth": 2, "min_similarity": 0.7},
        {"k": 5, "max_depth": 2, "min_similarity": 0.6},
        {"k": 5, "max_depth": 2, "min_similarity": 0.8},
    ]
    
    print("Testing parameter combinations:")
    for i, params in enumerate(param_sets, 1):
        print(f"\n{i}. k={params['k']}, depth={params['max_depth']}, sim={params['min_similarity']}")
        
        # TODO: Call relation extraction here
        print("   🚧 Extraction not implemented")
    
    print("\n✓ Parameter tuning test complete")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test Phase 1D relation extraction with normalized entities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic test (2 entities)
    python tests/relation_extraction_test_updated.py
    
    # Parameter tuning (1 entity, 6 configs)
    python tests/relation_extraction_test_updated.py --tune-params
    
    # Custom parameters
    python tests/relation_extraction_test_updated.py --k 10 --depth 3 --similarity 0.75

Default file locations:
    entities: data/interim/entities/normalized_entities.json
    chunks: data/interim/chunks/chunks_embedded.json
        """
    )
    
    parser.add_argument(
        '--entities',
        type=str,
        default='data/interim/entities/normalized_entities.json',
        help='Path to normalized entities file'
    )
    parser.add_argument(
        '--chunks',
        type=str,
        default='data/interim/chunks/chunks_embedded.json',
        help='Path to chunks file'
    )
    parser.add_argument(
        '--tune-params',
        action='store_true',
        help='Run parameter tuning test (1 entity, 6 configs)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Top-k chunks to retrieve (default: 5)'
    )
    parser.add_argument(
        '--depth',
        type=int,
        default=2,
        help='BFS expansion depth (default: 2)'
    )
    parser.add_argument(
        '--similarity',
        type=float,
        default=0.7,
        help='Minimum similarity threshold (default: 0.7)'
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    if not Path(args.entities).exists():
        print(f"❌ Entities file not found: {args.entities}")
        return 1
    
    if not Path(args.chunks).exists():
        print(f"❌ Chunks file not found: {args.chunks}")
        return 1
    
    # Run appropriate test
    if args.tune_params:
        test_parameter_tuning(args.entities, args.chunks)
    else:
        test_basic(
            args.entities, 
            args.chunks,
            k=args.k,
            max_depth=args.depth,
            min_similarity=args.similarity
        )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())