"""
Phase 1D - Relation Extraction Test (Merged Version)
Combines robust entity selection with actual LLM extraction

Features:
- Uses exact entity keys (no fragile indices!)
- Integrates RAKGRelationExtractor
- Systematic parameter testing (threshold + lambda grid)
- Clear results comparison

Usage:
    # Basic test (2 entities, default params)
    python tests/relation_extraction_test.py
    
    # Parameter tuning (1 entity, systematic grid search)
    python tests/relation_extraction_test.py --tune-params
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PREFERRED TEST ENTITIES (From actual normalized_entities.json)
# ============================================================================

PREFERRED_ENTITIES = [
    ("AI System", "Technology"),           # 4469 chunks
    ("European Union", "Organization"),    # 1063 chunks
    ("Chat-GPT", "AI System"),             # 840 chunks
    ("EU AI Act", "Regulation"),           # 474 chunks
    ("AI governance", "Concept"),          # 370 chunks
    ("transparency", "Concept"),           # 338 chunks
    ("stakeholders", "Stakeholder Group"), # 314 chunks
    ("US", "Country"),                     # 304 chunks
]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_entities() -> List[Dict]:
    """Load normalized entities with fallback paths"""
    paths = [
        PROJECT_ROOT / 'data/interim/entities/normalized_entities.json',
        PROJECT_ROOT / 'data/processed/entities/normalized_entities.json',
        PROJECT_ROOT / 'data/processed/normalized_entities.json',
    ]
    
    for path in paths:
        if path.exists():
            print(f"Loading entities from: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different formats
            if isinstance(data, dict) and 'entities' in data:
                entities = data['entities']
            elif isinstance(data, list):
                entities = data
            else:
                raise ValueError(f"Unknown entity format in {path}")
            
            print(f"✓ Loaded {len(entities)} entities\n")
            return entities
    
    raise FileNotFoundError("normalized_entities.json not found")


def load_chunks() -> List[Dict]:
    """Load chunks with fallback paths"""
    paths = [
        PROJECT_ROOT / 'data/interim/chunks/chunks_embedded.json',
        PROJECT_ROOT / 'data/processed/chunks_embedded.json',
    ]
    
    for path in paths:
        if path.exists():
            print(f"Loading chunks from: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different formats
            if isinstance(data, dict) and 'chunks' in data:
                chunks = data['chunks']
            elif isinstance(data, list):
                chunks = data
            elif isinstance(data, dict):
                chunks = list(data.values())
            else:
                raise ValueError(f"Unknown chunk format in {path}")
            
            print(f"✓ Loaded {len(chunks)} chunks\n")
            return chunks
    
    raise FileNotFoundError("chunks_embedded.json not found")


# ============================================================================
# ROBUST ENTITY SELECTION
# ============================================================================

def build_entity_map(entities: List[Dict]) -> Dict[Tuple[str, str], Dict]:
    """Build (name, type) -> entity lookup map"""
    return {(e['name'], e['type']): e for e in entities}


def get_test_entities(entities: List[Dict], 
                      n: int = 2,
                      min_chunks: int = 100) -> List[Dict]:
    """
    Get test entities with robust fallback
    
    Strategy:
    1. Try PREFERRED_ENTITIES list (exact keys)
    2. Fallback to top-N by chunk_count
    
    Returns:
        List of n high-quality entities
    """
    entity_map = build_entity_map(entities)
    
    # Try preferred entities
    selected = []
    for key in PREFERRED_ENTITIES:
        if key in entity_map:
            entity = entity_map[key]
            if len(entity.get('chunk_ids', [])) >= min_chunks:
                selected.append(entity)
                if len(selected) >= n:
                    break
    
    # Fallback: top by chunk_count
    if len(selected) < n:
        print(f"⚠️ Only found {len(selected)}/{n} preferred entities")
        print(f"   Using top entities by chunk count...\n")
        
        sorted_entities = sorted(
            entities,
            key=lambda e: len(e.get('chunk_ids', [])),
            reverse=True
        )
        
        for entity in sorted_entities:
            if len(entity.get('chunk_ids', [])) >= min_chunks:
                key = (entity['name'], entity['type'])
                if key not in [(e['name'], e['type']) for e in selected]:
                    selected.append(entity)
                    if len(selected) >= n:
                        break
    
    if len(selected) < n:
        print(f"❌ Could not find {n} entities with {min_chunks}+ chunks")
        print(f"   Only found: {len(selected)}")
    
    return selected


# ============================================================================
# BASIC TEST
# ============================================================================

def test_basic(entities: List[Dict], chunks: List[Dict]):
    """
    Basic test: Extract relations for 2 entities with default params
    
    Default params:
    - semantic_threshold: 0.85
    - mmr_lambda: 0.55
    - num_chunks: 20
    """
    print("=" * 80)
    print("PHASE 1D - BASIC TEST (2 Entities)")
    print("=" * 80)
    print()
    
    from src.phase1_graph_construction.relation_extractor import RAKGRelationExtractor
    
    # Get test entities
    print("Selecting test entities...")
    test_entities = get_test_entities(entities, n=2, min_chunks=200)
    
    if len(test_entities) < 2:
        print("❌ Could not find 2 good entities")
        return
    
    print(f"\n✓ Selected {len(test_entities)} entities:")
    for i, e in enumerate(test_entities, 1):
        print(f"  {i}. {e['name']} [{e['type']}] - {len(e.get('chunk_ids', []))} chunks")
    print()
    
    # Initialize extractor with defaults
    print("Initializing extractor...")
    extractor = RAKGRelationExtractor(
        model_name="Qwen/Qwen2.5-7B-Instruct-Turbo",
        semantic_threshold=0.85,
        mmr_lambda=0.55,
        num_chunks=20
    )
    print("✓ Ready (threshold=0.85, lambda=0.55, chunks=20)\n")
    
    # Extract relations
    print(f"Extracting relations for {len(test_entities)} entities...\n")
    
    for i, entity in enumerate(test_entities, 1):
        print(f"Entity {i}: {entity['name']} [{entity['type']}]")
        print(f"  Chunks available: {len(entity.get('chunk_ids', []))}")
        
        try:
            relations = extractor.extract_relations_for_entity(entity, chunks)
            
            print(f"  ✓ Extracted {len(relations)} relations")
            
            if relations:
                # Show sample relations
                print(f"\n  Sample relations (first 5):")
                for j, rel in enumerate(relations[:5], 1):
                    subj = rel.get('subject', '?')
                    pred = rel.get('predicate', '?')
                    obj = rel.get('object', '?')
                    print(f"    {j}. ({subj}) --[{pred}]--> ({obj})")
                
                # Show predicate diversity
                unique_preds = set(r.get('predicate') for r in relations)
                print(f"\n  Unique predicates: {len(unique_preds)}")
                print(f"  Top predicates: {', '.join(list(unique_preds)[:5])}")
            else:
                print(f"  ℹ️ No relations found (try lowering threshold)")
        
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
        
        print()


# ============================================================================
# PARAMETER TUNING TEST
# ============================================================================

def test_parameter_tuning(entities: List[Dict], chunks: List[Dict]):
    """
    Parameter tuning: Test threshold + lambda combinations on 1 entity
    
    Tests a 3x3 grid:
    - Thresholds: 0.80, 0.85, 0.90
    - Lambdas: 0.45, 0.55, 0.65
    
    Total: 9 combinations (~$0.15, ~5-10 minutes)
    """
    print("=" * 80)
    print("PHASE 1D - PARAMETER TUNING (Grid Search)")
    print("=" * 80)
    print()
    
    from src.phase1_graph_construction.relation_extractor import RAKGRelationExtractor
    
    # Get ONE high-quality entity
    print("Selecting test entity...")
    test_entities = get_test_entities(entities, n=1, min_chunks=300)
    
    if not test_entities:
        print("❌ Could not find a good entity")
        return
    
    entity = test_entities[0]
    print(f"\n✓ Test entity: {entity['name']} [{entity['type']}]")
    print(f"  Chunks available: {len(entity.get('chunk_ids', []))}\n")
    
    # Parameter grid
    thresholds = [0.80, 0.85, 0.90]
    lambdas = [0.45, 0.55, 0.65]
    
    print(f"Testing {len(thresholds)}x{len(lambdas)} = {len(thresholds)*len(lambdas)} combinations:")
    print(f"  Thresholds: {thresholds}")
    print(f"  Lambdas: {lambdas}")
    print(f"  Cost: ~$0.15, Time: ~5-10 min\n")
    
    input("Press Enter to start testing...")
    print()
    
    # Run grid search
    results = []
    
    for threshold in thresholds:
        for lambda_val in lambdas:
            config_label = f"t={threshold}, λ={lambda_val}"
            print(f"Testing: {config_label}")
            
            try:
                extractor = RAKGRelationExtractor(
                    model_name="Qwen/Qwen2.5-7B-Instruct-Turbo",
                    semantic_threshold=threshold,
                    mmr_lambda=lambda_val,
                    num_chunks=20
                )
                
                relations = extractor.extract_relations_for_entity(entity, chunks)
                
                unique_preds = set(r.get('predicate') for r in relations)
                unique_objs = set(r.get('object') for r in relations)
                
                result = {
                    'threshold': threshold,
                    'lambda': lambda_val,
                    'relations': len(relations),
                    'unique_predicates': len(unique_preds),
                    'unique_objects': len(unique_objs),
                    'sample_relations': relations[:3]
                }
                results.append(result)
                
                print(f"  ✓ {len(relations)} relations, {len(unique_preds)} unique predicates")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                results.append({
                    'threshold': threshold,
                    'lambda': lambda_val,
                    'error': str(e)
                })
            
            print()
    
    # Display results grid
    print("\n" + "=" * 80)
    print("RESULTS GRID")
    print("=" * 80)
    print()
    
    # Header
    print(f"{'Threshold':<12}", end="")
    for lam in lambdas:
        print(f"λ={lam:<8}", end="")
    print()
    print("-" * 80)
    
    # Grid
    for threshold in thresholds:
        print(f"{threshold:<12}", end="")
        for lambda_val in lambdas:
            result = next((r for r in results 
                          if r['threshold'] == threshold and r['lambda'] == lambda_val), None)
            if result and 'error' not in result:
                print(f"{result['relations']:<10}", end="")
            else:
                print(f"{'ERROR':<10}", end="")
        print()
    
    print()
    
    # Analysis
    valid_results = [r for r in results if 'error' not in r]
    
    if valid_results:
        print("=" * 80)
        print("ANALYSIS")
        print("=" * 80)
        print()
        
        # Best configurations
        best_count = max(valid_results, key=lambda x: x['relations'])
        best_diversity = max(valid_results, key=lambda x: x['unique_predicates'])
        
        print("Recommendations:")
        print(f"  Most relations: threshold={best_count['threshold']}, lambda={best_count['lambda']}")
        print(f"    → {best_count['relations']} relations, {best_count['unique_predicates']} predicates")
        print()
        print(f"  Most diverse: threshold={best_diversity['threshold']}, lambda={best_diversity['lambda']}")
        print(f"    → {best_diversity['relations']} relations, {best_diversity['unique_predicates']} predicates")
        print()
        
        # Insights
        print("Observations:")
        
        # Threshold effect
        avg_by_threshold = {}
        for t in thresholds:
            t_results = [r for r in valid_results if r['threshold'] == t]
            if t_results:
                avg_by_threshold[t] = sum(r['relations'] for r in t_results) / len(t_results)
        
        if avg_by_threshold:
            print(f"  Threshold effect (avg relations):")
            for t, avg in sorted(avg_by_threshold.items()):
                print(f"    {t}: {avg:.1f} relations")
        
        # Lambda effect
        avg_by_lambda = {}
        for lam in lambdas:
            lam_results = [r for r in valid_results if r['lambda'] == lam]
            if lam_results:
                avg_by_lambda[lam] = sum(r['relations'] for r in lam_results) / len(lam_results)
        
        if avg_by_lambda:
            print(f"  Lambda effect (avg relations):")
            for lam, avg in sorted(avg_by_lambda.items()):
                diversity = "more diversity" if lam < 0.55 else ("balanced" if lam == 0.55 else "more relevance")
                print(f"    {lam} ({diversity}): {avg:.1f} relations")
        
        print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1D Relation Extraction Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic test (2 entities, default params)
    python tests/relation_extraction_test.py
    
    # Parameter tuning (1 entity, 3x3 grid)
    python tests/relation_extraction_test.py --tune-params

Parameter Ranges:
    semantic_threshold: 0.80-0.90 (higher = more selective)
    mmr_lambda: 0.45-0.65 (lower = more diversity, higher = more relevance)
        """
    )
    
    parser.add_argument(
        '--tune-params',
        action='store_true',
        help='Run parameter tuning grid search'
    )
    
    args = parser.parse_args()
    
    try:
        # Load data
        entities = load_entities()
        chunks = load_chunks()
        
        # Run appropriate test
        if args.tune_params:
            test_parameter_tuning(entities, chunks)
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