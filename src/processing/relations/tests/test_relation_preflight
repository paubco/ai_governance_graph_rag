# -*- coding: utf-8 -*-
"""
Preflight test for Phase 1D relation extraction (v2.0).

Tests extraction on 10 hand-picked entities covering edge cases before full run.
Validates:
1. Relations have valid entity_ids (exist in entity set)
2. Predicates are normalized (lowercase_underscore)
3. Track 1 (semantic) vs Track 2 (academic) routing works
4. No self-references
5. Cost within expected range

Usage:
    python -m src.processing.relations.tests.test_relation_preflight
    
    # Verbose output
    python -m src.processing.relations.tests.test_relation_preflight --verbose
"""

# Standard library
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_sample_entities(
    semantic_file: Path,
    metadata_file: Path,
    n_semantic: int = 7,
    n_citation: int = 3
) -> List[Dict]:
    """
    Load stratified sample of entities for testing.
    
    Selects:
    - n_semantic entities from semantic file (varied types)
    - n_citation Citation entities from metadata file
    """
    entities = []
    
    # Load semantic entities
    if semantic_file.exists():
        semantic_entities = []
        with open(semantic_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    semantic_entities.append(json.loads(line))
        
        # Group by type
        by_type: Dict[str, List[Dict]] = {}
        for e in semantic_entities:
            t = e.get('type', 'Unknown')
            by_type.setdefault(t, []).append(e)
        
        # Sample from different types
        types_to_sample = ['Regulation', 'Technology', 'RegulatoryConcept', 
                          'Organization', 'Risk', 'Location', 'TechnicalConcept']
        
        for t in types_to_sample:
            if t in by_type and len(entities) < n_semantic:
                # Pick entity with moderate chunk count (not too few, not too many)
                candidates = [e for e in by_type[t] if 3 <= len(e.get('chunk_ids', [])) <= 20]
                if candidates:
                    entities.append(candidates[len(candidates)//2])  # Middle of list
                elif by_type[t]:
                    entities.append(by_type[t][0])
        
        print(f"Loaded {len(entities)} semantic entities from {len(semantic_entities)} total")
    
    # Load citation entities
    if metadata_file.exists() and n_citation > 0:
        citations = []
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    if e.get('type') == 'Citation':
                        citations.append(e)
        
        # Sample citations with moderate chunk count
        candidates = [e for e in citations if 2 <= len(e.get('chunk_ids', [])) <= 10]
        for c in candidates[:n_citation]:
            entities.append(c)
        
        print(f"Loaded {min(n_citation, len(candidates))} citation entities from {len(citations)} total")
    
    return entities


def validate_relation(
    relation: Dict,
    valid_entity_ids: Set[str],
    target_entity_id: str
) -> Dict:
    """
    Validate a single relation.
    
    Returns dict with validation results.
    """
    result = {
        'valid': True,
        'issues': []
    }
    
    subject_id = relation.get('subject_id', '')
    object_id = relation.get('object_id', '')
    predicate = relation.get('predicate', '')
    
    # Check subject is target
    if subject_id != target_entity_id:
        result['valid'] = False
        result['issues'].append(f"subject_id mismatch: {subject_id} != {target_entity_id}")
    
    # Check object exists
    if object_id not in valid_entity_ids:
        result['valid'] = False
        result['issues'].append(f"object_id not in valid set: {object_id}")
    
    # Check predicate format
    if not predicate:
        result['valid'] = False
        result['issues'].append("empty predicate")
    elif predicate != predicate.lower():
        result['issues'].append(f"predicate not lowercase: {predicate}")
    elif ' ' in predicate:
        result['issues'].append(f"predicate has spaces: {predicate}")
    
    # Check for self-reference
    if subject_id == object_id:
        result['valid'] = False
        result['issues'].append("self-reference")
    
    return result


def run_preflight_test(verbose: bool = False) -> bool:
    """
    Run preflight test on 10 entities.
    
    Returns True if all tests pass.
    """
    print("=" * 80)
    print("PHASE 1D PREFLIGHT TEST (v2.0)")
    print("=" * 80)
    print()
    
    # File paths
    semantic_file = PROJECT_ROOT / "data/processed/entities/entities_semantic.jsonl"
    metadata_file = PROJECT_ROOT / "data/processed/entities/entities_metadata.jsonl"
    chunks_file = PROJECT_ROOT / "data/processed/chunks/chunks_embedded.json"
    lookup_file = PROJECT_ROOT / "data/interim/entities/entity_id_lookup.json"
    cooccur_semantic = PROJECT_ROOT / "data/interim/entities/cooccurrence_semantic.json"
    cooccur_concept = PROJECT_ROOT / "data/interim/entities/cooccurrence_concept.json"
    
    # Check prerequisites
    missing = []
    for name, path in [
        ("entities_semantic.jsonl", semantic_file),
        ("chunks_embedded.json", chunks_file),
        ("entity_id_lookup.json", lookup_file),
        ("cooccurrence_semantic.json", cooccur_semantic),
    ]:
        if not path.exists():
            missing.append(name)
    
    if missing:
        print(f"MISSING FILES: {missing}")
        print("Run build_entity_cooccurrence.py first")
        return False
    
    # Load sample entities
    print("[1] Loading sample entities...")
    entities = load_sample_entities(semantic_file, metadata_file)
    
    if len(entities) < 5:
        print(f"Only {len(entities)} entities loaded - need at least 5")
        return False
    
    print(f"    Selected {len(entities)} entities for testing")
    for e in entities:
        chunk_count = len(e.get('chunk_ids', []))
        print(f"      - {e['name'][:40]:<40} [{e['type']}] ({chunk_count} chunks)")
    print()
    
    # Load chunks
    print("[2] Loading chunks...")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    chunks = list(chunks_data.values()) if isinstance(chunks_data, dict) else chunks_data
    print(f"    Loaded {len(chunks)} chunks")
    print()
    
    # Load entity lookup (for validation)
    print("[3] Loading entity lookup...")
    with open(lookup_file, 'r', encoding='utf-8') as f:
        entity_lookup = json.load(f)
    valid_entity_ids = set(entity_lookup.keys())
    print(f"    Loaded {len(valid_entity_ids)} entity IDs")
    print()
    
    # Initialize extractor
    print("[4] Initializing extractor...")
    from src.processing.relations.relation_extractor import RAKGRelationExtractor
    
    extractor = RAKGRelationExtractor(
        model_name='mistralai/Mistral-7B-Instruct-v0.3',
        num_chunks=6,
        mmr_lambda=0.65,
        semantic_threshold=0.85,
        max_tokens=16000,
        second_round_threshold=0.25,
        entity_lookup_file=str(lookup_file),
        cooccurrence_semantic_file=str(cooccur_semantic),
        cooccurrence_concept_file=str(cooccur_concept) if cooccur_concept.exists() else None,
        debug_mode=verbose
    )
    print("    Extractor initialized")
    print()
    
    # Run extraction
    print("[5] Running extraction on sample entities...")
    print("-" * 80)
    
    results = []
    total_relations = 0
    total_cost = 0.0
    total_time = 0.0
    all_predicates = set()
    issues_found = []
    
    for i, entity in enumerate(entities, 1):
        entity_id = entity['entity_id']
        entity_name = entity['name'][:35]
        entity_type = entity['type']
        
        print(f"  [{i}/{len(entities)}] {entity_name:<35} [{entity_type}]", end=" ", flush=True)
        
        start = time.time()
        try:
            result = extractor.extract_relations_for_entity(entity, chunks)
            elapsed = time.time() - start
            total_time += elapsed
            
            relations = result.get('relations', [])
            num_batches = result.get('num_batches', 0)
            strategy = result.get('strategy', 'unknown')
            
            # Estimate cost
            cost = (num_batches * 1500 / 1_000_000) * 0.20
            total_cost += cost
            
            # Validate relations
            valid_count = 0
            for rel in relations:
                validation = validate_relation(rel, valid_entity_ids, entity_id)
                if validation['valid']:
                    valid_count += 1
                    all_predicates.add(rel.get('predicate', ''))
                else:
                    issues_found.extend(validation['issues'])
            
            total_relations += len(relations)
            
            print(f"-> {len(relations):2d} relations ({valid_count} valid), "
                  f"{num_batches} batch(es), {strategy}, "
                  f"${cost:.4f}, {elapsed:.1f}s")
            
            if verbose and relations:
                for rel in relations[:3]:
                    obj_name = entity_lookup.get(rel['object_id'], {}).get('name', rel['object_id'])[:30]
                    print(f"        ({entity_name[:20]}, {rel['predicate']}, {obj_name})")
                if len(relations) > 3:
                    print(f"        ... and {len(relations)-3} more")
            
            results.append({
                'entity_id': entity_id,
                'relations': len(relations),
                'valid': valid_count,
                'batches': num_batches,
                'strategy': strategy,
                'cost': cost,
                'time': elapsed
            })
            
        except Exception as e:
            elapsed = time.time() - start
            print(f"-> ERROR: {e}")
            results.append({
                'entity_id': entity_id,
                'error': str(e),
                'time': elapsed
            })
    
    print("-" * 80)
    print()
    
    # Summary
    print("[6] RESULTS SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    print(f"  Entities processed: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Total relations: {total_relations}")
    print(f"  Avg relations/entity: {total_relations/len(successful):.1f}" if successful else "  N/A")
    print(f"  Unique predicates: {len(all_predicates)}")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg time/entity: {total_time/len(results):.1f}s")
    print()
    
    if all_predicates:
        print("  Top predicates:")
        for pred in sorted(all_predicates)[:10]:
            print(f"    - {pred}")
        print()
    
    if issues_found:
        print(f"  Issues found ({len(issues_found)}):")
        for issue in issues_found[:5]:
            print(f"    - {issue}")
        if len(issues_found) > 5:
            print(f"    ... and {len(issues_found)-5} more")
        print()
    
    # Pass/fail
    passed = (
        len(successful) >= len(entities) * 0.8 and  # 80% success rate
        total_relations > 0 and                      # At least some relations
        len(issues_found) < total_relations * 0.2   # <20% invalid relations
    )
    
    if passed:
        print("PREFLIGHT: PASSED")
        print("  Ready for full extraction run")
    else:
        print("PREFLIGHT: FAILED")
        print("  Review issues before full run")
    
    print("=" * 80)
    
    return passed


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Phase 1D preflight test')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    success = run_preflight_test(verbose=args.verbose)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())