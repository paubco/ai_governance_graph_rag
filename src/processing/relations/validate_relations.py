#!/usr/bin/env python3
"""
Relation Validation - Co-occurrence Check

Validates that relation subject and object entities share at least one chunk.
Removes hallucinated relations where entities never co-occur.

Usage:
    # Validate existing files (standalone)
    python -m src.processing.relations.validate_relations
    
    # Validate specific file
    python -m src.processing.relations.validate_relations --input relations_semantic.jsonl
    
    # Dry run (report only, no output)
    python -m src.processing.relations.validate_relations --dry-run
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Set, List, Tuple

# Project root - standard pattern
# src/processing/relations/validate_relations.py → 4 levels up
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def load_cooccurrence_matrix(cooccurrence_file: Path) -> Dict[str, Set[str]]:
    """
    Load cooccurrence matrix and invert it to entity_id → set(chunk_ids).
    
    Input format: {chunk_id: [entity_ids]}
    Output format: {entity_id: set(chunk_ids)}
    """
    entity_chunks = defaultdict(set)
    
    with open(cooccurrence_file, 'r', encoding='utf-8') as f:
        cooccurrence = json.load(f)
    
    for chunk_id, entity_ids in cooccurrence.items():
        for entity_id in entity_ids:
            entity_chunks[entity_id].add(chunk_id)
    
    return dict(entity_chunks)


def validate_relation(rel: Dict, entity_chunks: Dict[str, Set[str]]) -> Tuple[bool, str]:
    """
    Check if subject and object share at least one chunk.
    
    Returns:
        (is_valid, reason)
    """
    subject_id = rel.get('subject_id', '')
    object_id = rel.get('object_id', '')
    
    # Check entities exist
    if subject_id not in entity_chunks:
        return False, 'subject_not_found'
    if object_id not in entity_chunks:
        return False, 'object_not_found'
    
    # Check co-occurrence
    subj_chunks = entity_chunks[subject_id]
    obj_chunks = entity_chunks[object_id]
    
    if not subj_chunks & obj_chunks:
        return False, 'no_cooccurrence'
    
    return True, 'valid'


def validate_nested_relations(
    input_file: Path,
    entity_chunks: Dict[str, Set[str]],
    output_file: Path = None,
    dry_run: bool = False
) -> Dict:
    """
    Validate relations from nested format (relations_semantic.jsonl).
    Flattens to one relation per line and deduplicates globally.
    
    Each input line is: {"relations": [...], "num_batches": N, ...}
    Output is flat: one relation dict per line.
    """
    stats = {
        'total_entities': 0,
        'total_relations': 0,
        'valid_relations': 0,
        'invalid_relations': 0,
        'after_dedup': 0,
        'reasons': Counter(),
        'predicates_removed': Counter(),
    }
    
    all_valid_relations = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            result = json.loads(line)
            stats['total_entities'] += 1
            
            for rel in result.get('relations', []):
                stats['total_relations'] += 1
                is_valid, reason = validate_relation(rel, entity_chunks)
                
                if is_valid:
                    stats['valid_relations'] += 1
                    all_valid_relations.append(rel)
                else:
                    stats['invalid_relations'] += 1
                    stats['reasons'][reason] += 1
                    stats['predicates_removed'][rel.get('predicate', 'unknown')] += 1
    
    # Global deduplication with chunk_id merging
    deduplicated = deduplicate_relations(all_valid_relations)
    stats['after_dedup'] = len(deduplicated)
    
    # Write output (flat format, one relation per line)
    if not dry_run and output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for rel in deduplicated:
                f.write(json.dumps(rel, ensure_ascii=False) + '\n')
        print(f"Wrote {len(deduplicated):,} deduplicated relations to: {output_file}")
    
    return stats


def validate_flat_relations(
    input_file: Path,
    entity_chunks: Dict[str, Set[str]],
    output_file: Path = None,
    dry_run: bool = False
) -> Dict:
    """
    Validate relations from flat format (relations_discusses.jsonl).
    Deduplicates globally with chunk_id merging.
    
    Each line is a single relation dict.
    """
    stats = {
        'total_relations': 0,
        'valid_relations': 0,
        'invalid_relations': 0,
        'after_dedup': 0,
        'reasons': Counter(),
        'predicates_removed': Counter(),
    }
    
    all_valid_relations = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            rel = json.loads(line)
            stats['total_relations'] += 1
            
            is_valid, reason = validate_relation(rel, entity_chunks)
            
            if is_valid:
                stats['valid_relations'] += 1
                all_valid_relations.append(rel)
            else:
                stats['invalid_relations'] += 1
                stats['reasons'][reason] += 1
                stats['predicates_removed'][rel.get('predicate', 'unknown')] += 1
    
    # Global deduplication with chunk_id merging
    deduplicated = deduplicate_relations(all_valid_relations)
    stats['after_dedup'] = len(deduplicated)
    
    # Write output
    if not dry_run and output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for rel in deduplicated:
                f.write(json.dumps(rel, ensure_ascii=False) + '\n')
        print(f"Wrote {len(deduplicated):,} deduplicated relations to: {output_file}")
    
    return stats


def print_stats(stats: Dict, label: str):
    """Print validation statistics."""
    print()
    print('=' * 60)
    print(f'{label} VALIDATION RESULTS')
    print('=' * 60)
    print()
    
    total = stats['total_relations']
    valid = stats['valid_relations']
    invalid = stats['invalid_relations']
    
    print(f"Total relations:    {total:,}")
    print(f"Valid relations:    {valid:,} ({100*valid/total:.1f}%)" if total else "N/A")
    print(f"Invalid relations:  {invalid:,} ({100*invalid/total:.1f}%)" if total else "N/A")
    
    if 'after_dedup' in stats:
        deduped = stats['after_dedup']
        print(f"After dedup:        {deduped:,} ({100*deduped/valid:.1f}% of valid)")
    
    print()
    
    if stats['reasons']:
        print("Rejection reasons:")
        for reason, count in stats['reasons'].most_common():
            print(f"  {reason:25s} {count:,}")
        print()
    
    if stats['predicates_removed']:
        print("Top predicates removed:")
        for pred, count in stats['predicates_removed'].most_common(10):
            print(f"  {pred:25s} {count:,}")
        print()
    
    print('=' * 60)


def deduplicate_relations(relations: List[Dict]) -> List[Dict]:
    """
    Global deduplication with chunk_id merging.
    
    Same (subject_id, predicate, object_id) → merge chunk_ids lists.
    """
    merged = {}
    
    for rel in relations:
        key = (rel['subject_id'], rel['predicate'], rel['object_id'])
        
        if key not in merged:
            merged[key] = {
                'subject_id': rel['subject_id'],
                'predicate': rel['predicate'],
                'object_id': rel['object_id'],
                'chunk_ids': list(rel.get('chunk_ids', [])),
                'extraction_strategy': rel.get('extraction_strategy', 'semantic')
            }
        else:
            # Merge chunk_ids (deduplicate)
            existing_chunks = set(merged[key]['chunk_ids'])
            for cid in rel.get('chunk_ids', []):
                if cid not in existing_chunks:
                    merged[key]['chunk_ids'].append(cid)
                    existing_chunks.add(cid)
    
    return list(merged.values())


def main():
    parser = argparse.ArgumentParser(description='Validate relations by co-occurrence')
    parser.add_argument('--input', type=str, help='Input relations file')
    parser.add_argument('--output', type=str, help='Output file for validated relations')
    parser.add_argument('--dry-run', action='store_true', help='Report only, no output')
    parser.add_argument('--flat', action='store_true', help='Input is flat format (one relation per line)')
    args = parser.parse_args()
    
    # Default paths - use cooccurrence_full.json (all entities)
    cooccurrence_file = PROJECT_ROOT / "data/interim/entities/cooccurrence_full.json"
    semantic_input = PROJECT_ROOT / "data/processed/relations/relations_semantic.jsonl"
    citation_input = PROJECT_ROOT / "data/processed/relations/relations_discusses.jsonl"
    
    # Check prerequisites
    if not cooccurrence_file.exists():
        print(f"ERROR: Cooccurrence matrix not found: {cooccurrence_file}")
        sys.exit(1)
    
    print("Loading cooccurrence matrix...")
    entity_chunks = load_cooccurrence_matrix(cooccurrence_file)
    print(f"  Loaded {len(entity_chunks):,} entities")
    
    # Single file mode
    if args.input:
        input_path = Path(args.input)
        output_path = Path(args.output) if args.output else input_path.with_suffix('.validated.jsonl')
        
        if args.flat:
            stats = validate_flat_relations(input_path, entity_chunks, output_path, args.dry_run)
        else:
            stats = validate_nested_relations(input_path, entity_chunks, output_path, args.dry_run)
        
        print_stats(stats, input_path.name)
        return
    
    # Default: validate both semantic and citation
    all_stats = {'valid': 0, 'invalid': 0, 'total': 0, 'dedup': 0}
    
    # Semantic track (nested format)
    if semantic_input.exists():
        output = semantic_input.with_name('relations_semantic_validated.jsonl')
        stats = validate_nested_relations(semantic_input, entity_chunks, output, args.dry_run)
        print_stats(stats, "SEMANTIC TRACK")
        all_stats['valid'] += stats['valid_relations']
        all_stats['invalid'] += stats['invalid_relations']
        all_stats['total'] += stats['total_relations']
        all_stats['dedup'] += stats.get('after_dedup', stats['valid_relations'])
    else:
        print(f"Skipping semantic (not found): {semantic_input}")
    
    # Citation track (flat format)
    if citation_input.exists():
        output = citation_input.with_name('relations_discusses_validated.jsonl')
        stats = validate_flat_relations(citation_input, entity_chunks, output, args.dry_run)
        print_stats(stats, "CITATION TRACK")
        all_stats['valid'] += stats['valid_relations']
        all_stats['invalid'] += stats['invalid_relations']
        all_stats['total'] += stats['total_relations']
        all_stats['dedup'] += stats.get('after_dedup', stats['valid_relations'])
    else:
        print(f"Skipping citation (not found): {citation_input}")
    
    # Combined summary
    print()
    print('=' * 60)
    print("COMBINED SUMMARY")
    print('=' * 60)
    print(f"Total relations:  {all_stats['total']:,}")
    print(f"Valid:            {all_stats['valid']:,} ({100*all_stats['valid']/all_stats['total']:.1f}%)" if all_stats['total'] else "N/A")
    print(f"After dedup:      {all_stats['dedup']:,} ({100*all_stats['dedup']/all_stats['valid']:.1f}% of valid)" if all_stats['valid'] else "N/A")
    print(f"Removed (invalid):{all_stats['invalid']:,}")
    print(f"Removed (dupes):  {all_stats['valid'] - all_stats['dedup']:,}")
    print('=' * 60)


if __name__ == "__main__":
    main()