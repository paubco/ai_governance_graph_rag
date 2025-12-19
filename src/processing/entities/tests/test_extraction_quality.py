# -*- coding: utf-8 -*-
"""
Extraction quality analysis for Phase 1B pre-entities (v1.2).

Run:
    python -m src.processing.entities.tests.test_extraction_quality --file data/interim/entities/pre_entities.jsonl
    python -m src.processing.entities.tests.test_extraction_quality --file data/interim/entities/pre_entities.jsonl --sample-type Citation --n 15
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict

# Academic types (no domain prefix)
ACADEMIC_TYPES = {"Citation", "Author", "Journal"}


def load_entities(filepath: Path) -> List[Dict]:
    """Load all entities from pre_entities JSONL file."""
    entities = []
    with open(filepath, 'r') as f:
        for line in f:
            record = json.loads(line)
            for e in record.get('entities', []):
                e['_chunk_id'] = record.get('chunk_id', 'unknown')
                entities.append(e)
    return entities


def type_distribution(entities: List[Dict]) -> None:
    """Print type distribution."""
    types = Counter(e.get('type', 'MISSING') for e in entities)
    
    print("\n=== TYPE DISTRIBUTION ===")
    print(f"{'Type':<25} {'Count':>8} {'Pct':>8}")
    print("-" * 45)
    
    total = len(entities)
    for t, c in types.most_common():
        pct = 100 * c / total if total > 0 else 0
        print(f"{t:<25} {c:>8} {pct:>7.1f}%")
    
    print("-" * 45)
    print(f"{'TOTAL':<25} {total:>8}")


def pass_distribution(entities: List[Dict]) -> None:
    """Print semantic vs academic pass distribution."""
    semantic = [e for e in entities if e.get('type') not in ACADEMIC_TYPES]
    academic = [e for e in entities if e.get('type') in ACADEMIC_TYPES]
    
    total = len(entities)
    
    print("\n=== PASS DISTRIBUTION ===")
    print(f"{'Pass':<15} {'Count':>8} {'Pct':>8}")
    print("-" * 35)
    print(f"{'Semantic':<15} {len(semantic):>8} {100*len(semantic)/total:>7.1f}%")
    print(f"{'Academic':<15} {len(academic):>8} {100*len(academic)/total:>7.1f}%")
    print("-" * 35)
    print(f"{'TOTAL':<15} {total:>8}")


def type_by_pass(entities: List[Dict]) -> None:
    """Print types grouped by pass."""
    semantic = [e for e in entities if e.get('type') not in ACADEMIC_TYPES]
    academic = [e for e in entities if e.get('type') in ACADEMIC_TYPES]
    
    print("\n=== SEMANTIC TYPES ===")
    types = Counter(e.get('type') for e in semantic)
    for t, c in types.most_common():
        print(f"  {t:<25} {c:>6}")
    
    print("\n=== ACADEMIC TYPES ===")
    types = Counter(e.get('type') for e in academic)
    for t, c in types.most_common():
        print(f"  {t:<25} {c:>6}")


def sample_by_type(entities: List[Dict], type_name: str, n: int = 10) -> None:
    """Show sample entities of a given type."""
    filtered = [e for e in entities if e.get('type') == type_name]
    
    print(f"\n=== SAMPLE: {type_name} ({len(filtered)} total, showing {min(n, len(filtered))}) ===")
    
    for e in filtered[:n]:
        name = e.get('name', '')[:60]
        desc = (e.get('description') or '')[:70]
        chunk = e.get('_chunk_id', '')
        
        print(f"\n  {name}")
        if desc:
            print(f"    desc: {desc}...")
        print(f"    chunk: {chunk}")


def show_duplicates(entities: List[Dict], min_count: int = 2) -> None:
    """Show entity names that appear multiple times."""
    name_counts = Counter(e.get('name', '') for e in entities)
    
    duplicates = [(name, count) for name, count in name_counts.items() 
                  if count >= min_count and name]
    duplicates.sort(key=lambda x: -x[1])
    
    print(f"\n=== DUPLICATE ENTITIES (appearing {min_count}+ times) ===")
    print(f"{'Count':>6} {'Name':<60}")
    print("-" * 70)
    
    for name, count in duplicates[:30]:
        print(f"{count:>6} {name[:60]}")


def show_short_entities(entities: List[Dict], max_len: int = 3) -> None:
    """Show suspiciously short entity names (likely garbage)."""
    short = [e for e in entities if len(e.get('name', '')) <= max_len]
    
    print(f"\n=== SHORT ENTITIES (len <= {max_len}) - {len(short)} total ===")
    
    by_type = defaultdict(list)
    for e in short:
        by_type[e.get('type', 'MISSING')].append(e.get('name', ''))
    
    for t, names in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"  {t}: {names[:15]}")


def show_rejected_types(entities: List[Dict]) -> None:
    """Show types that were invalid (for debugging prompt issues)."""
    # This would need the raw extraction log - skip for now
    pass


def full_report(entities: List[Dict]) -> None:
    """Print full analysis report."""
    print(f"\n{'='*70}")
    print(f"EXTRACTION QUALITY REPORT (v1.2)")
    print(f"Total entities: {len(entities)}")
    print(f"{'='*70}")
    
    type_distribution(entities)
    pass_distribution(entities)
    type_by_pass(entities)
    show_duplicates(entities)
    show_short_entities(entities)


def main():
    parser = argparse.ArgumentParser(description='Extraction quality analysis')
    parser.add_argument('--file', '-f', required=True, help='Path to pre_entities JSONL')
    parser.add_argument('--sample-type', '-t', help='Sample entities of this type')
    parser.add_argument('--n', type=int, default=10, help='Number of samples')
    parser.add_argument('--duplicates', action='store_true', help='Show duplicates only')
    parser.add_argument('--short', action='store_true', help='Show short entities only')
    
    args = parser.parse_args()
    
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    entities = load_entities(filepath)
    print(f"Loaded {len(entities)} entities from {filepath}")
    
    if args.sample_type:
        sample_by_type(entities, args.sample_type, args.n)
    elif args.duplicates:
        show_duplicates(entities)
    elif args.short:
        show_short_entities(entities)
    else:
        full_report(entities)


if __name__ == '__main__':
    main()