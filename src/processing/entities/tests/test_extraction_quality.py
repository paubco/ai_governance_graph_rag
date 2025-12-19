# -*- coding: utf-8 -*-
"""
Extraction quality analysis for Phase 1B pre-entities.

Manual evaluation utilities for inspecting entity extraction results.
Not automated tests - these are for human review during development.

Run:
    python -m src.processing.entities.tests.test_extraction_quality --file data/interim/entities/pre_entities.jsonl
    python -m src.processing.entities.tests.test_extraction_quality --file data/interim/entities/pre_entities.jsonl --crosstab
    python -m src.processing.entities.tests.test_extraction_quality --file data/interim/entities/pre_entities.jsonl --sample Citation 10

References:
    - Phase 1B spec
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Optional


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
    print(f"{'Type':<20} {'Count':>8} {'Pct':>8}")
    print("-" * 40)
    
    total = len(entities)
    for t, c in types.most_common():
        pct = 100 * c / total if total > 0 else 0
        print(f"{t:<20} {c:>8} {pct:>7.1f}%")
    
    print("-" * 40)
    print(f"{'TOTAL':<20} {total:>8}")


def domain_distribution(entities: List[Dict]) -> None:
    """Print domain distribution."""
    domains = Counter(e.get('domain') or 'NO_DOMAIN' for e in entities)
    
    print("\n=== DOMAIN DISTRIBUTION ===")
    print(f"{'Domain':<20} {'Count':>8} {'Pct':>8}")
    print("-" * 40)
    
    total = len(entities)
    for d, c in domains.most_common():
        pct = 100 * c / total if total > 0 else 0
        print(f"{d:<20} {c:>8} {pct:>7.1f}%")


def type_domain_crosstab(entities: List[Dict]) -> None:
    """Print type x domain crosstab matrix."""
    cross = Counter()
    for e in entities:
        t = e.get('type', 'MISSING')
        d = e.get('domain') or 'NO_DOMAIN'
        cross[(t, d)] += 1
    
    # Get unique domains and types
    domains = ['Regulatory', 'Political', 'Technical', 'General', 'NO_DOMAIN']
    types = sorted(set(t for t, d in cross.keys()))
    
    print("\n=== TYPE x DOMAIN CROSSTAB ===")
    header = f"{'Type':<15} | " + " | ".join(f"{d:<10}" for d in domains) + " | Total"
    print(header)
    print("-" * len(header))
    
    for t in types:
        row = [cross.get((t, d), 0) for d in domains]
        row_total = sum(row)
        row_str = " | ".join(f"{c:<10}" for c in row)
        print(f"{t:<15} | {row_str} | {row_total}")


def sample_by_type(entities: List[Dict], type_name: str, n: int = 10) -> None:
    """Show sample entities of a given type."""
    filtered = [e for e in entities if e.get('type') == type_name]
    
    print(f"\n=== SAMPLE: {type_name} ({len(filtered)} total, showing {min(n, len(filtered))}) ===")
    
    for e in filtered[:n]:
        domain = e.get('domain') or 'NO_DOMAIN'
        name = e.get('name', '')[:60]
        desc = (e.get('description') or '')[:80]
        chunk = e.get('_chunk_id', '')
        
        print(f"\n  [{domain}] {name}")
        if desc:
            print(f"    desc: {desc}...")
        print(f"    chunk: {chunk}")


def sample_by_domain(entities: List[Dict], domain_name: str, n: int = 10) -> None:
    """Show sample entities of a given domain."""
    if domain_name == 'NO_DOMAIN':
        filtered = [e for e in entities if not e.get('domain')]
    else:
        filtered = [e for e in entities if e.get('domain') == domain_name]
    
    print(f"\n=== SAMPLE: {domain_name} ({len(filtered)} total, showing {min(n, len(filtered))}) ===")
    
    for e in filtered[:n]:
        t = e.get('type', 'MISSING')
        name = e.get('name', '')[:60]
        desc = (e.get('description') or '')[:80]
        
        print(f"\n  [{t}] {name}")
        if desc:
            print(f"    desc: {desc}...")


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
        print(f"\n  {t}: {names[:20]}")


def full_report(entities: List[Dict]) -> None:
    """Print full analysis report."""
    print(f"\n{'='*70}")
    print(f"EXTRACTION QUALITY REPORT")
    print(f"Total entities: {len(entities)}")
    print(f"{'='*70}")
    
    type_distribution(entities)
    domain_distribution(entities)
    type_domain_crosstab(entities)
    show_duplicates(entities)
    show_short_entities(entities)


def main():
    parser = argparse.ArgumentParser(
        description='Extraction quality analysis for Phase 1B',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full report
    python -m src.processing.entities.tests.test_extraction_quality --file data/interim/entities/pre_entities.jsonl
    
    # Just crosstab
    python -m src.processing.entities.tests.test_extraction_quality --file data/interim/entities/pre_entities.jsonl --crosstab
    
    # Sample specific type
    python -m src.processing.entities.tests.test_extraction_quality --file data/interim/entities/pre_entities.jsonl --sample-type Citation --n 15
    
    # Sample specific domain
    python -m src.processing.entities.tests.test_extraction_quality --file data/interim/entities/pre_entities.jsonl --sample-domain Regulatory --n 10
    
    # Show duplicates
    python -m src.processing.entities.tests.test_extraction_quality --file data/interim/entities/pre_entities.jsonl --duplicates
        """
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        required=True,
        help='Path to pre_entities JSONL file'
    )
    parser.add_argument(
        '--crosstab',
        action='store_true',
        help='Show type x domain crosstab only'
    )
    parser.add_argument(
        '--sample-type',
        type=str,
        help='Show samples of specific type'
    )
    parser.add_argument(
        '--sample-domain',
        type=str,
        help='Show samples of specific domain'
    )
    parser.add_argument(
        '--n',
        type=int,
        default=10,
        help='Number of samples to show (default: 10)'
    )
    parser.add_argument(
        '--duplicates',
        action='store_true',
        help='Show duplicate entity names'
    )
    parser.add_argument(
        '--short',
        action='store_true',
        help='Show suspiciously short entity names'
    )
    
    args = parser.parse_args()
    
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    entities = load_entities(filepath)
    print(f"Loaded {len(entities)} entities from {filepath}")
    
    if args.crosstab:
        type_domain_crosstab(entities)
    elif args.sample_type:
        sample_by_type(entities, args.sample_type, args.n)
    elif args.sample_domain:
        sample_by_domain(entities, args.sample_domain, args.n)
    elif args.duplicates:
        show_duplicates(entities)
    elif args.short:
        show_short_entities(entities)
    else:
        full_report(entities)


if __name__ == "__main__":
    main()