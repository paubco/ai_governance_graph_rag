# -*- coding: utf-8 -*-
"""
Phase 1C Evaluation Script - Data-Driven Quality Metrics

Outputs human-readable metrics for manual evaluation:
1. Filter effectiveness by type
2. Random samples of filtered vs kept entities
3. Relation quality samples (PART_OF, SAME_AS)
4. Suspicious patterns detection
5. Deduplication effectiveness

Usage:
    python -m src.processing.entities.evaluate_phase1c
    python -m src.processing.entities.evaluate_phase1c --sample 20
"""

import json
import random
import argparse
from collections import Counter, defaultdict
from pathlib import Path

# Imports
from src.processing.entities.pre_entity_filter import PreEntityFilter, load_chunks_as_dict, is_garbage
from src.processing.entities.semantic_disambiguator import ExactDeduplicator, route_by_type
from src.processing.entities.metadata_disambiguator import MetadataDisambiguator


def load_entities(filepath: str) -> list:
    """Load entities with chunk_id inheritance."""
    entities = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                chunk_data = json.loads(line)
                chunk_id = chunk_data.get('chunk_id', '')
                for e in chunk_data.get('entities', []):
                    e['chunk_id'] = chunk_id
                    entities.append(e)
    return entities


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)


def evaluate_filtering(entities: list, chunks: dict, sample_size: int = 10):
    """Evaluate pre-entity filter effectiveness."""
    print_section("1. FILTER EFFECTIVENESS")
    
    # Track what gets filtered by type
    filtered_by_type = defaultdict(list)
    kept_by_type = defaultdict(list)
    
    for e in entities:
        name = e.get('name', '')
        etype = e.get('type', 'Unknown')
        if is_garbage(name, etype):
            filtered_by_type[etype].append(name)
        else:
            kept_by_type[etype].append(name)
    
    # Summary table
    print("\nFilter Summary by Type:")
    print(f"{'Type':<20} {'Input':>8} {'Filtered':>10} {'Kept':>8} {'Filter%':>8}")
    print("-" * 60)
    
    all_types = set(filtered_by_type.keys()) | set(kept_by_type.keys())
    for etype in sorted(all_types):
        filtered = len(filtered_by_type[etype])
        kept = len(kept_by_type[etype])
        total = filtered + kept
        pct = (filtered / total * 100) if total > 0 else 0
        flag = " ⚠️" if pct > 20 else ""  # Flag high filter rates
        print(f"{etype:<20} {total:>8} {filtered:>10} {kept:>8} {pct:>7.1f}%{flag}")
    
    # Show samples of what's being filtered
    print(f"\n\nSample FILTERED entities (random {sample_size} per type):")
    for etype in ['Document', 'DocumentSection', 'Technology', 'Citation']:
        if filtered_by_type[etype]:
            samples = random.sample(filtered_by_type[etype], 
                                   min(sample_size, len(filtered_by_type[etype])))
            print(f"\n  {etype} ({len(filtered_by_type[etype])} filtered):")
            for name in samples:
                print(f"    ✗ '{name}'")
    
    # Show samples of what's being kept (for validation)
    print(f"\n\nSample KEPT entities (random {sample_size} per type):")
    for etype in ['Document', 'DocumentSection']:
        if kept_by_type[etype]:
            samples = random.sample(kept_by_type[etype],
                                   min(sample_size, len(kept_by_type[etype])))
            print(f"\n  {etype} ({len(kept_by_type[etype])} kept):")
            for name in samples:
                print(f"    ✓ '{name}'")
    
    return filtered_by_type, kept_by_type


def evaluate_deduplication(semantic_entities: list, sample_size: int = 10):
    """Evaluate deduplication effectiveness."""
    print_section("2. DEDUPLICATION EFFECTIVENESS")
    
    dedup = ExactDeduplicator()
    canonical, aliases = dedup.deduplicate(semantic_entities)
    
    # Stats
    reduction = len(semantic_entities) - len(canonical)
    reduction_pct = (reduction / len(semantic_entities) * 100) if semantic_entities else 0
    
    print(f"\nDeduplication Stats:")
    print(f"  Input:     {len(semantic_entities):,}")
    print(f"  Output:    {len(canonical):,}")
    print(f"  Reduced:   {reduction:,} ({reduction_pct:.1f}%)")
    print(f"  Aliases:   {len(aliases):,}")
    
    # Show high-merge entities (entities that absorbed many duplicates)
    high_merge = sorted(canonical, key=lambda x: x.get('merge_count', 1), reverse=True)[:sample_size]
    print(f"\n\nTop {sample_size} entities by merge count:")
    for e in high_merge:
        print(f"  {e.get('merge_count', 1):4d}x  '{e['name']}' ({e['type']})")
        if e.get('aliases'):
            print(f"         aliases: {e['aliases'][:3]}")
    
    # Type distribution after dedup
    print("\n\nType distribution after dedup:")
    type_counts = Counter(e['type'] for e in canonical)
    for etype, count in type_counts.most_common():
        print(f"  {etype:<25} {count:>6}")
    
    return canonical, aliases


def evaluate_relations(semantic_entities: list, metadata_entities: list, sample_size: int = 15):
    """Evaluate PART_OF and SAME_AS relation quality."""
    print_section("3. RELATION QUALITY")
    
    # Add entity IDs
    for i, e in enumerate(semantic_entities):
        e['entity_id'] = f'ent_{i:05d}'
    
    # Process metadata
    disambiguator = MetadataDisambiguator(semantic_entities=semantic_entities)
    meta_out, part_of, same_as = disambiguator.process(metadata_entities)
    
    # Stats
    print(f"\nRelation Counts:")
    print(f"  PART_OF (DocumentSection → Document): {len(part_of):,}")
    print(f"  SAME_AS (Document ↔ Regulation):      {len(same_as):,}")
    
    # PART_OF samples - check for quality
    print(f"\n\nPART_OF Samples (random {sample_size}):")
    if part_of:
        samples = random.sample(part_of, min(sample_size, len(part_of)))
        for r in samples:
            # Quality check: is subject actually a subsection of object?
            subj = r['subject']
            obj = r['object']
            looks_valid = obj.lower() in subj.lower() or len(obj) < len(subj)
            flag = "✓" if looks_valid else "⚠️"
            print(f"  {flag} '{subj[:45]}' → '{obj[:25]}'")
    
    # SAME_AS samples
    print(f"\n\nSAME_AS Samples (random {sample_size}):")
    if same_as:
        samples = random.sample(same_as, min(sample_size, len(same_as)))
        for r in samples:
            conf = r.get('confidence', 0)
            flag = "✓" if conf >= 0.95 else "~" if conf >= 0.85 else "⚠️"
            print(f"  {flag} '{r['subject'][:30]}' ↔ '{r['object'][:25]}' (conf={conf:.2f})")
    
    # Check for potential issues
    print("\n\nPotential Issues:")
    
    # Self-references in PART_OF
    self_refs = [r for r in part_of if r['subject'].lower() == r['object'].lower()]
    print(f"  Self-references in PART_OF: {len(self_refs)}")
    if self_refs:
        for r in self_refs[:5]:
            print(f"    ⚠️ '{r['subject']}' → '{r['object']}'")
    
    # Very short objects in PART_OF (suspicious)
    short_objs = [r for r in part_of if len(r['object']) <= 5]
    print(f"  PART_OF with short object (<=5 chars): {len(short_objs)}")
    if short_objs:
        for r in short_objs[:5]:
            print(f"    ⚠️ '{r['subject'][:40]}' → '{r['object']}'")
    
    return part_of, same_as


def evaluate_suspicious_patterns(entities: list, sample_size: int = 20):
    """Detect suspicious patterns that might indicate extraction issues."""
    print_section("4. SUSPICIOUS PATTERNS")
    
    # Very short entity names by type
    print("\nShort entity names (<=3 chars) by type:")
    short_by_type = defaultdict(list)
    for e in entities:
        if len(e['name']) <= 3:
            short_by_type[e['type']].append(e['name'])
    
    for etype, names in sorted(short_by_type.items(), key=lambda x: -len(x[1])):
        unique = set(names)
        print(f"  {etype}: {len(names)} total, {len(unique)} unique")
        if len(unique) <= 10:
            print(f"    → {sorted(unique)}")
    
    # Duplicate names across types (potential misclassification)
    print(f"\n\nNames appearing in MULTIPLE types:")
    name_types = defaultdict(set)
    for e in entities:
        name_types[e['name']].add(e['type'])
    
    multi_type = {name: types for name, types in name_types.items() if len(types) > 1}
    print(f"  Total: {len(multi_type)} names appear in multiple types")
    
    # Show examples
    samples = list(multi_type.items())[:sample_size]
    for name, types in samples:
        print(f"    '{name[:40]}' → {types}")
    
    # Very frequent entities (might indicate extraction issues)
    print(f"\n\nMost frequent entities (potential over-extraction):")
    name_counts = Counter(e['name'] for e in entities)
    for name, count in name_counts.most_common(15):
        if count > 50:
            types = set(e['type'] for e in entities if e['name'] == name)
            print(f"  {count:5d}x '{name[:40]}' → {types}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Phase 1C quality')
    parser.add_argument('--sample', type=int, default=10, help='Sample size for examples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--input', type=str, 
                       default='data/interim/entities/pre_entities.jsonl',
                       help='Input pre-entities file')
    parser.add_argument('--chunks', type=str,
                       default='data/processed/chunks/chunks_embedded.jsonl',
                       help='Chunks file for provenance')
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("\n" + "="*70)
    print(" PHASE 1C EVALUATION - Data-Driven Quality Metrics")
    print("="*70)
    print(f"\nInput: {args.input}")
    print(f"Sample size: {args.sample}")
    print(f"Seed: {args.seed}")
    
    # Load data
    print("\nLoading data...")
    entities = load_entities(args.input)
    chunks = load_chunks_as_dict(args.chunks)
    print(f"  Loaded {len(entities):,} entities from {len(chunks):,} chunks")
    
    # 1. Filter evaluation
    filtered, kept = evaluate_filtering(entities, chunks, args.sample)
    
    # Get clean entities for further evaluation
    filt = PreEntityFilter(chunks=chunks)
    clean, _ = filt.filter(entities)
    semantic, metadata = route_by_type(clean)
    
    # 2. Deduplication evaluation
    canonical, aliases = evaluate_deduplication(semantic, args.sample)
    
    # 3. Relation evaluation
    part_of, same_as = evaluate_relations(canonical, metadata, args.sample)
    
    # 4. Suspicious patterns
    evaluate_suspicious_patterns(entities, args.sample)
    
    # Summary
    print_section("5. SUMMARY")
    print(f"""
Pipeline Flow:
  Raw entities:        {len(entities):>8,}
  After filter:        {len(clean):>8,}  (-{len(entities)-len(clean):,} filtered)
  Semantic path:       {len(semantic):>8,}
  Metadata path:       {len(metadata):>8,}
  After dedup:         {len(canonical):>8,}  (-{len(semantic)-len(canonical):,} merged)
  
Relations:
  PART_OF:             {len(part_of):>8,}
  SAME_AS:             {len(same_as):>8,}

Quality Checks:
  - Review FILTERED samples: Are we removing things we shouldn't?
  - Review KEPT samples: Are we keeping garbage?
  - Review PART_OF samples: Do the links make sense?
  - Review SAME_AS samples: Are Document↔Regulation pairs correct?
  - Check suspicious patterns: Multi-type names, over-extraction
""")


if __name__ == '__main__':
    main()