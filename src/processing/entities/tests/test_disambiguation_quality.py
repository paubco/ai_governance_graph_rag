# -*- coding: utf-8 -*-
"""
Phase

Comprehensive analysis of entity disambiguation outputs including:
- Alias distribution and largest clusters
- Entity type distribution
- PART_OF and SAME_AS relations quality
- Quality checks (short names, numeric patterns, high chunk counts)
- Merge quality validation

"""
"""
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple


# Paths
DATA_DIR = Path(__file__).resolve().parents[3] / 'data'
SEMANTIC_FILE = DATA_DIR / 'processed' / 'entities' / 'entities_semantic.jsonl'
METADATA_FILE = DATA_DIR / 'processed' / 'entities' / 'entities_metadata.jsonl'
ALIASES_FILE = DATA_DIR / 'processed' / 'entities' / 'aliases.json'
PART_OF_FILE = DATA_DIR / 'processed' / 'relations' / 'part_of_relations.jsonl'
SAME_AS_FILE = DATA_DIR / 'processed' / 'relations' / 'same_as_relations.jsonl'
SAMEJUDGE_PROGRESS = DATA_DIR / 'interim' / 'entities' / 'samejudge_progress.json'


def load_jsonl(path: Path) -> list:
    """Load JSONL file."""
    items = []
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
    return items


def analyze_aliases(aliases: dict) -> dict:
    """Analyze alias distribution. Returns metrics dict."""
    print("\n[1] ALIASES ANALYSIS")
    print("-" * 50)
    
    if not aliases:
        print("  No aliases found")
        return {}
    
    total_aliases = sum(len(v) for v in aliases.values())
    max_aliases = max(len(v) for v in aliases.values())
    alias_counts = [len(v) for v in aliases.values()]
    
    metrics = {
        'canonical_count': len(aliases),
        'total_aliases': total_aliases,
        'avg_aliases': total_aliases / len(aliases),
        'max_aliases': max_aliases,
        'median_aliases': sorted(alias_counts)[len(alias_counts)//2],
    }
    
    print(f"Total canonical entities with aliases: {metrics['canonical_count']}")
    print(f"Total aliases: {metrics['total_aliases']}")
    print(f"Avg aliases per canonical: {metrics['avg_aliases']:.1f}")
    print(f"Max aliases: {metrics['max_aliases']}")
    print(f"Median aliases: {metrics['median_aliases']}")
    
    # Distribution
    print(f"\nAlias count distribution:")
    for threshold in [1, 2, 5, 10, 15, 20, 50]:
        count = len([c for c in alias_counts if c >= threshold])
        print(f"  >= {threshold}: {count} canonicals")
    
    print("\nTop 10 canonicals by alias count:")
    top = sorted(aliases.items(), key=lambda x: -len(x[1]))[:10]
    for name, alias_list in top:
        print(f"  {name:<40} → {len(alias_list)} aliases")
        for a in alias_list[:3]:
            print(f"    - {a}")
        if len(alias_list) > 3:
            print(f"    ... +{len(alias_list)-3} more")
    
    return metrics


def analyze_entities(semantic: list, metadata: list) -> dict:
    """Analyze entity type distribution. Returns metrics dict."""
    print("\n[2] SEMANTIC ENTITIES BY TYPE")
    print("-" * 50)
    
    type_counts = defaultdict(int)
    for e in semantic:
        type_counts[e.get('type', 'Unknown')] += 1
    
    print(f"Total: {len(semantic)}")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:<25} {c:>5} ({100*c/len(semantic):>5.1f}%)")
    
    print("\n[2b] METADATA ENTITIES BY TYPE")
    print("-" * 50)
    
    meta_counts = defaultdict(int)
    for e in metadata:
        meta_counts[e.get('type', 'Unknown')] += 1
    
    print(f"Total: {len(metadata)}")
    for t, c in sorted(meta_counts.items(), key=lambda x: -x[1]):
        pct = 100*c/len(metadata) if metadata else 0
        print(f"  {t:<25} {c:>5} ({pct:>5.1f}%)")
    
    return {
        'semantic_total': len(semantic),
        'semantic_types': dict(type_counts),
        'metadata_total': len(metadata),
        'metadata_types': dict(meta_counts),
    }


def analyze_relations(part_of: list, same_as: list) -> dict:
    """Analyze PART_OF and SAME_AS relations quality."""
    print("\n[3] PART_OF RELATIONS")
    print("-" * 50)
    print(f"Total: {len(part_of)}")
    
    # Analyze sources
    sources = defaultdict(int)
    for r in part_of:
        sources[r.get('source', 'unknown')] += 1
    
    print(f"\nBy source:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count}")
    
    # Check for cross-chunk issues
    cross_chunk = [r for r in part_of if 'cross_chunk' in r.get('source', '')]
    print(f"\nCross-chunk relations: {len(cross_chunk)} ({100*len(cross_chunk)/max(1,len(part_of)):.1f}%)")
    
    if part_of:
        print(f"\nSample relations:")
        for r in part_of[:8]:
            # Handle multiple possible field names
            subj = r.get('subject', r.get('source_name', r.get('child', '?')))[:35]
            obj = r.get('object', r.get('target_name', r.get('parent', '?')))[:35]
            src = r.get('source', '?')
            print(f"  {subj:<35} → {obj} [{src}]")
    
    print("\n[4] SAME_AS RELATIONS")
    print("-" * 50)
    print(f"Total: {len(same_as)}")
    
    if same_as:
        # Check confidence distribution
        confidences = [r.get('confidence', 1.0) for r in same_as]
        print(f"\nConfidence distribution:")
        print(f"  Min: {min(confidences):.2f}, Max: {max(confidences):.2f}")
        print(f"  < 1.0: {len([c for c in confidences if c < 1.0])} relations")
        
        print(f"\nSample relations:")
        for r in same_as[:8]:
            subj = r.get('subject', r.get('source_name', r.get('entity1', '?')))[:35]
            obj = r.get('object', r.get('target_name', r.get('entity2', '?')))[:35]
            conf = r.get('confidence', 1.0)
            print(f"  {subj:<35} ↔ {obj} [{conf:.2f}]")
    
    return {
        'part_of_total': len(part_of),
        'part_of_cross_chunk': len(cross_chunk),
        'same_as_total': len(same_as),
    }


def quality_checks(semantic: list) -> dict:
    """Run quality checks on semantic entities."""
    print("\n[5] QUALITY CHECKS")
    print("-" * 50)
    
    issues = {}
    
    # High chunk counts
    print("\nHigh chunk counts (top 10):")
    chunk_counts = [(e['name'], len(e.get('chunk_ids', []))) for e in semantic]
    top_chunks = sorted(chunk_counts, key=lambda x: -x[1])[:10]
    for name, count in top_chunks:
        print(f"  {name:<50} {count} chunks")
    issues['max_chunk_count'] = top_chunks[0][1] if top_chunks else 0
    
    # Short names
    short_names = [e['name'] for e in semantic if len(e['name']) <= 3]
    print(f"\nShort names (<=3 chars):")
    print(f"  Count: {len(short_names)}")
    # Separate valid from garbage
    valid_short = [n for n in short_names if n.upper() in ['AI', 'EU', 'UK', 'US', 'ML', 'NLP', 'GDP', 'CEO', 'CTO']]
    garbage_short = [n for n in short_names if n not in valid_short]
    print(f"  Valid (AI, EU, etc): {len(valid_short)}")
    print(f"  Garbage: {len(garbage_short)}")
    print(f"  Garbage examples: {garbage_short[:15]}")
    issues['short_names_garbage'] = len(garbage_short)
    
    # Numeric patterns
    numeric = [e['name'] for e in semantic if re.match(r'^[\d\.\-\(\)\,]+$', e['name'])]
    print(f"\nNumeric patterns:")
    print(f"  Count: {len(numeric)}")
    print(f"  Examples: {numeric[:10]}")
    issues['numeric_garbage'] = len(numeric)
    
    # Very long names (potential garbage)
    long_names = [(e['name'], len(e['name'])) for e in semantic if len(e['name']) > 80]
    if long_names:
        print(f"\nVery long names (>80 chars):")
        print(f"  Count: {len(long_names)}")
        for name, length in sorted(long_names, key=lambda x: -x[1])[:5]:
            print(f"  [{length}] {name[:80]}...")
    issues['long_names'] = len(long_names)
    
    # Names with special characters
    special = [e['name'] for e in semantic if re.search(r'[^\w\s\-\'\.\,\(\)]', e['name'])]
    if special:
        print(f"\nSpecial character names:")
        print(f"  Count: {len(special)}")
        print(f"  Examples: {special[:10]}")
    issues['special_chars'] = len(special)
    
    return issues


def analyze_samejudge_progress() -> dict:
    """Analyze SameJudge LLM decisions if available."""
    print("\n[6] SAMEJUDGE LLM ANALYSIS")
    print("-" * 50)
    
    if not SAMEJUDGE_PROGRESS.exists():
        print("  No SameJudge progress file found")
        return {}
    
    progress = json.load(open(SAMEJUDGE_PROGRESS))
    
    total = progress.get('total', 0)
    processed = progress.get('processed', 0)
    merge_pairs = progress.get('merge_pairs', [])
    
    print(f"Total pairs: {total}")
    print(f"Processed: {processed}")
    print(f"Merges approved: {len(merge_pairs)} ({100*len(merge_pairs)/max(1,processed):.1f}%)")
    
    if merge_pairs:
        # Similarity distribution of approved merges
        sims = [p.get('similarity', 0) for p in merge_pairs]
        print(f"\nApproved merge similarity distribution:")
        print(f"  Min: {min(sims):.3f}, Max: {max(sims):.3f}")
        for threshold in [0.88, 0.89, 0.90, 0.91, 0.92, 0.95]:
            count = len([s for s in sims if s < threshold])
            print(f"  < {threshold}: {count}")
    
    return {
        'total_pairs': total,
        'processed': processed,
        'merges_approved': len(merge_pairs),
        'approval_rate': len(merge_pairs) / max(1, processed),
    }


def summary_report(alias_metrics: dict, entity_metrics: dict, 
                   relation_metrics: dict, quality_issues: dict,
                   samejudge_metrics: dict) -> None:
    """Print summary report."""
    print("\n" + "=" * 70)
    print(" SUMMARY REPORT")
    print("=" * 70)
    
    print("\n📊 COUNTS:")
    print(f"  Semantic entities: {entity_metrics.get('semantic_total', 0):,}")
    print(f"  Metadata entities: {entity_metrics.get('metadata_total', 0):,}")
    print(f"  Aliases tracked: {alias_metrics.get('canonical_count', 0):,}")
    print(f"  PART_OF relations: {relation_metrics.get('part_of_total', 0):,}")
    print(f"  SAME_AS relations: {relation_metrics.get('same_as_total', 0):,}")
    
    print("\n✅ QUALITY METRICS:")
    print(f"  Max aliases per entity: {alias_metrics.get('max_aliases', 0)}")
    if samejudge_metrics:
        print(f"  LLM approval rate: {100*samejudge_metrics.get('approval_rate', 0):.1f}%")
    
    print("\n⚠️  ISSUES TO ADDRESS:")
    print(f"  Garbage short names: {quality_issues.get('short_names_garbage', 0)}")
    print(f"  Numeric garbage: {quality_issues.get('numeric_garbage', 0)}")
    print(f"  Cross-chunk PART_OF: {relation_metrics.get('part_of_cross_chunk', 0)}")
    
    # Quality score
    max_aliases = alias_metrics.get('max_aliases', 999)
    if max_aliases <= 20:
        print("\n🎯 ALIAS QUALITY: EXCELLENT (max ≤ 20)")
    elif max_aliases <= 50:
        print("\n🎯 ALIAS QUALITY: GOOD (max ≤ 50)")
    else:
        print("\n🎯 ALIAS QUALITY: NEEDS WORK (max > 50)")


def main():
    """Run full analysis."""
    print("=" * 70)
    print(" PHASE 1C DISAMBIGUATION OUTPUT ANALYSIS")
    print("=" * 70)
    
    # Load data
    aliases = {}
    if ALIASES_FILE.exists():
        aliases = json.load(open(ALIASES_FILE))
    
    semantic = load_jsonl(SEMANTIC_FILE)
    metadata = load_jsonl(METADATA_FILE)
    part_of = load_jsonl(PART_OF_FILE)
    same_as = load_jsonl(SAME_AS_FILE)
    
    # Summary
    print(f"\nFiles loaded:")
    print(f"  Semantic entities: {len(semantic)}")
    print(f"  Metadata entities: {len(metadata)}")
    print(f"  Aliases: {len(aliases)} canonicals")
    print(f"  PART_OF: {len(part_of)}")
    print(f"  SAME_AS: {len(same_as)}")
    
    # Run analyses
    alias_metrics = analyze_aliases(aliases)
    entity_metrics = analyze_entities(semantic, metadata)
    relation_metrics = analyze_relations(part_of, same_as)
    quality_issues = quality_checks(semantic)
    samejudge_metrics = analyze_samejudge_progress()
    
    # Summary
    summary_report(alias_metrics, entity_metrics, relation_metrics, 
                   quality_issues, samejudge_metrics)
    
    print("\n" + "=" * 70)
    print(" ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()