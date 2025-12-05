# -*- coding: utf-8 -*-
"""
Normalize extracted relations by converting entity names to hash IDs.

Processes raw relation triplets from relation extraction output, replacing
entity names with deterministic hash-based IDs for graph construction. Performs
exact string lookup using entity_name_to_id.json, deduplicates relations with
identical (subject_id, predicate, object_id) triplets by merging chunk_ids,
and generates quality reports flagging self-references, bidirectional pairs,
and unmatched entities. Outputs human-readable JSON with both names and IDs
for Neo4j upload.

Input files:
    data/interim/entities/entity_name_to_id.json (canonical lookup)
    data/interim/relations/relations_output.jsonl (~221K raw relations)

Output files:
    data/interim/relations/relations_normalized.json (~105K normalized)
    data/interim/relations/unmatched_entities.json (quality report)

Runtime: ~10-15 minutes (full dataset)

Example:
    # Standard normalization (canonical names only)
    python src/processing/relations/normalize_relations.py
    
    # With alias expansion (paraphrase-tolerant)
    python src/processing/relations/normalize_relations.py --use-aliases

Warning:
    This is a POST-HOC FIX. Proper architecture would generate entity IDs
    BEFORE relation extraction. See ARCHITECTURE.md § 3.3 for details.
"""

# Standard library
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
from tqdm import tqdm


def load_entity_lookup(lookup_path: Path) -> Dict[str, str]:
    """
    Load entity name to ID lookup dictionary.
    
    Args:
        lookup_path: Path to entity_name_to_id.json
        
    Returns:
        Dictionary mapping entity name to hash ID
        
    Raises:
        FileNotFoundError: If lookup file doesn't exist
    """
    if not lookup_path.exists():
        raise FileNotFoundError(f"Entity lookup not found: {lookup_path}")
    
    print(f"📖 Loading entity lookup from {lookup_path}...")
    with open(lookup_path, 'r', encoding='utf-8') as f:
        lookup = json.load(f)
    
    print(f"✅ Loaded {len(lookup):,} entity mappings")
    return lookup


def flatten_relations(relations_jsonl_path: Path) -> List[Dict]:
    """
    Flatten JSONL file where each line contains an entity with multiple relations.
    
    Args:
        relations_jsonl_path: Path to relations_output.jsonl
        
    Returns:
        List of individual relation dictionaries
    """
    print(f"\n📖 Loading relations from {relations_jsonl_path}...")
    
    all_relations = []
    
    with open(relations_jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading JSONL"):
            if not line.strip():
                continue
            
            entity_record = json.loads(line)
            
            # Each line has an entity with a list of relations
            for relation in entity_record.get('relations', []):
                all_relations.append(relation)
    
    print(f"✅ Extracted {len(all_relations):,} total relations from JSONL")
    return all_relations


def normalize_relations(
    relations: List[Dict],
    entity_lookup: Dict[str, str]
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """
    Replace entity names with hash IDs and track unmatched entities.
    
    Performs exact string lookup only. If --use-aliases flag is set,
    the lookup includes alias mappings from provenance-based discovery.
    
    Args:
        relations: List of relation dictionaries with entity names
        entity_lookup: Mapping from entity name to hash ID
        
    Returns:
        Tuple of (normalized_relations, unmatched_entities_report)
    """
    print("\n🔄 Normalizing relations (name → ID)...")
    
    normalized = []
    unmatched_entities = defaultdict(lambda: {"count": 0, "example_chunks": set()})
    exact_matches = 0
    
    for relation in tqdm(relations, desc="Normalizing"):
        subject_name = relation['subject']
        object_name = relation['object']
        
        # Exact lookup
        subject_id = entity_lookup.get(subject_name)
        object_id = entity_lookup.get(object_name)
        
        # Track matches
        if subject_id:
            exact_matches += 1
        if object_id:
            exact_matches += 1
        
        # Track unmatched entities
        if subject_id is None:
            unmatched_entities[subject_name]["count"] += 1
            unmatched_entities[subject_name]["example_chunks"].update(relation['chunk_ids'][:3])
        
        if object_id is None:
            unmatched_entities[object_name]["count"] += 1
            unmatched_entities[object_name]["example_chunks"].update(relation['chunk_ids'][:3])
        
        # Only keep relations where both entities matched
        if subject_id and object_id:
            normalized.append({
                "subject": subject_name,
                "subject_id": subject_id,
                "predicate": relation['predicate'],
                "object": object_name,
                "object_id": object_id,
                "chunk_ids": relation['chunk_ids'],
                "extraction_strategy": relation['extraction_strategy']
            })
    
    # Convert sets to lists for JSON serialization
    unmatched_report = {
        name: {
            "count": data["count"],
            "example_chunks": list(data["example_chunks"])
        }
        for name, data in unmatched_entities.items()
    }
    
    print(f"✅ Normalized {len(normalized):,} relations")
    print(f"   Exact matches: {exact_matches // 2:,} pairs")
    print(f"⚠️  Found {len(unmatched_report):,} unmatched entity names")
    
    return normalized, unmatched_report


def deduplicate_relations(relations: List[Dict]) -> List[Dict]:
    """
    Deduplicate relations with same (subject_id, predicate, object_id).
    Merges chunk_ids and keeps unique extraction strategies.
    
    Args:
        relations: List of normalized relation dictionaries
        
    Returns:
        Deduplicated list of relations
    """
    print("\n🔀 Deduplicating relations...")
    
    # Group by triplet key
    triplet_groups = defaultdict(lambda: {
        "chunk_ids": set(),
        "strategies": set(),
        "first_relation": None
    })
    
    for relation in tqdm(relations, desc="Grouping"):
        key = (relation['subject_id'], relation['predicate'], relation['object_id'])
        
        if triplet_groups[key]["first_relation"] is None:
            triplet_groups[key]["first_relation"] = relation
        
        triplet_groups[key]["chunk_ids"].update(relation['chunk_ids'])
        triplet_groups[key]["strategies"].add(relation['extraction_strategy'])
    
    # Build deduplicated list
    deduplicated = []
    for key, data in triplet_groups.items():
        rel = data["first_relation"].copy()
        rel["chunk_ids"] = sorted(list(data["chunk_ids"]))
        rel["extraction_strategy"] = list(data["strategies"])
        deduplicated.append(rel)
    
    print(f"✅ Deduplicated to {len(deduplicated):,} unique relations")
    print(f"   (Merged {len(relations) - len(deduplicated):,} duplicates)")
    
    return deduplicated


def analyze_quality(relations: List[Dict], unmatched: Dict) -> Dict:
    """
    Generate quality statistics for the normalized relations.
    
    Args:
        relations: Deduplicated normalized relations
        unmatched: Unmatched entities report
        
    Returns:
        Dictionary of quality metrics
    """
    print("\n📊 Analyzing relation quality...")
    
    # Count predicates
    predicate_counts = defaultdict(int)
    strategy_counts = defaultdict(int)
    self_refs = []
    bidirectional_pairs = defaultdict(set)
    
    for rel in relations:
        predicate_counts[rel['predicate']] += 1
        
        # Strategies might be list or string
        strategies = rel['extraction_strategy'] if isinstance(rel['extraction_strategy'], list) else [rel['extraction_strategy']]
        for strat in strategies:
            strategy_counts[strat] += 1
        
        # Self-references
        if rel['subject_id'] == rel['object_id']:
            self_refs.append((rel['subject'], rel['predicate']))
        
        # Bidirectional detection (simplified)
        forward_key = (rel['subject_id'], rel['object_id'])
        bidirectional_pairs[forward_key].add(rel['predicate'])
    
    # Find bidirectional pairs
    bidirectional = []
    checked = set()
    for (subj_id, obj_id), predicates in bidirectional_pairs.items():
        reverse_key = (obj_id, subj_id)
        if reverse_key in bidirectional_pairs and (subj_id, obj_id) not in checked:
            bidirectional.append({
                "forward": (subj_id, list(predicates)),
                "reverse": (obj_id, list(bidirectional_pairs[reverse_key]))
            })
            checked.add((subj_id, obj_id))
            checked.add(reverse_key)
    
    stats = {
        "total_relations": len(relations),
        "unique_predicates": len(predicate_counts),
        "top_predicates": dict(sorted(predicate_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
        "extraction_strategies": dict(strategy_counts),
        "self_references": {
            "count": len(self_refs),
            "examples": self_refs[:10]
        },
        "bidirectional_pairs": {
            "count": len(bidirectional),
            "examples": bidirectional[:5]
        },
        "unmatched_entities": {
            "count": len(unmatched),
            "total_occurrences": sum(data['count'] for data in unmatched.values())
        }
    }
    
    return stats


def print_quality_report(stats: Dict):
    """
    Print human-readable quality report to console.
    
    Args:
        stats: Quality statistics dictionary
    """
    print("\n" + "="*70)
    print("RELATION QUALITY REPORT")
    print("="*70)
    
    print(f"\n📈 Overview:")
    print(f"   Total relations: {stats['total_relations']:,}")
    print(f"   Unique predicates: {stats['unique_predicates']}")
    
    print(f"\n🔝 Top 10 Predicates:")
    for pred, count in stats['top_predicates'].items():
        print(f"   {pred}: {count:,}")
    
    print(f"\n🎯 Extraction Strategies:")
    for strategy, count in stats['extraction_strategies'].items():
        print(f"   {strategy}: {count:,}")
    
    print(f"\n🔄 Self-references: {stats['self_references']['count']}")
    if stats['self_references']['examples']:
        print(f"   Examples:")
        for name, pred in stats['self_references']['examples'][:5]:
            print(f"      {name} --[{pred}]--> {name}")
    
    print(f"\n↔️  Bidirectional pairs: {stats['bidirectional_pairs']['count']}")
    if stats['bidirectional_pairs']['examples']:
        print(f"   Note: These are kept as separate relations (different predicates)")
    
    print(f"\n⚠️  Unmatched Entities:")
    print(f"   Unique entities: {stats['unmatched_entities']['count']:,}")
    print(f"   Total occurrences: {stats['unmatched_entities']['total_occurrences']:,}")
    if stats['unmatched_entities']['count'] > 0:
        unmatched_rate = stats['unmatched_entities']['total_occurrences'] / stats['total_relations'] * 100
        print(f"   Unmatched rate: {unmatched_rate:.2f}%")
        if unmatched_rate > 15:
            print(f"   ❌ HIGH UNMATCHED RATE - Review unmatched_entities.json")
        elif unmatched_rate > 5:
            print(f"   ⚠️  MODERATE UNMATCHED RATE - Check unmatched_entities.json")
    
    print("="*70 + "\n")


def main():
    """
    Main execution function.
    """
    parser = argparse.ArgumentParser(
        description='Normalize entity relations by converting names to hash IDs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard normalization (canonical names only)
  python normalize_relations.py
  
  # With alias expansion (post-hoc fix for paraphrased names)
  python normalize_relations.py --use-aliases
        """
    )
    parser.add_argument(
        '--use-aliases',
        action='store_true',
        help='Use alias-expanded lookup (includes paraphrased entity names)'
    )
    
    args = parser.parse_args()
    
    # File paths
    if args.use_aliases:
        LOOKUP_FILE = PROJECT_ROOT / "data/interim/entities/entity_name_to_id_with_aliases.json"
        lookup_type = "alias-expanded"
        
        if not LOOKUP_FILE.exists():
            print("❌ Error: --use-aliases specified but alias file not found")
            print(f"   Expected: {LOOKUP_FILE}")
            print(f"   Run: python src/processing/entities/build_alias_lookup_filtered.py")
            return 1
    else:
        LOOKUP_FILE = PROJECT_ROOT / "data/interim/entities/entity_name_to_id.json"
        lookup_type = "canonical"
    
    INPUT_FILE = PROJECT_ROOT / "data/interim/relations/relations_output.jsonl"
    OUTPUT_FILE = PROJECT_ROOT / "data/interim/relations/relations_normalized.json"
    UNMATCHED_FILE = PROJECT_ROOT / "data/interim/relations/unmatched_entities.json"
    
    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("RELATION NORMALIZATION PIPELINE")
    print("="*70)
    print(f"Lookup mode: {lookup_type}")
    if args.use_aliases:
        print("⚠️  Using alias-expanded lookup (post-hoc fix)")
    print("="*70)
    
    # Load entity lookup
    entity_lookup = load_entity_lookup(LOOKUP_FILE)
    
    # Flatten relations from JSONL
    all_relations = flatten_relations(INPUT_FILE)
    
    # Normalize (name → ID)
    normalized_relations, unmatched_report = normalize_relations(all_relations, entity_lookup)
    
    # Deduplicate
    deduplicated_relations = deduplicate_relations(normalized_relations)
    
    # Quality analysis
    stats = analyze_quality(deduplicated_relations, unmatched_report)
    
    # Save outputs
    print(f"\n💾 Saving normalized relations to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(deduplicated_relations, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saving unmatched entities to {UNMATCHED_FILE}...")
    with open(UNMATCHED_FILE, 'w', encoding='utf-8') as f:
        json.dump(unmatched_report, f, indent=2, ensure_ascii=False)
    
    # Print quality report
    print_quality_report(stats)
    
    print(f"✅ Normalization complete!")
    print(f"\nOutput files:")
    print(f"   📄 {OUTPUT_FILE}")
    print(f"   📄 {UNMATCHED_FILE}")


if __name__ == "__main__":
    main()