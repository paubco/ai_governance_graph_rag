# -*- coding: utf-8 -*-
"""
Build filtered high-confidence entity alias lookup using provenance coverage.

Improved alias discovery that checks if pre-entity chunks ACCOUNT FOR the
normalized entity's chunks, rather than just co-occurring. Uses provenance
coverage heuristic: if pre-entity covers >80% of normalized chunks, it was
likely merged during Phase 1C disambiguation.

Algorithm:
    For each normalized entity:
    1. Find all pre-entities appearing in its chunks
    2. Calculate coverage: what % of normalized chunks does each pre-entity have?
    3. Bidirectional check: filter out generic supersets (pre-entity with >>more chunks)
    4. Apply noise filters + string similarity
    5. High coverage + bidirectional + filters → alias confirmed
    
Key insight: Merged entities' chunks sum to normalized chunks. Co-occurring
entities appear together but don't account for all chunks. Generic supersets
have way more chunks than the specific normalized entity.

Input files:
    data/interim/entities/pre_entities.json (~200K from Phase 1B)
    data/interim/entities/normalized_entities.json (~55K from Phase 1C)
    data/interim/entities/entity_name_to_id.json (canonical mappings)

Output files:
    data/interim/entities/entity_name_to_id_with_aliases.json
    data/interim/entities/alias_discovery_report.json

Runtime: ~5-10 minutes

Example:
    python src/processing/entities/build_alias_lookup.py
    
    # Example result:
    # "Chat-GPT" (840 chunks normalized)
    # - "ChatGPT" (828 chunks, covers 828/840 = 98.6%) → ALIAS ✓
    # - "AI system" (345 chunks, covers 28/840 = 3.3%) → NOT ALIAS ✗ (generic superset)

Warning:
    Still a POST-HOC FIX. Proper implementation stores aliases during Phase 1C.
    This provenance-based approach with bidirectional checks is more accurate
    than overlap counting but still heuristic-based. See ARCHITECTURE.md § 3.3.
"""

# Standard library
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

# Project root
# Project root (one more .parent needed for src/processing/entities/ location)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
from tqdm import tqdm


def is_noise_alias(alias: str) -> bool:
    """
    Filter out noise patterns that are definitely not entity variants.
    
    Returns True if alias should be EXCLUDED.
    """
    # Too short (single letters, etc.)
    if len(alias) <= 2:
        return True
    
    # Years (1900-2099)
    if re.match(r'^(19|20)\d{2}[a-z]?$', alias):
        return True
    
    # Pure numbers
    if re.match(r'^[\d\s,.\-]+$', alias):
        return True
    
    # Date patterns
    if re.match(r'^\d+\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}$', alias):
        return True
    
    # Citation markers (footnotes, superscripts)
    if re.match(r'^[\[\(\$\^\{\}]+[\d,\s]+[\]\)\$\^\{\}]+$', alias):
        return True
    
    # Document codes (e.g., "2021/0106(COD)")
    if re.match(r'^\d{4}/\d+\([A-Z]+\)$', alias):
        return True
    
    # Percentages, monetary values
    if re.match(r'^[\d\s,.%$€£]+$', alias):
        return True
    
    # URLs, DOIs
    if 'http' in alias.lower() or 'doi.org' in alias.lower():
        return True
    
    # Single words that are too generic
    generic_words = {
        'AI', 'AI Act', 'Article', 'Chapter', 'Section', 'Regulation',
        'Directive', 'the', 'and', 'or', 'of', 'in', 'to', 'for'
    }
    if alias in generic_words:
        return True
    
    # Mostly punctuation/symbols
    alpha_chars = sum(c.isalpha() for c in alias)
    if alpha_chars < len(alias) * 0.5:  # <50% letters
        return True
    
    return False


def is_likely_variant(canonical: str, alias: str) -> bool:
    """
    Check if alias is likely a legitimate variant of canonical.
    
    Uses string similarity heuristics.
    """
    canonical_lower = canonical.lower()
    alias_lower = alias.lower()
    
    # Exact match (different case)
    if canonical_lower == alias_lower:
        return True
    
    # Very similar (substring)
    if canonical_lower in alias_lower or alias_lower in canonical_lower:
        return True
    
    # Same words, different order/punctuation
    canonical_words = set(re.findall(r'\w+', canonical_lower))
    alias_words = set(re.findall(r'\w+', alias_lower))
    
    # At least 50% word overlap
    if canonical_words and alias_words:
        overlap = len(canonical_words & alias_words)
        smaller = min(len(canonical_words), len(alias_words))
        if overlap / smaller >= 0.5:
            return True
    
    return False


def load_json(file_path: Path) -> any:
    """Load JSON with nested structure handling."""
    print(f"📖 Loading {file_path.name}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle nested pre_entities structure
    if isinstance(data, dict) and 'entities' in data:
        print(f"   (Detected nested structure with chunk grouping)")
        flat_entities = []
        for chunk_record in data['entities']:
            if 'entities' in chunk_record:
                flat_entities.extend(chunk_record['entities'])
        print(f"✅ Loaded {len(flat_entities):,} entities from {len(data['entities']):,} chunks")
        return flat_entities
    
    print(f"✅ Loaded {len(data):,} items")
    return data


def build_chunk_to_pre_entities_index(pre_entities: List[Dict]) -> Dict[str, Set[str]]:
    """Build reverse index: chunk_id → set of pre-entity names."""
    print("\n🔨 Building chunk → pre-entities index...")
    
    chunk_index = defaultdict(set)
    
    for entity in tqdm(pre_entities, desc="Indexing"):
        entity_name = entity['name']
        
        # Handle both chunk_id and chunk_ids
        chunk_ids = entity.get('chunk_ids', [])
        if not chunk_ids:
            chunk_id = entity.get('chunk_id')
            if chunk_id:
                chunk_ids = [chunk_id]
        
        for chunk_id in chunk_ids:
            chunk_index[chunk_id].add(entity_name)
    
    print(f"✅ Indexed {len(chunk_index):,} chunks")
    return chunk_index


def extract_high_confidence_aliases(
    normalized_entities: List[Dict],
    chunk_index: Dict[str, Set[str]],
    pre_entities: List[Dict],
    min_coverage: float = 0.80,
    min_chunks: int = 5
) -> Dict[str, Dict[str, any]]:
    """
    Extract high-confidence aliases using provenance coverage.
    
    Key insight: If pre-entity was merged into normalized entity, then
    pre-entity's chunks should account for a large percentage of the
    normalized entity's chunks.
    
    Args:
        normalized_entities: Normalized entities from Phase 1C
        chunk_index: Reverse index from build_chunk_to_pre_entities_index
        pre_entities: Pre-entities from Phase 1B (for chunk lookup)
        min_coverage: Minimum coverage of normalized chunks (default: 80%)
        min_chunks: Minimum absolute chunks (default: 5)
    
    Returns:
        Dict of alias info (only high-confidence aliases)
    """
    print(f"\n🔍 Extracting aliases using PROVENANCE COVERAGE...")
    print(f"   Algorithm: Check if pre-entity chunks account for normalized chunks")
    print(f"   Min coverage: {min_coverage*100:.0f}% of normalized chunks")
    print(f"   Min absolute: {min_chunks} chunks")
    print(f"   Filters: Bidirectional check + noise patterns + string similarity\n")
    
    # Build pre-entity chunk lookup for coverage calculation
    print("📊 Building pre-entity chunk lookup...")
    pre_entity_chunks = {}
    for entity in tqdm(pre_entities, desc="Indexing pre-entities"):
        name = entity['name']
        chunk_ids = entity.get('chunk_ids', [])
        if not chunk_ids:
            chunk_id = entity.get('chunk_id')
            if chunk_id:
                chunk_ids = [chunk_id]
        
        if name not in pre_entity_chunks:
            pre_entity_chunks[name] = set()
        pre_entity_chunks[name].update(chunk_ids)
    
    print(f"✅ Indexed {len(pre_entity_chunks):,} unique pre-entity names")
    
    # Extract aliases
    alias_map = {}
    total_aliases_found = 0
    filtered_noise = 0
    filtered_dissimilar = 0
    filtered_low_coverage = 0
    
    for entity in tqdm(normalized_entities, desc="Finding aliases"):
        canonical_name = entity['name']
        entity_id = entity.get('entity_id', canonical_name)
        normalized_chunks = set(entity.get('chunk_ids', []))
        
        if not normalized_chunks or len(normalized_chunks) < min_chunks:
            continue
        
        # Find all pre-entities that appear in normalized chunks
        candidates = {}
        
        for chunk_id in normalized_chunks:
            pre_names = chunk_index.get(chunk_id, set())
            for pre_name in pre_names:
                if pre_name != canonical_name:
                    if pre_name not in candidates:
                        candidates[pre_name] = set()
                    candidates[pre_name].add(chunk_id)
        
        # Test provenance coverage for each candidate
        high_conf_aliases = []
        
        for pre_name, pre_chunks_in_normalized in candidates.items():
            # Key test: What % of normalized chunks does this pre-entity cover?
            coverage = len(pre_chunks_in_normalized) / len(normalized_chunks)
            
            # Also get total chunks for this pre-entity (for reporting)
            total_pre_chunks_set = pre_entity_chunks.get(pre_name, set())
            total_pre_chunks = len(total_pre_chunks_set)
            
            # Filter 1: Coverage threshold
            if coverage < min_coverage:
                filtered_low_coverage += 1
                continue
            
            # Filter 2: Absolute minimum
            if len(pre_chunks_in_normalized) < min_chunks:
                filtered_low_coverage += 1
                continue
            
            # Filter 3: BIDIRECTIONAL CHECK (new!)
            # If pre-entity has WAY more chunks than normalized, it's probably
            # a generic term (superset), not a variant
            # Example: "AI system" (345 chunks) is NOT an alias of "Deployers of AI systems" (28 chunks)
            if total_pre_chunks > len(normalized_chunks) * 2:  # Pre-entity has >2x chunks
                # Check reverse coverage: what % of pre-entity chunks are in normalized?
                reverse_coverage = len(pre_chunks_in_normalized) / total_pre_chunks
                if reverse_coverage < 0.5:  # Pre-entity only has <50% chunks in normalized
                    filtered_dissimilar += 1  # It's a superset/generic term
                    continue
            
            # Filter 4: Noise patterns
            if is_noise_alias(pre_name):
                filtered_noise += 1
                continue
            
            # Filter 5: String similarity
            if not is_likely_variant(canonical_name, pre_name):
                filtered_dissimilar += 1
                continue
            
            # Passed all filters!
            high_conf_aliases.append({
                'name': pre_name,
                'coverage': coverage,
                'chunks_in_normalized': len(pre_chunks_in_normalized),
                'total_pre_chunks': total_pre_chunks
            })
        
        if high_conf_aliases:
            alias_map[canonical_name] = {
                "entity_id": entity_id,
                "aliases": [a['name'] for a in high_conf_aliases],
                "coverage_scores": {
                    a['name']: {
                        'coverage_pct': round(a['coverage'] * 100, 1),
                        'chunks_in_normalized': a['chunks_in_normalized'],
                        'total_pre_chunks': a['total_pre_chunks']
                    }
                    for a in high_conf_aliases
                },
                "total_normalized_chunks": len(normalized_chunks)
            }
            total_aliases_found += len(high_conf_aliases)
    
    print(f"\n✅ Provenance-based aliases extracted:")
    print(f"   Entities with aliases: {len(alias_map):,}")
    print(f"   Total aliases: {total_aliases_found:,}")
    print(f"\n   Filtering results:")
    print(f"   - Low coverage filtered: {filtered_low_coverage:,}")
    print(f"   - Noise filtered: {filtered_noise:,}")
    print(f"   - Dissimilar filtered: {filtered_dissimilar:,}")
    
    return alias_map


def build_expanded_lookup(
    canonical_lookup: Dict[str, str],
    alias_map: Dict[str, Dict]
) -> Dict[str, str]:
    """Build expanded lookup with high-confidence aliases only."""
    print(f"\n🔧 Building expanded lookup...")
    
    expanded = canonical_lookup.copy()
    aliases_added = 0
    conflicts_detected = 0
    
    for canonical_name, alias_info in tqdm(alias_map.items(), desc="Adding aliases"):
        entity_id = alias_info['entity_id']
        aliases = alias_info['aliases']
        
        for alias in aliases:
            # Check conflicts
            if alias in expanded:
                existing_id = expanded[alias]
                if existing_id != entity_id:
                    conflicts_detected += 1
                    continue
            
            expanded[alias] = entity_id
            aliases_added += 1
    
    print(f"✅ Expanded lookup built:")
    print(f"   Original entries: {len(canonical_lookup):,}")
    print(f"   Aliases added: {aliases_added:,}")
    print(f"   Total entries: {len(expanded):,}")
    print(f"   Expansion: {100*aliases_added/len(canonical_lookup):.1f}%")
    if conflicts_detected > 0:
        print(f"   ⚠️  Conflicts detected (skipped): {conflicts_detected:,}")
    
    return expanded


def analyze_quality(alias_map: Dict[str, Dict], top_n: int = 20):
    """Print quality analysis using provenance coverage."""
    print("\n" + "="*70)
    print("PROVENANCE-BASED ALIAS QUALITY ANALYSIS")
    print("="*70)
    
    by_alias_count = sorted(
        alias_map.items(),
        key=lambda x: len(x[1]['aliases']),
        reverse=True
    )
    
    print(f"\n🔝 Top {top_n} entities by alias count:")
    for i, (canonical, info) in enumerate(by_alias_count[:top_n], 1):
        print(f"\n{i}. {canonical[:70]}")
        print(f"   Normalized chunks: {info['total_normalized_chunks']}")
        print(f"   Aliases found: {len(info['aliases'])}")
        
        # Sort by coverage
        sorted_aliases = sorted(
            info['aliases'],
            key=lambda x: info['coverage_scores'][x]['coverage_pct'],
            reverse=True
        )[:5]
        
        for alias in sorted_aliases:
            score = info['coverage_scores'][alias]
            print(f"      - {alias[:50]}")
            print(f"        Coverage: {score['coverage_pct']:.1f}% ({score['chunks_in_normalized']}/{info['total_normalized_chunks']} chunks)")
            print(f"        Pre-entity total: {score['total_pre_chunks']} chunks")
    
    # Statistics
    total_high_coverage = sum(
        1 for info in alias_map.values()
        for alias in info['aliases']
        if info['coverage_scores'][alias]['coverage_pct'] >= 90
    )
    
    total_medium_coverage = sum(
        1 for info in alias_map.values()
        for alias in info['aliases']
        if 80 <= info['coverage_scores'][alias]['coverage_pct'] < 90
    )
    
    print(f"\n📊 Coverage Distribution:")
    print(f"   ≥90% coverage (very high confidence): {total_high_coverage:,}")
    print(f"   80-90% coverage (high confidence): {total_medium_coverage:,}")
    
    print("\n" + "="*70)


def main():
    """Main execution"""
    
    PRE_ENTITIES_FILE = PROJECT_ROOT / "data/interim/entities/pre_entities.json"
    NORMALIZED_FILE = PROJECT_ROOT / "data/interim/entities/normalized_entities.json"
    CANONICAL_LOOKUP = PROJECT_ROOT / "data/interim/entities/entity_name_to_id.json"
    
    OUTPUT_FILE = PROJECT_ROOT / "data/interim/entities/entity_name_to_id_with_aliases.json"
    ALIAS_REPORT = PROJECT_ROOT / "data/interim/entities/alias_discovery_report.json"
    
    print("="*70)
    print("FILTERED HIGH-CONFIDENCE ALIAS LOOKUP BUILDER")
    print("="*70)
    print("\n✅ Improved version with aggressive noise filtering")
    print("⚠️  Still a post-hoc fix - see ARCHITECTURE.md § 3.3\n")
    
    # Check prerequisites
    missing = []
    for path in [PRE_ENTITIES_FILE, NORMALIZED_FILE, CANONICAL_LOOKUP]:
        if not path.exists():
            missing.append(str(path))
    
    if missing:
        print("❌ Missing required files:")
        for path in missing:
            print(f"   - {path}")
        return 1
    
    # Load data
    pre_entities = load_json(PRE_ENTITIES_FILE)
    normalized_entities = load_json(NORMALIZED_FILE)
    canonical_lookup = load_json(CANONICAL_LOOKUP)
    
    # Build chunk index
    chunk_index = build_chunk_to_pre_entities_index(pre_entities)
    
    # Extract high-confidence aliases using PROVENANCE COVERAGE
    alias_map = extract_high_confidence_aliases(
        normalized_entities,
        chunk_index,
        pre_entities,           # Pass pre_entities for chunk lookup
        min_coverage=0.80,      # 80% coverage of normalized chunks
        min_chunks=5            # At least 5 chunks
    )
    
    # Analyze quality
    analyze_quality(alias_map, top_n=20)
    
    # Build expanded lookup
    expanded_lookup = build_expanded_lookup(canonical_lookup, alias_map)
    
    # Save outputs
    print(f"\n💾 Saving filtered expanded lookup to {OUTPUT_FILE}...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(expanded_lookup, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saving filtered alias report to {ALIAS_REPORT}...")
    with open(ALIAS_REPORT, 'w', encoding='utf-8') as f:
        json.dump(alias_map, f, indent=2, ensure_ascii=False)
    
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    
    print(f"\n✅ Provenance-based alias lookup built successfully!")
    print(f"\n📊 Algorithm: PROVENANCE COVERAGE")
    print(f"   Instead of counting overlap, we check:")
    print(f"   'Does pre-entity cover >80% of normalized chunks?'")
    print(f"   → If yes: pre-entity was merged → it's an alias")
    print(f"   → If no: pre-entity just co-occurs → not an alias")
    
    print(f"\nOutput files:")
    print(f"   📄 {OUTPUT_FILE} ({file_size_mb:.1f} MB)")
    print(f"   📄 {ALIAS_REPORT}")
    
    print(f"\n🎯 Next steps:")
    print(f"   1. Run: python src/processing/relations/normalize_relations.py --use-aliases")
    print(f"      (Will automatically use this provenance-filtered version)")
    print(f"   2. Check salvage improvement vs canonical-only baseline")
    print(f"   3. Decision: Keep if >2,000 additional relations salvaged")
    
    print(f"\n💡 Why this is better than overlap counting:")
    print(f"   ✓ Tests provenance (merged) not just co-occurrence")
    print(f"   ✓ Bidirectional check filters generic supersets")
    print(f"   ✓ Eliminates false positives from related entities")
    print(f"   ✓ Example: 'ChatGPT' (98.6% coverage) ✓ vs 'AI system' (superset) ✗")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())