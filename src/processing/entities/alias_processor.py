# -*- coding: utf-8 -*-
"""
Module: alias_processor.py
Package: src.processing.entities
Purpose: Pipeline orchestrator for entity alias discovery

Author: Pau Barba i Colomer
Created: 2025-12-05
Modified: 2025-12-05

References:
    - See docs/ARCHITECTURE.md § 3.3 for alias discovery context
    - Post-hoc fix for LLM paraphrasing (proper: store during Phase 1C)

Usage:
    python src/processing/entities/alias_processor.py
"""

# Standard library
import json
import sys
from pathlib import Path
from typing import Dict, List

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
from tqdm import tqdm

# Local
from src.processing.entities.alias_builder import (
    AliasBuilder,
    QualityAnalyzer
)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Input paths
PRE_ENTITIES_FILE = PROJECT_ROOT / "data/interim/entities/pre_entities.json"
NORMALIZED_FILE = PROJECT_ROOT / "data/interim/entities/normalized_entities.json"
CANONICAL_LOOKUP = PROJECT_ROOT / "data/interim/entities/entity_name_to_id.json"

# Output paths
OUTPUT_FILE = PROJECT_ROOT / "data/interim/entities/entity_name_to_id_with_aliases.json"
ALIAS_REPORT = PROJECT_ROOT / "data/interim/entities/alias_discovery_report.json"

# Algorithm parameters
MIN_COVERAGE = 0.80  # 80% coverage of normalized chunks
MIN_CHUNKS = 5       # At least 5 chunks


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_json(file_path: Path) -> any:
    """
    Load JSON with nested structure handling.
    
    Pre-entities may have nested structure from Phase 1B extraction.
    This function flattens it if needed.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded data (list or dict)
    """
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


# ==============================================================================
# PIPELINE ORCHESTRATION
# ==============================================================================

class AliasProcessor:
    """
    Orchestrates entity alias discovery pipeline.
    
    Loads data, builds chunk indices, discovers aliases using provenance
    coverage, generates quality report, and saves expanded lookup.
    """
    
    def __init__(self):
        """Initialize processor."""
        self.pre_entities = []
        self.normalized_entities = []
        self.canonical_lookup = {}
        self.alias_map = {}
        self.expanded_lookup = {}
        self.quality_report = {}
    
    def run(self):
        """Execute complete alias discovery pipeline."""
        print("="*70)
        print("ENTITY ALIAS DISCOVERY - PROVENANCE COVERAGE")
        print("="*70)
        print("\n✅ Improved version with provenance-based filtering")
        print("⚠️  Post-hoc fix - proper implementation in Phase 1C")
        print(f"\nAlgorithm: Check if pre-entity chunks ACCOUNT FOR normalized chunks")
        print(f"Min coverage: {MIN_COVERAGE*100:.0f}%")
        print(f"Min chunks: {MIN_CHUNKS}\n")
        
        # Check prerequisites
        if not self._check_prerequisites():
            return 1
        
        # Load data
        self._load_data()
        
        # Build alias builder
        builder = AliasBuilder(
            min_coverage=MIN_COVERAGE,
            min_chunks=MIN_CHUNKS
        )
        
        # Build chunk indices
        print(f"\n{'='*70}")
        print("Building Chunk Indices")
        print(f"{'='*70}")
        
        chunk_index = builder.build_chunk_index(self.pre_entities)
        print(f"✅ Built chunk→pre-entities index ({len(chunk_index):,} chunks)")
        
        pre_entity_chunks = builder.build_pre_entity_chunks(self.pre_entities)
        print(f"✅ Built pre-entity→chunks index ({len(pre_entity_chunks):,} entities)")
        
        # Extract aliases
        print(f"\n{'='*70}")
        print("Extracting Aliases")
        print(f"{'='*70}\n")
        
        self.alias_map = builder.extract_aliases(
            self.normalized_entities,
            chunk_index,
            pre_entity_chunks
        )
        
        stats = builder.get_stats()
        
        print(f"\n✅ Provenance-based aliases extracted:")
        print(f"   Entities with aliases: {len(self.alias_map):,}")
        print(f"   Total aliases: {stats['total_aliases_found']:,}")
        print(f"\n   Filtering results:")
        print(f"   - Low coverage filtered: {stats['filtered_low_coverage']:,}")
        print(f"   - Noise filtered: {stats['filtered_noise']:,}")
        print(f"   - Dissimilar filtered: {stats['filtered_dissimilar']:,}")
        
        # Build expanded lookup
        print(f"\n{'='*70}")
        print("Building Expanded Lookup")
        print(f"{'='*70}")
        
        self.expanded_lookup, aliases_added, conflicts = builder.build_expanded_lookup(
            self.canonical_lookup,
            self.alias_map
        )
        
        print(f"✅ Expanded lookup built:")
        print(f"   Original entries: {len(self.canonical_lookup):,}")
        print(f"   Aliases added: {aliases_added:,}")
        print(f"   Total entries: {len(self.expanded_lookup):,}")
        print(f"   Expansion: {100*aliases_added/len(self.canonical_lookup):.1f}%")
        if conflicts > 0:
            print(f"   ⚠️  Conflicts detected (skipped): {conflicts:,}")
        
        # Analyze quality
        self._analyze_quality()
        
        # Save outputs
        self._save_outputs()
        
        # Print summary
        self._print_summary()
        
        return 0
    
    def _check_prerequisites(self) -> bool:
        """Check if all required input files exist."""
        missing = []
        for path in [PRE_ENTITIES_FILE, NORMALIZED_FILE, CANONICAL_LOOKUP]:
            if not path.exists():
                missing.append(str(path))
        
        if missing:
            print("❌ Missing required files:")
            for path in missing:
                print(f"   - {path}")
            return False
        
        return True
    
    def _load_data(self):
        """Load all input data."""
        print(f"\n{'='*70}")
        print("Loading Input Data")
        print(f"{'='*70}\n")
        
        self.pre_entities = load_json(PRE_ENTITIES_FILE)
        self.normalized_entities = load_json(NORMALIZED_FILE)
        self.canonical_lookup = load_json(CANONICAL_LOOKUP)
    
    def _analyze_quality(self):
        """Generate quality analysis report."""
        print(f"\n{'='*70}")
        print("Quality Analysis")
        print(f"{'='*70}")
        
        self.quality_report = QualityAnalyzer.analyze(self.alias_map, top_n=20)
        
        print(f"\n📊 Coverage Distribution:")
        print(f"   ≥90% coverage (very high confidence): {self.quality_report['coverage_distribution']['high_confidence_90plus']:,}")
        print(f"   80-90% coverage (high confidence): {self.quality_report['coverage_distribution']['medium_confidence_80_90']:,}")
        
        print(f"\n🔝 Top 10 entities by alias count:")
        for i, entity in enumerate(self.quality_report['top_entities'][:10], 1):
            canonical = entity['canonical']
            alias_count = entity['alias_count']
            total_chunks = entity['total_chunks']
            
            print(f"\n{i}. {canonical[:70]}")
            print(f"   Normalized chunks: {total_chunks}")
            print(f"   Aliases found: {alias_count}")
            
            for alias_info in entity['top_aliases'][:3]:
                print(f"      - {alias_info['name'][:50]}")
                print(f"        Coverage: {alias_info['coverage_pct']:.1f}% ({alias_info['chunks_in_normalized']}/{total_chunks} chunks)")
    
    def _save_outputs(self):
        """Save expanded lookup and quality report."""
        print(f"\n{'='*70}")
        print("Saving Outputs")
        print(f"{'='*70}")
        
        # Save expanded lookup
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.expanded_lookup, f, indent=2, ensure_ascii=False)
        
        file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
        print(f"✅ Saved expanded lookup: {OUTPUT_FILE.name} ({file_size_mb:.1f} MB)")
        
        # Save quality report
        with open(ALIAS_REPORT, 'w', encoding='utf-8') as f:
            json.dump(self.quality_report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved quality report: {ALIAS_REPORT.name}")
    
    def _print_summary(self):
        """Print final summary."""
        print(f"\n{'='*70}")
        print("ALIAS DISCOVERY COMPLETE")
        print(f"{'='*70}")
        
        print(f"\n📊 Algorithm: PROVENANCE COVERAGE")
        print(f"   Instead of counting overlap, we check:")
        print(f"   'Does pre-entity cover >{MIN_COVERAGE*100:.0f}% of normalized chunks?'")
        print(f"   → If yes: pre-entity was merged → it's an alias")
        print(f"   → If no: pre-entity just co-occurs → not an alias")
        
        print(f"\n📄 Output files:")
        print(f"   - {OUTPUT_FILE}")
        print(f"   - {ALIAS_REPORT}")
        
        print(f"\n🎯 Next steps:")
        print(f"   1. Run: python src/processing/relations/normalize_relations.py")
        print(f"      (Will automatically use entity_name_to_id_with_aliases.json)")
        print(f"   2. Check relation improvement vs canonical-only baseline")
        print(f"   3. Decision: Keep if >2% improvement in matched relations")
        
        print(f"\n💡 Why this is better than overlap counting:")
        print(f"   ✓ Tests provenance (merged) not just co-occurrence")
        print(f"   ✓ Bidirectional check filters generic supersets")
        print(f"   ✓ Eliminates false positives from related entities")
        print(f"   ✓ Theoretically justified (80% = coverage threshold)")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """Run alias discovery pipeline."""
    processor = AliasProcessor()
    return processor.run()


if __name__ == "__main__":
    sys.exit(main())
