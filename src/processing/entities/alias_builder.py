# -*- coding: utf-8 -*-
"""
Module: alias_builder.py
Package: src.processing.entities
Purpose: Core alias discovery using provenance coverage heuristic

Author: Pau Barba i Colomer
Created: 2025-12-05
Modified: 2025-12-05

References:
    - Post-hoc fix for LLM paraphrasing (proper: store during Phase 1C)
    - See docs/ARCHITECTURE.md § 3.3 for context
    
Algorithm:
    For each normalized entity:
    1. Find all pre-entities appearing in its chunks
    2. Calculate coverage: what % of normalized chunks does pre-entity cover?
    3. Bidirectional check: filter out generic supersets (pre-entity with >>more chunks)
    4. Apply noise filters + string similarity
    5. High coverage + bidirectional + filters → alias confirmed
    
Key insight: Merged entities' chunks sum to normalized chunks. Co-occurring
entities appear together but don't account for all chunks. Generic supersets
have way more chunks than the specific normalized entity.
"""

# Standard library
import re
from typing import Dict, List, Set
from collections import defaultdict

# Third-party
from tqdm import tqdm


# ==============================================================================
# NOISE FILTERS
# ==============================================================================

class NoiseFilter:
    """Filters for excluding non-alias noise patterns."""
    
    # Generic single words to exclude
    GENERIC_WORDS = {
        'AI', 'AI Act', 'Article', 'Chapter', 'Section', 'Regulation',
        'Directive', 'the', 'and', 'or', 'of', 'in', 'to', 'for'
    }
    
    @staticmethod
    def is_noise(alias: str) -> bool:
        """
        Check if alias should be excluded as noise.
        
        Args:
            alias: Candidate alias string
            
        Returns:
            True if alias should be excluded
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
        
        # Generic words
        if alias in NoiseFilter.GENERIC_WORDS:
            return True
        
        # Mostly punctuation/symbols
        alpha_chars = sum(c.isalpha() for c in alias)
        if alpha_chars < len(alias) * 0.5:  # <50% letters
            return True
        
        return False


# ==============================================================================
# STRING SIMILARITY
# ==============================================================================

class StringSimilarity:
    """String similarity checks for alias validation."""
    
    @staticmethod
    def is_likely_variant(canonical: str, alias: str) -> bool:
        """
        Check if alias is likely a legitimate variant of canonical.
        
        Uses string similarity heuristics: exact match, substring, word overlap.
        
        Args:
            canonical: Canonical entity name
            alias: Candidate alias
            
        Returns:
            True if likely a variant
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


# ==============================================================================
# ALIAS BUILDER
# ==============================================================================

class AliasBuilder:
    """
    Discovers entity aliases using provenance coverage heuristic.
    
    Core insight: If pre-entity was merged into normalized entity during
    Phase 1C disambiguation, then pre-entity's chunks should account for
    a large percentage of normalized entity's chunks (not just co-occur).
    """
    
    def __init__(
        self,
        min_coverage: float = 0.80,
        min_chunks: int = 5
    ):
        """
        Initialize alias builder.
        
        Args:
            min_coverage: Minimum coverage of normalized chunks (default: 80%)
            min_chunks: Minimum absolute chunks (default: 5)
        """
        self.min_coverage = min_coverage
        self.min_chunks = min_chunks
        
        # Statistics tracking
        self.stats = {
            'filtered_low_coverage': 0,
            'filtered_noise': 0,
            'filtered_dissimilar': 0,
            'total_aliases_found': 0
        }
    
    def build_chunk_index(
        self,
        pre_entities: List[Dict]
    ) -> Dict[str, Set[str]]:
        """
        Build reverse index: chunk_id → set of pre-entity names.
        
        Args:
            pre_entities: List of pre-entity dicts from Phase 1B
            
        Returns:
            Dict mapping chunk_id to set of entity names
        """
        chunk_index = defaultdict(set)
        
        for entity in pre_entities:
            entity_name = entity['name']
            
            # Handle both chunk_id and chunk_ids
            chunk_ids = entity.get('chunk_ids', [])
            if not chunk_ids:
                chunk_id = entity.get('chunk_id')
                if chunk_id:
                    chunk_ids = [chunk_id]
            
            for chunk_id in chunk_ids:
                chunk_index[chunk_id].add(entity_name)
        
        return chunk_index
    
    def build_pre_entity_chunks(
        self,
        pre_entities: List[Dict]
    ) -> Dict[str, Set[str]]:
        """
        Build mapping: pre-entity name → set of chunk IDs.
        
        Args:
            pre_entities: List of pre-entity dicts from Phase 1B
            
        Returns:
            Dict mapping entity name to set of chunk IDs
        """
        pre_entity_chunks = {}
        
        for entity in pre_entities:
            name = entity['name']
            chunk_ids = entity.get('chunk_ids', [])
            if not chunk_ids:
                chunk_id = entity.get('chunk_id')
                if chunk_id:
                    chunk_ids = [chunk_id]
            
            if name not in pre_entity_chunks:
                pre_entity_chunks[name] = set()
            pre_entity_chunks[name].update(chunk_ids)
        
        return pre_entity_chunks
    
    def extract_aliases(
        self,
        normalized_entities: List[Dict],
        chunk_index: Dict[str, Set[str]],
        pre_entity_chunks: Dict[str, Set[str]]
    ) -> Dict[str, Dict]:
        """
        Extract high-confidence aliases using provenance coverage.
        
        Args:
            normalized_entities: Normalized entities from Phase 1C
            chunk_index: Reverse index (chunk_id → pre-entity names)
            pre_entity_chunks: Mapping (pre-entity name → chunk IDs)
            
        Returns:
            Dict mapping canonical name to alias info with coverage scores
        """
        alias_map = {}
        
        # Reset stats
        self.stats = {k: 0 for k in self.stats}
        
        for entity in tqdm(normalized_entities, desc="Finding aliases"):
            canonical_name = entity['name']
            entity_id = entity.get('entity_id', canonical_name)
            normalized_chunks = set(entity.get('chunk_ids', []))
            
            if not normalized_chunks or len(normalized_chunks) < self.min_chunks:
                continue
            
            # Find all pre-entities appearing in normalized chunks
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
                # Key test: What % of normalized chunks does pre-entity cover?
                coverage = len(pre_chunks_in_normalized) / len(normalized_chunks)
                
                # Get total chunks for pre-entity
                total_pre_chunks_set = pre_entity_chunks.get(pre_name, set())
                total_pre_chunks = len(total_pre_chunks_set)
                
                # Filter 1: Coverage threshold
                if coverage < self.min_coverage:
                    self.stats['filtered_low_coverage'] += 1
                    continue
                
                # Filter 2: Absolute minimum
                if len(pre_chunks_in_normalized) < self.min_chunks:
                    self.stats['filtered_low_coverage'] += 1
                    continue
                
                # Filter 3: BIDIRECTIONAL CHECK
                # If pre-entity has WAY more chunks than normalized, it's probably
                # a generic term (superset), not a variant
                if total_pre_chunks > len(normalized_chunks) * 2:
                    # Check reverse coverage: what % of pre-entity chunks are in normalized?
                    reverse_coverage = len(pre_chunks_in_normalized) / total_pre_chunks
                    if reverse_coverage < 0.5:
                        self.stats['filtered_dissimilar'] += 1
                        continue
                
                # Filter 4: Noise patterns
                if NoiseFilter.is_noise(pre_name):
                    self.stats['filtered_noise'] += 1
                    continue
                
                # Filter 5: String similarity
                if not StringSimilarity.is_likely_variant(canonical_name, pre_name):
                    self.stats['filtered_dissimilar'] += 1
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
                self.stats['total_aliases_found'] += len(high_conf_aliases)
        
        return alias_map
    
    def build_expanded_lookup(
        self,
        canonical_lookup: Dict[str, str],
        alias_map: Dict[str, Dict]
    ) -> Dict[str, str]:
        """
        Build expanded name→ID lookup including aliases.
        
        Note: Entity IDs come from Phase 1C (add_entity_ids.py), which uses
        src.utils.id_generator.generate_entity_id() for consistency.
        
        Args:
            canonical_lookup: Original name→entity_id mapping
            alias_map: Alias information from extract_aliases()
            
        Returns:
            Expanded lookup with aliases added
        """
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
        
        return expanded, aliases_added, conflicts_detected
    
    def get_stats(self) -> Dict:
        """Get filtering statistics."""
        return self.stats.copy()


# ==============================================================================
# QUALITY ANALYZER
# ==============================================================================

class QualityAnalyzer:
    """Analyzes quality of discovered aliases."""
    
    @staticmethod
    def analyze(alias_map: Dict[str, Dict], top_n: int = 20) -> Dict:
        """
        Generate quality analysis report.
        
        Args:
            alias_map: Alias map from AliasBuilder
            top_n: Number of top entities to report
            
        Returns:
            Analysis report dict
        """
        # Sort by alias count
        by_alias_count = sorted(
            alias_map.items(),
            key=lambda x: len(x[1]['aliases']),
            reverse=True
        )
        
        # Coverage distribution
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
        
        # Top entities
        top_entities = []
        for canonical, info in by_alias_count[:top_n]:
            # Sort aliases by coverage
            sorted_aliases = sorted(
                info['aliases'],
                key=lambda x: info['coverage_scores'][x]['coverage_pct'],
                reverse=True
            )[:5]
            
            top_entities.append({
                'canonical': canonical,
                'total_chunks': info['total_normalized_chunks'],
                'alias_count': len(info['aliases']),
                'top_aliases': [
                    {
                        'name': alias,
                        'coverage_pct': info['coverage_scores'][alias]['coverage_pct'],
                        'chunks_in_normalized': info['coverage_scores'][alias]['chunks_in_normalized'],
                        'total_pre_chunks': info['coverage_scores'][alias]['total_pre_chunks']
                    }
                    for alias in sorted_aliases
                ]
            })
        
        return {
            'entities_with_aliases': len(alias_map),
            'total_aliases': sum(len(info['aliases']) for info in alias_map.values()),
            'coverage_distribution': {
                'high_confidence_90plus': total_high_coverage,
                'medium_confidence_80_90': total_medium_coverage
            },
            'top_entities': top_entities
        }
