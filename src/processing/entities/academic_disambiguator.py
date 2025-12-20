# -*- coding: utf-8 -*-
"""
Academic entity disambiguation for Phase 1C.

Handles the ACADEMIC path (4 types):
    Citation, Author, Journal, Affiliation

Key differences from semantic path:
    - NO merge (would incorrectly cluster different authors/citations)
    - Generate entity IDs only
    - Create PART_OF relations for article/section citations

PART_OF scope:
    - "Article 5 of EU AI Act" → PART_OF → "EU AI Act" ✓
    - "Floridi (2018)" → NO PART_OF (author citation)
    - "Floridi page 22" → "page 22" PART_OF → "Floridi" ✓ (if found)

Note: No embeddings generated for academic entities (Scopus matching uses fuzzy string).

Example:
    disambiguator = AcademicDisambiguator(semantic_entities, chunk_entity_map)
    academic_entities, part_of_relations = disambiguator.process(academic_raw)
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Optional: rapidfuzz for fuzzy matching
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

ACADEMIC_TYPES = {'Citation', 'Author', 'Journal', 'Affiliation'}

# Pattern for article/section/chapter citations (NOT author citations)
ARTICLE_SECTION_PATTERN = re.compile(
    r'(Article|Section|Chapter|Part|Annex|Recital|Title|Paragraph|Clause)\s+\d+',
    re.IGNORECASE
)

# Pattern for page references
PAGE_PATTERN = re.compile(
    r'(page|pages|p\.|pp\.)\s*\d+',
    re.IGNORECASE
)

# Minimum similarity for PART_OF fuzzy matching
PART_OF_THRESHOLD = 0.80


# =============================================================================
# ACADEMIC LINKER
# =============================================================================

class AcademicDisambiguator:
    """
    Handles academic entities: no merge, generates IDs, creates PART_OF relations.
    
    Usage:
        disambiguator = AcademicDisambiguator(semantic_entities, chunk_entity_map)
        processed, relations = disambiguator.process(academic_entities)
    """
    
    def __init__(self, 
                 semantic_entities: List[Dict] = None,
                 chunk_entity_map: Dict[str, List[Dict]] = None):
        """
        Initialize linker.
        
        Args:
            semantic_entities: Already-disambiguated semantic entities
            chunk_entity_map: {chunk_id: [semantic_entities_in_chunk]}
        """
        self.semantic_entities = semantic_entities or []
        self.chunk_entity_map = chunk_entity_map or {}
        
        # Build lookup structures
        self._build_lookups()
        
        self.stats = {
            'input_count': 0,
            'output_count': 0,
            'part_of_relations': 0,
            'article_citations': 0,
        }
    
    def _build_lookups(self):
        """Build efficient lookup structures for semantic entities."""
        # Name → entity (for fuzzy matching)
        self.semantic_by_name = {}
        for entity in self.semantic_entities:
            name = entity.get('name', '')
            if name:
                self.semantic_by_name[name.lower()] = entity
        
        logger.info(f"AcademicLinker: {len(self.semantic_by_name)} semantic entities indexed")
    
    def process(self, academic_entities: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Process academic entities: generate IDs, find PART_OF relations.
        
        Args:
            academic_entities: Academic pre-entities (no disambiguation)
            
        Returns:
            (processed_entities, part_of_relations)
        """
        from src.utils.id_generator import generate_entity_id
        
        logger.info(f"Processing {len(academic_entities)} academic entities...")
        
        self.stats['input_count'] = len(academic_entities)
        
        processed = []
        relations = []
        
        for entity in academic_entities:
            name = entity.get('name', '')
            entity_type = entity.get('type', '')
            chunk_id = entity.get('chunk_id', '')
            
            # Generate entity ID
            entity_id = generate_entity_id(name, entity_type)
            
            processed_entity = {
                'entity_id': entity_id,
                'name': name,
                'type': entity_type,
                'description': entity.get('description', ''),
                'chunk_ids': [chunk_id] if chunk_id else [],
            }
            processed.append(processed_entity)
            
            # Check for PART_OF relations (only for article/section citations)
            if self._is_linkable_citation(name):
                self.stats['article_citations'] += 1
                
                # Find semantic entities in same chunk
                chunk_semantics = self.chunk_entity_map.get(chunk_id, [])
                
                # Also check globally for regulation names in citation
                for semantic in chunk_semantics:
                    semantic_name = semantic.get('name', '')
                    semantic_id = semantic.get('entity_id', '')
                    
                    if self._is_part_of(name, semantic_name):
                        relations.append({
                            'subject': name,
                            'subject_id': entity_id,
                            'predicate': 'PART_OF',
                            'object': semantic_name,
                            'object_id': semantic_id,
                            'chunk_id': chunk_id,
                            'source': 'academic_linker',
                        })
        
        self.stats['output_count'] = len(processed)
        self.stats['part_of_relations'] = len(relations)
        
        logger.info(f"Academic processing complete:")
        logger.info(f"  Entities: {len(processed)}")
        logger.info(f"  Article/section citations: {self.stats['article_citations']}")
        logger.info(f"  PART_OF relations: {len(relations)}")
        
        return processed, relations
    
    def _is_linkable_citation(self, name: str) -> bool:
        """
        Check if citation should generate PART_OF relations.
        
        Returns True for:
            - Article/section/chapter references
            - Page references
            
        Returns False for:
            - Author citations like "Floridi (2018)"
        """
        # Check for article/section pattern
        if ARTICLE_SECTION_PATTERN.search(name):
            return True
        
        # Check for page pattern
        if PAGE_PATTERN.search(name):
            return True
        
        return False
    
    def _is_part_of(self, academic_name: str, semantic_name: str) -> bool:
        """
        Check if academic entity references semantic entity.
        
        Examples:
            "Article 5 of EU AI Act" contains "EU AI Act" → True
            "Article 5" + "EU AI Act" → False (no overlap)
        """
        if not semantic_name:
            return False
        
        academic_lower = academic_name.lower()
        semantic_lower = semantic_name.lower()
        
        # Fast path: exact substring containment
        if semantic_lower in academic_lower:
            return True
        
        # Slow path: fuzzy match for variations
        if RAPIDFUZZ_AVAILABLE:
            score = fuzz.partial_ratio(semantic_lower, academic_lower) / 100.0
            return score >= PART_OF_THRESHOLD
        else:
            # Fallback: word overlap
            semantic_words = set(semantic_lower.split())
            academic_words = set(academic_lower.split())
            if not semantic_words:
                return False
            overlap = len(semantic_words & academic_words) / len(semantic_words)
            return overlap >= PART_OF_THRESHOLD


def build_chunk_entity_map(entities: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Build mapping from chunk_id to entities in that chunk.
    
    Args:
        entities: List of entity dicts with 'chunk_ids' field
        
    Returns:
        {chunk_id: [entity1, entity2, ...]}
    """
    chunk_map = defaultdict(list)
    
    for entity in entities:
        chunk_ids = entity.get('chunk_ids', [])
        if isinstance(chunk_ids, str):
            chunk_ids = [chunk_ids]
        
        for chunk_id in chunk_ids:
            if chunk_id:
                chunk_map[chunk_id].append(entity)
    
    logger.info(f"Built chunk map: {len(chunk_map)} chunks with entities")
    return dict(chunk_map)