# -*- coding: utf-8 -*-
"""
Metadata

Handles the METADATA path (6 types):
    Citation, Author, Journal, Affiliation, Document, DocumentSection

Examples:
disambiguator = MetadataDisambiguator(semantic_entities)
    metadata_entities, part_of, same_as = disambiguator.process(metadata_raw)

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

METADATA_TYPES = {'Citation', 'Author', 'Journal', 'Affiliation', 'Document', 'DocumentSection'}

# Minimum similarity for fuzzy matching
SAME_AS_THRESHOLD = 0.85    # Document ↔ Regulation matching
PART_OF_THRESHOLD = 0.80    # DocumentSection → Document matching


# =============================================================================
# METADATA DISAMBIGUATOR
# =============================================================================

class MetadataDisambiguator:
    """
    Handles metadata entities: generates IDs, creates PART_OF and SAME_AS relations.
    
    Usage:
        disambiguator = MetadataDisambiguator(semantic_entities)
        metadata_out, part_of, same_as = disambiguator.process(metadata_entities)
    """
    
    def __init__(self, semantic_entities: List[Dict] = None):
        """
        Initialize disambiguator.
        
        Args:
            semantic_entities: Disambiguated semantic entities (for SAME_AS matching)
        """
        self.semantic_entities = semantic_entities or []
        
        # Build lookup for Regulation entities (for Document ↔ Regulation SAME_AS)
        self.regulations = [
            e for e in self.semantic_entities 
            if e.get('type') == 'Regulation'
        ]
        
        self.stats = {
            'input_count': 0,
            'output_count': 0,
            'documents': 0,
            'document_sections': 0,
            'part_of_relations': 0,
            'same_as_relations': 0,
        }
    
    def process(self, metadata_entities: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Process metadata entities: generate IDs, find PART_OF and SAME_AS relations.
        
        Args:
            metadata_entities: Metadata pre-entities
            
        Returns:
            (processed_entities, part_of_relations, same_as_relations)
        """
        from src.utils.id_generator import generate_entity_id
        
        logger.info(f"Processing {len(metadata_entities)} metadata entities...")
        
        self.stats['input_count'] = len(metadata_entities)
        
        processed = []
        part_of_relations = []
        same_as_relations = []
        
        # First pass: process all entities, collect Documents for PART_OF lookup
        documents_by_chunk = defaultdict(list)  # {chunk_id: [Document entities]}
        
        for entity in metadata_entities:
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
            
            # Track Documents for PART_OF lookup
            if entity_type == 'Document':
                self.stats['documents'] += 1
                documents_by_chunk[chunk_id].append(processed_entity)
                
                # Check for SAME_AS with Regulation
                same_as = self._find_same_as_regulation(name, entity_id)
                if same_as:
                    same_as_relations.append(same_as)
            
            elif entity_type == 'DocumentSection':
                self.stats['document_sections'] += 1
        
        # Second pass: find PART_OF relations (DocumentSection → Document)
        for entity in processed:
            if entity['type'] == 'DocumentSection':
                chunk_id = entity['chunk_ids'][0] if entity['chunk_ids'] else ''
                
                # Look for Document entities in same chunk
                for doc in documents_by_chunk.get(chunk_id, []):
                    if self._is_part_of(entity['name'], doc['name']):
                        part_of_relations.append({
                            'subject': entity['name'],
                            'subject_id': entity['entity_id'],
                            'subject_type': 'DocumentSection',
                            'predicate': 'PART_OF',
                            'object': doc['name'],
                            'object_id': doc['entity_id'],
                            'object_type': 'Document',
                            'chunk_id': chunk_id,
                            'source': 'metadata_disambiguator',
                        })
                
                # NOTE: Cross-chunk matching disabled - too many false positives
                # with fuzzy matching. DocumentSections should reference their
                # parent Document in the same chunk via explicit naming.
                # 
                # Original logic (disabled):
                # for doc_list in documents_by_chunk.values():
                #     for doc in doc_list:
                #         if doc['chunk_ids'] == entity['chunk_ids']:
                #             continue
                #         if self._is_part_of(entity['name'], doc['name']):
                #             part_of_relations.append({...})
        
        # Deduplicate relations (same pair might be found multiple times)
        part_of_relations = self._dedupe_relations(part_of_relations)
        same_as_relations = self._dedupe_relations(same_as_relations)
        
        self.stats['output_count'] = len(processed)
        self.stats['part_of_relations'] = len(part_of_relations)
        self.stats['same_as_relations'] = len(same_as_relations)
        
        logger.info(f"Metadata processing complete:")
        logger.info(f"  Entities: {len(processed):,}")
        logger.info(f"  Documents: {self.stats['documents']:,}")
        logger.info(f"  DocumentSections: {self.stats['document_sections']:,}")
        logger.info(f"  PART_OF relations: {len(part_of_relations):,}")
        logger.info(f"  SAME_AS relations: {len(same_as_relations):,}")
        
        return processed, part_of_relations, same_as_relations
    
    def _find_same_as_regulation(self, doc_name: str, doc_id: str) -> Optional[Dict]:
        """
        Find matching Regulation for a Document entity.
        
        Examples:
            "EU AI Act" (Document) ↔ "EU AI Act" (Regulation) → SAME_AS
            "GDPR" (Document) ↔ "GDPR" (Regulation) → SAME_AS
        """
        doc_lower = doc_name.lower().strip()
        
        for reg in self.regulations:
            reg_name = reg.get('name', '')
            reg_lower = reg_name.lower().strip()
            
            # Exact match
            if doc_lower == reg_lower:
                return {
                    'subject': doc_name,
                    'subject_id': doc_id,
                    'subject_type': 'Document',
                    'predicate': 'SAME_AS',
                    'object': reg_name,
                    'object_id': reg.get('entity_id', ''),
                    'object_type': 'Regulation',
                    'confidence': 1.0,
                    'source': 'metadata_disambiguator',
                }
            
            # Fuzzy match
            if RAPIDFUZZ_AVAILABLE:
                score = fuzz.ratio(doc_lower, reg_lower) / 100.0
                if score >= SAME_AS_THRESHOLD:
                    return {
                        'subject': doc_name,
                        'subject_id': doc_id,
                        'subject_type': 'Document',
                        'predicate': 'SAME_AS',
                        'object': reg_name,
                        'object_id': reg.get('entity_id', ''),
                        'object_type': 'Regulation',
                        'confidence': score,
                        'source': 'metadata_disambiguator',
                    }
        
        return None
    
    def _is_part_of(self, section_name: str, doc_name: str) -> bool:
        """
        Check if DocumentSection is part of Document.
        
        Logic: Section name should contain or reference the Document name,
        OR Document name should be a meaningful substring of section context.
        
        Examples:
            "Article 5 of EU AI Act" → "EU AI Act" ✓
            "Article 5" → "EU AI Act" ✗ (no explicit link in name)
            "Section 3.2 of GDPR" → "GDPR" ✓
            "audit data" → "DA" ✗ (too short, spurious match)
        """
        if not doc_name or not section_name:
            return False
        
        section_lower = section_name.lower().strip()
        doc_lower = doc_name.lower().strip()
        
        # Skip if same name (not a part-of relationship)
        if section_lower == doc_lower:
            return False
        
        # Skip if doc is too short (spurious substring matches)
        # Minimum 5 chars to avoid 'DA', 'NN', 'RA' matching everything
        if len(doc_lower) < 5:
            return False
        
        # Skip if doc is longer than section (parent must be referenced IN child)
        if len(doc_lower) >= len(section_lower):
            return False
        
        # Fast path: doc name is substring of section name
        if doc_lower in section_lower:
            return True
        
        # Fuzzy match
        if RAPIDFUZZ_AVAILABLE:
            score = fuzz.partial_ratio(doc_lower, section_lower) / 100.0
            return score >= PART_OF_THRESHOLD
        
        return False
    
    def _dedupe_relations(self, relations: List[Dict]) -> List[Dict]:
        """Remove duplicate relations (same subject-predicate-object triple)."""
        seen = set()
        unique = []
        
        for rel in relations:
            key = (rel['subject'], rel['predicate'], rel['object'])
            if key not in seen:
                seen.add(key)
                unique.append(rel)
        
        return unique


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

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