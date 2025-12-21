# -*- coding: utf-8 -*-
"""
Citation Entity Identification and Matching.

Identifies citation entities from extracted entities using discusses relations,
then matches them to Scopus references using provenance-constrained fuzzy matching.

Author: Pau Barba i Colomer
Created: 2025-12-21
Modified: 2025-12-21

References:
    - See ARCHITECTURE.md § 3.2.1 for Phase 2A context
    - See PHASE_2A_DESIGN.md for matching pipeline
"""

# Standard library
import re
from typing import Dict, List, Optional, Tuple

# Third-party
from rapidfuzz import fuzz

# Project imports
from src.utils.id_generator import generate_l2_publication_id
from src.utils.logger import get_logger

# Config - import with fallback
try:
    from src.config.extraction import ENRICHMENT_CONFIG
except ImportError:
    ENRICHMENT_CONFIG = {}

logger = get_logger(__name__)


# =============================================================================
# CITATION ENTITY IDENTIFIER
# =============================================================================

class CitationEntityIdentifier:
    """
    Identify citation entities that have discusses relations.
    
    Logic: Entity must be SUBJECT of a 'discusses' relation AND have an academic type.
    This ensures every identified citation has topic information.
    
    Example:
        identifier = CitationEntityIdentifier()
        citations = identifier.identify(entities, relations)
    """
    
    # Academic types that indicate citations/papers
    ACADEMIC_TYPES = {
        'Citation', 'Paper', 'Publication', 'Book', 'Article',
        'Author', 'Journal', 'Conference Paper', 'Academic Citations',
        'Legal Citation', 'Source Citation', 'Journal Publication'
    }
    
    def identify(
        self,
        entities: List[Dict],
        relations: List[Dict]
    ) -> Dict[str, Dict]:
        """
        Identify citation entities using discusses relations + type filtering.
        
        Strategy:
        1. Find all entities that are SUBJECTS of 'discusses' relations
        2. Filter to those with academic types
        3. Return only entities that satisfy BOTH conditions
        
        Args:
            entities: List of normalized entities
            relations: List of relations (to find discusses subjects)
            
        Returns:
            Dict mapping entity_id -> {name, chunk_ids, type, discusses_objects}
        """
        # Step 1: Build discusses subjects lookup
        discusses_subjects = {}
        discusses_count = 0
        
        for r in relations:
            if r.get('predicate') == 'discusses':
                discusses_count += 1
                subj_id = r.get('subject_id')
                if subj_id:
                    if subj_id not in discusses_subjects:
                        discusses_subjects[subj_id] = []
                    discusses_subjects[subj_id].append(r.get('object_id', ''))
        
        logger.debug(f"Found {discusses_count} discusses relations, "
                    f"{len(discusses_subjects)} unique subjects")
        
        # Step 2: Build entity lookup by ID
        entity_by_id = {}
        for entity in entities:
            eid = entity.get('entity_id')
            if eid:
                entity_by_id[eid] = entity
        
        # Step 3: Find overlap
        overlap_ids = set(discusses_subjects.keys()) & set(entity_by_id.keys())
        logger.debug(f"Entities that are discusses subjects: {len(overlap_ids)}")
        
        # Step 4: Filter to academic types
        citation_entities = {}
        type_counts = {}
        
        for entity_id in overlap_ids:
            entity = entity_by_id[entity_id]
            entity_type = entity.get('type', '')
            
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
            
            if entity_type in self.ACADEMIC_TYPES:
                citation_entities[entity_id] = {
                    'name': entity['name'],
                    'chunk_ids': entity.get('chunk_ids', []),
                    'type': entity_type,
                    'discusses_objects': discusses_subjects[entity_id]
                }
        
        logger.info(f"Identified {len(citation_entities)} citation entities "
                   f"(from {len(overlap_ids)} discusses subjects)")
        
        # Log top types for debugging
        sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])[:5]
        for t, c in sorted_types:
            marker = "✓" if t in self.ACADEMIC_TYPES else " "
            logger.debug(f"  {marker} {t}: {c}")
        
        return citation_entities


# =============================================================================
# CITATION MATCHER
# =============================================================================

class CitationMatcher:
    """
    Matches citation entities to publication references using fuzzy matching.
    
    Implements provenance-constrained matching: entities are matched only
    against references from their source L1 paper.
    
    Example:
        matcher = CitationMatcher(refs_lookup, chunk_to_l1, l1_pubs)
        result = matcher.match_entity_to_reference(entity_id, entity_data)
    """
    
    def __init__(
        self,
        references_lookup: Dict[str, List[Dict]],
        chunk_to_l1: Dict[str, str],
        l1_publications: List[Dict],
        config: Dict = None
    ):
        """
        Initialize matcher with reference data.
        
        Args:
            references_lookup: Dict of scopus_id -> parsed references
            chunk_to_l1: Mapping from chunk_id to scopus_id
            l1_publications: List of L1 publication dicts
            config: Optional config override (defaults to ENRICHMENT_CONFIG)
        """
        self.references_lookup = references_lookup
        self.chunk_to_l1 = chunk_to_l1
        self.l1_publications = l1_publications
        
        # Load thresholds from config
        cfg = config or ENRICHMENT_CONFIG
        self.type_author_threshold = cfg.get('type_author_threshold', 0.80)
        self.type_journal_threshold = cfg.get('type_journal_threshold', 0.80)
        self.type_title_threshold = cfg.get('type_title_threshold', 0.75)
        self.fuzzy_fallback_threshold = cfg.get('fuzzy_fallback_threshold', 0.65)
        self.l1_overlap_threshold = cfg.get('l1_overlap_threshold', 0.90)
        self.partial_year_surname = cfg.get('partial_year_surname_score', 0.65)
        self.partial_surname_start = cfg.get('partial_surname_start_score', 0.60)
        self.partial_year_only = cfg.get('partial_year_only_score', 0.50)
        
        # L2 deduplication registry
        self.l2_registry = {}  # (author_norm, year, title_norm) -> l2_pub_id
        self.l2_data_cache = {}  # l2_pub_id -> parsed_ref
    
    def match_entity_to_reference(
        self,
        entity_id: str,
        entity_data: Dict
    ) -> Optional[Tuple[Dict, Dict, float, str]]:
        """
        Match entity to references using 3-tier strategy.
        
        Tier 1: Type-aware matching (Author→author, Journal→journal, Title→title)
        Tier 2: Fuzzy fallback across ALL fields
        Tier 3: Partial matching (year + surname combinations)
        
        Args:
            entity_id: Entity identifier
            entity_data: Entity data with name, chunk_ids, type
            
        Returns:
            Tuple of (matched_ref, match_result, confidence, method) or None
        """
        entity_name = entity_data['name']
        entity_type = entity_data.get('type', '')
        chunk_ids = entity_data['chunk_ids']
        
        # Check chunk provenance
        if not chunk_ids:
            return None
        
        # Get source L1 paper (provenance constraint)
        source_chunk = chunk_ids[0]
        source_l1_id = self.chunk_to_l1.get(source_chunk)
        
        if not source_l1_id:
            return None
        
        # Get references for this L1 paper ONLY
        l1_refs = self.references_lookup.get(source_l1_id, [])
        
        if not l1_refs:
            return None
        
        # Tier 1: Type-aware matching
        matched_ref, match_conf, match_method = self._type_aware_match(
            entity_name, entity_type, l1_refs
        )
        
        # Tier 2: Fuzzy fallback if Tier 1 failed
        if not matched_ref:
            matched_ref, match_conf, match_method = self._fuzzy_fallback(
                entity_name, l1_refs
            )
        
        # Tier 3: Partial match if Tier 2 failed
        if not matched_ref:
            matched_ref, match_conf, match_method = self._partial_match(
                entity_name, l1_refs
            )
        
        if not matched_ref:
            return None
        
        # Check L1 overlap (is this reference actually one of our L1 papers?)
        is_l1, target_id, overlap_conf = self._check_l1_overlap(matched_ref)
        
        match_result = {
            'is_l1': is_l1,
            'target_id': target_id,
            'overlap_conf': overlap_conf,
            'source_l1_id': source_l1_id
        }
        
        return (matched_ref, match_result, match_conf, match_method)
    
    def _type_aware_match(
        self,
        entity_name: str,
        entity_type: str,
        references: List[Dict]
    ) -> Tuple[Optional[Dict], float, str]:
        """
        Tier 1: Type-aware matching.
        
        Author entities → match author field
        Journal entities → match journal field
        Citation/Paper/Article → match title field
        """
        entity_norm = self._normalize_name(entity_name)
        
        best_match = None
        best_score = 0.0
        best_method = ""
        
        # Author entities
        if entity_type in ['Author']:
            surname = self._extract_author_surname(entity_name)
            if not surname:
                return (None, 0.0, "")
            
            for ref in references:
                ref_author = self._normalize_name(ref.get('author', ''))
                if not ref_author:
                    continue
                
                author_sim = fuzz.partial_ratio(surname.lower(), ref_author.lower()) / 100.0
                
                if ref_author.lower().startswith(surname.lower()):
                    author_sim = min(1.0, author_sim + 0.1)
                
                if author_sim >= self.type_author_threshold and author_sim > best_score:
                    best_match = ref
                    best_score = author_sim
                    best_method = 'type_author'
        
        # Journal entities
        elif entity_type in ['Journal']:
            for ref in references:
                ref_journal = self._normalize_name(ref.get('journal', ''))
                if not ref_journal:
                    continue
                
                journal_sim = fuzz.ratio(entity_norm, ref_journal) / 100.0
                
                if journal_sim >= self.type_journal_threshold and journal_sim > best_score:
                    best_match = ref
                    best_score = journal_sim
                    best_method = 'type_journal'
        
        # Citation/Paper/Article entities
        elif entity_type in ['Citation', 'Paper', 'Article', 'Publication', 'Book']:
            for ref in references:
                ref_title = self._normalize_name(ref.get('title', ''))
                if not ref_title:
                    continue
                
                title_sim = fuzz.ratio(entity_norm, ref_title) / 100.0
                
                if title_sim >= self.type_title_threshold and title_sim > best_score:
                    best_match = ref
                    best_score = title_sim
                    best_method = 'type_title'
        
        return (best_match, best_score, best_method)
    
    def _fuzzy_fallback(
        self,
        entity_name: str,
        references: List[Dict]
    ) -> Tuple[Optional[Dict], float, str]:
        """
        Tier 2: Fuzzy match across ALL fields.
        
        If type-aware matching fails, search entity name against
        author, title, and journal fields with lower threshold.
        """
        entity_norm = self._normalize_name(entity_name)
        
        best_match = None
        best_score = 0.0
        best_method = ""
        
        for ref in references:
            ref_author = self._normalize_name(ref.get('author', ''))
            ref_title = self._normalize_name(ref.get('title', ''))
            ref_journal = self._normalize_name(ref.get('journal', ''))
            
            for field_name, field_value in [
                ('author', ref_author),
                ('title', ref_title),
                ('journal', ref_journal)
            ]:
                if not field_value:
                    continue
                
                sim = fuzz.partial_ratio(entity_norm, field_value) / 100.0
                
                if sim >= self.fuzzy_fallback_threshold and sim > best_score:
                    best_match = ref
                    best_score = sim
                    best_method = f'fuzzy_{field_name}'
        
        return (best_match, best_score, best_method)
    
    def _partial_match(
        self,
        entity_name: str,
        references: List[Dict]
    ) -> Tuple[Optional[Dict], float, str]:
        """
        Tier 3: Partial matching using year + surname combinations.
        """
        entity_year = self._extract_year(entity_name)
        entity_surname = self._extract_author_surname(entity_name)
        
        if not entity_year and not entity_surname:
            return (None, 0.0, "")
        
        best_match = None
        best_score = 0.0
        best_method = ""
        
        for ref in references:
            ref_year = ref.get('year')
            ref_author = self._normalize_name(ref.get('author', ''))
            
            score = 0.0
            
            # Year + surname match
            if entity_year and entity_surname:
                if ref_year == entity_year:
                    if entity_surname in ref_author.lower():
                        score = self.partial_year_surname
                        best_method = 'partial_year_surname'
            
            # Just surname with high confidence
            elif entity_surname and not entity_year:
                if entity_surname in ref_author.lower():
                    if ref_author.lower().startswith(entity_surname):
                        score = self.partial_surname_start
                        best_method = 'partial_surname_start'
            
            # Just year (very weak)
            elif entity_year and not entity_surname:
                if ref_year == entity_year:
                    score = self.partial_year_only
                    best_method = 'partial_year_only'
            
            if score > best_score:
                best_match = ref
                best_score = score
        
        return (best_match, best_score, best_method)
    
    def _check_l1_overlap(
        self,
        parsed_ref: Dict
    ) -> Tuple[bool, Optional[str], float]:
        """
        Check if parsed reference matches an L1 publication.
        
        Returns:
            (is_l1, scopus_id_or_none, confidence)
        """
        ref_year = parsed_ref.get('year')
        ref_title = self._normalize_name(parsed_ref.get('title', ''))
        
        if not ref_title:
            return (False, None, 0.0)
        
        for l1 in self.l1_publications:
            if l1.get('year') != ref_year:
                continue
            
            l1_title = self._normalize_name(l1.get('title', ''))
            similarity = fuzz.ratio(ref_title, l1_title) / 100.0
            
            if similarity >= self.l1_overlap_threshold:
                return (True, l1['scopus_id'], similarity)
        
        return (False, None, 0.0)
    
    def get_or_create_l2(self, parsed_ref: Dict) -> str:
        """
        Get existing L2 publication ID or create new one.
        
        Uses deduplication registry to prevent duplicate L2 nodes.
        
        Args:
            parsed_ref: Parsed reference dict
            
        Returns:
            L2 publication ID
        """
        author_norm = self._normalize_name(parsed_ref.get('author', ''))
        year = parsed_ref.get('year')
        title_norm = self._normalize_name(parsed_ref.get('title', ''))
        
        # Check registry for dedup
        registry_key = (author_norm, year, title_norm)
        
        if registry_key in self.l2_registry:
            return self.l2_registry[registry_key]
        
        # Create new L2
        l2_pub_id = generate_l2_publication_id(parsed_ref['raw'])
        self.l2_registry[registry_key] = l2_pub_id
        
        # Cache full data for later retrieval
        self.l2_data_cache[l2_pub_id] = parsed_ref
        
        return l2_pub_id
    
    def get_l2_publications(self) -> List[Dict]:
        """
        Get all L2 publications created during matching.
        
        Returns:
            List of L2 publication dicts
        """
        l2_pubs = []
        
        for l2_pub_id, parsed_ref in self.l2_data_cache.items():
            l2_pubs.append({
                'publication_id': l2_pub_id,
                'title': parsed_ref.get('title', ''),
                'author': parsed_ref.get('author', ''),
                'year': parsed_ref.get('year'),
                'journal': parsed_ref.get('journal', ''),
                'raw_reference': parsed_ref['raw'],
                'node_type': 'cited_publication'
            })
        
        return l2_pubs
    
    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize text for matching."""
        name = name.lower().strip()
        name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation
        name = re.sub(r'\s+', ' ', name)      # Collapse whitespace
        return name
    
    @staticmethod
    def _extract_year(text: str) -> Optional[int]:
        """Extract 4-digit year from text."""
        match = re.search(r'\b(19|20)\d{2}\b', text)
        return int(match.group()) if match else None
    
    @staticmethod
    def _extract_author_surname(text: str) -> Optional[str]:
        """Extract likely author surname from citation string."""
        patterns = [
            r'^[\'""]?([A-Z][a-zà-ÿ]+)',       # Capitalized word at start
            r'([A-Z][a-zà-ÿ]+)\s+\(\d{4}\)',   # Name before (year)
            r'([A-Z][a-zà-ÿ]+)\s+et al',       # Name before et al
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).lower()
        
        return None