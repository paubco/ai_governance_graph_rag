# -*- coding: utf-8 -*-
"""
Citation and Metadata Entity Matching.

Identifies and matches metadata entities to structured nodes:
- Citation entities → L2 Publications (via Scopus references)
- Author entities → Scopus Author nodes
- Journal entities → Scopus Journal nodes
- Document entities → L1 Publications OR Jurisdictions

Author: Pau Barba i Colomer
Created: 2025-12-21
Modified: 2025-12-22

References:
    - See ARCHITECTURE.md § 3.2.1 for Phase 2A context
    - See PHASE_2A_DESIGN.md for matching pipeline
"""

# Standard library
import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from difflib import SequenceMatcher
from tqdm import tqdm

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


# =============================================================================
# METADATA ENTITY MATCHER
# =============================================================================

class ProvenanceConstrainedMatcher:
    """
    Matches metadata entities against ONLY their source paper's metadata.
    
    Two-tier matching strategy:
    1. Structured match (high confidence): Against parsed Scopus CSV metadata
       - Authors: Paper's own authors from Scopus
       - Journal: Paper's publication venue from Scopus
       - Title: Paper's title from Scopus
    2. Fallback match (lower confidence): Against raw reference strings
       - Searches paper_references.json for surname/title/journal mentions
    
    Uses same normalization and thresholds as MinerUMatcher (96% match rate).
    
    Example:
        matcher = ProvenanceConstrainedMatcher(
            paper_mapping=paper_mapping,
            paper_references=references,
            chunks=chunks
        )
        results = matcher.match_all(metadata_entities)
    """
    
    # Thresholds aligned with MinerUMatcher
    TITLE_THRESHOLD_HIGH = 0.85
    TITLE_THRESHOLD_MEDIUM = 0.70
    TITLE_THRESHOLD_LOW = 0.60
    
    AUTHOR_THRESHOLD_EXACT = 1.0
    AUTHOR_THRESHOLD_FUZZY = 0.85
    
    JOURNAL_THRESHOLD_HIGH = 0.85
    JOURNAL_THRESHOLD_MEDIUM = 0.70
    
    def __init__(
        self,
        paper_mapping: Dict[str, Dict],
        paper_references: Dict[str, List[str]],
        chunks: List[Dict],
        threshold: float = 0.70  # Aligned with TITLE_THRESHOLD_MEDIUM
    ):
        """
        Initialize matcher with source data.
        
        Args:
            paper_mapping: paper_mapping.json with scopus_metadata per paper
                {paper_id: {scopus_metadata: {eid, title, authors, journal, year, ...}}}
            paper_references: Raw reference strings by paper_id
                {paper_id: [ref_string, ...]}
            chunks: Chunks with document_id for provenance
                [{chunk_id, document_id, ...}]
            threshold: Minimum similarity for fuzzy match (default 0.85)
        """
        self.threshold = threshold
        self.paper_mapping = paper_mapping
        
        # Build chunk_id → paper_id mapping
        self.chunk_to_paper = self._build_chunk_to_paper(chunks, paper_mapping)
        
        # Build paper_id → metadata index
        self.paper_metadata = self._build_paper_metadata(paper_mapping, paper_references)
        
        logger.info(f"ProvenanceConstrainedMatcher initialized:")
        logger.info(f"  Chunks mapped: {len(self.chunk_to_paper)}")
        logger.info(f"  Papers indexed: {len(self.paper_metadata)}")
    
    def _build_chunk_to_paper(self, chunks: List[Dict], paper_mapping: Dict) -> Dict[str, str]:
        """
        Build chunk_id → paper_id mapping.
        
        Academic chunks use paper_XXX pattern in chunk_id.
        Regulatory chunks use reg_XX pattern (not matched here).
        
        Note: chunks_embedded.jsonl uses 'chunk_ids' (list) not 'chunk_id' (str)
        """
        mapping = {}
        
        # Build eid → paper_id reverse lookup
        eid_to_paper = {}
        for paper_id, data in paper_mapping.items():
            eid = data.get('scopus_metadata', {}).get('eid', '')
            if eid:
                eid_to_paper[eid] = paper_id
        
        for chunk in chunks:
            # Handle both 'chunk_id' (str) and 'chunk_ids' (list) formats
            chunk_id = chunk.get('chunk_id') or (chunk.get('chunk_ids', [None])[0] if chunk.get('chunk_ids') else None)
            if not chunk_id:
                continue
                
            doc_id = chunk.get('document_id') or (chunk.get('document_ids', [None])[0] if chunk.get('document_ids') else None)
            
            # Try to extract paper_id from chunk_id pattern (paper_XXX_CHUNK_YYY)
            if chunk_id.startswith('paper_'):
                parts = chunk_id.split('_CHUNK_')
                if len(parts) >= 1:
                    paper_id = parts[0]
                    if paper_id in paper_mapping:
                        mapping[chunk_id] = paper_id
                        continue
            
            # Try document_id as eid
            if doc_id and doc_id in eid_to_paper:
                mapping[chunk_id] = eid_to_paper[doc_id]
            
            # Regulatory chunks (reg_XX) won't match - that's fine
        
        return mapping
    
    def _build_paper_metadata(
        self, 
        paper_mapping: Dict[str, Dict],
        references: Dict[str, List[str]]
    ) -> Dict[str, Dict]:
        """
        Build paper_id → {authors, journal, title, references} index.
        """
        metadata = {}
        
        for paper_id, data in paper_mapping.items():
            scopus = data.get('scopus_metadata', {})
            if not scopus:
                continue
            
            # Parse authors string "Surname1, I1; Surname2, I2; ..."
            authors = []
            authors_str = scopus.get('authors', '')
            if authors_str:
                for author_part in authors_str.split(';'):
                    author_part = author_part.strip()
                    if author_part:
                        surname = self._extract_surname(author_part)
                        initial = self._extract_initial(author_part)
                        if surname:
                            authors.append({
                                'name': author_part,
                                'surname': surname,
                                'initial': initial
                            })
            
            # Normalize journal name
            journal = self._normalize_text(scopus.get('journal', ''))
            
            # Normalize title
            title = self._normalize_text(scopus.get('title', ''))
            
            # Get raw references
            paper_refs = references.get(paper_id, [])
            
            metadata[paper_id] = {
                'eid': scopus.get('eid', ''),
                'authors': authors,
                'journal': journal,
                'title': title,
                'year': scopus.get('year'),
                'references': paper_refs,
                'references_normalized': [self._normalize_text(r) for r in paper_refs]
            }
        
        return metadata
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        
        Aligned with MinerUMatcher._normalize_text() for consistency.
        """
        if not text or text == "Unknown":
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _get_confidence_tier(self, score: float, match_type: str) -> str:
        """
        Get confidence tier based on score and match type.
        
        Aligned with MinerUMatcher confidence tiers.
        """
        if match_type == 'title' or match_type == 'document':
            if score >= self.TITLE_THRESHOLD_HIGH:
                return 'high'
            elif score >= self.TITLE_THRESHOLD_MEDIUM:
                return 'medium'
            elif score >= self.TITLE_THRESHOLD_LOW:
                return 'low'
        elif match_type == 'journal':
            if score >= self.JOURNAL_THRESHOLD_HIGH:
                return 'high'
            elif score >= self.JOURNAL_THRESHOLD_MEDIUM:
                return 'medium'
        elif match_type == 'author':
            if score >= self.AUTHOR_THRESHOLD_EXACT:
                return 'high'
            elif score >= self.AUTHOR_THRESHOLD_FUZZY:
                return 'medium'
        return 'low'
    
    def _extract_surname(self, name: str) -> str:
        """Extract surname from author name."""
        if not name:
            return ""
        # Handle "Last, First" format
        if ',' in name:
            return self._normalize_text(name.split(',')[0])
        # Handle "First Last" format
        parts = name.strip().split()
        if parts:
            return self._normalize_text(parts[-1])
        return ""
    
    def _extract_initial(self, name: str) -> str:
        """Extract first initial from author name."""
        if not name:
            return ""
        # Handle "Last, First" format
        if ',' in name:
            parts = name.split(',')
            if len(parts) > 1 and parts[1].strip():
                return parts[1].strip()[0].lower()
        # Handle "First Last" format
        parts = name.strip().split()
        if len(parts) > 1:
            return parts[0][0].lower()
        return ""
    
    def _similarity(self, a: str, b: str) -> float:
        """Calculate similarity ratio between two strings."""
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()
    
    def _get_source_paper(self, entity: Dict) -> Optional[str]:
        """Get source paper ID from entity's chunk provenance."""
        chunk_ids = entity.get('chunk_ids', [])
        for chunk_id in chunk_ids:
            if chunk_id in self.chunk_to_paper:
                return self.chunk_to_paper[chunk_id]
        return None
    
    def match_entity(self, entity: Dict) -> Optional[Dict]:
        """
        Match a metadata entity against its source paper's metadata.
        
        Returns:
            Match result dict or None if no match
        """
        entity_type = entity.get('type', '')
        
        # Get source paper
        paper_id = self._get_source_paper(entity)
        if not paper_id or paper_id not in self.paper_metadata:
            return None
        
        paper = self.paper_metadata[paper_id]
        
        # Try structured match first, then fallback to references
        if entity_type == 'Author':
            return self._match_author(entity, paper, paper_id)
        elif entity_type == 'Journal':
            return self._match_journal(entity, paper, paper_id)
        elif entity_type in ('Document', 'Citation'):
            return self._match_document(entity, paper, paper_id)
        
        return None
    
    def _match_author(self, entity: Dict, paper: Dict, paper_id: str) -> Optional[Dict]:
        """
        Match Author entity against paper's authors + references.
        
        Tier 1: Exact match against paper's own Scopus authors
        Tier 2: Fuzzy match against paper's own Scopus authors  
        Tier 3: Surname found in paper's reference strings
        """
        name = entity.get('name', '')
        surname = self._extract_surname(name)
        initial = self._extract_initial(name)
        
        if not surname or len(surname) < 2:
            return None
        
        # Tier 1 & 2: Match against paper's own authors
        for author in paper['authors']:
            # Exact surname + initial match
            if initial and author['initial']:
                if surname == author['surname'] and initial == author['initial']:
                    return {
                        'entity_id': entity['entity_id'],
                        'entity_name': name,
                        'target_id': f"author_{paper_id}_{author['surname']}_{author['initial']}",
                        'target_name': author['name'],
                        'target_type': 'Author',
                        'confidence': 1.0,
                        'confidence_tier': 'high',
                        'method': 'structured_exact',
                        'paper_id': paper_id
                    }
            
            # Fuzzy surname + initial match (using class threshold)
            if initial and author['initial'] == initial:
                score = self._similarity(surname, author['surname'])
                if score >= self.AUTHOR_THRESHOLD_FUZZY:
                    return {
                        'entity_id': entity['entity_id'],
                        'entity_name': name,
                        'target_id': f"author_{paper_id}_{author['surname']}_{author['initial']}",
                        'target_name': author['name'],
                        'target_type': 'Author',
                        'confidence': score,
                        'confidence_tier': self._get_confidence_tier(score, 'author'),
                        'method': 'structured_fuzzy',
                        'paper_id': paper_id
                    }
        
        # Tier 3: Fallback - check if surname appears in references
        if paper['references_normalized']:
            for i, ref_norm in enumerate(paper['references_normalized']):
                # Look for surname pattern (with word boundaries)
                if re.search(rf'\b{re.escape(surname)}\b', ref_norm):
                    # If we have initial, check for it too
                    if initial:
                        # Look for patterns like "surname, i." or "i. surname"
                        pattern1 = rf'\b{re.escape(surname)}\s*,\s*{re.escape(initial)}'
                        pattern2 = rf'\b{re.escape(initial)}\s*\.\s*{re.escape(surname)}'
                        if re.search(pattern1, ref_norm) or re.search(pattern2, ref_norm):
                            return {
                                'entity_id': entity['entity_id'],
                                'entity_name': name,
                                'target_id': f"ref_author_{paper_id}_{surname}_{initial}",
                                'target_name': f"{surname}, {initial}.",
                                'target_type': 'ReferenceAuthor',
                                'confidence': 0.80,
                                'confidence_tier': 'medium',
                                'method': 'reference_surname_initial',
                                'paper_id': paper_id,
                                'reference_index': i
                            }
                    else:
                        # Surname only - lower confidence
                        return {
                            'entity_id': entity['entity_id'],
                            'entity_name': name,
                            'target_id': f"ref_author_{paper_id}_{surname}",
                            'target_name': surname,
                            'target_type': 'ReferenceAuthor',
                            'confidence': 0.70,
                            'confidence_tier': 'low',
                            'method': 'reference_surname_only',
                            'paper_id': paper_id,
                            'reference_index': i
                        }
        
        return None
    
    def _match_journal(self, entity: Dict, paper: Dict, paper_id: str) -> Optional[Dict]:
        """
        Match Journal entity against paper's journal + references.
        
        Tier 1: Exact/fuzzy match against paper's own journal
        Tier 2: Journal name found in paper's reference strings
        """
        name = self._normalize_text(entity.get('name', ''))
        
        if not name or len(name) < 3:
            return None
        
        # Tier 1: Match against paper's own journal
        if paper['journal']:
            # Exact match
            if name == paper['journal']:
                return {
                    'entity_id': entity['entity_id'],
                    'entity_name': entity.get('name', ''),
                    'target_id': f"journal_{paper_id}",
                    'target_name': paper['journal'],
                    'target_type': 'Journal',
                    'confidence': 1.0,
                    'confidence_tier': 'high',
                    'method': 'structured_exact',
                    'paper_id': paper_id
                }
            
            # Fuzzy match (using class threshold)
            score = self._similarity(name, paper['journal'])
            if score >= self.JOURNAL_THRESHOLD_MEDIUM:
                return {
                    'entity_id': entity['entity_id'],
                    'entity_name': entity.get('name', ''),
                    'target_id': f"journal_{paper_id}",
                    'target_name': paper['journal'],
                    'target_type': 'Journal',
                    'confidence': score,
                    'confidence_tier': self._get_confidence_tier(score, 'journal'),
                    'method': 'structured_fuzzy',
                    'paper_id': paper_id
                }
        
        # Tier 2: Fallback - check references for journal name
        if paper['references_normalized'] and len(name) >= 5:
            for i, ref_norm in enumerate(paper['references_normalized']):
                if name in ref_norm:
                    return {
                        'entity_id': entity['entity_id'],
                        'entity_name': entity.get('name', ''),
                        'target_id': f"ref_journal_{paper_id}_{i}",
                        'target_name': entity.get('name', ''),
                        'target_type': 'ReferenceJournal',
                        'confidence': 0.75,
                        'confidence_tier': 'medium',
                        'method': 'reference_contains',
                        'paper_id': paper_id,
                        'reference_index': i
                    }
        
        return None
    
    def _match_document(self, entity: Dict, paper: Dict, paper_id: str) -> Optional[Dict]:
        """
        Match Document/Citation entity against paper's title + references.
        
        Tier 1: Fuzzy match against paper's own title (using MinerUMatcher thresholds)
        Tier 2: Significant keyword overlap with reference strings
        """
        name = entity.get('name', '')
        name_norm = self._normalize_text(name)
        
        if not name_norm or len(name_norm) < 5:
            return None
        
        # Tier 1: Match against paper's own title (lowered threshold to 0.70)
        if paper['title']:
            score = self._similarity(name_norm, paper['title'])
            if score >= self.TITLE_THRESHOLD_MEDIUM:
                return {
                    'entity_id': entity['entity_id'],
                    'entity_name': name,
                    'target_id': f"pub_{paper_id}",
                    'target_name': paper['title'][:50],
                    'target_type': 'Publication',
                    'confidence': score,
                    'confidence_tier': self._get_confidence_tier(score, 'document'),
                    'method': 'structured_title',
                    'paper_id': paper_id
                }
        
        # Tier 2: Fallback - keyword overlap with references
        if paper['references_normalized']:
            # Extract significant words (>3 chars)
            keywords = set(w for w in name_norm.split() if len(w) > 3)
            if len(keywords) < 2:
                return None
            
            best_match = None
            best_overlap = 0
            
            for i, ref_norm in enumerate(paper['references_normalized']):
                ref_words = set(ref_norm.split())
                overlap = len(keywords & ref_words)
                overlap_ratio = overlap / len(keywords) if keywords else 0
                
                if overlap >= 2 and overlap_ratio >= 0.5 and overlap > best_overlap:
                    best_overlap = overlap
                    conf = min(0.85, 0.5 + overlap_ratio * 0.35)
                    best_match = {
                        'entity_id': entity['entity_id'],
                        'entity_name': name,
                        'target_id': f"ref_doc_{paper_id}_{i}",
                        'target_name': paper['references'][i][:60] if i < len(paper['references']) else '',
                        'target_type': 'ReferenceDocument',
                        'confidence': conf,
                        'confidence_tier': self._get_confidence_tier(conf, 'document'),
                        'method': f'reference_keywords_{overlap}',
                        'paper_id': paper_id,
                        'reference_index': i
                    }
            
            if best_match:
                return best_match
        
        return None
    
    def match_all(self, entities: List[Dict]) -> Dict:
        """
        Match all metadata entities with provenance constraints.
        
        Returns:
            Dict with matches, unmatched, and stats
        """
        results = {
            'author_matches': [],
            'journal_matches': [],
            'document_matches': [],
            'unmatched': [],
            'stats': {
                'author_total': 0,
                'author_matched': 0,
                'journal_total': 0,
                'journal_matched': 0,
                'document_total': 0,
                'document_matched': 0,
                'no_provenance': 0
            }
        }
        
        for entity in tqdm(entities, desc="Matching"):
            entity_type = entity.get('type', '')
            
            # Track totals
            if entity_type == 'Author':
                results['stats']['author_total'] += 1
            elif entity_type == 'Journal':
                results['stats']['journal_total'] += 1
            elif entity_type in ('Document', 'Citation'):
                results['stats']['document_total'] += 1
            else:
                continue  # Skip non-metadata types
            
            # Check provenance
            paper_id = self._get_source_paper(entity)
            if not paper_id:
                results['stats']['no_provenance'] += 1
                results['unmatched'].append(entity)
                continue
            
            # Try to match
            match = self.match_entity(entity)
            
            if match:
                if entity_type == 'Author':
                    results['author_matches'].append(match)
                    results['stats']['author_matched'] += 1
                elif entity_type == 'Journal':
                    results['journal_matches'].append(match)
                    results['stats']['journal_matched'] += 1
                else:
                    results['document_matches'].append(match)
                    results['stats']['document_matched'] += 1
            else:
                results['unmatched'].append(entity)
        
        return results
    
    def generate_same_as_relations(self, results: Dict) -> List[Dict]:
        """
        Generate SAME_AS relations from matching results.
        
        Note: Only structured matches (to Scopus nodes) get SAME_AS.
        Reference matches are for validation/filtering only.
        """
        relations = []
        
        for match in results['author_matches']:
            # Only structured matches link to Scopus Author nodes
            if match['method'].startswith('structured'):
                relations.append({
                    'relation_type': 'SAME_AS',
                    'source_id': match['entity_id'],
                    'target_id': match['target_id'],
                    'target_type': 'Author',
                    'properties': {
                        'confidence': match['confidence'],
                        'method': match['method']
                    }
                })
        
        for match in results['journal_matches']:
            if match['method'].startswith('structured'):
                relations.append({
                    'relation_type': 'SAME_AS',
                    'source_id': match['entity_id'],
                    'target_id': match['target_id'],
                    'target_type': 'Journal',
                    'properties': {
                        'confidence': match['confidence'],
                        'method': match['method']
                    }
                })
        
        for match in results['document_matches']:
            if match['method'].startswith('structured'):
                relations.append({
                    'relation_type': 'SAME_AS',
                    'source_id': match['entity_id'],
                    'target_id': match['target_id'],
                    'target_type': 'Publication',
                    'properties': {
                        'confidence': match['confidence'],
                        'method': match['method']
                    }
                })
        
        return relations


# Keep old class name as alias for backward compatibility
MetadataMatcher = ProvenanceConstrainedMatcher