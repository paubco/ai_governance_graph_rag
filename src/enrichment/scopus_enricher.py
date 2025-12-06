# -*- coding: utf-8 -*-
"""
Module: scopus_enricher.py
Package: src.enrichment
Purpose: Core classes for Scopus metadata enrichment and citation matching

Author: Pau Barba i Colomer
Created: 2025-12-05
Modified: 2025-12-05

References:
    - See docs/PHASE_2A_DESIGN.md for full specification
    - See docs/ARCHITECTURE.md § 4 for graph schema
"""

# Standard library
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

# Project root
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
from rapidfuzz import fuzz

# Local
from src.utils.id_generator import (
    generate_publication_id,
    generate_author_id,
    generate_journal_id
)


# ==============================================================================
# SCOPUS CSV PARSER
# ==============================================================================

class ScopusParser:
    """
    Parser for Scopus CSV export files.
    
    Extracts L1 publications, authors, journals, and references from
    Scopus bibliometric data.
    """
    
    def __init__(self, csv_path: Path):
        """
        Initialize parser with CSV path.
        
        Args:
            csv_path: Path to Scopus export CSV
        """
        self.csv_path = csv_path
    
    def parse_publications(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Parse CSV to extract publications, authors, and journals.
        
        Returns:
            Tuple of (publications, authors, journals) as lists of dicts
        """
        publications = []
        authors_dict = {}
        journals_dict = {}
        
        with open(self.csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                scopus_id = row.get('EID', '').strip()
                if not scopus_id:
                    continue
                
                # Extract publication
                pub = self._extract_publication(row, scopus_id)
                publications.append(pub)
                
                # Extract authors
                self._extract_authors(row, authors_dict)
                
                # Extract journal
                self._extract_journal(row, journals_dict)
        
        return publications, list(authors_dict.values()), list(journals_dict.values())
    
    def _extract_publication(self, row: Dict, scopus_id: str) -> Dict:
        """Extract publication data from CSV row."""
        return {
            'publication_id': generate_publication_id("pub_l1", scopus_id),
            'scopus_id': scopus_id,
            'title': row.get('Title', '').strip(),
            'year': int(row.get('Year', 0)) if row.get('Year') else None,
            'doi': row.get('DOI', '').strip(),
            'source_title': row.get('Source title', '').strip(),
            'volume': row.get('Volume', '').strip(),
            'cited_by': int(row.get('Cited by', 0)) if row.get('Cited by') else 0,
            'abstract': row.get('Abstract', '').strip(),
            'keywords': row.get('Author Keywords', '').strip(),
            'document_type': row.get('Document Type', '').strip(),
            'references': row.get('References', '').strip(),
            'node_type': 'source_publication'
        }
    
    def _extract_authors(self, row: Dict, authors_dict: Dict):
        """Extract authors from CSV row into authors_dict."""
        author_ids = row.get('Author(s) ID', '').strip()
        author_names = row.get('Author full names', '').strip()
        
        if not (author_ids and author_names):
            return
        
        id_list = [a.strip() for a in author_ids.split(';')]
        name_list = [a.strip() for a in author_names.split(';')]
        
        for author_id, author_name in zip(id_list, name_list):
            if author_id and author_name and author_id not in authors_dict:
                # Parse "Surname, Firstname (ID)"
                name_clean = re.sub(r'\s*\(\d+\)', '', author_name).strip()
                
                authors_dict[author_id] = {
                    'author_id': generate_author_id(author_id),
                    'scopus_author_id': author_id,
                    'name': name_clean
                }
    
    def _extract_journal(self, row: Dict, journals_dict: Dict):
        """Extract journal from CSV row into journals_dict."""
        journal_name = row.get('Source title', '').strip()
        issn = row.get('ISSN', '').strip()
        
        if not journal_name:
            return
        
        journal_id = generate_journal_id(journal_name)
        if journal_id not in journals_dict:
            journals_dict[journal_id] = {
                'journal_id': journal_id,
                'name': journal_name,
                'issn': issn
            }


# ==============================================================================
# REFERENCE PARSER
# ==============================================================================

class ReferenceParser:
    """
    Parser for Scopus References field.
    
    Extracts citation metadata from semicolon-delimited reference strings.
    """
    
    @staticmethod
    def parse_reference_string(ref_string: str) -> Dict:
        """
        Parse a single reference string.
        
        Scopus format (consistent):
            "Surname, Given Names, Title, Journal, Volume, Pages, (Year)"
        
        Args:
            ref_string: Raw reference string
            
        Returns:
            Dict with parsed fields: author, title, year, journal, raw
        """
        ref_string = ref_string.strip()
        
        # Extract year
        year = ReferenceParser._extract_year(ref_string)
        
        # Split by comma
        parts = [p.strip() for p in ref_string.split(',') if p.strip()]
        
        # Remove year part (YYYY) or (YYYY)
        parts = [p for p in parts if not re.match(r'^\(?\d{4}\)?$', p.strip())]
        
        # Default empty
        author = ""
        title = ""
        journal = ""
        
        # Scopus format: parts[0]=Surname, parts[1]=Given, parts[2]=Title, parts[3]=Journal, ...
        if len(parts) >= 2:
            # Combine first two as author
            author = f"{parts[0]}, {parts[1]}"
        elif len(parts) == 1:
            # Just surname
            author = parts[0]
        
        if len(parts) >= 3:
            title = parts[2]
        
        if len(parts) >= 4:
            journal = parts[3]
        
        return {
            'author': author,
            'title': title,
            'year': year,
            'journal': journal,
            'raw': ref_string
        }
    
    @staticmethod
    def _extract_year(text: str) -> Optional[int]:
        """Extract 4-digit year from text."""
        match = re.search(r'\b(19|20)\d{2}\b', text)
        return int(match.group()) if match else None
    
    def parse_all_references(
        self,
        publications: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """
        Parse References field for all publications.
        
        Args:
            publications: List of publication dicts
            
        Returns:
            Dict mapping scopus_id -> list of parsed references
        """
        references_lookup = {}
        
        for pub in publications:
            scopus_id = pub['scopus_id']
            refs_field = pub.get('references', '')
            
            if not refs_field:
                references_lookup[scopus_id] = []
                continue
            
            # Split by semicolon (each ref ends with ;)
            ref_strings = [r.strip() for r in refs_field.split(';') if r.strip()]
            
            parsed_refs = []
            for ref_str in ref_strings:
                parsed = self.parse_reference_string(ref_str)
                parsed['source_scopus_id'] = scopus_id
                parsed_refs.append(parsed)
            
            references_lookup[scopus_id] = parsed_refs
        
        return references_lookup


# ==============================================================================
# CITATION ENTITY IDENTIFIER
# ==============================================================================

class CitationEntityIdentifier:
    """
    Identify citation entities that have discusses relations.
    
    Logic: Entity must be SUBJECT of a 'discusses' relation AND have an academic type.
    This ensures every identified citation has topic information.
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
                    discusses_subjects[subj_id].append(r.get('object', ''))
        
        print(f"  [DEBUG] Found {discusses_count} discusses relations")
        print(f"  [DEBUG] Unique subjects in discusses: {len(discusses_subjects)}")
        
        # Step 2: Build entity lookup by ID
        entity_by_id = {}
        for entity in entities:
            eid = entity.get('entity_id')
            if not eid:
                from src.utils.id_generator import generate_entity_id
                eid = generate_entity_id(entity['name'])
            entity_by_id[eid] = entity
        
        print(f"  [DEBUG] Entities with IDs: {len(entity_by_id)}")
        
        # Step 3: Find overlap
        overlap_ids = set(discusses_subjects.keys()) & set(entity_by_id.keys())
        print(f"  [DEBUG] Overlap (entities that are discusses subjects): {len(overlap_ids)}")
        
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
        
        print(f"  [DEBUG] Type distribution of discusses subjects (top 10):")
        sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])[:10]
        for t, c in sorted_types:
            marker = "✓" if t in self.ACADEMIC_TYPES else " "
            print(f"    {marker} {t}: {c}")
        
        print(f"  [DEBUG] After academic type filter: {len(citation_entities)}")
        
        return citation_entities


# ==============================================================================
# CITATION MATCHER
# ==============================================================================

class CitationMatcher:
    """
    Matches citation entities to publication references using fuzzy matching.
    
    Implements provenance-constrained matching: entities are matched only
    against references from their source L1 paper.
    """
    
    # Match confidence thresholds
    CONFIDENCE = {
        'doi_exact': 1.0,
        'year_exact': 1.0,        # Exact year match
        'author_year': 0.85,      # Author surname + year
        'title_fuzzy': 0.75       # Title fuzzy match
    }
    
    # Fuzzy match threshold
    TITLE_FUZZY_THRESHOLD = 0.90
    L1_OVERLAP_THRESHOLD = 0.90
    
    def __init__(
        self,
        references_lookup: Dict[str, List[Dict]],
        chunk_to_l1: Dict[str, str],
        l1_publications: List[Dict]
    ):
        """
        Initialize matcher with reference data.
        
        Args:
            references_lookup: Dict of scopus_id -> parsed references
            chunk_to_l1: Mapping from chunk_id to scopus_id
            l1_publications: List of L1 publication dicts
        """
        self.references_lookup = references_lookup
        self.chunk_to_l1 = chunk_to_l1
        self.l1_publications = l1_publications
        
        # L2 deduplication registry
        self.l2_registry = {}  # (author_norm, year, title_norm) -> l2_pub_id
        self.l2_data_cache = {}  # l2_pub_id -> parsed_ref (for later retrieval)
    
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
        
        # Get source L1 paper
        source_chunk = chunk_ids[0]
        source_l1_id = self.chunk_to_l1.get(source_chunk)
        
        if not source_l1_id:
            return None
        
        # Get references for this L1 paper
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
        
        # Check L1 overlap
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
        
        Returns:
            (matched_ref, confidence, method) or (None, 0.0, "")
        """
        entity_norm = self._normalize_name(entity_name)
        
        best_match = None
        best_score = 0.0
        best_method = ""
        
        # Author entities
        if entity_type in ['Author']:
            # Extract surname from entity name
            surname = self._extract_author_surname(entity_name)
            if not surname:
                return (None, 0.0, "")
            
            for ref in references:
                ref_author = self._normalize_name(ref.get('author', ''))
                if not ref_author:
                    continue
                
                # Use partial_ratio for substring matching
                author_sim = fuzz.partial_ratio(surname.lower(), ref_author.lower()) / 100.0
                
                # Boost if surname appears at start
                if ref_author.lower().startswith(surname.lower()):
                    author_sim = min(1.0, author_sim + 0.1)
                
                if author_sim >= 0.80:  # Slightly lower threshold for partial matching
                    if author_sim > best_score:
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
                
                if journal_sim >= 0.80:
                    if journal_sim > best_score:
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
                
                if title_sim >= 0.75:
                    if title_sim > best_score:
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
        Uses partial_ratio for substring matching.
        
        Returns:
            (matched_ref, confidence, method) or (None, 0.0, "")
        """
        entity_norm = self._normalize_name(entity_name)
        
        best_match = None
        best_score = 0.0
        best_method = ""
        
        for ref in references:
            ref_author = self._normalize_name(ref.get('author', ''))
            ref_title = self._normalize_name(ref.get('title', ''))
            ref_journal = self._normalize_name(ref.get('journal', ''))
            
            # Try all fields with partial_ratio for substring matching
            for field_name, field_value in [
                ('author', ref_author),
                ('title', ref_title),
                ('journal', ref_journal)
            ]:
                if not field_value:
                    continue
                
                # Use partial_ratio for better substring matching
                sim = fuzz.partial_ratio(entity_norm, field_value) / 100.0
                
                if sim >= 0.65:  # Lower threshold for partial matching
                    if sim > best_score:
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
        
        Extract year and surname from entity, match combinations
        with references.
        
        Returns:
            (matched_ref, confidence, method) or (None, 0.0, "")
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
                        score = 0.65  # Partial confidence
                        best_method = 'partial_year_surname'
            
            # Just surname with high confidence
            elif entity_surname and not entity_year:
                if entity_surname in ref_author.lower():
                    # Boost if surname appears at start
                    if ref_author.lower().startswith(entity_surname):
                        score = 0.60
                        best_method = 'partial_surname_start'
            
            # Just year (very weak)
            elif entity_year and not entity_surname:
                if ref_year == entity_year:
                    score = 0.50
                    best_method = 'partial_year_only'
            
            if score > best_score:
                best_match = ref
                best_score = score
        
        return (best_match, best_score, best_method)
    
    def _fuzzy_match(
        self,
        entity_name: str,
        references: List[Dict]
    ) -> Tuple[Optional[Dict], float, str]:
        """
        Fuzzy match entity name to list of references.
        
        Returns:
            (matched_ref, confidence, method) or (None, 0.0, "")
        """
        entity_year = self._extract_year(entity_name)
        entity_author = self._extract_author_surname(entity_name)
        entity_norm = self._normalize_name(entity_name)
        
        best_match = None
        best_score = 0.0
        best_method = ""
        
        for ref in references:
            ref_year = ref.get('year')
            ref_author = self._normalize_name(ref.get('author', ''))
            ref_title = self._normalize_name(ref.get('title', ''))
            
            # Strategy 1: Year exact + Author
            if entity_year and ref_year == entity_year:
                # Year match is certain
                if entity_author and (entity_author in ref_author or 
                                     ref_author.startswith(entity_author)):
                    # Author + year match
                    score = self.CONFIDENCE['author_year']
                    if score > best_score:
                        best_match = ref
                        best_score = score
                        best_method = 'author_year'
                else:
                    # Just year match
                    score = self.CONFIDENCE['year_exact']
                    if score > best_score and ref_title:
                        # Need some title similarity too
                        title_sim = fuzz.ratio(entity_norm, ref_title) / 100.0
                        if title_sim >= 0.5:  # Loose threshold for year match
                            best_match = ref
                            best_score = score
                            best_method = 'year_exact'
            
            # Strategy 2: Title fuzzy match
            if ref_title:
                title_sim = fuzz.ratio(entity_norm, ref_title) / 100.0
                
                if title_sim >= self.TITLE_FUZZY_THRESHOLD:
                    score = self.CONFIDENCE['title_fuzzy']
                    if score > best_score:
                        best_match = ref
                        best_score = score
                        best_method = 'title_fuzzy'
        
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
            # Year must match
            if l1['year'] != ref_year:
                continue
            
            # Fuzzy match on title
            l1_title = self._normalize_name(l1['title'])
            similarity = fuzz.ratio(ref_title, l1_title) / 100.0
            
            if similarity >= self.L1_OVERLAP_THRESHOLD:
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
        
        # Check registry
        registry_key = (author_norm, year, title_norm)
        
        if registry_key in self.l2_registry:
            return self.l2_registry[registry_key]
        
        # Create new L2
        l2_pub_id = generate_publication_id("pub_l2", parsed_ref['raw'])
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