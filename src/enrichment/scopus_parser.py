# -*- coding: utf-8 -*-
"""
Scopus bibliometric data parser for L1 publication metadata integration.

Parses Scopus CSV exports to extract L1 publications, authors, and journals for
Phase 2A metadata enrichment. The ScopusParser class reads Scopus export CSVs,
extracting publication metadata (title, DOI, year, citations), author data with
Scopus IDs, and journal information. The ReferenceParser class parses semicolon-
delimited reference strings from the References field to extract citation metadata
(author, title, year, journal) for citation matching.

The parser handles Scopus-specific CSV format with BOM encoding (utf-8-sig), splits
multi-value fields (authors, references) on semicolons, and deduplicates authors
and journals across publications. Author names are cleaned to remove ID suffixes.
Reference strings follow Scopus format: "Surname, Given Names, Title, Journal,
Volume, Pages, (Year)". All extracted entities receive generated IDs for graph
node creation.

Examples:
    # Initialize parser with Scopus CSV export
    from src.enrichment.scopus_parser import ScopusParser, ReferenceParser
    from pathlib import Path

    parser = ScopusParser(Path('data/raw/scopus_export.csv'))

    # Parse publications, authors, and journals
    pubs, authors, journals = parser.parse_publications()
    print(f"Extracted {len(pubs)} publications")  # 50 publications
    print(f"Found {len(authors)} unique authors")  # 120 authors
    print(f"Found {len(journals)} unique journals")  # 30 journals

    # Inspect publication metadata
    pub = pubs[0]
    print(pub['title'])  # "AI Ethics in Healthcare Systems"
    print(pub['year'])  # 2023
    print(pub['cited_by'])  # 15

    # Parse references for citation matching
    ref_parser = ReferenceParser()
    refs_lookup = ref_parser.parse_all_references(pubs)

    # Get references for a specific publication
    scopus_id = pubs[0]['scopus_id']
    refs = refs_lookup[scopus_id]
    print(f"Publication has {len(refs)} references")

    # Inspect parsed reference
    ref = refs[0]
    print(ref['author'])  # "Smith, J."
    print(ref['title'])  # "Machine Learning in Medicine"
    print(ref['year'])  # 2022

References:
    Scopus CSV Export: Scopus bibliometric database export format with BOM
    ARCHITECTURE.md § 3.2.1: Phase 2A metadata enrichment design
    PHASE_2A_DESIGN.md: Citation matching pipeline with L1/L2 layers
    src.utils.id_generator: Publication/author/journal ID generation
"""
# Standard library
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Project imports
from src.utils.id_generator import (
    generate_publication_id,
    generate_author_id,
    generate_journal_id
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# SCOPUS CSV PARSER
# =============================================================================

class ScopusParser:
    """
    Parser for Scopus CSV export files.
    
    Extracts L1 publications, authors, journals, and references from
    Scopus bibliometric data.
    
    Example:
        parser = ScopusParser(Path('data/raw/scopus.csv'))
        pubs, authors, journals = parser.parse_publications()
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
        
        logger.info(f"Parsing Scopus CSV: {self.csv_path}")
        
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
                self._extract_authors(row, scopus_id, authors_dict)
                
                # Extract journal
                self._extract_journal(row, journals_dict)
        
        logger.info(f"Parsed {len(publications)} publications, "
                   f"{len(authors_dict)} authors, {len(journals_dict)} journals")
        
        return publications, list(authors_dict.values()), list(journals_dict.values())
    
    def _extract_publication(self, row: Dict, scopus_id: str) -> Dict:
        """Extract publication data from CSV row."""
        return {
            'publication_id': generate_publication_id(scopus_id, layer=1),
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
    
    def _extract_authors(self, row: Dict, scopus_id: str, authors_dict: Dict):
        """Extract authors from CSV row into authors_dict."""
        author_ids = row.get('Author(s) ID', '').strip()
        author_names = row.get('Author full names', '').strip()
        
        if not (author_ids and author_names):
            return
        
        id_list = [a.strip() for a in author_ids.split(';') if a.strip()]
        name_list = [a.strip() for a in author_names.split(';') if a.strip()]
        
        # Handle case where we have more IDs than names or vice versa
        for i, author_id in enumerate(id_list):
            if not author_id or author_id in authors_dict:
                continue
            
            # Get corresponding name if available
            if i < len(name_list):
                author_name = name_list[i]
                # Parse "Surname, Firstname (ID)" - remove ID suffix
                name_clean = re.sub(r'\s*\(\d+\)', '', author_name).strip()
            else:
                name_clean = f"Author {author_id}"
            
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


# =============================================================================
# REFERENCE PARSER
# =============================================================================

class ReferenceParser:
    """
    Parser for Scopus References field.
    
    Extracts citation metadata from semicolon-delimited reference strings.
    
    Example:
        parser = ReferenceParser()
        refs = parser.parse_all_references(publications)
        # Returns: {"scopus_id": [{"author": "...", "title": "...", ...}, ...]}
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
        
        # Scopus format: parts[0]=Surname, parts[1]=Given, parts[2]=Title, parts[3]=Journal
        if len(parts) >= 2:
            author = f"{parts[0]}, {parts[1]}"
        elif len(parts) == 1:
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
            publications: List of publication dicts with 'references' field
            
        Returns:
            Dict mapping scopus_id -> list of parsed references
        """
        references_lookup = {}
        total_refs = 0
        
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
            total_refs += len(parsed_refs)
        
        logger.info(f"Parsed {total_refs} references from {len(publications)} publications")
        
        return references_lookup