# -*- coding: utf-8 -*-
"""
Module: citations.py
Package: src/utils
Purpose: Citation formatting for papers and regulations

Provides proper attribution for retrieved chunks:
- Scopus papers: Authors, Year, Title, Journal, DOI/URL
- DLA Piper regulations: Jurisdiction, URL

Usage:
    from src.utils.citations import CitationFormatter
    
    formatter = CitationFormatter()  # Loads paper_mapping.json
    citation = formatter.format(doc_id='paper_001', doc_type='paper')
    # Returns: {'authors': 'Smith et al.', 'year': 2023, 'title': '...', ...}

Author: Pau Barba i Colomer
Created: 2025-12-27
"""

import json
from pathlib import Path
from typing import Dict, Optional, Any


class CitationFormatter:
    """
    Format citations for papers and regulations.
    
    Loads paper_mapping.json once and provides citation metadata
    for any doc_id.
    """
    
    # Default paths
    DEFAULT_PAPER_MAPPING = Path('data/raw/academic/scopus_2023/paper_mapping.json')
    DLA_PIPER_BASE_URL = 'https://intelligence.dlapiper.com/artificial-intelligence/?t=01-law&c='
    
    def __init__(self, paper_mapping_path: Path = None, project_root: Path = None):
        """
        Initialize citation formatter.
        
        Args:
            paper_mapping_path: Path to paper_mapping.json
            project_root: Project root directory (for resolving relative paths)
        """
        self.project_root = project_root or Path(__file__).resolve().parent.parent.parent
        
        # Resolve paper mapping path
        if paper_mapping_path:
            self.paper_mapping_path = Path(paper_mapping_path)
        else:
            self.paper_mapping_path = self.project_root / self.DEFAULT_PAPER_MAPPING
        
        # Load paper mapping
        self.paper_mapping = {}
        self._load_paper_mapping()
    
    def _load_paper_mapping(self):
        """Load paper metadata from JSON file."""
        if self.paper_mapping_path.exists():
            with open(self.paper_mapping_path) as f:
                self.paper_mapping = json.load(f)
            print(f"CitationFormatter: Loaded {len(self.paper_mapping)} paper citations")
        else:
            print(f"CitationFormatter: Warning - Paper mapping not found at {self.paper_mapping_path}")
    
    def format(self, doc_id: str, doc_type: str, jurisdiction: str = None) -> Dict[str, Any]:
        """
        Format citation metadata for a document.
        
        Args:
            doc_id: Document identifier (e.g., 'paper_001', 'FR')
            doc_type: Type of document ('paper' or 'regulation')
            jurisdiction: Jurisdiction code for regulations
            
        Returns:
            Dict with citation fields:
                - authors: Author string
                - year: Publication year
                - title: Document title
                - journal: Journal name (papers only)
                - doi: DOI string (papers only)
                - url: Direct URL to source
                - source_type: 'paper' or 'regulation'
        """
        if doc_type == 'paper':
            return self._format_paper(doc_id)
        elif doc_type == 'regulation':
            return self._format_regulation(doc_id, jurisdiction)
        else:
            return self._format_unknown(doc_id)
    
    def _format_paper(self, doc_id: str) -> Dict[str, Any]:
        """Format citation for a Scopus paper."""
        if doc_id in self.paper_mapping:
            meta = self.paper_mapping[doc_id].get('scopus_metadata', {})
            
            # Process authors
            authors = meta.get('authors', 'Unknown')
            if authors.count(';') > 2:
                first_author = authors.split(';')[0].strip()
                authors = f"{first_author} et al."
            
            # Build URL
            doi = meta.get('doi', '')
            link = meta.get('link', '')
            url = f"https://doi.org/{doi}" if doi else link
            
            return {
                'authors': authors,
                'year': meta.get('year'),
                'title': meta.get('title', 'Untitled'),
                'journal': meta.get('journal', ''),
                'doi': doi,
                'url': url,
                'source_type': 'paper'
            }
        else:
            return {
                'authors': None,
                'year': None,
                'title': doc_id,
                'journal': None,
                'doi': None,
                'url': None,
                'source_type': 'paper'
            }
    
    def _format_regulation(self, doc_id: str, jurisdiction: str = None) -> Dict[str, Any]:
        """Format citation for a DLA Piper regulation."""
        jur = jurisdiction or doc_id
        return {
            'authors': 'DLA Piper',
            'year': 2024,
            'title': f'AI Laws of the World - {jur}',
            'journal': None,
            'doi': None,
            'url': f"{self.DLA_PIPER_BASE_URL}{jur}#insight",
            'source_type': 'regulation'
        }
    
    def _format_unknown(self, doc_id: str) -> Dict[str, Any]:
        """Format citation for unknown document type."""
        return {
            'authors': None,
            'year': None,
            'title': doc_id,
            'journal': None,
            'doi': None,
            'url': None,
            'source_type': 'unknown'
        }
    
    def format_string(self, doc_id: str, doc_type: str, jurisdiction: str = None) -> str:
        """
        Format citation as a display string.
        
        Args:
            doc_id: Document identifier
            doc_type: Type of document
            jurisdiction: Jurisdiction code for regulations
            
        Returns:
            Formatted citation string (e.g., "Smith et al. (2023). Title. Journal.")
        """
        cit = self.format(doc_id, doc_type, jurisdiction)
        
        if cit['authors'] and cit['year']:
            result = f"{cit['authors']} ({cit['year']}). \"{cit['title']}\""
            if cit['journal']:
                result += f" {cit['journal']}."
            return result
        else:
            return cit['title'] or doc_id
    
    def enrich_chunks(self, chunks: list) -> list:
        """
        Add citation metadata to a list of chunk dicts.
        
        Args:
            chunks: List of chunk dicts with 'doc_id', 'doc_type', 'jurisdiction'
            
        Returns:
            Same list with 'citation' field added to each chunk
        """
        for chunk in chunks:
            chunk['citation'] = self.format(
                doc_id=chunk.get('doc_id', ''),
                doc_type=chunk.get('doc_type', ''),
                jurisdiction=chunk.get('jurisdiction')
            )
        return chunks


# Convenience function for one-off use
def format_citation(doc_id: str, doc_type: str, jurisdiction: str = None, 
                    paper_mapping: dict = None) -> Dict[str, Any]:
    """
    Format a single citation without instantiating CitationFormatter.
    
    For repeated use, instantiate CitationFormatter directly.
    
    Args:
        doc_id: Document identifier
        doc_type: Type of document ('paper' or 'regulation')
        jurisdiction: Jurisdiction code for regulations
        paper_mapping: Pre-loaded paper mapping dict (optional)
        
    Returns:
        Citation metadata dict
    """
    if doc_type == 'paper' and paper_mapping and doc_id in paper_mapping:
        meta = paper_mapping[doc_id].get('scopus_metadata', {})
        authors = meta.get('authors', 'Unknown')
        if authors.count(';') > 2:
            first_author = authors.split(';')[0].strip()
            authors = f"{first_author} et al."
        
        doi = meta.get('doi', '')
        link = meta.get('link', '')
        
        return {
            'authors': authors,
            'year': meta.get('year'),
            'title': meta.get('title', 'Untitled'),
            'journal': meta.get('journal', ''),
            'doi': doi,
            'url': f"https://doi.org/{doi}" if doi else link,
            'source_type': 'paper'
        }
    
    elif doc_type == 'regulation':
        jur = jurisdiction or doc_id
        return {
            'authors': 'DLA Piper',
            'year': 2024,
            'title': f'AI Laws of the World - {jur}',
            'journal': None,
            'doi': None,
            'url': f"https://intelligence.dlapiper.com/artificial-intelligence/?t=01-law&c={jur}#insight",
            'source_type': 'regulation'
        }
    
    else:
        return {
            'authors': None,
            'year': None,
            'title': doc_id,
            'journal': None,
            'doi': None,
            'url': None,
            'source_type': doc_type or 'unknown'
        }