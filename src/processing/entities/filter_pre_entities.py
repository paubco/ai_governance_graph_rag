# -*- coding: utf-8 -*-
"""
Pre-Entity Quality Filter (Stage 0) - Updated with Academic Type Normalization
Location: scripts/filter_pre_entities.py
Runs BEFORE Phase 1C disambiguation pipeline

NEW: Collapses 121 academic entity types to ~15 canonical types
NEW: Expanded banned types with all identifier types

Purpose: Remove metadata entities, normalize academic types, filter low-quality extractions
Approach: Conservative - only remove obvious junk

Usage:
    python scripts/filter_pre_entities.py \\
        --input data/interim/entities/pre_entities.json \\
        --output data/interim/entities/pre_entities_clean.json
    
    # With stricter filtering
    python scripts/filter_pre_entities.py \\
        --input data/interim/entities/pre_entities.json \\
        --output data/interim/entities/pre_entities_clean.json \\
        --min-chunks 3
"""

import json
import argparse
import logging
import unicodedata
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import entity type classification
try:
    from src.utils.entity_type_classification import SKIP_TYPES
except ImportError:
    # Fallback if running from different directory
    SKIP_TYPES = {
        'DOI', 'Digital Object Identifier', 'Digital Object Identifier (DOI)',
        'ORCID', 'ISBN', 'ISSN', 'Chunk ID'
    }

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PreEntityFilter:
    """
    Conservative pre-entity quality filter with academic type normalization
    
    Filters:
    0. ACADEMIC TYPE NORMALIZATION: Collapse 121 academic types to ~15 canonical types
    1. TYPE-BASED: Remove metadata types (years, pages, identifiers, etc.)
    2. CHARACTER CLEANING: Fix unicode/control characters
    3. LENGTH VALIDATION: Remove unreasonably short/long entities
    4. CONTENT VALIDATION: Remove empty or non-textual entities
    5. CHUNK QUALITY: Remove single-mention entities (optional)
    
    Philosophy: Conservative - preserve legitimate entities, only remove obvious junk
    """
    
    def __init__(self):
        """
        Initialize filter with conservative defaults
        
        Single-mention filtering: ONLY removes entities that are BOTH:
        - Single mention (appears in 1 chunk only)
        - Short name (< 4 characters)
        
        Rationale: Legitimate entities often appear once (e.g., specific regulations),
        but short single-mention entities are usually junk/typos.
        """
        
        # ACADEMIC TYPE CANONICALIZATION MAP
        # Reduces 121 academic types → 15 canonical types
        self.academic_type_map = {
            # CITATIONS (all variants → "Citation")
            'Academic Citation': 'Citation',
            'Reference': 'Citation',
            'Academic Reference': 'Citation',
            'Citation': 'Citation',
            
            # AUTHORS (all variants → "Author")
            'Author': 'Author',
            'Authors': 'Author',
            'Author(s)': 'Author',
            'Author Group': 'Author',
            'Author List': 'Author',
            'Academic Authors': 'Author',
            'Author and Year': 'Citation',  # This is actually a citation
            'Author (Year)': 'Citation',
            'Author (2012)': 'Citation',
            'Author (2022)': 'Citation',
            'Authors (2019)': 'Citation',
            
            # EDITORS (all variants → "Editor")
            'Editor': 'Editor',
            'Editors': 'Editor',
            'Editor(s)': 'Editor',
            'Academic Editor': 'Editor',
            'Academic Editors': 'Editor',
            'Editorial Team': 'Editor',
            'Editor Citation': 'Editor',
            'Editors (2012)': 'Editor',
            
            # JOURNALS (all variants → "Journal")
            'Journal': 'Journal',
            'Journal Article': 'Journal',
            'Journal Article Reference': 'Journal',
            'Journal Name': 'Journal',
            'Journal Citation': 'Journal',
            'Journal Section': 'Journal',
            'Academic Journal': 'Journal',
            'Academic Journal Article': 'Journal',
            'Journal or Publication': 'Journal',
            'Journal and Publication Date': 'Journal',
            'Journal and Year': 'Journal',
            
            # PUBLICATIONS (general)
            'Publication': 'Publication',
            'Publication Source': 'Publication',
            'Publication Title': 'Publication',
            'Publication Type': 'Publication',
            'Publication Volume': 'Publication',
            'Publication Issue': 'Publication',
            'Publication Edition': 'Publication',
            'Publication Series': 'Publication',
            'Publication Status': 'Publication',
            'Publication Model': 'Publication',
            'Publication Month': 'Publication',
            'Publication Information': 'Publication',
            'Publication Metadata': 'Publication',
            'Publication Reference': 'Publication',
            'Publication Venue': 'Publication',
            'Academic Publication': 'Publication',
            
            # BOOKS (all variants → "Book")
            'Book': 'Book',
            'Book Title': 'Book',
            'Book Series': 'Book',
            'Book Chapter': 'Book',
            'Book Chapter Title': 'Book',
            'Book Subtitle': 'Book',
            'Book Volume': 'Book',
            'Book Edition': 'Book',
            'Book Section': 'Book',
            
            # PAPERS (all variants → "Paper")
            'Paper': 'Paper',
            'Paper Title': 'Paper',
            'Research Paper': 'Paper',
            'Research Paper Title': 'Paper',
            'Academic Paper': 'Paper',
            'Academic Paper Title': 'Paper',
            'Working Paper': 'Paper',
            'White Paper': 'Paper',
            'Discussion Paper': 'Paper',
            'Position Paper': 'Paper',
            'Review Paper': 'Paper',
            'Policy Paper': 'Paper',
            
            # ARTICLES (all variants → "Article")
            'Article Title': 'Article',
            'Article Subtitle': 'Article',
            'Academic Article': 'Article',
            'Journal Article': 'Article',
            'News Article': 'Article',
            
            # REPORTS (all variants → "Report")
            'Report Title': 'Report',
            'Research Report': 'Report',
            'Technical Report': 'Report',
            'Technical Report Title': 'Report',
            
            # CONFERENCES (all variants → "Conference")
            'Conference': 'Conference',
            'Conference Proceedings': 'Conference',
            'Conference Title': 'Conference',
            'Conference Full Name': 'Conference',
            'Conference Abbreviation': 'Conference',
            'Conference Acronym': 'Conference',
            'Conference or Event': 'Conference',
            'Conference or Journal': 'Conference',
            'Conference/Book Title': 'Conference',
            'Conference/Workshop': 'Conference',
            'Academic Conference': 'Conference',
            
            # THESIS/DISSERTATION
            'Thesis': 'Thesis',
            'Thesis Title': 'Thesis',
            'Dissertation': 'Thesis',
            'Manuscript': 'Thesis',
            
            # PREPRINTS
            'Preprint': 'Preprint',
            'Preprint Identifier': 'Preprint',
            
            # DOCUMENT TITLES (generic)
            'Document Title': 'Document',
            'Study Title': 'Document',
            'Research Title': 'Document',
            'Project Title': 'Document',
            
            # ACADEMIC INSTITUTIONS/DEPARTMENTS (keep as-is for now)
            'Academic Department': 'Academic Department',
            'Academic Institution': 'Academic Institution',
            'Academic Field': 'Academic Field',
            'Academic Discipline': 'Academic Discipline',
            
            # LITERATURE (general)
            'Literature': 'Literature',
            'Literature Type': 'Literature',
            'Literary Work': 'Literature',
            'Academic Literature': 'Literature',
            
            # PUBLISHER
            'Publisher': 'Publisher',
        }
        
        # TYPE-BASED FILTERING: Metadata types to ban
        # NEW: Includes all SKIP_TYPES identifiers
        self.banned_types = {
            # IDENTIFIERS (from SKIP_TYPES + expansions)
            'DOI', 'Digital Object Identifier', 'Digital Object Identifier (DOI)',
            'ORCID', 'ISBN', 'ISSN', 'Chunk ID',
            'arXiv Identifier', 'PubMed ID', 'PubMed Identifier',
            'Grant Number', 'Grant ID', 'Grant Agreement', 'Grant Identifier',
            'Protocol Code', 'Approval ID', 'Study Identifier',
            'Document ID', 'Identifier', 'Article ID', 'Article Identifier',
            'Article Number', 'Publication Identifier',
            'Database Identifier', 'Repository Name',
            
            # Time metadata
            'Year', 'Date', 'Publication Year', 'Publication Date', 
            'Time Period', 'Time Frame', 'Month', 'Decade',
            'Date Range', 'Time Range', 'Timeline', 'Time Span',
            'Publication Month', 'Access Date', 'Received Date', 
            'Accepted Date', 'Revised Date', 'Published Date',
            
            # Document structure metadata
            'Page Range', 'Page Number', 'Pages', 'Article Pages',
            'Volume', 'Volume Number', 'Volume and Issue', 
            'Journal Volume and Issue', 'Journal Volume',
            'Journal Volume and Pages', 'Journal Volume and Number',
            'Article Volume and Issue', 'Volume Identifier',
            'Volume and Pages', 'Volume or Edition',
            'Issue', 'Journal Issue', 'Issue Number', 'Publication Issue',
            'Section', 'Document Section', 'Regulatory Document Section', 
            'Section Title', 'Section Heading', 'Subsection', 'Section Reference',
            'Chapter', 'Chapter Title', 'Book Chapter Title',
            'Appendix', 'Footnote', 'Note',
            'Page', 'Page Reference', 'Page Indicator',
            
            # Visual/formatting metadata
            'Figure', 'Figure Reference', 'Figure Caption', 'Figure Title',
            'Visual Reference', 'Visual Element', 'Visual Aid',
            'Table', 'Table Reference', 'Table Title', 'Table Section',
            'Diagram', 'Graph', 'Chart', 'Image',
            'Mathematical Notation', 'Mathematical Expression',
            'Mathematical Formula', 'Mathematical Equation',
            'Equation', 'Equation Reference', 'Formula',
            
            # Reference metadata (keep citation/author types, ban metadata)
            'Article Reference', 'Journal Reference', 'Document Reference', 
            'Publication Details', 'Article Details',
            'Document Identifier',
            'Journal and Publication Date', 'Journal and Year',
            'Publication Information', 'Publication Metadata',
            'Journal Citation', 'Journal Section',
            
            # Contact/web metadata
            'Email', 'Email Address', 'Contact Email',
            'URL', 'Website', 'Web Resource', 'Web Page',
            'Contact Information', 'Address', 'Location Address',
            
            # Social media junk
            'Twitter Handle', 'Social Media Handle', 'Social Media Account',
            'Hashtag', 'Campaign', 'Social Media Platform',
            
            # Math/technical junk
            'Variable', 'Parameter', 'Technical Parameter',
            'Mathematical Concept', 'Mathematical Constraint',
            'Matrix', 'Mathematical Matrix', 'Algorithm Component',
            'Model Parameter', 'Threshold', 'Numerical Value',
            'Data Size', 'Sample Size', 'Dataset Size',
            'Hyperparameter', 'Hyper-Parameter',
            'Coefficient', 'P-value', 'Statistical Significance',
            'Statistical Parameter', 'Parameter Value', 'Parameter Setting',
            
            # Formatting/structure
            'Edition', 'Book Edition', 'Publication Edition',
            'Version', 'Software Version', 'Document Version',
            'Format', 'File Format', 'Data Format',
            'Subtitle', 'Book Subtitle', 'Article Subtitle',
            'Copyright', 'License', 'License URL',
            'Publisher Information', 'Publisher and Year',
            'Publisher and Location', 'Publisher Statement',
            'Imprint',
            
            # Other junk
            'Unknown', 'Unknown Entity', 'Unclear',
            'Journal Abbreviation', 'Conference Abbreviation',
            'Abbreviation', 'Acronym',
            'Document Element', 'Document Component', 'Document Structure',
            'Metadata', 'Document Metadata', 'Publication Metadata',
        }
        
        # PROTECTED TYPES: Keep even if single-appearance + short name
        # Rationale: These types often have legitimate 2-3 letter names (acronyms)
        self.protected_single_appearance = {
            # Academic entities (will enrich with Scopus)
            'Citation', 'Academic Citation', 'Reference',
            'Author', 'Authors', 'Author(s)', 'Editor',
            'Journal', 'Publication',
            'Book', 'Conference', 'Article', 'Paper', 'Document',
            
            # Regulatory documents (unique documents are legitimate)
            'Regulation', 'Regulatory Document', 'Legal Document',
            'Report', 'Policy', 'Directive', 'Law', 'Standard',
            
            # Organizations (often have acronyms: EU, UN, WHO, etc.)
            'Organization', 'Institution', 'Company',
            'Government Agency', 'Regulatory Body', 'Government Body',
            'International Organization',
            
            # Geographic entities (often have acronyms: EU, US, UK, etc.)
            'Country', 'Region', 'Location', 'City',
            
            # Legal/regulatory concepts (e.g., IP, AI, ML)
            'Legal Concept', 'Regulatory Concept', 'Legal Term',
            
            # Technologies (e.g., AI, ML, IT, IoT)
            'Technology', 'AI System', 'Framework', 'Methodology',
            
            # Core concepts (e.g., AI, ML)
            'Concept', 'Technical Term', 'Technical Concept',
        }
        
        # Statistics tracking
        self.stats = {
            'input_entities': 0,
            'academic_types_normalized': 0,  # NEW
            'banned_type': 0,
            'name_too_short': 0,
            'name_too_long': 0,
            'description_too_short': 0,
            'description_too_long': 0,
            'empty_name': 0,
            'no_letters': 0,
            'single_appear_short': 0,  # Single + short combo only
            'social_media_pattern': 0,
            'latex_notation': 0,
            'math_notation_dollars': 0,
            'high_special_char_density': 0,
            'output_entities': 0,
            'character_fixes': 0,
        }
    
    def normalize_academic_type(self, entity_type: str) -> str:
        """
        Normalize academic entity types to canonical forms
        
        Collapses 121 academic types → ~15 canonical types:
        - Citation (all citation/reference variants)
        - Author (all author variants)
        - Editor (all editor variants)
        - Journal (all journal variants)
        - Publication (generic publications)
        - Book (all book variants)
        - Paper (all paper variants)
        - Article (all article variants)
        - Report (all report variants)
        - Conference (all conference variants)
        - Thesis (thesis/dissertation/manuscript)
        - Preprint (preprints)
        - Document (generic document titles)
        - Literature (general literature)
        - Publisher (publishers)
        
        Args:
            entity_type: Original entity type
            
        Returns:
            Canonical type (or original if not in map)
        """
        return self.academic_type_map.get(entity_type, entity_type)
    
    def clean_string(self, text: str) -> Tuple[str, bool]:
        """
        Clean text while preserving valid content
        
        Fixes:
        - Control characters (0x00-0x1F except newline/tab)
        - Zero-width characters (invisible comparison issues)
        - Unicode normalize to NFC (canonical form)
        - Collapse whitespace to single spaces
        
        Args:
            text: Raw string
            
        Returns:
            (cleaned_text, was_modified)
        """
        original = text
        
        # Remove control characters (keep newline, tab)
        text = ''.join(ch for ch in text if ord(ch) >= 32 or ch in '\n\t')
        
        # Remove zero-width characters
        zero_width_chars = ['\u200B', '\u200C', '\u200D', '\uFEFF']
        for zw in zero_width_chars:
            text = text.replace(zw, '')
        
        # Unicode normalize to NFC (canonical composition)
        text = unicodedata.normalize('NFC', text)
        
        # Collapse whitespace
        text = ' '.join(text.split())
        text = text.strip()
        
        was_modified = (text != original)
        return text, was_modified
    
    def check_type_filter(self, entity: Dict) -> Tuple[bool, str]:
        """
        Check if entity type is banned metadata
        
        Returns:
            (keep: bool, reason: str)
        """
        entity_type = entity.get('type', '')
        
        if entity_type in self.banned_types:
            return False, 'banned_type'
        
        return True, 'OK'
    
    def check_pattern_bans(self, entity: Dict) -> Tuple[bool, str]:
        """
        Pattern-based entity bans (social media, math notation)
        
        Filters:
        - Social media: Starts with @ or #
        - LaTeX notation: Contains backslash
        - Math notation: Starts AND ends with $
        - High special char density: >40% special characters (with DOI exception)
        
        Returns:
            (keep: bool, reason: str)
        """
        name = entity.get('name', '')
        entity_type = entity.get('type', '')
        
        # Social media (@ or #)
        if name.startswith('@') or name.startswith('#'):
            return False, 'social_media_pattern'
        
        # LaTeX notation (backslash)
        if '\\' in name:
            return False, 'latex_notation'
        
        # Math notation: starts AND ends with $
        if name.startswith('$') and name.endswith('$'):
            return False, 'math_notation_dollars'
        
        # High special character density (>40%)
        # Exception: DOI (already filtered in banned_types, but double-check)
        if entity_type not in ['DOI', 'Digital Object Identifier']:
            letters = sum(1 for c in name if c.isalpha())
            special = sum(1 for c in name if not c.isalnum() and not c.isspace())
            
            if len(name) > 0 and special / len(name) > 0.4:
                return False, 'high_special_char_density'
        
        return True, 'OK'
    
    def check_length_validity(self, entity: Dict) -> Tuple[bool, str]:
        """
        Check entity name and description length validity
        
        Conservative defaults:
        - Name: 2-200 chars (allows acronyms, prevents sentences)
        - Description: 10-1000 chars (must be informative)
        
        Returns:
            (keep: bool, reason: str)
        """
        name = entity.get('name', '')
        description = entity.get('description', '')
        
        # Name length (2-200 chars)
        if len(name) < 2:
            return False, 'name_too_short'
        if len(name) > 200:
            return False, 'name_too_long'
        
        # Description length (10-1000 chars)
        if len(description) < 10:
            return False, 'description_too_short'
        if len(description) > 1000:
            return False, 'description_too_long'
        
        return True, 'OK'
    
    def check_content_validity(self, entity: Dict) -> Tuple[bool, str]:
        """
        Check entity name content validity
        
        Filters:
        - Empty name (after cleaning)
        - No alphabetic characters in name (e.g., "###", "123")
        
        Returns:
            (keep: bool, reason: str)
        """
        name = entity.get('name', '').strip()
        
        # Empty name
        if not name:
            return False, 'empty_name'
        
        # No letters (e.g., "###", "123", punctuation-only)
        has_letter = any(c.isalpha() for c in name)
        if not has_letter:
            return False, 'no_letters'
        
        return True, 'OK'
    
    def check_chunk_quality(self, entity: Dict) -> Tuple[bool, str]:
        """
        Check chunk mention quality
        
        CONSERVATIVE: Only remove single-mention entities if they're ALSO short (<4 chars)
        
        Rationale:
        - Single mention + long name (≥4 chars) = legitimate entity (keep)
        - Single mention + short name (<4 chars) = likely junk/typo (remove)
        - Exception: Protected types (citations, authors, journals) kept regardless
        
        Returns:
            (keep: bool, reason: str)
        """
        chunk_ids = entity.get('chunk_ids', [])
        name = entity.get('name', '')
        entity_type = entity.get('type', '')
        
        num_chunks = len(chunk_ids)
        
        # Protected types: keep even if single appearance
        if entity_type in self.protected_single_appearance:
            return True, 'OK'
        
        # ONLY filter: single appearance + short name = likely junk
        if num_chunks == 1 and len(name) < 4:
            return False, 'single_appear_short'
        
        # All other entities (including single-mention with name ≥4 chars): KEEP
        return True, 'OK'
    
    def _flatten_entities(self, nested_data: Dict) -> List[Dict]:
        """
        Flatten nested structure: {chunk_id: {entities: [...]}} -> flat list
        
        Adds chunk_id to each entity for later reconstruction
        """
        flat = []
        
        for chunk_obj in nested_data.get('entities', []):
            chunk_id = chunk_obj.get('chunk_id')
            
            for entity in chunk_obj.get('entities', []):
                entity['chunk_id'] = chunk_id
                flat.append(entity)
        
        return flat
    
    def filter_entities(self, nested_data: Dict) -> Dict:
        """
        Main filtering pipeline with academic type normalization
        
        Pipeline:
        0. Normalize academic types (collapse variants)
        1. Clean strings (unicode, whitespace)
        2. Type-based filtering (banned types)
        3. Pattern-based filtering (social media, math notation)
        4. Length validation (name, description)
        5. Content validation (empty, no letters)
        6. Chunk quality (single mention, short name)
        
        Args:
            nested_data: Nested structure from Phase 1B
            
        Returns:
            Filtered nested structure
        """
        # Flatten
        flat_entities = self._flatten_entities(nested_data)
        self.stats['input_entities'] = len(flat_entities)
        
        logger.info(f"Input entities: {len(flat_entities):,}")
        logger.info("")
        logger.info("Filtering...")
        
        # Filter entities
        clean_entities = []
        
        for entity in flat_entities:
            # STEP 0: Normalize academic types
            original_type = entity.get('type', '')
            normalized_type = self.normalize_academic_type(original_type)
            
            if normalized_type != original_type:
                entity['type'] = normalized_type
                entity['original_type'] = original_type  # Preserve for debugging
                self.stats['academic_types_normalized'] += 1
            
            # STEP 1: Clean strings
            name_clean, name_modified = self.clean_string(entity.get('name', ''))
            desc_clean, desc_modified = self.clean_string(entity.get('description', ''))
            
            entity['name'] = name_clean
            entity['description'] = desc_clean
            
            if name_modified or desc_modified:
                self.stats['character_fixes'] += 1
            
            # STEP 2: Type filter
            keep, reason = self.check_type_filter(entity)
            if not keep:
                self.stats[reason] += 1
                continue
            
            # STEP 3: Pattern-based filters
            keep, reason = self.check_pattern_bans(entity)
            if not keep:
                self.stats[reason] += 1
                continue
            
            # STEP 4: Length checks
            keep, reason = self.check_length_validity(entity)
            if not keep:
                self.stats[reason] += 1
                continue
            
            # STEP 5: Content checks
            keep, reason = self.check_content_validity(entity)
            if not keep:
                self.stats[reason] += 1
                continue
            
            # STEP 6: Chunk quality
            keep, reason = self.check_chunk_quality(entity)
            if not keep:
                self.stats[reason] += 1
                continue
            
            # Passed all filters
            clean_entities.append(entity)
        
        self.stats['output_entities'] = len(clean_entities)
        
        # Reconstruct nested structure
        output_data = self._reconstruct_nested(nested_data, clean_entities)
        
        # Log statistics
        self._log_stats()
        
        return output_data
    
    def _reconstruct_nested(self, original_data: Dict, flat_entities: List[Dict]) -> Dict:
        """
        Reconstruct nested structure from flat entity list
        
        Groups entities back by chunk_id to preserve original structure
        """
        # Build lookup: chunk_id -> list of entities
        chunk_map = {}
        for entity in flat_entities:
            chunk_id = entity.get('chunk_id')
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = []
            chunk_map[chunk_id].append(entity)
        
        # Reconstruct
        output = {
            'metadata': original_data.get('metadata', {}),
            'entities': []
        }
        
        # Update metadata
        output['metadata']['filtered'] = True
        output['metadata']['filter_stats'] = self.stats.copy()
        
        # Rebuild nested structure
        for chunk_obj in original_data.get('entities', []):
            chunk_id = chunk_obj.get('chunk_id')
            
            if chunk_id in chunk_map and chunk_map[chunk_id]:
                # Keep chunk with filtered entities
                new_chunk = {
                    'chunk_id': chunk_id,
                    'chunk_text': chunk_obj.get('chunk_text', ''),
                    'entities': chunk_map[chunk_id]
                }
                output['entities'].append(new_chunk)
        
        return output
    
    def _log_stats(self):
        """Log filtering statistics"""
        logger.info("")
        logger.info("=" * 80)
        logger.info("FILTERING COMPLETE")
        logger.info("=" * 80)
        logger.info("")
        logger.info(f"Input entities:    {self.stats['input_entities']:>8,}")
        logger.info(f"Output entities:   {self.stats['output_entities']:>8,}")
        logger.info(f"Removed:           {self.stats['input_entities'] - self.stats['output_entities']:>8,} "
                   f"({100 * (self.stats['input_entities'] - self.stats['output_entities']) / self.stats['input_entities']:.1f}%)")
        logger.info("")
        logger.info("Academic type normalization:")
        logger.info(f"  Types normalized:  {self.stats['academic_types_normalized']:>8,} "
                   f"(121 types → ~15 canonical)")
        logger.info("")
        logger.info("Removal reasons:")
        logger.info(f"  Banned type:          {self.stats['banned_type']:>8,}")
        logger.info(f"  Social media (@, #):  {self.stats['social_media_pattern']:>8,}")
        logger.info(f"  LaTeX notation (\\):   {self.stats['latex_notation']:>8,}")
        logger.info(f"  Math notation ($...$): {self.stats['math_notation_dollars']:>8,}")
        logger.info(f"  High special chars:   {self.stats['high_special_char_density']:>8,}")
        logger.info(f"  Name too short:       {self.stats['name_too_short']:>8,}")
        logger.info(f"  Name too long:        {self.stats['name_too_long']:>8,}")
        logger.info(f"  Description short:    {self.stats['description_too_short']:>8,}")
        logger.info(f"  Description long:     {self.stats['description_too_long']:>8,}")
        logger.info(f"  Empty name:           {self.stats['empty_name']:>8,}")
        logger.info(f"  No letters in name:   {self.stats['no_letters']:>8,}")
        logger.info(f"  Single + short (<4):  {self.stats['single_appear_short']:>8,}")
        logger.info("")
        logger.info(f"Character fixes:       {self.stats['character_fixes']:>8,}")
        logger.info("")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Pre-Entity Quality Filter with Academic Type Normalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python scripts/filter_pre_entities.py \\
        --input data/interim/entities/pre_entities.json \\
        --output data/interim/entities/pre_entities_clean.json

Conservative defaults:
    - Min name length: 2 chars (allows acronyms)
    - Max name length: 200 chars (prevents sentences)
    - Min description: 10 chars (must be informative)
    - Single-mention filtering: Only removes if BOTH single-mention AND short (<4 chars)
    
NEW: Academic type normalization
    - Collapses 121 academic types → 15 canonical types
    - Example: "Book Title", "Book Chapter Title" → "Book"
    - Example: "Author", "Authors", "Author(s)" → "Author"
    - Preserves original_type field for debugging
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input file (pre_entities.json from Phase 1B)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file (pre_entities_clean.json)'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info("")
    
    # Load data
    logger.info("Loading entities...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter
    filter = PreEntityFilter()
    filtered_data = filter.filter_entities(data)
    
    # Save
    logger.info(f"Saving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    
    logger.info("✓ Complete!")
    logger.info("")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())