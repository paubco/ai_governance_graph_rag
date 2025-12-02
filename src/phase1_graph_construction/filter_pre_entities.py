"""
Pre-Entity Quality Filter (Stage 0)
Location: scripts/filter_pre_entities.py
Runs BEFORE Phase 1C disambiguation pipeline

Purpose: Remove metadata entities and low-quality extractions
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
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PreEntityFilter:
    """
    Conservative pre-entity quality filter
    
    Filters:
    1. TYPE-BASED: Remove metadata types (years, pages, DOIs, etc.)
    2. CHARACTER CLEANING: Fix unicode/control characters
    3. LENGTH VALIDATION: Remove unreasonably short/long entities
    4. CONTENT VALIDATION: Remove empty or non-textual entities
    5. CHUNK QUALITY: Remove single-mention entities (optional)
    
    Philosophy: Conservative - preserve legitimate entities, only remove obvious junk
    """
    
    def __init__(self, min_chunks: int = 1):
        """
        Initialize filter with conservative defaults
        
        Args:
            min_chunks: Minimum chunk mentions (default: 1 for Phase 1B, 
                       deduplication in Stage 1 will merge multi-mention entities)
        """
        self.min_chunks = min_chunks
        
        # TYPE-BASED FILTERING: Metadata types to ban
        self.banned_types = {
            # Time metadata
            'Year', 'Date', 'Publication Year', 'Publication Date', 
            'Time Period', 'Time Frame',
            
            # Document structure metadata
            'DOI', 'Digital Object Identifier', 'Digital Object Identifier (DOI)',
            'Page Range', 'Page Number', 'Pages', 'Article Pages',
            'Volume', 'Volume Number', 'Volume and Issue', 
            'Journal Volume and Issue', 'Journal Volume',
            'Issue', 'Journal Issue',
            'Section', 'Document Section', 'Regulatory Document Section', 'Section Title',
            'Article Number', 'Article ID', 'Article Identifier',
            'Chapter Title',
            
            # Visual/formatting metadata
            'Figure', 'Figure Reference', 'Figure Caption', 'Visual Reference', 'Visual Element',
            'Table', 'Footnote',
            'Mathematical Notation', 'Mathematical Expression',
            
            # Reference metadata
            'Article Reference', 'Journal Reference', 'Document Reference', 
            'Publication Details', 'Publication Identifier', 'Article Details',
            'Document Identifier', 'Identifier',
            
            # Contact/web metadata
            'Email', 'URL', 'Website', 'Contact Information', 'Address',
            
            # Other junk
            'Unknown',  # Entities with "Unknown" type
            'Journal Abbreviation',
            'Document Element',
        }
        
        # Statistics tracking
        self.stats = {
            'input_entities': 0,
            'banned_type': 0,
            'name_too_short': 0,
            'name_too_long': 0,
            'description_too_short': 0,
            'description_too_long': 0,
            'empty_name': 0,
            'no_letters': 0,
            'single_mention': 0,
            'output_entities': 0,
            'character_fixes': 0,
        }
    
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
    
    def check_length_validity(self, entity: Dict) -> Tuple[bool, str]:
        """
        Check if entity fields have reasonable lengths (conservative)
        
        Conservative thresholds:
        - Name: 2-200 chars (allows acronyms, prevents full sentences)
        - Description: 10-2000 chars (informative but not paragraphs)
        
        Returns:
            (keep: bool, reason: str)
        """
        name = entity.get('name', '')
        description = entity.get('description', '')
        
        # Name too short (< 2 chars) - likely extraction error
        if len(name) < 2:
            return False, 'name_too_short'
        
        # Name too long (> 200 chars) - probably full sentence
        if len(name) > 200:
            return False, 'name_too_long'
        
        # Description too short (< 10 chars) - not informative
        if len(description) < 30:
            return False, 'description_too_short'
        
        # Description too long (> 2000 chars) - probably paragraph
        if len(description) > 500:
            return False, 'description_too_long'
        
        return True, 'OK'
    
    def check_content_validity(self, entity: Dict) -> Tuple[bool, str]:
        """
        Check if entity has meaningful textual content
        
        Conservative checks:
        - Not empty
        - Contains at least one letter (allows "GPT-4", "3D", etc.)
        
        Returns:
            (keep: bool, reason: str)
        """
        name = entity.get('name', '').strip()
        
        # Empty or null
        if not name:
            return False, 'empty_name'
        
        # Only punctuation/numbers - must have at least one letter
        if not any(c.isalpha() for c in name):
            return False, 'no_letters'
        
        return True, 'OK'
    
    def check_chunk_quality(self, entity: Dict) -> Tuple[bool, str]:
        """
        Check entity has sufficient mentions (optional, configurable)
        
        NOTE: Phase 1B outputs 'chunk_id' (singular), but after deduplication
        entities will have 'chunk_ids' (plural). Handle both formats.
        
        Args:
            entity: Entity dict
            
        Returns:
            (keep: bool, reason: str)
        """
        # Try plural first (after deduplication)
        chunk_ids = entity.get('chunk_ids', [])
        
        # Phase 1B uses singular 'chunk_id' (before deduplication)
        if not chunk_ids and 'chunk_id' in entity:
            chunk_ids = [entity['chunk_id']]
        
        # Handle string format
        if isinstance(chunk_ids, str):
            chunk_ids = [chunk_ids]
        
        # Check minimum mentions
        if len(chunk_ids) < self.min_chunks:
            return False, 'single_mention'
        
        return True, 'OK'
    
    def flatten_entities(self, nested_data: Dict) -> List[Dict]:
        """
        Flatten nested entity structure from Phase 1B output
        
        Input structure:
        {
            "metadata": {...},
            "entities": [
                {
                    "chunk_id": "...",
                    "chunk_text": "...",
                    "entities": [
                        {"name": "...", "type": "...", "description": "...", "chunk_id": "..."}
                    ]
                }
            ]
        }
        
        Output: Flat list of entity dicts
        """
        flat = []
        
        entities_array = nested_data.get('entities', [])
        
        for chunk_obj in entities_array:
            chunk_entities = chunk_obj.get('entities', [])
            for entity in chunk_entities:
                # Ensure chunk_id is present
                if 'chunk_id' not in entity and 'chunk_id' in chunk_obj:
                    entity['chunk_id'] = chunk_obj['chunk_id']
                flat.append(entity)
        
        return flat
    
    def filter_entities(self, nested_data: Dict) -> Dict:
        """
        Main filtering pipeline
        
        Steps:
        1. Flatten nested structure
        2. Clean strings (unicode, control chars)
        3. Apply filters (type, length, content, chunks)
        4. Collect statistics
        5. Return in same nested format
        
        Args:
            nested_data: Input dict with nested entity structure
            
        Returns:
            Filtered data in same nested structure
        """
        logger.info("=" * 80)
        logger.info("PRE-ENTITY QUALITY FILTER (STAGE 0)")
        logger.info("=" * 80)
        logger.info("")
        
        # Flatten
        flat_entities = self.flatten_entities(nested_data)
        self.stats['input_entities'] = len(flat_entities)
        
        logger.info(f"Input: {len(flat_entities)} entities")
        logger.info(f"Minimum chunks: {self.min_chunks} (Phase 1B has single chunk_id per entity)")
        logger.info("")
        logger.info("Filtering...")
        
        # Filter entities
        clean_entities = []
        
        for entity in flat_entities:
            # Clean strings FIRST
            name_clean, name_modified = self.clean_string(entity.get('name', ''))
            desc_clean, desc_modified = self.clean_string(entity.get('description', ''))
            
            entity['name'] = name_clean
            entity['description'] = desc_clean
            
            if name_modified or desc_modified:
                self.stats['character_fixes'] += 1
            
            # Type filter
            keep, reason = self.check_type_filter(entity)
            if not keep:
                self.stats[reason] += 1
                continue
            
            # Length checks
            keep, reason = self.check_length_validity(entity)
            if not keep:
                self.stats[reason] += 1
                continue
            
            # Content checks
            keep, reason = self.check_content_validity(entity)
            if not keep:
                self.stats[reason] += 1
                continue
            
            # Chunk quality (uses self.min_chunks)
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
        logger.info("Removal reasons:")
        logger.info(f"  Banned type:          {self.stats['banned_type']:>8,}")
        logger.info(f"  Name too short:       {self.stats['name_too_short']:>8,}")
        logger.info(f"  Name too long:        {self.stats['name_too_long']:>8,}")
        logger.info(f"  Description short:    {self.stats['description_too_short']:>8,}")
        logger.info(f"  Description long:     {self.stats['description_too_long']:>8,}")
        logger.info(f"  Empty name:           {self.stats['empty_name']:>8,}")
        logger.info(f"  No letters in name:   {self.stats['no_letters']:>8,}")
        logger.info(f"  Single mention:       {self.stats['single_mention']:>8,}")
        logger.info("")
        logger.info(f"Character fixes:       {self.stats['character_fixes']:>8,}")
        logger.info("")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Pre-Entity Quality Filter (Conservative)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (conservative defaults)
    python scripts/filter_pre_entities.py \\
        --input data/interim/entities/pre_entities.json \\
        --output data/interim/entities/pre_entities_clean.json
    
    # Stricter filtering (3+ chunks required)
    python scripts/filter_pre_entities.py \\
        --input data/interim/entities/pre_entities.json \\
        --output data/interim/entities/pre_entities_clean.json \\
        --min-chunks 3
    
    # Keep all entities (default)
    python scripts/filter_pre_entities.py \\
        --input data/interim/entities/pre_entities.json \\
        --output data/interim/entities/pre_entities_clean.json

Conservative defaults:
    - Min name length: 2 chars (allows acronyms)
    - Max name length: 200 chars (prevents sentences)
    - Min description: 10 chars (must be informative)
    - Min chunks: 1 mention (Phase 1B has single chunk_id per entity)
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
    parser.add_argument(
        '--min-chunks',
        type=int,
        default=1,
        help='Minimum chunk mentions (default: 1, use 2+ for stricter filtering after deduplication)'
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
    filter = PreEntityFilter(min_chunks=args.min_chunks)
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