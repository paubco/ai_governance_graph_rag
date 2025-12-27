# -*- coding: utf-8 -*-
"""
Two-stage pre-entity filtering for garbage removal and provenance validation.

Filters extracted pre-entities using blacklist patterns and source text provenance
checks. Stage 1 removes obvious garbage using regex patterns (single characters,
numbers only, markdown artifacts, special characters). Stage 2 verifies entities
appear in source chunk text via case-insensitive substring matching with special
character normalization. Tracks filter statistics by entity type and discard reason
for quality monitoring.

The filter loads chunk text into memory for fast provenance lookups. Blacklist
patterns catch LLM hallucinations and extraction artifacts (", *, #, etc.). Provenance
check catches entities mentioned in prompts or examples but not in actual chunk text.
All discarded entities are logged with reasons (blacklist match or provenance failure)
for analysis. Pass rate and discard distribution inform extraction quality.

Examples:
    # Initialize with chunk lookup
    from src.processing.entities.pre_entity_filter import PreEntityFilter

    chunks_by_id = {c.chunk_id: c.text for c in chunks}
    filter = PreEntityFilter(chunks_by_id)

    # Filter pre-entities
    clean_entities, stats = filter.filter(pre_entities)
    print(f"Kept: {len(clean_entities)}, Discarded: {stats['total_discarded']}")
    print(f"Pass rate: {stats['pass_rate']:.1f}%")

    # Inspect discard reasons
    for reason, count in stats['discard_reasons'].items():
        print(f"{reason}: {count}")

    # Type-specific statistics
    for type_name, type_stats in stats['by_type'].items():
        print(f"{type_name}: {type_stats['kept']}/{type_stats['total']}")

References:
    config.extraction_config.ENTITY_FILTER_CONFIG: Blacklist patterns
    src.utils.dataclasses.PreEntity: Pre-entity dataclass with chunk IDs
"""
import re
import logging
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# Import config (single source of truth)
from config.extraction_config import PRE_ENTITY_FILTER_CONFIG

# Optional: rapidfuzz for provenance check
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# LOAD PATTERNS FROM CONFIG
# =============================================================================

# Compile patterns once at module load
_PATTERNS_CASE_SENSITIVE = [
    re.compile(p) for p in PRE_ENTITY_FILTER_CONFIG['blacklist_case_sensitive']
]
_PATTERNS_CASE_INSENSITIVE = [
    re.compile(p, re.IGNORECASE) for p in PRE_ENTITY_FILTER_CONFIG['blacklist_case_insensitive']
]
_NUMERIC_PATTERNS = [
    re.compile(p) for p in PRE_ENTITY_FILTER_CONFIG['numeric_patterns']
]
_NUMERIC_ALLOWED_TYPES: Set[str] = set(PRE_ENTITY_FILTER_CONFIG['numeric_allowed_types'])
_PROVENANCE_THRESHOLD: float = PRE_ENTITY_FILTER_CONFIG['provenance_threshold']

# Type-specific blacklists (v2.0)
_DOCUMENT_PATTERNS = [
    re.compile(p) for p in PRE_ENTITY_FILTER_CONFIG.get('document_blacklist', [])
]
_DOCUMENT_PATTERNS_CI = [
    re.compile(p, re.IGNORECASE) for p in PRE_ENTITY_FILTER_CONFIG.get('document_blacklist_ci', [])
]
_DOCUMENT_SECTION_PATTERNS = [
    re.compile(p) for p in PRE_ENTITY_FILTER_CONFIG.get('document_section_blacklist', [])
]


# =============================================================================
# FILTER FUNCTIONS
# =============================================================================

def is_garbage(name: str, entity_type: str = None) -> bool:
    """
    Check if entity name matches garbage patterns.
    
    Type-aware filtering:
        - Numeric patterns only apply outside Citation type
        - Document/DocumentSection have specific blacklists (v2.0)
    
    Args:
        name: Entity name to check
        entity_type: Entity type (for type-aware filtering)
        
    Returns:
        True if entity is garbage, False otherwise
    """
    name = name.strip()
    if not name:
        return True
    
    # Case-sensitive patterns
    for pattern in _PATTERNS_CASE_SENSITIVE:
        if pattern.match(name):
            return True
    
    # Case-insensitive patterns
    for pattern in _PATTERNS_CASE_INSENSITIVE:
        if pattern.match(name):
            return True
    
    # Numeric patterns (skip for allowed types like Citation)
    if entity_type not in _NUMERIC_ALLOWED_TYPES:
        for pattern in _NUMERIC_PATTERNS:
            if pattern.match(name):
                return True
    
    # Type-specific blacklists (v2.0)
    if entity_type == 'Document':
        for pattern in _DOCUMENT_PATTERNS:
            if pattern.match(name):
                return True
        for pattern in _DOCUMENT_PATTERNS_CI:
            if pattern.match(name):
                return True
    
    elif entity_type == 'DocumentSection':
        for pattern in _DOCUMENT_SECTION_PATTERNS:
            if pattern.match(name):
                return True
    
    return False


def check_provenance(entity_name: str, chunk_text: str) -> Tuple[bool, float]:
    """
    Verify entity appears in source chunk using fuzzy matching.
    
    Handles variations like:
        - "EU AI Act" in "the EU AI Act regulates..." → exact match
        - "European AI Act" in "EU AI Act" → fuzzy match
    
    Args:
        entity_name: Entity name to verify
        chunk_text: Source chunk text
        
    Returns:
        (is_valid, similarity_score)
    """
    if not chunk_text:
        return True, 1.0  # No chunk text = can't verify, assume valid
    
    # Fast path: exact substring match (case-insensitive)
    if entity_name.lower() in chunk_text.lower():
        return True, 1.0
    
    # Slow path: fuzzy match for paraphrasing
    if RAPIDFUZZ_AVAILABLE:
        score = fuzz.partial_ratio(entity_name.lower(), chunk_text.lower()) / 100.0
        return score >= _PROVENANCE_THRESHOLD, score
    else:
        # Fallback: simple word overlap
        entity_words = set(entity_name.lower().split())
        chunk_words = set(chunk_text.lower().split())
        if not entity_words:
            return False, 0.0
        overlap = len(entity_words & chunk_words) / len(entity_words)
        return overlap >= _PROVENANCE_THRESHOLD, overlap


class PreEntityFilter:
    """
    Two-stage pre-entity filter: blacklist patterns + provenance verification.
    
    Usage:
        chunks = load_chunks_as_dict()  # {chunk_id: text}
        filter = PreEntityFilter(chunks)
        clean_entities, stats = filter.filter(pre_entities)
    """
    
    def __init__(self, chunks: Dict[str, str] = None, verify_provenance: bool = None):
        """
        Initialize filter.
        
        Args:
            chunks: Dict mapping chunk_id → chunk_text (for provenance check)
            verify_provenance: Override config setting (default from config)
        """
        self.chunks = chunks or {}
        
        # Use config default if not specified
        if verify_provenance is None:
            verify_provenance = PRE_ENTITY_FILTER_CONFIG['verify_provenance']
        self.verify_provenance = verify_provenance and bool(chunks)
        
        self.stats = {
            'input_count': 0,
            'blacklist_removed': 0,
            'provenance_failed': 0,
            'output_count': 0,
            'by_type': defaultdict(lambda: {'input': 0, 'blacklist': 0, 'provenance': 0, 'output': 0}),
        }
        self.hallucinations = []  # For thesis analysis
        
        if self.verify_provenance:
            logger.info(f"PreEntityFilter initialized with {len(chunks)} chunks for provenance")
            if not RAPIDFUZZ_AVAILABLE:
                logger.warning("rapidfuzz not available, using word overlap for provenance")
        else:
            logger.info("PreEntityFilter initialized (blacklist only, no provenance check)")
    
    def filter(self, entities: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Filter entities through blacklist + provenance.
        
        Args:
            entities: List of pre-entity dicts with 'name', 'type', 'chunk_id'
            
        Returns:
            (clean_entities, stats_dict)
        """
        logger.info(f"Filtering {len(entities)} entities...")
        
        self.stats['input_count'] = len(entities)
        self.stats['provenance_checked'] = 0
        self.stats['provenance_skipped_no_chunk'] = 0
        self.stats['provenance_exact_match'] = 0
        self.stats['provenance_fuzzy_match'] = 0
        clean = []
        
        for entity in entities:
            name = entity.get('name', '')
            entity_type = entity.get('type', '')
            chunk_id = entity.get('chunk_id', '')
            
            # Track by type
            self.stats['by_type'][entity_type]['input'] += 1
            
            # Stage 1: Blacklist
            if is_garbage(name, entity_type):
                self.stats['blacklist_removed'] += 1
                self.stats['by_type'][entity_type]['blacklist'] += 1
                continue
            
            # Stage 2: Provenance
            if self.verify_provenance:
                chunk_text = self.chunks.get(chunk_id, '')
                if chunk_text:
                    self.stats['provenance_checked'] += 1
                    is_valid, score = check_provenance(name, chunk_text)
                    
                    # Track match type for diagnostics
                    if is_valid and score == 1.0:
                        self.stats['provenance_exact_match'] += 1
                    elif is_valid:
                        self.stats['provenance_fuzzy_match'] += 1
                    
                    if not is_valid:
                        self.stats['provenance_failed'] += 1
                        self.stats['by_type'][entity_type]['provenance'] += 1
                        self.hallucinations.append({
                            'name': name,
                            'type': entity_type,
                            'chunk_id': chunk_id,
                            'score': round(score, 3),
                        })
                        continue
                else:
                    self.stats['provenance_skipped_no_chunk'] += 1
            
            # Passed all filters
            clean.append(entity)
            self.stats['by_type'][entity_type]['output'] += 1
        
        self.stats['output_count'] = len(clean)
        
        # Convert defaultdict to regular dict for JSON serialization
        self.stats['by_type'] = dict(self.stats['by_type'])
        
        # Log summary
        logger.info(f"Filtering complete:")
        logger.info(f"  Input:      {self.stats['input_count']:,}")
        logger.info(f"  Blacklist:  {self.stats['blacklist_removed']:,} removed")
        logger.info(f"  Provenance: {self.stats['provenance_failed']:,} failed")
        logger.info(f"    - Checked:     {self.stats['provenance_checked']:,}")
        logger.info(f"    - Exact match: {self.stats['provenance_exact_match']:,}")
        logger.info(f"    - Fuzzy match: {self.stats['provenance_fuzzy_match']:,}")
        logger.info(f"    - No chunk:    {self.stats['provenance_skipped_no_chunk']:,}")
        logger.info(f"  Output:     {self.stats['output_count']:,}")
        
        return clean, dict(self.stats)
    
    def get_hallucinations(self) -> List[Dict]:
        """Get list of entities that failed provenance check."""
        return self.hallucinations


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_chunks_as_dict(chunks_file: str) -> Dict[str, str]:
    """
    Load chunks from JSONL into {chunk_id: text} dict.
    
    Handles both formats:
        - chunk_id (string) — old format
        - chunk_ids (list) — new format (v1.1+)
    
    Args:
        chunks_file: Path to chunks JSONL file
        
    Returns:
        Dict mapping chunk_id → chunk text
    """
    import json
    
    chunks = {}
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunk = json.loads(line)
                text = chunk.get('text', chunk.get('content', ''))
                
                if not text:
                    continue
                
                # Handle both formats
                if 'chunk_ids' in chunk:
                    # New format: chunk_ids is a list
                    chunk_ids = chunk['chunk_ids']
                    if isinstance(chunk_ids, list):
                        for cid in chunk_ids:
                            if cid:
                                chunks[cid] = text
                    elif chunk_ids:
                        chunks[chunk_ids] = text
                elif 'chunk_id' in chunk:
                    # Old format: chunk_id is a string
                    chunk_id = chunk['chunk_id']
                    if chunk_id:
                        chunks[chunk_id] = text
    
    logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")
    return chunks