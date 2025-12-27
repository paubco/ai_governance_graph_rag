# -*- coding: utf-8 -*-
"""
Test Enrichment

# ==============================================================================
# REFERENCE PARSING TESTS
# ==============================================================================

"""
"""
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from src.enrichment.scopus_enricher import (
    ReferenceParser,
    CitationMatcher,
    CitationEntityIdentifier
)


# ==============================================================================
# REFERENCE PARSING TESTS
# ==============================================================================

def test_parse_reference_with_year():
    """Reference parsing extracts year correctly."""
    ref = "Floridi, L., Digital Ethics, Philosophy & Technology, 2018"
    parsed = ReferenceParser.parse_reference_string(ref)
    
    assert parsed['year'] == 2018
    assert 'Floridi' in parsed['author']
    assert 'L.' in parsed['author']
    # New parser: part[2] is title
    assert parsed['title'] == 'Digital Ethics'
    assert parsed['raw'] == ref


def test_parse_reference_year_in_parentheses():
    """Reference with year in (2018) format."""
    ref = "Zhou, Y. (2025). RAKG Construction"
    parsed = ReferenceParser.parse_reference_string(ref)
    
    assert parsed['year'] == 2025
    assert 'Zhou' in parsed['author']


def test_parse_reference_without_year():
    """Reference without year returns None."""
    ref = "Smith, J.; Some Paper; Nature"
    parsed = ReferenceParser.parse_reference_string(ref)
    
    assert parsed['year'] is None
    # Parser takes first semicolon-delimited part as author
    assert 'Smith' in parsed['author']
    assert parsed['raw'] == ref


# ==============================================================================
# CITATION MATCHING TESTS
# ==============================================================================

def test_match_with_author_and_year():
    """Entity with author + year matches reference."""
    # Setup references lookup
    refs = [{
        'author': 'Floridi, L.',
        'year': 2018,
        'title': 'Digital Ethics',
        'raw': 'Floridi, L.; Digital Ethics; Philosophy; 2018',
        'source_scopus_id': '2-s2.0-85123456'
    }]
    refs_lookup = {'2-s2.0-85123456': refs}
    
    # Setup chunk mapping
    chunk_mapping = {'paper_001_CHUNK_042': '2-s2.0-85123456'}
    
    # Create matcher
    matcher = CitationMatcher(refs_lookup, chunk_mapping, [])
    
    # Entity data
    entity_data = {
        'name': 'Floridi (2018)',
        'chunk_ids': ['paper_001_CHUNK_042']
    }
    
    # Test matching
    result = matcher.match_entity_to_reference('ent_test', entity_data)
    
    assert result is not None
    matched_ref, match_result, confidence, method = result
    assert matched_ref['year'] == 2018
    assert confidence >= 0.85  # author_year or year_exact


def test_match_requires_chunk_provenance():
    """Matching fails without chunk IDs (no provenance)."""
    matcher = CitationMatcher({}, {}, [])
    
    entity_data = {
        'name': 'Some Citation',
        'chunk_ids': []  # No chunks
    }
    
    result = matcher.match_entity_to_reference('ent_test', entity_data)
    
    assert result is None


def test_match_requires_academic_chunk():
    """Matching fails if chunk not from L1 publication."""
    matcher = CitationMatcher({}, {}, [])
    
    entity_data = {
        'name': 'Some Citation',
        'chunk_ids': ['reg_EU_CHUNK_001']  # Regulation chunk
    }
    
    result = matcher.match_entity_to_reference('ent_test', entity_data)
    
    assert result is None


# ==============================================================================
# L1 OVERLAP DETECTION TESTS
# ==============================================================================

def test_l1_overlap_detection_positive():
    """Corpus self-citation is detected (cited paper is also source paper)."""
    l1_pubs = [
        {
            'scopus_id': '2-s2.0-85987654321',
            'title': 'Digital Ethics and AI Systems',
            'year': 2018,
            'publication_id': 'pub_l1_85987654321'
        }
    ]
    
    matcher = CitationMatcher({}, {}, l1_pubs)
    
    # Reference that matches L1 publication
    ref = {
        'title': 'Digital Ethics and AI Systems',
        'year': 2018,
        'author': 'Floridi, L.'
    }
    
    is_l1, scopus_id, conf = matcher._check_l1_overlap(ref)
    
    assert is_l1 == True
    assert scopus_id == '2-s2.0-85987654321'
    assert conf >= 0.90  # High confidence title match


def test_l1_overlap_detection_negative():
    """Non-overlapping reference returns False."""
    l1_pubs = [
        {
            'scopus_id': '2-s2.0-111',
            'title': 'Completely Different Paper',
            'year': 2020,
            'publication_id': 'pub_l1_111'
        }
    ]
    
    matcher = CitationMatcher({}, {}, l1_pubs)
    
    ref = {
        'title': 'Digital Ethics',
        'year': 2018,
        'author': 'Floridi, L.'
    }
    
    is_l1, scopus_id, conf = matcher._check_l1_overlap(ref)
    
    assert is_l1 == False


# ==============================================================================
# CONFIDENCE SCORING TESTS
# ==============================================================================

def test_confidence_year_exact_is_1_0():
    """CRITICAL: Year match confidence must be 1.0 (year is certain)."""
    matcher = CitationMatcher({}, {}, [])
    
    assert matcher.CONFIDENCE['year_exact'] == 1.0
    assert matcher.CONFIDENCE['author_year'] == 0.85
    assert matcher.CONFIDENCE['title_fuzzy'] == 0.75


def test_entity_identification_confidence_levels():
    """Entity identification requires discusses + academic type."""
    identifier = CitationEntityIdentifier()
    
    # New logic uses ACADEMIC_TYPES set
    assert 'Citation' in identifier.ACADEMIC_TYPES
    assert 'Author' in identifier.ACADEMIC_TYPES
    assert 'Paper' in identifier.ACADEMIC_TYPES


# ==============================================================================
# ENTITY IDENTIFICATION TESTS
# ==============================================================================

def test_identify_typed_citations():
    """Entities need discusses relation + academic type."""
    identifier = CitationEntityIdentifier()
    
    entities = [
        {
            'entity_id': 'ent_1',
            'name': 'Floridi (2018)',
            'type': 'Citation',  # Academic type
            'chunk_ids': ['chunk_001']
        }
    ]
    
    relations = [
        {
            'subject_id': 'ent_1',
            'subject': 'Floridi (2018)',
            'predicate': 'discusses',
            'object_id': 'ent_2',
            'object': 'digital ethics'
        }
    ]
    
    citation_entities = identifier.identify(entities, relations)
    
    assert 'ent_1' in citation_entities
    assert citation_entities['ent_1']['type'] == 'Citation'
    assert 'discusses_objects' in citation_entities['ent_1']


def test_identify_discusses_subjects():
    """Entities need discusses relation + academic type."""
    identifier = CitationEntityIdentifier()
    
    entities = [
        {
            'entity_id': 'ent_1',
            'name': 'Zhou et al. (2025)',
            'type': 'Citation',  # Now has academic type
            'chunk_ids': ['chunk_001']
        }
    ]
    
    relations = [
        {
            'subject_id': 'ent_1',
            'subject': 'Zhou et al. (2025)',
            'predicate': 'discusses',
            'object_id': 'ent_2',
            'object': 'knowledge graphs'
        }
    ]
    
    citation_entities = identifier.identify(entities, relations)
    
    assert 'ent_1' in citation_entities
    assert citation_entities['ent_1']['type'] == 'Citation'
    assert 'knowledge graphs' in citation_entities['ent_1']['discusses_objects']


# ==============================================================================
# STRING UTILITY TESTS
# ==============================================================================

def test_normalize_name():
    """Name normalization for matching."""
    matcher = CitationMatcher({}, {}, [])
    
    assert matcher._normalize_name("EU AI Act") == "eu ai act"
    assert matcher._normalize_name("Chat-GPT") == "chatgpt"
    assert matcher._normalize_name("  Spaces  ") == "spaces"


def test_extract_year():
    """Year extraction from text."""
    matcher = CitationMatcher({}, {}, [])
    
    assert matcher._extract_year("Floridi (2018)") == 2018
    assert matcher._extract_year("Published in 2023") == 2023
    assert matcher._extract_year("No year here") is None


def test_extract_author_surname():
    """Author surname extraction."""
    matcher = CitationMatcher({}, {}, [])
    
    assert matcher._extract_author_surname("Floridi (2018)") == "floridi"
    assert matcher._extract_author_surname("Smith et al.") == "smith"
    # Quoted titles: extracts first word (acceptable heuristic)
    assert matcher._extract_author_surname('"Some Title"') == "some"


# ==============================================================================
# TIERED MATCHING TESTS
# ==============================================================================

def test_tier1_author_matching():
    """Tier 1: Author entity matches author field."""
    refs = [
        {'author': 'Floridi, Luciano', 'title': 'Digital Ethics', 'year': 2018, 'journal': 'Philosophy & Technology'},
        {'author': 'Smith, John', 'title': 'Other Paper', 'year': 2019, 'journal': 'Nature'}
    ]
    
    matcher = CitationMatcher({}, {}, [])
    
    # Test author entity matching
    matched, conf, method = matcher._type_aware_match('Floridi', 'Author', refs)
    
    assert matched is not None
    assert matched['author'] == 'Floridi, Luciano'
    assert method == 'type_author'
    assert conf >= 0.85


def test_tier1_journal_matching():
    """Tier 1: Journal entity matches journal field."""
    refs = [
        {'author': 'Floridi, L.', 'title': 'Paper 1', 'year': 2018, 'journal': 'Nature'},
        {'author': 'Smith, J.', 'title': 'Paper 2', 'year': 2019, 'journal': 'Science'}
    ]
    
    matcher = CitationMatcher({}, {}, [])
    
    # Test journal entity matching
    matched, conf, method = matcher._type_aware_match('Nature', 'Journal', refs)
    
    assert matched is not None
    assert matched['journal'] == 'Nature'
    assert method == 'type_journal'
    assert conf >= 0.80


def test_tier1_title_matching():
    """Tier 1: Citation entity matches title field."""
    refs = [
        {'author': 'Floridi, L.', 'title': 'Digital Ethics and AI', 'year': 2018, 'journal': 'Nature'},
        {'author': 'Smith, J.', 'title': 'Other Topic', 'year': 2019, 'journal': 'Science'}
    ]
    
    matcher = CitationMatcher({}, {}, [])
    
    # Test citation/paper entity matching title
    matched, conf, method = matcher._type_aware_match('Digital Ethics and AI', 'Citation', refs)
    
    assert matched is not None
    assert 'Digital Ethics' in matched['title']
    assert method == 'type_title'
    assert conf >= 0.75


def test_tier2_fuzzy_fallback():
    """Tier 2: Entity matches any field when type is wrong/unclear."""
    refs = [
        {'author': 'Floridi, Luciano', 'title': 'Digital Ethics', 'year': 2018, 'journal': 'Philosophy & Technology'}
    ]
    
    matcher = CitationMatcher({}, {}, [])
    
    # Entity name is "Floridi" but type is wrong (e.g., "Paper" instead of "Author")
    # Should still match via fallback
    matched, conf, method = matcher._fuzzy_fallback('Floridi', refs)
    
    assert matched is not None
    assert method.startswith('fuzzy_')
    assert conf >= 0.70


def test_tier2_fuzzy_matches_any_field():
    """Tier 2: Fuzzy matching checks all fields."""
    refs = [
        {'author': 'Smith, J.', 'title': 'AI Governance', 'year': 2020, 'journal': 'Nature Communications'}
    ]
    
    matcher = CitationMatcher({}, {}, [])
    
    # Should match "Nature" in journal field
    matched, conf, method = matcher._fuzzy_fallback('Nature Communications', refs)
    assert matched is not None
    assert method == 'fuzzy_journal'
    
    # Should match "Governance" in title field
    matched, conf, method = matcher._fuzzy_fallback('AI Governance', refs)
    assert matched is not None
    assert method == 'fuzzy_title'


def test_tier3_partial_year_surname():
    """Tier 3: Partial matching with year + surname."""
    refs = [
        {'author': 'Floridi, Luciano', 'title': 'Some Paper', 'year': 2018, 'journal': 'Nature'}
    ]
    
    matcher = CitationMatcher({}, {}, [])
    
    # Entity has year + surname but no full match
    matched, conf, method = matcher._partial_match('Floridi (2018)', refs)
    
    assert matched is not None
    assert method == 'partial_year_surname'
    assert conf >= 0.65


def test_tier3_partial_surname_only():
    """Tier 3: Partial matching with surname only."""
    refs = [
        {'author': 'Floridi, Luciano', 'title': 'Some Paper', 'year': 2018, 'journal': 'Nature'}
    ]
    
    matcher = CitationMatcher({}, {}, [])
    
    # Entity has surname but no year
    matched, conf, method = matcher._partial_match('Floridi', refs)
    
    assert matched is not None
    assert method == 'partial_surname_start'
    assert conf >= 0.50


def test_tiered_integration():
    """Integration: Tiered matching tries all levels."""
    refs = [
        {'author': 'Floridi, Luciano', 'title': 'Digital Ethics', 'year': 2018, 'journal': 'Philosophy & Technology'}
    ]
    
    references_lookup = {'L1_001': refs}
    chunk_to_l1 = {'chunk_001': 'L1_001'}
    l1_pubs = [{'scopus_id': 'L1_001', 'title': 'Source Paper', 'year': 2020}]
    
    matcher = CitationMatcher(references_lookup, chunk_to_l1, l1_pubs)
    
    # Test with Author entity (should use Tier 1)
    entity_data = {
        'name': 'Floridi',
        'type': 'Author',
        'chunk_ids': ['chunk_001']
    }
    
    result = matcher.match_entity_to_reference('ent_001', entity_data)
    
    assert result is not None
    matched_ref, match_result, conf, method = result
    assert method == 'type_author'
    assert conf >= 0.85


def test_tiered_fallback_when_tier1_fails():
    """Integration: Falls back to Tier 2 when Tier 1 fails."""
    refs = [
        {'author': 'Floridi, Luciano', 'title': 'Digital Ethics', 'year': 2018, 'journal': 'Philosophy & Technology'}
    ]
    
    references_lookup = {'L1_001': refs}
    chunk_to_l1 = {'chunk_001': 'L1_001'}
    l1_pubs = [{'scopus_id': 'L1_001', 'title': 'Source Paper', 'year': 2020}]
    
    matcher = CitationMatcher(references_lookup, chunk_to_l1, l1_pubs)
    
    # Entity type is "Journal" but name is actually an author
    # Tier 1 should fail, Tier 2 should succeed
    entity_data = {
        'name': 'Floridi',
        'type': 'Journal',  # Wrong type!
        'chunk_ids': ['chunk_001']
    }
    
    result = matcher.match_entity_to_reference('ent_001', entity_data)
    
    assert result is not None
    matched_ref, match_result, conf, method = result
    assert method.startswith('fuzzy_') or method.startswith('partial_')


# ==============================================================================
# RUN TESTS
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])