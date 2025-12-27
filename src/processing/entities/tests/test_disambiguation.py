# -*- coding: utf-8 -*-
"""
Tests

Tests cover:
    - Pre-entity filtering (blacklist + provenance)
    - Semantic disambiguation (dedup, FAISS, thresholds)
    - Metadata disambiguation (Document/DocumentSection relations)
    - Alias tracking

"""
import pytest
import json
from typing import Dict, List
from unittest.mock import patch, MagicMock

# Import modules under test
from src.processing.entities.pre_entity_filter import (
    is_garbage, check_provenance, PreEntityFilter
)
from src.processing.entities.semantic_disambiguator import (
    ExactDeduplicator, TieredThresholdFilter, 
    route_by_type, apply_merges, get_entity_key, METADATA_TYPES
)
from src.processing.entities.metadata_disambiguator import (
    MetadataDisambiguator, build_chunk_entity_map
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_entities():
    """Sample entities for testing."""
    return [
        {'name': 'EU AI Act', 'type': 'Regulation', 'chunk_id': 'chunk_001', 
         'description': 'European regulation on AI'},
        {'name': 'European AI Act', 'type': 'Regulation', 'chunk_id': 'chunk_002',
         'description': 'The EU AI Act'},
        {'name': 'AI Act', 'type': 'Legislation', 'chunk_id': 'chunk_003',
         'description': 'Regulation on AI systems'},
        {'name': 'GDPR', 'type': 'Regulation', 'chunk_id': 'chunk_001',
         'description': 'Data protection regulation'},
        {'name': 'machine learning', 'type': 'Technology', 'chunk_id': 'chunk_002',
         'description': 'ML algorithms'},
        {'name': 'Machine Learning', 'type': 'TechnicalConcept', 'chunk_id': 'chunk_003',
         'description': 'Statistical learning'},
    ]


@pytest.fixture
def sample_chunks():
    """Sample chunks for provenance testing."""
    return {
        'chunk_001': 'The EU AI Act establishes rules for AI systems. GDPR applies.',
        'chunk_002': 'The European AI Act and machine learning regulation.',
        'chunk_003': 'AI Act covers Machine Learning systems.',
    }


@pytest.fixture
def metadata_entities():
    """Sample metadata entities for testing (v2.0)."""
    return [
        {'name': 'Article 5 of the EU AI Act', 'type': 'DocumentSection', 'chunk_id': 'chunk_001',
         'description': 'Prohibited AI practices'},
        {'name': 'EU AI Act', 'type': 'Document', 'chunk_id': 'chunk_001',
         'description': 'European AI regulation'},
        {'name': 'Floridi (2018)', 'type': 'Citation', 'chunk_id': 'chunk_002',
         'description': 'Author citation'},
        {'name': 'Nature', 'type': 'Journal', 'chunk_id': 'chunk_003',
         'description': 'Scientific journal'},
        {'name': 'Section 3.2', 'type': 'DocumentSection', 'chunk_id': 'chunk_002',
         'description': 'Document section'},
    ]


# =============================================================================
# GARBAGE FILTER TESTS
# =============================================================================

class TestGarbagePatterns:
    """Test garbage pattern detection."""
    
    def test_ellipsis_garbage(self):
        """Test that ellipsis patterns are garbage."""
        assert is_garbage('...') is True
        assert is_garbage('..') is True
        assert is_garbage('…') is True  # Unicode ellipsis
    
    def test_na_garbage(self):
        """Test that N/A patterns are garbage."""
        assert is_garbage('N/A') is True
        assert is_garbage('NA') is True
        assert is_garbage('n/a') is True
    
    def test_percentage_garbage(self):
        """Test that percentages are garbage."""
        assert is_garbage('80%') is True
        assert is_garbage('41%') is True
        assert is_garbage('100%') is True
    
    def test_single_digit_garbage(self):
        """Test that single digits are garbage (outside Citation)."""
        assert is_garbage('4', 'Technology') is True
        assert is_garbage('5', 'Organization') is True
        # But valid in Citation
        assert is_garbage('18', 'Citation') is False
    
    def test_question_words_garbage(self):
        """Test that question words are garbage."""
        assert is_garbage('who') is True
        assert is_garbage('why') is True
        assert is_garbage('how') is True
        assert is_garbage('what') is True
    
    def test_rq_patterns_garbage(self):
        """Test that research question patterns are garbage."""
        assert is_garbage('RQ1') is True
        assert is_garbage('RQ2') is True
        assert is_garbage('RQ') is True
    
    def test_single_letter_garbage(self):
        """Test that single letters are garbage."""
        assert is_garbage('R') is True
        assert is_garbage('Ra') is True
        assert is_garbage('k') is True
    
    def test_valid_acronyms_kept(self):
        """Test that valid acronyms are NOT garbage."""
        assert is_garbage('AI') is False
        assert is_garbage('ML') is False
        assert is_garbage('EU') is False
        assert is_garbage('UK') is False
        assert is_garbage('US') is False
        assert is_garbage('UAE') is False
        assert is_garbage('GDPR') is False
        assert is_garbage('FDA') is False
        assert is_garbage('G20') is False
        assert is_garbage('FP7') is False
    
    def test_type_aware_numeric(self):
        """Test type-aware numeric filtering."""
        # Numbers are garbage outside Citation
        assert is_garbage('18', 'Author') is True
        assert is_garbage('46', 'Affiliation') is True
        assert is_garbage('[3]', 'Journal') is True
        
        # Numbers are valid in Citation
        assert is_garbage('18', 'Citation') is False
        assert is_garbage('[3]', 'Citation') is False


class TestProvenanceCheck:
    """Test provenance verification."""
    
    def test_exact_match(self):
        """Test exact substring matching."""
        is_valid, score = check_provenance('EU AI Act', 'The EU AI Act regulates AI.')
        assert is_valid is True
        assert score == 1.0
    
    def test_case_insensitive_match(self):
        """Test case-insensitive matching."""
        is_valid, score = check_provenance('eu ai act', 'The EU AI Act regulates AI.')
        assert is_valid is True
    
    def test_no_match(self):
        """Test when entity doesn't appear in chunk."""
        is_valid, score = check_provenance('Nature', 'The EU AI Act regulates AI.')
        assert is_valid is False
        assert score < 0.8
    
    def test_empty_chunk(self):
        """Test with empty chunk text (assumes valid)."""
        is_valid, score = check_provenance('EU AI Act', '')
        assert is_valid is True  # Can't verify, assume valid


class TestPreEntityFilter:
    """Test full garbage filter."""
    
    def test_filter_basic(self, sample_entities, sample_chunks):
        """Test basic filtering."""
        filter = PreEntityFilter(chunks=sample_chunks)
        clean, stats = filter.filter(sample_entities)
        
        assert len(clean) <= len(sample_entities)
        assert stats['input_count'] == len(sample_entities)
        assert stats['output_count'] == len(clean)
    
    def test_filter_removes_garbage(self, sample_chunks):
        """Test that garbage is removed."""
        entities = [
            {'name': '...', 'type': 'Citation', 'chunk_id': 'chunk_001'},
            {'name': 'N/A', 'type': 'Author', 'chunk_id': 'chunk_001'},
            {'name': 'EU AI Act', 'type': 'Regulation', 'chunk_id': 'chunk_001'},
        ]
        
        filter = PreEntityFilter(chunks=sample_chunks)
        clean, stats = filter.filter(entities)
        
        assert len(clean) == 1
        assert clean[0]['name'] == 'EU AI Act'
        assert stats['blacklist_removed'] == 2
    
    def test_provenance_check(self, sample_chunks):
        """Test that provenance check catches hallucinations."""
        entities = [
            {'name': 'EU AI Act', 'type': 'Regulation', 'chunk_id': 'chunk_001'},  # Present
            {'name': 'Nature', 'type': 'Journal', 'chunk_id': 'chunk_001'},  # Not present
        ]
        
        filter = PreEntityFilter(chunks=sample_chunks, verify_provenance=True)
        clean, stats = filter.filter(entities)
        
        assert len(clean) == 1
        assert clean[0]['name'] == 'EU AI Act'
        assert stats['provenance_failed'] == 1
        assert len(filter.hallucinations) == 1


# =============================================================================
# ENTITY DISAMBIGUATOR TESTS
# =============================================================================

class TestExactDeduplicator:
    """Test exact deduplication with alias tracking."""
    
    def test_basic_dedup(self, sample_entities):
        """Test basic deduplication."""
        dedup = ExactDeduplicator()
        canonical, aliases = dedup.deduplicate(sample_entities)
        
        # Should have fewer entities after dedup
        assert len(canonical) <= len(sample_entities)
        assert dedup.stats['input_count'] == len(sample_entities)
    
    def test_alias_tracking(self):
        """Test that aliases are tracked during merge."""
        entities = [
            {'name': 'EU AI Act', 'type': 'Regulation', 'chunk_id': 'c1'},
            {'name': 'eu ai act', 'type': 'Legislation', 'chunk_id': 'c2'},
            {'name': 'EU AI ACT', 'type': 'Regulation', 'chunk_id': 'c3'},
        ]
        
        dedup = ExactDeduplicator()
        canonical, aliases = dedup.deduplicate(entities)
        
        assert len(canonical) == 1
        # Should have aliases for the merged variants
        assert len(aliases) > 0 or len(canonical[0].get('aliases', [])) > 0
    
    def test_type_voting(self):
        """Test that most frequent type wins."""
        entities = [
            {'name': 'AI Act', 'type': 'Regulation', 'chunk_id': 'c1'},
            {'name': 'AI Act', 'type': 'Regulation', 'chunk_id': 'c2'},
            {'name': 'AI Act', 'type': 'Legislation', 'chunk_id': 'c3'},
        ]
        
        dedup = ExactDeduplicator()
        canonical, _ = dedup.deduplicate(entities)
        
        assert len(canonical) == 1
        assert canonical[0]['type'] == 'Regulation'  # Most frequent
    
    def test_chunk_ids_merged(self):
        """Test that chunk_ids are combined."""
        entities = [
            {'name': 'GDPR', 'type': 'Regulation', 'chunk_id': 'c1'},
            {'name': 'GDPR', 'type': 'Regulation', 'chunk_id': 'c2'},
        ]
        
        dedup = ExactDeduplicator()
        canonical, _ = dedup.deduplicate(entities)
        
        assert len(canonical) == 1
        assert 'c1' in canonical[0]['chunk_ids']
        assert 'c2' in canonical[0]['chunk_ids']


class TestTypeRouting:
    """Test entity type routing."""
    
    def test_route_by_type(self):
        """Test routing entities to semantic/metadata paths."""
        entities = [
            {'name': 'EU AI Act', 'type': 'Regulation'},
            {'name': 'Floridi (2018)', 'type': 'Citation'},
            {'name': 'AI', 'type': 'Technology'},
            {'name': 'Nature', 'type': 'Journal'},
            {'name': 'MIT', 'type': 'Affiliation'},
            {'name': 'EU AI Act', 'type': 'Document'},
            {'name': 'Article 5', 'type': 'DocumentSection'},
        ]
        
        semantic, metadata = route_by_type(entities)
        
        assert len(semantic) == 2  # Regulation, Technology
        assert len(metadata) == 5  # Citation, Journal, Affiliation, Document, DocumentSection
        
        # Check correct routing
        semantic_types = {e['type'] for e in semantic}
        metadata_types = {e['type'] for e in metadata}
        
        assert 'Regulation' in semantic_types
        assert 'Technology' in semantic_types
        assert 'Citation' in metadata_types
        assert 'Journal' in metadata_types
        assert 'Document' in metadata_types
        assert 'DocumentSection' in metadata_types


class TestTieredThresholdFilter:
    """Test tiered threshold filtering."""
    
    def test_auto_merge(self):
        """Test auto-merge for high similarity pairs."""
        pairs = [
            {'entity1_key': ('A', 'T'), 'entity2_key': ('B', 'T'), 'similarity': 0.98},
        ]
        
        filter = TieredThresholdFilter(auto_merge_threshold=0.95)
        result = filter.filter_pairs(pairs)
        
        assert len(result['merged']) == 1
        assert len(result['uncertain']) == 0
    
    def test_auto_reject(self):
        """Test auto-reject for low similarity pairs."""
        pairs = [
            {'entity1_key': ('A', 'T'), 'entity2_key': ('B', 'T'), 'similarity': 0.80},
        ]
        
        filter = TieredThresholdFilter(auto_reject_threshold=0.85)
        result = filter.filter_pairs(pairs)
        
        assert len(result['rejected']) == 1
        assert len(result['uncertain']) == 0
    
    def test_uncertain(self):
        """Test uncertain pairs sent to LLM."""
        pairs = [
            {'entity1_key': ('A', 'T'), 'entity2_key': ('B', 'T'), 'similarity': 0.90},
        ]
        
        filter = TieredThresholdFilter(
            auto_merge_threshold=0.95,
            auto_reject_threshold=0.85
        )
        result = filter.filter_pairs(pairs)
        
        assert len(result['uncertain']) == 1


# =============================================================================
# METADATA DISAMBIGUATOR TESTS (v2.0)
# =============================================================================

class TestMetadataDisambiguator:
    """Test metadata entity disambiguation."""
    
    def test_document_section_part_of_document(self):
        """Test PART_OF for DocumentSection → Document."""
        semantic_entities = [
            {'entity_id': 'ent_001', 'name': 'EU AI Act', 'type': 'Regulation', 
             'chunk_ids': ['chunk_001']},
        ]
        
        disambiguator = MetadataDisambiguator(semantic_entities=semantic_entities)
        
        metadata = [
            {'name': 'Article 5 of the EU AI Act', 'type': 'DocumentSection', 'chunk_id': 'chunk_001'},
            {'name': 'EU AI Act', 'type': 'Document', 'chunk_id': 'chunk_001'},
        ]
        
        processed, part_of, same_as = disambiguator.process(metadata)
        
        assert len(processed) == 2
        assert len(part_of) == 1
        assert part_of[0]['predicate'] == 'PART_OF'
        assert part_of[0]['subject_type'] == 'DocumentSection'
        assert part_of[0]['object_type'] == 'Document'
    
    def test_document_same_as_regulation(self):
        """Test SAME_AS for Document ↔ Regulation."""
        semantic_entities = [
            {'entity_id': 'ent_001', 'name': 'EU AI Act', 'type': 'Regulation', 
             'chunk_ids': ['chunk_001']},
            {'entity_id': 'ent_002', 'name': 'GDPR', 'type': 'Regulation',
             'chunk_ids': ['chunk_002']},
        ]
        
        disambiguator = MetadataDisambiguator(semantic_entities=semantic_entities)
        
        metadata = [
            {'name': 'EU AI Act', 'type': 'Document', 'chunk_id': 'chunk_001'},
            {'name': 'GDPR', 'type': 'Document', 'chunk_id': 'chunk_002'},
        ]
        
        processed, part_of, same_as = disambiguator.process(metadata)
        
        assert len(processed) == 2
        assert len(same_as) == 2
        assert all(r['predicate'] == 'SAME_AS' for r in same_as)
        assert all(r['subject_type'] == 'Document' for r in same_as)
        assert all(r['object_type'] == 'Regulation' for r in same_as)
    
    def test_citation_no_relations(self):
        """Test that Citation entities don't get PART_OF or SAME_AS."""
        semantic_entities = [
            {'entity_id': 'ent_001', 'name': 'AI Ethics', 'type': 'TechnicalConcept',
             'chunk_ids': ['chunk_001']},
        ]
        
        disambiguator = MetadataDisambiguator(semantic_entities=semantic_entities)
        
        metadata = [
            {'name': 'Floridi (2018)', 'type': 'Citation', 'chunk_id': 'chunk_001'},
        ]
        
        processed, part_of, same_as = disambiguator.process(metadata)
        
        assert len(processed) == 1
        assert len(part_of) == 0
        assert len(same_as) == 0
    
    def test_entity_id_generation(self):
        """Test that entity IDs are generated."""
        disambiguator = MetadataDisambiguator()
        
        metadata = [
            {'name': 'Nature', 'type': 'Journal', 'chunk_id': 'chunk_001'},
        ]
        
        processed, _, _ = disambiguator.process(metadata)
        
        assert 'entity_id' in processed[0]
        assert processed[0]['entity_id'].startswith('ent_')


class TestBuildChunkEntityMap:
    """Test chunk→entity map building."""
    
    def test_basic_map(self):
        """Test building basic chunk map."""
        entities = [
            {'name': 'A', 'chunk_ids': ['c1', 'c2']},
            {'name': 'B', 'chunk_ids': ['c1']},
            {'name': 'C', 'chunk_ids': ['c3']},
        ]
        
        chunk_map = build_chunk_entity_map(entities)
        
        assert 'c1' in chunk_map
        assert len(chunk_map['c1']) == 2  # A and B
        assert len(chunk_map['c2']) == 1  # A only
        assert len(chunk_map['c3']) == 1  # C only


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_full_pipeline_sample(self, sample_entities, sample_chunks, metadata_entities):
        """Test full pipeline on sample data."""
        all_entities = sample_entities + metadata_entities
        
        # Filter
        pre_filter = PreEntityFilter(chunks=sample_chunks)
        clean, _ = pre_filter.filter(all_entities)
        
        # Route
        semantic, metadata = route_by_type(clean)
        
        # Deduplicate semantic
        dedup = ExactDeduplicator()
        canonical, aliases = dedup.deduplicate(semantic)
        
        # Add entity IDs for SAME_AS matching
        for i, e in enumerate(canonical):
            e['entity_id'] = f'ent_{i:05d}'
        
        # Process metadata (mock ID generator)
        with patch('src.processing.entities.metadata_disambiguator.generate_entity_id') as mock_id:
            mock_id.return_value = 'ent_test123'
            
            disambiguator = MetadataDisambiguator(semantic_entities=canonical)
            metadata_out, part_of, same_as = disambiguator.process(metadata)
        
        # Assertions
        assert len(canonical) > 0
        assert len(metadata_out) > 0
        assert isinstance(aliases, dict)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])