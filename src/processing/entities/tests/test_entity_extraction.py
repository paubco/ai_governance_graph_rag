# -*- coding: utf-8 -*-
"""
Test suite for Phase 1B entity extraction.

Tests dual-pass entity extraction with Type x Domain schema validation.
Uses pytest with fixtures for extractor and sample chunks.

Run:
    pytest src/processing/entities/tests/test_entity_extraction.py -v
    pytest src/processing/entities/tests/test_entity_extraction.py -v -k "test_semantic"
    pytest src/processing/entities/tests/test_entity_extraction.py -v -k "TestLiveAPI"

References:
    - CONTRIBUTING.md Section 2.3 (Testing standards)
    - Phase 1B spec
"""

import json
import pytest
from pathlib import Path
from typing import List, Dict
from unittest.mock import Mock, patch, MagicMock

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

from src.utils.dataclasses import PreEntity
from config.extraction_config import (
    SEMANTIC_ENTITY_TYPES,
    ACADEMIC_ENTITY_TYPES,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def semantic_types() -> set:
    """Valid semantic entity types."""
    return set(SEMANTIC_ENTITY_TYPES)


@pytest.fixture
def academic_types() -> set:
    """Valid academic entity types."""
    return set(ACADEMIC_ENTITY_TYPES)


@pytest.fixture
def sample_regulation_chunk() -> Dict:
    """Sample regulation chunk for testing."""
    return {
        "chunk_id": "reg_EU_chunk_0042",
        "text": """The EU AI Act establishes a risk-based framework for artificial intelligence 
        regulation. High-risk AI systems must undergo conformity assessment procedures before 
        being placed on the European market. The European Commission will establish a European 
        Artificial Intelligence Board to ensure consistent application across Member States.""",
        "doc_type": "regulation",
    }


@pytest.fixture
def sample_paper_chunk() -> Dict:
    """Sample paper chunk for testing (includes citations)."""
    return {
        "chunk_id": "paper_042_chunk_0015",
        "text": """Recent work by Floridi (2018) and Jobin et al. (2019) has examined the 
        proliferation of AI ethics guidelines across jurisdictions. The authors argue that 
        algorithmic transparency is essential for accountability. This study extends their 
        framework to regulatory compliance contexts, examining how organizations implement 
        fairness requirements in machine learning systems.""",
        "doc_type": "paper",
    }


@pytest.fixture
def mock_semantic_response() -> str:
    """Mock LLM response for semantic extraction (v1.2 - domain-fused types)."""
    return json.dumps({"entities": [
        {
            "name": "EU AI Act",
            "type": "Regulation",
            "description": "European Union regulation establishing rules for AI systems"
        },
        {
            "name": "conformity assessment",
            "type": "RegulatoryProcess",
            "description": "Procedure to verify AI system compliance with requirements"
        },
        {
            "name": "European Commission",
            "type": "Organization",
            "description": "Executive branch of the European Union"
        },
        {
            "name": "data governance",
            "type": "RegulatoryConcept",
            "description": "Framework for managing data in compliance with regulations"
        },
        {
            "name": "neural networks",
            "type": "Technology",
            "description": "AI architecture for pattern recognition"
        },
    ]})


@pytest.fixture
def mock_academic_response() -> str:
    """Mock LLM response for academic extraction."""
    return json.dumps({"entities": [
        {
            "name": "Floridi (2018)",
            "type": "Citation",
            "description": "Reference to Floridi's 2018 work on digital ethics"
        },
        {
            "name": "Jobin et al. (2019)",
            "type": "Citation",
            "description": "Reference to Jobin et al.'s survey of AI ethics guidelines"
        },
        {
            "name": "the authors",
            "type": "Self-Reference",
            "description": "Self-reference to the paper's authors"
        },
    ]})


# ============================================================================
# UNIT TESTS - TYPE VALIDATION
# ============================================================================

class TestTypeValidation:
    """Tests for type validation (v1.2 - domain-fused types)."""
    
    def test_semantic_types_count(self, semantic_types):
        """Verify we have exactly 11 semantic types."""
        assert len(semantic_types) == 11
    
    def test_semantic_types_expected(self, semantic_types):
        """Verify expected semantic types are present."""
        expected = {
            "RegulatoryConcept", "TechnicalConcept", "PoliticalConcept",
            "RegulatoryProcess", "TechnicalProcess", "PoliticalProcess",
            "Regulation", "Technology", "Organization", "Location", "Principle"
        }
        assert semantic_types == expected
    
    def test_academic_types_count(self, academic_types):
        """Verify we have exactly 4 academic types."""
        assert len(academic_types) == 4
    
    def test_academic_types_expected(self, academic_types):
        """Verify expected academic types are present."""
        expected = {"Citation", "Author", "Journal", "Self-Reference"}
        assert academic_types == expected


# ============================================================================
# UNIT TESTS - PREENTITY DATACLASS
# ============================================================================

class TestPreEntityDataclass:
    """Tests for PreEntity dataclass (v1.2 - no domain field)."""
    
    def test_semantic_preentity_creation(self):
        """Test creating semantic PreEntity."""
        entity = PreEntity(
            name="EU AI Act",
            type="Regulation",
            description="European AI regulation",
            chunk_id="reg_EU_chunk_001",
            embedding_text="EU AI Act(Regulation)",
        )
        
        assert entity.name == "EU AI Act"
        assert entity.type == "Regulation"
        assert entity.embedding_text == "EU AI Act(Regulation)"
    
    def test_academic_preentity_creation(self):
        """Test creating academic PreEntity."""
        entity = PreEntity(
            name="Floridi (2018)",
            type="Citation",
            description="Reference to Floridi's work",
            chunk_id="paper_042_chunk_015",
            embedding_text="Floridi (2018)(Citation)",
        )
        
        assert entity.name == "Floridi (2018)"
        assert entity.type == "Citation"
        assert entity.embedding_text == "Floridi (2018)(Citation)"
    
    def test_compute_embedding_text(self):
        """Test embedding text computation."""
        entity = PreEntity(
            name="conformity assessment",
            type="RegulatoryProcess",
            description="Compliance procedure",
            chunk_id="reg_EU_chunk_001",
        )
        
        result = entity.compute_embedding_text()
        assert result == "conformity assessment(RegulatoryProcess)"


# ============================================================================
# UNIT TESTS - JSON PARSING
# ============================================================================

class TestJSONParsing:
    """Tests for JSON response parsing."""
    
    def test_parse_clean_json_object(self):
        """Test parsing clean JSON object with entities array."""
        from src.processing.entities.pre_entity_extractor import DualPassEntityExtractor
        
        with patch.object(DualPassEntityExtractor, '__init__', lambda x, **kw: None):
            extractor = DualPassEntityExtractor()
            extractor.semantic_types = set(SEMANTIC_ENTITY_TYPES)
            extractor.academic_types = set(ACADEMIC_ENTITY_TYPES)
            
            content = '{"entities": [{"name": "test", "type": "RegulatoryConcept", "description": "desc"}]}'
            result = extractor._parse_json_response(content, "chunk_001", "semantic")
            
            assert len(result) == 1
            assert result[0]["name"] == "test"
    
    def test_parse_markdown_wrapped_json(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        from src.processing.entities.pre_entity_extractor import DualPassEntityExtractor
        
        with patch.object(DualPassEntityExtractor, '__init__', lambda x, **kw: None):
            extractor = DualPassEntityExtractor()
            extractor.semantic_types = set(SEMANTIC_ENTITY_TYPES)
            
            extractor.academic_types = set(ACADEMIC_ENTITY_TYPES)
            
            content = '```json\n{"entities": [{"name": "test", "type": "RegulatoryConcept", "description": "desc"}]}\n```'
            result = extractor._parse_json_response(content, "chunk_001", "semantic")
            
            assert len(result) == 1
            assert result[0]["name"] == "test"
    
    def test_parse_empty_array(self):
        """Test parsing empty JSON array."""
        from src.processing.entities.pre_entity_extractor import DualPassEntityExtractor
        
        with patch.object(DualPassEntityExtractor, '__init__', lambda x, **kw: None):
            extractor = DualPassEntityExtractor()
            extractor.semantic_types = set(SEMANTIC_ENTITY_TYPES)
            
            extractor.academic_types = set(ACADEMIC_ENTITY_TYPES)
            
            # Test empty array
            result = extractor._parse_json_response("[]", "chunk_001", "semantic")
            assert result == []
            
            # Test empty entities in object format
            result = extractor._parse_json_response('{"entities": []}', "chunk_001", "semantic")
            assert result == []
    
    def test_parse_malformed_json(self):
        """Test handling malformed JSON gracefully."""
        from src.processing.entities.pre_entity_extractor import DualPassEntityExtractor
        
        with patch.object(DualPassEntityExtractor, '__init__', lambda x, **kw: None):
            extractor = DualPassEntityExtractor()
            extractor.semantic_types = set(SEMANTIC_ENTITY_TYPES)
            
            extractor.academic_types = set(ACADEMIC_ENTITY_TYPES)
            
            result = extractor._parse_json_response("not valid json", "chunk_001", "semantic")
            assert result == []


# ============================================================================
# UNIT TESTS - ENTITY VALIDATION
# ============================================================================

class TestEntityValidation:
    """Tests for entity validation during creation."""
    
    def test_reject_invalid_type(self):
        """Test that invalid types are rejected."""
        from src.processing.entities.pre_entity_extractor import DualPassEntityExtractor
        
        with patch.object(DualPassEntityExtractor, '__init__', lambda x, **kw: None):
            extractor = DualPassEntityExtractor()
            extractor.semantic_types = set(SEMANTIC_ENTITY_TYPES)
            
            raw = {"name": "test", "type": "InvalidType", "description": "desc"}
            result = extractor._create_semantic_entity(raw, "chunk_001")
            
            assert result is None
    
    def test_accept_valid_semantic_entity(self):
        """Test that valid semantic entities are accepted."""
        from src.processing.entities.pre_entity_extractor import DualPassEntityExtractor
        
        with patch.object(DualPassEntityExtractor, '__init__', lambda x, **kw: None):
            extractor = DualPassEntityExtractor()
            extractor.semantic_types = set(SEMANTIC_ENTITY_TYPES)
            
            raw = {"name": "EU AI Act", "type": "Regulation", "description": "EU law"}
            result = extractor._create_semantic_entity(raw, "chunk_001")
            
            assert result is not None
            assert result.name == "EU AI Act"
            assert result.type == "Regulation"
            assert result.embedding_text == "EU AI Act(Regulation)"
    
    def test_accept_valid_academic_entity(self):
        """Test that valid academic entities are accepted."""
        from src.processing.entities.pre_entity_extractor import DualPassEntityExtractor
        
        with patch.object(DualPassEntityExtractor, '__init__', lambda x, **kw: None):
            extractor = DualPassEntityExtractor()
            extractor.academic_types = set(ACADEMIC_ENTITY_TYPES)
            
            raw = {"name": "Floridi (2018)", "type": "Citation", "description": "Reference"}
            result = extractor._create_academic_entity(raw, "chunk_001")
            
            assert result is not None
            assert result.name == "Floridi (2018)"
            assert result.type == "Citation"
            assert result.embedding_text == "Floridi (2018)(Citation)"


# ============================================================================
# UNIT TESTS - DOC TYPE INFERENCE
# ============================================================================

class TestDocTypeInference:
    """Tests for document type inference from chunk ID."""
    
    def test_infer_regulation(self):
        """Test inferring regulation doc type."""
        from src.processing.entities.pre_entity_extractor import DualPassEntityExtractor
        
        with patch.object(DualPassEntityExtractor, '__init__', lambda x, **kw: None):
            extractor = DualPassEntityExtractor()
            
            assert extractor._infer_doc_type("reg_EU_chunk_001") == "regulation"
            assert extractor._infer_doc_type("reg_US_CA_chunk_042") == "regulation"
    
    def test_infer_paper(self):
        """Test inferring paper doc type."""
        from src.processing.entities.pre_entity_extractor import DualPassEntityExtractor
        
        with patch.object(DualPassEntityExtractor, '__init__', lambda x, **kw: None):
            extractor = DualPassEntityExtractor()
            
            assert extractor._infer_doc_type("paper_042_chunk_015") == "paper"
            assert extractor._infer_doc_type("paper_001_chunk_001") == "paper"
    
    def test_infer_unknown_defaults_to_regulation(self):
        """Test that unknown prefixes default to regulation."""
        from src.processing.entities.pre_entity_extractor import DualPassEntityExtractor
        
        with patch.object(DualPassEntityExtractor, '__init__', lambda x, **kw: None):
            extractor = DualPassEntityExtractor()
            
            assert extractor._infer_doc_type("unknown_chunk_001") == "regulation"


# ============================================================================
# INTEGRATION TESTS (with mocked API)
# ============================================================================

class TestIntegrationMocked:
    """Integration tests with mocked API responses."""
    
    def test_extract_regulation_chunk(
        self,
        sample_regulation_chunk,
        mock_semantic_response,
    ):
        """Test full extraction flow for regulation chunk."""
        from src.processing.entities.pre_entity_extractor import DualPassEntityExtractor
        
        with patch('src.processing.entities.pre_entity_extractor.Together') as mock_together:
            # Setup mock
            mock_client = MagicMock()
            mock_together.return_value = mock_client
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content=mock_semantic_response))]
            )
            
            extractor = DualPassEntityExtractor(api_key="test_key")
            entities = extractor.extract_entities(
                sample_regulation_chunk["text"],
                sample_regulation_chunk["chunk_id"],
                sample_regulation_chunk["doc_type"],
            )
            
            # Should only have semantic entities (no academic pass for regulations)
            assert len(entities) == 5
            
            # Verify specific entities
            names = {e.name for e in entities}
            assert "EU AI Act" in names
            assert "European Commission" in names
    
    def test_extract_paper_chunk_dual_pass(
        self,
        sample_paper_chunk,
        mock_semantic_response,
        mock_academic_response,
    ):
        """Test full extraction flow for paper chunk (both passes)."""
        from src.processing.entities.pre_entity_extractor import DualPassEntityExtractor
        
        with patch('src.processing.entities.pre_entity_extractor.Together') as mock_together:
            # Setup mock to return different responses for each call
            mock_client = MagicMock()
            mock_together.return_value = mock_client
            mock_client.chat.completions.create.side_effect = [
                MagicMock(choices=[MagicMock(message=MagicMock(content=mock_semantic_response))]),
                MagicMock(choices=[MagicMock(message=MagicMock(content=mock_academic_response))]),
            ]
            
            extractor = DualPassEntityExtractor(api_key="test_key")
            entities = extractor.extract_entities(
                sample_paper_chunk["text"],
                sample_paper_chunk["chunk_id"],
                sample_paper_chunk["doc_type"],
            )
            
            # Should have both semantic and academic entities
            # Use type to distinguish (academic types: Citation, Author, Journal, Self-Reference)
            semantic_entities = [e for e in entities if e.type not in ACADEMIC_ENTITY_TYPES]
            academic_entities = [e for e in entities if e.type in ACADEMIC_ENTITY_TYPES]
            
            assert len(semantic_entities) == 5
            assert len(academic_entities) == 3
            
            # Verify academic entities
            academic_names = {e.name for e in academic_entities}
            assert "Floridi (2018)" in academic_names
            assert "Jobin et al. (2019)" in academic_names


# ============================================================================
# LIVE API TESTS (optional, requires --live-api flag)
# ============================================================================

# ============================================================================
# LIVE API TESTS
# ============================================================================

class TestLiveAPI:
    """Live API tests - requires TOGETHER_API_KEY in .env."""
    
    def test_live_semantic_extraction(self, sample_regulation_chunk):
        """Test semantic extraction against live API."""
        from src.processing.entities.pre_entity_extractor import DualPassEntityExtractor
        
        extractor = DualPassEntityExtractor()
        entities = extractor.extract_entities(
            sample_regulation_chunk["text"],
            sample_regulation_chunk["chunk_id"],
            "regulation",
        )
        
        # Should extract entities
        assert len(entities) > 0
        
        # All should have valid types and embedding_text
        all_types = set(SEMANTIC_ENTITY_TYPES) | set(ACADEMIC_ENTITY_TYPES)
        for e in entities:
            assert e.type in all_types
            assert e.embedding_text is not None
    
    def test_live_academic_extraction(self, sample_paper_chunk):
        """Test academic extraction against live API."""
        from src.processing.entities.pre_entity_extractor import DualPassEntityExtractor
        
        extractor = DualPassEntityExtractor()
        entities = extractor.extract_entities(
            sample_paper_chunk["text"],
            sample_paper_chunk["chunk_id"],
            "paper",
        )
        
        # Should have both semantic and academic entities
        semantic = [e for e in entities if e.type not in ACADEMIC_ENTITY_TYPES]
        academic = [e for e in entities if e.type in ACADEMIC_ENTITY_TYPES]
        
        assert len(semantic) > 0
        assert len(academic) > 0
        
        # Verify academic entities have correct format
        for e in academic:
            assert e.type in ACADEMIC_ENTITY_TYPES
            assert e.embedding_text == f"{e.name}({e.type})"