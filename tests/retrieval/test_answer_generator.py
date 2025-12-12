# -*- coding: utf-8 -*-
"""
Module: test_answer_generator.py
Package: tests.retrieval
Purpose: Unit tests for answer generation from retrieval results

Author: Pau Barba i Colomer
Created: 2025-12-09
Modified: 2025-12-12

Tests:
- Prompt formatting with token budgeting
- Mock API calls
- Cost estimation
- Error handling
"""

# Standard library
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
import pytest

# Local
from src.retrieval.config import (
    RetrievalResult,
    GraphSubgraph,
    Relation,
    RankedChunk
)
from src.retrieval.answer_generator import AnswerGenerator, GeneratedAnswer


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_api_key():
    """Provide mock API key."""
    return "sk-test-key-12345"


@pytest.fixture
def sample_retrieval_result():
    """Create sample retrieval result for testing."""
    
    # Subgraph with relations
    subgraph = GraphSubgraph(
        entities=[
            'ent_001',
            'ent_002',
            'ent_003'
        ],
        relations=[
            Relation(
                source_id='ent_001',
                source_name='EU AI Act',
                target_id='ent_002',
                target_name='Risk Classification',
                predicate='defines',
                confidence=0.9,
                chunk_ids=['chunk_001']
            ),
            Relation(
                source_id='ent_002',
                source_name='Risk Classification',
                target_id='ent_003',
                target_name='High-Risk AI System',
                predicate='includes',
                confidence=0.85,
                chunk_ids=['chunk_002']
            )
        ]
    )
    
    # Ranked chunks
    chunks = [
        RankedChunk(
            chunk_id='chunk_001',
            text='The EU AI Act establishes a risk-based regulatory framework...',
            doc_id='EU_AI_Act_2024',
            doc_type='regulation',
            jurisdiction='EU',
            score=0.95,
            retrieval_method='graphrag'
        ),
        RankedChunk(
            chunk_id='chunk_002',
            text='High-risk AI systems include those used in critical infrastructure...',
            doc_id='EU_AI_Act_2024',
            doc_type='regulation',
            jurisdiction='EU',
            score=0.90,
            retrieval_method='naive'
        )
    ]
    
    return RetrievalResult(
        query='What are high-risk AI systems under the EU AI Act?',
        resolved_entities=['EU AI Act', 'Risk Classification', 'High-Risk AI System'],
        subgraph=subgraph,
        chunks=chunks
    )


@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic API response."""
    mock_response = Mock()
    mock_response.content = [Mock(text="High-risk AI systems under the EU AI Act include...")]
    mock_response.usage = Mock(input_tokens=500, output_tokens=150)
    return mock_response


# ============================================================================
# TESTS: INITIALIZATION
# ============================================================================

def test_init_with_api_key(mock_api_key):
    """Test initialization with explicit API key."""
    generator = AnswerGenerator(api_key=mock_api_key)
    assert generator.api_key == mock_api_key
    assert generator.client is not None


def test_init_with_env_var(monkeypatch, mock_api_key):
    """Test initialization with API key from environment."""
    monkeypatch.setenv('ANTHROPIC_API_KEY', mock_api_key)
    generator = AnswerGenerator()
    assert generator.api_key == mock_api_key


def test_init_no_api_key_raises(monkeypatch):
    """Test that missing API key raises error."""
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not found"):
        AnswerGenerator()


# ============================================================================
# TESTS: PROMPT FORMATTING
# ============================================================================

def test_format_graph_structure(mock_api_key, sample_retrieval_result):
    """Test graph structure formatting."""
    generator = AnswerGenerator(api_key=mock_api_key)
    
    graph_str = generator._format_graph_structure(
        sample_retrieval_result.subgraph,
        max_tokens=500
    )
    
    # Should contain relations in bullet format
    assert 'EU AI Act' in graph_str
    assert 'defines' in graph_str
    assert 'Risk Classification' in graph_str


def test_format_entity_context(mock_api_key, sample_retrieval_result):
    """Test entity context formatting."""
    generator = AnswerGenerator(api_key=mock_api_key)
    
    entity_str = generator._format_entity_context(
        sample_retrieval_result.subgraph,
        max_tokens=500
    )
    
    # Should contain entity IDs
    assert 'ent_001' in entity_str or 'EU AI Act' in entity_str


def test_format_sources_with_budget(mock_api_key, sample_retrieval_result):
    """Test source formatting with token budget."""
    generator = AnswerGenerator(api_key=mock_api_key)
    
    sources_str, chunks_included = generator._format_sources_with_budget(
        sample_retrieval_result.chunks,
        max_tokens=1000
    )
    
    # Should include both chunks (small token budget)
    assert len(chunks_included) == 2
    assert '[1]' in sources_str
    assert '[2]' in sources_str
    assert 'EU_AI_Act_2024' in sources_str


def test_format_sources_budget_exceeded(mock_api_key, sample_retrieval_result):
    """Test source formatting when budget is exceeded."""
    generator = AnswerGenerator(api_key=mock_api_key)
    
    # Very small budget - should stop after first chunk
    sources_str, chunks_included = generator._format_sources_with_budget(
        sample_retrieval_result.chunks,
        max_tokens=50  # Extremely low
    )
    
    # Might include 0 or 1 chunk depending on exact token count
    assert len(chunks_included) <= 1


def test_format_prompt_with_budget(mock_api_key, sample_retrieval_result):
    """Test complete prompt formatting with budget."""
    generator = AnswerGenerator(api_key=mock_api_key)
    
    formatted = generator._format_prompt_with_budget(sample_retrieval_result)
    
    # Should have all required fields
    assert 'user_prompt' in formatted
    assert 'chunks_included' in formatted
    assert 'tokens_used' in formatted
    
    # User prompt should contain query
    assert sample_retrieval_result.query in formatted['user_prompt']
    
    # Tokens should be tracked
    assert formatted['tokens_used']['total'] > 0


# ============================================================================
# TESTS: ANSWER GENERATION (MOCKED)
# ============================================================================

@patch('src.retrieval.answer_generator.Anthropic')
def test_generate_answer(mock_anthropic_class, mock_api_key, sample_retrieval_result, mock_anthropic_response):
    """Test answer generation with mocked API."""
    # Setup mock
    mock_client = Mock()
    mock_client.messages.create.return_value = mock_anthropic_response
    mock_anthropic_class.return_value = mock_client
    
    # Generate answer
    generator = AnswerGenerator(api_key=mock_api_key)
    answer = generator.generate(sample_retrieval_result)
    
    # Verify answer structure
    assert isinstance(answer, GeneratedAnswer)
    assert answer.answer == "High-risk AI systems under the EU AI Act include..."
    assert answer.query == sample_retrieval_result.query
    assert answer.input_tokens == 500
    assert answer.output_tokens == 150
    assert answer.cost_usd > 0
    
    # Verify API was called
    mock_client.messages.create.assert_called_once()


# ============================================================================
# TESTS: COST ESTIMATION
# ============================================================================

def test_estimate_cost_for_query(mock_api_key, sample_retrieval_result):
    """Test cost estimation without API call."""
    generator = AnswerGenerator(api_key=mock_api_key)
    
    estimate = generator.estimate_cost_for_query(sample_retrieval_result)
    
    # Should have cost estimate
    assert 'estimated_input_tokens' in estimate
    assert 'estimated_output_tokens' in estimate
    assert 'estimated_cost_usd' in estimate
    assert estimate['estimated_cost_usd'] > 0
    
    # Should track chunks
    assert estimate['chunks_that_fit'] > 0
    assert estimate['total_chunks_available'] == 2


# ============================================================================
# TESTS: EDGE CASES
# ============================================================================

def test_empty_subgraph(mock_api_key):
    """Test handling of empty subgraph."""
    generator = AnswerGenerator(api_key=mock_api_key)
    
    empty_subgraph = GraphSubgraph(entities=[], relations=[])
    
    graph_str = generator._format_graph_structure(empty_subgraph, max_tokens=500)
    assert "No relations" in graph_str
    
    entity_str = generator._format_entity_context(empty_subgraph, max_tokens=500)
    assert "No entities" in entity_str


def test_empty_chunks(mock_api_key):
    """Test handling of empty chunk list."""
    generator = AnswerGenerator(api_key=mock_api_key)
    
    sources_str, chunks = generator._format_sources_with_budget([], max_tokens=1000)
    assert "No sources" in sources_str
    assert len(chunks) == 0


def test_very_long_chunk_truncation(mock_api_key):
    """Test that very long chunks are truncated."""
    generator = AnswerGenerator(api_key=mock_api_key)
    generator.config['truncate_chunk_chars'] = 100
    
    long_chunk = RankedChunk(
        chunk_id='chunk_long',
        text='A' * 500,  # 500 chars
        doc_id='test_doc',
        doc_type='regulation',
        jurisdiction='EU',
        score=0.9,
        retrieval_method='naive'
    )
    
    sources_str, chunks = generator._format_sources_with_budget(
        [long_chunk],
        max_tokens=5000
    )
    
    # Should be truncated with ellipsis
    assert '...' in sources_str
    assert len(chunks) == 1