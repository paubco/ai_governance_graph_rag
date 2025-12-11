# -*- coding: utf-8 -*-
"""
test_retrieval_complete.py - Complete Phase 3 Retrieval Test Suite

Tests Phase 3.3.1 (Query Understanding) + Phase 3.3.2 (Context Retrieval)
with BOTH mock tests and real integration tests.

Run quick tests only:
    pytest tests/retrieval/test_retrieval_complete.py -v -m "not integration"

Run all tests (including slow integration):
    pytest tests/retrieval/test_retrieval_complete.py -v

Run only integration tests:
    pytest tests/retrieval/test_retrieval_complete.py -v -m integration
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import json
import tempfile
from pathlib import Path
import numpy as np
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.retrieval.config import (
    ExtractedEntity,
    ResolvedEntity,
    QueryFilters,
    ParsedQuery,
    Subgraph,
    Relation,
    Chunk,
    RankedChunk,
    RetrievalResult,
)


# ============================================================================
# PYTEST MARKERS
# ============================================================================

pytestmark = pytest.mark.retrieval


# ============================================================================
# MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_embedder():
    """Mock BGE-M3 embedder."""
    embedder = MagicMock()
    embedder.embed_single.return_value = np.random.rand(1024)
    return embedder


@pytest.fixture
def mock_faiss_index():
    """Mock FAISS index."""
    index = MagicMock()
    index.search.return_value = (
        np.array([[0.1, 0.2, 0.3]]),  # distances
        np.array([[0, 1, 2]])          # indices
    )
    index.reconstruct.return_value = np.random.rand(1024)
    return index


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver."""
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = None
    return driver, session


@pytest.fixture
def sample_entity_map():
    """Sample entity ID mapping."""
    return {
        'ent_001': 0,
        'ent_002': 1,
        'ent_042': 2,
        'ent_099': 3
    }


@pytest.fixture
def sample_normalized_entities():
    """Sample normalized entities."""
    return [
        {
            'entity_id': 'ent_001',
            'name': 'EU AI Act',
            'type': 'Regulation',
            'source_ids': ['chunk_042']
        },
        {
            'entity_id': 'ent_002',
            'name': 'GDPR',
            'type': 'Regulation',
            'source_ids': ['chunk_089']
        },
        {
            'entity_id': 'ent_042',
            'name': 'facial recognition',
            'type': 'Technology',
            'source_ids': ['chunk_042', 'chunk_089']
        }
    ]


# ============================================================================
# PHASE 3.3.1 TESTS (Query Understanding)
# ============================================================================

class TestQueryParser:
    """Test QueryParser with mocks."""
    
    @patch('src.retrieval.query_parser.Together')
    def test_parse_extracts_entities(self, mock_together_class, mock_embedder):
        """Test LLM entity extraction."""
        from src.retrieval.query_parser import QueryParser
        
        # Mock Together client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "entities": [
                {"name": "EU AI Act", "type": "Regulation"},
                {"name": "facial recognition", "type": "Technology"}
            ]
        })
        mock_client.chat.completions.create.return_value = mock_response
        mock_together_class.return_value = mock_client
        
        parser = QueryParser(mock_embedder)
        parsed = parser.parse("What does EU AI Act say about facial recognition?")
        
        assert parsed.raw_query == "What does EU AI Act say about facial recognition?"
        assert len(parsed.extracted_entities) == 2
        assert parsed.extracted_entities[0].name == "EU AI Act"
        assert parsed.query_embedding.shape == (1024,)
    
    def test_parse_detects_filters(self, mock_embedder):
        """Test jurisdiction and doc type filter detection."""
        from src.retrieval.query_parser import QueryParser
        
        with patch('src.retrieval.query_parser.Together'):
            parser = QueryParser(mock_embedder)
            
            # Mock LLM response
            with patch.object(parser, '_extract_entities_llm', return_value=[]):
                parsed = parser.parse("What EU regulations mention GDPR?")
                
                assert parsed.filters.jurisdiction_hints == ['EU']
                assert parsed.filters.doc_type_hints == ['regulation']


class TestEntityResolver:
    """Test EntityResolver with mocks."""
    
    def test_exact_match(self, mock_faiss_index, mock_embedder, sample_entity_map, sample_normalized_entities):
        """Test exact entity name matching."""
        from src.retrieval.entity_resolver import EntityResolver
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_entity_map, f)
            entity_ids_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_normalized_entities, f)
            entities_path = f.name
        
        try:
            with patch('src.retrieval.entity_resolver.faiss.read_index', return_value=mock_faiss_index):
                resolver = EntityResolver(
                    faiss_index_path='entities.index',
                    entity_ids_path=entity_ids_path,
                    normalized_entities_path=entities_path,
                    embedding_model=mock_embedder
                )
                
                entities = [ExtractedEntity(name="EU AI Act", type="Regulation")]
                resolved = resolver.resolve(entities)
                
                assert len(resolved) > 0
                assert resolved[0].name == "EU AI Act"
                assert resolved[0].match_type == "exact"
        finally:
            Path(entity_ids_path).unlink()
            Path(entities_path).unlink()
    
    def test_fuzzy_match(self, mock_faiss_index, mock_embedder, sample_entity_map, sample_normalized_entities):
        """Test fuzzy matching via FAISS."""
        from src.retrieval.entity_resolver import EntityResolver
        
        # Mock FAISS to return high similarity
        mock_faiss_index.search.return_value = (
            np.array([[0.15]]),  # distance (1 - 0.15/2 = 0.925 similarity > 0.75)
            np.array([[0]])       # index points to ent_001
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_entity_map, f)
            entity_ids_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_normalized_entities, f)
            entities_path = f.name
        
        try:
            with patch('src.retrieval.entity_resolver.faiss.read_index', return_value=mock_faiss_index):
                resolver = EntityResolver(
                    faiss_index_path='entities.index',
                    entity_ids_path=entity_ids_path,
                    normalized_entities_path=entities_path,
                    embedding_model=mock_embedder,
                    threshold=0.75
                )
                
                entities = [ExtractedEntity(name="European AI Act", type="Regulation")]
                resolved = resolver.resolve(entities)
                
                assert len(resolved) > 0
                assert resolved[0].match_type == "fuzzy"
                assert resolved[0].confidence >= 0.75
        finally:
            Path(entity_ids_path).unlink()
            Path(entities_path).unlink()


# ============================================================================
# PHASE 3.3.2 TESTS (Context Retrieval)
# ============================================================================

class TestGraphExpander:
    """Test GraphExpander with mocks."""
    
    @patch('src.retrieval.graph_expander.GraphDatabase.driver')
    @patch('src.retrieval.graph_expander.faiss.read_index')
    def test_single_entity_no_pcst(self, mock_faiss, mock_driver_class, sample_entity_map):
        """Test single entity returns candidates without PCST."""
        from src.retrieval.graph_expander import GraphExpander
        
        # Setup mocks
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver_class.return_value = mock_driver
        
        # Mock GDS exists check
        mock_result = MagicMock()
        mock_result.single.return_value = {'exists': True}
        mock_session.run.return_value = mock_result
        
        mock_index = MagicMock()
        mock_index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[1, 2]]))
        mock_index.reconstruct.return_value = np.random.rand(1024)
        mock_faiss.return_value = mock_index
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_entity_map, f)
            entity_map_path = f.name
        
        try:
            expander = GraphExpander(
                neo4j_uri='bolt://localhost:7687',
                neo4j_user='neo4j',
                neo4j_password='password',
                entity_index_path='entities.index',
                entity_id_map_path=entity_map_path
            )
            
            entities = [ResolvedEntity(
                entity_id='ent_001',
                name='EU AI Act',
                type='Regulation',
                confidence=0.95,
                match_type='exact'
            )]
            
            subgraph = expander.expand(entities)
            
            # Single entity → no PCST
            assert len(subgraph.entities) > 0
            assert len(subgraph.relations) == 0
        finally:
            Path(entity_map_path).unlink()
    
    @patch('src.retrieval.graph_expander.GraphDatabase.driver')
    @patch('src.retrieval.graph_expander.faiss.read_index')
    def test_multiple_entities_runs_pcst(self, mock_faiss, mock_driver_class, sample_entity_map):
        """Test multiple entities trigger PCST."""
        from src.retrieval.graph_expander import GraphExpander
        
        # Setup driver mock
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver_class.return_value = mock_driver
        
        # Mock GDS projection exists
        gds_result = MagicMock()
        gds_result.single.return_value = {'exists': True}
        
        # Mock PCST result
        pcst_result = MagicMock()
        pcst_result.single.return_value = {'entity_ids': ['ent_001', 'ent_042', 'ent_099']}
        
        # Mock relations query
        rel_result = MagicMock()
        rel_result.__iter__.return_value = iter([{
            'source_id': 'ent_001',
            'source_name': 'EU AI Act',
            'predicate': 'prohibits',
            'target_id': 'ent_099',
            'target_name': 'biometric ID',
            'chunk_ids': ['chunk_042']
        }])
        
        mock_session.run.side_effect = [gds_result, pcst_result, rel_result]
        
        mock_index = MagicMock()
        mock_index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[1, 2]]))
        mock_index.reconstruct.return_value = np.random.rand(1024)
        mock_faiss.return_value = mock_index
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_entity_map, f)
            entity_map_path = f.name
        
        try:
            expander = GraphExpander(
                neo4j_uri='bolt://localhost:7687',
                neo4j_user='neo4j',
                neo4j_password='password',
                entity_index_path='entities.index',
                entity_id_map_path=entity_map_path
            )
            
            entities = [
                ResolvedEntity('ent_001', 'EU AI Act', 'Regulation', 0.95, 'exact'),
                ResolvedEntity('ent_042', 'facial recognition', 'Technology', 0.87, 'fuzzy')
            ]
            
            subgraph = expander.expand(entities)
            
            assert len(subgraph.entities) == 3
            assert len(subgraph.relations) == 1
            assert subgraph.relations[0].predicate == 'prohibits'
        finally:
            Path(entity_map_path).unlink()


class TestResultRanker:
    """Test ResultRanker scoring."""
    
    def test_provenance_bonus_applied(self):
        """Test relation provenance gets highest score."""
        from src.retrieval.result_ranker import ResultRanker
        
        subgraph = Subgraph(
            entities=['ent_001', 'ent_042'],
            relations=[Relation(
                source_id='ent_001',
                source_name='EU AI Act',
                predicate='prohibits',
                target_id='ent_042',
                target_name='facial recognition',
                chunk_ids=['chunk_042']
            )]
        )
        
        chunks_a = [
            Chunk('chunk_042', 'Article 5 prohibits...', 'doc_001', 'regulation', 'EU',
                  metadata={'entities': ['ent_001'], 'is_relation_provenance': True}),
            Chunk('chunk_089', 'Facial recognition systems...', 'doc_001', 'regulation', 'EU',
                  metadata={'entities': ['ent_042'], 'is_relation_provenance': False})
        ]
        
        chunks_b = [
            Chunk('chunk_123', 'Privacy concerns...', 'doc_002', 'paper', None,
                  metadata={'faiss_rank': 0})
        ]
        
        ranker = ResultRanker()
        result = ranker.rank(chunks_a, chunks_b, subgraph, QueryFilters(), "test")
        
        # chunk_042 has provenance bonus → highest score
        scores = {c.chunk_id: c.score for c in result.chunks}
        assert scores['chunk_042'] > scores['chunk_089']
        assert scores['chunk_042'] > scores.get('chunk_123', 0)
    
    def test_deduplication(self):
        """Test same chunk from both paths deduplicated."""
        from src.retrieval.result_ranker import ResultRanker
        
        subgraph = Subgraph(entities=[], relations=[])
        
        duplicate = Chunk('chunk_042', 'Text', 'doc_001', 'regulation', 'EU', {})
        chunks_a = [duplicate]
        chunks_b = [duplicate]
        
        ranker = ResultRanker()
        result = ranker.rank(chunks_a, chunks_b, subgraph, QueryFilters(), "test")
        
        chunk_ids = [c.chunk_id for c in result.chunks]
        assert chunk_ids.count('chunk_042') == 1


# ============================================================================
# REAL INTEGRATION TESTS (marked with @pytest.mark.integration)
# ============================================================================

@pytest.fixture
def real_data_available():
    """Check if real data files exist."""
    required_files = [
        'data/processed/faiss/entity_embeddings.index',
        'data/processed/faiss/entity_id_map.json',
        'data/interim/entities/normalized_entities_with_ids.json',
        'data/processed/faiss/chunk_embeddings.index',
        'data/processed/faiss/chunk_id_map.json',
    ]
    
    missing = [f for f in required_files if not Path(f).exists()]
    
    if missing:
        pytest.skip(f"Missing real data files: {missing}")
    
    return True


@pytest.fixture
def real_embedder():
    """Load real BGE-M3 embedder."""
    try:
        from src.utils.embedder import BGEEmbedder
        embedder = BGEEmbedder()
        
        # Test it works
        test_emb = embedder.embed_single("test")
        assert test_emb.shape == (1024,), f"Wrong shape: {test_emb.shape}"
        
        return embedder
    except Exception as e:
        pytest.skip(f"Cannot load embedder: {e}")


@pytest.fixture
def neo4j_available():
    """Check if Neo4j is accessible."""
    from neo4j import GraphDatabase
    
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD')
    
    if not password:
        pytest.skip("NEO4J_PASSWORD not set")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            assert result.single()['test'] == 1
        driver.close()
        return (uri, user, password)
    except Exception as e:
        pytest.skip(f"Cannot connect to Neo4j: {e}")


@pytest.mark.integration
class TestRealIntegration:
    """Integration tests with real data."""
    
    def test_real_query_parser(self, real_embedder, real_data_available):
        """Test QueryParser with real BGE-M3 and Mistral API."""
        from src.retrieval.query_parser import QueryParser
        
        parser = QueryParser(real_embedder)
        
        query = "What does the EU AI Act say about facial recognition?"
        parsed = parser.parse(query)
        
        print(f"\n✅ REAL QueryParser Test:")
        print(f"   Query: {query}")
        print(f"   Entities: {[(e.name, e.type) for e in parsed.extracted_entities]}")
        print(f"   Filters: jurisdictions={parsed.filters.jurisdiction_hints}, "
              f"doc_types={parsed.filters.doc_type_hints}")
        print(f"   Embedding shape: {parsed.query_embedding.shape}")
        
        assert parsed.query_embedding.shape == (1024,)
        assert len(parsed.extracted_entities) > 0
    
    def test_real_entity_resolver(self, real_embedder, real_data_available):
        """Test EntityResolver with real 55K entity graph."""
        from src.retrieval.entity_resolver import EntityResolver
        
        resolver = EntityResolver(
            faiss_index_path='data/processed/faiss/entity_embeddings.index',
            entity_ids_path='data/processed/faiss/entity_id_map.json',
            normalized_entities_path='data/interim/entities/normalized_entities_with_ids.json',
            embedding_model=real_embedder,
            threshold=0.75,
            top_k=3
        )
        
        entities = [
            ExtractedEntity(name="EU AI Act", type="Regulation"),
            ExtractedEntity(name="facial recognition", type="Technology")
        ]
        
        resolved = resolver.resolve(entities)
        
        print(f"\n✅ REAL EntityResolver Test:")
        print(f"   Input: {[(e.name, e.type) for e in entities]}")
        print(f"   Resolved: {len(resolved)} entities")
        for e in resolved:
            print(f"     - {e.name} ({e.type}): {e.match_type}, confidence={e.confidence:.3f}")
        
        assert len(resolved) > 0
        assert all(e.confidence >= 0.75 for e in resolved if e.match_type == 'fuzzy')
    
    def test_real_graph_expander(self, real_embedder, real_data_available, neo4j_available):
        """Test GraphExpander with real Neo4j PCST using actual resolved entities."""
        from src.retrieval.graph_expander import GraphExpander
        from src.retrieval.entity_resolver import EntityResolver
        from src.retrieval.config import QueryFilters
        
        uri, user, password = neo4j_available
        
        # First: Resolve entities to get real IDs from the dataset
        resolver = EntityResolver(
            faiss_index_path='data/processed/faiss/entity_embeddings.index',
            entity_ids_path='data/processed/faiss/entity_id_map.json',
            normalized_entities_path='data/interim/entities/normalized_entities_with_ids.json',
            embedding_model=real_embedder,
            top_k=3  # Limit to 3 matches per entity
        )
        
        # Use real extracted entities (from query parser test pattern)
        extracted = [
            ('EU AI Act', 'Regulatory Concept'),
            ('facial recognition', 'Technical Term')
        ]
        
        filters = QueryFilters(
            jurisdiction_hints=['EU'],
            doc_type_hints=['regulation']
        )
        
        # Resolve to get real entity IDs from the 55K entity dataset
        resolved = resolver.resolve(extracted, filters)
        
        # Take top 2 for graph expansion
        entities_to_expand = resolved[:2]
        
        print(f"\n✅ Using real entities for expansion:")
        for e in entities_to_expand:
            print(f"   - {e.name} ({e.entity_id}): {e.match_type}, confidence={e.confidence:.3f}")
        
        # Now test GraphExpander with real entities
        expander = GraphExpander(
            neo4j_uri=uri,
            neo4j_user=user,
            neo4j_password=password,
            entity_index_path='data/processed/faiss/entity_embeddings.index',
            entity_id_map_path='data/processed/faiss/entity_id_map.json'
        )
        
        try:
            subgraph = expander.expand(entities_to_expand)
            
            print(f"\n✅ REAL GraphExpander Test:")
            print(f"   Input entities: {len(entities_to_expand)}")
            print(f"   Expanded to: {len(subgraph.entity_ids)} entities")
            print(f"   Relations found: {len(subgraph.relations)}")
            
            # Validation: PCST should expand but stay bounded
            assert len(subgraph.entity_ids) >= len(entities_to_expand), "Should at least have input entities"
            assert len(subgraph.entity_ids) <= 50, "PCST should limit expansion (hub node control)"
            
            # Show sample expanded entities
            if subgraph.entity_ids:
                print(f"   Sample expanded entities:")
                for eid in list(subgraph.entity_ids)[:5]:
                    print(f"     - {eid}")
            
            # Show sample relations
            if subgraph.relations:
                print(f"   Sample relations:")
                for rel in subgraph.relations[:3]:
                    print(f"     - {rel['subject_id']} --{rel['predicate']}--> {rel['object_id']}")
            
            assert isinstance(subgraph, Subgraph)
        
        finally:
            expander.close()
    
    def test_real_full_pipeline(self, real_embedder, real_data_available, neo4j_available):
        """Test complete pipeline end-to-end with real data."""
        from src.retrieval.retrieval_processor import RetrievalProcessor
        
        uri, user, password = neo4j_available
        
        processor = RetrievalProcessor(
            embedding_model=real_embedder,
            faiss_entity_index_path=Path('data/processed/faiss/entity_embeddings.index'),
            entity_ids_path=Path('data/processed/faiss/entity_id_map.json'),
            normalized_entities_path=Path('data/interim/entities/normalized_entities_with_ids.json'),
            faiss_chunk_index_path=Path('data/processed/faiss/chunk_embeddings.index'),
            chunk_ids_path=Path('data/processed/faiss/chunk_id_map.json'),
            neo4j_uri=uri,
            neo4j_user=user,
            neo4j_password=password
        )
        
        try:
            query = "What does the EU AI Act say about facial recognition?"
            
            print(f"\n✅ REAL Full Pipeline Test:")
            print(f"   Query: {query}")
            
            result = processor.retrieve(query)
            
            print(f"   Chunks retrieved: {len(result.chunks)}")
            print(f"   Entities in subgraph: {len(result.subgraph.entities)}")
            print(f"   Relations in subgraph: {len(result.subgraph.relations)}")
            
            if result.chunks:
                print(f"\n   Top 3 chunks:")
                for i, chunk in enumerate(result.chunks[:3], 1):
                    print(f"     [{i}] Score: {chunk.score:.3f}, Path: {chunk.source_path}")
                    print(f"         {chunk.text[:100]}...")
            
            if result.subgraph.relations:
                print(f"\n   Relations:")
                for rel in result.subgraph.relations[:5]:
                    print(f"     • {rel.source_name} --{rel.predicate}--> {rel.target_name}")
            
            # Validate structure
            assert isinstance(result, RetrievalResult)
            assert len(result.chunks) > 0
            assert result.query == query
            
            # Test prompt formatting
            from src.retrieval.result_ranker import ResultRanker
            ranker = ResultRanker()
            prompt_data = ranker.format_for_prompt(result)
            
            assert 'graph_structure' in prompt_data
            assert 'sources' in prompt_data
            assert 'query' in prompt_data
            
            print(f"\n   Prompt formatted successfully")
            print(f"   Graph structure length: {len(prompt_data['graph_structure'])} chars")
            print(f"   Sources length: {len(prompt_data['sources'])} chars")
        
        finally:
            processor.close()


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    # Run with: python test_retrieval_complete.py
    # Or: pytest test_retrieval_complete.py -v -m "not integration"
    pytest.main([__file__, '-v', '-s'])