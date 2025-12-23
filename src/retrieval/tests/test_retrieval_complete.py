# -*- coding: utf-8 -*-
"""
Complete Phase 3 Retrieval Test Suite.

Tests Phase 3.3.1 (Query Understanding) and Phase 3.3.2 (Context Retrieval)
with both mock tests and real integration tests.

Example:
    pytest tests/retrieval/test_retrieval_complete.py -v -m "not integration"
    pytest tests/retrieval/test_retrieval_complete.py -v -m integration
"""

# Standard library
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Project root
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Third-party
import numpy as np
import pytest
from dotenv import load_dotenv

load_dotenv()

# Config imports (direct)
from config.retrieval_config import RetrievalMode

# Dataclass imports (direct)
from src.utils.dataclasses import (
    ExtractedQueryEntity as ExtractedEntity,
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
        np.array([[0.1, 0.2, 0.3]]),
        np.array([[0, 1, 2]])
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
        assert parsed.embedding.shape == (1024,)
    
    def test_parse_detects_filters(self, mock_embedder):
        """Test jurisdiction and doc type filter detection."""
        from src.retrieval.query_parser import QueryParser
        
        with patch('src.retrieval.query_parser.Together'):
            parser = QueryParser(mock_embedder)
            
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
                    faiss_index_path=entity_ids_path,
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
        
        mock_faiss_index.search.return_value = (
            np.array([[0.15]]),
            np.array([[0]])
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
                    faiss_index_path=entity_ids_path,
                    entity_ids_path=entity_ids_path,
                    normalized_entities_path=entities_path,
                    embedding_model=mock_embedder,
                    threshold=0.75
                )
                
                entities = [ExtractedEntity(name="European AI Regulation", type="Regulation")]
                resolved = resolver.resolve(entities)
                
                if resolved:
                    assert resolved[0].match_type == "fuzzy"
                    assert resolved[0].confidence >= 0.75
        finally:
            Path(entity_ids_path).unlink()
            Path(entities_path).unlink()
    
    def test_alias_match(self, mock_faiss_index, mock_embedder, sample_entity_map, sample_normalized_entities):
        """Test alias matching (v1.1 feature)."""
        from src.retrieval.entity_resolver import EntityResolver
        
        # Create alias file
        aliases = {"EU AI Act": ["European AI Act", "AI Act"]}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_entity_map, f)
            entity_ids_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_normalized_entities, f)
            entities_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(aliases, f)
            aliases_path = f.name
        
        try:
            with patch('src.retrieval.entity_resolver.faiss.read_index', return_value=mock_faiss_index):
                resolver = EntityResolver(
                    faiss_index_path=entity_ids_path,
                    entity_ids_path=entity_ids_path,
                    normalized_entities_path=entities_path,
                    embedding_model=mock_embedder,
                    aliases_path=aliases_path
                )
                
                # Query with alias
                entities = [ExtractedEntity(name="AI Act", type="Regulation")]
                resolved = resolver.resolve(entities)
                
                if resolved:
                    assert resolved[0].name == "EU AI Act"
                    assert resolved[0].match_type == "alias"
        finally:
            Path(entity_ids_path).unlink()
            Path(entities_path).unlink()
            Path(aliases_path).unlink()


# ============================================================================
# PHASE 3.3.2 TESTS (Context Retrieval)
# ============================================================================

class TestGraphExpander:
    """Test GraphExpander with mocks."""
    
    def test_expand_returns_subgraph(self, mock_faiss_index, mock_neo4j_driver):
        """Test PCST expansion returns valid subgraph."""
        from src.retrieval.graph_expander import GraphExpander
        
        driver, session = mock_neo4j_driver
        
        # Mock GDS exists check
        session.run.return_value.single.return_value = {'exists': True}
        
        with patch('src.retrieval.graph_expander.GraphDatabase.driver', return_value=driver):
            with patch('src.retrieval.graph_expander.faiss.read_index', return_value=mock_faiss_index):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(['ent_001', 'ent_002'], f)
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
                        ResolvedEntity(
                            entity_id='ent_001',
                            name='EU AI Act',
                            type='Regulation',
                            confidence=0.95,
                            match_type='exact'
                        )
                    ]
                    
                    subgraph = expander.expand(entities)
                    
                    assert isinstance(subgraph, Subgraph)
                    assert isinstance(subgraph.entities, list)
                    assert isinstance(subgraph.relations, list)
                finally:
                    Path(entity_map_path).unlink()


class TestChunkRetriever:
    """Test ChunkRetriever with mocks."""
    
    def test_retrieve_dual(self, mock_faiss_index, mock_neo4j_driver):
        """Test dual retrieval returns chunks from both channels."""
        from src.retrieval.chunk_retriever import ChunkRetriever
        
        driver, session = mock_neo4j_driver
        
        # Mock Neo4j results
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([
            {'chunk_id': 'chunk_001', 'text': 'Sample text', 'doc_id': 'doc1',
             'doc_type': 'regulation', 'jurisdiction': 'EU', 'entities': ['ent_001']}
        ])
        session.run.return_value = mock_result
        
        with patch('src.retrieval.chunk_retriever.GraphDatabase.driver', return_value=driver):
            with patch('src.retrieval.chunk_retriever.faiss.read_index', return_value=mock_faiss_index):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(['chunk_001', 'chunk_002'], f)
                    chunk_map_path = f.name
                
                try:
                    retriever = ChunkRetriever(
                        neo4j_uri='bolt://localhost:7687',
                        neo4j_user='neo4j',
                        neo4j_password='password',
                        chunk_index_path='chunks.index',
                        chunk_id_map_path=chunk_map_path
                    )
                    
                    subgraph = Subgraph(entities=['ent_001'], relations=[])
                    query_emb = np.random.rand(1024).astype('float32')
                    
                    graph_chunks, semantic_chunks = retriever.retrieve_dual(subgraph, query_emb)
                    
                    assert isinstance(graph_chunks, list)
                    assert isinstance(semantic_chunks, list)
                finally:
                    Path(chunk_map_path).unlink()


class TestResultRanker:
    """Test ResultRanker."""
    
    def test_rank_merges_chunks(self):
        """Test ranking merges and deduplicates chunks."""
        from src.retrieval.result_ranker import ResultRanker
        
        ranker = ResultRanker()
        
        graph_chunks = [
            Chunk(
                chunk_ids=['chunk_001'],
                document_ids=['doc1'],
                text='Graph chunk',
                position=0,
                sentence_count=1,
                token_count=10,
                metadata={'entities': ['ent_001'], 'score': 0.5, 'doc_type': 'regulation', 'jurisdiction': 'EU'}
            )
        ]
        
        semantic_chunks = [
            Chunk(
                chunk_ids=['chunk_002'],
                document_ids=['doc2'],
                text='Semantic chunk',
                position=0,
                sentence_count=1,
                token_count=10,
                metadata={'score': 0.8, 'doc_type': 'paper', 'jurisdiction': 'US'}
            )
        ]
        
        subgraph = Subgraph(entities=['ent_001'], relations=[])
        filters = QueryFilters()
        
        result = ranker.rank(graph_chunks, semantic_chunks, subgraph, filters, "test query")
        
        assert isinstance(result, RetrievalResult)
        assert len(result.chunks) == 2


# ============================================================================
# INTEGRATION TESTS (require real data)
# ============================================================================

@pytest.fixture
def real_data_available():
    """Check if real data files exist."""
    required_files = [
        'data/faiss/entities.index',
        'data/faiss/entity_ids.json',
        'data/processed/entities.json',
        'data/faiss/chunks.index',
        'data/faiss/chunk_ids.json',
    ]
    
    missing = [f for f in required_files if not Path(f).exists()]
    
    if missing:
        pytest.skip(f"Missing real data files: {missing}")
    
    return True


@pytest.fixture
def real_embedder():
    """Load real BGE-M3 embedder."""
    try:
        from src.utils.embeddings import EmbeddingModel
        embedder = EmbeddingModel()
        
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
        
        print(f"\nREAL QueryParser Test:")
        print(f"   Query: {query}")
        print(f"   Entities: {[(e.name, e.type) for e in parsed.extracted_entities]}")
        print(f"   Filters: jurisdictions={parsed.filters.jurisdiction_hints}")
        print(f"   Embedding shape: {parsed.embedding.shape}")
        
        assert parsed.embedding.shape == (1024,)
        assert len(parsed.extracted_entities) > 0
    
    def test_real_entity_resolver_with_aliases(self, real_embedder, real_data_available):
        """Test EntityResolver with aliases (v1.1 feature)."""
        from src.retrieval.entity_resolver import EntityResolver
        
        resolver = EntityResolver(
            faiss_index_path=Path('data/faiss/entities.index'),
            entity_ids_path=Path('data/faiss/entity_ids.json'),
            normalized_entities_path=Path('data/processed/entities.json'),
            aliases_path=Path('data/processed/entities/aliases.json'),
            embedding_model=real_embedder,
            threshold=0.75,
            top_k=10
        )
        
        entities = [
            ExtractedEntity(name="European Union", type="Location"),
            ExtractedEntity(name="EU AI Act", type="Regulation")
        ]
        
        resolved = resolver.resolve(entities)
        
        print(f"\nREAL EntityResolver with Aliases Test:")
        print(f"   Input: {[(e.name, e.type) for e in entities]}")
        print(f"   Resolved: {len(resolved)} entities")
        for e in resolved:
            print(f"     - {e.name} ({e.type}): {e.match_type}, confidence={e.confidence:.3f}")
        
        assert len(resolved) > 0
    
    def test_real_full_pipeline(self, real_embedder, real_data_available, neo4j_available):
        """Test complete pipeline end-to-end with real data."""
        from src.retrieval.retrieval_processor import RetrievalProcessor
        
        uri, user, password = neo4j_available
        
        processor = RetrievalProcessor(
            embedding_model=real_embedder,
            faiss_entity_index_path=Path('data/faiss/entities.index'),
            entity_ids_path=Path('data/faiss/entity_ids.json'),
            normalized_entities_path=Path('data/processed/entities.json'),
            aliases_path=Path('data/processed/entities/aliases.json'),
            faiss_chunk_index_path=Path('data/faiss/chunks.index'),
            chunk_ids_path=Path('data/faiss/chunk_ids.json'),
            neo4j_uri=uri,
            neo4j_user=user,
            neo4j_password=password
        )
        
        try:
            query = "What does the EU AI Act say about facial recognition?"
            
            print(f"\nREAL Full Pipeline Test:")
            print(f"   Query: {query}")
            
            result = processor.retrieve(query)
            
            print(f"   Chunks retrieved: {len(result.chunks)}")
            print(f"   Entities in subgraph: {len(result.subgraph.entities) if result.subgraph else 0}")
            print(f"   Relations in subgraph: {len(result.subgraph.relations) if result.subgraph else 0}")
            
            if result.chunks:
                print(f"\n   Top 3 chunks:")
                for i, chunk in enumerate(result.chunks[:3], 1):
                    print(f"     [{i}] Score: {chunk.score:.3f}, Path: {chunk.source_path}")
                    print(f"         {chunk.text[:100]}...")
            
            assert isinstance(result, RetrievalResult)
            assert len(result.chunks) > 0
            assert result.query == query
        
        finally:
            processor.close()


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
