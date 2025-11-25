"""
Unit tests for chunk embedder
Tests model loading, embedding dimensions, and batch processing

Run: pytest tests/test_embedder.py -v
"""
import sys
import pytest
import numpy as np
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.phase1_graph_construction.embedder import ChunkEmbedder
from src.phase1_graph_construction.embed_processor import EmbedProcessor


class TestChunkEmbedder:
    """Test ChunkEmbedder class"""
    
    @pytest.fixture
    def embedder(self):
        """Create embedder instance for tests"""
        return ChunkEmbedder(model_name='BAAI/bge-m3')
    
    def test_model_initialization(self, embedder):
        """Test model loads correctly"""
        assert embedder.embedding_dim == 1024
        assert embedder.model is not None
    
    def test_single_embedding(self, embedder):
        """Test embedding a single chunk"""
        text = "AI systems should be transparent and explainable."
        
        embedding = embedder.embed_single(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1024,)
        assert not np.all(embedding == 0)  # Should not be all zeros
    
    def test_batch_embedding(self, embedder):
        """Test batch embedding"""
        texts = [
            "The EU AI Act regulates high-risk systems.",
            "Transparency is a key principle.",
            "Human oversight must be ensured."
        ]
        
        embeddings = embedder.embed_batch(texts, batch_size=2, show_progress=False)
        
        assert embeddings.shape == (3, 1024)
        assert not np.all(embeddings == 0)
    
    def test_embedding_consistency(self, embedder):
        """Test same text produces same embedding"""
        text = "AI governance requires clear regulations."
        
        emb1 = embedder.embed_single(text)
        emb2 = embedder.embed_single(text)
        
        # Should be nearly identical (within floating point precision)
        assert np.allclose(emb1, emb2, rtol=1e-5)


class TestEmbedProcessor:
    """Test EmbedProcessor class"""
    
    @pytest.fixture
    def sample_chunks(self, tmp_path):
        """Create sample chunks file"""
        chunks = {
            "test_chunk_001": {
                "chunk_id": "test_chunk_001",
                "text": "This is a test chunk about AI regulation.",
                "metadata": {"source": "test"}
            },
            "test_chunk_002": {
                "chunk_id": "test_chunk_002",
                "text": "Another chunk discussing transparency.",
                "metadata": {"source": "test"}
            }
        }
        
        filepath = tmp_path / "test_chunks.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chunks, f)
        
        return filepath
    
    @pytest.fixture
    def processor(self):
        """Create processor with embedder"""
        embedder = ChunkEmbedder(model_name='BAAI/bge-m3')
        return EmbedProcessor(embedder, checkpoint_freq=100)
    
    def test_load_chunks(self, processor, sample_chunks):
        """Test loading chunks from JSON"""
        chunks = processor.load_chunks(sample_chunks)
        
        assert len(chunks) == 2
        assert "test_chunk_001" in chunks
        assert "text" in chunks["test_chunk_001"]
    
    def test_process_chunks(self, processor, sample_chunks):
        """Test full embedding pipeline"""
        chunks = processor.load_chunks(sample_chunks)
        enriched_chunks = processor.process_chunks(chunks, batch_size=2)
        
        # Check embeddings were added
        for chunk_id, chunk_data in enriched_chunks.items():
            assert "embedding" in chunk_data
            assert len(chunk_data["embedding"]) == 1024
            assert isinstance(chunk_data["embedding"], list)
    
    def test_verification(self, processor, sample_chunks):
        """Test embedding verification"""
        chunks = processor.load_chunks(sample_chunks)
        enriched_chunks = processor.process_chunks(chunks, batch_size=2)
        
        stats = processor.verify_embeddings(enriched_chunks)
        
        assert stats['total_chunks'] == 2
        assert stats['chunks_with_embeddings'] == 2
        assert stats['success_rate'] == 100.0


@pytest.mark.integration
class TestRealChunks:
    """Integration test with actual project chunks"""
    
    def test_sample_real_chunks(self):
        """Test on 5 real chunks from project"""
        # Load actual chunks
        chunks_path = Path("data/interim/chunks/chunks_text.json")
        
        if not chunks_path.exists():
            pytest.skip("Real chunks file not found")
        
        with open(chunks_path, 'r', encoding='utf-8') as f:
            all_chunks = json.load(f)
        
        # Take first 5 chunks
        sample_ids = list(all_chunks.keys())[:5]
        sample_chunks = {cid: all_chunks[cid] for cid in sample_ids}
        
        # Process
        embedder = ChunkEmbedder()
        processor = EmbedProcessor(embedder)
        enriched = processor.process_chunks(sample_chunks, batch_size=5)
        
        # Verify
        stats = processor.verify_embeddings(enriched)
        assert stats['success_rate'] == 100.0