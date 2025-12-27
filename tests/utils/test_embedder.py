# -*- coding: utf-8 -*-
"""
Universal

Tests BGEEmbedder and EmbedProcessor with both chunk and entity embedding
using the universal architecture. Includes unit tests, integration tests
with real data, and performance benchmarks.

"""
"""
import sys
import pytest
import numpy as np
from pathlib import Path
import json
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.embedder import BGEEmbedder
from src.utils.embed_processor import EmbedProcessor


class TestBGEEmbedder:
    """Test universal BGEEmbedder class"""
    
    @pytest.fixture
    def embedder(self):
        """Create embedder instance for tests"""
        return BGEEmbedder(model_name='BAAI/bge-m3')
    
    def test_model_initialization(self, embedder):
        """Test model loads correctly"""
        assert embedder.embedding_dim == 1024
        assert embedder.model is not None
    
    def test_single_embedding(self, embedder):
        """Test embedding a single text"""
        text = "AI systems should be transparent and explainable."
        
        embedding = embedder.embed_single(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1024,)
        assert not np.all(embedding == 0)  # Should not be all zeros
    
    def test_batch_embedding(self, embedder):
        """Test batch embedding multiple texts"""
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
    
    def test_chunk_text_embedding(self, embedder):
        """Test embedding chunk-like text (full paragraphs)"""
        chunk_text = """
        Article 1: Purpose and Scope
        This regulation establishes harmonized rules for artificial intelligence systems.
        It applies to providers and users of AI systems in the European Union.
        """
        
        embedding = embedder.embed_single(chunk_text)
        
        assert embedding.shape == (1024,)
        assert not np.all(embedding == 0)
    
    def test_entity_text_embedding(self, embedder):
        """Test embedding entity-formatted text ('name [type]')"""
        entity_texts = [
            "GDPR [Regulation]",
            "European Union [Organization]",
            "data protection [Concept]"
        ]
        
        embeddings = embedder.embed_batch(entity_texts, batch_size=3, show_progress=False)
        
        assert embeddings.shape == (3, 1024)
        # Verify embeddings are different for different entities
        assert not np.allclose(embeddings[0], embeddings[1])


class TestEmbedProcessor:
    """Test universal EmbedProcessor class"""
    
    @pytest.fixture
    def embedder(self):
        """Create embedder for processor tests"""
        return BGEEmbedder(model_name='BAAI/bge-m3')
    
    @pytest.fixture
    def processor(self, embedder):
        """Create processor with embedder"""
        return EmbedProcessor(embedder, checkpoint_freq=100)
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks"""
        return {
            "chunk_001": {
                "chunk_id": "chunk_001",
                "text": "This is a test chunk about AI regulation.",
                "metadata": {"source": "test"}
            },
            "chunk_002": {
                "chunk_id": "chunk_002",
                "text": "Another chunk discussing transparency.",
                "metadata": {"source": "test"}
            },
            "chunk_003": {
                "chunk_id": "chunk_003",
                "text": "Third chunk about human oversight requirements.",
                "metadata": {"source": "test"}
            }
        }
    
    @pytest.fixture
    def sample_entities(self):
        """Create sample entities"""
        return {
            "entity_001": {
                "entity_id": "entity_001",
                "name": "GDPR",
                "type": "Regulation",
                "description": "General Data Protection Regulation",
                "chunk_ids": ["chunk_001"]
            },
            "entity_002": {
                "entity_id": "entity_002",
                "name": "European Union",
                "type": "Organization",
                "description": "Political and economic union of European states",
                "chunk_ids": ["chunk_002"]
            },
            "entity_003": {
                "entity_id": "entity_003",
                "name": "transparency",
                "type": "Concept",
                "description": "Principle of openness and accountability",
                "chunk_ids": ["chunk_002", "chunk_003"]
            }
        }
    
    def test_load_items_dict(self, processor):
        """Test loading items from dict format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = {"item_1": {"text": "test"}, "item_2": {"text": "test2"}}
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            items = processor.load_items(Path(temp_path))
            assert len(items) == 2
            assert "item_1" in items
        finally:
            Path(temp_path).unlink()
    
    def test_load_items_list(self, processor):
        """Test loading items from list format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = [{"text": "test1"}, {"text": "test2"}]
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            items = processor.load_items(Path(temp_path))
            assert len(items) == 2
            assert 0 in items
            assert 1 in items
        finally:
            Path(temp_path).unlink()
    
    def test_process_chunks(self, processor, sample_chunks):
        """Test processing chunks (full text)"""
        enriched = processor.process_items(
            sample_chunks,
            text_key='text',
            batch_size=2
        )
        
        # Check embeddings were added
        for chunk_id, chunk_data in enriched.items():
            assert "embedding" in chunk_data
            assert len(chunk_data["embedding"]) == 1024
            assert isinstance(chunk_data["embedding"], list)
            # Original fields preserved
            assert "text" in chunk_data
            assert "metadata" in chunk_data
    
    def test_process_entities(self, processor, sample_entities):
        """Test processing entities (name [type] format)"""
        # First format entities
        for entity_id, entity in sample_entities.items():
            sample_entities[entity_id]['formatted_text'] = f"{entity['name']} [{entity['type']}]"
        
        enriched = processor.process_items(
            sample_entities,
            text_key='formatted_text',
            batch_size=2
        )
        
        # Check embeddings were added
        for entity_id, entity_data in enriched.items():
            assert "embedding" in entity_data
            assert len(entity_data["embedding"]) == 1024
            assert isinstance(entity_data["embedding"], list)
            # Original fields preserved
            assert "name" in entity_data
            assert "type" in entity_data
            assert "description" in entity_data
            assert "formatted_text" in entity_data
    
    def test_verification_chunks(self, processor, sample_chunks):
        """Test verification for chunks"""
        enriched = processor.process_items(sample_chunks, text_key='text', batch_size=2)
        stats = processor.verify_embeddings(enriched)
        
        assert stats['total_items'] == 3
        assert stats['items_with_embeddings'] == 3
        assert stats['items_correct_dim'] == 3
        assert stats['success_rate'] == 100.0
    
    def test_verification_entities(self, processor, sample_entities):
        """Test verification for entities"""
        # Format entities
        for entity_id, entity in sample_entities.items():
            sample_entities[entity_id]['formatted_text'] = f"{entity['name']} [{entity['type']}]"
        
        enriched = processor.process_items(sample_entities, text_key='formatted_text', batch_size=2)
        stats = processor.verify_embeddings(enriched)
        
        assert stats['total_items'] == 3
        assert stats['items_with_embeddings'] == 3
        assert stats['items_correct_dim'] == 3
        assert stats['success_rate'] == 100.0
    
    def test_checkpoint_save(self, processor, sample_chunks, tmp_path):
        """Test checkpoint saving"""
        checkpoint_dir = tmp_path / "checkpoints"
        
        enriched = processor.process_items(
            sample_chunks,
            text_key='text',
            batch_size=2,
            checkpoint_dir=checkpoint_dir
        )
        
        # Checkpoints should exist if item count > checkpoint_freq
        # (won't trigger with only 3 items and checkpoint_freq=100)
        assert enriched is not None


class TestEntityFormatting:
    """Test entity-specific formatting for RAKG"""
    
    def test_entity_format(self):
        """Test entity formatting matches RAKG Eq. 19"""
        entity = {
            "name": "GDPR",
            "type": "Regulation",
            "description": "General Data Protection Regulation"
        }
        
        formatted = f"{entity['name']} [{entity['type']}]"
        
        assert formatted == "GDPR [Regulation]"
        assert entity['description'] not in formatted  # Description NOT included
    
    def test_entity_embedding_format(self):
        """Test entities are embedded with correct format"""
        embedder = BGEEmbedder()
        
        entities = [
            {"name": "EU", "type": "Organization"},
            {"name": "AI Act", "type": "Regulation"},
            {"name": "transparency", "type": "Concept"}
        ]
        
        formatted_texts = [f"{e['name']} [{e['type']}]" for e in entities]
        
        embeddings = embedder.embed_batch(formatted_texts, batch_size=3, show_progress=False)
        
        assert embeddings.shape == (3, 1024)
        # Different entities should have different embeddings
        assert not np.allclose(embeddings[0], embeddings[1])
        assert not np.allclose(embeddings[1], embeddings[2])


@pytest.mark.integration
class TestRealData:
    """Integration tests with actual project data (if available)"""
    
    def test_real_chunks_sample(self):
        """Test on sample of real chunks from project"""
        chunks_path = Path("data/interim/chunks/chunks_text.json")
        
        if not chunks_path.exists():
            pytest.skip("Real chunks file not found")
        
        with open(chunks_path, 'r', encoding='utf-8') as f:
            all_chunks = json.load(f)
        
        # Take first 5 chunks
        sample_ids = list(all_chunks.keys())[:5]
        sample_chunks = {cid: all_chunks[cid] for cid in sample_ids}
        
        # Process
        embedder = BGEEmbedder()
        processor = EmbedProcessor(embedder)
        enriched = processor.process_items(sample_chunks, text_key='text', batch_size=5)
        
        # Verify
        stats = processor.verify_embeddings(enriched)
        assert stats['success_rate'] == 100.0
        
        # Check embedding values are reasonable
        sample_embedding = next(iter(enriched.values()))['embedding']
        assert min(sample_embedding) < 0  # Should have negative values
        assert max(sample_embedding) > 0  # Should have positive values
        assert abs(np.mean(sample_embedding)) < 0.1  # Mean should be near zero
    
    def test_real_entities_sample(self):
        """Test on sample of real entities from project (if Phase 1B complete)"""
        entities_path = Path("data/interim/entities/pre_entities.json")
        
        if not entities_path.exists():
            pytest.skip("Real entities file not found (Phase 1B not complete)")
        
        with open(entities_path, 'r', encoding='utf-8') as f:
            all_entities = json.load(f)
        
        # Handle list or dict format
        if isinstance(all_entities, list):
            sample_entities = {i: all_entities[i] for i in range(min(5, len(all_entities)))}
        else:
            sample_ids = list(all_entities.keys())[:5]
            sample_entities = {eid: all_entities[eid] for eid in sample_ids}
        
        # Format entities
        for entity_id, entity in sample_entities.items():
            sample_entities[entity_id]['formatted_text'] = f"{entity['name']} [{entity['type']}]"
        
        # Process
        embedder = BGEEmbedder()
        processor = EmbedProcessor(embedder)
        enriched = processor.process_items(sample_entities, text_key='formatted_text', batch_size=5)
        
        # Verify
        stats = processor.verify_embeddings(enriched)
        assert stats['success_rate'] == 100.0


@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks"""
    
    def test_batch_size_performance(self):
        """Compare performance of different batch sizes"""
        embedder = BGEEmbedder()
        
        # Generate test texts
        texts = [f"This is test text number {i}" for i in range(100)]
        
        import time
        
        # Test batch_size=8
        start = time.time()
        embedder.embed_batch(texts, batch_size=8, show_progress=False)
        time_batch_8 = time.time() - start
        
        # Test batch_size=32
        start = time.time()
        embedder.embed_batch(texts, batch_size=32, show_progress=False)
        time_batch_32 = time.time() - start
        
        # batch_size=32 should be faster (or similar) for 100 texts
        # Just verify both complete without error
        assert time_batch_8 > 0
        assert time_batch_32 > 0
        
        print(f"\nBatch 8: {time_batch_8:.2f}s, Batch 32: {time_batch_32:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
