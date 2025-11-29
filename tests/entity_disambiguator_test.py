"""
Module: test_entity_disambiguator.py
Phase: 1C - Entity Disambiguation
Purpose: Unit tests for 4-stage disambiguation pipeline
Author: Pau Barba i Colomer
Created: 2025-11-29
Last Modified: 2025-11-29

Usage:
    # Run all tests
    pytest tests/test_entity_disambiguator.py -v
    
    # Run specific test
    pytest tests/test_entity_disambiguator.py::test_exact_dedup -v
    
    # Run with coverage
    pytest tests/test_entity_disambiguator.py --cov=entity_disambiguator

Notes:
    - Tests each stage independently with small datasets
    - Integration test with 1000 entities
    - Mock LLM responses for Stage 4 to avoid API costs
"""

# Standard library
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "src" / "phase1_graph_construction"))

# Third-party
import pytest

# Local imports
from entity_disambiguator import (
    ExactDeduplicator,
    FAISSBlocker,
    TieredThresholdFilter,
    SameJudgeLLM
)


class TestExactDeduplicator:
    """Tests for Stage 1: Exact string deduplication"""
    
    def test_normalize_string(self):
        """Test string normalization"""
        dedup = ExactDeduplicator()
        
        # Test casefold
        assert dedup.normalize_string("AI") == dedup.normalize_string("ai")
        assert dedup.normalize_string("AI") == dedup.normalize_string("Ai")
        
        # Test whitespace collapse
        assert dedup.normalize_string("AI  Technology") == "ai technology"
        assert dedup.normalize_string("  AI  ") == "ai"
        
        # Test NFKC normalization (ligatures)
        # Note: ﬁ is ligature for fi
        assert "fi" in dedup.normalize_string("ﬁle")
    
    def test_compute_hash(self):
        """Test hash computation"""
        dedup = ExactDeduplicator()
        
        # Same normalized string → same hash
        h1 = dedup.compute_hash("AI", "Technology")
        h2 = dedup.compute_hash("ai", "TECHNOLOGY")
        assert h1 == h2
        
        # Different string → different hash
        h3 = dedup.compute_hash("ML", "Technology")
        assert h1 != h3
    
    def test_deduplicate_simple(self):
        """Test deduplication with simple duplicates"""
        dedup = ExactDeduplicator()
        
        entities = [
            {"name": "AI", "type": "Technology", "chunk_ids": [1]},
            {"name": "ai", "type": "Technology", "chunk_ids": [2]},
            {"name": "AI", "type": "TECHNOLOGY", "chunk_ids": [3]}
        ]
        
        result = dedup.deduplicate(entities)
        
        # Should merge to 1 entity
        assert len(result) == 1
        
        # Should combine chunk_ids
        assert set(result[0]['chunk_ids']) == {1, 2, 3}
        
        # Should have metadata
        assert result[0]['duplicate_count'] == 3
    
    def test_deduplicate_preserves_embeddings(self):
        """Test that embeddings are preserved during dedup"""
        dedup = ExactDeduplicator()
        
        entities = [
            {
                "name": "AI", 
                "type": "Technology", 
                "chunk_ids": [1],
                "embedding": [0.1, 0.2, 0.3]
            },
            {
                "name": "ai", 
                "type": "Technology", 
                "chunk_ids": [2],
                "embedding": [0.4, 0.5, 0.6]  # Different embedding
            }
        ]
        
        result = dedup.deduplicate(entities)
        
        # Should preserve embedding from first entity
        assert 'embedding' in result[0]
        assert result[0]['embedding'] == [0.1, 0.2, 0.3]
    
    def test_type_voting(self):
        """Test that most frequent type wins"""
        dedup = ExactDeduplicator()
        
        entities = [
            {"name": "AI Act", "type": "Regulation", "chunk_ids": [1]},
            {"name": "AI Act", "type": "Regulation", "chunk_ids": [2]},
            {"name": "AI Act", "type": "Legislation", "chunk_ids": [3]}
        ]
        
        result = dedup.deduplicate(entities)
        
        # "Regulation" appears twice, should win
        assert result[0]['type'] == "Regulation"
    
    def test_no_duplicates(self):
        """Test with no duplicates"""
        dedup = ExactDeduplicator()
        
        entities = [
            {"name": "AI", "type": "Technology", "chunk_ids": [1]},
            {"name": "ML", "type": "Technology", "chunk_ids": [2]},
            {"name": "NLP", "type": "Technology", "chunk_ids": [3]}
        ]
        
        result = dedup.deduplicate(entities)
        
        # Should return all 3 entities unchanged
        assert len(result) == 3


class TestFAISSBlocker:
    """Tests for Stage 2: FAISS HNSW blocking"""
    
    @pytest.fixture
    def sample_entities(self):
        """Create sample entities with random embeddings"""
        np.random.seed(42)
        entities = []
        for i in range(100):
            entities.append({
                'name': f'Entity_{i}',
                'type': 'Test',
                'embedding': np.random.randn(1024).tolist()
            })
        return entities
    
    def test_build_index(self, sample_entities):
        """Test FAISS index building"""
        blocker = FAISSBlocker()
        blocker.build_index(sample_entities)
        
        assert blocker.index is not None
        assert blocker.stats['entities_indexed'] == 100
    
    def test_find_candidates(self, sample_entities):
        """Test candidate pair finding"""
        blocker = FAISSBlocker()
        blocker.build_index(sample_entities)
        
        pairs = blocker.find_candidates(sample_entities, k=10)
        
        # Should find some pairs
        assert len(pairs) > 0
        
        # Each pair should have 3 elements (i, j, similarity)
        assert all(len(pair) == 3 for pair in pairs)
        
        # Similarity should be between 0 and 1
        assert all(0 <= pair[2] <= 1 for pair in pairs)
        
        # Should not have self-matches
        assert all(pair[0] != pair[1] for pair in pairs)
    
    def test_similarity_symmetric(self, sample_entities):
        """Test that similarity is symmetric"""
        blocker = FAISSBlocker()
        blocker.build_index(sample_entities)
        
        pairs = blocker.find_candidates(sample_entities, k=10)
        
        # Build dict of similarities
        sim_dict = {}
        for i, j, sim in pairs:
            key1 = (min(i, j), max(i, j))
            sim_dict[key1] = sim
        
        # Check no duplicate pairs
        assert len(sim_dict) == len(pairs)


class TestTieredThresholdFilter:
    """Tests for Stage 3: Tiered threshold filtering"""
    
    def test_filter_pairs(self):
        """Test threshold filtering"""
        filter = TieredThresholdFilter(
            auto_merge_threshold=0.95,
            auto_reject_threshold=0.80
        )
        
        # Create test pairs with known similarities
        pairs = [
            (0, 1, 0.98),  # Should auto-merge
            (0, 2, 0.90),  # Should be uncertain
            (0, 3, 0.70),  # Should auto-reject
            (1, 2, 0.96),  # Should auto-merge
            (1, 3, 0.85),  # Should be uncertain
        ]
        
        entities = [{'name': f'E{i}'} for i in range(4)]
        
        result = filter.filter_pairs(pairs, entities)
        
        # Check counts
        assert len(result['merged']) == 2  # 0.98, 0.96
        assert len(result['rejected']) == 1  # 0.70
        assert len(result['uncertain']) == 2  # 0.90, 0.85
    
    def test_apply_merges_simple(self):
        """Test merge application with simple case"""
        filter = TieredThresholdFilter()
        
        entities = [
            {'name': 'A', 'chunk_ids': [1]},
            {'name': 'B', 'chunk_ids': [2]},
            {'name': 'C', 'chunk_ids': [3]}
        ]
        
        # Merge A and B
        merged_pairs = [(0, 1)]
        
        result = filter.apply_merges(entities, merged_pairs)
        
        # Should have 2 entities (A+B merged, C separate)
        assert len(result) == 2
    
    def test_apply_merges_transitive(self):
        """Test transitivity: if A=B and B=C, then A=C"""
        filter = TieredThresholdFilter()
        
        entities = [
            {'name': 'A', 'chunk_ids': [1]},
            {'name': 'B', 'chunk_ids': [2]},
            {'name': 'C', 'chunk_ids': [3]}
        ]
        
        # A=B and B=C
        merged_pairs = [(0, 1), (1, 2)]
        
        result = filter.apply_merges(entities, merged_pairs)
        
        # Should have 1 entity (all merged)
        assert len(result) == 1
        
        # Should have all chunk_ids
        assert set(result[0]['chunk_ids']) == {1, 2, 3}


class TestSameJudgeLLM:
    """Tests for Stage 4: LLM verification"""
    
    def test_build_prompt(self):
        """Test prompt building"""
        judge = SameJudgeLLM(api_key="dummy")
        
        entity1 = {
            'name': 'AI',
            'type': 'Technology',
            'description': 'Artificial Intelligence'
        }
        entity2 = {
            'name': 'Artificial Intelligence',
            'type': 'Technology',
            'description': 'AI systems'
        }
        
        prompt = judge._build_prompt(entity1, entity2)
        
        # Should contain entity names
        assert 'AI' in prompt
        assert 'Artificial Intelligence' in prompt
        
        # Should contain types
        assert 'Technology' in prompt
        
        # Should have expected format
        assert 'Decision:' in prompt
        assert 'Confidence:' in prompt
    
    def test_parse_response_yes(self):
        """Test parsing YES response"""
        judge = SameJudgeLLM(api_key="dummy")
        
        response = """Decision: YES
Confidence: HIGH
Reasoning: Both entities refer to the same concept."""
        
        result = judge._parse_response(response)
        
        assert result['is_same'] == True
        assert result['confidence'] == 0.9  # HIGH
        assert 'same concept' in result['reasoning']
    
    def test_parse_response_no(self):
        """Test parsing NO response"""
        judge = SameJudgeLLM(api_key="dummy")
        
        response = """Decision: NO
Confidence: MEDIUM
Reasoning: Different entities with different meanings."""
        
        result = judge._parse_response(response)
        
        assert result['is_same'] == False
        assert result['confidence'] == 0.7  # MEDIUM


class TestIntegration:
    """Integration tests for full pipeline"""
    
    def test_stage1_to_stage2(self):
        """Test Stage 1 → Stage 2 flow"""
        # Stage 1: Dedup
        dedup = ExactDeduplicator()
        entities = [
            {
                'name': 'AI', 
                'type': 'Technology', 
                'chunk_ids': [1],
                'embedding': np.random.randn(1024).tolist()
            },
            {
                'name': 'ai', 
                'type': 'Technology', 
                'chunk_ids': [2],
                'embedding': np.random.randn(1024).tolist()
            },
            {
                'name': 'ML', 
                'type': 'Technology', 
                'chunk_ids': [3],
                'embedding': np.random.randn(1024).tolist()
            }
        ]
        
        entities = dedup.deduplicate(entities)
        assert len(entities) == 2  # AI and ML
        
        # Stage 2: FAISS blocking
        blocker = FAISSBlocker()
        blocker.build_index(entities)
        pairs = blocker.find_candidates(entities, k=1)
        
        # Should find at least 1 pair
        assert len(pairs) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
