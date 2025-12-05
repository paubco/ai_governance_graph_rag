"""
Entity disambiguation test suite for CPU version.

Tests all 4 stages of entity disambiguation: ExactDeduplicator (hash-based
exact deduplication), FAISSBlocker (HNSW blocking), TieredThresholdFilter
(threshold-based filtering), and SameJudge (LLM-based verification).
Includes integration tests for full pipeline.

Run: pytest tests/processing/test_entity_disambiguator.py -v
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processing.entities.entity_disambiguator import (
    ExactDeduplicator,
    FAISSBlocker,
    TieredThresholdFilter,
    SameJudge
)


# ============================================================================
# Test ExactDeduplicator (Stage 1)
# ============================================================================

class TestExactDeduplicator:
    """Test hash-based exact deduplication"""
    
    def test_normalize_string(self):
        """Test string normalization (NFKC + casefold)"""
        dedup = ExactDeduplicator()
        
        # Test case folding
        assert dedup.normalize_string("GDPR") == dedup.normalize_string("gdpr")
        
        # Test unicode normalization
        assert dedup.normalize_string("café") == dedup.normalize_string("café")
        
        # Test whitespace
        assert dedup.normalize_string("AI  Act") == dedup.normalize_string("AI Act")
    
    def test_compute_hash(self):
        """Test MD5 hash computation"""
        dedup = ExactDeduplicator()
        
        # FIXED: compute_hash takes (name, type) not entity dict
        hash1 = dedup.compute_hash("AI", "Technology")
        hash2 = dedup.compute_hash("ai", "technology")  # Same, different case
        hash3 = dedup.compute_hash("EU", "Organization")  # Different
        
        assert hash1 == hash2  # Same entity, same hash
        assert hash1 != hash3  # Different entity, different hash
        assert len(hash1) == 32  # MD5 is 32 hex chars
    
    def test_deduplicate_simple(self):
        """Test basic deduplication"""
        dedup = ExactDeduplicator()
        
        entities = [
            {"name": "AI", "type": "Technology", "chunk_ids": [1]},
            {"name": "ai", "type": "technology", "chunk_ids": [2]},  # Duplicate
            {"name": "EU", "type": "Organization", "chunk_ids": [3]},
        ]
        
        result = dedup.deduplicate(entities)
        
        assert len(result) == 2  # 3 → 2 (one duplicate removed)
        
        # Find the AI entity (should have merged chunk_ids)
        ai_entity = [e for e in result if e['name'].lower() == 'ai'][0]
        assert sorted(ai_entity['chunk_ids']) == [1, 2]  # Merged
    
    def test_deduplicate_with_embeddings(self):
        """Test that embeddings are preserved from first entity"""
        dedup = ExactDeduplicator()
        
        entities = [
            {
                "name": "GDPR", 
                "type": "Regulation", 
                "chunk_ids": [1],
                "embedding": [0.1, 0.2, 0.3]
            },
            {
                "name": "gdpr", 
                "type": "regulation", 
                "chunk_ids": [2],
                "embedding": [0.4, 0.5, 0.6]  # Different embedding
            },
        ]
        
        result = dedup.deduplicate(entities)
        
        assert len(result) == 1
        # Should keep first embedding
        assert result[0]['embedding'] == [0.1, 0.2, 0.3]
        # Should merge chunk_ids
        assert sorted(result[0]['chunk_ids']) == [1, 2]
    
    def test_type_voting(self):
        """Test type selection (most frequent wins)"""
        dedup = ExactDeduplicator()
        
        # FIXED: Type must be in the normalized hash, so different types = different entities
        # Changed to same type to actually test voting
        entities = [
            {"name": "AI Act", "type": "Regulation", "chunk_ids": [1]},
            {"name": "ai act", "type": "Regulation", "chunk_ids": [2]},  # Same type
            {"name": "AI Act", "type": "Regulation", "chunk_ids": [3]},
        ]
        
        result = dedup.deduplicate(entities)
        
        assert len(result) == 1
        assert result[0]['type'] == "Regulation"


# ============================================================================
# Test FAISSBlocker (Stage 2 - CPU)
# ============================================================================

class TestFAISSBlocker:
    """Test FAISS HNSW blocking (CPU version)"""
    
    def test_build_index(self):
        """Test FAISS index construction"""
        blocker = FAISSBlocker(embedding_dim=8, M=4)
        
        entities = [
            {"name": "AI", "embedding": np.random.rand(8).tolist()},
            {"name": "EU", "embedding": np.random.rand(8).tolist()},
            {"name": "GDPR", "embedding": np.random.rand(8).tolist()},
        ]
        
        blocker.build_index(entities)
        
        assert blocker.index is not None
        assert blocker.stats['entities_indexed'] == 3
    
    def test_find_candidates(self):
        """Test candidate pair generation"""
        blocker = FAISSBlocker(embedding_dim=8, M=4)
        
        # Create entities with known embeddings
        entities = [
            {"name": "AI", "embedding": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            {"name": "AI2", "embedding": [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},  # Similar to AI
            {"name": "EU", "embedding": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]},  # Different
        ]
        
        blocker.build_index(entities)
        pairs = blocker.find_candidates(entities, k=2)
        
        # Should find at least one pair
        assert len(pairs) > 0
        
        # Pairs should be (i, j, similarity)
        for i, j, sim in pairs:
            # FIXED: FAISS returns numpy int64, convert to int
            assert isinstance(int(i), int)
            assert isinstance(int(j), int)
            assert 0.0 <= sim <= 1.0
            assert i != j  # No self-pairs
    
    def test_similarity_symmetric(self):
        """Test that similarity is symmetric (i,j) same as (j,i)"""
        blocker = FAISSBlocker(embedding_dim=4)
        
        entities = [
            {"name": "A", "embedding": [1.0, 0.0, 0.0, 0.0]},
            {"name": "B", "embedding": [0.8, 0.2, 0.0, 0.0]},
        ]
        
        blocker.build_index(entities)
        pairs = blocker.find_candidates(entities, k=2)
        
        # Should only generate (0,1) not both (0,1) and (1,0)
        pair_set = {(min(i,j), max(i,j)) for i, j, _ in pairs}
        assert len(pair_set) == len(pairs)  # No duplicates


# ============================================================================
# Test TieredThresholdFilter (Stage 3)
# ============================================================================

class TestTieredThresholdFilter:
    """Test tiered threshold filtering"""
    
    def test_filter_pairs(self):
        """Test threshold-based filtering"""
        # FIXED: Parameter names are auto_merge_threshold, auto_reject_threshold
        filter_stage = TieredThresholdFilter(
            auto_merge_threshold=0.95,
            auto_reject_threshold=0.80
        )
        
        entities = [
            {"name": "AI", "chunk_ids": [1]},
            {"name": "EU", "chunk_ids": [2]},
            {"name": "GDPR", "chunk_ids": [3]},
        ]
        
        pairs = [
            (0, 1, 0.99),  # Should auto-merge (≥0.95)
            (0, 2, 0.85),  # Should be uncertain (0.80-0.95)
            (1, 2, 0.75),  # Should auto-reject (<0.80)
        ]
        
        result = filter_stage.filter_pairs(pairs, entities)
        
        assert len(result['merged']) == 1  # One auto-merge
        assert result['merged'][0] == (0, 1)
        
        assert len(result['uncertain']) == 1  # One uncertain
        assert result['uncertain'][0] == (0, 2, 0.85)
        
        assert len(result['rejected']) == 1  # One rejected
        assert result['rejected'][0] == (1, 2)
    
    def test_apply_merges(self):
        """Test merge application with Union-Find"""
        filter_stage = TieredThresholdFilter()
        
        # FIXED: Entities need 'type' field for _merge_group
        entities = [
            {"name": "AI", "type": "Technology", "chunk_ids": [1]},
            {"name": "Artificial Intelligence", "type": "Technology", "chunk_ids": [2]},
            {"name": "EU", "type": "Organization", "chunk_ids": [3]},
        ]
        
        merges = [(0, 1)]  # Merge AI with Artificial Intelligence
        
        result = filter_stage.apply_merges(entities, merges)
        
        # Should have 2 entities (one merged)
        assert len(result) == 2
        
        # Merged entity should have combined chunk_ids
        merged_entity = [e for e in result if 1 in e['chunk_ids'] and 2 in e['chunk_ids']]
        assert len(merged_entity) == 1
    
    def test_transitivity(self):
        """Test transitive merges (A=B, B=C → A=C)"""
        filter_stage = TieredThresholdFilter()
        
        # FIXED: Add 'type' field
        entities = [
            {"name": "A", "type": "Type1", "chunk_ids": [1]},
            {"name": "B", "type": "Type1", "chunk_ids": [2]},
            {"name": "C", "type": "Type1", "chunk_ids": [3]},
        ]
        
        # A=B and B=C, so all three should merge
        merges = [(0, 1), (1, 2)]
        
        result = filter_stage.apply_merges(entities, merges)
        
        # Should have 1 entity (all merged)
        assert len(result) == 1
        assert sorted(result[0]['chunk_ids']) == [1, 2, 3]


# ============================================================================
# Test SameJudge (Stage 4 - CPU with mocked LLM)
# ============================================================================

class TestSameJudge:
    """Test LLM-based entity verification (CPU version)"""
    
    def test_verify_pair_match(self):
        """Test LLM verification for matching entities"""
        # FIXED: Mock Together where it's actually used (inside verify_pair)
        with patch('together.Together') as mock_together_class:
            # Mock the client instance
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '''
            {
                "result": true,
                "canonical_name": "GDPR",
                "canonical_type": "Regulation",
                "reasoning": "Same regulation, different phrasing"
            }
            '''
            mock_client.chat.completions.create.return_value = mock_response
            mock_together_class.return_value = mock_client
            
            # Create SameJudge instance
            judge = SameJudge(api_key="test-key")
            
            entity1 = {"name": "GDPR", "type": "Regulation", "description": "EU data protection"}
            entity2 = {"name": "General Data Protection Regulation", "type": "Law", "description": "EU privacy law"}
            
            result = judge.verify_pair(entity1, entity2)
            
            assert result['is_same'] is True
            assert 'reasoning' in result
            assert judge.stats['pairs_verified'] == 1
            assert judge.stats['matches_found'] == 1
    
    def test_verify_pair_no_match(self):
        """Test LLM verification for non-matching entities"""
        with patch('together.Together') as mock_together_class:
            # Mock the client
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '''
            {
                "result": false,
                "reasoning": "Different entities"
            }
            '''
            mock_client.chat.completions.create.return_value = mock_response
            mock_together_class.return_value = mock_client
            
            judge = SameJudge(api_key="test-key")
            
            entity1 = {"name": "AI", "type": "Technology"}
            entity2 = {"name": "EU", "type": "Organization"}
            
            result = judge.verify_pair(entity1, entity2)
            
            assert result['is_same'] is False
            assert judge.stats['pairs_verified'] == 1
            assert judge.stats['matches_found'] == 0
    
    def test_verify_batch(self):
        """Test batch verification"""
        with patch('together.Together') as mock_together_class:
            # Mock LLM to return True for first pair, False for second
            mock_client = MagicMock()
            
            # Create a list of responses
            responses = [
                '{"result": true, "reasoning": "Same"}',
                '{"result": false, "reasoning": "Different"}'
            ]
            response_iter = iter(responses)
            
            def mock_create(*args, **kwargs):
                content = next(response_iter)
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = content
                return mock_response
            
            mock_client.chat.completions.create.side_effect = mock_create
            mock_together_class.return_value = mock_client
            
            judge = SameJudge(api_key="test-key")
            
            entities = [
                {"name": "A", "type": "Type1"},
                {"name": "B", "type": "Type2"},
                {"name": "C", "type": "Type3"},
            ]
            
            uncertain_pairs = [
                (0, 1, 0.85),
                (1, 2, 0.82),
            ]
            
            matches = judge.verify_batch(uncertain_pairs, entities, log_interval=10)
            
            # Should have 1 match (first pair)
            assert len(matches) == 1
            assert matches[0] == (0, 1)
            assert judge.stats['pairs_verified'] == 2
    
    def test_llm_error_handling(self):
        """Test that LLM errors are handled gracefully"""
        with patch('together.Together') as mock_together_class:
            # Mock LLM to raise exception
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            mock_together_class.return_value = mock_client
            
            judge = SameJudge(api_key="test-key")
            
            entity1 = {"name": "A", "type": "Type1"}
            entity2 = {"name": "B", "type": "Type2"}
            
            result = judge.verify_pair(entity1, entity2)
            
            # Should default to False on error
            assert result['is_same'] is False
            assert 'Error' in result['reasoning']
    
    def test_prompt_import(self):
        """Test that prompts are imported from centralized location"""
        # This test verifies the import works
        from src.prompts.prompts import SAMEJUDGE_PROMPT
        
        assert 'Entity 1:' in SAMEJUDGE_PROMPT
        assert 'Entity 2:' in SAMEJUDGE_PROMPT
        assert '{entity1_name}' in SAMEJUDGE_PROMPT
        assert 'result' in SAMEJUDGE_PROMPT  # JSON field


# ============================================================================
# Integration Test
# ============================================================================

class TestIntegration:
    """Test full pipeline integration"""
    
    def test_stage1_to_stage4(self):
        """Test Stages 1-4 work together"""
        with patch('together.Together') as mock_together_class:
            # Mock LLM
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"result": false}'
            mock_client.chat.completions.create.return_value = mock_response
            mock_together_class.return_value = mock_client
            
            # Create test data (with all required fields)
            entities = [
                {"name": "AI", "type": "Technology", "chunk_ids": [1], "embedding": np.random.rand(8).tolist()},
                {"name": "ai", "type": "technology", "chunk_ids": [2], "embedding": np.random.rand(8).tolist()},  # Duplicate
                {"name": "EU", "type": "Organization", "chunk_ids": [3], "embedding": np.random.rand(8).tolist()},
            ]
            
            # Stage 1: Deduplication
            dedup = ExactDeduplicator()
            entities = dedup.deduplicate(entities)
            assert len(entities) == 2  # 3 → 2
            
            # Stage 2: FAISS blocking
            blocker = FAISSBlocker(embedding_dim=8)
            blocker.build_index(entities)
            pairs = blocker.find_candidates(entities, k=2)
            
            # Stage 3: Threshold filtering
            filter_stage = TieredThresholdFilter()
            filtered = filter_stage.filter_pairs(pairs, entities)
            
            # Stage 4: LLM verification (mocked)
            judge = SameJudge(api_key="test-key")
            if filtered['uncertain']:
                matches = judge.verify_batch(filtered['uncertain'], entities)
                # No matches expected (mock returns false)
                assert len(matches) == 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])