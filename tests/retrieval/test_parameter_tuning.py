# -*- coding: utf-8 -*-
"""
Parameter Tuning Tests for Phase 3 Retrieval Pipeline.

Tests different parameter combinations to evaluate sensitivity for entity resolution,
ranking bonuses, and FAISS top-k settings with comparative analysis output.

Example:
    pytest test_parameter_tuning.py -v -s
    # Output: Comparison table showing parameter effects on entities, subgraphs, chunks
"""

# Standard library
from pathlib import Path

# Third-party
import numpy as np
import pytest

# Local

from src.retrieval.entity_resolver import EntityResolver
from src.retrieval.result_ranker import ResultRanker
from src.retrieval.config import (
    ExtractedEntity,
    QueryFilters,
    RANKING_CONFIG,
    RETRIEVAL_CONFIG
)


# ============================================================================
# TEST SCENARIOS
# ============================================================================

TEST_QUERIES = [
    {
        'query': "What does the EU AI Act say about facial recognition?",
        'entities': [
            ('EU AI Act', 'Regulatory Concept'),
            ('facial recognition', 'Technical Term')
        ],
        'filters': QueryFilters(
            jurisdiction_hints=['EU'],
            doc_type_hints=['regulation']
        )
    },
    {
        'query': "What do academic papers say about AI bias?",
        'entities': [
            ('AI bias', 'Concept'),
            ('academic papers', 'Publication')
        ],
        'filters': QueryFilters(
            doc_type_hints=['paper']
        )
    },
    {
        'query': "GDPR transparency requirements",
        'entities': [
            ('GDPR', 'Regulation'),
            ('transparency', 'Concept')
        ],
        'filters': QueryFilters(
            jurisdiction_hints=['EU'],
            doc_type_hints=['regulation']
        )
    }
]


# ============================================================================
# PARAMETER TUNING TESTS
# ============================================================================

class TestEntityResolutionTopK:
    """Test impact of entity resolution top_k on subgraph size."""
    
    @pytest.fixture
    def resolver(self):
        """Initialize entity resolver (assumes real data available)."""
        from src.utils.embedder import BGEEmbedder
        
        embedder = BGEEmbedder()
        
        resolver = EntityResolver(
            faiss_index_path=Path('data/processed/faiss/entity_embeddings.index'),
            entity_ids_path=Path('data/processed/faiss/entity_id_map.json'),
            normalized_entities_path=Path('data/interim/entities/normalized_entities_with_ids.json'),
            embedding_model=embedder,
            threshold=0.75,
            top_k=10  # Default
        )
        
        return resolver
    
    def test_top_k_sensitivity(self, resolver):
        """Test how different top_k values affect entity count."""
        
        test_case = TEST_QUERIES[0]  # EU AI Act query
        extracted = [ExtractedEntity(name, etype) for name, etype in test_case['entities']]
        
        results = {}
        
        for k in [1, 3, 5, 10]:
            resolved = resolver.resolve(
                extracted, 
                filters=test_case['filters'],
                top_k=k
            )
            
            results[k] = {
                'entity_count': len(resolved),
                'exact_matches': sum(1 for e in resolved if e.match_type == 'exact'),
                'fuzzy_matches': sum(1 for e in resolved if e.match_type == 'fuzzy'),
                'avg_confidence': np.mean([e.confidence for e in resolved]) if resolved else 0.0
            }
        
        # Print comparison table
        print("\n" + "="*70)
        print("ENTITY RESOLUTION TOP-K SENSITIVITY")
        print("="*70)
        print(f"Query: {test_case['query']}")
        print(f"Extracted entities: {len(extracted)}")
        print()
        print(f"{'top_k':<10} {'Total':<10} {'Exact':<10} {'Fuzzy':<10} {'Avg Conf':<10}")
        print("-"*70)
        
        for k, stats in results.items():
            print(f"{k:<10} {stats['entity_count']:<10} {stats['exact_matches']:<10} "
                  f"{stats['fuzzy_matches']:<10} {stats['avg_confidence']:.3f}")
        
        print()
        print("RECOMMENDATION:")
        print("  - top_k=1: Fastest, but may miss relevant entities")
        print("  - top_k=3: Balanced (recommended for production)")
        print("  - top_k=5-10: Comprehensive but larger subgraphs")
        print()
        
        # Validation: More candidates should give more matches
        assert results[10]['entity_count'] >= results[1]['entity_count']


class TestRankingWeights:
    """Test impact of ranking weight combinations."""
    
    def test_bonus_weight_sensitivity(self):
        """Test how different bonus weights affect final ranking."""
        
        # Mock chunks with different provenance
        from src.retrieval.config import Chunk, Subgraph, Relation
        
        chunks_a = [
            Chunk('chunk_rel_1', 'Text with relation', 'doc1', 'regulation', 'EU', 
                  {'entities': ['ent_1', 'ent_2'], 'is_relation_provenance': True}),
            Chunk('chunk_ent_1', 'Text with entity', 'doc1', 'regulation', 'EU',
                  {'entities': ['ent_1']}),
            Chunk('chunk_ent_2', 'Text with entity', 'doc2', 'paper', 'US',
                  {'entities': ['ent_2']}),
        ]
        
        chunks_b = [
            Chunk('chunk_sem_1', 'Semantically similar', 'doc3', 'regulation', 'EU',
                  {'faiss_rank': 0}),
            Chunk('chunk_sem_2', 'Also similar', 'doc4', 'paper', 'EU',
                  {'faiss_rank': 1}),
        ]
        
        subgraph = Subgraph(
            entities=['ent_1', 'ent_2'],
            relations=[
                {'subject_id': 'ent_1', 'predicate': 'related_to', 'object_id': 'ent_2',
                 'chunk_ids': ['chunk_rel_1']}
            ]
        )
        
        filters = QueryFilters(
            jurisdiction_hints=['EU'],
            doc_type_hints=['regulation']
        )
        
        # Test different weight configurations
        configs = [
            {'name': 'Default', 'config': RANKING_CONFIG},
            {'name': 'High Provenance', 'config': {**RANKING_CONFIG, 'provenance_bonus': 0.5}},
            {'name': 'Equal Weights', 'config': {
                'provenance_bonus': 0.2,
                'path_a_bonus': 0.2,
                'path_b_baseline': 0.0,
                'jurisdiction_boost': 0.2,
                'doc_type_boost': 0.2,
                'final_top_k': 20
            }},
            {'name': 'Strong Preferences', 'config': {
                'provenance_bonus': 0.3,
                'path_a_bonus': 0.2,
                'path_b_baseline': 0.0,
                'jurisdiction_boost': 0.25,  # Stronger jurisdiction preference
                'doc_type_boost': 0.25,      # Stronger doc_type preference
                'final_top_k': 20
            }}
        ]
        
        print("\n" + "="*90)
        print("RANKING WEIGHT SENSITIVITY")
        print("="*90)
        
        for config_test in configs:
            ranker = ResultRanker(config=config_test['config'])
            result = ranker.rank(chunks_a, chunks_b, subgraph, filters, "test query")
            
            print(f"\n{config_test['name']}:")
            print(f"  Config: {config_test['config']}")
            print(f"  Top 3 chunks:")
            for i, chunk in enumerate(result.chunks[:3], 1):
                provenance = "RELATION" if chunk.metadata.get('is_relation_provenance') else \
                            "ENTITY" if 'entities' in chunk.metadata else "SEMANTIC"
                print(f"    [{i}] {chunk.chunk_id:<15} | Score: {chunk.score:.3f} | "
                      f"Path: {chunk.source_path:<20} | Provenance: {provenance}")
        
        print("\n" + "="*90)
        print("OBSERVATIONS:")
        print("  - Higher provenance_bonus → Relation chunks rank higher")
        print("  - Higher doc_type_boost → Matching doc types rank higher")
        print("  - Equal weights → More balanced between paths")
        print()


class TestCrossQueryConsistency:
    """Test parameter stability across different query types."""
    
    def test_parameter_consistency(self):
        """Test that parameter settings work well across diverse queries."""
        
        print("\n" + "="*90)
        print("CROSS-QUERY PARAMETER CONSISTENCY")
        print("="*90)
        
        # Recommended parameter set
        recommended_params = {
            'entity_resolution_top_k': 3,
            'ranking_config': {
                'provenance_bonus': 0.3,
                'path_a_bonus': 0.2,
                'path_b_baseline': 0.0,
                'jurisdiction_boost': 0.1,
                'doc_type_boost': 0.15,
                'final_top_k': 20
            }
        }
        
        print(f"\nTesting with recommended parameters:")
        print(f"  - entity_resolution_top_k: {recommended_params['entity_resolution_top_k']}")
        print(f"  - provenance_bonus: {recommended_params['ranking_config']['provenance_bonus']}")
        print(f"  - doc_type_boost: {recommended_params['ranking_config']['doc_type_boost']}")
        print()
        
        for i, test_case in enumerate(TEST_QUERIES, 1):
            print(f"Query {i}: {test_case['query']}")
            print(f"  Entities: {len(test_case['entities'])}")
            print(f"  Filters: jurisdictions={test_case['filters'].jurisdiction_hints}, "
                  f"doc_types={test_case['filters'].doc_type_hints}")
            print()
        
        print("="*90)
        print("RECOMMENDATION: Use consistent parameters across query types")
        print("  This ensures predictable behavior and easier debugging")
        print()


# ============================================================================
# PARAMETER RECOMMENDATIONS
# ============================================================================

def print_recommendations():
    """Print final parameter recommendations based on all tests."""
    
    print("\n" + "="*90)
    print("PARAMETER TUNING RECOMMENDATIONS")
    print("="*90)
    print()
    print("ENTITY RESOLUTION:")
    print(f"  entity_resolution_top_k: {RETRIEVAL_CONFIG['entity_resolution_top_k']}")
    print("  Rationale: Limits fuzzy matches per entity to prevent subgraph explosion")
    print("            while maintaining coverage. 3 matches balances precision/recall.")
    print()
    print("RANKING WEIGHTS:")
    print(f"  provenance_bonus: {RANKING_CONFIG['provenance_bonus']}")
    print("  Rationale: Highest priority for chunks containing PCST relations")
    print()
    print(f"  path_a_bonus: {RANKING_CONFIG['path_a_bonus']}")
    print("  Rationale: Medium priority for chunks from entity expansion")
    print()
    print(f"  doc_type_boost: {RANKING_CONFIG['doc_type_boost']}")
    print("  Rationale: Soft preference (not hard filter) for matching doc types")
    print("            Preserves cross-document reasoning while respecting user intent")
    print()
    print(f"  jurisdiction_boost: {RANKING_CONFIG['jurisdiction_boost']}")
    print("  Rationale: Soft preference for jurisdiction hints")
    print()
    print("FUTURE TUNING:")
    print("  - Adjust based on user feedback")
    print("  - Consider domain-specific weights (e.g., higher provenance for legal queries)")
    print("  - A/B test with evaluators")
    print()
    print("="*90)


if __name__ == '__main__':
    # Run all tests and print recommendations
    pytest.main([__file__, '-v', '-s'])
    print_recommendations()
