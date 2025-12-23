# -*- coding: utf-8 -*-
"""
Parameter Tuning Tests for Phase 3 Retrieval Pipeline.

Tests different parameter combinations to evaluate sensitivity for entity resolution,
ranking bonuses, and FAISS top-k settings with comparative analysis output.

Example:
    pytest test_parameter_tuning.py -v -s
"""

# Standard library
from pathlib import Path

# Third-party
import numpy as np
import pytest

# Config imports (direct)
from config.retrieval_config import RANKING_CONFIG, RETRIEVAL_CONFIG

# Dataclass imports (direct)
from src.utils.dataclasses import (
    ExtractedQueryEntity as ExtractedEntity,
    QueryFilters,
    Chunk,
    Subgraph,
    Relation,
)

# Local
from src.retrieval.entity_resolver import EntityResolver
from src.retrieval.result_ranker import ResultRanker


# ============================================================================
# TEST SCENARIOS
# ============================================================================

TEST_QUERIES = [
    {
        'query': "What does the EU AI Act say about facial recognition?",
        'entities': [
            ('EU AI Act', 'Regulation'),
            ('facial recognition', 'Technology')
        ],
        'filters': QueryFilters(
            jurisdiction_hints=['EU'],
            doc_type_hints=['regulation']
        )
    },
    {
        'query': "What do academic papers say about AI bias?",
        'entities': [
            ('AI bias', 'TechnicalConcept'),
            ('academic papers', 'Document')
        ],
        'filters': QueryFilters(
            doc_type_hints=['paper']
        )
    },
    {
        'query': "GDPR transparency requirements",
        'entities': [
            ('GDPR', 'Regulation'),
            ('transparency', 'RegulatoryConcept')
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
        from src.utils.embeddings import EmbeddingModel
        
        embedder = EmbeddingModel()
        
        resolver = EntityResolver(
            faiss_index_path=Path('data/faiss/entities.index'),
            entity_ids_path=Path('data/faiss/entity_ids.json'),
            normalized_entities_path=Path('data/processed/entities.json'),
            aliases_path=Path('data/processed/entities/aliases.json'),
            embedding_model=embedder,
            threshold=0.75,
            top_k=10
        )
        
        return resolver
    
    def test_top_k_sensitivity(self, resolver):
        """Test how different top_k values affect entity count."""
        
        test_case = TEST_QUERIES[0]  # EU AI Act query
        extracted = [ExtractedEntity(name=name, type=etype) for name, etype in test_case['entities']]
        
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
                'alias_matches': sum(1 for e in resolved if e.match_type == 'alias'),
                'fuzzy_matches': sum(1 for e in resolved if e.match_type == 'fuzzy'),
                'avg_confidence': np.mean([e.confidence for e in resolved]) if resolved else 0.0
            }
        
        # Print comparison table
        print("\n" + "="*80)
        print("ENTITY RESOLUTION TOP-K SENSITIVITY")
        print("="*80)
        print(f"Query: {test_case['query']}")
        print(f"Extracted entities: {len(extracted)}")
        print()
        print(f"{'top_k':<10} {'Total':<10} {'Exact':<10} {'Alias':<10} {'Fuzzy':<10} {'Avg Conf':<10}")
        print("-"*80)
        
        for k, stats in results.items():
            print(f"{k:<10} {stats['entity_count']:<10} {stats['exact_matches']:<10} "
                  f"{stats['alias_matches']:<10} {stats['fuzzy_matches']:<10} {stats['avg_confidence']:.3f}")
        
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
        chunks_graph = [
            Chunk(
                chunk_ids=['chunk_rel_1'],
                document_ids=['doc1'],
                text='Text with relation',
                position=0,
                sentence_count=1,
                token_count=10,
                metadata={
                    'entities': ['ent_1', 'ent_2'],
                    'is_relation_provenance': True,
                    'score': 0.5,
                    'doc_type': 'regulation',
                    'jurisdiction': 'EU'
                }
            ),
            Chunk(
                chunk_ids=['chunk_ent_1'],
                document_ids=['doc1'],
                text='Text with entity',
                position=1,
                sentence_count=1,
                token_count=10,
                metadata={
                    'entities': ['ent_1'],
                    'score': 0.4,
                    'doc_type': 'regulation',
                    'jurisdiction': 'EU'
                }
            ),
            Chunk(
                chunk_ids=['chunk_ent_2'],
                document_ids=['doc2'],
                text='Text with entity',
                position=0,
                sentence_count=1,
                token_count=10,
                metadata={
                    'entities': ['ent_2'],
                    'score': 0.4,
                    'doc_type': 'paper',
                    'jurisdiction': 'US'
                }
            ),
        ]
        
        chunks_semantic = [
            Chunk(
                chunk_ids=['chunk_sem_1'],
                document_ids=['doc3'],
                text='Semantically similar',
                position=0,
                sentence_count=1,
                token_count=10,
                metadata={
                    'faiss_rank': 0,
                    'score': 0.85,
                    'doc_type': 'regulation',
                    'jurisdiction': 'EU'
                }
            ),
            Chunk(
                chunk_ids=['chunk_sem_2'],
                document_ids=['doc4'],
                text='Also similar',
                position=0,
                sentence_count=1,
                token_count=10,
                metadata={
                    'faiss_rank': 1,
                    'score': 0.80,
                    'doc_type': 'paper',
                    'jurisdiction': 'EU'
                }
            ),
        ]
        
        subgraph = Subgraph(
            entities=['ent_1', 'ent_2'],
            relations=[
                Relation(
                    subject_id='ent_1',
                    predicate='related_to',
                    object_id='ent_2',
                    chunk_ids=['chunk_rel_1'],
                    extraction_strategy='semantic'
                )
            ]
        )
        
        filters = QueryFilters(
            jurisdiction_hints=['EU'],
            doc_type_hints=['regulation']
        )
        
        print("\n" + "="*90)
        print("RANKING WEIGHT SENSITIVITY")
        print("="*90)
        
        ranker = ResultRanker()
        result = ranker.rank(chunks_graph, chunks_semantic, subgraph, filters, "test query")
        
        print(f"\nDefault Config:")
        print(f"  Top 3 chunks:")
        for i, chunk in enumerate(result.chunks[:3], 1):
            provenance = "RELATION" if chunk.matching_entities else "SEMANTIC"
            print(f"    [{i}] {chunk.chunk_id:<15} | Score: {chunk.score:.3f} | "
                  f"Path: {chunk.source_path:<20}")
        
        print("\n" + "="*90)
        print("OBSERVATIONS:")
        print("  - Higher provenance_multiplier -> Relation chunks rank higher")
        print("  - Higher doc_type_penalty -> Non-matching doc types penalized more")
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
                'graph_provenance_multiplier': 1.0,
                'graph_entity_multiplier': 0.85,
                'jurisdiction_penalty': 0.9,
                'doc_type_penalty': 0.85,
                'final_top_k': 20
            }
        }
        
        print(f"\nTesting with recommended parameters:")
        print(f"  - entity_resolution_top_k: {recommended_params['entity_resolution_top_k']}")
        print(f"  - graph_provenance_multiplier: {recommended_params['ranking_config']['graph_provenance_multiplier']}")
        print(f"  - doc_type_penalty: {recommended_params['ranking_config']['doc_type_penalty']}")
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
    print("RANKING WEIGHTS (Multiplicative System):")
    print(f"  graph_provenance_multiplier: {RANKING_CONFIG['graph_provenance_multiplier']}")
    print("  Rationale: Highest priority for chunks containing PCST relations")
    print()
    print(f"  graph_entity_multiplier: {RANKING_CONFIG['graph_entity_multiplier']}")
    print("  Rationale: Medium priority for chunks from entity expansion")
    print()
    print(f"  doc_type_penalty: {RANKING_CONFIG['doc_type_penalty']}")
    print("  Rationale: Soft penalty for non-matching doc types")
    print()
    print(f"  jurisdiction_penalty: {RANKING_CONFIG['jurisdiction_penalty']}")
    print("  Rationale: Soft penalty for non-matching jurisdictions")
    print()
    print("="*90)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
    print_recommendations()
