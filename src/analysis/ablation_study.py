#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified ablation study with comprehensive evaluation metrics.

Compares semantic, graph, and dual retrieval modes with RAGAS metrics.

Example:
    python src/analysis/ablation_study.py
    python src/analysis/ablation_study.py --quick
    python src/analysis/ablation_study.py --no-ragas
"""

# Standard library
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# Project root (src/analysis/ablation_study.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
import anthropic

# Config imports (direct)
from config.retrieval_config import RetrievalMode

# Local
from src.retrieval.retrieval_processor import RetrievalProcessor
from src.retrieval.answer_generator import AnswerGenerator
from src.utils.embedder import BGEEmbedder
from src.utils.logger import get_logger

# Analysis metrics
from src.analysis.retrieval_metrics import (
    TestResult,
    EntityResolutionMetrics,
    GraphUtilizationMetrics,
    RetrievalMetrics,
    RAGASMetrics,
    PerformanceMetrics,
    CoverageMetrics,
    compute_entity_resolution_metrics,
    compute_graph_utilization_metrics,
    compute_retrieval_metrics,
    compute_coverage_metrics
)

logger = get_logger(__name__)


# ============================================================================
# RAGAS METRICS (INTEGRATED)
# ============================================================================

class RAGASEvaluator:
    """RAGAS metrics evaluator using Claude as LLM judge."""
    
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.model = "claude-sonnet-4-20250514"
    
    def faithfulness(self, answer: str, context: str) -> Dict:
        """Measure if all claims in answer are supported by context."""
        prompt = f"""Evaluate faithfulness: are all claims in ANSWER supported by CONTEXT?

CONTEXT:
{context}

ANSWER:
{answer}

Extract claims from ANSWER. For each, check if supported by CONTEXT.
Output JSON only (no markdown):
{{
    "claims": [
        {{"claim": "...", "supported": true, "evidence": "..."}},
        ...
    ],
    "score": 0.XX,
    "explanation": "summary"
}}

Be strict: only mark supported if clearly inferable."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = response.content[0].text.strip()
            
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            
            return {
                'score': float(result['score']),
                'supported': sum(1 for c in result['claims'] if c['supported']),
                'total': len(result['claims']),
                'explanation': result['explanation']
            }
        except Exception as e:
            logger.error(f"Faithfulness evaluation failed: {e}")
            return {'score': 0.0, 'supported': 0, 'total': 0, 'explanation': f'Error: {e}'}
    
    def answer_relevancy(self, query: str, answer: str) -> Dict:
        """Measure how well answer addresses query."""
        prompt = f"""Rate how well ANSWER addresses QUERY.

QUERY: {query}
ANSWER: {answer}

Scoring:
- 1.0: Fully addresses all aspects
- 0.8-0.9: Addresses main question
- 0.6-0.7: Partial, significant gaps
- 0.4-0.5: Tangentially related
- 0.0-0.3: Irrelevant

Output JSON only:
{{
    "score": 0.XX,
    "explanation": "what's covered/missing?"
}}"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = response.content[0].text.strip()
            
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            
            return {
                'score': float(result['score']),
                'explanation': result['explanation']
            }
        except Exception as e:
            logger.error(f"Relevancy evaluation failed: {e}")
            return {'score': 0.0, 'explanation': f'Error: {e}'}


# ============================================================================
# TEST QUERIES
# ============================================================================

TEST_QUERIES = [
    {
        'id': 'q1',
        'query': 'What is the EU AI Act?',
        'category': 'simple_factual',
        'description': 'Major regulation entity'
    },
    {
        'id': 'q2',
        'query': 'What are high-risk AI systems?',
        'category': 'simple_factual',
        'description': 'Core concept in AI governance'
    },
    {
        'id': 'q3',
        'query': 'Which jurisdictions regulate facial recognition?',
        'category': 'cross_jurisdictional',
        'description': 'Multi-jurisdiction entity traversal'
    },
    {
        'id': 'q4',
        'query': 'Compare China and US AI governance',
        'category': 'cross_jurisdictional_comparison',
        'description': 'Cross-jurisdiction comparison'
    },
    {
        'id': 'q5',
        'query': 'What academic research discusses algorithmic bias?',
        'category': 'multi_hop_research',
        'description': 'Research papers to concept'
    },
    {
        'id': 'q6',
        'query': "What is Snoopy's arch enemy?",
        'category': 'out_of_domain',
        'description': 'Should fail gracefully'
    }
]


# ============================================================================
# INTEGRATED TEST SUITE
# ============================================================================

class AblationTestSuite:
    """Unified test suite for retrieval mode ablation."""
    
    def __init__(self, enable_ragas: bool = True):
        self.enable_ragas = enable_ragas
        self.processor = None
        self.generator = None
        self.ragas = None
        self.results = []
    
    def load_pipeline(self):
        """Load all pipeline components."""
        print("Loading pipeline components...")
        
        data_dir = PROJECT_ROOT / 'data'
        
        # Embedding model
        embedding_model = BGEEmbedder()
        
        # Retrieval processor
        self.processor = RetrievalProcessor(
            embedding_model=embedding_model,
            faiss_entity_index_path=data_dir / 'processed' / 'faiss' / 'entity_embeddings.index',
            entity_ids_path=data_dir / 'processed' / 'faiss' / 'entity_id_map.json',
            normalized_entities_path=data_dir / 'processed' / 'entities' / 'entities_semantic_embedded.jsonl',
            aliases_path=data_dir / 'processed' / 'entities' / 'aliases.json',
            faiss_chunk_index_path=data_dir / 'processed' / 'faiss' / 'chunk_embeddings.index',
            chunk_ids_path=data_dir / 'processed' / 'faiss' / 'chunk_id_map.json',
            neo4j_uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            neo4j_user=os.getenv('NEO4J_USER', 'neo4j'),
            neo4j_password=os.getenv('NEO4J_PASSWORD')
        )
        
        # Answer generator
        self.generator = AnswerGenerator()
        
        # RAGAS evaluator
        if self.enable_ragas:
            print("Initializing RAGAS evaluator...")
            self.ragas = RAGASEvaluator()
        
        print("Pipeline loaded\n")
    
    def run_single_test(self, query_def: Dict, mode: RetrievalMode, test_num: int, total_tests: int) -> TestResult:
        """Run a single test with comprehensive metrics collection."""
        
        print(f"\n{'='*80}")
        print(f"TEST {test_num}/{total_tests}: {mode.value.upper()}")
        print(f"{'='*80}")
        print(f"Query: {query_def['query']}")
        print(f"Category: {query_def['category']}")
        print()
        
        try:
            start_time = time.time()
            
            # 1. RETRIEVAL
            print("1. RETRIEVAL")
            retrieval_start = time.time()
            
            retrieval_result = self.processor.retrieve(query_def['query'], mode=mode)
            
            retrieval_time = time.time() - retrieval_start
            
            # Compute entity resolution metrics
            extracted = retrieval_result.parsed_query.extracted_entities if retrieval_result.parsed_query else []
            resolved = retrieval_result.resolved_entities if retrieval_result.resolved_entities else []
            
            entity_metrics = compute_entity_resolution_metrics(extracted, resolved)
            
            print(f"   Extracted: {entity_metrics.extracted_count} entities")
            print(f"   Resolved: {entity_metrics.resolved_count} ({entity_metrics.resolution_rate:.1%})")
            
            # Compute graph utilization metrics
            graph_metrics = compute_graph_utilization_metrics(retrieval_result.subgraph, retrieval_result.chunks)
            
            print(f"   Subgraph: {graph_metrics.entities_in_subgraph} entities, "
                  f"{graph_metrics.relations_in_subgraph} relations")
            
            # Compute retrieval metrics
            query_emb = retrieval_result.parsed_query.embedding if retrieval_result.parsed_query else None
            retrieval_metrics = compute_retrieval_metrics(
                retrieval_result.chunks,
                query_emb,
                self.processor.chunk_retriever
            )
            
            print(f"   Retrieved: {retrieval_metrics.total_chunks} chunks")
            print(f"   Sources: {retrieval_metrics.chunks_by_source}")
            
            # 2. ANSWER GENERATION
            print("\n2. ANSWER GENERATION")
            answer_start = time.time()
            
            answer_result = self.generator.generate(retrieval_result)
            
            answer_time = time.time() - answer_start
            
            print(f"   Tokens: {answer_result.output_tokens}")
            print(f"   Cost: ${answer_result.cost_usd:.4f}")
            print(f"   Answer preview: {answer_result.answer[:150]}...")
            
            # Compute coverage metrics
            coverage_metrics = compute_coverage_metrics(
                retrieval_result.subgraph,
                answer_result.answer,
                resolved
            )
            
            print(f"   Coverage: {coverage_metrics.entity_coverage_rate:.1%} entities")
            
            # 3. RAGAS EVALUATION
            if self.ragas:
                print("\n3. RAGAS EVALUATION")
                
                context_text = "\n\n".join([chunk.text for chunk in retrieval_result.chunks[:10]])
                
                print("   Evaluating faithfulness...")
                faith = self.ragas.faithfulness(answer_result.answer, context_text)
                
                time.sleep(2)  # Rate limit protection
                
                print("   Evaluating relevancy...")
                rel = self.ragas.answer_relevancy(query_def['query'], answer_result.answer)
                
                print(f"   Faithfulness: {faith['score']:.3f} ({faith['supported']}/{faith['total']} claims)")
                print(f"   Relevancy: {rel['score']:.3f}")
                
                ragas_metrics = RAGASMetrics(
                    faithfulness_score=faith['score'],
                    faithfulness_details={
                        'supported_claims': faith['supported'],
                        'total_claims': faith['total'],
                        'explanation': faith['explanation']
                    },
                    relevancy_score=rel['score'],
                    relevancy_explanation=rel['explanation']
                )
            else:
                ragas_metrics = RAGASMetrics(
                    faithfulness_score=0.0,
                    faithfulness_details={},
                    relevancy_score=0.0,
                    relevancy_explanation="RAGAS disabled"
                )
            
            # 4. PACKAGE RESULT
            total_time = time.time() - start_time
            
            performance_metrics = PerformanceMetrics(
                retrieval_time=retrieval_time,
                answer_time=answer_time,
                total_time=total_time,
                answer_tokens=answer_result.output_tokens,
                cost_usd=answer_result.cost_usd
            )
            
            test_result = TestResult(
                test_id=f"{query_def['id']}_{mode.value}",
                query=query_def['query'],
                mode=mode.value,
                category=query_def['category'],
                timestamp=datetime.now().isoformat(),
                entity_resolution=entity_metrics,
                graph_utilization=graph_metrics,
                coverage=coverage_metrics,
                retrieval=retrieval_metrics,
                ragas=ragas_metrics,
                performance=performance_metrics,
                answer_text=answer_result.answer,
                success=True,
                error=None
            )
            
            print(f"\nTest completed in {total_time:.1f}s")
            
            return test_result
            
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            print(f"\nTEST FAILED: {e}")
            
            return TestResult(
                test_id=f"{query_def['id']}_{mode.value}",
                query=query_def['query'],
                mode=mode.value,
                category=query_def['category'],
                timestamp=datetime.now().isoformat(),
                entity_resolution=EntityResolutionMetrics(0, 0, 0.0, 0.0, [], {}),
                graph_utilization=GraphUtilizationMetrics(0, 0, {}, []),
                coverage=CoverageMetrics(0, 0, 0.0, 0, 0, 0.0, [], []),
                retrieval=RetrievalMetrics(0, {}, 0.0, 0.0, {}, []),
                ragas=RAGASMetrics(0.0, {}, 0.0, ""),
                performance=PerformanceMetrics(0.0, 0.0, 0.0, 0, 0.0),
                answer_text="",
                success=False,
                error=str(e)
            )
    
    def run_full_suite(self, queries: List[Dict], modes: List[RetrievalMode]):
        """Run complete ablation study."""
        
        print(f"\n{'='*80}")
        print("GRAPHRAG RETRIEVAL ABLATION STUDY v1.1")
        print(f"{'='*80}\n")
        
        print(f"Configuration:")
        print(f"  Queries: {len(queries)}")
        print(f"  Modes: {[m.value for m in modes]}")
        print(f"  Total tests: {len(queries) * len(modes)}")
        print(f"  RAGAS evaluation: {'ENABLED' if self.enable_ragas else 'DISABLED'}")
        print()
        
        test_num = 0
        total_tests = len(queries) * len(modes)
        
        for query_def in queries:
            for mode in modes:
                test_num += 1
                result = self.run_single_test(query_def, mode, test_num, total_tests)
                self.results.append(result)
        
        return self.results
    
    def analyze_results(self):
        """Generate analysis summary."""
        
        print(f"\n\n{'='*80}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*80}\n")
        
        successful_tests = [r for r in self.results if r.success]
        
        print(f"Tests completed: {len(successful_tests)}/{len(self.results)}\n")
        
        for mode in ['semantic', 'graph', 'dual']:
            mode_results = [r for r in successful_tests if r.mode == mode]
            if mode_results:
                avg_chunks = sum(r.retrieval.total_chunks for r in mode_results) / len(mode_results)
                avg_entities = sum(r.graph_utilization.entities_in_subgraph for r in mode_results) / len(mode_results)
                
                print(f"{mode.upper():10} -> {avg_chunks:.1f} chunks, {avg_entities:.1f} entities")
                
                if self.enable_ragas:
                    avg_faith = sum(r.ragas.faithfulness_score for r in mode_results) / len(mode_results)
                    avg_rel = sum(r.ragas.relevancy_score for r in mode_results) / len(mode_results)
                    print(f"           Faithfulness: {avg_faith:.3f}, Relevancy: {avg_rel:.3f}")
    
    def save_results(self, output_dir: Path):
        """Save results to JSON and markdown."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Convert to JSON-serializable format
        results_dicts = []
        for r in self.results:
            result_dict = {
                'test_id': r.test_id,
                'query': r.query,
                'mode': r.mode,
                'category': r.category,
                'success': r.success,
                'error': r.error,
                'entity_resolution': {
                    'extracted_count': r.entity_resolution.extracted_count,
                    'resolved_count': r.entity_resolution.resolved_count,
                    'resolution_rate': r.entity_resolution.resolution_rate,
                    'match_types': r.entity_resolution.match_types
                },
                'graph_utilization': {
                    'entities_in_subgraph': r.graph_utilization.entities_in_subgraph,
                    'relations_in_subgraph': r.graph_utilization.relations_in_subgraph,
                    'relation_types': r.graph_utilization.relation_types,
                    'jurisdictions_covered': r.graph_utilization.jurisdictions_covered
                },
                'retrieval': {
                    'total_chunks': r.retrieval.total_chunks,
                    'chunks_by_source': r.retrieval.chunks_by_source,
                    'avg_chunk_score': r.retrieval.avg_chunk_score,
                    'avg_query_similarity': r.retrieval.avg_query_similarity,
                    'source_diversity': r.retrieval.source_diversity,
                    'jurisdiction_diversity': r.retrieval.jurisdiction_diversity
                },
                'coverage': {
                    'entities_in_subgraph': r.coverage.entities_in_subgraph,
                    'entities_in_answer': r.coverage.entities_in_answer,
                    'entity_coverage_rate': r.coverage.entity_coverage_rate,
                    'relations_in_subgraph': r.coverage.relations_in_subgraph,
                    'relations_mentioned': r.coverage.relations_mentioned,
                    'relation_coverage_rate': r.coverage.relation_coverage_rate
                },
                'ragas': {
                    'faithfulness_score': r.ragas.faithfulness_score,
                    'relevancy_score': r.ragas.relevancy_score
                },
                'performance': {
                    'total_time': r.performance.total_time,
                    'cost_usd': r.performance.cost_usd
                }
            }
            results_dicts.append(result_dict)
        
        json_file = output_dir / f'ablation_results_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(results_dicts, f, indent=2, default=str)
        
        print(f"\nResults saved to: {json_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Retrieval ablation study')
    parser.add_argument('--quick', action='store_true', help='Quick test with 2 queries')
    parser.add_argument('--queries', type=int, help='Number of queries')
    parser.add_argument('--no-ragas', action='store_true', help='Skip RAGAS')
    parser.add_argument('-o', '--output', type=str, default='data/analysis/results', help='Output dir')
    
    args = parser.parse_args()
    
    try:
        if args.quick:
            queries = TEST_QUERIES[:2]
        elif args.queries:
            queries = TEST_QUERIES[:args.queries]
        else:
            queries = TEST_QUERIES
        
        modes = [RetrievalMode.SEMANTIC, RetrievalMode.GRAPH, RetrievalMode.DUAL]
        
        suite = AblationTestSuite(enable_ragas=not args.no_ragas)
        suite.load_pipeline()
        suite.run_full_suite(queries, modes)
        suite.analyze_results()
        suite.save_results(Path(args.output))
        
        print("\nAblation study complete!\n")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Ablation study failed: %s", str(e), exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()