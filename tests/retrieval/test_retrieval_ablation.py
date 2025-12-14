#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: test_retrieval_ablation_v1.py
Purpose: Unified ablation study with comprehensive evaluation metrics

Single test suite that:
1. Compares naive, graphrag, and dual retrieval modes
2. Evaluates with RAGAS metrics (faithfulness + relevancy)
3. Tracks comprehensive metrics aligned with thesis objectives
4. Generates detailed analysis reports

v1.0 Features:
- Complete entity resolution tracking
- Graph utilization metrics
- Retrieval effectiveness measurement
- RAGAS answer quality evaluation
- Performance/cost tracking

Usage:
    python tests/retrieval/test_retrieval_ablation_v1.py              # Full suite (18 tests)
    python tests/retrieval/test_retrieval_ablation_v1.py --quick      # Quick test (2 queries)
    python tests/retrieval/test_retrieval_ablation_v1.py --no-ragas   # Skip RAGAS (faster)

Author: Pau Barba i Colomer
Created: 2025-12-14
"""

import sys
import os
from pathlib import Path
import argparse
import json
import anthropic
import time
from datetime import datetime
from typing import List, Dict, Optional

# Project root (script in tests/retrieval/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.retrieval_processor import RetrievalProcessor
from src.retrieval.answer_generator import AnswerGenerator
from src.retrieval.config import RetrievalMode
from src.utils.embedder import BGEEmbedder
from src.utils.logger import get_logger

# Import test metrics
from src.retrieval.test_metrics import (
    TestResult,
    EntityResolutionMetrics,
    GraphUtilizationMetrics,
    RetrievalMetrics,
    RAGASMetrics,
    PerformanceMetrics,
    compute_entity_resolution_metrics,
    compute_graph_utilization_metrics,
    compute_retrieval_metrics
)

logger = get_logger(__name__)


# ============================================================================
# RAGAS METRICS (INTEGRATED)
# ============================================================================

class RAGASEvaluator:
    """
    RAGAS metrics evaluator using Claude as LLM judge.
    
    Implements reference-free metrics:
    - Faithfulness: Claims supported by context
    - Answer Relevancy: Answer addresses query
    """
    
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.model = "claude-sonnet-4-20250514"
    
    def faithfulness(self, answer: str, context: str) -> Dict:
        """
        Measure if all claims in answer are supported by context.
        
        Returns: {'score': float, 'supported': int, 'total': int, 'explanation': str}
        """
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
            
            # Extract JSON
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
        """
        Measure how well answer addresses query.
        
        Returns: {'score': float, 'explanation': str}
        """
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
    # SIMPLE FACTUAL - Big entities in corpus
    {
        'id': 'q1',
        'query': 'What is the EU AI Act?',
        'category': 'simple_factual',
        'description': 'Major regulation entity - should work well in all modes'
    },
    {
        'id': 'q2',
        'query': 'What are high-risk AI systems?',
        'category': 'simple_factual',
        'description': 'Core concept in AI governance'
    },
    
    # CROSS-JURISDICTIONAL - Should favor GraphRAG
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
        'description': 'Requires connecting entities across jurisdictions'
    },
    
    # MULTI-HOP RESEARCH - Academic entities → domain concepts
    {
        'id': 'q5',
        'query': 'What academic research discusses algorithmic bias?',
        'category': 'multi_hop_research',
        'description': 'Research papers → algorithmic bias concept'
    },
    
    # MOCK FAILURE CASE - Out of domain
    {
        'id': 'q6',
        'query': 'What is Snoopy\'s arch enemy?',
        'category': 'out_of_domain',
        'description': 'Should fail gracefully - tests error handling'
    }
]


# ============================================================================
# INTEGRATED TEST SUITE
# ============================================================================

class AblationTestSuite:
    """Unified test suite for retrieval mode ablation with comprehensive metrics."""
    
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
        faiss_dir = data_dir / 'processed' / 'faiss'
        interim_dir = data_dir / 'interim' / 'entities'
        
        # Embedding model
        embedding_model = BGEEmbedder()
        
        # Retrieval processor
        self.processor = RetrievalProcessor(
            embedding_model=embedding_model,
            faiss_entity_index_path=faiss_dir / 'entity_embeddings.index',
            entity_ids_path=faiss_dir / 'entity_id_map.json',
            normalized_entities_path=interim_dir / 'normalized_entities_with_ids.json',
            faiss_chunk_index_path=faiss_dir / 'chunk_embeddings.index',
            chunk_ids_path=faiss_dir / 'chunk_id_map.json',
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
        
        print("✓ Pipeline loaded\n")
    
    def run_single_test(self, query_def: Dict, mode: RetrievalMode, test_num: int, total_tests: int) -> TestResult:
        """Run a single test with comprehensive metrics collection."""
        
        print(f"\n{'='*80}")
        print(f"TEST {test_num}/{total_tests}: {mode.value.upper()}")
        print(f"{'='*80}")
        print(f"Query: {query_def['query']}")
        print(f"Category: {query_def['category']}")
        print()
        
        try:
            # Track overall timing
            start_time = time.time()
            
            # ====================
            # 1. RETRIEVAL
            # ====================
            print("1. RETRIEVAL")
            retrieval_start = time.time()
            
            retrieval_result = self.processor.retrieve(query_def['query'], mode=mode.value)
            
            retrieval_time = time.time() - retrieval_start
            
            # Compute entity resolution metrics
            entity_metrics = compute_entity_resolution_metrics(
                retrieval_result.extracted_entities,
                retrieval_result.resolved_entities
            )
            
            print(f"   Extracted: {entity_metrics.extracted_count} entities")
            print(f"   Resolved: {entity_metrics.resolved_count} entities ({entity_metrics.resolution_rate:.2%})")
            if entity_metrics.entity_names:
                print(f"   → {entity_metrics.entity_names[:5]}")
            
            # Compute graph utilization metrics
            graph_metrics = compute_graph_utilization_metrics(retrieval_result.subgraph)
            
            print(f"   Subgraph: {graph_metrics.entities_in_subgraph} entities, "
                  f"{graph_metrics.relations_in_subgraph} relations")
            
            # Compute retrieval metrics
            retrieval_metrics = compute_retrieval_metrics(retrieval_result.chunks)
            
            print(f"   Retrieved: {retrieval_metrics.total_chunks} chunks")
            print(f"   Sources: {retrieval_metrics.chunks_by_source}")
            
            # ====================
            # 2. ANSWER GENERATION
            # ====================
            print("\n2. ANSWER GENERATION")
            answer_start = time.time()
            
            answer_result = self.generator.generate(retrieval_result)
            
            answer_time = time.time() - answer_start
            
            print(f"   Tokens: {answer_result.output_tokens}")
            print(f"   Cost: ${answer_result.cost_usd:.4f}")
            print(f"   Answer preview: {answer_result.answer[:150]}...")
            
            # ====================
            # 3. RAGAS EVALUATION
            # ====================
            if self.ragas:
                print("\n3. RAGAS EVALUATION")
                
                # Combine top 10 chunks as context
                context_text = "\n\n".join([
                    chunk.text for chunk in retrieval_result.chunks[:10]
                ])
                
                # Faithfulness
                print("   Evaluating faithfulness...")
                faith = self.ragas.faithfulness(answer_result.answer, context_text)
                
                # Relevancy
                print("   Evaluating relevancy...")
                rel = self.ragas.answer_relevancy(query_def['query'], answer_result.answer)
                
                print(f"   → Faithfulness: {faith['score']:.3f} "
                      f"({faith['supported']}/{faith['total']} claims)")
                print(f"   → Relevancy: {rel['score']:.3f}")
                
                if faith['score'] < 0.7:
                    print(f"   ⚠️  LOW FAITHFULNESS: {faith['explanation']}")
                if rel['score'] < 0.6:
                    print(f"   ⚠️  LOW RELEVANCY: {rel['explanation']}")
                
                # Package RAGAS metrics
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
                # No RAGAS - create empty metrics
                ragas_metrics = RAGASMetrics(
                    faithfulness_score=0.0,
                    faithfulness_details={},
                    relevancy_score=0.0,
                    relevancy_explanation="RAGAS disabled"
                )
            
            # ====================
            # 4. PACKAGE COMPLETE TEST RESULT
            # ====================
            total_time = time.time() - start_time
            
            performance_metrics = PerformanceMetrics(
                retrieval_time=retrieval_time,
                answer_time=answer_time,
                total_time=total_time,
                answer_tokens=answer_result.output_tokens,
                cost_usd=answer_result.cost_usd
            )
            
            test_result = TestResult(
                # Metadata
                test_id=f"{query_def['id']}_{mode.value}",
                query=query_def['query'],
                mode=mode.value,
                category=query_def['category'],
                timestamp=datetime.now().isoformat(),
                
                # Comprehensive metrics
                entity_resolution=entity_metrics,
                graph_utilization=graph_metrics,
                retrieval=retrieval_metrics,
                ragas=ragas_metrics,
                performance=performance_metrics,
                
                # Raw data
                answer_text=answer_result.answer,
                success=True,
                error=None
            )
            
            print(f"\n✓ Test completed in {total_time:.1f}s")
            
            return test_result
            
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            print(f"\n❌ TEST FAILED: {e}")
            
            # Return failed test result with empty metrics
            return TestResult(
                test_id=f"{query_def['id']}_{mode.value}",
                query=query_def['query'],
                mode=mode.value,
                category=query_def['category'],
                timestamp=datetime.now().isoformat(),
                entity_resolution=EntityResolutionMetrics(0, 0, 0.0, 0.0, [], {}),
                graph_utilization=GraphUtilizationMetrics(0, 0, {}, []),
                retrieval=RetrievalMetrics(0, {}, 0.0, {}, []),
                ragas=RAGASMetrics(0.0, {}, 0.0, ""),
                performance=PerformanceMetrics(0.0, 0.0, 0.0, 0, 0.0),
                answer_text="",
                success=False,
                error=str(e)
            )
    
    def run_full_suite(self, queries: List[Dict], modes: List[RetrievalMode]):
        """Run complete ablation study."""
        
        print(f"\n{'='*80}")
        print("GRAPHRAG RETRIEVAL ABLATION STUDY v1.0")
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
        """Generate comprehensive analysis using rich metrics."""
        
        print(f"\n\n{'='*80}")
        print("COMPREHENSIVE ANALYSIS")
        print(f"{'='*80}\n")
        
        successful_tests = [r for r in self.results if r.success]
        
        print(f"Tests completed: {len(successful_tests)}/{len(self.results)}\n")
        
        # ====================
        # 1. ENTITY RESOLUTION ANALYSIS
        # ====================
        print("1. ENTITY RESOLUTION QUALITY")
        print("-" * 60)
        
        for mode in ['naive', 'graphrag', 'dual']:
            mode_results = [r for r in successful_tests if r.mode == mode]
            if mode_results:
                avg_extracted = sum(r.entity_resolution.extracted_count for r in mode_results) / len(mode_results)
                avg_resolved = sum(r.entity_resolution.resolved_count for r in mode_results) / len(mode_results)
                avg_resolution_rate = sum(r.entity_resolution.resolution_rate for r in mode_results) / len(mode_results)
                avg_confidence = sum(r.entity_resolution.avg_confidence for r in mode_results) / len(mode_results)
                
                print(f"{mode.upper():10} → Extracted: {avg_extracted:.1f}, Resolved: {avg_resolved:.1f} "
                      f"({avg_resolution_rate:.1%}), Confidence: {avg_confidence:.3f}")
        
        # ====================
        # 2. GRAPH UTILIZATION ANALYSIS
        # ====================
        print(f"\n2. GRAPH UTILIZATION")
        print("-" * 60)
        
        for mode in ['naive', 'graphrag', 'dual']:
            mode_results = [r for r in successful_tests if r.mode == mode]
            if mode_results:
                avg_entities = sum(r.graph_utilization.entities_in_subgraph for r in mode_results) / len(mode_results)
                avg_relations = sum(r.graph_utilization.relations_in_subgraph for r in mode_results) / len(mode_results)
                
                print(f"{mode.upper():10} → {avg_entities:.1f} entities, {avg_relations:.1f} relations")
        
        # ====================
        # 3. RETRIEVAL EFFECTIVENESS
        # ====================
        print(f"\n3. RETRIEVAL EFFECTIVENESS")
        print("-" * 60)
        
        for mode in ['naive', 'graphrag', 'dual']:
            mode_results = [r for r in successful_tests if r.mode == mode]
            if mode_results:
                avg_chunks = sum(r.retrieval.total_chunks for r in mode_results) / len(mode_results)
                avg_score = sum(r.retrieval.avg_chunk_score for r in mode_results) / len(mode_results)
                
                # Aggregate chunks by source
                all_sources = {}
                for r in mode_results:
                    for source, count in r.retrieval.chunks_by_source.items():
                        all_sources[source] = all_sources.get(source, 0) + count
                
                print(f"{mode.upper():10} → {avg_chunks:.1f} chunks, avg score: {avg_score:.3f}")
                if all_sources:
                    print(f"             Sources: {all_sources}")
        
        # ====================
        # 4. RAGAS ANSWER QUALITY
        # ====================
        if self.enable_ragas:
            print(f"\n4. RAGAS ANSWER QUALITY")
            print("-" * 60)
            
            for mode in ['naive', 'graphrag', 'dual']:
                mode_results = [r for r in successful_tests if r.mode == mode]
                if mode_results:
                    avg_faith = sum(r.ragas.faithfulness_score for r in mode_results) / len(mode_results)
                    avg_rel = sum(r.ragas.relevancy_score for r in mode_results) / len(mode_results)
                    
                    print(f"{mode.upper():10} → Faithfulness: {avg_faith:.3f}, Relevancy: {avg_rel:.3f}")
            
            # Flag low-scoring cases
            low_faith = [r for r in successful_tests if r.ragas.faithfulness_score < 0.7]
            low_rel = [r for r in successful_tests if r.ragas.relevancy_score < 0.6]
            
            if low_faith or low_rel:
                print(f"\n   Quality Concerns:")
                print(f"   - Low faithfulness (<0.7): {len(low_faith)} cases")
                print(f"   - Low relevancy (<0.6): {len(low_rel)} cases")
                
                if low_faith:
                    print(f"\n   Lowest faithfulness cases:")
                    for r in sorted(low_faith, key=lambda x: x.ragas.faithfulness_score)[:3]:
                        print(f"   - {r.query[:50]}... ({r.mode}): {r.ragas.faithfulness_score:.2f}")
        
        # ====================
        # 5. COST & EFFICIENCY
        # ====================
        print(f"\n5. COST & EFFICIENCY")
        print("-" * 60)
        
        for mode in ['naive', 'graphrag', 'dual']:
            mode_results = [r for r in successful_tests if r.mode == mode]
            if mode_results:
                avg_cost = sum(r.performance.cost_usd for r in mode_results) / len(mode_results)
                avg_time = sum(r.performance.total_time for r in mode_results) / len(mode_results)
                avg_retrieval_time = sum(r.performance.retrieval_time for r in mode_results) / len(mode_results)
                avg_answer_time = sum(r.performance.answer_time for r in mode_results) / len(mode_results)
                total_cost = sum(r.performance.cost_usd for r in mode_results)
                
                print(f"{mode.upper():10} → ${avg_cost:.4f}/query (${total_cost:.4f} total), "
                      f"{avg_time:.1f}s avg (ret: {avg_retrieval_time:.1f}s, ans: {avg_answer_time:.1f}s)")
        
        # ====================
        # 6. PER-CATEGORY BREAKDOWN
        # ====================
        print(f"\n6. PERFORMANCE BY QUERY CATEGORY")
        print("-" * 60)
        
        categories = set(r.category for r in successful_tests)
        for cat in sorted(categories):
            cat_results = [r for r in successful_tests if r.category == cat]
            print(f"\n{cat}:")
            
            for mode in ['naive', 'graphrag', 'dual']:
                mode_cat = [r for r in cat_results if r.mode == mode]
                if mode_cat:
                    avg_chunks = sum(r.retrieval.total_chunks for r in mode_cat) / len(mode_cat)
                    if self.enable_ragas:
                        avg_faith = sum(r.ragas.faithfulness_score for r in mode_cat) / len(mode_cat)
                        print(f"  {mode:10} → {avg_chunks:.1f} chunks, faithfulness: {avg_faith:.3f}")
                    else:
                        print(f"  {mode:10} → {avg_chunks:.1f} chunks")
    
    def save_results(self, output_dir: Path):
        """Save results with comprehensive metrics."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Convert TestResult objects to dicts for JSON serialization
        results_dicts = []
        for r in self.results:
            result_dict = {
                'test_id': r.test_id,
                'query': r.query,
                'mode': r.mode,
                'category': r.category,
                'timestamp': r.timestamp,
                'success': r.success,
                'error': r.error,
                'answer_text': r.answer_text,
                
                'entity_resolution': {
                    'extracted_count': r.entity_resolution.extracted_count,
                    'resolved_count': r.entity_resolution.resolved_count,
                    'resolution_rate': r.entity_resolution.resolution_rate,
                    'avg_confidence': r.entity_resolution.avg_confidence,
                    'entity_names': r.entity_resolution.entity_names,
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
                    'source_diversity': r.retrieval.source_diversity,
                    'jurisdiction_diversity': r.retrieval.jurisdiction_diversity
                },
                
                'ragas': {
                    'faithfulness_score': r.ragas.faithfulness_score,
                    'faithfulness_details': r.ragas.faithfulness_details,
                    'relevancy_score': r.ragas.relevancy_score,
                    'relevancy_explanation': r.ragas.relevancy_explanation
                },
                
                'performance': {
                    'retrieval_time': r.performance.retrieval_time,
                    'answer_time': r.performance.answer_time,
                    'total_time': r.performance.total_time,
                    'answer_tokens': r.performance.answer_tokens,
                    'cost_usd': r.performance.cost_usd
                }
            }
            results_dicts.append(result_dict)
        
        # Save JSON results
        json_file = output_dir / f'ablation_results_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(results_dicts, f, indent=2, default=str)
        
        # Generate markdown report
        report_file = output_dir / f'ablation_report_{timestamp}.md'
        self._generate_report(report_file, timestamp)
        
        print(f"\n\n{'='*80}")
        print(f"RESULTS SAVED")
        print(f"{'='*80}")
        print(f"JSON results: {json_file}")
        print(f"Report: {report_file}")
        print(f"{'='*80}\n")
    
    def _generate_report(self, output_file: Path, timestamp: str):
        """Generate comprehensive markdown summary report."""
        
        successful_tests = [r for r in self.results if r.success]
        
        report = f"""# GraphRAG Retrieval Ablation Study - v1.0 Results

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Tests:** {len(self.results)}  
**Successful:** {len(successful_tests)}  
**Modes Compared:** naive, graphrag, dual  

---

## Executive Summary

This ablation study compares three retrieval strategies across {len(set(r.query for r in self.results))} test queries:

- **NAIVE:** Vector similarity search only (baseline RAG)
- **GRAPHRAG:** Entity-centric graph traversal only
- **DUAL:** Hybrid approach combining both strategies

### Evaluation Metrics

**Comprehensive metrics aligned with thesis objectives:**
- Entity Resolution Quality (factual accuracy)
- Graph Utilization (effective source use)
- Retrieval Effectiveness (relevance)
- RAGAS Answer Quality (faithfulness + relevancy)
- Cost & Efficiency

---

## Key Findings

"""
        
        # Add summary statistics by mode
        by_mode = {}
        for mode in ['naive', 'graphrag', 'dual']:
            by_mode[mode] = [r for r in successful_tests if r.mode == mode]
        
        for mode, mode_results in by_mode.items():
            if not mode_results:
                continue
            
            avg_resolved = sum(r.entity_resolution.resolved_count for r in mode_results) / len(mode_results)
            avg_entities = sum(r.graph_utilization.entities_in_subgraph for r in mode_results) / len(mode_results)
            avg_relations = sum(r.graph_utilization.relations_in_subgraph for r in mode_results) / len(mode_results)
            avg_chunks = sum(r.retrieval.total_chunks for r in mode_results) / len(mode_results)
            
            report += f"""
### {mode.upper()} Mode

- Average entities resolved: {avg_resolved:.1f}
- Average subgraph size: {avg_entities:.1f} entities, {avg_relations:.1f} relations
- Average chunks retrieved: {avg_chunks:.1f}
"""
            
            if self.enable_ragas and mode_results:
                avg_faith = sum(r.ragas.faithfulness_score for r in mode_results) / len(mode_results)
                avg_rel = sum(r.ragas.relevancy_score for r in mode_results) / len(mode_results)
                
                report += f"""- Faithfulness: {avg_faith:.3f}
- Relevancy: {avg_rel:.3f}
"""
        
        # Test queries
        report += """
---

## Test Queries

"""
        unique_queries = {}
        for r in self.results:
            if r.query not in unique_queries:
                unique_queries[r.query] = r.category
        
        for i, (query, category) in enumerate(sorted(unique_queries.items()), 1):
            report += f"{i}. **{query}** ({category})\n"
        
        # Limitations
        report += """
---

## Methodology Notes

**v1.0 Capabilities:**
- ✓ Comprehensive entity resolution tracking
- ✓ Graph utilization metrics
- ✓ Retrieval effectiveness measurement
- ✓ RAGAS answer quality evaluation
- ✓ Performance/cost tracking

**v1.0 Limitations:**
- ⚠️ No ground truth annotations (planned for v1.1)
- ⚠️ No precision/recall metrics (require ground truth)
- ⚠️ Small sample size (6 queries, 18 tests)
- ⚠️ No statistical significance testing

**Future Work (v1.1):**
- Manual ground truth annotation for all test queries
- Precision@k, Recall@k, F1@k metrics
- Statistical significance testing (paired t-tests)
- Expanded test set (25+ queries)

---

## Conclusion

v1.0 demonstrates:
✓ Functional three-mode retrieval architecture
✓ Comprehensive evaluation framework aligned with thesis objectives
✓ Entity resolution and graph utilization tracking
✓ RAGAS metrics for answer quality assessment
✓ Foundation for rigorous v1.1 evaluation with ground truth

---

*Generated by test_retrieval_ablation_v1.py*
"""
        
        with open(output_file, 'w') as f:
            f.write(report)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified retrieval ablation study with comprehensive metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/retrieval/test_retrieval_ablation_v1.py              # Full suite (18 tests)
  python tests/retrieval/test_retrieval_ablation_v1.py --quick      # Quick test (2 queries)
  python tests/retrieval/test_retrieval_ablation_v1.py --no-ragas   # Skip RAGAS (faster)
  python tests/retrieval/test_retrieval_ablation_v1.py -o results/  # Custom output dir

Metrics Tracked:
  - Entity Resolution: extraction, resolution rate, confidence
  - Graph Utilization: entities, relations, connectivity
  - Retrieval: chunks by source, diversity, scores
  - RAGAS: faithfulness, answer relevancy
  - Performance: time, tokens, cost
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with 2 queries only')
    parser.add_argument('--queries', type=int,
                       help='Number of queries to test (default: all 6)')
    parser.add_argument('--no-ragas', action='store_true',
                       help='Skip RAGAS evaluation (faster but less informative)')
    parser.add_argument('-o', '--output', type=str, default='evaluation_v1.0',
                       help='Output directory (default: evaluation_v1.0)')
    
    args = parser.parse_args()
    
    try:
        # Determine queries to test
        if args.quick:
            queries = TEST_QUERIES[:2]
        elif args.queries:
            queries = TEST_QUERIES[:args.queries]
        else:
            queries = TEST_QUERIES
        
        # Modes to compare
        modes = [RetrievalMode.NAIVE, RetrievalMode.GRAPHRAG, RetrievalMode.DUAL]
        
        # Initialize test suite
        suite = AblationTestSuite(enable_ragas=not args.no_ragas)
        
        # Load pipeline
        suite.load_pipeline()
        
        # Run tests
        suite.run_full_suite(queries, modes)
        
        # Analyze
        suite.analyze_results()
        
        # Save results
        output_dir = Path(args.output)
        suite.save_results(output_dir)
        
        print("\n✓ Ablation study complete!\n")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Ablation study failed: %s", str(e), exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()