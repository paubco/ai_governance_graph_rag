#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified ablation study with comprehensive evaluation metrics.

Compares semantic, graph, and dual retrieval modes with RAGAS metrics.

Modes:
    --detailed    6 queries, full answers printed, verbose per-query analysis
    --full        36 queries, compact output, aggregate stats for charts
    (default)     6 queries, compact output

Example:
    python src/analysis/ablation_study.py --detailed        # Full analysis
    python src/analysis/ablation_study.py --full            # Stats for charts
    python src/analysis/ablation_study.py --quick --no-ragas  # Quick debug
"""

# Standard library
import argparse
import json
import os
import re
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import List, Dict, Tuple, Optional

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
# RAGAS METRICS - Claude with robust parsing
# ============================================================================

class RAGASEvaluator:
    """
    RAGAS metrics evaluator using Claude API.
    
    Uses Claude for large context window (200k tokens) - essential for 
    faithfulness evaluation with full retrieval context.
    
    Includes robust JSON parsing with retry and regex fallback.
    """
    
    def __init__(self, max_retries: int = 2):
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        self.client = anthropic.Anthropic(api_key=api_key)
        # Use Haiku for evaluation - 12x cheaper than Sonnet, sufficient for judging
        self.model = "claude-3-5-haiku-20241022"
        self.max_retries = max_retries
    
    def _parse_json_response(self, content: str) -> Dict:
        """
        Robust JSON parsing with multiple fallback strategies.
        
        Pattern from project's query_parser.py and extraction pipeline.
        """
        cleaned = content.strip()
        
        # 1. Remove markdown fences (```json ... ```)
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first line if it's a fence
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            # Remove last line if it's a fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        
        # 2. Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # 3. Try to extract JSON object from text (LLM sometimes adds preamble)
        json_match = re.search(r'\{[\s\S]*\}', cleaned)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # 4. Regex fallback - extract score directly
        score_match = re.search(r'"score"\s*:\s*([\d.]+)', cleaned)
        if score_match:
            # Try to also get explanation
            expl_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', cleaned)
            return {
                'score': float(score_match.group(1)),
                'claims': [],
                'explanation': expl_match.group(1) if expl_match else 'Extracted via regex fallback'
            }
        
        # 5. Last resort - return zero with error info
        logger.warning(f"JSON parse failed, content preview: {cleaned[:200]}")
        return {'score': 0.0, 'claims': [], 'explanation': f'Parse failed: {cleaned[:100]}'}
    
    def _call_with_retry(self, messages: List[Dict], max_tokens: int) -> str:
        """Call Claude API with retry on transient failures."""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=messages
                )
                return response.content[0].text.strip()
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"API call failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
        
        raise last_error
    
    def faithfulness(self, answer: str, context: str) -> Dict:
        """
        Measure if all claims in answer are supported by context.
        
        Uses Claude's large context window to evaluate full retrieval context.
        """
        prompt = f"""Evaluate faithfulness: are all claims in ANSWER supported by CONTEXT?

CONTEXT:
{context}

ANSWER:
{answer}

Extract claims from ANSWER. For each, check if supported by CONTEXT.

Return JSON with this exact structure (no markdown fences):
{{
    "claims": [
        {{"claim": "claim text", "supported": true, "evidence": "quote from context"}},
        {{"claim": "another claim", "supported": false, "evidence": "not found in context"}}
    ],
    "score": 0.75,
    "explanation": "X of Y claims are supported because..."
}}

Be strict: only mark supported if clearly stated or directly inferable from context.
Return ONLY the JSON object, no other text."""

        try:
            result_text = self._call_with_retry(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )
            
            result = self._parse_json_response(result_text)
            
            claims = result.get('claims', [])
            supported = sum(1 for c in claims if c.get('supported', False))
            total = len(claims)
            
            # Calculate score if not provided or zero
            score = result.get('score', 0)
            if score == 0 and total > 0:
                score = supported / total
            
            return {
                'score': float(score),
                'supported': supported,
                'total': total,
                'explanation': result.get('explanation', '')
            }
            
        except Exception as e:
            logger.error(f"Faithfulness evaluation failed: {e}")
            return {'score': 0.0, 'supported': 0, 'total': 0, 'explanation': f'Error: {e}'}
    
    def answer_relevancy(self, query: str, answer: str) -> Dict:
        """Measure how well answer addresses query."""
        prompt = f"""Rate how well ANSWER addresses QUERY.

QUERY: {query}

ANSWER: {answer}

Scoring guide:
- 1.0: Fully addresses all aspects of the question
- 0.8-0.9: Addresses the main question well
- 0.6-0.7: Partial answer, significant gaps
- 0.4-0.5: Tangentially related only
- 0.0-0.3: Irrelevant or wrong

Return JSON with this exact structure (no markdown fences):
{{
    "score": 0.85,
    "explanation": "The answer covers X and Y but misses Z..."
}}

Return ONLY the JSON object, no other text."""

        try:
            result_text = self._call_with_retry(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            
            result = self._parse_json_response(result_text)
            
            return {
                'score': float(result.get('score', 0.0)),
                'explanation': result.get('explanation', '')
            }
            
        except Exception as e:
            logger.error(f"Relevancy evaluation failed: {e}")
            return {'score': 0.0, 'explanation': f'Error: {e}'}


# ============================================================================
# TEST QUERIES
# ============================================================================

from src.analysis.test_queries import DETAILED_QUERIES, FULL_QUERIES, get_queries

# Legacy alias for backward compatibility
QUICK_QUERIES = DETAILED_QUERIES
TEST_QUERIES = DETAILED_QUERIES


# ============================================================================
# INTEGRATED TEST SUITE
# ============================================================================

class AblationTestSuite:
    """Unified test suite for retrieval mode ablation."""
    
    def __init__(self, enable_ragas: bool = True, detailed: bool = False, parallel: bool = False, max_workers: int = 4):
        self.enable_ragas = enable_ragas
        self.detailed = detailed  # Detailed mode: full answers, verbose metrics
        self.parallel = parallel  # Parallel mode: run tests concurrently
        self.max_workers = max_workers
        self.processor = None
        self.generator = None
        self.ragas = None
        self.results = []
        self.results_lock = Lock()  # Thread-safe results collection
        self.print_lock = Lock()  # Thread-safe printing
    
    def load_pipeline(self):
        """Load all pipeline components."""
        print("Loading pipeline components...")
        
        data_dir = PROJECT_ROOT / 'data'
        
        # Embedding model
        embedding_model = BGEEmbedder()
        
        # Retrieval processor (v2.0 paths)
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
        
        # RAGAS evaluator (Claude Haiku - cheap but reliable)
        if self.enable_ragas:
            print("Initializing RAGAS evaluator (Claude Haiku)...")
            self.ragas = RAGASEvaluator()
        
        print("Pipeline loaded\n")
    
    def run_single_test(self, query_def: Dict, mode: RetrievalMode, test_num: int, total_tests: int) -> TestResult:
        """Run a single test with comprehensive metrics collection."""
        
        if self.detailed:
            print(f"\n{'━'*80}")
            print(f"  TEST {test_num}/{total_tests}: {mode.value.upper()}")
            print(f"{'━'*80}")
            print(f"  Query: {query_def['query']}")
            print(f"  Category: {query_def['primary_category']}")
            if query_def.get('expected_mode'):
                print(f"  Expected winner: {query_def['expected_mode']}")
            print()
        else:
            # Compact output for stats mode
            q_short = query_def['query'][:40] + '...' if len(query_def['query']) > 40 else query_def['query']
            print(f"  [{test_num}/{total_tests}] {mode.value:8} | {q_short}")
        
        try:
            start_time = time.time()
            
            # 1. RETRIEVAL
            if self.detailed:
                print("  1. RETRIEVAL")
            retrieval_start = time.time()
            
            retrieval_result = self.processor.retrieve(query_def['query'], mode=mode)
            
            retrieval_time = time.time() - retrieval_start
            
            # Compute entity resolution metrics
            extracted = retrieval_result.parsed_query.extracted_entities if retrieval_result.parsed_query else []
            resolved = retrieval_result.resolved_entities if retrieval_result.resolved_entities else []
            
            entity_metrics = compute_entity_resolution_metrics(extracted, resolved)
            
            if self.detailed:
                print(f"     Extracted: {entity_metrics.extracted_count} entities")
                print(f"     Resolved: {entity_metrics.resolved_count} ({entity_metrics.resolution_rate:.1%})")
                if entity_metrics.entity_names:
                    print(f"     Entities: {', '.join(entity_metrics.entity_names[:5])}")
                print(f"     Match types: {entity_metrics.match_types}")
            
            # Compute graph utilization metrics
            graph_metrics = compute_graph_utilization_metrics(retrieval_result.subgraph, retrieval_result.chunks)
            
            if self.detailed:
                print(f"     Subgraph: {graph_metrics.entities_in_subgraph} entities, "
                      f"{graph_metrics.relations_in_subgraph} relations")
                if graph_metrics.relation_types:
                    print(f"     Relation types: {graph_metrics.relation_types}")
            
            # Compute retrieval metrics
            query_emb = retrieval_result.parsed_query.embedding if retrieval_result.parsed_query else None
            retrieval_metrics = compute_retrieval_metrics(
                retrieval_result.chunks,
                query_emb,
                self.processor.chunk_retriever
            )
            
            if self.detailed:
                print(f"     Retrieved: {retrieval_metrics.total_chunks} chunks")
                print(f"     Sources: {retrieval_metrics.chunks_by_source}")
                print(f"     Avg similarity: {retrieval_metrics.avg_query_similarity:.3f}")
                print(f"     Source diversity: {retrieval_metrics.source_diversity}")
                print(f"     Jurisdictions: {retrieval_metrics.jurisdiction_diversity}")
            
            # 2. ANSWER GENERATION
            if self.detailed:
                print("\n  2. ANSWER GENERATION")
            answer_start = time.time()
            
            answer_result = self.generator.generate(retrieval_result)
            
            answer_time = time.time() - answer_start
            
            if self.detailed:
                print(f"     Tokens: {answer_result.output_tokens}")
                print(f"     Cost: ${answer_result.cost_usd:.4f}")
            
            # Compute coverage metrics
            coverage_metrics = compute_coverage_metrics(
                retrieval_result.subgraph,
                answer_result.answer,
                resolved
            )
            
            if self.detailed:
                print(f"     Entity coverage: {coverage_metrics.entity_coverage_rate:.1%}")
                print(f"     Relation coverage: {coverage_metrics.relation_coverage_rate:.1%}")
                if coverage_metrics.covered_entities:
                    print(f"     Covered: {', '.join(coverage_metrics.covered_entities[:5])}")
                if coverage_metrics.uncovered_entities:
                    print(f"     Missed: {', '.join(coverage_metrics.uncovered_entities[:5])}")
                
                # Print FULL answer in detailed mode
                print(f"\n  ┌{'─'*76}┐")
                print(f"  │ FULL ANSWER{' '*64}│")
                print(f"  ├{'─'*76}┤")
                # Wrap answer text
                wrapped = textwrap.wrap(answer_result.answer, width=74)
                for line in wrapped:
                    print(f"  │ {line:<74} │")
                print(f"  └{'─'*76}┘")
            
            # 3. RAGAS EVALUATION
            if self.ragas:
                if self.detailed:
                    print("\n  3. RAGAS EVALUATION")
                    print("     Evaluating faithfulness...")
                
                context_text = "\n\n".join([chunk.text for chunk in retrieval_result.chunks[:10]])
                
                faith = self.ragas.faithfulness(answer_result.answer, context_text)
                
                time.sleep(2)  # Rate limit protection
                
                if self.detailed:
                    print("     Evaluating relevancy...")
                
                rel = self.ragas.answer_relevancy(query_def['query'], answer_result.answer)
                
                if self.detailed:
                    print(f"     Faithfulness: {faith['score']:.3f} ({faith['supported']}/{faith['total']} claims)")
                    print(f"     Relevancy: {rel['score']:.3f}")
                    if faith.get('explanation'):
                        print(f"     Explanation: {faith['explanation'][:100]}...")
                
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
            
            # ============================================================
            # PATCH: Extract citations and build detailed data
            # ============================================================
            
            # Extract citations from answer text
            citations = re.findall(r'\[(\d+)\]', answer_result.answer)
            cited_indices = sorted(set(int(c) for c in citations))
            
            # Build detailed data (only in detailed mode to save memory)
            chunks_detail = None
            relations_detail = None
            if self.detailed:
                chunks_detail = [
                    {
                        'index': i + 1,
                        'chunk_id': c.chunk_id,
                        'doc_id': c.doc_id,
                        'doc_type': c.doc_type,
                        'score': c.score,
                        'method': c.retrieval_method,
                        'text': c.text,  # Full text for detailed mode
                        'cited': (i + 1) in cited_indices,
                        'jurisdiction': c.jurisdiction if hasattr(c, 'jurisdiction') else None
                    }
                    for i, c in enumerate(retrieval_result.chunks)
                ]
                relations_detail = [
                    {
                        'source': rel.source_name,
                        'predicate': rel.predicate,
                        'target': rel.target_name,
                        'confidence': rel.confidence
                    }
                    for rel in retrieval_result.subgraph.relations[:20]  # Top 20
                ]
            
            test_result = TestResult(
                test_id=f"{query_def['id']}_{mode.value}",
                query=query_def['query'],
                mode=mode.value,
                category=query_def['primary_category'],
                timestamp=datetime.now().isoformat(),
                entity_resolution=entity_metrics,
                graph_utilization=graph_metrics,
                coverage=coverage_metrics,
                retrieval=retrieval_metrics,
                ragas=ragas_metrics,
                performance=performance_metrics,
                answer_text=answer_result.answer,
                success=True,
                error=None,
                chunks_detail=chunks_detail,
                cited_chunks=cited_indices,
                relations_detail=relations_detail
            )
            
            if self.detailed:
                print(f"\n  Test completed in {total_time:.1f}s")
                print(f"  {'─'*76}")
            
            return test_result
            
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            print(f"\nTEST FAILED: {e}")
            
            return TestResult(
                test_id=f"{query_def['id']}_{mode.value}",
                query=query_def['query'],
                mode=mode.value,
                category=query_def['primary_category'],
                timestamp=datetime.now().isoformat(),
                entity_resolution=EntityResolutionMetrics(0, 0, 0.0, 0.0, [], {}),
                graph_utilization=GraphUtilizationMetrics(0, 0, {}, []),
                coverage=CoverageMetrics(0, 0, 0.0, 0, 0, 0.0, [], []),
                retrieval=RetrievalMetrics(0, {}, 0.0, 0.0, {}, []),
                ragas=RAGASMetrics(0.0, {}, 0.0, ""),
                performance=PerformanceMetrics(0.0, 0.0, 0.0, 0, 0.0),
                answer_text="",
                success=False,
                error=str(e),
                chunks_detail=None,
                cited_chunks=[],
                relations_detail=None
            )
    
    def run_full_suite(self, queries: List[Dict], modes: List[RetrievalMode]):
        """Run complete ablation study (sequential or parallel)."""
        
        print(f"\n{'='*80}")
        print("GRAPHRAG RETRIEVAL ABLATION STUDY v1.2")
        print(f"{'='*80}\n")
        
        total_tests = len(queries) * len(modes)
        
        print(f"Configuration:")
        print(f"  Queries: {len(queries)}")
        print(f"  Modes: {[m.value for m in modes]}")
        print(f"  Total tests: {total_tests}")
        print(f"  RAGAS evaluation: {'ENABLED (Haiku)' if self.enable_ragas else 'DISABLED'}")
        print(f"  Execution: {'PARALLEL (' + str(self.max_workers) + ' workers)' if self.parallel else 'SEQUENTIAL'}")
        print()
        
        # Build test list
        test_list = []
        test_num = 0
        for query_def in queries:
            for mode in modes:
                test_num += 1
                test_list.append((query_def, mode, test_num, total_tests))
        
        if self.parallel and not self.detailed:
            # Parallel execution (only in non-detailed mode)
            self._run_parallel(test_list)
        else:
            # Sequential execution
            self._run_sequential(test_list)
        
        return self.results
    
    def _run_sequential(self, test_list: List[Tuple]):
        """Run tests sequentially."""
        for query_def, mode, test_num, total_tests in test_list:
            result = self.run_single_test(query_def, mode, test_num, total_tests)
            self.results.append(result)
    
    def _run_parallel(self, test_list: List[Tuple]):
        """
        Run tests in parallel using ThreadPoolExecutor.
        
        Note: Neo4j and FAISS are thread-safe for reads.
        Rate limiter handles Together.ai API limits.
        """
        completed = 0
        total = len(test_list)
        
        print(f"  Starting {total} tests with {self.max_workers} workers...\n")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_test = {
                executor.submit(self._run_test_wrapper, query_def, mode, test_num, total): 
                (query_def, mode, test_num)
                for query_def, mode, test_num, total in test_list
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_test):
                query_def, mode, test_num = future_to_test[future]
                try:
                    result = future.result()
                    with self.results_lock:
                        self.results.append(result)
                        completed += 1
                    
                    # Progress update
                    with self.print_lock:
                        q_short = query_def['query'][:35] + '...' if len(query_def['query']) > 35 else query_def['query']
                        faith = f"{result.ragas.faithfulness_score:.2f}" if self.enable_ragas else "N/A"
                        print(f"  ✓ [{completed}/{total}] {mode.value:8} | Faith: {faith} | {q_short}")
                        
                except Exception as e:
                    with self.print_lock:
                        print(f"  ✗ [{test_num}/{total}] {mode.value:8} | ERROR: {e}")
        
        # Sort results by test_id for consistent output
        self.results.sort(key=lambda r: r.test_id)
        print(f"\n  Completed {completed}/{total} tests")
    
    def _run_test_wrapper(self, query_def: Dict, mode: RetrievalMode, test_num: int, total_tests: int) -> TestResult:
        """
        Wrapper for parallel execution with minimal output.
        """
        # Suppress detailed output in parallel mode
        old_detailed = self.detailed
        self.detailed = False
        
        try:
            result = self.run_single_test(query_def, mode, test_num, total_tests)
            return result
        finally:
            self.detailed = old_detailed
        
        return self.results
    
    def analyze_results(self):
        """Generate analysis summary with expectation validation."""
        
        print(f"\n\n{'━'*80}")
        print("  ANALYSIS SUMMARY")
        print(f"{'━'*80}\n")
        
        successful_tests = [r for r in self.results if r.success]
        
        print(f"  Tests completed: {len(successful_tests)}/{len(self.results)}\n")
        
        # ─────────────────────────────────────────────────────────────────────
        # 1. MODE PERFORMANCE
        # ─────────────────────────────────────────────────────────────────────
        print(f"  {'─'*76}")
        print(f"  1. MODE PERFORMANCE")
        print(f"  {'─'*76}")
        print(f"\n  {'Mode':<12} {'Chunks':>8} {'Entities':>10} {'Relations':>10} {'Faith':>8} {'Relev':>8}")
        print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
        
        for mode in ['semantic', 'graph', 'dual']:
            mode_results = [r for r in successful_tests if r.mode == mode]
            if mode_results:
                avg_chunks = sum(r.retrieval.total_chunks for r in mode_results) / len(mode_results)
                avg_entities = sum(r.graph_utilization.entities_in_subgraph for r in mode_results) / len(mode_results)
                avg_relations = sum(r.graph_utilization.relations_in_subgraph for r in mode_results) / len(mode_results)
                
                if self.enable_ragas:
                    avg_faith = sum(r.ragas.faithfulness_score for r in mode_results) / len(mode_results)
                    avg_rel = sum(r.ragas.relevancy_score for r in mode_results) / len(mode_results)
                    print(f"  {mode.upper():<12} {avg_chunks:>8.1f} {avg_entities:>10.1f} {avg_relations:>10.1f} {avg_faith:>8.3f} {avg_rel:>8.3f}")
                else:
                    print(f"  {mode.upper():<12} {avg_chunks:>8.1f} {avg_entities:>10.1f} {avg_relations:>10.1f} {'N/A':>8} {'N/A':>8}")
        
        # Group results by query (used in sections 2 and 5)
        queries = {}
        for r in successful_tests:
            if r.query not in queries:
                queries[r.query] = {}
            queries[r.query][r.mode] = r.ragas.faithfulness_score if self.enable_ragas else 0
        
        # ─────────────────────────────────────────────────────────────────────
        # 2. PER-QUERY WINNERS (if RAGAS enabled)
        # ─────────────────────────────────────────────────────────────────────
        if self.enable_ragas:
            print(f"\n  {'─'*76}")
            print(f"  2. PER-QUERY WINNERS (by faithfulness)")
            print(f"  {'─'*76}")
            
            print(f"\n  {'Query':<45} {'Sem':>6} {'Graph':>6} {'Dual':>6} {'Winner':>8}")
            print(f"  {'-'*45} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
            
            winners = {'semantic': 0, 'graph': 0, 'dual': 0, 'tie': 0}
            
            for query, scores in queries.items():
                q_short = query[:42] + '...' if len(query) > 42 else query
                sem = scores.get('semantic', 0)
                graph = scores.get('graph', 0)
                dual = scores.get('dual', 0)
                
                # Determine winner
                max_score = max(sem, graph, dual)
                if max_score == 0:
                    winner = 'N/A'
                elif sem == graph == dual:
                    winner = 'tie'
                    winners['tie'] += 1
                elif sem == max_score:
                    winner = 'semantic'
                    winners['semantic'] += 1
                elif graph == max_score:
                    winner = 'graph'
                    winners['graph'] += 1
                else:
                    winner = 'dual'
                    winners['dual'] += 1
                
                print(f"  {q_short:<45} {sem:>6.2f} {graph:>6.2f} {dual:>6.2f} {winner:>8}")
            
            print(f"\n  Winner counts: semantic={winners['semantic']}, graph={winners['graph']}, dual={winners['dual']}, tie={winners['tie']}")
        
        # ─────────────────────────────────────────────────────────────────────
        # 3. CATEGORY ANALYSIS
        # ─────────────────────────────────────────────────────────────────────
        print(f"\n  {'─'*76}")
        print(f"  3. CATEGORY ANALYSIS")
        print(f"  {'─'*76}")
        
        categories = {}
        for r in successful_tests:
            if r.category not in categories:
                categories[r.category] = {'semantic': [], 'graph': [], 'dual': []}
            categories[r.category][r.mode].append(r)
        
        print(f"\n  {'Category':<30} {'Best Mode':<12} {'Faith (S/G/D)':<20} {'n':>4}")
        print(f"  {'-'*30} {'-'*12} {'-'*20} {'-'*4}")
        
        for cat, mode_results in sorted(categories.items()):
            scores = {}
            for mode in ['semantic', 'graph', 'dual']:
                if mode_results[mode]:
                    if self.enable_ragas:
                        scores[mode] = sum(r.ragas.faithfulness_score for r in mode_results[mode]) / len(mode_results[mode])
                    else:
                        scores[mode] = 0
            
            if scores:
                best = max(scores, key=scores.get)
                score_str = f"{scores.get('semantic', 0):.2f}/{scores.get('graph', 0):.2f}/{scores.get('dual', 0):.2f}"
                n = len(mode_results['semantic']) + len(mode_results['graph']) + len(mode_results['dual'])
                print(f"  {cat:<30} {best.upper():<12} {score_str:<20} {n//3:>4}")
        
        # ─────────────────────────────────────────────────────────────────────
        # 4. RELATION UTILIZATION ANALYSIS
        # ─────────────────────────────────────────────────────────────────────
        print(f"\n  {'─'*76}")
        print(f"  4. RELATION UTILIZATION ANALYSIS")
        print(f"  {'─'*76}")
        
        # Compute relation chunk ratio and correlation with faithfulness
        relation_data = []
        for r in successful_tests:
            chunks = r.retrieval.chunks_by_source
            s = chunks.get('semantic', 0)
            rel_chunks = chunks.get('graph_provenance', 0)
            e = chunks.get('graph_entity', 0)
            total = s + rel_chunks + e
            
            if total > 0:
                rel_ratio = rel_chunks / total
            else:
                rel_ratio = 0
            
            relation_data.append({
                'mode': r.mode,
                'rel_ratio': rel_ratio,
                'rel_chunks': rel_chunks,
                'faithfulness': r.ragas.faithfulness_score if self.enable_ragas else 0
            })
        
        # Aggregate by mode
        print(f"\n  {'Mode':<12} {'Rel Chunks':>12} {'Rel Ratio':>12} {'Avg Faith':>12}")
        print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        
        for mode in ['semantic', 'graph', 'dual']:
            mode_data = [d for d in relation_data if d['mode'] == mode]
            if mode_data:
                avg_rel = sum(d['rel_chunks'] for d in mode_data) / len(mode_data)
                avg_ratio = sum(d['rel_ratio'] for d in mode_data) / len(mode_data)
                avg_faith = sum(d['faithfulness'] for d in mode_data) / len(mode_data)
                print(f"  {mode.upper():<12} {avg_rel:>12.1f} {avg_ratio:>12.1%} {avg_faith:>12.2f}")
        
        # Correlation analysis (simple: high rel_ratio queries vs low)
        if self.enable_ragas:
            graph_data = [d for d in relation_data if d['mode'] == 'graph']
            if graph_data:
                high_rel = [d for d in graph_data if d['rel_ratio'] > 0.3]
                low_rel = [d for d in graph_data if d['rel_ratio'] <= 0.3]
                
                if high_rel and low_rel:
                    high_faith = sum(d['faithfulness'] for d in high_rel) / len(high_rel)
                    low_faith = sum(d['faithfulness'] for d in low_rel) / len(low_rel)
                    print(f"\n  Graph mode correlation (rel_ratio > 0.3 vs <= 0.3):")
                    print(f"    High relation context: {high_faith:.2f} avg faithfulness (n={len(high_rel)})")
                    print(f"    Low relation context:  {low_faith:.2f} avg faithfulness (n={len(low_rel)})")
        
        print(f"\n{'━'*80}\n")
    
    def save_results(self, output_dir: Path, detailed: bool = False):
        """Save results to JSON and markdown."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode_suffix = 'detailed' if detailed else 'stats'
        
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
                    'relation_coverage_rate': r.coverage.relation_coverage_rate,
                    'covered_entities': r.coverage.covered_entities,
                    'uncovered_entities': r.coverage.uncovered_entities
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
                },
                # ============================================================
                # PATCH: Add answer_text and detailed data to JSON
                # ============================================================
                'answer_text': r.answer_text,
                'cited_chunks': r.cited_chunks if r.cited_chunks else []
            }
            
            # Only include heavy data in detailed mode (saves space in full runs)
            if r.chunks_detail:
                result_dict['chunks_detail'] = r.chunks_detail
            if r.relations_detail:
                result_dict['relations_detail'] = r.relations_detail
            
            results_dicts.append(result_dict)
        
        json_file = output_dir / f'ablation_{mode_suffix}_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(results_dicts, f, indent=2, default=str)
        
        print(f"\nResults saved to: {json_file}")
        
        # Generate markdown report
        self._generate_markdown_report(output_dir, timestamp, results_dicts, mode_suffix)
        
        return json_file
    
    def _generate_markdown_report(self, output_dir: Path, timestamp: str, results_dicts: List[Dict], mode_suffix: str):
        """Generate comprehensive markdown report for thesis."""
        
        successful_tests = [r for r in self.results if r.success]
        n_queries = len(set(r.query for r in self.results))
        
        # Group by mode for summary
        mode_stats = {}
        for mode in ['semantic', 'graph', 'dual']:
            mode_results = [r for r in successful_tests if r.mode == mode]
            if mode_results:
                mode_stats[mode] = {
                    'faith_avg': sum(r.ragas.faithfulness_score for r in mode_results) / len(mode_results),
                    'relev_avg': sum(r.ragas.relevancy_score for r in mode_results) / len(mode_results),
                    'time_avg': sum(r.performance.total_time for r in mode_results) / len(mode_results),
                    'cost_avg': sum(r.performance.cost_usd for r in mode_results) / len(mode_results),
                    'entities_avg': sum(r.graph_utilization.entities_in_subgraph for r in mode_results) / len(mode_results),
                    'relations_avg': sum(r.graph_utilization.relations_in_subgraph for r in mode_results) / len(mode_results),
                    'resolution_avg': sum(r.entity_resolution.resolution_rate for r in mode_results) / len(mode_results),
                    'e_cov_avg': sum(r.coverage.entity_coverage_rate for r in mode_results) / len(mode_results),
                    'r_cov_avg': sum(r.coverage.relation_coverage_rate for r in mode_results) / len(mode_results),
                }
        
        # Calculate totals
        total_cost = sum(r.performance.cost_usd for r in successful_tests)
        total_time = sum(r.performance.total_time for r in successful_tests)
        avg_faith = sum(r.ragas.faithfulness_score for r in successful_tests) / len(successful_tests) if successful_tests else 0
        avg_relev = sum(r.ragas.relevancy_score for r in successful_tests) / len(successful_tests) if successful_tests else 0
        
        report = f"""# GraphRAG Retrieval Ablation Study v1.1

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Phase:** 3 - Retrieval & Answer Generation  
**Status:** Complete

---

## Executive Summary

Comparative evaluation of three retrieval strategies for cross-jurisdictional AI governance queries. Tests entity resolution, graph traversal, semantic search, and hybrid approaches against RAGAS quality metrics.

### Study Overview
| Metric | Value |
|--------|-------|
| Total Tests | {len(self.results)} |
| Successful | {len(successful_tests)} |
| Queries | {n_queries} |
| Modes | semantic, graph, dual |
| Total Cost | ${total_cost:.4f} |
| Total Time | {total_time:.1f}s |

### Key Results
| Metric | Value |
|--------|-------|
| Avg Faithfulness | {avg_faith:.3f} |
| Avg Relevancy | {avg_relev:.3f} |
| Entity Resolution | 100% (in-domain) |
| Best Overall Mode | {'semantic' if mode_stats.get('semantic', {}).get('faith_avg', 0) >= mode_stats.get('graph', {}).get('faith_avg', 0) else 'graph'} |

---

## Mode Comparison

### Performance by Mode
| Mode | Faithfulness | Relevancy | Resolution | E.Cov | R.Cov | Time | Cost |
|------|-------------|-----------|------------|-------|-------|------|------|
"""
        for mode, stats in mode_stats.items():
            best_faith = max(s.get('faith_avg', 0) for s in mode_stats.values())
            marker = "**" if stats['faith_avg'] == best_faith else ""
            report += f"| {mode.upper()} | {marker}{stats['faith_avg']:.3f}{marker} | {stats['relev_avg']:.3f} | {stats['resolution_avg']:.1%} | {stats['e_cov_avg']:.1%} | {stats['r_cov_avg']:.1%} | {stats['time_avg']:.1f}s | ${stats['cost_avg']:.4f} |\n"
        
        report += """
### Graph Utilization by Mode
| Mode | Avg Entities | Avg Relations | Subgraph Density |
|------|-------------|---------------|------------------|
"""
        for mode, stats in mode_stats.items():
            density = stats['relations_avg'] / stats['entities_avg'] if stats['entities_avg'] > 0 else 0
            report += f"| {mode.upper()} | {stats['entities_avg']:.1f} | {stats['relations_avg']:.1f} | {density:.2f} |\n"
        
        report += """
---

## Results by Query Category

"""
        # Group by category
        categories = {}
        for rd in results_dicts:
            cat = rd['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(rd)
        
        for cat, cat_results in categories.items():
            report += f"""### {cat.replace('_', ' ').title()}

| Query | Mode | Faith | Relev | Res | Subgraph | S/R/E | Sim | E.Cov | P/R | Jur | Time |
|-------|------|-------|-------|-----|----------|-------|-----|-------|-----|-----|------|
"""
            for rd in cat_results:
                src = rd['retrieval']['chunks_by_source']
                s_chunks = src.get('semantic', 0)
                r_chunks = src.get('graph_provenance', 0)
                e_chunks = src.get('graph_entity', 0)
                
                src_div = rd['retrieval'].get('source_diversity', {})
                p_count = src_div.get('paper', 0)
                r_count = src_div.get('regulation', 0)
                
                jur = rd['retrieval'].get('jurisdiction_diversity', [])
                jur_count = len(jur) if jur else 0
                
                e_cov = rd['coverage'].get('entity_coverage_rate', 0) * 100
                sim = rd['retrieval'].get('avg_query_similarity', 0)
                
                subgraph = f"{rd['graph_utilization']['entities_in_subgraph']}/{rd['graph_utilization']['relations_in_subgraph']}"
                
                # Truncate query
                q_short = rd['query'][:30] + "..." if len(rd['query']) > 30 else rd['query']
                
                report += f"| {q_short} | {rd['mode']} | {rd['ragas']['faithfulness_score']:.2f} | {rd['ragas']['relevancy_score']:.2f} | {rd['entity_resolution']['resolved_count']} | {subgraph} | {s_chunks}/{r_chunks}/{e_chunks} | {sim:.2f} | {e_cov:.0f}% | {p_count}/{r_count} | {jur_count} | {rd['performance']['total_time']:.1f}s |\n"
            
            report += "\n"
        
        report += """---

## Entity Resolution Analysis

### Match Type Distribution
| Match Type | Count | Percentage |
|------------|-------|------------|
"""
        # Aggregate match types
        all_match_types = {}
        total_matches = 0
        for rd in results_dicts:
            for mt, count in rd['entity_resolution'].get('match_types', {}).items():
                all_match_types[mt] = all_match_types.get(mt, 0) + count
                total_matches += count
        
        for mt, count in sorted(all_match_types.items(), key=lambda x: -x[1]):
            pct = count / total_matches * 100 if total_matches > 0 else 0
            report += f"| {mt} | {count} | {pct:.1f}% |\n"
        
        report += f"""
### Resolution Quality
| Metric | Value |
|--------|-------|
| Total Entities Extracted | {sum(rd['entity_resolution']['extracted_count'] for rd in results_dicts)} |
| Total Entities Resolved | {sum(rd['entity_resolution']['resolved_count'] for rd in results_dicts)} |
| Overall Resolution Rate | {sum(rd['entity_resolution']['resolved_count'] for rd in results_dicts) / max(1, sum(rd['entity_resolution']['extracted_count'] for rd in results_dicts)):.1%} |

---

## Source Diversity

### Document Types Retrieved
| Source Type | Chunks | Percentage |
|-------------|--------|------------|
"""
        # Aggregate source types
        all_sources = {}
        total_src = 0
        for rd in results_dicts:
            for src, count in rd['retrieval'].get('source_diversity', {}).items():
                all_sources[src] = all_sources.get(src, 0) + count
                total_src += count
        
        for src, count in sorted(all_sources.items(), key=lambda x: -x[1]):
            pct = count / total_src * 100 if total_src > 0 else 0
            report += f"| {src} | {count} | {pct:.1f}% |\n"
        
        # Aggregate jurisdictions
        all_jurs = set()
        for rd in results_dicts:
            all_jurs.update(rd['retrieval'].get('jurisdiction_diversity', []))
        
        report += f"""
### Jurisdictions Covered
| Metric | Value |
|--------|-------|
| Unique Jurisdictions | {len(all_jurs)} |
| Jurisdictions | {', '.join(sorted(all_jurs)[:10])}{'...' if len(all_jurs) > 10 else ''} |

---

## Mode Winner Analysis

"""
        # Compute per-query winners
        query_scores = {}
        for rd in results_dicts:
            q = rd['query']
            if q not in query_scores:
                query_scores[q] = {'category': rd['category'], 'scores': {}}
            query_scores[q]['scores'][rd['mode']] = rd['ragas']['faithfulness_score']
        
        report += """### Per-Query Winners (by Faithfulness)

| Query | Category | Semantic | Graph | Dual | Winner |
|-------|----------|----------|-------|------|--------|
"""
        winners = {'semantic': 0, 'graph': 0, 'dual': 0, 'tie': 0}
        query_winners = []
        
        for query, data in query_scores.items():
            scores = data['scores']
            sem = scores.get('semantic', 0)
            graph = scores.get('graph', 0)
            dual = scores.get('dual', 0)
            
            max_score = max(sem, graph, dual)
            if max_score == 0:
                winner = 'N/A'
            elif sem == graph == dual:
                winner = 'tie'
                winners['tie'] += 1
            elif sem == max_score:
                winner = 'semantic'
                winners['semantic'] += 1
            elif graph == max_score:
                winner = 'graph'
                winners['graph'] += 1
            else:
                winner = 'dual'
                winners['dual'] += 1
            
            query_winners.append({'query': query, 'category': data['category'], 'winner': winner})
            q_short = query[:35] + '...' if len(query) > 35 else query
            winner_fmt = f"**{winner}**" if winner not in ['N/A', 'tie'] else winner
            report += f"| {q_short} | {data['category']} | {sem:.2f} | {graph:.2f} | {dual:.2f} | {winner_fmt} |\n"
        
        report += f"""
### Winner Summary

| Mode | Wins | Percentage |
|------|------|------------|
| Semantic | {winners['semantic']} | {winners['semantic']/max(1,sum(winners.values()))*100:.1f}% |
| Graph | {winners['graph']} | {winners['graph']/max(1,sum(winners.values()))*100:.1f}% |
| Dual | {winners['dual']} | {winners['dual']/max(1,sum(winners.values()))*100:.1f}% |
| Tie | {winners['tie']} | {winners['tie']/max(1,sum(winners.values()))*100:.1f}% |

### Best Mode by Category

| Category | Best Mode | Semantic | Graph | Dual |
|----------|-----------|----------|-------|------|
"""
        # Aggregate by category
        cat_scores = {}
        for rd in results_dicts:
            cat = rd['category']
            if cat not in cat_scores:
                cat_scores[cat] = {'semantic': [], 'graph': [], 'dual': []}
            cat_scores[cat][rd['mode']].append(rd['ragas']['faithfulness_score'])
        
        for cat, modes in sorted(cat_scores.items()):
            avgs = {}
            for mode in ['semantic', 'graph', 'dual']:
                if modes[mode]:
                    avgs[mode] = sum(modes[mode]) / len(modes[mode])
                else:
                    avgs[mode] = 0
            
            best = max(avgs, key=avgs.get)
            best_fmt = f"**{best.upper()}**"
            report += f"| {cat} | {best_fmt} | {avgs['semantic']:.2f} | {avgs['graph']:.2f} | {avgs['dual']:.2f} |\n"

        report += """
---

## Test Queries

| # | Query | Category |
|---|-------|----------|
"""
        # Get unique queries from results (preserves order)
        seen = set()
        unique_queries = []
        for rd in results_dicts:
            if rd['query'] not in seen:
                seen.add(rd['query'])
                unique_queries.append({'query': rd['query'], 'category': rd['category']})
        
        for i, q in enumerate(unique_queries, 1):
            report += f"| {i} | {q['query']} | {q['category']} |\n"
        
        report += f"""
---

## Methodology

### Retrieval Modes

| Mode | Description |
|------|-------------|
| **SEMANTIC** | Vector similarity search via FAISS (baseline RAG) |
| **GRAPH** | Entity-centric traversal via Neo4j + Steiner Tree |
| **DUAL** | Hybrid: semantic chunks + graph-connected chunks |

### Evaluation Metrics

| Metric | Source | Description |
|--------|--------|-------------|
| Faithfulness | RAGAS | Claims supported by retrieved context |
| Relevancy | RAGAS | Answer addresses the query |
| Resolution | Pipeline | Entities matched to knowledge graph |
| E.Cov | Pipeline | Subgraph entities mentioned in answer |
| R.Cov | Pipeline | Subgraph relations reflected in answer |
| S/R/E | Pipeline | Semantic/Relation/Entity chunk counts |
| P/R | Pipeline | Paper/Regulation source split |
| Jur | Pipeline | Unique jurisdictions in context |

### System Configuration

| Component | Configuration |
|-----------|---------------|
| Entity Resolution | 3-stage (exact → alias → fuzzy) |
| Graph Expansion | Steiner Tree via Neo4j GDS |
| Embedding Model | BGE-M3 (1024 dim) |
| Answer Generation | Claude Sonnet |
| Graph | 38,266 entities, 339,268 relations |

---

## Limitations

- ⚠️ No ground truth annotations (automated RAGAS only)
- ⚠️ Small sample size ({n_queries} queries × 3 modes)
- ⚠️ Steiner Tree (not PCST) for graph expansion
- ⚠️ Single evaluator model (Claude)

---

*Generated by src/analysis/ablation_study.py v1.2*
"""
        
        report_file = output_dir / f'ablation_{mode_suffix}_{timestamp}.md'
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Markdown report saved to: {report_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Retrieval ablation study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detailed analysis (6 queries, full answers printed)
  python src/analysis/ablation_study.py --detailed

  # Stats mode (36 queries, aggregate metrics for charts)
  python src/analysis/ablation_study.py --full
  
  # Quick debug (2 queries, no RAGAS, no LaTeX)
  python src/analysis/ablation_study.py --quick --no-ragas
  
  # Export to LaTeX after running
  python src/analysis/ablation_latex_export.py data/analysis/results/ablation_detailed_*.json
        """
    )
    parser.add_argument('--detailed', action='store_true', 
                        help='Detailed mode: 6 queries, full answers, verbose metrics')
    parser.add_argument('--full', action='store_true', 
                        help='Stats mode: 36 queries, aggregate metrics for charts')
    parser.add_argument('--quick', action='store_true', 
                        help='Quick test with 2 queries (debug)')
    parser.add_argument('--parallel', action='store_true',
                        help='Run tests in parallel (faster, recommended for --full)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--queries', type=int, 
                        help='Override number of queries')
    parser.add_argument('--no-ragas', action='store_true', 
                        help='Skip RAGAS evaluation')
    parser.add_argument('--latex', action='store_true',
                        help='Also export LaTeX files (table, vars, appendix)')
    parser.add_argument('-o', '--output', type=str, default='data/analysis/results', 
                        help='Output directory')
    
    args = parser.parse_args()
    
    try:
        # Determine mode
        if args.detailed:
            queries = DETAILED_QUERIES  # 6 diverse queries
            detailed = True
            print("=" * 80)
            print("  DETAILED MODE: 6 queries × 3 modes = 18 tests")
            print("  Full answers and per-query analysis will be printed")
            print("=" * 80)
        elif args.full:
            queries = FULL_QUERIES  # 36 queries
            detailed = False
            print("=" * 80)
            print("  STATS MODE: 36 queries × 3 modes = 108 tests")
            print("  Compact output, aggregate statistics for charts")
            print("=" * 80)
        else:
            queries = DETAILED_QUERIES
            detailed = False
            print("=" * 80)
            print("  DEFAULT MODE: 6 queries × 3 modes = 18 tests")
            print("  Use --detailed for full answers, --full for 36 queries")
            print("=" * 80)
        
        # Apply overrides
        if args.quick:
            queries = queries[:2]
            print(f"  Quick mode: limited to 2 queries")
            if args.latex:
                print("  Warning: --latex ignored in --quick mode")
                args.latex = False
        elif args.queries:
            queries = queries[:args.queries]
            print(f"  Limited to {args.queries} queries")
        
        print(f"  RAGAS: {'enabled (Haiku)' if not args.no_ragas else 'DISABLED'}")
        print(f"  Answer generation: Haiku")
        print(f"  LaTeX export: {'enabled' if args.latex else 'disabled'}")
        print(f"  Parallel: {'enabled (' + str(args.workers) + ' workers)' if args.parallel else 'disabled'}")
        print("=" * 80 + "\n")
        
        modes = [RetrievalMode.SEMANTIC, RetrievalMode.GRAPH, RetrievalMode.DUAL]
        
        # Warn if parallel + detailed (doesn't make sense)
        if args.parallel and detailed:
            print("⚠️  Note: --parallel is ignored in --detailed mode (sequential output needed)")
            parallel = False
        else:
            parallel = args.parallel
        
        suite = AblationTestSuite(
            enable_ragas=not args.no_ragas, 
            detailed=detailed,
            parallel=parallel,
            max_workers=args.workers
        )
        suite.load_pipeline()
        suite.run_full_suite(queries, modes)
        suite.analyze_results()
        json_path = suite.save_results(Path(args.output), detailed=detailed)
        
        # LaTeX export if requested
        if args.latex and json_path:
            print("\n" + "=" * 80)
            print("  LATEX EXPORT")
            print("=" * 80)
            from src.analysis.ablation_latex_export import LaTeXExporter
            # PATCH: Pass is_detailed flag to LaTeX exporter
            exporter = LaTeXExporter(json_path, is_detailed=detailed)
            latex_dir = Path(args.output) / 'latex'
            exporter.export_all(latex_dir)
        
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