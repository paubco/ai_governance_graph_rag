#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: test_retrieval_ablation_v1.py
Purpose: Unified ablation study with integrated RAGAS metrics

Single test suite that:
1. Compares naive, graphrag, and dual retrieval modes
2. Evaluates with RAGAS metrics (faithfulness + relevancy)
3. Generates comprehensive analysis report

v1.0 Features:
- Reference-free RAGAS metrics (no ground truth needed)
- Entity resolution quality analysis
- Retrieval volume comparison
- Cost/efficiency tracking

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
    """Unified test suite for retrieval mode ablation with RAGAS evaluation."""
    
    def __init__(self, enable_ragas: bool = True):
        self.enable_ragas = enable_ragas
        self.processor = None
        self.generator = None
        self.ragas = None
        self.entity_lookup = {}
        self.results = []
    
    def load_pipeline(self):
        """Load all pipeline components."""
        print("Loading pipeline components...")
        
        data_dir = PROJECT_ROOT / 'data'
        faiss_dir = data_dir / 'processed' / 'faiss'
        interim_dir = data_dir / 'interim' / 'entities'
        
        # Embedding model
        embedding_model = BGEEmbedder()
        
        # Entity lookup
        normalized_entities_path = interim_dir / 'normalized_entities_with_ids.json'
        if normalized_entities_path.exists():
            with open(normalized_entities_path, 'r') as f:
                entities_data = json.load(f)
                for entity in entities_data:
                    self.entity_lookup[entity['entity_id']] = entity['name']
        
        # Retrieval processor
        self.processor = RetrievalProcessor(
            embedding_model=embedding_model,
            faiss_entity_index_path=faiss_dir / 'entity_embeddings.index',
            entity_ids_path=faiss_dir / 'entity_id_map.json',
            normalized_entities_path=normalized_entities_path,
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
    
    def run_single_test(self, query_def: Dict, mode: RetrievalMode, test_num: int, total_tests: int):
        """Run a single test: retrieve + generate + evaluate."""
        
        print(f"\n{'='*80}")
        print(f"TEST {test_num}/{total_tests}: {mode.value.upper()}")
        print(f"{'='*80}")
        print(f"Query: {query_def['query']}")
        print(f"Category: {query_def['category']}")
        print()
        
        result = {
            'test_id': f"{query_def['id']}_{mode.value}",
            'query_id': query_def['id'],
            'query': query_def['query'],
            'mode': mode.value,
            'category': query_def['category'],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Step 1: Retrieval
            print("1. RETRIEVAL")
            retrieval_result = self.processor.retrieve(query_def['query'], mode=mode)
            
            # Log entity extraction
            extracted_entities = [e.name for e in retrieval_result.extracted_entities]
            resolved_entity_names = [
                self.entity_lookup.get(eid, 'unknown') 
                for eid in retrieval_result.resolved_entities
            ]
            
            print(f"   Extracted entities: {extracted_entities}")
            print(f"   Resolved to: {len(resolved_entity_names)} entities")
            print(f"   → {resolved_entity_names[:5]}")
            print(f"   Subgraph: {len(retrieval_result.subgraph.entities)} entities, "
                  f"{len(retrieval_result.subgraph.relations)} relations")
            print(f"   Retrieved: {len(retrieval_result.chunks)} chunks")
            
            # Check for hallucinations
            blockchain_hallucination = any(
                'blockchain' in name.lower() 
                for name in resolved_entity_names
            ) and 'blockchain' not in query_def['query'].lower()
            
            if blockchain_hallucination:
                print(f"   ⚠️  WARNING: Blockchain entity hallucination detected!")
            
            result['retrieval'] = {
                'extracted_entities': extracted_entities,
                'resolved_entities': resolved_entity_names,
                'num_entities': len(resolved_entity_names),
                'num_relations': len(retrieval_result.subgraph.relations),
                'num_chunks': len(retrieval_result.chunks),
                'blockchain_hallucination': blockchain_hallucination
            }
            
            # Step 2: Answer Generation
            print("\n2. ANSWER GENERATION")
            answer_result = self.generator.generate(retrieval_result)
            
            print(f"   Tokens: {answer_result.output_tokens}")
            print(f"   Cost: ${answer_result.cost_usd:.4f}")
            print(f"   Answer preview: {answer_result.answer[:150]}...")
            
            result['answer'] = {
                'text': answer_result.answer,
                'tokens': answer_result.output_tokens,
                'cost': answer_result.cost_usd
            }
            
            # Step 3: RAGAS Evaluation
            if self.ragas:
                print("\n3. RAGAS EVALUATION")
                
                # Combine top 10 chunks as context
                context_text = "\n\n".join([
                    chunk.text for chunk in retrieval_result.chunks[:10]
                ])
                
                # Faithfulness
                print("   Evaluating faithfulness...")
                faith = self.ragas.faithfulness(answer_result.answer, context_text)
                print(f"   → Faithfulness: {faith['score']:.3f} "
                      f"({faith['supported']}/{faith['total']} claims supported)")
                
                # Relevancy
                print("   Evaluating relevancy...")
                rel = self.ragas.answer_relevancy(query_def['query'], answer_result.answer)
                print(f"   → Relevancy: {rel['score']:.3f}")
                
                if faith['score'] < 0.7:
                    print(f"   ⚠️  LOW FAITHFULNESS: {faith['explanation']}")
                if rel['score'] < 0.6:
                    print(f"   ⚠️  LOW RELEVANCY: {rel['explanation']}")
                
                result['ragas'] = {
                    'faithfulness': {
                        'score': faith['score'],
                        'supported_claims': faith['supported'],
                        'total_claims': faith['total'],
                        'explanation': faith['explanation']
                    },
                    'relevancy': {
                        'score': rel['score'],
                        'explanation': rel['explanation']
                    }
                }
            
            result['success'] = True
            
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            print(f"\n❌ TEST FAILED: {e}")
            result['success'] = False
            result['error'] = str(e)
        
        self.results.append(result)
        return result
    
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
                self.run_single_test(query_def, mode, test_num, total_tests)
        
        return self.results
    
    def analyze_results(self):
        """Generate comprehensive analysis."""
        
        print(f"\n\n{'='*80}")
        print("ANALYSIS")
        print(f"{'='*80}\n")
        
        successful_tests = [r for r in self.results if r.get('success', False)]
        
        print(f"Tests completed: {len(successful_tests)}/{len(self.results)}\n")
        
        # 1. Retrieval Volume Analysis
        print("1. RETRIEVAL VOLUME BY MODE")
        print("-" * 60)
        
        by_mode = {}
        for mode in ['naive', 'graphrag', 'dual']:
            by_mode[mode] = [r for r in successful_tests if r['mode'] == mode]
        
        for mode, mode_results in by_mode.items():
            if not mode_results:
                continue
            
            avg_chunks = sum(r['retrieval']['num_chunks'] for r in mode_results) / len(mode_results)
            avg_entities = sum(r['retrieval']['num_entities'] for r in mode_results) / len(mode_results)
            avg_relations = sum(r['retrieval']['num_relations'] for r in mode_results) / len(mode_results)
            
            print(f"{mode.upper():10} → {avg_chunks:.1f} chunks, {avg_entities:.1f} entities, {avg_relations:.1f} relations")
        
        # 2. RAGAS Metrics Analysis
        if self.enable_ragas:
            ragas_results = [r for r in successful_tests if 'ragas' in r]
            
            if ragas_results:
                print(f"\n2. RAGAS METRICS BY MODE")
                print("-" * 60)
                
                for mode in ['naive', 'graphrag', 'dual']:
                    mode_ragas = [r for r in ragas_results if r['mode'] == mode]
                    if mode_ragas:
                        avg_faith = sum(r['ragas']['faithfulness']['score'] for r in mode_ragas) / len(mode_ragas)
                        avg_rel = sum(r['ragas']['relevancy']['score'] for r in mode_ragas) / len(mode_ragas)
                        
                        print(f"{mode.upper():10} → Faithfulness: {avg_faith:.3f}, Relevancy: {avg_rel:.3f}")
                
                # Flag quality issues
                low_faith = [r for r in ragas_results if r['ragas']['faithfulness']['score'] < 0.7]
                low_rel = [r for r in ragas_results if r['ragas']['relevancy']['score'] < 0.6]
                
                print(f"\n   Quality Concerns:")
                print(f"   - Low faithfulness (<0.7): {len(low_faith)} cases")
                print(f"   - Low relevancy (<0.6): {len(low_rel)} cases")
                
                if low_faith:
                    print(f"\n   Lowest faithfulness cases:")
                    for r in sorted(low_faith, key=lambda x: x['ragas']['faithfulness']['score'])[:3]:
                        print(f"   - {r['query'][:50]}... ({r['mode']}): {r['ragas']['faithfulness']['score']:.2f}")
        
        # 3. Quality Issues
        print(f"\n3. KNOWN QUALITY ISSUES")
        print("-" * 60)
        
        blockchain_cases = [r for r in successful_tests if r['retrieval'].get('blockchain_hallucination', False)]
        print(f"Blockchain entity hallucinations: {len(blockchain_cases)}/{len(successful_tests)} tests")
        
        if blockchain_cases:
            print(f"   Affected queries:")
            for r in blockchain_cases:
                print(f"   - {r['query']} ({r['mode']})")
        
        # 4. Cost Analysis
        if successful_tests and 'answer' in successful_tests[0]:
            print(f"\n4. COST EFFICIENCY")
            print("-" * 60)
            
            for mode in ['naive', 'graphrag', 'dual']:
                mode_results = [r for r in successful_tests if r['mode'] == mode and 'answer' in r]
                if mode_results:
                    avg_cost = sum(r['answer']['cost'] for r in mode_results) / len(mode_results)
                    avg_tokens = sum(r['answer']['tokens'] for r in mode_results) / len(mode_results)
                    total_cost = sum(r['answer']['cost'] for r in mode_results)
                    
                    print(f"{mode.upper():10} → ${avg_cost:.4f}/query, {avg_tokens:.0f} tokens avg, ${total_cost:.4f} total")
        
        # 5. By Category
        print(f"\n5. PERFORMANCE BY QUERY CATEGORY")
        print("-" * 60)
        
        categories = set(r['category'] for r in successful_tests)
        for cat in sorted(categories):
            cat_results = [r for r in successful_tests if r['category'] == cat]
            print(f"\n{cat}:")
            
            for mode in ['naive', 'graphrag', 'dual']:
                mode_cat = [r for r in cat_results if r['mode'] == mode]
                if mode_cat:
                    avg_chunks = sum(r['retrieval']['num_chunks'] for r in mode_cat) / len(mode_cat)
                    if self.enable_ragas and 'ragas' in mode_cat[0]:
                        avg_faith = sum(r['ragas']['faithfulness']['score'] for r in mode_cat) / len(mode_cat)
                        print(f"  {mode:10} → {avg_chunks:.1f} chunks, faithfulness: {avg_faith:.3f}")
                    else:
                        print(f"  {mode:10} → {avg_chunks:.1f} chunks")
    
    def save_results(self, output_dir: Path):
        """Save results and generate report."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON results
        json_file = output_dir / f'ablation_results_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate markdown report
        report_file = output_dir / f'ablation_report_{timestamp}.md'
        self._generate_report(report_file)
        
        print(f"\n\n{'='*80}")
        print(f"RESULTS SAVED")
        print(f"{'='*80}")
        print(f"JSON results: {json_file}")
        print(f"Report: {report_file}")
        print(f"{'='*80}\n")
    
    def _generate_report(self, output_file: Path):
        """Generate markdown summary report."""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""# GraphRAG Retrieval Ablation Study - v1.0 Results

**Generated:** {timestamp}  
**Total Tests:** {len(self.results)}  
**Successful:** {len([r for r in self.results if r.get('success', False)])}  
**Modes Compared:** naive, graphrag, dual  

---

## Executive Summary

This ablation study compares three retrieval strategies:
- **NAIVE:** Vector similarity search only (baseline RAG)
- **GRAPHRAG:** Entity-centric graph traversal only
- **DUAL:** Hybrid approach combining both strategies

### Evaluation Metrics

"""
        
        if self.enable_ragas:
            report += """**RAGAS Metrics (Reference-Free):**
- Faithfulness: Claims supported by retrieved context (0-1)
- Answer Relevancy: Answer addresses query (0-1)

"""
        
        report += """**Retrieval Metrics:**
- Chunks retrieved
- Entities resolved
- Graph relations traversed

---

## Key Findings

"""
        
        # Add statistics by mode
        successful_tests = [r for r in self.results if r.get('success', False)]
        
        by_mode = {}
        for mode in ['naive', 'graphrag', 'dual']:
            by_mode[mode] = [r for r in successful_tests if r['mode'] == mode]
        
        for mode, mode_results in by_mode.items():
            if not mode_results:
                continue
            
            avg_chunks = sum(r['retrieval']['num_chunks'] for r in mode_results) / len(mode_results)
            avg_entities = sum(r['retrieval']['num_entities'] for r in mode_results) / len(mode_results)
            
            report += f"""
### {mode.upper()} Mode

- Average chunks retrieved: {avg_chunks:.1f}
- Average entities resolved: {avg_entities:.1f}
"""
            
            if self.enable_ragas and 'ragas' in mode_results[0]:
                avg_faith = sum(r['ragas']['faithfulness']['score'] for r in mode_results) / len(mode_results)
                avg_rel = sum(r['ragas']['relevancy']['score'] for r in mode_results) / len(mode_results)
                
                report += f"""- Faithfulness: {avg_faith:.3f}
- Relevancy: {avg_rel:.3f}
"""
        
        # Known issues
        blockchain_cases = [r for r in successful_tests if r['retrieval'].get('blockchain_hallucination', False)]
        
        report += f"""
---

## Known Quality Issues

### Entity Resolution Hallucinations
- **Blockchain entity incorrectly resolved:** {len(blockchain_cases)}/{len(successful_tests)} tests
- Affects queries about non-blockchain topics
- Root cause: Entity embedding similarity (requires investigation)

"""
        
        if self.enable_ragas:
            ragas_results = [r for r in successful_tests if 'ragas' in r]
            low_faith = [r for r in ragas_results if r['ragas']['faithfulness']['score'] < 0.7]
            
            if low_faith:
                report += f"""
### Answer Quality Concerns
- **Low faithfulness (<0.7):** {len(low_faith)} cases
- Indicates potential hallucinations or unsupported claims
- Requires deeper analysis in v1.1
"""
        
        # Test queries
        report += """
---

## Test Queries

"""
        unique_queries = {}
        for r in self.results:
            if r['query_id'] not in unique_queries:
                unique_queries[r['query_id']] = {
                    'query': r['query'],
                    'category': r['category']
                }
        
        for i, (qid, qdata) in enumerate(sorted(unique_queries.items()), 1):
            report += f"{i}. **{qdata['query']}** ({qdata['category']})\n"
        
        # Limitations
        report += """
---

## Limitations (v1.0)

⚠️ **No ground truth annotations**
- Cannot measure precision/recall
- Cannot validate entity resolution accuracy
- Qualitative comparison only

⚠️ **Small sample size**
- 5 queries, 15 tests total
- No statistical significance testing
- Results are preliminary

⚠️ **Reference-free metrics only**
- RAGAS faithfulness and relevancy implemented
- Context precision/recall require ground truth (v1.1)

---

## Next Steps (v1.1)

**Planned Improvements:**
1. Ground truth annotation (25+ queries)
2. Retrieval metrics (P@k, R@k, F1@k)
3. Full RAGAS metric suite
4. Statistical significance testing
5. Entity resolution quality validation

**Timeline:** 6 weeks, ~80 hours estimated

See `EVALUATION_V1_IMPLEMENTATION_GUIDE.md` for full roadmap.

---

## Conclusion

v1.0 demonstrates:
✓ Functional three-mode architecture
✓ RAGAS metrics successfully integrated
✓ Quality issues identified for improvement
✓ Foundation for comprehensive v1.1 evaluation

This is a **proof-of-concept evaluation** showing the system works and can be
rigorously evaluated with proper ground truth in v1.1.

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
        description='Unified retrieval ablation study with RAGAS metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/retrieval/test_retrieval_ablation_v1.py              # Full suite (18 tests)
  python tests/retrieval/test_retrieval_ablation_v1.py --quick      # Quick test (2 queries)
  python tests/retrieval/test_retrieval_ablation_v1.py --no-ragas   # Skip RAGAS (faster)
  python tests/retrieval/test_retrieval_ablation_v1.py -o results/  # Custom output dir

Metrics:
  - RAGAS Faithfulness: Claims supported by context (0-1)
  - RAGAS Relevancy: Answer addresses query (0-1)
  - Retrieval volume: Chunks/entities/relations
  - Cost tracking: Tokens and USD per query
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with 2 queries only')
    parser.add_argument('--queries', type=int,
                       help='Number of queries to test (default: all 5)')
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