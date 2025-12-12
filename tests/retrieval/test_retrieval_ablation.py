#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: test_retrieval_ablation.py
Purpose: Ablation study comparing naive, graphrag, and dual retrieval modes

Tests 5 queries (2 simple, 3 complex) across 3 retrieval modes to compare:
- Entity resolution quality
- Chunk retrieval effectiveness  
- Mode-specific strengths and weaknesses

Bonus: Logs entity extraction pipeline for debugging resolution issues.

Usage:
    python scripts/test_retrieval_ablation.py                    # Full ablation (15 tests)
    python scripts/test_retrieval_ablation.py --no-answer        # Skip LLM (faster)
    python scripts/test_retrieval_ablation.py --queries 3        # Test first 3 only

Author: Pau Barba i Colomer
Created: 2025-12-12
"""

import sys
import os
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import List, Dict

# Project root
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
# TEST QUERIES
# ============================================================================

TEST_QUERIES = [
    {
        'id': 'q1',
        'query': 'What is the EU AI Act?',
        'category': 'simple_factual',
        'expected_entities': ['EU AI Act'],
        'expected_strength': 'graphrag'  # Should resolve well to regulation
    },
    {
        'id': 'q2',
        'query': 'EU AI Act requirements',
        'category': 'simple_topic',
        'expected_entities': ['EU AI Act'],
        'expected_strength': 'graphrag'
    },
    {
        'id': 'q3',
        'query': 'What research supports algorithmic fairness requirements?',
        'category': 'complex_research',
        'expected_entities': ['algorithmic fairness', 'research'],
        'expected_strength': 'dual',
        'known_bug': 'Resolves to blockchain instead of fairness concepts'
    },
    {
        'id': 'q4',
        'query': 'Compare GDPR and CCPA transparency requirements',
        'category': 'complex_cross_jurisdictional',
        'expected_entities': ['GDPR', 'CCPA', 'transparency'],
        'expected_strength': 'dual'
    },
    {
        'id': 'q5',
        'query': 'How do different jurisdictions regulate high-risk AI systems?',
        'category': 'complex_multi_entity',
        'expected_entities': ['high-risk AI systems', 'jurisdictions'],
        'expected_strength': 'dual'
    }
]


# ============================================================================
# ENTITY EXTRACTION DEBUGGER
# ============================================================================

class EntityExtractionDebugger:
    """Wrapper to log entity extraction pipeline."""
    
    def __init__(self, processor: RetrievalProcessor):
        self.processor = processor
        self.extraction_logs = []
    
    def retrieve_with_logging(self, query: str, mode: RetrievalMode) -> Dict:
        """Run retrieval and log entity extraction steps."""
        log_entry = {
            'query': query,
            'mode': mode.value,
            'timestamp': datetime.now().isoformat()
        }
        
        # Monkey-patch query parser to log extraction
        original_parse = self.processor.query_parser.parse
        
        def logged_parse(q):
            result = original_parse(q)
            
            # Log extracted entities (before resolution)
            log_entry['extracted_entities'] = [
                {'name': e.name, 'type': e.type}
                for e in result.extracted_entities
            ]
            
            return result
        
        self.processor.query_parser.parse = logged_parse
        
        # Monkey-patch entity resolver to log resolution
        original_resolve = self.processor.entity_resolver.resolve
        
        def logged_resolve(entities):
            resolved = original_resolve(entities)
            
            # Log resolved entities (after FAISS)
            log_entry['resolved_entities'] = [
                {
                    'entity_id': e.entity_id,
                    'name': e.name,
                    'type': e.type,
                    'confidence': e.confidence,
                    'match_type': e.match_type
                }
                for e in resolved
            ]
            
            return resolved
        
        self.processor.entity_resolver.resolve = logged_resolve
        
        # Run retrieval
        try:
            result = self.processor.retrieve(query, mode=mode)
            
            log_entry['success'] = True
            log_entry['resolved_entity_ids'] = result.resolved_entities
            log_entry['subgraph_size'] = {
                'entities': len(result.subgraph.entities),
                'relations': len(result.subgraph.relations)
            }
            log_entry['chunks'] = {
                'total': len(result.chunks),
                'graphrag': sum(1 for c in result.chunks if c.retrieval_method == 'graphrag'),
                'naive': sum(1 for c in result.chunks if c.retrieval_method == 'naive')
            }
            
        except Exception as e:
            log_entry['success'] = False
            log_entry['error'] = str(e)
            result = None
        
        finally:
            # Restore original methods
            self.processor.query_parser.parse = original_parse
            self.processor.entity_resolver.resolve = original_resolve
        
        self.extraction_logs.append(log_entry)
        return result


# ============================================================================
# TEST RUNNER
# ============================================================================

def load_pipeline():
    """Load pipeline components."""
    data_dir = PROJECT_ROOT / 'data'
    faiss_dir = data_dir / 'processed' / 'faiss'
    interim_dir = data_dir / 'interim' / 'entities'
    
    embedding_model = BGEEmbedder()
    
    # Load entity lookup
    entity_lookup = {}
    normalized_entities_path = interim_dir / 'normalized_entities_with_ids.json'
    if normalized_entities_path.exists():
        with open(normalized_entities_path, 'r') as f:
            entities_data = json.load(f)
            for entity in entities_data:
                entity_lookup[entity['entity_id']] = entity['name']
    
    processor = RetrievalProcessor(
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
    
    generator = AnswerGenerator()
    
    return processor, generator, entity_lookup


def run_test_suite(num_queries: int = None, skip_answer: bool = False):
    """Run retrieval mode ablation study."""
    print(f"\n{'='*80}")
    print("RETRIEVAL MODE ABLATION STUDY")
    print(f"{'='*80}\n")
    
    # Load pipeline
    print("Loading pipeline...")
    processor, generator, entity_lookup = load_pipeline()
    debugger = EntityExtractionDebugger(processor)
    
    # Select queries
    queries_to_test = TEST_QUERIES[:num_queries] if num_queries else TEST_QUERIES
    modes = [RetrievalMode.NAIVE, RetrievalMode.GRAPHRAG, RetrievalMode.DUAL]
    
    print(f"Comparing {len(modes)} retrieval modes across {len(queries_to_test)} queries")
    print(f"Total tests: {len(queries_to_test) * len(modes)}")
    print(f"Answer generation: {'SKIPPED' if skip_answer else 'ENABLED'}")
    print(f"\nModes: {', '.join(m.value for m in modes)}\n")
    
    # Run tests
    results = []
    test_num = 0
    
    for query_def in queries_to_test:
        for mode in modes:
            test_num += 1
            
            print(f"\n{'-'*80}")
            print(f"TEST {test_num}/{len(queries_to_test) * len(modes)}")
            print(f"{'-'*80}")
            print(f"Query: {query_def['query']}")
            print(f"Mode: {mode.value}")
            print(f"Category: {query_def['category']}")
            if 'known_bug' in query_def:
                print(f"Known Bug: {query_def['known_bug']}")
            print()
            
            # Run retrieval
            retrieval_result = debugger.retrieve_with_logging(query_def['query'], mode)
            
            if not retrieval_result:
                print("❌ RETRIEVAL FAILED")
                continue
            
            # Display entity extraction pipeline
            log_entry = debugger.extraction_logs[-1]
            
            print("Entity Extraction Pipeline:")
            print(f"  1. LLM Extracted: {log_entry.get('extracted_entities', [])}")
            print(f"  2. FAISS Resolved: {len(log_entry.get('resolved_entities', []))} entities")
            
            # Show resolved entity names
            resolved_names = []
            for ent in log_entry.get('resolved_entities', []):
                name = entity_lookup.get(ent['entity_id'], 'unknown')
                resolved_names.append(f"{name} (conf={ent['confidence']:.2f})")
            print(f"     → {resolved_names[:5]}")
            
            print(f"  3. Subgraph: {log_entry['subgraph_size']['entities']} entities, {log_entry['subgraph_size']['relations']} relations")
            print(f"  4. Retrieved: {log_entry['chunks']['total']} chunks (GraphRAG: {log_entry['chunks']['graphrag']}, Naive: {log_entry['chunks']['naive']})")
            
            # Check for blockchain hallucination
            blockchain_found = any('blockchain' in name.lower() for name in resolved_names)
            if blockchain_found and 'blockchain' not in query_def['query'].lower():
                print("  ⚠️  WARNING: 'blockchain' entity resolved but not in query!")
            
            # Answer generation (optional)
            answer_result = None
            if not skip_answer:
                try:
                    answer_result = generator.generate(retrieval_result)
                    print(f"  5. Answer: {answer_result.output_tokens} tokens, ${answer_result.cost_usd:.4f}")
                except Exception as e:
                    print(f"  5. Answer: FAILED - {e}")
            
            # Store results
            test_result = {
                'test_id': f"{query_def['id']}_{mode.value}",
                'query': query_def['query'],
                'mode': mode.value,
                'category': query_def['category'],
                'extraction_log': log_entry,
                'blockchain_hallucinated': blockchain_found and 'blockchain' not in query_def['query'].lower()
            }
            
            if answer_result:
                test_result['answer'] = {
                    'tokens': answer_result.output_tokens,
                    'cost': answer_result.cost_usd
                }
            
            results.append(test_result)
    
    return results, debugger.extraction_logs


def analyze_results(results: List[Dict], extraction_logs: List[Dict]):
    """Analyze test results and identify patterns."""
    print(f"\n\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}\n")
    
    # Blockchain hallucination analysis
    blockchain_cases = [r for r in results if r.get('blockchain_hallucinated', False)]
    
    print(f"Blockchain Hallucination:")
    print(f"  Found in: {len(blockchain_cases)}/{len(results)} tests")
    
    if blockchain_cases:
        print(f"\n  Affected queries:")
        for case in blockchain_cases:
            print(f"    - {case['query']} (mode: {case['mode']})")
    
    # Entity resolution quality
    print(f"\nEntity Resolution Quality:")
    for log in extraction_logs:
        extracted = log.get('extracted_entities', [])
        resolved = log.get('resolved_entities', [])
        
        print(f"  Query: {log['query'][:50]}...")
        print(f"    Extracted: {len(extracted)} entities")
        print(f"    Resolved: {len(resolved)} entities")
        if extracted and resolved:
            print(f"    Resolution rate: {len(resolved)/len(extracted):.2f}x")
    
    # Mode performance
    print(f"\nMode Performance:")
    for mode in ['naive', 'graphrag', 'dual']:
        mode_results = [r for r in results if r['mode'] == mode]
        if mode_results:
            avg_chunks = sum(r['extraction_log']['chunks']['total'] for r in mode_results) / len(mode_results)
            print(f"  {mode.upper()}: avg {avg_chunks:.1f} chunks")


def save_results(results: List[Dict], output_dir: Path):
    """Save ablation results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'ablation_results_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n\nAblation results saved to: {output_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Ablation study: compare naive, graphrag, and dual retrieval modes',
        epilog="""
Examples:
  python scripts/test_retrieval_ablation.py                 # Full ablation (5 queries × 3 modes)
  python scripts/test_retrieval_ablation.py --no-answer     # Skip LLM generation (faster)
  python scripts/test_retrieval_ablation.py --queries 2     # Test first 2 queries only
        """
    )
    parser.add_argument('--queries', type=int, help='Number of queries to test (default: all 5)')
    parser.add_argument('--no-answer', action='store_true', help='Skip answer generation')
    parser.add_argument('--output', type=str, default='ablation_results', help='Output directory')
    
    args = parser.parse_args()
    
    try:
        # Run tests
        results, extraction_logs = run_test_suite(
            num_queries=args.queries,
            skip_answer=args.no_answer
        )
        
        # Analyze
        analyze_results(results, extraction_logs)
        
        # Save
        save_results(results, Path(args.output))
        
        print(f"\n{'='*80}\n")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Ablation test suite failed: %s", str(e), exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()