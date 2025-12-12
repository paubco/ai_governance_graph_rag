#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: run_query.py
Package: scripts
Purpose: CLI interface for GraphRAG retrieval pipeline

Author: Pau Barba i Colomer
Created: 2025-12-12
Modified: 2025-12-12

Usage:
    python scripts/run_query.py "What is the EU AI Act?"
    python scripts/run_query.py "Compare GDPR and CCPA" --mode dual --output results.json
    python scripts/run_query.py "High-risk AI systems" --verbose --no-answer

References:
    - PHASE_3_DESIGN.md § 4 (Retrieval Pipeline)
    - PHASE_3_DESIGN.md § 6 (Evaluation)
"""

import sys
import os
from pathlib import Path
import argparse
import json
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local imports
from src.retrieval.retrieval_processor import RetrievalProcessor
from src.retrieval.answer_generator import AnswerGenerator
from src.retrieval.config import RetrievalMode
from src.utils.embedder import BGEEmbedder
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GraphRAG Query Pipeline for AI Governance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_query.py "What is the EU AI Act?"
  python scripts/run_query.py "Compare GDPR and CCPA" --mode dual
  python scripts/run_query.py "High-risk AI systems" --output results.json --verbose
  python scripts/run_query.py "Test query" --no-answer  # Skip LLM call
        """
    )
    
    parser.add_argument(
        'query',
        type=str,
        help='Query string to process'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['dual', 'graphrag', 'naive'],
        default='dual',
        help='Retrieval mode (default: dual)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to JSON file (optional)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed intermediate results'
    )
    
    parser.add_argument(
        '--no-answer',
        action='store_true',
        help='Skip answer generation (retrieval only, for testing)'
    )
    
    return parser.parse_args()


# ============================================================================
# PIPELINE LOADING
# ============================================================================

def load_pipeline():
    """
    Load FAISS indices and pipeline components.
    
    Returns:
        Tuple of (RetrievalProcessor, AnswerGenerator)
    """
    logger.info("Loading pipeline components...")
    
    # Data paths
    data_dir = PROJECT_ROOT / 'data' / 'processed'
    faiss_dir = data_dir / 'faiss'
    
    # Load embedding model
    embedding_model = BGEEmbedder()
    
    # Retrieval processor
    processor = RetrievalProcessor(
        embedding_model=embedding_model,
        # Phase 3.3.1 paths
        faiss_entity_index_path=faiss_dir / 'entities.faiss',
        entity_ids_path=faiss_dir / 'entity_ids.json',
        normalized_entities_path=data_dir / 'entities_normalized.json',
        # Phase 3.3.2 paths
        faiss_chunk_index_path=faiss_dir / 'chunks.faiss',
        chunk_ids_path=faiss_dir / 'chunk_ids.json',
        # Neo4j connection
        neo4j_uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        neo4j_user=os.getenv('NEO4J_USER', 'neo4j'),
        neo4j_password=os.getenv('NEO4J_PASSWORD')
    )
    
    # Answer generator
    generator = AnswerGenerator()
    
    logger.info("Pipeline loaded successfully")
    return processor, generator


# ============================================================================
# QUERY EXECUTION
# ============================================================================

def run_query(query: str, mode: str, verbose: bool, skip_answer: bool):
    """
    Execute query through full pipeline.
    
    Args:
        query: Query string.
        mode: Retrieval mode (dual, graphrag, naive).
        verbose: Show detailed output.
        skip_answer: Skip answer generation.
        
    Returns:
        Dict with all results and metadata.
    """
    start_time = datetime.now()
    
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"MODE:  {mode}")
    print(f"{'='*80}\n")
    
    # Load pipeline
    processor, generator = load_pipeline()
    
    # Convert mode string to enum
    retrieval_mode = RetrievalMode[mode.upper()]
    
    # Step 1-5: Retrieval
    print("🔍 Running retrieval pipeline...")
    retrieval_result = processor.retrieve(query, mode=retrieval_mode)
    
    # Display retrieval results
    print(f"\n✅ Retrieval complete:")
    print(f"   • Resolved entities: {len(retrieval_result.resolved_entities)}")
    print(f"   • Subgraph entities: {len(retrieval_result.subgraph.entities)}")
    print(f"   • Subgraph relations: {len(retrieval_result.subgraph.relations)}")
    print(f"   • Chunks retrieved: {len(retrieval_result.chunks)}")
    
    if verbose:
        print(f"\n📋 Resolved Entities:")
        for entity in retrieval_result.resolved_entities[:10]:
            print(f"   • {entity}")
        
        print(f"\n🔗 Sample Relations:")
        for rel in retrieval_result.subgraph.relations[:5]:
            print(f"   • {rel.source_name} --{rel.predicate}--> {rel.target_name}")
        
        print(f"\n📄 Top Chunks:")
        for i, chunk in enumerate(retrieval_result.chunks[:3], 1):
            print(f"   [{i}] {chunk.doc_id} (score: {chunk.score:.3f}, method: {chunk.retrieval_method})")
            print(f"       {chunk.text[:150]}...")
    
    # Step 6: Answer generation
    answer_result = None
    if not skip_answer:
        print(f"\n💬 Generating answer...")
        
        # Estimate cost first
        estimate = generator.estimate_cost_for_query(retrieval_result)
        print(f"   • Estimated cost: ${estimate['estimated_cost_usd']:.4f}")
        print(f"   • Chunks that fit: {estimate['chunks_that_fit']}/{estimate['total_chunks_available']}")
        
        # Generate answer
        answer_result = generator.generate(retrieval_result)
        
        print(f"\n✅ Answer generated:")
        print(f"   • Input tokens: {answer_result.input_tokens}")
        print(f"   • Output tokens: {answer_result.output_tokens}")
        print(f"   • Actual cost: ${answer_result.cost_usd:.4f}")
        print(f"\n{'─'*80}")
        print(answer_result.answer)
        print(f"{'─'*80}")
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print(f"\n⏱️  Total time: {elapsed:.2f}s")
    
    # Build results dict
    results = {
        'query': query,
        'mode': mode,
        'timestamp': start_time.isoformat(),
        'elapsed_seconds': elapsed,
        'retrieval': {
            'resolved_entities': retrieval_result.resolved_entities,
            'subgraph': {
                'entity_count': len(retrieval_result.subgraph.entities),
                'relation_count': len(retrieval_result.subgraph.relations),
                'entities': retrieval_result.subgraph.entities,
                'relations': [
                    {
                        'source': rel.source_name,
                        'predicate': rel.predicate,
                        'target': rel.target_name,
                        'confidence': rel.confidence,
                    }
                    for rel in retrieval_result.subgraph.relations
                ]
            },
            'chunks': [
                {
                    'chunk_id': chunk.chunk_id,
                    'doc_id': chunk.doc_id,
                    'doc_type': chunk.doc_type,
                    'score': chunk.score,
                    'retrieval_method': chunk.retrieval_method,
                    'text': chunk.text[:500]  # Truncate for JSON
                }
                for chunk in retrieval_result.chunks
            ]
        }
    }
    
    if answer_result:
        results['answer'] = {
            'text': answer_result.answer,
            'input_tokens': answer_result.input_tokens,
            'output_tokens': answer_result.output_tokens,
            'cost_usd': answer_result.cost_usd,
            'model': answer_result.model,
            'chunks_used': answer_result.retrieval_chunks_used,
        }
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Run query
        results = run_query(
            query=args.query,
            mode=args.mode,
            verbose=args.verbose,
            skip_answer=args.no_answer
        )
        
        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Results saved to: {output_path}")
        
        print(f"\n{'='*80}\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Pipeline failed: %s", str(e), exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()