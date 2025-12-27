# -*- coding: utf-8 -*-
"""
Test Chunking

def parse_args():

Examples:
# Quick local test with 10 docs (uses config defaults)
    python test_chunking.py --mode local --sample 10 --seed 42
    
    # Override threshold
    python test_chunking.py --mode local --threshold 0.50 --sample 20 --seed 42
    
    # Full server run
    python test_chunking.py --mode server

References:
    See ARCHITECTURE.md § 3.1.1 for Phase 1A design
    See extraction_config.py CHUNKING_CONFIG for default parameters

"""
# Standard library
import argparse
import json
import logging
import random
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Config
from config.extraction_config import CHUNKING_CONFIG

# Local
from src.processing.chunks.chunk_processor import ChunkProcessor
from src.utils.logger import setup_logging
from src.utils.io import load_jsonl

# Load defaults from config
DEFAULT_THRESHOLD = CHUNKING_CONFIG.get('similarity_threshold', 0.45)
DEFAULT_MIN_SENTENCES = CHUNKING_CONFIG.get('min_sentences', 3)
DEFAULT_MAX_TOKENS = CHUNKING_CONFIG.get('max_tokens', 1500)
DEFAULT_MIN_COHERENCE = CHUNKING_CONFIG.get('min_coherence', 0.30)
DEFAULT_MIN_TOKENS = CHUNKING_CONFIG.get('min_tokens', 15)
DEFAULT_MIN_TOKENS_PER_SENTENCE = CHUNKING_CONFIG.get('min_tokens_per_sentence', 10)
DEFAULT_DEDUP_THRESHOLD = CHUNKING_CONFIG.get('dedup_threshold', 0.95)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run semantic chunking pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Defaults loaded from CHUNKING_CONFIG:
    threshold:              {DEFAULT_THRESHOLD}
    min-sentences:          {DEFAULT_MIN_SENTENCES}
    max-tokens:             {DEFAULT_MAX_TOKENS}
    min-coherence:          {DEFAULT_MIN_COHERENCE}
    min-tokens:             {DEFAULT_MIN_TOKENS}
    min-tokens-per-sentence:{DEFAULT_MIN_TOKENS_PER_SENTENCE}
    dedup-threshold:        {DEFAULT_DEDUP_THRESHOLD}

Examples:
    # Quick local test with 10 docs (uses config defaults)
    python test_chunking.py --mode local --sample 10 --seed 42
    
    # Override threshold
    python test_chunking.py --mode local --threshold 0.50 --sample 20 --seed 42
    
    # Full server run
    python test_chunking.py --mode server
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['local', 'server'],
        default='local',
        help='Processing mode: local (chunking only) or server (with embedding + dedup)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f'Sentence similarity threshold for chunk boundaries (default: {DEFAULT_THRESHOLD})'
    )
    
    parser.add_argument(
        '--min-sentences',
        type=int,
        default=DEFAULT_MIN_SENTENCES,
        help=f'Minimum sentences per chunk (default: {DEFAULT_MIN_SENTENCES})'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f'Maximum tokens per chunk (default: {DEFAULT_MAX_TOKENS})'
    )
    
    parser.add_argument(
        '--min-coherence',
        type=float,
        default=DEFAULT_MIN_COHERENCE,
        help=f'Minimum coherence score to keep chunk (default: {DEFAULT_MIN_COHERENCE})'
    )
    
    parser.add_argument(
        '--min-tokens',
        type=int,
        default=DEFAULT_MIN_TOKENS,
        help=f'Minimum tokens for single-sentence chunks (default: {DEFAULT_MIN_TOKENS})'
    )
    
    parser.add_argument(
        '--min-tokens-per-sentence',
        type=float,
        default=DEFAULT_MIN_TOKENS_PER_SENTENCE,
        help=f'Below this + no header = garbage (default: {DEFAULT_MIN_TOKENS_PER_SENTENCE})'
    )
    
    parser.add_argument(
        '--sample',
        type=int,
        default=0,
        help='Number of documents to sample (0 = all documents)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible sampling'
    )
    
    parser.add_argument(
        '--dedup-threshold',
        type=float,
        default=DEFAULT_DEDUP_THRESHOLD,
        help=f'Cosine similarity threshold for deduplication (server mode, default: {DEFAULT_DEDUP_THRESHOLD})'
    )
    
    parser.add_argument(
        '--show-samples',
        type=int,
        default=5,
        help='Number of sample chunks to display (default: 5)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def print_chunk_samples(chunks_path: Path, discards_path: Path, n_samples: int = 5):
    """
    Print samples of kept and discarded chunks for inspection.
    
    Args:
        chunks_path: Path to chunks.jsonl
        discards_path: Path to discarded_chunks.jsonl
        n_samples: Number of samples to show
    """
    print("\n" + "=" * 80)
    print("SAMPLE KEPT CHUNKS")
    print("=" * 80)
    
    if chunks_path.exists():
        chunks = load_jsonl(chunks_path)
        
        # Sample randomly
        if len(chunks) > n_samples:
            samples = random.sample(chunks, n_samples)
        else:
            samples = chunks
        
        for i, chunk in enumerate(samples, 1):
            print(f"\n--- Kept #{i} [{chunk['chunk_id']}] ---")
            print(f"Source: {chunk['document_id']} | Section: {chunk.get('section_header', 'None')}")
            print(f"Tokens: {chunk['token_count']} | Sentences: {chunk['sentence_count']} | Coherence: {chunk['metadata'].get('coherence', 'N/A')}")
            text_preview = chunk['text'][:400] + "..." if len(chunk['text']) > 400 else chunk['text']
            print(f"Text: {text_preview}")
    else:
        print("No chunks file found.")
    
    print("\n" + "=" * 80)
    print("SAMPLE DISCARDED CHUNKS")
    print("=" * 80)
    
    if discards_path.exists():
        discards = load_jsonl(discards_path)
        
        # Group by reason and show samples from each
        by_reason = {}
        for d in discards:
            reason = d['reason'].split(' (')[0]
            if reason not in by_reason:
                by_reason[reason] = []
            by_reason[reason].append(d)
        
        for reason, items in by_reason.items():
            print(f"\n### Reason: {reason} ({len(items)} total) ###")
            
            samples = items[:min(n_samples, len(items))]
            for i, d in enumerate(samples, 1):
                print(f"\n--- Discarded #{i} [{d['document_id']}] ---")
                print(f"Section: {d.get('section_header', 'None')}")
                print(f"Reason: {d['reason']}")
                print(f"Tokens: {d.get('token_count', 'N/A')} | Sentences: {d.get('sentence_count', 'N/A')} | Coherence: {d.get('coherence', 'N/A')}")
                text_preview = d['text'][:300] + "..." if len(d['text']) > 300 else d['text']
                print(f"Text: {text_preview}")
    else:
        print("No discards file found.")
    
    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    # Set random seed for sample display
    if args.seed is not None:
        random.seed(args.seed)
    
    logger.info("=" * 60)
    logger.info("CHUNKING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Min sentences: {args.min_sentences}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Min coherence: {args.min_coherence}")
    logger.info(f"Min tokens: {args.min_tokens}")
    logger.info(f"Min tokens/sentence: {args.min_tokens_per_sentence}")
    logger.info(f"Sample: {args.sample if args.sample > 0 else 'all'}")
    logger.info(f"Seed: {args.seed}")
    if args.mode == 'server':
        logger.info(f"Dedup threshold: {args.dedup_threshold}")
    logger.info("=" * 60)
    
    # Initialize processor
    processor = ChunkProcessor(
        threshold=args.threshold,
        min_sentences=args.min_sentences,
        max_tokens=args.max_tokens,
        min_coherence=args.min_coherence,
        min_tokens=args.min_tokens,
        min_tokens_per_sentence=args.min_tokens_per_sentence,
        mode=args.mode,
        dedup_threshold=args.dedup_threshold
    )
    
    # Run pipeline
    try:
        report = processor.process(
            sample=args.sample,
            seed=args.seed
        )
        
        # Show sample chunks
        if args.show_samples > 0:
            data_root = PROJECT_ROOT / 'data'
            if args.mode == 'server':
                chunks_path = data_root / 'processed' / 'chunks' / 'chunks_embedded.jsonl'
            else:
                chunks_path = data_root / 'processed' / 'chunks' / 'chunks.jsonl'
            discards_path = data_root / 'interim' / 'chunks' / 'discarded_chunks.jsonl'
            
            print_chunk_samples(chunks_path, discards_path, args.show_samples)
        
        logger.info("Pipeline completed successfully")
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e}")
        logger.error("Make sure Phase 0B has been run first.")
        return 1
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())