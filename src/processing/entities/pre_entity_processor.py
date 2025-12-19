# -*- coding: utf-8 -*-
"""
Parallel pre-entity extraction processor with checkpoint/resume.

Batch orchestration for Phase 1B entity extraction. Wraps DualPassEntityExtractor
with threading, checkpointing, rate limiting, and progress tracking.

Workflow:
    1. Load chunks from JSONL file
    2. Resume from checkpoint if available
    3. Extract entities in parallel (N worker threads)
    4. Save checkpoints every N chunks
    5. Output pre_entities.jsonl

Config:
    --workers N: Parallel workers (3 for Tier 1, 10 for Tier 2+ rate limits)
    --limit N: Process only N chunks (for testing)
    --sample N: Random sample of N chunks (for testing)
    --resume: Resume from checkpoint

Example:
    python -m src.processing.entities.pre_entity_processor --workers 4
    python -m src.processing.entities.pre_entity_processor --sample 50 --seed 42
    python -m src.processing.entities.pre_entity_processor --resume

References:
    - ARCHITECTURE.md Section 3.1.2
    - v1.0 entity_processor.py (parallelism pattern)
"""

# Standard library
import argparse
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import List, Dict, Optional

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
from dotenv import load_dotenv

# Local
from src.processing.entities.pre_entity_extractor import DualPassEntityExtractor
from src.utils.dataclasses import PreEntity
from src.utils.io import load_jsonl, save_jsonl
from src.utils.logger import setup_logging
from config.extraction_config import ENTITY_EXTRACTION_CONFIG

# Load environment
load_dotenv(PROJECT_ROOT / '.env')

# Setup logging
setup_logging(log_file='logs/pre_entity_extraction.log')
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default paths
DEFAULT_INPUT = PROJECT_ROOT / "data/processed/chunks/chunks_embedded.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/interim/entities/pre_entities.jsonl"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "data/interim/entities/.pre_entities_checkpoint.jsonl"

# Processing defaults
DEFAULT_WORKERS = 4
DEFAULT_CHECKPOINT_INTERVAL = 100


# ============================================================================
# PROCESSOR CLASS
# ============================================================================

class PreEntityProcessor:
    """
    Parallel pre-entity extraction with checkpoint/resume.
    
    Uses ThreadPoolExecutor for I/O-bound API calls.
    Implements exponential backoff for rate limiting.
    Saves checkpoints for long-running extractions.
    
    Attributes:
        extractor: DualPassEntityExtractor instance
        num_workers: Number of parallel threads
        checkpoint_interval: Save checkpoint every N chunks
        output_file: Path to output JSONL
        checkpoint_file: Path to checkpoint file
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        num_workers: int = DEFAULT_WORKERS,
        checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
        output_file: Path = DEFAULT_OUTPUT,
        checkpoint_file: Path = DEFAULT_CHECKPOINT,
    ):
        """
        Initialize processor.
        
        Args:
            api_key: Together.ai API key
            num_workers: Parallel worker threads
            checkpoint_interval: Save every N chunks
            output_file: Output JSONL path
            checkpoint_file: Checkpoint path
        """
        self.extractor = DualPassEntityExtractor(api_key=api_key)
        self.num_workers = num_workers
        self.checkpoint_interval = checkpoint_interval
        self.output_file = Path(output_file)
        self.checkpoint_file = Path(checkpoint_file)
        
        # Thread-safe state
        self.results: List[Dict] = []
        self.results_lock = Lock()
        self.processed_chunk_ids: set = set()
        
        # Statistics
        self.stats = {
            'chunks_processed': 0,
            'total_entities': 0,
            'semantic_entities': 0,
            'academic_entities': 0,
            'errors': 0,
        }
        
        # Ensure output directories exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PreEntityProcessor initialized with {num_workers} workers")
    
    def _get_chunk_id(self, chunk: Dict) -> str:
        """Get chunk ID, handling both chunk_id and chunk_ids (deduplicated)."""
        if 'chunk_id' in chunk:
            return chunk['chunk_id']
        elif 'chunk_ids' in chunk:
            return chunk['chunk_ids'][0] if chunk['chunk_ids'] else 'unknown'
        return 'unknown'
    
    def _get_doc_type(self, chunk: Dict) -> str:
        """Get doc type from chunk, handling both document_id and document_ids."""
        # Try document_id first
        if 'document_id' in chunk:
            doc_id = chunk['document_id']
        elif 'document_ids' in chunk:
            doc_id = chunk['document_ids'][0] if chunk['document_ids'] else ''
        else:
            doc_id = ''
        
        # Infer from ID prefix
        if doc_id.startswith("reg_"):
            return "regulation"
        elif doc_id.startswith("paper_"):
            return "paper"
        
        # Fallback: try chunk_id prefix
        chunk_id = self._get_chunk_id(chunk)
        return self._infer_doc_type(chunk_id)
    
    def load_chunks(self, input_file: Path) -> List[Dict]:
        """Load chunks from JSONL file."""
        logger.info(f"Loading chunks from {input_file}...")
        
        chunks = load_jsonl(input_file)
        
        # Handle both list and dict formats
        if isinstance(chunks, dict):
            chunks = list(chunks.values())
        
        logger.info(f"Loaded {len(chunks):,} chunks")
        return chunks
    
    def load_checkpoint(self) -> int:
        """
        Load checkpoint if available.
        
        Returns:
            Number of chunks already processed
        """
        if not self.checkpoint_file.exists():
            return 0
        
        logger.info(f"Loading checkpoint from {self.checkpoint_file}...")
        
        try:
            checkpoint_data = load_jsonl(self.checkpoint_file)
            
            for record in checkpoint_data:
                chunk_id = record.get('chunk_id')
                if chunk_id:
                    self.processed_chunk_ids.add(chunk_id)
                    self.results.append(record)
                    
                    # Update stats
                    entities = record.get('entities', [])
                    self.stats['total_entities'] += len(entities)
                    self.stats['semantic_entities'] += sum(1 for e in entities if e.get('domain'))
                    self.stats['academic_entities'] += sum(1 for e in entities if not e.get('domain'))
            
            self.stats['chunks_processed'] = len(self.processed_chunk_ids)
            
            logger.info(f"Resumed from checkpoint: {len(self.processed_chunk_ids):,} chunks, "
                       f"{self.stats['total_entities']:,} entities")
            
            return len(self.processed_chunk_ids)
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return 0
    
    def save_checkpoint(self, final: bool = False):
        """
        Save current progress to checkpoint.
        
        Note: Caller must hold results_lock!
        
        Args:
            final: If True, save to output_file instead of checkpoint
        """
        target = self.output_file if final else self.checkpoint_file
        
        try:
            save_jsonl(self.results, target)
            
            if final:
                # Remove checkpoint after successful final save
                if self.checkpoint_file.exists():
                    self.checkpoint_file.unlink()
                logger.info(f"Final output saved to {target}")
            else:
                logger.debug(f"Checkpoint saved: {len(self.results)} chunks")
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def extract_chunk_with_retry(
        self,
        chunk: Dict,
        max_retries: int = 3,
    ) -> Dict:
        """
        Extract entities from chunk with exponential backoff.
        
        Args:
            chunk: Chunk dict with chunk_id/chunk_ids, text, and optionally doc_type
            max_retries: Maximum retry attempts
            
        Returns:
            Result dict with chunk_id, entities list
        """
        chunk_id = self._get_chunk_id(chunk)
        chunk_text = chunk['text']
        doc_type = self._get_doc_type(chunk)
        
        for attempt in range(max_retries):
            try:
                entities = self.extractor.extract_entities(
                    chunk_text, chunk_id, doc_type
                )
                
                # Convert PreEntity to dict for JSON serialization
                entities_dict = [asdict(e) for e in entities]
                
                return {
                    'chunk_id': chunk_id,
                    'entities': entities_dict,
                }
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Rate limit: exponential backoff
                if '429' in str(e) or 'rate limit' in error_msg:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limit for {chunk_id}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                # Other errors: log and return empty
                logger.error(f"Error extracting {chunk_id}: {e}")
                with self.results_lock:
                    self.stats['errors'] += 1
                
                return {
                    'chunk_id': chunk_id,
                    'entities': [],
                    'error': str(e),
                }
        
        # Max retries exhausted
        logger.error(f"Failed {chunk_id} after {max_retries} attempts")
        with self.results_lock:
            self.stats['errors'] += 1
        
        return {
            'chunk_id': chunk_id,
            'entities': [],
            'error': 'max_retries_exhausted',
        }
    
    def _infer_doc_type(self, chunk_id: str) -> str:
        """Infer document type from chunk ID prefix."""
        if chunk_id.startswith("reg_"):
            return "regulation"
        elif chunk_id.startswith("paper_"):
            return "paper"
        return "regulation"
    
    def process_chunks(
        self,
        chunks: List[Dict],
        limit: Optional[int] = None,
        sample: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Process chunks in parallel with progress tracking.
        
        Args:
            chunks: List of chunk dicts
            limit: Process only first N chunks
            sample: Random sample of N chunks
            seed: Random seed for sampling
        """
        # Filter out already-processed chunks
        chunks_to_process = [
            c for c in chunks
            if self._get_chunk_id(c) not in self.processed_chunk_ids
        ]
        
        logger.info(f"Chunks remaining: {len(chunks_to_process):,} "
                   f"(skipped {len(self.processed_chunk_ids):,} from checkpoint)")
        
        # Apply sampling/limit
        if sample:
            if seed is not None:
                random.seed(seed)
            chunks_to_process = random.sample(
                chunks_to_process, 
                min(sample, len(chunks_to_process))
            )
            logger.info(f"Sampled {len(chunks_to_process)} chunks (seed={seed})")
        elif limit:
            chunks_to_process = chunks_to_process[:limit]
            logger.info(f"Limited to {len(chunks_to_process)} chunks")
        
        if not chunks_to_process:
            logger.info("No chunks to process")
            return
        
        total = len(chunks_to_process)
        start_time = datetime.now()
        chunks_this_run = 0
        last_checkpoint = len(self.results)
        
        logger.info(f"Processing {total:,} chunks with {self.num_workers} workers...")
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_chunk = {
                executor.submit(self.extract_chunk_with_retry, chunk): chunk
                for chunk in chunks_to_process
            }
            
            for future in as_completed(future_to_chunk):
                result = future.result()
                chunk_id = result['chunk_id']
                entities = result.get('entities', [])
                
                with self.results_lock:
                    self.results.append(result)
                    self.processed_chunk_ids.add(chunk_id)
                    
                    chunks_this_run += 1
                    self.stats['chunks_processed'] += 1
                    self.stats['total_entities'] += len(entities)
                    self.stats['semantic_entities'] += sum(1 for e in entities if e.get('domain'))
                    self.stats['academic_entities'] += sum(1 for e in entities if not e.get('domain'))
                    
                    # Progress logging
                    if chunks_this_run % 50 == 0:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        rate = chunks_this_run / elapsed if elapsed > 0 else 0
                        avg_entities = self.stats['total_entities'] / self.stats['chunks_processed']
                        
                        logger.info(
                            f"Progress: {chunks_this_run}/{total} ({100*chunks_this_run/total:.1f}%) | "
                            f"Entities: {self.stats['total_entities']:,} | "
                            f"Avg: {avg_entities:.1f}/chunk | "
                            f"Rate: {rate:.1f} chunks/sec"
                        )
                    
                    # Checkpoint
                    if len(self.results) - last_checkpoint >= self.checkpoint_interval:
                        logger.info(f"Saving checkpoint at {len(self.results)} chunks...")
                        self.save_checkpoint(final=False)
                        last_checkpoint = len(self.results)
        
        # Final save
        logger.info("Saving final output...")
        with self.results_lock:
            self.save_checkpoint(final=True)
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        self._print_summary(chunks_this_run, elapsed)
    
    def _print_summary(self, chunks_this_run: int, elapsed: float):
        """Print extraction summary."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("EXTRACTION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Chunks processed this run: {chunks_this_run:,}")
        logger.info(f"Total chunks in output: {self.stats['chunks_processed']:,}")
        logger.info(f"Total entities: {self.stats['total_entities']:,}")
        logger.info(f"  - Semantic: {self.stats['semantic_entities']:,}")
        logger.info(f"  - Academic: {self.stats['academic_entities']:,}")
        logger.info(f"Average per chunk: {self.stats['total_entities']/max(1, self.stats['chunks_processed']):.1f}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"Time: {elapsed/60:.1f} minutes")
        logger.info(f"Rate: {chunks_this_run/max(1, elapsed):.2f} chunks/sec")
        logger.info(f"Output: {self.output_file}")
        logger.info("=" * 70)


# ============================================================================
# CLI
# ============================================================================

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Phase 1B: Pre-Entity Extraction (Dual-Pass)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full extraction with 4 workers
    python -m src.processing.entities.pre_entity_processor --workers 4
    
    # Test on 50 random chunks
    python -m src.processing.entities.pre_entity_processor --sample 50 --seed 42
    
    # Process first 100 chunks
    python -m src.processing.entities.pre_entity_processor --limit 100
    
    # Resume from checkpoint
    python -m src.processing.entities.pre_entity_processor --resume
    
    # Custom input/output
    python -m src.processing.entities.pre_entity_processor \\
        --input data/processed/chunks/chunks.jsonl \\
        --output data/interim/entities/pre_entities.jsonl
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=str(DEFAULT_INPUT),
        help='Input chunks JSONL file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=str(DEFAULT_OUTPUT),
        help='Output pre-entities JSONL file'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=DEFAULT_WORKERS,
        help=f'Parallel workers (default: {DEFAULT_WORKERS})'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Process only first N chunks'
    )
    parser.add_argument(
        '--sample',
        type=int,
        help='Random sample of N chunks'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help=f'Save checkpoint every N chunks (default: {DEFAULT_CHECKPOINT_INTERVAL})'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='Together.ai API key (or set TOGETHER_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    # Validate API key
    api_key = args.api_key or os.getenv('TOGETHER_API_KEY')
    if not api_key:
        logger.error("TOGETHER_API_KEY not set!")
        logger.error("Set env var or use --api-key")
        sys.exit(1)
    
    # Header
    logger.info("=" * 70)
    logger.info("PHASE 1B: PRE-ENTITY EXTRACTION (v1.1 Dual-Pass)")
    logger.info("=" * 70)
    logger.info(f"Model: {ENTITY_EXTRACTION_CONFIG['model_name']}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info("")
    
    # Initialize processor
    processor = PreEntityProcessor(
        api_key=api_key,
        num_workers=args.workers,
        checkpoint_interval=args.checkpoint_interval,
        output_file=Path(args.output),
    )
    
    # Load chunks
    chunks = processor.load_chunks(Path(args.input))
    
    # Load checkpoint if resuming
    if args.resume:
        processor.load_checkpoint()
    
    # Process
    processor.process_chunks(
        chunks,
        limit=args.limit,
        sample=args.sample,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()