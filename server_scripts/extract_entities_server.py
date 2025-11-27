#!/usr/bin/env python3
"""
Parallel Entity Extraction Server Script
Uses threading for fast parallel API calls to Together.ai

Features:
- Multi-threaded extraction (3-10 workers based on rate limit tier)
- Auto-resume from checkpoints
- Rate limit handling with exponential backoff
- Progress tracking and statistics
- Integrates with existing EntityProcessor

Author: Pau Barba i Colomer
Server: RTX 3060 GPU (for future embedding, not used here)
Usage: python extract_entities_server.py [--workers N] [--limit N]
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import defaultdict
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase1_graph_construction.entity_extractor import RAKGEntityExtractor
from dotenv import load_dotenv
import json

# Load environment
load_dotenv(PROJECT_ROOT / '.env')

# Setup logging with immediate flush
import sys
log_handler_stream = logging.StreamHandler(sys.stdout)
log_handler_stream.setLevel(logging.INFO)
log_handler_file = logging.FileHandler('logs/entity_extraction_server.log', mode='a')
log_handler_file.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler_stream.setFormatter(formatter)
log_handler_file.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[log_handler_stream, log_handler_file]
)
logger = logging.getLogger(__name__)

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None


class ParallelEntityProcessor:
    """
    Parallel entity extraction with checkpoint/resume capability.
    
    Uses ThreadPoolExecutor for I/O-bound API calls.
    Respects Together.ai rate limits with adaptive workers.
    """
    
    def __init__(
        self,
        api_key: str,
        num_workers: int = 3,
        checkpoint_interval: int = 1000,
        output_file: str = "data/interim/entities/pre_entities.json"
    ):
        """
        Initialize parallel processor.
        
        Args:
            api_key: Together.ai API key
            num_workers: Number of parallel workers (3-10 recommended)
            checkpoint_interval: Save every N chunks
            output_file: Output path for results
        """
        self.extractor = RAKGEntityExtractor(api_key=api_key)
        self.num_workers = num_workers
        self.checkpoint_interval = checkpoint_interval
        self.output_file = Path(output_file)
        
        # Thread-safe data structures
        self.results = []
        self.results_lock = Lock()
        self.chunks_processed = 0
        self.total_entities = 0
        self.errors = 0
        
        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized with {num_workers} workers")
    
    def load_chunks(self, chunks_file: str = "data/interim/chunks/chunks_text.json"):
        """Load chunks from JSON file."""
        logger.info(f"Loading chunks from {chunks_file}...")
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert dict to list if needed
        if isinstance(data, dict):
            chunks = list(data.values())
        else:
            chunks = data
        
        logger.info(f"✅ Loaded {len(chunks)} chunks")
        return chunks
    
    def load_checkpoint(self):
        """Load existing results from checkpoint if available."""
        if self.output_file.exists():
            logger.info(f"Found checkpoint: {self.output_file}")
            
            with open(self.output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both formats: with metadata or raw list
            if isinstance(data, dict) and 'entities' in data:
                entities_data = data['entities']
            elif isinstance(data, list):
                entities_data = data
            else:
                self.results = []
                return 0
            
            # Detect format: flat (entity_processor) vs nested (server script)
            if len(entities_data) > 0:
                first_item = entities_data[0]
                
                # Check if it's nested format (has 'chunk_text' and nested 'entities')
                if isinstance(first_item, dict) and 'chunk_text' in first_item and 'entities' in first_item:
                    # Nested format - use directly
                    self.results = entities_data
                    
                    # Calculate totals
                    self.chunks_processed = len(self.results)
                    self.total_entities = sum(len(chunk['entities']) for chunk in self.results)
                    
                    logger.info(f"🔄 Resuming from checkpoint: {len(self.results)} chunks already processed")
                    logger.info(f"   Total entities from checkpoint: {self.total_entities}")
                    return len(self.results)
                
                # Flat format (entity_processor) - need to group by chunk_id
                elif isinstance(first_item, dict) and 'name' in first_item and 'chunk_id' in first_item:
                    logger.info("📦 Detected flat format checkpoint - converting to nested format...")
                    
                    # Load chunks to get chunk_text
                    chunks_file = "data/interim/chunks/chunks_text.json"
                    with open(chunks_file, 'r', encoding='utf-8') as f:
                        chunks_data = json.load(f)
                    
                    # Convert to dict for lookup
                    if isinstance(chunks_data, dict):
                        chunks = chunks_data
                    else:
                        chunks = {chunk['chunk_id']: chunk for chunk in chunks_data}
                    
                    # Group entities by chunk_id
                    entities_by_chunk = defaultdict(list)
                    
                    for entity in entities_data:
                        chunk_id = entity['chunk_id']
                        entities_by_chunk[chunk_id].append({
                            'name': entity['name'],
                            'type': entity['type'],
                            'description': entity['description'],
                            'chunk_id': entity['chunk_id']
                        })
                    
                    # Build nested format
                    self.results = []
                    for chunk_id, chunk_entities in entities_by_chunk.items():
                        if chunk_id in chunks:
                            self.results.append({
                                'chunk_id': chunk_id,
                                'chunk_text': chunks[chunk_id]['text'],
                                'entities': chunk_entities
                            })
                    
                    # Update counters
                    self.chunks_processed = len(self.results)
                    self.total_entities = len(entities_data)
                    
                    logger.info(f"✅ Converted {len(entities_data)} flat entities into {len(self.results)} chunks")
                    logger.info(f"🔄 Resuming from checkpoint: {len(self.results)} chunks already processed")
                    return len(self.results)
            
            self.results = []
            return 0
        
        return 0
    
    def save_checkpoint(self, final: bool = False):
        """Save current progress to checkpoint with validation."""
        with self.results_lock:
            data = {
                'metadata': {
                    'final': final,
                    'chunks_processed': len(self.results),
                    'total_entities': self.total_entities,
                    'errors': self.errors,
                    'timestamp': datetime.now().isoformat()
                },
                'entities': self.results
            }
            
            # Save to temp file first, then rename (atomic)
            temp_file = self.output_file.with_suffix('.tmp')
            
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                # Validate JSON by reading it back
                with open(temp_file, 'r', encoding='utf-8') as f:
                    json.load(f)  # Will raise error if invalid
                
                # If valid, replace main file
                temp_file.replace(self.output_file)
                
                if not final:
                    logger.info(f"✓ Checkpoint saved: {len(self.results)} chunks, {self.total_entities} entities")
                else:
                    logger.info(f"✓ FINAL output saved: {len(self.results)} chunks, {self.total_entities} entities")
                    logger.info(f"✓ JSON validated successfully")
                
                sys.stdout.flush()
                
            except Exception as e:
                logger.error(f"❌ Failed to save checkpoint: {e}")
                if temp_file.exists():
                    temp_file.unlink()
                raise
    
    def extract_chunk_with_retry(self, chunk: dict, max_retries: int = 3):
        """
        Extract entities from a single chunk with retry logic.
        
        Args:
            chunk: Chunk dictionary with 'chunk_id' and 'text'
            max_retries: Maximum number of retries on rate limit
            
        Returns:
            Tuple of (chunk_id, chunk_text, entities)
        """
        chunk_id = chunk['chunk_id']
        chunk_text = chunk['text']
        
        for attempt in range(max_retries):
            try:
                entities = self.extractor.extract_entities(chunk_text, chunk_id)
                return chunk_id, chunk_text, entities
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if rate limit error
                if '429' in error_msg or 'rate limit' in error_msg.lower():
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"Rate limit hit for {chunk_id}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Non-rate-limit error
                    logger.error(f"Error processing {chunk_id}: {e}")
                    with self.results_lock:
                        self.errors += 1
                    return chunk_id, chunk_text, []
        
        # Max retries exhausted
        logger.error(f"Failed to process {chunk_id} after {max_retries} attempts")
        with self.results_lock:
            self.errors += 1
        return chunk_id, chunk_text, []
    
    def process_chunks(self, chunks: list, start_index: int = 0, limit: int = None):
        """
        Process chunks in parallel with progress tracking.
        
        Args:
            chunks: List of chunk dictionaries
            start_index: Resume from this index
            limit: Process only N chunks (for testing)
        """
        # Apply limit and start index
        if limit:
            chunks_to_process = chunks[start_index:start_index + limit]
        else:
            chunks_to_process = chunks[start_index:]
        
        total = len(chunks_to_process)
        logger.info(f"Processing {total} chunks with {self.num_workers} parallel workers")
        logger.info(f"Starting from index {start_index} (checkpoint offset)")
        
        # Statistics
        start_time = datetime.now()
        last_checkpoint = len(self.results)  # Track from current checkpoint position
        chunks_processed_this_run = 0
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(self.extract_chunk_with_retry, chunk): chunk
                for chunk in chunks_to_process
            }
            
            # Process completed tasks
            for future in as_completed(future_to_chunk):
                chunk_id, chunk_text, entities = future.result()
                
                # Store result
                with self.results_lock:
                    self.results.append({
                        'chunk_id': chunk_id,
                        'chunk_text': chunk_text,
                        'entities': entities
                    })
                    
                    chunks_processed_this_run += 1
                    self.total_entities += len(entities)
                    
                    # Calculate true totals (checkpoint + new)
                    total_chunks_done = len(self.results)
                    
                    # Progress update every 100 chunks
                    if chunks_processed_this_run % 100 == 0:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        rate = chunks_processed_this_run / elapsed if elapsed > 0 else 0
                        avg_entities = self.total_entities / total_chunks_done if total_chunks_done > 0 else 0
                        progress_pct = (chunks_processed_this_run / total * 100) if total > 0 else 0
                        
                        logger.info(
                            f"Progress: {chunks_processed_this_run}/{total} NEW chunks ({progress_pct:.1f}%) | "
                            f"Total done: {total_chunks_done}/{len(chunks)} | "
                            f"Entities: {self.total_entities} | "
                            f"Avg: {avg_entities:.1f}/chunk | "
                            f"Rate: {rate:.1f} chunks/sec"
                        )
                        
                        # Force flush
                        sys.stdout.flush()
                        sys.stderr.flush()
                    
                    # Checkpoint save every N chunks
                    if total_chunks_done - last_checkpoint >= self.checkpoint_interval:
                        logger.info(f"\n[CHECKPOINT TRIGGER] Saving at {total_chunks_done} total chunks...")
                        self.save_checkpoint(final=False)
                        last_checkpoint = total_chunks_done
                        logger.info(f"[CHECKPOINT SAVED] Continuing...\n")
                        sys.stdout.flush()
        
        # Final save
        logger.info(f"\n{'='*70}")
        logger.info(f"Saving final results...")
        self.save_checkpoint(final=True)
        
        # Print summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ EXTRACTION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Chunks processed this run: {chunks_processed_this_run}")
        logger.info(f"Total chunks in output: {len(self.results)}")
        logger.info(f"Total entities: {self.total_entities}")
        logger.info(f"Average per chunk: {self.total_entities/len(self.results):.1f}")
        logger.info(f"Time this run: {elapsed/60:.1f} minutes")
        logger.info(f"Rate: {chunks_processed_this_run/elapsed:.2f} chunks/sec")
        logger.info(f"Output: {self.output_file}")
        logger.info("=" * 70)


def main():
    """Main execution with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Parallel entity extraction on server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full extraction with 3 workers (Tier 1 rate limit)
  python extract_entities_server.py --workers 3
  
  # Fast extraction with 10 workers (Tier 2+ rate limit)
  python extract_entities_server.py --workers 10
  
  # Test with 100 chunks
  python extract_entities_server.py --workers 3 --limit 100
  
  # Resume from checkpoint
  python extract_entities_server.py --workers 3
        """
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=3,
        help='Number of parallel workers (3 for Tier 1, 10 for Tier 2+)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Process only N chunks (for testing)'
    )
    parser.add_argument(
        '--chunks-file',
        type=str,
        default='data/interim/chunks/chunks_text.json',
        help='Input chunks file'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='data/interim/entities/pre_entities.json',
        help='Output entities file'
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = os.getenv('TOGETHER_API_KEY')
    if not api_key:
        logger.error("❌ TOGETHER_API_KEY not set in environment!")
        logger.error("   Set it with: export TOGETHER_API_KEY='your_key'")
        sys.exit(1)
    
    # Header
    logger.info("=" * 70)
    logger.info("PARALLEL ENTITY EXTRACTION (Together.ai)")
    logger.info("=" * 70)
    logger.info(f"Model: Qwen2.5-72B-Instruct-Turbo")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Input: {args.chunks_file}")
    logger.info(f"Output: {args.output_file}")
    logger.info("")
    
    # Initialize processor
    processor = ParallelEntityProcessor(
        api_key=api_key,
        num_workers=args.workers,
        output_file=args.output_file
    )
    
    # Load chunks
    chunks = processor.load_chunks(args.chunks_file)
    
    # Check for checkpoint
    start_index = processor.load_checkpoint()
    
    # Process chunks
    logger.info("-" * 70)
    processor.process_chunks(chunks, start_index=start_index, limit=args.limit)


if __name__ == "__main__":
    main()