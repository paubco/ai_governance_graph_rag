#!/usr/bin/env python3
"""
Parallel Relation Extraction Server Script
Uses threading for fast parallel API calls to Together.ai

Features:
- Multi-threaded extraction (3-10 workers based on rate limit tier)
- Auto-resume from checkpoints
- Rate limit handling with exponential backoff
- Progress tracking and statistics
- Prompt validation and full logging
- Integrates enhanced RAKGRelationExtractor with bug fixes

Author: Pau Barba i Colomer
Server: RTX 3060 GPU (for future embedding, not used here)
Usage: python extract_relations_server.py [--workers N] [--limit N]
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
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase1_graph_construction.relation_extractor import RAKGRelationExtractor
from dotenv import load_dotenv

# Load environment
load_dotenv(PROJECT_ROOT / '.env')

# Setup logging with immediate flush
log_handler_stream = logging.StreamHandler(sys.stdout)
log_handler_stream.setLevel(logging.INFO)
log_handler_file = logging.FileHandler('logs/relation_extraction_server.log', mode='a')
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


class ParallelRelationProcessor:
    """
    Parallel relation extraction with checkpoint/resume capability.
    
    Uses ThreadPoolExecutor for I/O-bound API calls.
    Respects Together.ai rate limits with adaptive workers.
    Incorporates all Phase 1D bug fixes: prompt validation, logging, error handling.
    """
    
    def __init__(
        self,
        api_key: str,
        num_workers: int = 3,
        checkpoint_interval: int = 100,
        output_file: str = "data/interim/relations/relations.json",
        semantic_threshold: float = 0.85,
        mmr_lambda: float = 0.55,
        num_chunks: int = 20
    ):
        """
        Initialize parallel processor.
        
        Args:
            api_key: Together.ai API key
            num_workers: Number of parallel workers (3-10 recommended)
            checkpoint_interval: Save every N entities
            output_file: Output path for results
            semantic_threshold: Semantic similarity threshold
            mmr_lambda: MMR diversity parameter
            num_chunks: Chunk count for MMR selection
        """
        self.extractor = RAKGRelationExtractor(
            model_name="Qwen/Qwen2.5-7B-Instruct-Turbo",
            api_key=api_key,
            semantic_threshold=semantic_threshold,
            mmr_lambda=mmr_lambda,
            num_chunks=num_chunks
        )
        self.num_workers = num_workers
        self.checkpoint_interval = checkpoint_interval
        self.output_file = Path(output_file)
        
        # Thread-safe data structures
        self.results = []
        self.results_lock = Lock()
        self.entities_processed = 0
        self.total_relations = 0
        self.errors = 0
        self.skipped = 0  # Oversized prompts
        
        # Statistics tracking
        self.stats = {
            'entities_processed': 0,
            'relations_extracted': 0,
            'errors': 0,
            'skipped_oversized': 0,
            'empty_responses': 0,
            'avg_relations_per_entity': 0.0,
            'avg_processing_time': 0.0
        }
        
        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized with {num_workers} workers")
        logger.info(f"  Semantic threshold: {semantic_threshold}")
        logger.info(f"  MMR lambda: {mmr_lambda}")
        logger.info(f"  Chunks for MMR: {num_chunks}")
    
    def load_normalized_entities(self, entities_file: str = "data/interim/entities/normalized_entities.json"):
        """Load normalized entities from JSON file.
        
        Expected format: {entity_id: {entity_object}, ...} or {"entities": [...]}
        Converts dict to list if needed.
        """
        logger.info(f"Loading entities from {entities_file}...")
        
        with open(entities_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle dict with 'entities' key (metadata wrapper)
        if isinstance(data, dict) and 'entities' in data:
            entities = data['entities']
        # Handle dict with entity_ids as keys
        elif isinstance(data, dict):
            entities = list(data.values())
        # Handle direct list
        elif isinstance(data, list):
            entities = data
        else:
            raise ValueError(f"Unexpected format in {entities_file}")
        
        logger.info(f"✅ Loaded {len(entities)} entities")
        return entities
    
    def load_chunks_embedded(self, chunks_file: str = "data/interim/chunks/chunks_embedded.json"):
        """Load embedded chunks from JSON file.
        
        Expected format: {"chunk_id": {chunk_object}, ...}
        Converts to list of chunk objects for processing.
        """
        logger.info(f"Loading chunks from {chunks_file}...")
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert dict of {chunk_id: chunk_obj} to list
        if isinstance(data, dict):
            chunks = list(data.values())
        elif isinstance(data, list):
            chunks = data
        else:
            raise ValueError(f"Expected dict or list, got {type(data).__name__}")
        
        logger.info(f"✅ Loaded {len(chunks)} chunks")
        return chunks
    
    def load_checkpoint(self):
        """Load existing results from checkpoint if available."""
        if self.output_file.exists():
            logger.info(f"Found checkpoint: {self.output_file}")
            
            with open(self.output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle format
            if isinstance(data, dict) and 'results' in data:
                self.results = data['results']
                self.stats = data.get('stats', self.stats)
            elif isinstance(data, dict) and 'relations' in data:
                # Old format: list of relations
                # Group by entity
                relations_by_entity = defaultdict(list)
                for relation in data['relations']:
                    entity_id = relation.get('extracted_from_entity', 'unknown')
                    relations_by_entity[entity_id].append(relation)
                
                self.results = [
                    {
                        'entity_id': entity_id,
                        'relations': rels
                    }
                    for entity_id, rels in relations_by_entity.items()
                ]
            elif isinstance(data, list):
                self.results = data
            
            # Calculate stats
            self.entities_processed = len(self.results)
            self.total_relations = sum(len(r.get('relations', [])) for r in self.results)
            
            logger.info(f"🔄 Resuming from checkpoint:")
            logger.info(f"   Entities already processed: {self.entities_processed}")
            logger.info(f"   Relations already extracted: {self.total_relations}")
            
            return self.entities_processed
        
        return 0
    
    def save_checkpoint(self, final: bool = False):
        """Save current progress to checkpoint.
        
        NOTE: Caller must hold self.results_lock before calling this!
        """
        try:
            # Update stats
            self.stats['entities_processed'] = len(self.results)
            self.stats['relations_extracted'] = sum(len(r.get('relations', [])) for r in self.results)
            self.stats['errors'] = self.errors
            self.stats['skipped_oversized'] = self.skipped
            
            if self.stats['entities_processed'] > 0:
                self.stats['avg_relations_per_entity'] = (
                    self.stats['relations_extracted'] / self.stats['entities_processed']
                )
            
            output_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'entities_processed': self.stats['entities_processed'],
                    'relations_extracted': self.stats['relations_extracted'],
                    'model': 'Qwen/Qwen2.5-7B-Instruct-Turbo',
                    'final': final
                },
                'stats': self.stats,
                'results': self.results
            }
            
            # Write to temp file first, then atomic rename
            temp_file = self.output_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_file.replace(self.output_file)
            
            if final:
                logger.info(f"✅ Final results saved: {self.output_file}")
            else:
                logger.debug(f"Checkpoint saved: {self.output_file}")
        
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def extract_entity_with_retry(
        self,
        entity: dict,
        all_chunks: list,
        max_retries: int = 3
    ) -> tuple:
        """
        Extract relations for one entity with retry logic.
        
        Returns:
            (entity_id, entity_name, relations, success, error_msg)
        """
        entity_id = entity.get('id', entity.get('name', 'unknown'))
        entity_name = entity.get('name', 'Unknown')
        
        try:
            for attempt in range(max_retries):
                try:
                    # Extract relations with enhanced extractor
                    relations = self.extractor.extract_relations_for_entity(
                        entity,
                        all_chunks,
                        save_prompt=False  # Disable individual prompt logging in parallel mode
                    )
                    
                    return entity_id, entity_name, relations, True, None
                    
                except Exception as e:
                    error_msg = str(e)
                    
                    # Check if rate limit error
                    if '429' in error_msg or 'rate limit' in error_msg.lower():
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Rate limit hit for {entity_name}, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Non-rate-limit error
                        logger.error(f"Error processing {entity_name}: {e}")
                        with self.results_lock:
                            self.errors += 1
                        return entity_id, entity_name, [], False, error_msg
            
            # Max retries exhausted
            logger.error(f"Failed to process {entity_name} after {max_retries} attempts")
            with self.results_lock:
                self.errors += 1
            return entity_id, entity_name, [], False, "Max retries exceeded"
        
        except Exception as e:
            # Outer exception handler for unexpected errors
            logger.error(f"Unexpected error processing {entity_name}: {e}")
            with self.results_lock:
                self.errors += 1
            return entity_id, entity_name, [], False, str(e)
    
    def process_entities(
        self,
        entities: list,
        all_chunks: list,
        start_index: int = 0,
        limit: int = None
    ):
        """
        Process entities in parallel with progress tracking.
        
        Args:
            entities: List of entity dictionaries
            all_chunks: All available chunks
            start_index: Resume from this index
            limit: Process only N entities (for testing)
        """
        # Apply limit and start index
        if limit:
            entities_to_process = entities[start_index:start_index + limit]
        else:
            entities_to_process = entities[start_index:]
        
        total = len(entities_to_process)
        logger.info(f"\nProcessing {total} entities with {self.num_workers} parallel workers")
        logger.info(f"Starting from index {start_index} (checkpoint offset)")
        
        # Statistics
        start_time = datetime.now()
        last_checkpoint = len(self.results)
        entities_processed_this_run = 0
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_entity = {
                executor.submit(self.extract_entity_with_retry, entity, all_chunks): entity
                for entity in entities_to_process
            }
            
            # Process completed tasks
            for future in as_completed(future_to_entity):
                entity_id, entity_name, relations, success, error_msg = future.result()
                
                # Store result
                with self.results_lock:
                    self.results.append({
                        'entity_id': entity_id,
                        'entity_name': entity_name,
                        'relations': relations,
                        'relation_count': len(relations),
                        'success': success,
                        'error': error_msg,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    entities_processed_this_run += 1
                    self.total_relations += len(relations)
                    
                    # Calculate totals
                    total_entities_done = len(self.results)
                    
                    # Progress update every 10 entities
                    if entities_processed_this_run % 10 == 0:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        rate = entities_processed_this_run / elapsed if elapsed > 0 else 0
                        avg_relations = self.total_relations / total_entities_done if total_entities_done > 0 else 0
                        progress_pct = (entities_processed_this_run / total * 100) if total > 0 else 0
                        
                        logger.info(
                            f"Progress: {entities_processed_this_run}/{total} NEW entities ({progress_pct:.1f}%) | "
                            f"Total done: {total_entities_done}/{len(entities)} | "
                            f"Relations: {self.total_relations} | "
                            f"Avg: {avg_relations:.1f}/entity | "
                            f"Rate: {rate:.2f} entities/sec | "
                            f"Errors: {self.errors}"
                        )
                        
                        # Force flush
                        sys.stdout.flush()
                        sys.stderr.flush()
                    
                    # Checkpoint save every N entities
                    if total_entities_done - last_checkpoint >= self.checkpoint_interval:
                        logger.info(f"\n[CHECKPOINT] Saving at {total_entities_done} entities...")
                        self.save_checkpoint(final=False)
                        last_checkpoint = total_entities_done
                        logger.info(f"[CHECKPOINT] Saved. Continuing...\n")
                        sys.stdout.flush()
        
        # Final save
        logger.info(f"\n{'='*80}")
        logger.info(f"Saving final results...")
        with self.results_lock:
            self.save_checkpoint(final=True)
        
        # Print summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ EXTRACTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Entities processed this run: {entities_processed_this_run}")
        logger.info(f"Total entities in output: {len(self.results)}")
        logger.info(f"Total relations: {self.total_relations}")
        logger.info(f"Average per entity: {self.total_relations/len(self.results):.1f}")
        logger.info(f"Errors: {self.errors}")
        logger.info(f"Success rate: {(len(self.results)-self.errors)/len(self.results)*100:.1f}%")
        logger.info(f"Time this run: {elapsed/60:.1f} minutes")
        logger.info(f"Rate: {entities_processed_this_run/elapsed:.2f} entities/sec")
        logger.info(f"Output: {self.output_file}")
        logger.info("=" * 80)


def main():
    """Main execution with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Parallel relation extraction on server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full extraction with 3 workers (Tier 1 rate limit)
  python extract_relations_server.py --workers 3
  
  # Fast extraction with 10 workers (Tier 2+ rate limit)
  python extract_relations_server.py --workers 10
  
  # Test with 100 entities
  python extract_relations_server.py --workers 3 --limit 100
  
  # Resume from checkpoint
  python extract_relations_server.py --workers 3
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
        help='Process only N entities (for testing)'
    )
    parser.add_argument(
        '--entities-file',
        type=str,
        default='data/interim/entities/normalized_entities.json',
        help='Input normalized entities file'
    )
    parser.add_argument(
        '--chunks-file',
        type=str,
        default='data/interim/chunks/chunks_embedded.json',
        help='Input embedded chunks file'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='data/interim/relations/relations.json',
        help='Output relations file'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.85,
        help='Semantic similarity threshold (default: 0.85)'
    )
    parser.add_argument(
        '--lambda',
        type=float,
        default=0.55,
        dest='mmr_lambda',
        help='MMR lambda parameter (default: 0.55)'
    )
    parser.add_argument(
        '--num-chunks',
        type=int,
        default=20,
        help='Chunk count for MMR selection (default: 20)'
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = os.getenv('TOGETHER_API_KEY')
    if not api_key:
        logger.error("❌ TOGETHER_API_KEY not set in environment!")
        logger.error("   Set it with: export TOGETHER_API_KEY='your_key'")
        sys.exit(1)
    
    # Header
    logger.info("=" * 80)
    logger.info("PARALLEL RELATION EXTRACTION (Together.ai)")
    logger.info("=" * 80)
    logger.info(f"Model: Qwen2.5-7B-Instruct-Turbo")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"MMR Lambda: {args.mmr_lambda}")
    logger.info(f"Chunks for MMR: {args.num_chunks}")
    logger.info(f"Entities: {args.entities_file}")
    logger.info(f"Chunks: {args.chunks_file}")
    logger.info(f"Output: {args.output_file}")
    logger.info("")
    
    # Initialize processor
    processor = ParallelRelationProcessor(
        api_key=api_key,
        num_workers=args.workers,
        output_file=args.output_file,
        semantic_threshold=args.threshold,
        mmr_lambda=args.mmr_lambda,
        num_chunks=args.num_chunks
    )
    
    # Load data
    entities = processor.load_normalized_entities(args.entities_file)
    chunks = processor.load_chunks_embedded(args.chunks_file)
    
    # Check for checkpoint
    start_index = processor.load_checkpoint()
    
    # Process entities
    logger.info("-" * 80)
    processor.process_entities(
        entities,
        chunks,
        start_index=start_index,
        limit=args.limit
    )


if __name__ == "__main__":
    main()