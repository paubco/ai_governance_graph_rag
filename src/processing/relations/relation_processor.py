# -*- coding: utf-8 -*-
"""
Parallel relation extraction processor with checkpoint resume.

Coordinates parallel extraction of relations from normalized entities using
multithreaded processing with rate limiting and progress tracking. Uses
ThreadPoolExecutor with 40 configurable workers, 3000 RPM rate limiting for
API compliance, and JSONL append-only output for resilience with automatic
checkpoint/resume capability.

Example:
    processor = ParallelRelationProcessor(
        extractor=RAKGRelationExtractor(...),
        num_workers=40,
        output_dir=Path("data/interim/relations")
    )
    processor.process_all_entities(entities_list)
"""

# Standard library
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
from tqdm import tqdm

# Local
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.rate_limiter import RateLimiter
from src.utils.logger import setup_logging

# Setup logging - only configure if not already done
setup_logging()
logger = logging.getLogger(__name__)


class ParallelRelationProcessor:
    """
    Parallel orchestrator for relation extraction.
    
    Architecture:
    - ThreadPoolExecutor with 40 workers (configurable)
    - Each worker processes one entity at a time
    - Rate limiter ensures 3000 RPM compliance
    - Checkpoint manager handles progress tracking
    - Resume capability via completed entity detection
    
    Usage:
        processor = ParallelRelationProcessor(
            extractor=RAKGRelationExtractor(...),
            num_workers=40,
            output_dir=Path("data/interim/relations")
        )
        
        processor.process_all_entities(entities_list)
    """
    
    def __init__(
        self,
        extractor,                              # RAKGRelationExtractor instance
        all_chunks: List[Dict],                 # All chunks for extraction
        num_workers: int = 40,
        checkpoint_freq: int = 100,
        rate_limit_rpm: int = 2900,
        output_dir: Path = Path("data/interim/relations"),
        config: Optional[Dict] = None
    ):
        """
        Initialize parallel processor.
        
        Args:
            extractor: RAKGRelationExtractor instance (must be thread-safe)
            all_chunks: List of all chunk dicts (passed to extractor)
            num_workers: Number of parallel threads
            checkpoint_freq: Save progress every N entities
            rate_limit_rpm: Together.ai rate limit (conservative: 2900 < 3000)
            output_dir: Output directory for results
            config: Optional config dict for extraction parameters
        """
        self.extractor = extractor
        self.all_chunks = all_chunks
        self.num_workers = num_workers
        self.config = config or {}
        
        # Components
        self.checkpoint_manager = CheckpointManager(
            output_dir=output_dir,
            checkpoint_freq=checkpoint_freq
        )
        self.rate_limiter = RateLimiter(max_calls_per_minute=rate_limit_rpm)
        
        # Logging
        log_file = output_dir / "extraction.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
        
        logger.info(f"Initialized ParallelRelationProcessor:")
        logger.info(f"  Workers: {num_workers}")
        logger.info(f"  Rate limit: {rate_limit_rpm} RPM")
        logger.info(f"  Checkpoint freq: {checkpoint_freq} entities")
        logger.info(f"  Output dir: {output_dir}")
    
    def process_all_entities(
        self,
        entities: List[Dict],
        test_mode: bool = False,
        max_entities: Optional[int] = None
    ):
        """
        Process all entities in parallel with resume capability.
        
        Args:
            entities: List of entity dicts with keys: entity_id, name, type
            test_mode: If True, only process first 100 entities
            max_entities: Optional limit on number to process
        """
        # Test mode override
        if test_mode:
            entities = entities[:100]
            logger.info("🧪 TEST MODE: Processing only first 100 entities")
        elif max_entities:
            entities = entities[:max_entities]
            logger.info(f"🎯 Processing first {max_entities} entities")
        
        # Resume detection
        completed_ids = self.checkpoint_manager.load_completed_entities()
        remaining = [e for e in entities if e.get('entity_id') not in completed_ids]
        
        # Status report
        print("\n" + "="*80)
        print("📊 PARALLEL RELATION EXTRACTION")
        print("="*80)
        print(f"Total entities: {len(entities)}")
        print(f"Already completed: {len(completed_ids)}")
        print(f"Remaining to process: {len(remaining)}")
        print(f"Workers: {self.num_workers}")
        print(f"Rate limit: {self.rate_limiter.max_calls} RPM")
        print(f"Output: {self.checkpoint_manager.relations_file}")
        print("="*80 + "\n")
        
        if not remaining:
            print("✅ All entities already processed!")
            return
        
        # Start timer
        self.checkpoint_manager.set_start_time(datetime.now())
        start_time = time.time()
        
        # Progress bar
        with tqdm(total=len(remaining), desc="Extracting relations", unit="entity") as pbar:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(
                        self._process_single_entity_with_retry,
                        entity
                    ): entity
                    for entity in remaining
                }
                
                # Process as they complete
                for future in as_completed(futures):
                    entity = futures[future]
                    
                    try:
                        result = future.result()
                        
                        if result:  # Success
                            self.checkpoint_manager.append_result(result)
                            self.checkpoint_manager.update_progress(
                                cost=result['cost'],
                                success=True,
                                total_entities=len(remaining),
                                retries=result.get('retry_attempts', 0),
                                had_second_batch=(result.get('num_batches', 1) > 1)
                            )
                        else:  # Failed after retries
                            self.checkpoint_manager.update_progress(
                                cost=0.0,
                                success=False,
                                total_entities=len(remaining)
                            )
                        
                    except Exception as e:
                        logger.error(f"Unexpected error for {entity.get('name')}: {e}")
                        self.checkpoint_manager.log_failure(entity, e)
                        self.checkpoint_manager.update_progress(
                            cost=0.0,
                            success=False,
                            total_entities=len(remaining)
                        )
                    
                    finally:
                        pbar.update(1)
        
        # Final checkpoint
        self.checkpoint_manager.save_checkpoint(total_entities=len(remaining))
        
        # Summary
        elapsed = time.time() - start_time
        stats = self.checkpoint_manager.get_stats()
        rate_stats = self.rate_limiter.get_stats()
        
        print("\n" + "="*80)
        print("✅ EXTRACTION COMPLETE")
        print("="*80)
        print(f"Completed: {stats['completed']}")
        print(f"Failed: {stats['failed']}")
        print(f"Success rate: {100 * stats['completed'] / stats['total_processed']:.1f}%")
        print(f"Retry attempts: {stats['retry_attempts']}")
        print(f"Second batches: {stats['second_batch_count']}")
        print(f"Total cost: ${stats['cost_usd']:.2f}")
        print(f"Elapsed time: {elapsed/3600:.1f} hours ({elapsed/60:.1f} minutes)")
        print(f"Avg per entity: {stats['elapsed_sec']/stats['total_processed']:.1f}s")
        print(f"\nRate limiter stats:")
        print(f"  Total API calls: {rate_stats['total_calls']}")
        print(f"  Total wait time: {rate_stats['total_wait_time_sec']}s")
        print(f"  Utilization: {rate_stats['utilization_pct']}%")
        print("="*80 + "\n")
        
        # Failed entities report
        if stats['failed'] > 0:
            print(f"⚠️  {stats['failed']} entities failed - see {self.checkpoint_manager.failed_file}")
    
    def _process_single_entity_with_retry(
        self,
        entity: Dict,
        max_retries: int = 3
    ) -> Optional[Dict]:
        """
        Process one entity with exponential backoff retry logic.
        
        Args:
            entity: Entity dict with entity_id, name, type
            max_retries: Number of retry attempts on failure
            
        Returns:
            Result dict if successful, None if all retries failed
        """
        for attempt in range(max_retries):
            try:
                # Rate limiting (thread-safe)
                wait_time = self.rate_limiter.acquire()
                
                if wait_time > 1.0:
                    logger.debug(f"Rate limit wait: {wait_time:.1f}s for {entity.get('name')}")
                
                # Extract relations (main work)
                extraction_result = self.extractor.extract_relations_for_entity(
                    entity,
                    self.all_chunks
                )
                
                # Handle dict return with metadata
                if isinstance(extraction_result, dict):
                    relations_list = extraction_result['relations']
                    num_batches = extraction_result['num_batches']
                    chunks_used = extraction_result['chunks_used']
                else:
                    # Backward compatibility (shouldn't happen)
                    relations_list = extraction_result
                    num_batches = 1
                    chunks_used = len(entity.get('chunk_ids', []))
                
                # Estimate cost (rough: $0.20 per 1M tokens)
                # Assuming ~1500 tokens per batch average
                tokens_used = num_batches * 1500
                cost = (tokens_used / 1_000_000) * 0.20
                
                # Format result for JSONL
                result = {
                    'entity_id': entity['entity_id'],
                    'entity_name': entity['name'],
                    'entity_type': entity['type'],
                    'relations': relations_list,
                    'num_batches': num_batches,
                    'chunks_used': chunks_used,
                    'cost': round(cost, 6),
                    'retry_attempts': attempt,
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.debug(
                    f"✓ {entity['name']}: {len(result['relations'])} relations, "
                    f"{num_batches} batches, ${cost:.4f}"
                )
                
                return result
                
            except Exception as e:
                # Retry logic
                if attempt < max_retries - 1:
                    wait = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(
                        f"Retry {attempt+1}/{max_retries} for {entity.get('name')} "
                        f"after {wait}s (error: {type(e).__name__})"
                    )
                    time.sleep(wait)
                else:
                    # Final failure
                    logger.error(
                        f"Failed after {max_retries} attempts: {entity.get('name')} - {e}"
                    )
                    self.checkpoint_manager.log_failure(entity, e)
                    return None
    
    def estimate_cost_and_time(
        self,
        num_entities: int,
        second_round_rate: float = 0.30
    ) -> Dict:
        """
        Estimate total cost and processing time.
        
        Args:
            num_entities: Number of entities to process
            second_round_rate: % of entities triggering second batch (0.0-1.0)
            
        Returns:
            Dict with estimated cost, time, API calls
        """
        # Calculations
        single_batch_entities = int(num_entities * (1 - second_round_rate))
        double_batch_entities = int(num_entities * second_round_rate)
        
        # Tokens (rough estimate: 1500 per batch)
        tokens_single = single_batch_entities * 1500
        tokens_double = double_batch_entities * 3000
        total_tokens = tokens_single + tokens_double
        
        # Cost ($0.20 per 1M tokens)
        cost = (total_tokens / 1_000_000) * 0.20
        
        # Time (17s per entity average, with parallelization)
        sequential_time = num_entities * 17
        parallel_time = sequential_time / self.num_workers
        
        return {
            'num_entities': num_entities,
            'single_batch_entities': single_batch_entities,
            'double_batch_entities': double_batch_entities,
            'total_tokens': f"{total_tokens:,}",
            'estimated_cost_usd': round(cost, 2),
            'sequential_time_hours': round(sequential_time / 3600, 1),
            'parallel_time_hours': round(parallel_time / 3600, 1),
            'workers': self.num_workers
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n🧪 TESTING PARALLEL PROCESSOR")
    print("="*80)
    
    # Mock extractor for testing
    class MockExtractor:
        def extract_relations(self, entity, **kwargs):
            time.sleep(0.1)  # Simulate API call
            return {
                'relations': [
                    {'subject': entity['name'], 'predicate': 'relates_to', 'object': 'concept'}
                ],
                'num_batches': 1,
                'chunks': ['chunk1', 'chunk2']
            }
    
    # Create processor
    processor = ParallelRelationProcessor(
        extractor=MockExtractor(),
        num_workers=4,  # Small for testing
        checkpoint_freq=5,
        output_dir=Path("test_parallel_output")
    )
    
    # Mock entities
    entities = [
        {'entity_id': f'entity_{i}', 'name': f'Entity {i}', 'type': 'Concept'}
        for i in range(20)
    ]
    
    # Cost estimate
    estimate = processor.estimate_cost_and_time(num_entities=55695)
    print("\nCost estimate for 55,695 entities:")
    for key, value in estimate.items():
        print(f"  {key}: {value}")
    
    # Test run
    print("\n" + "="*80)
    processor.process_all_entities(entities, test_mode=False, max_entities=20)