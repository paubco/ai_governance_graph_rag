# -*- coding: utf-8 -*-
"""
Parallel relation extraction processor with checkpoint resume (v2.0).

Coordinates parallel extraction of relations from normalized entities using
multithreaded processing with rate limiting and progress tracking.

v2.0 Changes:
- Uses entity_id throughout (no post-hoc name normalization)
- Updated input paths to Phase 1C outputs
- Loads both semantic and metadata entities

Usage:
    # Full extraction
    python -m src.processing.relations.relation_processor
    
    # Test run (100 entities)
    python -m src.processing.relations.relation_processor --entities 100
    
    # Resume from checkpoint
    python -m src.processing.relations.relation_processor --resume
    
    # Semantic track only
    python -m src.processing.relations.relation_processor --track semantic
"""

# Standard library
import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
from tqdm import tqdm

# Local
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.rate_limiter import RateLimiter
from src.utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class ParallelRelationProcessor:
    """
    Parallel orchestrator for relation extraction (v2.0).
    
    Features:
    - ThreadPoolExecutor with configurable workers
    - Rate limiter for API compliance (2900 RPM)
    - Checkpoint manager for resume capability
    - JSONL append-only output
    """
    
    def __init__(
        self,
        extractor,
        all_chunks: List[Dict],
        num_workers: int = 40,
        checkpoint_freq: int = 100,
        rate_limit_rpm: int = 2900,
        output_dir: Path = None,
        config: Optional[Dict] = None
    ):
        self.extractor = extractor
        self.all_chunks = all_chunks
        self.num_workers = num_workers
        self.config = config or {}
        
        if output_dir is None:
            output_dir = PROJECT_ROOT / "data/processed/relations"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_manager = CheckpointManager(
            output_dir=self.output_dir,
            checkpoint_freq=checkpoint_freq
        )
        self.rate_limiter = RateLimiter(max_calls_per_minute=rate_limit_rpm)
        
        logger.info(f"Initialized ParallelRelationProcessor v2.0:")
        logger.info(f"  Workers: {num_workers}")
        logger.info(f"  Rate limit: {rate_limit_rpm} RPM")
        logger.info(f"  Checkpoint freq: {checkpoint_freq}")
        logger.info(f"  Output dir: {self.output_dir}")
    
    def estimate_cost_and_time(
        self,
        num_entities: int,
        second_round_rate: float = 0.35
    ) -> Dict:
        avg_tokens_per_batch = 1500
        batches_per_entity = 1 + second_round_rate
        total_batches = num_entities * batches_per_entity
        total_tokens = total_batches * avg_tokens_per_batch
        
        cost_usd = (total_tokens / 1_000_000) * 0.20
        rpm = self.rate_limiter.max_calls
        time_hours = (total_batches / rpm) / 60
        
        return {
            'num_entities': num_entities,
            'estimated_batches': int(total_batches),
            'estimated_tokens': int(total_tokens),
            'estimated_cost_usd': f"${cost_usd:.2f}",
            'estimated_time_hours': f"{time_hours:.1f}h",
            'rate_limit_rpm': rpm,
        }
    
    def process_all_entities(
        self,
        entities: List[Dict],
        max_entities: Optional[int] = None
    ):
        if max_entities:
            entities = entities[:max_entities]
            logger.info(f"Processing first {max_entities} entities")
        
        completed_ids = self.checkpoint_manager.load_completed_entities()
        remaining = [e for e in entities if e.get('entity_id') not in completed_ids]
        
        print("\n" + "=" * 80)
        print("PARALLEL RELATION EXTRACTION (v2.0)")
        print("=" * 80)
        print(f"Total entities: {len(entities)}")
        print(f"Already completed: {len(completed_ids)}")
        print(f"Remaining: {len(remaining)}")
        print(f"Workers: {self.num_workers}")
        print(f"Rate limit: {self.rate_limiter.max_calls} RPM")
        print(f"Output: {self.checkpoint_manager.relations_file}")
        print("=" * 80 + "\n")
        
        if not remaining:
            print("All entities already processed!")
            return
        
        self.checkpoint_manager.set_start_time(datetime.now())
        start_time = time.time()
        
        with tqdm(total=len(remaining), desc="Extracting relations", unit="entity") as pbar:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(self._process_single_entity, entity): entity
                    for entity in remaining
                }
                
                for future in as_completed(futures):
                    entity = futures[future]
                    
                    try:
                        result = future.result()
                        
                        if result:
                            self.checkpoint_manager.append_result(result)
                            self.checkpoint_manager.update_progress(
                                cost=result.get('cost', 0),
                                success=True,
                                total_entities=len(remaining),
                                retries=result.get('retry_attempts', 0),
                                had_second_batch=(result.get('num_batches', 1) > 1)
                            )
                        else:
                            self.checkpoint_manager.update_progress(
                                cost=0.0, success=False, total_entities=len(remaining)
                            )
                    
                    except Exception as e:
                        logger.error(f"Error for {entity.get('name')}: {e}")
                        self.checkpoint_manager.log_failure(entity, e)
                        self.checkpoint_manager.update_progress(
                            cost=0.0, success=False, total_entities=len(remaining)
                        )
                    
                    finally:
                        pbar.update(1)
        
        self.checkpoint_manager.save_checkpoint(total_entities=len(remaining))
        
        elapsed = time.time() - start_time
        stats = self.checkpoint_manager.get_stats()
        
        print("\n" + "=" * 80)
        print("EXTRACTION COMPLETE")
        print("=" * 80)
        print(f"Completed: {stats['completed']}")
        print(f"Failed: {stats['failed']}")
        if stats['total_processed'] > 0:
            print(f"Success rate: {100 * stats['completed'] / stats['total_processed']:.1f}%")
        print(f"Second batches: {stats['second_batch_count']}")
        print(f"Total cost: ${stats['cost_usd']:.2f}")
        print(f"Elapsed: {elapsed/3600:.1f}h ({elapsed/60:.1f}m)")
        print("=" * 80 + "\n")
    
    def _process_single_entity(
        self,
        entity: Dict,
        max_retries: int = 3
    ) -> Optional[Dict]:
        for attempt in range(max_retries):
            try:
                self.rate_limiter.acquire()
                
                result = self.extractor.extract_relations_for_entity(
                    entity, self.all_chunks
                )
                
                num_batches = result.get('num_batches', 1)
                cost = (num_batches * 1500 / 1_000_000) * 0.20
                
                return {
                    'entity_id': entity['entity_id'],
                    'entity_name': entity['name'],
                    'entity_type': entity['type'],
                    'relations': result['relations'],
                    'num_batches': num_batches,
                    'chunks_used': result.get('chunks_used', 0),
                    'strategy': result.get('strategy', 'unknown'),
                    'cost': round(cost, 6),
                    'retry_attempts': attempt,
                    'timestamp': datetime.now().isoformat()
                }
            
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for {entity.get('name')}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        
        return None


# ============================================================================
# CLI HELPERS
# ============================================================================

def load_entities(track: str) -> List[Dict]:
    """Load entities based on track selection."""
    entities = []
    
    semantic_file = PROJECT_ROOT / "data/processed/entities/entities_semantic.jsonl"
    metadata_file = PROJECT_ROOT / "data/processed/entities/entities_metadata.jsonl"
    
    if track in ('semantic', 'all'):
        if semantic_file.exists():
            with open(semantic_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entities.append(json.loads(line))
            logger.info(f"Loaded {len(entities)} semantic entities")
    
    if track in ('citation', 'all'):
        if metadata_file.exists():
            count = 0
            with open(metadata_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entity = json.loads(line)
                        if track == 'citation':
                            if entity.get('type') == 'Citation':
                                entities.append(entity)
                                count += 1
                        else:
                            entities.append(entity)
                            count += 1
            logger.info(f"Loaded {count} metadata entities")
    
    return entities


def load_chunks() -> List[Dict]:
    """Load embedded chunks."""
    chunks_file = PROJECT_ROOT / "data/processed/chunks/chunks_embedded.json"
    
    if not chunks_file.exists():
        logger.error(f"Chunks file not found: {chunks_file}")
        return []
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = list(data.values()) if isinstance(data, dict) else data
    logger.info(f"Loaded {len(chunks)} chunks")
    return chunks


def validate_prerequisites() -> bool:
    """Validate required input files exist."""
    required = [
        ("entities_semantic.jsonl", "data/processed/entities/entities_semantic.jsonl"),
        ("chunks_embedded.json", "data/processed/chunks/chunks_embedded.json"),
        ("cooccurrence_semantic.json", "data/interim/entities/cooccurrence_semantic.json"),
        ("entity_id_lookup.json", "data/interim/entities/entity_id_lookup.json"),
    ]
    
    missing = [(n, p) for n, p in required if not (PROJECT_ROOT / p).exists()]
    
    if missing:
        print("\nMISSING PREREQUISITES:")
        for name, path in missing:
            print(f"  - {name}")
        print("\nRun: python -m src.processing.relations.build_entity_cooccurrence")
        return False
    
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Phase 1D: Parallel relation extraction (v2.0)'
    )
    parser.add_argument('--workers', type=int, default=40)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--entities', type=int, default=None)
    parser.add_argument('--track', choices=['semantic', 'citation', 'all'], default='all')
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    
    if not validate_prerequisites():
        return 1
    
    from src.processing.relations.relation_extractor import RAKGRelationExtractor
    from config.extraction_config import RELATION_EXTRACTION_CONFIG
    
    config = RELATION_EXTRACTION_CONFIG.copy()
    
    print("\n" + "=" * 80)
    print("PHASE 1D: RELATION EXTRACTION (v2.0)")
    print("=" * 80)
    print(f"Track: {args.track} | Workers: {args.workers} | Limit: {args.entities or 'full'}")
    print("=" * 80 + "\n")
    
    try:
        entities = load_entities(args.track)
        chunks = load_chunks()
        
        if not entities or not chunks:
            return 1
        
        if args.entities:
            step = max(1, len(entities) // args.entities)
            entities = entities[::step][:args.entities]
        
        extractor = RAKGRelationExtractor(
            model_name=config['model_name'],
            num_chunks=config.get('chunks_per_entity', 6),
            mmr_lambda=config.get('mmr_lambda', 0.65),
            semantic_threshold=config.get('semantic_threshold', 0.85),
            max_tokens=config.get('max_tokens', 16000),
            second_round_threshold=config.get('second_round_threshold', 0.25),
            entity_lookup_file=str(PROJECT_ROOT / "data/interim/entities/entity_id_lookup.json"),
            cooccurrence_semantic_file=str(PROJECT_ROOT / "data/interim/entities/cooccurrence_semantic.json"),
            cooccurrence_concept_file=str(PROJECT_ROOT / "data/interim/entities/cooccurrence_concept.json"),
            debug_mode=args.debug
        )
        
        processor = ParallelRelationProcessor(
            extractor=extractor,
            all_chunks=chunks,
            num_workers=args.workers,
            checkpoint_freq=100,
            rate_limit_rpm=config.get('requests_per_minute', 2900),
        )
        
        estimate = processor.estimate_cost_and_time(len(entities))
        print("ESTIMATE:", estimate)
        
        if not args.resume and args.entities is None:
            if input("Proceed? [y/N]: ").lower() != 'y':
                return 0
        
        processor.process_all_entities(entities)
        return 0
    
    except KeyboardInterrupt:
        print("\n\nInterrupted. Use --resume to continue.")
        return 130
    except Exception as e:
        logger.exception("Failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())