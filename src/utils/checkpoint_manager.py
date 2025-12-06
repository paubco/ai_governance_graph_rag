# -*- coding: utf-8 -*-
"""
Checkpoint manager for parallel extraction with thread-safe operations.

Handles progress tracking and JSONL append operations for parallel relation extraction
with concurrent writes. Features O(1) append-only JSONL output, progress state tracking
(counts, cost, ETA), rolling checkpoints (keeps N most recent), thread-safe file operations,
and resume capability to skip completed entities.

Example:
    manager = CheckpointManager(output_dir=Path("data/interim/relations"))
    manager.set_start_time(datetime.now())
    manager.append_result(result_dict)
    manager.update_progress(cost=0.001, success=True)
    completed = manager.load_completed_entities()  # Resume from checkpoint
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional
from threading import Lock

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpointing for parallel relation extraction.
    
    Features:
    - JSONL append-only output (O(1) per entity)
    - Progress state tracking (counts, cost, ETA)
    - Rolling checkpoints (keep N most recent)
    - Thread-safe file operations
    - Resume capability (skip completed entities)
    
    Files managed:
    - relations_output.jsonl: Main results (append-only)
    - progress_state.json: Current progress summary
    - checkpoint_NNNN_count.json: Rolling checkpoints
    - failed_entities.json: Entities that errored
    """
    
    def __init__(
        self,
        output_dir: Path,
        checkpoint_freq: int = 100,
        max_checkpoints: int = 2
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            output_dir: Directory for all output files
            checkpoint_freq: Save checkpoint every N entities
            max_checkpoints: Number of rolling checkpoints to keep
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_freq = checkpoint_freq
        self.max_checkpoints = max_checkpoints
        
        # File paths
        self.relations_file = self.output_dir / "relations_output.jsonl"
        self.progress_file = self.output_dir / "progress_state.json"
        self.failed_file = self.output_dir / "failed_entities.json"
        
        # Thread-safe locks
        self.file_lock = Lock()
        self.progress_lock = Lock()
        
        # Progress counters
        self.completed_count = 0
        self.failed_count = 0
        self.total_cost = 0.0
        self.retry_count = 0  # Total retry attempts
        self.second_batch_count = 0  # Entities that triggered second batch
        self.start_time = None
        self.checkpoint_counter = 0
    
    def load_completed_entities(self) -> Set[str]:
        """
        Load set of already-completed entity IDs from JSONL.
        
        Enables resume capability - skip entities already processed.
        
        Returns:
            Set of entity_id strings
        """
        if not self.relations_file.exists():
            logger.info("No existing output file found - starting fresh")
            return set()
        
        completed = set()
        line_count = 0
        
        try:
            with open(self.relations_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line_count += 1
                    if line.strip():  # Skip empty lines
                        data = json.loads(line)
                        completed.add(data['entity_id'])
            
            logger.info(f"Loaded {len(completed)} completed entities from {line_count} lines")
            
        except json.JSONDecodeError as e:
            logger.error(f"Error reading JSONL at line {line_count}: {e}")
            logger.warning("Proceeding with partial resume data")
        
        return completed
    
    def append_result(self, result: Dict):
        """
        Thread-safe append of result to JSONL file.
        
        Args:
            result: Dictionary with entity extraction results
                Required keys: entity_id, entity_name, relations, cost
        """
        with self.file_lock:
            try:
                with open(self.relations_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
            except Exception as e:
                logger.error(f"Failed to append result for {result.get('entity_id')}: {e}")
                raise
    
    def update_progress(
        self,
        cost: float,
        success: bool = True,
        total_entities: Optional[int] = None,
        retries: int = 0,
        had_second_batch: bool = False
    ):
        """
        Thread-safe progress counter update.
        
        Args:
            cost: API cost for this entity (USD)
            success: True if extraction succeeded
            total_entities: Total entities to process (for ETA)
            retries: Number of retry attempts for this entity
            had_second_batch: Whether entity triggered second batch
        """
        with self.progress_lock:
            if success:
                self.completed_count += 1
                self.total_cost += cost
            else:
                self.failed_count += 1
            
            self.retry_count += retries
            if had_second_batch:
                self.second_batch_count += 1
            
            # Checkpoint at intervals
            total = self.completed_count + self.failed_count
            if total % self.checkpoint_freq == 0:
                self.save_checkpoint(total_entities)
    
    def save_checkpoint(self, total_entities: Optional[int] = None):
        """
        Save current progress state and create rolling checkpoint.
        
        Args:
            total_entities: Total entities to process (for progress %)
        """
        # Calculate statistics
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        total_processed = self.completed_count + self.failed_count
        avg_time = elapsed / total_processed if total_processed > 0 else 0
        
        # ETA calculation
        eta_sec = None
        if total_entities and total_processed > 0:
            remaining = total_entities - total_processed
            eta_sec = remaining * avg_time
        
        # Progress state
        progress = {
            'last_updated': datetime.now().isoformat(),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'total_entities': total_entities,
            'completed_entities': self.completed_count,
            'failed_entities': self.failed_count,
            'success_rate_pct': round(100 * self.completed_count / total_processed, 2) if total_processed > 0 else 0,
            'total_cost_usd': round(self.total_cost, 4),
            'retry_attempts': self.retry_count,
            'second_batch_count': self.second_batch_count,
            'avg_time_per_entity_sec': round(avg_time, 2),
            'elapsed_time_sec': round(elapsed, 1),
            'eta_remaining_sec': round(eta_sec, 1) if eta_sec else None
        }
        
        # Save progress state (small file, frequent updates)
        with self.file_lock:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, indent=2, ensure_ascii=False)
        
        # Create rolling checkpoint
        self.checkpoint_counter += 1
        checkpoint_file = self.output_dir / f"checkpoint_{self.checkpoint_counter:04d}_{total_processed}.json"
        
        with self.file_lock:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, indent=2, ensure_ascii=False)
        
        # Cleanup old checkpoints (keep only N most recent)
        self._cleanup_old_checkpoints()
        
        logger.info(
            f"Checkpoint {self.checkpoint_counter}: "
            f"{self.completed_count} completed, "
            f"{self.failed_count} failed, "
            f"{self.retry_count} retries, "
            f"{self.second_batch_count} second batches, "
            f"${self.total_cost:.2f} cost"
        )
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoint files, keeping only N most recent"""
        checkpoints = sorted(self.output_dir.glob("checkpoint_*.json"))
        
        if len(checkpoints) > self.max_checkpoints:
            for old_checkpoint in checkpoints[:-self.max_checkpoints]:
                try:
                    old_checkpoint.unlink()
                    logger.debug(f"Deleted old checkpoint: {old_checkpoint.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete {old_checkpoint.name}: {e}")
    
    def log_failure(self, entity: Dict, error: Exception):
        """
        Log failed entity for later inspection.
        
        Args:
            entity: Entity dict with id, name, type
            error: Exception that occurred
        """
        failure_record = {
            'entity_id': entity.get('entity_id'),
            'entity_name': entity.get('name'),
            'entity_type': entity.get('type'),
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        }
        
        with self.file_lock:
            # Append to failed entities file (JSONL)
            with open(self.failed_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(failure_record, ensure_ascii=False) + '\n')
        
        logger.error(f"Failed entity: {entity.get('name')} - {error}")
    
    def load_failed_entities(self) -> List[Dict]:
        """
        Load list of failed entities for manual inspection.
        
        Returns:
            List of failure records
        """
        if not self.failed_file.exists():
            return []
        
        failures = []
        with open(self.failed_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    failures.append(json.loads(line))
        
        return failures
    
    def set_start_time(self, start_time: datetime):
        """Set processing start time for ETA calculation"""
        self.start_time = start_time
    
    def get_stats(self) -> Dict:
        """
        Get current progress statistics.
        
        Returns:
            Dictionary with counts, cost, timing
        """
        with self.progress_lock:
            elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            total_processed = self.completed_count + self.failed_count
            
            return {
                'completed': self.completed_count,
                'failed': self.failed_count,
                'total_processed': total_processed,
                'cost_usd': round(self.total_cost, 4),
                'retry_attempts': self.retry_count,
                'second_batch_count': self.second_batch_count,
                'elapsed_sec': round(elapsed, 1)
            }


# Example usage
if __name__ == "__main__":
    from src.utils.logger import setup_logging
    # Test checkpoint manager
    setup_logging()
    
    manager = CheckpointManager(
        output_dir=Path("test_checkpoints"),
        checkpoint_freq=5
    )
    
    # Simulate processing
    manager.set_start_time(datetime.now())
    
    for i in range(12):
        result = {
            'entity_id': f'entity_{i}',
            'entity_name': f'Entity {i}',
            'relations': [{'subject': 'A', 'predicate': 'relates_to', 'object': 'B'}],
            'cost': 0.001
        }
        manager.append_result(result)
        manager.update_progress(cost=0.001, total_entities=12)
    
    print("\nFinal stats:", manager.get_stats())
    print("Completed entities:", len(manager.load_completed_entities()))