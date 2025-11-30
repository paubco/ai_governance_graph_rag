"""
Server Script: disambiguation_server.py
Phase: 1C - Entity Disambiguation  
Purpose: GPU-optimized disambiguation with multithreading for server deployment
Author: Pau Barba i Colomer
Created: 2025-11-29
Last Modified: 2025-11-29

**GPU Optimization**:
    - Uses faiss-gpu for GPU-accelerated HNSW index
    - Multithreaded FAISS search (4-8 workers)
    - Thread-safe with explicit locks
    - Optimized for RTX 3060 / server GPU

Thread Safety:
    - FAISS index built once (main thread)
    - Search parallelized across workers (read-only, safe)
    - Result collection with locks
    - Checkpoint saves atomic
    
Usage:
    # On server with GPU
    nohup python server_scripts/disambiguation_server.py \
        --input data/interim/entities/stage1_deduplicated_embedded.json \
        --output data/interim/entities/stage3_output.json \
        --workers 4 \
        --gpu-id 0 \
        > logs/disambiguation.log 2>&1 &
    
    # Monitor
    tail -f logs/disambiguation.log

Notes:
    - Runs Stages 2-3 only (assumes entities already deduplicated+embedded)
    - For CPU version, use src/phase1_graph_construction/disambiguation_processor.py
    - Auto-detects GPU, falls back to CPU if unavailable
"""

# Standard library
import argparse
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict, Counter
import time

# Third-party
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import local modules
from src.phase1_graph_construction.entity_disambiguator import (
    ExactDeduplicator,
    TieredThresholdFilter
)

# Load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use environment variables directly

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/disambiguation_server.log')
    ]
)
logger = logging.getLogger(__name__)


class FAISSBlockerGPU:
    """
    GPU-accelerated FAISS HNSW blocking with multithreading
    
    Thread Safety:
        - Index built once in main thread
        - Search is read-only (thread-safe)
        - Result collection protected by lock
    
    GPU Optimization:
        - Uses faiss-gpu for CUDA acceleration
        - Batched search for GPU efficiency
        - Falls back to CPU if GPU unavailable
    """
    
    def __init__(self, 
                 embedding_dim: int = 1024, 
                 M: int = 32,
                 gpu_id: int = 0):
        """
        Initialize GPU-accelerated FAISS blocker
        
        Args:
            embedding_dim: Embedding dimension (1024 for BGE-M3)
            M: HNSW connections per node
            gpu_id: CUDA device ID (0 = first GPU)
        """
        self.embedding_dim = embedding_dim
        self.M = M
        self.gpu_id = gpu_id
        self.index = None
        self.use_gpu = False
        
        # Thread safety
        self.index_lock = threading.Lock()
        self.result_lock = threading.Lock()
        
        self.stats = {
            'entities_indexed': 0,
            'candidate_pairs': 0,
            'avg_neighbors': 0.0,
            'device': 'cpu'
        }
        
        # Try to import faiss-gpu
        try:
            import faiss
            if faiss.get_num_gpus() > 0:
                self.use_gpu = True
                self.stats['device'] = f'cuda:{gpu_id}'
                logger.info(f"GPU detected: Using CUDA device {gpu_id}")
            else:
                logger.warning("No GPU detected, falling back to CPU")
        except Exception as e:
            logger.warning(f"GPU initialization failed: {e}, using CPU")
    
    def build_index(self, entities: List[Dict], ef_construction: int = 200):
        """
        Build HNSW index on GPU (if available)
        
        Thread Safety: Called once in main thread before parallel search
        
        Args:
            entities: List of entities with 'embedding' field
            ef_construction: HNSW parameter (higher = better recall)
        """
        import faiss
        
        logger.info(f"Building FAISS HNSW index on {self.stats['device']}...")
        
        # Extract embeddings
        embeddings = np.array([e['embedding'] for e in entities]).astype('float32')
        
        # CRITICAL: Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create CPU index first
        index_cpu = faiss.IndexHNSWFlat(self.embedding_dim, self.M)
        index_cpu.hnsw.efConstruction = ef_construction
        index_cpu.add(embeddings)
        
        # Move to GPU if available
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, self.gpu_id, index_cpu)
                logger.info(f"Index moved to GPU {self.gpu_id}")
            except Exception as e:
                logger.warning(f"GPU transfer failed: {e}, using CPU index")
                self.index = index_cpu
                self.use_gpu = False
                self.stats['device'] = 'cpu'
        else:
            self.index = index_cpu
        
        self.stats['entities_indexed'] = len(entities)
        
        logger.info(f"FAISS index built:")
        logger.info(f"  Device: {self.stats['device']}")
        logger.info(f"  Entities: {len(entities)}")
        logger.info(f"  M: {self.M}, ef_construction: {ef_construction}")
    
    def find_candidates_batch(self,
                              entities: List[Dict],
                              batch_indices: List[int],
                              k: int = 50,
                              ef_search: int = 64) -> List[Tuple[int, int, float]]:
        """
        Find candidates for a batch of entities (thread-safe worker function)
        
        Thread Safety:
            - Index is read-only (safe for concurrent reads)
            - Returns results to main thread
        
        Args:
            entities: Full entity list
            batch_indices: Indices of entities to process in this batch
            k: Number of neighbors
            ef_search: HNSW search parameter
            
        Returns:
            List of (i, j, similarity) tuples for this batch
        """
        import faiss
        
        # Get embeddings for this batch
        batch_embeddings = np.array([entities[i]['embedding'] for i in batch_indices]).astype('float32')
        faiss.normalize_L2(batch_embeddings)
        
        # Search (read-only operation, thread-safe)
        with self.index_lock:  # Lock for GPU safety
            self.index.hnsw.efSearch = ef_search
            distances, indices = self.index.search(batch_embeddings, k + 1)
        
        # Convert to pairs
        pairs = []
        for batch_idx, (i, (dists, neighbors)) in enumerate(zip(batch_indices, zip(distances, indices))):
            for neighbor_idx, dist in zip(neighbors, dists):
                if neighbor_idx == i:  # Skip self
                    continue
                
                # Convert L2 distance to cosine similarity
                similarity = float(1 - (dist ** 2) / 2)
                
                # Filter out dissimilar pairs (negative cosine similarity)
                if similarity < 0.0:
                    continue
                
                # Only keep unique pairs (i < j)
                if i < neighbor_idx:
                    pairs.append((i, neighbor_idx, similarity))
        
        return pairs
    
    def find_candidates_parallel(self,
                                entities: List[Dict],
                                k: int = 50,
                                ef_search: int = 64,
                                num_workers: int = 4) -> List[Tuple[int, int, float]]:
        """
        Find candidates with parallel search across workers
        
        Thread Safety:
            - Index read-only (safe)
            - Each worker processes different entity batch
            - Results collected with lock
        
        Args:
            entities: List of entities
            k: Number of neighbors per entity
            ef_search: HNSW search parameter
            num_workers: Number of parallel threads
            
        Returns:
            List of unique (i, j, similarity) tuples
        """
        logger.info(f"Searching k={k} neighbors with {num_workers} workers...")
        
        # Split entities into batches for workers
        n_entities = len(entities)
        batch_size = (n_entities + num_workers - 1) // num_workers
        batches = [
            list(range(i, min(i + batch_size, n_entities)))
            for i in range(0, n_entities, batch_size)
        ]
        
        logger.info(f"Split {n_entities} entities into {len(batches)} batches")
        
        # Process batches in parallel
        all_pairs = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self.find_candidates_batch, entities, batch, k, ef_search)
                for batch in batches
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="FAISS search"):
                batch_pairs = future.result()
                with self.result_lock:
                    all_pairs.extend(batch_pairs)
        
        # Remove duplicates (same pair from different batches)
        unique_pairs = list(set(all_pairs))
        
        self.stats['candidate_pairs'] = len(unique_pairs)
        self.stats['avg_neighbors'] = len(unique_pairs) / len(entities) if entities else 0
        
        logger.info(f"Found {len(unique_pairs)} unique candidate pairs")
        logger.info(f"Average neighbors per entity: {self.stats['avg_neighbors']:.1f}")
        
        return unique_pairs


class SameJudgeGPU:
    """
    Stage 4: Multithreaded LLM-based entity verification (GPU version)
    
    Thread Safety:
        - Each worker gets own API client
        - Checkpoint saves protected by lock
        - Result collection via as_completed() (atomic)
    
    Optimization:
        - 8 parallel workers for GPU inference
        - Exponential backoff retry
        - Atomic checkpoint saves
        - Thread-safe result collection
    
    References:
        - Zhang et al. (2025) RAKG - SameJudge methodology
    """
    
    def __init__(self, 
                 model: str = "Qwen/Qwen2-7B-Instruct",
                 api_key: str = None,
                 num_workers: int = 8,
                 checkpoint_interval: int = 100):
        """
        Initialize GPU-optimized SameJudge
        
        Args:
            model: LLM model identifier
            api_key: Together.ai API key
            num_workers: Number of parallel threads (8 for GPU)
            checkpoint_interval: Save checkpoint every N pairs
        """
        self.model = model
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        self.num_workers = num_workers
        self.checkpoint_interval = checkpoint_interval
        
        # Thread safety
        self.checkpoint_lock = threading.Lock()
        
        self.stats = {
            'pairs_verified': 0,
            'matches_found': 0,
            'total_cost': 0.0
        }
        self.cost_per_call = 0.00003
        
        logger.info(f"SameJudgeGPU initialized:")
        logger.info(f"  Model: {model}")
        logger.info(f"  Workers: {num_workers}")
        logger.info(f"  Thread-safe: Yes (with locks)")
    
    def _get_client(self):
        """Get thread-local API client"""
        from together import Together
        return Together(api_key=self.api_key)
    
    def verify_pair(self, i: int, j: int, entities: List[Dict]) -> Tuple[int, int, bool]:
        """
        Verify a single pair (thread-safe worker function)
        
        Args:
            i, j: Entity indices
            entities: Full entity list (read-only, thread-safe)
            
        Returns:
            (i, j, is_same) tuple
        """
        # Import prompt from centralized location (FAIL HARD if missing)
        from src.prompts.prompts import SAMEJUDGE_PROMPT
        
        entity1 = entities[i]
        entity2 = entities[j]
        
        prompt = SAMEJUDGE_PROMPT.format(
            entity1_name=entity1['name'],
            entity1_type=entity1['type'],
            entity1_desc=entity1.get('description', 'N/A')[:150],
            entity2_name=entity2['name'],
            entity2_type=entity2['type'],
            entity2_desc=entity2.get('description', 'N/A')[:150]
        )
        
        # Thread-local client
        client = self._get_client()
        
        # Retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=300
                )
                
                response_text = response.choices[0].message.content.strip()
                response_text = response_text.replace('```json', '').replace('```', '').strip()
                result = json.loads(response_text)
                
                is_same = result.get('result', False)
                return (i, j, is_same)
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    logger.error(f"Failed pair {i}-{j}: {e}")
                    return (i, j, False)  # Default to not matching on error
    
    def verify_batch(self, 
                    uncertain_pairs: List[Tuple[int, int, float]],
                    entities: List[Dict],
                    checkpoint_path: Path = None) -> List[Tuple[int, int]]:
        """
        Verify batch of uncertain pairs with multithreading
        
        Thread Safety:
            - Entities list is read-only (safe)
            - Each worker processes different pairs
            - Results collected atomically
            - Checkpoints saved with lock
        
        Args:
            uncertain_pairs: List of (i, j, similarity) tuples
            entities: List of entities (read-only)
            checkpoint_path: Optional checkpoint file
            
        Returns:
            List of (i, j) pairs confirmed as matches
        """
        logger.info(f"Stage 4: Verifying {len(uncertain_pairs)} pairs with {self.num_workers} workers...")
        logger.info(f"Estimated cost: ${len(uncertain_pairs) * self.cost_per_call:.2f}")
        
        matches = []
        processed = 0
        start_time = time.time()
        
        # Check for checkpoint
        if checkpoint_path and checkpoint_path.exists():
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            processed = checkpoint.get('processed', 0)
            matches = checkpoint.get('matches', [])
            logger.info(f"Resuming from: {processed} pairs processed")
        
        # Process remaining
        remaining_pairs = uncertain_pairs[processed:]
        
        # Multithreaded processing
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            futures = [
                executor.submit(self.verify_pair, i, j, entities)
                for i, j, sim in remaining_pairs
            ]
            
            # Collect results as they complete (main thread - atomic)
            with tqdm(total=len(remaining_pairs), desc="SameJudge GPU") as pbar:
                for future in as_completed(futures):
                    i, j, is_same = future.result()
                    
                    if is_same:
                        matches.append((i, j))
                    
                    processed += 1
                    pbar.update(1)
                    
                    # Checkpoint with lock
                    if checkpoint_path and processed % self.checkpoint_interval == 0:
                        with self.checkpoint_lock:
                            self._save_checkpoint(checkpoint_path, matches, processed, len(uncertain_pairs))
        
        # Final checkpoint
        if checkpoint_path:
            with self.checkpoint_lock:
                self._save_checkpoint(checkpoint_path, matches, processed, len(uncertain_pairs), final=True)
        
        elapsed = time.time() - start_time
        self.stats['pairs_verified'] = len(uncertain_pairs)
        self.stats['matches_found'] = len(matches)
        self.stats['total_cost'] = len(uncertain_pairs) * self.cost_per_call
        
        logger.info(f"LLM verification complete:")
        logger.info(f"  Pairs: {len(uncertain_pairs)}")
        logger.info(f"  Matches: {len(matches)} ({100 * len(matches) / len(uncertain_pairs):.1f}%)")
        logger.info(f"  Time: {elapsed/60:.1f} minutes")
        logger.info(f"  Cost: ${self.stats['total_cost']:.2f}")
        
        return matches
    
    def _save_checkpoint(self, path: Path, matches: List, processed: int, total: int, final: bool = False):
        """Save checkpoint (called with lock held)"""
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'processed': processed,
            'total': total,
            'matches': matches,
            'final': final
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Atomic write
        temp_path = path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        temp_path.replace(path)


class DisambiguationServerProcessor:
    """
    GPU-optimized orchestrator for full 4-stage pipeline
    
    Assumes:
        - Entities already deduplicated+embedded (Stage 1 complete)
        - Running on server with GPU
        - Needs fast processing for large datasets
    
    Stages:
        Stage 2: FAISS GPU blocking (4 workers)
        Stage 3: Tiered thresholds
        Stage 4: SameJudge GPU (8 workers)
    """
    
    def __init__(self, 
                 data_dir: str = "data/interim/entities",
                 faiss_workers: int = 4,
                 samejudge_workers: int = 8,
                 gpu_id: int = 0,
                 together_api_key: str = None):
        """
        Initialize server processor
        
        Args:
            data_dir: Directory for intermediate files
            faiss_workers: Workers for FAISS search (4 recommended)
            samejudge_workers: Workers for LLM calls (8 recommended)
            gpu_id: CUDA device ID
            together_api_key: API key for Stage 4
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize GPU-optimized stages
        self.stage2 = FAISSBlockerGPU(gpu_id=gpu_id)
        self.stage3 = TieredThresholdFilter()
        self.stage4 = SameJudgeGPU(
            api_key=together_api_key,
            num_workers=samejudge_workers
        )
        
        self.faiss_workers = faiss_workers
        self.stats = {}
        
        logger.info(f"Server processor initialized:")
        logger.info(f"  FAISS workers: {faiss_workers}")
        logger.info(f"  SameJudge workers: {samejudge_workers}")
        logger.info(f"  GPU: {gpu_id}")
    
    def load_json(self, filepath: str) -> List[Dict]:
        """Load JSON file (handles metadata+entities structure)"""
        logger.info(f"Loading {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle your actual format: {"metadata": {...}, "entities": [...]}
        if isinstance(data, dict):
            if 'entities' in data:
                logger.info("Extracting entities array from file...")
                data = data['entities']
            else:
                # Fallback: entity IDs as keys
                logger.info("Converting dict format to list...")
                data = list(data.values())
        
        logger.info(f"Loaded {len(data)} entities")
        return data
    
    def save_json(self, data: List[Dict], filepath: str):
        """Save JSON file"""
        logger.info(f"Saving to {filepath}...")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(data)} items")
    
    def run_stage2(self, entities: List[Dict]) -> List[tuple]:
        """Run Stage 2: GPU-accelerated FAISS blocking"""
        logger.info("="*60)
        logger.info("STAGE 2: FAISS BLOCKING (GPU)")
        logger.info("="*60)
        
        # Build index
        self.stage2.build_index(entities)
        
        # Parallel search
        pairs = self.stage2.find_candidates_parallel(
            entities,
            k=50,
            num_workers=self.faiss_workers
        )
        
        # Save intermediate result
        output_file = self.data_dir / "stage2_candidate_pairs.json"
        pairs_list = [[int(i), int(j), float(sim)] for i, j, sim in pairs]
        with open(output_file, 'w') as f:
            json.dump(pairs_list, f)
        logger.info(f"Saved {len(pairs)} candidate pairs")
        
        self.stats['stage2'] = self.stage2.stats
        
        return pairs
    
    def run_stage3(self, 
                   pairs: List[tuple], 
                   entities: List[Dict]) -> Tuple[Dict, List[Dict]]:
        """Run Stage 3: Tiered threshold filtering"""
        logger.info("="*60)
        logger.info("STAGE 3: TIERED THRESHOLD FILTERING")
        logger.info("="*60)
        
        filtered = self.stage3.filter_pairs(pairs, entities)
        
        # Apply auto-merges
        logger.info("Applying auto-merge decisions...")
        entities = self.stage3.apply_merges(entities, filtered['merged'])
        
        # Save intermediate results
        output_file = self.data_dir / "stage3_filtered_pairs.json"
        filtered_json = {
            'merged': [[int(i), int(j)] for i, j in filtered['merged']],
            'rejected': [[int(i), int(j)] for i, j in filtered['rejected']],
            'uncertain': [[int(i), int(j), float(sim)] for i, j, sim in filtered['uncertain']]
        }
        with open(output_file, 'w') as f:
            json.dump(filtered_json, f, indent=2)
        
        # Save entities after auto-merges
        output_file = self.data_dir / "stage3_entities_after_automerge.json"
        self.save_json(entities, str(output_file))
        
        self.stats['stage3'] = self.stage3.stats
        
        return filtered, entities
    
    
    def run_stage4(self, 
                   uncertain_pairs: List[tuple],
                   entities: List[Dict]) -> List[Dict]:
        """Run Stage 4: GPU-accelerated LLM verification"""
        logger.info("="*60)
        logger.info("STAGE 4: LLM VERIFICATION (SAMEJUDGE GPU)")
        logger.info("="*60)
        
        # Verify uncertain pairs with multithreaded LLM
        checkpoint_path = self.data_dir / "samejudge_checkpoint.json"
        llm_matches = self.stage4.verify_batch(
            uncertain_pairs, 
            entities,
            checkpoint_path=checkpoint_path
        )
        
        # Apply LLM match decisions
        logger.info("Applying LLM match decisions...")
        entities = self.stage3.apply_merges(entities, llm_matches)
        
        self.stats['stage4'] = self.stage4.stats
        
        return entities
    
    def run_pipeline(self, input_file: str, output_file: str = None) -> Dict:
        """Execute full 4-stage GPU-optimized pipeline"""
        logger.info("="*60)
        logger.info("GPU-OPTIMIZED DISAMBIGUATION (ALL 4 STAGES)")
        logger.info("="*60)
        
        # Load entities (already deduplicated+embedded)
        entities = self.load_json(input_file)
        
        # Verify embeddings exist
        if 'embedding' not in entities[0]:
            raise ValueError("Entities missing embeddings! Run Stage 1 first.")
        
        # Stage 2: GPU FAISS blocking
        pairs = self.run_stage2(entities)
        
        # Stage 3: Tiered thresholds
        filtered, entities = self.run_stage3(pairs, entities)
        
        # Stage 4: GPU LLM verification
        entities = self.run_stage4(filtered['uncertain'], entities)
        
        # Save final output
        if output_file:
            self.save_json(entities, output_file)
        else:
            output_file = self.data_dir / "normalized_entities.json"
            self.save_json(entities, str(output_file))
        
        # Compile stats
        self.stats['final_entity_count'] = len(entities)
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETE!")
        logger.info("="*60)
        logger.info(f"Final entity count: {len(entities)}")
        logger.info("")
        logger.info("Statistics:")
        for stage, stats in self.stats.items():
            logger.info(f"  {stage}: {stats}")
        
        return self.stats


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="GPU-Optimized Entity Disambiguation (All 4 Stages)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run on server with GPU (all 4 stages)
    export TOGETHER_API_KEY="your-key"
    nohup python server_scripts/disambiguation_server.py \\
        --input data/interim/entities/stage1_deduplicated_embedded.json \\
        --output data/interim/entities/normalized_entities.json \\
        --faiss-workers 4 \\
        --samejudge-workers 8 \\
        --gpu-id 0 \\
        > logs/disambiguation.log 2>&1 &
    
    # Monitor
    tail -f logs/disambiguation.log

Note: For CPU version, use src/phase1_graph_construction/disambiguation_processor.py
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input file (stage1_deduplicated_embedded.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/interim/entities/normalized_entities.json',
        help='Output file (normalized_entities.json)'
    )
    parser.add_argument(
        '--faiss-workers',
        type=int,
        default=4,
        help='Workers for FAISS search (default: 4)'
    )
    parser.add_argument(
        '--samejudge-workers',
        type=int,
        default=8,
        help='Workers for SameJudge LLM (default: 8)'
    )
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='CUDA device ID (default: 0)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='Together.ai API key (or set TOGETHER_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    # Get API key
    import os
    api_key = args.api_key or os.getenv('TOGETHER_API_KEY')
    if not api_key:
        logger.error("No API key provided! Set TOGETHER_API_KEY or use --api-key")
        sys.exit(1)
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("GPU-Optimized Disambiguation Server (All 4 Stages)")
    logger.info("=" * 70)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"FAISS workers: {args.faiss_workers}")
    logger.info(f"SameJudge workers: {args.samejudge_workers}")
    logger.info(f"GPU ID: {args.gpu_id}")
    
    # Run pipeline
    processor = DisambiguationServerProcessor(
        faiss_workers=args.faiss_workers,
        samejudge_workers=args.samejudge_workers,
        gpu_id=args.gpu_id,
        together_api_key=api_key
    )
    
    try:
        start_time = time.time()
        stats = processor.run_pipeline(
            input_file=args.input,
            output_file=args.output
        )
        elapsed = time.time() - start_time
        
        logger.info(f"\nCompleted in {elapsed/60:.1f} minutes")
        logger.info(f"Output saved to: {args.output}")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()