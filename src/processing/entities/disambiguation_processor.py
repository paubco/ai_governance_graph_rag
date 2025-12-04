"""
Entity Disambiguation Server - GPU Version

GPU-optimized pipeline for Phase 1C entity disambiguation.
Runs all 4 stages: dedup -> embed -> FAISS -> thresholds -> LLM

CURRENT WORKFLOW (December 2025):
    1. Phase 1B outputs: pre_entities.json (~143k raw entities)
    2. Phase 1C-0 filters: pre_entities_clean.json (~21k clean entities) 
    3. THIS SCRIPT processes: pre_entities_clean.json -> normalized_entities.json
    
    Default: Start from Stage 1 (dedup raw entities)
    Recommended: Use pre_entities_clean.json as input for better quality

STAGES:
    Stage 1:   Exact deduplication (name normalization)
    Stage 1.5: BGE-M3 embedding (1024-dim, GPU-accelerated)
    Stage 2:   FAISS HNSW blocking (GPU, parallel search)
    Stage 3:   Tiered threshold filtering (auto-merge high confidence)
    Stage 4:   SameJudge LLM verification (multithreaded API calls)

WORKER CONFIGURATION:
    --faiss-workers: Parallel threads for FAISS search (recommended: 4-8)
        - Higher = faster search but more memory
        - 4 workers good for RTX 3060 (12GB VRAM)
        
    --samejudge-workers: Parallel threads for LLM calls (recommended: 8-12)
        - Higher = faster verification but more API concurrency
        - 8 workers = ~480 pairs/min with Together.ai
        - Limited by API rate limits, not compute

STAGE CONTROL:
    --start-from-stage: Where to begin processing
        1 = Start from raw entities (run all 4 stages)
        2 = Start from embedded entities (skip dedup+embed, run FAISS+LLM)
        
    --stop-at-stage: Where to stop (for debugging/inspection)
        1 = Stop after dedup (check entity counts)
        2 = Stop after embedding (verify embeddings)
        3 = Stop after thresholds (inspect auto-merges)
        4 = Complete pipeline (default)

This is the production GPU-accelerated version for Phase 1C disambiguation.
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
from src.processing.entities.entity_disambiguator import (
    ExactDeduplicator,
    TieredThresholdFilter,
    normalize_key
)
from src.utils.embedder import BGEEmbedder
from src.utils.embed_processor import EmbedProcessor

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
        
        # Set efSearch ONCE here for all subsequent searches (thread-safe)
        # Higher efSearch = better recall but slower (64 is good default)
        if hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = 64
        
        self.stats['entities_indexed'] = len(entities)
        
        logger.info(f"FAISS index built:")
        logger.info(f"  Device: {self.stats['device']}")
        logger.info(f"  Entities: {len(entities)}")
        logger.info(f"  M: {self.M}, ef_construction: {ef_construction}, efSearch: 64")
    
    def find_candidates_batch(self,
                              entities: List[Dict],
                              batch_indices: List[int],
                              k: int = 50) -> List[Dict]:
        """
        Find candidates for a batch of entities (thread-safe worker function)
        
        Returns pairs with entity KEYS instead of indices.
        
        Thread Safety:
            - Index search is read-only (thread-safe, no lock needed)
            - efSearch set once in build_index()
            - Each worker processes different batch
        
        Args:
            entities: Full entity list (must have _index field!)
            batch_indices: Indices of entities to process in this batch
            k: Number of neighbors
            
        Returns:
            List of pair dicts with entity1_key, entity2_key, similarity
        """
        import faiss
        from src.processing.entities.entity_disambiguator import get_entity_key
        
        # Get embeddings for this batch
        batch_embeddings = np.array([entities[i]['embedding'] for i in batch_indices]).astype('float32')
        faiss.normalize_L2(batch_embeddings)
        
        # Search (read-only, thread-safe - no lock needed!)
        # efSearch already set in build_index()
        distances, indices = self.index.search(batch_embeddings, k + 1)
        
        # Convert to pairs with entity KEYS
        pairs = []
        seen_pairs = set()  # Track unique pairs using entity keys
        
        for batch_idx, (i, (dists, neighbors)) in enumerate(zip(batch_indices, zip(distances, indices))):
            entity1 = entities[i]
            entity1_key = get_entity_key(entity1)
            
            for neighbor_idx, dist in zip(neighbors, dists):
                if neighbor_idx == i:  # Skip self
                    continue
                
                entity2 = entities[neighbor_idx]
                entity2_key = get_entity_key(entity2)
                
                # Convert L2 distance to cosine similarity
                similarity = float(1 - (dist ** 2) / 2)
                
                # Filter out dissimilar pairs (negative cosine similarity)
                if similarity < 0.0:
                    continue
                
                # Only keep unique pairs (use sorted keys)
                pair_key = tuple(sorted([entity1_key, entity2_key]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    pairs.append({
                        'entity1_key': entity1_key,
                        'entity2_key': entity2_key,
                        'similarity': similarity
                    })
        
        return pairs
    
    def find_candidates_parallel(self,
                                entities: List[Dict],
                                k: int = 50,
                                num_workers: int = 4) -> List[Dict]:
        """
        Find candidates with parallel search across workers
        
        Returns pairs with entity KEYS - much cleaner than indices!
        
        Thread Safety:
            - Index search is read-only and thread-safe
            - Each worker processes different entity batch
            - No locks needed for search (efSearch set in build_index)
        
        Args:
            entities: List of entities
            k: Number of neighbors per entity
            num_workers: Number of parallel threads
            
        Returns:
            List of pair dicts with entity1_key, entity2_key, similarity
        """
        logger.info(f"Searching k={k} neighbors with {num_workers} workers...")
        
        # TEMPORARY: Add _index for FAISS operations only
        for i, entity in enumerate(entities):
            entity['_index'] = i
        
        # Split entities into batches for workers
        n_entities = len(entities)
        batch_size = (n_entities + num_workers - 1) // num_workers
        batches = [
            list(range(i, min(i + batch_size, n_entities)))
            for i in range(0, n_entities, batch_size)
        ]
        
        logger.info(f"Split {n_entities} entities into {len(batches)} batches")
        
        # Process batches in parallel (TRUE parallelism - no locks!)
        all_pairs = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self.find_candidates_batch, entities, batch, k)
                for batch in batches
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="FAISS search"):
                batch_pairs = future.result()
                all_pairs.extend(batch_pairs)
        
        # Remove duplicates using entity keys
        unique_pairs_dict = {}
        for pair in all_pairs:
            # Create unique key from sorted entity keys
            pair_key = tuple(sorted([pair['entity1_key'], pair['entity2_key']]))
            if pair_key not in unique_pairs_dict:
                unique_pairs_dict[pair_key] = pair
        
        unique_pairs = list(unique_pairs_dict.values())
        
        # CLEANUP: Remove temporary _index field
        for entity in entities:
            entity.pop('_index', None)
        
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
                 model: str = "Qwen/Qwen2.5-7B-Instruct-Turbo",
                 api_key: str = None,
                 num_workers: int = 8,
                 checkpoint_interval: int = 500):
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
    
    def _parse_llm_response(self, response_text: str):
        """
        Robust JSON parsing with regex fallback
        
        Handles: markdown blocks, unescaped quotes/backslashes, malformed JSON
        Returns: bool (is_same) or None if parsing failed
        """
        import re
        
        # Strategy 1: Clean and parse JSON
        try:
            text = response_text.replace('```json', '').replace('```', '').strip()
            if '{' in text and '}' in text:
                text = text[text.find('{'):text.rfind('}')+1]
            result = json.loads(text)
            return result.get('result', False)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Regex extraction of "result" field
        try:
            match = re.search(r'"result"\s*:\s*(true|false)', response_text, re.IGNORECASE)
            if match:
                return match.group(1).lower() == 'true'
        except Exception:
            pass
        
        # Strategy 3: Look for YES/NO patterns in text
        try:
            lower_text = response_text.lower()
            if '"result": true' in lower_text or 'same entity' in lower_text:
                return True
            if '"result": false' in lower_text or 'different entit' in lower_text:
                return False
        except Exception:
            pass
        
        return None  # Parsing failed
    
    def verify_pair(self, 
                    entity1_key: Tuple[str, str], 
                    entity2_key: Tuple[str, str],
                    entity_map: Dict[Tuple[str, str], Dict]) -> Tuple[Tuple[str, str], Tuple[str, str], bool]:
        """
        Verify a single pair (thread-safe worker function)
        
        Uses entity KEYS instead of indices - much cleaner!
        
        Note: LLM can decide that entities with different types are actually the same
        (e.g., "EU" [Organization] vs "European Union" [Political Union])
        
        Args:
            entity1_key: (name, type) tuple for first entity
            entity2_key: (name, type) tuple for second entity
            entity_map: Entity lookup dict (read-only, thread-safe)
            
        Returns:
            (entity1_key, entity2_key, is_same) tuple
        """
        # Defensive key check (should be pre-filtered, but just in case)
        if entity1_key not in entity_map or entity2_key not in entity_map:
            return (entity1_key, entity2_key, False)
        
        entity1 = entity_map[entity1_key]
        entity2 = entity_map[entity2_key]
        
        # Import prompt from centralized location (FAIL HARD if missing)
        from src.prompts.prompts import SAMEJUDGE_PROMPT
        
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
                
                # Use robust parser with fallbacks
                is_same = self._parse_llm_response(response_text)
                
                if is_same is None:
                    # Parsing failed even with fallbacks
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        logger.error(f"Parse failed {entity1_key}-{entity2_key}: {response_text[:80]}")
                        return (entity1_key, entity2_key, False)
                
                return (entity1_key, entity2_key, is_same)
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    logger.error(f"Failed pair {entity1_key}-{entity2_key}: {e}")
                    return (entity1_key, entity2_key, False)  # Default to not matching on error
    
    def verify_batch(self, 
                    uncertain_pairs: List[Dict],
                    entities: List[Dict],
                    checkpoint_path: Path = None) -> List[Dict]:
        """
        Verify batch of uncertain pairs with multithreading
        
        Works with entity KEYS - much cleaner than indices!
        Pre-filters pairs with missing keys (from Stage 3 merges).
        
        Thread Safety:
            - Entity map is read-only (safe)
            - Each worker processes different pairs
            - Results collected atomically
            - Checkpoints saved with lock
        
        Args:
            uncertain_pairs: List of pair dicts with entity1_key, entity2_key, similarity
            entities: List of entities (for building map)
            checkpoint_path: Optional checkpoint file
            
        Returns:
            List of pair dicts confirmed as matches
        """
        from src.processing.entities.entity_disambiguator import build_entity_map
        
        # Build entity map once for all workers (read-only, thread-safe)
        entity_map = build_entity_map(entities)
        
        # Pre-filter pairs: remove those with missing keys (merged in Stage 3)
        valid_pairs = []
        skipped = 0
        for pair in uncertain_pairs:
            key1 = pair['entity1_key']
            key2 = pair['entity2_key']
            if key1 in entity_map and key2 in entity_map:
                valid_pairs.append(pair)
            else:
                skipped += 1
        
        if skipped > 0:
            logger.warning(f"Skipped {skipped} pairs with missing entity keys (merged in Stage 3)")
        
        logger.info(f"Stage 4: Verifying {len(valid_pairs)} pairs with {self.num_workers} workers...")
        logger.info(f"Estimated cost: ${len(valid_pairs) * self.cost_per_call:.2f}")
        
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
            
            # Normalize keys from JSON (tuples become lists in JSON)
            matches = [
                {
                    'entity1_key': normalize_key(m['entity1_key']),
                    'entity2_key': normalize_key(m['entity2_key'])
                }
                for m in matches
            ]
            
            logger.info(f"Resuming from: {processed} pairs processed")
        
        # Process remaining
        remaining_pairs = valid_pairs[processed:]
        
        if not remaining_pairs:
            logger.info("No pairs to process")
            return matches
        
        # Multithreaded processing
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            futures = [
                executor.submit(self.verify_pair, pair['entity1_key'], pair['entity2_key'], entity_map)
                for pair in remaining_pairs
            ]
            
            # Collect results as they complete (main thread - atomic)
            with tqdm(total=len(remaining_pairs), desc="SameJudge") as pbar:
                for future in as_completed(futures):
                    entity1_key, entity2_key, is_same = future.result()
                    
                    if is_same:
                        matches.append({
                            'entity1_key': entity1_key,
                            'entity2_key': entity2_key
                        })
                    
                    processed += 1
                    pbar.update(1)
                    
                    # Progress logging every 500 pairs
                    if processed % 500 == 0:
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else 0
                        eta = (len(remaining_pairs) - processed) / rate if rate > 0 else 0
                        logger.info(
                            f"Progress: {processed}/{len(remaining_pairs)} "
                            f"({100*processed/len(remaining_pairs):.1f}%) | "
                            f"Matches: {len(matches)} | Rate: {rate:.1f}/s | ETA: {eta/60:.1f}min"
                        )
                    
                    # Checkpoint with lock
                    if checkpoint_path and processed % self.checkpoint_interval == 0:
                        with self.checkpoint_lock:
                            self._save_checkpoint(checkpoint_path, matches, processed, len(valid_pairs))
        
        # Final checkpoint
        if checkpoint_path:
            with self.checkpoint_lock:
                self._save_checkpoint(checkpoint_path, matches, processed, len(valid_pairs), final=True)
        
        elapsed = time.time() - start_time
        self.stats['pairs_verified'] = processed
        self.stats['matches_found'] = len(matches)
        self.stats['total_cost'] = processed * self.cost_per_call
        
        logger.info(f"LLM verification complete:")
        logger.info(f"  Pairs processed: {processed}/{len(valid_pairs)}")
        logger.info(f"  Matches: {len(matches)} ({100 * len(matches) / processed if processed > 0 else 0:.1f}%)")
        logger.info(f"  Time: {elapsed/60:.1f} minutes")
        logger.info(f"  Rate: {processed/elapsed if elapsed > 0 else 0:.1f} pairs/sec")
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

CURRENT WORKFLOW:
    Input: pre_entities_clean.json (after Phase 1C-0 filtering)
    Output: normalized_entities.json (canonical entities ready for Phase 1D)
    
    Processing: ~21k clean entities -> ~18-20k normalized entities
    Time: 4-6 hours (with GPU + 8 LLM workers)
    Cost: ~$6-8 for SameJudge API calls

RECOMMENDED CONFIGURATION (RTX 3060, 12GB VRAM):
    - faiss_workers: 4 (parallel FAISS search)
    - samejudge_workers: 8 (parallel LLM calls)
    - gpu_id: 0 (first GPU)

STAGES:
    Stage 1:   Exact dedup (name normalization)
    Stage 1.5: BGE-M3 embedding (GPU-accelerated)
    Stage 2:   FAISS GPU blocking (4 workers)
    Stage 3:   Tiered thresholds (auto-merge)
    Stage 4:   SameJudge GPU (8 workers, Together.ai)

STAGE CONTROL:
    Use --start-from-stage and --stop-at-stage for partial runs:
    - Start from 1: Process raw entities (all stages)
    - Start from 2: Skip dedup+embed (if already done)
    - Stop at 1/2/3: Inspect intermediate results
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
        
        # Initialize ALL stages (1-4) for comprehensive pipeline
        self.stage1 = ExactDeduplicator()
        self.embedder = BGEEmbedder()  # BGE-M3 1024-dim
        self.embed_processor = EmbedProcessor(
            embedder=self.embedder,
            checkpoint_freq=1000
        )
        self.stage2 = FAISSBlockerGPU(gpu_id=gpu_id)
        self.stage3 = TieredThresholdFilter()
        self.stage4 = SameJudgeGPU(
            api_key=together_api_key,
            num_workers=samejudge_workers
        )
        
        self.faiss_workers = faiss_workers
        self.stats = {}
        
        logger.info(f"GPU server initialized (ALL 4 stages):")
        logger.info(f"  FAISS workers: {faiss_workers}")
        logger.info(f"  SameJudge workers: {samejudge_workers}")
        logger.info(f"  GPU: {gpu_id}")
    
    def load_json(self, filepath: str) -> List[Dict]:
        """Load JSON file (handles nested chunk+entities structure from Phase 1A)"""
        logger.info(f"Loading {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle Phase 1A format: {"metadata": {...}, "entities": [chunks]}
        if isinstance(data, dict):
            if 'entities' in data:
                logger.info("Extracting entities from chunks...")
                chunks = data['entities']
                
                # Flatten: each chunk has nested 'entities' array
                entities = []
                for chunk in chunks:
                    if isinstance(chunk, dict) and 'entities' in chunk:
                        # This is a chunk with nested entities
                        entities.extend(chunk['entities'])
                    else:
                        # Direct entity
                        entities.append(chunk)
                
                data = entities
                logger.info(f"Flattened {len(chunks)} chunks into {len(entities)} entities")
            else:
                # Fallback: entity IDs as keys
                logger.info("Converting dict format to list...")
                data = list(data.values())
        
        logger.info(f"Loaded {len(data)} entities")
        return data
    
    def save_json(self, data: List[Dict], filepath: str):
        """Save JSON file with numpy array handling"""
        logger.info(f"Saving to {filepath}...")
        
        # Convert numpy arrays to lists for JSON serialization
        data_serializable = []
        for item in data:
            item_copy = item.copy()
            if 'embedding' in item_copy and isinstance(item_copy['embedding'], np.ndarray):
                item_copy['embedding'] = item_copy['embedding'].tolist()
            data_serializable.append(item_copy)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_serializable, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(data)} items")
    
    def run_stage1(self, input_file: str) -> List[Dict]:
        """
        Run Stage 1: Exact deduplication
        
        Args:
            input_file: Path to pre_entities.json
            
        Returns:
            Deduplicated entities
        """
        logger.info("="*60)
        logger.info("STAGE 1: EXACT DEDUPLICATION")
        logger.info("="*60)
        
        entities = self.load_json(input_file)
        entities = self.stage1.deduplicate(entities)
        
        # Save intermediate result
        output_file = self.data_dir / "stage1_deduplicated.json"
        self.save_json(entities, str(output_file))
        
        self.stats['stage1'] = self.stage1.stats
        
        return entities
    
    def run_stage1_5(self, entities: List[Dict]) -> List[Dict]:
        """
        Run Stage 1.5: Embed deduplicated entities using BGE-M3
        
        Args:
            entities: Deduplicated entities (without embeddings)
            
        Returns:
            Entities with embeddings
        """
        logger.info("="*60)
        logger.info("STAGE 1.5: EMBEDDING DEDUPLICATED ENTITIES (GPU)")
        logger.info("="*60)
        logger.info("Using BGE-M3 embedder (1024-dim)")
        
        # Format entities as "name [type]" for embedding (RAKG standard)
        logger.info("Formatting entities as 'name [type]' for embedding...")
        
        for entity in entities:
            entity['formatted_text'] = f"{entity['name']} [{entity['type']}]"
        
        # Batch process with checkpoints
        entities_dict = {i: entity for i, entity in enumerate(entities)}
        
        embedded_dict = self.embed_processor.process_items(
            items=entities_dict,
            text_key='formatted_text',
            batch_size=32,
            checkpoint_dir=self.data_dir / "embed_checkpoints"
        )
        
        # Convert back to list
        entities = list(embedded_dict.values())
        
        # Remove temporary formatted_text field
        for entity in entities:
            entity.pop('formatted_text', None)
        
        # Save intermediate result
        output_file = self.data_dir / "stage1_deduplicated_embedded.json"
        self.save_json(entities, str(output_file))
        
        # Verify embeddings
        stats = self.embed_processor.verify_embeddings(embedded_dict)
        self.stats['stage1_5'] = stats
        
        logger.info(f"âœ“ Stage 1.5 complete: {len(entities)} entities embedded")
        
        return entities
    
    def run_stage2(self, entities: List[Dict]) -> List[Dict]:
        """Run Stage 2: GPU-accelerated FAISS blocking"""
        logger.info("="*60)
        logger.info("STAGE 2: FAISS BLOCKING (GPU)")
        logger.info("="*60)
        
        # Build index
        self.stage2.build_index(entities)
        
        # Parallel search (returns pairs with entity keys already!)
        pairs = self.stage2.find_candidates_parallel(
            entities,
            k=50,
            num_workers=self.faiss_workers
        )
        
        # Save intermediate result (already has entity keys!)
        output_file = self.data_dir / "stage2_candidate_pairs.json"
        with open(output_file, 'w') as f:
            json.dump(pairs, f, indent=2)
        logger.info(f"Saved {len(pairs)} candidate pairs")
        
        self.stats['stage2'] = self.stage2.stats
        
        return pairs
    
    def run_stage3(self, 
                   pairs: List[Dict], 
                   entities: List[Dict]) -> Tuple[Dict, List[Dict]]:
        """
        Run Stage 3: Tiered threshold filtering
        
        MUCH SIMPLER with entity keys - no index remapping needed!
        """
        logger.info("="*60)
        logger.info("STAGE 3: TIERED THRESHOLD FILTERING")
        logger.info("="*60)
        
        # Filter pairs into merged/rejected/uncertain
        filtered = self.stage3.filter_pairs(pairs, entities)
        
        # Save filtered pairs (already have entity keys!)
        output_file = self.data_dir / "stage3_filtered_pairs.json"
        with open(output_file, 'w') as f:
            json.dump(filtered, f, indent=2)
        
        # Apply auto-merges and get key mapping
        logger.info("Applying auto-merge decisions...")
        entities, key_mapping = self.stage3.apply_merges(entities, filtered['merged'])
        
        # Update uncertain pairs with canonical entity keys
        # Build entity map for validation (entities may have been removed in merging)
        from src.processing.entities.entity_disambiguator import build_entity_map
        entity_map = build_entity_map(entities)
        
        updated_uncertain = []
        skipped_same_entity = 0
        skipped_missing = 0
        
        for pair in filtered['uncertain']:
            key1 = pair['entity1_key']
            key2 = pair['entity2_key']
            
            # Remap to canonical keys
            canonical_key1 = key_mapping.get(key1, key1)
            canonical_key2 = key_mapping.get(key2, key2)
            
            # Skip if both merged into same entity
            if canonical_key1 == canonical_key2:
                skipped_same_entity += 1
                continue
            
            # Skip if either entity no longer exists (merged away)
            if canonical_key1 not in entity_map or canonical_key2 not in entity_map:
                skipped_missing += 1
                continue
            
            # Keep pair with updated keys
            updated_uncertain.append({
                'entity1_key': canonical_key1,
                'entity2_key': canonical_key2,
                'similarity': pair['similarity']
            })
        
        logger.info(f"Updated uncertain pairs: {len(filtered['uncertain'])} â†’ {len(updated_uncertain)}")
        logger.info(f"  Removed {skipped_same_entity} pairs (merged into same entity)")
        logger.info(f"  Removed {skipped_missing} pairs (entity no longer exists)")
        filtered['uncertain'] = updated_uncertain
        
        # Save entities after auto-merges
        output_file = self.data_dir / "stage3_entities_after_automerge.json"
        self.save_json(entities, str(output_file))
        
        self.stats['stage3'] = self.stage3.stats
        
        return filtered, entities
    
    
    def run_stage4(self, 
                   uncertain_pairs: List[Dict],
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
        
        # Apply LLM match decisions (discard key mapping - final stage)
        logger.info("Applying LLM match decisions...")
        entities, _ = self.stage3.apply_merges(entities, llm_matches)
        
        self.stats['stage4'] = self.stage4.stats
        
        return entities
    
    def run_pipeline(self, 
                     input_file: str, 
                     output_file: str = None,
                     start_from_stage: int = 1,
                     stop_at_stage: int = None) -> Dict:
        """
Execute comprehensive GPU-optimized pipeline

TYPICAL USAGE:
    # Full pipeline (recommended)
    processor.run_pipeline(
        input_file='data/interim/entities/pre_entities_clean.json',
        output_file='data/interim/entities/normalized_entities.json',
        start_from_stage=1,  # Run all stages
        stop_at_stage=None   # Complete pipeline
    )
    
    # Resume from embedded entities (if Stage 1-1.5 already done)
    processor.run_pipeline(
        input_file='data/interim/entities/stage1_deduplicated_embedded.json',
        start_from_stage=2,  # Skip dedup+embed
        stop_at_stage=None
    )
    
    # Debug: Stop after FAISS to inspect candidates
    processor.run_pipeline(
        input_file='data/interim/entities/pre_entities_clean.json',
        start_from_stage=1,
        stop_at_stage=2  # Inspect FAISS candidates before LLM
    )

Args:
    input_file: Path to input entities
        - For start_from_stage=1: pre_entities_clean.json (recommended)
        - For start_from_stage=2: stage1_deduplicated_embedded.json
    
    output_file: Path to save final output (default: normalized_entities.json)
    
    start_from_stage: Where to begin processing
        1 = Start from raw entities (run: dedup + embed + FAISS + LLM)
        2 = Start from embedded entities (run: FAISS + LLM only)
    
    stop_at_stage: Where to stop (None = complete pipeline)
        1 = Stop after exact dedup (inspect entity count reduction)
        2 = Stop after embedding (verify embeddings exist)
        3 = Stop after thresholds (inspect auto-merge decisions)
        4 = Complete pipeline (default if None)
        
Returns:
    Pipeline statistics dict with per-stage metrics
"""
        logger.info("="*60)
        logger.info("GPU-OPTIMIZED DISAMBIGUATION PIPELINE")
        logger.info("="*60)
        logger.info(f"Start from stage: {start_from_stage}")
        if stop_at_stage:
            logger.info(f"Stop at stage: {stop_at_stage}")
        
        if start_from_stage == 1:
            # Full pipeline: dedup â†’ embed â†’ FAISS â†’ thresholds â†’ LLM
            if not input_file:
                raise ValueError("input_file required when starting from Stage 1")
            
            # Stage 1: Exact dedup
            entities = self.run_stage1(input_file)
            
            if stop_at_stage == 1:
                logger.info("âœ“ Stopping after Stage 1 (exact dedup)")
                return self.stats
            
            # Stage 1.5: Embed
            entities = self.run_stage1_5(entities)
            
            if stop_at_stage == 2:
                logger.info("âœ“ Stopping after Stage 1.5 (embedding complete)")
                return self.stats
                
        elif start_from_stage == 2:
            # Start from embedded entities
            entities = self.load_json(input_file)
            
            # Verify embeddings exist
            if 'embedding' not in entities[0]:
                raise ValueError("Entities missing embeddings! Run Stages 1-1.5 first.")
            
            logger.info("Loaded embedded entities (skipping Stages 1-1.5)")
        
        else:
            raise ValueError(f"Invalid start_from_stage: {start_from_stage}")
        
        # Stage 2: GPU FAISS blocking
        pairs = self.run_stage2(entities)
        
        if stop_at_stage == 2:
            logger.info("âœ“ Stopping after Stage 2 (FAISS blocking)")
            return self.stats
        
        # Stage 3: Tiered thresholds
        filtered, entities = self.run_stage3(pairs, entities)
        
        if stop_at_stage == 3:
            logger.info("âœ“ Stopping after Stage 3 (threshold filtering)")
            return self.stats
        
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

    # RECOMMENDED: Full pipeline from filtered entities (Phase 1C-0 output)
    export TOGETHER_API_KEY="your-key"
    nohup python src/processing/entities/disambiguation_processor.py \\
        --input data/interim/entities/pre_entities_clean.json \\
        --output data/interim/entities/normalized_entities.json \\
        --start-from-stage 1 \\
        --faiss-workers 4 \\
        --samejudge-workers 8 \\
        --gpu-id 0 \\
        > logs/disambiguation.log 2>&1 &

    # Resume from embedded entities (if Stage 1-1.5 already done)
    python src/processing/entities/disambiguation_processor.py \\
        --input data/interim/entities/stage1_deduplicated_embedded.json \\
        --start-from-stage 2 \\
        --faiss-workers 4 \\
        --samejudge-workers 8

    # Debug: Run only FAISS blocking (stop before expensive LLM)
    python src/processing/entities/disambiguation_processor.py \\
        --input data/interim/entities/pre_entities_clean.json \\
        --start-from-stage 1 \\
        --stop-at-stage 2 \\
        --faiss-workers 4

    # Monitor progress
    tail -f logs/disambiguation.log

    # Check GPU usage
    watch -n 1 nvidia-smi

WORKER TUNING:
    RTX 3060 (12GB):  --faiss-workers 4  --samejudge-workers 8
    RTX 3090 (24GB):  --faiss-workers 8  --samejudge-workers 12
    RTX 4090 (24GB):  --faiss-workers 8  --samejudge-workers 16

Note: This is the production GPU-accelerated version.
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
        '--start-from-stage',
        type=int,
        choices=[1, 2],
        default=2,
        help='Start from Stage 1 (raw entities) or Stage 2 (embedded) - default: 2'
    )
    parser.add_argument(
        '--stop-at-stage',
        type=int,
        choices=[1, 2, 3, 4],
        help='Stop after this stage (default: run all stages)'
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
            output_file=args.output,
            start_from_stage=args.start_from_stage,
            stop_at_stage=args.stop_at_stage
        )
        elapsed = time.time() - start_time
        
        if args.stop_at_stage:
            logger.info(f"\nâœ“ Stopped at Stage {args.stop_at_stage} (completed in {elapsed/60:.1f} minutes)")
        else:
            logger.info(f"\nâœ“ Pipeline completed in {elapsed/60:.1f} minutes")
            logger.info(f"Output saved to: {args.output}")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()