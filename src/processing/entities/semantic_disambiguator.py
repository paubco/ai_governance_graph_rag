# -*- coding: utf-8 -*-
"""
Semantic entity disambiguation with alias tracking (v1.1).

Handles the SEMANTIC path (9 types):
    RegulatoryConcept, TechnicalConcept, PoliticalConcept, EconomicConcept,
    Regulation, Technology, Organization, Location, Risk

Stages:
    Stage 1: ExactDeduplicator (hash-based exact deduplication + alias tracking)
    Stage 2: FAISSBlocker (HNSW approximate nearest neighbors)
    Stage 3: TieredThresholdFilter (similarity-based filtering with auto-merge)
    Stage 4: SameJudge (LLM verification)

v1.1 Changes:
    - Alias tracking during merge (not post-hoc)
    - Type routing helpers for two-path architecture
    - Uses Mistral-7B for SameJudge (not Qwen due to JSON bugs)

Example:
    deduplicator = ExactDeduplicator()
    entities, aliases = deduplicator.deduplicate(raw_entities)
"""

# Standard library
import hashlib
import unicodedata
import logging
import json
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter

# Third-party
import numpy as np

# Logger
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS (v2.0 - Semantic + Metadata)
# =============================================================================

METADATA_TYPES = {'Citation', 'Author', 'Journal', 'Affiliation', 'Document', 'DocumentSection'}

SEMANTIC_TYPES = {
    'RegulatoryConcept', 'TechnicalConcept', 'PoliticalConcept', 
    'EconomicConcept', 'Regulation', 'Technology', 'Organization', 
    'Location', 'Risk'
}


# =============================================================================
# ENTITY KEY UTILITIES
# =============================================================================

def get_entity_key(entity: Dict) -> Tuple[str, str]:
    """
    Get hashable key for entity (name, type) tuple.
    
    Args:
        entity: Entity dict with 'name' and 'type' fields
        
    Returns:
        (name, type) tuple - hashable and human-readable
    """
    return (entity['name'], entity['type'])


def normalize_key(key):
    """
    Normalize entity key to hashable tuple.
    
    Handles JSON deserialization where tuples become lists.
    """
    if isinstance(key, list):
        return tuple(key)
    return key


def build_entity_map(entities: List[Dict]) -> Dict[Tuple[str, str], Dict]:
    """
    Build O(1) lookup map from entity keys to entity dicts.
    """
    return {get_entity_key(e): e for e in entities}


def route_by_type(entities: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Route entities to semantic or metadata path.
    
    Args:
        entities: List of entity dicts
        
    Returns:
        (semantic_entities, metadata_entities)
    """
    semantic = []
    metadata = []
    
    for entity in entities:
        if entity.get('type') in METADATA_TYPES:
            metadata.append(entity)
        else:
            semantic.append(entity)
    
    logger.info(f"Routed {len(entities)} entities: {len(semantic)} semantic, {len(metadata)} metadata")
    return semantic, metadata


# =============================================================================
# STAGE 1: EXACT DEDUPLICATION WITH ALIAS TRACKING
# =============================================================================

class ExactDeduplicator:
    """
    Stage 1: Hash-based exact string deduplication with ALIAS TRACKING.
    
    Groups entities by NAME ONLY (type-agnostic).
    When merging: most frequent type wins, tracks all surface forms as aliases.
    
    v1.1: Alias tracking during merge, not post-hoc.
    """
    
    def __init__(self):
        """Initialize deduplicator with stats and alias tracking."""
        self.stats = {
            'input_count': 0,
            'output_count': 0,
            'duplicates_removed': 0,
            'largest_group_size': 0,
            'type_consolidations': 0,
            'entities_with_aliases': 0,
        }
        self.alias_map = {}  # canonical_name → [variant_names]
    
    def normalize_string(self, text: str) -> str:
        """
        Normalize string for exact matching.
        
        Steps:
            1. NFKC normalization (collapses ligatures, fullwidth chars)
            2. Casefold (better than lower() for Unicode)
            3. Whitespace collapse
        """
        normalized = unicodedata.normalize('NFKC', text)
        normalized = normalized.casefold()
        normalized = normalized.strip()
        normalized = ' '.join(normalized.split())
        return normalized
    
    def compute_hash(self, name: str) -> str:
        """
        Compute MD5 hash of normalized name only.
        
        Same entity with different types → same hash → will merge.
        """
        normalized = self.normalize_string(name)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def deduplicate(self, entities: List[Dict]) -> Tuple[List[Dict], Dict[str, List[str]]]:
        """
        Deduplicate entities with alias tracking.
        
        Args:
            entities: List of entity dicts
            
        Returns:
            (canonical_entities, alias_map)
            alias_map: {canonical_name: [variant1, variant2, ...]}
        """
        logger.info(f"Stage 1: Deduplicating {len(entities)} entities...")
        
        # Group entities by hash
        hash_groups = defaultdict(list)
        for entity in entities:
            h = self.compute_hash(entity['name'])
            hash_groups[h].append(entity)
        
        # Track largest group
        largest_group = max(len(group) for group in hash_groups.values()) if hash_groups else 0
        
        # Track type consolidations
        type_consolidations = sum(
            1 for group in hash_groups.values()
            if len(set(e['type'] for e in group)) > 1
        )
        
        # Merge each group with alias tracking
        canonical = []
        self.alias_map = {}
        
        for h, group in hash_groups.items():
            merged, aliases = self._merge_group(group)
            canonical.append(merged)
            if aliases:
                self.alias_map[merged['name']] = aliases
        
        # Update stats
        self.stats['input_count'] = len(entities)
        self.stats['output_count'] = len(canonical)
        self.stats['duplicates_removed'] = len(entities) - len(canonical)
        self.stats['largest_group_size'] = largest_group
        self.stats['type_consolidations'] = type_consolidations
        self.stats['entities_with_aliases'] = len(self.alias_map)
        
        logger.info(f"Deduplication: {len(entities)} → {len(canonical)} entities")
        logger.info(f"  Duplicates removed: {self.stats['duplicates_removed']}")
        logger.info(f"  Type consolidations: {type_consolidations}")
        logger.info(f"  Entities with aliases: {len(self.alias_map)}")
        
        return canonical, self.alias_map
    
    def _merge_group(self, group: List[Dict]) -> Tuple[Dict, List[str]]:
        """
        Merge duplicate entities with alias tracking.
        
        Args:
            group: List of duplicate entities
            
        Returns:
            (merged_entity, list_of_aliases)
            
        Note: Only tracks meaningful aliases (not just case variants).
              Real semantic aliases come from FAISS merge, not hash-dedup.
        """
        # Name selection: most frequent surface form
        name_counts = Counter(e['name'] for e in group)
        canonical_name = name_counts.most_common(1)[0][0]
        
        # Type: most frequent wins (voting)
        type_counts = Counter(e['type'] for e in group)
        canonical_type = type_counts.most_common(1)[0][0]
        
        # ALIAS TRACKING: Only meaningful differences (not case-only)
        # Hash-dedup aliases are always case variants (by definition of hash)
        # Real semantic aliases (EU AI Act ↔ European AI Act) come from FAISS merge
        all_names = set(e['name'] for e in group)
        aliases = sorted([
            n for n in all_names 
            if n != canonical_name and n.lower() != canonical_name.lower()
        ])
        
        # Combine all chunk_ids
        all_chunk_ids = set()
        for entity in group:
            if 'chunk_id' in entity:
                all_chunk_ids.add(entity['chunk_id'])
            if 'chunk_ids' in entity:
                chunk_ids = entity['chunk_ids']
                if isinstance(chunk_ids, list):
                    all_chunk_ids.update(chunk_ids)
                elif chunk_ids:
                    all_chunk_ids.add(chunk_ids)
        
        # Description: longest (most informative)
        descriptions = [e.get('description', '') for e in group if e.get('description')]
        best_description = max(descriptions, key=len) if descriptions else ''
        
        merged = {
            'name': canonical_name,
            'type': canonical_type,
            'description': best_description,
            'chunk_ids': sorted(list(all_chunk_ids)),
            'aliases': aliases,
            'merge_count': len(group),
        }
        
        # Preserve embedding if exists (will re-embed later)
        for entity in group:
            if 'embedding' in entity:
                merged['embedding'] = entity['embedding']
                break
        
        return merged, aliases


# =============================================================================
# STAGE 2: FAISS BLOCKING
# =============================================================================

class FAISSBlocker:
    """
    Stage 2: FAISS HNSW blocking for approximate nearest neighbor search.
    
    Uses Hierarchical Navigable Small World (HNSW) graph to find similar entities
    efficiently. Reduces N² comparisons to N×k comparisons.
    """
    
    def __init__(self, embedding_dim: int = 1024, M: int = 32):
        """
        Initialize FAISS blocker.
        
        Args:
            embedding_dim: Embedding dimension (1024 for BGE-M3)
            M: HNSW connections per node (32 standard for 1024-dim)
        """
        self.embedding_dim = embedding_dim
        self.M = M
        self.index = None
        self.stats = {
            'entities_indexed': 0,
            'candidate_pairs': 0,
            'avg_neighbors': 0.0
        }
    
    def build_index(self, entities: List[Dict], ef_construction: int = 200):
        """
        Build HNSW index from entity embeddings.
        
        Args:
            entities: List of entities with 'embedding' field
            ef_construction: HNSW parameter (higher = better recall, slower build)
        """
        import faiss
        
        logger.info(f"Building FAISS index for {len(entities)} entities...")
        
        # Extract embeddings
        embeddings = []
        for entity in entities:
            if 'embedding' in entity:
                emb = entity['embedding']
                if isinstance(emb, list):
                    emb = np.array(emb, dtype=np.float32)
                embeddings.append(emb)
            else:
                embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
        
        embeddings = np.vstack(embeddings).astype(np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Build HNSW index
        self.index = faiss.IndexHNSWFlat(self.embedding_dim, self.M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.add(embeddings)
        
        self.stats['entities_indexed'] = len(entities)
        logger.info(f"Index built with {len(entities)} entities")
    
    def find_candidates(self, 
                       entities: List[Dict],
                       k: int = 50,
                       threshold: float = 0.70,
                       ef_search: int = 64) -> List[Dict]:
        """
        Find candidate pairs above similarity threshold.
        
        Args:
            entities: List of entities (same order as build_index)
            k: Number of neighbors to search
            threshold: Minimum similarity threshold
            ef_search: HNSW search parameter
            
        Returns:
            List of pair dicts with entity1_key, entity2_key, similarity
        """
        import faiss
        
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index first.")
        
        logger.info(f"Stage 2: Finding candidates (k={k}, threshold={threshold})...")
        
        # Extract and normalize embeddings
        embeddings = []
        for entity in entities:
            if 'embedding' in entity:
                emb = entity['embedding']
                if isinstance(emb, list):
                    emb = np.array(emb, dtype=np.float32)
                embeddings.append(emb)
            else:
                embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
        
        embeddings = np.vstack(embeddings).astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # Set search parameter
        self.index.hnsw.efSearch = ef_search
        
        # Search
        distances, indices = self.index.search(embeddings, k)
        
        # Convert to pairs (using entity keys, not indices)
        pairs = []
        seen = set()
        
        for i, (dists, idxs) in enumerate(zip(distances, indices)):
            key_i = get_entity_key(entities[i])
            
            for dist, j in zip(dists, idxs):
                if i >= j:  # Skip self and duplicates
                    continue
                
                # Convert L2 distance to cosine similarity
                similarity = 1.0 - (dist / 2.0)
                
                if similarity < threshold:
                    continue
                
                key_j = get_entity_key(entities[j])
                pair_key = (min(key_i, key_j), max(key_i, key_j))
                
                if pair_key not in seen:
                    seen.add(pair_key)
                    pairs.append({
                        'entity1_key': key_i,
                        'entity2_key': key_j,
                        'similarity': float(similarity),
                    })
        
        self.stats['candidate_pairs'] = len(pairs)
        self.stats['avg_neighbors'] = len(pairs) / len(entities) if entities else 0.0
        
        logger.info(f"Found {len(pairs)} candidate pairs")
        return pairs


# =============================================================================
# STAGE 3: TIERED THRESHOLD FILTERING
# =============================================================================

class TieredThresholdFilter:
    """
    Stage 3: Tiered threshold filtering for auto-merge/reject decisions.
    
    Thresholds:
        - High (≥0.95): Auto-merge without LLM
        - Low (<0.85): Auto-reject without LLM
        - Medium (0.85-0.95): Send to LLM for verification
    """
    
    def __init__(self, 
                 auto_merge_threshold: float = 0.95,
                 auto_reject_threshold: float = 0.85):
        """
        Initialize threshold filter.
        
        Args:
            auto_merge_threshold: Similarity ≥ this → auto-merge
            auto_reject_threshold: Similarity < this → auto-reject
        """
        self.merge_threshold = auto_merge_threshold
        self.reject_threshold = auto_reject_threshold
        self.stats = {
            'input_pairs': 0,
            'auto_merged': 0,
            'auto_rejected': 0,
            'uncertain': 0
        }
    
    def filter_pairs(self, pairs: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Classify pairs into auto-merge, auto-reject, uncertain.
        
        Args:
            pairs: List of pair dicts with similarity
            
        Returns:
            {'merged': [...], 'rejected': [...], 'uncertain': [...]}
        """
        logger.info(f"Stage 3: Filtering {len(pairs)} pairs...")
        
        merged = []
        rejected = []
        uncertain = []
        
        for pair in pairs:
            sim = pair['similarity']
            
            if sim >= self.merge_threshold:
                merged.append(pair)
            elif sim < self.reject_threshold:
                rejected.append(pair)
            else:
                uncertain.append(pair)
        
        self.stats['input_pairs'] = len(pairs)
        self.stats['auto_merged'] = len(merged)
        self.stats['auto_rejected'] = len(rejected)
        self.stats['uncertain'] = len(uncertain)
        
        logger.info(f"  Auto-merge (≥{self.merge_threshold}): {len(merged)}")
        logger.info(f"  Auto-reject (<{self.reject_threshold}): {len(rejected)}")
        logger.info(f"  Uncertain: {len(uncertain)} → LLM")
        
        return {
            'merged': merged,
            'rejected': rejected,
            'uncertain': uncertain
        }


# =============================================================================
# STAGE 4: LLM SAME JUDGE
# =============================================================================

class SameJudge:
    """
    Stage 4: LLM-based entity verification.
    
    Uses Mistral-7B (not Qwen due to JSON parsing issues).
    Integrates with CheckpointManager and RateLimiter from project utils.
    """
    
    def __init__(self, 
                 model: str = "mistralai/Mistral-7B-Instruct-v0.3",
                 api_key: str = None,
                 max_rpm: int = 2900):
        """
        Initialize LLM judge.
        
        Args:
            model: Model name for Together API
            api_key: API key (uses env var if not provided)
            max_rpm: Max requests per minute (default 2900, buffer below 3000 limit)
        """
        import os
        from dotenv import load_dotenv
        from together import Together
        from src.utils.rate_limiter import RateLimiter
        
        load_dotenv()
        
        self.model = model
        self.client = Together(api_key=api_key or os.environ.get('TOGETHER_API_KEY'))
        self.rate_limiter = RateLimiter(max_calls_per_minute=max_rpm)
        self.stats = {
            'pairs_judged': 0,
            'same_decisions': 0,
            'different_decisions': 0,
            'errors': 0,
        }
    
    def judge_pair(self, entity1: Dict, entity2: Dict) -> Tuple[bool, str]:
        """
        Judge if two entities refer to the same real-world entity.
        
        Args:
            entity1, entity2: Entity dicts with name, type, description
            
        Returns:
            (is_same, reasoning)
        """
        prompt = self._build_prompt(entity1, entity2)
        
        # Rate limit
        self.rate_limiter.acquire()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.0,
            )
            
            result = response.choices[0].message.content.strip().upper()
            
            self.stats['pairs_judged'] += 1
            
            if 'YES' in result or 'SAME' in result:
                self.stats['same_decisions'] += 1
                return True, result
            else:
                self.stats['different_decisions'] += 1
                return False, result
                
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"SameJudge error: {e}")
            return False, str(e)
    
    def _build_prompt(self, entity1: Dict, entity2: Dict) -> str:
        """Build prompt for same-entity judgment."""
        from src.prompts.prompts import SAMEJUDGE_PROMPT
        
        return SAMEJUDGE_PROMPT.format(
            entity1_name=entity1.get('name', ''),
            entity1_type=entity1.get('type', ''),
            entity1_desc=entity1.get('description', '')[:150] or 'N/A',
            entity2_name=entity2.get('name', ''),
            entity2_type=entity2.get('type', ''),
            entity2_desc=entity2.get('description', '')[:150] or 'N/A',
        )
    
    def judge_pairs(self, 
                   pairs: List[Dict], 
                   entity_map: Dict[Tuple[str, str], Dict],
                   checkpoint_dir: str = None,
                   checkpoint_freq: int = 1000,
                   max_workers: int = 8) -> List[Dict]:
        """
        Judge multiple pairs, returning those that should merge.
        
        Uses CheckpointManager for resume capability and progress tracking.
        Parallelized with ThreadPoolExecutor.
        
        Args:
            pairs: List of pair dicts with entity1_key, entity2_key
            entity_map: {(name, type): entity_dict}
            checkpoint_dir: Directory for checkpoints (enables resume)
            checkpoint_freq: Save checkpoint every N pairs
            max_workers: Number of parallel workers
            
        Returns:
            List of pairs that should merge
        """
        from tqdm import tqdm
        from datetime import datetime
        from pathlib import Path
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from threading import Lock
        from src.utils.io import load_json, save_json
        
        logger.info(f"Stage 4: Judging {len(pairs)} uncertain pairs (workers={max_workers})...")
        
        merge_pairs = []
        start_idx = 0
        results_lock = Lock()
        
        # Setup checkpoint if directory provided
        progress_file = None
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            progress_file = checkpoint_path / 'samejudge_progress.json'
            
            # Resume from checkpoint if exists
            if progress_file.exists():
                progress = load_json(progress_file)
                start_idx = progress.get('processed', 0)
                merge_pairs = progress.get('merge_pairs', [])
                self.stats = progress.get('stats', self.stats)
                logger.info(f"Resuming from checkpoint: {start_idx}/{len(pairs)} processed")
        
        remaining_pairs = pairs[start_idx:]
        processed_count = start_idx
        
        def process_pair(pair):
            """Process single pair - thread-safe via rate_limiter."""
            key1 = normalize_key(pair['entity1_key'])
            key2 = normalize_key(pair['entity2_key'])
            
            entity1 = entity_map.get(key1, {})
            entity2 = entity_map.get(key2, {})
            
            is_same, reasoning = self.judge_pair(entity1, entity2)
            return pair, is_same
        
        # Process with thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_pair, pair): pair for pair in remaining_pairs}
            
            with tqdm(total=len(pairs), initial=start_idx, desc="SameJudge") as pbar:
                for future in as_completed(futures):
                    pair, is_same = future.result()
                    
                    with results_lock:
                        if is_same:
                            merge_pairs.append(pair)
                        processed_count += 1
                        pbar.update(1)
                        
                        # Checkpoint at intervals
                        if checkpoint_dir and processed_count % checkpoint_freq == 0:
                            save_json({
                                'processed': processed_count,
                                'total': len(pairs),
                                'merge_pairs': merge_pairs,
                                'stats': self.stats,
                                'rate_limiter': self.rate_limiter.get_stats(),
                                'timestamp': datetime.now().isoformat(),
                            }, str(progress_file))
                            logger.info(f"Checkpoint: {processed_count}/{len(pairs)}, {len(merge_pairs)} merges")
        
        # Final checkpoint
        if checkpoint_dir:
            save_json({
                'processed': len(pairs),
                'total': len(pairs),
                'merge_pairs': merge_pairs,
                'stats': self.stats,
                'rate_limiter': self.rate_limiter.get_stats(),
                'complete': True,
                'timestamp': datetime.now().isoformat(),
            }, str(progress_file))
        
        logger.info(f"  LLM approved {len(merge_pairs)} merges")
        logger.info(f"  Rate limiter stats: {self.rate_limiter.get_stats()}")
        return merge_pairs


# =============================================================================
# MERGE APPLICATION WITH ALIAS TRACKING
# =============================================================================

def apply_merges(entities: List[Dict], 
                merged_pairs: List[Dict],
                existing_aliases: Dict[str, List[str]] = None) -> Tuple[List[Dict], Dict[str, List[str]]]:
    """
    Apply merge decisions using Union-Find with alias tracking.
    
    Args:
        entities: List of entities
        merged_pairs: List of pair dicts to merge
        existing_aliases: Existing alias map to extend
        
    Returns:
        (canonical_entities, updated_alias_map)
    """
    logger.info(f"Applying {len(merged_pairs)} merges...")
    
    aliases = dict(existing_aliases) if existing_aliases else {}
    
    # Build Union-Find structure
    parent = {}
    
    for entity in entities:
        key = get_entity_key(entity)
        parent[key] = key
    
    def find(x):
        x = normalize_key(x)
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        x = normalize_key(x)
        y = normalize_key(y)
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px
    
    # Union all merged pairs
    for pair in merged_pairs:
        key1 = normalize_key(pair['entity1_key'])
        key2 = normalize_key(pair['entity2_key'])
        union(key1, key2)
    
    # Group entities by root
    groups = defaultdict(list)
    for entity in entities:
        key = get_entity_key(entity)
        root = find(key)
        groups[root].append(entity)
    
    # Merge each group with alias tracking
    deduplicator = ExactDeduplicator()
    canonical = []
    
    for root in sorted(groups.keys()):
        group = groups[root]
        merged, new_aliases = deduplicator._merge_group(group)
        canonical.append(merged)
        
        # Extend alias map
        if new_aliases:
            canonical_name = merged['name']
            if canonical_name in aliases:
                # Combine with existing aliases
                existing = set(aliases[canonical_name])
                existing.update(new_aliases)
                aliases[canonical_name] = sorted(list(existing))
            else:
                aliases[canonical_name] = new_aliases
    
    logger.info(f"Merged {len(entities)} → {len(canonical)} entities")
    logger.info(f"Total aliases tracked: {len(aliases)}")
    
    return canonical, aliases