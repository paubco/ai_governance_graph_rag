"""
Entity Disambiguation - Core Classes

Stage 1: ExactDeduplicator - Hash-based exact deduplication
Stage 2: FAISSBlocker - HNSW approximate nearest neighbors (CPU)
Stage 3: TieredThresholdFilter - Similarity-based filtering with auto-merge
Stage 4: SameJudge - LLM verification (CPU, single-threaded)

For GPU versions, use server_scripts/disambiguation_server.py
"""

# Standard library
import hashlib
import unicodedata
import logging
import json
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter

# Third-party
import numpy as np

# Logger
logger = logging.getLogger(__name__)


# ============================================================================
# ENTITY KEY UTILITIES
# ============================================================================

def get_entity_key(entity: Dict) -> Tuple[str, str]:
    """
    Get hashable key for entity (name, type) tuple
    
    This is THE canonical way to identify entities throughout the pipeline.
    Replaces fragile index-based lookups with human-readable keys.
    
    Args:
        entity: Entity dict with 'name' and 'type' fields
        
    Returns:
        (name, type) tuple - hashable and human-readable
        
    Example:
        >>> entity = {'name': 'UAE', 'type': 'Country'}
        >>> get_entity_key(entity)
        ('UAE', 'Country')
    """
    return (entity['name'], entity['type'])


def build_entity_map(entities: List[Dict]) -> Dict[Tuple[str, str], Dict]:
    """
    Build O(1) lookup map from entity keys to entity dicts
    
    Args:
        entities: List of entity dicts
        
    Returns:
        Dict mapping (name, type) -> entity dict
        
    Example:
        >>> entities = [{'name': 'UAE', 'type': 'Country'}, ...]
        >>> entity_map = build_entity_map(entities)
        >>> entity_map[('UAE', 'Country')]
        {'name': 'UAE', 'type': 'Country', ...}
    """
    return {get_entity_key(e): e for e in entities}


# ============================================================================
# STAGE 1: EXACT DEDUPLICATION
# ============================================================================


class ExactDeduplicator:
    """
    Stage 1: Hash-based exact string deduplication
    
    Uses NFKC normalization + casefold + MD5 hashing to group identical entities.
    Merges duplicates by combining chunk_ids and using most frequent type.
    
    References:
        - Christen (2012) "Data Matching" - Standard normalization pipeline
        - Python recordlinkage library - NFKC best practice
    """
    
    def __init__(self):
        """Initialize deduplicator with stats tracking"""
        self.stats = {
            'input_count': 0,
            'output_count': 0,
            'duplicates_removed': 0,
            'largest_group_size': 0
        }
    
    def normalize_string(self, text: str) -> str:
        """
        Normalize string for exact matching
        
        Steps:
            1. NFKC normalization (collapses ligatures, fullwidth chars)
            2. Casefold (better than lower() for Unicode)
            3. Whitespace collapse
        
        Args:
            text: Raw string
            
        Returns:
            Normalized string
        """
        # NFKC normalization: ﬁ→fi, fullwidth→normal
        normalized = unicodedata.normalize('NFKC', text)
        
        # Casefold: handles special cases like German ß→ss
        normalized = normalized.casefold()
        
        # Strip and collapse whitespace
        normalized = normalized.strip()
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def compute_hash(self, name: str, entity_type: str) -> str:
        """
        Compute MD5 hash of normalized 'name [type]'
        
        Args:
            name: Entity name
            entity_type: Entity type
            
        Returns:
            32-character hex hash
        """
        normalized = self.normalize_string(f"{name} [{entity_type}]")
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def deduplicate(self, entities: List[Dict]) -> List[Dict]:
        """
        Deduplicate entities by exact string matching
        
        Process:
            1. Group entities by hash
            2. Merge each group (combine chunk_ids, vote on type)
            3. Preserve embeddings if they exist
        
        Args:
            entities: List of entity dicts with 'name', 'type', 'chunk_ids'
            
        Returns:
            List of canonical entities (deduplicated)
        """
        logger.info(f"Stage 1: Deduplicating {len(entities)} entities...")
        
        # Group entities by hash
        hash_groups = defaultdict(list)
        for entity in entities:
            h = self.compute_hash(entity['name'], entity['type'])
            hash_groups[h].append(entity)
        
        # Track largest group for stats
        largest_group = max(len(group) for group in hash_groups.values())
        
        # Merge each group
        canonical = []
        for h, group in hash_groups.items():
            merged = self._merge_group(group)
            canonical.append(merged)
        
        # Update stats
        self.stats['input_count'] = len(entities)
        self.stats['output_count'] = len(canonical)
        self.stats['duplicates_removed'] = len(entities) - len(canonical)
        self.stats['largest_group_size'] = largest_group
        
        logger.info(f"Deduplication complete: {len(entities)} → {len(canonical)} entities")
        logger.info(f"Removed {self.stats['duplicates_removed']} duplicates ({100 * self.stats['duplicates_removed'] / len(entities):.1f}%)")
        logger.info(f"Largest duplicate group: {largest_group} entities")
        
        return canonical
    
    def _merge_group(self, group: List[Dict]) -> Dict:
        """
        Merge duplicate entities
        
        Merge strategy:
            - Name: From first entity (canonical)
            - Type: Most frequent across group
            - Description: Longest (most informative)
            - chunk_ids: Union of all chunk_ids
            - Embedding: From first entity (if exists)
        
        Args:
            group: List of duplicate entities
            
        Returns:
            Merged canonical entity
        """
        # Start with first entity as base
        canonical = group[0].copy()
        
        # Remove chunk_id if present (redundant - we have chunk_ids)
        canonical.pop('chunk_id', None)
        
        # Combine all chunk_ids
        all_chunk_ids = set()
        for entity in group:
            # Handle both chunk_id (singular from Phase 1B) and chunk_ids (plural from merges)
            if 'chunk_id' in entity:
                all_chunk_ids.add(entity['chunk_id'])
            if 'chunk_ids' in entity:
                chunk_ids = entity['chunk_ids']
                if isinstance(chunk_ids, list):
                    all_chunk_ids.update(chunk_ids)
                elif chunk_ids:
                    all_chunk_ids.add(chunk_ids)
        canonical['chunk_ids'] = sorted(list(all_chunk_ids))
        
        # Type: Most frequent wins (voting)
        type_counts = Counter([e['type'] for e in group])
        canonical['type'] = type_counts.most_common(1)[0][0]
        
        # Description: Most frequent wins (voting, like type)
        descriptions = [e.get('description', '') for e in group if e.get('description', '')]
        if descriptions:
            desc_counts = Counter(descriptions)
            canonical['description'] = desc_counts.most_common(1)[0][0]
        else:
            canonical['description'] = ''
        
        # Preserve embedding if exists (from first entity)
        if 'embedding' in group[0]:
            canonical['embedding'] = group[0]['embedding']
        
        # Add metadata about merge
        canonical['duplicate_count'] = len(group)
        
        return canonical


class FAISSBlocker:
    """
    Stage 2: FAISS HNSW blocking for approximate nearest neighbor search
    
    **NOTE**: This is the CPU-only version for local development.
    For GPU-accelerated version with multithreading, use:
        server_scripts/disambiguation_server.py
    
    Uses Hierarchical Navigable Small World (HNSW) graph to find similar entities
    efficiently. Reduces N² comparisons to N×k comparisons.
    
    References:
        - Malkov & Yashunin (2020) - HNSW algorithm
        - Johnson et al. (2024) - FAISS library
        - Li et al. (VLDB 2023) - Benchmark validation
    """
    
    def __init__(self, embedding_dim: int = 1024, M: int = 32):
        """
        Initialize FAISS blocker (CPU version)
        
        Args:
            embedding_dim: Embedding dimension (1024 for BGE-M3)
            M: HNSW connections per node (32 standard for 1024-dim)
        """
        import faiss
        
        self.embedding_dim = embedding_dim
        self.M = M
        self.index = None
        self.stats = {
            'entities_indexed': 0,
            'candidate_pairs': 0,
            'avg_neighbors': 0.0
        }
        
        logger.info("FAISSBlocker initialized (CPU version)")
        logger.info("For GPU version, use server_scripts/disambiguation_server.py")
    
    def build_index(self, entities: List[Dict], ef_construction: int = 200):
        """
        Build HNSW index from entity embeddings
        
        Args:
            entities: List of entities with 'embedding' field
            ef_construction: HNSW parameter (higher = better recall, slower build)
        """
        import faiss
        
        logger.info(f"Stage 2: Building FAISS HNSW index for {len(entities)} entities...")
        
        # Extract embeddings
        embeddings = np.array([e['embedding'] for e in entities]).astype('float32')
        
        # CRITICAL: Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create HNSW index
        self.index = faiss.IndexHNSWFlat(self.embedding_dim, self.M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.add(embeddings)
        
        self.stats['entities_indexed'] = len(entities)
        
        logger.info(f"FAISS index built successfully:")
        logger.info(f"  Type: HNSW, M={self.M}, ef_construction={ef_construction}")
        logger.info(f"  Entities: {len(entities)}")
        logger.info(f"  Dimension: {self.embedding_dim}")
    
    def find_candidates(self, 
                       entities: List[Dict], 
                       k: int = 50,
                       ef_search: int = 64) -> List[Dict]:
        """
        Find candidate pairs using approximate nearest neighbor search
        
        Returns pairs with entity KEYS (name, type) instead of indices.
        This makes the output human-readable and robust to list reordering.
        
        Args:
            entities: List of entities with embeddings
            k: Number of neighbors to retrieve per entity
            ef_search: HNSW parameter (higher = better recall, slower search)
            
        Returns:
            List of pair dicts:
            {
                'entity1_key': (name, type),
                'entity2_key': (name, type),
                'similarity': float
            }
        """
        logger.info(f"Searching k={k} nearest neighbors (ef_search={ef_search})...")
        
        self.index.hnsw.efSearch = ef_search
        
        # Extract and normalize embeddings
        embeddings = np.array([e['embedding'] for e in entities]).astype('float32')
        import faiss
        faiss.normalize_L2(embeddings)
        
        # TEMPORARY: Add _index for FAISS operations only
        for i, entity in enumerate(entities):
            entity['_index'] = i
        
        # Search k+1 neighbors (includes self-match)
        distances, indices = self.index.search(embeddings, k + 1)
        
        # Convert to candidate pairs with entity KEYS
        pairs = []
        seen_pairs = set()  # Track unique pairs
        
        for i, (dists, neighbors) in enumerate(zip(distances, indices)):
            entity1 = entities[i]
            entity1_key = get_entity_key(entity1)
            
            for neighbor_idx, dist in zip(neighbors, dists):
                # Skip self-matches
                if neighbor_idx == i:
                    continue
                
                entity2 = entities[neighbor_idx]
                entity2_key = get_entity_key(entity2)
                
                # Convert L2 distance to cosine similarity
                # Since vectors are normalized: cos_sim = 1 - (L2_dist^2 / 2)
                similarity = float(1 - (dist ** 2) / 2)
                
                # Filter out dissimilar pairs (negative cosine similarity)
                if similarity < 0.0:
                    continue
                
                # Only keep unique pairs (use sorted keys to avoid duplicates)
                pair_key = tuple(sorted([entity1_key, entity2_key]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    pairs.append({
                        'entity1_key': entity1_key,
                        'entity2_key': entity2_key,
                        'similarity': similarity
                    })
        
        # CLEANUP: Remove temporary _index field
        for entity in entities:
            entity.pop('_index', None)
        
        self.stats['candidate_pairs'] = len(pairs)
        self.stats['avg_neighbors'] = len(pairs) / len(entities) if entities else 0.0
        
        logger.info(f"Found {len(pairs)} candidate pairs")
        logger.info(f"Average neighbors per entity: {self.stats['avg_neighbors']:.1f}")
        
        return pairs


class TieredThresholdFilter:
    """
    Stage 3: Tiered threshold filtering for auto-merge/reject decisions
    
    Applies three similarity thresholds:
        - High (≥0.98): Auto-merge without LLM (only near-identical)
        - Low (<0.90): Auto-reject without LLM (clearly different)
        - Medium (0.90-0.98): Send to LLM for verification
    
    References:
        - Papadakis et al. (2021) - Multi-tier blocking survey
        - Splink (2024) - Multi-threshold approach
    """
    
    def __init__(self, 
                 auto_merge_threshold: float = 0.98,
                 auto_reject_threshold: float = 0.90):
        """
        Initialize threshold filter
        
        Args:
            auto_merge_threshold: Similarity ≥ this → auto-merge (default 0.98)
            auto_reject_threshold: Similarity < this → auto-reject (default 0.90)
        """
        self.merge_threshold = auto_merge_threshold
        self.reject_threshold = auto_reject_threshold
        self.stats = {
            'input_pairs': 0,
            'auto_merged': 0,
            'auto_rejected': 0,
            'uncertain': 0
        }
    
    def filter_pairs(self, 
                    pairs: List[Dict],
                    entities: List[Dict]) -> Dict:
        """
        Classify pairs into auto-merge, auto-reject, uncertain
        
        Works with entity KEYS (name, type) instead of indices for robustness.
        
        Args:
            pairs: List of pair dicts with entity1_key, entity2_key, similarity
            entities: List of entities (for context, not used in filtering)
            
        Returns:
            {
                'merged': List of pair dicts,
                'rejected': List of pair dicts,
                'uncertain': List of pair dicts
            }
        """
        logger.info(f"Stage 3: Filtering {len(pairs)} pairs with tiered thresholds...")
        
        merged = []
        rejected = []
        uncertain = []
        
        for pair in pairs:
            sim = pair['similarity']
            
            # Apply similarity thresholds
            # Note: Different types are allowed through - LLM will decide in Stage 4
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
        
        logger.info(f"Filtering complete:")
        logger.info(f"  Auto-merge (≥{self.merge_threshold}): {len(merged)} pairs")
        logger.info(f"  Auto-reject (<{self.reject_threshold}): {len(rejected)} pairs")
        logger.info(f"  Uncertain ({self.reject_threshold}-{self.merge_threshold}): {len(uncertain)} pairs")
        
        return {
            'merged': merged,
            'rejected': rejected,
            'uncertain': uncertain
        }
    
    def apply_merges(self, 
                    entities: List[Dict],
                    merged_pairs: List[Dict]) -> Tuple[List[Dict], Dict[Tuple[str, str], Tuple[str, str]]]:
        """
        Apply merge decisions using Union-Find algorithm
        
        Uses Union-Find to handle transitivity:
            If A=B and B=C, then A=C
        
        Now works with entity KEYS instead of indices - much cleaner!
        Returns key mapping so uncertain pairs can be updated.
        
        Args:
            entities: List of entities
            merged_pairs: List of pair dicts with entity1_key, entity2_key
            
        Returns:
            Tuple of:
            - List of canonical entities after merging
            - Dict mapping old entity keys to canonical entity keys
        """
        logger.info(f"Applying {len(merged_pairs)} merge decisions...")
        
        # Build Union-Find structure using entity KEYS
        parent = {}
        
        # Initialize: each entity is its own parent
        for entity in entities:
            key = get_entity_key(entity)
            parent[key] = key
        
        def find(x):
            """Find root with path compression"""
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            """Union two sets"""
            px, py = find(x), find(y)
            if px != py:
                parent[py] = px
        
        # Union all merged pairs (using entity keys - no index validation needed!)
        for pair in merged_pairs:
            key1 = pair['entity1_key']
            key2 = pair['entity2_key']
            union(key1, key2)
        
        # Group entities by root
        groups = defaultdict(list)
        for entity in entities:
            key = get_entity_key(entity)
            root = find(key)
            groups[root].append(entity)
        
        # Merge each group using ExactDeduplicator logic
        deduplicator = ExactDeduplicator()
        canonical = []
        for root in sorted(groups.keys()):
            group = groups[root]
            merged = deduplicator._merge_group(group)
            canonical.append(merged)
        
        # Build key mapping: old_key → canonical_key
        key_mapping = {}
        for entity in entities:
            old_key = get_entity_key(entity)
            root_key = find(old_key)
            key_mapping[old_key] = root_key
        
        logger.info(f"Merged {len(entities)} → {len(canonical)} entities")
        
        return canonical, key_mapping



class SameJudge:
    """
    Stage 4: LLM-based entity verification (CPU version)
    
    Simple, single-threaded implementation for development/testing.
    For GPU-optimized multithreaded version, use:
        server_scripts/disambiguation_server.py
    
    Uses centralized prompt from src/prompts/prompts.py
    
    References:
        - Zhang et al. (2025) RAKG - SameJudge methodology
        - Wu et al. (2020) BLINK - LLM entity linking
    """
    
    def __init__(self, 
                 model: str = "Qwen/Qwen2.5-7B-Instruct-Turbo",
                 api_key: str = None):
        """
        Initialize LLM judge (CPU version)
        
        Args:
            model: LLM model identifier
            api_key: Together.ai API key
        """
        from together import Together
        
        self.client = Together(api_key=api_key)
        self.model = model
        self.stats = {
            'pairs_verified': 0,
            'matches_found': 0,
            'total_cost': 0.0
        }
        self.cost_per_call = 0.00003  # Approximate for 7B model
        
        logger.info("SameJudge initialized (CPU version - single-threaded)")
        logger.info("For GPU version with 8 workers, use server_scripts/disambiguation_server.py")
    
    def verify_pair(self, entity1: Dict, entity2: Dict) -> Dict:
        """
        Use LLM to determine if two entities are the same
        
        Note: LLM can decide that entities with different types are actually the same
        (e.g., "EU" [Organization] vs "European Union" [Political Union])
        
        Args:
            entity1: First entity dict
            entity2: Second entity dict
            
        Returns:
            {
                'is_same': bool,
                'confidence': float (0-1),
                'reasoning': str
            }
        """
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
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=150
            )
            
            response_text = response.choices[0].message.content.strip()
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            result = json.loads(response_text)
            
            parsed = {
                'is_same': result.get('result', False),
                'confidence': 0.9 if result.get('result', False) else 0.5,
                'reasoning': result.get('reasoning', '')
            }
            
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            parsed = {
                'is_same': False,
                'confidence': 0.0,
                'reasoning': f"Error: {str(e)}"
            }
        
        self.stats['pairs_verified'] += 1
        if parsed['is_same']:
            self.stats['matches_found'] += 1
        self.stats['total_cost'] += self.cost_per_call
        
        return parsed
    
    def verify_batch(self, 
                    uncertain_pairs: List[Tuple[int, int, float]],
                    entities: List[Dict],
                    log_interval: int = 100) -> List[Dict]:
        """
        Verify batch of uncertain pairs (single-threaded)
        
        Works with entity KEYS - much cleaner than index-based approach!
        
        Args:
            uncertain_pairs: List of pair dicts with entity1_key, entity2_key, similarity
            entities: List of entities
            log_interval: Log progress every N pairs
            
        Returns:
            List of pair dicts confirmed as matches
        """
        logger.info(f"Stage 4: Verifying {len(uncertain_pairs)} pairs with LLM...")
        logger.info(f"Estimated cost: ${len(uncertain_pairs) * self.cost_per_call:.2f}")
        logger.info("Single-threaded - for faster processing, use disambiguation_server.py")
        
        # Build entity map for O(1) lookups (no index validation needed!)
        entity_map = build_entity_map(entities)
        
        matches = []
        
        for idx, pair in enumerate(uncertain_pairs):
            key1 = pair['entity1_key']
            key2 = pair['entity2_key']
            
            # Lookup entities (KeyError only if entity was removed, which shouldn't happen)
            entity1 = entity_map[key1]
            entity2 = entity_map[key2]
            
            result = self.verify_pair(entity1, entity2)
            
            if result['is_same']:
                matches.append(pair)
            
            # Log progress
            if (idx + 1) % log_interval == 0 or (idx + 1) == len(uncertain_pairs):
                logger.info(f"Progress: {idx + 1}/{len(uncertain_pairs)} pairs verified")
                logger.info(f"  Matches found: {len(matches)} ({100 * len(matches) / (idx + 1):.1f}%)")
                logger.info(f"  Cost so far: ${self.stats['total_cost']:.2f}")
        
        logger.info(f"LLM verification complete:")
        logger.info(f"  Total pairs: {len(uncertain_pairs)}")
        logger.info(f"  Matches: {len(matches)} ({100 * len(matches) / len(uncertain_pairs) if uncertain_pairs else 0:.1f}%)")
        logger.info(f"  Total cost: ${self.stats['total_cost']:.2f}")
        
        return matches