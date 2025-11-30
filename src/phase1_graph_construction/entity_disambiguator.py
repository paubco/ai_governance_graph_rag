"""
Module: entity_disambiguator.py
Phase: 1C - Entity Disambiguation
Purpose: 4-stage entity disambiguation pipeline (exact dedup, FAISS blocking, tiered thresholds, LLM verification)
Author: Pau Barba i Colomer
Created: 2025-11-29
Last Modified: 2025-11-29

Dependencies:
    - faiss-cpu or faiss-gpu: FAISS HNSW index for Stage 2
    - together: LLM API for Stage 4
    - numpy: Array operations
    
Usage:
    from entity_disambiguator import ExactDeduplicator, FAISSBlocker, TieredThresholdFilter, SameJudgeLLM
    
    # Stage 1: Exact deduplication
    dedup = ExactDeduplicator()
    canonical = dedup.deduplicate(entities)
    
    # Stage 2: FAISS blocking
    blocker = FAISSBlocker()
    blocker.build_index(canonical)
    pairs = blocker.find_candidates(canonical, k=50)
    
    # Stage 3: Tiered thresholds
    filter = TieredThresholdFilter()
    filtered = filter.filter_pairs(pairs, canonical)
    
    # Stage 4: LLM verification
    judge = SameJudgeLLM(api_key="your-key")
    matches = judge.verify_batch(filtered['uncertain'], canonical)

Notes:
    - Implements 4-stage pipeline from PHASE_1C_DESIGN_JUSTIFICATION.md
    - Based on Malkov & Yashunin (2020) HNSW, Papadakis et al. (2021) blocking survey
    - Extends RAKG methodology with FAISS blocking for scalability
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
        
        # Combine all chunk_ids
        all_chunk_ids = set()
        for entity in group:
            chunk_ids = entity.get('chunk_ids', [])
            if isinstance(chunk_ids, list):
                all_chunk_ids.update(chunk_ids)
            else:
                all_chunk_ids.add(chunk_ids)
        canonical['chunk_ids'] = sorted(list(all_chunk_ids))
        
        # Type: Most frequent wins (voting)
        type_counts = Counter([e['type'] for e in group])
        canonical['type'] = type_counts.most_common(1)[0][0]
        
        # Description: Keep longest
        descriptions = [e.get('description', '') for e in group]
        if descriptions:
            canonical['description'] = max(descriptions, key=len)
        
        # Preserve embedding if exists (from first entity)
        if 'embedding' in group[0]:
            canonical['embedding'] = group[0]['embedding']
        
        # Add metadata about merge
        canonical['duplicate_count'] = len(group)
        canonical['merged_types'] = list(type_counts.keys()) if len(type_counts) > 1 else None
        
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
                       ef_search: int = 64) -> List[Tuple[int, int, float]]:
        """
        Find candidate pairs using approximate nearest neighbor search
        
        Args:
            entities: List of entities with embeddings
            k: Number of neighbors to retrieve per entity
            ef_search: HNSW parameter (higher = better recall, slower search)
            
        Returns:
            List of (entity_i_idx, entity_j_idx, similarity) tuples
        """
        logger.info(f"Searching k={k} nearest neighbors (ef_search={ef_search})...")
        
        self.index.hnsw.efSearch = ef_search
        
        # Extract and normalize embeddings
        embeddings = np.array([e['embedding'] for e in entities]).astype('float32')
        import faiss
        faiss.normalize_L2(embeddings)
        
        # Search k+1 neighbors (includes self-match)
        distances, indices = self.index.search(embeddings, k + 1)
        
        # Convert to candidate pairs
        pairs = []
        seen_pairs = set()  # Track unique pairs
        
        for i, (dists, neighbors) in enumerate(zip(distances, indices)):
            for neighbor_idx, dist in zip(neighbors, dists):
                # Skip self-matches
                if neighbor_idx == i:
                    continue
                
                # Convert L2 distance to cosine similarity
                # Since vectors are normalized: cos_sim = 1 - (L2_dist^2 / 2)
                similarity = float(1 - (dist ** 2) / 2)
                
                # Filter out dissimilar pairs (negative cosine similarity)
                # For entity disambiguation, we only care about similar entities
                if similarity < 0.0:
                    continue
                
                # Only keep unique pairs (i < j to avoid duplicates)
                pair_key = (min(i, neighbor_idx), max(i, neighbor_idx))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    pairs.append((pair_key[0], pair_key[1], similarity))
        
        self.stats['candidate_pairs'] = len(pairs)
        self.stats['avg_neighbors'] = len(pairs) / len(entities) if entities else 0.0
        
        logger.info(f"Found {len(pairs)} candidate pairs")
        logger.info(f"Average neighbors per entity: {self.stats['avg_neighbors']:.1f}")
        
        return pairs


class TieredThresholdFilter:
    """
    Stage 3: Tiered threshold filtering for auto-merge/reject decisions
    
    Applies three similarity thresholds:
        - High (≥0.95): Auto-merge without LLM
        - Low (<0.80): Auto-reject without LLM
        - Medium (0.80-0.95): Send to LLM for verification
    
    References:
        - Papadakis et al. (2021) - Multi-tier blocking survey
        - Splink (2024) - Multi-threshold approach
    """
    
    def __init__(self, 
                 auto_merge_threshold: float = 0.95,
                 auto_reject_threshold: float = 0.80):
        """
        Initialize threshold filter
        
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
    
    def filter_pairs(self, 
                    pairs: List[Tuple[int, int, float]],
                    entities: List[Dict]) -> Dict:
        """
        Classify pairs into auto-merge, auto-reject, uncertain
        
        Args:
            pairs: List of (i, j, similarity) tuples
            entities: List of entities (for context)
            
        Returns:
            {
                'merged': List[Tuple[int, int]],
                'rejected': List[Tuple[int, int]],
                'uncertain': List[Tuple[int, int, float]]
            }
        """
        logger.info(f"Stage 3: Filtering {len(pairs)} pairs with tiered thresholds...")
        
        merged = []
        rejected = []
        uncertain = []
        
        for i, j, sim in pairs:
            if sim >= self.merge_threshold:
                merged.append((i, j))
            elif sim < self.reject_threshold:
                rejected.append((i, j))
            else:
                uncertain.append((i, j, sim))
        
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
                    merged_pairs: List[Tuple[int, int]]) -> List[Dict]:
        """
        Apply merge decisions using Union-Find algorithm
        
        Uses Union-Find to handle transitivity:
            If A=B and B=C, then A=C
        
        Args:
            entities: List of entities
            merged_pairs: List of (i, j) pairs to merge
            
        Returns:
            List of canonical entities after merging
        """
        logger.info(f"Applying {len(merged_pairs)} merge decisions...")
        
        # Build Union-Find structure
        parent = {i: i for i in range(len(entities))}
        
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
        
        # Union all merged pairs
        for i, j in merged_pairs:
            union(i, j)
        
        # Group entities by root
        groups = defaultdict(list)
        for i in range(len(entities)):
            root = find(i)
            groups[root].append(entities[i])
        
        # Merge each group using ExactDeduplicator logic
        deduplicator = ExactDeduplicator()
        canonical = []
        for group in groups.values():
            merged = deduplicator._merge_group(group)
            canonical.append(merged)
        
        logger.info(f"Merged {len(entities)} → {len(canonical)} entities")
        
        return canonical



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
                 model: str = "Qwen/Qwen2-7B-Instruct",
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
        # Import prompt from centralized location
        try:
            from prompts.prompts import SAMEJUDGE_PROMPT
        except ImportError:
            # Fallback if prompts.py not found
            logger.warning("prompts.py not found, using fallback prompt")
            SAMEJUDGE_PROMPT = """Are these two entities the SAME real-world entity?

Entity 1:
- Name: {entity1_name}
- Type: {entity1_type}
- Description: {entity1_desc}

Entity 2:
- Name: {entity2_name}
- Type: {entity2_type}
- Description: {entity2_desc}

Respond ONLY with valid JSON:
{{
  "result": true or false,
  "canonical_name": "most official name if same",
  "canonical_type": "standardized type if same",
  "reasoning": "brief explanation"
}}

JSON:"""
        
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
                    log_interval: int = 100) -> List[Tuple[int, int]]:
        """
        Verify batch of uncertain pairs (single-threaded)
        
        Args:
            uncertain_pairs: List of (i, j, similarity) tuples
            entities: List of entities
            log_interval: Log progress every N pairs
            
        Returns:
            List of (i, j) pairs confirmed as matches
        """
        logger.info(f"Stage 4: Verifying {len(uncertain_pairs)} pairs with LLM...")
        logger.info(f"Estimated cost: ${len(uncertain_pairs) * self.cost_per_call:.2f}")
        logger.info("Single-threaded - for faster processing, use disambiguation_server.py")
        
        matches = []
        
        for idx, (i, j, sim) in enumerate(uncertain_pairs):
            result = self.verify_pair(entities[i], entities[j])
            
            if result['is_same']:
                matches.append((i, j))
            
            # Log progress
            if (idx + 1) % log_interval == 0 or (idx + 1) == len(uncertain_pairs):
                logger.info(f"Progress: {idx + 1}/{len(uncertain_pairs)} pairs verified")
                logger.info(f"  Matches found: {len(matches)} ({100 * len(matches) / (idx + 1):.1f}%)")
                logger.info(f"  Cost so far: ${self.stats['total_cost']:.2f}")
        
        logger.info(f"LLM verification complete:")
        logger.info(f"  Total pairs: {len(uncertain_pairs)}")
        logger.info(f"  Matches: {len(matches)} ({100 * len(matches) / len(uncertain_pairs):.1f}%)")
        logger.info(f"  Total cost: ${self.stats['total_cost']:.2f}")
        
        return matches