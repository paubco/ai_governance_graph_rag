"""
Module: relation_extractor.py
Phase: 1D - Relation Extraction  
Purpose: RAKG-style relation extraction with MMR diversity-aware chunk selection
Author: Pau Barba i Colomer
Created: 2025-12-01
Last Modified: 2025-12-01

Dependencies:
    - numpy: Vector operations, cosine similarity
    - together: Together.ai API client (or compatible LLM client)
    - json: JSON parsing

Usage:
    from src.phase1_graph_construction.relation_extractor import RAKGRelationExtractor
    
    extractor = RAKGRelationExtractor(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        api_key="your_key"
    )
    
    relations = extractor.extract_relations_for_entity(
        entity=entity_dict,
        all_chunks=chunks_list
    )

Notes:
    - Implements RAKG Equation 24: rel(ei) = LLMrel(entity, retriever(ei))
    - Uses MMR (Carbonell & Goldstein, 1998) for diversity-aware chunk selection
    - Permissive extraction (no entity list constraints), validation in Phase 2B
    - Configurable parameters for testing (similarity threshold, lambda, k)
    
Reference:
    Zhang et al. (2025) - RAKG: Document-level Retrieval Augmented KG Construction
    Carbonell & Goldstein (1998) - Maximal Marginal Relevance (MMR)
"""

# Standard library
import json
import os
import time
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Third-party
import numpy as np
from together import Together
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# Create prompt logging directory
PROMPT_LOG_DIR = Path('logs/phase1d_prompts')
PROMPT_LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Prompt logging directory: {PROMPT_LOG_DIR}")


# ============================================================================
# PROMPT VALIDATION & LOGGING UTILITIES
# ============================================================================

def estimate_token_count(text: str) -> int:
    """
    Rough token count estimate (4 chars ≈ 1 token)
    
    Args:
        text: Input text
        
    Returns:
        int: Estimated token count
    """
    return len(text) // 4


def validate_prompt_size(
    prompt: str, 
    entity_name: str,
    safe_limit: int = 8000,
    warning_limit: int = 6000
) -> Tuple[bool, int]:
    """
    Validate prompt size before LLM call
    
    Thresholds:
    - < 6000 tokens: ✓ OK
    - 6000-8000 tokens: ⚠️ WARNING (log but proceed)
    - > 8000 tokens: 🚨 ERROR (should reduce chunks)
    
    Args:
        prompt: Full prompt text
        entity_name: Entity name (for logging)
        safe_limit: Hard limit (default: 8000)
        warning_limit: Warning threshold (default: 6000)
        
    Returns:
        Tuple[bool, int]: (should_proceed, token_estimate)
    """
    token_estimate = estimate_token_count(prompt)
    
    if token_estimate < warning_limit:
        logger.debug(f"  Prompt size OK: ~{token_estimate} tokens")
        return True, token_estimate
    elif token_estimate < safe_limit:
        logger.warning(f"  ⚠️ Large prompt: ~{token_estimate} tokens for '{entity_name}'")
        logger.warning(f"     (Consider reducing chunks if quality issues occur)")
        return True, token_estimate
    else:
        logger.error(f"  🚨 Prompt exceeds safe limit: ~{token_estimate} tokens for '{entity_name}'")
        logger.error(f"     Hard limit: {safe_limit} tokens")
        logger.error(f"     This prompt is TOO LARGE and should be reduced")
        return False, token_estimate


def save_prompt_to_file(prompt: str, entity_name: str, token_count: int) -> None:
    """
    Save full prompt to log file for debugging
    
    Args:
        prompt: Full prompt text
        entity_name: Entity name (used in filename)
        token_count: Estimated token count
    """
    # Sanitize filename
    safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in entity_name)
    safe_name = safe_name.replace(' ', '_')[:50]  # Limit length
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = PROMPT_LOG_DIR / f"{safe_name}_{timestamp}.txt"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"ENTITY: {entity_name}\n")
            f.write(f"TIMESTAMP: {timestamp}\n")
            f.write(f"ESTIMATED TOKENS: ~{token_count}\n")
            f.write(f"CHARACTERS: {len(prompt)}\n")
            f.write("=" * 80 + "\n\n")
            f.write(prompt)
        
        logger.debug(f"  Prompt saved to: {filename}")
    except Exception as e:
        logger.warning(f"  Failed to save prompt to file: {e}")


# ============================================================================
# CONSTANTS (configurable via constructor)
# ============================================================================

DEFAULT_SEMANTIC_THRESHOLD = 0.85  # For semantic neighbors retrieval
DEFAULT_MMR_LAMBDA = 0.55          # Balance: 0.5=balanced, 1.0=pure relevance
DEFAULT_NUM_CHUNKS = 20            # Final chunks to select per entity
DEFAULT_CANDIDATE_POOL = 200       # Pre-filter size before MMR
                                   # Why 200? Gives MMR a large enough pool for diversity
                                   # while keeping computation manageable
                                   # Formula: O(num_chunks² × candidate_pool) for MMR
                                   # 200 candidates → 20 final is good diversity/speed trade-off


# ============================================================================
# PROMPT BUILDING HELPERS
# ============================================================================

CHUNK_TEMPLATE = """--- Chunk ID: {chunk_id} ---
{chunk_text}
"""

def format_chunks_for_prompt(chunks: list) -> str:
    """
    Format chunks for relation extraction prompt
    
    Args:
        chunks: List of chunk dicts with 'chunk_id' and 'text'
    
    Returns:
        str: Formatted chunks text with separators
    """
    formatted = []
    for chunk in chunks:
        formatted.append(CHUNK_TEMPLATE.format(
            chunk_id=chunk.get('chunk_id', 'unknown'),
            chunk_text=chunk.get('text', '')
        ))
    return "\n".join(formatted)


def build_relation_extraction_prompt(entity: dict, chunks: list, detected_entities: list = None) -> str:
    """
    Build relation extraction prompt for Phase 1D
    
    Args:
        entity: Entity dict with name, type, description
        chunks: List of chunk dicts
        detected_entities: List of detected entity dicts (optional)
    
    Returns:
        str: Complete prompt ready for LLM
    """
    from src.prompts.prompts import RELATION_EXTRACTION_PROMPT
    
    chunks_text = format_chunks_for_prompt(chunks)
    
    # Format detected entities list
    if detected_entities:
        detected_list = "\n".join([
            f"- {e['name']} [{e['type']}]"
            for e in detected_entities
        ])
    else:
        detected_list = "(Entity detection not available - extract any entities)"
    
    return RELATION_EXTRACTION_PROMPT.format(
        entity_name=entity.get('name', 'Unknown'),
        entity_type=entity.get('type', 'Unknown'),
        entity_description=entity.get('description', 'No description'),
        detected_entities_list=detected_list,
        chunks_text=chunks_text
    )


def extract_relations_json(response_text: str) -> str:
    """
    Extract JSON from LLM response (removes markdown wrappers only)
    
    Args:
        response_text: Raw LLM response
    
    Returns:
        str: Cleaned JSON string
    """
    # Remove markdown code blocks
    cleaned = response_text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "")
    cleaned = cleaned.strip()
    
    return cleaned


# ============================================================================
# VECTOR SIMILARITY HELPERS
# ============================================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        float: Cosine similarity in [0, 1]
    
    Example:
        >>> vec1 = np.array([1.0, 0.0, 0.0])
        >>> vec2 = np.array([1.0, 0.0, 0.0])
        >>> cosine_similarity(vec1, vec2)
        1.0
    """
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    
    if norm_product == 0:
        return 0.0
    
    return dot_product / norm_product


def batch_cosine_similarity(query_vec: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and multiple vectors (vectorized)
    
    Args:
        query_vec: Query vector (d,)
        vectors: Matrix of vectors (n, d)
        
    Returns:
        np.ndarray: Similarities (n,)
    
    Example:
        >>> query = np.array([1.0, 0.0])
        >>> vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> batch_cosine_similarity(query, vecs)
        array([1., 0.])
    """
    # Normalize query
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    
    # Normalize all vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    vectors_norm = vectors / norms
    
    # Dot product
    similarities = np.dot(vectors_norm, query_norm)
    
    return similarities


# ============================================================================
# MAIN CLASS
# ============================================================================

class RAKGRelationExtractor:
    """
    RAKG-style relation extraction with MMR diversity-aware chunk selection
    
    Implements the relation extraction component of RAKG methodology:
    1. Gather candidate chunks (direct + semantic neighbors)
    2. MMR selection for diversity (Maximal Marginal Relevance)
    3. LLM-based OPENIE extraction
    4. Post-processing and validation
    
    CANDIDATE_POOL_SIZE vs NUM_CHUNKS:
    ------------------------------------
    - candidate_pool_size: Pre-filter pool (default: 200)
      → How many chunks to consider BEFORE MMR selection
      → Larger pool = more diversity options for MMR
      → But slower MMR computation: O(k² × pool_size)
      
    - num_chunks: Final selection (default: 20)
      → How many chunks actually sent to LLM
      → Limited by context window (~20 chunks = 1100 tokens)
      
    Example flow for entity "AI" with 1,780 direct chunks:
      1. Gather candidates: 1,780 direct + semantic → 1,780 total (too many!)
      2. Pre-filter: Take top 200 by relevance → 200 candidates
      3. MMR select: Choose 20 diverse chunks → 20 final chunks
      4. LLM: Extract relations from these 20 chunks
    
    Why not just top-20 by relevance?
      → Would get 20 very similar chunks (all about same aspect)
      → MMR needs larger pool (200) to find diverse chunks
      → Trade-off: 200 is large enough for diversity, small enough for speed
    
    Attributes:
        model_name (str): Together.ai model name
        api_key (str): API key (auto-loaded from .env if not provided)
        semantic_threshold (float): Cosine similarity threshold for semantic neighbors
        mmr_lambda (float): MMR balance parameter (relevance vs diversity)
        num_chunks (int): Final number of chunks to select
        candidate_pool_size (int): Pre-filter pool size before MMR
        temperature (float): LLM temperature (0.0 for deterministic)
        max_tokens (int): Max tokens for LLM response
        
    Example:
        >>> # API key from .env file
        >>> extractor = RAKGRelationExtractor(
        ...     model_name="Qwen/Qwen2.5-7B-Instruct",
        ...     semantic_threshold=0.85,
        ...     mmr_lambda=0.55,
        ...     num_chunks=20,
        ...     candidate_pool_size=200  # Pre-filter pool
        ... )
        >>> relations = extractor.extract_relations_for_entity(entity, chunks)
    
    Reference:
        RAKG paper Section D: Relationship Network Construction
    """
    
    def __init__(
        self,
        model_name: str,
        api_key: str = None,
        semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
        mmr_lambda: float = DEFAULT_MMR_LAMBDA,
        num_chunks: int = DEFAULT_NUM_CHUNKS,
        candidate_pool_size: int = DEFAULT_CANDIDATE_POOL,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        entity_cooccurrence_file: str = None,
        normalized_entities_file: str = None
    ):
        """
        Initialize RAKG Relation Extractor
        
        Args:
            model_name: Together.ai model (e.g., "Qwen/Qwen2.5-7B-Instruct-Turbo")
            api_key: Together.ai API key (if None, loads from .env TOGETHER_API_KEY)
            semantic_threshold: Cosine similarity threshold (default: 0.85)
            mmr_lambda: MMR balance (0.5=balanced, 1.0=relevance only)
            num_chunks: Final chunks to select (default: 20)
            candidate_pool_size: Pre-filter pool size (default: 200)
            temperature: LLM temperature (default: 0.0 for deterministic)
            max_tokens: Max LLM response tokens (default: 4000)
            entity_cooccurrence_file: Path to entity co-occurrence JSON (optional)
            normalized_entities_file: Path to normalized entities JSON (optional)
        """
        self.model_name = model_name
        
        # API key: use provided, or load from .env
        if api_key is None:
            api_key = os.getenv('TOGETHER_API_KEY')
            if not api_key:
                raise ValueError(
                    "TOGETHER_API_KEY not found. Either:\n"
                    "1. Pass api_key parameter, or\n"
                    "2. Add TOGETHER_API_KEY to .env file"
                )
        self.api_key = api_key
        
        self.semantic_threshold = semantic_threshold
        self.mmr_lambda = mmr_lambda
        self.num_chunks = num_chunks
        self.candidate_pool_size = candidate_pool_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize Together client
        self.client = Together(api_key=self.api_key)
        
        # Load entity co-occurrence matrix (optional, for entity-aware diversity)
        self.entity_cooccurrence = None
        self.normalized_entities = None
        self.entity_map = None
        
        if entity_cooccurrence_file:
            self._load_entity_cooccurrence(entity_cooccurrence_file)
        
        if normalized_entities_file:
            self._load_normalized_entities(normalized_entities_file)
        
        logger.info(f"Initialized RAKGRelationExtractor")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Semantic threshold: {semantic_threshold}")
        logger.info(f"  MMR lambda: {mmr_lambda}")
        logger.info(f"  Chunks per entity: {num_chunks}")
        logger.info(f"  Candidate pool: {candidate_pool_size}")
        if self.entity_cooccurrence:
            logger.info(f"  Entity co-occurrence: Loaded ({len(self.entity_cooccurrence)} chunks)")
        if self.normalized_entities:
            logger.info(f"  Normalized entities: Loaded ({len(self.normalized_entities)} entities)")
    
    def _load_entity_cooccurrence(self, cooccurrence_file: str):
        """Load entity co-occurrence matrix"""
        with open(cooccurrence_file, 'r', encoding='utf-8') as f:
            self.entity_cooccurrence = json.load(f)
    
    def _load_normalized_entities(self, entities_file: str):
        """Load normalized entities and build lookup map"""
        with open(entities_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different formats
        if isinstance(data, list):
            entities = data
        elif isinstance(data, dict) and 'entities' in data:
            entities = data['entities']
        elif isinstance(data, dict):
            entities = list(data.values())
        else:
            raise ValueError(f"Unexpected format in {entities_file}")
        
        self.normalized_entities = entities
        self.entity_map = {e['id']: e for e in entities}
    
    # ========================================================================
    # STEP 1: CANDIDATE GATHERING
    # ========================================================================
    
    def gather_candidate_chunks(
        self,
        entity: Dict,
        all_chunks: List[Dict]
    ) -> List[Dict]:
        """
        Gather candidate chunks for entity (direct + semantic neighbors)
        
        Strategy:
        1. Direct chunks: Where entity was originally extracted (chunk_ids)
        2. Semantic neighbors: High cosine similarity to entity embedding
        3. Deduplicate by chunk_id
        4. Limit to candidate_pool_size
        
        Args:
            entity: Entity dict with 'embedding', 'chunk_ids', 'name'
            all_chunks: List of all chunk dicts with 'chunk_id', 'embedding', 'text'
        
        Returns:
            List[Dict]: Candidate chunks (up to candidate_pool_size)
        
        Example:
            >>> entity = {"name": "GDPR", "embedding": [...], "chunk_ids": ["c1", "c2"]}
            >>> candidates = extractor.gather_candidate_chunks(entity, all_chunks)
            >>> len(candidates)
            200
        
        Note:
            - Direct chunks take priority (guaranteed to mention entity)
            - Semantic search adds chunks discussing entity without explicit mention
        """
        logger.debug(f"Gathering candidates for entity: {entity.get('name', 'Unknown')}")
        
        entity_embedding = np.array(entity['embedding'])
        direct_chunk_ids = set(entity.get('chunk_ids', []))
        
        # Build chunk lookup
        chunks_dict = {chunk['chunk_id']: chunk for chunk in all_chunks}
        
        # Step 1: Direct chunks
        direct_chunks = []
        for chunk_id in direct_chunk_ids:
            if chunk_id in chunks_dict:
                direct_chunks.append(chunks_dict[chunk_id])
        
        logger.debug(f"  Direct chunks: {len(direct_chunks)}")
        
        # Step 2: Semantic neighbors (if we need more chunks)
        semantic_chunks = []
        if len(direct_chunks) < self.candidate_pool_size:
            # Vectorized similarity computation
            chunk_embeddings = np.array([chunk['embedding'] for chunk in all_chunks])
            similarities = batch_cosine_similarity(entity_embedding, chunk_embeddings)
            
            # Filter by threshold and exclude direct chunks
            for i, similarity in enumerate(similarities):
                if similarity >= self.semantic_threshold:
                    chunk = all_chunks[i]
                    if chunk['chunk_id'] not in direct_chunk_ids:
                        semantic_chunks.append((chunk, similarity))
            
            # Sort by similarity (descending)
            semantic_chunks.sort(key=lambda x: x[1], reverse=True)
            semantic_chunks = [chunk for chunk, _ in semantic_chunks]
            
            logger.debug(f"  Semantic neighbors: {len(semantic_chunks)}")
        
        # Combine: direct first (higher priority), then semantic
        candidates = direct_chunks + semantic_chunks
        
        # Limit to pool size
        candidates = candidates[:self.candidate_pool_size]
        
        logger.debug(f"  Total candidates: {len(candidates)}")
        
        return candidates
    
    # ========================================================================
    # STEP 2: MMR DIVERSITY SELECTION
    # ========================================================================
    
    def mmr_select_chunks(
        self,
        entity_embedding: np.ndarray,
        candidate_chunks: List[Dict],
        k: int = None
    ) -> List[Dict]:
        """
        Select k diverse chunks using Maximal Marginal Relevance (MMR)
        
        MMR Score = λ × Relevance - (1-λ) × MaxSimilarity_to_selected
        
        Where:
        - Relevance = cosine_similarity(chunk, entity)
        - MaxSimilarity_to_selected = max similarity to already selected chunks
        - λ = balance parameter (self.mmr_lambda)
        
        Args:
            entity_embedding: Entity embedding vector
            candidate_chunks: Pool of candidate chunks
            k: Number of chunks to select (default: self.num_chunks)
        
        Returns:
            List[Dict]: Selected diverse chunks
        
        Example:
            >>> entity_emb = np.array([1.0, 0.0, ...])
            >>> candidates = [...]  # 200 chunks
            >>> selected = extractor.mmr_select_chunks(entity_emb, candidates, k=20)
            >>> len(selected)
            20
        
        Reference:
            Carbonell & Goldstein (1998) - The Use of MMR, Diversity-Based
            Reranking for Reordering Documents and Producing Summaries
        """
        if k is None:
            k = self.num_chunks
        
        if len(candidate_chunks) <= k:
            logger.debug(f"  Candidates ({len(candidate_chunks)}) <= k ({k}), returning all")
            return candidate_chunks
        
        logger.debug(f"  MMR selection: {len(candidate_chunks)} candidates → {k} chunks")
        
        # Precompute chunk embeddings matrix
        chunk_embeddings = np.array([chunk['embedding'] for chunk in candidate_chunks])
        
        # Precompute relevance scores (similarity to entity)
        relevance_scores = batch_cosine_similarity(entity_embedding, chunk_embeddings)
        
        # MMR selection loop
        selected_indices = []
        selected_embeddings = []
        remaining_indices = list(range(len(candidate_chunks)))
        
        for iteration in range(k):
            best_score = -1.0
            best_idx = None
            
            for idx in remaining_indices:
                # Relevance component
                relevance = relevance_scores[idx]
                
                # Diversity component
                if selected_embeddings:
                    # Max similarity to any selected chunk
                    similarities_to_selected = [
                        cosine_similarity(chunk_embeddings[idx], sel_emb)
                        for sel_emb in selected_embeddings
                    ]
                    max_sim_to_selected = max(similarities_to_selected)
                else:
                    max_sim_to_selected = 0.0
                
                # MMR score
                mmr_score = (
                    self.mmr_lambda * relevance - 
                    (1 - self.mmr_lambda) * max_sim_to_selected
                )
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            # Add best chunk to selected
            selected_indices.append(best_idx)
            selected_embeddings.append(chunk_embeddings[best_idx])
            remaining_indices.remove(best_idx)
        
        # Return selected chunks in order
        selected_chunks = [candidate_chunks[i] for i in selected_indices]
        
        logger.debug(f"  MMR selection complete: {len(selected_chunks)} chunks")
        
        return selected_chunks
    
    def two_stage_mmr_select(
        self,
        entity: Dict,
        candidate_chunks: List[Dict]
    ) -> List[Dict]:
        """
        Two-stage chunk selection: semantic diversity + entity coverage
        
        Novel contribution: Entity-aware diversity on top of semantic MMR.
        
        Stage 1: Semantic MMR (200 candidates → 40 diverse chunks)
          - Standard MMR using embedding similarity
          - Selects semantically diverse and relevant chunks
        
        Stage 2: Entity Coverage Maximization (40 → 20 final chunks)
          - Greedily select chunks that introduce new entities
          - Maximizes entity diversity for richer relation extraction
        
        This improves over pure semantic MMR by ensuring selected chunks
        cover a diverse set of entities, not just semantically diverse
        expressions of the same entities.
        
        Args:
            entity: Entity dict with 'id', 'embedding'
            candidate_chunks: Pool of candidate chunks
        
        Returns:
            List[Dict]: 20 chunks with both semantic and entity diversity
        
        Note:
            Requires entity_cooccurrence to be loaded. Falls back to
            standard MMR if co-occurrence not available.
        """
        # Stage 1: Semantic diversity (200 → 40)
        entity_embedding = np.array(entity['embedding'])
        semantic_diverse = self.mmr_select_chunks(
            entity_embedding,
            candidate_chunks,
            k=min(40, len(candidate_chunks))
        )
        
        # If no co-occurrence data, fall back to taking first 20
        if not self.entity_cooccurrence:
            logger.debug(f"  No co-occurrence data, using first {self.num_chunks} from MMR")
            return semantic_diverse[:self.num_chunks]
        
        # Stage 2: Entity coverage maximization (40 → 20)
        logger.debug(f"  Stage 2: Entity coverage maximization ({len(semantic_diverse)} → {self.num_chunks})")
        
        entity_id = entity['id']
        selected = []
        seen_entities = set([entity_id])  # Start with target entity
        
        # Score each chunk by number of new entities it introduces
        chunk_scores = []
        for chunk in semantic_diverse:
            chunk_id = chunk.get('chunk_id', chunk.get('id', ''))
            cooccurring = set(self.entity_cooccurrence.get(chunk_id, []))
            
            # Remove target entity
            cooccurring.discard(entity_id)
            
            # Count new entities
            new_entities = cooccurring - seen_entities
            score = len(new_entities)
            
            chunk_scores.append((chunk, cooccurring, score))
        
        # Greedily select chunks with highest new entity count
        for _ in range(min(self.num_chunks, len(chunk_scores))):
            if not chunk_scores:
                break
            
            # Find chunk with most new entities
            best_idx = max(range(len(chunk_scores)), key=lambda i: chunk_scores[i][2])
            best_chunk, best_entities, _ = chunk_scores.pop(best_idx)
            
            selected.append(best_chunk)
            seen_entities.update(best_entities)
            
            # Recalculate scores for remaining chunks
            for i in range(len(chunk_scores)):
                chunk, cooccurring, _ = chunk_scores[i]
                new_entities = cooccurring - seen_entities
                chunk_scores[i] = (chunk, cooccurring, len(new_entities))
        
        logger.debug(f"  Entity coverage: {len(seen_entities) - 1} unique entities (excluding target)")
        
        return selected
    
    # ========================================================================
    # STEP 3: LLM EXTRACTION
    # ========================================================================
    
    def extract_relations_llm(
        self,
        prompt: str,
        stop_sequences: List[str] = None
    ) -> Dict:
        """
        Call LLM to extract relations from prompt
        
        Args:
            prompt: Complete extraction prompt
            stop_sequences: Stop sequences for LLM (optional)
        
        Returns:
            Dict: Parsed JSON response with 'relations' key
        
        Raises:
            json.JSONDecodeError: If LLM response is not valid JSON
            Exception: If API call fails
        
        Example:
            >>> prompt = "Extract relations from..."
            >>> response = extractor.extract_relations_llm(prompt)
            >>> response['relations']
            [{"subject": "GDPR", "predicate": "regulates", "object": "data"}]
        
        Note:
            - Uses temperature=0.0 for deterministic output
            - Implements graceful error handling
            - Logs response details for debugging
        """
        if stop_sequences is None:
            stop_sequences = ["```", "\n\nNote:", "Explanation:"]
        
        try:
            # Call API - Use CHAT endpoint for instruct models
            logger.debug(f"  Calling LLM API...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop_sequences
            )
            
            # Extract raw response from chat format
            raw_text = response.choices[0].message.content.strip()
            
            # Log response details
            logger.debug(f"  ✓ Response received: {len(raw_text)} chars")
            logger.debug(f"  Response preview (first 200 chars):")
            logger.debug(f"    {raw_text[:200]}")
            
            # Check for truncation (incomplete response)
            if not raw_text.rstrip().endswith('}'):
                logger.error(f"  ✗ Response truncated (doesn't end with '}}')!")
                logger.error(f"    Response length: {len(raw_text)} chars")
                logger.error(f"    Last 100 chars: ...{raw_text[-100:]}")
                logger.error(f"    This means max_tokens ({self.max_tokens}) was too small")
                logger.error(f"    Try increasing max_tokens or reducing num_chunks")
                return {"relations": []}  # Graceful fallback
            
            # Handle empty response
            if not raw_text:
                logger.error(f"  ✗ Empty response from LLM")
                logger.error(f"    This usually means:")
                logger.error(f"    1. Prompt too large (check token count)")
                logger.error(f"    2. API timeout")
                logger.error(f"    3. Model refused to respond")
                return {"relations": []}  # Graceful fallback
            
            # Clean and parse JSON
            cleaned = extract_relations_json(raw_text)
            
            if not cleaned or cleaned == "{}":
                logger.warning(f"  ⚠️ Empty JSON after cleaning")
                logger.debug(f"  Raw response was: {raw_text[:500]}")
                return {"relations": []}
            
            # Parse JSON
            parsed = json.loads(cleaned)
            
            # Validate structure
            if 'relations' not in parsed:
                logger.warning(f"  ⚠️ Response missing 'relations' key")
                logger.debug(f"  Keys found: {list(parsed.keys())}")
                return {"relations": []}
            
            logger.debug(f"  ✓ Parsed {len(parsed.get('relations', []))} relations")
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"  ✗ JSON parse error: {e}")
            logger.error(f"  Cleaned JSON (first 800 chars): {cleaned[:800]}")
            logger.error(f"  Raw response (first 800 chars): {raw_text[:800] if 'raw_text' in locals() else 'N/A'}")
            
            # Save full malformed response to file for debugging
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            error_file = PROMPT_LOG_DIR / f"ERROR_malformed_json_{timestamp}.txt"
            try:
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write("MALFORMED JSON RESPONSE\n")
                    f.write("=" * 80 + "\n\n")
                    f.write("ERROR:\n")
                    f.write(str(e) + "\n\n")
                    f.write("CLEANED JSON:\n")
                    f.write(cleaned + "\n\n")
                    f.write("RAW RESPONSE:\n")
                    f.write(raw_text if 'raw_text' in locals() else 'N/A')
                logger.error(f"  Saved malformed response to: {error_file}")
            except Exception as save_error:
                logger.debug(f"  Could not save error file: {save_error}")
            
            return {"relations": []}  # Graceful fallback
        
        except Exception as e:
            logger.error(f"  ✗ LLM API error: {e}")
            import traceback
            logger.debug(f"  Traceback: {traceback.format_exc()}")
            return {"relations": []}  # Graceful fallback
    
    # ========================================================================
    # MAIN EXTRACTION METHOD
    # ========================================================================
    
    def extract_relations_for_entity(
        self,
        entity: Dict,
        all_chunks: List[Dict],
        prompt_template: str = None,
        save_prompt: bool = True
    ) -> List[Dict]:
        """
        Extract all relations for a given entity (end-to-end pipeline)
        
        Pipeline:
        1. Gather candidate chunks (direct + semantic)
        2. MMR selection for diversity
        3. Build prompt with selected chunks
        4. Validate prompt size
        5. LLM extraction
        6. Post-processing
        
        Args:
            entity: Entity dict with name, type, description, embedding, chunk_ids
            all_chunks: List of all chunks with chunk_id, text, embedding
            prompt_template: Custom prompt (optional, uses default if None)
            save_prompt: Save full prompt to log file (default: True)
        
        Returns:
            List[Dict]: Extracted relations with subject, predicate, object, chunk_ids
        
        Example:
            >>> entity = {"name": "GDPR", "type": "Regulation", ...}
            >>> relations = extractor.extract_relations_for_entity(entity, all_chunks)
            >>> len(relations)
            15
        
        Note:
            - Returns empty list if extraction fails
            - Logs errors but doesn't raise (graceful degradation)
            - Validates prompt size before LLM call
        """
        entity_name = entity.get('name', 'Unknown')
        entity_type = entity.get('type', 'Unknown')
        logger.info(f"\nEntity: {entity_name} [{entity_type}]")
        
        try:
            # Step 1: Gather candidates
            logger.info(f"  Gathering candidates...")
            start_time = time.time()
            candidates = self.gather_candidate_chunks(entity, all_chunks)
            logger.info(f"    ✓ Gathered {len(candidates)} candidates ({time.time() - start_time:.2f}s)")
            
            if not candidates:
                logger.warning(f"  ⚠️ No candidates found for {entity_name}")
                return []
            
            # Step 2: Two-stage MMR selection (semantic + entity diversity)
            logger.info(f"  Two-stage MMR selection...")
            start_time = time.time()
            selected_chunks = self.two_stage_mmr_select(entity, candidates)
            logger.info(f"    ✓ Selected {len(selected_chunks)} chunks ({time.time() - start_time:.2f}s)")
            
            # Step 2.5: Get detected entities from selected chunks
            detected_entities = []
            if self.entity_cooccurrence and self.entity_map:
                entity_id = entity['id']
                detected_entity_ids = set()
                
                for chunk in selected_chunks:
                    chunk_id = chunk.get('chunk_id', chunk.get('id', ''))
                    cooccurring = self.entity_cooccurrence.get(chunk_id, [])
                    detected_entity_ids.update(cooccurring)
                
                # Remove target entity
                detected_entity_ids.discard(entity_id)
                
                # Resolve entity names
                detected_entities = [
                    self.entity_map[eid]
                    for eid in detected_entity_ids
                    if eid in self.entity_map
                ]
                
                logger.info(f"    ✓ Detected {len(detected_entities)} co-occurring entities")
            
            # Step 3: Build prompt with detected entities
            logger.info(f"  Building prompt...")
            prompt = build_relation_extraction_prompt(entity, selected_chunks, detected_entities)
            prompt_chars = len(prompt)
            logger.info(f"    ✓ Prompt built: {prompt_chars:,} chars")
            
            # Step 4: Validate prompt size
            logger.info(f"  Validating prompt size...")
            should_proceed, token_estimate = validate_prompt_size(prompt, entity_name)
            
            if not should_proceed:
                logger.error(f"  ✗ Prompt too large (~{token_estimate} tokens), skipping entity")
                return []
            
            # Log prompt preview
            logger.debug(f"  Prompt preview (first 500 chars):")
            logger.debug(f"    {prompt[:500]}...")
            
            # Save full prompt to file
            if save_prompt:
                save_prompt_to_file(prompt, entity_name, token_estimate)
            
            # Step 5: LLM extraction
            logger.info(f"  Calling LLM (~{token_estimate} tokens)...")
            start_time = time.time()
            response = self.extract_relations_llm(prompt)
            logger.info(f"    ✓ LLM responded ({time.time() - start_time:.2f}s)")
            
            # Step 6: Post-processing
            relations = response.get('relations', [])
            
            # Add metadata
            for relation in relations:
                relation['extracted_from_entity'] = entity.get('id', entity_name)
            
            if len(relations) == 0:
                logger.warning(f"  ⚠️ No relations extracted (try lowering threshold)")
            else:
                logger.info(f"  ✓ Extracted {len(relations)} relations")
            
            return relations
            
        except Exception as e:
            logger.error(f"  ✗ Failed to extract relations for {entity_name}: {e}")
            import traceback
            logger.debug(f"  Traceback: {traceback.format_exc()}")
            return []