# -*- coding: utf-8 -*-
"""
RAKG-style relation extractor with entity_id-based output (v2.0).

Implements retrieval-augmented relation extraction using two-stage MMR for semantic
and entity diversity in chunk selection. v2.0 uses entity_ids throughout to avoid
post-hoc name→ID normalization that caused 52% entity mismatch in v1.0.

Key v2.0 Changes:
- Entity lookup keyed by entity_id (not name)
- Co-occurrence matrices contain entity_ids (not names)
- LLM prompt constrains output to valid entity_ids
- Relations output subject_id/object_id directly

Algorithm:
    1. Semantic MMR: Select top-k chunks by entity embedding similarity with diversity
    2. Entity MMR: Add chunks containing co-occurring entities (diversified selection)
    3. Threshold check: Trigger second batch if first batch yields < threshold relations
    4. LLM extraction: OpenIE triplet extraction with ID-constrained JSON schema
    5. Validation: Verify output IDs exist in detected entity set

Example:
    extractor = RAKGRelationExtractor(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        num_chunks=6, mmr_lambda=0.65
    )
    relations = extractor.extract_relations_for_entity(entity, chunks)
"""

# Standard library
import json
import os
import re
import time
import logging
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path

# Third-party
import numpy as np
from together import Together
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logging
from src.prompts.prompts import RELATION_EXTRACTION_PROMPT, METADATA_RELATION_EXTRACTION_PROMPT

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_SEMANTIC_THRESHOLD = 0.85
DEFAULT_MMR_LAMBDA = 0.65
DEFAULT_NUM_CHUNKS = 6
DEFAULT_CANDIDATE_POOL = 200
DEFAULT_SECOND_ROUND_THRESHOLD = 0.25

# Entity type classification
SEMANTIC_TYPES = {
    'RegulatoryConcept', 'TechnicalConcept', 'PoliticalConcept', 'EconomicConcept',
    'Regulation', 'Technology', 'Organization', 'Location', 'Risk'
}
ACADEMIC_TYPES = {'Citation', 'Author', 'Journal', 'Affiliation'}
SKIP_TYPES = {'Document', 'DocumentSection'}
CONCEPT_TYPES = {
    'RegulatoryConcept', 'TechnicalConcept', 'PoliticalConcept', 'EconomicConcept'
}


# ============================================================================
# VECTOR HELPERS
# ============================================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if norm_product == 0:
        return 0.0
    return dot_product / norm_product


def batch_cosine_similarity(query_vec: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and multiple vectors."""
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    vectors_norm = vectors / norms
    return np.dot(vectors_norm, query_norm)


# ============================================================================
# PROMPT HELPERS
# ============================================================================

def format_chunks_for_prompt(chunks: List[Dict]) -> str:
    """Format chunks for relation extraction prompt."""
    formatted = []
    for chunk in chunks:
        chunk_id = chunk.get('chunk_id', 'unknown')
        text = chunk.get('text', '')
        formatted.append(f"--- Chunk ID: {chunk_id} ---\n{text}")
    return "\n\n".join(formatted)


def format_detected_entities_with_ids(entities: List[Dict]) -> str:
    """
    Format detected entities with IDs for prompt.
    
    v2.0: Includes entity_id so LLM can use it in output.
    
    Format: "- ent_abc123: Entity Name [Type]"
    """
    lines = []
    for entity in entities:
        entity_id = entity.get('entity_id', 'unknown')
        name = entity.get('name', 'Unknown')
        entity_type = entity.get('type', 'Unknown')
        lines.append(f"- {entity_id}: {name} [{entity_type}]")
    return "\n".join(lines)


def build_relation_prompt(
    entity: Dict,
    chunks: List[Dict],
    detected_entities: List[Dict],
    track: str = 'semantic'
) -> str:
    """
    Build relation extraction prompt with entity IDs.
    
    Args:
        entity: Target entity dict with entity_id, name, type, description
        chunks: List of context chunks
        detected_entities: List of detected entity dicts in chunks
        track: 'semantic' or 'academic'
    
    Returns:
        Formatted prompt string
    """
    chunks_text = format_chunks_for_prompt(chunks)
    detected_list = format_detected_entities_with_ids(detected_entities)
    
    if track == 'academic':
        return METADATA_RELATION_EXTRACTION_PROMPT.format(
            entity_id=entity.get('entity_id', ''),
            entity_name=entity.get('name', 'Unknown'),
            entity_type=entity.get('type', 'Unknown'),
            detected_entities_list=detected_list,
            chunks_text=chunks_text
        )
    else:
        return RELATION_EXTRACTION_PROMPT.format(
            entity_id=entity.get('entity_id', ''),
            entity_name=entity.get('name', 'Unknown'),
            entity_type=entity.get('type', 'Unknown'),
            entity_description=entity.get('description', 'No description'),
            detected_entities_list=detected_list,
            chunks_text=chunks_text
        )


def normalize_predicate(predicate: str) -> str:
    """
    Normalize predicate to lowercase_underscore format.
    
    Examples:
        "can be a substitute for" → "can_be_substitute_for"
        "Is Related To" → "is_related_to"
    """
    pred = predicate.lower().strip()
    pred = re.sub(r'[^a-z0-9\s]', '', pred)
    pred = re.sub(r'\s+', '_', pred)
    pred = re.sub(r'_+', '_', pred)
    return pred.strip('_')


# ============================================================================
# MAIN EXTRACTOR CLASS
# ============================================================================

class RAKGRelationExtractor:
    """
    RAKG-style relation extraction with entity_id-based output (v2.0).
    
    Key differences from v1.0:
    - entity_lookup keyed by entity_id (not name)
    - cooccurrence matrices contain entity_ids
    - Output relations use subject_id/object_id directly
    - No post-hoc normalization needed
    """
    
    def __init__(
        self,
        model_name: str = 'mistralai/Mistral-7B-Instruct-v0.3',
        api_key: str = None,
        semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
        mmr_lambda: float = DEFAULT_MMR_LAMBDA,
        num_chunks: int = DEFAULT_NUM_CHUNKS,
        candidate_pool_size: int = DEFAULT_CANDIDATE_POOL,
        temperature: float = 0.0,
        max_tokens: int = 16000,
        second_round_threshold: float = DEFAULT_SECOND_ROUND_THRESHOLD,
        entity_lookup_file: str = None,
        cooccurrence_semantic_file: str = None,
        cooccurrence_concept_file: str = None,
        debug_mode: bool = False
    ):
        """
        Initialize RAKG Relation Extractor v2.0.
        
        Args:
            model_name: Together.ai model
            api_key: API key (loads from .env if None)
            semantic_threshold: Cosine similarity for semantic neighbors
            mmr_lambda: MMR balance (0.5=balanced, 1.0=relevance only)
            num_chunks: Chunks per extraction batch
            candidate_pool_size: Pre-filter pool before MMR
            temperature: LLM temperature
            max_tokens: Max LLM response tokens
            second_round_threshold: Distance threshold for second batch
            entity_lookup_file: Path to entity_id_lookup.json
            cooccurrence_semantic_file: Path to cooccurrence_semantic.json
            cooccurrence_concept_file: Path to cooccurrence_concept.json
            debug_mode: Enable prompt/response logging
        """
        self.model_name = model_name
        self.debug_mode = debug_mode
        
        # API key
        if api_key is None:
            api_key = os.getenv('TOGETHER_API_KEY')
            if not api_key:
                raise ValueError("TOGETHER_API_KEY not found in environment")
        
        self.client = Together(api_key=api_key)
        
        # Parameters
        self.semantic_threshold = semantic_threshold
        self.mmr_lambda = mmr_lambda
        self.num_chunks = num_chunks
        self.candidate_pool_size = candidate_pool_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.second_round_threshold = second_round_threshold
        
        # Entity lookup (entity_id → entity dict)
        self.entity_lookup: Dict[str, Dict] = {}
        if entity_lookup_file:
            self._load_entity_lookup(entity_lookup_file)
        
        # Co-occurrence matrices (chunk_id → [entity_ids])
        self.cooccurrence_semantic: Dict[str, List[str]] = {}
        self.cooccurrence_concept: Dict[str, List[str]] = {}
        
        if cooccurrence_semantic_file:
            self._load_cooccurrence(cooccurrence_semantic_file, 'semantic')
        if cooccurrence_concept_file:
            self._load_cooccurrence(cooccurrence_concept_file, 'concept')
        
        logger.info(f"Initialized RAKGRelationExtractor v2.0")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Entity lookup: {len(self.entity_lookup):,} entities")
        logger.info(f"  Semantic cooccurrence: {len(self.cooccurrence_semantic):,} chunks")
        logger.info(f"  Concept cooccurrence: {len(self.cooccurrence_concept):,} chunks")
    
    def _load_entity_lookup(self, filepath: str):
        """Load entity_id → entity lookup."""
        logger.info(f"Loading entity lookup from {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            self.entity_lookup = json.load(f)
        logger.info(f"  Loaded {len(self.entity_lookup):,} entities")
    
    def _load_cooccurrence(self, filepath: str, matrix_type: str):
        """Load co-occurrence matrix."""
        logger.info(f"Loading {matrix_type} cooccurrence from {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if matrix_type == 'semantic':
            self.cooccurrence_semantic = data
        elif matrix_type == 'concept':
            self.cooccurrence_concept = data
        
        logger.info(f"  Loaded {len(data):,} chunks")
    
    def _classify_entity(self, entity: Dict) -> str:
        """
        Classify entity extraction strategy.
        
        Returns:
            'semantic': Full OpenIE extraction (Track 1)
            'academic': Subject-constrained discusses (Track 2)
            'skip': No extraction needed
        """
        entity_type = entity.get('type', '')
        
        if entity_type in SKIP_TYPES:
            return 'skip'
        elif entity_type in ACADEMIC_TYPES:
            return 'academic'
        elif entity_type in SEMANTIC_TYPES:
            return 'semantic'
        else:
            return 'skip'
    
    def _get_cooccurrence_matrix(self, strategy: str) -> Dict[str, List[str]]:
        """Get appropriate co-occurrence matrix for strategy."""
        if strategy == 'academic':
            return self.cooccurrence_concept  # Objects must be concepts
        else:
            return self.cooccurrence_semantic
    
    # ========================================================================
    # CHUNK SELECTION
    # ========================================================================
    
    def gather_candidate_chunks(
        self,
        entity: Dict,
        all_chunks: List[Dict]
    ) -> List[Dict]:
        """
        Gather candidate chunks for entity (direct + semantic neighbors).
        
        Returns up to candidate_pool_size chunks.
        """
        entity_embedding = np.array(entity.get('embedding', []))
        direct_chunk_ids = set(entity.get('chunk_ids', []))
        
        # Build chunk lookup
        chunks_dict = {chunk['chunk_id']: chunk for chunk in all_chunks}
        
        # Direct chunks first
        direct_chunks = [chunks_dict[cid] for cid in direct_chunk_ids if cid in chunks_dict]
        
        # Semantic neighbors if needed
        semantic_chunks = []
        if len(direct_chunks) < self.candidate_pool_size and len(entity_embedding) > 0:
            chunk_embeddings = np.array([c['embedding'] for c in all_chunks])
            similarities = batch_cosine_similarity(entity_embedding, chunk_embeddings)
            
            for i, sim in enumerate(similarities):
                if sim >= self.semantic_threshold:
                    chunk = all_chunks[i]
                    if chunk['chunk_id'] not in direct_chunk_ids:
                        semantic_chunks.append((chunk, sim))
            
            semantic_chunks.sort(key=lambda x: x[1], reverse=True)
            semantic_chunks = [c for c, _ in semantic_chunks]
        
        candidates = direct_chunks + semantic_chunks
        return candidates[:self.candidate_pool_size]
    
    def mmr_select_chunks(
        self,
        entity_embedding: np.ndarray,
        candidate_chunks: List[Dict],
        k: int = None
    ) -> List[Dict]:
        """Select k diverse chunks using MMR."""
        if k is None:
            k = self.num_chunks
        
        if len(candidate_chunks) <= k:
            return candidate_chunks
        
        chunk_embeddings = np.array([c['embedding'] for c in candidate_chunks])
        relevance_scores = batch_cosine_similarity(entity_embedding, chunk_embeddings)
        
        selected_indices = []
        selected_embeddings = []
        remaining = list(range(len(candidate_chunks)))
        
        for _ in range(k):
            best_score = -1.0
            best_idx = None
            
            for idx in remaining:
                relevance = relevance_scores[idx]
                
                if selected_embeddings:
                    sims = [cosine_similarity(chunk_embeddings[idx], e) for e in selected_embeddings]
                    max_sim = max(sims)
                else:
                    max_sim = 0.0
                
                mmr = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_sim
                
                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                selected_embeddings.append(chunk_embeddings[best_idx])
                remaining.remove(best_idx)
        
        return [candidate_chunks[i] for i in selected_indices]
    
    def two_stage_mmr_select(
        self,
        entity: Dict,
        candidate_chunks: List[Dict],
        cooccurrence_matrix: Dict[str, List[str]]
    ) -> List[Dict]:
        """
        Two-stage chunk selection: semantic diversity + entity coverage.
        
        Stage 1: Semantic MMR (candidates → 2*num_chunks)
        Stage 2: Entity coverage maximization (→ num_chunks)
        """
        entity_embedding = np.array(entity.get('embedding', []))
        entity_id = entity.get('entity_id', '')
        
        # Stage 1: Semantic diversity
        stage1_k = min(self.num_chunks * 2, len(candidate_chunks))
        semantic_diverse = self.mmr_select_chunks(entity_embedding, candidate_chunks, k=stage1_k)
        
        if not cooccurrence_matrix:
            return semantic_diverse[:self.num_chunks]
        
        # Stage 2: Entity coverage maximization
        selected = []
        seen_entity_ids: Set[str] = {entity_id}
        
        chunk_scores = []
        for chunk in semantic_diverse:
            chunk_id = chunk.get('chunk_id', '')
            cooccurring_ids = set(cooccurrence_matrix.get(chunk_id, []))
            cooccurring_ids.discard(entity_id)
            new_entities = cooccurring_ids - seen_entity_ids
            chunk_scores.append((chunk, cooccurring_ids, len(new_entities)))
        
        for _ in range(min(self.num_chunks, len(chunk_scores))):
            if not chunk_scores:
                break
            
            best_idx = max(range(len(chunk_scores)), key=lambda i: chunk_scores[i][2])
            best_chunk, best_entities, _ = chunk_scores.pop(best_idx)
            
            selected.append(best_chunk)
            seen_entity_ids.update(best_entities)
            
            # Recalculate scores
            for i in range(len(chunk_scores)):
                chunk, cooccurring, _ = chunk_scores[i]
                new_count = len(cooccurring - seen_entity_ids)
                chunk_scores[i] = (chunk, cooccurring, new_count)
        
        return selected
    
    def _should_do_second_round(
        self,
        entity: Dict,
        selected_chunks: List[Dict]
    ) -> Tuple[bool, float]:
        """Check if second round of extraction is needed."""
        if not selected_chunks:
            return False, 0.0
        
        entity_embedding = np.array(entity.get('embedding', []))
        if len(entity_embedding) == 0:
            return False, 0.0
        
        chunk_embeddings = np.array([c['embedding'] for c in selected_chunks])
        centroid = np.mean(chunk_embeddings, axis=0)
        
        distance = 1.0 - cosine_similarity(entity_embedding, centroid)
        return distance > self.second_round_threshold, distance
    
    # ========================================================================
    # ENTITY DETECTION
    # ========================================================================
    
    def _get_detected_entities_from_chunks(
        self,
        chunks: List[Dict],
        exclude_entity_id: str,
        cooccurrence_matrix: Dict[str, List[str]]
    ) -> List[Dict]:
        """
        Get detected entities from chunks using entity_ids.
        
        v2.0: Returns full entity dicts from entity_lookup.
        """
        detected_ids: Set[str] = set()
        
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id', '')
            entity_ids = cooccurrence_matrix.get(chunk_id, [])
            detected_ids.update(entity_ids)
        
        detected_ids.discard(exclude_entity_id)
        
        # Resolve to full entity dicts
        detected_entities = []
        for eid in detected_ids:
            if eid in self.entity_lookup:
                detected_entities.append(self.entity_lookup[eid])
        
        return detected_entities
    
    # ========================================================================
    # LLM EXTRACTION
    # ========================================================================
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM and return response text."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _parse_relations_response(
        self,
        response_text: str,
        valid_entity_ids: Set[str],
        target_entity_id: str,
        chunk_ids: List[str]
    ) -> List[Dict]:
        """
        Parse LLM response and validate entity IDs.
        
        v2.0: Validates that subject_id and object_id exist.
        """
        # Clean response
        cleaned = response_text.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return []
        
        relations = data.get('relations', [])
        validated = []
        
        for rel in relations:
            subject_id = rel.get('subject_id', '')
            object_id = rel.get('object_id', '')
            predicate = rel.get('predicate', '')
            rel_chunk_ids = rel.get('chunk_ids', chunk_ids)
            
            # Validate subject is target entity
            if subject_id != target_entity_id:
                logger.debug(f"  Skipping relation: subject_id {subject_id} != target {target_entity_id}")
                continue
            
            # Validate object exists
            if object_id not in valid_entity_ids:
                logger.debug(f"  Skipping relation: object_id {object_id} not in valid set")
                continue
            
            # Normalize predicate
            predicate = normalize_predicate(predicate)
            if not predicate:
                continue
            
            validated.append({
                'subject_id': subject_id,
                'predicate': predicate,
                'object_id': object_id,
                'chunk_ids': rel_chunk_ids,
            })
        
        return validated
    
    # ========================================================================
    # MAIN EXTRACTION
    # ========================================================================
    
    def extract_relations_for_entity(
        self,
        entity: Dict,
        all_chunks: List[Dict]
    ) -> Dict:
        """
        Extract relations for a single entity.
        
        Returns:
            Dict with keys: relations, num_batches, chunks_used, strategy
        """
        entity_id = entity.get('entity_id', '')
        entity_name = entity.get('name', 'Unknown')
        entity_type = entity.get('type', 'Unknown')
        
        logger.debug(f"Processing: {entity_name} [{entity_type}] ({entity_id})")
        
        # Classify strategy
        strategy = self._classify_entity(entity)
        if strategy == 'skip':
            return {'relations': [], 'num_batches': 0, 'chunks_used': 0, 'strategy': 'skip'}
        
        # Get appropriate co-occurrence matrix
        cooccurrence = self._get_cooccurrence_matrix(strategy)
        
        # Gather and select chunks
        candidates = self.gather_candidate_chunks(entity, all_chunks)
        if not candidates:
            logger.warning(f"  No candidates for {entity_name}")
            return {'relations': [], 'num_batches': 0, 'chunks_used': 0, 'strategy': strategy}
        
        selected_chunks = self.two_stage_mmr_select(entity, candidates, cooccurrence)
        
        # Check for second round
        second_round_chunks = []
        if strategy == 'semantic':
            should_second, distance = self._should_do_second_round(entity, selected_chunks)
            if should_second:
                logger.debug(f"  Second round triggered (distance: {distance:.3f})")
                selected_ids = {c['chunk_id'] for c in selected_chunks}
                remaining = [c for c in candidates if c['chunk_id'] not in selected_ids]
                if remaining:
                    second_round_chunks = self.two_stage_mmr_select(entity, remaining, cooccurrence)
        
        # Extract relations
        all_relations = []
        
        # Batch 1
        detected_entities = self._get_detected_entities_from_chunks(
            selected_chunks, entity_id, cooccurrence
        )
        valid_ids = {e['entity_id'] for e in detected_entities}
        valid_ids.add(entity_id)
        
        chunk_ids = [c['chunk_id'] for c in selected_chunks]
        prompt = build_relation_prompt(entity, selected_chunks, detected_entities, strategy)
        
        if self.debug_mode:
            logger.debug(f"  Prompt length: {len(prompt)} chars")
        
        response = self._call_llm(prompt)
        batch1_relations = self._parse_relations_response(response, valid_ids, entity_id, chunk_ids)
        all_relations.extend(batch1_relations)
        
        # Batch 2 (if triggered)
        if second_round_chunks:
            detected_entities_2 = self._get_detected_entities_from_chunks(
                second_round_chunks, entity_id, cooccurrence
            )
            valid_ids_2 = {e['entity_id'] for e in detected_entities_2}
            valid_ids_2.add(entity_id)
            
            chunk_ids_2 = [c['chunk_id'] for c in second_round_chunks]
            prompt_2 = build_relation_prompt(entity, second_round_chunks, detected_entities_2, strategy)
            
            response_2 = self._call_llm(prompt_2)
            batch2_relations = self._parse_relations_response(response_2, valid_ids_2, entity_id, chunk_ids_2)
            all_relations.extend(batch2_relations)
        
        # Deduplicate
        seen = set()
        deduplicated = []
        for rel in all_relations:
            key = (rel['subject_id'], rel['predicate'], rel['object_id'])
            if key not in seen:
                seen.add(key)
                rel['extraction_strategy'] = strategy
                deduplicated.append(rel)
        
        num_batches = 2 if second_round_chunks else 1
        total_chunks = len(selected_chunks) + len(second_round_chunks)
        
        logger.debug(f"  Extracted {len(deduplicated)} relations ({num_batches} batches, {total_chunks} chunks)")
        
        return {
            'relations': deduplicated,
            'num_batches': num_batches,
            'chunks_used': total_chunks,
            'strategy': strategy
        }