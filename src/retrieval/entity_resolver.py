# -*- coding: utf-8 -*-
"""
Module: entity_resolver.py
Package: src.retrieval
Purpose: Resolve entity mentions to canonical entity IDs (Phase 3.3.1)

Author: Pau Barba i Colomer
Created: 2025-12-07
Modified: 2025-12-07

References:
    - PHASE_3_DESIGN.md § 4.3 (Entity resolution approach)
    - RAKG methodology for entity matching
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

import faiss
import numpy as np

from .config import (
    ResolvedEntity,
    ExtractedEntity,
    QueryFilters,
    ENTITY_RESOLUTION_CONFIG,
)


class EntityResolver:
    """
    Resolve entity mentions to canonical entity IDs.
    
    Strategy:
    1. Exact match: O(1) lookup via entity name dictionary
    2. Fuzzy match: FAISS similarity search with threshold
    """
    
    def __init__(
        self,
        faiss_index_path: Path,
        entity_ids_path: Path,
        normalized_entities_path: Path,
        embedding_model,
        threshold: float = 0.75,  # From ENTITY_RESOLUTION_CONFIG
        top_k: int = 3,           # Limit fuzzy matches to avoid explosion
    ):
        """
        Initialize entity resolver.
        
        Args:
            faiss_index_path: Path to FAISS entity index.
            entity_ids_path: Path to entity IDs JSON mapping.
            normalized_entities_path: Path to normalized entities JSON.
            embedding_model: BGE-M3 embedding model with embed_single() method.
            threshold: Fuzzy match threshold (default from config).
            top_k: Number of candidates to retrieve (default from config).
        """
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.top_k = top_k
        
        # Load FAISS index
        self.faiss_index = faiss.read_index(str(faiss_index_path))
        
        # Load entity IDs mapping
        # Format: list where index = FAISS position, value = entity_id
        with open(entity_ids_path, 'r', encoding='utf-8') as f:
            entity_id_list = json.load(f)
        
        # Create reverse mapping: FAISS index → entity_id
        if isinstance(entity_id_list, list):
            # List format: ["ent_001", "ent_002", ...]
            self.entity_ids = {idx: eid for idx, eid in enumerate(entity_id_list)}
        else:
            # Dict format: {"ent_001": 0, "ent_002": 1, ...}
            self.entity_ids = {v: k for k, v in entity_id_list.items()}
        
        # Load normalized entities for metadata
        entities_list = []
        try:
            with open(normalized_entities_path, 'r', encoding='utf-8') as f:
                entities_list = json.load(f)
            # Build lookup: entity_id → entity metadata
            self.entities_by_id = {e['entity_id']: e for e in entities_list}
        except MemoryError:
            print("⚠️  Entity file too large for memory, using minimal mode")
            # Fallback: build minimal entities from IDs only
            self.entities_by_id = {}
            for entity_id in (self.entity_ids.values() if isinstance(self.entity_ids, dict) 
                             else [self.entity_ids[i] for i in range(len(self.entity_ids))]):
                self.entities_by_id[entity_id] = {
                    'entity_id': entity_id,
                    'name': entity_id,  # Use ID as name
                    'type': 'Unknown'   # Type unknown in minimal mode
                }
                entities_list.append(self.entities_by_id[entity_id])
        except Exception as e:
            print(f"⚠️  Error loading entities: {e}")
            self.entities_by_id = {}
        
        # Build exact match lookup: lowercase name → entity_id
        self.name_to_id = self._build_name_lookup(entities_list)
    
    def resolve(
        self, 
        entities: List[ExtractedEntity], 
        filters: QueryFilters = None,
        top_k: int = None
    ) -> List[ResolvedEntity]:
        """
        Resolve extracted entities to canonical entity IDs.
        
        Tries exact match first, then falls back to fuzzy matching.
        Returns top_k matches per entity to limit subgraph explosion.
        
        Args:
            entities: List of ExtractedEntity objects from query parsing.
            filters: Optional QueryFilters for future use (not currently applied).
            top_k: Override default top_k for this query (default: use constructor value).
            
        Returns:
            List of ResolvedEntity objects (may be empty).
        """
        # Use override or default
        k = top_k if top_k is not None else self.top_k
        
        resolved = []
        
        for entity in entities:
            # Try exact match first
            exact_match = self._exact_match(entity.name)
            if exact_match:
                resolved.append(exact_match)
                continue
            
            # Fall back to fuzzy match (limited to top_k)
            fuzzy_matches = self._fuzzy_match(entity.name, entity.type, top_k=k)
            resolved.extend(fuzzy_matches)
        
        print(f"  Resolved {len(entities)} mentions → {len(resolved)} entities (top_k={k})")
        return resolved
    
    def _exact_match(self, mention: str) -> Optional[ResolvedEntity]:
        """
        Attempt exact match via name lookup.
        
        Args:
            mention: Entity mention string.
            
        Returns:
            ResolvedEntity if exact match found, else None.
        """
        mention_lower = mention.lower().strip()
        
        if mention_lower in self.name_to_id:
            entity_id = self.name_to_id[mention_lower]
            entity = self.entities_by_id[entity_id]
            
            return ResolvedEntity(
                entity_id=entity_id,
                name=entity['name'],
                type=entity['type'],
                confidence=1.0,
                match_type='exact',
            )
        
        return None
    
    def _fuzzy_match(self, mention: str, entity_type: str = None, top_k: int = None) -> List[ResolvedEntity]:
        """
        Attempt fuzzy match via FAISS similarity search.
        
        Args:
            mention: Entity mention string.
            entity_type: Optional entity type for filtering (future use).
            top_k: Number of candidates to retrieve (uses self.top_k if None).
            
        Returns:
            List of ResolvedEntity objects above threshold.
        """
        # Use provided top_k or default
        k = top_k if top_k is not None else self.top_k
        
        # Embed the mention
        mention_embedding = self.embedding_model.embed_single(mention)  # FIXED: embed_single not embed
        
        # Ensure correct shape for FAISS (1, 1024)
        if mention_embedding.ndim == 1:
            mention_embedding = mention_embedding.reshape(1, -1)
        
        # FAISS search
        distances, indices = self.faiss_index.search(
            mention_embedding.astype('float32'),
            k
        )
        
        # Convert distances to cosine similarities
        # FAISS inner product returns negative of cosine distance
        # For normalized vectors: similarity = 1 - distance/2
        similarities = 1 - (distances[0] / 2)
        
        resolved = []
        
        for idx, similarity in zip(indices[0], similarities):
            # Skip if below threshold
            if similarity < self.threshold:
                continue
            
            # Skip invalid indices
            if idx < 0 or idx >= len(self.entity_ids):
                continue
            
            entity_id = self.entity_ids[idx]
            entity = self.entities_by_id.get(entity_id)
            
            if entity:
                resolved.append(ResolvedEntity(
                    entity_id=entity_id,
                    name=entity['name'],
                    type=entity['type'],
                    confidence=float(similarity),
                    match_type='fuzzy',
                ))
        
        return resolved
    
    def _build_name_lookup(self, entities: List[Dict]) -> Dict[str, str]:
        """
        Build lowercase name → entity_id lookup for exact matching.
        
        Args:
            entities: List of entity dictionaries.
            
        Returns:
            Dictionary mapping lowercase names to entity IDs.
        """
        name_to_id = {}
        
        for entity in entities:
            name_lower = entity['name'].lower().strip()
            entity_id = entity['entity_id']
            
            # Use first occurrence (most frequent/canonical)
            if name_lower not in name_to_id:
                name_to_id[name_lower] = entity_id
        
        return name_to_id