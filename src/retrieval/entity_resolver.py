# -*- coding: utf-8 -*-
"""
Entity resolver for AI governance GraphRAG pipeline.

Resolves entity mentions from queries to canonical entity IDs in the knowledge
graph. Uses three-stage matching:
    1. Exact match on entity name
    2. Alias match using aliases.json lookup
    3. FAISS semantic similarity for fuzzy matching

v1.1 Changes:
    - Added alias support (1,310 clusters, 2,594 aliases)
    - Alias lookup: alias → canonical_name → entity_id
"""

# Standard library
import json
from pathlib import Path
from typing import List, Dict, Optional

# Third-party
import faiss
import numpy as np

# Config imports (direct)
from config.retrieval_config import ENTITY_RESOLUTION_CONFIG

# Dataclass imports (direct)
from src.utils.dataclasses import (
    ResolvedEntity,
    ExtractedQueryEntity as ExtractedEntity,
    QueryFilters,
)


class EntityResolver:
    """
    Resolve entity mentions to canonical entity IDs.
    
    Strategy:
    1. Exact match: O(1) lookup via entity name dictionary
    2. Alias match: O(1) lookup via alias → canonical_name → entity_id
    3. Fuzzy match: FAISS similarity search with threshold
    """
    
    def __init__(
        self,
        faiss_index_path: Path,
        entity_ids_path: Path,
        normalized_entities_path: Path,
        embedding_model,
        aliases_path: Path = None,
        threshold: float = None,
        top_k: int = 3,
    ):
        """
        Initialize entity resolver.
        
        Args:
            faiss_index_path: Path to FAISS entity index.
            entity_ids_path: Path to entity IDs JSON mapping.
            normalized_entities_path: Path to normalized entities JSON.
            embedding_model: BGE-M3 embedding model with embed_single() method.
            aliases_path: Path to aliases.json (optional, enables alias matching).
            threshold: Fuzzy match threshold (default from config).
            top_k: Number of candidates to retrieve for fuzzy matching.
        """
        self.embedding_model = embedding_model
        self.threshold = threshold or ENTITY_RESOLUTION_CONFIG['fuzzy_threshold']
        self.top_k = top_k
        
        # Load FAISS index
        self.faiss_index = faiss.read_index(str(faiss_index_path))
        
        # Load entity IDs mapping (FAISS index → entity_id)
        with open(entity_ids_path, 'r', encoding='utf-8') as f:
            entity_id_list = json.load(f)
        
        if isinstance(entity_id_list, list):
            self.entity_ids = {idx: eid for idx, eid in enumerate(entity_id_list)}
        else:
            self.entity_ids = {v: k for k, v in entity_id_list.items()}
        
        # Load normalized entities for metadata
        self.entity_ids_by_id = {}
        self.name_to_id = {}
        
        try:
            with open(normalized_entities_path, 'r', encoding='utf-8') as f:
                entities_list = json.load(f)
            
            self.entity_ids_by_id = {e['entity_id']: e for e in entities_list}
            self.name_to_id = self._build_name_lookup(entities_list)
            
        except Exception as e:
            print(f"Warning: Error loading entities: {e}")
        
        # Load aliases (v1.1)
        self.alias_to_canonical = {}
        if aliases_path and Path(aliases_path).exists():
            self.alias_to_canonical = self._build_alias_lookup(aliases_path)
            print(f"Loaded {len(self.alias_to_canonical)} alias mappings")
    
    def resolve(
        self, 
        entities: List[ExtractedEntity], 
        filters: QueryFilters = None,
        top_k: int = None
    ) -> List[ResolvedEntity]:
        """
        Resolve extracted entities to canonical entity IDs.
        
        Tries exact match first, then alias match, then fuzzy matching.
        
        Args:
            entities: List of ExtractedEntity objects from query parsing.
            filters: Optional QueryFilters (reserved for future use).
            top_k: Override default top_k for this query.
            
        Returns:
            List of ResolvedEntity objects (may be empty).
        """
        k = top_k if top_k is not None else self.top_k
        resolved = []
        
        for entity in entities:
            # 1. Try exact name match
            exact_match = self._exact_match(entity.name)
            if exact_match:
                exact_match.query_mention = entity.name
                resolved.append(exact_match)
                continue
            
            # 2. Try alias match (v1.1)
            alias_match = self._alias_match(entity.name)
            if alias_match:
                alias_match.query_mention = entity.name
                resolved.append(alias_match)
                continue
            
            # 3. Fall back to fuzzy match
            fuzzy_matches = self._fuzzy_match(entity.name, entity.type, top_k=k)
            for match in fuzzy_matches:
                match.query_mention = entity.name
            resolved.extend(fuzzy_matches)
        
        print(f"  Resolved {len(entities)} mentions → {len(resolved)} entities")
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
            entity = self.entity_ids_by_id.get(entity_id)
            
            if entity:
                return ResolvedEntity(
                    entity_id=entity_id,
                    name=entity['name'],
                    type=entity['type'],
                    confidence=1.0,
                    match_type='exact',
                )
        
        return None
    
    def _alias_match(self, mention: str) -> Optional[ResolvedEntity]:
        """
        Attempt alias match via alias → canonical_name → entity_id.
        
        Args:
            mention: Entity mention string.
            
        Returns:
            ResolvedEntity if alias match found, else None.
        """
        mention_lower = mention.lower().strip()
        
        # Check if mention is an alias
        if mention_lower in self.alias_to_canonical:
            canonical_name = self.alias_to_canonical[mention_lower]
            canonical_lower = canonical_name.lower().strip()
            
            # Now find entity_id for canonical name
            if canonical_lower in self.name_to_id:
                entity_id = self.name_to_id[canonical_lower]
                entity = self.entity_ids_by_id.get(entity_id)
                
                if entity:
                    return ResolvedEntity(
                        entity_id=entity_id,
                        name=entity['name'],
                        type=entity['type'],
                        confidence=0.95,  # Slightly lower than exact match
                        match_type='alias',
                    )
        
        return None
    
    def _fuzzy_match(
        self, 
        mention: str, 
        entity_type: str = None, 
        top_k: int = None
    ) -> List[ResolvedEntity]:
        """
        Attempt fuzzy match via FAISS similarity search.
        
        Args:
            mention: Entity mention string.
            entity_type: Optional entity type for filtering (future use).
            top_k: Number of candidates to retrieve.
            
        Returns:
            List of ResolvedEntity objects above threshold.
        """
        k = top_k if top_k is not None else self.top_k
        
        # Embed the mention
        mention_embedding = self.embedding_model.embed_single(mention)
        
        if mention_embedding.ndim == 1:
            mention_embedding = mention_embedding.reshape(1, -1)
        
        # FAISS search
        distances, indices = self.faiss_index.search(
            mention_embedding.astype('float32'),
            k
        )
        
        # Convert distances to cosine similarities
        # For normalized vectors: similarity = 1 - distance/2
        similarities = 1 - (distances[0] / 2)
        
        resolved = []
        
        for idx, similarity in zip(indices[0], similarities):
            if similarity < self.threshold:
                continue
            
            if idx < 0 or idx not in self.entity_ids:
                continue
            
            entity_id = self.entity_ids[idx]
            entity = self.entity_ids_by_id.get(entity_id)
            
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
        """Build lowercase name → entity_id lookup for exact matching."""
        name_to_id = {}
        
        for entity in entities:
            name_lower = entity['name'].lower().strip()
            entity_id = entity['entity_id']
            
            if name_lower not in name_to_id:
                name_to_id[name_lower] = entity_id
        
        return name_to_id
    
    def _build_alias_lookup(self, aliases_path: Path) -> Dict[str, str]:
        """
        Build alias → canonical_name lookup from aliases.json.
        
        aliases.json format: {canonical_name: [alias1, alias2, ...]}
        
        Returns:
            Dict mapping lowercase alias → canonical_name
        """
        alias_to_canonical = {}
        
        try:
            with open(aliases_path, 'r', encoding='utf-8') as f:
                aliases_data = json.load(f)
            
            for canonical_name, aliases in aliases_data.items():
                # Also map canonical name to itself (for case-insensitive matching)
                canonical_lower = canonical_name.lower().strip()
                alias_to_canonical[canonical_lower] = canonical_name
                
                # Map each alias to the canonical name
                for alias in aliases:
                    alias_lower = alias.lower().strip()
                    if alias_lower not in alias_to_canonical:
                        alias_to_canonical[alias_lower] = canonical_name
            
        except Exception as e:
            print(f"Warning: Error loading aliases: {e}")
        
        return alias_to_canonical