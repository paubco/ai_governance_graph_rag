# -*- coding: utf-8 -*-
"""
Module: graph_expander.py
Package: src.retrieval
Purpose: Graph expansion via Prize-Collecting Steiner Tree (PCST) optimization

Author: Pau Barba i Colomer
Created: 2025-12-07
Modified: 2025-12-07

References:
    - He et al. (2024) "G-Retriever" - PCST for GraphRAG (NeurIPS 2024)
    - Neo4j GDS documentation: Prize-Collecting Steiner Tree
    - PHASE_3.3.2_OPEN_QUESTIONS.md (Design rationale)

Algorithm:
    Two-stage expansion to avoid hub node explosion:
    1. FAISS k-NN: Get top-K similar entities per query entity (broad net)
    2. PCST: Find minimal connecting subgraph among candidates (focused)

Key insight: PCST automatically avoids hub nodes by optimizing
prize-cost balance, finding shortest meaningful paths between query entities.
"""

import numpy as np
from typing import List, Dict, Set
from neo4j import GraphDatabase
import faiss

from .config import (
    ResolvedEntity,
    GraphSubgraph,
    Relation,
    PCST_CONFIG,
)


# ============================================================================
# GRAPH EXPANDER
# ============================================================================

class GraphExpander:
    """
    Expand from query entities using PCST optimization.
    
    Two-stage process:
    1. Get k-NN candidates via FAISS (entity similarity)
    2. Run PCST to find minimal connecting subgraph
    
    Prevents hub node explosion while discovering bridging concepts.
    """
    
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        entity_index_path: str,
        entity_id_map_path: str,
    ):
        """
        Initialize graph expander.
        
        Args:
            neo4j_uri: Neo4j connection string
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            entity_index_path: Path to FAISS entity embeddings index
            entity_id_map_path: Path to entity ID mapping JSON
        """
        # Neo4j connection
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # Load FAISS index
        self.faiss_index = faiss.read_index(entity_index_path)
        
        # Load entity ID mapping
        import json
        with open(entity_id_map_path, 'r') as f:
            entity_id_data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(entity_id_data, list):
            # List format: ["ent_001", "ent_002", ...] where index = FAISS position
            self.entity_id_map = {eid: idx for idx, eid in enumerate(entity_id_data)}
            self.index_to_entity = {idx: eid for idx, eid in enumerate(entity_id_data)}
        else:
            # Dict format: {"ent_001": 0, "ent_002": 1, ...}
            self.entity_id_map = entity_id_data
            self.index_to_entity = {v: k for k, v in entity_id_data.items()}
        
        # Config
        self.config = PCST_CONFIG
        
        # Ensure GDS projection exists
        self._ensure_gds_projection()
    
    def _ensure_gds_projection(self):
        """Create Neo4j GDS projection if not exists."""
        with self.driver.session() as session:
            # Check if projection exists
            result = session.run("""
                CALL gds.graph.exists('entity-graph')
                YIELD exists
                RETURN exists
            """)
            exists = result.single()['exists']
            
            if not exists:
                print("Creating Neo4j GDS projection 'entity-graph'...")
                session.run("""
                    CALL gds.graph.project(
                        'entity-graph',
                        'Entity',
                        {
                            RELATION: {
                                orientation: 'UNDIRECTED',
                                properties: ['confidence']
                            }
                        }
                    )
                """)
                print("✓ GDS projection created")
    
    def expand(self, resolved_entities: List[ResolvedEntity]) -> GraphSubgraph:
        """
        Main expansion method.
        
        Args:
            resolved_entities: Entities resolved from query (Phase 3.3.1)
        
        Returns:
            GraphSubgraph with entities and relations from PCST
        
        Note:
            Graph has 16K components (65% in main component). PCST may fail
            if query entities are in disconnected components. This is a known
            limitation due to missing hierarchical relations (e.g., law articles
            not linked to parent regulations). Falls back to k-NN candidates.
        """
        if not resolved_entities:
            return GraphSubgraph(entities=[], relations=[])
        
        # Stage 1: Get k-NN candidates via FAISS
        candidate_ids = self._get_faiss_candidates(resolved_entities)
        
        # If no valid candidates, return empty subgraph
        if not candidate_ids:
            print("⚠️  No valid entities found in index")
            return GraphSubgraph(entities=[], relations=[])
        
        # Stage 2: Run PCST to find minimal connecting subgraph
        if len(resolved_entities) == 1:
            # Single entity: no paths to find, just return candidates
            subgraph_entity_ids = candidate_ids[:self.config['max_entities']]
            relations = []
        else:
            # Multiple entities: find connecting paths
            subgraph_entity_ids, relations = self._run_pcst(
                terminal_ids=[e.entity_id for e in resolved_entities],
                candidate_ids=candidate_ids
            )
        
        return GraphSubgraph(
            entities=subgraph_entity_ids,
            relations=relations
        )
    
    def _get_faiss_candidates(self, resolved_entities: List[ResolvedEntity]) -> List[str]:
        """
        Stage 1: Get k-NN candidates via FAISS similarity.
        
        Returns:
            List of entity IDs (deduplicated)
        """
        candidates = set()
        
        for entity in resolved_entities:
            # Skip entities not in FAISS index
            if entity.entity_id not in self.entity_id_map:
                print(f"⚠️  Skipping entity not in index: {entity.entity_id}")
                continue
            
            # Add the entity itself
            candidates.add(entity.entity_id)
            
            # Get entity embedding
            entity_idx = self.entity_id_map[entity.entity_id]
            entity_embedding = self.faiss_index.reconstruct(entity_idx).reshape(1, -1)
            
            # FAISS search for similar entities
            k = self.config['k_candidates']
            distances, indices = self.faiss_index.search(entity_embedding, k + 1)  # +1 to exclude self
            
            # Convert indices to entity IDs
            for idx in indices[0]:
                if idx in self.index_to_entity:
                    candidate_id = self.index_to_entity[idx]
                    candidates.add(candidate_id)
        
        return list(candidates)
    
    def _run_pcst(
        self,
        terminal_ids: List[str],
        candidate_ids: List[str]
    ) -> tuple[List[str], List[Relation]]:
        """
        Stage 2: Run PCST to find minimal connecting subgraph.
        
        Args:
            terminal_ids: Query entities (must be connected)
            candidate_ids: All candidate entities from FAISS
        
        Returns:
            (subgraph_entity_ids, relations)
        """
        with self.driver.session() as session:
            # Try to run PCST
            try:
                result = session.run("""
                    // Get internal node IDs for terminals
                    MATCH (n:Entity)
                    WHERE n.entity_id IN $terminal_ids
                    WITH collect(id(n)) AS terminalNodeIds
                    
                    // Run PCST
                    CALL gds.beta.steinerTree.stream('entity-graph', {
                        sourceNode: terminalNodeIds[0],
                        targetNodes: terminalNodeIds[1..],
                        delta: $delta
                    })
                    YIELD nodeId
                    
                    // Return entities in subgraph
                    WITH collect(nodeId) AS subgraphNodeIds
                    MATCH (e:Entity)
                    WHERE id(e) IN subgraphNodeIds
                    RETURN collect(e.entity_id) AS entity_ids
                """, terminal_ids=terminal_ids, delta=self.config['delta'])
                
                record = result.single()
                if record and record['entity_ids']:
                    subgraph_entity_ids = record['entity_ids']
                else:
                    # PCST failed or found nothing - fall back to candidates
                    subgraph_entity_ids = candidate_ids[:self.config['max_entities']]
                
            except Exception as e:
                print(f"⚠️  PCST failed ({e}), falling back to k-NN candidates")
                subgraph_entity_ids = candidate_ids[:self.config['max_entities']]
            
            # Get relations for the subgraph
            relations = self._get_subgraph_relations(subgraph_entity_ids)
            
            return subgraph_entity_ids, relations
    
    def _get_subgraph_relations(self, entity_ids: List[str]) -> List[Relation]:
        """
        Extract relations between entities in subgraph.
        
        Args:
            entity_ids: Entity IDs in the subgraph
        
        Returns:
            List of Relation objects with provenance (chunk_ids)
        """
        if not entity_ids:
            return []
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e1:Entity)-[r:RELATION]->(e2:Entity)
                WHERE e1.entity_id IN $entity_ids
                  AND e2.entity_id IN $entity_ids
                RETURN 
                    e1.entity_id AS source_id,
                    e1.name AS source_name,
                    r.predicate AS predicate,
                    e2.entity_id AS target_id,
                    e2.name AS target_name,
                    r.chunk_ids AS chunk_ids
                LIMIT 100
            """, entity_ids=entity_ids)
            
            relations = []
            for record in result:
                relations.append(Relation(
                    source_id=record['source_id'],
                    source_name=record['source_name'],
                    predicate=record['predicate'],
                    target_id=record['target_id'],
                    target_name=record['target_name'],
                    chunk_ids=record['chunk_ids'] or []
                ))
            
            return relations
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()