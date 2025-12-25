# -*- coding: utf-8 -*-
"""
Graph expander for AI governance GraphRAG pipeline.

Layered expansion strategy:
- Single entity: k-NN expansion via FAISS (bounded by k_candidates)
- Multi-entity: Steiner Tree (connect) + k-NN expansion (context)

Note: PCST expansion requires GDS 2.5+ (current: 2.4.6). Left as future work.
"""

# Standard library
from typing import List, Dict, Set, Tuple
import json
import logging

# Third-party
import numpy as np
from neo4j import GraphDatabase
import faiss

# Config imports (direct)
from config.retrieval_config import PCST_CONFIG

# Dataclass imports (direct)
from src.utils.dataclasses import ResolvedEntity, Subgraph, Relation

logger = logging.getLogger(__name__)


class ExpansionMetrics:
    """Diagnostics for graph expansion."""
    def __init__(self):
        self.terminals_requested = 0
        self.terminals_connected = 0
        self.steiner_nodes_added = 0
        self.expansion_nodes_added = 0
        self.total_subgraph_nodes = 0
        self.total_relations = 0
        self.disconnected_terminals = []
        self.algorithm_used = ""
    
    def __str__(self):
        return (
            f"Expansion: {self.algorithm_used} | "
            f"Terminals: {self.terminals_connected}/{self.terminals_requested} | "
            f"Steiner: +{self.steiner_nodes_added} | "
            f"Expansion: +{self.expansion_nodes_added} | "
            f"Total: {self.total_subgraph_nodes} nodes, {self.total_relations} rels"
        )


class GraphExpander:
    """
    Expand from query entities using layered strategy.
    
    Single entity: k-NN via FAISS to find similar entities
    Multi-entity: Steiner Tree (connect) + k-NN (expand context)
    
    Future: PCST for single-entity expansion (requires GDS 2.5+)
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
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        self.faiss_index = faiss.read_index(entity_index_path)
        
        with open(entity_id_map_path, 'r') as f:
            entity_id_data = json.load(f)
        
        if isinstance(entity_id_data, list):
            self.entity_id_map = {eid: idx for idx, eid in enumerate(entity_id_data)}
            self.index_to_entity = {idx: eid for idx, eid in enumerate(entity_id_data)}
        else:
            self.entity_id_map = entity_id_data
            self.index_to_entity = {v: k for k, v in entity_id_data.items()}
        
        self.config = PCST_CONFIG
        self.last_metrics = None
        self._ensure_gds_projection()
    
    def _ensure_gds_projection(self):
        """Create Neo4j GDS projection if not exists."""
        with self.driver.session() as session:
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
                print("GDS projection created")
    
    def expand(self, resolved_entities: List[ResolvedEntity]) -> Subgraph:
        """
        Main expansion method with layered strategy.
        
        Args:
            resolved_entities: Entities resolved from query
        
        Returns:
            Subgraph with entities and relations
        """
        self.last_metrics = ExpansionMetrics()
        
        if not resolved_entities:
            return Subgraph(entity_ids=[], relations=[])
        
        self.last_metrics.terminals_requested = len(resolved_entities)
        terminal_ids = [e.entity_id for e in resolved_entities]
        
        if len(resolved_entities) == 1:
            # Single entity: PCST expansion
            subgraph_entity_ids, relations = self._expand_single(
                resolved_entities[0]
            )
            self.last_metrics.algorithm_used = "kNN"
        else:
            # Multi-entity: Steiner Tree + k-NN expansion
            subgraph_entity_ids, relations = self._expand_multi(
                resolved_entities
            )
            self.last_metrics.algorithm_used = "SteinerTree+kNN"
        
        self.last_metrics.total_subgraph_nodes = len(subgraph_entity_ids)
        self.last_metrics.total_relations = len(relations)
        
        # Log warning for disconnected terminals only
        if self.last_metrics.disconnected_terminals:
            print(f"   WARNING: Disconnected terminals: {self.last_metrics.disconnected_terminals}")
        
        return Subgraph(
            entity_ids=subgraph_entity_ids,
            relations=relations
        )
    
    def _expand_single(self, entity: ResolvedEntity) -> Tuple[List[str], List[Relation]]:
        """
        Single entity expansion using k-NN.
        
        Note: PCST expansion requires GDS 2.5+ (we have 2.4.6).
        Using k-NN as bounded alternative - finds similar entities via FAISS.
        """
        entity_id = entity.entity_id
        
        if entity_id not in self.entity_id_map:
            print(f"Warning: Entity not in index: {entity_id}")
            return [entity_id], []
        
        # Get k-NN candidates (bounded by k_candidates config)
        candidates = self._get_faiss_candidates([entity])
        subgraph_ids = candidates[:self.config['max_entities']]
        
        self.last_metrics.terminals_connected = 1
        self.last_metrics.expansion_nodes_added = len(subgraph_ids) - 1
        
        # Get relations between subgraph nodes
        relations = self._get_subgraph_relations(subgraph_ids)
        
        return subgraph_ids, relations
    
    def _expand_multi(self, resolved_entities: List[ResolvedEntity]) -> Tuple[List[str], List[Relation]]:
        """
        Multi-entity expansion: Steiner Tree (connect) + k-NN (expand).
        
        Layer 1: Find minimum tree connecting all terminals
        Layer 2: Expand from spine nodes to add context
        """
        terminal_ids = [e.entity_id for e in resolved_entities]
        
        # Layer 1: Steiner Tree to connect terminals
        spine_ids = self._run_steiner_tree(terminal_ids)
        
        # Track connectivity metrics
        connected_terminals = set(terminal_ids) & set(spine_ids)
        self.last_metrics.terminals_connected = len(connected_terminals)
        self.last_metrics.steiner_nodes_added = len(set(spine_ids) - set(terminal_ids))
        self.last_metrics.disconnected_terminals = list(set(terminal_ids) - connected_terminals)
        
        if not spine_ids:
            # Steiner Tree failed, fall back to k-NN for each entity
            print("   Steiner Tree returned empty, using k-NN fallback")
            all_candidates = set()
            for entity in resolved_entities:
                candidates = self._get_faiss_candidates([entity])
                all_candidates.update(candidates[:self.config['k_expansion']])
            spine_ids = list(all_candidates)
        
        # Layer 2: k-NN expansion from spine nodes
        expanded_ids = self._expand_from_spine(spine_ids, terminal_ids)
        self.last_metrics.expansion_nodes_added = len(expanded_ids) - len(spine_ids)
        
        # Get relations between all subgraph nodes
        relations = self._get_subgraph_relations(expanded_ids)
        
        return expanded_ids, relations
    
    def _run_steiner_tree(self, terminal_ids: List[str]) -> List[str]:
        """Run Steiner Tree to connect terminal nodes."""
        if len(terminal_ids) < 2:
            return terminal_ids
            
        with self.driver.session() as session:
            try:
                # First, get Neo4j internal node IDs for terminals
                node_result = session.run("""
                    MATCH (n:Entity)
                    WHERE n.entity_id IN $terminal_ids
                    RETURN id(n) AS nodeId, n.entity_id AS entityId
                """, terminal_ids=terminal_ids)
                
                node_records = list(node_result)
                if len(node_records) < 2:
                    logger.warning(f"Only {len(node_records)} terminals found in graph")
                    return terminal_ids
                
                neo4j_node_ids = [r['nodeId'] for r in node_records]
                source_node = neo4j_node_ids[0]
                target_nodes = neo4j_node_ids[1:]
                
                # Run Steiner Tree with explicit node IDs
                result = session.run("""
                    CALL gds.beta.steinerTree.stream('entity-graph', {
                        sourceNode: $sourceNode,
                        targetNodes: $targetNodes,
                        delta: $delta
                    })
                    YIELD nodeId
                    
                    WITH collect(nodeId) AS subgraphNodeIds
                    MATCH (e:Entity)
                    WHERE id(e) IN subgraphNodeIds
                    RETURN collect(e.entity_id) AS entity_ids
                """, sourceNode=source_node, targetNodes=target_nodes, delta=self.config['delta'])
                
                record = result.single()
                if record and record['entity_ids']:
                    return record['entity_ids']
                else:
                    return terminal_ids  # Fallback to just terminals
                    
            except Exception as e:
                logger.warning(f"Steiner Tree failed: {e}")
                print(f"   Steiner Tree failed ({e}), returning terminals only")
                return terminal_ids
    
    def _expand_from_spine(self, spine_ids: List[str], terminal_ids: List[str]) -> List[str]:
        """Expand from spine nodes using k-NN to add context."""
        expanded = set(spine_ids)
        
        k_expansion = self.config.get('k_expansion', 3)
        max_entities = self.config['max_entities']
        
        for entity_id in spine_ids:
            if len(expanded) >= max_entities:
                break
                
            if entity_id not in self.entity_id_map:
                continue
            
            entity_idx = self.entity_id_map[entity_id]
            entity_embedding = self.faiss_index.reconstruct(entity_idx).reshape(1, -1)
            
            distances, indices = self.faiss_index.search(entity_embedding, k_expansion + 1)
            
            for idx in indices[0]:
                if len(expanded) >= max_entities:
                    break
                if idx in self.index_to_entity:
                    expanded.add(self.index_to_entity[idx])
        
        return list(expanded)
    
    def _get_faiss_candidates(self, resolved_entities: List[ResolvedEntity]) -> List[str]:
        """Get k-NN candidates via FAISS similarity."""
        candidates = set()
        
        for entity in resolved_entities:
            if entity.entity_id not in self.entity_id_map:
                continue
            
            candidates.add(entity.entity_id)
            
            entity_idx = self.entity_id_map[entity.entity_id]
            entity_embedding = self.faiss_index.reconstruct(entity_idx).reshape(1, -1)
            
            k = self.config['k_candidates']
            distances, indices = self.faiss_index.search(entity_embedding, k + 1)
            
            for idx in indices[0]:
                if idx in self.index_to_entity:
                    candidates.add(self.index_to_entity[idx])
        
        return list(candidates)
    
    def _get_subgraph_relations(self, entity_ids: List[str]) -> List[Relation]:
        """Extract relations between entities in subgraph."""
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
                    subject_id=record['source_id'],
                    predicate=record['predicate'],
                    object_id=record['target_id'],
                    chunk_ids=record['chunk_ids'] or [],
                    extraction_strategy='semantic'
                ))
            
            return relations
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()