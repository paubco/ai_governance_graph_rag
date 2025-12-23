# -*- coding: utf-8 -*-
"""
Graph expander for AI governance GraphRAG pipeline.

Expands from query entities using Prize-Collecting Steiner Tree (PCST)
optimization to find minimal connecting subgraph.
"""

# Standard library
from typing import List, Dict, Set
import json

# Third-party
import numpy as np
from neo4j import GraphDatabase
import faiss

# Config imports (direct)
from config.retrieval_config import PCST_CONFIG

# Dataclass imports (direct)
from src.utils.dataclasses import ResolvedEntity, Subgraph, Relation


class GraphExpander:
    """
    Expand from query entities using PCST optimization.
    
    Two-stage process:
    1. Get k-NN candidates via FAISS (entity similarity)
    2. Run PCST to find minimal connecting subgraph
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
        Main expansion method.
        
        Args:
            resolved_entities: Entities resolved from query (Phase 3.3.1)
        
        Returns:
            Subgraph with entities and relations from PCST
        """
        if not resolved_entities:
            return Subgraph(entity_ids=[], relations=[])
        
        candidate_ids = self._get_faiss_candidates(resolved_entities)
        
        if not candidate_ids:
            print("Warning: No valid entities found in index")
            return Subgraph(entity_ids=[], relations=[])
        
        if len(resolved_entities) == 1:
            subgraph_entity_ids = candidate_ids[:self.config['max_entities']]
            relations = []
        else:
            subgraph_entity_ids, relations = self._run_pcst(
                terminal_ids=[e.entity_id for e in resolved_entities],
                candidate_ids=candidate_ids
            )
        
        return Subgraph(
            entity_ids=subgraph_entity_ids,
            relations=relations
        )
    
    def _get_faiss_candidates(self, resolved_entities: List[ResolvedEntity]) -> List[str]:
        """Get k-NN candidates via FAISS similarity."""
        candidates = set()
        
        for entity in resolved_entities:
            if entity.entity_id not in self.entity_id_map:
                print(f"Warning: Skipping entity not in index: {entity.entity_id}")
                continue
            
            candidates.add(entity.entity_id)
            
            entity_idx = self.entity_id_map[entity.entity_id]
            entity_embedding = self.faiss_index.reconstruct(entity_idx).reshape(1, -1)
            
            k = self.config['k_candidates']
            distances, indices = self.faiss_index.search(entity_embedding, k + 1)
            
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
        """Run PCST to find minimal connecting subgraph."""
        with self.driver.session() as session:
            try:
                result = session.run("""
                    MATCH (n:Entity)
                    WHERE n.entity_id IN $terminal_ids
                    WITH collect(id(n)) AS terminalNodeIds
                    
                    CALL gds.beta.steinerTree.stream('entity-graph', {
                        sourceNode: terminalNodeIds[0],
                        targetNodes: terminalNodeIds[1..],
                        delta: $delta
                    })
                    YIELD nodeId
                    
                    WITH collect(nodeId) AS subgraphNodeIds
                    MATCH (e:Entity)
                    WHERE id(e) IN subgraphNodeIds
                    RETURN collect(e.entity_id) AS entity_ids
                """, terminal_ids=terminal_ids, delta=self.config['delta'])
                
                record = result.single()
                if record and record['entity_ids']:
                    subgraph_entity_ids = record['entity_ids']
                else:
                    subgraph_entity_ids = candidate_ids[:self.config['max_entities']]
                
            except Exception as e:
                print(f"Warning: PCST failed ({e}), falling back to k-NN candidates")
                subgraph_entity_ids = candidate_ids[:self.config['max_entities']]
            
            relations = self._get_subgraph_relations(subgraph_entity_ids)
            
            return subgraph_entity_ids, relations
    
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