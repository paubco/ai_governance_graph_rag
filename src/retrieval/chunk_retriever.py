# -*- coding: utf-8 -*-
"""
Module: chunk_retriever.py
Package: src.retrieval
Purpose: Dual-path chunk retrieval (Path A: GraphRAG, Path B: Naive RAG)

Author: Pau Barba i Colomer
Created: 2025-12-07
Modified: 2025-12-12

References:
    - RAKG (Zhang et al., 2025) - Corpus retrospective retrieval
    - RAGulating (Agarwal et al., 2025) - Provenance tracking
    - PHASE_3_DESIGN.md § 5.2 (Dual-channel architecture)

Retrieval Paths:
    Path A (GraphRAG): Entity expansion → Corpus retrospective
        - Get all chunks mentioning PCST entities (EXTRACTED_FROM)
        - PLUS chunks containing PCST relations (provenance)
    
    Path B (Naive RAG): Direct semantic search
        - FAISS similarity search on chunk embeddings
        - Baseline for comparison
"""

import numpy as np
from typing import List
from neo4j import GraphDatabase
import faiss
import json

from .config import (
    GraphSubgraph,
    Chunk,
    RETRIEVAL_CONFIG,
)


# ============================================================================
# CHUNK RETRIEVER
# ============================================================================

class ChunkRetriever:
    """
    Dual-path chunk retrieval.
    
    Combines:
    - Path A: Entity-centric retrieval (GraphRAG)
    - Path B: Semantic search (Naive RAG)
    """
    
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        chunk_index_path: str,
        chunk_id_map_path: str,
    ):
        """
        Initialize chunk retriever.
        
        Args:
            neo4j_uri: Neo4j connection string.
            neo4j_user: Neo4j username.
            neo4j_password: Neo4j password.
            chunk_index_path: Path to FAISS chunk embeddings index.
            chunk_id_map_path: Path to chunk ID mapping JSON.
        """
        # Neo4j connection
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # Load FAISS chunk index
        self.faiss_index = faiss.read_index(chunk_index_path)
        
        # Load chunk ID mapping
        with open(chunk_id_map_path, 'r') as f:
            chunk_id_data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(chunk_id_data, list):
            # List format: ["chunk_001", "chunk_002", ...] where index = FAISS position
            self.chunk_id_map = {cid: idx for idx, cid in enumerate(chunk_id_data)}
            self.index_to_chunk = {idx: cid for idx, cid in enumerate(chunk_id_data)}
        else:
            # Dict format: {"chunk_001": 0, "chunk_002": 1, ...}
            self.chunk_id_map = chunk_id_data
            self.index_to_chunk = {v: k for k, v in chunk_id_data.items()}
        
        # Config
        self.config = RETRIEVAL_CONFIG
    
    def retrieve_dual(
        self,
        subgraph: GraphSubgraph,
        query_embedding: np.ndarray
    ) -> tuple[List[Chunk], List[Chunk]]:
        """
        Retrieve chunks via both paths.
        
        Args:
            subgraph: PCST subgraph from graph expansion.
            query_embedding: Query embedding for Naive path.
        
        Returns:
            (graphrag_chunks, naive_chunks)
        """
        # GraphRAG: Entity-centric retrieval
        graphrag_chunks = self._retrieve_path_a(subgraph)
        
        # Naive: Semantic search
        naive_chunks = self._retrieve_path_b(query_embedding)
        
        return graphrag_chunks, naive_chunks
    
    def _retrieve_path_a(self, subgraph: GraphSubgraph) -> List[Chunk]:
        """
        GraphRAG Path: Corpus retrospective + relation provenance.
        
        Strategy:
        1. Get all chunks mentioning PCST entities (EXTRACTED_FROM edges)
        2. Prioritize chunks containing PCST relations (provenance)
        
        Args:
            subgraph: PCST subgraph with entities and relations.
        
        Returns:
            List of Chunk objects with metadata.
        """
        if not subgraph.entities:
            return []
        
        # Get relation chunk IDs (for provenance tracking)
        relation_chunk_ids = set()
        for rel in subgraph.relations:
            relation_chunk_ids.update(rel.chunk_ids)
        
        with self.driver.session() as session:
            # Corpus retrospective: Get all chunks mentioning expanded entities
            result = session.run("""
                MATCH (e:Entity)-[:EXTRACTED_FROM]->(c:Chunk)
                WHERE e.entity_id IN $entity_ids
                
                // Get document info
                OPTIONAL MATCH (doc)-[:CONTAINS]->(c)
                WHERE doc:Jurisdiction OR doc:Publication
                
                WITH DISTINCT c, doc, e
                
                RETURN 
                    c.chunk_id AS chunk_id,
                    c.text AS text,
                    CASE 
                        WHEN doc:Jurisdiction THEN doc.code
                        WHEN doc:Publication THEN doc.scopus_id
                        ELSE NULL
                    END AS doc_id,
                    CASE 
                        WHEN doc:Jurisdiction THEN 'regulation'
                        WHEN doc:Publication THEN 'paper'
                        ELSE 'unknown'
                    END AS doc_type,
                    doc.code AS jurisdiction,
                    collect(e.entity_id) AS entities
            """, entity_ids=subgraph.entities)
            
            chunks = []
            for record in result:
                # Score based on number of entities matched
                # Lower base (0.40-0.60) so coverage bonus is more meaningful
                num_entities = len(record['entities'])
                base_score = min(0.40 + (num_entities * 0.05), 0.60)
                
                chunk = Chunk(
                    chunk_id=record['chunk_id'],
                    text=record['text'],
                    doc_id=record['doc_id'],
                    doc_type=record['doc_type'],
                    jurisdiction=record['jurisdiction'],
                    score=base_score,
                    metadata={
                        'entities': record['entities'],
                        'is_relation_provenance': record['chunk_id'] in relation_chunk_ids
                    }
                )
                chunks.append(chunk)
            
            return chunks
    
    def _retrieve_path_b(self, query_embedding: np.ndarray) -> List[Chunk]:
        """
        Naive RAG Path: Direct semantic search via FAISS.
        
        Args:
            query_embedding: Query embedding (BGE-M3, 1024-dim).
        
        Returns:
            Top-K most similar chunks.
        """
        k = self.config['path_b_top_k']
        
        # FAISS search
        query_vec = query_embedding.reshape(1, -1)
        distances, indices = self.faiss_index.search(query_vec, k)
        
        # Convert distances to similarity scores
        # For inner product: distance IS similarity
        # For L2: similarity = 1 / (1 + distance)
        # Normalize to 0-1 range
        similarities = distances[0]
        if similarities.max() > 1.0:  # Likely L2 distance
            similarities = 1.0 / (1.0 + similarities)
        
        # Convert to chunk IDs with scores
        chunk_ids = []
        chunk_scores = {}
        for idx, similarity in zip(indices[0], similarities):
            if idx in self.index_to_chunk:
                chunk_id = self.index_to_chunk[idx]
                chunk_ids.append(chunk_id)
                chunk_scores[chunk_id] = float(similarity)
        
        # Fetch chunk details from Neo4j
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Chunk)
                WHERE c.chunk_id IN $chunk_ids
                
                // Get document info
                OPTIONAL MATCH (doc)-[:CONTAINS]->(c)
                WHERE doc:Jurisdiction OR doc:Publication
                
                RETURN 
                    c.chunk_id AS chunk_id,
                    c.text AS text,
                    CASE 
                        WHEN doc:Jurisdiction THEN doc.code
                        WHEN doc:Publication THEN doc.scopus_id
                        ELSE NULL
                    END AS doc_id,
                    CASE 
                        WHEN doc:Jurisdiction THEN 'regulation'
                        WHEN doc:Publication THEN 'paper'
                        ELSE 'unknown'
                    END AS doc_type,
                    doc.code AS jurisdiction
            """, chunk_ids=chunk_ids)
            
            # Preserve FAISS order (important for scoring)
            chunk_dict = {}
            for record in result:
                chunk_id = record['chunk_id']
                chunk_dict[chunk_id] = Chunk(
                    chunk_id=chunk_id,
                    text=record['text'],
                    doc_id=record['doc_id'],
                    doc_type=record['doc_type'],
                    jurisdiction=record['jurisdiction'],
                    score=chunk_scores.get(chunk_id, 0.5),  # FAISS similarity
                    metadata={'faiss_rank': chunk_ids.index(chunk_id)}
                )
            
            # Return in FAISS order
            chunks = [chunk_dict[cid] for cid in chunk_ids if cid in chunk_dict]
            return chunks
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()