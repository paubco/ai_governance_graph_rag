# -*- coding: utf-8 -*-
"""
Dual-channel chunk retrieval for AI governance GraphRAG pipeline.

Implements combined graph-based and semantic retrieval for context gathering.
"""

# Standard library
from typing import List
import json

# Third-party
import numpy as np
from neo4j import GraphDatabase
import faiss

# Config imports (direct)
from config.retrieval_config import RETRIEVAL_CONFIG

# Dataclass imports (direct)
from src.utils.dataclasses import Subgraph, Chunk


class ChunkRetriever:
    """
    Dual-channel chunk retrieval.
    
    Combines:
    - Graph Retrieval: Entity-centric via PCST expansion
    - Semantic Retrieval: Vector similarity via FAISS
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
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        self.faiss_index = faiss.read_index(chunk_index_path)
        
        with open(chunk_id_map_path, 'r') as f:
            chunk_id_data = json.load(f)
        
        if isinstance(chunk_id_data, list):
            self.chunk_id_map = {cid: idx for idx, cid in enumerate(chunk_id_data)}
            self.index_to_chunk = {idx: cid for idx, cid in enumerate(chunk_id_data)}
        else:
            self.chunk_id_map = chunk_id_data
            self.index_to_chunk = {v: k for k, v in chunk_id_data.items()}
        
        self.config = RETRIEVAL_CONFIG
    
    def retrieve_dual(
        self,
        subgraph: Subgraph,
        query_embedding: np.ndarray
    ) -> tuple[List[Chunk], List[Chunk]]:
        """
        Retrieve chunks via both channels.
        
        Args:
            subgraph: PCST subgraph from graph expansion.
            query_embedding: Query embedding for semantic retrieval.
        
        Returns:
            (graph_chunks, semantic_chunks)
        """
        graph_chunks = self._retrieve_graph(subgraph)
        semantic_chunks = self._retrieve_semantic(query_embedding)
        
        return graph_chunks, semantic_chunks
    
    def _retrieve_graph(self, subgraph: Subgraph) -> List[Chunk]:
        """
        Graph Retrieval: Corpus retrospective + relation provenance.
        
        Uses EXTRACTED_FROM edges (35,650 total) as primary retrieval path.
        """
        if not subgraph.entities:
            return []
        
        # Get relation chunk IDs for provenance tracking
        relation_chunk_ids = set()
        for rel in subgraph.relations:
            relation_chunk_ids.update(rel.chunk_ids)
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)-[:EXTRACTED_FROM]->(c:Chunk)
                WHERE e.entity_id IN $entity_ids
                
                OPTIONAL MATCH (doc)-[:CONTAINS]->(c)
                WHERE doc:Jurisdiction OR doc:Publication
                
                WITH DISTINCT c, doc, collect(e.entity_id) AS entities
                
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
                    entities
            """, entity_ids=list(subgraph.entities))
            
            chunks = []
            for record in result:
                num_entities = len(record['entities'])
                base_score = min(0.40 + (num_entities * 0.05), 0.60)
                
                chunk = Chunk(
                    chunk_ids=[record['chunk_id']],
                    document_ids=[record['doc_id']] if record['doc_id'] else [],
                    text=record['text'],
                    position=0,
                    sentence_count=0,
                    token_count=0,
                    metadata={
                        'entities': record['entities'],
                        'is_relation_provenance': record['chunk_id'] in relation_chunk_ids,
                        'score': base_score,
                        'doc_type': record['doc_type'],
                        'jurisdiction': record['jurisdiction'],
                    }
                )
                chunks.append(chunk)
            
            return chunks
    
    def _retrieve_semantic(self, query_embedding: np.ndarray) -> List[Chunk]:
        """
        Semantic Retrieval: Direct vector similarity search via FAISS.
        """
        k = self.config['semantic_top_k']
        
        query_vec = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.faiss_index.search(query_vec, k)
        
        # Convert distances to similarity scores
        similarities = distances[0]
        if similarities.max() > 1.0:  # Likely L2 distance
            similarities = 1.0 / (1.0 + similarities)
        
        chunk_ids = []
        chunk_scores = {}
        for idx, similarity in zip(indices[0], similarities):
            if idx in self.index_to_chunk:
                chunk_id = self.index_to_chunk[idx]
                chunk_ids.append(chunk_id)
                chunk_scores[chunk_id] = float(similarity)
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Chunk)
                WHERE c.chunk_id IN $chunk_ids
                
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
            
            chunk_dict = {}
            for record in result:
                chunk_id = record['chunk_id']
                chunk_dict[chunk_id] = Chunk(
                    chunk_ids=[chunk_id],
                    document_ids=[record['doc_id']] if record['doc_id'] else [],
                    text=record['text'],
                    position=0,
                    sentence_count=0,
                    token_count=0,
                    metadata={
                        'score': chunk_scores.get(chunk_id, 0.5),
                        'faiss_rank': chunk_ids.index(chunk_id),
                        'doc_type': record['doc_type'],
                        'jurisdiction': record['jurisdiction'],
                    }
                )
            
            # Return in FAISS order
            chunks = [chunk_dict[cid] for cid in chunk_ids if cid in chunk_dict]
            return chunks
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()