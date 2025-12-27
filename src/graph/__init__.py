# -*- coding: utf-8 -*-
"""
Graph construction package for Neo4j import and FAISS index building.

Contains neo4j_importer (batched UNWIND import engine), neo4j_import_processor
(orchestrator with checkpointing), and faiss_builder (HNSW index construction for
entity and chunk embeddings).
"""
