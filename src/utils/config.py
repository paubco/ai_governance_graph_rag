# src/utils/config.py

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
ENRICHED_DATA_PATH = DATA_PATH / "enriched"

# API Keys
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
SCOPUS_API_KEY = os.getenv("SCOPUS_API_KEY")
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", None)  # Optional

# LLM Configuration (Mistral-7B via Together.ai)
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
LLM_TEMPERATURE = 0.0  # Deterministic for entity extraction
LLM_MAX_TOKENS = 1024

# Embedding Configuration (BGE-M3)
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024
EMBEDDING_BATCH_SIZE = 32

# Text Processing
SEMANTIC_CHUNK_THRESHOLD = 0.7  # Cosine similarity for sentence grouping
MIN_CHUNK_LENGTH = 100  # words
MAX_CHUNK_LENGTH = 300  # words

# RAKG Entity Normalization
VECJUDGE_THRESHOLD_SINGLE = 0.96  # Single-word entities
VECJUDGE_THRESHOLD_MULTI = 0.75   # Multi-word entities
SAMEJUDGE_TEMPERATURE = 0.0       # LLM refinement

# Neo4j Vector Index
VECTOR_INDEX_NAME = "entity_embeddings"
VECTOR_SIMILARITY_METRIC = "cosine"

# Data Acquisition
DLA_PIPER_BASE_URL = "https://www.dlapiperintelligence.com/aialgorithms"
SCOPUS_MAX_PAPERS = 50
RATE_LIMIT_DELAY = 2.0  # seconds between requests

# Debug
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
