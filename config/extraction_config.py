# -*- coding: utf-8 -*-
"""
Module: extraction_config.py
Package: src.config
Purpose: Configuration for extraction phases (1A-1D, 2A)

Author: Pau Barba i Colomer
Created: 2025-12-18
Modified: 2025-12-18

References:
    - ARCHITECTURE.md § 3-4 (Phases 1-2)
    - ARCHITECTURE.md § 7.5 (Type normalization)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# API KEYS (from .env)
# ============================================================================

TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
SCOPUS_API_KEY = os.getenv('SCOPUS_API_KEY')


# ============================================================================
# v1.1 TYPE SYSTEM (TBD - pending empirical testing)
# ============================================================================
# Design: Type × Domain matrix for semantic extraction
# All entities get Type × Domain treatment; exact values TBD after testing
#
# Candidate domains: Regulatory, Technical, General
# Candidate base types: Concept, Document, Process, Organization, Person, 
#                       Location, Technology, Group
#
# Academic extraction uses separate simpler schema (single-dimension, TBD)
# ============================================================================

# Placeholder - will be populated after type matrix testing
SEMANTIC_ENTITY_TYPES = []  # TBD: e.g., ["Regulatory Concept", "Technical Concept", ...]
SEMANTIC_DOMAINS = []       # TBD: e.g., ["Regulatory", "Technical", "General"]

# Academic entity types (separate extraction pass)
ACADEMIC_ENTITY_TYPES = [
    "Citation",      # e.g., "Floridi (2018)", "Zhang et al. 2025"
    "Self-Reference", # e.g., "the authors", "this study"
    "Publication",   # e.g., "Nature", "NeurIPS"
]


# ============================================================================
# PHASE 1A: CHUNKING (v1.1 - empirically derived from BGE-small analysis)
# ============================================================================
# Similarity distribution (BGE-small-en-v1.5 on 10 papers + 10 regulations):
#   Papers:      mean=0.629, p5=0.455, p25=0.550, p50=0.632
#   Regulations: mean=0.666, p5=0.505, p25=0.592, p50=0.666
# 
# Threshold at p25 (~0.55) creates meaningful semantic breaks
# Coherence filter at p5 (~0.40) catches garbage without discarding content
# ============================================================================

CHUNKING_CONFIG = {
    # Boundary detection model (same family as final embeddings)
    'boundary_model': 'BAAI/bge-small-en-v1.5',
    
    # Similarity threshold for chunk boundaries
    # Set at ~25th percentile: breaks when similarity drops below this
    # Lower = larger chunks, Higher = smaller chunks
    'similarity_threshold': 0.45,
    
    # Chunk sizing constraints
    'min_sentences': 3,               # Minimum sentences per chunk
    'max_tokens': 1500,               # Maximum tokens per chunk
    
    # Coherence filtering (mean adjacent-sentence similarity)
    # Set at ~5th percentile: only discards true garbage
    'min_coherence': 0.30,
    'min_tokens': 15,                 # Discard tiny single-sentence chunks
    
    # Density filtering (tokens per sentence)
    # Low density + no header = likely reference list / NLTK artifacts
    'min_tokens_per_sentence': 10,    # Below this + orphan = garbage
    
    # Deduplication (server mode only, uses BGE-M3 embeddings)
    'dedup_threshold': 0.95,
    'dedup_threshold': 0.95,
    
    # Source-type aware header detection
    'header_patterns': {
        'regulation': [
            r'^#{1,6}\s+.+$',           # Markdown headers
            r'^Article\s+\d+',          # Article 5
            r'^Section\s+\d+',          # Section 3
            r'^\d+\.\s+[A-Z]',          # 1. Title
            r'^[A-Z][A-Z\s]{3,}$',      # ALL CAPS
        ],
        'paper': [
            r'^#\s+\d+(?:\.\d+)*\.?\s+.+$',  # Numbered: # 1. Introduction
            r'^#\s+(?:Introduction|Conclusion|Discussion|Results|Methods|'
            r'Methodology|Background|Literature|Related\s+Work|Theoretical|'
            r'Empirical|Analysis|Findings|Implications|Limitations|Future).*$',
        ],
    },
    
    # Garbage headers to skip entirely (papers)
    'garbage_headers': {
        'ARTICLEINFO', 'ARTICLE INFO', 'KEYWORDS', 'ORCID', 'OPEN ACCESS',
        'ACKNOWLEDGMENTS', 'ACKNOWLEDGEMENTS', 'Acknowledgments', 'Acknowledgements',
        'Funding', 'Author contributions', 'Correspondence', 'Data availability',
        'Data availability statement', 'Conflict of interest', 'Competing interests',
        'Declaration of competing interest', 'Declarations', 'Disclosure statement',
        "Publisher's note", "Publisher's Note", 'CCS CONCEPTS', 'References',
        'REFERENCES', 'Bibliography',
    },
}


# ============================================================================
# EMBEDDING CONFIGURATION
# ============================================================================

EMBEDDING_CONFIG = {
    # Final embeddings (retrieval, deduplication)
    'model_name': 'BAAI/bge-m3',
    'dimension': 1024,
    'batch_size': 32,
    'device': 'cuda',                # or 'cpu'
    'normalize': True,               # L2 normalize embeddings
    
    # Entity embedding format: "{name}({type})"
    # Per RAKG methodology - description NOT included in embedding
    'entity_format': '{name}({type})',
}


# ============================================================================
# PHASE 1B: ENTITY EXTRACTION
# ============================================================================

ENTITY_EXTRACTION_CONFIG = {
    # Model
    'model_name': 'Qwen/Qwen2.5-72B-Instruct-Turbo',  # v1.0 used this
    # 'model_name': 'mistralai/Mistral-7B-Instruct-v0.3',  # v1.1 recommendation
    
    # LLM parameters
    'temperature': 0.0,              # Deterministic for consistency
    'max_tokens': 4096,              # Response limit
    'top_p': 0.95,
    
    # Batch processing
    'batch_size': 10,                # Chunks per API call
    'max_workers': 4,                # Parallel API calls
    'retry_attempts': 3,
    'retry_delay': 2.0,              # seconds
    
    # Rate limiting
    'requests_per_minute': 60,
    'tokens_per_minute': 100000,
    
    # Checkpointing
    'checkpoint_frequency': 100,     # Save every N chunks
}


# ============================================================================
# PHASE 1C: ENTITY DISAMBIGUATION
# ============================================================================

DISAMBIGUATION_CONFIG = {
    # Model (for SameJudge refinement)
    'model_name': 'Qwen/Qwen2-7B-Instruct',  # v1.0
    # 'model_name': 'mistralai/Mistral-7B-Instruct-v0.3',  # v1.1 recommendation
    
    # FAISS blocking
    'similarity_threshold': 0.85,    # Minimum cosine for candidate pairs
    'top_k_candidates': 20,          # Max candidates per entity
    
    # Tiered thresholds (RAKG-inspired)
    'auto_merge_threshold': 0.95,    # Above this: auto-merge without LLM
    'llm_review_threshold': 0.85,    # Between 0.85-0.95: LLM decides
    'reject_threshold': 0.85,        # Below this: definitely different
    
    # LLM parameters
    'temperature': 0.0,
    'max_tokens': 256,
    
    # Batch processing
    'batch_size': 50,                # Entity pairs per batch
    'max_workers': 4,
    'checkpoint_frequency': 500,
}


# ============================================================================
# PHASE 1D: RELATION EXTRACTION
# ============================================================================

RELATION_EXTRACTION_CONFIG = {
    # Model (Mistral works better for JSON than Qwen)
    'model_name': 'mistralai/Mistral-7B-Instruct-v0.3',
    
    # LLM parameters
    'temperature': 0.0,
    'max_tokens': 2048,
    
    # Corpus retrospective retrieval
    'chunks_per_entity': 10,         # Max chunks to retrieve per entity
    'mmr_lambda': 0.7,               # Diversity vs relevance balance
    'similarity_threshold': 0.6,     # Min similarity for chunk retrieval
    
    # Two-track extraction (v1.0)
    'semantic_track': True,          # Full OpenIE for semantic entities
    'citation_track': True,          # Constrained extraction for citations
    
    # Batch processing
    'batch_size': 5,                 # Entities per batch
    'max_workers': 2,                # Lower for rate limiting
    'checkpoint_frequency': 50,
    
    # Rate limiting
    'requests_per_minute': 30,
}


# ============================================================================
# PHASE 2A: SCOPUS ENRICHMENT
# ============================================================================

ENRICHMENT_CONFIG = {
    # Citation matching
    'citation_match_threshold': 0.8,  # Fuzzy match score
    'author_match_threshold': 0.9,
    
    # L1 → L2 matching
    'title_similarity_threshold': 0.85,
    'doi_exact_match': True,
    
    # Metadata extraction
    'extract_authors': True,
    'extract_journals': True,
    'extract_references': True,
    'extract_keywords': True,
}


# ============================================================================
# NEO4J CONFIGURATION
# ============================================================================

NEO4J_CONFIG = {
    'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
    'user': os.getenv('NEO4J_USER', 'neo4j'),
    'password': os.getenv('NEO4J_PASSWORD'),
    'database': 'neo4j',
    
    # Import settings
    'batch_size': 1000,              # Nodes/relations per transaction
    'use_periodic_commit': True,
}


# ============================================================================
# SCRAPER CONFIGURATION (Phase 0)
# ============================================================================

SCRAPER_CONFIG = {
    'base_url': 'https://intelligence.dlapiper.com/artificial-intelligence/',
    'delay_between_requests': 2,     # seconds
    'timeout': 10,
    'retry_attempts': 3,
    'headers': {
        'User-Agent': 'Mozilla/5.0 (Educational Research Bot)',
    },
}


# ============================================================================
# LOGGING
# ============================================================================

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout',
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console'],
    },
}