# -*- coding: utf-8 -*-
"""
Module: extraction_config.py
Package: config
Purpose: Configuration for all extraction phases (0B, 1A-1D, 2A)

Author: Pau Barba i Colomer
Created: 2025-12-18
Modified: 2025-12-19

References:
    - ARCHITECTURE.md § 3-4 (Phases 1-2)
    - ARCHITECTURE.md § 7.5 (Type normalization)
    - Phase 1B spec: Type × Domain schema
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
# PHASE 0B: PREPROCESSING
# ============================================================================

PREPROCESSING_CONFIG = {
    # Sections to remove entirely from papers (heading + content to next heading)
    # These contain metadata, not content useful for entity extraction
    'garbage_sections': [
        # Article metadata
        'ARTICLEINFO', 'ARTICLE INFO',
        
        # Keywords (redundant with Scopus metadata)
        'KEYWORDS', 'Keywords', 'Keywords:',
        
        # Acknowledgments (all case variants)
        'ACKNOWLEDGMENTS', 'ACKNOWLEDGEMENTS',
        'Acknowledgments', 'Acknowledgements', 
        'Acknowledgment', 'ACKNOWLEDGEMENT',
        
        # Author/contribution info
        'Author contributions', 'Author Contributions',
        'Credit authorship contribution statement',
        'Author biographies',
        
        # Funding
        'Funding', 'Funding information',
        'Information for funding/support or the lack of it',
        
        # Conflicts/declarations
        'Conflict of interest', 'Conflicts of interest',
        'CONFLICT OF INTEREST', 'CONFLICT OF INTEREST STATEMENT',
        'Declaration of competing interest', 'Declaration of conflicting interests',
        'Declarations', 'Declaration',
        'DECLARATION OF INTERESTS',
        'Competing interests',
        'Disclosure statement',
        
        # Data availability
        'Data availability', 'Data Availability',
        'DATA AVAILABILITY STATEMENT',
        'Availability of data and materials',
        
        # Author identifiers/contact
        'ORCID', 'ORCID iDs', 'ORCID:',
        'Correspondence', 'Corresponding author', 'Corresponding author:',
        
        # Publisher boilerplate
        'Open Access', 'OPEN ACCESS',
        "Publisher's note",
        'Check for updates',
        'Additional information',
        
        # Technical metadata
        'CCS CONCEPTS',
        'Ethics statement',
        
        # Misc
        'Abbreviations',
        'Notes',
    ],
    
    # Sections to EXTRACT (saved separately) then remove from text
    # These are bibliography/reference sections
    'reference_sections': [
        'References', 'REFERENCES',
        'Bibliography', 'BIBLIOGRAPHY',
        'Works Cited',
        'ACM Reference Format:',
    ],
    
    # Patterns to strip from chunk text (v1.1 - post Phase 1B testing)
    'garbage_patterns': [
        r'Table\s+\d+',              # Table 1, Table 2
        r'Figure\s+\d+',             # Figure 1, Figure 2
        r'Fig\.\s*\d+',              # Fig. 1, Fig.2
        r'doi:\s*\S+',               # DOI references
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
        r'Vol\.\s*\d+',              # Volume references
        r'pp?\.\s*\d+[-–]\d+',       # Page ranges (p. 1-10, pp. 1-10)
    ],
}


# ============================================================================
# v1.2 TYPE SYSTEM - Domain-Fused (Single Source of Truth)
# ============================================================================

SEMANTIC_ENTITY_TYPES = {
    # Concepts (domain-fused)
    "RegulatoryConcept": "Legal/compliance ideas (governance, privacy, requirements)",
    "TechnicalConcept": "AI/ML ideas (training data, model architecture, algorithms)",
    "PoliticalConcept": "Governance ideas (policy frameworks, institutional design)",
    # Processes (domain-fused)
    "RegulatoryProcess": "Compliance procedures (conformity assessment, auditing)",
    "TechnicalProcess": "Technical procedures (model training, evaluation)",
    "PoliticalProcess": "Policy procedures (legislative process, consultation)",
    # Core types
    "Regulation": "Legally binding documents (EU AI Act, GDPR, Article 5)",
    "Technology": "AI systems/tools/models (ChatGPT, BERT, neural networks)",
    "Organization": "Formal institutions (European Commission, NIST)",
    "Location": "Geographic/jurisdictional (EU, California, China)",
    # Values and concerns
    "EthicalPrinciple": "Normative values (transparency, fairness, accountability, human dignity)",
    "Risk": "Adverse outcomes regulations address (bias, discrimination, safety, cybersecurity)",
}

ACADEMIC_ENTITY_TYPES = {
    "Citation": "In-text references: 'Author (Year)', 'Author et al. (Year)'",
    "Author": "Named researchers (full names only)",
    "Journal": "Publication venues, conferences",
}

# Convenience lists for validation
SEMANTIC_TYPE_NAMES = list(SEMANTIC_ENTITY_TYPES.keys())
ACADEMIC_TYPE_NAMES = list(ACADEMIC_ENTITY_TYPES.keys())


# ============================================================================
# PHASE 1A: CHUNKING (v1.1 - empirically derived from BGE-small analysis)
# ============================================================================

CHUNKING_CONFIG = {
    # Boundary detection model (same family as final embeddings)
    'boundary_model': 'BAAI/bge-small-en-v1.5',
    
    # Similarity threshold for chunk boundaries
    'similarity_threshold': 0.45,
    
    # Chunk sizing constraints
    'min_sentences': 3,
    'max_tokens': 1500,
    
    # Coherence filtering
    'min_coherence': 0.30,
    'min_tokens': 15,
    
    # Density filtering
    'min_tokens_per_sentence': 10,
    
    # Merge duplicates
    'merge_threshold': 0.98,
    'dedup_threshold': 0.95,
    
    # Header patterns
    'header_patterns': {
        'regulation': [
            r'^#{1,6}\s+.+$',
            r'^Article\s+\d+',
            r'^Section\s+\d+',
            r'^\d+\.\s+[A-Z]',
            r'^[A-Z][A-Z\s]{3,}$',
        ],
        'paper': [
            r'^#\s+\d+(?:\.\d+)*\.?\s+.+$',
            r'^#\s+(?:Introduction|Conclusion|Discussion|Results|Methods|'
            r'Methodology|Background|Literature|Related\s+Work|Theoretical|'
            r'Empirical|Analysis|Findings|Implications|Limitations|Future).*$',
        ],
    },
    
    # Garbage headers to skip
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
    'model_name': 'BAAI/bge-m3',
    'dimension': 1024,
    'batch_size': 32,
    'device': 'cuda',
    'normalize': True,
    
    # Entity embedding formats (v1.1)
    'semantic_format': '{name}({domain} {type})',   # e.g., "EU AI Act(Regulatory Regulation)"
    'academic_format': '{name}({type})',            # e.g., "Floridi (2018)(Citation)"
}


# ============================================================================
# PHASE 1B: ENTITY EXTRACTION (v1.1)
# ============================================================================

ENTITY_EXTRACTION_CONFIG = {
    # Model - v1.1 uses Mistral-7B (better JSON, lower cost than Qwen-72B)
    'model_name': 'mistralai/Mistral-7B-Instruct-v0.3',
    
    # LLM parameters
    'temperature': 0.0,              # Deterministic for consistency
    'max_tokens': 4096,
    'top_p': 0.95,
    
    # JSON mode - Together.ai supports response_format={"type": "json_object"}
    'use_json_mode': True,
    
    # Dual-pass extraction
    'semantic_pass': True,           # Always run for all chunks
    'academic_pass': True,           # Only for paper chunks
    
    # Batch processing
    'batch_size': 10,
    'max_workers': 4,
    'retry_attempts': 3,
    'retry_delay': 2.0,
    
    # Rate limiting
    'requests_per_minute': 60,
    'tokens_per_minute': 100000,
    
    # Checkpointing
    'checkpoint_frequency': 100,
}


# ============================================================================
# PHASE 1C: ENTITY DISAMBIGUATION
# ============================================================================

DISAMBIGUATION_CONFIG = {
    # Model - v1.1 uses Mistral-7B for consistency
    'model_name': 'mistralai/Mistral-7B-Instruct-v0.3',
    
    # FAISS blocking
    'similarity_threshold': 0.85,
    'top_k_candidates': 20,
    
    # Tiered thresholds (RAKG-inspired)
    'auto_merge_threshold': 0.95,
    'llm_review_threshold': 0.85,
    'reject_threshold': 0.85,
    
    # LLM parameters
    'temperature': 0.0,
    'max_tokens': 256,
    
    # Batch processing
    'batch_size': 50,
    'max_workers': 4,
    'checkpoint_frequency': 500,
}


# ============================================================================
# PHASE 1D: RELATION EXTRACTION
# ============================================================================

RELATION_EXTRACTION_CONFIG = {
    'model_name': 'mistralai/Mistral-7B-Instruct-v0.3',
    
    'temperature': 0.0,
    'max_tokens': 2048,
    
    # Corpus retrospective retrieval
    'chunks_per_entity': 10,
    'mmr_lambda': 0.7,
    'similarity_threshold': 0.6,
    
    # Two-track extraction
    'semantic_track': True,
    'citation_track': True,
    
    # Batch processing
    'batch_size': 5,
    'max_workers': 2,
    'checkpoint_frequency': 50,
    
    'requests_per_minute': 30,
}


# ============================================================================
# PHASE 2A: SCOPUS ENRICHMENT
# ============================================================================

ENRICHMENT_CONFIG = {
    'citation_match_threshold': 0.8,
    'author_match_threshold': 0.9,
    'title_similarity_threshold': 0.85,
    'doi_exact_match': True,
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
    'batch_size': 1000,
    'use_periodic_commit': True,
}


# ============================================================================
# SCRAPER CONFIGURATION (Phase 0)
# ============================================================================

SCRAPER_CONFIG = {
    'base_url': 'https://intelligence.dlapiper.com/artificial-intelligence/',
    'delay_between_requests': 2,
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