# -*- coding: utf-8 -*-
"""
Module: extraction_config.py
Package: src.config
Purpose: Configuration for extraction phases (1A-1D, 2A)

Author: Pau Barba i Colomer
Created: 2025-12-18
Modified: 2025-12-19

References:
    - ARCHITECTURE.md Section 3-4 (Phases 1-2)
    - ARCHITECTURE.md Section 7.5 (Type normalization)
    - Phase 1B spec: Type x Domain schema
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
# v1.1 TYPE SYSTEM - Type x Domain Matrix
# ============================================================================
# Design: Semantic entities get Type x Domain treatment
# Academic entities use simpler single-dimension schema (no domain)
#
# Embedding formats:
#   Semantic: "{name}({domain} {type})" -> "conformity assessment(Regulatory Process)"
#   Academic: "{name}({type})" -> "Floridi (2018)(Citation)"
# ============================================================================

# -----------------------------------------------------------------------------
# DOMAINS (4) - For semantic entities only
# -----------------------------------------------------------------------------
SEMANTIC_DOMAINS = [
    "Regulatory",   # Legal requirements, compliance, policy rules
    "Political",    # Governance actors, policy-making bodies
    "Technical",    # AI/ML systems, methods, algorithms, tools
    "General",      # Domain-agnostic or cross-cutting concepts
]

# Domain descriptions for prompt context
DOMAIN_DESCRIPTIONS = {
    "Regulatory": "Legal requirements, compliance, policy rules - laws, directives, requirements, legal obligations",
    "Political": "Governance actors, policy-making bodies - governments, ministries, commissions, political entities",
    "Technical": "AI/ML systems, methods, algorithms, tools - models, algorithms, software, technical processes",
    "General": "Domain-agnostic or cross-cutting concepts - abstract principles that span multiple domains",
}

# -----------------------------------------------------------------------------
# SEMANTIC TYPES (12) - Used with domains
# -----------------------------------------------------------------------------
SEMANTIC_ENTITY_TYPES = [
    "Concept",       # Abstract ideas, principles, terms (NOT processes, NOT normative principles)
    "Regulation",    # Legally binding documents, laws, directives
    "Technology",    # AI systems, models, algorithms, tools
    "Organization",  # Institutions, companies, agencies with formal structure
    "Person",        # Named individuals
    "Location",      # Geographic/jurisdictional entities
    "Process",       # Procedures, methodologies with defined steps
    "Document",      # Reports, standards, non-binding publications
    "Group",         # Collectives, categories without formal structure
    "Metric",        # Measurable quantities, KPIs
    "Principle",     # Normative values, rights, ethical concepts
    "Event",         # Conferences, milestones, temporal occurrences
]

# Type descriptions with disambiguation hints
TYPE_DESCRIPTIONS = {
    "Concept": "Abstract ideas, principles, terms. NOT a Process (no steps), NOT a Principle (not normative).",
    "Regulation": "Legally binding documents, laws, directives. NOT Document (must be legally binding).",
    "Technology": "AI systems, models, algorithms, tools. NOT Process (it's a thing, not steps).",
    "Organization": "Institutions, companies, agencies with formal structure. NOT Group (has formal structure).",
    "Person": "Named individuals. NOT Organization, NOT Role.",
    "Location": "Geographic/jurisdictional entities. NOT Organization (it's a place).",
    "Process": "Procedures, methodologies with defined steps. NOT Concept (has defined steps).",
    "Document": "Reports, standards, non-binding publications. NOT Regulation (not legally binding).",
    "Group": "Collectives, categories without formal structure. NOT Organization (no formal structure).",
    "Metric": "Measurable quantities, KPIs. NOT Concept (must be quantifiable).",
    "Principle": "Normative values, rights, ethical concepts. NOT Concept (has normative/ethical weight).",
    "Event": "Conferences, milestones, temporal occurrences. NOT Process (point in time, not steps).",
}

# -----------------------------------------------------------------------------
# ACADEMIC TYPES (4) - No domain, used for paper chunks
# -----------------------------------------------------------------------------
ACADEMIC_ENTITY_TYPES = [
    "Citation",       # In-text references: "Author (Year)", "Author et al. (Year)"
    "Author",         # Named researchers/writers from citations or author lists
    "Journal",        # Publication venues, conference proceedings
    "Self-Reference", # Meta-references: "this study", "the authors", "our approach", "we"
]

ACADEMIC_TYPE_DESCRIPTIONS = {
    "Citation": "In-text references like 'Author (Year)' or 'Author et al. (Year)'",
    "Author": "Named researchers or writers - full names from citations or author lists",
    "Journal": "Publication venues, conference proceedings",
    "Self-Reference": "References to current work: 'this study', 'the authors', 'we propose'",
}

# -----------------------------------------------------------------------------
# DISAMBIGUATION RULES (embedded in prompts)
# -----------------------------------------------------------------------------
DOMAIN_DISAMBIGUATION_RULES = """
- If it's a LAW, DIRECTIVE, REQUIREMENT, or LEGAL OBLIGATION -> Regulatory
- If it's a GOVERNMENT, MINISTRY, COMMISSION, or POLITICAL BODY -> Political
- If it's an AI SYSTEM, ALGORITHM, MODEL, or TECHNICAL METHOD -> Technical
- If it could apply to multiple domains equally -> General
- When in doubt between Regulatory and Political:
  - Regulatory = the rule itself
  - Political = the body that makes/enforces rules
"""

TYPE_DISAMBIGUATION_RULES = """
- Concept vs Principle: Does it have normative/ethical weight? -> Principle. Otherwise -> Concept.
- Concept vs Process: Does it have defined steps/stages? -> Process. Otherwise -> Concept.
- Regulation vs Document: Is it legally binding? -> Regulation. Otherwise -> Document.
- Technology vs Process: Is it a thing you use, or steps you follow? Thing -> Technology. Steps -> Process.
- Organization vs Group: Does it have formal structure (leadership, legal entity)? -> Organization. Otherwise -> Group.
- Person vs Organization: Is it an individual human? -> Person. Otherwise -> Organization.
"""


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
    # Model - v1.1 uses Mistral-7B (better JSON, lower cost)
    'model_name': 'mistralai/Mistral-7B-Instruct-v0.3',
    
    # LLM parameters
    'temperature': 0.0,              # Deterministic for consistency
    'max_tokens': 4096,
    'top_p': 0.95,
    
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