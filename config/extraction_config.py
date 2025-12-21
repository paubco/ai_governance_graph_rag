# -*- coding: utf-8 -*-
"""
Module: extraction_config.py
Package: config
Purpose: Configuration for all extraction phases (0B, 1A-1D, 2A)

Author: Pau Barba i Colomer
Created: 2025-12-18
Modified: 2025-12-20

References:
    - ARCHITECTURE.md Â§ 3-4 (Phases 1-2)
    - ARCHITECTURE.md Â§ 7.5 (Type normalization)
    - Phase 1B v2.0: Semantic + Metadata schema
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
        r'pp?\.\s*\d+[-â€“]\d+',       # Page ranges (p. 1-10, pp. 1-10)
    ],
}


# ============================================================================
# v2.0 TYPE SYSTEM - Semantic + Metadata (Single Source of Truth)
# ============================================================================

SEMANTIC_ENTITY_TYPES = {
    # Concepts (domain-fused, includes procedures and principles)
    "RegulatoryConcept": "Compliance ideas and principles (governance, transparency, accountability, conformity assessment)",
    "TechnicalConcept": "AI/ML ideas and methods (training, evaluation, model architecture, algorithms)",
    "PoliticalConcept": "Policy/governance ideas (institutional design, legislative process)",
    "EconomicConcept": "Financial/market ideas (portfolio management, asset allocation, market risk)",
    # Core types
    "Regulation": "Legally binding documents (EU AI Act, GDPR)",
    "Technology": "AI systems/tools/models (ChatGPT, BERT, neural networks)",
    "Organization": "Formal institutions (European Commission, NIST)",
    "Location": "Geographic/jurisdictional ONLY (EU, California, China) NOT languages",
    "Risk": "Adverse outcomes regulations address (bias, discrimination, safety, cybersecurity)",
}

METADATA_ENTITY_TYPES = {
    # Bibliographic
    "Citation": "References to external works: Author (Year), [1], [2]",
    "Author": "Researcher names ONLY, NOT organizations or AI tools",
    "Journal": "Publication venues (journals, conferences)",
    "Affiliation": "Institutional affiliations (universities, research centers, companies)",
    # Document structure
    "Document": "Named documents referenced structurally (EU AI Act, GDPR, research papers)",
    "DocumentSection": "Structural parts of documents (Article 5, Section 3, Annex A, page 12)",
}

# Convenience lists for validation
SEMANTIC_TYPE_NAMES = list(SEMANTIC_ENTITY_TYPES.keys())
METADATA_TYPE_NAMES = list(METADATA_ENTITY_TYPES.keys())


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
    
    # Entity embedding formats (v2.0)
    'semantic_format': '{name}({type})',    # e.g., "EU AI Act(Regulation)"
    'metadata_format': '{name}({type})',    # e.g., "Floridi (2018)(Citation)"
}


# ============================================================================
# PHASE 1B: ENTITY EXTRACTION (v2.0)
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
    
    # Dual-pass extraction (v2.0: semantic + metadata)
    'semantic_pass': True,           # Always run for all chunks
    'metadata_pass': True,           # Run for all chunks (was academic_pass)
    
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
# PHASE 1C: PRE-ENTITY FILTERING (v2.0 - type-specific)
# ============================================================================

PRE_ENTITY_FILTER_CONFIG = {
    # Provenance check settings
    'verify_provenance': True,
    'provenance_threshold': 0.80,      # rapidfuzz partial_ratio
    
    # Case-SENSITIVE patterns (preserves AI, ML, EU, G20, etc.)
    'blacklist_case_sensitive': [
        r'^\.+$',                    # ... or ..
        r'^\u2026$',                 # Unicode ellipsis (…)
        r'^\d+%$',                   # 80%, 41%, 83%
        r'^[a-z]$',                  # Single lowercase letter (k, q)
        r'^[RP]\d+$',                # R12, P9
        r'^n_[a-z]$',                # n_s, n_p (math variables)
        r'^[A-Z][a-z]?$',            # Single upper or Upper+lower (R, Ra, Re)
        r'^[a-z]+[A-Z]',             # Weird casing like "hEN"
        r'^[A-Z]\.$',                # Single letter + period (I.)
        r'^Eq\.$',                   # Equation marker
        r'^d\*$',                    # d* math notation
        r'^[a-z]{2}$',               # 2-char lowercase (do)
        r'^[_a-z]\d$',               # lowercase/underscore + digit (h1, _2)
        r'^[A-Z]\d$',                # Single uppercase + digit (B1, B2) - NOT G20
    ],
    
    # Case-INSENSITIVE patterns
    'blacklist_case_insensitive': [
        r'^N/?A$',                   # N/A, NA, n/a
        r'^RQ\d*$',                  # RQ, RQ1, RQ2
        r'^(who|why|how|what|when|use)$',
        r'^(res|age|fee|unl|tax)$',
        r'^(law|sex|act|noa)$',
        r'^ext$',                    # 3-char "ext"
    ],
    
    # Numeric patterns - ONLY apply outside Citation type
    'numeric_patterns': [
        r'^\d{1,3}$',                # 1-3 digit numbers (4, 18, 481)
        r'^\[\d+\]$',                # Bracketed refs [3], [1]
    ],
    
    # Types where numbers ARE valid (skip numeric patterns)
    'numeric_allowed_types': ['Citation'],
    
    # =======================================================================
    # TYPE-SPECIFIC BLACKLISTS (v2.0)
    # =======================================================================
    
    # Document type: Named documents, NOT figures/tables/technologies
    'document_blacklist': [
        # PDF artifacts
        r'^\.+$',                    # ...
        r'^\u2026$',                 # Unicode ellipsis
        
        # Figure/Table references (not documents)
        r'^Table\s*\d+$',            # Table 1, Table 2
        r'^Figure\s*\d+$',           # Figure 1, Figure 2
        r'^Fig\.\s*\d+$',            # Fig. 1, Fig.2
        r'^Algorithm\s*\d+$',        # Algorithm 1
        
        # Technologies misclassified as Document
        r'^ChatGPT\d*$',             # ChatGPT, ChatGPT3
        r'^GPT-?\d*$',               # GPT-3, GPT4
        r'^CNN$',
        r'^LSTM$',
        r'^XGBoost$',
        r'^RF$',
        r'^DRM$',
        r'^Chatbot$',
        
        # Years (not documents)
        r'^(2018|2019|2020|2021|2022|2023|2024|2025)$',
        
        # Too generic
        r'^Article$',                # "Article" alone
        r'^AI$',                     # Too short
        r'^(this|This)\s+(paper|study)$',  # "this paper", "This study"
        r'^education$',
        r'^healthcare$',
        
        # Geographic (should be Location)
        r'^China$',
        r'^European Union$',
        
        # Narrative/Contribution refs
        r'^Narrative\s*\d+$',
        r'^Contribution\s*\d+$',
        r'^Contributions\s*\d+$',
        
        # Single numbers
        r'^\d{1,2}$',
        
        # Too short - causes spurious PART_OF matches (v2.0)
        r'^.{1,3}$',                 # 1-3 char names: DA, NN, RA, RED
        
        # Generic English words misclassified as Document (case-insensitive patterns below)
    ],
    
    # Case-insensitive document blacklist (generic words)
    'document_blacklist_ci': [
        r'^systems?$',               # system, systems
        r'^graph$',
        r'^rights?$',                # right, rights
        r'^learning$',
        r'^models?$',                # model, models
        r'^data$',
        r'^governments?$',           # government, governments
        r'^regulations?$',           # regulation, regulations
        r'^education$',
        r'^research$',
        r'^safety$',
        r'^frameworks?$',
        r'^services?$',
        r'^study$',
        r'^health$',
        r'^content$',
        r'^ethics$',
        r'^tools$',
        r'^papers?$',
        r'^students?$',
        r'^impact$',
        r'^article$',
        r'^governance$',
        r'^training$',
        r'^testing$',
        r'^industry$',
        r'^transport$',
        r'^records?$',
    ],
    
    # DocumentSection type: Article 5, Section 3, Annex I — NOT figures/tables
    'document_section_blacklist': [
        # PDF artifacts
        r'^\.+$',
        r'^\u2026$',
        r'^Article\.+$',             # Article...
        
        # Figure/Table references
        r'^Table\s*\d+$',
        r'^Figure\s*\d+$',
        r'^Fig\.\s*\d+$',
        
        # Meta sections (not document structure)
        r'^Abstract$',
        r'^ABSTRACT$',
        r'^Keywords$',
        r'^KEYWORDS$',
        
        # Too generic
        r'^Article$',                # "Article" alone (not "Article 5")
        r'^Article\s*\(not specified\)$',
        r'^Section$',
        
        # Single numbers (not valid sections)
        r'^\d{1,2}$',
        
        # Too short (v2.0)
        r'^.{1,3}$',                 # 1-3 char names
    ],
}


# ============================================================================
# PHASE 1C: ENTITY DISAMBIGUATION (v2.0)
# ============================================================================

DISAMBIGUATION_CONFIG = {
    # Model - Mistral-7B for SameJudge (Qwen has JSON bugs)
    'model_name': 'mistralai/Mistral-7B-Instruct-v0.3',
    
    # FAISS blocking
    'faiss_k': 50,                   # Neighbors to retrieve
    'faiss_threshold': 0.70,         # Blocking threshold (permissive)
    'faiss_M': 32,                   # HNSW connections per node
    'faiss_ef_construction': 200,    # Build quality
    'faiss_ef_search': 64,           # Search quality
    
    # Tiered thresholds (tuned from manual review)
    'auto_merge_threshold': 0.98,    # >= this: auto-merge
    'auto_reject_threshold': 0.885,  # < this: auto-reject (raised from 0.88)
    # Between 0.885-0.98: LLM decides
    
    # LLM SameJudge
    'temperature': 0.0,
    'max_tokens': 256,
    'max_llm_pairs': 25000,          # Cost control limit
    'max_workers': 8,                # Parallel workers for LLM calls
    
    # Cluster breaking (betweenness centrality)
    # Cuts high-betweenness edges (bridges between subcommunities) with low similarity
    'min_cluster_size': 5,           # Only analyze clusters >= this size
    'betweenness_threshold': 0.3,    # Cut edges with betweenness > this
    'similarity_ceiling': 0.91,      # Only cut if similarity < this
    'hard_max_size': 20,             # Force-cut weakest edge if cluster exceeds this
    
    # Batch processing
    'batch_size': 50,
    'checkpoint_frequency': 1000,
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