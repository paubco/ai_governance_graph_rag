# -*- coding: utf-8 -*-
"""
Retrieval Config

# ============================================================================
# ENTITY TYPES (v2.0 - imported from extraction_config.py for DRY)
# ============================================================================

References:
    ARCHITECTURE.md Â§ 5 (Phase 3)
    PHASE_3_DESIGN.md

"""
"""
import os
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# RETRIEVAL MODE
# ============================================================================

class RetrievalMode(Enum):
    """
    Retrieval strategy modes for ablation studies.
    
    SEMANTIC: Semantic retrieval only (vector similarity via FAISS)
    GRAPH: Graph retrieval only (entity-centric with PCST)
    DUAL: Both channels merged (default)
    """
    SEMANTIC = "semantic"
    GRAPH = "graph"
    DUAL = "dual"


# ============================================================================
# ENTITY TYPES (v2.0 - imported from extraction_config.py for DRY)
# ============================================================================

from config.extraction_config import SEMANTIC_TYPE_NAMES, METADATA_TYPE_NAMES

ENTITY_TYPES = SEMANTIC_TYPE_NAMES + METADATA_TYPE_NAMES


# ============================================================================
# QUERY UNDERSTANDING (Phase 3.3.1)
# ============================================================================

QUERY_PARSING_CONFIG = {
    # Model for entity extraction from query
    'model_name': 'mistralai/Mistral-7B-Instruct-v0.3',
    'temperature': 0.0,
    'max_tokens': 512,
    
    # Entity resolution
    'fuzzy_threshold': 0.75,         # Min cosine for entity match
    'top_k_per_entity': 10,          # FAISS candidates per query entity
    'max_entities_per_query': 5,     # Limit extracted entities
}


# ============================================================================
# ENTITY RESOLUTION (Phase 3.3.1b)
# ============================================================================

ENTITY_RESOLUTION_CONFIG = {
    'fuzzy_threshold': 0.75,         # Cosine similarity threshold
    'top_k_per_entity': 10,          # Candidates per query entity
    'exact_match_boost': 0.1,        # Score bonus for exact name match
}


# ============================================================================
# GRAPH EXPANSION - PCST (Phase 3.3.2a)
# ============================================================================

PCST_CONFIG = {
    'k_candidates': 5,               # Top-K similar entities per seed (FAISS)
    'delta': 0.5,                    # Neo4j PCST prize-cost balance
    'prize_strategy': 'uniform',     # 'uniform' or 'frequency'
    'cost_strategy': 'uniform',      # 'uniform', 'frequency', or 'similarity'
    'max_entities': 50,              # Global cap on expanded entities
    'max_hops': 2,                   # Maximum traversal depth
}


# ============================================================================
# CHUNK RETRIEVAL (Phase 3.3.2b)
# ============================================================================

RETRIEVAL_CONFIG = {
    # Graph retrieval
    'graph_all_entities': True,      # Get all chunks from PCST entities
    
    # Semantic retrieval
    'semantic_top_k': 15,            # Semantic search chunk limit
    
    # Entity resolution
    'entity_resolution_top_k': 3,    # Max entities per extracted mention
    
    # PCST limits
    'pcst_max_entities': 50,         # Hub node control
    
    # Token budget
    'chunk_token_limit': 8000,       # Approximate token budget for context
}


# ============================================================================
# RANKING & SCORING (Phase 3.3.2c)
# ============================================================================
# Multiplicative system - all scores bounded [0,1]

RANKING_CONFIG = {
    # Graph bonus multipliers
    'graph_provenance_multiplier': 1.0,   # Chunks with PCST relations
    'graph_entity_multiplier': 0.85,      # Chunks from entity expansion
    
    # Hint-based adjustments (soft, not hard filters)
    'jurisdiction_penalty': 0.9,          # Multiply if NO jurisdiction match
    'doc_type_penalty': 0.85,             # Multiply if NO doc_type match
    
    # Final selection
    'final_top_k': 20,                    # Final chunks for LLM context
    
    # Deduplication
    'dedup_similarity_threshold': 0.95,   # Near-duplicate removal
}


# ============================================================================
# ANSWER GENERATION (Phase 3.3.4)
# ============================================================================

ANSWER_GENERATION_CONFIG = {
    # LLM provider and model
    'provider': 'anthropic',
    'model': 'claude-3-5-haiku-20241022',
    'api_key': os.getenv('ANTHROPIC_API_KEY'),
    
    # Token budgets
    'max_input_tokens': 15000,
    'token_budget': {
        'system_prompt': 1000,
        'graph_structure': 500,
        'entity_context': 500,
        'source_chunks': 10000,
        'instructions': 500,
        'buffer': 1000,
    },
    
    # Generation parameters
    'max_output_tokens': 2000,
    'temperature': 0.0,              # Deterministic for evaluation
    'top_p': 0.9,
    
    # Formatting
    'max_chunks_to_format': 40,
    'truncate_chunk_chars': 1000,
}

# Alternative: Mistral configuration (8k context, cheaper)
ANSWER_GENERATION_CONFIG_MISTRAL = {
    'provider': 'together',
    'model': 'mistralai/Mistral-7B-Instruct-v0.3',
    'api_key': os.getenv('TOGETHER_API_KEY'),
    
    'max_input_tokens': 6000,
    'token_budget': {
        'system_prompt': 500,
        'graph_structure': 300,
        'entity_context': 200,
        'source_chunks': 2000,
        'instructions': 200,
        'buffer': 500,
    },
    
    'max_output_tokens': 1500,
    'temperature': 0.0,
    'top_p': 0.9,
    
    'max_chunks_to_format': 10,
    'truncate_chunk_chars': 400,
}


# ============================================================================
# JURISDICTION PATTERNS (for query parsing)
# ============================================================================

JURISDICTION_PATTERNS = {
    # Europe - EU & Members
    r'\b(EU|European Union|Europe)\b': 'EU',
    r'\b(Austria|Austrian)\b': 'AT',
    r'\b(Belgium|Belgian)\b': 'BE',
    r'\b(Czech Republic|Czech|Czechia)\b': 'CZ',
    r'\b(Denmark|Danish)\b': 'DK',
    r'\b(Finland|Finnish)\b': 'FI',
    r'\b(France|French)\b': 'FR',
    r'\b(Germany|German)\b': 'DE',
    r'\b(Hungary|Hungarian)\b': 'HU',
    r'\b(Ireland|Irish)\b': 'IE',
    r'\b(Italy|Italian)\b': 'IT',
    r'\b(Luxembourg)\b': 'LU',
    r'\b(Netherlands|Dutch|Holland)\b': 'NL',
    r'\b(Norway|Norwegian)\b': 'NO',
    r'\b(Poland|Polish)\b': 'PL',
    r'\b(Portugal|Portuguese)\b': 'PT',
    r'\b(Romania|Romanian)\b': 'RO',
    r'\b(Spain|Spanish)\b': 'ES',
    r'\b(Sweden|Swedish)\b': 'SE',
    r'\b(Switzerland|Swiss)\b': 'CH',
    r'\b(UK|United Kingdom|Britain|British)\b': 'GB',
    
    # Americas
    r'\b(US|USA|United States|America|American)\b': 'US',
    r'\b(Argentina|Argentine)\b': 'AR',
    r'\b(Brazil|Brazilian)\b': 'BR',
    r'\b(Canada|Canadian)\b': 'CA',
    r'\b(Chile|Chilean)\b': 'CL',
    r'\b(Colombia|Colombian)\b': 'CO',
    r'\b(Mexico|Mexican)\b': 'MX',
    r'\b(Peru|Peruvian)\b': 'PE',
    
    # Asia-Pacific
    r'\b(Australia|Australian)\b': 'AU',
    r'\b(China|Chinese|PRC)\b': 'CN',
    r'\b(Hong Kong|HK)\b': 'HK',
    r'\b(India|Indian)\b': 'IN',
    r'\b(Indonesia|Indonesian)\b': 'ID',
    r'\b(Israel|Israeli)\b': 'IL',
    r'\b(Japan|Japanese)\b': 'JP',
    r'\b(Malaysia|Malaysian)\b': 'MY',
    r'\b(New Zealand|NZ|Kiwi)\b': 'NZ',
    r'\b(Pakistan|Pakistani)\b': 'PK',
    r'\b(Philippines|Filipino)\b': 'PH',
    r'\b(Singapore|Singaporean)\b': 'SG',
    r'\b(South Korea|Korea|Korean)\b': 'KR',
    r'\b(Taiwan|Taiwanese)\b': 'TW',
    
    # Middle East & Africa
    r'\b(Egypt|Egyptian)\b': 'EG',
    r'\b(Kenya|Kenyan)\b': 'KE',
    r'\b(Nigeria|Nigerian)\b': 'NG',
    r'\b(Saudi Arabia|Saudi)\b': 'SA',
    r'\b(South Africa|South African)\b': 'ZA',
    r'\b(UAE|United Arab Emirates|Dubai|Emirati)\b': 'AE',
    
    # Regulation aliases → jurisdiction
    r'\b(GDPR)\b': 'EU',
    r'\b(AI Act|EU AI Act)\b': 'EU',
    r'\b(DSA|Digital Services Act)\b': 'EU',
    r'\b(DMA|Digital Markets Act)\b': 'EU',
    r'\b(CCPA)\b': 'US',
    r'\b(HIPAA)\b': 'US',
}


# ============================================================================
# DOCUMENT TYPE PATTERNS (for query parsing)
# ============================================================================

DOC_TYPE_PATTERNS = {
    # Regulatory sources → filters to DLA Piper docs
    r'\b(regulations?|regulatory|laws?|legal|legislation)\b': 'regulation',
    r'\b(acts?|directives?|statutes?|ordinances?)\b': 'regulation',
    r'\b(rules?|requirements?|compliance|mandatory)\b': 'regulation',
    r'\b(jurisdiction|government|official)\b': 'regulation',
    
    # Academic sources → filters to Scopus papers
    r'\b(papers?|articles?|publications?)\b': 'academic',
    r'\b(stud(?:y|ies)|research|researchers?)\b': 'academic',
    r'\b(journals?|proceedings?|conference)\b': 'academic',
    r'\b(authors?|scholars?|academics?|literature)\b': 'academic',
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

EVALUATION_CONFIG = {
    # RAGAS metrics
    'metrics': ['faithfulness', 'answer_relevancy', 'context_precision'],
    
    # Ablation modes
    'ablation_modes': [RetrievalMode.SEMANTIC, RetrievalMode.GRAPH, RetrievalMode.DUAL],
    
    # Test queries path
    'test_queries_path': 'data/evaluation/test_queries.json',
    'results_path': 'data/evaluation/results/',
}


# ============================================================================
# HELPER FUNCTIONS (for query parsing)
# ============================================================================

import re
from typing import List


def parse_jurisdictions(query: str) -> List[str]:
    """
    Extract jurisdiction hints from query using regex patterns.
    
    Args:
        query: User query string.
        
    Returns:
        List of jurisdiction codes (e.g., ['EU', 'US']).
    """
    jurisdictions = []
    for pattern, code in JURISDICTION_PATTERNS.items():
        if re.search(pattern, query, re.IGNORECASE):
            if code not in jurisdictions:
                jurisdictions.append(code)
    return jurisdictions


def parse_doc_types(query: str) -> List[str]:
    """
    Extract document type hints from query using regex patterns.
    
    Args:
        query: User query string.
        
    Returns:
        List of doc types (e.g., ['regulation', 'paper']).
    """
    doc_types = []
    for pattern, doc_type in DOC_TYPE_PATTERNS.items():
        if re.search(pattern, query, re.IGNORECASE):
            if doc_type not in doc_types:
                doc_types.append(doc_type)
    return doc_types