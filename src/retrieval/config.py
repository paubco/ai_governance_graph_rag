# -*- coding: utf-8 -*-
"""
Module: config.py
Package: src.retrieval
Purpose: Data classes and configuration for Phase 3 retrieval pipeline

Author: Pau Barba i Colomer
Created: 2025-12-07
Modified: 2025-12-12

References:
    - PHASE_3_DESIGN.md § 5 (Implementation)
    - PHASE_3.3.2_OPEN_QUESTIONS.md (Graph expansion design decisions)
    - RAKG paper § 4.2 (Entity Coverage metric)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from enum import Enum
import re
import numpy as np


# ============================================================================
# RETRIEVAL MODE (for testing/evaluation)
# ============================================================================

class RetrievalMode(Enum):
    """
    Retrieval strategy modes for ablation studies.
    
    NAIVE: Semantic FAISS search only
    GRAPHRAG: Entity-centric with PCST only
    DUAL: Both paths merged (default)
    """
    NAIVE = "naive"
    GRAPHRAG = "graphrag"
    DUAL = "dual"


# ============================================================================
# ENTITY TYPES (from Phase 1 actual data)
# ============================================================================

ENTITY_TYPES = [
    "Concept", "Author", "Journal", "Technology", "Article", "Organization",
    "Process", "Publication", "Document", "Book", "Institution", "Person",
    "Regulatory Concept", "Regulation", "Paper", "Legal Concept", "Methodology",
    "Title", "Regulatory Document", "Technical Term", "Role", "Location",
    "Metric", "Conference"
]


# ============================================================================
# QUERY UNDERSTANDING (Phase 3.3.1)
# ============================================================================

@dataclass
class ExtractedEntity:
    """Entity extracted from query by LLM."""
    name: str
    type: str


@dataclass
class QueryFilters:
    """
    Hints for ranking (NOT hard filters).
    
    Note: Jurisdictions/doc_types are soft hints that boost scores,
    not hard constraints. This enables cross-jurisdictional comparison.
    """
    jurisdiction_hints: Optional[List[str]] = None
    doc_type_hints: Optional[List[str]] = None


@dataclass
class ParsedQuery:
    """Output of query parsing (Phase 3.3.1)."""
    raw_query: str
    extracted_entities: List[ExtractedEntity]
    filters: QueryFilters
    query_embedding: np.ndarray  # For semantic search


@dataclass
class ResolvedEntity:
    """Entity after FAISS resolution."""
    entity_id: str
    name: str
    type: str
    confidence: float  # Cosine similarity score
    match_type: Literal["exact", "fuzzy"]


# ============================================================================
# GRAPH EXPANSION (Phase 3.3.2a - PCST)
# ============================================================================

@dataclass
class Relation:
    """
    Relation from PCST subgraph.
    
    Used for:
    1. LLM context (formatted in prompt as structured knowledge)
    2. Provenance tracking (chunk_ids point to where relation was extracted)
    3. Answer grounding (citation trail)
    """
    source_id: str
    source_name: str
    predicate: str
    target_id: str
    target_name: str
    confidence: float = 0.0
    chunk_ids: List[str] = field(default_factory=list)  # Where this relation was extracted


@dataclass
class GraphSubgraph:
    """
    Output of PCST graph expansion.
    
    Contains:
    - entities: For corpus retrospective (chunk retrieval)
    - relations: For LLM context + provenance tracking
    """
    entities: List[str]  # Entity IDs from PCST
    relations: List[Relation]  # Edge structure from PCST


# ============================================================================
# CHUNK RETRIEVAL (Phase 3.3.2b)
# ============================================================================

@dataclass
class Chunk:
    """Text chunk from Neo4j."""
    chunk_id: str
    text: str
    doc_id: str
    doc_type: str  # 'regulation' or 'paper'
    score: float = 0.5  # Base score from entity matches or FAISS
    jurisdiction: Optional[str] = None  # For regulations
    metadata: dict = field(default_factory=dict)


@dataclass
class RankedChunk:
    """
    Chunk after ranking with scoring metadata.
    
    Used for:
    - Final top-K selection
    - Prompt assembly
    - Ablation analysis (retrieval_method tracking)
    """
    chunk_id: str
    text: str
    score: float
    retrieval_method: Literal["graphrag", "naive"]
    doc_id: str = ""
    doc_type: str = ""
    jurisdiction: Optional[str] = None
    entities: List[str] = field(default_factory=list)  # Which entities led here
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """
    Complete retrieval output.
    
    Contains both chunks and subgraph for prompt assembly.
    Subgraph relations will be formatted as GRAPH STRUCTURE section.
    """
    query: str
    resolved_entities: List[str]
    subgraph: GraphSubgraph
    chunks: List[RankedChunk]


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# PCST Graph Expansion
PCST_CONFIG = {
    'k_candidates': 10,        # Top-K similar entities per seed (FAISS) - increased for path coverage
    'delta': 0.5,              # Neo4j PCST prize-cost balance parameter
    'prize_strategy': 'uniform',  # 'uniform' or 'frequency' (start simple)
    'cost_strategy': 'uniform',   # 'uniform', 'frequency', or 'similarity'
    'max_entities': 50,        # Global cap on expanded entities
}

# Chunk Retrieval
RETRIEVAL_CONFIG = {
    'path_a_all_entities': True,    # Get all chunks from PCST entities (not top-K)
    'path_b_top_k': 15,              # Semantic search chunk limit
    'chunk_token_limit': 8000,       # Approximate token budget for context
    'entity_resolution_top_k': 3,    # Max entities per extracted mention (avoid explosion)
    'pcst_max_entities': 50,         # PCST expansion limit (hub node control)
}

# Ranking & Scoring (Entity Coverage-Based)
RANKING_CONFIG = {
    # GraphRAG scoring components
    'entity_coverage_bonus': 0.40,    # Max bonus for 100% entity coverage
    'provenance_bonus': 0.15,         # Flat bonus if chunk contains PCST relation
    
    # Naive scoring: Pure FAISS similarity (no bonuses)
    
    # Final selection
    'final_top_k': 20,                # Final chunks for LLM context
}

# RANKING RATIONALE (Entity Coverage):
# 
# GraphRAG chunks scored by entity coverage:
#   score = base_score + (entities_in_chunk / total_resolved * 0.40) + provenance_bonus
# 
# This penalizes chunks mentioning random entities without full context.
# 
# Examples with 4 resolved entities [A, B, C, D]:
# 
#   Chunk with [A]:          base 0.45 + (1/4 * 0.40) = 0.55
#   Chunk with [A, B]:       base 0.50 + (2/4 * 0.40) = 0.70
#   Chunk with [A,B,C,D]:    base 0.60 + (4/4 * 0.40) = 1.00
#   Above + PCST relation:   base 0.60 + 0.40 + 0.15  = 1.15
# 
# Naive chunks: Pure FAISS similarity (0.0-1.0), no bonuses
# 
# Creates fair competition where both paths can reach similar max scores.
# Based on RAKG's Entity Coverage (EC) metric from paper § 4.2.

# Entity Resolution (from Phase 3.3.1)
ENTITY_RESOLUTION_CONFIG = {
    'fuzzy_threshold': 0.75,      # Cosine similarity threshold
    'top_k_per_entity': 10,       # Candidates per query entity
}


# ============================================================================
# ANSWER GENERATION CONFIGURATION (Phase 3.3.4)
# ============================================================================

ANSWER_GENERATION_CONFIG = {
    # LLM provider and model
    'provider': 'anthropic',           # 'anthropic' or 'together'
    'model': 'claude-3-5-haiku-20241022',  # Claude Haiku (fast + cheap)
    
    # Generation parameters
    'max_output_tokens': 2000,         # Answer length limit
    'temperature': 0.0,                # Deterministic for evaluation
    'top_p': 1.0,
    
    # Formatting
    'max_chunks_to_format': 20,        # Max chunks even if more retrieved
    'truncate_chunk_chars': 1000,      # Max chars per chunk in prompt
    
    # Token budget breakdown (for Claude's 200K context)
    'token_budget': {
        'graph_structure': 500,        # Relations formatted as bullets
        'entity_context': 500,         # Key entity list
        'source_chunks': 10000,        # ~40 chunks at 250 tokens each
    }
}


# Alternative: Mistral configuration (8k context)
ANSWER_GENERATION_CONFIG_MISTRAL = {
    'provider': 'together',
    'model': 'mistralai/Mistral-7B-Instruct-v0.3',
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
    'temperature': 0.3,
    'top_p': 0.9,
    'max_chunks_to_format': 10,
    'truncate_chunk_chars': 400,
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_jurisdictions(query: str) -> List[str]:
    """
    Extract jurisdiction hints from query.
    
    Examples:
        "EU regulations" -> ["EU"]
        "GDPR in California" -> ["EU", "US-CA"]
    """
    jurisdictions = []
    
    # Common jurisdiction patterns
    patterns = {
        r'\b(EU|European Union)\b': 'EU',
        r'\b(US|USA|United States)\b': 'US',
        r'\b(UK|United Kingdom)\b': 'UK',
        r'\b(GDPR)\b': 'EU',  # GDPR implies EU
        r'\bCalifornia\b': 'US-CA',
        r'\bNew York\b': 'US-NY',
    }
    
    for pattern, code in patterns.items():
        if re.search(pattern, query, re.IGNORECASE):
            jurisdictions.append(code)
    
    return list(set(jurisdictions))  # Deduplicate


def parse_doc_types(query: str) -> List[str]:
    """
    Extract document type hints from query.
    
    Examples:
        "regulations about" -> ["regulation"]
        "papers on" -> ["paper"]
    """
    doc_types = []
    
    # Document type patterns (match singular and plural)
    if re.search(r'\b(regulations?|laws?|acts?|directives?)\b', query, re.IGNORECASE):
        doc_types.append('regulation')
    
    if re.search(r'\b(papers?|articles?|stud(y|ies)|research)\b', query, re.IGNORECASE):
        doc_types.append('paper')
    
    return doc_types