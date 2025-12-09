# -*- coding: utf-8 -*-
"""
Module: config.py
Package: src.retrieval
Purpose: Data classes and configuration for Phase 3 retrieval pipeline

Author: Pau Barba i Colomer
Created: 2025-12-07
Modified: 2025-12-07

References:
    - PHASE_3_DESIGN.md § 5 (Implementation)
    - PHASE_3.3.2_OPEN_QUESTIONS.md (Graph expansion design decisions)
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
    
    NAIVE: Path B only (semantic FAISS search)
    GRAPHRAG: Path A only (entity-centric with PCST)
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
    query_embedding: np.ndarray  # For Path B semantic search


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
    chunk_ids: List[str] = field(default_factory=list)  # Where this relation was extracted


@dataclass
class Subgraph:
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
    jurisdiction: Optional[str] = None  # For regulations
    metadata: dict = field(default_factory=dict)


@dataclass
class RankedChunk:
    """
    Chunk after ranking with scoring metadata.
    
    Used for:
    - Final top-K selection
    - Prompt assembly
    - Ablation analysis (source_path tracking)
    """
    chunk_id: str
    text: str
    score: float
    source_path: Literal["graphrag_relation", "graphrag_entity", "naive"]
    entities: List[str] = field(default_factory=list)  # Which entities led here
    doc_type: str = ""
    jurisdiction: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """
    Complete retrieval output.
    
    Contains both chunks and subgraph for prompt assembly.
    Subgraph relations will be formatted as GRAPH STRUCTURE section.
    """
    chunks: List[RankedChunk]
    subgraph: Subgraph  # Pass through for prompt builder
    query: str


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# PCST Graph Expansion
PCST_CONFIG = {
    'k_candidates': 10,        # Top-K similar entities per seed (FAISS)
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
}

# Ranking & Scoring
RANKING_CONFIG = {
    'provenance_bonus': 0.3,      # Chunks containing PCST relations (highest)
    'path_a_bonus': 0.2,          # Chunks from entity expansion (medium)
    'path_b_baseline': 0.0,       # Semantic search chunks (baseline)
    'jurisdiction_boost': 0.1,    # Bonus if chunk matches jurisdiction hint
    'final_top_k': 20,            # Final chunks for LLM context
}

# Entity Resolution (from Phase 3.3.1)
ENTITY_RESOLUTION_CONFIG = {
    'fuzzy_threshold': 0.75,      # Cosine similarity threshold
    'top_k_per_entity': 10,       # Candidates per query entity
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