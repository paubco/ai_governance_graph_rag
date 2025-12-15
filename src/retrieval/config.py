# -*- coding: utf-8 -*-
"""
Module: config.py
Package: src.retrieval
Purpose: Data classes and configuration for Phase 3 retrieval pipeline

MODIFIED: Extended RetrievalResult with entity metadata for evaluation/testing

Author: Pau Barba i Colomer
Created: 2025-12-07
Modified: 2025-12-14 (evaluation extension + config merge)

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
    """
    Text chunk from Neo4j.
    
    MODIFIED: Added score field (default 0.5) for base scoring before ranking.
    """
    chunk_id: str
    text: str
    doc_id: str
    doc_type: str  # 'regulation' or 'paper'
    jurisdiction: Optional[str] = None  # For regulations
    score: float = 0.5  # Base score (FAISS similarity or entity match score)
    metadata: dict = field(default_factory=dict)


@dataclass
class RankedChunk:
    """
    Chunk after ranking with scoring metadata.
    
    Used for:
    - Final top-K selection
    - Prompt assembly
    - Ablation analysis (source_path tracking)
    
    Source paths:
    - graph_relation: Retrieved via graph traversal + contains PCST relation (provenance)
    - graph_entity: Retrieved via graph traversal (entity-centric)
    - semantic: Retrieved via vector similarity search (FAISS)
    """
    chunk_id: str
    text: str
    score: float
    source_path: Literal["graph_relation", "graph_entity", "semantic"] = "semantic"
    retrieval_method: str = ""  
    doc_id: str = "" 
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
    
    MODIFIED: Added extracted_entities and resolved_entities for evaluation/testing.
    Not used in production prompt assembly, but critical for ablation studies.
    """
    query: str
    chunks: List[RankedChunk]
    subgraph: Subgraph
    
    # Evaluation metadata (added for Phase 3 testing)
    extracted_entities: List[ExtractedEntity] = field(default_factory=list)  # Raw LLM extraction
    resolved_entities: List[ResolvedEntity] = field(default_factory=list)    # After FAISS disambiguation


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# PCST Graph Expansion
PCST_CONFIG = {
    'k_candidates': 5,         # Top-K similar entities per seed (FAISS) - reduced for performance
    'delta': 0.5,              # Neo4j PCST prize-cost balance parameter
    'prize_strategy': 'uniform',  # 'uniform' or 'frequency' (start simple)
    'cost_strategy': 'uniform',   # 'uniform', 'frequency', or 'similarity'
    'max_entities': 50,        # Global cap on expanded entities
}

# Chunk Retrieval (MERGED from both sections)
RETRIEVAL_CONFIG = {
    'graph_all_entities': True,      # Get all chunks from PCST entities (not top-K)
    'semantic_top_k': 15,             # Semantic search chunk limit
    'chunk_token_limit': 8000,        # Approximate token budget for context
    'entity_resolution_top_k': 3,     # Max entities per extracted mention
    'pcst_max_entities': 50,          # PCST expansion limit (hub node control)
}

# Ranking & Scoring (MERGED + FIXED with entity_coverage_bonus)
RANKING_CONFIG = {
    'provenance_bonus': 0.3,          # Chunks containing PCST relations (highest)
    'graph_bonus': 0.2,               # Chunks from graph entity expansion (medium)
    'semantic_baseline': 0.0,         # Semantic search chunks (baseline)
    'jurisdiction_boost': 0.1,        # Bonus if chunk matches jurisdiction hint
    'doc_type_boost': 0.15,           # Bonus if chunk matches doc_type hint
    'entity_coverage_bonus': 0.2,     # FIXED: Bonus based on entity coverage
    'final_top_k': 20,                # Final chunks for LLM context
}

# Entity Resolution (from Phase 3.3.1)
ENTITY_RESOLUTION_CONFIG = {
    'fuzzy_threshold': 0.75,      # Cosine similarity threshold
    'top_k_per_entity': 10,       # Candidates per query entity
}

# Answer Generation Configuration (Phase 3.3.4)
ANSWER_GENERATION_CONFIG = {
    # LLM provider and model
    'provider': 'anthropic',           # 'anthropic' or 'together'
    'model': 'claude-3-5-haiku-20241022',  # Claude Haiku (fast + cheap)
    
    # Token budgets (for 200k context)
    'max_input_tokens': 15000,         # Total input budget
    'token_budget': {
        'system_prompt': 1000,
        'graph_structure': 500,
        'entity_context': 500,
        'source_chunks': 10000,        # Most of budget goes here
        'instructions': 500,
        'buffer': 1000,
    },
    
    # Generation parameters
    'max_output_tokens': 2000,         # Answer length limit
    'temperature': 0.0,                # Deterministic for evaluation (was 0.3)
    'top_p': 0.9,
    
    # Formatting
    'max_chunks_to_format': 40,        # Max chunks even if more retrieved
    'truncate_chunk_chars': 1000,      # Max chars per chunk in prompt
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
    'temperature': 0.0,  # Deterministic for evaluation (was 0.3)
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