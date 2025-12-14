# -*- coding: utf-8 -*-
"""
Module: test_metrics.py
Package: tests.retrieval
Purpose: Comprehensive evaluation metrics for retrieval ablation study

Aligned with thesis objectives:
1. Factual Accuracy → EntityResolutionMetrics, RAGASMetrics
2. Query Relevance → RAGASMetrics, RetrievalMetrics  
3. Effective Use of Sources → GraphUtilizationMetrics, RetrievalMetrics

Author: Pau Barba i Colomer
Created: 2025-12-14
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ============================================================================
# ENTITY RESOLUTION METRICS (Objective 1: Factual Accuracy)
# ============================================================================

@dataclass
class EntityResolutionMetrics:
    """
    Measures entity pipeline quality.
    
    Key questions:
    - Did LLM extract correct entities from query?
    - How many resolved to actual graph entities?
    - What was the confidence?
    """
    extracted_count: int              # How many entities LLM extracted
    resolved_count: int               # How many matched to graph
    resolution_rate: float            # resolved / extracted
    avg_confidence: float             # Average FAISS match score
    entity_names: List[str]           # Resolved entity names (for inspection)
    match_types: Dict[str, int]       # {"exact": 2, "fuzzy": 3}


# ============================================================================
# GRAPH UTILIZATION METRICS (Objective 3: Effective Use of Sources)
# ============================================================================

@dataclass
class GraphUtilizationMetrics:
    """
    Measures how well graph structure was used.
    
    Key questions:
    - How much of the graph was activated?
    - Were cross-jurisdictional connections found?
    - How deep did traversal go?
    """
    entities_in_subgraph: int
    relations_in_subgraph: int
    relation_types: Dict[str, int]        # {"discusses": 5, "regulates": 2}
    jurisdictions_covered: List[str]      # For cross-jurisdictional queries


# ============================================================================
# RETRIEVAL METRICS (Objective 2 & 3: Relevance + Source Use)
# ============================================================================

@dataclass
class RetrievalMetrics:
    """
    Measures retrieval effectiveness.
    
    Key questions:
    - How many chunks retrieved per path?
    - What's the source diversity?
    - Which path contributed more?
    """
    total_chunks: int
    chunks_by_source: Dict[str, int]      # {"graphrag_relation": 5, "graphrag_entity": 3, "naive": 7}
    avg_chunk_score: float                # Mean ranking score
    source_diversity: Dict[str, int]      # {"regulation": 8, "paper": 7}
    jurisdiction_diversity: List[str]     # Jurisdictions represented in chunks


# ============================================================================
# RAGAS METRICS (Objective 1 & 2: Accuracy + Relevance)
# ============================================================================

@dataclass
class RAGASMetrics:
    """
    Answer quality from RAGAS framework.
    
    Key questions:
    - Are claims supported by context? (faithfulness)
    - Does answer address the query? (relevancy)
    """
    faithfulness_score: float
    faithfulness_details: Dict            # {supported_claims, total_claims, explanation}
    relevancy_score: float
    relevancy_explanation: str


# ============================================================================
# PERFORMANCE METRICS (Efficiency)
# ============================================================================

@dataclass
class PerformanceMetrics:
    """
    Cost and efficiency tracking.
    
    Key questions:
    - How expensive was this query?
    - Where's the time spent?
    """
    retrieval_time: float
    answer_time: float
    total_time: float
    answer_tokens: int
    cost_usd: float


# ============================================================================
# COMPLETE TEST RESULT
# ============================================================================

@dataclass
class TestResult:
    """
    Complete test result for one query × one mode.
    
    Captures all data needed for:
    - Per-query analysis
    - Mode comparison
    - Category-based evaluation
    - Results section tables/figures
    """
    # Metadata
    test_id: str                      # "q1_naive", "q1_graphrag", etc.
    query: str
    mode: str                         # "naive" | "graphrag" | "dual"
    category: str                     # Query type/complexity
    timestamp: str
    
    # Metrics (aligned with thesis objectives)
    entity_resolution: EntityResolutionMetrics
    graph_utilization: GraphUtilizationMetrics
    retrieval: RetrievalMetrics
    ragas: RAGASMetrics
    performance: PerformanceMetrics
    
    # Raw data for inspection
    answer_text: str
    success: bool
    error: Optional[str] = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_entity_resolution_metrics(
    extracted_entities: List,  # List[ExtractedEntity]
    resolved_entities: List    # List[ResolvedEntity]
) -> EntityResolutionMetrics:
    """
    Compute entity resolution metrics from retrieval result.
    
    Args:
        extracted_entities: Raw LLM extraction
        resolved_entities: After FAISS disambiguation
        
    Returns:
        EntityResolutionMetrics
    """
    extracted_count = len(extracted_entities)
    resolved_count = len(resolved_entities)
    
    resolution_rate = resolved_count / extracted_count if extracted_count > 0 else 0.0
    
    avg_confidence = (
        sum(e.confidence for e in resolved_entities) / resolved_count
        if resolved_count > 0 else 0.0
    )
    
    entity_names = [e.name for e in resolved_entities]
    
    match_types = {}
    for e in resolved_entities:
        match_types[e.match_type] = match_types.get(e.match_type, 0) + 1
    
    return EntityResolutionMetrics(
        extracted_count=extracted_count,
        resolved_count=resolved_count,
        resolution_rate=resolution_rate,
        avg_confidence=avg_confidence,
        entity_names=entity_names,
        match_types=match_types
    )


def compute_graph_utilization_metrics(subgraph) -> GraphUtilizationMetrics:
    """
    Compute graph utilization metrics from subgraph.
    
    Args:
        subgraph: Subgraph object with entities and relations
        
    Returns:
        GraphUtilizationMetrics
    """
    relation_types = {}
    for rel in subgraph.relations:
        relation_types[rel.predicate] = relation_types.get(rel.predicate, 0) + 1
    
    # Extract jurisdictions (would need chunk metadata)
    jurisdictions_covered = []  # TODO: Extract from chunks
    
    return GraphUtilizationMetrics(
        entities_in_subgraph=len(subgraph.entities),
        relations_in_subgraph=len(subgraph.relations),
        relation_types=relation_types,
        jurisdictions_covered=jurisdictions_covered
    )


def compute_retrieval_metrics(chunks: List) -> RetrievalMetrics:
    """
    Compute retrieval metrics from ranked chunks.
    
    Args:
        chunks: List[RankedChunk]
        
    Returns:
        RetrievalMetrics
    """
    total_chunks = len(chunks)
    
    # Count by source path
    chunks_by_source = {}
    for chunk in chunks:
        source = chunk.source_path
        chunks_by_source[source] = chunks_by_source.get(source, 0) + 1
    
    # Average score
    avg_chunk_score = sum(c.score for c in chunks) / total_chunks if total_chunks > 0 else 0.0
    
    # Source diversity (doc_type)
    source_diversity = {}
    for chunk in chunks:
        doc_type = chunk.doc_type
        source_diversity[doc_type] = source_diversity.get(doc_type, 0) + 1
    
    # Jurisdiction diversity
    jurisdictions = set()
    for chunk in chunks:
        if chunk.jurisdiction:
            jurisdictions.add(chunk.jurisdiction)
    
    return RetrievalMetrics(
        total_chunks=total_chunks,
        chunks_by_source=chunks_by_source,
        avg_chunk_score=avg_chunk_score,
        source_diversity=source_diversity,
        jurisdiction_diversity=list(jurisdictions)
    )