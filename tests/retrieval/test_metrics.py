# -*- coding: utf-8 -*-
"""
Comprehensive evaluation metrics for retrieval ablation study.

Provides metrics aligned with thesis objectives for factual accuracy, query relevance,
and effective use of sources. Includes entity resolution, graph utilization, and
RAGAS metrics for comprehensive retrieval system evaluation.

Example:
    metrics = EntityResolutionMetrics(extracted_count=5, resolved_count=4)
"""

# Standard library
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Third-party
import numpy as np


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
# COVERAGE METRICS (Objective 3: Information Utilization)
# ============================================================================

@dataclass
class CoverageMetrics:
    """
    Measures how well answer utilized retrieved information.
    
    Key questions:
    - Did answer use entities from the subgraph?
    - Were graph relations reflected in answer?
    - How much of retrieved context was utilized?
    """
    entities_in_subgraph: int
    entities_in_answer: int
    entity_coverage_rate: float       # in_answer / in_subgraph
    
    relations_in_subgraph: int
    relations_mentioned: int
    relation_coverage_rate: float     # mentioned / in_subgraph
    
    # Example entities that appeared in answer
    covered_entities: List[str] = field(default_factory=list)
    uncovered_entities: List[str] = field(default_factory=list)


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
    - How relevant are chunks to the query (pure semantic similarity)?
    """
    total_chunks: int
    chunks_by_source: Dict[str, int]      # {"graph_relation": 5, "graph_entity": 3, "semantic": 7}
    avg_chunk_score: float                # Mean ranking score
    avg_query_similarity: float           # Mean cosine sim(chunk_emb, query_emb) - pure relevance
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
    coverage: CoverageMetrics                   # NEW: Answer utilization
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


def compute_retrieval_metrics(
    chunks: List,
    query_embedding: np.ndarray,
    chunk_retriever
) -> RetrievalMetrics:
    """
    Compute retrieval metrics from ranked chunks.
    
    Args:
        chunks: List[RankedChunk]
        query_embedding: Query embedding for similarity computation
        chunk_retriever: ChunkRetriever instance (for FAISS index access)
        
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
    
    # Compute average query similarity (pure semantic relevance)
    avg_query_similarity = 0.0
    if total_chunks > 0:
        similarities = []
        for chunk in chunks:
            # Look up FAISS position for this chunk
            if chunk.chunk_id in chunk_retriever.chunk_id_map:
                faiss_idx = chunk_retriever.chunk_id_map[chunk.chunk_id]
                # Get embedding from FAISS index
                chunk_emb = chunk_retriever.faiss_index.reconstruct(int(faiss_idx))
                # Compute cosine similarity with query
                similarity = np.dot(query_embedding, chunk_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb)
                )
                similarities.append(float(similarity))
        
        avg_query_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    
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
        avg_query_similarity=avg_query_similarity,
        source_diversity=source_diversity,
        jurisdiction_diversity=list(jurisdictions)
    )


def compute_coverage_metrics(
    subgraph,
    answer_text: str,
    resolved_entities: List = None,  # NEW: Actual entity names
    client=None  # Anthropic client for LLM extraction
) -> CoverageMetrics:
    """
    Compute coverage metrics: how well did answer utilize retrieved information?
    
    Args:
        subgraph: Subgraph object with entities and relations
        answer_text: Generated answer text
        resolved_entities: List of ResolvedEntity objects with actual names
        client: Optional Anthropic client for entity extraction from answer
        
    Returns:
        CoverageMetrics
    """
    # Entities in subgraph (what was available)
    entities_in_subgraph = len(subgraph.entities)
    relations_in_subgraph = len(subgraph.relations)
    
    # Build entity name lookup from resolved_entities
    entity_names_map = {}
    if resolved_entities:
        for entity in resolved_entities:
            # Map entity_id to name
            entity_names_map[entity.entity_id] = entity.name
    
    # Extract entity names from subgraph for matching
    subgraph_entity_names = set()
    for entity_id in subgraph.entities:
        if entity_id in entity_names_map:
            # Use actual entity name
            name = entity_names_map[entity_id]
            subgraph_entity_names.add(name.lower())
        else:
            # Fallback: use ID (shouldn't happen with resolved_entities)
            subgraph_entity_names.add(entity_id.lower())
    
    # Simple heuristic: check if entity names appear in answer
    # (Better approach: use LLM to extract entities from answer)
    answer_lower = answer_text.lower()
    
    covered_entities = []
    uncovered_entities = []
    
    for entity_name in subgraph_entity_names:
        if entity_name in answer_lower:
            covered_entities.append(entity_name)
        else:
            uncovered_entities.append(entity_name)
    
    entities_in_answer = len(covered_entities)
    entity_coverage_rate = entities_in_answer / entities_in_subgraph if entities_in_subgraph > 0 else 0.0
    
    # Relation coverage: check if relation predicates appear in answer
    relation_predicates = set()
    relations_mentioned = 0
    
    for rel in subgraph.relations:
        predicate_lower = rel.predicate.lower().replace('_', ' ')
        if predicate_lower not in relation_predicates:
            relation_predicates.add(predicate_lower)
            # Check if both source and target entities appear near this predicate
            if (rel.source_name.lower() in answer_lower and 
                rel.target_name.lower() in answer_lower):
                relations_mentioned += 1
    
    relation_coverage_rate = relations_mentioned / relations_in_subgraph if relations_in_subgraph > 0 else 0.0
    
    return CoverageMetrics(
        entities_in_subgraph=entities_in_subgraph,
        entities_in_answer=entities_in_answer,
        entity_coverage_rate=entity_coverage_rate,
        relations_in_subgraph=relations_in_subgraph,
        relations_mentioned=relations_mentioned,
        relation_coverage_rate=relation_coverage_rate,
        covered_entities=covered_entities[:10],  # Limit to 10 examples
        uncovered_entities=uncovered_entities[:10]
    )