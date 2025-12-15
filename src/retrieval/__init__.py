# -*- coding: utf-8 -*-
"""
AI Governance GraphRAG Pipeline - Retrieval package.

Phase 3 retrieval pipeline for query understanding, graph expansion, and context
retrieval with dual-channel architecture.
"""

from .config import (
    # Enums
    RetrievalMode,
    # Data classes
    ExtractedEntity,
    ResolvedEntity,
    QueryFilters,
    ParsedQuery,
    Relation,
    Subgraph,  # FIXED: was GraphSubgraph
    Chunk,
    RankedChunk,
    RetrievalResult,
    # Configuration constants
    ENTITY_TYPES,
    PCST_CONFIG,
    RETRIEVAL_CONFIG,
    RANKING_CONFIG,
    ENTITY_RESOLUTION_CONFIG,
    # Utility functions
    parse_jurisdictions,
    parse_doc_types,
)

from .query_parser import QueryParser
from .entity_resolver import EntityResolver
from .graph_expander import GraphExpander
from .chunk_retriever import ChunkRetriever
from .result_ranker import ResultRanker
from .retrieval_processor import RetrievalProcessor, QueryUnderstanding

__all__ = [
    # Enums
    'RetrievalMode',
    # Data classes
    'ExtractedEntity',
    'ResolvedEntity',
    'QueryFilters',
    'ParsedQuery',
    'Relation',
    'Subgraph',  # FIXED: was GraphSubgraph
    'Chunk',
    'RankedChunk',
    'RetrievalResult',
    'QueryUnderstanding',
    # Components
    'QueryParser',
    'EntityResolver',
    'GraphExpander',
    'ChunkRetriever',
    'ResultRanker',
    'RetrievalProcessor',
    # Config
    'ENTITY_TYPES',
    'PCST_CONFIG',
    'RETRIEVAL_CONFIG',
    'RANKING_CONFIG',
    'ENTITY_RESOLUTION_CONFIG',
    'parse_jurisdictions',
    'parse_doc_types',
]