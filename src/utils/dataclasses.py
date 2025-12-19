# -*- coding: utf-8 -*-
"""
Core data structures for AI Governance GraphRAG Pipeline.

Single source of truth for all pipeline data structures. Import from here,
not from individual modules.

v1.1 Changes:
- Simplified PreEntity (flat, no nesting, no confidence)
- Added aliases field to Entity
- Standardized Relation to use IDs not names
- Added merge support to Chunk (chunk_ids, document_ids lists)

Example:
    from src.utils.dataclasses import Entity, PreEntity, Relation, Chunk
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any, Union
from enum import Enum
import numpy as np


# ============================================================================
# ENUMS
# ============================================================================

class DocType(Enum):
    """Document source types."""
    REGULATION = "regulation"
    PAPER = "paper"


class ExtractionStrategy(Enum):
    """Relation extraction strategies (Phase 1D)."""
    ACADEMIC = "academic"    # Constrained predicates for citations
    SEMANTIC = "semantic"    # OpenIE for domain entities


class RetrievalMode(Enum):
    """Retrieval strategy modes for ablation studies."""
    SEMANTIC = "semantic"
    GRAPH = "graph"
    DUAL = "dual"


# ============================================================================
# PHASE 1A: CHUNKS
# ============================================================================

@dataclass
class Chunk:
    """
    Text chunk from document, with optional merge provenance.
    
    Single chunk:
        chunk_id = "reg_EU_CHUNK_0042"
        document_id = "reg_EU"
        chunk_ids = None (or empty)
        document_ids = None (or empty)
    
    Merged chunk (duplicates across jurisdictions):
        chunk_id = "reg_AT_CHUNK_0000"  (canonical - first seen)
        document_id = "reg_AT"           (canonical)
        chunk_ids = ["reg_AT_CHUNK_0000", "reg_BE_CHUNK_0000", "reg_BG_CHUNK_0000"]
        document_ids = ["reg_AT", "reg_BE", "reg_BG"]
    """
    chunk_id: str                           # Canonical ID
    document_id: str                        # Canonical doc ID
    text: str
    position: int                           # 0-indexed position in document
    sentence_count: int
    token_count: int
    section_header: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Merge provenance (populated for deduplicated chunks)
    chunk_ids: List[str] = field(default_factory=list)       # All merged chunk IDs
    document_ids: List[str] = field(default_factory=list)    # All merged doc IDs
    
    @property
    def is_merged(self) -> bool:
        """Check if this chunk represents merged duplicates."""
        return len(self.chunk_ids) > 1
    
    @property
    def merge_count(self) -> int:
        """Number of chunks merged (1 if not merged)."""
        return len(self.chunk_ids) if self.chunk_ids else 1
    
    @property
    def doc_type(self) -> str:
        """Infer document type from ID prefix."""
        if self.document_id.startswith("reg_"):
            return "regulation"
        elif self.document_id.startswith("paper_"):
            return "paper"
        return "unknown"
    
    @property
    def jurisdiction(self) -> Optional[str]:
        """Extract jurisdiction code from regulation document ID."""
        if self.doc_type == "regulation":
            parts = self.document_id.split("_")
            if len(parts) >= 2:
                return parts[1]
        return None
    
    @property
    def jurisdictions(self) -> List[str]:
        """Get all jurisdictions for merged regulation chunks."""
        if not self.document_ids:
            j = self.jurisdiction
            return [j] if j else []
        
        jurisdictions = []
        for doc_id in self.document_ids:
            if doc_id.startswith("reg_"):
                parts = doc_id.split("_")
                if len(parts) >= 2:
                    jurisdictions.append(parts[1])
        return jurisdictions


@dataclass
class EmbeddedChunk(Chunk):
    """Chunk with embedding vector for FAISS indexing."""
    embedding: Optional[np.ndarray] = None  # 1024-dim BGE-M3


# ============================================================================
# PHASE 1B: PRE-ENTITIES (Raw Extraction)
# ============================================================================

@dataclass
class PreEntity:
    """
    Raw entity extraction from LLM (Phase 1B).
    
    Flat structure - one record per extraction. Multiple PreEntities
    may refer to the same real-world entity (resolved in Phase 1C).
    
    v1.1: Simplified - no nesting, no confidence score.
    """
    name: str
    type: str
    description: str
    chunk_id: str


# ============================================================================
# PHASE 1C: CANONICAL ENTITIES (After Disambiguation)
# ============================================================================

@dataclass
class Entity:
    """
    Canonical entity after disambiguation (Phase 1C).
    
    Represents a unique real-world entity. Multiple PreEntities
    may have been merged into this single canonical form.
    
    v1.1: Added aliases field for surface form mapping.
    """
    entity_id: str                          # ent_<12-char-hash>
    name: str                               # Canonical name (best form)
    type: str                               # Entity type
    description: str                        # Best description from merges
    chunk_ids: List[str]                    # All source chunks
    aliases: List[str] = field(default_factory=list)  # Surface forms
    merge_count: int = 1                    # PreEntities merged into this


@dataclass 
class EmbeddedEntity(Entity):
    """Entity with embedding vector for FAISS indexing."""
    embedding: Optional[np.ndarray] = None  # 1024-dim BGE-M3


# ============================================================================
# PHASE 1D: RELATIONS
# ============================================================================

@dataclass
class Relation:
    """
    Extracted relation triplet (Phase 1D).
    
    Links two canonical entities via a predicate.
    Uses entity IDs (not names) for reliable linking.
    """
    subject_id: str
    predicate: str
    object_id: str
    chunk_ids: List[str]                    # Provenance
    extraction_strategy: str = "semantic"   # "academic" or "semantic"


# ============================================================================
# PHASE 3: RETRIEVAL
# ============================================================================

@dataclass
class ExtractedQueryEntity:
    """Entity extracted from user query by LLM."""
    name: str
    type: str


@dataclass
class ResolvedEntity:
    """Entity after FAISS resolution from query."""
    entity_id: str
    name: str
    type: str
    confidence: float                       # Cosine similarity
    match_type: Literal["exact", "fuzzy"]
    query_mention: str = ""                 # Original text that matched


@dataclass
class QueryFilters:
    """Soft hints for retrieval ranking (not hard filters)."""
    jurisdiction_hints: List[str] = field(default_factory=list)
    doc_type_hints: List[str] = field(default_factory=list)


@dataclass
class ParsedQuery:
    """Output of query parsing (Phase 3.3.1)."""
    raw_query: str
    extracted_entities: List[ExtractedQueryEntity]
    resolved_entities: List[ResolvedEntity]
    filters: QueryFilters
    embedding: Optional[np.ndarray] = None


@dataclass
class Subgraph:
    """Output of Steiner Tree graph expansion."""
    entity_ids: List[str]
    relations: List[Relation]


@dataclass
class RankedChunk:
    """Chunk after retrieval ranking with scoring metadata."""
    chunk_id: str
    text: str
    score: float
    source_path: Literal["graph_provenance", "graph_entity", "semantic"]
    doc_id: str
    doc_type: str
    jurisdiction: Optional[str] = None
    matching_entities: List[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """Complete retrieval output for a query."""
    query: str
    chunks: List[RankedChunk]
    subgraph: Subgraph
    parsed_query: Optional[ParsedQuery] = None