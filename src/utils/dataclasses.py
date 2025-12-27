# -*- coding: utf-8 -*-
"""
Core data structures for AI Governance GraphRAG Pipeline

Single source of truth for all pipeline data structures. Defines dataclasses for
chunks, entities, relations, and retrieval components used throughout the pipeline.
Import from this module rather than individual modules for consistency.

Examples:
# Import core data structures
    from src.utils.dataclasses import Entity, PreEntity, Relation, Chunk

    # Create entity
    entity = Entity(
        entity_id="ent_abc123",
        name="GDPR",
        type="Regulation",
        description="General Data Protection Regulation",
        chunk_ids=["chunk_001"]
    )

"""
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any
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
    Text chunk from document.
    
    Lists-first design for merged chunk provenance (Phase 1A deduplication).
    Convenience properties provide singular access for downstream compatibility.
    """
    chunk_ids: List[str]                    # Primary - supports merged chunks
    document_ids: List[str]                 # Primary - multi-doc provenance
    text: str
    position: int                           # 0-indexed position in document
    sentence_count: int
    token_count: int
    section_header: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def chunk_id(self) -> str:
        """Convenience - first chunk ID."""
        return self.chunk_ids[0] if self.chunk_ids else ""
    
    @property
    def document_id(self) -> str:
        """Convenience - first document ID."""
        return self.document_ids[0] if self.document_ids else ""
    
    @property
    def doc_type(self) -> str:
        """Infer document type from ID prefix."""
        doc_id = self.document_id
        if doc_id.startswith("reg_"):
            return "regulation"
        elif doc_id.startswith("paper_"):
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
    
    v1.2 Changes:
    - Removed domain field (now baked into type name: RegulatoryConcept, etc.)
    - Simplified embedding_text: "{name}({type})"
    """
    name: str
    type: str
    description: str
    chunk_id: str
    embedding_text: Optional[str] = None
    
    def compute_embedding_text(self) -> str:
        """Compute embedding text: '{name}({type})'."""
        return f"{self.name}({self.type})"


# ============================================================================
# PHASE 1C: CANONICAL ENTITIES (After Disambiguation)
# ============================================================================

@dataclass
class Entity:
    """
    Canonical entity after disambiguation (Phase 1C).
    
    Represents a unique real-world entity. Multiple PreEntities
    may have been merged into this single canonical form.
    
    v1.2: Removed domain field (now baked into type name).
    """
    entity_id: str                          # ent_<12-char-hash>
    name: str                               # Canonical name (best form)
    type: str                               # Entity type (domain-fused: RegulatoryConcept, etc.)
    description: str                        # Best description from merges
    chunk_ids: List[str]                    # All source chunks
    aliases: List[str] = field(default_factory=list)
    merge_count: int = 1


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
    resolved_entities: List[ResolvedEntity] = field(default_factory=list)
    resolved_entities: List[ResolvedEntity] = None