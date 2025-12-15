# v1.1 Implementation Plan

**Goal**: Full corpus reprocess with improved entity quality, structural relations, and pipeline consolidation  
**Timeline**: ~1 week  
**Scope**: Phases 1A through 2B (retrieval unchanged)

---

## Overview

### Why Reprocess?

v1.0 entities have critical issues:
- No alias tracking (lost surface forms during disambiguation)
- No structural relations (Article 5 ↔ EU AI Act disconnected)
- Inconsistent types ("Regulatory Requirement" vs "Legislation")
- Garbage chunks reaching extraction (OCR artifacts, boilerplate)
- Late ID generation (unstable identifiers through pipeline)
- Non-English content not translated (Spanish, Turkish docs skipped or garbled)

### Key Changes from v1.0

| Aspect | v1.0 | v1.1 |
|--------|------|------|
| LLM | Qwen-72B (1B), Qwen-7B (1C), Mistral-7B (1D) | Mistral-7B throughout |
| Schema enforcement | Regex JSON parsing | Pydantic + LLM JSON mode |
| Entity IDs | Generated post-disambiguation | Hash at extraction (Phase 1B) |
| Structural relations | None | PART_OF (pattern-based), SAME_AS (LLM) |
| Alias tracking | None | Preserved through disambiguation |
| Pre-processing | None | Translation + ftfy + datatrove quality |
| Chunk quality | None | datatrove Gopher heuristics |
| Type vocabulary | Freeform | Canonical 9 types enforced |
| File pattern | Tripartite (core/processor/runner) | Single processor per phase |
| Orchestration | Manual phase-by-phase | Single `run_pipeline.py` |
| Translation | None | `deep-translator` for ES/TR (fallback: argos-translate) |

---

## 1. Shared Infrastructure

### 1.1 Pydantic Models

Define once, use everywhere. Schema enforced at LLM output and file I/O.

**Location**: `src/models/`

```
src/models/
├── __init__.py
├── chunk.py          # Chunk, ChunkMetadata
├── entity.py         # PreEntity, Entity, StructuralRelation
├── relation.py       # Relation, RelationProvenance
└── query.py          # ParsedQuery, RetrievalResult (exists, may need updates)
```

**Core models**:

```python
# chunk.py
class ChunkMetadata(BaseModel):
    doc_id: str
    doc_type: Literal["regulation", "academic_paper"]
    jurisdiction: str | None = None
    scopus_id: str | None = None
    section_title: str | None = None

class Chunk(BaseModel):
    chunk_id: str
    text: str
    embedding: list[float] | None = None
    metadata: ChunkMetadata
    quality_score: float | None = None  # NEW: from quality gate

# entity.py
class StructuralRelation(BaseModel):
    relation_type: Literal["PART_OF", "SAME_AS"]
    target_name: str

class PreEntity(BaseModel):
    pre_entity_id: str              # NEW: hash(name + type + chunk_id)
    name: str
    type: str                        # Enforced canonical types
    description: str
    chunk_id: str
    structural_relations: list[StructuralRelation] = []

class Entity(BaseModel):
    entity_id: str                   # Deterministic hash(name + type)
    name: str
    type: str
    description: str
    aliases: list[str] = []          # NEW: surface forms that merged here
    alias_ids: list[str] = []        # NEW: pre_entity_ids that merged here
    embedding: list[float] | None = None
    chunk_ids: list[str] = []        # Provenance

# relation.py  
class Relation(BaseModel):
    subject_id: str
    predicate: str
    object_id: str
    chunk_id: str                    # Provenance
    confidence: float = 1.0
```

**Canonical entity types** (enforced in prompts and validation):

```python
CANONICAL_TYPES = [
    "Concept",
    "Organization",
    "Technology",
    "Regulation",
    "Person",
    "Country",
    "Citation",
    "Author",
    "Journal"
]
```

### 1.2 Consolidated Utils

**Location**: `src/utils/`

| File | Purpose | Consolidates |
|------|---------|--------------|
| `llm.py` | LLM client with retry, JSON schema enforcement | Scattered Together.ai setup |
| `io.py` | `load_json`, `load_jsonl`, `save_json`, `save_jsonl` | Repeated patterns |
| `batch.py` | `BatchProcessor` base class with progress, checkpoints | Duplicated ThreadPoolExecutor |
| `embedder.py` | Keep as-is | Already consolidated |
| `checkpoint.py` | Keep as-is | Already exists |

**`llm.py` key interface**:

```python
class LLMClient:
    def __init__(self, model: str = "mistral-7b"):
        ...
    
    def prompt(self, messages: list[dict], temperature: float = 0.3) -> str:
        """Raw text completion with retry."""
        ...
    
    def prompt_json(self, messages: list[dict], schema: type[BaseModel]) -> BaseModel:
        """Schema-enforced JSON completion. Returns validated Pydantic model."""
        ...
```

### 1.3 Centralized Config

**Location**: `src/config/`

```
src/config/
├── __init__.py
├── base.py           # Paths, model names, shared constants
├── extraction.py     # Phase 1B params (prompts, types)
├── disambiguation.py # Phase 1C params (thresholds)
├── relations.py      # Phase 1D params (MMR, sampling)
└── retrieval.py      # Phase 3 params (exists, keep)
```

**`base.py`**:

```python
# Paths
DATA_RAW = Path("data/raw")
DATA_INTERIM = Path("data/interim")
DATA_PROCESSED = Path("data/processed")

# Models
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
EMBEDDING_MODEL = "BAAI/bge-m3"

# Canonical types
CANONICAL_TYPES = ["Concept", "Organization", ...]
```

---

## 2. Pipeline Orchestration

**Location**: `scripts/run_pipeline.py`

Single entry point for full pipeline or phase ranges.

### Interface

```bash
# Full pipeline
python scripts/run_pipeline.py --start 1a --end 2b

# Single phase
python scripts/run_pipeline.py --phase 1b

# Range (inclusive both ends)
python scripts/run_pipeline.py --start 1c --end 1d

# With options
python scripts/run_pipeline.py --phase 1b --workers 10 --checkpoint resume
```

### Phase Registry

```python
PHASES = {
    "1a": ("Phase 1A: Chunking", chunking_processor.run),
    "1b": ("Phase 1B: Entity Extraction", entity_processor.run),
    "1c": ("Phase 1C: Disambiguation", disambiguation_processor.run),
    "1d": ("Phase 1D: Relations", relation_processor.run),
    "2a": ("Phase 2A: Enrichment", enrichment_processor.run),
    "2b": ("Phase 2B: Import", import_processor.run),
}

PHASE_ORDER = ["1a", "1b", "1c", "1d", "2a", "2b"]
```

### Data Flow Validation

Before each phase, validate inputs exist:

```python
PHASE_INPUTS = {
    "1a": ["data/raw/dlapiper/*.json", "data/raw/academic/*/full.md"],
    "1b": ["data/interim/chunks/chunks_embedded.json"],
    "1c": ["data/interim/entities/pre_entities.json"],
    "1d": ["data/interim/entities/normalized_entities.json"],
    "2a": ["data/interim/relations/relations_normalized.json"],
    "2b": ["data/processed/faiss/*.index"],
}
```

---

## 3. Phase 1A: Chunking

**File**: `src/processing/chunking/chunking_processor.py`

**Input**: `data/raw/`  
**Output**: `data/interim/chunks/chunks_embedded.json`

### Changes from v1.0

| Change | Rationale |
|--------|-----------|
| Add pre-processing (Phase 0) | Clean HTML, translate, dedupe before chunking |
| Use datatrove quality filters | Industry-standard quality signals from RedPajama/Dolma |
| Consider larger chunks | Current ~500 tokens may be too granular |
| Store quality_score | For debugging and future filtering |

### Phase 0: Pre-Processing (NEW)

Run BEFORE chunking to clean raw text:

```python
# src/processing/preprocessing/document_cleaner.py

from datatrove.pipeline.filters import (
    GopherQualityFilter,
    LanguageFilter,
)
from deep_translator import GoogleTranslator
import ftfy

def preprocess_document(text: str, doc_lang: str = "auto") -> str | None:
    """Clean and translate document. Returns None if garbage."""
    
    # 1. Fix encoding issues (mojibake, etc.)
    text = ftfy.fix_text(text)
    
    # 2. Detect language and translate non-English
    if doc_lang == "auto":
        doc_lang = detect_language(text)
    
    if doc_lang not in ["en", "english"]:
        # Spanish, Turkish, Ukrainian, etc. → English
        try:
            translator = GoogleTranslator(source=doc_lang, target="en")
            text = translator.translate(text)
        except Exception as e:
            logger.warning(f"Translation failed for {doc_lang}: {e}")
            return None  # Skip untranslatable
    
    # 3. Strip HTML/markdown artifacts
    text = strip_html_tags(text)
    text = strip_markdown_artifacts(text)
    
    # 4. Deduplicate repeated paragraphs within document
    text = dedupe_paragraphs(text)
    
    return text
```

### Quality Gate (datatrove-based)

Replace hand-rolled rules with battle-tested quality signals from RedPajama/Dolma:

```python
from datatrove.pipeline.filters import GopherQualityFilter

# Gopher quality heuristics (used by RedPajama, Dolma)
QUALITY_THRESHOLDS = {
    "min_doc_words": 50,
    "max_doc_words": 100000,
    "min_avg_word_length": 3,
    "max_avg_word_length": 10,
    "max_symbol_word_ratio": 0.1,      # Catches "###" spam
    "max_bullet_lines_ratio": 0.9,     # Catches bullet-only docs
    "max_ellipsis_lines_ratio": 0.3,   # Catches "..." spam
    "max_non_alpha_words_ratio": 0.8,  # Catches OCR garbage
    "min_unique_words_ratio": 0.2,     # Catches repetition
}

def passes_quality_gate(text: str) -> tuple[bool, dict]:
    """Returns (passes, quality_signals)."""
    
    words = text.split()
    
    signals = {
        "word_count": len(words),
        "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
        "unique_words_ratio": len(set(words)) / len(words) if words else 0,
        "non_alpha_ratio": sum(1 for w in words if not w.isalpha()) / len(words) if words else 0,
        # ... additional signals
    }
    
    passes = all([
        signals["word_count"] >= QUALITY_THRESHOLDS["min_doc_words"],
        signals["avg_word_length"] >= QUALITY_THRESHOLDS["min_avg_word_length"],
        signals["unique_words_ratio"] >= QUALITY_THRESHOLDS["min_unique_words_ratio"],
        signals["non_alpha_ratio"] <= QUALITY_THRESHOLDS["max_non_alpha_words_ratio"],
    ])
    
    return passes, signals
```

### Translation Approach

| Language | Source | Method |
|----------|--------|--------|
| Spanish | DLA Piper (some jurisdictions) | `deep-translator` (Google API) |
| Turkish | Academic papers | `deep-translator` |
| Ukrainian/Cyrillic | OCR garbage | Skip (quality gate filters) |

**Fallback**: If `deep-translator` API fails or is rate-limited, use `argos-translate` (local, open-source NLLB-based).

```bash
pip install deep-translator argos-translate
```

### Chunk Size Decision

**Current**: ~500 tokens  
**Consider**: 750-1000 tokens

Larger chunks mean:
- Fewer chunks total (25K → ~15K)
- More context per chunk (better for relation extraction)
- Potentially more entities per chunk (co-occurrence)

Trade-off: Retrieval precision may decrease. Test both.

**Recommendation**: Parameterize chunk size, run comparison.

---

## 4. Phase 1B: Entity Extraction

**File**: `src/processing/entities/entity_processor.py`

**Input**: `data/interim/chunks/chunks_embedded.json`  
**Output**: `data/interim/entities/pre_entities.json`

### Changes from v1.0

| Change | Rationale |
|--------|-----------|
| Mistral-7B (not Qwen-72B) | Cost, consistency, no JSON bugs |
| Pydantic schema enforcement | No more regex JSON parsing |
| Extract structural relations | PART_OF, SAME_AS captured at source |
| Early ID hashing | Stable IDs from birth |
| Canonical type enforcement | Fixed vocabulary in prompt |
| Split compound entities | "transparency and consent" → two entities |

### Prompt Changes

**v1.0 prompt** (simplified):
```
Extract entities with name, type, description.
```

**v1.1 prompt** (SAME_AS only — PART_OF is pattern-based):
```
Extract entities from this text. For each entity:

1. name: Canonical form (split "X and Y" into separate entities)
2. type: One of [Concept, Organization, Technology, Regulation, Person, Country, Citation, Author, Journal]
3. description: Brief disambiguation context
4. same_as: (optional) If this is an ALIAS/abbreviation of a more canonical entity:
   {"target_name": "<canonical entity>"}
   Example: "AI Act" → same_as: "EU Artificial Intelligence Act"

Output JSON array of entities.
```

### PART_OF Detection (Pattern-Based, Post-Extraction)

PART_OF relations are detected deterministically after LLM extraction — not via prompt. This is more reliable for structured references.

**Reference**: Grounded in Akoma Ntoso (OASIS legal XML standard) and TAIR ontology (arxiv 2408.11925).

```python
LEGAL_HIERARCHY_PATTERNS = [
    # "Article X of [Parent]" — most common
    r"^(Article|Section|Chapter|Paragraph|Title|Annex|Schedule|Recital)\s+[\d\w\(\)]+\s+of\s+(?:the\s+)?(.+)$",
    
    # "X Act, Article Y" — inverted form
    r"^(.+?),?\s+(Article|Section)\s+[\d\w\(\)]+$",
    
    # "Article X PARENT" — no "of" (e.g., "GDPR Article 17")
    r"^(Article|Section)\s+[\d\w\(\)]+\s+(?!of\b)(.+)$",
]

ACADEMIC_HIERARCHY_PATTERNS = [
    # "Chapter X of [Citation]"
    r"^(Chapter|Section|Part)\s+[\d\w]+\s+of\s+(.+)$",
    
    # "Figure/Table X in [Citation]"
    r"^(Figure|Table|Appendix)\s+[\d\w]+\s+(?:in|from)\s+(.+)$",
]

def detect_part_of(entity_name: str) -> StructuralRelation | None:
    """Apply patterns to detect PART_OF parent."""
    for pattern in LEGAL_HIERARCHY_PATTERNS + ACADEMIC_HIERARCHY_PATTERNS:
        match = re.match(pattern, entity_name, re.IGNORECASE)
        if match:
            parent_name = match.group(2).strip()
            # Strip leading "the"
            parent_name = re.sub(r"^the\s+", "", parent_name, flags=re.IGNORECASE)
            return StructuralRelation(type="PART_OF", target_name=parent_name)
    return None
```

**Pattern match examples**:

| Input | Child (full entity) | Parent (extracted) |
|-------|---------------------|-------------------|
| Article 22 of the EU AI Act | Article 22 of the EU AI Act | EU AI Act |
| Section 5(b)(ii) of the Data Protection Act 2018 | Section 5(b)(ii) of the Data Protection Act 2018 | Data Protection Act 2018 |
| Recital 47 of Regulation (EU) 2024/1689 | Recital 47 of Regulation (EU) 2024/1689 | Regulation (EU) 2024/1689 |
| GDPR, Article 17 | GDPR, Article 17 | GDPR |
| Chapter 3 of Floridi (2018) | Chapter 3 of Floridi (2018) | Floridi (2018) |

### Pre-Entity ID Generation

Immediately after extraction, before any processing:

```python
def generate_pre_entity_id(name: str, type: str, chunk_id: str) -> str:
    """Deterministic ID from extraction context."""
    content = f"{name}:{type}:{chunk_id}"
    return "pre_" + hashlib.sha256(content.encode()).hexdigest()[:12]
```

### Two-Track Extraction (Preserved)

Still separate domain vs. academic entity types:

| Track | Types | Extraction |
|-------|-------|------------|
| Domain | Concept, Organization, Technology, Regulation, Person, Country | Full extraction + PART_OF patterns + SAME_AS prompt |
| Academic | Citation, Author, Journal | Extraction only (relations in Phase 1D, Scopus linkage in 1C) |

---

## 5. Phase 1C: Disambiguation

**File**: `src/processing/entities/disambiguation_processor.py`

**Input**: `data/interim/entities/pre_entities.json`  
**Output**: `data/interim/entities/normalized_entities.json`

### Changes from v1.0

| Change | Rationale |
|--------|-----------|
| Two-track disambiguation | Domain (embedding) vs Academic (Scopus linkage) |
| Alias tracking | Store merged surface forms |
| Alias ID tracking | Store merged pre_entity_ids |
| Type normalization | Map variants to canonical |
| Re-embed canonical | Embedding represents final form, not arbitrary variant |
| Preserve structural relations | Carry through from merged entities |

### Two-Track Disambiguation (NEW)

Domain and academic entities need **different disambiguation methods**:

| Track | Entity Types | Method | Why |
|-------|--------------|--------|-----|
| Domain | Concept, Organization, Technology, Regulation, Person, Country | Embedding similarity (FAISS + tiered thresholds) | Fuzzy matching needed ("AI Act" ≈ "EU AI Act") |
| Academic | Citation, Author, Journal | Scopus record linkage (DOI, author ID, ISSN) | Authoritative identifiers exist |

**Domain track** (unchanged from v1.0):
```python
# FAISS candidate generation → tiered threshold merging
candidates = faiss_search(entity.embedding, top_k=10, threshold=0.70)
# Then tiered passes at 0.90, 0.85, 0.82
```

**Academic track** (NEW — cleaner):
```python
def disambiguate_citation(citation: PreEntity, scopus_index: dict) -> Entity:
    """Match citation to Scopus record by DOI or title."""
    
    # 1. Try DOI exact match
    if citation.doi and citation.doi in scopus_index:
        return link_to_scopus(citation, scopus_index[citation.doi])
    
    # 2. Try title fuzzy match (Levenshtein, not embedding)
    best_match = fuzzy_title_match(citation.name, scopus_index)
    if best_match and best_match.score > 0.90:
        return link_to_scopus(citation, best_match)
    
    # 3. No match — keep as extracted (unlinked citation)
    return Entity.from_pre_entity(citation, linked=False)

def disambiguate_author(author: PreEntity, scopus_authors: dict) -> Entity:
    """Match author to Scopus author ID."""
    # Similar logic with author name normalization
    ...
```

### Alias Tracking

When merging entity A into canonical B:

```python
def merge_entity(canonical: Entity, variant: PreEntity) -> Entity:
    """Merge variant into canonical, preserving alias info."""
    
    # Track surface form
    if variant.name != canonical.name:
        canonical.aliases.append(variant.name)
    
    # Track original ID
    canonical.alias_ids.append(variant.pre_entity_id)
    
    # Merge chunk provenance
    canonical.chunk_ids.append(variant.chunk_id)
    
    # Merge structural relations (deduplicate)
    for rel in variant.structural_relations:
        if rel not in canonical.structural_relations:
            canonical.structural_relations.append(rel)
    
    return canonical
```

### PART_OF Parent Resolution

After disambiguation builds the entity index, resolve parent names to entity IDs:

```python
def resolve_structural_relations(entity: Entity, entity_index: dict[str, Entity]):
    """Resolve PART_OF target_name → target_id."""
    for struct_rel in entity.structural_relations:
        if struct_rel.type != "PART_OF":
            continue
            
        parent_name = struct_rel.target_name
        
        # 1. Exact match in index
        if parent_name in entity_index:
            struct_rel.target_id = entity_index[parent_name].entity_id
            continue
        
        # 2. Fuzzy match via FAISS (threshold 0.85)
        candidates = faiss_search(parent_name, top_k=3)
        if candidates and candidates[0].score > 0.85:
            struct_rel.target_id = candidates[0].entity_id
            continue
        
        # 3. Create parent entity if not found
        parent = Entity(
            name=parent_name,
            type=entity.type,  # Inherit type (Regulation → Regulation)
            description=f"Parent document containing {entity.name}",
            chunk_ids=entity.chunk_ids,  # Same provenance
        )
        entity_index[parent_name] = parent
        struct_rel.target_id = parent.entity_id
```

### Type Normalization

Map extracted types to canonical vocabulary:

```python
TYPE_MAPPING = {
    # Regulation variants
    "Regulatory Requirement": "Regulation",
    "Legal Provision": "Regulation",
    "Legislation": "Regulation",
    "Law": "Regulation",
    "Act": "Regulation",
    
    # Organization variants
    "Institution": "Organization",
    "Agency": "Organization",
    "Company": "Organization",
    
    # Technology variants
    "AI System": "Technology",
    "Algorithm": "Technology",
    "System": "Technology",
    
    # ... etc
}

def normalize_type(raw_type: str) -> str:
    return TYPE_MAPPING.get(raw_type, raw_type)
```

### Re-embed Canonical Forms

After disambiguation completes, re-embed using canonical representation:

```python
def embed_canonical(entity: Entity) -> Entity:
    """Embed the final canonical form, not inherited variant embedding."""
    text = f"{entity.name} ({entity.type}): {entity.description}"
    entity.embedding = embedder.embed(text)
    return entity
```

### Tiered Thresholds (Unchanged)

Keep v1.0 approach:

| Stage | Threshold | Purpose |
|-------|-----------|---------|
| 1 | > 0.70 | FAISS candidate generation |
| 2 | > 0.90 | Aggressive merge |
| 3 | > 0.85 | Conservative merge |
| 4 | > 0.82 | Final polish |

---

## 6. Phase 1D: Relations

**File**: `src/processing/relations/relation_processor.py`

**Input**: `data/interim/entities/normalized_entities.json`  
**Output**: `data/interim/relations/relations_normalized.json`

### Changes from v1.0

| Change | Rationale |
|--------|-----------|
| Import structural relations | PART_OF, SAME_AS from Phase 1B |
| Comprehensive citation coverage | All chunks, not MMR-sampled |
| Mistral-7B (already) | Keep |
| Pydantic output | Schema-enforced triplets |

### Structural Relations Import

Before LLM extraction, create relations from Phase 1B structural data:

```python
def import_structural_relations(entities: list[Entity]) -> list[Relation]:
    """Convert structural_relations to proper Relation objects."""
    relations = []
    
    for entity in entities:
        for struct_rel in entity.structural_relations:
            # Find target entity by name
            target = find_entity_by_name(struct_rel.target_name)
            if target:
                relations.append(Relation(
                    subject_id=entity.entity_id,
                    predicate=struct_rel.relation_type,  # "PART_OF" or "SAME_AS"
                    object_id=target.entity_id,
                    chunk_id=entity.chunk_ids[0],  # First provenance
                    confidence=1.0  # Structural = high confidence
                ))
    
    return relations
```

### Citation Coverage

**v1.0**: Citations only extracted when co-occurring with semantic entities in MMR-selected chunks.

**v1.1**: Citations get comprehensive coverage (all chunks where they appear).

```python
def get_chunks_for_entity(entity: Entity) -> list[str]:
    """Return chunk IDs for relation extraction."""
    
    if entity.type in ["Citation", "Author", "Journal"]:
        # Comprehensive: all chunks where entity appears
        return entity.chunk_ids
    else:
        # Semantic: MMR-selected diverse subset
        return mmr_select(entity.chunk_ids, k=10)
```

### Relation Extraction (Unchanged)

Keep v1.0 approach:

| Track | Predicate | Method |
|-------|-----------|--------|
| Semantic | OpenIE (free predicates) | LLM extraction |
| Academic | Fixed `discusses` | LLM confirms topic |

---

## 7. Phase 2A: Enrichment

**File**: `src/enrichment/enrichment_processor.py`

**Input**: `data/interim/relations/relations_normalized.json`  
**Output**: `data/processed/entities/`, `data/processed/relations/`

### Changes from v1.0

Minimal changes — Scopus enrichment works well.

| Change | Rationale |
|--------|-----------|
| Handle new entity schema | aliases, alias_ids fields |
| Validate against Pydantic | Consistent typing |

---

## 8. Phase 2B: Import

**File**: `src/graph/import_processor.py`

**Input**: `data/processed/`  
**Output**: Neo4j + FAISS indices

### Changes from v1.0

| Change | Rationale |
|--------|-----------|
| PART_OF edges | New relationship type |
| SAME_AS edges | New relationship type |
| Alias node property | Store aliases on Entity nodes |
| Rebuild FAISS with new IDs | ID scheme changed |

### Neo4j Schema Updates

```cypher
// New relationship types
CREATE (a:Entity)-[:PART_OF]->(b:Entity)
CREATE (a:Entity)-[:SAME_AS]->(b:Entity)

// Alias property on entities
CREATE (e:Entity {
  entity_id: $id,
  name: $name,
  type: $type,
  aliases: $aliases  // NEW: list of strings
})

// Index for alias lookup
CREATE INDEX entity_alias_idx FOR (e:Entity) ON (e.aliases)
```

### FAISS Rebuild

Same process as v1.0, but with new entity IDs:

```python
# Entity index
entity_embeddings = [e.embedding for e in entities]
entity_ids = [e.entity_id for e in entities]
faiss_index.add(np.array(entity_embeddings))
save_json(entity_ids, "entity_id_map.json")

# Chunk index (unchanged)
...
```

---

## 9. Validation & Testing

### Test Structure

```
tests/
├── models/
│   └── test_models.py           # Pydantic validation
├── utils/
│   └── test_llm.py              # LLM client, JSON enforcement
├── processing/
│   ├── test_chunking.py         # Quality gate
│   ├── test_entity_extraction.py
│   ├── test_disambiguation.py   # Alias tracking
│   └── test_relations.py        # Structural import
└── integration/
    └── test_pipeline.py         # End-to-end with sample data
```

### Validation Checks

After each phase, validate outputs:

```python
# Phase 1B: All pre-entities have IDs and valid types
assert all(e.pre_entity_id for e in pre_entities)
assert all(e.type in CANONICAL_TYPES for e in pre_entities)

# Phase 1C: Aliases tracked
merged_count = sum(len(e.aliases) for e in entities)
assert merged_count > 0, "No merges occurred?"

# Phase 1D: Structural relations imported
structural = [r for r in relations if r.predicate in ["PART_OF", "SAME_AS"]]
assert len(structural) > 0, "No structural relations?"

# Phase 2B: Graph connectivity
# Run: MATCH (a)-[:PART_OF]->(b) RETURN count(*)
```

### v1.0 vs v1.1 Comparison Metrics

| Metric | How to Measure |
|--------|----------------|
| Entity count | len(entities) |
| Alias coverage | sum(len(e.aliases) for e in entities) / len(entities) |
| Type consistency | len(set(e.type for e in entities)) — should be ≤9 |
| Structural relations | count of PART_OF + SAME_AS |
| Graph connectivity | Largest connected component % |
| Chunk quality | % chunks passing quality gate |

---

## 10. File Structure (Final)

```
src/
├── models/
│   ├── __init__.py
│   ├── chunk.py
│   ├── entity.py
│   └── relation.py
│
├── config/
│   ├── __init__.py
│   ├── base.py
│   ├── extraction.py
│   ├── disambiguation.py
│   └── relations.py
│
├── utils/
│   ├── llm.py              # NEW: consolidated LLM client
│   ├── io.py               # NEW: file I/O helpers
│   ├── batch.py            # NEW: batch processing base
│   ├── embedder.py         # KEEP
│   └── checkpoint.py       # KEEP
│
├── processing/
│   ├── chunking/
│   │   └── chunking_processor.py
│   ├── entities/
│   │   ├── entity_processor.py
│   │   └── disambiguation_processor.py
│   └── relations/
│       └── relation_processor.py
│
├── enrichment/
│   └── enrichment_processor.py
│
├── graph/
│   └── import_processor.py
│
└── retrieval/              # UNCHANGED
    └── ...

scripts/
├── run_pipeline.py         # NEW: orchestrator
└── run_query.py            # KEEP

tests/
├── models/
├── utils/
├── processing/
└── integration/
```

---

## Summary: What Changes, What Stays

### Changes

- [ ] Pre-processing: translation (deep-translator) + ftfy + datatrove quality
- [ ] Pydantic models for all data structures
- [ ] Single processor per phase (no tripartite)
- [ ] `run_pipeline.py` orchestrator with phase selection
- [ ] Chunk quality gate (datatrove Gopher heuristics)
- [ ] PART_OF relations (pattern-based, document structure)
- [ ] SAME_AS relations (LLM-based, aliases)
- [ ] Early pre_entity_id hashing
- [ ] Alias tracking through disambiguation
- [ ] Cleaner academic track (Scopus record linkage, not embedding similarity)
- [ ] Type normalization to canonical vocabulary
- [ ] Re-embed canonical entity forms
- [ ] Comprehensive citation relation coverage
- [ ] Mistral-7B throughout
- [ ] Consolidated utils (llm.py, io.py, batch.py)

### Stays

- [ ] BGE-M3 embeddings
- [ ] FAISS for vector search
- [ ] Neo4j for graph storage
- [ ] Tiered disambiguation thresholds (domain track)
- [ ] MMR for semantic entity chunk selection
- [ ] Two-track extraction (domain vs. academic)
- [ ] OpenIE for semantic relations (Phase 1D)
- [ ] Phase 3 retrieval system (unchanged)
- [ ] pytest for testing

---

## Explicit Scope: v1.1 vs v1.2

### v1.1 (This Plan)

| Phase | What's included |
|-------|-----------------|
| 1A | Pre-processing (translate, clean) + datatrove quality + chunking |
| 1B | Entity extraction + pattern-based PART_OF + LLM SAME_AS |
| 1C | Domain disambiguation (embedding) + Academic linkage (Scopus) |
| 1D | OpenIE relations (unchanged methodology) |
| 2A-2B | Import with PART_OF/SAME_AS edges |

### v1.2 (Future — NOT this plan)

| Feature | Rationale for deferral |
|---------|------------------------|
| SKOS concept taxonomy | Ontological, not text-extractable |
| Cross-jurisdictional mapping | Needs manual validation |
| Constrained predicates (replace OpenIE) | Polish, not core |
| Perplexity-based quality filtering | datatrove heuristics sufficient for now |

See ARCHITECTURE.md § 7.7.3 for v1.2 design notes.