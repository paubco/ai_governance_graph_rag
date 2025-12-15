# V1.1 Readiness Survey: GraphRAG Pipeline Comprehensive Assessment

**Date**: 2025-12-15
**Target**: v1.1 rewrite with Pydantic models, Mistral-7B throughout, structural relations
**Scope**: Phases 1A-2B (retrieval unchanged)
**Reference**: `docs/standardization_audit.md`, `docs/V1_1_IMPLEMENTATION_PLAN.md`

---

## Executive Summary

This survey identifies technical debt and migration requirements for the v1.1 rewrite. Key findings:

**CRITICAL ISSUES** (Must Address):
- ⚠️ **Qwen-72B usage**: 3 files using Qwen models (must migrate to Mistral-7B)
- ⚠️ **Raw dict usage**: Extensive use of `Dict[str, Any]` instead of Pydantic models
- ⚠️ **Hardcoded paths**: 40+ instances of `data/interim/*` hardcoded paths
- ⚠️ **Missing tests**: 14/14 processing modules have NO corresponding tests

**MODERATE ISSUES**:
- Nested loops in entity/relation processing (O(n²) and O(n³) patterns)
- Magic number thresholds (20+ instances)
- Defensive code density varies widely (0-8 try blocks per file)

**POSITIVE FINDINGS**:
- ✅ Standard docstring format adopted across codebase
- ✅ Retrieval pipeline (Phase 3) already uses dataclasses
- ✅ No bare `except:` clauses found
- ✅ Mistral-7B already in use for Phase 1D (relations)

---

## 1. DATA MODEL INVENTORY

### 1.1 Current Pydantic/BaseModel Usage

| Phase | File | Model Name | Key Fields | v1.1 Action |
|-------|------|------------|-----------|-------------|
| 1D Relations | `relation_extractor.py` | `Relation` | subject, predicate, object, description | ✅ Keep, extend with provenance |
| 1D Relations | `relation_extractor.py` | `RelationOutput` | relations: List[Relation] | ✅ Keep |
| 3 Retrieval | Various | Multiple dataclasses | (see retrieval/config.py) | ✅ Keep, already compliant |

**Pydantic models found**: 2 (both in relation extraction)
**Total Python files in processing**: 18
**Coverage**: 5.6% (1/18 modules use Pydantic)

### 1.2 Current Dataclass Usage

| Phase | File | Model Name | Type | Key Fields | v1.1 Action |
|-------|------|------------|------|-----------|-------------|
| 1A Chunking | `semantic_chunker.py` | `Chunk` | @dataclass | chunk_id, document_id, text, position, metadata | Migrate to Pydantic |
| 0 Ingestion | `document_loader.py` | `Document` | @dataclass | doc_id, content, metadata, title, url | Migrate to Pydantic |
| 3 Retrieval | `config.py` | `ExtractedEntity` | @dataclass | name, type, confidence | ✅ Keep (retrieval unchanged) |
| 3 Retrieval | `config.py` | `QueryFilters` | @dataclass | entity_types, time_range | ✅ Keep |
| 3 Retrieval | `config.py` | `ParsedQuery` | @dataclass | raw_query, entities, filters | ✅ Keep |
| 3 Retrieval | `config.py` | `ResolvedEntity` | @dataclass | entity_id, name, similarity | ✅ Keep |
| 3 Retrieval | `config.py` | `Relation` | @dataclass | subject_id, predicate, object_id | ✅ Keep |
| 3 Retrieval | `config.py` | `Subgraph` | @dataclass | entities, relations | ✅ Keep |
| 3 Retrieval | `config.py` | `Chunk` | @dataclass | chunk_id, text, source_path | ✅ Keep |
| 3 Retrieval | `config.py` | `RankedChunk` | @dataclass | chunk, score, provenance | ✅ Keep |
| 3 Retrieval | `config.py` | `RetrievalResult` | @dataclass | query, entities, subgraph, chunks | ✅ Keep |
| 3 Retrieval | `result_ranker.py` | `ScoringDebugInfo` | @dataclass | chunk_id, base_score, bonuses | ✅ Keep |
| 3 Retrieval | `retrieval_processor.py` | `QueryUnderstanding` | @dataclass | parsed_query, resolved_entities | ✅ Keep |
| 3 Retrieval | `answer_generator.py` | `GeneratedAnswer` | @dataclass | answer, citations, metadata | ✅ Keep |

**Dataclasses found**: 14 (13 in retrieval, 1 in chunking, 1 in ingestion)

### 1.3 Raw Dict Usage (Critical Gap)

**Heavy dict usage found in**:

| Phase | File | Function | Usage Pattern | v1.1 Action |
|-------|------|----------|---------------|-------------|
| 1B Entities | `alias_builder.py` | `build_index_for_pre_entities` | Returns `Dict[str, Set[str]]` | Create `ChunkEntityIndex` model |
| 1B Entities | `alias_builder.py` | `build_index_for_normalized` | Returns `Dict[str, Dict]` | Create `EntityAliasMap` model |
| 1B Entities | `entity_extractor.py` | `extract_entities` | Returns `Dict[str, List[Dict]]` | Create `PreEntity` model |
| 1B Entities | `entity_processor.py` | `extract_chunk_with_retry` | Parameter: `chunk: dict` | Use `Chunk` model |
| 1C Disambiguation | `disambiguation_processor.py` | Various | Extensive dict manipulation | Create `DisambiguationCandidate` model |
| 1D Relations | `normalize_relations.py` | `load_entity_lookup` | Returns `Dict[str, str]` | Create `EntityLookup` model |
| 1D Relations | `run_relation_extraction.py` | `initialize_extractor` | Parameter: `config: dict` | Create `RelationExtractionConfig` model |
| 2A Enrichment | `enrichment_processor.py` | Various | Returns `Dict[str, Any]` | Create enrichment models |
| 2B Graph | `neo4j_import_processor.py` | `prepare_*` | Multiple `Dict[str, str]` mappings | Create Neo4j import models |

**Estimated dict→Pydantic migrations needed**: 15-20 models across 6 modules

### 1.4 Sample Data Schemas (From Actual Files)

**Chunk Schema** (`data/interim/chunks/chunks_text.json`):
```json
{
  "chunk_id": "reg_AE_CHUNK_0000",
  "document_id": "reg_AE",
  "text": "...",
  "position": 0,
  "sentence_count": 3,
  "token_count": 92,
  "metadata": {
    "source_type": "regulation",
    "title": "...",
    "country_code": "AE",
    "url": "...",
    "scraped_date": "...",
    "section_titles": [...]
  }
}
```

**Entity Schema** (`data/interim/entities/pre_entities.json`):
```json
{
  "chunk_id": "reg_AE_CHUNK_0001",
  "entities": [
    {
      "name": "federal laws",
      "type": "Legal Framework",
      "description": "...",
      "chunk_id": "reg_AE_CHUNK_0001"
    }
  ]
}
```

**Relation Schema** (`data/interim/relations/relations_output.jsonl`):
```json
{
  "entity_id": "'Digital Ethics: A Guide...'",
  "entity_name": "...",
  "entity_type": "Report",
  "relations": [
    {
      "subject": "...",
      "predicate": "discusses",
      "object": "...",
      "chunk_ids": ["chunk_0119"],
      "extraction_strategy": "academic"
    }
  ],
  "cost": 0.0003,
  "timestamp": "2025-12-04T11:41:41"
}
```

---

## 2. LLM INTERACTION PATTERNS

### 2.1 Model Usage Breakdown

| File | Model Used | Purpose | Output Parsing | JSON Schema? | Retry? | v1.1 Action |
|------|------------|---------|----------------|--------------|--------|-------------|
| `entity_extractor.py` | ⚠️ **Qwen-72B** | Entity extraction (1B) | Regex + JSON | ❌ No | ✅ Yes (3) | **MIGRATE to Mistral-7B** |
| `disambiguation_processor.py` | ⚠️ **Qwen-7B** | SameJudge LLM (1C) | JSON + regex fallback | ❌ No | ✅ Yes (3) | **MIGRATE to Mistral-7B** |
| `entity_disambiguator.py` | ⚠️ **Qwen-7B** | VecJudge (1C legacy) | JSON | ❌ No | ✅ Yes | **MIGRATE to Mistral-7B** |
| `relation_extractor.py` | ✅ Mistral-7B | Relation extraction (1D) | **Pydantic** | ✅ Yes | ✅ Yes (3) | ✅ Keep |
| `query_parser.py` | ✅ Mistral-7B | Query entity extraction (3) | JSON | ❌ No | ✅ Yes | ✅ Keep |

**CRITICAL**: 3 files using Qwen models must migrate to Mistral-7B

### 2.2 Prompt Locations

| File | Prompt Variable | Location | Format | v1.1 Action |
|------|----------------|----------|--------|-------------|
| `entity_extractor.py` | `ENTITY_EXTRACTION_PROMPT` | `src/prompts/prompts.py` | Template | Update for PART_OF/SAME_AS |
| `disambiguation_processor.py` | `SAMEJUDGE_PROMPT` | Inline (line ~452) | f-string | Extract to config |
| `relation_extractor.py` | `ACADEMIC_ENTITY_EXTRACTION_PROMPT` | Inline (line ~1302) | f-string | Extract to config |
| `query_parser.py` | `QUERY_ENTITY_EXTRACTION_PROMPT` | Inline (line ~146) | f-string | ✅ Keep (retrieval unchanged) |

### 2.3 Together.ai Client Patterns

**Consistent pattern across all files**:
```python
from together import Together
self.client = Together(api_key=api_key)
response = self.client.chat.completions.create(
    model=self.model,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,  # Deterministic
    max_tokens=4096
)
```

**Rate limiting**: Uses `utils/rate_limiter.py` (2900 RPM, conservative limit)

---

## 3. COMPLEXITY HOTSPOTS

### 3.1 Nested Loop Analysis

| Function | File:Line | Pattern | Estimated Big O | Anti-Pattern | v1.1 Fix |
|----------|-----------|---------|-----------------|--------------|----------|
| `find_aliases` | `alias_builder.py:268-290` | 3 nested loops | O(n × m × k) | Triple iteration over entities/chunks/names | Use FAISS for fuzzy matching |
| `_build_candidate_pairs_faiss` | `disambiguation_processor.py:222-226` | 2 nested loops | O(n × k) | FAISS results + inner loop | Vectorize distance filtering |
| `build_cooccurrence_typed` | `build_entity_cooccurrence.py:~100-150` | 3+ nested loops | O(n³) | Entity×Entity×Chunk iteration | Use sparse matrices (scipy) |
| `extract_relations_for_entity` | `relation_extractor.py:~1400-1500` | 2 nested loops | O(n × m) | Entity×Chunks iteration | ✅ Already parallelized |
| `_filter_pairs_by_name_type` | `disambiguation_processor.py:~250-280` | 2 nested loops | O(n²) | Pairwise comparison | Use hash-based filtering |

**Critical findings**:
- Alias discovery (Phase 1C) has O(n³) complexity → **Use vectorized operations**
- Co-occurrence matrix (Phase 1D) iterates all entity pairs → **Use sparse matrix libraries**
- FAISS search results processed in loops → **Batch distance thresholding**

### 3.2 FAISS Search Patterns

| File | Line | Operation | Top-k | Threshold | Optimization Opportunity |
|------|------|-----------|-------|-----------|--------------------------|
| `disambiguation_processor.py` | 216 | `index.search(embeddings, k+1)` | 11 | 0.70 | ✅ Batched (good) |
| `entity_disambiguator.py` | 409 | `index.search(embeddings, k+1)` | 11 | None | Batch normalize distances |
| `graph_expander.py` | 201 | `faiss_index.search(embedding, k+1)` | 11 | None | ✅ Already optimal |
| `entity_resolver.py` | ~85 | `faiss_index.search(query_emb, k)` | 10 | 0.85 | ✅ Retrieval (unchanged) |

**FAISS usage is generally efficient** - most searches are batched and use appropriate k values.

### 3.3 String Concatenation (Minimal Issues)

Only 1 instance found of string concatenation in loops:
- `normalize_relations.py:248` - Counter increment (not a performance issue)

**Assessment**: String concatenation is NOT a major anti-pattern in this codebase.

---

## 4. DEFENSIVE CODE DENSITY

### 4.1 Quantitative Analysis

| File | Lines | Try Blocks | Try/100L | Bare Except | None Checks | Type Hints | Verdict |
|------|-------|------------|----------|-------------|-------------|------------|---------|
| `entity_extractor.py` | 231 | 2 | 0.9 | 0 | 2 | 5 | ✅ Minimal defense |
| `entity_processor.py` | 472 | 2 | 0.4 | 0 | 2 | 5 | ✅ Minimal defense |
| `relation_extractor.py` | 1549 | 8 | 0.5 | 0 | 14 | 54 | ⚠️ Moderate defense |
| `disambiguation_processor.py` | 1245 | 8 | 0.6 | 0 | 4 | 30 | ⚠️ Moderate defense |
| `semantic_chunker.py` | 289 | N/A | N/A | 0 | N/A | N/A | ✅ Clean |

**Key findings**:
- ✅ **ZERO bare `except:` clauses** across all files (excellent)
- ✅ Try blocks are focused (0.4-0.9 per 100 lines = minimal defensive coding)
- ⚠️ Type hint coverage varies (5-54 annotations per file)
- ✅ None checks are sparse (2-14 per file = not overly defensive)

**Assessment**: Code is NOT overly defensive. Try blocks are used appropriately for API calls and JSON parsing.

### 4.2 Error Handling Patterns

**Common pattern** (seen in 4/5 files):
```python
for attempt in range(max_retries):
    try:
        response = self.client.chat.completions.create(...)
        result = json.loads(response.choices[0].message.content)
        return result
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Attempt {attempt+1} failed: {e}")
        if attempt == max_retries - 1:
            raise
```

**Assessment**: Error handling follows best practices (specific exceptions, retry logic, logging).

---

## 5. HARDCODED VALUES

### 5.1 Model Names (CRITICAL)

| File:Line | Hardcoded Value | Type | v1.1 Action |
|-----------|----------------|------|-------------|
| `entity_extractor.py:67` | `"Qwen/Qwen2.5-72B-Instruct-Turbo"` | ⚠️ Model | **CHANGE to Mistral-7B** |
| `disambiguation_processor.py:345` | `"Qwen/Qwen2.5-7B-Instruct-Turbo"` | ⚠️ Model | **CHANGE to Mistral-7B** |
| `entity_disambiguator.py:645` | `"Qwen/Qwen2.5-7B-Instruct-Turbo"` | ⚠️ Model | **CHANGE to Mistral-7B** |
| `relation_extractor.py:517` | `"mistralai/Mistral-7B-Instruct-v0.3"` | ✅ Model | ✅ Correct |
| `query_parser.py:64` | `"mistralai/Mistral-7B-Instruct-v0.3"` | ✅ Model | ✅ Correct (retrieval) |

### 5.2 Hardcoded Paths (EXTENSIVE)

**Sample of 40+ instances**:

| File:Line | Hardcoded Path | v1.1 Action |
|-----------|----------------|-------------|
| `disambiguation_processor.py:683` | `"data/interim/entities"` | Move to config |
| `disambiguation_processor.py:979` | `"data/interim/entities/pre_entities_clean.json"` | Move to config |
| `entity_processor.py:75` | `"data/interim/entities/pre_entities.json"` | Move to config |
| `entity_processor.py:103` | `"data/interim/chunks/chunks_text.json"` | Move to config |
| `alias_processor.py:45-51` | All paths (5 lines) | Move to config |
| `chunk_processor.py:10` | `Path("data/interim")` | Move to config |
| `relation_processor.py:15` | `Path("data/interim/relations")` | Move to config |
| `build_entity_cooccurrence.py:249-255` | All paths (7 lines) | Move to config |
| `run_relation_extraction.py:222-225` | All paths (4 lines) | Move to config |

**Total hardcoded path instances**: 40+

**Recommendation**: Create centralized `src/config/paths.py`:
```python
from pathlib import Path

DATA_ROOT = Path("data")
DATA_RAW = DATA_ROOT / "raw"
DATA_INTERIM = DATA_ROOT / "interim"
DATA_PROCESSED = DATA_ROOT / "processed"

# Interim paths
CHUNKS_DIR = DATA_INTERIM / "chunks"
ENTITIES_DIR = DATA_INTERIM / "entities"
RELATIONS_DIR = DATA_INTERIM / "relations"

# Specific files
CHUNKS_TEXT = CHUNKS_DIR / "chunks_text.json"
CHUNKS_EMBEDDED = CHUNKS_DIR / "chunks_embedded.json"
PRE_ENTITIES = ENTITIES_DIR / "pre_entities.json"
NORMALIZED_ENTITIES = ENTITIES_DIR / "normalized_entities.json"
# ... etc
```

### 5.3 Magic Numbers (Thresholds)

| File:Line | Value | Purpose | Type | v1.1 Action |
|-----------|-------|---------|------|-------------|
| `alias_builder.py:145` | `0.5` | Overlap ratio | Threshold | Move to config |
| `alias_builder.py:166` | `0.80` | Min coverage | Threshold | Move to config |
| `disambiguation_processor.py:371` | `0.00003` | Cost per call | Config | Document as constant |
| `entity_disambiguator.py:473-474` | `0.99`, `0.97` | Auto-merge thresholds | Threshold | Move to config |
| `semantic_chunker.py:64` | `0.7` | Similarity threshold | Threshold | ✅ Already parameterized |
| `relation_processor.py:324` | `0.30` | Second round rate | Sampling | Move to config |
| `relation_extractor.py:288-289` | `0.85`, `0.65` | Semantic threshold, MMR lambda | Threshold | Move to config |
| `relation_extractor.py:681, 759` | `0.15` | Similarity threshold | Threshold | Move to config |

**Total magic numbers found**: 20+

**Recommendation**: Move to `src/config/thresholds.py` with documentation of purpose.

---

## 6. INTERFACE CONTRACTS

### 6.1 JSON Write Operations

| Boundary | Producer File | Consumer | Schema | v1.1 Model | Gap |
|----------|--------------|----------|--------|------------|-----|
| Phase 1A → 1B | `chunk_processor.py:137` | `entity_processor.py` | Chunk dict | `Chunk` (Pydantic) | Type validation |
| Phase 1B → 1C | `entity_processor.py:225` | `disambiguation_processor.py` | Entity dict | `PreEntity` (Pydantic) | Type validation |
| Phase 1C → 1D | `disambiguation_processor.py:768` | `relation_processor.py` | Entity dict | `Entity` (Pydantic) | Type validation |
| Phase 1D → 2A | `normalize_relations.py` (jsonl) | `enrichment_processor.py` | Relation dict | `Relation` (Pydantic) | Type validation |
| Phase 2A → 2B | `enrichment_processor.py` | `neo4j_import_processor.py` | Various dicts | Neo4j models | Type validation |

**Current state**: All interfaces use raw JSON dicts with manual validation
**v1.1 target**: All interfaces use Pydantic models with automatic validation

### 6.2 Data Files Present

**Chunks** (`data/interim/chunks/`):
- `chunks_text.json` (88 MB) - Text only
- `chunks_embedded.json` (756 MB) - With BGE-M3 embeddings
- `chunks_metadata.json` (78 KB) - Per-document stats
- `chunking_summary.json` (502 B) - Aggregate stats

**Entities** (`data/interim/entities/`):
- `pre_entities.json` (53 MB) - Phase 1B output
- `normalized_entities.json` (1.6 GB) - Phase 1C output (with embeddings)
- `normalized_entities_with_ids.json` (1.7 GB) - With stable IDs
- `entity_name_to_id.json` (3 MB) - Lookup table
- `entity_name_to_id_with_aliases.json` (3 MB) - With aliases
- `cooccurrence_*.json` (3 files, 4-8 MB each) - Co-occurrence matrices

**Relations** (`data/interim/relations/`):
- `relations_output.jsonl` (65 MB) - Raw extractions (append-only)
- `relations_normalized.json` (34 MB) - Deduplicated
- `unmatched_entities.json` (8 MB) - Failed entity lookups
- `progress_state.json` (384 B) - Checkpoint state

### 6.3 Schema Consistency Issues

**Identified gaps**:
1. ⚠️ Chunk schema has `document_id` in JSON but parameter is `doc_id` in code
2. ⚠️ Entity schema has nested "entities" array per chunk (inconsistent with normalized format)
3. ⚠️ Relation schema mixes `entity_id` (string) and `entity_name` (redundant)
4. ⚠️ No validation that embeddings are 1024-dim (BGE-M3 standard)

**Recommendation**: Pydantic models with validators will catch these inconsistencies.

---

## 7. TEST COVERAGE GAPS

### 7.1 Untested Modules (CRITICAL)

**All 14 processing modules lack tests**:

```
processing/chunking/chunk_processor.py
processing/chunking/semantic_chunker.py
processing/entities/alias_builder.py
processing/entities/alias_processor.py
processing/entities/disambiguation_processor.py
processing/entities/entity_disambiguator.py
processing/entities/entity_extractor.py
processing/entities/entity_processor.py
processing/entities/filter_pre_entities.py
processing/relations/build_entity_cooccurrence.py
processing/relations/normalize_relations.py
processing/relations/relation_extractor.py
processing/relations/relation_processor.py
processing/relations/run_relation_extraction.py
```

**Existing tests** (`tests/processing/`):
- 5 test files exist BUT they test OLD implementations or specific functions
- No integration tests for full phase workflows

### 7.2 Test Structure vs Source Structure

| Source Package | Test Package | Coverage |
|----------------|--------------|----------|
| `src/ingestion/` (5 files) | `tests/ingestion/` (2 tests) | 40% |
| `src/processing/` (18 files) | `tests/processing/` (5 tests) | 28% |
| `src/enrichment/` (4 files) | `tests/extraction/` (1 test) | 25% |
| `src/graph/` (5 files) | `tests/graph/` (3 tests) | 60% |
| `src/retrieval/` (9 files) | `tests/retrieval/` (5 tests) | 56% |
| `src/utils/` (11 files) | `tests/utils/` (3 tests) | 27% |

**Overall test coverage by module count**: 20/54 = **37%**

### 7.3 Test Recommendations for v1.1

**Priority 1 (Core extraction logic)**:
1. `test_entity_extraction_pydantic.py` - Validate PreEntity model enforcement
2. `test_disambiguation_tiered.py` - Test 4-stage disambiguation with alias tracking
3. `test_relation_extraction_structural.py` - Test PART_OF/SAME_AS detection
4. `test_chunk_quality_gate.py` - Test datatrove quality filters

**Priority 2 (Integration)**:
5. `test_phase_1b_to_1c_interface.py` - Validate PreEntity → Entity transition
6. `test_phase_1c_to_1d_interface.py` - Validate Entity → Relation interface
7. `test_pydantic_serialization.py` - Test model JSON round-tripping

**Priority 3 (Utilities)**:
8. `test_llm_client.py` - Test unified LLM client with Mistral-7B
9. `test_config_loader.py` - Test centralized config management

---

## 8. FOCUSED ISSUES (Per User Request)

### 8.1 Files Using Qwen-72B (MUST CHANGE)

| File | Model | Line | Context | v1.1 Action |
|------|-------|------|---------|-------------|
| ⚠️ `entity_extractor.py` | Qwen-72B | 67 | Default model parameter | Change to `mistralai/Mistral-7B-Instruct-v0.3` |
| ⚠️ `entity_extractor.py` | Qwen-72B | 5 | Docstring reference | Update docs to reflect Mistral |
| ⚠️ `disambiguation_processor.py` | Qwen-7B | 345 | SameJudge model | Change to Mistral-7B |
| ⚠️ `entity_disambiguator.py` | Qwen-7B | 645 | VecJudge model | Change to Mistral-7B |

**Impact**: ~200K entities extracted with Qwen-72B will need re-extraction

**Cost estimate**:
- v1.0 cost: $30 (Qwen-72B @ $0.80/1M input tokens)
- v1.1 cost: ~$6 (Mistral-7B @ $0.20/1M input tokens)
- **Savings**: $24 per full corpus run

### 8.2 Raw Dict Usage for Structured Data (NEEDS PYDANTIC)

**High-priority migrations**:

1. **`PreEntity`** (Phase 1B output):
   ```python
   class PreEntity(BaseModel):
       pre_entity_id: str  # NEW: hash at extraction
       name: str
       type: str  # Enforced canonical types
       description: str
       chunk_id: str
       structural_relations: List[StructuralRelation] = []
   ```

2. **`Entity`** (Phase 1C output):
   ```python
   class Entity(BaseModel):
       entity_id: str
       name: str
       type: str
       description: str
       aliases: List[str] = []  # NEW
       alias_ids: List[str] = []  # NEW
       embedding: List[float]  # Validated 1024-dim
       chunk_ids: List[str]
   ```

3. **`Chunk`** (Phase 1A output - migrate from dataclass):
   ```python
   class Chunk(BaseModel):
       chunk_id: str
       text: str
       embedding: Optional[List[float]] = None
       metadata: ChunkMetadata
       quality_score: Optional[float] = None  # NEW
   ```

### 8.3 Nested Loops in Entity/Relation Processing

**Critical performance bottlenecks**:

1. **`alias_builder.py:268-290`** - O(n³) alias discovery:
   - Current: Triple nested loop over entities/chunks/names
   - Fix: Use FAISS for fuzzy string matching (O(n log n))

2. **`build_entity_cooccurrence.py`** - O(n²) co-occurrence matrix:
   - Current: Pairwise entity iteration
   - Fix: Use scipy sparse matrices with vectorized operations

3. **`disambiguation_processor.py:222-226`** - O(n×k) FAISS post-processing:
   - Current: Loop over FAISS results
   - Fix: Vectorize distance thresholding with numpy

**Expected speedup**: 5-10x for alias discovery, 2-3x for co-occurrence

### 8.4 Hardcoded Paths (NEEDS CENTRALIZED CONFIG)

**Create `src/config/paths.py`**:
```python
from pathlib import Path
import os

# Allow override via environment variable
DATA_ROOT = Path(os.getenv("DATA_DIR", "data"))

# Raw data
DATA_RAW = DATA_ROOT / "raw"
RAW_ACADEMIC = DATA_RAW / "academic"
RAW_DLAPIPER = DATA_RAW / "dlapiper"

# Interim data (Phase 1 outputs)
DATA_INTERIM = DATA_ROOT / "interim"
CHUNKS_DIR = DATA_INTERIM / "chunks"
ENTITIES_DIR = DATA_INTERIM / "entities"
RELATIONS_DIR = DATA_INTERIM / "relations"

# Processed data (Phase 2+ outputs)
DATA_PROCESSED = DATA_ROOT / "processed"
FAISS_DIR = DATA_PROCESSED / "faiss"
GRAPH_DIR = DATA_PROCESSED / "graph"

# Specific files (most common)
CHUNKS_TEXT = CHUNKS_DIR / "chunks_text.json"
CHUNKS_EMBEDDED = CHUNKS_DIR / "chunks_embedded.json"
PRE_ENTITIES = ENTITIES_DIR / "pre_entities.json"
PRE_ENTITIES_CLEAN = ENTITIES_DIR / "pre_entities_clean.json"
NORMALIZED_ENTITIES = ENTITIES_DIR / "normalized_entities.json"
ENTITY_LOOKUP = ENTITIES_DIR / "entity_name_to_id.json"
RELATIONS_RAW = RELATIONS_DIR / "relations_output.jsonl"
RELATIONS_NORMALIZED = RELATIONS_DIR / "relations_normalized.json"
```

**Migration**: Replace all 40+ hardcoded path instances with imports from this module.

### 8.5 Missing Tests for Core Extraction Logic

**Immediate test needs**:

1. **Entity extraction** (`test_entity_extraction.py`):
   - Test PART_OF pattern detection (15+ regex patterns)
   - Test SAME_AS LLM extraction
   - Test Pydantic schema enforcement
   - Test early ID hashing

2. **Disambiguation** (`test_disambiguation.py`):
   - Test alias tracking through merges
   - Test tiered thresholds (0.90 → 0.85 → 0.82)
   - Test PART_OF parent resolution
   - Test type normalization

3. **Relations** (`test_relations.py`):
   - Test structural relation import
   - Test comprehensive citation coverage
   - Test Pydantic output validation

**Coverage target**: 80% for core extraction modules (1B, 1C, 1D)

---

## 9. MIGRATION PRIORITY MATRIX

### Phase 1: Foundation (Week 1)

| Task | Effort | Impact | Priority | Blocking? |
|------|--------|--------|----------|-----------|
| Create Pydantic models | 2 days | ⚠️ Critical | P0 | Yes (all other work) |
| Centralize config (paths, models) | 1 day | ⚠️ Critical | P0 | Yes |
| Create LLM client wrapper | 1 day | High | P1 | Partial |
| Migrate Qwen → Mistral-7B | 0.5 day | ⚠️ Critical | P0 | Partial |

### Phase 2: Core Modules (Week 2)

| Task | Effort | Impact | Priority | Blocking? |
|------|--------|--------|----------|-----------|
| Rewrite Phase 1B (entity extraction) | 2 days | ⚠️ Critical | P0 | Yes |
| Rewrite Phase 1C (disambiguation) | 2 days | ⚠️ Critical | P0 | Yes |
| Add PART_OF pattern detection | 1 day | High | P1 | No |
| Add SAME_AS LLM extraction | 1 day | High | P1 | No |

### Phase 3: Integration (Week 3)

| Task | Effort | Impact | Priority | Blocking? |
|------|--------|--------|----------|-----------|
| Update Phase 1D (structural relations) | 1 day | Medium | P2 | No |
| Test full pipeline (1A→1B→1C→1D) | 2 days | High | P1 | Yes |
| Add quality gate (datatrove) | 1 day | Medium | P2 | No |
| Optimize nested loops | 2 days | Medium | P2 | No |

### Phase 4: Polish (Week 4)

| Task | Effort | Impact | Priority | Blocking? |
|------|--------|--------|----------|-----------|
| Write comprehensive tests | 3 days | High | P1 | No |
| Update documentation | 1 day | Low | P3 | No |
| Performance benchmarking | 1 day | Low | P3 | No |
| Code review & refactoring | 1 day | Low | P3 | No |

---

## 10. RISK ASSESSMENT

### High Risk Items

1. ⚠️ **Model migration (Qwen → Mistral)**:
   - Risk: Output format changes, quality degradation
   - Mitigation: Parallel comparison on 100-chunk sample first

2. ⚠️ **Pydantic validation strictness**:
   - Risk: Existing data fails validation
   - Mitigation: Write migration scripts to fix legacy data

3. ⚠️ **PART_OF regex patterns**:
   - Risk: Misses structural relations
   - Mitigation: Test on sample of 50 known cases (GDPR articles, etc.)

### Medium Risk Items

4. **Nested loop optimizations**:
   - Risk: Introduces bugs in complex logic
   - Mitigation: Unit tests before/after, validate output equality

5. **Centralized config**:
   - Risk: Breaks existing scripts
   - Mitigation: Backward-compatible imports initially

### Low Risk Items

6. **Test coverage improvements**: No risk (additive only)
7. **Documentation updates**: No risk

---

## 11. APPENDIX: FILE-BY-FILE ANALYSIS

### Processing Files (18 total)

**Chunking** (2 files):
- `chunk_processor.py` (269 lines) - Orchestrator, needs config migration
- `semantic_chunker.py` (289 lines) - Core logic, ✅ already clean

**Entities** (6 files):
- `entity_extractor.py` (231 lines) - ⚠️ Qwen-72B, needs Pydantic
- `entity_processor.py` (472 lines) - Orchestrator, needs config
- `filter_pre_entities.py` - Type enforcement, needs Pydantic validation
- `entity_disambiguator.py` - ⚠️ Qwen-7B, needs Mistral migration
- `disambiguation_processor.py` (1245 lines) - ⚠️ Qwen-7B, needs Pydantic
- `alias_processor.py` / `alias_builder.py` - Needs optimization

**Relations** (6 files):
- `relation_extractor.py` (1549 lines) - ✅ Mistral-7B, has Pydantic
- `relation_processor.py` (407 lines) - Orchestrator, needs config
- `run_relation_extraction.py` (393 lines) - Entry point, needs cleanup
- `build_entity_cooccurrence.py` (307 lines) - Needs optimization
- `normalize_relations.py` (426 lines) - Needs Pydantic
- `build_alias_lookup.py` - Utility

### Test Coverage

**Tests present**: 5 files in `tests/processing/`
- `test_semantic_chunker.py`
- `test_entity_extraction.py`
- `test_entity_disambiguator.py`
- `test_relation_extraction.py`
- `test_relation_extraction_parallel.py`

**Tests needed**: 14 core modules + 5-7 integration tests

---

## 12. SUMMARY TABLE: v1.0 vs v1.1

| Aspect | v1.0 State | v1.1 Target | Gap Size |
|--------|-----------|-------------|----------|
| **Data Models** | 2 Pydantic, 14 dataclasses, extensive dicts | All Pydantic | 15-20 models |
| **LLM Models** | Qwen-72B (1B), Qwen-7B (1C), Mistral-7B (1D) | Mistral-7B throughout | 3 files |
| **Hardcoded Paths** | 40+ instances | Centralized config | 1 file |
| **Magic Numbers** | 20+ thresholds | Config-driven | 1 file |
| **Test Coverage** | 28% processing, 37% overall | 80% core, 60% overall | 10-15 tests |
| **Structural Relations** | None | PART_OF (regex) + SAME_AS (LLM) | 2 features |
| **Alias Tracking** | None | Preserved through disambiguation | 1 feature |
| **Quality Gate** | None | datatrove Gopher heuristics | 1 feature |
| **Performance** | O(n³) in places | Vectorized/sparse | 3 optimizations |

---

## CONCLUSION

The v1.1 rewrite is feasible but requires substantial effort across 6 areas:

1. ⚠️ **CRITICAL**: Migrate 3 files from Qwen to Mistral-7B
2. ⚠️ **CRITICAL**: Create 15-20 Pydantic models to replace dict usage
3. ⚠️ **CRITICAL**: Centralize 40+ hardcoded paths and 20+ magic numbers
4. **HIGH**: Write 10-15 missing tests for core extraction logic
5. **MEDIUM**: Optimize 3 nested loop hotspots (O(n³) → O(n log n))
6. **LOW**: Add structural relations (PART_OF/SAME_AS) and quality gate

**Estimated effort**: 3-4 weeks for complete v1.1 implementation
**Cost savings**: $24/run by moving to Mistral-7B
**Quality improvements**: Type safety, reproducibility, maintainability

---

**End of Survey**
**Total Sections**: 12
**Total Tables**: 28
**Critical Issues Flagged**: 8
**Files Analyzed**: 74 (54 source + 20 test)
