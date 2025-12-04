# GraphRAG Pipeline Execution Map

**Purpose:** Document the actual execution order of the RAKG pipeline as implemented.

**Status:** As of Phase 1D completion (December 2024)

---

## Phase 0: Data Ingestion

**Goal:** Load academic papers and regulatory documents into standardized format.

### Execution Order:

1. **Raw Data Preparation** (manual)
   - MinerU parsed papers → `data/raw/academic/scopus_2023/MinerU_parsed_papers/`
   - Scopus metadata CSV → `data/raw/academic/scopus_2023/`

2. **Document Loading** (script: `src/phase0_data_ingestion/document_loader.py`)
   - Input: Raw markdown files + Scopus CSV
   - Output: `data/interim/academic/loaded_documents.json`
   - Combines paper content with metadata (DOI, authors, citations, etc.)

---

## Phase 1: Graph Construction

**Goal:** Build knowledge graph from documents via chunking → extraction → disambiguation → relations.

### Phase 1A: Semantic Chunking

**Script:** `src/phase1_graph_construction/chunk_processor.py`

**Input:**
- `data/interim/academic/loaded_documents.json`

**Process:**
- Core logic: `semantic_chunker.py` (SemanticChunker class)
- Splits documents into semantically coherent chunks using BGE-M3 embeddings
- Tracks metadata: chunk_id, document_id, position, word_count

**Output:**
- `data/interim/chunks/chunks_text.json` (chunk text + metadata)
- `data/processed/chunks/chunk_embeddings.npz` (BGE-M3 embeddings for retrieval)

**Test:** `tests/semantic_chunk_tester.py`

---

### Phase 1B: Entity Extraction

**Script:** `server_scripts/extract_entities_server.py` (production)
**Alternative:** `src/phase1_graph_construction/entity_processor.py` (original version)

**Input:**
- `data/interim/chunks/chunks_text.json`

**Process:**
- Core logic: `entity_extractor.py` (RAKGEntityExtractor class)
- LLM-based extraction (Together AI: Meta-Llama-3.1-70B-Instruct-Turbo)
- Parallel processing with checkpoint management
- Extracts: name, type, description, chunk_ids

**Output:**
- `data/interim/entities/pre_entities_raw.json` (raw LLM output)
- Checkpoint: `data/interim/entities/extraction_checkpoint.json`

**Test:** `tests/entity_extraction_test.py`

---

### Phase 1B+: Entity Filtering (Preprocessing)

**Script:** `src/phase1_graph_construction/filter_pre_entities.py`

**Input:**
- `data/interim/entities/pre_entities_raw.json`

**Process:**
- Statistical filtering (remove low-frequency noise)
- Type-based filtering (remove generic/vague entities)
- Quality filtering (length, format validation)

**Output:**
- `data/interim/entities/pre_entities_filtered.json`
- Reduces ~55K raw entities → ~15K high-quality entities

**Test:** None (standalone preprocessing step)

---

### Phase 1C: Entity Disambiguation

**Script:** `server_scripts/disambiguation_server.py` (GPU production version)
**Alternative:** `src/phase1_graph_construction/disambiguation_processor.py` (CPU version)

**Input:**
- `data/interim/entities/pre_entities_filtered.json`

**Process:**
- Core logic: `entity_disambiguator.py` (4-stage pipeline)
  1. **ExactDeduplicator:** Merge exact matches by (name, type)
  2. **FAISSBlocker:** Find embedding-similar candidates (BGE-M3 + FAISS GPU)
  3. **TieredThresholdFilter:** Type-specific similarity thresholds
  4. **SameJudge:** Heuristic disambiguation rules
- Utility: `src/utils/embedder.py` (BGEEmbedder), `src/utils/embed_processor.py`
- Assigns unique entity IDs, merges duplicate mentions

**Output:**
- `data/processed/entities/normalized_entities.json`
- `data/processed/entities/entity_id_lookup.json` (canonical entity mapping)
- `data/processed/embeddings/entity_embeddings.npz` (BGE-M3 embeddings)
- Reduces ~15K filtered → ~5K unique entities

**Test:** `tests/entity_disambiguator_test.py`

---

### Phase 1C+: Build Co-occurrence Matrix (Preprocessing)

**Script:** `src/phase1_graph_construction/build_entity_cooccurrence.py`

**Input:**
- `data/processed/entities/normalized_entities.json`
- `data/interim/chunks/chunks_text.json`

**Process:**
- Build entity co-occurrence graph from chunk-level co-mentions
- Type-aware filtering (semantic, concept, academic entity types)
- Utility: `src/utils/entity_type_classification.py`

**Output:**
- `data/processed/graph/entity_cooccurrence.json`
- Format: {entity_id: {neighbor_id: co-mention_count}}
- Used to focus relation extraction on likely-related entities

**Test:** None (standalone preprocessing step)

---

### Phase 1D: Relation Extraction

**Script:** `src/phase1_graph_construction/run_relation_extraction.py`
**Parallel Processor:** `server_scripts/relation_processor_server.py`

**Input:**
- `data/processed/entities/normalized_entities.json`
- `data/processed/graph/entity_cooccurrence.json`
- `data/interim/chunks/chunks_text.json`

**Process:**
- Core logic: `relation_extractor.py` (RAKGRelationExtractor class)
- LLM-based extraction (Together AI: Mistral-7B)
- Parallel processing with checkpoint and rate limiting
- Extracts: source_entity, target_entity, relation_type, description, evidence_chunks
- Utilities:
  - `src/utils/checkpoint_manager.py` (progress tracking)
  - `src/utils/rate_limiter.py` (API throttling)

**Output:**
- `data/processed/graph/entity_relations.json`
- Checkpoint: `data/processed/graph/relation_extraction_checkpoint.json`

**Test:** `tests/relation_extraction_test.py`, `tests/relation_extraction_parallel_test.py`

---

## Phase 2: Index Construction

**Status:** Partially implemented (embedding complete, validation/enrichment TODO)

### Phase 2A: Entity/Chunk Embedding

**Already completed in Phase 1:**
- Chunk embeddings: `data/processed/chunks/chunk_embeddings.npz` (Phase 1A)
- Entity embeddings: `data/processed/embeddings/entity_embeddings.npz` (Phase 1C)

**Embedder:** `src/utils/embedder.py` (BGEEmbedder class, BAAI/bge-m3, 1024-dim)

---

### Phase 2B: Entity Normalization (TODO)

**Script:** `src/phase2_index_construction/entity_normalizer.py`

**Purpose:** Final entity quality checks and canonical form standardization.

---

### Phase 2C: Triplet Validation (TODO)

**Script:** `src/phase2_index_construction/triplet_validator.py`

**Purpose:** Validate (entity1, relation, entity2) triplets for consistency.

---

### Phase 2D: Wikipedia Enrichment (TODO)

**Script:** `src/phase2_index_construction/wikipedia_enrichment.py`

**Purpose:** Augment entities with Wikipedia metadata.

---

## Phase 3: Operator Configuration (TODO)

**Purpose:** Configure retrieval operators for query processing.

**Scripts:**
- `src/phase3_operator_config/metadata_filters.py`
- `src/phase3_operator_config/retrieval_operators.py`

---

## Phase 4: Retrieval & Generation (TODO)

**Purpose:** Query interface for GraphRAG system.

**Scripts:**
- `src/phase4_retrieval_generation/prompt_builder.py`
- `src/phase4_retrieval_generation/generator.py`

---

## Utility Modules

**Location:** `src/utils/`

**Active Utilities:**
- `embedder.py` - BGE-M3 embedder (used in Phase 1A, 1C)
- `embed_processor.py` - Batch embedding with progress tracking
- `checkpoint_manager.py` - Checkpoint/resume for long-running tasks
- `rate_limiter.py` - Token bucket rate limiting for API calls
- `entity_type_classification.py` - Entity type categorization logic
- `logger.py` - Logging setup
- `neo4j_utils.py` - Neo4j graph database operations (TODO)

**Orphaned:**
- `add_entity_ids.py` - Utility script (git untracked, unclear usage)

---

## Server Scripts

**Location:** `server_scripts/` (root level, outside src/)

**Purpose:** Optimized production versions with parallelization.

**Active Scripts:**
1. `extract_entities_server.py` - Parallel entity extraction (Phase 1B)
2. `disambiguation_server.py` - GPU-accelerated disambiguation (Phase 1C)
3. `relation_processor_server.py` - Parallel relation extraction (Phase 1D)

**Status:** These duplicate functionality from `src/phase1_graph_construction/*_processor.py` but add:
- GPU support (disambiguation)
- Better parallelization
- Production-grade error handling

**Note:** Consider consolidation in future refactor.

---

## Configuration

**Config Files:**
- `configs/config.py` - Central configuration (currently only used by dlapiper_scraper.py)
- `environment-cpu.yml` - Conda environment for CPU execution
- `environment-gpu.yml` - Conda environment for GPU execution

**Prompts:**
- `src/prompts/prompts.py` - LLM prompt templates

---

## Data Flow Summary

```
RAW PAPERS (MinerU .md + Scopus .csv)
  ↓
[Phase 0] document_loader.py
  ↓
LOADED DOCUMENTS (.json)
  ↓
[Phase 1A] chunk_processor.py
  ↓
CHUNKS (.json) + CHUNK EMBEDDINGS (.npz)
  ↓
[Phase 1B] extract_entities_server.py
  ↓
PRE-ENTITIES RAW (.json)
  ↓
[Phase 1B+] filter_pre_entities.py
  ↓
PRE-ENTITIES FILTERED (.json)
  ↓
[Phase 1C] disambiguation_server.py
  ↓
NORMALIZED ENTITIES (.json) + ENTITY EMBEDDINGS (.npz) + ID LOOKUP (.json)
  ↓
[Phase 1C+] build_entity_cooccurrence.py
  ↓
CO-OCCURRENCE MATRIX (.json)
  ↓
[Phase 1D] run_relation_extraction.py
  ↓
ENTITY RELATIONS (.json)
  ↓
[Phase 2-4] TODO
```

---

## Execution Commands (Phase 1 Complete Pipeline)

```bash
# Phase 0: Load documents
python src/phase0_data_ingestion/document_loader.py

# Phase 1A: Chunk documents
python src/phase1_graph_construction/chunk_processor.py

# Phase 1B: Extract entities
python server_scripts/extract_entities_server.py

# Phase 1B+: Filter entities
python src/phase1_graph_construction/filter_pre_entities.py \
  --input data/interim/entities/pre_entities_raw.json \
  --output data/interim/entities/pre_entities_filtered.json

# Phase 1C: Disambiguate entities (GPU recommended)
python server_scripts/disambiguation_server.py \
  --input data/interim/entities/pre_entities_filtered.json \
  --output data/processed/entities/normalized_entities.json

# Phase 1C+: Build co-occurrence matrix
python src/phase1_graph_construction/build_entity_cooccurrence.py

# Phase 1D: Extract relations
python src/phase1_graph_construction/run_relation_extraction.py \
  --entities data/processed/entities/normalized_entities.json \
  --chunks data/interim/chunks/chunks_text.json \
  --cooccurrence data/processed/graph/entity_cooccurrence.json \
  --output data/processed/graph/entity_relations.json
```

---

## Key Design Patterns

### Three-File Pattern (Phase 1 Subphases)

Each subphase follows:
1. **Core Logic:** `*_extractor.py` or `*_disambiguator.py` or `*_chunker.py` (classes)
2. **Processor:** `*_processor.py` or `run_*.py` (main() entry point)
3. **Test:** `tests/*_test.py` (pytest)

**Exceptions:**
- `filter_pre_entities.py` - Preprocessing step (has main(), no core class)
- `build_entity_cooccurrence.py` - Preprocessing step (has main(), no core class)

---

## Testing

**Test Location:** `tests/`

**Coverage:**
- Phase 0: `document_loader_test.py`
- Phase 1A: `semantic_chunk_tester.py`
- Phase 1B: `entity_extraction_test.py`
- Phase 1C: `entity_disambiguator_test.py`, `embedder_test.py`
- Phase 1D: `relation_extraction_test.py`, `relation_extraction_parallel_test.py`
- Debug: `test_api_debug.py`

**Run Tests:**
```bash
pytest tests/ -v
```

---

## Recent Changes (This Session)

**2024-12-04 Cleanup:**
- ✓ Fixed filename typos: `matrcher→matcher`, `extration→extraction`
- ✓ Removed duplicate `src/phase2_index_construction/embedder.py` (empty file)
- ✓ Removed empty directories: `configs/prompts/`, `data/enriched/`, `data/external/`

---

**Last Updated:** December 4, 2024
**Pipeline Status:** Phase 1 (A-D) complete, Phase 2-4 in planning
