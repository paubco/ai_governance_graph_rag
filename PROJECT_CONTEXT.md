# Graph_Rag Project - Context Documentation

**Last Updated**: 2025-10-21
**Branch**: main
**Status**: Active Development - Refactoring Phase
**Python Version**: 3.10

---

## Project Overview

Graph_Rag is a **Graph-based Retrieval-Augmented Generation (RAG) system** that extracts, enriches, and links academic research metadata to construct knowledge graphs. The system focuses on AI policy and data governance research from Scopus.

**Core Pipeline**: Academic Papers → Text Chunking → Entity Annotation (DBpedia) → Entity Enrichment → Validation → Neo4j Graph Storage

---

## Current Architecture

### Directory Structure
```
Graph_Rag/
├── src/
│   ├── main.py                    # Pipeline orchestrator
│   ├── data_intake.py             # CSV loading & normalization
│   ├── text_preprocessing.py      # Text chunking utilities
│   ├── dbpedia_pipeline.py        # Entity annotation & enrichment
│   ├── build_graph.py             # [EMPTY - Not implemented]
│   └── utils/
│       ├── config.py              # Configuration management
│       ├── logger.py              # Dual-logger setup
│       ├── neo4j_utils.py         # Graph database interface
│       └── dbpedia_cleaner.py     # Entity validation heuristics
├── data/
│   ├── abstract_sample_data.csv   # Input: 23 papers
│   ├── annotations_raw.csv        # Output: Raw entities
│   └── entities_clean.csv         # Output: Validated entities
├── logs/                          # Application & cleaning logs
└── requirements.txt               # Python dependencies
```

### Data Flow

**Step 1 - Data Intake** ([data_intake.py](src/data_intake.py))
- Loads `abstract_sample_data.csv` (23 Scopus papers on AI regulation)
- Splits into `metadata_df` (authorship, journal, DOI) and `text_df` (titles, abstracts)
- Normalizes nested fields (affiliations, links)

**Step 2 - Text Preprocessing** ([text_preprocessing.py](src/text_preprocessing.py))
- Chunks abstracts into 150-word segments with 20-word overlap
- Result: 23 documents → 41 chunks for entity extraction

**Step 3 - Entity Annotation** ([dbpedia_pipeline.py](src/dbpedia_pipeline.py))
- Calls DBpedia Spotlight API with confidence=0.50
- Extracts entities with URIs (e.g., "European Union" → `dbpedia.org/resource/European_Union`)
- Saves `annotations_raw.csv`

**Step 4 - Entity Enrichment** ([dbpedia_pipeline.py](src/dbpedia_pipeline.py))
- Fetches full DBpedia JSON for each entity
- Parses: labels, abstracts, types (ontology classes), relations, Wikidata IDs
- Optimized: Fetch each unique URI only once

**Step 5 - Entity Validation** ([utils/dbpedia_cleaner.py](src/utils/dbpedia_cleaner.py))
- Filters entities using multi-layered heuristics:
  - **Domain whitelist**: Company, GovernmentAgency, Software, Book, etc.
  - **Keyword matching**: "regulation", "algorithm", "governance", "AI", etc.
  - **Adaptive thresholds**: Single-word (0.96), two-word (0.75), multi-word (0.70)
  - **Shape rules**: Reject lowercase singles, accept acronyms (AI, OECD)
- Saves `entities_clean.csv` and logs decisions to `entity_cleaning.log`

**Step 6 - Neo4j Storage** ([utils/neo4j_utils.py](src/utils/neo4j_utils.py))
- Creates graph nodes: Publication, Author, Journal, Affiliation
- Creates relationships: WROTE, PUBLISHED_IN, AFFILIATED_WITH
- Cloud instance: `neo4j+s://d333fc3d.databases.neo4j.io`

---

## Key Modules

### [main.py](src/main.py) - Pipeline Orchestrator (117 lines)
**Role**: Coordinates all processing steps sequentially

**Current Implementation**:
```python
# Step 1: Load data
metadata_df, text_df = load_local_data()

# Step 1b: Chunk text
text_chunks = chunk_dataframe(text_df)

# Step 2: Annotate with DBpedia Spotlight
raw_annotations = text_chunks.apply(annotate_text_spotlight)

# Step 3: Enrich & clean entities
clean_entities = extract_and_clean_entities(raw_annotations)  # ⚠️ ISSUE

# Step 4: Load to Neo4j
load_metadata_to_neo4j(driver, metadata_df)
```

**Known Issues**:
- Line 85: Calls `extract_and_clean_entities()` which doesn't exist in dbpedia_pipeline.py
- Line 36: `annotate_text_spotlight()` called with 1 arg, requires 3 (scopus_id, chunk_id, text)

### [dbpedia_pipeline.py](src/dbpedia_pipeline.py) - Core Entity Processing (155 lines)
**Key Functions**:
- `annotate_text_spotlight(scopus_id, chunk_id, text, confidence=0.50)` → Calls DBpedia Spotlight API
- `fetch_dbpedia_entity(uri)` → Retrieves RDF/JSON data
- `parse_dbpedia_entity(entity_json)` → Extracts structured fields
- `enrich_annotations(raw_annotations)` → Wraps annotations with full DBpedia data
- `clean_parsed_entities(entity_json_list)` → Validates entities via heuristics

**API Behavior**:
- Endpoint: `https://api.dbpedia-spotlight.org/en/annotate`
- Timeout: 30 seconds per request
- Error handling: Returns empty list on failure, logs warnings

### [utils/dbpedia_cleaner.py](src/utils/dbpedia_cleaner.py) - Entity Validation (140 lines)
**Validation Logic** (`is_valid_entity()`):
1. Empty surface form → REJECT
2. Similarity below threshold (0.96/0.75/0.70) → REJECT
3. Single lowercase word → REJECT
4. No types AND no keyword match → REJECT
5. Types not in domain whitelist → REJECT
6. Otherwise → ACCEPT

**Domain Whitelists**:
- **Type whitelist**: Company, GovernmentAgency, Software, Organisation, AcademicDiscipline, Book, WrittenWork, Technology, etc. (30+ classes)
- **Keywords**: "data", "algorithm", "neural network", "regulation", "law", "governance", "compliance", "blockchain", "economics", etc. (40+ terms)

**Logging**: All accept/reject decisions logged to `logs/entity_cleaning.log` with entity details

### [utils/config.py](src/utils/config.py) - Configuration (43 lines)
**Key Settings**:
```python
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"
NEO4J_URI = os.getenv("NEO4J_URI")
CHUNK_SIZE = 150  # Words per chunk
CHUNK_OVERLAP = 20  # Overlapping words
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
```

**Recent Changes**:
- Fixed DATA_PATH to use absolute paths (was relative)
- Updated chunking: 50→150 words, 10→20 overlap (better context)

### [utils/logger.py](src/utils/logger.py) - Logging (35 lines)
**Two Logger Instances**:
- `logger` → Main application events (`logs/project.log`)
- `cleaning_logger` → Entity validation decisions (`logs/entity_cleaning.log`)

**Log Levels**: DEBUG for files, INFO for console

---

## Recent Changes (Git Status)

**Modified Files** (Uncommitted):
- `requirements.txt` → Added ipykernel dependency
- `src/data_intake.py` → Refactored logger imports
- `src/main.py` → Complete pipeline rewrite
- `src/utils/config.py` → Fixed paths, updated chunk sizes
- `src/utils/logger.py` → Refactored to dual-logger pattern

**Deleted Files**:
- `src/enrich_entities.py` → Merged into dbpedia_pipeline.py
- `src/link_entities.py` → Replaced by Neo4j utilities

**New Untracked Files**:
- `src/__init__.py` → Package initialization
- `src/dbpedia_pipeline.py` → New core module
- `src/text_preprocessing.py` → New chunking utilities
- `src/utils/__init.py__` → ⚠️ MALFORMED (wrong filename)
- `src/utils/dbpedia_cleaner.py` → New validation module
- `src/notebooks/dbpedia_pipeline_test.ipynb` → Testing notebook

**Commit History**:
- `45516af` - "added main to structure" (HEAD)
- `70b7242` - "refactored original prototyping notebooks"
- `89a802b` - "Initial commit environment and logger setup"

---

## Dependencies (requirements.txt)

| Package | Version | Purpose |
|---------|---------|---------|
| python-dotenv | 1.0.1 | Environment variables |
| pandas | latest | Data manipulation |
| requests | 2.32.3 | HTTP API calls |
| sentence-transformers | 3.1.1 | Embeddings |
| spacy | 3.7.5 | NLP pipeline |
| neo4j | 6.0.2 | Graph database |
| networkx | 3.3 | Graph analysis |
| rdflib | 7.0.0 | RDF processing |
| farm-haystack | 1.26.3 | RAG framework |
| jupyter, ipykernel | latest | Notebooks |

---

## Environment Variables (.env)

```
DATA_PATH=data/
NEO4J_URI=neo4j+s://d333fc3d.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=vfplvQhlsLvW4WIajmdUomOJF9FZ6P28gEwPP33abxU
SCOPUS_API_KEY=2a609096808a7a3452824d7c57388036
SPACY_MODEL=en_core_web_sm
DEBUG_MODE=true
```

---

## Known Issues & Blockers

### Critical Issues (Prevent Execution):

1. **Function Import Error** ([main.py:85](src/main.py#L85))
   - Calls `extract_and_clean_entities()` which doesn't exist
   - Should use `enrich_annotations()` + `clean_parsed_entities()` instead

2. **Function Signature Mismatch** ([main.py:36](src/main.py#L36))
   - Calls `annotate_text_spotlight(row["chunk_text"])`
   - Requires `(scopus_id, chunk_id, text, confidence, support)` parameters
   - Missing 2 required positional arguments

3. **Malformed Module** ([src/utils/__init.py__](src/utils/__init.py__))
   - Wrong filename (double underscores at end)
   - Should rename to `__init__.py`

### Non-Critical Issues:

4. **Empty Module** ([build_graph.py](src/build_graph.py))
   - File exists but is empty (0 lines)
   - Graph building logic not implemented

5. **API Timeout Issues**
   - DBpedia requests timeout on large entities (United States, China)
   - No retry logic or caching implemented

6. **No Documentation** ([docs/readme.md](docs/readme.md))
   - Empty documentation file

---

## Testing & Validation

**Notebook Testing** ([notebooks/dbpedia_pipeline_test.ipynb](src/notebooks/dbpedia_pipeline_test.ipynb)):
- Tested on first 20 chunks (of 41 total)
- Result: 159 enriched entities extracted
- Entity types: Company, GovernmentAgency, Place, Software, Music (diverse, needs filtering)
- Validation reduced set significantly via domain heuristics

**Current Dataset**:
- **Input**: 23 academic papers from Scopus (AI regulation domain)
- **Papers**: Topics include EU AI Act, data governance, algorithmic accountability
- **Chunks**: 41 overlapping segments
- **Output**: ~150+ validated entities after cleaning

---

## Technical Decisions

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| **DBpedia Spotlight** | Open-source, well-maintained, supports REST API | Lower precision than commercial NER tools |
| **150-word chunks** | Balance between context and entity boundaries | Increased from 50 (more context) |
| **Domain whitelist** | Improves precision for AI/policy domain | Manual curation, maintenance burden |
| **Adaptive thresholds** | Different confidence for single vs. multi-word | More complex heuristics |
| **Neo4j cloud** | Managed service, no local setup | Cost, network dependency |
| **Word-level chunking** | Simple, fast, reproducible | May split sentences awkwardly |

---

## Next Steps (Anticipated Changes)

Based on current state, short-term changes likely include:

1. **Fix Critical Bugs**:
   - Correct function calls in main.py (lines 36, 85)
   - Rename `__init.py__` to `__init__.py`

2. **Complete Implementation**:
   - Implement `build_graph.py` logic for entity-to-graph conversion
   - Add retry logic for DBpedia API timeouts

3. **Testing & Validation**:
   - Run full pipeline end-to-end
   - Validate Neo4j graph structure
   - Test on larger dataset (100+ papers)

4. **Documentation**:
   - Populate `docs/readme.md`
   - Add docstrings to key functions

5. **Optimization** (if scaling):
   - Implement async API calls
   - Add entity caching (avoid re-fetching URIs)
   - Parallelize chunking and annotation

---

## Contact & Resources

**Project Type**: Master's Thesis (UOC TFM)
**Domain**: AI Regulation & Data Governance
**Data Source**: Scopus API
**External APIs**: DBpedia Spotlight, DBpedia RDF/JSON
**Cloud Services**: Neo4j Aura (cloud database)

**Key URLs**:
- DBpedia Spotlight: `https://api.dbpedia-spotlight.org/en/annotate`
- DBpedia Data: `https://dbpedia.org/data/{entity}.json`
- Neo4j Instance: `neo4j+s://d333fc3d.databases.neo4j.io`

---

## Summary

The Graph_Rag project is in **active refactoring** after migrating from prototype notebooks to production-ready modules. The architecture is well-structured with clear separation of concerns, sophisticated entity validation, and proper logging. However, **the pipeline currently has critical bugs** (function mismatches) that prevent execution. Once these are fixed, the system should work end-to-end from CSV input to Neo4j graph storage.

**Strengths**: Modular design, robust entity validation, centralized configuration
**Blockers**: Function signature mismatches, malformed module names
**Readiness**: ~80% complete - needs debugging before full deployment
