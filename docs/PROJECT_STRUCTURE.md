# Graph RAG Project Structure

## Overview
This is a Graph-based Retrieval Augmented Generation (Graph-RAG) system for academic paper analysis, focused on AI regulation research. The system extracts entities and relationships from academic papers, builds a knowledge graph in Neo4j, and enables semantic retrieval and generation.

## Quick Reference: File Naming Conventions

### Key File Names You'll Encounter
- **`scopus_csv_loader.py`** - Loads Scopus CSV exports
- **`paper_to_scopus_metadata_matrcher.py`** - [Note: "matrcher" is a typo, should be "matcher"] Matches parsed papers to Scopus
- **`dlapiper_scraper.py`** - Scrapes DLA Piper AI Laws website
- **`scopus_export_{YEAR}_raw.csv`** - Raw Scopus export (e.g., scopus_export_2023_raw.csv)
- **`scopus_metadata_{YEAR}.csv`** - Cleaned Scopus metadata (e.g., scopus_metadata_2023.csv)
- **`paper_scopus_matches.csv`** - Matching results between MinerU papers and Scopus
- **`paper_mapping.json`** - Maps MinerU paper IDs to metadata (located in MinerU_parsed_papers/)

### Important Path Patterns
- Raw data: `data/raw/{source}/{year}/`
- Interim data: `data/interim/{source}/`
- Processed data: `data/processed/{type}/`
- Logs: `data/logs/{component}/`

## Project Root
```
Graph_Rag/
├── archive/              # Archived v1 implementation (DBpedia approach)
├── configs/              # Configuration files and prompts
├── data/                 # Data directories (organized by pipeline stage)
├── docs/                 # Documentation
├── logs/                 # Application logs
├── src/                  # Source code (organized by pipeline phase)
├── .env                  # Environment variables (API keys, credentials)
├── .gitignore
├── environment.yml       # Conda environment specification
└── requirements.txt      # Python dependencies
```

## Source Code Structure (`src/`)

### IMPORTANT: Current Implementation Status
This project is in **active migration from V1 to V6**. Many files exist as placeholders (empty or minimal implementation). Below is the ACTUAL current state of each file.

### Pipeline Phases

#### **Phase 1: Graph Construction** (`src/phase1_graph_construction/`)
**STATUS**: ✅ ACTIVE DEVELOPMENT - Core data acquisition scripts are implemented

**Implemented Files:**

1. **`scopus_csv_loader.py`** (314 lines) - ✅ FULLY IMPLEMENTED
   - **Class**: `ScopusCSVLoader`
   - **Purpose**: Load and clean Scopus export CSV files (handles UTF-8 BOM encoding)
   - **Key Methods**:
     - `load()` - Load Scopus CSV with UTF-8-sig encoding
     - `clean_and_structure()` - Extract 15+ metadata fields (EID, DOI, title, authors, abstract, keywords, citations, etc.)
     - `validate_data()` - Generate statistics on data quality
     - `save()` - Save cleaned data to `data/interim/academic/`
     - `run_full_pipeline()` - Complete ETL pipeline
   - **Input**: `data/raw/academic/scopus_{YEAR}/scopus_export_{YEAR}_raw.csv`
   - **Output**: `data/interim/academic/scopus_metadata_{YEAR}.csv`
   - **CLI Usage**: `python scopus_csv_loader.py [YEAR]`

2. **`paper_to_scopus_metadata_matrcher.py`** (388 lines) - ✅ FULLY IMPLEMENTED
   - **Class**: `MinerUMatcher`
   - **Purpose**: Link MinerU-parsed papers to Scopus metadata using DOI/title/abstract matching
   - **Matching Strategy** (3-tier fallback):
     1. DOI extraction from `content_list.json` or markdown (confidence=1.0)
     2. Multi-language title matching using SequenceMatcher (threshold=0.85)
     3. Abstract similarity matching (threshold=0.90, fallback)
   - **Key Methods**:
     - `extract_doi()` - Extract DOI from PDF footer/header/content
     - `extract_titles()` - Get all titles from content_list.json (handles multi-language)
     - `extract_abstract()` - Parse abstract from full.md
     - `match_by_titles()` - Fuzzy title matching across Scopus dataset
     - `match_by_abstract()` - Fallback abstract similarity
     - `match_all()` - Run complete matching pipeline with statistics
   - **Input**:
     - `data/raw/academic/scopus_2023/MinerU_parsed_papers/` (paper_001, paper_002, etc.)
     - `data/raw/academic/scopus_2023/scopus_export_2023_raw.csv`
   - **Output**:
     - `data/interim/academic/paper_scopus_matches.csv` - Match results report
     - `data/interim/academic/manual_review_needed.csv` - Unmatched papers
     - Updates `paper_mapping.json` with Scopus metadata
   - **CLI Usage**: `python paper_to_scopus_metadata_matrcher.py`

3. **`dlapiper_scraper.py`** (453 lines) - ✅ FULLY IMPLEMENTED
   - **Class**: `DLAPiperScraper`
   - **Purpose**: Scrape AI regulation content from DLA Piper AI Laws of the World
   - **Architecture**:
     - Get country list from dropdown (extracts both code and name)
     - Scrape each country page with accordion sections
     - Extract sections, subsections, notes, and links
     - Save as JSON files per country
   - **Key Methods**:
     - `get_country_list()` - Parse country dropdown from main page
     - `scrape_country()` - Scrape all sections for a specific country
     - `_extract_sections()` - Parse accordion items
     - `_parse_section()` - Extract title, content, notes, subsections, links
     - `_extract_subsections()` - Handle US states, Canadian provinces (H4 headings)
     - `_extract_notes()` - Country-specific notes (EU countries)
     - `save_country_data()` - Save to `{COUNTRY_CODE}.json`
     - `scrape_all_countries()` - Full pipeline with rate limiting
   - **Configuration**: Uses `configs/config.py` → `SCRAPER_CONFIG`
   - **Rate Limiting**: 2 seconds between requests (configurable)
   - **Input**: Web scraping from https://intelligence.dlapiper.com/artificial-intelligence/
   - **Output**: `data/raw/dlapiper/{COUNTRY_CODE}.json` (one per country)
   - **Logging**: `data/logs/scraper/dlapiper_scraper_{DATE}.log`
   - **CLI Usage**:
     - Test single country: `test_single_country('FR')`
     - Production: `python dlapiper_scraper.py`

**Deleted Files** (V1 → V6 migration):
- `entity_extraction.py` - Removed during refactoring
- `relation_extraction.py` - Removed during refactoring
- `text_processing.py` - Removed during refactoring
- `data_acquisition.py` - Removed during refactoring

---

#### **Phase 2: Index Construction** (`src/phase2_index_construction/`)
**STATUS**: ⚠️ PLACEHOLDER - All files are empty stubs (1 line each)

**Files** (NOT YET IMPLEMENTED):
- `embedder.py` - Empty placeholder
- `entity_normalizer.py` - Empty placeholder
- `wikipedia_enrichment.py` - Not yet created
- `triplet_validator.py` - Empty placeholder

**Intended Purpose** (when implemented):
- Generate embeddings for entities and text chunks
- Normalize entity names and deduplicate
- Enrich entities with Wikipedia data
- Validate extracted knowledge triplets

---

#### **Phase 3: Operator Configuration** (`src/phase3_operator_config/`)
**STATUS**: ⚠️ PLACEHOLDER - All files are empty stubs (1 line each)

**Files** (NOT YET IMPLEMENTED):
- `retrieval_operators.py` - Empty placeholder
- `metadata_filters.py` - Not yet created

**Intended Purpose** (when implemented):
- Configure graph traversal and retrieval operators
- Implement metadata filters (year, citations, journal, etc.)

---

#### **Phase 4: Retrieval & Generation** (`src/phase4_retrieval_generation/`)
**STATUS**: ⚠️ PLACEHOLDER - All files are empty stubs (1 line each)

**Files** (NOT YET IMPLEMENTED):
- `prompt_builder.py` - Not yet created
- `generator.py` - Empty placeholder

**Intended Purpose** (when implemented):
- Build prompts for LLM generation
- Generate answers using retrieved context from Neo4j

---

### Utilities (`src/utils/`)
**STATUS**: Mixed (one implemented, others empty)

**Implemented Files:**

1. **`neo4j_utils.py`** (73 lines) - ✅ PARTIALLY IMPLEMENTED (V1 legacy code)
   - **Functions**:
     - `get_driver(uri, user, password)` - Create Neo4j driver connection
     - `create_publication_graph(tx, row)` - Create Publication, Author, Journal, Affiliation nodes
     - `load_metadata_to_neo4j(driver, metadata_df)` - Bulk load metadata to Neo4j
     - `close_driver(driver)` - Close Neo4j connection
   - **Graph Schema**:
     - Nodes: `Publication`, `Author`, `Journal`, `Affiliation`
     - Relationships: `WROTE`, `PUBLISHED_IN`, `AFFILIATED_WITH`
   - **Note**: This is V1 legacy code, may need refactoring for V6

**Empty Files:**
- `logger.py` - Empty placeholder
- `config.py` - Empty placeholder (config moved to `configs/config.py`)
- `dbpedia_cleaner.py` - V1 legacy utility (DBpedia no longer used)

---

### Main Entry Point
- **`src/main.py`** - Empty placeholder (not yet implemented)

### Storage
- **`src/storage/`** - Directory exists, no files yet

### Notebooks (`src/notebooks/`)
**STATUS**: Development/testing notebooks (not tracked in git)

**Files:**
- `scraping_tests.ipynb` - Testing notebook for web scraping (3.6 KB)

## Data Organization (`data/`)

### Data Pipeline Flow
```
raw/ → interim/ → processed/ → enriched/
```

### Directory Structure

#### **Raw Data** (`data/raw/`)
- `academic/scopus_2023/` - Scopus exports and parsed papers
  - `MinerU_parsed_papers/` - Papers parsed by MinerU (paper_001 to paper_025+)
  - CSV files with Scopus metadata

#### **Interim Data** (`data/interim/`)
- `academic/paper_scopus_matches.csv` - Matched papers with Scopus metadata
- Intermediate processing results

#### **Processed Data** (`data/processed/`)
Organized by data type:
- `chunks/` - Text chunks from papers
- `embeddings/` - Vector embeddings
- `entities/` - Extracted entities
- `graph/` - Graph structure exports
- `metadata/` - Paper and entity metadata

#### **Enriched Data** (`data/enriched/`)
- Wikipedia-enriched entities
- External knowledge integrations

#### **Logs** (`data/logs/`)
- `extraction/` - Entity/relation extraction logs
- `scraper/` - Web scraping logs

## Configuration (`configs/`)

### Main Configuration File

**`configs/config.py`** (131 lines) - ✅ FULLY IMPLEMENTED
- **Purpose**: Centralized configuration for the entire pipeline
- **Architecture**: Loads secrets from `.env`, defines application logic in Python

**Key Configuration Sections**:

1. **Base Paths** (from .env):
   - `BASE_DIR` - Project root
   - `DATA_PATH` - Main data directory

2. **Derived Paths** (calculated):
   - Data directories: `RAW_DATA_PATH`, `INTERIM_DATA_PATH`, `PROCESSED_DATA_PATH`, `EXTERNAL_DATA_PATH`
   - Data sources: `DLAPIPER_RAW_PATH`, `WIKIPEDIA_RAW_PATH`, `ACADEMIC_RAW_PATH`
   - Processed outputs: `ENTITIES_PATH`, `EMBEDDINGS_PATH`, `GRAPH_DATA_PATH`
   - Logs: `LOGS_PATH`, `SCRAPER_LOGS_PATH`, `EXTRACTION_LOGS_PATH`

3. **`SCRAPER_CONFIG`** - DLA Piper scraper settings:
   - `base_url` - DLA Piper AI Laws URL
   - `output_dir` - Where to save scraped data
   - `delay_between_requests` - 2 seconds (rate limiting)
   - `timeout` - 10 seconds
   - `retry_attempts` - 3
   - `headers` - User-Agent for requests

4. **`EXTRACTION_CONFIG`** - Entity extraction settings:
   - `model_name` - "mistralai/Mistral-7B-Instruct-v0.1"
   - `api_key` - Together AI API key (from .env)
   - `temperature` - 0.0 (deterministic)
   - `max_tokens` - 2048
   - `chunk_size` - 512
   - `chunk_overlap` - 50

5. **`EMBEDDING_CONFIG`** - Embedding settings:
   - `model_name` - "BAAI/bge-m3"
   - `dimension` - 1024
   - `batch_size` - 32
   - `device` - "cuda" or "cpu"

6. **`NEO4J_CONFIG`** - Graph database settings:
   - `uri` - Neo4j connection URI (from .env, default: bolt://localhost:7687)
   - `user` - Neo4j username (from .env, default: neo4j)
   - `password` - Neo4j password (from .env, REQUIRED)
   - `database` - "neo4j"

7. **`LOGGING_CONFIG`** - Logging configuration:
   - Console handler (INFO level)
   - Rotating file handler (DEBUG level, 10MB, 5 backups)
   - Output: `data/logs/pipeline.log`

**Note**: All directories are automatically created on import.

### Prompts Directory
- **`prompts/`** - Directory exists for LLM prompt templates (not yet populated)

## Archive (`archive/`)

### V1 DBpedia Approach (`archive/v1_dbpedia_approach/`)
Initial implementation using DBpedia for entity linking (deprecated).

**Files:**
- `main_old.py` - Original main script
- `dbpedia_pipeline.py` - DBpedia integration pipeline
- `data_intake.py` - Data loading utilities
- `text_preprocessing.py` - Text preprocessing

## Key Technologies

### Core Stack
- **Graph Database**: Neo4j (v5.0+)
- **LLM Provider**: Together AI
- **Embeddings**: FlagEmbedding, sentence-transformers
- **NLP**: NLTK, langdetect, deep-translator
- **Data**: pandas, numpy
- **Web Scraping**: requests, beautifulsoup4

### Data Sources
- **Scopus**: Academic paper metadata via pybliometrics
- **MinerU**: PDF parsing for academic papers
- **DLA Piper**: Legal resources (web scraping)
- **Wikipedia**: Entity enrichment

## Development Workflow

### Current Status (based on git status)
- **Modified**: `requirements.txt`
- **Deleted**: Original phase1 scripts (data_acquisition, entity_extraction, relation_extraction, text_processing)
- **New/Untracked**:
  - `configs/` directory
  - `data/interim/` directory
  - `src/notebooks/` directory
  - Phase 1 new scripts (dlapiper_scraper, paper_to_scopus_metadata_matrcher, scopus_csv_loader)

### Recent Commits
- Complete Pipeline V6 migration (Day 1)
- Fix critical import errors and consolidate codebase structure
- Refactored original prototyping notebooks
- Added main to structure

## Git Branch Info
- **Current branch**: `main`
- **Main development branch**: Not specified (likely `main`)

## Environment Setup

### Required Environment Variables (`.env`)
- Neo4j credentials (URI, username, password)
- Together AI API key
- Scopus API keys
- Other API credentials

### Installation
```bash
# Using conda
conda env create -f environment.yml

# Or using pip
pip install -r requirements.txt
```

## Notebooks (`src/notebooks/`)
Development and prototyping notebooks (untracked in git).

## Implementation Status Summary

### ✅ FULLY IMPLEMENTED (Ready to Use)
1. **Phase 1: Data Acquisition**
   - ✅ Scopus CSV loading and cleaning (`scopus_csv_loader.py`)
   - ✅ MinerU paper to Scopus matching (`paper_to_scopus_metadata_matrcher.py`)
   - ✅ DLA Piper web scraping (`dlapiper_scraper.py`)
   - ✅ Centralized configuration system (`configs/config.py`)

2. **Utilities**
   - ✅ Neo4j connection utilities (V1 legacy, may need updates for V6)

### ⚠️ PLACEHOLDERS (Not Yet Implemented)
1. **Phase 2: Index Construction** - All files are empty stubs
2. **Phase 3: Operator Configuration** - All files are empty stubs
3. **Phase 4: Retrieval & Generation** - All files are empty stubs
4. **Main Pipeline** - `src/main.py` is empty

### 📊 Data Status
- **Raw Data**: Scopus CSV exports and MinerU parsed papers (paper_001 to paper_025+)
- **Interim Data**: Matching results available in `paper_scopus_matches.csv`
- **Processed Data**: Directories created but empty
- **Graph Database**: Neo4j utilities exist, schema may need V6 updates

### 🚧 Current Development Phase
- **Phase**: V1 → V6 Migration (Day 1 completed)
- **Focus**: Data acquisition layer is complete
- **Next Steps**: Implement Phase 2 (embeddings, entity normalization, enrichment)

## Critical Notes for Online Claude

1. **File Naming**: `paper_to_scopus_metadata_matrcher.py` has a typo ("matrcher" not "matcher") - this is the actual filename in the codebase
2. **Empty Files**: If you see a file with only 1 line, it's a placeholder - don't assume functionality exists
3. **Configuration**: Always use `configs/config.py` (NOT `src/utils/config.py` which is empty)
4. **Neo4j Code**: `src/utils/neo4j_utils.py` is V1 legacy code - may need refactoring for V6 schema
5. **Migration Context**: Original Phase 1 files (entity_extraction, relation_extraction, text_processing, data_acquisition) were deleted and replaced with new architecture
