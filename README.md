# Architecture

**Project**: AI Governance GraphRAG Pipeline  
**Version**: 1.1  
**Last Updated**: December 18, 2025

---

## 1. Overview

### 1.1 Goal

Build a knowledge graph for AI governance research that enables cross-jurisdictional regulatory queries by combining 48 regulatory documents with 158 academic papers. The system implements entity-centric corpus retrospective retrieval following RAKG methodology (Zhang et al., 2025), adapted for the regulatory compliance domain.

### 1.2 Data Sources

| Source | Content | Count | Origin |
|--------|---------|-------|--------|
| **Regulations** | Jurisdiction metadata + legal text | 48 | DLA Piper (2024) web scrape |
| **Academic Papers** | Scopus metadata CSV + PDFs (MinerU-parsed) | 158 | Scopus export (142/148 matched to metadata) |
| **Derived Metadata** | Authors, Journals, References | 572 / 119 / 1,513 | Scopus CSV |

### 1.3 Tech Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| **LLM (Extraction)** | Qwen-72B via Together.ai | Phase 1B entity extraction |
| **LLM (Disambiguation)** | Qwen-7B via Together.ai | Phase 1C |
| **LLM (Relations)** | Mistral-7B via Together.ai | Phase 1D (Qwen has JSON bug) |
| **LLM (Generation)** | Configurable (Claude/Qwen) | Phase 3 |
| **Embeddings** | BGE-M3 (1024-dim, multilingual) | All phases |
| **Graph DB** | Neo4j 5.12 Enterprise + GDS | Docker container, PCST algorithm |
| **Vector Store** | FAISS (HNSW) | Entity + chunk indices |
| **Hardware** | RTX 3060 GPU (12GB VRAM) | Local embedding computation |

### 1.4 Methodology Sources

| Component | Source | Reference |
|-----------|--------|-----------|
| Entity extraction | RAKG | Zhang et al. (2025) |
| Entity disambiguation | Adapted RAKG + Fellegi-Sunter | Papadakis et al. (2021), Fellegi & Sunter (1969) |
| Relation extraction | RAGulating Compliance | Agarwal et al. (2025) |
| Retrieval | RAKG corpus retrospective | Zhang et al. (2025) |
| Subgraph expansion | Steiner Tree | Neo4j GDS |
| Chunk diversity | MMR | Carbonell & Goldstein (1998) |
| Embeddings | BGE-M3 | Chen et al. (2024) |
| Evaluation | RAGAS | Es et al. (2023) |

---

## 2. System Architecture

### 2.1 Code Structure

```
src/
â”œâ”€â”€ ingestion/                    # Phase 0A: Data loading
â”‚   â”œâ”€â”€ document_loader.py        # Unified loader for DLA Piper + Scopus
â”‚   â”œâ”€â”€ scopus_parser.py          # CSV metadata extraction
â”‚   â””â”€â”€ paper_to_scopus_metadata_matcher.py  # Paper-Scopus linkage
â”‚
â”œâ”€â”€ preprocessing/                # Phase 0B: Text preprocessing
â”‚   â”œâ”€â”€ text_cleaner.py           # Encoding fixes (ftfy), HTML/LaTeX stripping
â”‚   â”œâ”€â”€ translator.py             # Google Translate API with file caching
â”‚   â””â”€â”€ preprocessing_processor.py # Orchestrator: clean â†’ detect â†’ translate
â”‚
â”œâ”€â”€ processing/
â”‚   â”œâ”€â”€ chunks/                   # Phase 1A
â”‚   â”‚   â”œâ”€â”€ semantic_chunker.py   # BGE-small boundary detection + filtering
â”‚   â”‚   â””â”€â”€ chunk_processor.py    # Orchestrator with resume mode
â”‚   â”‚
â”‚   â”œâ”€â”€ entities/                 # Phase 1B-1C
â”‚   â”‚   â”œâ”€â”€ pre_entity_extractor.py  # Mistral-7B dual-pass extraction
â”‚   â”‚   â”œâ”€â”€ pre_entity_processor.py  # Parallel orchestrator with checkpoints
â”‚   â”‚   â”œâ”€â”€ disambiguation.py     # FAISS blocking + tiered thresholds
â”‚   â”‚   â””â”€â”€ add_entity_ids.py     # Deterministic hash ID generation
â”‚   â”‚
â”‚   â””â”€â”€ relations/
â”‚       â”œâ”€â”€ build_entity_cooccurrence.py  # 3 typed matrices
â”‚       â”œâ”€â”€ run_relation_extraction.py    # OpenIE triplets (Mistral-7B)
â”‚       â””â”€â”€ normalize_relations.py        # ID mapping + Neo4j format
â”‚
â”œâ”€â”€ enrichment/
â”‚   â”œâ”€â”€ enrichment_processor.py   # Orchestrator (10-step pipeline)
â”‚   â”œâ”€â”€ scopus_enricher.py        # Citation matching, author/journal nodes
â”‚   â””â”€â”€ jurisdiction_matcher.py   # Country entity â†’ Jurisdiction linking
â”‚
â”œâ”€â”€ graph/
â”‚   â”œâ”€â”€ neo4j_import_processor.py # Batch import to Neo4j
â”‚   â””â”€â”€ faiss_builder.py          # Build HNSW indices
â”‚
â””â”€â”€ utils/
    â”œâ”€â”€ dataclasses.py            # Core data structures
    â”œâ”€â”€ constants.py              # Entity types, jurisdiction codes
    â”œâ”€â”€ id_generator.py           # Deterministic hash IDs
    â”œâ”€â”€ io.py                     # JSON/JSONL utilities
    â”œâ”€â”€ embeddings.py             # BGE-M3 wrapper
    â””â”€â”€ embed_processor.py        # OOM-resilient batch embedding

data/
â”œâ”€â”€ raw/                          # Original inputs (read-only)
â”‚   â”œâ”€â”€ dlapiper/                 # 48 jurisdiction JSONs
â”‚   â””â”€â”€ scopus/                   # CSV + MinerU parsed papers
â”œâ”€â”€ interim/                      # Checkpoints (resumable)
â”‚   â”œâ”€â”€ preprocessed/             # Phase 0B output
â”‚   â”‚   â”œâ”€â”€ documents_cleaned.jsonl
â”‚   â”‚   â”œâ”€â”€ paper_references.json     # 5,240 refs from 119 papers
â”‚   â”‚   â””â”€â”€ preprocessing_report.json
â”‚   â”œâ”€â”€ translation_cache/        # Cached translations
â”‚   â”œâ”€â”€ chunks/                   # Phase 1A intermediates
â”‚   â”‚   â”œâ”€â”€ discarded_chunks.jsonl
â”‚   â”‚   â””â”€â”€ chunking_report.json
â”‚   â”œâ”€â”€ entities/
â”‚   â””â”€â”€ relations/
â””â”€â”€ processed/                    # Final outputs
    â”œâ”€â”€ chunks/                   # chunks_embedded.jsonl (2,718)
    â”œâ”€â”€ entities/
    â”œâ”€â”€ relations/
    â”œâ”€â”€ neo4j/
    â””â”€â”€ faiss/
```

### 2.2 Pipeline Overview

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                     PHASE 0: DATA PREPARATION                               â”‚
â”‚                                                                             â”‚
â”‚   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”   â”‚
â”‚   â”‚  0A: SCOPUS MATCHING                                                â”‚   â”‚
â”‚   â”‚  paper_to_scopus_metadata_matcher.py â†’ 142/148 matched (95.9%)      â”‚   â”‚
â”‚   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜   â”‚
â”‚                                â–¼                                            â”‚
â”‚   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”   â”‚
â”‚   â”‚  0B: PREPROCESSING                                                  â”‚   â”‚
â”‚   â”‚  clean + extract refs + translate â†’ 206 docs (10.4M â†’ 8.3M chars)   â”‚   â”‚
â”‚   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                 â”‚
â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
                    PHASE 1: KNOWLEDGE GRAPH CONSTRUCTION
â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
                                 â”‚
                                 â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  1A: CHUNKING â†’ 2,718 chunks (14% discarded, 14% deduped)                   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                 â”‚
                                 â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  1B: ENTITY EXTRACTION â†’ 62,048 pre-entities (15 types, Mistral-7B)         â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                 â”‚
                                 â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
│  1C: ENTITY DISAMBIGUATION → 42K entities (21K semantic + 21K metadata)    │
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                 â”‚
                                 â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
│  1D: RELATION EXTRACTION → 339,268 semantic + 5,745 discusses = 345K total  │
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                 â”‚
â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
                       PHASE 2: ENRICHMENT & STORAGE
â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
                                 â”‚
                                 â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
│  2A: SCOPUS ENRICHMENT → 872 L2 pubs, 900 CITES, 592 SAME_AS            │
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                 â”‚
                                 â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
│  2B: STORAGE → 38,266 entities, 339,268 relations, 58K cross-layer     │
│  3: RETRIEVAL → 3 modes (semantic/graph/dual), RAGAS evaluation          │
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## 3. Pipeline Phases

### 3.0 Phase 0: Data Preparation

#### 3.0.1 Phase 0A: Scopus Matching

**Status**: Complete

**Purpose**: Link MinerU-parsed PDFs to Scopus metadata.

**Files**:
- `src/ingestion/paper_to_scopus_metadata_matcher.py`
- Output: `data/interim/paper_mapping.json`

**Results**:

| Match Type | Count |
|------------|-------|
| DOI exact | 117 |
| Title (high confidence, â‰¥0.85) | 21 |
| Title (low confidence, 0.70-0.85) | 3 |
| Abstract fallback | 1 |
| **Total matched** | **142/148 (95.9%)** |

#### 3.0.2 Phase 0B: Preprocessing

**Status**: Complete

**Purpose**: Clean text, extract references, and translate non-English documents.

**Files**:
- `src/preprocessing/text_cleaner.py` â€” ftfy encoding fixes, HTML/LaTeX stripping, garbage section removal
- `src/preprocessing/translator.py` â€” Google Translate API with file caching
- `src/preprocessing/preprocessing_processor.py` â€” Orchestrator

**Pipeline**: `raw text â†’ clean_text() â†’ extract_references() â†’ langdetect() â†’ translate() â†’ cleaned text`

**Results**:

| Metric | Value |
|--------|-------|
| Documents processed | 206 |
| By source | 48 regulations, 158 papers |
| By language | 184 English, 5 Spanish, 1 German |
| Translated | 6 |
| Chars before | 10,410,535 |
| Chars after | 8,316,081 (20% reduction) |
| References extracted | 5,240 from 119 papers |
| Garbage sections removed | 328 |

**Cleaning fixes applied**: encoding (9.7M), HTML (84K), LaTeX (7K), images (756), emails (180)

**Output files**:
- `data/interim/preprocessed/documents_cleaned.jsonl` â€” Cleaned documents
- `data/interim/preprocessed/paper_references.json` â€” Extracted references
- `data/interim/preprocessed/preprocessing_report.json` â€” Statistics

**Documents excluded**:

| Doc ID | Reason |
|--------|--------|
| paper_046 | OCR corrupted (Ukrainian) |
| paper_055 | OCR corrupted (Russian) |
| paper_025 | Duplicate of paper_021 |
| paper_065 | Duplicate of paper_040 |
| paper_083 | Duplicate of paper_019 |
| paper_132 | Duplicate of paper_048 |

---

### 3.1 Phase 1: Knowledge Graph Construction

#### 3.1.1 Phase 1A: Chunking

**Status**: Complete

**Purpose**: Split preprocessed documents into semantic chunks with quality filtering and embeddings.

**Files**:
- `src/processing/chunks/semantic_chunker.py` â€” BGE-small boundary detection, coherence filtering
- `src/processing/chunks/chunk_processor.py` â€” Pipeline orchestrator with resume mode
- `src/utils/embed_processor.py` â€” OOM-resilient BGE-M3 embedding

**Method**: Sentences are embedded with BGE-small-en-v1.5. Adjacent sentences with similarity above threshold are grouped into chunks. Three filtering stages remove low-quality output.

**Parameters**:
```python
similarity_threshold = 0.45   # Boundary detection (p25 of distribution)
min_coherence = 0.30          # Minimum mean adjacent similarity
min_tokens_per_sentence = 10  # Density filter for reference lists
merge_threshold = 0.98        # Cross-doc duplicate detection
```

**Filtering pipeline**:
1. Coherence â€” mean adjacent similarity < 0.30 â†’ discard
2. Density â€” tokens/sentence < 10 + no header â†’ discard  
3. Cross-doc merge â€” similarity â‰¥0.98 â†’ keep first occurrence

**Results**:

| Metric | Value |
|--------|-------|
| Documents processed | 206 |
| Chunks produced | 2,718 |
| Avg tokens/chunk | 410.9 |
| Avg coherence | 0.619 |
| Discarded | 450 (14.2%) |
| Duplicates merged | 451 (14.2%) |

| Source | Docs | Kept | Discarded |
|--------|------|------|-----------|
| Regulations | 48 | 660 | 386 |
| Papers | 158 | 2,509 | 64 |

**Input**: `data/interim/preprocessed/documents_cleaned.jsonl`  
**Output**: `data/processed/chunks/chunks_embedded.jsonl`

**CLI**:
```bash
python -m src.processing.chunks.chunk_processor --mode server
python -m src.processing.chunks.chunk_processor --mode server --resume
```

---

#### 3.1.2 Phase 1B: Entity Extraction

**Status**: Complete (v2.0 schema)

**Purpose**: Extract pre-entities from chunks using dual-pass LLM extraction.

**Files**:
- `src/processing/entities/pre_entity_extractor.py` â€” Mistral-7B dual-pass extraction
- `src/processing/entities/pre_entity_processor.py` â€” Parallel orchestrator with checkpoints
- `src/prompts/prompts.py` â€” Mistral-optimized prompts
- `config/extraction_config.py` â€” Type definitions (single source of truth)

**Model**: Mistral-7B-Instruct-v0.3 via Together.ai (JSON mode)

**Dual-Pass Architecture**:

| Pass | Entity Types | Purpose |
|------|--------------|---------|
| Semantic | RegulatoryConcept, TechnicalConcept, PoliticalConcept, EconomicConcept, Regulation, Technology, Organization, Location, Risk | Domain knowledge backbone |
| Metadata | Citation, Author, Journal, Affiliation, Document, DocumentSection | Bibliographic + structural entities |

**Schema v2.0**: 15 types total (9 semantic + 6 metadata)

| Semantic Types (9) | Metadata Types (6) |
|--------------------|-------------------|
| RegulatoryConcept | Citation |
| TechnicalConcept | Author |
| PoliticalConcept | Journal |
| EconomicConcept | Affiliation |
| Regulation | **Document** |
| Technology | **DocumentSection** |
| Organization | |
| Location | |
| Risk | |

**v1.9 â†’ v2.0 Schema Change** (major):
- Added `Document` type â€” Named documents referenced structurally (EU AI Act, GDPR)
- Added `DocumentSection` type â€” Structural parts (Article 5, Section 3.2, Annex I)
- Renamed "Academic" â†’ "Metadata" â€” Document structure isn't academic
- Enables document traversal: `DocumentSection â†’ PART_OF â†’ Document â†’ SAME_AS â†’ Regulation`

**Embedding format**: `"{name}({type})"`

**Full Run Results (v2.0)**:

| Metric | Value |
|--------|-------|
| Total entities | 62,048 |
| Semantic entities | 37,879 (61%) |
| Metadata entities | 24,169 (39%) |
| Avg per chunk | 22.8 |
| Processing time | ~20 min (24 workers) |

**Type Distribution**:

| Type | Count | % |
|------|-------|---|
| Citation | 7,389 | 11.9% |
| DocumentSection | 6,870 | 11.1% |
| Organization | 6,320 | 10.2% |
| Technology | 5,944 | 9.6% |
| Document | 5,839 | 9.4% |
| PoliticalConcept | 4,861 | 7.8% |
| Risk | 4,371 | 7.0% |
| RegulatoryConcept | 3,888 | 6.3% |
| TechnicalConcept | 3,814 | 6.1% |
| Location | 3,658 | 5.9% |
| EconomicConcept | 3,581 | 5.8% |
| Affiliation | 2,145 | 3.5% |
| Regulation | 1,442 | 2.3% |
| Author | 1,327 | 2.1% |
| Journal | 599 | 1.0% |

**Known Garbage (Phase 1C handles)**:
- "..." artifacts: 259
- Short entities â‰¤3 chars: 3,472 (mix valid/garbage)
- Fig/Table in DocumentSection: ~200
- Year numbers as Citation: ~400

**Input**: `data/processed/chunks/chunks_embedded.jsonl`  
**Output**: `data/interim/entities/pre_entities.jsonl`

**CLI**:
```bash
# Test run
python -m src.processing.entities.pre_entity_processor --sample 50 --seed 42

# Full extraction (24 workers)
python -m src.processing.entities.pre_entity_processor --workers 24

# Resume
python -m src.processing.entities.pre_entity_processor --resume
```

---

#### 3.1.3 Phase 1C: Entity Disambiguation

**Status**: Complete

**Purpose**: Dual-path disambiguation + document structure linking.

**Method**: FAISS blocking + tiered thresholds + LLM SameJudge + betweenness centrality cluster breaking.

**Dual-Path Architecture**:

| Path | Entity Types | Strategy |
|------|--------------|----------|
| Semantic | 9 domain types | Filter -> ExactDedup -> Embed -> FAISS -> Threshold -> SameJudge -> Betweenness |
| Metadata | 6 structural types | Filter -> Dedup -> PART_OF/SAME_AS (same-chunk only) |

**Threshold Configuration**:

| Threshold | Value | Rationale |
|-----------|-------|-----------|
| `auto_merge` | 0.98 | False positives at 0.96 (author initials) |
| `auto_reject` | 0.885 | 0.85-0.88 is ~95% different |
| `betweenness_threshold` | 0.25 | Cut bridges connecting subcommunities |
| `similarity_ceiling` | 0.91 | Only cut if similarity also low |
| `hard_max_size` | 20 | Fallback: cut weakest edge |

**Key Innovation: Betweenness Centrality Bridge Detection**

Problem: LLM approves A-B and B-C, Union-Find chains A=B=C, creating 196-entity clusters.

Solution: Model merges as graph, compute edge betweenness centrality. High betweenness + low similarity = artificial bridge = CUT.

Result: 233 betweenness cuts + 267 size-limit cuts = max cluster 19.

**Final Results**:

| Metric | Value |
|--------|-------|
| Input pre-entities | 62,048 |
| After garbage filter | 57,455 |
| Semantic raw | 36,629 |
| Metadata raw | 20,826 |
| **Semantic output** | **21,368** (42% merged) |
| **Metadata output** | **20,826** |
| Alias clusters | 1,310 |
| Max aliases per entity | 19 (down from 196) |
| PART_OF relations | 131 |
| SAME_AS relations | 405 |

**Input**: `data/interim/entities/pre_entities.jsonl`  
**Outputs**:
- `data/processed/entities/entities_semantic.jsonl` (21,368)
- `data/processed/entities/entities_metadata.jsonl` (20,826)
- `data/processed/entities/aliases.json` (1,310 clusters)
- `data/processed/relations/part_of_relations.jsonl` (131)
- `data/processed/relations/same_as_relations.jsonl` (405)

---

#### 3.1.4 Phase 1D: Relation Extraction

**Status**: Complete

**Purpose**: Extract relations between entities via dual-track approach.

**Method**: Mistral-7B-Instruct-v0.3 with JSON mode enforcement.

**Two-Track Extraction**:

| Track | Entity Types | Method | Output |
|-------|--------------|--------|--------|
| Semantic | Concepts, Orgs, Tech, Regulations | OpenIE (schema-free predicates) | 339,268 relations |
| Citation | Citations, Authors, Documents | Fixed `discusses` predicate | 5,745 relations |

**Semantic Track**:
- Entities processed: 21,368
- Entity coverage: 98.7% (21,082 with relations)
- Unique predicates: 2,728
- Top 4 predicates: appliesto (15.2%), requires (8.0%), enforces (3.5%), refersto (3.2%) — cover 29.9%
- Adaptive chunk selection: 3–6 chunks per batch within 6,000 token budget
- Multi-round extraction triggered by centroid distance

**Citation Track**:
- Chunks processed: 988
- Relations: 5,745 (avg 7.7/chunk)
- Object distribution: PoliticalConcept (34.2%), TechnicalConcept (24.8%), EconomicConcept (20.9%), RegulatoryConcept (20.1%)

**Validation**:
- Co-occurrence constraint: subject/object must share at least one chunk
- ID resolution via hashed IDs including known aliases
- **Result: 345,013 relations (339,268 semantic + 5,745 discusses), 100% validation pass rate**

**Input**: `entities_semantic.jsonl`, `entities_metadata.jsonl`, `chunks_embedded.jsonl`  
**Output**: 
- `data/processed/relations/relations_semantic_validated.jsonl`
- `data/processed/relations/relations_discusses_validated.jsonl`

---

### 3.2 Phase 2: Enrichment & Storage

#### 3.2.1 Phase 2A: Scopus Enrichment

**Status**: Complete

**Purpose**: Provenance-constrained metadata matching + citation linking + jurisdiction mapping.

**Key Features**:
- Dual-track loading: semantic + metadata entities/relations merged at runtime
- Multi-jurisdiction chunk handling (EU harmonized content → multiple CONTAINS)
- Fuzzy L2 deduplication (80% author, 75% title thresholds)
- DISCUSSES filter for citation relevance

**Nodes Created**:

| Node Type | Count |
|-----------|-------|
| L1 Publications | 158 |
| L2 Publications | 872 |
| Authors | 572 |
| Journals | 119 |

**Relations Generated**:

| Relation | Count |
|----------|-------|
| CITES | 900 |
| SAME_AS (total) | 592 |
| MATCHED_TO | 219 |
| PUBLISHED_IN | 147 |

**SAME_AS Breakdown**:

| Target Type | Count |
|-------------|-------|
| Author | 359 |
| Jurisdiction | 194 |
| Journal | 39 |

**Files**:
- `src/enrichment/citation_matcher.py` - Fuzzy L2 deduplication
- `src/enrichment/enrichment_processor.py` - Dual-track loading
- `config/extraction_config.py` - Separate paths for semantic/metadata

---

#### 3.2.2 Phase 2B: Storage

**Status**: Complete

**Purpose**: Import graph to Neo4j, build FAISS indices for retrieval.

**Graph Overview**:

| Metric | Value |
|--------|-------|
| Jurisdictions | 48 |
| Publications (L1) | 158 |
| L2 Publications | 872 |
| Entities | 38,266 |
| Chunks | 2,718 |
| Semantic Relations | 339,268 |

**Node Breakdown**:

| Node Type | Count |
|-----------|-------|
| Entity | 38,266 |
| Chunk | 2,718 |
| L2 Publication | 872 |
| Author | 572 |
| Publication (L1) | 158 |
| Journal | 119 |
| Jurisdiction | 48 |

**Semantic Layer Metrics**:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Avg Degree (all edges) | 19.1 | |
| Avg Degree (RELATION only) | 17.7 | |
| Relation Density | 8.9 rels/entity | Very dense |
| Connectivity (any edge) | 100.0% | All entities reachable |
| Has RELATION edges | 55.7% | Semantic layer coverage |
| Truly Isolated | 0 | All have provenance |
| Type Homophily | 26.4% | <30% = rich cross-type connections |
| Clustering Coefficient | 1.13 | High - tight communities |
| Avg Path Length | 4.21 | Reasonable traversal |
| Top 4 Predicates | 29.86% | Good diversity |

**Cross-Layer Relations**:

| Relationship | Count |
|--------------|-------|
| EXTRACTED_FROM (Entity→Chunk) | 52,548 |
| CONTAINS (Publication→Chunk) | 2,507 |
| CITES (Publication→L2) | 900 |
| CONTAINS (Jurisdiction→Chunk) | 660 |
| AUTHORED_BY (Publication→Author) | 585 |
| SAME_AS (Entity→Author) | 359 |
| MATCHED_TO (Entity→Publication) | 219 |
| SAME_AS (Entity→Jurisdiction) | 194 |
| PUBLISHED_IN (Publication→Journal) | 147 |
| SAME_AS (Entity→Journal) | 39 |
| **Total Cross-Layer** | **58,158** |

**Top Connected Entities**:

| Entity | Type | Degree |
|--------|------|--------|
| AI | Technology | 3,706 |
| Machine-based system | Technology | 2,659 |
| transparency | RegulatoryConcept | 1,811 |
| discriminatory impacts | Risk | 1,482 |
| EU | Location | 1,207 |

**Metadata Layer**:

| Metric | Value |
|--------|-------|
| Avg co-authors per author | 14.2 |
| Avg citations per L2 | 1.03 |
| Max citations | 4 |

**Academic-Regulatory Bridging**: 392 entities (1.8%) bridge both domains

**FAISS Indices**:
- `entities.faiss` - 38,266 entity embeddings (HNSW, M=32, ef=200)
- `chunks.faiss` - 2,718 chunk embeddings (HNSW, M=32, ef=200)

**Files**:
- `src/graph/neo4j_importer.py` - Batched UNWIND imports
- `src/graph/neo4j_import_processor.py` - Multi-jurisdiction CONTAINS
- `src/graph/faiss_builder.py` - HNSW index building
- `src/graph/graph_analytics.py` - Analytics queries

---
### 3.3 Phase 3: Retrieval & Generation

#### 3.3.1 Phase 3A: Query Understanding

**Status**: Complete

**Purpose**: Parse user queries into structured form for retrieval.

**Components**:
- **Query Parser**: Extracts entity mentions, jurisdiction hints, doc type hints
- **Entity Resolver**: 3-stage matching (exact → alias → fuzzy)

**Entity Resolution**:

| Stage | Method | Threshold |
|-------|--------|-----------|
| Exact | Neo4j name match | 1.0 |
| Alias | Neo4j alias lookup | 1.0 |
| Fuzzy | FAISS embedding similarity | 0.75 |

**Alias Query** (Neo4j):
```cypher
MATCH (e:Entity)
WHERE toLower(e.name) = toLower($name)
   OR toLower($name) IN [a IN e.aliases | toLower(a)]
RETURN e.entity_id, e.name, e.type, e.aliases
```

**Files**:
- `src/retrieval/query_parser.py` - ParsedQuery dataclass
- `src/retrieval/entity_resolver.py` - 3-stage resolution

---

#### 3.3.2 Phase 3B: Context Retrieval

**Status**: Complete

**Purpose**: Retrieve relevant chunks via three modes for ablation study.

**Retrieval Modes**:

| Mode | Source | Method |
|------|--------|--------|
| SEMANTIC | FAISS | Vector similarity on query embedding |
| GRAPH | Neo4j | Steiner Tree expansion → EXTRACTED_FROM |
| DUAL | Both | Combined with coverage-proportional ranking |

**Chunk Source Paths**:
- `semantic`: FAISS vector similarity
- `graph_provenance`: Chunks linked via relation's chunk_ids
- `graph_entity`: Chunks linked via entity's EXTRACTED_FROM

**Graph Expansion** (Steiner Tree):
```python
if len(resolved_entities) == 1:
    # k-NN expansion only
    subgraph = candidate_ids[:max_entities]
else:
    # Steiner Tree via Neo4j GDS
    CALL gds.beta.steinerTree.stream(...)
```

**GDS Projection**: 38,266 nodes, 678,536 relations (undirected)

**Files**:
- `src/retrieval/semantic_retriever.py` - FAISS search
- `src/retrieval/graph_expander.py` - Steiner Tree via GDS
- `src/retrieval/retrieval_processor.py` - Mode orchestration

---

#### 3.3.3 Phase 3C: Ranking & Generation

**Status**: Complete

**Purpose**: Rank chunks and generate grounded answers.

**Multiplicative Scoring**:
```python
score = base_similarity
score *= graph_provenance_multiplier if from_graph else 1.0
score *= jurisdiction_match_bonus if jurisdiction_match else 1.0
score *= doc_type_match_bonus if doc_type_match else 1.0
```

**Answer Generation**:
- Model: Claude 3.5 Haiku
- Max input: 15,000 tokens
- Max output: 2,000 tokens
- Temperature: 0.0 (deterministic for evaluation)

**Citation System**:
- Papers: Authors, Year, Title, Journal, DOI from `paper_mapping.json`
- Regulations: DLA Piper URL constructed from jurisdiction code

**Files**:
- `src/retrieval/result_ranker.py` - Multiplicative scoring
- `src/retrieval/answer_generator.py` - Claude integration
- `src/utils/citations.py` - CitationFormatter

---

#### 3.3.4 Phase 3D: Evaluation

**Status**: Complete

**Purpose**: Ablation study comparing retrieval modes.

**Test Query Categories**:

| Category | n | Description |
|----------|---|-------------|
| regulation_only | 6 | Regulatory concepts, compliance |
| academic_only | 6 | Research methods, frameworks |
| cross_domain | 6 | Academic critiques of regulations |
| metadata_provenance | 6 | Chunk co-occurrence, entity linking |
| applied_scenario | 6 | Practical compliance questions |
| edge_cases | 6 | OOD, aliases, ambiguous, specific |

**Test Matrix**: 36 queries × 3 modes = 108 tests

**Metrics** (RAGAS with Claude Sonnet 4):
- Faithfulness: Grounding in retrieved context
- Relevancy: Topical appropriateness
- Terminal Coverage: Query entities mentioned in answer

**Results** (8 representative queries):

| Mode | Faithfulness | Relevancy | Best Count |
|------|--------------|-----------|------------|
| SEMANTIC | 0.77 | 0.86 | 2/8 |
| GRAPH | 0.58 | 0.88 | 2/8 |
| DUAL | **0.89** | 0.81 | **5/8** |

**Key Finding**: Dual mode excels on cross-domain and multi-hop queries. Graph mode excels when entities resolve cleanly to high-degree nodes. Semantic provides consistent baseline.

**LaTeX Export**:
- `ablation_vars.tex` - Macros for inline citations
- `ablation_table.tex` - Results table
- `ablation_appendix.tex` - Full I/O per query
- `ablation_data.tex` - pgfplots data

**Files**:
- `src/analysis/retrieval_metrics.py` - Evaluation dataclasses
- `src/analysis/ablation_study.py` - 3-mode comparison runner
- `src/analysis/ablation_latex_export.py` - Thesis figures
- `src/analysis/test_queries.py` - Query definitions

---

### 3.4 Code Structure (Complete)

```
src/
├── ingestion/                    # Phase 0A
├── preprocessing/                # Phase 0B
├── processing/
│   ├── chunks/                   # Phase 1A
│   ├── entities/                 # Phase 1B-1C
│   └── relations/                # Phase 1D
├── enrichment/                   # Phase 2A
├── graph/                        # Phase 2B
├── retrieval/                    # Phase 3A-3C
│   ├── query_parser.py
│   ├── entity_resolver.py
│   ├── semantic_retriever.py
│   ├── graph_expander.py
│   ├── result_ranker.py
│   ├── answer_generator.py
│   ├── retrieval_processor.py
│   └── tests/
├── analysis/                     # Phase 3D
│   ├── retrieval_metrics.py
│   ├── ablation_study.py
│   ├── ablation_latex_export.py
│   └── test_queries.py
└── utils/
    ├── dataclasses.py
    ├── embedder.py
    ├── citations.py
    └── ...

config/
├── extraction_config.py
└── retrieval_config.py
```

---

## 4. Data Structures

See `src/utils/dataclasses.py` for canonical definitions.

Core types:
- `Chunk` â€” Text segment with embedding
- `PreEntity` â€” Raw extraction before disambiguation
- `Entity` â€” Canonical entity with ID and aliases
- `Relation` â€” Subject-predicate-object triplet

---

## 5. Dependencies

### Phase 0B

```bash
pip install clean-text[gpl] langdetect requests pandas
```

**Google Translate API**:
1. Enable Cloud Translation API at console.cloud.google.com
2. Create API key
3. Add to `.env`: `GOOGLE_TRANSLATE_API_KEY=xxx`

### Phase 1+

[TO UPDATE as phases complete]

---

## 6. References

**Primary Methodology**:
1. **Zhang, H., et al. (2025)**. "RAKG: Document-level Retrieval Augmented Knowledge Graph Construction." *arXiv*.
2. **Agarwal, B., et al. (2025)**. "RAGulating Compliance: Leveraging AI for Multi-Jurisdictional Regulatory Knowledge Graphs." *arXiv*.

**Entity Resolution**:
3. **Papadakis, G., et al. (2021)**. "Blocking and Filtering Techniques for Entity Resolution." *ACM Computing Surveys*.
4. **Fellegi, I. & Sunter, A. (1969)**. "A Theory for Record Linkage." *JASA*. — Tiered threshold theory for disambiguation.

**Embeddings & Retrieval**:
5. **Chen, J., et al. (2024)**. "BGE M3-Embedding: Multi-Functionality, Multi-Linguality, and Multi-Granularity Text Embeddings." *arXiv*.
6. **Carbonell, J. & Goldstein, J. (1998)**. "The Use of MMR, Diversity-Based Reranking." *SIGIR*. — Chunk diversity selection.

**Graph Algorithms**:
7. **Neo4j GDS Documentation**. Steiner Tree algorithm for subgraph extraction.
8. **Newman, M. (2004)**. "Analysis of Weighted Networks." *Physical Review E*. — Betweenness centrality.

**Evaluation**:
9. **Es, S., et al. (2023)**. "RAGAS: Automated Evaluation of Retrieval Augmented Generation." *arXiv*. — Faithfulness/relevancy metrics.