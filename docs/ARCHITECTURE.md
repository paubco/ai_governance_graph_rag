# Architecture

**Project**: AI Governance GraphRAG Pipeline  
**Version**: 3.0  
**Last Updated**: December 7, 2025

---

## 1. Overview

### 1.1 Goal

Build a knowledge graph for AI governance research that enables cross-jurisdictional regulatory queries by combining 48 regulatory documents with 158 academic papers. The system implements entity-centric corpus retrospective retrieval following RAKG methodology (Zhang et al., 2025), adapted for the regulatory compliance domain.

### 1.2 Data Sources

| Source | Content | Count | Origin |
|--------|---------|-------|--------|
| **Regulations** | Jurisdiction metadata + legal text | 48 | DLA Piper (2024) web scrape |
| **Academic Papers** | Scopus metadata CSV + PDFs (MinerU-parsed) | 158 | Scopus export |
| **Derived Metadata** | Authors, Journals, References | 572 / 119 / 1,513 | Scopus CSV |

### 1.3 Tech Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| **LLM (Extraction)** | Qwen-72B via Together.ai | Phase 1B entity extraction |
| **LLM (Disambiguation)** | Qwen-7B via Together.ai | Phase 1C |
| **LLM (Relations)** | Mistral-7B via Together.ai | Phase 1D (Qwen has JSON bug) |
| **LLM (Generation)** | Configurable (Claude/Qwen) | Phase 3 |
| **Embeddings** | BGE-M3 (1024-dim, multilingual) | All phases |
| **Graph DB** | Neo4j Aura Free | Graph traversal, ~100MB |
| **Vector Store** | FAISS (HNSW) | Entity + chunk indices, ~320MB |
| **Hardware** | RTX 3060 GPU (12GB VRAM) | Local embedding computation |

### 1.4 Methodology Sources

| Component | Source | Reference |
|-----------|--------|-----------|
| Entity extraction | RAKG | Zhang et al. (2025) |
| Entity disambiguation | Adapted RAKG | See § 4.2 |
| Relation extraction | RAGulating Compliance | Agarwal et al. (2025) |
| Retrieval | RAKG corpus retrospective | Zhang et al. (2025) |
| Blocking | Entity resolution literature | Papadakis et al. (2021) |
| Embeddings | BGE-M3 | Chen et al. (2024) |

---

## 2. System Architecture

### 2.1 Code Structure

```
src/
├── ingestion/                    # Data loading
│   ├── document_loader.py        # Unified loader for DLA Piper + Scopus
│   └── scopus_parser.py          # CSV metadata extraction
│
├── processing/
│   ├── chunking/
│   │   ├── semantic_chunker.py   # Sentence-boundary chunking
│   │   └── chunk_processor.py    # Orchestrator + embedding
│   │
│   ├── entities/
│   │   ├── entity_extractor.py   # RAKG pre-entity extraction (Qwen-72B)
│   │   ├── entity_processor.py   # Parallel processing orchestrator
│   │   ├── disambiguation.py     # FAISS blocking + tiered thresholds
│   │   └── add_entity_ids.py     # Deterministic hash ID generation
│   │
│   └── relations/
│       ├── build_entity_cooccurrence.py  # 3 typed matrices
│       ├── run_relation_extraction.py    # OpenIE triplets (Mistral-7B)
│       └── normalize_relations.py        # ID mapping + Neo4j format
│
├── enrichment/
│   ├── enrichment_processor.py   # Orchestrator (10-step pipeline)
│   ├── scopus_enricher.py        # Citation matching, author/journal nodes
│   └── jurisdiction_matcher.py   # Country entity → Jurisdiction linking
│
├── graph/
│   ├── neo4j_import_processor.py # Batch import to Neo4j Aura
│   └── faiss_builder.py          # Build HNSW indices
│
├── retrieval/                    # Phase 3: Query-time
│   ├── pipeline.py               # RetrievalPipeline (mode: naive/graphrag)
│   ├── query_parser.py           # Embed query, extract mentions, filters
│   ├── entity_resolver.py        # Query mentions → canonical entities
│   ├── corpus_retriever.py       # Provenance + similarity retrieval
│   ├── graph_expander.py         # Relations + 1-hop neighbors
│   ├── ranker.py                 # Scoring, dedup, context fitting
│   ├── prompt_builder.py         # Structured prompt construction
│   └── generator.py              # LLM call + response parsing
│
└── utils/
    ├── config.py                 # Paths, API keys, thresholds
    ├── embeddings.py             # BGE-M3 wrapper
    └── llm_client.py             # Together.ai / Anthropic client

data/
├── raw/                          # Original inputs (read-only)
│   ├── dla_piper/                # 48 jurisdiction JSONs
│   └── scopus/                   # CSV + PDFs
├── interim/                      # Checkpoints (resumable)
│   ├── chunks/
│   ├── entities/
│   └── relations/
└── processed/                    # Final outputs
    ├── chunks/                   # chunks_embedded.json
    ├── entities/                 # normalized_entities_with_ids.json
    ├── relations/                # neo4j_edges.jsonl
    ├── neo4j/                    # Import-ready files
    └── faiss/                    # .faiss indices + ID mappings
```

### 2.2 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA PREPARATION                                    │
│                                                                             │
│   DLA Piper (48 jurisdictions)          Scopus (158 papers)                │
│   • Web scrape → JSON                   • PDF → MinerU → Markdown           │
│                     └──────────┬──────────┘                                 │
│                                ▼                                            │
│                    DocumentLoader → 206 unified documents                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
═══════════════════════════════════════════════════════════════════════════════
                    PHASE 1: KNOWLEDGE GRAPH CONSTRUCTION
═══════════════════════════════════════════════════════════════════════════════
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1A: CHUNKING                                                         │
│  SemanticChunker → 25,131 chunks with BGE-M3 embeddings                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1B: ENTITY EXTRACTION                                                │
│  Qwen-72B + RAKG method → ~200K pre-entities                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1C: ENTITY DISAMBIGUATION                                            │
│  FAISS blocking + tiered thresholds → 55,695 normalized entities           │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1D: RELATION EXTRACTION                                              │
│  Mistral-7B + OpenIE (two-track) → 105,456 validated relations             │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
═══════════════════════════════════════════════════════════════════════════════
                       PHASE 2: ENRICHMENT & STORAGE
═══════════════════════════════════════════════════════════════════════════════
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2A: SCOPUS ENRICHMENT                                                │
│  Add Authors, Journals, L2Publications; match citations; link jurisdictions │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2B: STORAGE                                                          │
│  Neo4j (graph structure) + FAISS (vector indices)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
═══════════════════════════════════════════════════════════════════════════════
                      PHASE 3: RETRIEVAL & GENERATION
═══════════════════════════════════════════════════════════════════════════════
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Query → Parse → Resolve Entities → Retrieve Context → Rank → Generate     │
│                                                                             │
│  Modes: naive (baseline) | graphrag (full) | graphrag_lite (ablation)      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Dual Storage Architecture

```
┌─────────────────────────────────┐     ┌─────────────────────────────────┐
│  FAISS (Vector Search)          │     │  Neo4j (Graph Traversal)        │
│                                 │     │                                 │
│  entities.faiss                 │     │  :Jurisdiction (48)             │
│  • 55,695 × 1024-dim            │◄───►│  :Publication (715)             │
│  • HNSW index                   │     │  :Chunk (25,131)                │
│                                 │     │  :Entity (55,695)               │
│  chunks.faiss                   │     │  :Author (572)                  │
│  • 25,131 × 1024-dim            │     │  :Journal (119)                 │
│  • HNSW index                   │     │                                 │
│                                 │     │  Relationships:                 │
│  Linked via entity_id/chunk_id  │     │  RELATION, EXTRACTED_FROM,      │
│                                 │     │  CONTAINS, AUTHORED_BY,         │
│  Size: ~320MB                   │     │  PUBLISHED_IN, CITES, etc.      │
└─────────────────────────────────┘     │                                 │
                                        │  Size: ~100MB                   │
                                        └─────────────────────────────────┘
```

---

## 3. Pipeline Phases

### 3.0 Data Preparation

Raw data consists of 48 jurisdiction JSONs scraped from DLA Piper's "AI Laws of the World" database (2024) and 158 academic PDFs processed through MinerU for markdown extraction. Scopus CSV provides bibliometric metadata (authors, journals, references). The pipeline assumes these exist in `data/raw/`.

---

### 3.1 Knowledge Graph Construction

#### 3.1.1 Phase 1A: Chunking

**Method**: Semantic chunking at sentence boundaries, respecting document structure.

**Input**: 206 unified documents  
**Output**: 25,131 chunks (~500 tokens avg)

Each chunk contains:
- `chunk_id`: Unique identifier
- `text`: Chunk content
- `embedding`: 1024-dim BGE-M3 vector
- `metadata`: `{doc_type, jurisdiction|scopus_id, section_title}`

**Embedding model**: BGE-M3 (Chen et al., 2024) — chosen for multilingual support across regulatory texts.

---

#### 3.1.2 Phase 1B: Entity Extraction

**Method**: RAKG pre-entity extraction (Zhang et al., 2025)

**Model**: Qwen-72B via Together.ai

The LLM analyzes each chunk to identify entities with:
- `name`: Canonical form
- `type`: One of [Concept, Organization, Technology, Regulation, Person, Country, Academic Citation, ...]
- `description`: Brief disambiguation context
- `chunk_id`: Provenance link

**Output**: ~200K pre-entities  
**Cost**: ~$30 (6-8 hours)

---

#### 3.1.3 Phase 1C: Entity Disambiguation

**Problem**: Same entity appears with variant surface forms ("EU AI Act" vs "AI Act" vs "Regulation 2024/1689").

**Method**: Two-stage process:

1. **Exact deduplication**: Normalize case/whitespace (200K → 143K)
2. **Semantic clustering**: FAISS HNSW blocking + tiered thresholds (143K → 55,695)

**Tiered threshold pipeline**:

| Stage | Cosine Threshold | Purpose |
|-------|------------------|---------|
| 1 | > 0.70 | FAISS candidate generation |
| 2 | > 0.90 | Aggressive merge (obvious duplicates) |
| 3 | > 0.85 | Conservative merge |
| 4 | > 0.82 | Final polish |

**ID generation**: Deterministic SHA-256 hash of `{name}:{type}` for reproducibility.

**Output**: 55,695 normalized entities with embeddings and IDs

---

#### 3.1.4 Phase 1D: Relation Extraction

**Method**: OpenIE-style triplet extraction per RAGulating Compliance (Agarwal et al., 2025)

**Model**: Mistral-7B via Together.ai (Qwen-7B has JSON infinite loop bug at temperature=0)

**Two-track strategy**:

| Track | Entity Types | Predicates | Rationale |
|-------|--------------|------------|-----------|
| **Semantic** | Concepts, Orgs, Tech, Regulations | Any (OpenIE) | Domain knowledge |
| **Academic** | Citations, Authors, Journals | `discusses` only | Avoid duplicating Scopus |

**Validation**: Relations only emitted if both subject and object exist in the entity index — construction-time grounding.

**Output**: 105,456 validated relations (20,832 unique predicates)

---

### 3.2 Enrichment & Storage

#### 3.2.1 Phase 2A: Scopus Metadata Enrichment

**Purpose**: Add structured metadata that text extraction cannot provide reliably.

**Created nodes**:
- 572 Author nodes
- 119 Journal nodes  
- 557 L2Publication nodes (cited works)

**Matching**:
- Citation entities → Scopus references (provenance-constrained matching)
- Country entities → Jurisdiction nodes (41 SAME_AS links)

**Generated relationships**: `AUTHORED_BY`, `PUBLISHED_IN`, `CITES`, `MATCHED_TO`, `SAME_AS`

---

#### 3.2.2 Phase 2B: Graph & Vector Import

**Neo4j schema**:

```cypher
// Document layer
(:Jurisdiction {code, name, url})
(:Publication {scopus_id, title, year, doi})
(:L2Publication {title, year, doi, matched_confidence})

// Content layer
(:Chunk {chunk_id, text, metadata})
(:Entity {entity_id, name, type, description, chunk_ids, frequency})

// Metadata layer
(:Author {author_id, name, scopus_id})
(:Journal {journal_id, name, issn})

// Key relationships
(:Jurisdiction)-[:CONTAINS]->(:Chunk)
(:Publication)-[:CONTAINS]->(:Chunk)
(:Entity)-[:EXTRACTED_FROM]->(:Chunk)
(:Entity)-[:RELATION {predicate, chunk_ids}]->(:Entity)
(:Publication)-[:AUTHORED_BY]->(:Author)
(:Publication)-[:PUBLISHED_IN]->(:Journal)
(:Publication)-[:CITES]->(:L2Publication)
(:Entity)-[:MATCHED_TO]->(:L2Publication)
(:Entity)-[:SAME_AS]->(:Jurisdiction)
```

**FAISS indices**:
- `entities.faiss`: 55,695 entity embeddings (HNSW)
- `chunks.faiss`: 25,131 chunk embeddings (HNSW)
- ID mapping JSONs link FAISS index positions to graph IDs

---

### 3.3 Retrieval & Generation

**Method**: Entity-centric corpus retrospective retrieval (Zhang et al., 2025), adapted for QA.

**Core principle**: Entities are the entry point. Chunks retrieved via provenance (high confidence) and similarity (medium confidence).

**Retrieval modes**:

| Mode | Description | Use Case |
|------|-------------|----------|
| `naive` | Query → FAISS chunk search → Top-K → Generate | Baseline |
| `graphrag` | Full entity-centric pipeline | Primary |
| `graphrag_lite` | Entities + provenance only | Ablation |

---

#### 3.3.1 Query Understanding

**Steps**:
1. **Embed query** using BGE-M3
2. **Extract entity mentions** from query text (rule-based + NER)
3. **Parse filters** (jurisdiction, doc_type, date range)
4. **Resolve mentions → canonical entities**:
   - Exact match: `MATCH (e:Entity) WHERE toLower(e.name) = toLower($mention)`
   - Fuzzy match: FAISS entity search with cosine threshold

**Output**: Set of canonical entity IDs + query embedding + filters

---

#### 3.3.2 Context Retrieval

**Two retrieval paths per entity**:

| Path | Method | Confidence |
|------|--------|------------|
| **Provenance** | `(e:Entity)-[:EXTRACTED_FROM]->(c:Chunk)` | High — entity definitely here |
| **Similarity** | FAISS chunk search with entity embedding | Medium — entity likely relevant |

**Graph expansion** (optional):
- Inter-entity relations: `(e1)-[:RELATION]->(e2)` where both matched
- 1-hop neighbors: Related entities not in query
- Jurisdiction context via chunk metadata

**Filtering**: Jurisdiction/doc_type filters applied to both paths.

---

#### 3.3.3 Ranking & Prompt Assembly

**Scoring function**:
```
score = w1 * vector_similarity
      + w2 * entity_mention_count  
      + w3 * jurisdiction_match_bonus
      + w4 * is_provenance_chunk
```

**Assembly**:
1. Deduplicate chunks (keep highest score)
2. Sort by score, take top-K within token budget
3. Build structured prompt:

```
ENTITIES:
  - EU AI Act (Legislation): Regulation (EU) 2024/1689...
  - emotion recognition (Technology): AI systems that...

RELATIONS:
  - EU AI Act --regulates--> emotion recognition [3 chunks]

SOURCES:
  [EU-Art50] "Deployers of emotion recognition systems..."
  [ES-Impl] "Spain designates the AESIA as..."

QUESTION: {user_query}
```

---

#### 3.3.4 Answer Generation

**Model**: Configurable (Claude Sonnet / Qwen-72B / local)

**Parameters**:
- Temperature: 0.3 (factual)
- Max tokens: 2000
- System prompt: "Answer based only on provided context. Cite sources."

**Output**:
- Answer with inline citations `[EU-Art50]`
- Provenance summary (entities, jurisdictions, chunk count)
- "Unable to answer from context" if insufficient evidence

---

#### 3.3.5 Evaluation Strategy

**Test set**: 20-30 questions covering:
- Single-jurisdiction factual: "What is Spain's definition of AI system?"
- Cross-jurisdictional: "How do EU and US approaches to facial recognition differ?"
- Academic-regulatory bridge: "What research supports the EU AI Act's transparency requirements?"
- Negative: "What does Brazil say about quantum computing?" (not in corpus)

**Metrics**:

| Metric | Description |
|--------|-------------|
| Answer relevance | Manual 1-5 rating |
| Factual accuracy | Binary: claims supported by citations? |
| Citation precision | % of citations that support their claims |
| Source diversity | # unique jurisdictions/papers |

**Comparison**: Same metrics across `naive` vs `graphrag` vs `graphrag_lite` modes.

---

## 4. Methodological Contributions

### 4.1 Two-Track Relation Extraction

**Novel contribution**: Separating semantic entities from academic entities with different extraction strategies.

| Track | Entity Types | Extraction | Why |
|-------|--------------|------------|-----|
| Semantic | Concepts, Orgs, Tech, Regs | Full OpenIE | Domain knowledge backbone |
| Academic | Citations, Authors, Journals | `discusses` only | Scopus provides citation networks |

**Key insight**: Academic-to-academic relations (citation networks) come from structured Scopus metadata, not text extraction. This avoids redundancy and leverages higher-quality source data.

---

### 4.2 Scalable Entity Disambiguation via Blocking

**Problem**: RAKG's VecJudge produces O(n²) candidate pairs. Our corpus density (755 entities/doc vs RAKG's 50-150) caused explosion to 15.2M pairs (77 hours, $300-400).

**Solution**: FAISS HNSW blocking + tiered thresholds from entity resolution literature (Papadakis et al., 2021).

| Approach | Candidate Pairs | Time | Cost |
|----------|-----------------|------|------|
| RAKG VecJudge | 15.2M | 77 hrs | $300-400 |
| **Our blocking** | 250K | 6 hrs | $6 |

**Contribution**: Demonstrated RAKG's brittleness at higher entity densities; validated blocking as compatible extension.

---

### 4.3 Construction-Time Relation Validation

**RAKG approach**: "LLM as Judge" validates at query time.

**Our approach**: Validate at construction time:
- Relations only created if both entities exist in index
- Every entity has `EXTRACTED_FROM` provenance links
- Result: 100% entity grounding, zero hallucinated references

**Trade-off**: Less flexible than query-time validation, but zero runtime cost and simpler implementation.

---

## 5. Cost & Timeline

### 5.1 Construction (One-Time)

| Stage | Time | Cost |
|-------|------|------|
| Ingestion + Chunking | 15 min | $0 |
| Embedding (chunks) | 30 min | $0 |
| Entity Extraction | 6-8 hrs | ~$30 |
| Entity Disambiguation | 6 hrs | ~$6 |
| Relation Extraction | 2-3 days | ~$7 |
| Enrichment + Import | 2 hrs | $0 |
| **Total** | **~4-5 days** | **~$43** |

### 5.2 Query (Per Request)

| Step | Time | Cost |
|------|------|------|
| Parse + Resolve | <1 sec | $0 |
| Retrieval + Expansion | <1 sec | $0 |
| Ranking | <1 sec | $0 |
| Generation | 5-15 sec | $0.01-0.05 |
| **Total** | **~10-20 sec** | **~$0.02-0.10** |

---

## 6. References

### Core Methodologies

1. **Zhang, H., et al. (2025)**. "RAKG: Document-level Retrieval Augmented Knowledge Graph Construction." arXiv.

2. **Agarwal, B., et al. (2025)**. "RAGulating Compliance: Leveraging AI for Multi-Jurisdictional Regulatory Knowledge Graphs."

### Entity Resolution

3. **Papadakis, G., et al. (2021)**. "Blocking and Filtering Techniques for Entity Resolution: A Survey." ACM Computing Surveys.

4. **Malkov, Y., & Yashunin, D. (2020)**. "Efficient and Robust Approximate Nearest Neighbor Search Using HNSW Graphs." IEEE TPAMI.

### Embeddings

5. **Chen, J., et al. (2024)**. "BGE M3-Embedding: Multi-Functionality, Multi-Linguality, and Multi-Granularity Text Embeddings." arXiv:2402.05816.

### Data Sources

6. **DLA Piper (2024)**. "AI Laws of the World." https://www.dlapiper.com/en/insights/artificial-intelligence

7. **Scopus (Elsevier)**. Bibliometric database for academic paper metadata.
