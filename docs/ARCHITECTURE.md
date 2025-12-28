# Architecture

**Project**: AI Governance GraphRAG Pipeline  
**Version**: 1.1  
**Date**: December 2025

---

## 1. Overview

### 1.1 Purpose

A knowledge graph and question-answering system for cross-jurisdictional AI governance research. Integrates 48 regulatory documents from DLA Piper's "AI Laws of the World" with 158 academic papers from Scopus, enabling structured queries via three retrieval modes.

### 1.2 Data Sources

| Source | Content | Count | Origin |
|--------|---------|-------|--------|
| Regulatory | Jurisdiction summaries | 48 | DLA Piper web scrape |
| Academic | PDFs (MinerU-parsed) | 158 | Scopus export |
| Derived | Authors / Journals / L2 Pubs | 572 / 119 / 872 | Scopus CSV |

### 1.3 Technology Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| LLM (Extraction) | Mistral-7B-Instruct-v0.3 via Together.ai | Entity extraction |
| LLM (Disambiguation) | Mistral-7B-Instruct-v0.3 via Together.ai | Merge verification |
| LLM (Relations) | Mistral-7B-Instruct-v0.3 via Together.ai | Relation extraction |
| LLM (Generation) | Claude Haiku 3.5 | Answer generation |
| Embeddings | BGE-M3 (1024-dim) | Entity/chunk embeddings |
| Embeddings (chunking) | BGE-small-en-v1.5 (384-dim) | Sentence boundaries |
| Graph Database | Neo4j 5.12 + GDS | Docker, Steiner Tree |
| Vector Store | FAISS (HNSW) | M=32, ef=200 |

### 1.4 Methodology Sources

| Component | Approach | Reference |
|-----------|----------|-----------|
| Entity extraction | Pre-entity + disambiguation | RAKG [1] |
| Disambiguation | FAISS blocking + tiered thresholds | Papadakis [3], Fellegi-Sunter |
| Relation extraction | Schema-free OpenIE | RAGulating [2] |
| Subgraph expansion | Prize-Collecting Steiner Tree | Neo4j GDS [6] |
| Dual retrieval | Graph + semantic fusion | KG-RAG [7], KG2RAG [8] |
| Embeddings | BGE-M3 | Chen et al. [9] |

---

## 2. Code Structure

```
src/
├── ingestion/
│   ├── document_loader.py          # Unified loader for DLA Piper + Scopus
│   └── scopus_parser.py            # CSV metadata extraction
│
├── preprocessing/
│   └── preprocessing_processor.py  # Text cleaning, ref extraction, translation
│
├── processing/
│   ├── chunks/
│   │   ├── semantic_chunker.py     # Sentence-boundary chunking
│   │   └── chunk_processor.py      # Orchestrator + BGE-M3 embedding
│   │
│   ├── entities/
│   │   ├── entity_extractor.py     # RAKG pre-entity extraction (Mistral-7B)
│   │   ├── pre_entity_processor.py # Parallel extraction orchestrator
│   │   ├── disambiguation.py       # FAISS blocking + tiered merge
│   │   └── add_entity_ids.py       # Deterministic SHA-256 hashing
│   │
│   └── relations/
│       ├── build_entity_cooccurrence.py  # Entity-chunk matrices
│       ├── relation_extractor.py         # OpenIE triplets (Mistral-7B)
│       ├── relation_processor.py         # Orchestrator + checkpointing
│       └── validate_relations.py         # Co-occurrence validation
│
├── enrichment/
│   ├── enrichment_processor.py     # 10-step metadata pipeline
│   ├── citation_matcher.py         # Provenance-constrained matching
│   └── jurisdiction_matcher.py     # Country → Jurisdiction linking
│
├── graph/
│   ├── neo4j_import_processor.py   # Batch UNWIND imports
│   └── faiss_builder.py            # HNSW index construction
│
├── retrieval/
│   ├── query_parser.py             # Entity mentions, jurisdiction hints
│   ├── entity_resolver.py          # 3-stage resolution (exact/alias/fuzzy)
│   ├── semantic_retriever.py       # FAISS chunk similarity
│   ├── graph_expander.py           # Steiner Tree + EXTRACTED_FROM
│   ├── result_ranker.py            # Multiplicative scoring
│   ├── answer_generator.py         # Claude Haiku generation
│   └── retrieval_processor.py      # Pipeline orchestrator
│
├── analysis/
│   ├── graph_analytics.py          # Neo4j metrics extraction
│   └── ablation_study.py           # RAGAS evaluation
│
└── utils/
    ├── dataclasses.py              # Chunk, Entity, Relation types
    ├── embedder.py                 # BGE-M3 wrapper
    ├── rate_limiter.py             # API throttling
    └── checkpoint_manager.py       # Resume support

config/
├── extraction_config.py            # LLM params, thresholds
└── retrieval_config.py             # Mode settings, budgets

scripts/
└── run_query.py                    # CLI entry point
```

---

## 3. Knowledge Graph

### 3.1 Statistics

| Metric | Value |
|--------|-------|
| Documents | 206 |
| Chunks | 2,718 |
| Entities | 38,266 (21,368 semantic + 16,898 metadata) |
| Semantic Relations | 339,268 |
| Cross-Layer Relations | 58,158 |
| Unique Predicates | 2,728 |

### 3.2 Semantic Layer Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Avg Degree (all) | 19.1 | High connectivity |
| Avg Degree (RELATION) | 17.7 | Dense semantic links |
| Connectivity (any edge) | 100% | No isolated entities |
| Has RELATION edges | 55.7% | Semantic coverage |
| Type Homophily | 26.4% | Rich cross-type links |
| Clustering Coefficient | 0.38 | Tight communities |
| Avg Path Length | 2.9 hops | Efficient traversal |

### 3.3 Top Predicates

| Predicate | Count | % |
|-----------|-------|---|
| appliesto | 51,522 | 15.2% |
| requires | 27,027 | 8.0% |
| enforces | 11,985 | 3.5% |
| refersto | 10,765 | 3.2% |
| *Top 4 total* | | *29.9%* |

### 3.4 Hub Entities

| Entity | Type | Degree |
|--------|------|--------|
| AI | Technology | 3,706 |
| Machine-based system | Technology | 2,659 |
| transparency | RegulatoryConcept | 1,811 |
| discriminatory impacts | Risk | 1,482 |
| EU | Location | 1,207 |

### 3.5 Cross-Layer Relations

| Relation | Count | Source |
|----------|-------|--------|
| EXTRACTED_FROM | 52,548 | Entity provenance |
| CONTAINS (Pub→Chunk) | 2,507 | Document structure |
| CITES | 900 | Scopus references |
| CONTAINS (Jur→Chunk) | 660 | Jurisdiction mapping |
| AUTHORED_BY | 585 | Scopus metadata |
| SAME_AS (→Author) | 359 | Entity matching |
| MATCHED_TO | 219 | Citation matching |
| SAME_AS (→Jurisdiction) | 194 | Country matching |
| PUBLISHED_IN | 147 | Scopus metadata |
| SAME_AS (→Journal) | 39 | Entity matching |

### 3.6 Domain Integration

| Metric | Value |
|--------|-------|
| Academic-only entities | 35,928 |
| Regulatory-only entities | 1,946 |
| Bridging entities | 392 (1.0%) |
| Max cross-jurisdictional | 44 jurisdictions |

---

## 4. Pipeline

### 4.0 Data Preparation

**Files**: `src/ingestion/`, `src/preprocessing/`

| Step | Input | Output | Implementation |
|------|-------|--------|----------------|
| Ingestion | Raw JSONs + PDFs | Unified documents | `document_loader.py` |
| Preprocessing | 206 documents | Cleaned text | `preprocessing_processor.py` |

**Preprocessing operations**:
- Encoding fixes (UTF-8 normalization)
- HTML/LaTeX removal
- Reference extraction via regex
- Non-English translation (Google Translate API)
- Output: 8.3M chars (20% reduction)

---

### 4.1 Phase 1A: Chunking

**Files**: `src/processing/chunks/`

| Metric | Value |
|--------|-------|
| Input | 206 documents |
| Output | 2,718 chunks |
| Avg tokens | 411 |
| Discarded | 450 (14.2%) |
| Merged duplicates | 451 (14.2%) |

**Implementation** (`semantic_chunker.py`):
```python
# Sentence-boundary chunking with quality filtering
chunks = semantic_chunker.chunk(
    text,
    max_tokens=512,
    min_coherence=0.4,  # Quality threshold
    overlap=50
)
```

**Embedding** (`chunk_processor.py`):
- Model: BGE-M3 (1024-dim)
- Batch size: 32
- Output: `chunks_embedded.jsonl`

---

### 4.2 Phase 1B: Entity Extraction

**Files**: `src/processing/entities/`

| Metric | Value |
|--------|-------|
| Pre-entities | 62,048 |
| Entity types | 15 (9 semantic + 6 metadata) |
| Cost | ~$12 |

**Dual-pass extraction** (`pre_entity_processor.py`):

| Pass | Types | Focus |
|------|-------|-------|
| Semantic | Technology, Organization, Risk, RegulatoryConcept, TechnicalConcept, Location, EconomicConcept, PoliticalConcept, Legislation | Domain knowledge |
| Metadata | Citation, Author, Document, Affiliation, DocumentSection, Journal | Bibliographic |

**Prompt structure** (`prompts.py`):
```python
ENTITY_EXTRACTION_PROMPT = """
Extract entities from this regulatory/academic text.
For each entity provide:
- name: Canonical form
- type: One of {types}
- description: Brief context (1-2 sentences)
...
"""
```

---

### 4.3 Phase 1C: Entity Disambiguation

**Files**: `src/processing/entities/disambiguation.py`

| Metric | Value |
|--------|-------|
| Input | 62,048 pre-entities |
| Output | 38,266 canonical entities |
| Reduction | 38% |
| Alias clusters | 1,310 |
| Total aliases | 2,594 |

**Tiered thresholds** (Fellegi-Sunter decision theory):

| Similarity | Action | Rationale |
|------------|--------|-----------|
| ≥0.98 | Auto-merge | Obvious duplicates |
| 0.885–0.98 | LLM verify | Mistral-7B confirmation |
| <0.885 | Auto-reject | Distinct entities |

**FAISS blocking** (`disambiguation.py`):
```python
# Build HNSW index for candidate generation
index = faiss.IndexHNSWFlat(1024, 32)
index.hnsw.efSearch = 200

# Query returns top-k candidates per entity
distances, indices = index.search(embeddings, k=50)

# Filter by threshold, then verify
candidates = [(i, j) for i, j, d in pairs if d >= 0.70]
```

**Merge order**: Betweenness centrality prioritization — merge high-connectivity entities first to propagate canonical forms.

---

### 4.4 Phase 1D: Relation Extraction

**Files**: `src/processing/relations/`

| Track | Relations | Method |
|-------|-----------|--------|
| Semantic | 339,268 | OpenIE (Mistral-7B) |
| Citation | 5,745 | Typed `discusses` |
| **Total** | **345,013** | |

**Two-track strategy** (`relation_processor.py`):

| Track | Entity Types | Predicates | Rationale |
|-------|--------------|------------|-----------|
| Semantic | Concepts, Orgs, Tech, Regs | Schema-free | Domain knowledge |
| Citation | Citations → Concepts | `discusses` only | Avoid Scopus duplication |

**Co-occurrence constraint** (`build_entity_cooccurrence.py`):
```python
# Build entity-chunk matrix
cooccurrence = defaultdict(set)
for entity in entities:
    for chunk_id in entity['chunk_ids']:
        cooccurrence[entity['id']].add(chunk_id)

# Relations only extracted between co-occurring entities
valid_pairs = [(s, o) for s, o in pairs 
               if cooccurrence[s] & cooccurrence[o]]
```

**Validation** (`validate_relations.py`):
- 100% pass rate (by construction)
- All relations grounded in shared chunks

---

### 4.5 Phase 2A: Scopus Enrichment

**Files**: `src/enrichment/`

| Output | Count |
|--------|-------|
| L2 Publications | 872 |
| CITES relations | 900 |
| SAME_AS (Author) | 359 |
| SAME_AS (Jurisdiction) | 194 |
| SAME_AS (Journal) | 39 |

**Provenance-constrained matching** (`citation_matcher.py`):
```python
# Entities only match against their SOURCE paper's metadata
def match_citation(entity, source_paper_id):
    paper_refs = paper_references[source_paper_id]
    for ref in paper_refs:
        if fuzzy_match(entity['name'], ref['title']) >= 0.85:
            return create_matched_to(entity, ref)
```

**L2 deduplication**:
- 80% author surname overlap OR
- 75% title similarity
- Prevents duplicate L2 nodes from multiple L1 citations

**Author ID resolution** (`citation_matcher.py`):
```python
# Build surname → author_id index from authors.json
author_index = defaultdict(list)
for author in authors:
    surname = author['name'].split()[-1].lower()
    author_index[surname].append(author['id'])

# Resolve entity authors against index
def resolve_author(entity_name, source_paper_id):
    surname = entity_name.split()[-1].lower()
    candidates = author_index.get(surname, [])
    # Filter by source paper authorship
    ...
```

---

### 4.6 Phase 2B: Storage

**Files**: `src/graph/`

**Neo4j Import** (`neo4j_import_processor.py`):
```python
# Batch UNWIND for performance
IMPORT_ENTITIES = """
UNWIND $batch AS e
MERGE (n:Entity {entity_id: e.entity_id})
SET n.name = e.name, n.type = e.type, ...
"""

# Multi-jurisdiction CONTAINS handling
for chunk in chunks:
    for jur_code in chunk['document_ids']:  # May be multiple
        create_contains(jur_code, chunk['chunk_id'])
```

**FAISS Index Building** (`faiss_builder.py`):
```python
# Entity index
entity_index = faiss.IndexHNSWFlat(1024, 32)
entity_index.hnsw.efConstruction = 200
entity_index.add(entity_embeddings)  # 38,266 vectors

# Chunk index
chunk_index = faiss.IndexHNSWFlat(1024, 32)
chunk_index.add(chunk_embeddings)  # 2,718 vectors
```

**GDS Projection** (for Steiner Tree):
```cypher
CALL gds.graph.project(
    'entity_graph',
    ['Entity', 'Chunk'],
    {
        RELATION: {orientation: 'UNDIRECTED'},
        EXTRACTED_FROM: {orientation: 'UNDIRECTED'}
    }
)
-- 38,266 nodes, 678,536 relationships
```

---

### 4.7 Phase 3: Retrieval

**Files**: `src/retrieval/`

#### Query Understanding (`query_parser.py`)

```python
def parse_query(query: str) -> ParsedQuery:
    embedding = embedder.embed(query)
    mentions = extract_entity_mentions(query)  # Rule-based + NER
    jurisdiction = detect_jurisdiction_hint(query)  # Pattern matching
    doc_type = detect_doc_type_hint(query)  # "regulation" vs "academic"
    return ParsedQuery(embedding, mentions, jurisdiction, doc_type)
```

**Jurisdiction patterns**: 48 countries + aliases (GDPR→EU, CCPA→US, etc.)

**Doc type patterns**:
- `regulation`: laws, acts, directives, compliance, enforcement
- `academic`: papers, research, journals, authors, literature

#### Entity Resolution (`entity_resolver.py`)

3-stage resolution:

```python
def resolve(mention: str) -> Optional[Entity]:
    # Stage 1: Exact match
    if entity := exact_match(mention):
        return entity
    
    # Stage 2: Alias lookup
    if entity := alias_lookup(mention):
        return entity
    
    # Stage 3: Fuzzy FAISS
    candidates = faiss_search(embed(mention), k=5)
    if candidates[0].similarity >= 0.75:
        return candidates[0].entity
    
    return None
```

#### Retrieval Modes (`retrieval_processor.py`)

| Mode | Implementation | Use Case |
|------|----------------|----------|
| SEMANTIC | FAISS chunk similarity only | Broad lexical |
| GRAPH | Steiner Tree → EXTRACTED_FROM | Entity-centric |
| DUAL | Combined with coverage ranking | Cross-domain |

**SEMANTIC mode** (`semantic_retriever.py`):
```python
def retrieve_semantic(query_embedding, k=20):
    distances, indices = chunk_index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]
```

**GRAPH mode** (`graph_expander.py`):
```python
def retrieve_graph(entity_ids: List[str], k=20):
    # Steiner Tree connects resolved entities
    subgraph = run_steiner_tree(entity_ids)
    
    # Get chunks via provenance
    chunks = []
    for entity_id in subgraph.entity_ids:
        chunks.extend(get_extracted_from(entity_id))
    
    return deduplicate(chunks)[:k]
```

**DUAL mode**:
```python
def retrieve_dual(query_embedding, entity_ids, k=20):
    semantic_chunks = retrieve_semantic(query_embedding, k)
    graph_chunks = retrieve_graph(entity_ids, k)
    
    # Combine with coverage-proportional ranking
    all_chunks = merge_by_score(semantic_chunks, graph_chunks)
    return all_chunks[:k]
```

#### Ranking (`result_ranker.py`)

Multiplicative scoring:
```python
score = (
    base_similarity *
    (1 + graph_bonus if from_graph else 1) *
    (1 + jurisdiction_bonus if matches_hint else 1) *
    (1 + doc_type_bonus if matches_hint else 1)
)
```

#### Generation (`answer_generator.py`)

```python
response = anthropic.messages.create(
    model="claude-3-5-haiku-20241022",
    max_tokens=2000,
    temperature=0.0,
    system="Answer based only on provided context. Cite sources.",
    messages=[{"role": "user", "content": prompt}]
)
```

**Context budget**: 15,000 tokens input, 2,000 tokens output

---

## 5. Evaluation

**Files**: `src/analysis/ablation_study.py`

### 5.1 Test Design

- 8 representative queries across 6 categories
- 3 modes × 8 queries = 24 evaluations
- RAGAS metrics with Claude Sonnet as judge

**Query Categories**:
1. Regulation-only
2. Academic-only
3. Cross-domain
4. Metadata provenance
5. Applied scenario
6. Edge cases (out-of-domain)

### 5.2 Results

| Mode | Faithfulness | Relevancy | Best Count |
|------|--------------|-----------|------------|
| Semantic | 0.77 | 0.86 | 2/8 |
| Graph | 0.58 | 0.88 | 2/8 |
| **Dual** | **0.89** | 0.81 | **5/8** |

### 5.3 Key Findings

1. **Dual mode excels on cross-domain queries** — combines semantic breadth with graph-derived entity specificity

2. **Graph mode best for entity-centric lookups** — when entities resolve cleanly to high-degree nodes

3. **Semantic provides consistent baseline** — effective for queries using corpus vocabulary

4. **Subgraph size does not predict quality** — retrieval precision matters more than recall

5. **Relation context correlation** — high relation coverage (>30%) correlates with higher faithfulness

---

## 6. References

### Core Methodologies

1. **Zhang, H., et al. (2025)**. "RAKG: Document-level Retrieval Augmented Knowledge Graph Construction." arXiv.

2. **Agarwal, B., et al. (2025)**. "RAGulating Compliance: Leveraging AI for Multi-Jurisdictional Regulatory Knowledge Graphs."

### Entity Resolution

3. **Papadakis, G., et al. (2021)**. "Blocking and Filtering Techniques for Entity Resolution: A Survey." ACM Computing Surveys, 53(2), 1-42.

4. **Malkov, Y., & Yashunin, D. (2020)**. "Efficient and Robust Approximate Nearest Neighbor Search Using HNSW Graphs." IEEE TPAMI.

### Graph Retrieval

5. **He, X., et al. (2024)**. "G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering." NeurIPS 2024. arXiv:2402.07630.

6. **Neo4j (2024)**. "Prize-Collecting Steiner Tree." Neo4j Graph Data Science Library. https://neo4j.com/docs/graph-data-science/current/algorithms/prize-collecting-steiner-tree/

### Dual-Channel Retrieval

7. **Sanmartin, D. (2024)**. "KG-RAG: Bridging the Gap Between Knowledge and Creativity." arXiv:2405.12035.

8. **Feng, Z., et al. (2025)**. "KG2RAG: Knowledge Graph-Guided Retrieval Augmented Generation."

### Embeddings

9. **Chen, J., et al. (2024)**. "BGE M3-Embedding: Multi-Functionality, Multi-Linguality, and Multi-Granularity Text Embeddings." arXiv:2402.05816.

### Data Sources

10. **DLA Piper (2024)**. "AI Laws of the World." https://www.dlapiper.com/en/insights/artificial-intelligence

11. **Scopus (Elsevier)**. Bibliometric database for academic paper metadata.