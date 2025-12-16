# Architecture

**Project**: AI Governance GraphRAG Pipeline  
**Version**: 1.0  
**Last Updated**: December 2025

---

## 1. Overview

### 1.1 Goal

Build a knowledge graph for AI governance research that enables cross-jurisdictional regulatory queries by combining 48 regulatory documents with 156 academic papers. The system implements entity-centric corpus retrospective retrieval following RAKG methodology (Zhang et al., 2025), adapted for the regulatory compliance domain.

### 1.2 Data Sources

| Source | Content | Count | Origin |
|--------|---------|-------|--------|
| **Regulations** | Jurisdiction metadata + legal text | 48 | DLA Piper (2024) web scrape |
| **Academic Papers** | Scopus metadata CSV + PDFs (MinerU-parsed) | 156 | Scopus export |
| **Derived Metadata** | Authors, Journals, References | 572 / 119 / 1,513 | Scopus CSV |

### 1.3 Tech Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| **LLM (Extraction)** | Qwen-72B via Together.ai | Phase 1B entity extraction |
| **LLM (Disambiguation)** | Qwen-7B via Together.ai | Phase 1C |
| **LLM (Relations)** | Mistral-7B via Together.ai | Phase 1D (Qwen has JSON bug) |
| **LLM (Generation)** | Claude Haiku 3.5 via Anthropic | Phase 3 |
| **Embeddings** | BGE-M3 (1024-dim, multilingual) | All phases |
| **Graph DB** | Neo4j 5.12 Enterprise + GDS | Docker container, PCST algorithm |
| **Vector Store** | FAISS (HNSW) | Entity + chunk indices, ~320MB |
| **Frontend** | Streamlit | Docker container |
| **Deployment** | Docker Compose | Single-command startup |
| **Hardware** | RTX 3060 GPU (12GB VRAM) | Local embedding computation |

**Note**: Neo4j Aura Free was initially used but lacks Graph Data Science (GDS) plugin required for PCST algorithm. Migrated to Docker-hosted Neo4j Enterprise with GDS.

### 1.4 Methodology Sources

| Component | Source | Reference |
|-----------|--------|-----------|
| Entity extraction | RAKG | Zhang et al. (2025) |
| Entity disambiguation | Adapted RAKG | See S 4.2 |
| Relation extraction | RAGulating Compliance | Agarwal et al. (2025) |
| Retrieval | RAKG corpus retrospective | Zhang et al. (2025) |
| Blocking | Entity resolution literature | Papadakis et al. (2021) |
| Embeddings | BGE-M3 | Chen et al. (2024) |

---

## 2. System Architecture

### 2.1 Code Structure

```
src/
 ingestion/                    # Data loading
+-- document_loader.py        # Unified loader for DLA Piper + Scopus
+-- scopus_parser.py          # CSV metadata extraction
|
 processing/
+-- chunking/
|   +-- semantic_chunker.py   # Sentence-boundary chunking
|   +-- chunk_processor.py    # Orchestrator + embedding
|   |
+-- entities/
|   +-- entity_extractor.py   # RAKG pre-entity extraction (Qwen-72B)
|   +-- entity_processor.py   # Parallel processing orchestrator
|   +-- disambiguation.py     # FAISS blocking + tiered thresholds
|   +-- add_entity_ids.py     # Deterministic hash ID generation
|   |
+-- relations/
+-- build_entity_cooccurrence.py  # 3 typed matrices
+-- run_relation_extraction.py    # OpenIE triplets (Mistral-7B)
+-- normalize_relations.py        # ID mapping + Neo4j format
|
 enrichment/
+-- enrichment_processor.py   # Orchestrator (10-step pipeline)
+-- scopus_enricher.py        # Citation matching, author/journal nodes
+-- jurisdiction_matcher.py   # Country entity  Jurisdiction linking
|
 graph/
+-- neo4j_import_processor.py # Batch import to Neo4j Aura
+-- faiss_builder.py          # Build HNSW indices
|
 retrieval/                    # Phase 3: Query-time
+-- pipeline.py               # RetrievalPipeline (mode: naive/graphrag)
+-- query_parser.py           # Embed query, extract mentions, filters
+-- entity_resolver.py        # Query mentions  canonical entities
+-- corpus_retriever.py       # Provenance + similarity retrieval
+-- graph_expander.py         # Relations + 1-hop neighbors
+-- ranker.py                 # Scoring, dedup, context fitting
+-- prompt_builder.py         # Structured prompt construction
+-- generator.py              # LLM call + response parsing
|
 utils/
     config.py                 # Paths, API keys, thresholds
     embeddings.py             # BGE-M3 wrapper
     llm_client.py             # Together.ai / Anthropic client

data/
 raw/                          # Original inputs (read-only)
+-- dla_piper/                # 48 jurisdiction JSONs
+-- scopus/                   # CSV + PDFs
 interim/                      # Checkpoints (resumable)
+-- chunks/
+-- entities/
+-- relations/
 processed/                    # Final outputs
     chunks/                   # chunks_embedded.json
     entities/                 # normalized_entities_with_ids.json
     relations/                # neo4j_edges.jsonl
     neo4j/                    # Import-ready files
     faiss/                    # .faiss indices + ID mappings
```

### 2.2 Pipeline Overview

```

+-- DATA PREPARATION                                    |
|                                                                             |
+-- DLA Piper (48 jurisdictions)          Scopus (156 papers)                |
+-- Web scrape  JSON                    PDF  MinerU  Markdown           |
|                                                      |
|                                                                            |
+-- DocumentLoader  204 unified documents                   |

                                 |
*******************************************************************************
                    PHASE 1: KNOWLEDGE GRAPH CONSTRUCTION
*******************************************************************************
                                 |
                                 

|  PHASE 1A: CHUNKING                                                         |
|  SemanticChunker  25,131 chunks with BGE-M3 embeddings                    |

                                 |
                                 

|  PHASE 1B: ENTITY EXTRACTION                                                |
|  Qwen-72B + RAKG method  ~200K pre-entities                               |

                                 |
                                 

|  PHASE 1C: ENTITY DISAMBIGUATION                                            |
|  FAISS blocking + tiered thresholds  55,695 normalized entities           |

                                 |
                                 

|  PHASE 1D: RELATION EXTRACTION                                              |
|  Mistral-7B + OpenIE (two-track)  105,456 validated relations             |

                                 |
*******************************************************************************
                       PHASE 2: ENRICHMENT & STORAGE
*******************************************************************************
                                 |
                                 

|  PHASE 2A: SCOPUS ENRICHMENT                                                |
|  Add Authors, Journals, L2Publications; match citations; link jurisdictions |

                                 |
                                 

|  PHASE 2B: STORAGE                                                          |
|  Neo4j (graph structure) + FAISS (vector indices)                          |

                                 |
*******************************************************************************
                      PHASE 3: RETRIEVAL & GENERATION
*******************************************************************************
                                 |
                                 

|  Query  Parse  Resolve Entities  Retrieve Context  Rank  Generate     |
|                                                                             |
|  Modes: naive (baseline) | graphrag (full) | graphrag_lite (ablation)      |

```

### 2.3 Dual Storage Architecture

```
     
|  FAISS (Vector Search)          |     |  Neo4j (Graph Traversal)        |
|                                 |     |                                 |
|  entities.faiss                 |     |  :Jurisdiction (48)             |
+-- 55,695 -- 1024-dim            |--|  :Publication (715)             |
+-- HNSW index                   |     |  :Chunk (25,131)                |
|                                 |     |  :Entity (55,695)               |
|  chunks.faiss                   |     |  :Author (572)                  |
+-- 25,131 -- 1024-dim            |     |  :Journal (119)                 |
+-- HNSW index                   |     |                                 |
|                                 |     |  Relationships:                 |
|  Linked via entity_id/chunk_id  |     |  RELATION, EXTRACTED_FROM,      |
|                                 |     |  CONTAINS, AUTHORED_BY,         |
|  Size: ~320MB                   |     |  PUBLISHED_IN, CITES, etc.      |
     |                                 |
                                        |  Size: ~100MB                   |
                                        
```

---

## 3. Pipeline Phases

### 3.0 Data Preparation

Raw data consists of 48 jurisdiction JSONs scraped from DLA Piper's "AI Laws of the World" database (2024) and 156 academic PDFs processed through MinerU for markdown extraction. Scopus CSV provides bibliometric metadata (authors, journals, references). The pipeline assumes these exist in `data/raw/`.

---

### 3.1 Knowledge Graph Construction

#### 3.1.1 Phase 1A: Chunking

**Method**: Semantic chunking at sentence boundaries, respecting document structure.

**Input**: 204 unified documents  
**Output**: 25,131 chunks (~500 tokens avg)

Each chunk contains:
- `chunk_id`: Unique identifier
- `text`: Chunk content
- `embedding`: 1024-dim BGE-M3 vector
- `metadata`: `{doc_type, jurisdiction|scopus_id, section_title}`

**Embedding model**: BGE-M3 (Chen et al., 2024)  chosen for multilingual support across regulatory texts.

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

1. **Exact deduplication**: Normalize case/whitespace (200K  143K)
2. **Semantic clustering**: FAISS HNSW blocking + tiered thresholds (143K  55,695)

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

**Validation**: Relations only emitted if both subject and object exist in the entity index  construction-time grounding.

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
- Citation entities  Scopus references (provenance-constrained matching)
- Country entities  Jurisdiction nodes (41 SAME_AS links)

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

**Method**: Dual-strategy retrieval combining entity-centric graph traversal with semantic similarity search. This applies RAKG's retrieval strategies  Corpus Retrospective and Graph Structure Retrieval  at query time.

**Literature grounding**:
- Entity extraction from queries: GraphRAG survey (Han et al., 2025), E2GraphRAG (2025)
- Dual-channel retrieval: KG-RAG (Nature, 2025), KG2RAG (2025)
- Corpus Retrospective + Graph Structure: RAKG (Zhang et al., 2025)  applied at query time rather than construction time

**Implementation modes** (for ablation testing, see S 3.3.5):

| Mode | Entity-Centric | Semantic Similarity | Purpose |
|------|----------------|---------------------|---------|
| `naive` |  | ... | Baseline: vector search only |
| `graphrag` | ... |  | Graph traversal only |
| `hybrid` | ... | ... | Full system: both strategies merged |

---

#### 3.3.1 Query Understanding

The retrieval system implements two complementary strategies:

| Strategy | Mechanism | Strength |
|----------|-----------|----------|
| **Entity-Centric** | Extract entities  resolve to graph  traverse relationships  retrieve grounded chunks | Multi-hop reasoning, cross-document connections |
| **Semantic Similarity** | Embed query  vector search  retrieve similar chunks | Direct lexical/semantic matches |

Both strategies share a common **query parsing** step, then diverge.

---

**Common: Query Parsing**

Every query is first parsed to extract:
- **Query embedding**: Full query embedded via BGE-M3 (used by both strategies)
- **Filters**: Jurisdiction codes and document types detected via regex

```python
class ParsedQuery:
    raw_query: str
    embedding: np.ndarray          # BGE-M3 1024-dim
    filters: QueryFilters          # {jurisdictions: [], doc_types: []}
```

---

**Strategy 1: Entity-Centric Retrieval**

This strategy leverages the knowledge graph structure for multi-hop reasoning.

**Step 1: Entity Extraction**

LLM extracts entity mentions from the query (consistent with Phase 1B methodology):
- Model: Mistral-7B via Together.ai
- Output: List of `{name, type}` mentions

```
Query: "What does the EU AI Act say about facial recognition?"

Extracted:
   EU AI Act (Legislation)
   facial recognition (Technology)
```

**Step 2: Entity Resolution**

Map extracted mentions to canonical graph entities:
- Embed each mention (BGE-M3)
- FAISS k-NN search against entity index
- Threshold filter (cosine > 0.75)
- Return top-k matches per mention (default k=3)

**Multiple matches are intentional**: A single query mention like "EU AI Act" may resolve to multiple related entities (e.g., "AI Act", "EU AI Act Article 5", "European AI Act"). This improves recall  the graph traversal phase will find connections from any of these entry points.

```
Query: "What does the EU AI Act say about facial recognition?"

Resolved (top 3 per mention):
   "EU AI Act" 
      - AI Act (Legislation): confidence=0.92
      - Article 5 of the EU AI Act: confidence=0.80
      - European AI Act: confidence=0.78
   "facial recognition" 
      - facial recognition (Technology): confidence=0.76
      - face detection (Technology): confidence=0.71
```

**Step 3: Graph Traversal** (continues in S 3.3.2)

From resolved entities, the system traverses the knowledge graph to find connecting paths and retrieve grounded chunks.

---

**Strategy 2: Semantic Similarity Retrieval**

This strategy bypasses the graph entirely  traditional vector search.

**Step 1: Query Embedding**

The full query is embedded (already computed in parsing step).

**Step 2: Chunk Search**

FAISS k-NN search against chunk embedding index:
- Return top-K chunks by cosine similarity
- Apply jurisdiction/doc_type filters from parsed query

```
Query embedding  FAISS  Top 15 chunks (cosine > 0.65)
```

This path retrieves chunks that are semantically similar to the query but may not contain the specific entities mentioned.

---

**Strategy Comparison**

| Aspect | Entity-Centric | Semantic Similarity |
|--------|----------------|---------------------|
| **Input** | Extracted entity mentions | Full query text |
| **Search space** | Entity index  Graph  Chunks | Chunk index directly |
| **Grounding** | Chunks guaranteed to contain entities | Chunks may drift from query intent |
| **Multi-hop** | ... Traverses entity relationships |  Single-hop similarity only |
| **Coverage** | May miss chunks without extracted entities | May find relevant chunks entity extraction missed |

**Why both?** Entity-centric excels at structured queries ("How does X relate to Y?") but depends on successful entity extraction. Semantic similarity provides fallback coverage and catches relevant chunks that don't mention entities explicitly.

**Retrieval Configuration Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `entity_resolution_top_k` | 3 | Max fuzzy matches per extracted entity mention |
| `entity_resolution_threshold` | 0.75 | Min cosine similarity for entity match |
| `chunk_retrieval_top_k` | 15 | Max chunks from semantic similarity search |
| `chunk_similarity_threshold` | 0.65 | Min cosine similarity for chunk retrieval |
| `provenance_relation_bonus` | 0.3 | Score boost for chunks with PCST relations |
| `provenance_entity_bonus` | 0.2 | Score boost for chunks with query entities |
| `jurisdiction_match_bonus` | 0.1 | Score boost for jurisdiction filter match |
| `doc_type_match_bonus` | 0.15 | Score boost for doc type filter match |

These parameters are tunable via `RetrievalConfig` and tested for sensitivity in `test_parameter_tuning.py`.

---

#### 3.3.2 Context Retrieval

**Entity-Centric Strategy (continued)**

**Step 1: Graph Traversal via PCST**

The hub node problem (entities like "AI system" with 17K+ relations) is solved using **Prize-Collecting Steiner Tree (PCST)** optimization, following G-Retriever (He et al., 2024, NeurIPS).

**Core insight**: Instead of expanding from each entity independently (explosion), find the **minimal subgraph connecting all query entities**. This automatically prunes irrelevant relations  only paths that connect query entities survive.

**Algorithm**:
1. Assign **prizes** to resolved query entities (high) and their neighbors (decaying)
2. Assign **costs** to edges (constant or inverse-frequency weighted)
3. Solve PCST: find connected subgraph maximizing prizes minus costs
4. Return subgraph entities and relations

**Implementation**: Neo4j Graph Data Science library provides PCST (2-approximate algorithm).

*Note: Actual implementation uses Python GDS client (`graphdatascience` library), not raw Cypher. The queries below illustrate the logic:*

```cypher
// Project subgraph around query entities
CALL gds.graph.project.cypher(
  'queryGraph',
  'MATCH (n:Entity) WHERE n.entity_id IN $seed_ids 
   OR (n)-[:RELATION*1..2]-(:Entity {entity_id: $seed_ids})
   RETURN id(n) AS id, 
          CASE WHEN n.entity_id IN $seed_ids THEN 10.0 ELSE 1.0 END AS prize',
  'MATCH (n)-[r:RELATION]-(m) RETURN id(n) AS source, id(m) AS target, 1.0 AS cost'
)

// Run PCST to find connecting subgraph
CALL gds.steinerTree.stream('queryGraph', {
  sourceNode: $primary_entity,
  targetNodes: $other_entities,
  relationshipWeightProperty: 'cost'
})
YIELD nodeId, parentId, weight
```

*Prize/cost values are illustrative  actual implementation uses GDS defaults.*

**Why PCST solves hub explosion**:

| Query | Without PCST | With PCST |
|-------|--------------|-----------|
| "EU AI Act + facial recognition" | Expand "AI system"  17K relations | Find 3 paths connecting EU AI Act to facial recognition |
| "GDPR + transparency" | Each entity expands independently | Minimal subgraph linking both concepts |

**Literature grounding**:
- G-Retriever (He et al., 2024): "formulate subgraph retrieval as a Prize-Collecting Steiner Tree optimization problem"
- Neo4j GDS: Production-ready 2-approximate PCST implementation

**Fallback**: If PCST returns empty (disconnected entities), fall back to independent 1-hop expansion with degree limits.

**Design Note: MMR vs Provenance Ranking**

The pipeline uses two different ranking strategies at different phases:

| Phase | Strategy | Goal | Method |
|-------|----------|------|--------|
| **Phase 1D** (Construction) | MMR | Diversity | Select diverse chunks as ground truth for relation extraction |
| **Phase 3** (Retrieval) | Provenance | Precision | Rank chunks by entity grounding, not just similarity |

**Why two strategies?**

- **MMR (Maximal Marginal Relevance)** at construction time ensures broad coverage during relation extraction. It avoids the "echo chamber" problem where similar chunks produce redundant relations.

- **Provenance ranking** at query time prioritizes chunks that contain the query entities (graph-grounded) over chunks that are merely semantically similar (potential drift).

**Complementarity**: MMR builds diverse relation knowledge  Provenance exploits that structure at query time. They combat different problems: MMR fights redundancy in training, Provenance fights semantic drift in retrieval.

**Step 2: Corpus Retrospective** (find chunks for each entity):

```cypher
MATCH (e:Entity {entity_id: $eid})-[:EXTRACTED_FROM]->(c:Chunk)
RETURN c.chunk_id, c.text, c.metadata
```

This retrieves chunks where entities were originally extracted  guaranteed entity presence.

**Semantic Similarity Strategy (continued)**

The chunks retrieved via vector search (S 3.3.1) are added to the candidate pool. These may surface relevant content that entity extraction missed.

**Jurisdiction filtering**: Applied to both strategies via chunk metadata.

---

#### 3.3.3 Ranking & Prompt Assembly

**Scoring function**:

```python
score = base_similarity                    # Cosine similarity to query
      + provenance_bonus                   # +0.3 relation, +0.2 entity, +0.0 semantic
      + jurisdiction_match_bonus           # +0.1 if matches filter
      + doc_type_match_bonus               # +0.15 if matches filter (soft preference)
```

**Provenance bonus tiers** (entity-centric chunks ranked higher):

| Source | Bonus | Rationale |
|--------|-------|-----------|
| **Relation provenance** | +0.3 | Chunk where PCST relation was extracted  directly evidences the graph path |
| **Entity provenance** | +0.2 | Chunk from EXTRACTED_FROM  contains query entity but not necessarily the relation |
| **Semantic only** | +0.0 | Vector similarity match, no graph grounding |

**Filter bonuses** (soft preferences, not hard filters):

| Filter | Bonus | Behavior |
|--------|-------|----------|
| Jurisdiction match | +0.1 | Chunks from matching jurisdiction ranked higher |
| Doc type match | +0.15 | Chunks from matching doc type (regulation/academic) ranked higher |

**Note**: These are bonuses, not filters. Chunks that don't match still appear in results but rank lower. This prevents over-filtering when relevant content exists in unexpected sources.

**Assembly**:
1. Deduplicate chunks (keep highest score)
2. Sort by score, take top-K within token budget (~15-20 chunks)
3. Build structured prompt (format below)

**Prompt format** *(design  implementation pending in Phase 3.3.4)*:

```
GRAPH STRUCTURE (relationships discovered):
   EU AI Act --prohibits--> real-time biometric identification
   real-time biometric identification --includes--> facial recognition

KEY ENTITIES:
   EU AI Act (Regulation): Regulation (EU) 2024/1689...
   facial recognition (Technology): AI systems that identify...

RELEVANT TEXT SOURCES:
  [1] "Article 5 of the EU AI Act prohibits the use of..."
  [2] "Facial recognition systems fall under..."

QUESTION: {user_query}

INSTRUCTIONS: Answer using the graph structure and source text. Cite sources by number.
```

**Why this format matters**: The LLM sees graph paths (structural knowledge), entity context (semantic grounding), and source text (verbatim evidence). This is what makes it GraphRAG  the graph structure informs the answer, not just the chunks.

---

#### 3.3.4 Answer Generation

**Model**: Claude Haiku 3.5 via Anthropic

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

**Ablation Study Design**

The dual-strategy architecture enables controlled comparison of retrieval approaches:

| Mode | Entity-Centric | Semantic Similarity | Purpose |
|------|----------------|---------------------|---------|
| `naive` |  Off | ... On | Baseline: chunk embedding similarity only |
| `graphrag` | ... On |  Off | Graph traversal: PCST + corpus retrospective only |
| `hybrid` | ... On | ... On | Full system: both strategies + provenance ranking |

**What each mode tests**:

- **Naive RAG** (`naive`): Traditional transformer-based retrieval. Query embedded  FAISS chunk search  top-K by cosine similarity. No entity extraction, no graph traversal. This is the "what if we didn't build a graph" baseline.

- **GraphRAG** (`graphrag`): Pure entity-centric retrieval. Query entities extracted  resolved to graph  PCST finds connecting paths  corpus retrospective retrieves chunks. Tests whether graph structure alone provides relevant context.

- **Hybrid** (`hybrid`): Both strategies contribute chunks, merged via provenance ranking (entity-centric weighted higher). Tests whether combining approaches outperforms either alone.

**Test Set**: 20-30 questions covering:
- Single-jurisdiction factual: "What is Spain's definition of AI system?"
- Cross-jurisdictional: "How do EU and US approaches to facial recognition differ?"
- Academic-regulatory bridge: "What research supports the EU AI Act's transparency requirements?"
- Multi-hop reasoning: "Which regulations cite transparency requirements that academic papers criticize?"
- Negative: "What does Brazil say about quantum computing?" (not in corpus)

**Metrics**:

| Metric | Description | Applies To |
|--------|-------------|------------|
| **Retrieval precision** | % of retrieved chunks containing query entities | All modes |
| **Retrieval recall** | % of relevant chunks retrieved (manual annotation) | All modes |
| **Answer relevance** | Manual 1-5 rating | All modes |
| **Factual accuracy** | Binary: claims supported by citations? | All modes |
| **Citation precision** | % of citations that support their claims | All modes |
| **Source diversity** | # unique jurisdictions/papers in response | All modes |
| **Strategy contribution** | % of final chunks from entity-centric vs semantic | `hybrid` only |

**Hypothesis**: 
- `graphrag` > `naive` for multi-hop and cross-jurisdictional queries (graph paths matter)
- `naive`  `graphrag` for simple factual queries (either finds the chunk)
- `hybrid`  both for robustness (catches what either misses)

**Parameter Sensitivity Testing**

Beyond mode comparison, `test_parameter_tuning.py` validates configuration robustness:

| Test | Parameters Varied | Validates |
|------|-------------------|-----------|
| Entity top_k sensitivity | k=1,3,5,10 | Impact on subgraph size, diminishing returns |
| Ranking weight sensitivity | Provenance/jurisdiction/doc_type bonuses | Score distribution stability |
| Cross-query consistency | Fixed params across query types | Parameters generalize, not overfit to specific queries |

---

## 4. Methodological Contributions

### 4.1 Integrated Academic-Semantic Knowledge Graph

**The core contribution**: A knowledge graph that integrates bibliographic structure with semantic context, enabling queries that traverse both citation networks and conceptual relationships.

Traditional approaches separate these concerns:
- **Bibliometric systems** (citation networks): Track who cites whom, but lose semantic context
- **Semantic extraction** (NLP pipelines): Extract concepts and relations, but ignore academic structure

Our approach integrates both through shared chunk provenance:

```

|  CHUNK: "Floridi's work on digital ethics (Floridi, 2018)      |
+-- influenced the EU AI Act's transparency requirements"   |
|                                                                 |
|  Contains BOTH:                                                 |
+-- Citation entity: "Floridi (2018)"                         |
+-- Semantic entities: "digital ethics", "transparency"        |
+-- Regulation entity: "EU AI Act"                            |
|                                                                 |
|  Extracted relations:                                           |
+-- Floridi (2018) --discusses--> digital ethics              |
+-- EU AI Act --requires--> transparency                       |
+-- digital ethics --influenced--> transparency requirements   |

```

**Key design decisions**:

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Citation extraction | From semantic chunks, not reference lists | Captures what the citation *discusses* in context |
| Citation relations | Fixed `discusses` predicate | Semantic meaning; actual citation links come from Scopus |
| Scopus enrichment | Match extracted citations to bibliographic metadata | Adds L2 publications, journals, CITES structure |
| Chunk overlap | Preserved, not deduplicated | Bridge between academic and semantic entity types |

**What this enables**:
- "What does Floridi's work say about ethics?"  Traverses citation  `discusses`  concept
- "What research supports the EU AI Act?"  Traverses regulation  shared chunks  citations
- "Compare academic and regulatory perspectives on transparency"  Both paths through shared chunks

---

### 4.2 Dual-Strategy Retrieval with Ablation

**Design choice**: Implement both entity-centric (graph traversal) and semantic similarity (vector search) as parallel strategies with configurable modes.

This isn't novel  dual-channel retrieval appears in KG-RAG [Sanmartin, 2024], KG2RAG [Feng et al., 2025], and others. Our contribution is using it for **controlled evaluation**:

| Mode | Entity-Centric | Semantic | Purpose |
|------|----------------|----------|---------|
| `naive` |  | ... | Baseline: "What if we didn't build the graph?" |
| `graphrag` | ... |  | Test: "Does graph structure alone suffice?" |
| `hybrid` | ... | ... | Production: Best of both |

**Why this matters for thesis**: Clean ablation answers whether the graph construction effort (55K entities, 105K relations, ~$43 API cost) actually improves retrieval over naive chunking.

---

### 4.3 Applied Techniques (Credited)

We combine established techniques from the literature. Credit where due:

| Technique | Source | Our Application |
|-----------|--------|-----------------|
| **RAKG methodology** | Zhang et al. (2025) | Pre-entity extraction, corpus retrospective retrieval |
| **RAGulating domain patterns** | Agarwal et al. (2025) | OpenIE for regulatory text, multi-jurisdictional handling |
| **FAISS blocking** | Papadakis et al. (2021) | O(n log n) candidate filtering for disambiguation |
| **Tiered thresholds** | Entity resolution literature | 0.90  0.85  0.82 progressive clustering |
| **PCST subgraph extraction** | He et al. (2024), G-Retriever | Hub node problem via Neo4j GDS |
| **BGE-M3 embeddings** | Chen et al. (2024) | Multilingual 1024-dim vectors |

**Adaptation note**: RAKG assumes 50-150 entities per document. Our corpus averages 755 entities/doc (academic papers are denser). This required the FAISS blocking adaptation  not a novel technique, but a necessary scaling fix.

---

### 4.4 Cross-Jurisdictional Entity Linking

48 jurisdictions with overlapping regulatory concepts required entity normalization across legal systems.

| Challenge | Solution |
|-----------|----------|
| Same concept, different names | "high-risk AI" (EU)  "critical AI" (Canada)  merged via embedding similarity |
| Country mentions  Jurisdiction | `SAME_AS` relations link country entities to jurisdiction nodes |
| Multi-jurisdiction chunks | DLA Piper summaries reference multiple jurisdictions  chunk metadata tracks all |

**Not a methodological contribution**  standard entity resolution. Documented for reproducibility.

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

## 6. Deployment

### 6.1 Architecture

The system runs entirely in Docker for reproducibility and ease of evaluation.

```

|  Docker Compose                                              |
|                                                              |
|               |
|  | app                 |       | neo4j               |      |
|  |                     |       |                     |      |
|  |  Python 3.10       |  |  Neo4j 5.12        |      |
|  |  Streamlit UI      | Bolt  |  GDS Plugin        |      |
|  |  FAISS indices     |       |  PCST Algorithm    |      |
|  |                     |       |                     |      |
|  | Port: 8501          |       | Port: 7474, 7687    |      |
|               |
|           |                             |                    |
|                                                            |
|      |
|  | Mounted Volume: ./data                              |    |
|  +-- faiss/           (indices ~320MB)              |    |
|  +-- processed/       (entities, relations)         |    |
|  +-- neo4j_import/    (initial graph data)          |    |
|      |

         |
          External API

| Together.ai         |
| (LLM: Mistral-7B)   |

```

### 6.2 Why Docker?

| Requirement | Solution |
|-------------|----------|
| Neo4j GDS plugin (for PCST) | Not available on Aura Free; Docker Enterprise includes it |
| Reproducible evaluation | Evaluator runs `docker-compose up`, everything works |
| No manual installation | All dependencies bundled in containers |
| Consistent environment | Same setup on any machine |

### 6.3 Container Details

**neo4j container**:
- Image: `neo4j:5.12.0-enterprise`
- Plugins: Graph Data Science (GDS) for PCST algorithm
- Memory: 2-4GB heap, 2GB page cache
- Ports: 7474 (browser), 7687 (Bolt protocol)
- Data: Persisted in Docker volume

**app container**:
- Base: `python:3.10-slim`
- Includes: Streamlit, FAISS, neo4j driver, Together.ai client
- Ports: 8501 (Streamlit UI)
- Mounts: `./data` volume for indices and processed files

### 6.4 Data Artifacts

Pre-built artifacts required (not regenerated at startup):

| Artifact | Size | Location | Contents |
|----------|------|----------|----------|
| FAISS chunk index | ~250MB | `data/faiss/chunk_embeddings.index` | 25K chunk vectors |
| FAISS entity index | ~60MB | `data/faiss/entity_embeddings.index` | 55K entity vectors |
| ID maps | ~5MB | `data/faiss/*_id_map.json` | Index  ID mappings |
| Normalized entities | ~30MB | `data/processed/normalized_entities.json` | 55K entities with metadata |
| Relations | ~20MB | `data/processed/relations.jsonl` | 105K triplets |
| Neo4j import | ~50MB | `data/neo4j_import/` | Cypher import scripts |

**Distribution**: Artifacts hosted on cloud storage (Google Drive/Dropbox), downloaded via script.

### 6.5 Startup Sequence

```bash
# 1. Clone repository
git clone https://github.com/<user>/graphrag-thesis
cd graphrag-thesis

# 2. Download pre-built data artifacts
./scripts/download_data.sh

# 3. Configure API key
echo "TOGETHER_API_KEY=your_key_here" > .env

# 4. Start everything
docker-compose up -d

# 5. Wait for Neo4j to initialize (~60 seconds)
docker-compose logs -f neo4j  # Watch for "Started"

# 6. Access
# Web UI: http://localhost:8501
# Neo4j Browser: http://localhost:7474 (neo4j/graphrag2024)
```

### 6.6 Validation Commands

```bash
# Check services running
docker-compose ps

# Verify GDS plugin
docker exec graphrag_neo4j cypher-shell -u neo4j -p graphrag2024 \
  "CALL gds.version() YIELD version RETURN version"

# Verify entity count
docker exec graphrag_neo4j cypher-shell -u neo4j -p graphrag2024 \
  "MATCH (n:Entity) RETURN count(n) AS entities"
# Expected: 55695

# Verify relation count  
docker exec graphrag_neo4j cypher-shell -u neo4j -p graphrag2024 \
  "MATCH ()-[r:RELATION]->() RETURN count(r) AS relations"
# Expected: ~105000
```

### 6.7 Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8GB | 16GB |
| Disk | 10GB | 20GB |
| CPU | 2 cores | 4 cores |
| Docker | 4.0+ | Latest |

---

## 7. Limitations & Technical Debt

This section documents known issues and lessons learned, intended for a post-submission cleanup branch.

### 7.1 Phase 1A: Chunking Issues

| Issue | Impact | Fix |
|-------|--------|-----|
| **Garbage content not filtered** | Some chunks contain OCR artifacts, malformed text, or non-human-readable content (especially from PDF parsing) | Pre-chunking pass with perplexity/coherence check  reject chunks that fail basic "is this human text?" heuristics |
| **Duplicate regulatory content** | EU member state documents repeat identical EU AI Act passages | Chunk-level deduplication via high-threshold embedding similarity (>0.95) with provenance tracking to retain one canonical source |
| **No chunk quality pass** | Can't assess informativeness or filter low-signal chunks | Merge with garbage filtering into unified "chunk quality gate" before entity extraction |

**Note**: Non-English content comes from academic papers (parsed via MinerU), not regulations. DLA Piper regulatory text is English-only.

### 7.2 Phase 1B: Entity Extraction Issues

| Issue | Impact | Fix |
|-------|--------|-----|
| **Compound entities not split** | "transparency and consent requirements" extracted as one entity instead of two | Prompt engineering: instruct LLM to split "X and Y" patterns into separate entities |
| **Modifiers not normalized** | "enhanced transparency requirements", "minimum transparency rules" are separate entities instead of merging to "transparency requirements" | Extract base concept + modifier as attribute, or strip modifiers entirely |
| **Prompt not designed for academic entities** | Pre-entity extraction prompt was generic; didn't anticipate academic entity types (Citations, Authors, Journals) | Consider two-pass extraction: (1) general entities, (2) academic-constrained pass with type restrictions |
| **72B model overkill for actual use** | Chose Qwen-72B for "better definitions" but descriptions ended up playing minimal role  only used in disambiguation and as minor context | Use Mistral-7B throughout; definitions don't justify 6x cost |
| **No type normalization** | "Organization" vs "organisation" vs "org" inconsistencies propagate downstream | Add canonical type mapping as part of pre-entity filtering pipeline (see S 6.2.1) |

**Cascading effect**: Compound entities can't be fixed by disambiguation  "transparency and consent requirements" has a different embedding than "transparency requirements", so they never cluster together. The fix must happen at extraction time.

#### 7.2.1 Type Normalization & Pre-Entity Filtering

This deserves more serious treatment. Should be a unified pipeline:

```
Raw pre-entities 
   Type normalization (canonical mapping)
   Quality filter (garbage removal)  
   Deduplication (exact + fuzzy)
   Output: clean pre-entities for disambiguation
```

The same type normalization should inform query parsing  resolved entities should match canonical types.

### 7.3 Phase 1C: Disambiguation Issues

| Issue | Impact | Fix |
|-------|--------|-----|
| **Embeddings inherited from pre-entities** | Normalized entities retain embeddings computed on PRE-entity text, not final canonical form. The merged entity "EU AI Act" carries embedding from whichever variant was processed first  potential bias | Normalized entities should be **embeddingless**; re-embed canonical `{name} ({type}): {description}` after disambiguation |
| **Hash IDs generated post-1D** | Deterministic IDs come too late in pipeline; earlier phases use unstable identifiers | Generate hash IDs immediately after disambiguation, before relation extraction |
| **No alias tracking** | Lost which surface forms merged into canonical | Store `aliases: List[str]` on each entity for provenance and query matching |

#### 7.3.1 Proposed Embedding Architecture

```
Phase 1C Output:
  normalized_entities.json        # NO embeddings, just {id, name, type, description, aliases}
  
Phase 1C.5 (new):
  normalized_entities_embedded.json  # Add embeddings to canonical form

Benefits:
  - Phase 2 loading is cheaper (no dragging 1024-dim vectors through JSON)
  - Embeddings represent final entity, not arbitrary pre-entity variant
  - Clear separation: structure vs. vectors
```

### 7.4 Architectural Issues

| Issue | Impact | Fix |
|-------|--------|-----|
| **Inconsistent naming conventions** | Confusing codebase: `embedded` vs `embedding`, `retrieval` vs `retriever`, `_processor` vs no suffix | Standardize (see S 6.4.1) |
| **No file naming standard** | Tests named inconsistently: `test_X.py`, `X_test.py`, `test_X_integration.py` | Convention: `test_{module}.py` for unit, `test_{feature}_integration.py` for integration |
| **Nested JSON metadata** | Chunk metadata nested inside chunk object; hard to query and wastes space on repeated fields | Separate `chunk_metadata.json` keyed by chunk_id |
| **No common config** | Thresholds, model names, paths scattered across files | Centralized `src/config/` with phase-specific configs inheriting from base |
| **Minimal code reuse** | Only `embeddings.py` is truly shared; everything else duplicated | Extract to `src/utils/`: see S 6.4.2 |
| **Inconsistent file formats** | Mix of `.json` and `.jsonl`, inconsistent schemas | Standardize: JSONL for streaming/large files, JSON for lookups; document all schemas |

#### 7.4.1 Naming Conventions (Proposed)

| Category | Convention | Examples |
|----------|------------|----------|
| **Modules** | Noun or verb-noun, lowercase | `embeddings.py`, `chunk_retriever.py` |
| **Classes** | Role suffix: `*Processor` (orchestrates), `*Retriever` (fetches), `*Builder` (constructs), `*Parser` (parses), `*Pipeline` (end-to-end flow) | `EntityProcessor`, `ChunkRetriever`, `PromptBuilder` |
| **Embeddings** | `embedding` (singular = one vector), `embeddings` (plural = module/collection) | `query_embedding`, `embeddings.py` |
| **Data files** | `{entity}_{state}.{ext}` | `entities_normalized.json`, `chunks_embedded.jsonl` |
| **Test files** | `test_{module}.py` (unit), `test_{feature}_integration.py` (integration) | `test_query_parser.py`, `test_retrieval_integration.py` |
| **Config** | `{phase}_config.py` or `config.py` per module | `retrieval_config.py` |

**Cleanup task**: Full codebase audit against these conventions in `cleanup/technical-debt` branch.

#### 7.4.2 Code That Should Be Utils

| Pattern | Current State | Should Be |
|---------|---------------|-----------|
| **LLM prompting** | Each processor has own Together.ai client setup, prompt formatting | `src/utils/llm.py`: `LLMClient` with `prompt()`, `prompt_json()`, retry logic |
| **JSON parsing from LLM** | Every LLM call has own try/except, regex cleanup for markdown fences, schema validation | `src/utils/llm.py`: `parse_llm_json()` with fence stripping, fallback extraction, schema validation |
| **Batch processing** | Duplicated ThreadPoolExecutor patterns, progress bars, checkpointing | `src/utils/batch.py`: `BatchProcessor` base class |
| **Testing fixtures** | Each test file creates own mocks, temp files | `src/utils/testing.py`: shared fixtures, mock factories |
| **File I/O** | Repeated JSON/JSONL loading patterns | `src/utils/io.py`: `load_json()`, `load_jsonl()`, `save_checkpoint()` |
| **Embeddings** | ... Already extracted | Keep as-is |

### 7.5 Embedding & Retrieval Issues

| Issue | Impact | Root Cause |
|-------|--------|------------|
| **Type dominates retrieval similarity** | Query "GDPR transparency" returns 11x "Regulatory Requirement" entities instead of GDPR itself | Type strings not normalized  index has inconsistent types ("Regulatory Requirement", "Legal Provision", "Legislation") that don't align with query extraction types |
| **Query types don't match index types** | LLM extracts "Regulatory Concept", index has "Legislation" | No canonical type vocabulary enforced across extraction and resolution |
| **No query caching** | Repeated queries re-run LLM extraction (~2-5 sec) | Missing LRU cache for entity extraction and embeddings |

**Key insight**: Type-aware retrieval is correct and desirable. The problem is type normalization was never done (S 7.2.1), so types are inconsistent garbage.

**Fix chain**:
1. Normalize types in entity index to canonical 9 types
2. Enforce same vocabulary in query extraction prompt
3. Then type-aware retrieval works as intended

**Short-term workaround**: Fuzzy type matching (map "Regulatory Concept"  "Regulation") at query time.

### 7.6 Model Selection

**Lesson learned**: Should have used **Mistral-7B-Instruct-v0.3** throughout.

| Phase | Used | Should Have Used | Why |
|-------|------|------------------|-----|
| 1B Entity extraction | Qwen-72B | Mistral-7B | 72B overkill  descriptions not leveraged |
| 1C Disambiguation | Qwen-7B | Mistral-7B | Consistency |
| 1D Relations | Mistral-7B  |  | Already correct |
| 3.3.1 Query extraction | Mistral-7B  |  | Already correct |

**Cost implication**: Qwen-72B cost ~$30 for Phase 1B. Mistral-7B would have been ~$5.

### 7.7 Future Work (Post-Submission Branch)

Priority order for cleanup:

1. **Chunk quality gate**  Garbage filtering + deduplication in one pass
2. **Type normalization pipeline**  Serious pre-entity cleaning  
3. **Re-embed canonical entities**  Fix embedding inheritance problem
4. **Comprehensive citation extraction**  See below
5. **Code utils extraction**  LLM client, batch processing, testing
6. **Common config**  Centralized thresholds and paths
7. **File format standardization**  JSONL + schema docs

#### 7.7.1 Comprehensive Citation Extraction

**Current limitation**: Citation entities only get `discusses` relations extracted when they co-occur with semantic entities in MMR-selected chunks. This misses many valid citation relations.

**Problem**: "Floridi (2018) discusses digital ethics" is valuable even if that chunk doesn't mention "EU AI Act" or other query entities. Current two-track approach under-extracts citation networks.

**Proposed fix**: Citation track should use **comprehensive coverage** (all chunks where citation appears), not MMR sampling:

| Track | Entity Types | Chunk Selection | Rationale |
|-------|--------------|-----------------|-----------|
| Semantic | Concepts, Orgs, Tech, Regs | MMR-selected (diverse) | Avoid redundant semantic relations |
| **Citation** | Citations, Authors | **Comprehensive** (all EXTRACTED_FROM) | Citation networks should be complete |

**Implementation options**:
1. **Backtrace via provenance**: If `entity.type in ['Citation', 'Author']`, process ALL chunks from `EXTRACTED_FROM` instead of MMR sample
2. **Separate pass**: Run citation extraction as independent phase after semantic extraction
3. **Flag-based**: Add `contains_citation` flag to chunks, ensure all flagged chunks processed

**Why this matters**: Complete citation networks enable academic queries like "What papers cite research on transparency?" without gaps from sampling.

**Branch plan**: Create `cleanup/technical-debt` branch after Phase 3 delivery, before final thesis submission.

---

## 8. References

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
