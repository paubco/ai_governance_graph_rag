# Architecture

**Project**: AI Governance GraphRAG Pipeline  
**Version**: 2.0  
**Last Updated**: December 4, 2025

---

## 1. Overview

### Goal

Build a knowledge graph for AI governance research using adapted RAKG methodology (Zhou et al., 2025) to answer complex regulatory queries across jurisdictions.

### Data Sources

| Source | Content | Count | Origin |
|--------|---------|-------|--------|
| **Regulations** | Jurisdiction metadata + legal text | 48 | DLA Piper (2024) web scrape |
| **Academic Papers** | Scopus metadata CSV + PDFs (MinerU-parsed) | 158 | Scopus export |
| **Derived Metadata** | Authors, Journals, References | 572 / 119 / 1,513 | Scopus CSV |

**Note**: Both sources provide metadata AND text. DLA Piper provides jurisdiction codes and scraped legal text. Scopus provides bibliometric metadata and PDFs which are parsed to markdown via MinerU.

### Tech Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| **LLM (Extraction)** | Qwen-72B via Together.ai | Phase 1B entity extraction ($30) |
| **LLM (Disambiguation)** | Qwen-7B via Together.ai | Phase 1C |
| **LLM (Relations)** | Mistral-7B via Together.ai | Phase 1D (Qwen-7B has JSON infinite loop bug) |
| **Embeddings** | BGE-M3 (1024-dim, multilingual) | All phases |
| **Blocking** | FAISS HNSW | Phase 1C clustering/candidate generation |
| **Graph DB** | Neo4j 5.x | Graph traversal, no embeddings |
| **Vector Store** | TBD (FAISS file or external) | Embeddings too large for Neo4j (1.5GB) |
| **Hardware** | RTX 3060 GPU (12GB VRAM) | Local embedding computation |

### Methodology

| Component | Source | Notes |
|-----------|--------|-------|
| Entity extraction | RAKG (Zhou et al., 2025) | Pre-entity extraction per chunk |
| Entity disambiguation | **Adapted** from RAKG | FAISS blocking replaces VecJudge (see § 3.1) |
| Relation extraction base | RAGulating Compliance (Chen et al., 2024) | OpenIE-style triplets |
| Two-track extraction | **Novel contribution** | Semantic vs academic entity separation |
| Blocking approach | Papadakis et al. (2021), Malkov & Yashunin (2020) | Entity resolution literature |

---

## 2. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                               │
│                      src/ingestion/                                 │
│                                                                     │
│  ┌─────────────────────────┐    ┌─────────────────────────────────┐│
│  │ DLA Piper (48 regs)     │    │ Scopus (158 papers)             ││
│  │ • Jurisdiction metadata │    │ • Bibliometric CSV              ││
│  │ • Legal text (scraped)  │    │ • PDFs → MinerU → markdown      ││
│  └───────────┬─────────────┘    └────────────────┬────────────────┘│
│              └────────────────┬──────────────────┘                 │
│                               ▼                                     │
│                   ┌───────────────────┐                            │
│                   │  DocumentLoader   │ → 206 unified documents    │
│                   └─────────┬─────────┘                            │
└─────────────────────────────┼──────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  PHASE 1A: CHUNKING                                 │
│              src/processing/chunking/                               │
│                                                                     │
│  SemanticChunker → ChunkProcessor → 25,131 chunks                  │
│                                                                     │
│  Each chunk has metadata:                                           │
│  • doc_type: "regulation" | "academic_paper"                       │
│  • jurisdiction: "ES" (regulations only)                           │
│  • scopus_id: "85123456" (papers only)                             │
│  • section_title: "User transparency"                              │
│                                                                     │
│  Output: data/processed/chunks/chunks_embedded.json                 │
└─────────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  PHASE 1B: ENTITY EXTRACTION                        │
│              src/processing/entities/                               │
│                                                                     │
│  Model: Qwen-72B via Together.ai ($30)                             │
│  Method: RAKG pre-entity extraction (Zhou et al., 2025)            │
│                                                                     │
│  RAKGEntityExtractor → ParallelEntityProcessor → ~200K pre-entities│
│                                                                     │
│  Output: data/interim/entities/pre_entities.json                    │
│  ⚠️  Expensive to re-run - checkpoint frequently                   │
└─────────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  PHASE 1C: ENTITY DISAMBIGUATION                    │
│              src/processing/entities/                               │
│                                                                     │
│  *** ADAPTED FROM RAKG - See § 3.1 ***                             │
│                                                                     │
│  Model: Qwen-7B (disambiguation decisions)                         │
│  Blocking: FAISS HNSW (candidate pair generation)                  │
│                                                                     │
│  Pipeline:                                                          │
│  ┌────────────────────┐                                            │
│  │filter_pre_entities │ → Exact dedup (200K → 143K)                │
│  └─────────┬──────────┘                                            │
│            ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  4-Stage Tiered Threshold Pipeline                        │      │
│  │  Stage 1: FAISS HNSW blocking (cosine > 0.70)            │      │
│  │  Stage 2: Aggressive Merge (cosine > 0.90)               │      │
│  │  Stage 3: Conservative Merge (cosine > 0.85)             │      │
│  │  Stage 4: Final Polish (cosine > 0.82) → 76K final       │      │
│  └──────────────────────────────────────────────────────────┘      │
│            ▼                                                        │
│  ┌────────────────────┐                                            │
│  │  add_entity_ids    │ → Deterministic hash IDs for O(1) lookup   │
│  └────────────────────┘                                            │
│                                                                     │
│  Output: data/processed/entities/normalized_entities_with_ids.json  │
│          data/processed/entities/entity_name_to_id.json             │
└─────────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  PHASE 1D: RELATION EXTRACTION                      │
│              src/processing/relations/                              │
│                                                                     │
│  *** TWO-TRACK STRATEGY (Novel) - See § 3.2 ***                    │
│                                                                     │
│  Model: Mistral-7B (Qwen-7B has JSON infinite loop bug)            │
│  Base method: OpenIE triplets per RAGulating (Chen et al., 2024)   │
│                                                                     │
│  ┌────────────────────────┐                                        │
│  │build_entity_cooccurrence│ → 3 typed matrices                    │
│  └───────────┬────────────┘                                        │
│              ▼                                                      │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Track 1 (Semantic): Concepts, Orgs, Tech, Regulations   │      │
│  │    • Full OpenIE extraction                              │      │
│  │    • Any predicate discovered                            │      │
│  │    • ~120K relations                                     │      │
│  ├──────────────────────────────────────────────────────────┤      │
│  │  Track 2 (Academic): Citations, Authors, Journals        │      │
│  │    • Constrained extraction                              │      │
│  │    • Fixed predicates: discusses, publishes_about        │      │
│  │    • Concept-only objects                                │      │
│  │    • ~35K relations                                      │      │
│  └──────────────────────────────────────────────────────────┘      │
│            ▼                                                        │
│  ┌────────────────────┐                                            │
│  │normalize_relations │ → Add IDs via name lookup, Neo4j format    │
│  └────────────────────┘                                            │
│                                                                     │
│  ⚠️  Relations extracted before add_entity_ids ran - IDs added     │
│      post-hoc via name matching in normalize_relations.py          │
│                                                                     │
│  Output: data/processed/neo4j_edges.jsonl (105K relations)         │
└─────────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  PHASE 2A: SCOPUS ENRICHMENT                        │
│                  src/enrichment/                                    │
│                                                                     │
│  Files: enrichment_processor.py (orchestrator)                      │
│         scopus_enricher.py (core matching)                          │
│         jurisdiction_matcher.py (country linking)                   │
│         tests/test_enrichment.py (16 unit tests)                    │
│                                                                     │
│  Pipeline (10 steps):                                               │
│  1. Parse Scopus CSV (158 pubs, 572 authors, 119 journals)         │
│  2. Parse references field (1,513 references)                       │
│  3. Identify citation entities (3-tier: type/discusses/pattern)     │
│  4. Build chunk→L1 mapping (uses 'eid' field)                       │
│  5. Match citations to references (provenance-constrained)          │
│  6. Match jurisdiction entities (41 SAME_AS links)                  │
│  7. Generate 5 relation types                                       │
│  8. Quality report                                                  │
│  9. Save outputs                                                    │
│                                                                     │
│  Output: data/processed/{entities,relations,reports}/*.json         │
└─────────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  PHASE 2B: NEO4J STORAGE                            │
│                                                                     │
│  ⚠️  Embeddings (1.5GB) stored in FAISS, NOT in Neo4j              │
│                                                                     │
│  Neo4j stores graph structure only:                                 │
│  • :Jurisdiction nodes (48) - top-level for regulations            │
│  • :Publication nodes (158 L1 + L2s) - top-level for papers        │
│  • :Chunk nodes (25K) - no embeddings, metadata only               │
│  • :Entity nodes (76K) - no embeddings                             │
│  • :Author nodes (572)                                              │
│  • :Journal nodes (119)                                             │
│                                                                     │
│  Relationships:                                                     │
│  • :CONTAINS (Jurisdiction/Publication → Chunk)                    │
│  • :EXTRACTED_FROM (Entity → Chunk) - provenance                   │
│  • :RELATION {predicate, chunk_ids} (Entity → Entity)              │
│  • :AUTHORED_BY (Publication → Author)                             │
│  • :PUBLISHED_IN (Publication → Journal)                           │
│  • :MATCHED_TO (Citation Entity → L2Publication)                   │
│  • :CITES (L1 Publication → L2Publication)                         │
│  • :SAME_AS (Country Entity → Jurisdiction)                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Methodology Adaptations

### 3.1 Phase 1C: Deviation from RAKG VecJudge

**Problem discovered**: RAKG's VecJudge methodology produced 15.2M candidate pairs on our corpus (expected: 500K-1M), requiring 77 hours and $300-400 in LLM costs.

**Root cause**: RAKG assumes 50-150 entities per document. Our corpus averages 755 entities/document due to regulatory text density, causing quadratic explosion in pairwise comparisons.

**Adaptation**: Replaced full VecJudge with FAISS HNSW blocking + tiered thresholds.

| Approach | Candidate Pairs | Time | Cost |
|----------|-----------------|------|------|
| Original RAKG VecJudge | 15.2M | 77 hrs | $300-400 |
| **Our adaptation** | 250K | 6 hrs | $6 |

**Literature grounding**:
- FAISS HNSW algorithm: Malkov, Y., & Yashunin, D. (2020). "Efficient and Robust Approximate Nearest Neighbor Search Using HNSW Graphs." IEEE TPAMI.
- Blocking strategies: Papadakis, G., et al. (2021). "Blocking and Filtering Techniques for Entity Resolution: A Survey." ACM Computing Surveys.
- BGE-M3 cross-lingual matching: Chen, J., et al. (2024). "BGE M3-Embedding." arXiv:2402.03216.

**Academic contribution**: Demonstrated RAKG's brittleness at higher entity densities and validated hybrid blocking approaches from entity resolution literature as compatible extensions.

---

### 3.2 Phase 1D: Two-Track Relation Extraction (Novel)

**Base methodology**: OpenIE-style triplet extraction per RAGulating Compliance (Chen et al., 2024).

**Novel contribution**: Two-track strategy separating semantic from academic entities.

| Track | Entity Types | Predicate | Objects | Rationale |
|-------|-------------|-----------|---------|-----------|
| **Track 1** | Concepts, Orgs, Tech, Regulations | Any (OpenIE) | Any semantic | Domain knowledge backbone |
| **Track 2** | Citations, Authors, Journals | `discusses` | Concepts only | Literature→concept mapping |

**Key principles (novel)**:

1. **Bidirectionality Principle**: Relations missed during entity A's extraction may be captured when co-occurring entities are processed as subjects. This justifies reducing chunks per entity (10 instead of 20) without losing coverage.

2. **Typed Co-occurrence Matrices**: Pre-compute three matrices (semantic, concept, full) to enable clean track separation without runtime filtering.

3. **No academic-to-academic relations**: Citation networks are handled by Scopus metadata (Phase 2A), not text extraction. Avoids duplicating what structured data provides better.

---

### 3.3 Phase 2A: Scopus Metadata Enrichment

**Design principle**: Phase 1D captures WHAT papers discuss (text-derived). Phase 2A captures WHO wrote them and WHERE published (metadata-derived). Complementary, not redundant.

**Source paper vs. mentioned paper distinction**:
- **Source papers** (158): Papers we parsed with MinerU. Have full Scopus metadata.
- **Mentioned papers** (7,023 entities): Papers cited in source papers' text. May match Scopus references.

**Matching approach**:
1. Find which chunk the citation entity came from (provenance)
2. Identify the source paper for that chunk
3. Match against Scopus metadata for papers cited BY that source paper (References field)

**Matching strategies** (confidence order):
1. DOI exact match (1.0)
2. Author surname + year (0.95)
3. Title fuzzy via embedding (0.80)
4. References field cross-reference (0.75)

**Jurisdiction entity linking** (optional):

Country/region entities like "European Union" need linking to Jurisdiction nodes:
```
Entity("European Union") --[SAME_AS]--> Jurisdiction(EU)
```

| Entity Type | How It's Linked |
|-------------|-----------------|
| Country/region entities | SAME_AS → Jurisdiction (direct name match) |
| Regulatory entities (GDPR, EU AI Act) | Via provenance: EXTRACTED_FROM → Chunk ← CONTAINS Jurisdiction |
| Organizations (CNIL, FTC) | Not linked (would require external knowledge) |

Implementation is simple post-disambiguation lookup: 41 matches expected.

---

## 4. Graph Schema

```cypher
// ============================================
// TOP-LEVEL DOCUMENT NODES
// ============================================

(:Jurisdiction {
  code: "ES",
  name: "Spain",
  url: "https://...",
  scraped_date: "2025-11-06",
  source_file: "ES.json"
})

(:Publication {
  scopus_id: "85123456",
  title: "Algorithmic Fairness...",
  authors: ["Smith, J."],
  year: 2023,
  journal: "Nature",
  doi: "10.1234/...",
  source_file: "paper_123"
})

// ============================================
// CONTENT LAYER
// ============================================

(:Chunk {
  chunk_id: "chunk_ES_001",
  text: "Article 50...",
  // NO embedding - stored externally
  metadata: {
    doc_type: "regulation" | "academic_paper",
    jurisdiction: "ES",         // Only for regulations
    eid: "2-s2.0-85123456",     // Only for papers (Scopus EID)
    section_title: "User transparency"
  }
})

(:Entity {
  entity_id: "ent_a3f4e9c2d5b1",
  name: "EU AI Act",
  type: "Regulation",
  description: "...",
  chunk_ids: ["chunk_ES_001", ...],
  frequency: 127
  // NO embedding - stored externally
})

// ============================================
// METADATA NODES (Phase 2A)
// ============================================

(:Author {
  author_id: "author_12345",
  name: "Luciano Floridi",
  scopus_id: "12345678"
})

(:Journal {
  journal_id: "journal_abc",
  name: "Nature",
  issn: "1234-5678"
})

// ============================================
// RELATIONSHIPS
// ============================================

// Document containment
(:Jurisdiction)-[:CONTAINS]->(:Chunk)
(:Publication)-[:CONTAINS]->(:Chunk)

// Provenance
(:Entity)-[:EXTRACTED_FROM]->(:Chunk)

// Knowledge graph
(:Entity)-[:RELATION {predicate: "regulates", chunk_ids: [...]}]->(:Entity)

// Academic metadata
(:Publication)-[:AUTHORED_BY]->(:Author)
(:Publication)-[:PUBLISHED_IN]->(:Journal)
(:Entity {type: "Academic Citation"})-[:MATCHED_TO]->(:Publication)

// Jurisdiction linking (country entities only)
(:Entity {type: "Country"})-[:SAME_AS]->(:Jurisdiction)
```

---

## 5. Storage Architecture

```
┌─────────────────────────────────────┐
│  External Vector Store (TBD)        │  ← Semantic similarity search
│  • Entity embeddings (76K × 1024)   │
│  • Chunk embeddings (25K × 1024)    │
│  Size: ~1.5GB                       │
│  Options: FAISS file, Pinecone,     │
│           Weaviate, Qdrant          │
└─────────────────────────────────────┘
            ↕ (linked by entity_id / chunk_id)
┌─────────────────────────────────────┐
│  Neo4j Knowledge Graph              │  ← Graph traversal
│  • Jurisdictions: 48 nodes          │
│  • Publications: 158 nodes          │
│  • Chunks: 25K nodes (no embed)     │
│  • Entities: 76K nodes (no embed)   │
│  • Authors: 572 nodes               │
│  • Journals: 119 nodes              │
│  • Relations: 105K edges           │
│  Size: ~100MB                       │
└─────────────────────────────────────┘
```

---

## 6. Processing Timeline

| Stage | Input | Output | Time | Cost |
|-------|-------|--------|------|------|
| Ingestion | raw/ | documents | 3 sec | $0 |
| Chunking | documents | 25K chunks | 10 min | $0 |
| Embedding | chunks | embedded chunks | 30 min | $0 |
| Entity Extraction | chunks | 200K pre-entities | 6-8 hrs | ~$30 |
| Entity Filtering | 200K pre-entities | 143K entities | 2 min | $0 |
| Disambiguation | 143K entities | 76K entities | 6 hrs | ~$6 |
| ID Generation | 76K entities | entities + IDs | 1 min | $0 |
| Co-occurrence | 76K entities | 3 matrices | 30 min | $0 |
| Relation Extraction | entities + matrices | 105K relations | 2-3 days | ~$7 |
| Relation Normalization | relations | Neo4j files | 5 min | $0 |
| Scopus Enrichment | Neo4j + Scopus | enriched graph | 1-2 hrs | $0 |
| **Total** | | | ~4-5 days | **~$43** |

---

## 7. References

### Core Methodologies

1. **RAKG**: Zhou, Y., et al. (2025). "RAKG: Document-level Retrieval-Augmented Knowledge Graph Construction." arXiv:2504.09823.

2. **RAGulating Compliance**: Chen, X., et al. (2024). "Ontology-free Relation Extraction for Regulatory Documents."

### Entity Resolution / Blocking

3. **FAISS HNSW**: Malkov, Y., & Yashunin, D. (2020). "Efficient and Robust Approximate Nearest Neighbor Search Using HNSW Graphs." IEEE TPAMI.

4. **Blocking Survey**: Papadakis, G., et al. (2021). "Blocking and Filtering Techniques for Entity Resolution: A Survey." ACM Computing Surveys.

### Embeddings

5. **BGE-M3**: Chen, J., et al. (2024). "BGE M3-Embedding: Multi-Functionality, Multi-Linguality, and Multi-Granularity Text Embeddings." arXiv:2402.03216.

### Data Sources

6. **DLA Piper**: DLA Piper (2024). "AI Laws of the World." https://www.dlapiper.com/en/insights/artificial-intelligence
