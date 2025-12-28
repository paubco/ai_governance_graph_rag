# Changelog

## [1.1] - December 2025

Changes from v1.0 baseline to production system.

### Graph Statistics

| Metric | v1.0 | v1.1 |
|--------|------|------|
| Chunks | 25,131 | **2,718** |
| Pre-entities | ~200K | **62,048** |
| Entities | 55,695 | **38,266** |
| Relations | 105,456 | **339,268** |
| L2 Publications | 557 | **872** |
| Unique predicates | 20,832 | **2,728** |

### Infrastructure

- **Neo4j**: Migrated from Aura Free to Docker Enterprise + GDS plugin (required for Steiner Tree algorithm)
- **Generation**: Added Claude Haiku 3.5 as default (was "configurable")

### Phase 1: Knowledge Graph Construction

**1A Chunking**:
- Added quality filtering (coherence ≥ 0.4)
- Added cross-document deduplication
- Reduced from 25K to 2.7K high-quality chunks

**1B Entity Extraction**:
- Implemented dual-pass extraction (semantic + metadata types)
- 15 entity types (9 semantic + 6 metadata)

**1C Disambiguation**:
- Refined thresholds: 0.98 auto-merge, 0.885-0.98 LLM verify, <0.885 reject
- Added betweenness centrality for merge order prioritization
- 38% reduction (62K → 38K entities)

**1D Relation Extraction**:
- Fixed predicate concentration: 92.3% → **29.9%** in top 4 predicates
- Diversified prompt examples from 4 to 12
- Added co-occurrence validation (100% pass rate)

### Phase 2: Storage & Enrichment

**2A Scopus Enrichment**:
- Fixed Author SAME_AS matching (was 0, now 359)
- Fixed Journal SAME_AS matching (was 0, now 39)
- Added provenance-constrained metadata matching
- L2 deduplication: 80% author overlap, 75% title similarity

**2B Neo4j Import**:
- Fixed multi-jurisdiction CONTAINS handling (26 orphans → 0)
- Fixed entity/relation track loading (semantic + metadata)
- Added FAISS index building (HNSW, M=32, ef=200)

### Phase 3: Retrieval (New)

Implemented complete retrieval pipeline:

- **Query Parser**: Entity mention extraction, jurisdiction/doc-type hints
- **Entity Resolver**: 3-stage (exact → alias → fuzzy at ≥0.75)
- **Retrieval Modes**: Semantic, Graph, Dual
- **Graph Expansion**: Steiner Tree via Neo4j GDS
- **Answer Generation**: Claude Haiku 3.5 (15K context, 2K output)

### Evaluation (New)

Added RAGAS evaluation framework:

| Mode | Faithfulness | Relevancy | Best Count |
|------|--------------|-----------|------------|
| Semantic | 0.77 | 0.86 | 2/8 |
| Graph | 0.58 | 0.88 | 2/8 |
| **Dual** | **0.89** | 0.81 | **5/8** |

### Bug Fixes

- Entity ID mismatch between semantic/metadata tracks
- Missing `discusses` relations in Neo4j import
- Orphan jurisdictions from single-jurisdiction CONTAINS
- Author SAME_AS = 0 (ID resolution via authors.json)
- Journal SAME_AS using wrong ID format
- SEMANTIC mode running unnecessary graph expansion
- Subgraph constructor parameter name (`entity_ids` not `entities`)
- RankedChunk attribute access (`source_path` not `retrieval_method`)

### Analytics Corrections

| Metric | Reported (buggy) | Actual |
|--------|------------------|--------|
| Connectivity (any edge) | 55.7% | **100%** |
| Has RELATION edges | - | **55.7%** |
| Truly isolated | 16,953 | **0** |
| Top 4 predicates | 92.3% | **29.9%** |
| Avg degree (semantic) | 8.9 | **17.7** |
| Clustering coefficient | - | **0.38** |
| Avg path length | - | **2.9 hops** |

---

## [1.0] - December 2025

Initial architecture specification. See ARCHITECTURE.md v1.0 for baseline design.