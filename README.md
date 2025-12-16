# AI Governance GraphRAG Pipeline

A knowledge graph construction and retrieval system for cross-jurisdictional AI governance research. Combines 48 regulatory documents from DLA Piper's "AI Laws of the World" with 156 academic papers to enable structured queries across legal and scholarly sources.

**Master's Thesis** — Universitat Oberta de Catalunya (UOC)  
**Author**: Pau Barba i Colomer  
**Tutor**: Janneth Chicaiza Espinosa  
**Date**: December 2025  
**License**: MIT (code) / CC BY-NC 3.0 (thesis)

---

## Overview

This system implements a GraphRAG (Graph Retrieval-Augmented Generation) pipeline that:

1. **Extracts** entities and relations from regulatory and academic texts using LLM-based extraction
2. **Disambiguates** entities via FAISS blocking with tiered similarity thresholds
3. **Enriches** the graph with Scopus bibliometric metadata (authors, journals, citations)
4. **Retrieves** context using dual-channel search (graph traversal + semantic similarity)
5. **Generates** answers grounded in both graph structure and source text

---

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/paubco/ai_governance_graph_rag
cd ai_governance_graph_rag

# 2. Create environment
conda env create -f environment.yml
conda activate graphrag

# 3. Configure API keys
cp .env.example .env
# Edit .env with TOGETHER_API_KEY, ANTHROPIC_API_KEY

# 4. Start Neo4j (Docker)
docker-compose up -d neo4j

# 5. Run a query
python -m src.retrieval.run_query "What is the EU AI Act?" --mode dual
```

---

## Running Queries

The system supports three retrieval modes for ablation testing:

```bash
# Semantic-only (baseline)
python -m src.retrieval.run_query "What are high-risk AI systems?" --mode semantic

# Graph-only (entity traversal)
python -m src.retrieval.run_query "What are high-risk AI systems?" --mode graph

# Dual (both channels - recommended)
python -m src.retrieval.run_query "What are high-risk AI systems?" --mode dual
```

Query cost: ~$0.005 | Latency: ~70 seconds | Output: 300-500 tokens with 5-15 source citations

---

## Data Sources

| Source | Content | Count |
|--------|---------|-------|
| Regulations | DLA Piper jurisdiction summaries | 48 |
| Academic Papers | Scopus PDFs (MinerU-parsed) | 156 |
| **Total Documents** | | **204** |

---

## Pipeline Results

| Artifact | Count |
|----------|-------|
| Chunks | ~25,000 |
| Pre-entities | ~155,000 |
| Canonical Entities | ~56,000 |
| Relations | 150,000+ |
| L2 Publications | 557 |

---

## Project Structure

```
ai_governance_graph_rag/
├── src/
│   ├── ingestion/          # Data loading
│   ├── processing/
│   │   ├── chunking/       # Semantic chunking + BGE-M3 embeddings
│   │   ├── entities/       # Extraction + disambiguation
│   │   └── relations/      # OpenIE triplet extraction
│   ├── enrichment/         # Scopus metadata linking
│   ├── graph/              # Neo4j + FAISS import
│   └── retrieval/          # Query processing + generation
├── data/
│   ├── raw/                # Original inputs
│   ├── interim/            # Processing checkpoints
│   └── processed/          # Final outputs
└── docs/                   # Documentation
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM (Entity Extraction) | Qwen-72B via Together.ai |
| LLM (Disambiguation) | Qwen-7B via Together.ai |
| LLM (Relations) | Mistral-7B via Together.ai |
| LLM (Generation) | Claude Haiku 3.5 via Anthropic |
| Embeddings | BGE-M3 (1024-dim, multilingual) |
| Graph Database | Neo4j + GDS (Steiner Tree) |
| Vector Store | FAISS (HNSW) |

---

## Methodology

The implementation follows RAKG methodology (Zhang et al., 2025) with domain-specific adaptations. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for technical details including two-track extraction, tiered disambiguation, and dual-channel retrieval design.

---

## Evaluation

Three-way ablation across 6 query types (18 test scenarios):

| Mode | Mean Faithfulness | Best For |
|------|-------------------|----------|
| Semantic | 0.63 | Comparison queries |
| Graph | 0.66 | Entity-centric queries |
| Dual | 0.54 | Conceptual synthesis |

Evaluation uses RAGAS metrics (faithfulness, relevancy) with Claude Haiku 3.5 as judge.

---

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Technical specification, methodology, graph schema |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | Code standards, development guide |

---

## References

1. Zhang, H., et al. (2025). "RAKG: Document-level Retrieval Augmented Knowledge Graph Construction."
2. He, X., et al. (2024). "G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding." NeurIPS 2024.
3. Han, H., et al. (2025). "Retrieval-Augmented Generation with Graphs (GraphRAG)." arXiv.

---

## License

- **Code**: MIT License. See [LICENSE](LICENSE).
- **Thesis document**: CC BY-NC 3.0 (per UOC requirements).

---

## Acknowledgments

Developed as part of a Master's thesis in Data Science at Universitat Oberta de Catalunya (UOC), supervised by Janneth Chicaiza Espinosa.
