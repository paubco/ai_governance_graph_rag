# AI Governance GraphRAG Pipeline

**Master's Thesis** — Universitat Oberta de Catalunya  
**Author**: Pau Barba i Colomer  
**Date**: December 2025

---

## Overview

A knowledge graph construction pipeline for AI governance research, enabling cross-jurisdictional regulatory analysis through entity extraction, disambiguation, and relation extraction.

For methodology details, see **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

---

## Quick Start

```bash
# Setup
conda env create -f environment-gpu.yml
conda activate graphrag
echo "TOGETHER_API_KEY=your_key_here" > .env

# Run pipeline (in order)
python -m src.processing.chunking.chunk_processor
python -m src.processing.entities.entity_processor
python -m src.processing.entities.disambiguation_processor
python -m src.processing.entities.add_entity_ids
python -m src.processing.relations.build_entity_cooccurrence
python -m src.processing.relations.run_relation_extraction
python -m src.processing.relations.normalize_relations
python -m src.enrichment.enrichment_processor  # Phase 2A
python -m src.graph.neo4j_import_processor     # Phase 2B
python -m src.graph.faiss_builder              # Phase 2B
```

---

## Data Sources

| Source | Count | Origin |
|--------|-------|--------|
| AI Regulations | 48 | DLA Piper (2024) |
| Academic Papers | 158 | Scopus 2023 + MinerU |

---

## Results

| Artifact | Count |
|----------|-------|
| Chunks | 25,131 |
| Entities | 55,695 |
| Relations | ~105,456 |

---

## Project Structure

```
Graph_RAG/
├── src/
│   ├── ingestion/      # Data loading
│   ├── processing/     # Chunking, entities, relations
│   ├── enrichment/     # Scopus metadata, jurisdiction linking
│   ├── graph/          # Neo4j import, FAISS builder
│   └── utils/          # Shared utilities
├── data/
│   ├── raw/            # Original inputs
│   ├── interim/        # Checkpoints
│   └── processed/      # Final outputs (entities/, relations/, reports/)
└── docs/
    ├── ARCHITECTURE.md # Technical specification
    └── CONTRIBUTING.md # Code standards
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Pipeline design, methodology, graph schema |
| [CONTRIBUTING.md](docs/CONTRIBUTING.md) | Code standards, imports, docstrings |

---

## References

1. Zhou, Y., et al. (2025). "RAKG: Document-level Retrieval-Augmented Knowledge Graph Construction."
2. Chen, X., et al. (2024). "RAGulating Compliance: Ontology-free Relation Extraction."
3. DLA Piper (2024). "AI Laws of the World."
