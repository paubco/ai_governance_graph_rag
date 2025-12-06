# AI Governance GraphRAG

> **A knowledge graph that connects AI research with regulations across 48 countries.**

<p align="center">
  <img src="docs/images/graph_preview.png" alt="Knowledge Graph Preview" width="600">
</p>

## The Problem

AI regulations are emerging worldwide, but they're scattered across jurisdictions and disconnected from academic research. A compliance officer asking *"How do different countries define 'high-risk AI systems'?"* would need to manually read dozens of regulatory documents and cross-reference academic literature.

## The Solution

This project builds a **knowledge graph** that:

1. **Extracts concepts** from 158 academic papers and 48 countries' AI regulations
2. **Links related ideas** across sources (e.g., "transparency" in the EU AI Act ↔ "explainability" in research)
3. **Enables cross-domain queries** like:
   - *"What does academic research say about the transparency requirements in EU AI Act?"*
   - *"Which jurisdictions mention algorithmic bias?"*
   - *"How do US and EU approaches to AI risk differ?"*

### Key Finding

**512 "bridge concepts"** appear in both academic papers AND regulatory texts—including `AI System`, `transparency`, `human rights`, and `data protection`. These bridges connect previously siloed knowledge domains.

---

## What's a Knowledge Graph?

A knowledge graph represents information as a network of **entities** (things) and **relationships** (connections between things).

```
┌──────────────┐                      ┌──────────────┐
│   EU AI Act  │───── regulates ─────▶│  AI System   │
│ (Regulation) │                      │ (Technology) │
└──────────────┘                      └──────────────┘
       │                                     │
       │ requires                            │ discussed_in
       ▼                                     ▼
┌──────────────┐                      ┌──────────────┐
│ transparency │◀──── studied_by ─────│ Smith (2024) │
│  (Concept)   │                      │   (Paper)    │
└──────────────┘                      └──────────────┘
```

This structure lets you traverse connections that would be invisible in traditional search.

---

## Results

| What We Built | Count |
|---------------|-------|
| 🌐 Jurisdictions covered | 48 |
| 📄 Academic papers processed | 158 |
| 🔗 Entities extracted | 55,695 |
| 🕸️ Relationships discovered | 105,456 |
| 🌉 Cross-domain bridges | 512 |

### Network Structure

The graph exhibits **scale-free** properties (like the web or social networks):

- **4 super-hubs** with 500+ connections each
- **"AI System"** is the most connected concept (3,496 links)
- 82% of entities have ≤5 connections (long-tail distribution)

### Top Connected Concepts

| Concept | Type | Why It Matters |
|---------|------|----------------|
| AI System | Technology | Central to all regulations |
| transparency | Concept | Key requirement across jurisdictions |
| European Union | Organization | Most comprehensive AI framework |
| AI Act | Legislation | First major AI law |
| human rights | Legal Concept | Foundational principle |

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  📚 Scopus Academic Papers (158)    │    🌍 DLA Piper AI Regulations (48) │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: EXTRACTION                             │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Chunk documents into ~500 token segments                            │
│  2. Extract entities using LLM (Qwen-72B)                               │
│  3. Disambiguate duplicates (FAISS + embeddings)                        │
│  4. Extract relationships using LLM (Mistral-7B)                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         PHASE 2: ENRICHMENT                             │
├─────────────────────────────────────────────────────────────────────────┤
│  5. Link citations to Scopus metadata                                   │
│  6. Match entities to jurisdiction codes                                │
│  7. Build provenance chains (entity → chunk → source)                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         PHASE 3: GRAPH + RETRIEVAL                      │
├─────────────────────────────────────────────────────────────────────────┤
│  8. Import to Neo4j graph database                                      │
│  9. Build FAISS vector indices for semantic search                      │
│  10. Query interface (coming soon)                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Use Cases

### For Researchers
- Find which papers discuss specific regulatory concepts
- Discover connections between research topics and policy frameworks
- Identify gaps in academic coverage of emerging regulations

### For Policy Analysts
- Compare how different jurisdictions define key terms
- Trace the academic foundations of regulatory requirements
- Identify concepts that span multiple legal frameworks

### For Compliance Teams
- Map regulatory requirements to academic best practices
- Find authoritative sources for compliance documentation
- Track how concepts like "algorithmic transparency" are interpreted globally

---

## Technical Details

<details>
<summary><b>Stack</b></summary>

| Component | Technology |
|-----------|------------|
| Graph Database | Neo4j Aura |
| Vector Search | FAISS (HNSW) |
| Embeddings | BGE-M3 (1024-dim) |
| Entity Extraction | Qwen-72B via Together.ai |
| Relation Extraction | Mistral-7B via Together.ai |
| Language | Python 3.10+ |

</details>

<details>
<summary><b>Graph Schema</b></summary>

**Nodes:**
- `Entity` — Extracted concepts (55,695)
- `Chunk` — Text segments with provenance (25,131)  
- `Publication` — Academic papers (158)
- `L2Publication` — Cited works (557)
- `Jurisdiction` — Countries/regions (48)
- `Author` — Paper authors (572)
- `Journal` — Academic journals (119)

**Relationships:**
- `RELATION` — Semantic connections (105,456)
- `EXTRACTED_FROM` — Entity provenance (126,000)
- `CONTAINS` — Document structure (24,549)
- `CITES` — Citation links (579)
- `MATCHED_TO` — Entity-citation alignment (2,388)

</details>

<details>
<summary><b>Data Quality</b></summary>

| Metric | Value |
|--------|-------|
| Entity provenance coverage | 100% |
| Chunk-to-source attribution | 97.7% |
| Orphan nodes | 4 |
| Unique predicates | 20,832 |

</details>

<details>
<summary><b>Running the Pipeline</b></summary>

```bash
# Setup
conda env create -f environment.yml
conda activate graphrag
cp .env.example .env  # Add your API keys

# Run pipeline
python -m src.processing.chunking.chunk_processor
python -m src.processing.entities.entity_processor
python -m src.processing.entities.disambiguation_processor
python -m src.processing.relations.run_relation_extraction
python -m src.enrichment.enrichment_processor
python -m src.graph.neo4j_import_processor
```

</details>

---

## Project Context

**Master's Thesis** — MSc Data Science, Universitat Oberta de Catalunya (UOC)  
**Author**: Pau Calvet Milián  
**Date**: December 2025  
**Advisor**: [TBD]

### Methodology

This project combines techniques from two recent papers:

1. **RAKG** (Zhou et al., 2025) — Entity extraction and disambiguation using LLMs
2. **RAGulating Compliance** (Agarwal et al., 2025) — Ontology-free relation extraction for regulatory texts

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed methodology.

---

## Acknowledgments

- **DLA Piper** for the [AI Laws of the World](https://www.dlapiper.com/en-us/insights/publications/ai-laws-of-the-world) dataset
- **Scopus** for academic paper metadata and full texts
- **Together.ai** for affordable LLM API access

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Built with 🧠 and ☕ in Barcelona</i>
</p>
