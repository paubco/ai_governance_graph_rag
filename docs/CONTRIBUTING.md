# Contributing / Code Standards

**Project**: AI Governance GraphRAG Pipeline  
**Version**: 1.1  
**Last Updated**: December 2025

---

## 1. Project Structure

```
src/
+-- ingestion/              # Data loading
|   +-- document_loader.py
|   +-- scopus_csv_loader.py
|   +-- dlapiper_scraper.py
|   +-- mineru_metadata_matcher.py
|
+-- processing/
|   +-- chunking/
|   |   +-- semantic_chunker.py
|   |   +-- chunk_processor.py
|   +-- entities/
|   |   +-- entity_extractor.py
|   |   +-- entity_processor.py
|   |   +-- entity_disambiguator.py
|   |   +-- disambiguation_processor.py
|   |   +-- filter_pre_entities.py
|   +-- relations/
|       +-- relation_extractor.py
|       +-- relation_processor.py
|       +-- run_relation_extraction.py
|       +-- build_entity_cooccurrence.py
|       +-- normalize_relations.py
|
+-- enrichment/
|   +-- scopus_enricher.py
|   +-- author_journal_builder.py
|   +-- citation_matcher.py
|
+-- retrieval/              # Phase 3
|   +-- pipeline.py
|   +-- query_parser.py
|   +-- entity_resolver.py
|   +-- graph_expander.py
|   +-- ranker.py
|   +-- generator.py
|
+-- utils/                  # Foundation modules (v1.1)
    +-- __init__.py         # Package exports
    +-- dataclasses.py      # Core data structures
    +-- id_generator.py     # Deterministic ID generation
    +-- aliases.py          # Surface form -> canonical ID
    +-- io.py               # Save/load with embedding split
    +-- constants.py        # Entity types, jurisdictions
    +-- config.py           # Paths, env vars
    +-- embedder.py         # BGE-M3 wrapper
    +-- checkpoint_manager.py
    +-- rate_limiter.py
    +-- logger.py
```

---

## 2. Foundation Modules (v1.1)

### 2.1 Required Imports

All modules should import from centralized foundations:

```python
# Dataclasses - ALWAYS use these, never define locally
from src.utils.dataclasses import (
    Entity, PreEntity, EmbeddedEntity,
    Chunk, EmbeddedChunk,
    Relation,
)

# ID generation - NEVER hand-roll IDs
from src.utils.id_generator import (
    generate_entity_id,      # SHA-256 with name|type
    generate_chunk_id,       # doc_id + position
    generate_relation_id,    # subject|predicate|object
)

# I/O - simple helpers
from src.utils.io import (
    load_json,               # JSON files
    save_json,
    load_jsonl,              # JSONL streaming
    save_jsonl,
)

# Aliases - for entity resolution
from src.utils.aliases import AliasLookup

# Constants
from src.utils.constants import (
    ENTITY_TYPES_V1,
    JURISDICTION_CODES,
    JURISDICTION_ALIASES,
)
```

### 2.2 Dataclass Usage

```python
# CORRECT - use dataclasses
from src.utils.dataclasses import Entity

entity = Entity(
    entity_id=generate_entity_id(name, type),
    name="EU AI Act",
    type="Regulation",
    description="European AI legislation",
    chunk_ids=["reg_EU_CHUNK_0001"],
    aliases=["AI Act", "European AI Act"],
    merge_count=3,
)

# WRONG - raw dicts
entity = {
    "name": "EU AI Act",
    "type": "Regulation",
    ...
}
```

### 2.3 File I/O Pattern

```python
from src.utils.io import load_json, save_json, load_jsonl, save_jsonl

# JSON for entity/chunk lists
entities = load_json("data/processed/entities.json")
save_json(entities, "data/processed/entities_v2.json")

# JSONL for streaming (pre-entities, relations)
relations = load_jsonl("data/processed/relations.jsonl")
save_jsonl(relations, "data/processed/relations_v2.jsonl")

# v1.0 compatibility - nested pre-entity format
from src.utils.io import load_pre_entities_v1
pre_entities = load_pre_entities_v1("data/interim/pre_entities_v1.json")
```

---

## 3. File Header Template

```python
# -*- coding: utf-8 -*-
"""
Module: <filename>.py
Package: src.<module>.<submodule>
Purpose: <One-line description>

Author: Pau Barba i Colomer
Created: <YYYY-MM-DD>
Modified: <YYYY-MM-DD>

References:
    - <Paper if applicable>
    - See docs/ARCHITECTURE.md for context
"""
```

---

## 4. Import Structure

```python
# -*- coding: utf-8 -*-
"""..."""

# Standard library
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

# Third-party
import numpy as np
from tqdm import tqdm

# Local - foundation modules first
from src.utils.dataclasses import Entity, Chunk, Relation
from src.utils.id_generator import generate_entity_id
from src.utils.io import load_json, save_json
from src.utils.logger import get_logger

# Local - other modules
from src.processing.entities.entity_extractor import extract_entities

logger = get_logger(__name__)
```

**Note**: No more `sys.path` manipulation needed if running from project root.

---

## 5. Docstring Format (Google Style)

```python
def extract_entities(chunk_text: str, chunk_id: str) -> List[PreEntity]:
    """
    Extract named entities from a text chunk.
    
    Args:
        chunk_text: Text content to process.
        chunk_id: Unique identifier for provenance.
        
    Returns:
        List of PreEntity dataclass instances.
        
    Raises:
        json.JSONDecodeError: If LLM output invalid.
    """
```

---

## 6. Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Files | snake_case | `entity_extractor.py` |
| Classes | PascalCase | `RAKGEntityExtractor` |
| Functions | snake_case | `extract_entities()` |
| Constants | UPPER_SNAKE | `MAX_CHUNK_SIZE` |
| Private | _prefix | `_parse_json()` |
| Dataclasses | PascalCase | `Entity`, `PreEntity` |

---

## 7. File Operations

Always use `src.utils.io` for pipeline artifacts:

```python
# Pipeline artifacts - use io module
from src.utils.io import load_json, save_json, load_jsonl, save_jsonl

# JSON for lists (entities, chunks)
entities = load_json("data/processed/entities.json")
save_json(entities, "data/processed/entities_v2.json")

# JSONL for streaming (pre-entities, relations)  
relations = load_jsonl("data/processed/relations.jsonl")
save_jsonl(relations, "data/processed/relations_v2.jsonl")
```

For raw file access, always specify encoding:

```python
# Reading
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Scopus CSV (has BOM)
with open(path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
```

---

## 8. Data Directories

| Directory | Purpose | Format |
|-----------|---------|--------|
| `data/raw/` | Original inputs (read-only) | Various |
| `data/interim/` | Checkpoints, partial outputs | JSON/JSONL |
| `data/processed/` | Final outputs | JSON + embedded variants |
| `data/processed/faiss/` | Vector indices | .faiss + ID mappings |

---

## 9. ID Generation Rules

| Artifact | Generator | Format |
|----------|-----------|--------|
| Entity | `generate_entity_id(name, type)` | `ent_<12-hex>` |
| Chunk | `generate_chunk_id(doc_id, idx)` | `<doc_id>_CHUNK_<4-digit>` |
| Relation | `generate_relation_id(s, p, o)` | `rel_<12-hex>` |
| Publication | `generate_publication_id(scopus_id)` | `pub_l1_<12-hex>` |
| Author | `generate_author_id(scopus_id)` | `author_<scopus_id>` |
| Journal | `generate_journal_id(name)` | `journal_<12-hex>` |

**NEVER** hand-roll IDs. Always use generators for consistency.

---

## 10. Git Commits

```
<type>(<scope>): <description>

Types: feat, fix, refactor, docs, test, chore
Scopes: ingestion, chunking, entities, relations, enrichment, retrieval, utils
```

Examples:
```
feat(entities): add alias storage during disambiguation
fix(retrieval): correct FAISS index loading
refactor(utils): consolidate ID generation
docs(architecture): update dataclass documentation
```

---

## 11. Deprecated Files (v1.1)

These files are deprecated and should not be used:

| File | Replacement |
|------|-------------|
| `src/utils/add_entity_ids.py` | `src/utils/id_generator.py` |
| `src/utils/neo4j_utils.py` | Direct Neo4j driver usage |
| `src/retrieval/config.py` (dataclasses) | `src/utils/dataclasses.py` |

---

## 12. Checklist

When creating/modifying a file:

- [ ] `# -*- coding: utf-8 -*-` at top
- [ ] Module docstring with purpose
- [ ] Imports from foundation modules (dataclasses, id_generator, io)
- [ ] Type hints using dataclasses, not raw dicts
- [ ] `generate_*_id()` for all ID creation
- [ ] `save_*/load_*` from io module for artifacts
- [ ] Logger instead of print
- [ ] Pathlib for paths