# Contributing / Code Standards

**Project**: AI Governance GraphRAG Pipeline  
**Last Updated**: December 4, 2025

---

## 1. Project Structure

```
src/
├── ingestion/              # Data loading
│   ├── document_loader.py
│   ├── scopus_csv_loader.py
│   ├── dlapiper_scraper.py
│   └── mineru_metadata_matcher.py
│
├── processing/
│   ├── chunking/
│   │   ├── semantic_chunker.py
│   │   └── chunk_processor.py
│   ├── entities/
│   │   ├── entity_extractor.py
│   │   ├── entity_processor.py
│   │   ├── entity_disambiguator.py
│   │   ├── disambiguation_processor.py
│   │   ├── filter_pre_entities.py
│   │   └── add_entity_ids.py
│   └── relations/
│       ├── relation_extractor.py
│       ├── relation_processor.py
│       ├── run_relation_extraction.py
│       ├── build_entity_cooccurrence.py
│       └── normalize_relations.py
│
├── enrichment/
│   ├── scopus_enricher.py
│   ├── author_journal_builder.py
│   └── citation_matcher.py
│
└── utils/
    ├── embedder.py
    ├── checkpoint_manager.py
    ├── rate_limiter.py
    └── logger.py
```

---

## 2. File Header Template (Google Style, Tiered)

Scale complexity with the file. Use the simplest format that covers everything.

### Tier 1: Simple (utility files, single-purpose modules)

```python
# -*- coding: utf-8 -*-
"""
One-line summary.

Extended prose if needed. No headers, just flowing text.
"""
```

### Tier 2: Standard (most files)

```python
# -*- coding: utf-8 -*-
"""
One-line summary.

Extended description as flowing prose - methodology, key details, outputs.

Example:
    result = function_name("input")
    # Returns: expected_output
"""
```

### Tier 3: Complex (pipelines, multi-stage processors)

Use **short single-word headers** only when content is truly categorical:

```python
# -*- coding: utf-8 -*-
"""
One-line summary.

Extended prose description of what this does and why.

Workflow:
    1. Input: description
    2. Process: description  
    3. Output: description

Stages:
    Stage 1: What it does
    Stage 2: What it does

Config:
    --flag-name: What it controls
    --other-flag: What it controls

Example:
    python src/module/script.py --flag value
"""
```

### Header Rules

- **Single word + colon** (`Workflow:`, `Stages:`, `Config:`, `Features:`)
- **Only when categorical** - if it can be prose, make it prose
- **Terse content** - no full sentences under headers, just fragments
- **No metadata headers** - no `Author:`, `Created:`, `Modified:`, `References:`

### Examples

**Simple utility:**
```python
"""
Semantic chunker for AI governance GraphRAG pipeline.

Implements RAKG-style chunking with hierarchical boundaries: headers are hard
boundaries (respect document structure), within sections use sentence similarity
for semantic coherence, and sentences are never split (atomic units).
"""
```

**Standard with example:**
```python
"""
Hash-based entity ID generator for normalized entities.

Assigns unique, reproducible entity IDs using SHA-256 hashing. Uses first 12 
characters (48 bits, ~281 trillion combinations) with collision detection.
Outputs normalized_entities_with_ids.json and entity_name_to_id.json lookup.

Example:
    entity_id = generate_entity_id("GDPR", "Regulation")
    # Returns: "ent_a3f4e9c2d5b1"
"""
```

**Complex pipeline:**
```python
"""
GPU-optimized entity disambiguation with 4-stage pipeline.

Production pipeline for Phase 1C using GPU-accelerated embedding, FAISS HNSW 
blocking, and multithreaded LLM verification.

Workflow:
    1. pre_entities.json (~143k raw)
    2. pre_entities_clean.json (~21k filtered)
    3. normalized_entities.json (~18-20k disambiguated)

Stages:
    Stage 1:   Exact deduplication (name normalization)
    Stage 1.5: BGE-M3 embedding (1024-dim, GPU)
    Stage 2:   FAISS HNSW blocking (GPU, parallel)
    Stage 3:   Tiered threshold filtering (auto-merge high confidence)
    Stage 4:   SameJudge LLM verification (multithreaded)

Config:
    --faiss-workers: Parallel FAISS threads (4-8 recommended)
    --samejudge-workers: Parallel LLM threads (8-12 recommended)
    --start-from-stage: 1 (all) or 2 (skip dedup+embed)
    --stop-at-stage: 1-4 for partial runs
"""
```

**Filter with features:**
```python
"""
Pre-entity quality filter with academic type normalization.

Conservative filtering to remove metadata entities and low-quality extractions
before Phase 1C disambiguation.

Features:
    Academic type normalization (121 → 15 canonical types)
    Metadata entity removal (identifiers, dates, structural)
    Character cleaning and length validation
    Conservative single-mention filtering

Example:
    python src/processing/entities/filter_pre_entities.py \\
        --input data/interim/entities/pre_entities.json \\
        --output data/interim/entities/pre_entities_clean.json
"""
```

---

## 3. Import Structure

```python
# -*- coding: utf-8 -*-
"""..."""

# Standard library
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Project root (adjust .parent count by depth)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
import numpy as np
from tqdm import tqdm

# Local (always absolute from src/)
from src.utils.logger import get_logger
from src.utils.embedder import BGEEmbedder

logger = get_logger(__name__)
```

---

## 4. Docstring Format (Google Style)

```python
def extract_entities(chunk_text: str, chunk_id: str) -> List[Dict]:
    """
    Extract named entities from a text chunk.
    
    Args:
        chunk_text: Text content to process.
        chunk_id: Unique identifier for provenance.
        
    Returns:
        List of entity dicts with name, type, description.
        
    Raises:
        json.JSONDecodeError: If LLM output invalid.
    """
```

---

## 5. Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Files | snake_case | `entity_extractor.py` |
| Classes | PascalCase | `RAKGEntityExtractor` |
| Functions | snake_case | `extract_entities()` |
| Constants | UPPER_SNAKE | `MAX_CHUNK_SIZE` |
| Private | _prefix | `_parse_json()` |

---

## 6. File Operations

Always specify encoding:

```python
# Reading
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Writing
with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# Scopus CSV (has BOM)
with open(path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
```

---

## 7. Data Directories

| Directory | Purpose |
|-----------|---------|
| `data/raw/` | Original inputs (read-only) |
| `data/interim/` | Checkpoints, partial outputs |
| `data/processed/` | Final outputs |

---

## 8. Git Commits

```
<type>(<scope>): <description>

Types: feat, fix, refactor, docs, test, chore
Scopes: ingestion, chunking, entities, relations, enrichment, utils
```

---

## 9. Checklist

When creating/modifying a file:

- [ ] `# -*- coding: utf-8 -*-` at top
- [ ] Module docstring with purpose
- [ ] Imports in correct order
- [ ] Type hints on functions
- [ ] `encoding='utf-8'` on file ops
- [ ] Pathlib for paths
- [ ] Logger instead of print
