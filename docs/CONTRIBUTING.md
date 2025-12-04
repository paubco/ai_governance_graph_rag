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

## 2. File Header Template

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
