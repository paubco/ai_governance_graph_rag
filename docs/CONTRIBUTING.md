# Contributing / Code Standards

**Project**: AI Governance GraphRAG Pipeline  
**Version**: 1.0  
**Last Updated**: December 2025

---

## 1. Project Structure

```
src/
├── ingestion/                    # Phase 0: Data acquisition
│   ├── document_loader.py        # Main orchestrator
│   ├── dlapiper_scraper.py       # Regulation scraping
│   ├── scopus_csv_loader.py      # Academic metadata loading
│   └── paper_to_scopus_metadata_matcher.py
│
├── processing/                   # Phase 1: Core NLP pipeline
│   ├── chunking/                 # Phase 1A
│   │   ├── semantic_chunker.py   # BGE-M3 embeddings, cosine similarity
│   │   └── chunk_processor.py    # Batch processing orchestrator
│   │
│   ├── entities/                 # Phase 1B-1C
│   │   ├── entity_extractor.py   # LLM-based extraction (Qwen-72B)
│   │   ├── entity_processor.py   # Parallel extraction orchestrator
│   │   ├── filter_pre_entities.py
│   │   ├── entity_disambiguator.py
│   │   ├── disambiguation_processor.py
│   │   ├── alias_processor.py
│   │   └── add_entity_ids.py
│   │
│   └── relations/                # Phase 1D
│       ├── relation_extractor.py # LLM triplet extraction (Mistral-7B)
│       ├── relation_processor.py
│       ├── run_relation_extraction.py
│       ├── build_entity_cooccurrence.py
│       ├── normalize_relations.py
│       └── build_alias_lookup.py
│
├── enrichment/                   # Phase 2A: Scopus enrichment
│   ├── enrichment_processor.py   # 10-step pipeline orchestrator
│   ├── scopus_enricher.py        # Citation matching
│   └── jurisdiction_matcher.py   # Entity-to-jurisdiction linking
│
├── graph/                        # Phase 2B: Storage
│   ├── neo4j_import_processor.py # Batch Neo4j import
│   ├── neo4j_importer.py
│   ├── faiss_builder.py          # HNSW index construction
│   └── graph_analytics.py
│
├── retrieval/                    # Phase 3: Query processing
│   ├── retrieval_processor.py    # Pipeline orchestration
│   ├── query_parser.py           # Entity extraction from queries
│   ├── entity_resolver.py        # FAISS-based entity matching
│   ├── chunk_retriever.py        # Dual-channel retrieval
│   ├── graph_expander.py         # PCST subgraph extraction
│   ├── result_ranker.py          # Multiplicative scoring
│   ├── answer_generator.py       # LLM generation (Claude)
│   └── config.py                 # Dataclasses for retrieval
│
├── prompts/                      # LLM prompt templates
│   └── prompts.py
│
└── utils/                        # Cross-cutting utilities
    ├── embedder.py               # BGE-M3 wrapper
    ├── checkpoint_manager.py
    ├── rate_limiter.py
    ├── logger.py
    ├── id_generator.py
    ├── token_counter.py
    └── neo4j_utils.py

tests/
├── processing/                   # Phase 1 tests
├── retrieval/                    # Phase 3 tests
├── graph/                        # Neo4j + FAISS tests
├── ingestion/                    # Document loading tests
└── utils/                        # Utility tests
```

---

## 2. File Header Template

All Python files should use this header format:

```python
# -*- coding: utf-8 -*-
"""
Brief module title for AI governance GraphRAG pipeline.

Detailed description of functionality in 2-3 sentences. Workflow
or architecture details included where relevant.

Example:
    processor = ClassExample(...)
    result = processor.method()
"""

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

**Note**: Do NOT include:
- `Module: filename.py`
- `Package: src.xxx`
- `Purpose: ...`
- `Author: ...`
- `Created: ...` / `Modified: ...`
- `References: ...`
- `CRITICAL BUG FIX: ...` annotations

---

## 3. Import Structure

Imports should be organized in this order:

```python
# -*- coding: utf-8 -*-
"""Module docstring..."""

# Standard library
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
import numpy as np
from tqdm import tqdm

# Local (always absolute from src/)
from src.utils.logger import get_logger
from src.utils.embedder import BGEEmbedder
```

**Rules**:
- Use absolute imports from `src/` (not relative imports)
- Group imports with blank lines between sections
- Sort alphabetically within each section

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
| Classes | PascalCase with role suffix | `EntityProcessor`, `ChunkRetriever` |
| Functions | snake_case | `extract_entities()` |
| Constants | UPPER_SNAKE | `MAX_CHUNK_SIZE` |
| Private | _prefix | `_parse_json()` |

**Class role suffixes**:
- `*Processor`: Orchestrates batch operations
- `*Retriever`: Fetches data
- `*Builder`: Constructs artifacts
- `*Parser`: Parses input
- `*Pipeline`: End-to-end flow

**Embedding terminology**:
- `embedding` (singular): One vector
- `embeddings` (plural): Module or collection

---

## 6. File Operations

Always specify encoding:

```python
# Reading JSON
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Writing JSON
with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# Scopus CSV (has BOM)
with open(path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)

# JSONL streaming
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        record = json.loads(line)
```

---

## 7. Data Directories

| Directory | Purpose | Access |
|-----------|---------|--------|
| `data/raw/` | Original inputs | Read-only |
| `data/interim/` | Checkpoints, partial outputs | Read/write |
| `data/processed/` | Final outputs | Write once |

**File format conventions**:
- `.json`: Small files, lookups, configs
- `.jsonl`: Large files, streaming data
- `.index`: FAISS indices
- `.csv`: Tabular exports

---

## 8. Git Commits

```
<type>(<scope>): <description>

Types: feat, fix, refactor, docs, test, chore
Scopes: ingestion, chunking, entities, relations, enrichment, retrieval, utils
```

**Examples**:
```
feat(entities): add tiered threshold disambiguation
fix(relations): handle empty co-occurrence matrix
refactor(retrieval): extract scoring to separate module
docs(architecture): update Phase 3 retrieval diagram
test(entities): add alias tracking coverage
chore: update environment.yml dependencies
```

---

## 9. Testing

### Test File Naming

- Unit tests: `test_{module}.py`
- Integration tests: `test_{feature}_integration.py`

### Test Location

Tests mirror the `src/` structure:
```
tests/
├── processing/
│   ├── test_semantic_chunker.py
│   ├── test_entity_extraction.py
│   └── test_relation_extraction.py
├── retrieval/
│   ├── test_answer_generator.py
│   └── test_retrieval_complete.py
└── ...
```

### Running Tests

```bash
# All tests
pytest tests/

# Specific module
pytest tests/processing/test_entity_extraction.py

# With coverage
pytest --cov=src tests/
```

---

## 10. Code Quality Checklist

When creating/modifying a file:

- [ ] `# -*- coding: utf-8 -*-` at top
- [ ] Module docstring with brief description + example
- [ ] Imports in correct order (stdlib, project root, third-party, local)
- [ ] Type hints on all functions
- [ ] `encoding='utf-8'` on all file operations
- [ ] Pathlib for paths (not string concatenation)
- [ ] Logger instead of print statements
- [ ] No hardcoded paths (use `data/interim/...` pattern)
- [ ] Google-style docstrings for public functions

---

## 11. Environment Setup

```bash
# Create environment
conda env create -f environment.yml
conda activate graphrag

# Install dev dependencies
pip install pytest pytest-cov black isort

# Configure API keys
cp .env.example .env
# Edit .env with TOGETHER_API_KEY, ANTHROPIC_API_KEY
```

---

## 12. Common Patterns

### LLM Calls

```python
from together import Together

client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,
    max_tokens=4096
)
result = response.choices[0].message.content
```

### Batch Processing with Progress

```python
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(tqdm(
        executor.map(process_item, items),
        total=len(items),
        desc="Processing"
    ))
```

### Checkpoint Loading

```python
checkpoint_path = Path("data/interim/entities/checkpoint.json")
if checkpoint_path.exists():
    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        processed_ids = set(json.load(f))
else:
    processed_ids = set()
```

---

## 13. Known Issues to Avoid

| Anti-pattern | Why | Instead |
|--------------|-----|---------|
| `Dict[str, Any]` everywhere | No type safety | Use dataclasses or Pydantic models |
| Hardcoded paths | Breaks portability | Use `data/interim/...` pattern |
| Bare `except:` | Hides bugs | Catch specific exceptions |
| `print()` for logging | No levels, no file output | Use `logger.info()` |
| Relative imports | Fragile across moves | Use `from src.module import` |

---

## 14. Future Improvements (v1.1)

See `docs/V1_1_IMPLEMENTATION_PLAN.md` for planned improvements:

- Pydantic models for all data structures
- Centralized config in `src/config/`
- Consolidated LLM client in `src/utils/llm.py`
- Single processor per phase (no tripartite split)
- Mistral-7B throughout (cost savings)
