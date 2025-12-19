# -*- coding: utf-8 -*-
"""
LLM prompt templates - all types/descriptions from config.
"""

from config.extraction_config import (
    SEMANTIC_ENTITY_TYPES,
    ACADEMIC_ENTITY_TYPES,
    SEMANTIC_TYPE_NAMES,
    ACADEMIC_TYPE_NAMES,
)

# Build type strings from config
def _build_type_list(type_dict: dict) -> str:
    """Build formatted type list from config dict."""
    return "\n".join(f"- {name}: {desc}" for name, desc in type_dict.items())

_SEMANTIC_TYPES_LIST = _build_type_list(SEMANTIC_ENTITY_TYPES)
_ACADEMIC_TYPES_LIST = _build_type_list(ACADEMIC_ENTITY_TYPES)
_ALL_TYPE_NAMES = ", ".join(SEMANTIC_TYPE_NAMES + ACADEMIC_TYPE_NAMES)


# ============================================================================
# PHASE 1B: SEMANTIC ENTITY EXTRACTION
# ============================================================================

SEMANTIC_EXTRACTION_PROMPT = f"""Extract entities from the text. Use ONLY these types:

{_SEMANTIC_TYPES_LIST}

DO NOT EXTRACT (handled separately):
- Citations, author names, journal names
- DOIs, page numbers, affiliations
- Table/Figure references

DISAMBIGUATION:
- Concept vs Process: Has steps? -> Process
- Concept vs Principle: Normative/ethical? -> Principle
- Regulation: Must be legally binding

TEXT:
{{chunk_text}}

JSON only:
{{{{"entities": [{{"name": "...", "type": "...", "description": "brief"}}]}}}}"""


# ============================================================================
# PHASE 1B: ACADEMIC ENTITY EXTRACTION
# ============================================================================

ACADEMIC_EXTRACTION_PROMPT = f"""Extract academic entities. Types:

{_ACADEMIC_TYPES_LIST}

REDUNDANT EXTRACTION: Extract both "Floridi (2018) in Nature" AND "Floridi (2018)" AND "Nature"

TEXT:
{{chunk_text}}

JSON only:
{{{{"entities": [{{"name": "...", "type": "...", "description": "brief"}}]}}}}"""


# ============================================================================
# PHASE 1C: ENTITY DISAMBIGUATION
# ============================================================================

SAMEJUDGE_PROMPT = """Are these two entities the SAME real-world entity?

Entity 1:
- Name: {entity1_name}
- Type: {entity1_type}
- Description: {entity1_desc}

Entity 2:
- Name: {entity2_name}
- Type: {entity2_type}
- Description: {entity2_desc}

JSON only:
{{
  "result": true or false,
  "canonical_name": "most official name if same",
  "canonical_type": "standardized type if same",
  "reasoning": "brief explanation"
}}"""


# ============================================================================
# PHASE 1D: RELATION EXTRACTION
# ============================================================================

RELATION_EXTRACTION_PROMPT = """Extract relationships for target entity.

TARGET: {entity_name} ({entity_type})
Description: {entity_description}

DETECTED ENTITIES:
{detected_entities_list}

CHUNKS:
{chunks_text}

RULES:
- Subject/Object MUST be from detected entities or target
- Discover predicates from text (regulates, applies_to, requires)
- No duplicates, only explicit relations

JSON only:
{{
  "relations": [
    {{"subject": "...", "predicate": "...", "object": "...", "chunk_ids": ["..."]}}
  ]
}}"""


ACADEMIC_RELATION_EXTRACTION_PROMPT = """Extract what this citation discusses.

TARGET: {entity_name} ({entity_type})
Description: {entity_description}

DETECTED CONCEPTS:
{detected_entities_list}

CHUNKS:
{chunks_text}

RULES:
- Subject is ALWAYS the target citation
- Predicate is ALWAYS "discusses"
- Object MUST be from detected concepts

JSON only:
{{
  "relations": [
    {{"subject": "{entity_name}", "predicate": "discusses", "object": "...", "chunk_ids": ["..."]}}
  ]
}}"""


# ============================================================================
# PHASE 3: QUERY PARSING
# ============================================================================

QUERY_ENTITY_EXTRACTION_PROMPT = f"""Extract ONLY entities explicitly in the query.

Query: {{query}}

Types: {_ALL_TYPE_NAMES}

RULES:
- Extract ONLY literal phrases from query
- NO inferred concepts

JSON only:
{{{{
  "entities": [{{{{"name": "exact phrase", "type": "..."}}}}]
}}}}"""


# ============================================================================
# PHASE 3: ANSWER GENERATION
# ============================================================================

ANSWER_GENERATION_SYSTEM_PROMPT = """AI governance expert. Ground answers in sources, cite [1][2], acknowledge uncertainty."""

ANSWER_GENERATION_USER_PROMPT = """QUESTION: {query}

GRAPH: {graph_structure}
ENTITIES: {entity_context}
SOURCES: {sources}

Answer with citations. Note jurisdictional differences. State if info missing."""