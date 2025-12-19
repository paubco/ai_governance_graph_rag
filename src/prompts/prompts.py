# -*- coding: utf-8 -*-
"""
LLM prompt templates - Mistral-7B optimized, slim version.
"""

from config.extraction_config import (
    SEMANTIC_ENTITY_TYPES,
    ACADEMIC_ENTITY_TYPES,
    SEMANTIC_TYPE_NAMES,
    ACADEMIC_TYPE_NAMES,
)

def _build_type_list(type_dict: dict) -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in type_dict.items())

_SEMANTIC_TYPES_LIST = _build_type_list(SEMANTIC_ENTITY_TYPES)
_ACADEMIC_TYPES_LIST = _build_type_list(ACADEMIC_ENTITY_TYPES)
_ALL_TYPE_NAMES = ", ".join(SEMANTIC_TYPE_NAMES + ACADEMIC_TYPE_NAMES)


# ============================================================================
# PHASE 1B: SEMANTIC ENTITY EXTRACTION
# ============================================================================

SEMANTIC_EXTRACTION_PROMPT = f"""# Task
Extract named entities. Use ONLY these types:
{_SEMANTIC_TYPES_LIST}

# Rules
- Split compounds: "AI and ML" → extract "AI" AND "ML" separately
- Regulation = law itself (EU AI Act, Article 5)
- RegulatoryConcept = ideas about law (compliance, governance)
- Principle = ONLY ethical values (transparency, fairness) NOT metrics/skills

# NEVER EXTRACT
- Citations, authors, journals, DOIs, page numbers

# Examples
Input: "The EU AI Act requires conformity assessment."
Output: {{{{"entities": [{{{{"name": "EU AI Act", "type": "Regulation", "description": "EU AI law"}}}}, {{{{"name": "conformity assessment", "type": "RegulatoryProcess", "description": "Compliance procedure"}}}}]}}}}

Input: "AI and ML enable automated decision-making."
Output: {{{{"entities": [{{{{"name": "AI", "type": "Technology", "description": "Artificial intelligence"}}}}, {{{{"name": "ML", "type": "Technology", "description": "Machine learning"}}}}]}}}}

# Text
{{chunk_text}}

# Output
JSON only: {{{{"entities": [{{{{"name": "...", "type": "...", "description": "..."}}}}]}}}}"""


# ============================================================================
# PHASE 1B: ACADEMIC ENTITY EXTRACTION
# ============================================================================

ACADEMIC_EXTRACTION_PROMPT = f"""# Task
Extract academic references. Use ONLY these types:
{_ACADEMIC_TYPES_LIST}

# Rules
- Citation: "Author (Year)", [1], [2]
- Author: Full researcher names only
- Journal: Publication venues

# NEVER EXTRACT
- Concepts, regulations, technologies, organizations, locations

# Examples
Input: "Floridi (2018) published in Nature examines AI ethics."
Output: {{{{"entities": [{{{{"name": "Floridi (2018)", "type": "Citation", "description": "Floridi 2018 work"}}}}, {{{{"name": "Nature", "type": "Journal", "description": "Scientific venue"}}}}]}}}}

Input: "ChatGPT demonstrates remarkable capabilities."
Output: {{{{"entities": []}}}}

# Text
{{chunk_text}}

# Output
JSON only: {{{{"entities": [{{{{"name": "...", "type": "...", "description": "..."}}}}]}}}}"""


# ============================================================================
# PHASE 1C: ENTITY DISAMBIGUATION
# ============================================================================

SAMEJUDGE_PROMPT = """Are these the SAME real-world entity?

Entity 1: {entity1_name} ({entity1_type}) - {entity1_desc}
Entity 2: {entity2_name} ({entity2_type}) - {entity2_desc}

JSON only:
{{"result": true/false, "canonical_name": "...", "canonical_type": "...", "reasoning": "..."}}"""


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
{{"relations": [{{"subject": "...", "predicate": "...", "object": "...", "chunk_ids": ["..."]}}]}}"""


ACADEMIC_RELATION_EXTRACTION_PROMPT = """Extract what this citation discusses.

TARGET: {entity_name} ({entity_type})

DETECTED CONCEPTS:
{detected_entities_list}

CHUNKS:
{chunks_text}

JSON only:
{{"relations": [{{"subject": "{entity_name}", "predicate": "discusses", "object": "...", "chunk_ids": ["..."]}}]}}"""


# ============================================================================
# PHASE 3: QUERY PARSING
# ============================================================================

QUERY_ENTITY_EXTRACTION_PROMPT = f"""Extract entities from query. Types: {_ALL_TYPE_NAMES}

Query: {{query}}

JSON only: {{{{"entities": [{{{{"name": "...", "type": "..."}}}}]}}}}"""


# ============================================================================
# PHASE 3: ANSWER GENERATION
# ============================================================================

ANSWER_GENERATION_SYSTEM_PROMPT = """AI governance expert. Cite sources [1][2], acknowledge uncertainty."""

ANSWER_GENERATION_USER_PROMPT = """QUESTION: {query}

GRAPH: {graph_structure}
ENTITIES: {entity_context}
SOURCES: {sources}

Answer with citations. Note jurisdictional differences."""