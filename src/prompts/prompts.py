# -*- coding: utf-8 -*-
"""
LLM prompt templates - Mistral-7B optimized with examples.

v1.3 changes:
- Remove Self-Reference type
- Add entity splitting guidance
- Tighten Principle to ONLY ethical values
"""

from config.extraction_config import (
    SEMANTIC_ENTITY_TYPES,
    ACADEMIC_ENTITY_TYPES,
    SEMANTIC_TYPE_NAMES,
    ACADEMIC_TYPE_NAMES,
)

# Build type lists from config
def _build_type_list(type_dict: dict) -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in type_dict.items())

_SEMANTIC_TYPES_LIST = _build_type_list(SEMANTIC_ENTITY_TYPES)
_ACADEMIC_TYPES_LIST = _build_type_list(ACADEMIC_ENTITY_TYPES)
_ALL_TYPE_NAMES = ", ".join(SEMANTIC_TYPE_NAMES + ACADEMIC_TYPE_NAMES)


# ============================================================================
# PHASE 1B: SEMANTIC ENTITY EXTRACTION
# ============================================================================

SEMANTIC_EXTRACTION_PROMPT = f"""# Task
Extract named entities from the text below.

# Available Types
ONLY use these exact type names:
{_SEMANTIC_TYPES_LIST}

# Rules
- Use EXACT type names from the list above
- Do NOT invent new types
- Split compound entities: "AI and ML" → extract "AI" AND "ML" separately
- Regulation = the law/act ITSELF (e.g., "EU AI Act", "GDPR", "Article 5")
- RegulatoryConcept = IDEAS about regulation (e.g., "compliance", "data governance")
- Article/Amendment references are Regulation, NOT RegulatoryConcept
- Principle = ONLY ethical/normative values (transparency, fairness, accountability)
- Principle is NOT: languages, metrics, emotions, skills, returns

# DO NOT EXTRACT
- Citations: "Author (Year)", "et al.", "[1]", "[2]"
- Author names, journal names
- DOIs, page numbers, affiliations
- "this study", "we propose" (self-references)

# Examples

Input: "The EU AI Act requires conformity assessment for high-risk systems."
Output: {{{{"entities": [
  {{{{"name": "EU AI Act", "type": "Regulation", "description": "European AI regulation"}}}},
  {{{{"name": "conformity assessment", "type": "RegulatoryProcess", "description": "Compliance verification procedure"}}}},
  {{{{"name": "high-risk systems", "type": "TechnicalConcept", "description": "AI systems requiring strict oversight"}}}}
]}}}}

Input: "AI and ML technologies enable automated decision-making."
Output: {{{{"entities": [
  {{{{"name": "AI", "type": "Technology", "description": "Artificial intelligence"}}}},
  {{{{"name": "ML", "type": "Technology", "description": "Machine learning"}}}},
  {{{{"name": "automated decision-making", "type": "TechnicalProcess", "description": "Algorithmic decisions without human input"}}}}
]}}}}

Input: "Floridi (2018) argues that algorithmic fairness requires accountability mechanisms."
Output: {{{{"entities": [
  {{{{"name": "algorithmic fairness", "type": "TechnicalConcept", "description": "Fair treatment in algorithmic decisions"}}}},
  {{{{"name": "accountability", "type": "Principle", "description": "Normative value of responsibility"}}}}
]}}}}

# Text
{{chunk_text}}

# Output
Respond with JSON only:
{{{{"entities": [{{{{"name": "...", "type": "...", "description": "..."}}}}]}}}}"""


# ============================================================================
# PHASE 1B: ACADEMIC ENTITY EXTRACTION
# ============================================================================

ACADEMIC_EXTRACTION_PROMPT = f"""# Task
Extract academic reference entities from the text below.

# Available Types
ONLY use these exact type names:
{_ACADEMIC_TYPES_LIST}

# Rules
- Citation: "Author (Year)" patterns, bracketed references [1], [2]
- Author: Full researcher names only
- Journal: Publication venue names

# DO NOT EXTRACT
- Concepts, processes, principles
- Regulations, technologies, organizations
- Locations, dates, page numbers
- "this study", "we propose" (skip these entirely)

# Examples

Input: "Floridi (2018) published in Nature examines AI ethics."
Output: {{{{"entities": [
  {{{{"name": "Floridi (2018)", "type": "Citation", "description": "Reference to Floridi's 2018 work"}}}},
  {{{{"name": "Nature", "type": "Journal", "description": "Scientific publication venue"}}}}
]}}}}

Input: "As shown in [1], [2], and Jobin et al. (2019), the framework addresses key issues."
Output: {{{{"entities": [
  {{{{"name": "[1]", "type": "Citation", "description": "Numbered reference"}}}},
  {{{{"name": "[2]", "type": "Citation", "description": "Numbered reference"}}}},
  {{{{"name": "Jobin et al. (2019)", "type": "Citation", "description": "Reference to Jobin survey"}}}}
]}}}}

Input: "ChatGPT demonstrates remarkable capabilities in natural language processing."
Output: {{{{"entities": []}}}}

# Text
{{chunk_text}}

# Output
Respond with JSON only:
{{{{"entities": [{{{{"name": "...", "type": "...", "description": "..."}}}}]}}}}"""


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
{{{{"entities": [{{{{"name": "exact phrase", "type": "..."}}}}]}}}}"""


# ============================================================================
# PHASE 3: ANSWER GENERATION
# ============================================================================

ANSWER_GENERATION_SYSTEM_PROMPT = """AI governance expert. Ground answers in sources, cite [1][2], acknowledge uncertainty."""

ANSWER_GENERATION_USER_PROMPT = """QUESTION: {query}

GRAPH: {graph_structure}
ENTITIES: {entity_context}
SOURCES: {sources}

Answer with citations. Note jurisdictional differences. State if info missing."""