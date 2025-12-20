# -*- coding: utf-8 -*-
"""
LLM prompt templates - Mistral-7B optimized, v2.0 (semantic + metadata).
"""

from config.extraction_config import (
    SEMANTIC_ENTITY_TYPES,
    METADATA_ENTITY_TYPES,
    SEMANTIC_TYPE_NAMES,
    METADATA_TYPE_NAMES,
)

def _build_type_list(type_dict: dict) -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in type_dict.items())

_SEMANTIC_TYPES_LIST = _build_type_list(SEMANTIC_ENTITY_TYPES)
_METADATA_TYPES_LIST = _build_type_list(METADATA_ENTITY_TYPES)
_ALL_TYPE_NAMES = ", ".join(SEMANTIC_TYPE_NAMES + METADATA_TYPE_NAMES)


# ============================================================================
# PHASE 1B: SEMANTIC ENTITY EXTRACTION
# ============================================================================

SEMANTIC_EXTRACTION_PROMPT = f"""# Task
Extract named entities. Use ONLY these types:
{_SEMANTIC_TYPES_LIST}

# Rules
- MUST split compounds: "AI and ML" â†’ extract "AI" AND "ML" as separate entities
- Regulation = law documents (EU AI Act, GDPR)
- RegulatoryConcept = compliance ideas AND principles (governance, transparency, accountability)

# NEVER EXTRACT (metadata pass handles these)
- Document structure (Article X, Section Y, Annex Z, page N)
- Citations, authors, journals, affiliations, DOIs, [1], [2]

# Examples
Input: "The EU AI Act requires transparency and conformity assessment."
Output: {{{{"entities": [{{{{"name": "EU AI Act", "type": "Regulation", "description": "EU AI law"}}}}, {{{{"name": "transparency", "type": "RegulatoryConcept", "description": "Regulatory principle"}}}}, {{{{"name": "conformity assessment", "type": "RegulatoryConcept", "description": "Compliance procedure"}}}}]}}}}

Input: "Cybersecurity risks in AI and ML systems."
Output: {{{{"entities": [{{{{"name": "cybersecurity risks", "type": "Risk", "description": "Security threat"}}}}, {{{{"name": "AI", "type": "Technology", "description": "Artificial intelligence"}}}}, {{{{"name": "ML", "type": "Technology", "description": "Machine learning"}}}}]}}}}

# Text
{{chunk_text}}

# Output
JSON only: {{{{"entities": [{{{{"name": "...", "type": "...", "description": "..."}}}}]}}}}"""


# ============================================================================
# PHASE 1B: METADATA ENTITY EXTRACTION
# ============================================================================

METADATA_EXTRACTION_PROMPT = f"""# Task
Extract metadata entities. Use ONLY these types:
{_METADATA_TYPES_LIST}

# Rules
- Citation: "Author (Year)", [1], [2]
- Author: Researcher names ONLY
- Journal: Publication venues (journals, conferences)
- Affiliation: Universities, research centers, companies
- Document: Named documents when structurally referenced (EU AI Act, GDPR, research papers)
- DocumentSection: Structural parts (Article 5, Section 3, Annex A, page 12)

# NEVER EXTRACT (semantic pass handles these)
- Concepts, technologies, risks, organizations, locations

# Examples
Input: "Article 5 of the EU AI Act prohibits social scoring."
Output: {{{{"entities": [{{{{"name": "Article 5", "type": "DocumentSection", "description": "Section of EU AI Act"}}}}, {{{{"name": "EU AI Act", "type": "Document", "description": "EU AI regulation"}}}}]}}}}

Input: "AuthorName (2020) discusses this in Section 3.2."
Output: {{{{"entities": [{{{{"name": "AuthorName (2020)", "type": "Citation", "description": "AuthorName 2020 work"}}}}, {{{{"name": "Section 3.2", "type": "DocumentSection", "description": "Paper section"}}}}]}}}}

# Text
{{chunk_text}}

# Output
JSON only: {{{{"entities": [{{{{"name": "...", "type": "...", "description": "..."}}}}]}}}}"""


# ============================================================================
# PHASE 1C: ENTITY DISAMBIGUATION
# ============================================================================

SAMEJUDGE_PROMPT = """Are these the SAME real-world entity? Be strict about identifiers.

Entity 1: {entity1_name} ({entity1_type})
  Context: {entity1_desc}

Entity 2: {entity2_name} ({entity2_type})
  Context: {entity2_desc}

SAME (YES): name variations, abbreviations, singular/plural, translations
- "EU AI Act" = "European AI Act"
- "United States" = "USA"
- "technology" = "technologies"

DIFFERENT (NO): any identifier mismatch (numbers, dates, initials, versions)
- "Article 5" ≠ "Article 6"
- "March 2025" ≠ "April 2025"
- "O. V." ≠ "O. P."
- "Level 1" ≠ "Level 2"
- "Privacy Act" ≠ "Data Protection Act"

Answer YES or NO only:"""


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


METADATA_RELATION_EXTRACTION_PROMPT = """Extract what this citation discusses.

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