# -*- coding: utf-8 -*-
"""
LLM prompt templates - Mistral-7B optimized, v2.0 (semantic + metadata).

v2.1 Changes (Phase 1D):
- RELATION_EXTRACTION_PROMPT now uses entity_ids for constrained output
- METADATA_RELATION_EXTRACTION_PROMPT updated for Track 2 (discusses only)
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
- MUST split compounds: "AI and ML" → extract "AI" AND "ML" as separate entities
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

# Few-shot conversation for Mistral (more effective than single prompt)
SAMEJUDGE_SYSTEM = """Classify: are these two entities the EXACT SAME thing with different names?
Answer YES or NO only."""

# Interleaved YES/NO with descriptions for context
SAMEJUDGE_EXAMPLES = [
    # Pattern: abbreviation = full name → YES
    ("GDPR (Regulation) - EU data protection law", "General Data Protection Regulation (Regulation) - European privacy regulation", "YES"),
    # Pattern: specific ≠ generic → NO  
    ("GDPR (Regulation) - EU data protection law", "data protection laws (Regulation) - laws protecting personal data", "NO"),
    # Pattern: country abbreviation → YES
    ("USA (Location) - North American country", "United States (Location) - country in North America", "YES"),
    # Pattern: different numbers → NO
    ("Article 5 (DocumentSection) - prohibited practices", "Article 6 (DocumentSection) - high-risk systems", "NO"),
    # Pattern: tech abbreviation → YES
    ("AI (Technology) - artificial intelligence", "artificial intelligence (Technology) - machine intelligence", "YES"),
    # Pattern: antonyms → NO
    ("safety (Risk) - protection from harm", "risk (Risk) - potential for harm", "NO"),
    # Pattern: regulation abbreviation → YES
    ("EU AI Act (Regulation) - European AI law", "Artificial Intelligence Act (Regulation) - EU regulation on AI", "YES"),
    # Pattern: X ≠ X-issues → NO
    ("safety (Risk) - state of being safe", "safety issues (Risk) - problems with safety", "NO"),
    # Pattern: location abbreviation → YES
    ("EU (Location) - European Union", "European Union (Location) - union of European states", "YES"),
    # Pattern: specific ≠ generic → NO
    ("EU AI Act (Regulation) - European AI law", "AI regulations (Regulation) - laws governing AI generally", "NO"),
    # Pattern: tech abbreviation → YES
    ("ML (Technology) - machine learning", "machine learning (Technology) - subset of AI", "YES"),
    # Pattern: related but different → NO
    ("AI safety (Risk) - ensuring AI is safe", "AI risks (Risk) - dangers from AI", "NO"),
]

# Legacy single-prompt format (kept for reference)
SAMEJUDGE_PROMPT = """Do these refer to the IDENTICAL entity? Default to NO.

Entity 1: {entity1_name} ({entity1_type})
Entity 2: {entity2_name} ({entity2_type})

Answer YES or NO:"""


# ============================================================================
# PHASE 1D: RELATION EXTRACTION (v1.2 - ID-constrained)
# ============================================================================

# Track 1: Semantic entities - OpenIE with multi-chunk context
RELATION_EXTRACTION_PROMPT = """Extract relationships for the target entity.

TARGET: {entity_id}: {entity_name} ({entity_type})
Description: {entity_description}

DETECTED ENTITIES (use these IDs):
{detected_entities_list}

CHUNKS:
{chunks_text}

RULES:
- Subject MUST be target: {entity_id}
- Object MUST be an ID from detected entities
- Predicates: lowercase_underscore (regulates, applies_to, requires, enables)
- No duplicates, only explicit relations
- Each unique relation appears ONCE

JSON only:
{{"relations": [{{"subject_id": "{entity_id}", "predicate": "...", "object_id": "ent_...", "chunk_ids": ["..."]}}]}}"""


# Track 2: Citation entities - chunk-based "discusses" extraction
CITATION_DISCUSSES_PROMPT = """What does this citation discuss in this chunk?

CITATION: {entity_id}: {entity_name}

CONCEPTS IN CHUNK (use these IDs as objects):
{detected_entities_list}

CHUNK:
{chunk_text}

Output only concepts the citation explicitly discusses or supports in this context.

JSON only:
{{"relations": [{{"subject_id": "{entity_id}", "predicate": "discusses", "object_id": "ent_...", "chunk_ids": ["{chunk_id}"]}}]}}"""


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