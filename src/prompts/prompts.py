# -*- coding: utf-8 -*-
"""
LLM prompt templates for entity and relation extraction.

v1.2: Domain-fused types, imports from config, no domain field.
"""

from config.extraction_config import SEMANTIC_ENTITY_TYPES, ACADEMIC_ENTITY_TYPES

# Build type strings from config
_SEMANTIC_TYPES_STR = ", ".join(SEMANTIC_ENTITY_TYPES)
_ACADEMIC_TYPES_STR = ", ".join(ACADEMIC_ENTITY_TYPES)

# ============================================================================
# PHASE 1B: SEMANTIC ENTITY EXTRACTION (v1.2)
# ============================================================================

SEMANTIC_EXTRACTION_PROMPT = """Extract entities from the text. Use ONLY these types:

CONCEPT TYPES (ideas, not procedures):
- RegulatoryConcept: Legal/compliance ideas (data governance, privacy, requirements)
- TechnicalConcept: AI/ML ideas (training data, model architecture, algorithms)
- PoliticalConcept: Governance ideas (policy frameworks, institutional design)

PROCESS TYPES (procedures with steps):
- RegulatoryProcess: Compliance procedures (conformity assessment, auditing, certification)
- TechnicalProcess: Technical procedures (data analysis, model training, evaluation)
- PoliticalProcess: Policy procedures (legislative process, public consultation)

OTHER TYPES:
- Regulation: Legally binding documents (EU AI Act, GDPR, directives)
- Technology: AI systems/tools/models (ChatGPT, BERT, neural networks)
- Organization: Formal institutions (European Commission, NIST, UNESCO)
- Location: Geographic/jurisdictional (EU, California, China)
- Principle: Normative values (transparency, fairness, accountability)

DO NOT EXTRACT:
- Citations, author names, journal names (handled separately)
- DOIs, page numbers, publication dates, affiliations
- Table/Figure references, self-references ("this study")

DISAMBIGUATION:
- Concept vs Process: Has steps? -> Process. Abstract idea? -> Concept.
- Concept vs Principle: Normative/ethical weight? -> Principle.
- Regulation vs other: Legally binding? -> Regulation.

TEXT:
{chunk_text}

Respond with JSON:
{{"entities": [{{"name": "...", "type": "one of 11 types", "description": "brief"}}]}}"""


# ============================================================================
# PHASE 1B: ACADEMIC ENTITY EXTRACTION (v1.2)
# ============================================================================

ACADEMIC_EXTRACTION_PROMPT = f"""Extract academic entities. Types: {_ACADEMIC_TYPES_STR}

- Citation: "Author (Year)" or "Author et al. (Year)" - extract FULL blobs AND components
- Author: Named researchers (full names only)
- Journal: Publication venues, conferences
- Self-Reference: Multi-word phrases only ("this study", "we propose", "our approach")

REDUNDANT EXTRACTION: Extract both "page 12 of Floridi (2018) in Nature" AND "Floridi (2018)" AND "Nature"

TEXT:
{{chunk_text}}

Respond with JSON:
{{{{"entities": [{{"name": "...", "type": "one of 4 types", "description": "brief"}}]}}}}"""


# ============================================================================
# PHASE 1C: ENTITY DISAMBIGUATION (v1.2 - no domain field)
# ============================================================================

SAMEJUDGE_PROMPT = """Are these the SAME real-world entity?

Entity 1: {entity1_name} ({entity1_type}) - {entity1_desc}
Entity 2: {entity2_name} ({entity2_type}) - {entity2_desc}

JSON only:
{{"result": true/false, "canonical_name": "official name", "reasoning": "brief"}}"""


# ============================================================================
# PHASE 1D: RELATION EXTRACTION
# ============================================================================

RELATION_EXTRACTION_PROMPT = """Extract relationships for target entity.

TARGET: {entity_name} ({entity_type})
DETECTED ENTITIES: {detected_entities_list}

CHUNKS:
{chunks_text}

RULES:
- Subject/Object MUST be from detected entities or target
- Discover predicates from text (regulates, applies_to, requires, etc.)
- No duplicates, only explicit relations

JSON only:
{{"relations": [{{"subject": "...", "predicate": "...", "object": "...", "chunk_ids": ["..."]}}]}}"""


ACADEMIC_RELATION_EXTRACTION_PROMPT = """Extract what this citation discusses.

TARGET: {entity_name} ({entity_type})
DETECTED CONCEPTS: {detected_entities_list}

CHUNKS:
{chunks_text}

RULES:
- Subject is ALWAYS the target citation
- Predicate is ALWAYS "discusses"
- Object MUST be from detected concepts

JSON only:
{{"relations": [{{"subject": "{entity_name}", "predicate": "discusses", "object": "concept", "chunk_ids": ["..."]}}]}}"""


# ============================================================================
# PHASE 3: QUERY PARSING
# ============================================================================

QUERY_ENTITY_EXTRACTION_PROMPT = f"""Extract ONLY entities explicitly in the query.

Query: {{query}}

RULES:
- Extract ONLY literal phrases from query
- NO inferred/related concepts
- Types: {_SEMANTIC_TYPES_STR}, {_ACADEMIC_TYPES_STR}

JSON only:
{{{{"entities": [{{"name": "exact phrase", "type": "type"}}]}}}}"""


# ============================================================================
# PHASE 3: ANSWER GENERATION
# ============================================================================

ANSWER_GENERATION_SYSTEM_PROMPT = """AI governance expert. Ground answers in sources, cite everything [1][2], acknowledge uncertainty."""

ANSWER_GENERATION_USER_PROMPT = """QUESTION: {query}

GRAPH: {graph_structure}
ENTITIES: {entity_context}
SOURCES: {sources}

Answer with citations. Note cross-jurisdictional differences. State if info missing."""