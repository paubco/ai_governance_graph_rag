# -*- coding: utf-8 -*-
"""
LLM prompt templates for entity extraction, disambiguation, and relation extraction.

Centralized prompt templates for Phases 1B (entity extraction), 1C (disambiguation),
1D (relation extraction), and 3 (query parsing) with standardized formats.

v1.1 Changes:
- Replaced free-type ENTITY_EXTRACTION_PROMPT with constrained dual-pass prompts
- SEMANTIC_EXTRACTION_PROMPT: 12 types x 4 domains with disambiguation rules
- ACADEMIC_EXTRACTION_PROMPT: 4 types, no domain, citation-focused
"""

# ============================================================================
# PHASE 1B: SEMANTIC ENTITY EXTRACTION (v1.1)
# ============================================================================

SEMANTIC_EXTRACTION_PROMPT = """Extract entities from the following text chunk.

TYPES (choose exactly one):
- Concept: Abstract ideas, principles, terms (NOT processes, NOT normative principles)
- Regulation: Legally binding documents, laws, directives
- Technology: AI systems, models, algorithms, tools
- Organization: Institutions, companies, agencies with formal structure
- Person: Named individuals (NOT paper authors or cited researchers)
- Location: Geographic/jurisdictional entities
- Process: Procedures, methodologies with defined steps
- Document: Reports, standards, non-binding publications (NOT journal articles)
- Group: Collectives, categories without formal structure
- Principle: Normative values, rights, ethical concepts

DOMAINS (choose exactly one):
- Regulatory: Legal requirements, compliance, policy rules
- Political: Governance actors, policy-making bodies
- Technical: AI/ML systems, methods, algorithms
- General: Domain-agnostic or cross-cutting

DO NOT EXTRACT (handled separately):
- Citations: "Author (Year)", "Author et al. (Year)", any parenthetical references
- Journal names: "Nature", "AI & Society", "Frontiers in..."
- Author names from paper metadata or citations
- DOIs, page numbers, publication dates, affiliations
- Self-references: "this study", "we propose", "the authors"
- Table/Figure references: "Table 1", "Figure 2"

TYPE DISAMBIGUATION:
- Concept vs Principle: Has normative/ethical weight? -> Principle. Otherwise -> Concept.
- Concept vs Process: Has defined steps/stages? -> Process. Otherwise -> Concept.
- Regulation vs Document: Is it legally binding? -> Regulation. Otherwise -> Document.
- Technology vs Process: Is it a thing you use, or steps you follow? Thing -> Technology. Steps -> Process.
- Organization vs Group: Has formal structure (leadership, legal entity)? -> Organization. Otherwise -> Group.

DOMAIN DISAMBIGUATION:
- LAW, DIRECTIVE, REQUIREMENT, LEGAL OBLIGATION -> Regulatory
- GOVERNMENT, MINISTRY, COMMISSION, POLITICAL BODY -> Political
- AI SYSTEM, ALGORITHM, MODEL, TECHNICAL METHOD -> Technical
- Could apply to multiple domains equally -> General
- Regulatory = the rule itself. Political = the body that makes/enforces rules.

TEXT:
{chunk_text}

Respond with a JSON object containing an "entities" array:
{{"entities": [
  {{"name": "entity name", "type": "one of 10 types", "domain": "one of 4 domains", "description": "brief description"}}
]}}"""


# ============================================================================
# PHASE 1B: ACADEMIC ENTITY EXTRACTION (v1.1)
# ============================================================================

ACADEMIC_EXTRACTION_PROMPT = """Extract academic entities (citations, authors, journals) from the following text chunk.

TYPES (choose exactly one):
- Citation: In-text references like "Author (Year)" or "Author et al. (Year)"
- Author: Named researchers or writers (full names only)
- Journal: Publication venues, conference proceedings, journal names
- Self-Reference: ONLY multi-word phrases referring to current work

REDUNDANT EXTRACTION STRATEGY:
Extract BOTH complete reference blobs AND their components separately.
Example text: "as shown in page 12 of Floridi (2018) published in Nature"
Extract ALL of:
  - "page 12 of Floridi (2018) published in Nature" (Citation - full blob)
  - "Floridi (2018)" (Citation - core reference)
  - "Nature" (Journal)

This redundancy is intentional - extract overlapping entities.

SELF-REFERENCE RULES:
- Must be at least 2 words
- VALID: "this study", "the authors", "we propose", "our approach", "this paper", "our findings"
- INVALID: single words, numbers, "Citation", generic terms

TEXT:
{chunk_text}

Respond with a JSON object containing an "entities" array:
{{"entities": [
  {{"name": "entity name", "type": "one of 4 types", "description": "brief description"}}
]}}"""


# ============================================================================
# PHASE 1C: ENTITY DISAMBIGUATION
# ============================================================================

SAMEJUDGE_PROMPT = """Are these two entities the SAME real-world entity?

Entity 1:
- Name: {entity1_name}
- Type: {entity1_type}
- Domain: {entity1_domain}
- Description: {entity1_desc}

Entity 2:
- Name: {entity2_name}
- Type: {entity2_type}
- Domain: {entity2_domain}
- Description: {entity2_desc}

Respond ONLY with valid JSON:
{{
  "result": true or false,
  "canonical_name": "most official name if same",
  "canonical_type": "standardized type if same",
  "reasoning": "brief explanation"
}}

JSON:"""


# ============================================================================
# PHASE 1D: RELATION EXTRACTION (OPENIE)
# ============================================================================

RELATION_EXTRACTION_PROMPT = """You are a knowledge graph construction expert specializing in OpenIE (Open Information Extraction).

TARGET ENTITY:
Name: {entity_name}
Type: {entity_type}
Description: {entity_description}

DETECTED ENTITIES IN CONTEXT:
{detected_entities_list}

CONTEXT CHUNKS:
{chunks_text}

TASK:
Extract ALL relationships where "{entity_name}" is connected to the detected entities above.

RULES:
- Subject MUST be "{entity_name}" OR one of the detected entities
- Object MUST be one of the detected entities OR "{entity_name}"
- Use ONLY entities from the detected list above
- NO duplicate relations (same subject-predicate-object)
- NO predefined relation types - discover predicates from text (e.g., "regulates", "applies_to")
- Extract only relations explicitly stated in chunks
- If no valid relationships found, return empty list

OUTPUT FORMAT (JSON only, no other text):
{{
  "relations": [
    {{
      "subject": "entity_name_from_detected_list",
      "predicate": "verb_phrase",
      "object": "entity_name_from_detected_list",
      "chunk_ids": ["chunk_id"]
    }}
  ]
}}

JSON:"""


# ============================================================================
# PHASE 1D: ACADEMIC ENTITY EXTRACTION (Subject-Constrained)
# ============================================================================

ACADEMIC_RELATION_EXTRACTION_PROMPT = """You are a knowledge graph construction expert specializing in academic literature mapping.

TARGET ACADEMIC ENTITY:
Name: {entity_name}
Type: {entity_type}
Description: {entity_description}

DETECTED CONCEPTS IN CONTEXT:
{detected_entities_list}

CONTEXT CHUNKS:
{chunks_text}

TASK:
Extract what concepts "{entity_name}" discusses based on the context.

RULES:
- Subject is ALWAYS "{entity_name}" (the academic entity)
- Predicate is ALWAYS "discusses"
- Object MUST be one of the detected concepts from the list above
- DO NOT extract relations where "{entity_name}" is the object
- Use ONLY concepts from the detected list
- If no concepts found, return empty list

OUTPUT FORMAT (JSON only, no other text):
{{
  "relations": [
    {{
      "subject": "{entity_name}",
      "predicate": "discusses",
      "object": "concept_name_from_detected_list",
      "chunk_ids": ["chunk_id"]
    }}
  ]
}}

JSON:"""


# ============================================================================
# PHASE 3: QUERY ENTITY EXTRACTION
# ============================================================================

QUERY_ENTITY_EXTRACTION_PROMPT = """Extract ONLY entities explicitly mentioned in the query text.

Query: {query}

CRITICAL RULES:
1. Extract ONLY entities that appear literally in the query above
2. DO NOT generate related concepts or infer additional entities
3. DO NOT add context or background knowledge
4. If query says "algorithmic fairness", extract "algorithmic fairness" (not "bias", "data", "blockchain")
5. Use types from this list: {entity_types}

Return JSON (no other text):
{{
  "entities": [
    {{"name": "exact phrase from query", "type": "type from list"}}
  ]
}}

Examples:
Query: "What is the EU AI Act?" -> [{{"name": "EU AI Act", "type": "Regulation"}}]
Query: "Compare GDPR and CCPA" -> [{{"name": "GDPR", "type": "Regulation"}}, {{"name": "CCPA", "type": "Regulation"}}]

JSON:"""


# ============================================================================
# PHASE 3.3.4: ANSWER GENERATION
# ============================================================================

ANSWER_GENERATION_SYSTEM_PROMPT = """You are an AI governance expert assistant. Your role is to answer questions about AI regulations and research using a knowledge graph and source documents.

KEY PRINCIPLES:
1. Ground answers in provided sources - cite everything
2. Use graph structure to show relationships between concepts
3. Be precise about jurisdictions and regulatory distinctions
4. Acknowledge uncertainty when sources are incomplete
5. Compare across jurisdictions when relevant
6. When sources conflict, state both positions with citations

CITATION FORMAT:
- Use [1], [2], [3] to reference source chunks
- Multiple sources: [1, 2]
- Always cite specific claims

STRUCTURE YOUR ANSWER:
1. Direct answer (1-2 sentences)
2. Key details with citations
3. Graph-based insights (how concepts relate)
4. Cross-jurisdictional comparisons (if relevant)
5. Caveats or limitations"""


ANSWER_GENERATION_USER_PROMPT = """# QUESTION
{query}

# KNOWLEDGE GRAPH STRUCTURE
{graph_structure}

# KEY ENTITIES
{entity_context}

# SOURCE DOCUMENTS
{sources}

# INSTRUCTIONS
Answer the question using the graph structure and source documents above.

- Start with a direct answer
- Support all claims with citations [1], [2], etc.
- Highlight relationships from the graph
- Note any cross-jurisdictional differences
- If the answer requires information not in sources, state what's missing
- Be concise but comprehensive

ANSWER:"""