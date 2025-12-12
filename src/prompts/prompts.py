# -*- coding: utf-8 -*-
"""
LLM prompt templates for entity extraction, disambiguation, and relation extraction.

Centralized prompt templates for Phases 1B (entity extraction), 1C (disambiguation),
1D (relation extraction), and 3 (query parsing) with standardized formats. Contains:
- ENTITY_EXTRACTION_PROMPT: Phase 1B free-type entity discovery from chunks
- SAMEJUDGE_PROMPT: Phase 1C entity verification
- RELATION_EXTRACTION_PROMPT: Phase 1D OpenIE-style triplet extraction
- ACADEMIC_ENTITY_EXTRACTION_PROMPT: Phase 1D subject-constrained extraction
- QUERY_ENTITY_EXTRACTION_PROMPT: Phase 3 query entity extraction with type enforcement
"""

# ============================================================================
# PHASE 1B: ENTITY EXTRACTION
# ============================================================================

# Prompt for extracting entities from text chunks using free-type methodology.
# Used by RAKGEntityExtractor to discover entities with LLM-determined types.
# Instructs model to extract all significant entities including academic citations.
# Output: JSON list of entities with name, type, and description fields.
QUERY_ENTITY_EXTRACTION_PROMPT = """Extract entities explicitly mentioned in the user's query.

Query: {query}

RULES:
- Extract ONLY entities that appear verbatim or paraphrased in the query text
- DO NOT infer related concepts, background knowledge, or implicit connections
- DO NOT expand acronyms unless query contains both forms
- Assign types from this list: {entity_types}

Output JSON only (no markdown, no explanation):
{{
  "entities": [
    {{"name": "entity as written in query", "type": "type from list"}}
  ]
}}

JSON:"""

# ============================================================================
# PHASE 1C: ENTITY DISAMBIGUATION
# ============================================================================

# Prompt for determining if two entities refer to the same real-world entity.
# Used by SameJudge to perform pairwise entity comparison during disambiguation.
# Returns boolean result plus canonical name/type if entities match.
# Critical for clustering duplicate entities discovered across different chunks.
SAMEJUDGE_PROMPT = """Are these two entities the SAME real-world entity?

Entity 1:
- Name: {entity1_name}
- Type: {entity1_type}
- Description: {entity1_desc}

Entity 2:
- Name: {entity2_name}
- Type: {entity2_type}
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

# Prompt for extracting semantic relations using OpenIE methodology (Track 1).
# Used for semantic entities to discover relationships between co-occurring entities.
# Allows free-form predicates (not constrained to predefined relation types).
# Target entity can appear as either subject or object of relations.
# Output: Relations with subject, predicate, object, and chunk_id provenance.
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

# Prompt for extracting what concepts academic entities discuss (Track 2).
# Used for academic entities (citations, authors, journals) with fixed subject.
# Subject is always the academic entity; predicate is always "discusses".
# Object must be a concept from the co-occurring semantic entities.
# Enables mapping academic literature to the concepts they address.
ACADEMIC_ENTITY_EXTRACTION_PROMPT = """You are a knowledge graph construction expert specializing in academic literature mapping.

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

# Prompt for extracting entities from user queries with type enforcement.
# Used by QueryParser (Phase 3) to identify entities in natural language questions.
# Unlike Phase 1B which uses free-form types, this enforces predefined entity types
# from the knowledge graph schema for consistent matching and resolution.
# Output: JSON array of entities with name and type (type must be from allowed list).
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
Query: "What is the EU AI Act?" → [{{"name": "EU AI Act", "type": "Regulation"}}]
Query: "Compare GDPR and CCPA" → [{{"name": "GDPR", "type": "Regulation"}}, {{"name": "CCPA", "type": "Regulation"}}]

JSON:"""

# ============================================================================
# PHASE 3.3.4: ANSWER GENERATION
# ============================================================================

# System prompt for Claude to generate answers from retrieval results.
# Establishes role as AI governance expert with emphasis on citation and precision.
# Output: Structured answer with citations, graph insights, and jurisdictional comparisons.
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

# User prompt template for formatting query with retrieval context.
# Combines query, graph structure, entity context, and source documents.
# Instructs model to use citations and highlight graph relationships.
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