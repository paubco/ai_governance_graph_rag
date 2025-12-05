# -*- coding: utf-8 -*-
"""
LLM prompt templates for entity extraction, disambiguation, and relation extraction.

Centralized prompt templates for Phases 1B (entity extraction), 1C (disambiguation),
and 1D (relation extraction) with standardized formats. Contains ENTITY_EXTRACTION_PROMPT
for Phase 1B free-type entity discovery, SAMEJUDGE_PROMPT for Phase 1C entity verification,
RELATION_EXTRACTION_PROMPT for Phase 1D OpenIE-style triplet extraction, and
ACADEMIC_ENTITY_EXTRACTION_PROMPT for subject-constrained extraction.
"""

# ============================================================================
# PHASE 1B: ENTITY EXTRACTION
# ============================================================================

# Prompt for extracting entities from text chunks using free-type methodology.
# Used by RAKGEntityExtractor to discover entities with LLM-determined types.
# Instructs model to extract all significant entities including academic citations.
# Output: JSON list of entities with name, type, and description fields.
ENTITY_EXTRACTION_PROMPT = """You are an entity extraction assistant for AI governance and regulatory compliance documents.

Text: {text}

Instructions:
1. Extract ALL significant entities from the text
2. For each entity provide:
   - name: Exact name as written in the text
   - type: The category this entity belongs to (you decide based on context)
   - description: Brief contextual description (1-2 sentences)

3. Focus on entities useful for understanding regulatory compliance and AI governance
4. Include academic citations as entities with author and year (e.g., "Smith et al. (2020)", "Jones and Lee (2019)")
5. Preserve exact formatting and capitalization

Output ONLY valid JSON (no other text, no markdown):
{{
  "entities": [
    {{"name": "...", "type": "...", "description": "..."}},
    ...
  ]
}}

JSON output:"""


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
