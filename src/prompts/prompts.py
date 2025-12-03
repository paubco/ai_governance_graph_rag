"""
Extraction Prompts for Phases 1B, 1C, 1D
"""

# ============================================================================
# PHASE 1B: ENTITY EXTRACTION
# ============================================================================

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
Extract relationships where "{entity_name}" is connected to detected entities.

CRITICAL RULES:
1. ONLY extract relations EXPLICITLY STATED in the text
   - The predicate MUST be a verb or verb phrase that appears in the chunk
   - DO NOT infer relations from entities merely appearing together
   
2. Entity constraints:
   - Subject MUST be "{entity_name}" OR one of the detected entities
   - Object MUST be one of the detected entities OR "{entity_name}"
   
3. Quality controls:
   - NO duplicate relations (same subject-predicate-object)
   - NO predefined relation types - use actual verbs from text
   - If no explicit relationships found, return empty list

EXAMPLES:
✓ CORRECT (explicit):
  Text: "GDPR regulates data processing"
  Extract: {{"subject": "GDPR", "predicate": "regulates", "object": "data processing"}}

✗ WRONG (inferred):
  Text: "GDPR and transparency are important"
  DO NOT extract: {{"subject": "GDPR", "predicate": "relates_to", "object": "transparency"}}
  (No verb connecting them!)

OUTPUT FORMAT (JSON only):
{{
  "relations": [
    {{
      "subject": "entity_name",
      "predicate": "verb_from_text",
      "object": "entity_name",
      "chunk_ids": ["chunk_id"]
    }}
  ]
}}

JSON:"""


# ============================================================================
# PHASE 1D: ACADEMIC ENTITY EXTRACTION (Subject-Constrained)
# ============================================================================

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
Extract concepts that "{entity_name}" explicitly discusses in the text.

CRITICAL RULES:
1. ONLY extract if the text explicitly shows "{entity_name}" discussing/analyzing the concept
2. Subject is ALWAYS "{entity_name}"
3. Predicate is ALWAYS "discusses"
4. Object MUST be from the detected concepts list
5. DO NOT infer relations from mere co-occurrence

EXAMPLE:
✓ CORRECT: Text says "Smith et al. (2020) discusses algorithmic bias"
✗ WRONG: Text mentions both "Smith et al." and "bias" but no explicit discussion

OUTPUT FORMAT (JSON only):
{{
  "relations": [
    {{
      "subject": "{entity_name}",
      "predicate": "discusses",
      "object": "concept_name",
      "chunk_ids": ["chunk_id"]
    }}
  ]
}}

JSON:"""