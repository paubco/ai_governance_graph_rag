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

Your task: Extract relationship triplets from regulatory and academic text.

TARGET ENTITY:
Name: {entity_name}
Type: {entity_type}
Description: {entity_description}

DETECTED ENTITIES IN CONTEXT:
{detected_entities_list}

CONTEXT CHUNKS:
{chunks_text}

EXTRACTION TASK:
Extract ALL relationships where "{entity_name}" is connected to the detected entities listed above.

CRITICAL RULES:
- Subject MUST be "{entity_name}" OR one of the detected entities
- Object MUST be one of the detected entities OR "{entity_name}"
- DO NOT extract relations with entities NOT in the detected list
- DO NOT invent new entities - use ONLY the provided detected entities
- NO duplicate relations (same subject-predicate-object combination)
- If no valid relationships found, return empty list

OPENIE PRINCIPLES:
1. NO predefined relation types - discover relations from text
2. Predicates are verb phrases connecting entities (e.g., "regulates", "applies_to", "established_by")
3. Extract both directions if relevant: (A, predicate1, B) and (B, predicate2, A) are different
4. Only extract relations explicitly stated in the chunks

GOOD EXAMPLES:
Detected entities: ["GDPR", "EU", "data processing"]
Text: "The GDPR regulates data processing in the EU"
→ {{"subject": "GDPR", "predicate": "regulates", "object": "data processing", "chunk_ids": ["..."]}}
→ {{"subject": "GDPR", "predicate": "applies_in", "object": "EU", "chunk_ids": ["..."]}}

Detected entities: ["transparency", "AI systems"]  
Text: "Transparency is required for AI systems"
→ {{"subject": "transparency", "predicate": "is_required_for", "object": "AI systems", "chunk_ids": ["..."]}}

BAD EXAMPLES (DO NOT DO THIS):
❌ {{"subject": "GDPR", "predicate": "is", "object": "important"}} ← "important" not in detected entities
❌ {{"subject": "AI", "predicate": "has", "object": "applications"}} ← "applications" not in detected entities
❌ {{"subject": "policy", "predicate": "affects", "object": "citizens"}} ← neither entity in detected list

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

CRITICAL: Use ONLY entity names from the detected entities list above!

Extract all relations now (JSON only):"""