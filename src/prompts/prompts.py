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

Your task: Extract relationship triplets from regulatory and academic text without predefined schemas.

TARGET ENTITY:
Name: {entity_name}
Type: {entity_type}
Description: {entity_description}

CONTEXT CHUNKS:
{chunks_text}

EXTRACTION TASK:
Extract ALL relationships where "{entity_name}" is the subject OR object of the relation.

OPENIE PRINCIPLES:
1. NO predefined relation types - discover relations from text
2. Relation predicates are the linking phrases in text (e.g., "regulates", "applies to", "established by")
3. Extract both directions if relevant: (A, predicate1, B) and (B, predicate2, A) are different
4. Only extract relations explicitly stated or strongly implied in the chunks
5. Subject and object should be entities, organizations, concepts, or regulations

EXAMPLES:
- Text: "The GDPR regulates data processing in the EU"
  → ("GDPR", "regulates", "data processing")
  → ("GDPR", "applies_in", "EU")
  
- Text: "AI systems must comply with GDPR requirements"
  → ("AI systems", "must_comply_with", "GDPR")

OUTPUT FORMAT (JSON only, no other text):
{{
  "relations": [
    {{
      "subject": "entity_name",
      "predicate": "relationship_verb_phrase",
      "object": "entity_name",
      "chunk_ids": ["chunk_id_where_found"]
    }}
  ]
}}

CRITICAL RULES:
- Output ONLY valid JSON
- No explanations, notes, markdown, or code blocks
- Predicates should be clear verb phrases (2-4 words typical)
- If no relationships found, return: {{"relations": []}}

Extract all relations now:"""