"""
Entity extraction prompts for Phase 1B

Following RAKG methodology:
- Free entity types (LLM-discovered, not predefined)
- High coverage principle (over-extract, filter later)
- Citation-aware extraction
"""

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

"""
Entity disambiguation prompts for Phase 1C
"""

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