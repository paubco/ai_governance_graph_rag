# -*- coding: utf-8 -*-
"""
Query

Parses user queries into structured format with LLM-based entity extraction,
rule-based metadata filter extraction, and query embedding.

"""
# Standard library
import json
import os
from typing import List

# Third-party
import numpy as np
from together import Together
from dotenv import load_dotenv

# Config imports (direct)
from config.retrieval_config import (
    ENTITY_TYPES,
    parse_jurisdictions,
    parse_doc_types,
)

# Dataclass imports (direct)
from src.utils.dataclasses import (
    ParsedQuery,
    QueryFilters,
    ExtractedQueryEntity as ExtractedEntity,
)

# Prompts
from src.prompts.prompts import QUERY_ENTITY_EXTRACTION_PROMPT

# Load environment
load_dotenv()


class QueryParser:
    """
    Parse user queries into structured form.
    
    Combines:
    - LLM entity extraction (Mistral-7B with type enforcement)
    - Rule-based filter extraction (jurisdictions, doc types)
    - Query embedding (BGE-M3 for semantic retrieval)
    """
    
    def __init__(
        self,
        embedding_model,
        api_key: str = None,
        model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    ):
        """
        Initialize query parser.
        
        Args:
            embedding_model: BGE-M3 embedding model with embed_single() method.
            api_key: Together.ai API key (or from TOGETHER_API_KEY env var).
            model: LLM model for entity extraction.
        """
        self.embedding_model = embedding_model
        
        api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError(
                "Together.ai API key required. "
                "Set TOGETHER_API_KEY in .env file or pass api_key parameter."
            )
        
        self.client = Together(api_key=api_key)
        self.model = model
    
    def parse(self, query: str) -> ParsedQuery:
        """
        Parse user query into structured form.
        
        Args:
            query: Natural language query string.
            
        Returns:
            ParsedQuery with embedding, entities, and filters.
        """
        return ParsedQuery(
            raw_query=query,
            embedding=self._embed_query(query),
            extracted_entities=self._extract_entities_llm(query),
            resolved_entities=[],  # Filled by EntityResolver
            filters=QueryFilters(
                jurisdiction_hints=parse_jurisdictions(query),
                doc_type_hints=parse_doc_types(query)
            )
        )
    
    def _embed_query(self, query: str) -> np.ndarray:
        """Embed query using BGE-M3."""
        embedding = self.embedding_model.embed_single(query)
        if embedding.ndim == 2:
            embedding = embedding.squeeze()
        return embedding
    
    def _extract_entities_llm(self, query: str) -> List[ExtractedEntity]:
        """Extract entities from query using Mistral-7B."""
        prompt = QUERY_ENTITY_EXTRACTION_PROMPT.format(
            query=query,
            entity_types=", ".join(ENTITY_TYPES)
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
                stop=["</s>", "\n\nUser:", "\n\nAssistant:"]
            )
            
            content = response.choices[0].message.content.strip()
            entities_raw = self._parse_json_response(content)
            
            entities = []
            for entity_dict in entities_raw:
                if 'name' not in entity_dict or 'type' not in entity_dict:
                    continue
                if entity_dict['type'] not in ENTITY_TYPES:
                    continue
                entities.append(ExtractedEntity(
                    name=entity_dict['name'],
                    type=entity_dict['type']
                ))
            
            return entities
        
        except Exception as e:
            print(f"Error extracting entities from query: {e}")
            return []
    
    def _parse_json_response(self, content: str) -> List[dict]:
        """Parse JSON array from LLM response."""
        try:
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].strip().startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)
            
            content = content.strip()
            result = json.loads(content)
            
            if isinstance(result, list):
                return result
            elif isinstance(result, dict) and 'entities' in result:
                return result['entities']
            else:
                return []
        
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return []