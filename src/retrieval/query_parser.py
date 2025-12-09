# -*- coding: utf-8 -*-
"""
Module: query_parser.py
Package: src.retrieval
Purpose: Parse user queries with LLM entity extraction (Phase 3)

Author: Pau Barba i Colomer
Created: 2025-12-07
Modified: 2025-12-07

References:
    - PHASE_3_DESIGN.md § 5.1 (LLM entity extraction)
    - Consistent with Phase 1D methodology (Mistral-7B)
"""

import json
import os
from typing import List

import numpy as np
from together import Together
from dotenv import load_dotenv

from .config import (
    ParsedQuery,
    QueryFilters,
    ExtractedEntity,
    ENTITY_TYPES,
    parse_jurisdictions,
    parse_doc_types,
)
from src.prompts.prompts import QUERY_ENTITY_EXTRACTION_PROMPT

# Load environment variables from .env
load_dotenv()


# ============================================================================
# QUERY PARSER
# ============================================================================

class QueryParser:
    """
    Parse user queries into structured form.
    
    Combines:
    - LLM entity extraction (Mistral-7B with type enforcement)
    - Rule-based filter extraction (jurisdictions, doc types)
    - Query embedding (BGE-M3 for Path B)
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
            api_key: Together.ai API key (or from TOGETHER_API_KEY env var or .env file).
            model: LLM model for entity extraction.
        """
        self.embedding_model = embedding_model
        
        # Get API key from .env if not provided
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
        
        Steps:
        1. Embed query (for Path B naive retrieval)
        2. Extract entities via LLM (for Path A GraphRAG)
        3. Parse filters via rules (jurisdictions, doc types)
        
        Args:
            query: Natural language query string.
            
        Returns:
            ParsedQuery with embedding, entities, and filters.
        """
        return ParsedQuery(
            raw_query=query,
            query_embedding=self._embed_query(query),
            extracted_entities=self._extract_entities_llm(query),
            filters=QueryFilters(
                jurisdiction_hints=parse_jurisdictions(query),
                doc_type_hints=parse_doc_types(query)
            )
        )
    
    def _embed_query(self, query: str) -> np.ndarray:
        """
        Embed query using BGE-M3.
        
        Args:
            query: Query string to embed.
            
        Returns:
            Embedding vector (1024 dimensions).
        """
        embedding = self.embedding_model.embed_single(query)  # FIXED: embed_single not embed
        
        # Ensure correct shape (1024,)
        if embedding.ndim == 2:
            embedding = embedding.squeeze()
        
        return embedding
    
    def _extract_entities_llm(self, query: str) -> List[ExtractedEntity]:
        """
        Extract entities from query using Mistral-7B.
        
        Strategy:
        - LLM with type enforcement (must use ENTITY_TYPES)
        - JSON schema for reliable parsing
        - Temperature=0 for determinism
        
        Args:
            query: Query string to analyze.
            
        Returns:
            List of ExtractedEntity objects.
        """
        # Format prompt
        prompt = QUERY_ENTITY_EXTRACTION_PROMPT.format(
            query=query,
            entity_types=", ".join(ENTITY_TYPES)
        )
        
        try:
            # Call LLM API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},  # Enforce JSON output
                stop=["</s>", "\n\nUser:", "\n\nAssistant:"]
            )
            
            # Extract response content
            content = response.choices[0].message.content.strip()
            
            # Parse JSON
            entities_raw = self._parse_json_response(content)
            
            # Convert to ExtractedEntity objects
            entities = []
            for entity_dict in entities_raw:
                # Validate required fields
                if 'name' not in entity_dict or 'type' not in entity_dict:
                    continue
                
                # Validate type is in ENTITY_TYPES
                if entity_dict['type'] not in ENTITY_TYPES:
                    # Try to map to closest valid type
                    # For now, skip invalid types
                    continue
                
                entities.append(ExtractedEntity(
                    name=entity_dict['name'],
                    type=entity_dict['type']
                ))
            
            return entities
        
        except Exception as e:
            print(f"Error extracting entities from query: {e}")
            print(f"Query: {query}")
            return []
    
    def _parse_json_response(self, content: str) -> List[dict]:
        """
        Parse JSON array from LLM response.
        
        Handles:
        - Markdown code blocks (```json ... ```)
        - Extra whitespace
        - Malformed JSON
        
        Args:
            content: Raw LLM response.
            
        Returns:
            List of entity dictionaries.
        """
        try:
            # Strip markdown code blocks if present
            if content.startswith("```"):
                # Remove opening ```json or ```
                lines = content.split("\n")
                if lines[0].strip().startswith("```"):
                    lines = lines[1:]
                # Remove closing ```
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)
            
            content = content.strip()
            
            # Parse JSON
            result = json.loads(content)
            
            # Handle both array and object formats
            if isinstance(result, list):
                return result
            elif isinstance(result, dict) and 'entities' in result:
                return result['entities']
            else:
                print(f"Warning: Unexpected JSON structure: {type(result)}")
                return []
        
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Response content (first 200 chars): {content[:200]}...")
            return []
        except Exception as e:
            print(f"Unexpected error parsing response: {e}")
            return []