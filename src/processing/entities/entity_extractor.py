# -*- coding: utf-8 -*-
"""
RAKG-style entity extractor using LLM for free-type discovery.

Extracts pre-entities from text chunks using LLM (Qwen2.5-72B-Instruct-Turbo).
Following RAKG's approach: free entity types, high coverage, natural filtering later.
"""

# Standard library
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
from dotenv import load_dotenv
from together import Together

# Load environment
load_dotenv(PROJECT_ROOT / '.env')

class RAKGEntityExtractor:
    """
    Extract pre-entities from text chunks using LLM.

    Following RAKG methodology:
    - Free entity types (LLM-discovered, not predefined ontology)
    - High coverage (over-extract, filter in Phase 1C via corpus retrieval)
    - Citation-aware extraction
    - Temperature=0 for deterministic output

    Output format per chunk:
    {
        'chunk_id': 'doc_123_chunk_5',
        'entities': [
            {
                'name': 'GDPR',
                'type': 'Data Protection Regulation',  # LLM-discovered
                'description': 'EU regulation on data protection...',
                'chunk_id': 'doc_123_chunk_5'
            },
            {
                'name': 'Pani et al., 2021',
                'type': 'Citation',
                'description': 'Academic reference...',
                'chunk_id': 'doc_123_chunk_5'
            }
        ]
    }
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "Qwen/Qwen2.5-72B-Instruct-Turbo"
    ):
        """
        Initialize the entity extractor.

        Args:
            api_key: Together.ai API key (or from TOGETHER_API_KEY env var)
            model: Model to use for extraction
        """
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError(
                "Together.ai API key required. "
                "Set TOGETHER_API_KEY environment variable or pass api_key parameter."
            )

        self.client = Together(api_key=api_key)
        self.model = model

        # Import prompt
        from src.prompts.prompts import ENTITY_EXTRACTION_PROMPT
        self.prompt_template = ENTITY_EXTRACTION_PROMPT

    def extract_entities(
        self,
        chunk_text: str,
        chunk_id: str
    ) -> List[Dict]:
        """
        Extract entities from a single chunk.

        Args:
            chunk_text: Text content of the chunk
            chunk_id: Unique identifier for the chunk

        Returns:
            List of entity dictionaries with keys: name, type, description, chunk_id
            Returns empty list on error
        """
        # Format prompt with chunk text
        prompt = self.prompt_template.format(text=chunk_text)

        try:
            # Call LLM API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Deterministic
                max_tokens=2048,
                stop=["</s>", "\n\nUser:", "\n\nAssistant:"]
            )

            # Extract response content
            content = response.choices[0].message.content

            # Parse JSON response
            entities = self._parse_json_response(content, chunk_id)

            # Add chunk_id to each entity for traceability
            for entity in entities:
                entity['chunk_id'] = chunk_id

            return entities

        except Exception as e:
            print(f"Error extracting entities from chunk {chunk_id}: {e}")
            return []

    def _parse_json_response(
        self,
        content: str,
        chunk_id: str
    ) -> List[Dict]:
        """
        Parse JSON from LLM response.

        Handles common issues:
        - Markdown code blocks (```json ... ```)
        - Extra whitespace
        - Malformed JSON

        Args:
            content: Raw LLM response
            chunk_id: Chunk identifier (for error logging)

        Returns:
            List of entity dictionaries
        """
        try:
            # Strip markdown code blocks if present
            content = content.strip()
            if content.startswith("```"):
                # Remove opening ```json or ```
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                # Remove closing ```
                if content.endswith("```"):
                    content = content.rsplit("```", 1)[0]

            content = content.strip()

            # Parse JSON
            result = json.loads(content)

            # Validate structure
            if 'entities' not in result:
                print(f"Warning: No 'entities' key in response for chunk {chunk_id}")
                return []

            entities = result['entities']

            if not isinstance(entities, list):
                print(f"Warning: 'entities' is not a list for chunk {chunk_id}")
                return []

            # Validate each entity has required fields
            valid_entities = []
            for entity in entities:
                if all(key in entity for key in ['name', 'type', 'description']):
                    valid_entities.append(entity)
                else:
                    print(f"Warning: Entity missing required fields in chunk {chunk_id}: {entity}")

            return valid_entities

        except json.JSONDecodeError as e:
            print(f"JSON parse error for chunk {chunk_id}: {e}")
            print(f"Response content (first 200 chars): {content[:200]}...")
            return []
        except Exception as e:
            print(f"Unexpected error parsing response for chunk {chunk_id}: {e}")
            return []

    def extract_batch(
        self,
        chunks: List[Dict],
        verbose: bool = True
    ) -> Dict[str, List[Dict]]:
        """
        Extract entities from multiple chunks.

        Args:
            chunks: List of chunk dictionaries with 'chunk_id' and 'text' keys
            verbose: Whether to print progress

        Returns:
            Dictionary mapping chunk_id to list of entities
        """
        results = {}

        for i, chunk in enumerate(chunks):
            chunk_id = chunk['chunk_id']
            chunk_text = chunk['text']

            if verbose and (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(chunks)} chunks...")

            entities = self.extract_entities(chunk_text, chunk_id)
            results[chunk_id] = entities

        if verbose:
            total_entities = sum(len(ents) for ents in results.values())
            print(f"\nExtraction complete: {total_entities} entities from {len(chunks)} chunks")

        return results
