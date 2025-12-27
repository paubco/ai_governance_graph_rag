# -*- coding: utf-8 -*-
"""
Pre Entity Extractor

- Semantic pass: 9 types (concepts, regulations, technology, etc.)
- Metadata pass: 6 types (Citation, Author, Journal, Affiliation, Document, DocumentSection)

"""
# Standard library
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
from dotenv import load_dotenv
from together import Together

# Local imports
from src.utils.dataclasses import PreEntity
from config.extraction_config import (
    ENTITY_EXTRACTION_CONFIG,
    SEMANTIC_ENTITY_TYPES,
    METADATA_ENTITY_TYPES,
)
from src.prompts.prompts import (
    SEMANTIC_EXTRACTION_PROMPT,
    METADATA_EXTRACTION_PROMPT,
)

# Load environment
load_dotenv(PROJECT_ROOT / '.env')

# Logger
logger = logging.getLogger(__name__)


class DualPassEntityExtractor:
    """
    Extract pre-entities using dual-pass methodology.
    
    Pass 1 (Semantic): All chunks -> 11 domain-fused types
    Pass 2 (Academic): Paper chunks only -> 4 types
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the dual-pass entity extractor.
        
        Args:
            api_key: Together.ai API key (or from TOGETHER_API_KEY env var)
            model: Model to use (default from config)
        """
        api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError(
                "Together.ai API key required. "
                "Set TOGETHER_API_KEY env var or pass api_key parameter."
            )
        
        self.client = Together(api_key=api_key)
        self.model = model or ENTITY_EXTRACTION_CONFIG['model_name']
        
        # Valid types (v2.0 - semantic + metadata)
        self.semantic_types = set(SEMANTIC_ENTITY_TYPES)
        self.metadata_types = set(METADATA_ENTITY_TYPES)
        
        logger.info(f"DualPassEntityExtractor initialized with {self.model}")
    
    def extract_entities(
        self,
        chunk_text: str,
        chunk_id: str,
        doc_type: str = "regulation",
    ) -> List[PreEntity]:
        """
        Extract entities from a single chunk using dual-pass methodology.
        
        Args:
            chunk_text: Text content of the chunk
            chunk_id: Unique identifier for the chunk
            doc_type: "regulation" or "paper" (both run metadata pass in v2.0)
            
        Returns:
            List of PreEntity objects with embedding_text pre-computed
        """
        entities: List[PreEntity] = []
        
        # Pass 1: Semantic extraction (all chunks)
        semantic_entities = self._extract_semantic(chunk_text, chunk_id)
        entities.extend(semantic_entities)
        
        # Pass 2: Metadata extraction (all chunks in v2.0)
        metadata_entities = self._extract_metadata(chunk_text, chunk_id)
        entities.extend(metadata_entities)
        
        return entities
    
    def _extract_semantic(
        self,
        chunk_text: str,
        chunk_id: str,
    ) -> List[PreEntity]:
        """
        Pass 1: Extract semantic entities with domain-fused types.
        Retries once if >50% entities are rejected (type validation failures).
        """
        max_attempts = 2
        
        for attempt in range(max_attempts):
            prompt = SEMANTIC_EXTRACTION_PROMPT.format(chunk_text=chunk_text)
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=ENTITY_EXTRACTION_CONFIG['max_tokens'],
                    response_format={"type": "json_object"},
                )
                
                content = response.choices[0].message.content
                raw_entities = self._parse_json_response(content, chunk_id, "semantic")
                
                # Convert to PreEntity with validation
                entities = []
                for raw in raw_entities:
                    entity = self._create_semantic_entity(raw, chunk_id)
                    if entity:
                        entities.append(entity)
                
                # Retry if too many rejections
                if raw_entities and len(entities) / len(raw_entities) < 0.5:
                    if attempt == 0:
                        logger.warning(f"Low valid ratio for {chunk_id} ({len(entities)}/{len(raw_entities)}), retrying...")
                        continue
                
                return entities
                
            except Exception as e:
                logger.error(f"Semantic extraction failed for {chunk_id}: {e}")
                return []
        
        return entities
    
    def _extract_metadata(
        self,
        chunk_text: str,
        chunk_id: str,
    ) -> List[PreEntity]:
        """
        Pass 2: Extract metadata entities (citations, authors, documents, sections).
        Retries once if >50% entities are rejected.
        """
        max_attempts = 2
        
        for attempt in range(max_attempts):
            prompt = METADATA_EXTRACTION_PROMPT.format(chunk_text=chunk_text)
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=ENTITY_EXTRACTION_CONFIG['max_tokens'],
                    response_format={"type": "json_object"},
                )
                
                content = response.choices[0].message.content
                raw_entities = self._parse_json_response(content, chunk_id, "metadata")
                
                # Convert to PreEntity with validation
                entities = []
                for raw in raw_entities:
                    entity = self._create_metadata_entity(raw, chunk_id)
                    if entity:
                        entities.append(entity)
                
                # Retry if too many rejections
                if raw_entities and len(entities) / len(raw_entities) < 0.5:
                    if attempt == 0:
                        logger.warning(f"Low valid ratio (metadata) for {chunk_id} ({len(entities)}/{len(raw_entities)}), retrying...")
                        continue
                
                return entities
                
            except Exception as e:
                logger.error(f"Metadata extraction failed for {chunk_id}: {e}")
                return []
        
        return entities
    
    def _create_semantic_entity(
        self,
        raw: Dict,
        chunk_id: str,
    ) -> Optional[PreEntity]:
        """
        Create PreEntity from raw semantic extraction with validation.
        
        v1.2: No domain field - domain is baked into type name.
        Computes embedding_text as "{name}({type})".
        """
        name = raw.get('name', '').strip()
        entity_type = raw.get('type', '').strip()
        description = raw.get('description', '').strip()
        
        # Validate required fields
        if not name or not entity_type:
            logger.warning(f"Missing fields in semantic entity: {raw}")
            return None
        
        # Validate type
        if entity_type not in self.semantic_types:
            logger.warning(f"Invalid type '{entity_type}' for entity '{name}', skipping")
            return None
        
        # Compute embedding text: "{name}({type})"
        embedding_text = f"{name}({entity_type})"
        
        return PreEntity(
            name=name,
            type=entity_type,
            description=description,
            chunk_id=chunk_id,
            embedding_text=embedding_text,
        )
    
    def _create_metadata_entity(
        self,
        raw: Dict,
        chunk_id: str,
    ) -> Optional[PreEntity]:
        """
        Create PreEntity from raw metadata extraction with validation.
        
        Validates type against metadata types.
        Computes embedding_text as "{name}({type})".
        """
        name = raw.get('name', '').strip()
        entity_type = raw.get('type', '').strip()
        description = raw.get('description', '').strip()
        
        # Validate required fields
        if not name or not entity_type:
            logger.warning(f"Missing fields in metadata entity: {raw}")
            return None
        
        # Validate type
        if entity_type not in self.metadata_types:
            logger.warning(f"Invalid type '{entity_type}' for metadata entity '{name}', skipping")
            return None
        
        # Compute embedding text: "{name}({type})"
        embedding_text = f"{name}({entity_type})"
        
        return PreEntity(
            name=name,
            type=entity_type,
            description=description,
            chunk_id=chunk_id,
            embedding_text=embedding_text,
        )
    
    def _parse_json_response(
        self,
        content: str,
        chunk_id: str,
        pass_type: str,
    ) -> List[Dict]:
        """
        Parse JSON from LLM response.
        
        Handles:
        - Markdown code blocks (```json ... ```)
        - Extra whitespace
        - Malformed JSON
        
        Args:
            content: Raw LLM response
            chunk_id: Chunk identifier (for logging)
            pass_type: "semantic" or "metadata" (for logging)
            
        Returns:
            List of entity dictionaries
        """
        try:
            content = content.strip()
            
            # Strip markdown code blocks if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                if content.endswith("```"):
                    content = content.rsplit("```", 1)[0]
            
            content = content.strip()
            
            # Handle empty response
            if not content or content == "[]":
                return []
            
            # Parse JSON
            result = json.loads(content)
            
            # Handle both array and object formats
            if isinstance(result, list):
                return result
            elif isinstance(result, dict) and 'entities' in result:
                return result['entities']
            else:
                logger.warning(f"Unexpected JSON structure for {chunk_id} ({pass_type}): {type(result)}")
                return []
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error for {chunk_id} ({pass_type}): {e}")
            logger.debug(f"Content: {content[:200]}...")
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing {chunk_id} ({pass_type}): {e}")
            return []
    
    def extract_batch(
        self,
        chunks: List[Dict],
        verbose: bool = True,
    ) -> List[PreEntity]:
        """
        Extract entities from multiple chunks.
        
        Args:
            chunks: List of chunk dicts with 'chunk_id', 'text', and optionally 'doc_type'
            verbose: Print progress
            
        Returns:
            Flat list of all PreEntity objects
        """
        all_entities: List[PreEntity] = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = chunk['chunk_id']
            chunk_text = chunk['text']
            doc_type = chunk.get('doc_type', self._infer_doc_type(chunk_id))
            
            if verbose and (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(chunks)} chunks...")
            
            entities = self.extract_entities(chunk_text, chunk_id, doc_type)
            all_entities.extend(entities)
        
        if verbose:
            semantic_count = sum(1 for e in all_entities if e.type in self.semantic_types)
            metadata_count = len(all_entities) - semantic_count
            logger.info(f"Extraction complete: {len(all_entities)} entities "
                       f"({semantic_count} semantic, {metadata_count} metadata) "
                       f"from {len(chunks)} chunks")
        
        return all_entities
    
    def _infer_doc_type(self, chunk_id: str) -> str:
        """Infer document type from chunk ID prefix."""
        if chunk_id.startswith("reg_"):
            return "regulation"
        elif chunk_id.startswith("paper_"):
            return "paper"
        return "regulation"  # Default to regulation (semantic only)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_extractor(api_key: Optional[str] = None) -> DualPassEntityExtractor:
    """Factory function to create extractor with default config."""
    return DualPassEntityExtractor(api_key=api_key)


def extract_from_chunk(
    chunk_text: str,
    chunk_id: str,
    doc_type: str = "regulation",
    api_key: Optional[str] = None,
) -> List[PreEntity]:
    """
    Convenience function for single-chunk extraction.
    
    Creates extractor, extracts, returns. Use DualPassEntityExtractor
    directly for batch processing to avoid re-initialization.
    """
    extractor = DualPassEntityExtractor(api_key=api_key)
    return extractor.extract_entities(chunk_text, chunk_id, doc_type)