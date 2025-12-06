# -*- coding: utf-8 -*-
"""
Module: id_generator.py
Package: src.utils
Purpose: Consistent ID generation using deterministic hashing

Author: Pau Barba i Colomer
Created: 2025-12-05
Modified: 2025-12-05

References:
    - Used across entity disambiguation, alias discovery, and enrichment
    - See docs/ARCHITECTURE.md for ID schemes
"""

import hashlib
from typing import Optional


def generate_entity_id(entity_name: str, entity_type: Optional[str] = None) -> str:
    """
    Generate deterministic entity ID from name and optional type.
    
    Uses MD5 hash of normalized string to ensure consistent IDs across
    pipeline stages. Same entity name always generates same ID.
    
    Args:
        entity_name: Canonical entity name
        entity_type: Optional entity type for disambiguation
        
    Returns:
        Entity ID in format "ent_<12-char-hash>"
        
    Example:
        >>> generate_entity_id("EU AI Act")
        "ent_a3f4e9c2d5b1"
        >>> generate_entity_id("EU AI Act", "Regulation")
        "ent_a3f4e9c2d5b1"  # Same ID (type not used in hash)
    """
    # Normalize: lowercase, strip whitespace
    normalized = entity_name.lower().strip()
    
    # Generate hash
    hash_obj = hashlib.md5(normalized.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()[:12]
    
    return f"ent_{hash_hex}"


def generate_publication_id(prefix: str, source_string: str) -> str:
    """
    Generate deterministic publication ID from source string.
    
    Args:
        prefix: ID prefix ("pub_l1" or "pub_l2")
        source_string: Source identifier (scopus_id or reference string)
        
    Returns:
        Publication ID in format "<prefix>_<12-char-hash>"
        
    Example:
        >>> generate_publication_id("pub_l1", "2-s2.0-85123456789")
        "pub_l1_a3f4e9c2d5b1"
        >>> generate_publication_id("pub_l2", "Floridi, L. (2018). Digital Ethics...")
        "pub_l2_7b8c3d4e5f6a"
    """
    hash_obj = hashlib.md5(source_string.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()[:12]
    
    return f"{prefix}_{hash_hex}"


def generate_author_id(scopus_author_id: str) -> str:
    """
    Generate author ID from Scopus author ID.
    
    Args:
        scopus_author_id: Scopus author identifier
        
    Returns:
        Author ID in format "author_<scopus_id>"
        
    Example:
        >>> generate_author_id("56291236900")
        "author_56291236900"
    """
    return f"author_{scopus_author_id}"


def generate_journal_id(journal_name: str) -> str:
    """
    Generate deterministic journal ID from name.
    
    Args:
        journal_name: Journal name
        
    Returns:
        Journal ID in format "journal_<12-char-hash>"
        
    Example:
        >>> generate_journal_id("Nature")
        "journal_3c5d7e9f1a2b"
    """
    normalized = journal_name.lower().strip()
    hash_obj = hashlib.md5(normalized.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()[:12]
    
    return f"journal_{hash_hex}"


def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
    """
    Generate chunk ID from document ID and index.
    
    Args:
        doc_id: Document identifier (e.g., "paper_123", "reg_ES")
        chunk_index: Chunk position in document (0-indexed)
        
    Returns:
        Chunk ID in format "<doc_id>_CHUNK_<4-digit-index>"
        
    Example:
        >>> generate_chunk_id("paper_085", 42)
        "paper_085_CHUNK_0042"
        >>> generate_chunk_id("reg_ES", 1)
        "reg_ES_CHUNK_0001"
    """
    return f"{doc_id}_CHUNK_{chunk_index:04d}"
