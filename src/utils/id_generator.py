# -*- coding: utf-8 -*-
"""
Deterministic ID generation for pipeline artifacts.

Single source of truth for all ID generation. Uses SHA-256 hashing
with consistent normalization for reproducible IDs across runs.

v1.1: Consolidated from id_generator.py and add_entity_ids.py.
      Standardized on SHA-256 with "name|type" format.

Example:
    from src.utils.id_generator import generate_entity_id, generate_chunk_id
    
    entity_id = generate_entity_id("EU AI Act", "Regulation")
    # Returns: "ent_a3f4e9c2d5b1"
"""

import hashlib
from typing import Optional


def _hash_string(content: str, length: int = 12) -> str:
    """
    Generate truncated SHA-256 hash of content.
    
    Args:
        content: String to hash (should be pre-normalized)
        length: Number of hex characters to return (default 12 = 48 bits)
        
    Returns:
        Lowercase hex hash string
        
    Note:
        12 chars = 48 bits = ~281 trillion combinations.
        Collision probability is negligible for <1M entities.
    """
    hash_obj = hashlib.sha256(content.encode('utf-8'))
    return hash_obj.hexdigest()[:length]


# ============================================================================
# ENTITY IDS
# ============================================================================

def generate_entity_id(name: str, entity_type: str) -> str:
    """
    Generate deterministic entity ID from name and type.
    
    Uses SHA-256 hash of normalized "name|type" string.
    Same entity always generates same ID across pipeline runs.
    
    Args:
        name: Entity name (will be lowercased and stripped)
        entity_type: Entity type (will be lowercased and stripped)
        
    Returns:
        Entity ID in format "ent_<12-char-hex>"
        
    Example:
        >>> generate_entity_id("EU AI Act", "Regulation")
        "ent_7a3f2e9c4d1b"
        >>> generate_entity_id("eu ai act", "regulation")  # Same ID
        "ent_7a3f2e9c4d1b"
    """
    # Normalize: lowercase, strip whitespace
    normalized_name = name.lower().strip()
    normalized_type = entity_type.lower().strip()
    
    # Combine with separator
    content = f"{normalized_name}|{normalized_type}"
    
    return f"ent_{_hash_string(content)}"


def generate_entity_id_from_name_only(name: str) -> str:
    """
    Generate entity ID from name only (legacy compatibility).
    
    DEPRECATED: Use generate_entity_id() with type for v1.1+.
    This exists only for loading v1.0 data that used name-only hashing.
    
    Args:
        name: Entity name
        
    Returns:
        Entity ID in format "ent_<12-char-hex>"
    """
    normalized = name.lower().strip()
    return f"ent_{_hash_string(normalized)}"


# ============================================================================
# CHUNK IDS
# ============================================================================

def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
    """
    Generate chunk ID from document ID and position.
    
    Format: "<doc_id>_CHUNK_<4-digit-index>"
    
    Args:
        doc_id: Document identifier (e.g., "reg_EU", "paper_042")
        chunk_index: 0-indexed position in document
        
    Returns:
        Chunk ID string
        
    Example:
        >>> generate_chunk_id("reg_EU", 42)
        "reg_EU_CHUNK_0042"
        >>> generate_chunk_id("paper_085", 7)
        "paper_085_CHUNK_0007"
    """
    return f"{doc_id}_CHUNK_{chunk_index:04d}"


def parse_chunk_id(chunk_id: str) -> tuple:
    """
    Parse chunk ID into components.
    
    Args:
        chunk_id: Chunk ID string
        
    Returns:
        Tuple of (doc_id, chunk_index)
        
    Example:
        >>> parse_chunk_id("reg_EU_CHUNK_0042")
        ("reg_EU", 42)
    """
    parts = chunk_id.rsplit("_CHUNK_", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid chunk ID format: {chunk_id}")
    return parts[0], int(parts[1])


# ============================================================================
# PUBLICATION IDS (L1 = corpus papers, L2 = referenced papers)
# ============================================================================

def generate_publication_id(scopus_id: str, layer: int = 1) -> str:
    """
    Generate publication ID from Scopus ID.
    
    Args:
        scopus_id: Scopus document ID (e.g., "2-s2.0-85123456789")
        layer: 1 for corpus papers, 2 for referenced papers
        
    Returns:
        Publication ID in format "pub_l<layer>_<12-char-hash>"
        
    Example:
        >>> generate_publication_id("2-s2.0-85123456789", layer=1)
        "pub_l1_a3f4e9c2d5b1"
    """
    prefix = f"pub_l{layer}"
    return f"{prefix}_{_hash_string(scopus_id)}"


def generate_l2_publication_id(reference_string: str) -> str:
    """
    Generate L2 publication ID from reference string.
    
    For papers not in corpus (cited but not downloaded).
    
    Args:
        reference_string: Full reference text (e.g., "Floridi, L. (2018)...")
        
    Returns:
        Publication ID in format "pub_l2_<12-char-hash>"
        
    Example:
        >>> generate_l2_publication_id("Floridi, L. (2018). Digital Ethics...")
        "pub_l2_7b8c3d4e5f6a"
    """
    # Normalize to reduce near-duplicates
    normalized = reference_string.lower().strip()
    return f"pub_l2_{_hash_string(normalized)}"


# ============================================================================
# AUTHOR AND JOURNAL IDS
# ============================================================================

def generate_author_id(scopus_author_id: str) -> str:
    """
    Generate author ID from Scopus author ID.
    
    Uses Scopus ID directly (no hashing) for traceability.
    
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
    Generate journal ID from name.
    
    Args:
        journal_name: Journal name
        
    Returns:
        Journal ID in format "journal_<12-char-hash>"
        
    Example:
        >>> generate_journal_id("Nature Machine Intelligence")
        "journal_3c5d7e9f1a2b"
    """
    normalized = journal_name.lower().strip()
    return f"journal_{_hash_string(normalized)}"


# ============================================================================
# RELATION IDS (for Neo4j edge uniqueness)
# ============================================================================

def generate_relation_id(subject_id: str, predicate: str, object_id: str) -> str:
    """
    Generate unique relation ID from triplet.
    
    Used for deduplication and Neo4j edge identification.
    
    Args:
        subject_id: Source entity ID
        predicate: Relation predicate
        object_id: Target entity ID
        
    Returns:
        Relation ID in format "rel_<12-char-hash>"
        
    Example:
        >>> generate_relation_id("ent_abc123", "regulates", "ent_def456")
        "rel_9e8d7c6b5a4f"
    """
    content = f"{subject_id}|{predicate}|{object_id}"
    return f"rel_{_hash_string(content)}"


# ============================================================================
# VALIDATION
# ============================================================================

def is_valid_entity_id(entity_id: str) -> bool:
    """Check if string is a valid entity ID format."""
    if not entity_id.startswith("ent_"):
        return False
    hex_part = entity_id[4:]
    return len(hex_part) == 12 and all(c in '0123456789abcdef' for c in hex_part)


def is_valid_chunk_id(chunk_id: str) -> bool:
    """Check if string is a valid chunk ID format."""
    return "_CHUNK_" in chunk_id


def get_id_type(id_string: str) -> Optional[str]:
    """
    Determine the type of an ID string.
    
    Returns:
        One of: "entity", "chunk", "publication", "author", "journal", "relation", None
    """
    if id_string.startswith("ent_"):
        return "entity"
    elif "_CHUNK_" in id_string:
        return "chunk"
    elif id_string.startswith("pub_"):
        return "publication"
    elif id_string.startswith("author_"):
        return "author"
    elif id_string.startswith("journal_"):
        return "journal"
    elif id_string.startswith("rel_"):
        return "relation"
    return None
