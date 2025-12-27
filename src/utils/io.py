# -*- coding: utf-8 -*-
"""
I/O utilities for pipeline artifacts

Simple helpers for JSON and JSONL operations with consistent encoding and logging.
Provides convenience functions for loading and saving JSON files, streaming JSONL files,
and

Examples:
# JSON operations
    from src.utils.io import load_json, save_json, load_jsonl, save_jsonl
    entities = load_json("data/processed/entities.json")
    save_json(entities, "data/processed/entities_v2.json")

    # JSONL streaming
    for record in stream_jsonl("data/interim/relations.jsonl"):
        process(record)

"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Iterator, Union

logger = logging.getLogger(__name__)


# ============================================================================
# JSON (for lookups, configs, entity/chunk lists)
# ============================================================================

def load_json(path: Union[str, Path]) -> Any:
    """
    Load JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Parsed JSON content
    """
    path = Path(path)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded {path} ({_size_str(path)})")
    return data


def save_json(
    data: Any,
    path: Union[str, Path],
    indent: int = 2,
) -> str:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save (must be JSON-serializable)
        path: Output path
        indent: Indentation level (default 2)
        
    Returns:
        Path string
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=_serialize)
    
    logger.info(f"Saved {path} ({_size_str(path)})")
    return str(path)


# ============================================================================
# JSONL (for streaming: pre-entities, relations)
# ============================================================================

def load_jsonl(path: Union[str, Path]) -> List[Dict]:
    """
    Load JSONL file as list.
    
    Args:
        path: Path to JSONL file
        
    Returns:
        List of records
    """
    path = Path(path)
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} records from {path}")
    return records


def save_jsonl(
    records: List[Dict],
    path: Union[str, Path],
) -> str:
    """
    Save list to JSONL file.
    
    Args:
        records: List of dicts to save
        path: Output path
        
    Returns:
        Path string
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=_serialize) + '\n')
    
    logger.info(f"Saved {len(records)} records to {path}")
    return str(path)


def stream_jsonl(path: Union[str, Path]) -> Iterator[Dict]:
    """
    Stream JSONL file (memory efficient).
    
    Args:
        path: Path to JSONL file
        
    Yields:
        Records one at a time
    """
    path = Path(path)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def append_jsonl(
    record: Dict,
    path: Union[str, Path],
) -> None:
    """
    Append single record to JSONL file.
    
    Args:
        record: Dict to append
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False, default=_serialize) + '\n')


# ============================================================================
# V1.0 COMPATIBILITY: Pre-entity loader
# ============================================================================

def load_pre_entities_v1(path: Union[str, Path]) -> List[Dict]:
    """
    Load v1.0 pre-entities (nested format with metadata wrapper).
    
    Converts to flat list of pre-entity dicts.
    
    Args:
        path: Path to v1.0 pre-entities JSON
        
    Returns:
        Flat list of {name, type, description, chunk_id}
    """
    path = Path(path)
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # v1.0 format: {"metadata": {...}, "entities": [{chunk_id, entities: [...]}]}
    pre_entities = []
    for chunk_entry in data.get('entities', []):
        chunk_id = chunk_entry.get('chunk_id')
        for entity in chunk_entry.get('entities', []):
            pre_entities.append({
                'name': entity.get('name'),
                'type': entity.get('type'),
                'description': entity.get('description', ''),
                'chunk_id': entity.get('chunk_id') or chunk_id,
            })
    
    logger.info(f"Loaded {len(pre_entities)} pre-entities from v1.0 format")
    return pre_entities


# ============================================================================
# HELPERS
# ============================================================================

def _serialize(obj: Any) -> Any:
    """Convert non-JSON-serializable objects."""
    if hasattr(obj, 'tolist'):  # numpy array
        return obj.tolist()
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return str(obj)


def _size_str(path: Path) -> str:
    """Human-readable file size."""
    size = path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"