#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAISS Index Builder for GraphRAG Vector Store

Builds FAISS HNSW indexes for entity and chunk embeddings with parallel ID mapping.
Handles embedding extraction, index construction, and persistence with consistent
ordering between FAISS indices and ID maps for efficient similarity search.

Index Categories:
1. Entity Embeddings:
   - Source: normalized_entities_with_ids.json
   - Output: entity_embeddings.index + entity_id_map.json
   - Dimension: 1024 (BGE-M3)
2. Chunk Embeddings:
   - Source: chunks_embedded.json
   - Output: chunk_embeddings.index + chunk_id_map.json
   - Dimension: 1024 (BGE-M3)

HNSW Parameters:
- M=32: Number of neighbors per node in graph
- ef_construction=200: Dynamic candidate list size during construction

Usage:
    python -m src.graph.faiss_builder --data-dir data
    python -m src.graph.faiss_builder --entities-file data/interim/entities/normalized_entities_with_ids.json
"""

# Standard library
import json
from pathlib import Path
from typing import List, Dict, Tuple
import sys
import argparse

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
import numpy as np
import faiss
from tqdm import tqdm

# Local
from src.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


class FAISSIndexBuilder:
    """
    Build FAISS HNSW indexes with parallel ID mapping.
    
    Handles entity and chunk embeddings separately, maintaining
    consistent ordering between FAISS indices and ID maps.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize builder.
        
        Args:
            output_dir: Directory to save indexes and ID maps
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_entity_embeddings(self, entities_file: Path) -> Tuple[List[str], np.ndarray]:
        """
        Load entity embeddings from normalized entities file.
        
        Args:
            entities_file: Path to normalized_entities_with_ids.json
            
        Returns:
            Tuple of (entity_ids, embeddings_array)
        """
        logger.info(f"Loading entity embeddings from {entities_file}...")
        
        with open(entities_file, 'r', encoding='utf-8') as f:
            entities = json.load(f)
        
        # Extract in consistent order
        entity_ids = []
        embeddings = []
        
        for entity in tqdm(entities, desc="Extracting entity embeddings"):
            entity_ids.append(entity['entity_id'])
            embeddings.append(entity['embedding'])
        
        embeddings_array = np.array(embeddings, dtype='float32')
        
        logger.info(f"Loaded {len(entity_ids):,} entity embeddings, dim={embeddings_array.shape[1]}")
        
        return entity_ids, embeddings_array
    
    def load_chunk_embeddings(self, chunks_file: Path) -> Tuple[List[str], np.ndarray]:
        """
        Load chunk embeddings from chunks_embedded.json.
        
        Args:
            chunks_file: Path to chunks_embedded.json
            
        Returns:
            Tuple of (chunk_ids, embeddings_array)
        """
        logger.info(f"Loading chunk embeddings from {chunks_file}...")
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        # Handle dict structure: {chunk_id: chunk_object}
        if isinstance(chunks_data, dict):
            chunks = list(chunks_data.values())
        else:
            chunks = chunks_data
        
        # Extract in consistent order
        chunk_ids = []
        embeddings = []
        
        for chunk in tqdm(chunks, desc="Extracting chunk embeddings"):
            chunk_ids.append(chunk['chunk_id'])
            embeddings.append(chunk['embedding'])
        
        embeddings_array = np.array(embeddings, dtype='float32')
        
        logger.info(f"Loaded {len(chunk_ids):,} chunk embeddings, dim={embeddings_array.shape[1]}")
        
        return chunk_ids, embeddings_array
    
    def build_hnsw_index(self, embeddings: np.ndarray, m: int = 32, ef_construction: int = 200) -> faiss.Index:
        """
        Build FAISS HNSW index.
        
        Args:
            embeddings: Numpy array of embeddings (N, D)
            m: Number of neighbors per node in HNSW graph (default: 32)
            ef_construction: Size of dynamic candidate list during construction (default: 200)
            
        Returns:
            FAISS HNSW index
        """
        dim = embeddings.shape[1]
        logger.info(f"Building HNSW index: {len(embeddings):,} vectors, {dim} dimensions")
        logger.info(f"HNSW parameters: M={m}, ef_construction={ef_construction}")
        
        # Create HNSW index
        index = faiss.IndexHNSWFlat(dim, m)
        index.hnsw.efConstruction = ef_construction
        
        # Add embeddings
        logger.info("Adding vectors to index...")
        index.add(embeddings)
        
        logger.info(f"✓ Index built: {index.ntotal:,} vectors")
        
        return index
    
    def save_index_and_map(self, index: faiss.Index, id_map: List[str], 
                          index_file: str, map_file: str):
        """
        Save FAISS index and parallel ID map.
        
        Args:
            index: FAISS index
            id_map: List of IDs in same order as index
            index_file: Filename for index (e.g., 'entity_embeddings.index')
            map_file: Filename for ID map (e.g., 'entity_id_map.json')
        """
        # Verify consistency
        if index.ntotal != len(id_map):
            raise ValueError(
                f"Index/map size mismatch: {index.ntotal} vectors vs {len(id_map)} IDs"
            )
        
        # Save index
        index_path = self.output_dir / index_file
        logger.info(f"Saving index to {index_path}...")
        faiss.write_index(index, str(index_path))
        
        # Save ID map
        map_path = self.output_dir / map_file
        logger.info(f"Saving ID map to {map_path}...")
        with open(map_path, 'w', encoding='utf-8') as f:
            json.dump(id_map, f)
        
        logger.info(f"✓ Saved {len(id_map):,} mappings")
    
    def build_entity_index(self, entities_file: Path):
        """
        Build and save entity embedding index.
        
        Args:
            entities_file: Path to normalized_entities_with_ids.json
        """
        logger.info("\n=== BUILDING ENTITY INDEX ===")
        
        # Load embeddings
        entity_ids, embeddings = self.load_entity_embeddings(entities_file)
        
        # Build index
        index = self.build_hnsw_index(embeddings)
        
        # Save
        self.save_index_and_map(
            index=index,
            id_map=entity_ids,
            index_file='entity_embeddings.index',
            map_file='entity_id_map.json'
        )
        
        logger.info("✓ Entity index complete")
    
    def build_chunk_index(self, chunks_file: Path):
        """
        Build and save chunk embedding index.
        
        Args:
            chunks_file: Path to chunks_embedded.json
        """
        logger.info("\n=== BUILDING CHUNK INDEX ===")
        
        # Load embeddings
        chunk_ids, embeddings = self.load_chunk_embeddings(chunks_file)
        
        # Build index
        index = self.build_hnsw_index(embeddings)
        
        # Save
        self.save_index_and_map(
            index=index,
            id_map=chunk_ids,
            index_file='chunk_embeddings.index',
            map_file='chunk_id_map.json'
        )
        
        logger.info("✓ Chunk index complete")
    
    def build_all_indexes(self, entities_file: Path, chunks_file: Path):
        """
        Build both entity and chunk indexes.
        
        Args:
            entities_file: Path to normalized_entities_with_ids.json
            chunks_file: Path to chunks_embedded.json
        """
        logger.info("Starting FAISS index building...")
        
        # Build entity index
        self.build_entity_index(entities_file)
        
        # Build chunk index
        self.build_chunk_index(chunks_file)
        
        logger.info("\n" + "="*60)
        logger.info("FAISS INDEX BUILDING COMPLETE")
        logger.info("="*60)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("Files created:")
        logger.info("  - entity_embeddings.index")
        logger.info("  - entity_id_map.json")
        logger.info("  - chunk_embeddings.index")
        logger.info("  - chunk_id_map.json")


def main():
    """Main entry point for FAISS index building."""
    parser = argparse.ArgumentParser(
        description='Build FAISS indexes for entity and chunk embeddings'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=PROJECT_ROOT / 'data',
        help='Data directory root (default: PROJECT_ROOT/data)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROJECT_ROOT / 'data' / 'processed' / 'faiss',
        help='Output directory for indexes (default: data/processed/faiss)'
    )
    parser.add_argument(
        '--entities-file',
        type=Path,
        help='Path to normalized_entities_with_ids.json (default: data/processed/entities/...)'
    )
    parser.add_argument(
        '--chunks-file',
        type=Path,
        help='Path to chunks_embedded.json (default: data/processed/chunks/...)'
    )
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    if args.entities_file is None:
        args.entities_file = args.data_dir / 'interim' / 'entities' / 'normalized_entities_with_ids.json'
    
    if args.chunks_file is None:
        args.chunks_file = args.data_dir / 'interim' / 'chunks' / 'chunks_embedded.json'
    
    # Verify files exist
    if not args.entities_file.exists():
        logger.error(f"Entities file not found: {args.entities_file}")
        return 1
    
    if not args.chunks_file.exists():
        logger.error(f"Chunks file not found: {args.chunks_file}")
        return 1
    
    # Build indexes
    builder = FAISSIndexBuilder(args.output_dir)
    builder.build_all_indexes(args.entities_file, args.chunks_file)
    
    return 0


if __name__ == '__main__':
    exit(main())