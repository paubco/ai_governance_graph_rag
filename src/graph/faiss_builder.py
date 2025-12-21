# -*- coding: utf-8 -*-
"""
FAISS Index Builder for GraphRAG Vector Store.

Builds FAISS HNSW indexes for entity and chunk embeddings with parallel ID mapping.
Handles embedding extraction, index construction, and persistence.

Author: Pau Barba i Colomer
Created: 2025-12-21
Modified: 2025-12-21

References:
    - See ARCHITECTURE.md § 3.2.2 for Phase 2B context
    - See PHASE_2B_DESIGN.md for FAISS indices
"""

# Standard library
import json
from pathlib import Path
from typing import Dict, List, Tuple
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

# Project imports
from src.utils.io import read_jsonl
from src.utils.logger import get_logger

# Config - import with fallback
try:
    from src.config.extraction import FAISS_CONFIG
except ImportError:
    FAISS_CONFIG = {
        'hnsw_m': 32,
        'hnsw_ef_construction': 200,
        'hnsw_ef_search': 64,
        'output_dir': 'data/processed/faiss',
    }

logger = get_logger(__name__)


class FAISSIndexBuilder:
    """
    Build FAISS HNSW indexes with parallel ID mapping.
    
    Handles entity and chunk embeddings separately, maintaining
    consistent ordering between FAISS indices and ID maps.
    
    Example:
        builder = FAISSIndexBuilder(Path('data/processed/faiss'))
        builder.build_all_indexes(entities_file, chunks_file)
    """
    
    def __init__(self, output_dir: Path, config: Dict = None):
        """
        Initialize builder.
        
        Args:
            output_dir: Directory to save indexes and ID maps
            config: Optional config override (defaults to FAISS_CONFIG)
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        cfg = config or FAISS_CONFIG
        self.hnsw_m = cfg.get('hnsw_m', 32)
        self.hnsw_ef_construction = cfg.get('hnsw_ef_construction', 200)
        self.hnsw_ef_search = cfg.get('hnsw_ef_search', 64)
    
    def load_entity_embeddings(self, entities_file: Path) -> Tuple[List[str], np.ndarray]:
        """
        Load entity embeddings from file.
        
        Supports both JSON and JSONL formats.
        
        Args:
            entities_file: Path to entities file with embeddings
            
        Returns:
            Tuple of (entity_ids, embeddings_array)
        """
        logger.info(f"Loading entity embeddings from {entities_file}...")
        
        # Load data based on format
        if entities_file.suffix == '.jsonl':
            entities = list(read_jsonl(entities_file))
        else:
            with open(entities_file, 'r', encoding='utf-8') as f:
                entities = json.load(f)
        
        # Extract in consistent order
        entity_ids = []
        embeddings = []
        skipped = 0
        
        for entity in tqdm(entities, desc="Extracting entity embeddings"):
            entity_id = entity.get('entity_id')
            embedding = entity.get('embedding')
            
            if entity_id and embedding is not None:
                entity_ids.append(entity_id)
                embeddings.append(embedding)
            else:
                skipped += 1
        
        if skipped > 0:
            logger.warning(f"Skipped {skipped} entities without embeddings")
        
        embeddings_array = np.array(embeddings, dtype='float32')
        
        logger.info(f"Loaded {len(entity_ids):,} entity embeddings, dim={embeddings_array.shape[1]}")
        
        return entity_ids, embeddings_array
    
    def load_chunk_embeddings(self, chunks_file: Path) -> Tuple[List[str], np.ndarray]:
        """
        Load chunk embeddings from file.
        
        Supports both JSON and JSONL formats.
        
        Args:
            chunks_file: Path to chunks file with embeddings
            
        Returns:
            Tuple of (chunk_ids, embeddings_array)
        """
        logger.info(f"Loading chunk embeddings from {chunks_file}...")
        
        # Load data based on format
        if chunks_file.suffix == '.jsonl':
            chunks_data = list(read_jsonl(chunks_file))
        else:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            # Handle dict structure: {chunk_id: chunk_object}
            if isinstance(chunks_data, dict):
                chunks_data = list(chunks_data.values())
        
        # Extract in consistent order
        chunk_ids = []
        embeddings = []
        skipped = 0
        
        for chunk in tqdm(chunks_data, desc="Extracting chunk embeddings"):
            # Handle both formats
            chunk_id = chunk.get('chunk_id') or chunk.get('chunk_ids', [''])[0]
            embedding = chunk.get('embedding')
            
            if chunk_id and embedding is not None:
                chunk_ids.append(chunk_id)
                embeddings.append(embedding)
            else:
                skipped += 1
        
        if skipped > 0:
            logger.warning(f"Skipped {skipped} chunks without embeddings")
        
        embeddings_array = np.array(embeddings, dtype='float32')
        
        logger.info(f"Loaded {len(chunk_ids):,} chunk embeddings, dim={embeddings_array.shape[1]}")
        
        return chunk_ids, embeddings_array
    
    def build_hnsw_index(self, embeddings: np.ndarray, m: int = None, 
                         ef_construction: int = None) -> faiss.Index:
        """
        Build FAISS HNSW index.
        
        Args:
            embeddings: Numpy array of embeddings (N, D)
            m: Number of neighbors per node (default: from config)
            ef_construction: Size of dynamic candidate list (default: from config)
            
        Returns:
            FAISS HNSW index
        """
        m = m or self.hnsw_m
        ef_construction = ef_construction or self.hnsw_ef_construction
        
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
            index_file: Filename for index
            map_file: Filename for ID map
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
            entities_file: Path to entities file with embeddings
        """
        logger.info("\n=== BUILDING ENTITY INDEX ===")
        
        entity_ids, embeddings = self.load_entity_embeddings(entities_file)
        index = self.build_hnsw_index(embeddings)
        
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
            chunks_file: Path to chunks file with embeddings
        """
        logger.info("\n=== BUILDING CHUNK INDEX ===")
        
        chunk_ids, embeddings = self.load_chunk_embeddings(chunks_file)
        index = self.build_hnsw_index(embeddings)
        
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
            entities_file: Path to entities file with embeddings
            chunks_file: Path to chunks file with embeddings
        """
        logger.info("Starting FAISS index building...")
        
        self.build_entity_index(entities_file)
        self.build_chunk_index(chunks_file)
        
        logger.info("\n" + "=" * 60)
        logger.info("FAISS INDEX BUILDING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("Files created:")
        logger.info("  - entity_embeddings.index")
        logger.info("  - entity_id_map.json")
        logger.info("  - chunk_embeddings.index")
        logger.info("  - chunk_id_map.json")


# =============================================================================
# MAIN
# =============================================================================

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
        help='Path to entities file with embeddings'
    )
    parser.add_argument(
        '--chunks-file',
        type=Path,
        help='Path to chunks file with embeddings'
    )
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    if args.entities_file is None:
        # Try JSONL first, fall back to JSON
        jsonl_path = args.data_dir / 'interim' / 'entities' / 'entities_semantic.jsonl'
        json_path = args.data_dir / 'interim' / 'entities' / 'normalized_entities_with_ids.json'
        args.entities_file = jsonl_path if jsonl_path.exists() else json_path
    
    if args.chunks_file is None:
        # Try JSONL first, fall back to JSON
        jsonl_path = args.data_dir / 'interim' / 'chunks' / 'chunks_embedded.jsonl'
        json_path = args.data_dir / 'interim' / 'chunks' / 'chunks_embedded.json'
        args.chunks_file = jsonl_path if jsonl_path.exists() else json_path
    
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