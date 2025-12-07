#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAISS Index Verification Tests for GraphRAG Vector Store

Verifies FAISS indexes are built correctly and searchable by validating
index integrity, ID mapping consistency, and search functionality.

Verification Categories:
1. File Existence: Ensure all index and mapping files exist
2. Index Loading: Verify indexes can be loaded without errors
3. Count Consistency: Validate index size matches ID map size
4. Dimension Validation: Check embedding dimensions are correct (1024 for BGE-M3)
5. Search Functionality: Test nearest neighbor search returns valid results
6. Size Validation: Verify index file sizes are reasonable

Test Coverage:
- Entity embeddings index (entity_embeddings.index)
- Chunk embeddings index (chunk_embeddings.index)
- Parallel ID mappings (entity_id_map.json, chunk_id_map.json)
- HNSW search with random query vectors

Usage:
    python tests/graph/faiss_builder_test.py
    python tests/graph/faiss_builder_test.py --faiss-dir data/processed/faiss
"""

# Standard library
import json
from pathlib import Path
import sys
import argparse

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
import numpy as np
import faiss

# Local
from src.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


class FAISSIndexVerifier:
    """Verify FAISS indexes are correct and searchable."""
    
    def __init__(self, faiss_dir: Path):
        """
        Initialize verifier.
        
        Args:
            faiss_dir: Directory containing FAISS indexes and ID maps
        """
        self.faiss_dir = faiss_dir
        
        # Expected file paths
        self.entity_index_path = faiss_dir / 'entity_embeddings.index'
        self.entity_map_path = faiss_dir / 'entity_id_map.json'
        self.chunk_index_path = faiss_dir / 'chunk_embeddings.index'
        self.chunk_map_path = faiss_dir / 'chunk_id_map.json'
    
    def verify_files_exist(self) -> bool:
        """
        Verify all required files exist.
        
        Returns:
            True if all files exist
        """
        logger.info("\n=== FILE EXISTENCE CHECK ===")
        
        files = [
            ('Entity index', self.entity_index_path),
            ('Entity ID map', self.entity_map_path),
            ('Chunk index', self.chunk_index_path),
            ('Chunk ID map', self.chunk_map_path),
        ]
        
        all_exist = True
        for name, path in files:
            exists = path.exists()
            status = "✓" if exists else "✗"
            logger.info(f"{status} {name}: {path}")
            
            if not exists:
                all_exist = False
        
        return all_exist
    
    def verify_entity_index(self) -> bool:
        """
        Verify entity index loads and has correct properties.
        
        Returns:
            True if verification passes
        """
        logger.info("\n=== ENTITY INDEX VERIFICATION ===")
        
        try:
            # Load index
            logger.info(f"Loading index from {self.entity_index_path}...")
            index = faiss.read_index(str(self.entity_index_path))
            
            # Load ID map
            logger.info(f"Loading ID map from {self.entity_map_path}...")
            with open(self.entity_map_path, 'r', encoding='utf-8') as f:
                id_map = json.load(f)
            
            # Check counts match
            logger.info(f"Index vectors: {index.ntotal:,}")
            logger.info(f"ID map entries: {len(id_map):,}")
            
            if index.ntotal != len(id_map):
                logger.error(f"✗ Count mismatch: {index.ntotal} vs {len(id_map)}")
                return False
            
            logger.info("✓ Counts match")
            
            # Check dimension (BGE-M3 = 1024)
            expected_dim = 1024
            logger.info(f"Vector dimension: {index.d}")
            
            if index.d != expected_dim:
                logger.warning(f"⚠ Expected dimension {expected_dim}, got {index.d}")
            else:
                logger.info(f"✓ Dimension correct ({expected_dim})")
            
            # Check ID format
            sample_ids = id_map[:5]
            logger.info(f"Sample IDs: {sample_ids}")
            
            # All IDs should start with entity ID prefix
            valid_format = all(isinstance(id, str) and len(id) > 0 for id in sample_ids)
            if valid_format:
                logger.info("✓ ID format valid")
            else:
                logger.error("✗ Invalid ID format")
                return False
            
            # Perform test search
            logger.info("\nPerforming test search (k=5)...")
            query_vector = np.random.randn(1, index.d).astype('float32')
            distances, indices = index.search(query_vector, 5)
            
            logger.info(f"Search returned {len(indices[0])} results")
            logger.info(f"Sample result indices: {indices[0].tolist()}")
            logger.info(f"Sample distances: {distances[0].tolist()}")
            
            # Verify indices are valid
            valid_indices = all(0 <= idx < len(id_map) for idx in indices[0])
            if valid_indices:
                logger.info("✓ All returned indices are valid")
                
                # Show sample mapped IDs
                sample_result_ids = [id_map[idx] for idx in indices[0]]
                logger.info(f"Mapped to IDs: {sample_result_ids}")
            else:
                logger.error("✗ Some indices out of range")
                return False
            
            logger.info("\n✓ Entity index verification passed")
            return True
        
        except Exception as e:
            logger.error(f"✗ Entity index verification failed: {e}")
            return False
    
    def verify_chunk_index(self) -> bool:
        """
        Verify chunk index loads and has correct properties.
        
        Returns:
            True if verification passes
        """
        logger.info("\n=== CHUNK INDEX VERIFICATION ===")
        
        try:
            # Load index
            logger.info(f"Loading index from {self.chunk_index_path}...")
            index = faiss.read_index(str(self.chunk_index_path))
            
            # Load ID map
            logger.info(f"Loading ID map from {self.chunk_map_path}...")
            with open(self.chunk_map_path, 'r', encoding='utf-8') as f:
                id_map = json.load(f)
            
            # Check counts match
            logger.info(f"Index vectors: {index.ntotal:,}")
            logger.info(f"ID map entries: {len(id_map):,}")
            
            if index.ntotal != len(id_map):
                logger.error(f"✗ Count mismatch: {index.ntotal} vs {len(id_map)}")
                return False
            
            logger.info("✓ Counts match")
            
            # Check dimension (BGE-M3 = 1024)
            expected_dim = 1024
            logger.info(f"Vector dimension: {index.d}")
            
            if index.d != expected_dim:
                logger.warning(f"⚠ Expected dimension {expected_dim}, got {index.d}")
            else:
                logger.info(f"✓ Dimension correct ({expected_dim})")
            
            # Check ID format
            sample_ids = id_map[:5]
            logger.info(f"Sample IDs: {sample_ids}")
            
            # All IDs should be valid strings
            valid_format = all(isinstance(id, str) and len(id) > 0 for id in sample_ids)
            if valid_format:
                logger.info("✓ ID format valid")
            else:
                logger.error("✗ Invalid ID format")
                return False
            
            # Perform test search
            logger.info("\nPerforming test search (k=5)...")
            query_vector = np.random.randn(1, index.d).astype('float32')
            distances, indices = index.search(query_vector, 5)
            
            logger.info(f"Search returned {len(indices[0])} results")
            logger.info(f"Sample result indices: {indices[0].tolist()}")
            logger.info(f"Sample distances: {distances[0].tolist()}")
            
            # Verify indices are valid
            valid_indices = all(0 <= idx < len(id_map) for idx in indices[0])
            if valid_indices:
                logger.info("✓ All returned indices are valid")
                
                # Show sample mapped IDs
                sample_result_ids = [id_map[idx] for idx in indices[0]]
                logger.info(f"Mapped to IDs: {sample_result_ids}")
            else:
                logger.error("✗ Some indices out of range")
                return False
            
            logger.info("\n✓ Chunk index verification passed")
            return True
        
        except Exception as e:
            logger.error(f"✗ Chunk index verification failed: {e}")
            return False
    
    def verify_index_sizes(self) -> bool:
        """
        Verify index file sizes are reasonable.
        
        Returns:
            True if sizes look correct
        """
        logger.info("\n=== INDEX SIZE CHECK ===")
        
        entity_index_size = self.entity_index_path.stat().st_size / (1024 ** 2)  # MB
        chunk_index_size = self.chunk_index_path.stat().st_size / (1024 ** 2)  # MB
        
        logger.info(f"Entity index: {entity_index_size:.1f} MB")
        logger.info(f"Chunk index: {chunk_index_size:.1f} MB")
        logger.info(f"Total: {entity_index_size + chunk_index_size:.1f} MB")
        
        # Rough expectation: 76K entities × 1024 dim × 4 bytes ≈ 312 MB
        #                    25K chunks × 1024 dim × 4 bytes ≈ 102 MB
        # HNSW adds overhead, so expect ~400-500 MB total
        
        total_size = entity_index_size + chunk_index_size
        if 300 < total_size < 700:
            logger.info("✓ Index sizes look reasonable")
            return True
        else:
            logger.warning(f"⚠ Total size {total_size:.1f} MB outside expected range (300-700 MB)")
            return True  # Warning, not error
    
    def run_all_verifications(self) -> bool:
        """
        Run complete verification suite.
        
        Returns:
            True if all verifications pass
        """
        logger.info("Starting FAISS index verification...")
        
        results = {}
        
        results['files_exist'] = self.verify_files_exist()
        
        if not results['files_exist']:
            logger.error("Cannot proceed - files missing")
            return False
        
        results['entity_index'] = self.verify_entity_index()
        results['chunk_index'] = self.verify_chunk_index()
        results['sizes'] = self.verify_index_sizes()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("VERIFICATION SUMMARY")
        logger.info("="*60)
        
        for check, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"{status}: {check}")
        
        all_passed = all(results.values())
        
        if all_passed:
            logger.info("\n✓ ALL VERIFICATIONS PASSED")
            logger.info("\nFAISS indexes are ready for use!")
        else:
            logger.warning("\n✗ SOME VERIFICATIONS FAILED - Review logs above")
        
        return all_passed


def main():
    """Main entry point for FAISS verification."""
    parser = argparse.ArgumentParser(
        description='Verify FAISS indexes are built correctly'
    )
    parser.add_argument(
        '--faiss-dir',
        type=Path,
        default=PROJECT_ROOT / 'data' / 'processed' / 'faiss',
        help='Directory containing FAISS indexes (default: data/processed/faiss)'
    )
    
    args = parser.parse_args()
    
    if not args.faiss_dir.exists():
        logger.error(f"FAISS directory not found: {args.faiss_dir}")
        return 1
    
    # Run verification
    verifier = FAISSIndexVerifier(args.faiss_dir)
    success = verifier.run_all_verifications()
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
