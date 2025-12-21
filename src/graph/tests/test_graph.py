#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2B Graph Building Preflight Tests.

Quick sanity checks for Neo4j import and FAISS index building.
Run before graph construction to catch configuration and import errors early.

Author: Pau Barba i Colomer
Created: 2025-12-21

Usage:
    python tests/test_graph.py
    python tests/test_graph.py --test-neo4j  # Also test Neo4j connection
"""

# Standard library
import sys
from pathlib import Path
import tempfile
import json
import argparse
import os

# Third-party
from dotenv import load_dotenv

# Load .env before anything else
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """Test all graph modules can be imported."""
    print("\n=== TEST: Graph Module Imports ===")
    
    modules = [
        ('src.graph.neo4j_importer', 'Neo4jImporter'),
        ('src.graph.neo4j_import_processor', 'Neo4jImportProcessor'),
        ('src.graph.faiss_builder', 'FAISSIndexBuilder'),
    ]
    
    all_passed = True
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"  ✓ {class_name}")
        except Exception as e:
            print(f"  ✗ {class_name}: {e}")
            all_passed = False
    
    return all_passed


def test_dependencies():
    """Test required dependencies are installed."""
    print("\n=== TEST: Dependencies ===")
    
    deps = [
        ('neo4j', 'Neo4j driver'),
        ('faiss', 'FAISS'),
        ('numpy', 'NumPy'),
        ('tqdm', 'tqdm'),
    ]
    
    all_passed = True
    for module_name, display_name in deps:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ✗ {display_name} not installed")
            all_passed = False
    
    return all_passed


def test_neo4j_importer_methods():
    """Test Neo4jImporter has all required methods."""
    print("\n=== TEST: Neo4jImporter Methods ===")
    
    from src.graph.neo4j_importer import Neo4jImporter
    
    expected_methods = [
        # Node imports
        'import_jurisdictions',
        'import_publications',
        'import_authors',
        'import_journals',
        'import_chunks',
        'import_entities',
        'import_l2_publications',
        # Relationship imports
        'import_contains_jurisdiction',
        'import_contains_publication',
        'import_authored_by',
        'import_published_in',
        'import_extracted_from',
        'import_relations',
        'import_part_of',           # v1.1
        'import_same_as_entity',    # v1.1
        'import_same_as_jurisdiction',
        'import_matched_to',
        'import_cites_l2',
        'import_cites_l1',
    ]
    
    all_passed = True
    missing = []
    
    for method_name in expected_methods:
        if hasattr(Neo4jImporter, method_name):
            pass  # Don't print each one
        else:
            missing.append(method_name)
            all_passed = False
    
    if all_passed:
        print(f"  ✓ All {len(expected_methods)} import methods present")
        print("  ✓ v1.1 additions: import_part_of, import_same_as_entity")
    else:
        print(f"  ✗ Missing methods: {missing}")
    
    return all_passed


def test_neo4j_connection(uri: str, user: str, password: str):
    """Test actual Neo4j connection."""
    print("\n=== TEST: Neo4j Connection ===")
    
    from neo4j import GraphDatabase
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            value = result.single()['test']
            assert value == 1
        
        driver.close()
        print(f"  ✓ Connected to {uri}")
        return True
        
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        return False


def test_faiss_builder():
    """Test FAISSIndexBuilder with mock embeddings."""
    print("\n=== TEST: FAISSIndexBuilder ===")
    
    try:
        import numpy as np
        import faiss
        from src.graph.faiss_builder import FAISSIndexBuilder
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create mock entity data with embeddings
        entities = [
            {'entity_id': f'ent_{i:03d}', 'embedding': np.random.randn(1024).tolist()}
            for i in range(100)
        ]
        
        entities_file = tmpdir / 'entities.json'
        with open(entities_file, 'w') as f:
            json.dump(entities, f)
        
        # Create mock chunk data with embeddings
        chunks = [
            {'chunk_id': f'chunk_{i:03d}', 'embedding': np.random.randn(1024).tolist()}
            for i in range(50)
        ]
        
        chunks_file = tmpdir / 'chunks.json'
        with open(chunks_file, 'w') as f:
            json.dump(chunks, f)
        
        try:
            output_dir = tmpdir / 'faiss'
            builder = FAISSIndexBuilder(output_dir)
            builder.build_all_indexes(entities_file, chunks_file)
            
            # Verify outputs exist
            assert (output_dir / 'entity_embeddings.index').exists()
            assert (output_dir / 'entity_id_map.json').exists()
            assert (output_dir / 'chunk_embeddings.index').exists()
            assert (output_dir / 'chunk_id_map.json').exists()
            
            # Load and verify entity index
            index = faiss.read_index(str(output_dir / 'entity_embeddings.index'))
            assert index.ntotal == 100, f"Expected 100 vectors, got {index.ntotal}"
            
            # Load and verify ID map
            with open(output_dir / 'entity_id_map.json') as f:
                id_map = json.load(f)
            assert len(id_map) == 100
            
            # Test search
            query = np.random.randn(1, 1024).astype('float32')
            distances, indices = index.search(query, 5)
            assert len(indices[0]) == 5
            
            print(f"  ✓ Built entity index: 100 vectors, 1024 dims")
            print(f"  ✓ Built chunk index: 50 vectors")
            print(f"  ✓ Search working (returned 5 neighbors)")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_checkpoint_manager():
    """Test checkpoint functionality in import processor."""
    print("\n=== TEST: Checkpoint Manager ===")
    
    from src.graph.neo4j_import_processor import Neo4jImportProcessor
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = tmpdir / 'data'
        checkpoint_dir = tmpdir / 'checkpoints'
        data_dir.mkdir()
        checkpoint_dir.mkdir()
        
        try:
            processor = Neo4jImportProcessor(data_dir, checkpoint_dir)
            
            # Test marking steps complete
            processor.mark_completed('test_step_1')
            processor.mark_completed('test_step_2')
            
            assert processor.is_completed('test_step_1')
            assert processor.is_completed('test_step_2')
            assert not processor.is_completed('test_step_3')
            
            # Verify checkpoint file
            assert processor.checkpoint_file.exists()
            
            # Test reload
            processor2 = Neo4jImportProcessor(data_dir, checkpoint_dir)
            assert processor2.is_completed('test_step_1')
            assert processor2.is_completed('test_step_2')
            
            print(f"  ✓ Checkpoint save/load working")
            print(f"  ✓ Step completion tracking working")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False


def run_all_tests(test_neo4j: bool = False):
    """Run all graph building preflight tests."""
    print("=" * 60)
    print("PHASE 2B GRAPH BUILDING PREFLIGHT TESTS")
    print("=" * 60)
    
    results = {}
    
    results['imports'] = test_imports()
    results['dependencies'] = test_dependencies()
    results['neo4j_methods'] = test_neo4j_importer_methods()
    results['checkpoint'] = test_checkpoint_manager()
    results['faiss_builder'] = test_faiss_builder()
    
    # Optional Neo4j connection test
    if test_neo4j:
        uri = os.getenv('NEO4J_URI')
        user = os.getenv('NEO4J_USER', 'neo4j')
        password = os.getenv('NEO4J_PASSWORD')
        
        if uri and password:
            results['neo4j_connection'] = test_neo4j_connection(uri, user, password)
        else:
            print("\n=== TEST: Neo4j Connection ===")
            print("  ⚠ Skipped (NEO4J_URI or NEO4J_PASSWORD not set)")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL GRAPH PREFLIGHT TESTS PASSED")
        print("Ready to run:")
        print("  python -m src.graph.neo4j_import_processor")
        print("  python -m src.graph.faiss_builder")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        print("Fix the issues above before running Phase 2B.")
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 2B Graph Preflight Tests')
    parser.add_argument('--test-neo4j', action='store_true', 
                        help='Also test Neo4j connection (requires env vars)')
    args = parser.parse_args()
    
    exit(run_all_tests(test_neo4j=args.test_neo4j))