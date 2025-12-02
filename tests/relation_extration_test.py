"""
Phase 1D Real Integration Test

Usage:
    cd ~/Graph_RAG
    python tests/test_phase_1d_real.py
"""

import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_data():
    """Load real entities and chunks"""
    
    # Load entities
    entities_paths = [
        PROJECT_ROOT / 'data/processed/normalized_entities.json',
        PROJECT_ROOT / 'data/processed/entities/normalized_entities.json',
        PROJECT_ROOT / 'data/interim/entities/normalized_entities.json',
    ]
    
    entities_path = None
    for path in entities_paths:
        if path.exists():
            entities_path = path
            break
    
    if not entities_path:
        raise FileNotFoundError("normalized_entities.json not found")
    
    logger.info(f"Loading entities from: {entities_path}")
    with open(entities_path, 'r', encoding='utf-8') as f:
        entities_data = json.load(f)
    
    if isinstance(entities_data, dict) and 'entities' in entities_data:
        entities = entities_data['entities']
    elif isinstance(entities_data, list):
        entities = entities_data
    else:
        raise ValueError("Unknown entity format")
    
    logger.info(f"  Loaded: {len(entities)} entities")
    
    # Load chunks
    chunks_path = PROJECT_ROOT / 'data/interim/chunks/chunks_embedded.json'
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks_embedded.json not found at {chunks_path}")
    
    logger.info(f"Loading chunks from: {chunks_path}")
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    if isinstance(chunks_data, dict):
        chunks = list(chunks_data.values()) if 'chunks' not in chunks_data else chunks_data['chunks']
    elif isinstance(chunks_data, list):
        chunks = chunks_data
    else:
        raise ValueError("Unknown chunk format")
    
    logger.info(f"  Loaded: {len(chunks)} chunks")
    
    return entities, chunks


def test_real_extraction():
    """Test actual relation extraction with LLM"""
    print("=" * 80)
    print("PHASE 1D - REAL INTEGRATION TEST")
    print("=" * 80)
    print()
    
    from src.phase1_graph_construction.relation_extractor import RAKGRelationExtractor
    
    # Load data
    print("1. Loading data...")
    entities, chunks = load_data()
    
    # Initialize extractor (will use API key from .env)
    print("\n2. Initializing extractor...")
    extractor = RAKGRelationExtractor(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        semantic_threshold=0.85,
        mmr_lambda=0.55,
        num_chunks=20
    )
    print(f"   ✓ Extractor ready (API key loaded from .env)")
    
    # Test on 2 entities
    test_entities = entities[:2]
    
    print(f"\n3. Testing extraction on {len(test_entities)} entities:")
    for i, entity in enumerate(test_entities, 1):
        print(f"\n   Entity {i}: {entity.get('name', 'Unknown')} ({entity.get('type', 'Unknown')})")
        print(f"   Description: {entity.get('description', 'N/A')[:100]}...")
        
        try:
            print(f"   Extracting relations...")
            relations = extractor.extract_relations_for_entity(entity, chunks)
            
            print(f"   ✓ Extracted {len(relations)} relations")
            
            if relations:
                print(f"\n   Sample relations:")
                for j, rel in enumerate(relations[:3], 1):
                    print(f"     {j}. ({rel.get('subject')}) --[{rel.get('predicate')}]--> ({rel.get('object')})")
                    print(f"        From chunks: {rel.get('chunk_ids', [])[:2]}...")
            else:
                print(f"   (No relations found for this entity)")
                
        except Exception as e:
            print(f"   ✗ Extraction failed: {e}")
            raise
    
    print("\n" + "=" * 80)
    print("✅ REAL EXTRACTION TEST PASSED")
    print("=" * 80)
    print("\nThe pipeline works! Ready for full run:")
    print("  python server_scripts/relation_processor_server.py --workers 5 --test --test-size 10")
    print()


def main():
    try:
        test_real_extraction()
        return 0
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ Test failed: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())