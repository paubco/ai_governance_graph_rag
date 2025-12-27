# -*- coding: utf-8 -*-
"""
Entity

Tests entity extraction on manually selected chunks before full run.
Supports chunk selection by ID, random sampling, or custom test files.

"""
"""
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_chunks(chunks_file: str = "data/interim/chunks/chunks_text.json") -> List[Dict]:
    """Load all chunks from file."""
    with open(chunks_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def select_chunks_by_id(
    chunks: List[Dict],
    chunk_ids: List[str]
) -> List[Dict]:
    """Select specific chunks by ID."""
    chunk_dict = {c['chunk_id']: c for c in chunks}
    selected = []
    
    for chunk_id in chunk_ids:
        if chunk_id in chunk_dict:
            selected.append(chunk_dict[chunk_id])
        else:
            print(f"Warning: Chunk {chunk_id} not found")
    
    return selected


def sample_diverse_chunks(
    chunks: List[Dict],
    n: int = 10
) -> List[Dict]:
    """
    Sample N diverse chunks from different parts of the corpus.
    
    Strategy: Take chunks evenly distributed across the corpus
    to get regulations, papers, different jurisdictions, etc.
    """
    import random

    # Convert dict to list if needed
    if isinstance(chunks, dict):
        chunks = list(chunks.values())

    # Take from different parts
    step = len(chunks) // n
    indices = [i * step for i in range(n)]
    
    # Add some randomness
    random.seed(42)
    indices = [i + random.randint(-10, 10) for i in indices]
    indices = [max(0, min(i, len(chunks)-1)) for i in indices]
    
    return [chunks[i] for i in indices]


def test_extraction(
    chunks: List[Dict],
    api_key: str = None,
    output_file: str = None
):
    """
    Test entity extraction on selected chunks.
    
    Args:
        chunks: List of chunks to process
        api_key: Together.ai API key
        output_file: Where to save results (optional)
    """
    from src.processing.entities.entity_extractor import RAKGEntityExtractor
    
    print(f"\n{'='*60}")
    print("ENTITY EXTRACTION TEST")
    print(f"{'='*60}")
    print(f"Testing on {len(chunks)} chunks\n")
    
    # Initialize extractor
    extractor = RAKGEntityExtractor(api_key=api_key)
    
    # Process chunks
    all_results = []
    total_entities = 0
    
    for i, chunk in enumerate(chunks):
        chunk_id = chunk['chunk_id']
        chunk_text = chunk['text']
        
        print(f"\n{'-'*60}")
        print(f"Chunk {i+1}/{len(chunks)}: {chunk_id}")
        print(f"{'-'*60}")
        print(f"Text preview: {chunk_text[:150]}...")
        print()
        
        # Extract entities
        entities = extractor.extract_entities(chunk_text, chunk_id)
        
        # Display results
        print(f"Extracted {len(entities)} entities:")
        for j, entity in enumerate(entities, 1):
            print(f"  {j}. {entity['name']} ({entity['type']})")
            print(f"     Description: {entity['description']}")
        
        all_results.append({
            'chunk_id': chunk_id,
            'chunk_text': chunk_text,
            'entities': entities
        })
        
        total_entities += len(entities)
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Chunks processed: {len(chunks)}")
    print(f"Total entities: {total_entities}")
    print(f"Average per chunk: {total_entities/len(chunks):.1f}")
    print(f"{'='*60}\n")
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_file}\n")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Test entity extraction')
    
    # Chunk selection options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--chunk-ids', type=str, 
                      help='Comma-separated list of chunk IDs')
    group.add_argument('--sample', type=int, 
                      help='Random sample of N chunks')
    group.add_argument('--test-file', type=str,
                      help='JSON file with test chunks')
    
    # Other options
    parser.add_argument('--chunks-file', type=str,
                       default='data/interim/chunks/chunks_text.json',
                       help='Path to chunks file')
    parser.add_argument('--output', type=str,
                       help='Output file for results')
    parser.add_argument('--api-key', type=str,
                       help='Together.ai API key')
    
    args = parser.parse_args()
    
    # Load chunks
    if args.test_file:
        print(f"Loading test chunks from {args.test_file}...")
        with open(args.test_file, 'r', encoding='utf-8') as f:
            test_chunks = json.load(f)
    else:
        all_chunks = load_chunks(args.chunks_file)
        
        if args.chunk_ids:
            chunk_ids = args.chunk_ids.split(',')
            test_chunks = select_chunks_by_id(all_chunks, chunk_ids)
        elif args.sample:
            test_chunks = sample_diverse_chunks(all_chunks, args.sample)
    
    # Run test
    results = test_extraction(
        chunks=test_chunks,
        api_key=args.api_key,
        output_file=args.output
    )
    
    return results


if __name__ == "__main__":
    main()
