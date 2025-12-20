# -*- coding: utf-8 -*-
"""
Phase 1C Threshold Refinement - FAISS blocking + sample generation for human review.

Runs FAISS blocking to find similar entity pairs, outputs samples at different
similarity bands for human labeling to determine optimal thresholds.

Input:  entities_semantic_embedded.jsonl
Output: 
  - candidate_pairs.jsonl (all pairs above min threshold)
  - threshold_samples.json (samples per band for human labeling)
  - threshold_review.txt (human-readable format for labeling)

Usage:
    python -m src.processing.entities.test_threshold_refinement
    python -m src.processing.entities.test_threshold_refinement --k 50 --min-sim 0.70 --samples 20
"""

import json
import logging
import argparse
import random
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path('data')
INPUT_FILE = DATA_DIR / 'processed' / 'entities' / 'entities_semantic_embedded.jsonl'
PAIRS_FILE = DATA_DIR / 'interim' / 'entities' / 'candidate_pairs.jsonl'
SAMPLES_FILE = DATA_DIR / 'interim' / 'entities' / 'threshold_samples.json'

# Similarity bands for threshold tuning
SIMILARITY_BANDS = [
    ('0.98+', 0.98, 1.01),
    ('0.95-0.98', 0.95, 0.98),
    ('0.90-0.95', 0.90, 0.95),
    ('0.85-0.90', 0.85, 0.90),
    ('0.80-0.85', 0.80, 0.85),
    ('0.75-0.80', 0.75, 0.80),
    ('0.70-0.75', 0.70, 0.75),
]


def load_entities(filepath: Path) -> List[Dict]:
    """Load embedded entities from JSONL."""
    entities = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entity = json.loads(line)
                # Convert embedding list back to numpy
                if 'embedding' in entity and isinstance(entity['embedding'], list):
                    entity['embedding'] = np.array(entity['embedding'], dtype=np.float32)
                entities.append(entity)
    logger.info(f"Loaded {len(entities)} entities from {filepath}")
    
    # Check embeddings
    with_embeddings = sum(1 for e in entities if 'embedding' in e)
    logger.info(f"  {with_embeddings} have embeddings")
    
    return entities


def build_faiss_index(entities: List[Dict], embedding_dim: int = 1024, M: int = 32):
    """Build FAISS HNSW index from entity embeddings."""
    import faiss
    
    logger.info(f"Building FAISS index for {len(entities)} entities...")
    
    # Extract embeddings
    embeddings = []
    for entity in entities:
        if 'embedding' in entity:
            embeddings.append(entity['embedding'])
        else:
            embeddings.append(np.zeros(embedding_dim, dtype=np.float32))
    
    embeddings = np.vstack(embeddings).astype(np.float32)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Build HNSW index
    index = faiss.IndexHNSWFlat(embedding_dim, M)
    index.hnsw.efConstruction = 200
    index.add(embeddings)
    
    logger.info(f"Index built: {index.ntotal} vectors")
    return index, embeddings


def find_candidate_pairs(index, embeddings: np.ndarray, entities: List[Dict],
                         k: int = 50, min_similarity: float = 0.70,
                         ef_search: int = 64) -> List[Dict]:
    """
    Find candidate pairs above minimum similarity threshold.
    
    Args:
        index: FAISS index
        embeddings: Normalized embedding matrix
        entities: Entity list
        k: Number of neighbors to search
        min_similarity: Minimum cosine similarity
        ef_search: HNSW search parameter
        
    Returns:
        List of pair dicts with similarity scores
    """
    import faiss
    
    logger.info(f"Finding candidates (k={k}, min_sim={min_similarity})...")
    
    index.hnsw.efSearch = ef_search
    distances, indices = index.search(embeddings, k)
    
    # Convert to pairs
    pairs = []
    seen = set()
    
    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        for dist, j in zip(dists, idxs):
            if i >= j:  # Skip self and duplicates
                continue
            
            # Convert L2 distance to cosine similarity (for normalized vectors)
            similarity = 1.0 - (dist / 2.0)
            
            if similarity < min_similarity:
                continue
            
            pair_key = (min(i, j), max(i, j))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            
            pairs.append({
                'idx1': i,
                'idx2': j,
                'entity1_name': entities[i].get('name', ''),
                'entity1_type': entities[i].get('type', ''),
                'entity1_id': entities[i].get('entity_id', ''),
                'entity2_name': entities[j].get('name', ''),
                'entity2_type': entities[j].get('type', ''),
                'entity2_id': entities[j].get('entity_id', ''),
                'similarity': float(similarity),
            })
    
    # Sort by similarity descending
    pairs.sort(key=lambda x: -x['similarity'])
    
    logger.info(f"Found {len(pairs)} candidate pairs")
    return pairs


def analyze_similarity_distribution(pairs: List[Dict]) -> Dict:
    """Analyze similarity distribution of pairs."""
    if not pairs:
        return {'total_pairs': 0, 'bands': {}}
    
    sims = [p['similarity'] for p in pairs]
    
    # Count per band
    bands = {}
    for band_name, low, high in SIMILARITY_BANDS:
        count = sum(1 for s in sims if low <= s < high)
        bands[band_name] = count
    
    stats = {
        'total_pairs': len(pairs),
        'min_similarity': min(sims),
        'max_similarity': max(sims),
        'mean_similarity': sum(sims) / len(sims),
        'bands': bands,
    }
    
    return stats


def sample_pairs_for_tuning(pairs: List[Dict], samples_per_band: int = 15,
                            seed: int = 42) -> Dict[str, List[Dict]]:
    """
    Sample pairs from each similarity band for human review.
    
    Returns dict of band -> sample pairs for labeling.
    """
    random.seed(seed)
    
    # Group by band
    band_pairs = {band_name: [] for band_name, _, _ in SIMILARITY_BANDS}
    
    for pair in pairs:
        sim = pair['similarity']
        for band_name, low, high in SIMILARITY_BANDS:
            if low <= sim < high:
                band_pairs[band_name].append(pair)
                break
    
    # Sample from each band
    samples = {}
    for band_name in band_pairs:
        band_list = band_pairs[band_name]
        n_sample = min(samples_per_band, len(band_list))
        if n_sample > 0:
            samples[band_name] = random.sample(band_list, n_sample)
        else:
            samples[band_name] = []
    
    return samples


def format_samples_for_review(samples: Dict[str, List[Dict]]) -> str:
    """Format samples as human-readable text for labeling."""
    lines = []
    lines.append("="*70)
    lines.append("THRESHOLD TUNING - Label each pair: SAME / DIFF / UNSURE")
    lines.append("="*70)
    lines.append("")
    lines.append("Instructions:")
    lines.append("  SAME  = These refer to the same real-world entity")
    lines.append("  DIFF  = These are different entities")
    lines.append("  UNSURE = Cannot determine without more context")
    lines.append("")
    lines.append("Goal: Find thresholds where:")
    lines.append("  - auto_merge: lowest band where nearly ALL are SAME")
    lines.append("  - auto_reject: highest band where nearly ALL are DIFF")
    lines.append("  - uncertain band in between → send to LLM")
    
    for band_name, _, _ in SIMILARITY_BANDS:
        band_samples = samples.get(band_name, [])
        if not band_samples:
            continue
            
        lines.append(f"\n{'='*70}")
        lines.append(f"BAND: {band_name} ({len(band_samples)} samples)")
        lines.append("="*70)
        
        for i, pair in enumerate(band_samples, 1):
            lines.append(f"\n[{band_name}#{i}] sim={pair['similarity']:.4f}")
            lines.append(f"  A: {pair['entity1_name']} [{pair['entity1_type']}]")
            lines.append(f"  B: {pair['entity2_name']} [{pair['entity2_type']}]")
            lines.append(f"  Label: _______ (SAME / DIFF / UNSURE)")
    
    lines.append("\n" + "="*70)
    lines.append("SUMMARY - Fill in after labeling:")
    lines.append("="*70)
    lines.append("")
    for band_name, _, _ in SIMILARITY_BANDS:
        lines.append(f"  {band_name}: SAME=___ DIFF=___ UNSURE=___")
    lines.append("")
    lines.append("Recommended thresholds:")
    lines.append("  auto_merge_threshold:  ____")
    lines.append("  auto_reject_threshold: ____")
    lines.append("")
    lines.append("="*70)
    lines.append("END OF SAMPLES")
    lines.append("="*70)
    
    return "\n".join(lines)


def save_pairs(pairs: List[Dict], filepath: Path):
    """Save candidate pairs to JSONL."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for pair in pairs:
            # Remove idx fields for cleaner output
            pair_clean = {k: v for k, v in pair.items() if not k.startswith('idx')}
            f.write(json.dumps(pair_clean, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(pairs)} pairs to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='FAISS blocking + threshold tuning')
    parser.add_argument('--input', type=str, default=str(INPUT_FILE),
                       help='Input embedded entities JSONL')
    parser.add_argument('--k', type=int, default=50,
                       help='Number of neighbors to search')
    parser.add_argument('--min-sim', type=float, default=0.70,
                       help='Minimum similarity threshold')
    parser.add_argument('--samples', type=int, default=15,
                       help='Samples per band for tuning')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Load entities
    entities = load_entities(input_path)
    
    if not any('embedding' in e for e in entities):
        logger.error("No embeddings found! Run embed_entities.py first:")
        logger.error("  python -m src.processing.entities.embed_entities")
        return
    
    # Build index and find pairs
    index, embeddings = build_faiss_index(entities)
    pairs = find_candidate_pairs(index, embeddings, entities, 
                                  k=args.k, min_similarity=args.min_sim)
    
    # Analyze distribution
    stats = analyze_similarity_distribution(pairs)
    
    print("\n" + "="*50)
    print("SIMILARITY DISTRIBUTION")
    print("="*50)
    print(f"Total pairs:    {stats['total_pairs']:,}")
    if stats['total_pairs'] > 0:
        print(f"Min similarity: {stats['min_similarity']:.4f}")
        print(f"Max similarity: {stats['max_similarity']:.4f}")
        print(f"Mean:           {stats['mean_similarity']:.4f}")
        print("\nBands:")
        for band_name, _, _ in SIMILARITY_BANDS:
            count = stats['bands'].get(band_name, 0)
            pct = count / stats['total_pairs'] * 100 if stats['total_pairs'] > 0 else 0
            print(f"  {band_name}: {count:>6,} ({pct:>5.1f}%)")
    
    # Save all pairs
    save_pairs(pairs, PAIRS_FILE)
    
    # Sample for tuning
    samples = sample_pairs_for_tuning(pairs, samples_per_band=args.samples, seed=args.seed)
    
    # Save samples as JSON
    SAMPLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SAMPLES_FILE, 'w', encoding='utf-8') as f:
        # Clean samples for JSON (remove idx)
        samples_clean = {}
        for band, pairs_list in samples.items():
            samples_clean[band] = [
                {k: v for k, v in p.items() if not k.startswith('idx')}
                for p in pairs_list
            ]
        json.dump(samples_clean, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved samples to {SAMPLES_FILE}")
    
    # Save human-readable format
    review_file = SAMPLES_FILE.parent / 'threshold_review.txt'
    review_text = format_samples_for_review(samples)
    with open(review_file, 'w', encoding='utf-8') as f:
        f.write(review_text)
    logger.info(f"Saved review file to {review_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("OUTPUT FILES")
    print("="*50)
    print(f"All pairs:     {PAIRS_FILE}")
    print(f"Samples JSON:  {SAMPLES_FILE}")
    print(f"Review file:   {review_file}")
    print("\n" + "="*50)
    print("NEXT STEPS")
    print("="*50)
    print(f"1. Open {review_file}")
    print("2. Label each pair: SAME / DIFF / UNSURE")
    print("3. Count results per band")
    print("4. Determine thresholds:")
    print("   - auto_merge: lowest band where ~90%+ are SAME")
    print("   - auto_reject: highest band where ~90%+ are DIFF")
    print("5. Update ENTITY_DISAMBIGUATION_CONFIG in extraction_config.py")


if __name__ == '__main__':
    main()