"""
BGE-M3 Embedding Script for GPU Server
Processes chunks with CUDA acceleration for GraphRAG pipeline

Author: Pau Barba i Colomer
Project: AI Governance Knowledge Graph
Phase: Week 2 - Chunk Embedding
Model: BAAI/bge-m3 (1024-dim multilingual)

Usage:
    python embed_server.py /path/to/chunks_text.json /path/to/output.json

Expected Runtime: ~15 minutes for 25k chunks on RTX 3060
"""

import sys
import json
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('embedding.log')
    ]
)
logger = logging.getLogger(__name__)


def check_gpu():
    """Check GPU availability and specs"""
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {device}")
        logger.info(f"VRAM: {vram:.1f} GB")
        return 'cuda'
    else:
        logger.warning("CUDA not available, falling back to CPU")
        return 'cpu'


def main():
    # Parse arguments
    if len(sys.argv) != 3:
        print("Usage: python embed_server.py <input_chunks.json> <output_chunks.json>")
        print()
        print("Example:")
        print("  python embed_server.py /tmp/chunks_text.json /tmp/chunks_embedded.json")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    # Validate input
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    logger.info("=" * 70)
    logger.info("BGE-M3 CHUNK EMBEDDING - GRAPHRAG PIPELINE")
    logger.info("=" * 70)
    
    # Check device
    device = check_gpu()
    start_time = datetime.now()
    
    # Load model
    logger.info("Loading BGE-M3 model...")
    logger.info("  Model: BAAI/bge-m3")
    logger.info("  Dimensions: 1024")
    logger.info("  Multilingual: Yes (100+ languages)")
    
    try:
        model = SentenceTransformer('BAAI/bge-m3', device=device)
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Load chunks
    logger.info("-" * 70)
    logger.info(f"Loading chunks from: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load chunks: {e}")
        sys.exit(1)
    
    chunk_ids = list(chunks.keys())
    logger.info(f"✓ Loaded {len(chunk_ids):,} chunks")
    
    # Check if already embedded
    if 'embedding' in list(chunks.values())[0]:
        logger.warning("Chunks already contain embeddings!")
        response = input("Overwrite existing embeddings? [y/N]: ")
        if response.lower() != 'y':
            logger.info("Aborted by user")
            sys.exit(0)
    
    # Embedding configuration
    batch_size = 64 if device == 'cuda' else 16  # GPU can handle larger batches
    total_batches = (len(chunk_ids) + batch_size - 1) // batch_size
    
    logger.info("-" * 70)
    logger.info(f"Embedding configuration:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Total batches: {total_batches:,}")
    logger.info(f"  Device: {device}")
    logger.info("=" * 70)
    
    # Embed chunks
    logger.info("Starting embedding process...")
    logger.info("")
    
    embedded_count = 0
    
    try:
        for i in tqdm(range(0, len(chunk_ids), batch_size), 
                      desc="Embedding batches",
                      total=total_batches,
                      unit="batch"):
            
            batch_ids = chunk_ids[i:i+batch_size]
            batch_texts = [chunks[cid]['text'] for cid in batch_ids]
            
            # Embed batch
            embeddings = model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Add to chunks
            for chunk_id, emb in zip(batch_ids, embeddings):
                chunks[chunk_id]['embedding'] = emb.tolist()
                embedded_count += 1
            
            # Progress log every 10 batches
            if (i // batch_size + 1) % 10 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = embedded_count / elapsed
                remaining = (len(chunk_ids) - embedded_count) / rate
                logger.info(f"  Progress: {embedded_count:,}/{len(chunk_ids):,} chunks " 
                           f"({embedded_count/len(chunk_ids)*100:.1f}%) "
                           f"- ETA: {remaining/60:.1f} min")
        
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user!")
        logger.info(f"Embedded {embedded_count:,}/{len(chunk_ids):,} chunks before interruption")
        response = input("Save progress? [Y/n]: ")
        if response.lower() != 'n':
            logger.info("Saving partial results...")
        else:
            logger.info("Discarding progress")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error during embedding: {e}")
        logger.info(f"Embedded {embedded_count:,} chunks before error")
        sys.exit(1)
    
    # Verification
    logger.info("")
    logger.info("=" * 70)
    logger.info("Verifying embeddings...")
    
    total = len(chunks)
    with_embeddings = sum(1 for c in chunks.values() if 'embedding' in c)
    correct_dim = sum(1 for c in chunks.values() 
                     if 'embedding' in c and len(c['embedding']) == 1024)
    
    logger.info(f"  Total chunks: {total:,}")
    logger.info(f"  With embeddings: {with_embeddings:,}")
    logger.info(f"  Correct dimensions (1024): {correct_dim:,}")
    
    success_rate = (correct_dim / total) * 100
    
    if success_rate < 100:
        logger.warning(f"⚠️  Only {success_rate:.2f}% successfully embedded!")
    else:
        logger.info(f"✓ Success rate: {success_rate:.2f}%")
    
    # Save results
    logger.info("-" * 70)
    logger.info(f"Saving enriched chunks to: {output_path}")
    
    try:
        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        logger.info("✓ Save complete")
    except Exception as e:
        logger.error(f"Failed to save output: {e}")
        sys.exit(1)
    
    # Summary
    elapsed_time = datetime.now() - start_time
    minutes, seconds = divmod(elapsed_time.total_seconds(), 60)
    
    logger.info("=" * 70)
    logger.info("EMBEDDING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {int(minutes)}m {int(seconds)}s")
    logger.info(f"Chunks processed: {len(chunk_ids):,}")
    logger.info(f"Average speed: {len(chunk_ids)/elapsed_time.total_seconds():.1f} chunks/sec")
    logger.info(f"Output file size: {output_path.stat().st_size / 1024**2:.1f} MB")
    logger.info("")
    
    # Sample check
    sample = list(chunks.values())[0]
    logger.info(f"Sample chunk verification:")
    logger.info(f"  Chunk ID: {sample['chunk_id']}")
    logger.info(f"  Text length: {len(sample['text'])} chars")
    logger.info(f"  Embedding dimension: {len(sample['embedding'])}")
    logger.info(f"  Embedding sample (first 5 values): {sample['embedding'][:5]}")
    
    logger.info("")
    logger.info("✓ Ready for entity extraction (Phase 1B)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
