#!/usr/bin/env python3
"""
BGE-M3 Embedding Script - MEMORY SAFE VERSION
Saves periodically to avoid memory accumulation
"""

import sys
import os

# CRITICAL: Set BEFORE importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,garbage_collection_threshold:0.6'

import json
import torch
import gc
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path
import logging
from datetime import datetime

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
    if len(sys.argv) != 3:
        print("Usage: python embed_server.py <input_chunks.json> <output_chunks.json>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    logger.info("=" * 70)
    logger.info("BGE-M3 CHUNK EMBEDDING - MEMORY SAFE VERSION")
    logger.info("=" * 70)
    
    device = check_gpu()
    start_time = datetime.now()
    
    # Clear any existing GPU cache/fragmentation
    if device == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    # Load model with fp16 for half memory usage
    logger.info("Loading BGE-M3 model (fp16 mode for memory efficiency)...")
    try:
        model = SentenceTransformer('BAAI/bge-m3', device=device)
        if device == 'cuda':
            model = model.half()  # Use fp16 (half precision)
        logger.info("✓ Model loaded in fp16 mode")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Load chunks
    logger.info("-" * 70)
    logger.info(f"Loading chunks from: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)
    
    chunk_ids = list(all_chunks.keys())
    total_chunks = len(chunk_ids)
    logger.info(f"✓ Loaded {total_chunks:,} chunks")
    
    # Configuration
    batch_size = 16  # Moderate size with fragmentation fix
    save_every = 5000  # Save every 5000 chunks to free memory
    
    logger.info("-" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Save interval: {save_every} chunks")
    logger.info(f"  Precision: fp16 (half)")
    logger.info("=" * 70)
    logger.info("Starting embedding process...")
    logger.info("")
    
    embedded_count = 0
    
    try:
        # Process in batches
        for i in tqdm(range(0, total_chunks, batch_size), 
                      desc="Embedding",
                      unit="batch"):
            
            batch_ids = chunk_ids[i:i+batch_size]
            # Truncate very long texts to prevent memory issues
            batch_texts = []
            for cid in batch_ids:
                text = all_chunks[cid]['text']
                # Truncate to max 8192 characters (BGE-M3 limit is ~8192 tokens)
                if len(text) > 8192:
                    text = text[:8192]
                    logger.warning(f"Truncated chunk {cid} from {len(all_chunks[cid]['text'])} to 8192 chars")
                batch_texts.append(text)
            
            # Embed
            with torch.no_grad():  # Don't track gradients
                embeddings = model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_tensor=True,  # Keep as tensor first
                    normalize_embeddings=False
                )
                # Convert to numpy AFTER encoding
                embeddings = embeddings.cpu().numpy()
            
            # Add to chunks
            for chunk_id, emb in zip(batch_ids, embeddings):
                all_chunks[chunk_id]['embedding'] = emb.tolist()
                embedded_count += 1
            
            # Clear memory aggressively
            del embeddings
            del batch_texts
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            # Save periodically and reload to prevent memory accumulation
            if embedded_count % save_every == 0:
                logger.info(f"\n  💾 Checkpoint: Saving {embedded_count:,} chunks...")
                temp_path = output_path.parent / f"{output_path.stem}_temp.json"
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(all_chunks, f, ensure_ascii=False)
                
                # Clear and reload to free memory
                all_chunks.clear()
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
                with open(temp_path, 'r', encoding='utf-8') as f:
                    all_chunks = json.load(f)
                
                temp_path.unlink()  # Delete temp file
                logger.info(f"  ✓ Checkpoint saved, memory cleared")
            
            # Progress log
            if (i // batch_size + 1) % 20 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = embedded_count / elapsed
                remaining = (total_chunks - embedded_count) / rate
                logger.info(f"  Progress: {embedded_count:,}/{total_chunks:,} chunks "
                           f"({embedded_count/total_chunks*100:.1f}%) "
                           f"- ETA: {remaining/60:.1f} min")
        
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user!")
        logger.info(f"Embedded {embedded_count:,}/{total_chunks:,} chunks")
        response = input("Save progress? [Y/n]: ")
        if response.lower() == 'n':
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.info(f"Embedded {embedded_count:,} chunks before error")
        logger.info("Saving partial results...")
    
    # Final save
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Saving final results to: {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    logger.info("✓ Save complete")
    
    # Verification
    total = len(all_chunks)
    with_embeddings = sum(1 for c in all_chunks.values() if 'embedding' in c)
    correct_dim = sum(1 for c in all_chunks.values() 
                     if 'embedding' in c and len(c['embedding']) == 1024)
    
    logger.info("")
    logger.info(f"Verification:")
    logger.info(f"  Total chunks: {total:,}")
    logger.info(f"  With embeddings: {with_embeddings:,}")
    logger.info(f"  Correct dimensions: {correct_dim:,}")
    logger.info(f"  Success rate: {correct_dim/total*100:.2f}%")
    
    # Summary
    elapsed_time = datetime.now() - start_time
    minutes, seconds = divmod(elapsed_time.total_seconds(), 60)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("EMBEDDING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {int(minutes)}m {int(seconds)}s")
    logger.info(f"Chunks processed: {total:,}")
    logger.info(f"Average speed: {total/elapsed_time.total_seconds():.1f} chunks/sec")
    logger.info(f"Output size: {output_path.stat().st_size / 1024**2:.1f} MB")
    logger.info("")
    logger.info("✓ Ready for Phase 1B: Entity Extraction")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()