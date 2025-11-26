#!/usr/bin/env python3
"""
BGE-M3 Embedding Script - UNIVERSAL VERSION
Auto-detects progress and resumes or starts fresh
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
    """Check GPU availability"""
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
        print()
        print("Examples:")
        print("  # Start fresh:")
        print("  python embed_server.py data/interim/chunks/chunks_text.json data/interim/chunks/chunks_embedded.json")
        print()
        print("  # Resume (input = output):")
        print("  python embed_server.py data/interim/chunks/chunks_embedded.json data/interim/chunks/chunks_embedded.json")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    logger.info("=" * 70)
    logger.info("BGE-M3 CHUNK EMBEDDING - UNIVERSAL MODE")
    logger.info("=" * 70)
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info("")
    
    device = check_gpu()
    
    # Clear GPU cache before starting
    if device == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("✓ GPU cache cleared")
    
    # Load chunks
    logger.info("-" * 70)
    logger.info(f"Loading chunks from: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)
    
    total_chunks = len(all_chunks)
    logger.info(f"✓ Loaded {total_chunks:,} chunks")
    
    # Detect resume vs fresh start
    sample_chunk = next(iter(all_chunks.values()))
    is_resume = 'embedding' in sample_chunk
    
    if is_resume:
        # Count what's already done
        with_embeddings = sum(1 for c in all_chunks.values() if 'embedding' in c and len(c.get('embedding', [])) == 1024)
        missing_ids = [cid for cid, chunk in all_chunks.items() 
                      if 'embedding' not in chunk or len(chunk.get('embedding', [])) != 1024]
        
        logger.info("")
        logger.info("🔄 RESUME MODE DETECTED")
        logger.info(f"  Already embedded: {with_embeddings:,} chunks ({with_embeddings/total_chunks*100:.1f}%)")
        logger.info(f"  Remaining: {len(missing_ids):,} chunks ({len(missing_ids)/total_chunks*100:.1f}%)")
        
        if len(missing_ids) == 0:
            logger.info("")
            logger.info("✓ All chunks already embedded! Nothing to do.")
            sys.exit(0)
        
        chunk_ids = missing_ids
    else:
        logger.info("")
        logger.info("🆕 FRESH START MODE")
        chunk_ids = list(all_chunks.keys())
    
    # Load model
    logger.info("-" * 70)
    logger.info("Loading BGE-M3 model (fp16 mode for memory efficiency)...")
    
    try:
        model = SentenceTransformer('BAAI/bge-m3', device=device)
        if device == 'cuda':
            model = model.half()  # fp16
        logger.info("✓ Model loaded in fp16 mode")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Configuration - Conservative for stability
    batch_size = 8  # Small and safe
    save_interval = 1000  # Save every 1000 chunks
    
    logger.info("-" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Save interval: {save_interval} chunks")
    logger.info(f"  Precision: fp16 (half)")
    logger.info(f"  Memory management: max_split_size_mb=512")
    logger.info("=" * 70)
    logger.info(f"Starting embedding process for {len(chunk_ids):,} chunks...")
    logger.info("")
    
    start_time = datetime.now()
    embedded_count = 0
    
    try:
        for i in tqdm(range(0, len(chunk_ids), batch_size),
                      desc="Embedding batches",
                      unit="batch"):
            
            batch_ids = chunk_ids[i:i+batch_size]
            
            # Get texts and truncate if needed
            batch_texts = []
            for cid in batch_ids:
                text = all_chunks[cid]['text']
                # Truncate very long texts (BGE-M3 max is ~8192 tokens)
                if len(text) > 8192:
                    text = text[:8192]
                batch_texts.append(text)
            
            # Embed with fp16
            with torch.no_grad():
                embeddings = model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    normalize_embeddings=False
                )
                # Convert to numpy on CPU
                embeddings = embeddings.cpu().numpy()
            
            # Add to chunks
            for chunk_id, emb in zip(batch_ids, embeddings):
                all_chunks[chunk_id]['embedding'] = emb.tolist()
                embedded_count += 1
            
            # Aggressive memory cleanup after each batch
            del embeddings
            del batch_texts
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            # Save checkpoint periodically
            if embedded_count % save_interval == 0:
                logger.info(f"\n💾 Checkpoint: Saving progress ({embedded_count:,}/{len(chunk_ids):,})...")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(all_chunks, f, ensure_ascii=False)
                logger.info(f"✓ Checkpoint saved")
                
                # Extra memory cleanup at checkpoint
                if device == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
            
            # Progress logging every 20 batches
            if (i // batch_size + 1) % 20 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = embedded_count / elapsed if elapsed > 0 else 0
                remaining = (len(chunk_ids) - embedded_count) / rate if rate > 0 else 0
                logger.info(f"  Progress: {embedded_count:,}/{len(chunk_ids):,} chunks "
                           f"({embedded_count/len(chunk_ids)*100:.1f}%) "
                           f"- ETA: {remaining/60:.1f} min")
    
    except KeyboardInterrupt:
        logger.warning("\n⚠ Interrupted by user!")
        logger.info(f"Embedded {embedded_count:,}/{len(chunk_ids):,} chunks in this session")
        response = input("Save progress? [Y/n]: ")
        if response.lower() == 'n':
            logger.info("Progress not saved")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error during embedding: {e}")
        logger.info(f"Embedded {embedded_count:,} chunks before error")
        logger.info("Saving partial results...")
    
    # Final save
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Saving final results to: {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    logger.info("✓ Final save complete")
    
    # Verification
    logger.info("")
    logger.info("Verification:")
    total = len(all_chunks)
    with_embeddings = sum(1 for c in all_chunks.values() if 'embedding' in c)
    correct_dim = sum(1 for c in all_chunks.values() 
                     if 'embedding' in c and len(c['embedding']) == 1024)
    
    logger.info(f"  Total chunks: {total:,}")
    logger.info(f"  With embeddings: {with_embeddings:,}")
    logger.info(f"  Correct dimensions (1024): {correct_dim:,}")
    logger.info(f"  Success rate: {correct_dim/total*100:.2f}%")
    
    # Session summary
    elapsed_time = datetime.now() - start_time
    minutes, seconds = divmod(elapsed_time.total_seconds(), 60)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("EMBEDDING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Session time: {int(minutes)}m {int(seconds)}s")
    logger.info(f"Embeddings created this session: {embedded_count:,}")
    logger.info(f"Average speed: {embedded_count/elapsed_time.total_seconds():.1f} chunks/sec")
    logger.info(f"Output file size: {output_path.stat().st_size / 1024**2:.1f} MB")
    logger.info("")
    
    if correct_dim == total:
        logger.info("✅ SUCCESS: All chunks embedded!")
        logger.info("✓ Ready for Phase 1B: Entity Extraction")
    else:
        logger.warning(f"⚠ INCOMPLETE: {total - correct_dim:,} chunks still need embedding")
        logger.info("Run again to resume from where it stopped:")
        logger.info(f"  python embed_server.py {output_path} {output_path}")
    
    logger.info("=" * 70)

if __name__ == "__main__":
    main()