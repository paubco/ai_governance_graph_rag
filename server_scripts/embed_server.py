"""
BGE-M3 Embedding Script 
Handles both chunks AND entities
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

def detect_input_type(data):
    """
    Detect if input is chunks or entities and return appropriate structure
    
    Returns:
        (input_type, items_dict, text_key)
        - input_type: 'chunks' or 'entities'
        - items_dict: dict of items to embed
        - text_key: field to use for embedding text
    """
    # Case 1: Phase 1B output - nested entities structure
    if isinstance(data, dict) and 'metadata' in data and 'entities' in data:
        logger.info("Detected: Phase 1B entity extraction output (nested structure)")
        
        # Flatten: extract all entities from all chunks
        items_dict = {}
        entity_count = 0
        
        for chunk_data in data['entities']:
            for entity in chunk_data.get('entities', []):
                # Format entity as "name [type]" per RAKG Eq. 19
                formatted_text = f"{entity['name']} [{entity['type']}]"
                
                items_dict[f'entity_{entity_count:06d}'] = {
                    'name': entity['name'],
                    'type': entity['type'],
                    'description': entity.get('description', ''),
                    'chunk_id': entity['chunk_id'],
                    'formatted_text': formatted_text,
                    'text': formatted_text  # For compatibility with embed logic
                }
                entity_count += 1
        
        logger.info(f"✓ Flattened {entity_count:,} entities from nested structure")
        logger.info(f"Sample formatted entity: {items_dict['entity_000000']['formatted_text']}")
        return 'entities', items_dict, 'text'
    
    # Case 2: Already flattened entities (list or dict with name/type)
    elif isinstance(data, (list, dict)):
        sample = data[0] if isinstance(data, list) else next(iter(data.values()))
        
        if 'name' in sample and 'type' in sample:
            logger.info("Detected: Flattened entities")
            
            # Convert to dict if list
            if isinstance(data, list):
                items_dict = {f'entity_{i:06d}': item for i, item in enumerate(data)}
            else:
                items_dict = data
            
            # Add formatted text if not present
            for key, entity in items_dict.items():
                if 'text' not in entity:
                    formatted_text = f"{entity['name']} [{entity['type']}]"
                    entity['formatted_text'] = formatted_text
                    entity['text'] = formatted_text
            
            logger.info(f"✓ Loaded {len(items_dict):,} entities")
            return 'entities', items_dict, 'text'
        
        # Case 3: Chunks with 'text' field
        elif 'text' in sample:
            logger.info("Detected: Text chunks")
            
            # Convert to dict if list
            if isinstance(data, list):
                items_dict = {f'chunk_{i:06d}': item for i, item in enumerate(data)}
            else:
                items_dict = data
            
            logger.info(f"✓ Loaded {len(items_dict):,} chunks")
            return 'chunks', items_dict, 'text'
        
        else:
            raise ValueError(f"Unknown input format. Sample item keys: {list(sample.keys())}")
    
    else:
        raise ValueError(f"Unknown input format. Data type: {type(data)}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python embed_server.py <input.json> <output.json>")
        print()
        print("Supports:")
        print("  - Phase 1A chunks (with 'text' field)")
        print("  - Phase 1B entities (nested structure with metadata)")
        print("  - Flattened entities (with 'name' and 'type' fields)")
        print()
        print("Examples:")
        print("  python embed_server.py chunks_text.json chunks_embedded.json")
        print("  python embed_server.py pre_entities.json pre_entities_embedded.json")
        print("  python embed_server.py pre_entities_embedded.json pre_entities_embedded.json  # Resume")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    logger.info("=" * 70)
    logger.info("UNIVERSAL BGE-M3 EMBEDDING SERVER")
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
    
    # Load and detect input type
    logger.info("-" * 70)
    logger.info(f"Loading from: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    input_type, all_items, text_key = detect_input_type(raw_data)
    
    total_items = len(all_items)
    logger.info(f"Input type: {input_type.upper()}")
    logger.info(f"Total items: {total_items:,}")
    
    # Detect resume vs fresh start
    sample_item = next(iter(all_items.values()))
    is_resume = 'embedding' in sample_item
    
    if is_resume:
        # Count what's already done
        with_embeddings = sum(1 for item in all_items.values() 
                             if 'embedding' in item and len(item.get('embedding', [])) == 1024)
        missing_ids = [iid for iid, item in all_items.items() 
                      if 'embedding' not in item or len(item.get('embedding', [])) != 1024]
        
        logger.info("")
        logger.info("🔄 RESUME MODE DETECTED")
        logger.info(f"  Already embedded: {with_embeddings:,} ({with_embeddings/total_items*100:.1f}%)")
        logger.info(f"  Remaining: {len(missing_ids):,} ({len(missing_ids)/total_items*100:.1f}%)")
        
        if len(missing_ids) == 0:
            logger.info("")
            logger.info("✓ All items already embedded! Nothing to do.")
            sys.exit(0)
        
        item_ids = missing_ids
    else:
        logger.info("")
        logger.info("🆕 FRESH START MODE")
        item_ids = list(all_items.keys())
    
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
    
    # Configuration
    batch_size = 8
    save_interval = 1000
    
    logger.info("-" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Save interval: {save_interval} items")
    logger.info(f"  Precision: fp16 (half)")
    logger.info(f"  Text field: '{text_key}'")
    logger.info("=" * 70)
    logger.info(f"Starting embedding process for {len(item_ids):,} items...")
    logger.info("")
    
    start_time = datetime.now()
    embedded_count = 0
    
    try:
        for i in tqdm(range(0, len(item_ids), batch_size),
                      desc="Embedding batches",
                      unit="batch"):
            
            batch_ids = item_ids[i:i+batch_size]
            
            # Get texts
            batch_texts = []
            for iid in batch_ids:
                text = all_items[iid][text_key]
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
            
            # Add to items
            for item_id, emb in zip(batch_ids, embeddings):
                all_items[item_id]['embedding'] = emb.tolist()
                embedded_count += 1
            
            # Aggressive memory cleanup after each batch
            del embeddings
            del batch_texts
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            # Save checkpoint periodically
            if embedded_count % save_interval == 0:
                logger.info(f"\n💾 Checkpoint: Saving progress ({embedded_count:,}/{len(item_ids):,})...")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(all_items, f, ensure_ascii=False)
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
                remaining = (len(item_ids) - embedded_count) / rate if rate > 0 else 0
                logger.info(f"  Progress: {embedded_count:,}/{len(item_ids):,} "
                           f"({embedded_count/len(item_ids)*100:.1f}%) "
                           f"- ETA: {remaining/60:.1f} min")
    
    except KeyboardInterrupt:
        logger.warning("\n⚠ Interrupted by user!")
        logger.info(f"Embedded {embedded_count:,}/{len(item_ids):,} items in this session")
        logger.info("Saving progress...")
    
    except Exception as e:
        logger.error(f"Error during embedding: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info(f"Embedded {embedded_count:,} items before error")
        logger.info("Saving partial results...")
    
    # Final save
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Saving final results to: {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_items, f, ensure_ascii=False)
    
    logger.info("✓ Final save complete")
    
    # Verification
    logger.info("")
    logger.info("Verification:")
    total = len(all_items)
    with_embeddings = sum(1 for item in all_items.values() if 'embedding' in item)
    correct_dim = sum(1 for item in all_items.values() 
                     if 'embedding' in item and len(item['embedding']) == 1024)
    
    logger.info(f"  Total items: {total:,}")
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
    logger.info(f"Average speed: {embedded_count/elapsed_time.total_seconds():.1f} items/sec")
    logger.info(f"Output file size: {output_path.stat().st_size / 1024**2:.1f} MB")
    logger.info("")
    
    if correct_dim == total:
        logger.info(f"✅ SUCCESS: All {input_type} embedded!")
        if input_type == 'chunks':
            logger.info("✓ Ready for Phase 1B: Entity Extraction")
        else:
            logger.info("✓ Ready for Phase 1C-2: VecJudge Clustering")
    else:
        logger.warning(f"⚠ INCOMPLETE: {total - correct_dim:,} items still need embedding")
        logger.info("Run again to resume:")
        logger.info(f"  python embed_server.py {output_path} {output_path}")
    
    logger.info("=" * 70)

if __name__ == "__main__":
    main()