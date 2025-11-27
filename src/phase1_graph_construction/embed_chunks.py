"""
Chunk Embedding Wrapper - Phase 1A-2
Embeds semantic chunks for RAKG corpus retrospective retrieval

Author: Pau Barba i Colomer
Phase: 1A-2 Chunk Embedding
Input: chunks_text.json
Output: chunks_embedded.json
"""

import sys
import json
import logging
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.embedder import BGEEmbedder
from src.utils.embed_processor import EmbedProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def embed_chunks(input_path: str, output_path: str, 
                device: str = 'cuda', batch_size: int = 8,
                checkpoint_dir: str = None):
    """
    Embed semantic chunks for corpus retrospective retrieval (RAKG Phase 1A-2).
    
    Args:
        input_path: Path to chunks_text.json
        output_path: Path to save chunks_embedded.json
        device: 'cpu' or 'cuda'
        batch_size: Embedding batch size (8 for GPU, 32 for CPU)
        checkpoint_dir: Directory for checkpoints (optional)
    """
    logger.info("=" * 60)
    logger.info("PHASE 1A-2: CHUNK EMBEDDING")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {batch_size}")
    
    # Initialize embedder
    logger.info("\nInitializing BGE-M3 embedder...")
    embedder = BGEEmbedder(device=device)
    processor = EmbedProcessor(embedder, checkpoint_freq=1000)
    
    # Load chunks
    logger.info("\nLoading chunks...")
    chunks = processor.load_items(Path(input_path))
    
    # Process chunks (text is in 'text' key)
    logger.info("\nEmbedding chunks...")
    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else None
    chunks = processor.process_items(
        chunks, 
        text_key='text',  # Chunks have full text in 'text' field
        batch_size=batch_size,
        checkpoint_dir=checkpoint_path
    )
    
    # Verify embeddings
    logger.info("\nVerifying embeddings...")
    stats = processor.verify_embeddings(chunks)
    
    # Save enriched chunks
    logger.info("\nSaving enriched chunks...")
    processor.save_items(chunks, Path(output_path))
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ CHUNK EMBEDDING COMPLETE")
    logger.info(f"✓ {stats['total_items']} chunks embedded")
    logger.info(f"✓ Success rate: {stats['success_rate']:.2f}%")
    logger.info("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Embed semantic chunks (Phase 1A-2)'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to chunks_text.json'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to save chunks_embedded.json'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cpu', 'cuda'],
        help='Device to use for embedding'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for embedding (8 for GPU, 32 for CPU)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        default=None,
        help='Directory for saving checkpoints (optional)'
    )
    
    args = parser.parse_args()
    
    embed_chunks(
        input_path=args.input,
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir
    )
