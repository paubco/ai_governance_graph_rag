"""
Main script to embed all chunks
Run: python scripts/embed_chunks.py
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.phase1_graph_construction.embedder import ChunkEmbedder
from src.phase1_graph_construction.embed_processor import EmbedProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Embed all chunks with checkpoints"""
    
    # Paths
    chunks_path = Path("data/interim/chunks/chunks_text.json")
    output_path = Path("data/interim/chunks/chunks_text.json")  # Overwrite
    checkpoint_dir = Path("data/interim/chunks/checkpoints")
    
    # Initialize
    logger.info("=" * 60)
    logger.info("CHUNK EMBEDDING PIPELINE")
    logger.info("=" * 60)
    
    embedder = ChunkEmbedder(model_name='BAAI/bge-m3')
    processor = EmbedProcessor(embedder, checkpoint_freq=1000)
    
    # Load chunks
    chunks = processor.load_chunks(chunks_path)
    
    # Process with checkpoints
    enriched_chunks = processor.process_chunks(
        chunks,
        batch_size=8,
        checkpoint_dir=checkpoint_dir
    )
    
    # Verify
    stats = processor.verify_embeddings(enriched_chunks)
    
    if stats['success_rate'] < 100:
        logger.warning(f"⚠️  Only {stats['success_rate']:.2f}% success rate!")
    else:
        logger.info("✓ All chunks embedded successfully")
    
    # Save
    processor.save_chunks(enriched_chunks, output_path)
    
    logger.info("=" * 60)
    logger.info("EMBEDDING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()