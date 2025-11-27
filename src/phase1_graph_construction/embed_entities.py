"""
Entity Embedding Wrapper - Phase 1C-1
Embeds pre-entities for VecJudge clustering (RAKG disambiguation)

Author: Pau Barba i Colomer
Phase: 1C-1 Pre-Entity Embedding
Input: pre_entities.json
Output: pre_entities_embedded.json
Format: "entity_name [entity_type]" (NOT description)
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


def format_entity_text(entity: dict) -> str:
    """
    Format entity for embedding per RAKG Equation 19.
    
    Format: "entity_name [entity_type]"
    Example: "GDPR [Regulation]"
    
    Args:
        entity: Entity dict with 'name' and 'type' keys
        
    Returns:
        Formatted text string
    """
    return f"{entity['name']} [{entity['type']}]"


def embed_entities(input_path: str, output_path: str,
                  device: str = 'cuda', batch_size: int = 8,
                  checkpoint_dir: str = None):
    """
    Embed pre-entities for VecJudge clustering (RAKG Phase 1C-1).
    
    RAKG Methodology:
    - Embed "name + type" (NOT description) per Equation 19
    - Format: "GDPR [Regulation]", "EU [Organization]"
    - Used for VecJudge similarity clustering (Equation 20)
    
    Args:
        input_path: Path to pre_entities.json (from Phase 1B)
        output_path: Path to save pre_entities_embedded.json
        device: 'cpu' or 'cuda'
        batch_size: Embedding batch size (8 for GPU, 32 for CPU)
        checkpoint_dir: Directory for checkpoints (optional)
    """
    logger.info("=" * 60)
    logger.info("PHASE 1C-1: PRE-ENTITY EMBEDDING")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {batch_size}")
    
    # Initialize embedder
    logger.info("\nInitializing BGE-M3 embedder...")
    embedder = BGEEmbedder(device=device)
    processor = EmbedProcessor(embedder, checkpoint_freq=1000)
    
    # Load entities
    logger.info("\nLoading pre-entities...")
    with open(input_path, 'r', encoding='utf-8') as f:
        entities = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(entities, list):
        logger.info(f"Loaded {len(entities)} entities (list format)")
        entity_dict = {i: entity for i, entity in enumerate(entities)}
    else:
        logger.info(f"Loaded {len(entities)} entities (dict format)")
        entity_dict = entities
    
    # Format entity texts: "name [type]" per RAKG Eq. 19
    logger.info("\nFormatting entity texts...")
    for entity_id, entity in entity_dict.items():
        formatted = format_entity_text(entity)
        entity_dict[entity_id]['formatted_text'] = formatted
    
    logger.info(f"Example formatted entities:")
    for i, (entity_id, entity) in enumerate(list(entity_dict.items())[:3]):
        logger.info(f"  {i+1}. {entity['formatted_text']}")
    
    # Process entities
    logger.info("\nEmbedding entities...")
    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else None
    entity_dict = processor.process_items(
        entity_dict,
        text_key='formatted_text',  # Use formatted "name [type]"
        batch_size=batch_size,
        checkpoint_dir=checkpoint_path
    )
    
    # Verify embeddings
    logger.info("\nVerifying embeddings...")
    stats = processor.verify_embeddings(entity_dict)
    
    # Convert back to list if input was list
    if isinstance(entities, list):
        logger.info("\nConverting back to list format...")
        result = [entity_dict[i] for i in sorted(entity_dict.keys())]
    else:
        result = entity_dict
    
    # Save enriched entities
    logger.info("\nSaving enriched entities...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ ENTITY EMBEDDING COMPLETE")
    logger.info(f"✓ {stats['total_items']} entities embedded")
    logger.info(f"✓ Success rate: {stats['success_rate']:.2f}%")
    logger.info("✓ Ready for VecJudge clustering (Phase 1C-2)")
    logger.info("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Embed pre-entities for VecJudge clustering (Phase 1C-1)'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to pre_entities.json'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to save pre_entities_embedded.json'
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
    
    embed_entities(
        input_path=args.input,
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir
    )
