"""
Phase 1B: Entity Processor

Orchestrates entity extraction across all chunks with:
- Incremental saving (every 1000 chunks)
- Progress tracking
- Error recovery
- Statistics reporting
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class EntityProcessor:
    """
    Orchestrates entity extraction from all chunks.
    
    Features:
    - Loads chunks from chunks_text.json
    - Processes in batches with incremental saves
    - Tracks progress and statistics
    - Saves to data/interim/entities/pre_entities.json
    """
    
    def __init__(
        self,
        chunks_file: str = "data/interim/chunks/chunks_text.json",
        output_file: str = "data/interim/entities/pre_entities.json",
        checkpoint_interval: int = 1000
    ):
        """
        Initialize the processor.
        
        Args:
            chunks_file: Path to input chunks file
            output_file: Path to output entities file
            checkpoint_interval: Save every N chunks
        """
        self.chunks_file = chunks_file
        self.output_file = output_file
        self.checkpoint_interval = checkpoint_interval
        
        # Ensure output directory exists
        output_dir = Path(output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_chunks(self) -> List[Dict]:
        """
        Load chunks from JSON file.
        
        Returns:
            List of chunk dictionaries with 'chunk_id' and 'text'
        """
        print(f"Loading chunks from {self.chunks_file}...")
        
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Format: {chunk_id: {chunk_data}}
        # Convert dict values to list
        chunks = list(data.values())
        
        print(f"✅ Loaded {len(chunks)} chunks")
        return chunks
    
    def process_all(
        self,
        extractor,
        start_index: int = 0,
        limit: Optional[int] = None
    ):
        """
        Process all chunks with entity extraction.
        
        Args:
            extractor: RAKGEntityExtractor instance
            start_index: Start from this chunk index (for resuming)
            limit: Process only this many chunks (for testing)
        """
        # Load chunks
        chunks = self.load_chunks()
        
        # Apply limit if specified
        if limit:
            chunks = chunks[start_index:start_index + limit]
            print(f"Processing chunks {start_index} to {start_index + len(chunks)} (limit={limit})")
        else:
            chunks = chunks[start_index:]
            print(f"Processing {len(chunks)} chunks starting from index {start_index}")
        
        # Track results
        all_entities = []
        chunks_processed = 0
        total_entities_extracted = 0
        
        # Track statistics
        stats = {
            'start_time': datetime.now().isoformat(),
            'total_chunks': len(chunks),
            'chunks_processed': 0,
            'total_entities': 0,
            'errors': 0
        }
        
        print("\nStarting entity extraction...\n")
        
        # Process chunks
        for i, chunk in enumerate(chunks):
            chunk_id = chunk['chunk_id']
            chunk_text = chunk['text']
            
            # Extract entities
            try:
                entities = extractor.extract_entities(chunk_text, chunk_id)
                
                # Add to results
                for entity in entities:
                    all_entities.append(entity)
                
                chunks_processed += 1
                total_entities_extracted += len(entities)
                
                # Progress update
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(chunks)} chunks | "
                          f"Entities: {total_entities_extracted} | "
                          f"Avg: {total_entities_extracted/(i+1):.1f} per chunk")
                
                # Checkpoint save
                if (i + 1) % self.checkpoint_interval == 0:
                    self._save_checkpoint(all_entities, chunks_processed, total_entities_extracted)
            
            except Exception as e:
                print(f"Error processing chunk {chunk_id}: {e}")
                stats['errors'] += 1
                continue
        
        # Final save
        stats['chunks_processed'] = chunks_processed
        stats['total_entities'] = total_entities_extracted
        stats['end_time'] = datetime.now().isoformat()
        
        self._save_final(all_entities, stats)
        
        print("\n" + "="*60)
        print("ENTITY EXTRACTION COMPLETE")
        print("="*60)
        print(f"Chunks processed: {chunks_processed}")
        print(f"Total entities: {total_entities_extracted}")
        print(f"Average per chunk: {total_entities_extracted/chunks_processed:.1f}")
        print(f"Errors: {stats['errors']}")
        print(f"Output: {self.output_file}")
        print("="*60)
        
        return all_entities, stats
    
    def _save_checkpoint(
        self,
        entities: List[Dict],
        chunks_processed: int,
        total_entities: int
    ):
        """Save intermediate checkpoint."""
        checkpoint_file = self.output_file.replace('.json', f'_checkpoint_{chunks_processed}.json')
        
        data = {
            'metadata': {
                'checkpoint': True,
                'chunks_processed': chunks_processed,
                'total_entities': total_entities,
                'timestamp': datetime.now().isoformat()
            },
            'entities': entities
        }
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[CHECKPOINT] Saved {total_entities} entities to {checkpoint_file}\n")
    
    def _save_final(
        self,
        entities: List[Dict],
        stats: Dict
    ):
        """Save final results."""
        data = {
            'metadata': {
                'final': True,
                'statistics': stats,
                'entity_count': len(entities)
            },
            'entities': entities
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[FINAL] Saved {len(entities)} entities to {self.output_file}")


def main():
    """
    Main execution function.
    
    Usage:
        python entity_processor.py [--limit N] [--start INDEX]
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract entities from chunks')
    parser.add_argument('--limit', type=int, help='Process only N chunks (for testing)')
    parser.add_argument('--start', type=int, default=0, help='Start from chunk index')
    parser.add_argument('--api-key', type=str, help='Together.ai API key')
    
    args = parser.parse_args()
    
    # Get API key (from args or .env)
    api_key = args.api_key
    if not api_key:
        try:
            from dotenv import load_dotenv
            import os
            load_dotenv()
            api_key = os.getenv('TOGETHER_API_KEY')
        except ImportError:
            pass
    
    if not api_key:
        print("❌ Error: No API key!")
        print("   Use --api-key YOUR_KEY or set TOGETHER_API_KEY in .env")
        return None, None
    
    # Initialize extractor
    from entity_extractor import RAKGEntityExtractor
    
    print("Initializing RAKGEntityExtractor...")
    extractor = RAKGEntityExtractor(api_key=args.api_key)
    
    # Initialize processor
    processor = EntityProcessor()
    
    # Process chunks
    entities, stats = processor.process_all(
        extractor=extractor,
        start_index=args.start,
        limit=args.limit
    )
    
    return entities, stats


if __name__ == "__main__":
    main()