"""
Chunk Processor for AI Governance GraphRAG Pipeline
Orchestrates document loading, chunking, and storage
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from dataclasses import asdict
import logging

# Add project root to path (adjust based on where this file lives)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.processing.chunking.semantic_chunker import SemanticChunker, Chunk

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChunkProcessor:
    """
    Processes documents through semantic chunking pipeline
    
    Workflow:
    1. Load documents (from Phase 0 DocumentLoader)
    2. Chunk each document (SemanticChunker)
    3. Save chunks to interim storage (JSON)
    4. Generate statistics and logs
    """
    
    def __init__(
        self,
        interim_path: Path,
        threshold: float = 0.7,
        min_sentences: int = 3,
        max_tokens: int = 1500
    ):
        self.interim_path = Path(interim_path)
        self.chunks_dir = self.interim_path / 'chunks'
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize chunker
        self.chunker = SemanticChunker(
            threshold=threshold,
            min_sentences=min_sentences,
            max_tokens=max_tokens
        )
        
        logger.info(f"ChunkProcessor initialized")
        logger.info(f"Output directory: {self.chunks_dir}")
        logger.info(f"Threshold: {threshold}, Min sentences: {min_sentences}, Max tokens: {max_tokens}")
    
    def process_documents(self, documents: List) -> Dict:
        """
        Process all documents through chunking pipeline
        
        Args:
            documents: List of Document objects from DocumentLoader
        
        Returns:
            Processing statistics
        """
        logger.info(f"Processing {len(documents)} documents...")
        
        all_chunks = []
        doc_stats = {}
        
        for i, doc in enumerate(documents, 1):
            logger.info(f"[{i}/{len(documents)}] Chunking: {doc.doc_id}")
            
            try:
                # Chunk document
                chunks = self.chunker.chunk_document(
                    text=doc.text,
                    document_id=doc.doc_id,
                    metadata={
                        'source_type': doc.source_type,
                        'title': doc.title,
                        **doc.metadata
                    }
                )
                
                # Store stats
                doc_stats[doc.doc_id] = {
                    'chunk_count': len(chunks),
                    'original_length': len(doc.text),
                    'source_type': doc.source_type,
                    **self.chunker.get_statistics(chunks)
                }
                
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error processing {doc.doc_id}: {e}")
                continue
        
        # Save chunks and metadata
        logger.info(f"Saving {len(all_chunks)} chunks to disk...")
        self._save_chunks(all_chunks)
        self._save_metadata(doc_stats)
        
        # Generate summary statistics
        summary = self._generate_summary(all_chunks, doc_stats)
        self._save_summary(summary)
        
        logger.info("Chunking complete!")
        return summary
    
    def _save_chunks(self, chunks: List[Chunk]):
        """Save chunks to JSON file"""
        chunks_data = {}
        
        for chunk in chunks:
            chunk_dict = asdict(chunk)
            chunks_data[chunk.chunk_id] = chunk_dict
        
        output_file = self.chunks_dir / 'chunks_text.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved chunks to: {output_file}")
    
    def _save_metadata(self, doc_stats: Dict):
        """Save per-document metadata"""
        output_file = self.chunks_dir / 'chunks_metadata.json'
        
        metadata = {
            'chunking_timestamp': datetime.now().isoformat(),
            'documents': doc_stats
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved metadata to: {output_file}")
    
    def _generate_summary(self, chunks: List[Chunk], doc_stats: Dict) -> Dict:
        """Generate summary statistics"""
        total_tokens = sum(c.token_count for c in chunks)
        total_sentences = sum(c.sentence_count for c in chunks)
        
        # Group by source type
        by_source = {}
        for doc_id, stats in doc_stats.items():
            source_type = stats['source_type']
            if source_type not in by_source:
                by_source[source_type] = {'doc_count': 0, 'chunk_count': 0}
            by_source[source_type]['doc_count'] += 1
            by_source[source_type]['chunk_count'] += stats['chunk_count']
        
        summary = {
            'total_documents': len(doc_stats),
            'total_chunks': len(chunks),
            'total_tokens': total_tokens,
            'total_sentences': total_sentences,
            'avg_chunks_per_doc': len(chunks) / len(doc_stats) if doc_stats else 0,
            'avg_tokens_per_chunk': total_tokens / len(chunks) if chunks else 0,
            'avg_sentences_per_chunk': total_sentences / len(chunks) if chunks else 0,
            'by_source_type': by_source,
            'chunking_timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def _save_summary(self, summary: Dict):
        """Save summary statistics"""
        output_file = self.chunks_dir / 'chunking_summary.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved summary to: {output_file}")
        
        # Also print to console
        print("\n" + "="*60)
        print("CHUNKING SUMMARY")
        print("="*60)
        print(f"Documents processed: {summary['total_documents']}")
        print(f"Total chunks created: {summary['total_chunks']}")
        print(f"Avg chunks per document: {summary['avg_chunks_per_doc']:.1f}")
        print(f"Avg tokens per chunk: {summary['avg_tokens_per_chunk']:.1f}")
        print(f"Avg sentences per chunk: {summary['avg_sentences_per_chunk']:.1f}")
        print(f"\nBy source type:")
        for source_type, stats in summary['by_source_type'].items():
            print(f"  {source_type}: {stats['doc_count']} docs → {stats['chunk_count']} chunks")
        print("="*60 + "\n")
    
    def load_chunks(self) -> Dict:
        """Load previously chunked data"""
        chunks_file = self.chunks_dir / 'chunks_text.json'
        
        if not chunks_file.exists():
            raise FileNotFoundError(f"No chunks found at {chunks_file}")
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        logger.info(f"Loaded {len(chunks_data)} chunks from disk")
        return chunks_data


def main():
    """
    Main execution function for testing
    Run this to test chunking on a small sample
    """
    import sys
    from pathlib import Path
    
    # Import DocumentLoader (adjust path as needed)
    # Assuming you have: src/ingestion/document_loader.py
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root / 'src' / 'ingestion'))
    
    try:
        from src.ingestion.document_loader import DocumentLoader
        
        # Initialize
        loader = DocumentLoader(year='2023')
        processor = ChunkProcessor(
            interim_path=project_root / 'data' / 'interim',
            threshold=0.7
        )
        
        # Load documents
        print("Loading documents...")
        documents = loader.load_all_documents()
        print(f"Loaded {len(documents)} documents")
        
        # TEST MODE: Process only first 5 documents
        test_mode = input("\nTest mode (5 docs only)? [Y/n]: ").strip().lower()
        if test_mode != 'n':
            documents = documents[:5] + documents[-5:]
            print(f"TEST MODE: Processing {len(documents)} documents only")
                
        # Process
        summary = processor.process_documents(documents)
        
        print("\nDone! Check data/interim/chunks/ for output files.")
        
    except ImportError as e:
        print(f"Error importing DocumentLoader: {e}")
        print("Make sure document_loader.py exists in src/ingestion/")
        return
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()