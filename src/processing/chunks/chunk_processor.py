# -*- coding: utf-8 -*-
"""
Module: chunk_processor.py
Package: src.processing.chunks
Purpose: Orchestrate document chunking pipeline with mode-aware output

Author: Pau Barba i Colomer
Created: 2025-12-18
Modified: 2025-12-18

References:
    - See ARCHITECTURE.md § 3.1.1 for Phase 1A design
    - See CONTRIBUTING.md § 2 for foundation module usage
    - See extraction_config.py CHUNKING_CONFIG for parameters
"""

# Standard library
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
import numpy as np
from tqdm import tqdm

# Foundation
from src.utils.dataclasses import Chunk, EmbeddedChunk
from src.utils.io import load_jsonl, save_jsonl, save_json

# Config
from config.extraction_config import CHUNKING_CONFIG

# Local
from src.processing.chunks.semantic_chunker import SemanticChunker
from src.processing.chunks.chunk_deduplicator import ChunkDeduplicator

logger = logging.getLogger(__name__)

# Load defaults from config
DEFAULT_THRESHOLD = CHUNKING_CONFIG.get('similarity_threshold', 0.45)
DEFAULT_MIN_SENTENCES = CHUNKING_CONFIG.get('min_sentences', 3)
DEFAULT_MAX_TOKENS = CHUNKING_CONFIG.get('max_tokens', 1500)
DEFAULT_MIN_COHERENCE = CHUNKING_CONFIG.get('min_coherence', 0.30)
DEFAULT_MIN_TOKENS = CHUNKING_CONFIG.get('min_tokens', 15)
DEFAULT_MIN_TOKENS_PER_SENTENCE = CHUNKING_CONFIG.get('min_tokens_per_sentence', 10)
DEFAULT_DEDUP_THRESHOLD = CHUNKING_CONFIG.get('dedup_threshold', 0.95)


class ChunkProcessor:
    """
    Orchestrates document chunking pipeline.
    
    Modes:
        - local: Chunking only (BGE-small), outputs chunks.jsonl
        - server: Chunking + BGE-M3 embedding + dedup, outputs chunks_embedded.jsonl
    
    Parameters loaded from CHUNKING_CONFIG (extraction_config.py):
        threshold: Sentence similarity threshold for boundaries
        min_sentences: Minimum sentences per chunk
        max_tokens: Maximum tokens per chunk
        min_coherence: Minimum coherence score to keep chunk
        min_tokens: Minimum tokens for single-sentence chunks
        min_tokens_per_sentence: Below this + no header = garbage
        dedup_threshold: Cosine similarity for deduplication (server mode)
    """
    
    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        min_sentences: int = DEFAULT_MIN_SENTENCES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        min_coherence: float = DEFAULT_MIN_COHERENCE,
        min_tokens: int = DEFAULT_MIN_TOKENS,
        min_tokens_per_sentence: float = DEFAULT_MIN_TOKENS_PER_SENTENCE,
        mode: str = 'local',
        dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD
    ):
        self.threshold = threshold
        self.min_coherence = min_coherence
        self.min_tokens = min_tokens
        self.min_tokens_per_sentence = min_tokens_per_sentence
        self.mode = mode
        self.dedup_threshold = dedup_threshold
        
        # Paths
        self.data_root = PROJECT_ROOT / 'data'
        self.input_path = self.data_root / 'interim' / 'preprocessed' / 'documents_cleaned.jsonl'
        self.output_dir = self.data_root / 'processed' / 'chunks'
        self.report_dir = self.data_root / 'interim' / 'chunks'
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize chunker
        self.chunker = SemanticChunker(
            threshold=threshold,
            min_sentences=min_sentences,
            max_tokens=max_tokens,
            min_coherence=min_coherence,
            min_tokens=min_tokens,
            min_tokens_per_sentence=min_tokens_per_sentence
        )
        
        # Server mode: load embedder and deduplicator
        self.embedder = None
        self.deduplicator = None
        if mode == 'server':
            from src.utils.embedder import BGEEmbedder
            logger.info("Loading BGE-M3 embedder for server mode...")
            self.embedder = BGEEmbedder()
            self.deduplicator = ChunkDeduplicator(threshold=dedup_threshold)
        
        logger.info(
            f"ChunkProcessor initialized: mode={mode}, threshold={threshold}, "
            f"min_coherence={min_coherence}, min_tokens={min_tokens}, "
            f"min_tokens_per_sentence={min_tokens_per_sentence}, "
            f"dedup_threshold={dedup_threshold}"
        )
    
    def load_documents(
        self,
        sample: int = 0,
        seed: Optional[int] = None
    ) -> List[Dict]:
        """
        Load documents from Phase 0B output.
        
        Args:
            sample: Number of documents to sample (0 = all)
            seed: Random seed for reproducible sampling
            
        Returns:
            List of document dicts
        """
        logger.info(f"Loading documents from {self.input_path}")
        documents = load_jsonl(self.input_path)
        
        if sample > 0 and sample < len(documents):
            if seed is not None:
                np.random.seed(seed)
            indices = np.random.choice(len(documents), size=sample, replace=False)
            documents = [documents[i] for i in sorted(indices)]
            logger.info(f"Sampled {sample} documents (seed={seed})")
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def process(
        self,
        sample: int = 0,
        seed: Optional[int] = None
    ) -> Dict:
        """
        Run full chunking pipeline.
        
        Args:
            sample: Number of documents to sample (0 = all)
            seed: Random seed for reproducible sampling
            
        Returns:
            Processing report dict
        """
        # Load documents
        documents = self.load_documents(sample=sample, seed=seed)
        
        # Chunk all documents
        all_chunks = []
        all_discards = []
        doc_stats = {}
        
        for doc in tqdm(documents, desc="Chunking documents"):
            doc_id = doc['doc_id']
            text = doc['text']
            source_type = doc.get('source_type', 'regulation')
            
            # Build metadata from Phase 0B fields
            metadata = {
                'source_type': source_type,
                'title': doc.get('title', ''),
                'was_translated': doc.get('was_translated', False),
                'original_language': doc.get('original_language', 'en'),
            }
            # Merge any additional metadata
            if 'metadata' in doc:
                metadata.update(doc['metadata'])
            
            try:
                chunks, discards = self.chunker.chunk_document(
                    text=text,
                    document_id=doc_id,
                    source_type=source_type,
                    metadata=metadata
                )
                
                doc_stats[doc_id] = {
                    'chunk_count': len(chunks),
                    'discard_count': len(discards),
                    'source_type': source_type,
                    **self.chunker.get_statistics(chunks)
                }
                
                all_chunks.extend(chunks)
                all_discards.extend(discards)
                
            except Exception as e:
                logger.error(f"Error chunking {doc_id}: {e}")
                continue
        
        logger.info(
            f"Created {len(all_chunks)} chunks from {len(documents)} documents "
            f"({len(all_discards)} discarded)"
        )
        
        # Server mode: embed and deduplicate
        dedup_stats = {}
        if self.mode == 'server':
            all_chunks, dedup_stats = self._embed_and_deduplicate(all_chunks)
        
        # Save outputs
        self._save_chunks(all_chunks)
        self._save_discards(all_discards)
        
        # Generate and save report
        report = self._generate_report(
            all_chunks, all_discards, doc_stats, dedup_stats, sample, seed
        )
        self._save_report(report)
        
        return report
    
    def _embed_and_deduplicate(
        self,
        chunks: List[Chunk]
    ) -> tuple:
        """
        Embed chunks with BGE-M3 and deduplicate.
        
        Args:
            chunks: List of Chunk objects
            
        Returns:
            Tuple of (embedded_chunks, dedup_stats)
        """
        logger.info(f"Embedding {len(chunks)} chunks with BGE-M3...")
        
        # Extract texts and embed
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_batch(texts, batch_size=32)
        
        # Convert to EmbeddedChunk
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded = EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                position=chunk.position,
                sentence_count=chunk.sentence_count,
                token_count=chunk.token_count,
                section_header=chunk.section_header,
                metadata=chunk.metadata,
                embedding=embedding
            )
            embedded_chunks.append(embedded)
        
        # Deduplicate
        logger.info("Running deduplication...")
        filtered_chunks, dedup_stats = self.deduplicator.deduplicate(embedded_chunks)
        
        logger.info(
            f"Deduplication: {len(embedded_chunks)} -> {len(filtered_chunks)} "
            f"({dedup_stats['duplicates_removed']} duplicates removed)"
        )
        
        return filtered_chunks, dedup_stats
    
    def _save_chunks(self, chunks: List) -> None:
        """Save chunks to appropriate output file."""
        if self.mode == 'server':
            output_path = self.output_dir / 'chunks_embedded.jsonl'
        else:
            output_path = self.output_dir / 'chunks.jsonl'
        
        # Convert to dicts for serialization
        chunk_dicts = []
        for chunk in chunks:
            chunk_dict = asdict(chunk)
            # Convert numpy array to list for JSON
            if 'embedding' in chunk_dict and chunk_dict['embedding'] is not None:
                chunk_dict['embedding'] = chunk_dict['embedding'].tolist()
            chunk_dicts.append(chunk_dict)
        
        save_jsonl(chunk_dicts, output_path)
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")
    
    def _save_discards(self, discards: List[Dict]) -> None:
        """Save discarded chunks for inspection."""
        output_path = self.report_dir / 'discarded_chunks.jsonl'
        save_jsonl(discards, output_path)
        logger.info(f"Saved {len(discards)} discarded chunks to {output_path}")
    
    def _generate_report(
        self,
        chunks: List,
        discards: List[Dict],
        doc_stats: Dict,
        dedup_stats: Dict,
        sample: int,
        seed: Optional[int]
    ) -> Dict:
        """Generate chunking report."""
        token_counts = [c.token_count for c in chunks]
        sentence_counts = [c.sentence_count for c in chunks]
        coherence_scores = [c.metadata.get('coherence', 0) for c in chunks]
        
        # Group by source type
        by_source = {}
        for doc_id, stats in doc_stats.items():
            source_type = stats['source_type']
            if source_type not in by_source:
                by_source[source_type] = {'doc_count': 0, 'chunk_count': 0, 'discard_count': 0}
            by_source[source_type]['doc_count'] += 1
            by_source[source_type]['chunk_count'] += stats['chunk_count']
            by_source[source_type]['discard_count'] += stats['discard_count']
        
        # Group discards by reason
        discard_reasons = {}
        for d in discards:
            reason = d['reason'].split(' (')[0]  # Get base reason without details
            discard_reasons[reason] = discard_reasons.get(reason, 0) + 1
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'mode': self.mode,
            'parameters': {
                'threshold': self.threshold,
                'min_coherence': self.min_coherence,
                'dedup_threshold': self.dedup_threshold if self.mode == 'server' else None,
                'sample': sample,
                'seed': seed,
            },
            'summary': {
                'total_documents': len(doc_stats),
                'total_chunks_kept': len(chunks),
                'total_chunks_discarded': len(discards),
                'discard_rate': len(discards) / (len(chunks) + len(discards)) if chunks or discards else 0,
                'chunks_per_doc': {
                    'mean': float(np.mean([s['chunk_count'] for s in doc_stats.values()])),
                    'std': float(np.std([s['chunk_count'] for s in doc_stats.values()])),
                },
            },
            'tokens_per_chunk': {
                'mean': float(np.mean(token_counts)) if token_counts else 0,
                'median': float(np.median(token_counts)) if token_counts else 0,
                'std': float(np.std(token_counts)) if token_counts else 0,
                'min': int(min(token_counts)) if token_counts else 0,
                'max': int(max(token_counts)) if token_counts else 0,
            },
            'sentences_per_chunk': {
                'mean': float(np.mean(sentence_counts)) if sentence_counts else 0,
                'median': float(np.median(sentence_counts)) if sentence_counts else 0,
                'std': float(np.std(sentence_counts)) if sentence_counts else 0,
            },
            'coherence': {
                'mean': float(np.mean(coherence_scores)) if coherence_scores else 0,
                'median': float(np.median(coherence_scores)) if coherence_scores else 0,
                'min': float(min(coherence_scores)) if coherence_scores else 0,
                'max': float(max(coherence_scores)) if coherence_scores else 0,
            },
            'by_source_type': by_source,
            'discard_reasons': discard_reasons,
        }
        
        # Add dedup stats if server mode
        if dedup_stats:
            report['deduplication'] = dedup_stats
        
        return report
    
    def _save_report(self, report: Dict) -> None:
        """Save chunking report."""
        output_path = self.report_dir / 'chunking_report.json'
        save_json(report, output_path)
        logger.info(f"Saved report to {output_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("CHUNKING REPORT")
        print("=" * 60)
        print(f"Mode: {report['mode']}")
        print(f"Threshold: {report['parameters']['threshold']}")
        print(f"Min coherence: {report['parameters']['min_coherence']}")
        print(f"Documents: {report['summary']['total_documents']}")
        print(f"Chunks kept: {report['summary']['total_chunks_kept']}")
        print(f"Chunks discarded: {report['summary']['total_chunks_discarded']}")
        print(f"Discard rate: {report['summary']['discard_rate']:.1%}")
        print(f"Avg tokens/chunk: {report['tokens_per_chunk']['mean']:.1f}")
        print(f"Avg coherence: {report['coherence']['mean']:.3f}")
        print(f"\nBy source type:")
        for source_type, stats in report['by_source_type'].items():
            print(f"  {source_type}: {stats['doc_count']} docs -> {stats['chunk_count']} kept, {stats['discard_count']} discarded")
        print(f"\nDiscard reasons:")
        for reason, count in report['discard_reasons'].items():
            print(f"  {reason}: {count}")
        if 'deduplication' in report:
            print(f"\nDeduplication: {report['deduplication']['duplicates_removed']} duplicates removed")
        print("=" * 60 + "\n")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run Phase 1A semantic chunking pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Defaults from CHUNKING_CONFIG:
    threshold:              {DEFAULT_THRESHOLD}
    min-sentences:          {DEFAULT_MIN_SENTENCES}
    max-tokens:             {DEFAULT_MAX_TOKENS}
    min-coherence:          {DEFAULT_MIN_COHERENCE}
    min-tokens:             {DEFAULT_MIN_TOKENS}
    min-tokens-per-sentence:{DEFAULT_MIN_TOKENS_PER_SENTENCE}
    dedup-threshold:        {DEFAULT_DEDUP_THRESHOLD}

Examples:
    # Local mode (chunking only)
    python -m src.processing.chunks.chunk_processor --mode local
    
    # Sample for testing
    python -m src.processing.chunks.chunk_processor --mode local --sample 10 --seed 42
    
    # Server mode (with BGE-M3 embedding + dedup)
    python -m src.processing.chunks.chunk_processor --mode server
        """
    )
    
    parser.add_argument(
        '--mode', type=str, choices=['local', 'server'], default='local',
        help='Processing mode: local (chunking only) or server (with embedding + dedup)'
    )
    parser.add_argument(
        '--threshold', type=float, default=DEFAULT_THRESHOLD,
        help=f'Sentence similarity threshold for boundaries (default: {DEFAULT_THRESHOLD})'
    )
    parser.add_argument(
        '--min-sentences', type=int, default=DEFAULT_MIN_SENTENCES,
        help=f'Minimum sentences per chunk (default: {DEFAULT_MIN_SENTENCES})'
    )
    parser.add_argument(
        '--max-tokens', type=int, default=DEFAULT_MAX_TOKENS,
        help=f'Maximum tokens per chunk (default: {DEFAULT_MAX_TOKENS})'
    )
    parser.add_argument(
        '--min-coherence', type=float, default=DEFAULT_MIN_COHERENCE,
        help=f'Minimum coherence score (default: {DEFAULT_MIN_COHERENCE})'
    )
    parser.add_argument(
        '--min-tokens', type=int, default=DEFAULT_MIN_TOKENS,
        help=f'Minimum tokens for single-sentence chunks (default: {DEFAULT_MIN_TOKENS})'
    )
    parser.add_argument(
        '--min-tokens-per-sentence', type=float, default=DEFAULT_MIN_TOKENS_PER_SENTENCE,
        help=f'Below this + no header = garbage (default: {DEFAULT_MIN_TOKENS_PER_SENTENCE})'
    )
    parser.add_argument(
        '--dedup-threshold', type=float, default=DEFAULT_DEDUP_THRESHOLD,
        help=f'Deduplication threshold (server mode, default: {DEFAULT_DEDUP_THRESHOLD})'
    )
    parser.add_argument(
        '--sample', type=int, default=0,
        help='Number of documents to sample (0 = all)'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducible sampling'
    )
    parser.add_argument(
        '--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    import random
    from src.utils.logger import setup_logging
    
    args = parse_args()
    
    # Setup logging
    setup_logging(level=getattr(logging, args.log_level))
    
    # Set seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    logger.info("=" * 60)
    logger.info("PHASE 1A: SEMANTIC CHUNKING")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Min sentences: {args.min_sentences}")
    logger.info(f"Min coherence: {args.min_coherence}")
    logger.info(f"Min tokens/sentence: {args.min_tokens_per_sentence}")
    logger.info(f"Sample: {args.sample if args.sample > 0 else 'all'}")
    logger.info("=" * 60)
    
    # Initialize processor
    processor = ChunkProcessor(
        threshold=args.threshold,
        min_sentences=args.min_sentences,
        max_tokens=args.max_tokens,
        min_coherence=args.min_coherence,
        min_tokens=args.min_tokens,
        min_tokens_per_sentence=args.min_tokens_per_sentence,
        mode=args.mode,
        dedup_threshold=args.dedup_threshold
    )
    
    # Run pipeline
    report = processor.process(
        sample_size=args.sample if args.sample > 0 else None,
        random_seed=args.seed
    )
    
    logger.info("Pipeline completed successfully")
    return report


if __name__ == '__main__':
    main()