# -*- coding: utf-8 -*-
"""
Semantic chunking with sentence-similarity boundaries and coherence filtering

RAKG-style semantic chunker with hierarchical boundaries using BGE-small model for
sentence similarity during boundary detection. Splits documents into sections at
headers with source-type awareness (regulation vs paper), uses sentence similarity
for soft boundaries within sections, and filters chunks by coherence score and
token density to remove low-quality content.

References:
RAKG methodology: Hierarchical chunking approach
    ARCHITECTURE.md: Section 3.1.1 for design context
    extraction_config.py: CHUNKING_CONFIG for parameters

"""
# Standard library
import logging
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Foundation
from src.utils.dataclasses import Chunk
from src.utils.id_generator import generate_chunk_id

# Config
from config.extraction_config import CHUNKING_CONFIG

# Ensure NLTK data available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

logger = logging.getLogger(__name__)


# Load from config (with fallbacks)
HEADER_PATTERNS = CHUNKING_CONFIG.get('header_patterns', {
    'regulation': [
        r'^#{1,6}\s+.+$',
        r'^Article\s+\d+',
        r'^Section\s+\d+',
        r'^\d+\.\s+[A-Z]',
        r'^[A-Z][A-Z\s]{3,}$'
    ],
    'paper': [
        r'^#\s+\d+(?:\.\d+)*\.?\s+.+$',
        r'^#\s+(?:Introduction|Conclusion|Discussion|Results|Methods|Methodology|Background|Literature|Related\s+Work|Theoretical|Empirical|Analysis|Findings|Implications|Limitations|Future).*$',
    ]
})

GARBAGE_HEADERS = CHUNKING_CONFIG.get('garbage_headers', {
    'ARTICLEINFO', 'ARTICLE INFO', 'KEYWORDS', 'ORCID', 'OPEN ACCESS',
    'ACKNOWLEDGMENTS', 'ACKNOWLEDGEMENTS', 'Acknowledgments', 'Acknowledgements',
    'Funding', 'Author contributions', 'Correspondence', 'Data availability',
    'Data availability statement', 'Conflict of interest', 'Competing interests',
    'Declaration of competing interest', 'Declarations', 'Disclosure statement',
    "Publisher's note", "Publisher's Note", 'CCS CONCEPTS', 'References',
    'REFERENCES', 'Bibliography'
})

# Default values from config
DEFAULT_THRESHOLD = CHUNKING_CONFIG.get('similarity_threshold', 0.45)
DEFAULT_MIN_SENTENCES = CHUNKING_CONFIG.get('min_sentences', 3)
DEFAULT_MAX_TOKENS = CHUNKING_CONFIG.get('max_tokens', 1500)
DEFAULT_MIN_COHERENCE = CHUNKING_CONFIG.get('min_coherence', 0.30)
DEFAULT_MIN_TOKENS = CHUNKING_CONFIG.get('min_tokens', 15)
DEFAULT_MIN_TOKENS_PER_SENTENCE = CHUNKING_CONFIG.get('min_tokens_per_sentence', 10)
DEFAULT_MODEL = CHUNKING_CONFIG.get('boundary_model', 'BAAI/bge-small-en-v1.5')


class SemanticChunker:
    """
    RAKG-style semantic chunker with hierarchical boundaries and coherence filtering.
    
    Uses BGE-small model for sentence similarity during boundary detection
    (same family as BGE-M3 used for final embeddings).
    
    Algorithm:
        1. Split document into sections at headers (source-type aware)
        2. Within each section, use sentence similarity for soft boundaries
        3. Sentences are atomic - never split mid-sentence
        4. Filter chunks by coherence score and density (tokens/sentence)
    
    Parameters loaded from CHUNKING_CONFIG (extraction_config.py):
        threshold: Cosine similarity threshold for boundaries (default 0.45)
        min_sentences: Minimum sentences per chunk (default 3)
        max_tokens: Maximum tokens per chunk (default 1500)
        min_coherence: Minimum coherence score to keep chunk (default 0.30)
        min_tokens_per_sentence: Below this + no header = garbage (default 10)
        model_name: Sentence transformer for boundary detection (bge-small)
    """
    
    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        min_sentences: int = DEFAULT_MIN_SENTENCES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        min_coherence: float = DEFAULT_MIN_COHERENCE,
        min_tokens: int = DEFAULT_MIN_TOKENS,
        min_tokens_per_sentence: float = DEFAULT_MIN_TOKENS_PER_SENTENCE,
        model_name: str = DEFAULT_MODEL
    ):
        self.threshold = threshold
        self.min_sentences = min_sentences
        self.max_tokens = max_tokens
        self.min_coherence = min_coherence
        self.min_tokens = min_tokens
        self.min_tokens_per_sentence = min_tokens_per_sentence
        
        logger.info(f"Loading boundary detection model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        logger.info(
            f"SemanticChunker initialized: threshold={threshold}, "
            f"min_sentences={min_sentences}, max_tokens={max_tokens}, "
            f"min_coherence={min_coherence}, min_tokens={min_tokens}, "
            f"min_tokens_per_sentence={min_tokens_per_sentence}"
        )
    
    def is_header(self, line: str, source_type: str = 'regulation') -> bool:
        """
        Check if a line is a structural header.
        
        Args:
            line: Text line to check
            source_type: 'regulation' or 'paper' (different patterns)
        """
        line = line.strip()
        if not line:
            return False
        
        patterns = HEADER_PATTERNS.get(source_type, HEADER_PATTERNS['regulation'])
        
        for pattern in patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        return False
    
    def is_garbage_header(self, header: str) -> bool:
        """Check if header indicates garbage section (metadata, refs, etc.)."""
        if not header:
            return False
        
        # Strip markdown and whitespace
        clean = re.sub(r'^#+\s*', '', header).strip()
        
        # Check exact matches
        if clean in GARBAGE_HEADERS:
            return True
        
        # Check partial matches (case-insensitive)
        clean_upper = clean.upper()
        for garbage in GARBAGE_HEADERS:
            if garbage.upper() in clean_upper:
                return True
        
        return False
    
    def coherence_score(self, text: str) -> float:
        """
        Calculate semantic coherence of text as average adjacent sentence similarity.
        
        Args:
            text: Chunk text
            
        Returns:
            Float 0-1, higher = more coherent. Returns 0 for single sentences.
        """
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) < 2:
            return 0.0  # Can't measure coherence of single sentence
        
        embeddings = self.model.encode(sentences, show_progress_bar=False)
        
        similarities = []
        for i in range(len(sentences) - 1):
            sim = cosine_similarity(
                embeddings[i:i+1],
                embeddings[i+1:i+2]
            )[0][0]
            similarities.append(sim)
        
        return float(np.mean(similarities))
    
    def split_into_sections(self, text: str, source_type: str = 'regulation') -> List[Dict]:
        """
        Split document into sections at header boundaries.
        
        Args:
            text: Full document text
            source_type: 'regulation' or 'paper'
            
        Returns:
            List of {'header': str|None, 'content': str}
        """
        lines = text.split('\n')
        sections = []
        current_header = None
        current_content = []
        
        for line in lines:
            if self.is_header(line, source_type):
                # Save previous section
                if current_content:
                    sections.append({
                        'header': current_header,
                        'content': '\n'.join(current_content).strip()
                    })
                # Start new section
                current_header = line.strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Add final section
        if current_content:
            sections.append({
                'header': current_header,
                'content': '\n'.join(current_content).strip()
            })
        
        return sections
    
    def chunk_section_by_similarity(self, text: str) -> List[str]:
        """
        Chunk a section using sentence similarity.
        
        Args:
            text: Section text (no headers)
            
        Returns:
            List of chunk texts
        """
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) <= self.min_sentences:
            return [text] if text.strip() else []
        
        # Encode all sentences (fast with MiniLM)
        embeddings = self.model.encode(sentences, show_progress_bar=False)
        
        # Build chunks based on similarity
        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_tokens = len(sentences[0].split())
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_tokens = len(sentence.split())
            
            # Similarity with previous sentence
            similarity = cosine_similarity(
                embeddings[i:i+1],
                embeddings[i-1:i]
            )[0][0]
            
            # Decision: continue or break?
            would_exceed_max = (current_tokens + sentence_tokens) > self.max_tokens
            below_threshold = similarity < self.threshold
            meets_min = len(current_chunk_sentences) >= self.min_sentences
            
            if would_exceed_max or (below_threshold and meets_min):
                # Finalize current chunk
                chunks.append(' '.join(current_chunk_sentences))
                current_chunk_sentences = [sentence]
                current_tokens = sentence_tokens
            else:
                # Add to current chunk
                current_chunk_sentences.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk_sentences:
            chunks.append(' '.join(current_chunk_sentences))
        
        return chunks
    
    def chunk_document(
        self,
        text: str,
        document_id: str,
        source_type: str = 'regulation',
        metadata: Optional[Dict] = None
    ) -> Tuple[List[Chunk], List[Dict]]:
        """
        Chunk entire document with hierarchical approach and coherence filtering.
        
        Args:
            text: Full document text
            document_id: Unique document identifier
            source_type: 'regulation' or 'paper'
            metadata: Additional metadata to propagate to chunks
            
        Returns:
            Tuple of (kept_chunks, discarded_chunks)
            - kept_chunks: List of Chunk dataclass instances
            - discarded_chunks: List of dicts with chunk info and discard reason
        """
        if metadata is None:
            metadata = {}
        
        # Step 1: Split at headers
        sections = self.split_into_sections(text, source_type)
        logger.debug(f"Document {document_id}: {len(sections)} sections found")
        
        # Step 2: Chunk each section by similarity
        kept_chunks = []
        discarded_chunks = []
        chunk_position = 0
        
        for section in sections:
            section_header = section['header']
            section_content = section['content']
            
            if not section_content.strip():
                continue
            
            # Skip garbage sections entirely
            if self.is_garbage_header(section_header):
                discarded_chunks.append({
                    'document_id': document_id,
                    'text': section_content[:500],  # Truncate for logging
                    'section_header': section_header,
                    'reason': 'garbage_header',
                    'token_count': len(section_content.split()),
                })
                continue
            
            section_chunks = self.chunk_section_by_similarity(section_content)
            
            for chunk_text in section_chunks:
                if not chunk_text.strip():
                    continue
                
                sentences = nltk.sent_tokenize(chunk_text)
                tokens = chunk_text.split()
                token_count = len(tokens)
                sentence_count = len(sentences)
                
                # Calculate coherence
                coherence = self.coherence_score(chunk_text)
                
                # Decide keep or discard
                discard_reason = None
                
                # Calculate density (tokens per sentence)
                tokens_per_sentence = token_count / sentence_count if sentence_count > 0 else 0
                
                if coherence < self.min_coherence and sentence_count > 1:
                    discard_reason = f'low_coherence ({coherence:.3f} < {self.min_coherence})'
                elif token_count < self.min_tokens and sentence_count == 1:
                    discard_reason = f'too_short (tokens={token_count}, sentences=1)'
                elif tokens_per_sentence < self.min_tokens_per_sentence and section_header is None:
                    # Orphan low-density = likely reference list / NLTK artifacts
                    discard_reason = f'orphan_low_density ({tokens_per_sentence:.1f} tokens/sent < {self.min_tokens_per_sentence})'
                
                if discard_reason:
                    discarded_chunks.append({
                        'document_id': document_id,
                        'text': chunk_text[:500],
                        'section_header': section_header,
                        'reason': discard_reason,
                        'token_count': token_count,
                        'sentence_count': sentence_count,
                        'coherence': coherence,
                    })
                    continue
                
                # Keep chunk
                chunk_id = generate_chunk_id(document_id, chunk_position)
                chunk = Chunk(
                    chunk_ids=[chunk_id],
                    document_ids=[document_id],
                    text=chunk_text,
                    position=chunk_position,
                    sentence_count=sentence_count,
                    token_count=token_count,
                    section_header=section_header,
                    metadata={
                        **metadata, 
                        'coherence': round(coherence, 3),
                        'tokens_per_sentence': round(tokens_per_sentence, 1)
                    }
                )
                
                kept_chunks.append(chunk)
                chunk_position += 1
        
        logger.debug(
            f"Document {document_id}: {len(kept_chunks)} kept, "
            f"{len(discarded_chunks)} discarded"
        )
        return kept_chunks, discarded_chunks
    
    def get_statistics(self, chunks: List[Chunk]) -> Dict:
        """
        Calculate chunking statistics.
        
        Args:
            chunks: List of Chunk objects
            
        Returns:
            Dict with token/sentence statistics
        """
        if not chunks:
            return {}
        
        token_counts = [c.token_count for c in chunks]
        sentence_counts = [c.sentence_count for c in chunks]
        coherence_scores = [c.metadata.get('coherence', 0) for c in chunks]
        
        return {
            'total_chunks': len(chunks),
            'tokens': {
                'mean': float(np.mean(token_counts)),
                'median': float(np.median(token_counts)),
                'std': float(np.std(token_counts)),
                'min': int(min(token_counts)),
                'max': int(max(token_counts)),
            },
            'sentences': {
                'mean': float(np.mean(sentence_counts)),
                'median': float(np.median(sentence_counts)),
                'std': float(np.std(sentence_counts)),
            },
            'coherence': {
                'mean': float(np.mean(coherence_scores)),
                'median': float(np.median(coherence_scores)),
                'min': float(min(coherence_scores)),
                'max': float(max(coherence_scores)),
            },
            'sections_with_headers': sum(1 for c in chunks if c.section_header),
        }