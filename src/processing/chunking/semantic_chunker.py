# -*- coding: utf-8 -*-
"""
Semantic chunker for AI governance GraphRAG pipeline.

Implements RAKG-style chunking with hierarchical boundaries: headers are hard
boundaries (respect document structure), within sections use sentence similarity
for semantic coherence, and sentences are never split (atomic units).
"""

import re
import nltk
from typing import List, Dict, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download nltk data if not present
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


@dataclass
class Chunk:
    """Represents a semantic chunk with metadata"""
    chunk_id: str
    document_id: str
    text: str
    position: int
    sentence_count: int
    token_count: int
    metadata: Dict
    section_header: Optional[str] = None


class SemanticChunker:
    """
    RAKG-style semantic chunker with hierarchical boundaries
    
    Algorithm:
    1. Split document into sections at headers (HARD boundaries)
    2. Within each section, use sentence similarity for soft boundaries
    3. Sentences are atomic - never split mid-sentence
    4. Leftover sentences before headers attach to previous chunk
    
    Args:
        model_name: Sentence transformer model for embeddings
        threshold: Cosine similarity threshold (0.7 = sentences must be 70% similar to stay together)
        min_sentences: Minimum sentences per chunk (prevents tiny chunks)
        max_tokens: Maximum tokens per chunk (prevents huge chunks)
    """
    
    def __init__(
        self,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        threshold: float = 0.7,
        min_sentences: int = 3,
        max_tokens: int = 1500
    ):
        print(f"Loading sentence transformer: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.min_sentences = min_sentences
        self.max_tokens = max_tokens
        
        # Regex patterns for detecting headers
        self.header_patterns = [
            r'^#{1,6}\s+.+$',           # Markdown headers: # Title
            r'^Article\s+\d+',          # Article 5
            r'^Section\s+\d+',          # Section 3
            r'^\d+\.\s+[A-Z]',          # 1. Title
            r'^[A-Z][A-Z\s]{3,}$'       # ALL CAPS HEADERS
        ]
        
    def is_header(self, line: str) -> bool:
        """Check if a line is a structural header"""
        line = line.strip()
        if not line:
            return False
        
        for pattern in self.header_patterns:
            if re.match(pattern, line):
                return True
        return False
    
    def split_into_sections(self, text: str) -> List[Dict]:
        """
        Split document into sections at header boundaries
        
        Returns:
            List of {'header': str, 'content': str}
        """
        lines = text.split('\n')
        sections = []
        current_header = None
        current_content = []
        
        for line in lines:
            if self.is_header(line):
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
        Chunk a section using sentence similarity
        
        Args:
            text: Section text (no headers)
        
        Returns:
            List of chunk texts
        """
        # Tokenize into sentences
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) <= self.min_sentences:
            # Section too small - return as single chunk
            return [text]
        
        # Encode all sentences
        embeddings = self.model.encode(sentences, show_progress_bar=False)
        
        # Find semantic boundaries
        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_tokens = len(sentences[0].split())
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_tokens = len(sentence.split())
            
            # Calculate similarity with previous sentence
            similarity = cosine_similarity(
                embeddings[i:i+1], 
                embeddings[i-1:i]
            )[0][0]
            
            # Decision: Add to current chunk or start new?
            would_exceed_max = (current_tokens + sentence_tokens) > self.max_tokens
            below_threshold = similarity < self.threshold
            
            if would_exceed_max or (below_threshold and len(current_chunk_sentences) >= self.min_sentences):
                # Start new chunk
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
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Chunk entire document with hierarchical approach
        
        Args:
            text: Full document text
            document_id: Unique document identifier
            metadata: Additional metadata to propagate
        
        Returns:
            List of Chunk objects
        """
        if metadata is None:
            metadata = {}
        
        # Step 1: Split into sections at headers
        sections = self.split_into_sections(text)
        
        # Step 2: Chunk each section by similarity
        all_chunks = []
        chunk_position = 0
        
        for section in sections:
            section_header = section['header']
            section_content = section['content']
            
            if not section_content.strip():
                continue
            
            # Chunk this section
            section_chunks = self.chunk_section_by_similarity(section_content)
            
            # Create Chunk objects
            for chunk_text in section_chunks:
                sentences = nltk.sent_tokenize(chunk_text)
                tokens = chunk_text.split()
                
                chunk = Chunk(
                    chunk_id=f"{document_id}_CHUNK_{chunk_position:04d}",
                    document_id=document_id,
                    text=chunk_text,
                    position=chunk_position,
                    sentence_count=len(sentences),
                    token_count=len(tokens),
                    metadata=metadata.copy(),
                    section_header=section_header
                )
                
                all_chunks.append(chunk)
                chunk_position += 1
        
        return all_chunks
    
    def get_statistics(self, chunks: List[Chunk]) -> Dict:
        """Calculate chunking statistics"""
        if not chunks:
            return {}
        
        token_counts = [c.token_count for c in chunks]
        sentence_counts = [c.sentence_count for c in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_tokens_per_chunk': np.mean(token_counts),
            'median_tokens_per_chunk': np.median(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'avg_sentences_per_chunk': np.mean(sentence_counts),
            'total_sentences': sum(sentence_counts),
            'sections_with_headers': sum(1 for c in chunks if c.section_header),
        }


if __name__ == "__main__":
    # Quick test
    test_text = """
# Article 5: Prohibited AI Practices

Social scoring systems are banned. They violate human dignity. 
Such systems create discriminatory outcomes.

The weather is nice today. It's sunny outside.

# Article 6: High-Risk Systems

Biometric identification requires authorization. 
Law enforcement must follow strict protocols.
These systems pose significant risks.
"""
    
    chunker = SemanticChunker(threshold=0.7)
    chunks = chunker.chunk_document(test_text, "TEST_DOC")
    
    print(f"\nCreated {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"\n{chunk.chunk_id} (Section: {chunk.section_header})")
        print(f"  Tokens: {chunk.token_count}, Sentences: {chunk.sentence_count}")
        print(f"  Text: {chunk.text[:100]}...")
    
    stats = chunker.get_statistics(chunks)
    print(f"\nStatistics: {stats}")