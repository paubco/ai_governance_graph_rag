# -*- coding: utf-8 -*-
"""
Module: text_cleaner.py
Package: src.preprocessing
Purpose: Text cleaning with encoding fixes, HTML stripping, LaTeX removal, and image placeholder removal

Author: Pau Barba i Colomer
Created: 2025-12-18
Modified: 2025-12-18

Cleaning steps:
    1. Encoding fixes (ftfy via clean-text)
    2. HTML tag stripping
    3. LaTeX notation removal
    4. Markdown image placeholder removal
    5. Whitespace normalization

References:
    - See ARCHITECTURE.md § 7 for known issues (garbage chunks, non-English)
"""

# Standard library
import re
import sys
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass, field

# Project root (src/preprocessing/text_cleaner.py → 3 parents)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
from cleantext import clean

# Local
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CleaningResult:
    """Result of text cleaning operation."""
    text: str
    original_length: int
    cleaned_length: int
    fixes_applied: Dict[str, int] = field(default_factory=dict)


class TextCleaner:
    """
    Cleans text by applying multiple cleaning steps.
    
    Steps:
        1. Fix encoding issues (mojibake via ftfy)
        2. Strip HTML tags
        3. Remove LaTeX notation
        4. Remove markdown image placeholders
        5. Normalize whitespace
    
    Uses clean-text library for encoding/HTML and custom regex for LaTeX/images.
    """
    
    # LaTeX patterns to clean
    LATEX_PATTERNS = [
        # Math mode: $x^2$ or $1923^{40}$
        (r'\$([^$]+)\$', r'\1'),
        # Superscript: ^{40} or ^2
        (r'\^{([^}]+)}', r'\1'),
        (r'\^(\d+)', r''),  # Remove standalone superscripts (footnote refs)
        # Subscript: _{2}
        (r'_{([^}]+)}', r'\1'),
        # Commands: \textbf{text}, \textit{text}, \emph{text}
        (r'\\text(?:bf|it|rm|sf|tt){([^}]+)}', r'\1'),
        (r'\\emph{([^}]+)}', r'\1'),
        # Citations: \cite{ref} → remove
        (r'\\cite{[^}]+}', ''),
        # References: \ref{fig:1} → remove
        (r'\\ref{[^}]+}', ''),
        # Labels: \label{...} → remove
        (r'\\label{[^}]+}', ''),
        # Section commands → keep text
        (r'\\(?:section|subsection|subsubsection)\*?{([^}]+)}', r'\1'),
        # Remaining backslash commands → remove
        (r'\\[a-zA-Z]+(?:{[^}]*})?', ''),
        # Curly braces cleanup
        (r'(?<!\S){([^{}]+)}(?!\S)', r'\1'),
    ]
    
    # HTML table artifacts (from your samples)
    HTML_TABLE_PATTERN = re.compile(r'</?(?:table|tr|td|th|tbody|thead)[^>]*>', re.IGNORECASE)
    
    # Markdown image placeholders: ![alt](path) or ![](images/...)
    MARKDOWN_IMAGE_PATTERN = re.compile(r'!\[[^\]]*\]\([^)]+\)')
    
    def __init__(self):
        """Initialize cleaner with compiled regex patterns."""
        self.latex_compiled = [
            (re.compile(pattern), replacement) 
            for pattern, replacement in self.LATEX_PATTERNS
        ]
        logger.info("TextCleaner initialized")
    
    def clean(self, text: str) -> CleaningResult:
        """
        Clean text by applying all cleaning steps.
        
        Args:
            text: Raw text to clean
            
        Returns:
            CleaningResult with cleaned text and statistics
        """
        if not text:
            return CleaningResult(
                text="",
                original_length=0,
                cleaned_length=0,
                fixes_applied={}
            )
        
        original_length = len(text)
        fixes = {}
        
        # Step 1: Fix encoding (ftfy via clean-text)
        text, encoding_fixes = self._fix_encoding(text)
        if encoding_fixes > 0:
            fixes['encoding'] = encoding_fixes
        
        # Step 2: Strip HTML
        text, html_fixes = self._strip_html(text)
        if html_fixes > 0:
            fixes['html'] = html_fixes
        
        # Step 3: Clean LaTeX
        text, latex_fixes = self._clean_latex(text)
        if latex_fixes > 0:
            fixes['latex'] = latex_fixes
        
        # Step 4: Strip markdown images
        text, image_fixes = self._strip_markdown_images(text)
        if image_fixes > 0:
            fixes['images'] = image_fixes
        
        # Step 5: Normalize whitespace
        text = self._normalize_whitespace(text)
        
        return CleaningResult(
            text=text,
            original_length=original_length,
            cleaned_length=len(text),
            fixes_applied=fixes
        )
    
    def _fix_encoding(self, text: str) -> Tuple[str, int]:
        """
        Fix encoding issues using clean-text (wraps ftfy).
        
        Handles mojibake like: â€" → —, Ã© → é
        """
        # clean() from cleantext fixes encoding via ftfy
        cleaned = clean(
            text,
            fix_unicode=True,
            to_ascii=False,  # Keep unicode chars
            lower=False,
            no_line_breaks=False,
            no_urls=False,
            no_emails=False,
            no_phone_numbers=False,
            no_numbers=False,
            no_digits=False,
            no_currency_symbols=False,
            no_punct=False,
        )
        
        # Count differences (approximate fix count)
        fixes = sum(1 for a, b in zip(text, cleaned) if a != b)
        
        return cleaned, fixes
    
    def _strip_html(self, text: str) -> Tuple[str, int]:
        """
        Strip HTML tags from text.
        
        Handles table artifacts like: <table><tr><td>...
        """
        # Count HTML tags before removal
        html_tags = len(self.HTML_TABLE_PATTERN.findall(text))
        html_tags += text.count('<') - text.count('\\<')  # Rough count
        
        # Use clean-text HTML stripping
        cleaned = clean(
            text,
            fix_unicode=False,  # Already done
            to_ascii=False,
            lower=False,
            no_line_breaks=False,
            no_urls=False,
            no_emails=False,
            no_phone_numbers=False,
            no_numbers=False,
            no_digits=False,
            no_currency_symbols=False,
            no_punct=False,
        )
        
        # Additional: strip any remaining HTML-like tags
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        
        return cleaned, html_tags if html_tags > 0 else 0
    
    def _clean_latex(self, text: str) -> Tuple[str, int]:
        """
        Remove LaTeX notation from text.
        
        Handles: $1923^{40}$ → 1923, \textbf{AI} → AI
        """
        fixes = 0
        
        for pattern, replacement in self.latex_compiled:
            text, count = pattern.subn(replacement, text)
            fixes += count
        
        return text, fixes
    
    def _strip_markdown_images(self, text: str) -> Tuple[str, int]:
        """
        Remove markdown image placeholders.
        
        Handles: ![alt text](images/abc123.jpg) → (removed)
        """
        matches = self.MARKDOWN_IMAGE_PATTERN.findall(text)
        fixes = len(matches)
        
        if fixes > 0:
            text = self.MARKDOWN_IMAGE_PATTERN.sub('', text)
        
        return text, fixes
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace: multiple spaces → single, trim lines.
        """
        # Multiple spaces to single
        text = re.sub(r' +', ' ', text)
        # Multiple newlines to double (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Trim each line
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(lines).strip()