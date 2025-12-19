# -*- coding: utf-8 -*-
"""
Module: text_cleaner.py
Package: src.preprocessing
Purpose: Text cleaning with encoding fixes, HTML stripping, LaTeX removal, 
         image placeholder removal, email removal, and section extraction

Author: Pau Barba i Colomer
Created: 2025-12-18
Modified: 2025-12-18

Cleaning steps:
    1. Encoding fixes (ftfy via clean-text)
    2. HTML tag stripping
    3. LaTeX notation removal
    4. Markdown image placeholder removal
    5. Email address removal
    6. Irrelevant section removal (ARTICLEINFO, ACKNOWLEDGMENTS, etc.)
    7. Reference section extraction (saved separately, removed from text)
    8. Whitespace normalization

Configuration:
    - Garbage sections and reference sections defined in extraction_config.py
    
References:
    - See ARCHITECTURE.md § 7 for known issues
"""

# Standard library
import re
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field

# Project root (src/preprocessing/text_cleaner.py → 3 parents)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
from cleantext import clean

# Local
from src.utils.logger import get_logger
from config.extraction_config import PREPROCESSING_CONFIG

logger = get_logger(__name__)


# ============================================================================
# CONFIGURATION - From extraction_config.py
# ============================================================================

GARBAGE_SECTIONS = PREPROCESSING_CONFIG['garbage_sections']
REFERENCE_SECTION_NAMES = PREPROCESSING_CONFIG['reference_sections']


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CleaningResult:
    """Result of text cleaning operation."""
    text: str
    original_length: int
    cleaned_length: int
    fixes_applied: Dict[str, int] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)


# ============================================================================
# TEXT CLEANER
# ============================================================================

class TextCleaner:
    """
    Cleans text by applying multiple cleaning steps.
    
    Steps:
        1. Fix encoding issues (mojibake via ftfy)
        2. Strip HTML tags
        3. Remove LaTeX notation
        4. Remove markdown image placeholders
        5. Remove email addresses
        6. Remove irrelevant sections (ARTICLEINFO, ACKNOWLEDGMENTS, etc.)
        7. Extract and remove reference section (preserved in result)
        8. Normalize whitespace
    
    Uses clean-text library for encoding/HTML and custom regex for the rest.
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
    
    # HTML table artifacts
    HTML_TABLE_PATTERN = re.compile(r'</?(?:table|tr|td|th|tbody|thead)[^>]*>', re.IGNORECASE)
    
    # Markdown image placeholders: ![alt](path) or ![](images/...)
    MARKDOWN_IMAGE_PATTERN = re.compile(r'!\[[^\]]*\]\([^)]+\)')
    
    # Email pattern
    EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    
    def __init__(self, 
                 garbage_sections: Optional[List[str]] = None,
                 reference_sections: Optional[List[str]] = None):
        """
        Initialize cleaner with compiled regex patterns.
        
        Args:
            garbage_sections: List of section names to remove (default: GARBAGE_SECTIONS)
            reference_sections: List of reference section names to extract (default: REFERENCE_SECTION_NAMES)
        """
        self.garbage_sections = garbage_sections or GARBAGE_SECTIONS
        self.reference_sections = reference_sections or REFERENCE_SECTION_NAMES
        
        # Compile LaTeX patterns
        self.latex_compiled = [
            (re.compile(pattern), replacement) 
            for pattern, replacement in self.LATEX_PATTERNS
        ]
        
        # Build section removal pattern
        garbage_pattern = '|'.join(re.escape(s) for s in self.garbage_sections)
        self.garbage_section_pattern = re.compile(
            rf'^#\s*({garbage_pattern})\s*\n(.*?)(?=^# |\Z)',
            re.MULTILINE | re.DOTALL | re.IGNORECASE
        )
        
        # Build reference section pattern
        ref_pattern = '|'.join(re.escape(s) for s in self.reference_sections)
        self.reference_section_pattern = re.compile(
            rf'^#\s*({ref_pattern})\s*\n(.*?)(?=^# |\Z)',
            re.MULTILINE | re.DOTALL | re.IGNORECASE
        )
        
        logger.info(f"TextCleaner initialized (garbage sections: {len(self.garbage_sections)}, "
                    f"reference sections: {len(self.reference_sections)})")
    
    def clean(self, text: str, extract_references: bool = True) -> CleaningResult:
        """
        Clean text by applying all cleaning steps.
        
        Args:
            text: Raw text to clean
            extract_references: Whether to extract references (default: True)
            
        Returns:
            CleaningResult with cleaned text, statistics, and extracted references
        """
        if not text:
            return CleaningResult(
                text="",
                original_length=0,
                cleaned_length=0,
                fixes_applied={},
                references=[]
            )
        
        original_length = len(text)
        fixes = {}
        references = []
        
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
        
        # Step 5: Remove emails
        text, email_fixes = self._remove_emails(text)
        if email_fixes > 0:
            fixes['emails'] = email_fixes
        
        # Step 6: Remove garbage sections
        text, section_fixes = self._remove_garbage_sections(text)
        if section_fixes > 0:
            fixes['garbage_sections'] = section_fixes
        
        # Step 7: Extract and remove references
        if extract_references:
            text, references, ref_fixes = self._extract_references(text)
            if ref_fixes > 0:
                fixes['references_extracted'] = ref_fixes
        
        # Step 8: Normalize whitespace
        text = self._normalize_whitespace(text)
        
        return CleaningResult(
            text=text,
            original_length=original_length,
            cleaned_length=len(text),
            fixes_applied=fixes,
            references=references
        )
    
    def _fix_encoding(self, text: str) -> Tuple[str, int]:
        """
        Fix encoding issues using clean-text (wraps ftfy).
        
        Handles mojibake like: â€" → —, Ã© → é
        """
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
    
    def _remove_emails(self, text: str) -> Tuple[str, int]:
        """
        Remove email addresses from text.
        
        Handles: liadong@cityu.edu.hk → (removed)
        """
        emails = self.EMAIL_PATTERN.findall(text)
        fixes = len(emails)
        
        if fixes > 0:
            text = self.EMAIL_PATTERN.sub('', text)
        
        return text, fixes
    
    def _remove_garbage_sections(self, text: str) -> Tuple[str, int]:
        """
        Remove irrelevant sections (heading + content).
        
        Removes: ARTICLEINFO, ACKNOWLEDGMENTS, Author contributions, etc.
        Removal is from heading to next # heading or EOF.
        """
        fixes = 0
        
        while True:
            match = self.garbage_section_pattern.search(text)
            if not match:
                break
            
            section_name = match.group(1)
            logger.debug(f"Removing section: {section_name}")
            text = self.garbage_section_pattern.sub('', text, count=1)
            fixes += 1
        
        return text, fixes
    
    def _extract_references(self, text: str) -> Tuple[str, List[str], int]:
        """
        Extract reference section and parse into individual references.
        
        Returns:
            Tuple of (text_without_refs, list_of_references, section_count)
        """
        references = []
        fixes = 0
        
        match = self.reference_section_pattern.search(text)
        if not match:
            return text, references, fixes
        
        refs_text = match.group(2).strip()
        fixes = 1
        
        # Parse individual references
        references = self._parse_references(refs_text)
        
        # Remove section from text
        text = self.reference_section_pattern.sub('', text)
        
        logger.debug(f"Extracted {len(references)} references")
        
        return text, references, fixes
    
    def _parse_references(self, refs_text: str) -> List[str]:
        """
        Parse reference section text into individual references.
        
        Handles two formats:
            1. Numbered: [1] Author... [2] Author...
            2. Author-year: AuthorLastName, F. (2020). Title...
        """
        if not refs_text:
            return []
        
        # Pattern 1: Numbered references [1], [2], etc.
        if re.search(r'^\[\d+\]', refs_text, re.MULTILINE):
            refs = re.split(r'\n(?=\[\d+\])', refs_text)
        # Pattern 2: Author-year format (newline + capital letter starting author name)
        else:
            refs = re.split(r'\n(?=[A-Z][a-zA-Z\'\-]+,?\s+[A-Z])', refs_text)
        
        # Clean and filter
        refs = [r.strip() for r in refs if r.strip() and len(r.strip()) > 20]
        
        return refs
    
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