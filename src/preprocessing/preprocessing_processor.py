# -*- coding: utf-8 -*-
"""
Preprocessing Processor

@dataclass
class CleanedDocument:

References:
    See ARCHITECTURE.md § 7 for known issues

"""
"""
# Standard library
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

# Project root (src/preprocessing/preprocessing_processor.py → 3 parents)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
from langdetect import detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException

# Utils imports - use canonical types
from src.utils.dataclasses import DocType

# Local
from src.utils.logger import get_logger
from src.preprocessing.text_cleaner import TextCleaner, CleaningResult
from src.preprocessing.translator import Translator, TranslationResult

logger = get_logger(__name__)


@dataclass
class CleanedDocument:
    """Document after preprocessing."""
    doc_id: str
    source_type: DocType
    title: str
    text: str
    original_language: str
    language_confidence: float
    was_translated: bool
    translation_from_cache: bool
    original_text_length: int
    cleaned_text_length: int
    cleaning_fixes: Dict[str, int]
    references: List[str] = field(default_factory=list)  # Extracted bibliography
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "doc_id": self.doc_id,
            "source_type": self.source_type.value,  # Enum → string for JSON
            "title": self.title,
            "text": self.text,
            "original_language": self.original_language,
            "language_confidence": self.language_confidence,
            "was_translated": self.was_translated,
            "translation_from_cache": self.translation_from_cache,
            "original_text_length": self.original_text_length,
            "cleaned_text_length": self.cleaned_text_length,
            "cleaning_fixes": self.cleaning_fixes,
            "reference_count": len(self.references),  # Count only, refs saved separately
            "metadata": self.metadata,
        }


class PreprocessingProcessor:
    """
    Orchestrates document preprocessing pipeline.
    
    Pipeline:
        1. Load documents via DocumentLoader
        2. Clean text (encoding, HTML, LaTeX)
        3. Detect language
        4. Translate non-English to English
        5. Output cleaned JSONL + report
    """
    
    def __init__(
        self,
        document_loader,
        translator: Optional[Translator] = None,
        output_dir: str = "data/interim/preprocessed"
    ):
        """
        Initialize preprocessor.
        
        Args:
            document_loader: DocumentLoader instance
            translator: Translator instance (created if not provided)
            output_dir: Directory for output files
        """
        self.document_loader = document_loader
        self.text_cleaner = TextCleaner()
        self.translator = translator or Translator()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Output paths
        self.output_jsonl = self.output_dir / "documents_cleaned.jsonl"
        self.output_report = self.output_dir / "preprocessing_report.json"
        self.output_references = self.output_dir / "paper_references.json"
        
        # References storage (doc_id → list of refs)
        self.paper_references: Dict[str, List[str]] = {}
        
        # Statistics
        self.stats = {
            "total_documents": 0,
            "by_language": {},
            "by_source_type": {},
            "total_chars_before": 0,
            "total_chars_after": 0,
            "translated_count": 0,
            "cache_hits": 0,
            "cleaning_stats": {
                "encoding": 0,
                "html": 0,
                "latex": 0,
                "images": 0,
                "emails": 0,
                "garbage_sections": 0,
                "references_extracted": 0
            },
            "references_stats": {
                "papers_with_refs": 0,
                "total_refs_extracted": 0
            },
            "non_english_docs": [],
            "errors": []
        }
        
        logger.info(f"PreprocessingProcessor initialized, output: {self.output_dir}")
    
    def process_all(self) -> List[CleanedDocument]:
        """
        Process all documents through the preprocessing pipeline.
        
        Returns:
            List of CleanedDocument objects
        """
        logger.info("=" * 60)
        logger.info("STARTING PREPROCESSING PIPELINE")
        logger.info("=" * 60)
        
        # Load documents
        documents = self.document_loader.load_all_documents()
        self.stats["total_documents"] = len(documents)
        
        logger.info(f"Processing {len(documents)} documents...")
        
        # Process each document
        cleaned_documents = []
        
        for i, doc in enumerate(documents, 1):
            try:
                cleaned = self._process_document(doc)
                cleaned_documents.append(cleaned)
                
                # Progress logging
                if i % 20 == 0:
                    logger.info(f"Processed {i}/{len(documents)} documents...")
                    
            except Exception as e:
                logger.error(f"Failed to process {doc.doc_id}: {e}")
                self.stats["errors"].append({
                    "doc_id": doc.doc_id,
                    "error": str(e)
                })
        
        # Save outputs
        self._save_documents(cleaned_documents)
        self._save_references()
        self._save_report()
        
        logger.info("=" * 60)
        logger.info("PREPROCESSING COMPLETE")
        logger.info(f"Documents processed: {len(cleaned_documents)}")
        logger.info(f"Output: {self.output_jsonl}")
        logger.info(f"References: {self.output_references}")
        logger.info(f"Report: {self.output_report}")
        logger.info("=" * 60)
        
        return cleaned_documents
    
    def _process_document(self, doc) -> CleanedDocument:
        """
        Process a single document.
        
        Args:
            doc: Document from DocumentLoader
            
        Returns:
            CleanedDocument
        """
        # Step 1: Clean text (includes reference extraction for papers)
        is_paper = doc.source_type != "regulation"
        cleaning_result = self.text_cleaner.clean(doc.text, extract_references=is_paper)
        
        # Update cleaning stats
        for fix_type, count in cleaning_result.fixes_applied.items():
            self.stats["cleaning_stats"][fix_type] = \
                self.stats["cleaning_stats"].get(fix_type, 0) + count
        
        # Handle extracted references (papers only)
        references = []
        if is_paper and cleaning_result.references:
            references = cleaning_result.references
            self.paper_references[doc.doc_id] = references
            self.stats["references_stats"]["papers_with_refs"] += 1
            self.stats["references_stats"]["total_refs_extracted"] += len(references)
        
        # Step 2: Detect language
        detected_lang, confidence = self._detect_language(cleaning_result.text)
        
        # Update language stats
        self.stats["by_language"][detected_lang] = \
            self.stats["by_language"].get(detected_lang, 0) + 1
        
        # Track non-English docs for report
        if detected_lang != "en":
            self.stats["non_english_docs"].append({
                "doc_id": doc.doc_id,
                "detected_language": detected_lang,
                "confidence": confidence,
                "char_count": len(cleaning_result.text)
            })
        
        # Step 3: Translate if needed
        translation_result = self.translator.translate(
            text=cleaning_result.text,
            source_lang=detected_lang,
            doc_id=doc.doc_id
        )
        
        # Update translation stats
        if translation_result.was_translated:
            self.stats["translated_count"] += 1
            if translation_result.from_cache:
                self.stats["cache_hits"] += 1
        
        # Update char stats
        self.stats["total_chars_before"] += cleaning_result.original_length
        self.stats["total_chars_after"] += len(translation_result.text)
        
        # Convert source_type string to DocType enum
        # DocumentLoader uses "regulation" or "academic_paper"
        if doc.source_type == "regulation":
            doc_type = DocType.REGULATION
        else:
            doc_type = DocType.PAPER
        
        # Update source type stats (use enum value for JSON-safe keys)
        self.stats["by_source_type"][doc_type.value] = \
            self.stats["by_source_type"].get(doc_type.value, 0) + 1
        
        return CleanedDocument(
            doc_id=doc.doc_id,
            source_type=doc_type,
            title=doc.title,
            text=translation_result.text,
            original_language=detected_lang,
            language_confidence=confidence,
            was_translated=translation_result.was_translated,
            translation_from_cache=translation_result.from_cache,
            original_text_length=cleaning_result.original_length,
            cleaned_text_length=len(translation_result.text),
            cleaning_fixes=cleaning_result.fixes_applied,
            references=references,
            metadata=doc.metadata
        )
    
    def _detect_language(self, text: str) -> tuple:
        """
        Detect language of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (language_code, confidence)
        """
        if not text or len(text) < 20:
            return "unknown", 0.0
        
        try:
            # Get detailed detection with probabilities
            detections = detect_langs(text[:5000])  # Sample first 5k chars
            top = detections[0]
            return top.lang, top.prob
        except LangDetectException:
            return "unknown", 0.0
    
    def _save_documents(self, documents: List[CleanedDocument]):
        """Save cleaned documents to JSONL."""
        # Import here to avoid circular dependency
        import json
        
        with open(self.output_jsonl, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(documents)} documents to {self.output_jsonl}")
    
    def _save_references(self):
        """Save extracted references to JSON (separate file for later use)."""
        import json
        
        if not self.paper_references:
            logger.info("No references extracted (no papers or no reference sections)")
            return
        
        with open(self.output_references, 'w', encoding='utf-8') as f:
            json.dump(self.paper_references, f, indent=2, ensure_ascii=False)
        
        total_refs = sum(len(refs) for refs in self.paper_references.values())
        logger.info(f"Saved {total_refs} references from {len(self.paper_references)} papers to {self.output_references}")
    
    def _save_report(self):
        """Save preprocessing report to JSON."""
        import json
        
        report = {
            "run_timestamp": datetime.now().isoformat(),
            "pipeline_version": "0.1.0",
            **self.stats,
            "translation_cache_stats": self.translator.get_cache_stats()
        }
        
        with open(self.output_report, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved report to {self.output_report}")
        
        # Log summary
        self._log_summary()
    
    def _log_summary(self):
        """Log preprocessing summary."""
        logger.info("\n" + "=" * 60)
        logger.info("PREPROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total documents: {self.stats['total_documents']}")
        logger.info(f"By source type: {self.stats['by_source_type']}")
        logger.info(f"By language: {self.stats['by_language']}")
        logger.info(f"Translated: {self.stats['translated_count']}")
        logger.info(f"Translation cache hits: {self.stats['cache_hits']}")
        logger.info(f"Chars before: {self.stats['total_chars_before']:,}")
        logger.info(f"Chars after: {self.stats['total_chars_after']:,}")
        logger.info(f"Cleaning fixes: {self.stats['cleaning_stats']}")
        logger.info(f"References: {self.stats['references_stats']['total_refs_extracted']} from {self.stats['references_stats']['papers_with_refs']} papers")
        if self.stats['errors']:
            logger.warning(f"Errors: {len(self.stats['errors'])}")
        
        # LOUD WARNING if docs went untranslated
        untranslated = self.translator.untranslated_docs
        if untranslated:
            logger.warning("=" * 60)
            logger.warning(f"WARNING: {len(untranslated)} DOCUMENTS NOT TRANSLATED!")
            logger.warning("These documents remain in their original language:")
            for doc in untranslated:
                logger.warning(f"  - {doc['doc_id']} ({doc['source_lang']}, {doc['char_count']} chars)")
            logger.warning("Set GOOGLE_TRANSLATE_API_KEY and re-run to translate.")
            logger.warning("=" * 60)
        
        logger.info("=" * 60)


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """
    Run preprocessing pipeline from command line.
    
    Usage:
        python -m src.preprocessing.preprocessing_processor
    """
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Setup logging using foundation utility
    from src.utils.logger import setup_logging
    setup_logging(log_file="logs/preprocessing.log")
    
    # Import document loader
    from src.ingestion.document_loader import DocumentLoader
    
    # Initialize
    loader = DocumentLoader(year='2023')
    processor = PreprocessingProcessor(document_loader=loader)
    
    # Run pipeline
    cleaned_docs = processor.process_all()
    
    print(f"\nDone! Processed {len(cleaned_docs)} documents.")
    print(f"Output: data/interim/preprocessed/documents_cleaned.jsonl")
    print(f"References: data/interim/preprocessed/paper_references.json")


if __name__ == "__main__":
    main()