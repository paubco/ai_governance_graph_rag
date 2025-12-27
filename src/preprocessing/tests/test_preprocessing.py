# -*- coding: utf-8 -*-
"""
Test Preprocessing

# ============================================================================
# SAMPLE DATA (from actual v1.0 chunks)
# ============================================================================

"""
"""
# Standard library
import sys
from pathlib import Path

# Project root (src/preprocessing/tests/test_preprocessing.py → 4 parents)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
import pytest

# Utils imports
from src.utils.dataclasses import DocType

# Local imports (absolute, not relative)
from src.preprocessing.text_cleaner import TextCleaner, CleaningResult
from src.preprocessing.translator import Translator
from src.preprocessing.preprocessing_processor import PreprocessingProcessor, CleanedDocument


# ============================================================================
# SAMPLE DATA (from actual v1.0 chunks)
# ============================================================================

SAMPLE_HTML_TABLE = """<table><tr><td>Hex</td><td>Class</td><td>Error</td><td>Prob(sd)</td><td>Prem</td><td>Res</td><td>Age</td><td>Fee</td><td>Unl</td><td>Tax</td></tr><tr><td>H1</td><td>N(100%)</td><td"""

SAMPLE_SPANISH = """La inteligencia artificial está suscitando intensos debates en el Conjunto de la doctrina sobre su compatibiliad con la ética humanista imperante en la société occidental. Esta perspectiva antropocéntrica del cambio de paradigma planteado por la inteligencia artificial."""

SAMPLE_TURKISH = """Ikincisi isbe bu teknoljilerin toplumun tamamı ve kolektif fayda icin kullanilabilmesi amacıla mevcut Güç iliskilerini, yani teknooji alanindaki en buyüz oyuncularin ve bu oyuncilarin ekonomilerine entgre oldukları devletlerin."""

SAMPLE_LATEX = """Since its establishment in $1923^{40}$, the Republic of Turkey has made significant strides in various domains. In August 2021, Turkey unveiled"""

SAMPLE_ENCODING = "Artificial Intelligence \xe2\x80\x93 A New Era"  # Mojibake for em-dash

SAMPLE_CORRUPTED_CYRILLIC = """Пдхои, ЗakладetiВ Artificial Intelligence Act, 6e3nepequHo, Маюь Вжларе 3наченя дя ВИЗнчени"""


# ============================================================================
# TEXT CLEANER TESTS
# ============================================================================

class TestTextCleaner:
    """Tests for TextCleaner."""
    
    @pytest.fixture
    def cleaner(self):
        return TextCleaner()
    
    def test_clean_html_table(self, cleaner):
        """Should strip HTML table tags."""
        result = cleaner.clean(SAMPLE_HTML_TABLE)
        
        assert "<table>" not in result.text
        assert "<tr>" not in result.text
        assert "<td>" not in result.text
        assert result.fixes_applied.get("html", 0) > 0
    
    def test_clean_latex_superscript(self, cleaner):
        """Should clean LaTeX notation like $1923^{40}$."""
        result = cleaner.clean(SAMPLE_LATEX)
        
        assert "$" not in result.text
        assert "^{40}" not in result.text
        assert "1923" in result.text  # Year preserved
        assert "Turkey" in result.text
        assert result.fixes_applied.get("latex", 0) > 0
    
    def test_fix_encoding(self, cleaner):
        """Should fix mojibake like â€" to em-dash."""
        result = cleaner.clean(SAMPLE_ENCODING)
        
        # Should fix the encoding issue (mojibake bytes should be gone)
        assert "\xe2\x80\x93" not in result.text
        # Content preserved
        assert "Intelligence" in result.text
    
    def test_clean_preserves_content(self, cleaner):
        """Cleaning should preserve meaningful content."""
        result = cleaner.clean(SAMPLE_SPANISH)
        
        assert "inteligencia artificial" in result.text
        assert "debates" in result.text
        assert result.cleaned_length > 100  # Content preserved
    
    def test_empty_text(self, cleaner):
        """Should handle empty text gracefully."""
        result = cleaner.clean("")
        
        assert result.text == ""
        assert result.original_length == 0
        assert result.cleaned_length == 0
    
    def test_result_lengths(self, cleaner):
        """Result should track length changes."""
        result = cleaner.clean(SAMPLE_HTML_TABLE)
        
        assert result.original_length == len(SAMPLE_HTML_TABLE)
        assert result.cleaned_length <= result.original_length


# ============================================================================
# LANGUAGE DETECTION TESTS
# ============================================================================

class TestLanguageDetection:
    """Tests for language detection in PreprocessingProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create processor without document loader for unit tests."""
        # Mock document loader
        class MockLoader:
            def load_all_documents(self):
                return []
        
        return PreprocessingProcessor(
            document_loader=MockLoader(),
            translator=Translator(api_key=None)  # No API key for tests
        )
    
    def test_detect_spanish(self, processor):
        """Should detect Spanish text."""
        lang, confidence = processor._detect_language(SAMPLE_SPANISH)
        
        assert lang == "es"
        assert confidence > 0.8
    
    def test_detect_turkish(self, processor):
        """Should detect Turkish text."""
        lang, confidence = processor._detect_language(SAMPLE_TURKISH)
        
        assert lang == "tr"
        assert confidence > 0.5
    
    def test_detect_english(self, processor):
        """Should detect English text."""
        english_text = "The European Union has proposed comprehensive AI legislation."
        lang, confidence = processor._detect_language(english_text)
        
        assert lang == "en"
        assert confidence > 0.8
    
    def test_detect_short_text(self, processor):
        """Should handle very short text."""
        lang, confidence = processor._detect_language("Hi")
        
        # Too short for reliable detection
        assert lang == "unknown"
        assert confidence == 0.0
    
    def test_detect_mixed_script(self, processor):
        """Should detect (possibly incorrectly) corrupted Cyrillic-Latin mix."""
        lang, confidence = processor._detect_language(SAMPLE_CORRUPTED_CYRILLIC)
        
        # Will likely detect as some language, but with lower confidence
        assert lang != ""
        # Corrupted text often has lower confidence
        # Just check it doesn't crash


# ============================================================================
# TRANSLATOR TESTS (Cache only, no API calls)
# ============================================================================

class TestTranslatorCache:
    """Tests for Translator caching (no API calls)."""
    
    @pytest.fixture
    def translator(self, tmp_path):
        """Create translator with temp cache dir."""
        return Translator(
            api_key=None,  # No API key
            cache_dir=str(tmp_path / "translation_cache")
        )
    
    def test_english_passthrough(self, translator):
        """English text should pass through unchanged."""
        result = translator.translate(
            text="Hello world",
            source_lang="en",
            doc_id="test_001"
        )
        
        assert result.text == "Hello world"
        assert result.was_translated is False
        assert result.source_language == "en"
    
    def test_no_api_key_graceful(self, translator):
        """Should handle missing API key gracefully but loudly."""
        result = translator.translate(
            text=SAMPLE_SPANISH,
            source_lang="es",
            doc_id="test_002"
        )
        
        # Without API key, should return original text
        assert result.was_translated is False
        assert "inteligencia artificial" in result.text
        
        # Should track this as untranslated
        assert len(translator.untranslated_docs) == 1
        assert translator.untranslated_docs[0]["doc_id"] == "test_002"
    
    def test_cache_stats(self, translator):
        """Should track cache statistics and untranslated docs."""
        stats = translator.get_cache_stats()
        
        assert "cached_documents" in stats
        assert "total_cached_chars" in stats
        assert "languages" in stats
        assert "untranslated_docs" in stats
        assert "untranslated_count" in stats


# ============================================================================
# CLEANED DOCUMENT TESTS
# ============================================================================

class TestCleanedDocument:
    """Tests for CleanedDocument dataclass."""
    
    def test_to_dict(self):
        """Should serialize to dictionary."""
        doc = CleanedDocument(
            doc_id="reg_ES",
            source_type=DocType.REGULATION,
            title="AI Regulations - Spain",
            text="Cleaned text here...",
            original_language="es",
            language_confidence=0.95,
            was_translated=True,
            translation_from_cache=False,
            original_text_length=1000,
            cleaned_text_length=950,
            cleaning_fixes={"encoding": 5, "html": 2},
            metadata={"country_code": "ES"}
        )
        
        d = doc.to_dict()
        
        assert d["doc_id"] == "reg_ES"
        assert d["source_type"] == "regulation"  # Enum serialized to string
        assert d["was_translated"] is True
        assert d["cleaning_fixes"]["encoding"] == 5
        assert d["metadata"]["country_code"] == "ES"


# ============================================================================
# INTEGRATION TESTS (require actual data)
# ============================================================================

@pytest.mark.integration
class TestPreprocessingIntegration:
    """Integration tests requiring actual data files."""
    
    def test_full_pipeline(self):
        """
        Full pipeline test with real data.
        
        Skip if data files not present.
        """
        try:
            from src.ingestion.document_loader import DocumentLoader
        except ImportError:
            pytest.skip("DocumentLoader not available")
        
        data_path = Path("data/raw/dlapiper")
        if not data_path.exists():
            pytest.skip("Data directory not found")
        
        # Run pipeline on subset
        loader = DocumentLoader(year='2023')
        processor = PreprocessingProcessor(
            document_loader=loader,
            translator=Translator(api_key=None)
        )
        
        # Just verify it runs without error
        # (actual translation requires API key)
        cleaned_docs = processor.process_all()
        
        assert len(cleaned_docs) > 0
        assert all(isinstance(d, CleanedDocument) for d in cleaned_docs)