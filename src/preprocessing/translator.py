# -*- coding: utf-8 -*-
"""
Translator

@dataclass
class TranslationResult:

References:
    See ARCHITECTURE.md § 7 for known issues (non-English docs)

"""
"""
# Standard library
import os
import json
import hashlib
import sys
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass

# Project root (src/preprocessing/translator.py → 3 parents)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TranslationResult:
    """Result of translation operation."""
    text: str
    source_language: str
    was_translated: bool
    from_cache: bool
    char_count: int


class Translator:
    """
    Translates text to English using Google Cloud Translation API.
    
    Features:
    - File-based caching (avoids re-translating)
    - Handles long texts by chunking at sentence boundaries
    - Graceful degradation if API unavailable
    """
    
    # Google Translate API character limit per request
    MAX_CHARS_PER_REQUEST = 5000
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "data/interim/translation_cache"
    ):
        """
        Initialize translator.
        
        Args:
            api_key: Google Cloud API key (or set GOOGLE_TRANSLATE_API_KEY env var)
            cache_dir: Directory for translation cache files
        """
        self.api_key = api_key or os.getenv("GOOGLE_TRANSLATE_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache index for quick lookups
        self.cache_index_path = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
        
        # Track untranslated docs for reporting
        self.untranslated_docs = []
        
        # Initialize Google Translate client
        self.client = None
        if self.api_key:
            self._init_client()
        else:
            logger.warning("=" * 60)
            logger.warning("NO GOOGLE_TRANSLATE_API_KEY FOUND")
            logger.warning("Non-English documents will NOT be translated!")
            logger.warning("Set GOOGLE_TRANSLATE_API_KEY in .env to enable translation")
            logger.warning("=" * 60)
    
    def _init_client(self):
        """Verify API key works with a test call."""
        try:
            import requests
            
            # Test the API key with a simple request
            url = "https://translation.googleapis.com/language/translate/v2"
            params = {
                "key": self.api_key,
                "q": "test",
                "target": "en"
            }
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                self.client = True  # Just a flag - we use REST API
                logger.info("Google Translate API key verified")
            else:
                error = response.json().get("error", {}).get("message", "Unknown error")
                logger.error(f"Google Translate API error: {error}")
                self.client = None
                
        except ImportError:
            logger.error("requests library not installed. Run: pip install requests")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to verify Google Translate API: {e}")
            self.client = None
    
    def _load_cache_index(self) -> Dict:
        """Load cache index from disk."""
        if self.cache_index_path.exists():
            with open(self.cache_index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_cache_index(self):
        """Save cache index to disk."""
        with open(self.cache_index_path, 'w', encoding='utf-8') as f:
            json.dump(self.cache_index, f, indent=2, ensure_ascii=False)
    
    def _get_cache_key(self, doc_id: str, source_lang: str) -> str:
        """Generate cache key for a document."""
        return f"{doc_id}_{source_lang}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cached translation."""
        return self.cache_dir / f"{cache_key}.txt"
    
    def _get_from_cache(self, doc_id: str, source_lang: str) -> Optional[str]:
        """
        Retrieve translation from cache if exists.
        
        Args:
            doc_id: Document identifier
            source_lang: Source language code
            
        Returns:
            Cached translation or None
        """
        cache_key = self._get_cache_key(doc_id, source_lang)
        
        if cache_key in self.cache_index:
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return f.read()
        
        return None
    
    def _save_to_cache(self, doc_id: str, source_lang: str, translated_text: str):
        """
        Save translation to cache.
        
        Args:
            doc_id: Document identifier
            source_lang: Source language code
            translated_text: Translated text
        """
        cache_key = self._get_cache_key(doc_id, source_lang)
        cache_path = self._get_cache_path(cache_key)
        
        # Save text file
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(translated_text)
        
        # Update index
        self.cache_index[cache_key] = {
            "doc_id": doc_id,
            "source_lang": source_lang,
            "char_count": len(translated_text),
        }
        self._save_cache_index()
    
    def translate(
        self,
        text: str,
        source_lang: str,
        doc_id: str
    ) -> TranslationResult:
        """
        Translate text to English.
        
        Args:
            text: Text to translate
            source_lang: Source language code (e.g., "es", "tr")
            doc_id: Document ID for caching
            
        Returns:
            TranslationResult with translated text and metadata
        """
        # Check if already English
        if source_lang == "en":
            return TranslationResult(
                text=text,
                source_language="en",
                was_translated=False,
                from_cache=False,
                char_count=len(text)
            )
        
        # Check cache
        cached = self._get_from_cache(doc_id, source_lang)
        if cached is not None:
            logger.info(f"Cache hit for {doc_id} ({source_lang})")
            return TranslationResult(
                text=cached,
                source_language=source_lang,
                was_translated=True,
                from_cache=True,
                char_count=len(cached)
            )
        
        # Translate via API
        if not self.client:
            logger.warning(f"SKIPPING TRANSLATION: {doc_id} ({source_lang}) - No API key!")
            self.untranslated_docs.append({
                "doc_id": doc_id,
                "source_lang": source_lang,
                "char_count": len(text)
            })
            return TranslationResult(
                text=text,
                source_language=source_lang,
                was_translated=False,
                from_cache=False,
                char_count=len(text)
            )
        
        translated = self._translate_with_api(text, source_lang)
        
        # Cache result
        self._save_to_cache(doc_id, source_lang, translated)
        
        logger.info(f"Translated {doc_id}: {source_lang} → en ({len(text)} chars)")
        
        return TranslationResult(
            text=translated,
            source_language=source_lang,
            was_translated=True,
            from_cache=False,
            char_count=len(translated)
        )
    
    def _translate_with_api(self, text: str, source_lang: str) -> str:
        """
        Call Google Translate REST API.
        
        Handles long texts by splitting at sentence boundaries.
        """
        import requests
        
        url = "https://translation.googleapis.com/language/translate/v2"
        
        # If text is short enough, translate directly
        if len(text) <= self.MAX_CHARS_PER_REQUEST:
            params = {
                "key": self.api_key,
                "q": text,
                "source": source_lang,
                "target": "en",
                "format": "text"
            }
            response = requests.post(url, data=params)
            
            if response.status_code != 200:
                error = response.json().get("error", {}).get("message", "Unknown")
                raise Exception(f"Translation API error: {error}")
            
            return response.json()["data"]["translations"][0]["translatedText"]
        
        # Split long text into chunks at sentence boundaries
        chunks = self._split_into_chunks(text)
        translated_chunks = []
        
        for chunk in chunks:
            params = {
                "key": self.api_key,
                "q": chunk,
                "source": source_lang,
                "target": "en",
                "format": "text"
            }
            response = requests.post(url, data=params)
            
            if response.status_code != 200:
                error = response.json().get("error", {}).get("message", "Unknown")
                raise Exception(f"Translation API error: {error}")
            
            translated_chunks.append(
                response.json()["data"]["translations"][0]["translatedText"]
            )
        
        return " ".join(translated_chunks)
    
    def _split_into_chunks(self, text: str) -> list:
        """
        Split text into chunks respecting sentence boundaries.
        
        Args:
            text: Long text to split
            
        Returns:
            List of text chunks under MAX_CHARS_PER_REQUEST
        """
        import re
        
        # Split by sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > self.MAX_CHARS_PER_REQUEST:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about translation cache and skipped docs."""
        return {
            "cached_documents": len(self.cache_index),
            "total_cached_chars": sum(
                entry.get("char_count", 0) 
                for entry in self.cache_index.values()
            ),
            "languages": list(set(
                entry.get("source_lang", "unknown")
                for entry in self.cache_index.values()
            )),
            "untranslated_docs": self.untranslated_docs,
            "untranslated_count": len(self.untranslated_docs)
        }