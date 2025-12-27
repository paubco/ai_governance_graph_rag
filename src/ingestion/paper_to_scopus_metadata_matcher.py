# -*- coding: utf-8 -*-
"""
Paper To Scopus Metadata Matcher

# ============================================================================
# CONFIGURATION
# ============================================================================

Examples:
python -m src.ingestion.paper_to_scopus_metadata_matcher

"""
# Standard library
import json
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from difflib import SequenceMatcher

# Project root (src/ingestion/paper_to_scopus_metadata_matcher.py → 3 parents)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
import pandas as pd

# Local
from src.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Matching thresholds (lowered based on empirical testing)
TITLE_THRESHOLD_HIGH = 0.85      # Auto-accept
TITLE_THRESHOLD_MEDIUM = 0.70    # Accept with note
TITLE_THRESHOLD_LOW = 0.60       # Flag for review

ABSTRACT_THRESHOLD_HIGH = 0.80   # Auto-accept
ABSTRACT_THRESHOLD_MEDIUM = 0.60 # Accept with note

# DOI patterns
DOI_PATTERN = re.compile(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', re.IGNORECASE)
DOI_URL_PATTERN = re.compile(r'https?://doi\.org/(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', re.IGNORECASE)


# ============================================================================
# MATCHER CLASS
# ============================================================================

class MinerUMatcher:
    """
    Match MinerU papers to Scopus metadata using multi-strategy linking.
    
    Improvements over v1:
    - Lower title threshold (0.85 → 0.70) based on empirical testing
    - Fallback title extraction from full.md when content_list.json fails
    - Confidence tiers for transparency
    - Better abstract matching with lower threshold
    """
    
    def __init__(self, papers_path: str, scopus_csv: str):
        """
        Initialize matcher.
        
        Args:
            papers_path: Path to MinerU_parsed_papers directory
            scopus_csv: Path to Scopus export CSV
        """
        self.papers_path = Path(papers_path)
        self.scopus_csv = Path(scopus_csv)
        self.mapping_file = self.papers_path.parent / "paper_mapping.json"
        
        # Output directory
        self.output_dir = Path("data/interim/academic")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Scopus data
        logger.info(f"Loading Scopus CSV: {self.scopus_csv}")
        self.scopus_df = pd.read_csv(self.scopus_csv, encoding='utf-8-sig')
        logger.info(f"Loaded {len(self.scopus_df)} Scopus records")
        
        # Build paper mapping from folder structure
        self.paper_mapping = self._build_paper_mapping()
        logger.info(f"Found {len(self.paper_mapping)} paper folders")
    
    def _build_paper_mapping(self) -> Dict:
        """Build mapping from paper folders, preserving existing data."""
        mapping = {}
        
        # Load existing mapping if present
        if self.mapping_file.exists():
            try:
                with open(self.mapping_file, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                logger.info(f"Loaded existing mapping with {len(mapping)} entries")
            except Exception as e:
                logger.warning(f"Could not load existing mapping: {e}")
        
        # Ensure all paper folders are in mapping
        for paper_folder in sorted(self.papers_path.glob('paper_*')):
            if paper_folder.is_dir():
                paper_id = paper_folder.name
                if paper_id not in mapping:
                    mapping[paper_id] = {}
        
        return mapping
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text or text == "Unknown" or pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    # ========================================================================
    # EXTRACTION METHODS
    # ========================================================================
    
    def extract_titles(self, paper_folder: Path) -> List[str]:
        """
        Extract titles from content_list.json AND full.md.
        
        Returns list of candidate titles to try matching.
        """
        titles = []
        
        # Strategy 1: content_list.json (text_level: 1)
        json_path = paper_folder / "content_list.json"
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    content_list = json.load(f)
                
                for item in content_list:
                    if item.get('text_level') == 1:
                        text = item.get('text', '').strip()
                        if text and len(text) > 10:
                            titles.append(text)
            except Exception as e:
                logger.debug(f"Error reading content_list.json: {e}")
        
        # Strategy 2: full.md first heading (# Title)
        md_path = paper_folder / "full.md"
        if md_path.exists():
            try:
                with open(md_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:30]  # First 30 lines
                
                for line in lines:
                    line = line.strip()
                    # Match # Title (but not ## or ###)
                    if line.startswith('# ') and not line.startswith('## '):
                        title = line[2:].strip()
                        # Skip garbage titles
                        if len(title) > 15 and title.upper() not in [
                            'ABSTRACT', 'INTRODUCTION', 'REFERENCES', 
                            'ACKNOWLEDGEMENTS', 'CONCLUSION', 'METHODS'
                        ]:
                            if title not in titles:
                                titles.append(title)
            except Exception as e:
                logger.debug(f"Error reading full.md: {e}")
        
        return titles
    
    def extract_abstract(self, paper_folder: Path) -> Optional[str]:
        """Extract abstract from markdown file."""
        md_path = paper_folder / "full.md"
        if not md_path.exists():
            return None
        
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try multiple patterns
            patterns = [
                r'#\s*Abstract\s*\n+(.+?)(?=\n#|\n\n\n|\Z)',
                r'#\s*ABSTRACT\s*\n+(.+?)(?=\n#|\n\n\n|\Z)',
                r'Abstract[:\s]*\n+(.+?)(?=\n#|\n\n\n|\Z)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                if match:
                    abstract = match.group(1).strip()
                    # Clean markdown artifacts
                    abstract = re.sub(r'!\[.*?\]\(.*?\)', '', abstract)
                    abstract = re.sub(r'\s+', ' ', abstract)
                    if len(abstract) > 100:
                        return abstract[:1500]
            
            # Fallback: extract first substantial paragraph after title
            lines = content.split('\n')
            text_lines = []
            started = False
            for line in lines[3:50]:  # Skip first 3 lines (title area)
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('!'):
                    if started and text_lines:
                        break
                    continue
                started = True
                text_lines.append(line)
            
            if text_lines:
                fallback = ' '.join(text_lines)
                fallback = re.sub(r'\s+', ' ', fallback)
                if len(fallback) > 100:
                    return fallback[:1500]
        
        except Exception as e:
            logger.debug(f"Error extracting abstract: {e}")
        
        return None
    
    def extract_doi(self, paper_folder: Path) -> Optional[str]:
        """Extract DOI from content_list.json or markdown."""
        
        # Priority 1: content_list.json
        json_path = paper_folder / "content_list.json"
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    content_list = json.load(f)
                
                for item in content_list:
                    item_type = item.get('type', '')
                    text = item.get('text', '')
                    
                    if item_type in ['footer', 'header', 'text']:
                        # Try DOI URL
                        url_match = DOI_URL_PATTERN.search(text)
                        if url_match:
                            return url_match.group(1)
                        
                        # Try plain DOI (only if single occurrence)
                        doi_match = DOI_PATTERN.search(text)
                        if doi_match and text.count('10.') == 1:
                            return doi_match.group(0)
            
            except Exception as e:
                logger.debug(f"Error reading JSON for DOI: {e}")
        
        # Priority 2: markdown header area
        md_path = paper_folder / "full.md"
        if md_path.exists():
            try:
                with open(md_path, 'r', encoding='utf-8') as f:
                    text = f.read(5000)  # First 5000 chars
                
                # Try DOI URL first
                url_match = DOI_URL_PATTERN.search(text)
                if url_match:
                    return url_match.group(1)
                
                # Try plain DOI
                doi_match = DOI_PATTERN.search(text)
                if doi_match:
                    return doi_match.group(0)
            
            except Exception as e:
                logger.debug(f"Error reading markdown for DOI: {e}")
        
        return None
    
    # ========================================================================
    # MATCHING METHODS
    # ========================================================================
    
    def match_by_titles(self, paper_titles: List[str]) -> Optional[Tuple[str, float, str, str]]:
        """
        Match by title similarity.
        
        Returns: (eid, score, matched_title, confidence_tier)
        """
        if not paper_titles:
            return None
        
        best_eid = None
        best_score = 0
        best_title = None
        
        for paper_title in paper_titles[:5]:  # Try first 5 titles
            paper_norm = self._normalize_text(paper_title)
            if not paper_norm or len(paper_norm) < 10:
                continue
            
            for _, row in self.scopus_df.iterrows():
                scopus_title = self._normalize_text(str(row.get('Title', '')))
                if not scopus_title:
                    continue
                
                score = SequenceMatcher(None, paper_norm, scopus_title).ratio()
                
                if score > best_score:
                    best_score = score
                    best_eid = row['EID']
                    best_title = paper_title
        
        # Determine confidence tier
        if best_score >= TITLE_THRESHOLD_HIGH:
            return (best_eid, best_score, best_title, 'high')
        elif best_score >= TITLE_THRESHOLD_MEDIUM:
            return (best_eid, best_score, best_title, 'medium')
        elif best_score >= TITLE_THRESHOLD_LOW:
            return (best_eid, best_score, best_title, 'low')
        
        return None
    
    def match_by_abstract(self, paper_abstract: str) -> Optional[Tuple[str, float, str]]:
        """
        Match by abstract similarity.
        
        Returns: (eid, score, confidence_tier)
        """
        if not paper_abstract or len(paper_abstract) < 100:
            return None
        
        paper_norm = self._normalize_text(paper_abstract)[:800]
        
        best_eid = None
        best_score = 0
        
        for _, row in self.scopus_df.iterrows():
            scopus_abstract = self._normalize_text(str(row.get('Abstract', '')))
            if len(scopus_abstract) < 50:
                continue
            
            score = SequenceMatcher(None, paper_norm, scopus_abstract[:800]).ratio()
            
            if score > best_score:
                best_score = score
                best_eid = row['EID']
        
        # Determine confidence tier
        if best_score >= ABSTRACT_THRESHOLD_HIGH:
            return (best_eid, best_score, 'high')
        elif best_score >= ABSTRACT_THRESHOLD_MEDIUM:
            return (best_eid, best_score, 'medium')
        
        return None
    
    # ========================================================================
    # MAIN MATCHING LOGIC
    # ========================================================================
    
    def match_all(self):
        """Match all papers to Scopus metadata."""
        
        logger.info("=" * 70)
        logger.info("MATCHING MINERU PAPERS TO SCOPUS")
        logger.info("=" * 70)
        logger.info(f"Papers:         {len(self.paper_mapping)}")
        logger.info(f"Scopus records: {len(self.scopus_df)}")
        logger.info(f"Title threshold: {TITLE_THRESHOLD_MEDIUM}")
        logger.info(f"Abstract threshold: {ABSTRACT_THRESHOLD_MEDIUM}")
        logger.info("")
        
        stats = {
            'doi': 0, 
            'title_high': 0, 
            'title_medium': 0,
            'title_low': 0,
            'abstract': 0, 
            'unmatched': 0
        }
        results = []
        
        for paper_id in sorted(self.paper_mapping.keys()):
            paper_info = self.paper_mapping[paper_id]
            paper_folder = self.papers_path / paper_id
            
            if not paper_folder.exists():
                logger.warning(f"[{paper_id}] Folder not found, skipping")
                continue
            
            result = {
                'paper_id': paper_id,
                'extracted_title': None,
                'scopus_eid': None,
                'scopus_title': None,
                'doi_extracted': None,
                'match_method': None,
                'match_confidence': 0.0,
                'confidence_tier': None
            }
            
            # Extract titles for logging
            titles = self.extract_titles(paper_folder)
            result['extracted_title'] = titles[0] if titles else 'Unknown'
            
            logger.info(f"[{paper_id}] {result['extracted_title'][:55]}...")
            
            # Strategy 1: DOI (perfect match)
            doi = self.extract_doi(paper_folder)
            if doi:
                result['doi_extracted'] = doi
                matched = self.scopus_df[
                    self.scopus_df['DOI'].str.lower() == doi.lower()
                ]
                
                if not matched.empty:
                    scopus_row = matched.iloc[0]
                    result['scopus_eid'] = scopus_row['EID']
                    result['scopus_title'] = scopus_row['Title']
                    result['match_method'] = 'doi'
                    result['match_confidence'] = 1.0
                    result['confidence_tier'] = 'high'
                    
                    paper_info['title'] = scopus_row['Title']
                    paper_info['scopus_metadata'] = self._create_metadata(
                        scopus_row, 'doi', 1.0
                    )
                    
                    stats['doi'] += 1
                    logger.info(f"  ✓ DOI match: {doi}")
                    results.append(result)
                    continue
            
            # Strategy 2: Title matching
            title_match = self.match_by_titles(titles)
            if title_match:
                eid, score, matched_title, tier = title_match
                result['scopus_eid'] = eid
                result['match_method'] = 'title'
                result['match_confidence'] = score
                result['confidence_tier'] = tier
                
                scopus_row = self.scopus_df[self.scopus_df['EID'] == eid].iloc[0]
                result['scopus_title'] = scopus_row['Title']
                
                paper_info['title'] = scopus_row['Title']
                paper_info['scopus_metadata'] = self._create_metadata(
                    scopus_row, 'title', score
                )
                
                stats[f'title_{tier}'] += 1
                logger.info(f"  ✓ Title match ({tier}, {score:.2f}): {scopus_row['Title'][:50]}...")
                results.append(result)
                continue
            
            # Strategy 3: Abstract matching
            abstract = self.extract_abstract(paper_folder)
            abstract_match = self.match_by_abstract(abstract)
            
            if abstract_match:
                eid, score, tier = abstract_match
                result['scopus_eid'] = eid
                result['match_method'] = 'abstract'
                result['match_confidence'] = score
                result['confidence_tier'] = tier
                
                scopus_row = self.scopus_df[self.scopus_df['EID'] == eid].iloc[0]
                result['scopus_title'] = scopus_row['Title']
                
                paper_info['title'] = scopus_row['Title']
                paper_info['scopus_metadata'] = self._create_metadata(
                    scopus_row, 'abstract', score
                )
                
                stats['abstract'] += 1
                logger.info(f"  ✓ Abstract match ({tier}, {score:.2f}): {scopus_row['Title'][:50]}...")
                results.append(result)
                continue
            
            # No match found
            result['match_method'] = 'none'
            stats['unmatched'] += 1
            
            # Show best near-miss for debugging
            if titles:
                best_score = 0
                best_scopus = None
                for _, row in self.scopus_df.iterrows():
                    for t in titles[:3]:
                        score = SequenceMatcher(
                            None, 
                            self._normalize_text(t), 
                            self._normalize_text(str(row.get('Title', '')))
                        ).ratio()
                        if score > best_score:
                            best_score = score
                            best_scopus = row.get('Title', '')[:50]
                
                logger.warning(f"  ✗ No match (best: {best_score:.2f} '{best_scopus}...')")
            else:
                logger.warning(f"  ✗ No match (no titles extracted)")
            
            results.append(result)
        
        self._save_results(results, stats)
    
    def _create_metadata(self, scopus_row, method: str, confidence: float) -> dict:
        """Create scopus_metadata dict from Scopus row."""
        def safe_get(key, default=None):
            val = scopus_row.get(key)
            return val if pd.notna(val) else default
        
        return {
            'eid': safe_get('EID'),
            'doi': safe_get('DOI'),
            'title': safe_get('Title'),
            'authors': safe_get('Authors'),
            'year': int(safe_get('Year', 0)) if safe_get('Year') else None,
            'journal': safe_get('Source title'),
            'abstract': safe_get('Abstract'),
            'citations': int(safe_get('Cited by', 0)) if safe_get('Cited by') else 0,
            'affiliations': safe_get('Affiliations'),
            'author_keywords': safe_get('Author Keywords'),
            'index_keywords': safe_get('Index Keywords'),
            'link': safe_get('Link'),
            'match_method': method,
            'match_confidence': float(confidence),
            'source': 'Scopus + MinerU'
        }
    
    def _save_results(self, results: List[dict], stats: dict):
        """Save enhanced mapping and reports."""
        
        # Save enhanced paper_mapping.json
        with open(self.mapping_file, 'w', encoding='utf-8') as f:
            json.dump(self.paper_mapping, f, indent=2, ensure_ascii=False)
        
        # Save match report
        report_df = pd.DataFrame(results)
        report_path = self.output_dir / "paper_scopus_matches.csv"
        report_df.to_csv(report_path, index=False)
        
        # Calculate stats
        total = len(results)
        matched = (stats['doi'] + stats['title_high'] + stats['title_medium'] + 
                   stats['title_low'] + stats['abstract'])
        rate = (matched / total * 100) if total > 0 else 0
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("RESULTS")
        logger.info("=" * 70)
        logger.info(f"Total:           {total}")
        logger.info(f"Matched:         {matched} ({rate:.1f}%)")
        logger.info(f"  DOI:           {stats['doi']}")
        logger.info(f"  Title (high):  {stats['title_high']}")
        logger.info(f"  Title (medium):{stats['title_medium']}")
        logger.info(f"  Title (low):   {stats['title_low']}")
        logger.info(f"  Abstract:      {stats['abstract']}")
        logger.info(f"Unmatched:       {stats['unmatched']}")
        logger.info("")
        logger.info(f"Output: {self.mapping_file}")
        logger.info(f"Report: {report_path}")
        
        # Save unmatched for manual review
        if stats['unmatched'] > 0:
            unmatched_df = report_df[report_df['match_method'] == 'none'][
                ['paper_id', 'extracted_title', 'doi_extracted']
            ]
            manual_path = self.output_dir / "manual_review_needed.csv"
            unmatched_df.to_csv(manual_path, index=False)
            logger.info(f"Manual review: {manual_path}")
        
        # Save low-confidence matches for review
        low_conf = report_df[report_df['confidence_tier'] == 'low']
        if len(low_conf) > 0:
            low_path = self.output_dir / "low_confidence_matches.csv"
            low_conf.to_csv(low_path, index=False)
            logger.info(f"Low confidence: {low_path}")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Run matcher from command line."""
    from dotenv import load_dotenv
    load_dotenv()
    
    setup_logging(log_file="logs/scopus_matching.log")
    
    matcher = MinerUMatcher(
        papers_path="data/raw/academic/scopus_2023/MinerU_parsed_papers",
        scopus_csv="data/raw/academic/scopus_2023/scopus_export_2023_raw.csv"
    )
    
    matcher.match_all()


if __name__ == "__main__":
    main()