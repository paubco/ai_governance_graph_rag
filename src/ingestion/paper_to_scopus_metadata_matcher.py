# -*- coding: utf-8 -*-
"""
MINERU METADATA MATCHER
Links MinerU papers to Scopus metadata using:
1. DOI from content_list.json (footer/header)
2. Title matching (handles multi-language titles)
3. Abstract matching (fallback)

OUTPUT: data/interim/academic/
"""

import pandas as pd
import numpy as np
import json
import re
import logging
from pathlib import Path
from typing import Optional, Tuple, List
from difflib import SequenceMatcher

class MinerUMatcher:
    """Match MinerU papers to Scopus metadata"""
    
    def __init__(self, papers_path: str, scopus_csv: str):
        self.papers_path = Path(papers_path)
        self.scopus_csv = Path(scopus_csv)
        self.mapping_file = self.papers_path.parent / "paper_mapping.json"
        
        # Output to interim
        self.output_dir = Path("data/interim/academic")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Scopus data
        self.scopus_df = pd.read_csv(self.scopus_csv, encoding='utf-8-sig')
        
        # Create fresh mapping from paper folders (ignore old mapping)
        self.paper_mapping = {}
        for paper_folder in sorted(self.papers_path.glob('paper_*')):
            if paper_folder.is_dir():
                self.paper_mapping[paper_folder.name] = {}
    
        # DOI patterns
        self.doi_pattern = re.compile(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', re.IGNORECASE)
        self.doi_url_pattern = re.compile(r'https?://doi\.org/(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', re.IGNORECASE)
        
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        return logger
    
    def _normalize_text(self, text: str) -> str:
        """Lowercase, remove punctuation, normalize spaces"""
        if not text or text == "Unknown":
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_titles(self, paper_folder: Path) -> List[str]:
        """
        Extract all titles from content_list.json
        Papers may have multiple titles (e.g., different languages)
        """
        titles = []
        
        json_path = paper_folder / "content_list.json"
        if not json_path.exists():
            return titles
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                content_list = json.load(f)
            
            # Extract all text items with text_level: 1 (titles/main headings)
            for item in content_list:
                if item.get('text_level') == 1:
                    text = item.get('text', '').strip()
                    if text and len(text) > 10:  # Skip short headers
                        titles.append(text)
        
        except Exception as e:
            self.logger.debug(f"Error extracting titles: {e}")
        
        return titles
    
    def extract_abstract(self, paper_folder: Path) -> Optional[str]:
        """Extract abstract text from markdown"""
        md_path = paper_folder / "full.md"
        if not md_path.exists():
            return None
        
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find abstract section (case insensitive)
            pattern = r'#\s*ABSTRACT\s*\n\n(.+?)(?=\n#|$)'
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            
            if match:
                abstract = match.group(1).strip()
                # Clean up: remove extra whitespace, limit length
                abstract = re.sub(r'\s+', ' ', abstract)
                return abstract[:1000]  # First 1000 chars
        
        except Exception as e:
            self.logger.debug(f"Error extracting abstract: {e}")
        
        return None
    
    def extract_doi(self, paper_folder: Path) -> Optional[str]:
        """Extract DOI from content_list.json or markdown"""
        
        # Priority 1: content_list.json footer/header
        json_path = paper_folder / "content_list.json"
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    content_list = json.load(f)
                
                for item in content_list:
                    item_type = item.get('type', '')
                    text = item.get('text', '')
                    
                    if item_type in ['footer', 'header', 'text']:
                        # Try full DOI URL
                        url_match = self.doi_url_pattern.search(text)
                        if url_match:
                            return url_match.group(1)
                        
                        # Try plain DOI
                        doi_match = self.doi_pattern.search(text)
                        if doi_match and text.count('10.') == 1:
                            return doi_match.group(0)
            
            except Exception as e:
                self.logger.debug(f"Error reading JSON: {e}")
        
        # Priority 2: markdown fallback
        md_path = paper_folder / "full.md"
        if md_path.exists():
            try:
                with open(md_path, 'r', encoding='utf-8') as f:
                    lines = [f.readline() for _ in range(100)]
                    text = ''.join(lines)
                    
                    url_match = self.doi_url_pattern.search(text)
                    if url_match:
                        return url_match.group(1)
                    
                    doi_match = self.doi_pattern.search(text)
                    if doi_match:
                        return doi_match.group(0)
            
            except Exception as e:
                self.logger.debug(f"Error reading markdown: {e}")
        
        return None
    
    def match_by_titles(self, paper_titles: List[str]) -> Optional[Tuple[str, float, str]]:
        """
        Match by title similarity using multiple titles
        Returns: (eid, score, matched_title)
        """
        if not paper_titles:
            return None
        
        best_eid = None
        best_score = 0
        best_title = None
        
        # Try each title variant
        for paper_title in paper_titles[:3]:  # Try first 3 titles
            paper_norm = self._normalize_text(paper_title)
            if not paper_norm:
                continue
            
            for _, row in self.scopus_df.iterrows():
                scopus_title = self._normalize_text(str(row['Title']))
                score = SequenceMatcher(None, paper_norm, scopus_title).ratio()
                
                if score > best_score:
                    best_score = score
                    best_eid = row['EID']
                    best_title = paper_title
        
        # Threshold: 0.85
        if best_score >= 0.85:
            return (best_eid, best_score, best_title)
        
        return None
    
    def match_by_abstract(self, paper_abstract: str) -> Optional[Tuple[str, float]]:
        """
        Match by abstract similarity (fallback)
        Returns: (eid, score)
        """
        if not paper_abstract:
            return None
        
        paper_norm = self._normalize_text(paper_abstract)
        if len(paper_norm) < 50:  # Too short
            return None
        
        best_eid = None
        best_score = 0
        
        for _, row in self.scopus_df.iterrows():
            scopus_abstract = self._normalize_text(str(row.get('Abstract', '')))
            if len(scopus_abstract) < 50:
                continue
            
            # Compare first 500 chars of normalized abstracts
            score = SequenceMatcher(None, paper_norm[:500], scopus_abstract[:500]).ratio()
            
            if score > best_score:
                best_score = score
                best_eid = row['EID']
        
        # Higher threshold for abstracts (0.90)
        if best_score >= 0.90:
            return (best_eid, best_score)
        
        return None
    
    def match_all(self):
        """Match all papers to Scopus metadata"""
        
        self.logger.info("=" * 70)
        self.logger.info("🔗 MATCHING MINERU PAPERS TO SCOPUS")
        self.logger.info("=" * 70)
        self.logger.info(f"Papers:        {len(self.paper_mapping)}")
        self.logger.info(f"Scopus records: {len(self.scopus_df)}")
        self.logger.info("")
        
        stats = {'doi': 0, 'title': 0, 'abstract': 0, 'unmatched': 0}
        results = []
        
        for paper_id, paper_info in self.paper_mapping.items():
            paper_folder = self.papers_path / paper_id
            
            result = {
                'paper_id': paper_id,
                'original_title': paper_info.get('title', 'Unknown'),
                'scopus_eid': None,
                'doi_extracted': None,
                'match_method': None,
                'match_confidence': 0.0
            }
            
            self.logger.info(f"[{paper_id}] {result['original_title'][:60]}...")
            
            # Strategy 1: DOI
            doi = self.extract_doi(paper_folder)
            if doi:
                result['doi_extracted'] = doi
                matched = self.scopus_df[
                    self.scopus_df['DOI'].str.lower() == doi.lower()
                ]
                
                if not matched.empty:
                    scopus_row = matched.iloc[0]
                    result['scopus_eid'] = scopus_row['EID']
                    result['match_method'] = 'doi'
                    result['match_confidence'] = 1.0
                    
                    paper_info['scopus_metadata'] = self._create_metadata(scopus_row, 'doi', 1.0)
                    
                    stats['doi'] += 1
                    self.logger.info(f"  ✓ DOI: {doi}")
                    results.append(result)
                    continue
            
            # Strategy 2: Title matching (try multiple titles)
            titles = self.extract_titles(paper_folder)
            if not titles:
                titles = [paper_info.get('title', 'Unknown')]
            
            title_match = self.match_by_titles(titles)
            if title_match:
                eid, score, matched_title = title_match
                result['scopus_eid'] = eid
                result['match_method'] = 'title'
                result['match_confidence'] = score
                
                scopus_row = self.scopus_df[self.scopus_df['EID'] == eid].iloc[0]
                paper_info['scopus_metadata'] = self._create_metadata(scopus_row, 'title', score)
                
                stats['title'] += 1
                self.logger.info(f"  ✓ Title: {matched_title[:50]} (sim={score:.2f})")
                results.append(result)
                continue
            
            # Strategy 3: Abstract matching (fallback)
            abstract = self.extract_abstract(paper_folder)
            abstract_match = self.match_by_abstract(abstract)
            
            if abstract_match:
                eid, score = abstract_match
                result['scopus_eid'] = eid
                result['match_method'] = 'abstract'
                result['match_confidence'] = score
                
                scopus_row = self.scopus_df[self.scopus_df['EID'] == eid].iloc[0]
                paper_info['scopus_metadata'] = self._create_metadata(scopus_row, 'abstract', score)
                
                stats['abstract'] += 1
                self.logger.info(f"  ✓ Abstract (sim={score:.2f})")
                results.append(result)
                continue
            
            # No match
            result['match_method'] = 'none'
            stats['unmatched'] += 1
            self.logger.warning(f"  ✗ No match")
            results.append(result)
        
        self._save_results(results, stats)
    
    def _create_metadata(self, scopus_row, method: str, confidence: float) -> dict:
        """Create scopus_metadata dict from Scopus row"""
        return {
            'eid': scopus_row['EID'] if pd.notna(scopus_row['EID']) else None,
            'doi': scopus_row.get('DOI') if pd.notna(scopus_row.get('DOI')) else None,
            'authors': scopus_row.get('Authors') if pd.notna(scopus_row.get('Authors')) else None,
            'year': int(scopus_row['Year']) if pd.notna(scopus_row['Year']) else None,
            'journal': scopus_row.get('Source title') if pd.notna(scopus_row.get('Source title')) else None,
            'abstract': scopus_row.get('Abstract') if pd.notna(scopus_row.get('Abstract')) else None,
            'citations': int(scopus_row.get('Cited by', 0)) if pd.notna(scopus_row.get('Cited by')) else 0,
            'match_method': method,
            'match_confidence': float(confidence)
        }
    def _save_results(self, results, stats):
        """Save enhanced mapping and reports"""
        
        # Save enhanced paper_mapping.json
        with open(self.mapping_file, 'w', encoding='utf-8') as f:
            json.dump(self.paper_mapping, f, indent=2, ensure_ascii=False)
        
        # Save match report
        report_df = pd.DataFrame(results)
        report_path = self.output_dir / "paper_scopus_matches.csv"
        report_df.to_csv(report_path, index=False)
        
        # Print stats
        total = len(results)
        matched = stats['doi'] + stats['title'] + stats['abstract']
        rate = (matched / total * 100) if total > 0 else 0
        
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("📊 RESULTS")
        self.logger.info("=" * 70)
        self.logger.info(f"Total:     {total}")
        self.logger.info(f"Matched:   {matched} ({rate:.1f}%)")
        self.logger.info(f"  DOI:     {stats['doi']}")
        self.logger.info(f"  Title:   {stats['title']}")
        self.logger.info(f"  Abstract: {stats['abstract']}")
        self.logger.info(f"Unmatched: {stats['unmatched']}")
        self.logger.info("")
        self.logger.info(f"✓ Created: {self.mapping_file}")
        self.logger.info(f"✓ Report:   {report_path}")
        
        # Save unmatched for manual review
        if stats['unmatched'] > 0:
            unmatched_df = report_df[report_df['scopus_eid'].isna()][
                ['paper_id', 'original_title', 'doi_extracted']
            ]
            manual_path = self.output_dir / "manual_review_needed.csv"
            unmatched_df.to_csv(manual_path, index=False)
            self.logger.info(f"✓ Manual:   {manual_path}")


def main():
    matcher = MinerUMatcher(
        papers_path=r"data/raw/academic/scopus_2023/MinerU_parsed_papers",
        scopus_csv=r"data/raw/academic/scopus_2023/scopus_export_2023_raw.csv"
    )
    
    matcher.match_all()


if __name__ == "__main__":
    main()