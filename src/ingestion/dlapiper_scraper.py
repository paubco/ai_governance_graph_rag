# -*- coding: utf-8 -*-
"""
DLA

Scrapes regulatory content from DLA Piper's AI Laws of the World database.
Extracts country-specific AI regulations with section-level granularity.
Outputs JSON files per country with regulatory sections, notes, and subsections.

"""
# Standard library
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
import requests
from bs4 import BeautifulSoup

# Local
from src.configs.config import SCRAPER_CONFIG, SCRAPER_LOGS_PATH
from src.utils.logger import setup_logging


# ============================================================================
# LOGGING SETUP
# ============================================================================
setup_logging(
    log_file=str(SCRAPER_LOGS_PATH / f"dlapiper_scraper_{datetime.now().strftime('%Y%m%d')}.log")
)
logger = logging.getLogger(__name__)


# ============================================================================
# SCRAPER CLASS
# ============================================================================
class DLAPiperScraper:
    """
    Scraper for DLA Piper AI Laws of the World
    
    Simple architecture:
    - Get country list from dropdown (has BOTH code and name)
    - Scrape each country page
    - Extract sections from HTML
    - Save as JSON + TXT
    """
    
    def __init__(self):
        """Initialize scraper with project configuration"""
        self.base_url = SCRAPER_CONFIG["base_url"]
        self.output_dir = Path(SCRAPER_CONFIG["output_dir"])
        self.delay = SCRAPER_CONFIG["delay_between_requests"]
        self.timeout = SCRAPER_CONFIG["timeout"]
        self.retry_attempts = SCRAPER_CONFIG["retry_attempts"]
        self.headers = SCRAPER_CONFIG["headers"]
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DLA Piper Scraper initialized")
        logger.info(f"Output directory: {self.output_dir}")
    
    def get_country_list(self) -> List[Dict[str, str]]:
        """
        Extract country codes AND names from the main page dropdown
        
        Returns:
            List of dicts: [{'code': 'FR', 'name': 'France'}, ...]
        """
        logger.info("Fetching country list from main page...")
        
        try:
            response = requests.get(
                self.base_url,
                timeout=self.timeout,
                headers=self.headers
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            
            # The dropdown already has BOTH code and name!
            country_options = soup.select('select[name="c"] option[value]')
            
            countries = []
            for option in country_options:
                code = option.get('value')
                name = option.text.strip()
                
                if code and code != "":
                    countries.append({'code': code, 'name': name})
            
            logger.info(f"Found {len(countries)} countries")
            return countries
            
        except Exception as e:
            logger.error(f"Error fetching country list: {e}")
            return []
    
    def scrape_country(self, country_code: str, country_name: str) -> Optional[Dict]:
        """
        Scrape all sections for a specific country
        
        Args:
            country_code: Two-letter code (e.g., 'FR')
            country_name: Full name (e.g., 'France')
            
        Returns:
            Dictionary with country data or None if failed
        """
        url = f"{self.base_url}?c={country_code}"
        logger.info(f"Scraping {country_name} ({country_code})")
        
        for attempt in range(self.retry_attempts):
            try:
                response = requests.get(url, timeout=self.timeout, headers=self.headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'lxml')
                
                # Extract sections
                sections = self._extract_sections(soup)
                
                # Build country data
                country_data = {
                    'country_code': country_code,
                    'country_name': country_name,
                    'url': url,
                    'scraped_date': datetime.now().isoformat(),
                    'num_sections': len(sections),
                    'sections': sections,  # Links are already in each section!
                }
                
                logger.info(f"Successfully scraped {country_code}: {len(sections)} sections")
                return country_data
                
            except requests.Timeout:
                logger.warning(f"Timeout for {country_code} (attempt {attempt + 1}/{self.retry_attempts})")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.delay * 2)
                    continue
            except Exception as e:
                logger.error(f"Error scraping {country_code}: {e}")
                return None
        
        logger.error(f"Failed to scrape {country_code} after {self.retry_attempts} attempts")
        return None
    
    def _extract_sections(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract all accordion sections from the page"""
        accordion_items = soup.find_all('div', class_='page-nav-accordion__item')
        
        sections = []
        for item in accordion_items:
            section_data = self._parse_section(item)
            if section_data:
                # Filter out contacts section - we don't need it
                if section_data['title'].lower() in ['key contacts', 'contacts']:
                    logger.debug(f"Skipping contacts section")
                    continue
                sections.append(section_data)
        
        return sections
    
    def _parse_section(self, accordion_item) -> Optional[Dict]:
        """Parse a single accordion section"""
        try:
            # Get section title
            title_elem = accordion_item.find('a', class_='page-nav-accordion__title')
            if not title_elem:
                return None
            
            section_title = title_elem.text.strip()
            
            # Get section content
            content_elem = accordion_item.find('div', class_='page-nav-accordion__content')
            if not content_elem:
                return None
            
            # Extract heading
            h1 = content_elem.find('h1')
            section_heading = h1.text.strip() if h1 else section_title
            
            # Extract content (excluding notes)
            main_content = self._extract_content(content_elem, exclude_class='note')
            
            # Extract notes (EU countries use these)
            notes = self._extract_notes(content_elem)
            
            # Extract subsections (US states, Canadian provinces use these)
            subsections = self._extract_subsections(content_elem)
            
            # Extract links
            links = self._extract_section_links(content_elem)
            
            return {
                'title': section_title,
                'heading': section_heading,
                'main_content': main_content,
                'country_specific_notes': notes,
                'subsections': subsections,
                'links': links,
            }
            
        except Exception as e:
            logger.error(f"Error parsing section: {e}")
            return None
    
    def _extract_content(self, element, exclude_class: Optional[str] = None) -> str:
        """
        Extract clean text content from an element
        Excludes content within specified class and H4 subsections
        """
        content = element.find_all(['p', 'ul', 'ol', 'blockquote', 'h2', 'h3'])
        
        text_parts = []
        for elem in content:
            # Skip if inside excluded class
            if exclude_class and elem.find_parent(class_=exclude_class):
                continue
            
            # Skip if this element comes after an H4 (it's part of a subsection)
            # Check if there's an H4 before this element at the same level
            previous_h4 = None
            for sibling in elem.previous_siblings:
                if sibling.name == 'h4':
                    previous_h4 = sibling
                    break
                # If we hit another major heading, stop looking
                if sibling.name in ['h2', 'h3']:
                    break
            
            # If there's an H4 before us, skip this content (it belongs to subsection)
            if previous_h4:
                continue
            
            text = elem.get_text(strip=True)
            if text:
                text_parts.append(text)
        
        return '\n\n'.join(text_parts)
    
    def _extract_notes(self, content_elem) -> List[Dict]:
        """Extract country-specific notes"""
        notes = []
        note_divs = content_elem.find_all('div', class_='note')
        
        for note_div in note_divs:
            note_heading = note_div.find('h3')
            note_text = self._extract_content(note_div)
            notes.append({
                'heading': note_heading.text.strip() if note_heading else '',
                'content': note_text
            })
        
        return notes
    
    def _extract_subsections(self, content_elem) -> List[Dict]:
        """
        Extract subsections marked by H4 headings (e.g., US states, Canadian provinces)
        
        Returns:
            List of subsections with heading and content
        """
        subsections = []
        h4_elements = content_elem.find_all('h4')
        
        for h4 in h4_elements:
            heading = h4.text.strip()
            
            # Get content between this H4 and the next H4 (or end)
            content_parts = []
            for sibling in h4.find_next_siblings():
                # Stop at next H4 or major heading
                if sibling.name in ['h4', 'h2', 'h3']:
                    break
                
                # Collect text from paragraphs and lists
                if sibling.name in ['p', 'ul', 'ol']:
                    text = sibling.get_text(strip=True)
                    if text:
                        content_parts.append(text)
            
            if content_parts:
                subsections.append({
                    'heading': heading,
                    'content': '\n\n'.join(content_parts)
                })
        
        return subsections
    
    def _extract_section_links(self, content_elem) -> List[Dict]:
        """Extract links within a section"""
        links = []
        for a in content_elem.find_all('a', href=True):
            href = a.get('href')
            if href and not href.startswith('#'):
                links.append({
                    'text': a.text.strip(),
                    'url': href
                })
        return links
    
    
    def save_country_data(self, country_data: Dict) -> None:
        """
        Save scraped country data to JSON file
        
        Args:
            country_data: Dictionary with country data
        """
        if not country_data:
            return
        
        country_code = country_data['country_code']
        
        # Save JSON (for pipeline)
        json_path = self.output_dir / f"{country_code}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(country_data, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved JSON: {json_path}")
    
    def scrape_all_countries(self) -> Dict:
        """Scrape all countries with rate limiting"""
        countries = self.get_country_list()
        
        if not countries:
            logger.error("No countries found. Aborting.")
            return {
                'total_countries': 0,
                'successful': 0,
                'failed': 0,
                'countries': []
            }
        
        logger.info(f"Starting scrape of {len(countries)} countries...")
        logger.info(f"Estimated time: {len(countries) * self.delay / 60:.1f} minutes")
        
        results = {
            'total_countries': len(countries),
            'successful': 0,
            'failed': 0,
            'countries': [],
            'start_time': datetime.now().isoformat(),
        }
        
        for i, country in enumerate(countries, 1):
            logger.info(f"[{i}/{len(countries)}] Processing {country['name']}...")
            
            # Pass BOTH code and name to scrape_country()
            country_data = self.scrape_country(
                country['code'],
                country['name']  # ← We already have it from get_country_list()!
            )
            
            if country_data:
                self.save_country_data(country_data)
                results['successful'] += 1
                results['countries'].append({
                    'code': country['code'],
                    'name': country['name'],
                    'status': 'success',
                    'sections': country_data['num_sections']
                })
            else:
                results['failed'] += 1
                results['countries'].append({
                    'code': country['code'],
                    'name': country['name'],
                    'status': 'failed'
                })
            
            # Rate limiting
            if i < len(countries):
                time.sleep(self.delay)
        
        results['end_time'] = datetime.now().isoformat()
        
        # Save summary
        summary_path = self.output_dir / 'scraping_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Scraping complete!")
        logger.info(f"Successful: {results['successful']}/{results['total_countries']}")
        logger.info(f"Failed: {results['failed']}/{results['total_countries']}")
        logger.info(f"Summary saved to: {summary_path}")
        logger.info(f"{'=' * 80}\n")
        
        return results


# ============================================================================
# CONVENIENCE FUNCTIONS FOR TESTING
# ============================================================================
def test_single_country(country_code: str = 'FR') -> Optional[Dict]:
    """Test scraping a single country"""
    logger.info(f"Testing single country scrape: {country_code}")
    
    scraper = DLAPiperScraper()
    
    # For testing, we need to get the country name
    # In production, this comes from get_country_list()
    countries = scraper.get_country_list()
    country = next((c for c in countries if c['code'] == country_code), None)
    
    if not country:
        logger.error(f"Country {country_code} not found in dropdown")
        return None
    
    country_data = scraper.scrape_country(country['code'], country['name'])
    
    if country_data:
        scraper.save_country_data(country_data)
        logger.info(f"\nSuccessfully scraped {country_code}")
        logger.info(f"Found {country_data['num_sections']} sections")
        logger.info(f"Files saved to: {scraper.output_dir}")
        return country_data
    else:
        logger.error(f"\nFailed to scrape {country_code}")
        return None


def main():
    """Main scraping function for production use"""
    logger.info("=" * 80)
    logger.info("DLA PIPER AI LAWS SCRAPER - PRODUCTION MODE")
    logger.info("=" * 80)
    
    scraper = DLAPiperScraper()
    results = scraper.scrape_all_countries()
    
    return results


if __name__ == "__main__":
    # For testing: Scrape just France
    # test_single_country('FR')
    
    # For production: Scrape all countries
    main()