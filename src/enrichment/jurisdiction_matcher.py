# -*- coding: utf-8 -*-
"""
Jurisdiction

Maps country and region entities to jurisdiction codes via SAME_AS relationships.
Only links entities that represent the jurisdiction itself (not organizations
like CNIL or FTC).

Examples:
matcher = JurisdictionMatcher(valid_codes)
        matches = matcher.match_entities(entities)
        # Returns: [{"entity_id": "ent_123", "jurisdiction_code": "EU"}, ...]

References:
    See ARCHITECTURE.md § 3.2.1 for Phase 2A context
    See PHASE_2A_DESIGN.md for matching pipeline

"""
# Standard library
import json
from pathlib import Path
from typing import List, Dict, Set

# Project imports
from src.utils.id_generator import generate_entity_id
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# JURISDICTION NAME MAPPING
# =============================================================================

# Direct name → code mapping
# Built from scraping_summary.json + common variants (max 2-3 per jurisdiction)
JURISDICTION_MAP = {
    # Official names from DLA Piper
    "Australia": "AU",
    "Austria": "AT",
    "Belgium": "BE",
    "Brazil": "BR",
    "Bulgaria": "BG",
    "Canada": "CA",
    "Chile": "CL",
    "China": "CN",
    "Croatia": "HR",
    "Cyprus": "CY",
    "Czech Republic": "CZ",
    "Denmark": "DK",
    "Estonia": "EE",
    "European Union": "EU",
    "Finland": "FI",
    "France": "FR",
    "Germany": "DE",
    "Greece": "GR",
    "Hong Kong, SAR": "HK",
    "Hungary": "HU",
    "Ireland": "IE",
    "Italy": "IT",
    "Japan": "JP",
    "Latvia": "LV",
    "Lithuania": "LT",
    "Luxembourg": "LU",
    "Malta": "MT",
    "Mauritius": "MU",
    "Mexico": "MX",
    "Netherlands": "NL",
    "New Zealand": "NZ",
    "Nigeria": "NG",
    "Norway": "NO",
    "Peru": "PE",
    "Poland": "PL",
    "Portugal": "PT",
    "Romania": "RO",
    "Singapore": "SG",
    "Slovak Republic": "SK",
    "Slovenia": "SI",
    "South Korea": "KR",
    "Spain": "ES",
    "Sweden": "SE",
    "Thailand": "TH",
    "Turkey": "TR",
    "United Arab Emirates": "AE",
    "United Kingdom": "GB",
    "United States": "US",
    
    # Common variants (2-3 max per jurisdiction)
    "EU": "EU",
    "USA": "US",
    "US": "US",
    "United States of America": "US",
    "UK": "GB",
    "Great Britain": "GB",
    "Britain": "GB",
    "UAE": "AE",
    "Hong Kong": "HK",
    "HK": "HK",
    "PRC": "CN",
    "People's Republic of China": "CN",
    "Korea": "KR",
    "Czechia": "CZ",
    "Slovakia": "SK",
}


# =============================================================================
# JURISDICTION MATCHER
# =============================================================================

class JurisdictionMatcher:
    """
    Matches country/region entities to jurisdiction codes.
    
    Uses direct name lookup for entities that ARE the jurisdiction.
    Not for organizations (CNIL, FTC) - those stay unlinked.
    
    Example:
        matcher = JurisdictionMatcher(valid_codes)
        matches = matcher.match_entities(entities)
        # Returns: [{"entity_id": "ent_123", "jurisdiction_code": "EU"}, ...]
    """
    
    def __init__(self, valid_codes: Set[str]):
        """
        Initialize matcher with valid jurisdiction codes.
        
        Args:
            valid_codes: Set of valid 2-letter codes (AU, EU, US, etc.)
        """
        self.valid_codes = valid_codes
        self.name_map = JURISDICTION_MAP
    
    def match_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Match country/region entities to jurisdiction codes.
        
        Args:
            entities: Normalized entities from Phase 1C
            
        Returns:
            List of matches: [{"entity_id": "...", "jurisdiction_code": "..."}, ...]
        """
        matches = []
        
        for entity in entities:
            entity_name = entity['name']
            entity_id = entity.get('entity_id')
            entity_type = entity.get('type', '')
            
            # Generate entity_id if missing (legacy compatibility)
            if not entity_id:
                entity_id = generate_entity_id(entity_name, entity_type)
            
            # Direct name lookup
            if entity_name in self.name_map:
                code = self.name_map[entity_name]
                
                # Verify code is valid (in scraped data)
                if code in self.valid_codes:
                    matches.append({
                        'entity_id': entity_id,
                        'entity_name': entity_name,
                        'jurisdiction_code': code
                    })
        
        logger.info(f"Matched {len(matches)} entities to jurisdictions")
        
        return matches
    
    @staticmethod
    def load_valid_codes(scraping_summary_path: Path) -> Set[str]:
        """
        Load valid jurisdiction codes from scraping summary.
        
        Args:
            scraping_summary_path: Path to scraping_summary.json
            
        Returns:
            Set of 2-letter codes
        """
        with open(scraping_summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        codes = {country['code'] for country in data['countries']}
        logger.info(f"Loaded {len(codes)} valid jurisdiction codes")
        
        return codes