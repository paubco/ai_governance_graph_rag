# -*- coding: utf-8 -*-
"""
Shared constants for AI Governance GraphRAG Pipeline

Central location for entity types, jurisdiction codes, and other constants used
across multiple pipeline phases. Includes entity type definitions for 
48 jurisdiction codes from DLA Piper plus EU, and helper functions for normalization
and validation.

"""
from typing import Dict, List, Set


# ============================================================================
# ENTITY TYPES
# ============================================================================

# Current types from v1.0 data (24 types with overlaps)
# TODO: Consolidate during v1.1 type enforcement phase
ENTITY_TYPES_V1 = [
    "Concept",
    "Author",
    "Journal", 
    "Technology",
    "Article",
    "Organization",
    "Process",
    "Publication",
    "Document",
    "Book",
    "Institution",
    "Person",
    "Regulatory Concept",
    "Regulation",
    "Paper",
    "Legal Concept",
    "Methodology",
    "Title",
    "Regulatory Document",
    "Technical Term",
    "Role",
    "Location",
    "Metric",
    "Conference",
    "Country",
    "Funding",
    "Data Statistic",
    "Report",
]

# Proposed v1.1 canonical types (to be finalized)
# Clustering logic: maximize coverage, minimize overlap
ENTITY_TYPES_V1_1_DRAFT = [
    # Core domain types
    "Regulation",       # Laws, acts, directives (merges Regulatory Document, Legal Concept)
    "Concept",          # Abstract ideas (merges Regulatory Concept, Technical Term)
    "Technology",       # Tech/methods, AI systems
    "Organization",     # Orgs, institutions, companies (merges Institution)
    "Person",           # Named individuals (merges Author for non-Scopus)
    "Location",         # Countries, regions, cities (merges Country)
    
    # Academic types (from Scopus metadata)
    "Publication",      # Academic papers (merges Paper, Article, Document, Book)
    "Author",           # Scopus authors only (linked to Publication)
    "Journal",          # Publication venues (merges Conference)
    
    # Supporting types
    "Process",          # Procedures, workflows
    "Methodology",      # Research methods
    "Metric",           # Measurements, statistics (merges Data Statistic)
    "Role",             # Job titles, positions
]


# ============================================================================
# JURISDICTION CODES
# ============================================================================

# 48 jurisdictions from DLA Piper + EU
JURISDICTION_CODES: List[str] = [
    "AR",  # Argentina
    "AU",  # Australia
    "AT",  # Austria
    "BE",  # Belgium
    "BR",  # Brazil
    "CA",  # Canada
    "CL",  # Chile
    "CN",  # China
    "CO",  # Colombia
    "CZ",  # Czech Republic
    "DK",  # Denmark
    "EG",  # Egypt
    "EU",  # European Union
    "FI",  # Finland
    "FR",  # France
    "DE",  # Germany
    "GR",  # Greece
    "HK",  # Hong Kong
    "HU",  # Hungary
    "IN",  # India
    "ID",  # Indonesia
    "IE",  # Ireland
    "IL",  # Israel
    "IT",  # Italy
    "JP",  # Japan
    "KE",  # Kenya
    "LU",  # Luxembourg
    "MY",  # Malaysia
    "MX",  # Mexico
    "NL",  # Netherlands
    "NZ",  # New Zealand
    "NG",  # Nigeria
    "NO",  # Norway
    "PK",  # Pakistan
    "PE",  # Peru
    "PH",  # Philippines
    "PL",  # Poland
    "PT",  # Portugal
    "RO",  # Romania
    "SA",  # Saudi Arabia
    "SG",  # Singapore
    "ZA",  # South Africa
    "KR",  # South Korea
    "ES",  # Spain
    "SE",  # Sweden
    "CH",  # Switzerland
    "TW",  # Taiwan
    "AE",  # UAE
    "GB",  # United Kingdom
    "US",  # United States
]

# Common aliases for jurisdictions (for query parsing)
JURISDICTION_ALIASES: Dict[str, str] = {
    # Full names
    "European Union": "EU",
    "United States": "US",
    "United Kingdom": "GB",
    "United Arab Emirates": "AE",
    "South Korea": "KR",
    "South Africa": "ZA",
    "New Zealand": "NZ",
    "Hong Kong": "HK",
    "Saudi Arabia": "SA",
    "Czech Republic": "CZ",
    
    # Common abbreviations
    "UK": "GB",
    "USA": "US",
    "America": "US",
    "UAE": "AE",
    "Korea": "KR",
    
    # Country names
    "Germany": "DE",
    "France": "FR",
    "Spain": "ES",
    "Italy": "IT",
    "China": "CN",
    "Japan": "JP",
    "Brazil": "BR",
    "India": "IN",
    "Australia": "AU",
    "Canada": "CA",
    "Mexico": "MX",
    "Argentina": "AR",
    "Netherlands": "NL",
    "Belgium": "BE",
    "Switzerland": "CH",
    "Sweden": "SE",
    "Norway": "NO",
    "Denmark": "DK",
    "Finland": "FI",
    "Poland": "PL",
    "Austria": "AT",
    "Portugal": "PT",
    "Ireland": "IE",
    "Greece": "GR",
    "Hungary": "HU",
    "Romania": "RO",
    "Israel": "IL",
    "Egypt": "EG",
    "Nigeria": "NG",
    "Kenya": "KE",
    "Singapore": "SG",
    "Malaysia": "MY",
    "Indonesia": "ID",
    "Philippines": "PH",
    "Taiwan": "TW",
    "Pakistan": "PK",
    "Chile": "CL",
    "Colombia": "CO",
    "Peru": "PE",
    "Luxembourg": "LU",
}

# Reverse lookup: code -> full name
JURISDICTION_NAMES: Dict[str, str] = {
    "AR": "Argentina",
    "AU": "Australia",
    "AT": "Austria",
    "BE": "Belgium",
    "BR": "Brazil",
    "CA": "Canada",
    "CL": "Chile",
    "CN": "China",
    "CO": "Colombia",
    "CZ": "Czech Republic",
    "DK": "Denmark",
    "EG": "Egypt",
    "EU": "European Union",
    "FI": "Finland",
    "FR": "France",
    "DE": "Germany",
    "GR": "Greece",
    "HK": "Hong Kong",
    "HU": "Hungary",
    "IN": "India",
    "ID": "Indonesia",
    "IE": "Ireland",
    "IL": "Israel",
    "IT": "Italy",
    "JP": "Japan",
    "KE": "Kenya",
    "LU": "Luxembourg",
    "MY": "Malaysia",
    "MX": "Mexico",
    "NL": "Netherlands",
    "NZ": "New Zealand",
    "NG": "Nigeria",
    "NO": "Norway",
    "PK": "Pakistan",
    "PE": "Peru",
    "PH": "Philippines",
    "PL": "Poland",
    "PT": "Portugal",
    "RO": "Romania",
    "SA": "Saudi Arabia",
    "SG": "Singapore",
    "ZA": "South Africa",
    "KR": "South Korea",
    "ES": "Spain",
    "SE": "Sweden",
    "CH": "Switzerland",
    "TW": "Taiwan",
    "AE": "UAE",
    "GB": "United Kingdom",
    "US": "United States",
}


# ============================================================================
# DOCUMENT TYPES
# ============================================================================

DOC_TYPES = ["regulation", "paper"]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_jurisdiction(text: str) -> str:
    """
    Normalize jurisdiction text to ISO code.
    
    Args:
        text: Jurisdiction name or code
        
    Returns:
        ISO 2-letter code, or original text if not found
    """
    text_upper = text.upper().strip()
    
    # Already a code?
    if text_upper in JURISDICTION_CODES:
        return text_upper
    
    # Check aliases
    text_title = text.strip()
    if text_title in JURISDICTION_ALIASES:
        return JURISDICTION_ALIASES[text_title]
    
    return text


def is_valid_entity_type(entity_type: str, strict: bool = False) -> bool:
    """
    Check if entity type is valid.
    
    Args:
        entity_type: Type to check
        strict: If True, use v1.1 types only; otherwise accept v1.0 types
        
    Returns:
        True if valid type
    """
    types = ENTITY_TYPES_V1_1_DRAFT if strict else ENTITY_TYPES_V1
    return entity_type in types


def get_jurisdiction_set() -> Set[str]:
    """Get set of all valid jurisdiction codes."""
    return set(JURISDICTION_CODES)
