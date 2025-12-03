"""
Entity Type Classification for Phase 1D-0
Strategy: Define NON-SEMANTIC types, everything else defaults to semantic

Total entity types in normalized_entities.json: ~900
Approach: Negative template - classify what's NOT semantic
"""

# ============================================================================
# ACADEMIC TYPES - Track 2 subjects (excluded from semantic matrix)
# ============================================================================

ACADEMIC_TYPES = {
    # Core academic entities
    'Academic',
    'Academic Article',
    'Academic Authors',
    'Academic Citation',
    'Academic Conference',
    'Academic Department',
    'Academic Discipline',
    'Academic Editor',
    'Academic Editors',
    'Academic Field',
    'Academic Institution',
    'Academic Journal',
    'Academic Journal Article',
    'Academic Literature',
    'Academic Paper',
    'Academic Publication',
    'Academic Reference',
    'Academic Work',
    
    # Citations & References
    'Citation',
    'Reference',
    
    # Authors & Editors
    'Author',
    'Authors',
    'Author(s)',
    'Author Group',
    'Author List',
    'Author and Year',
    'Author (Year)',
    'Author (2012)',
    'Author (2022)',
    'Authors (2019)',
    'Authorship',
    'Editor',
    'Editors',
    'Editor(s)',
    'Editor Citation',
    'Editors (2012)',
    'Editorial Team',
    'News Article Authors',
    
    # Journals & Publications
    'Journal',
    'Journal Article',
    'Journal Article Reference',
    'Journal Name',
    'Journal Citation',
    'Journal Section',
    'Journal Volume and Pages',
    'Journal Volume and Number',
    'Journal and Publication Date',
    'Journal and Year',
    'Journal or Publication',
    'Publication',
    'Publication Source',
    'Publication Title',
    'Publication Type',
    'Publication Volume',
    'Publication Issue',
    'Publication Edition',
    'Publication Series',
    'Publication Status',
    'Publication Model',
    'Publication Month',
    'Publication Information',
    'Publication Metadata',
    'Publication Reference',
    'Publication Venue',
    'Academic Journal',
    'Academic Journal Article',
    
    # Books
    'Book',
    'Book Title',
    'Book Series',
    'Book Chapter',
    'Book Chapter Title',
    'Book Subtitle',
    'Book Volume',
    'Book Edition',
    'Book Section',
    
    # Papers & Articles
    'Paper',
    'Paper Title',
    'Article Title',
    'Article Subtitle',
    'Article Volume and Issue',
    'Research Paper',
    'Research Paper Title',
    'Academic Paper',
    'Working Paper',
    'White Paper',
    'Discussion Paper',
    'Position Paper',
    'Review Paper',
    'Policy Paper',
    'Technical Report',
    'Technical Report Title',
    'Research Report',
    
    # Conferences
    'Conference',
    'Conference Proceedings',
    'Conference Title',
    'Conference Full Name',
    'Conference Abbreviation',
    'Conference Acronym',
    'Conference or Event',
    'Conference or Journal',
    'Conference/Book Title',
    'Conference/Workshop',
    
    # Document metadata
    'Document Title',
    'Report Title',
    'Study Title',
    'Research Title',
    'Thesis Title',
    'Project Title',
    'Volume Identifier',
    'Volume and Pages',
    'Volume or Edition',
    'Issue Number',
    
    # Academic works
    'Thesis',
    'Dissertation',
    'Manuscript',
    'Preprint',
    'Preprint Identifier',
    
    # Literature
    'Literature',
    'Literature Type',
    'Literary Work',
    'Academic Literature',
    
    # Publisher info
    'Publisher',
    
    # Identifiers (borderline, but related to academic metadata)
    'arXiv Identifier',
    'PubMed ID',
    'PubMed Identifier',
}  # 118 types


# ============================================================================
# SKIP TYPES - No extraction performed
# ============================================================================

SKIP_TYPES = {
    # Digital identifiers
    'DOI',
    'Digital Object Identifier',
    'Digital Object Identifier (DOI)',
    'ORCID',
    'ISBN',
    'ISSN',
    
    # Phase 1B artifacts
    'Chunk ID',
}  # 7 types


# ============================================================================
# SKIP NAMES - Generic references (edge cases)
# ============================================================================

SKIP_NAMES = {
    'et al.',
    'et al',
}  # Keep minimal - most "author" variations are legitimate entities


# ============================================================================
# CONCEPT TYPES - For Track 2 objects (subset of semantic)
# ============================================================================

CONCEPT_TYPES = {
    # Core concepts
    'Concept',
    'Legal Concept',
    'Regulatory Concept',
    'Technical Term',
    'Technical Concept',
    'Economic Concept',
    'AI Concept',
    'Ethical Concept',
    'Social Concept',
    'Political Concept',
    'Philosophical Concept',
    'Theoretical Concept',
    
    # Principles & Values
    'Principle',
    'Value',
    'Right',
    'Obligation',
    'Legal Principle',
    'Ethical Principle',
    'Regulatory Principle',
    'Governance Principle',
    
    # Abstract ideas (not concrete entities)
    'Concept',
    'Sub-concept',
    'Key Concept',
    'General Concept',
    'Conceptual Framework',
    'Conceptual Tool',
}  # ~30 types (will expand based on actual usage)


# ============================================================================
# CLASSIFICATION FUNCTIONS
# ============================================================================

def is_semantic(entity_type: str, entity_name: str) -> bool:
    """
    Check if entity should be in semantic co-occurrence matrix
    
    Returns True unless entity is academic or skip type
    """
    if entity_name.lower() in SKIP_NAMES:
        return False
    
    if entity_type in SKIP_TYPES:
        return False
    
    if entity_type in ACADEMIC_TYPES:
        return False
    
    # Default: semantic
    return True


def is_concept(entity_type: str) -> bool:
    """
    Check if entity is a concept-type (for Track 2 objects)
    
    Returns True for abstract concepts/ideas
    """
    return entity_type in CONCEPT_TYPES


def is_academic(entity_type: str) -> bool:
    """
    Check if entity is academic-type (for Track 2 subjects)
    """
    return entity_type in ACADEMIC_TYPES


def is_skip(entity_type: str, entity_name: str) -> bool:
    """
    Check if entity should be skipped entirely
    """
    if entity_name.lower() in SKIP_NAMES:
        return True
    
    return entity_type in SKIP_TYPES


def get_extraction_strategy(entity: dict) -> str:
    """
    Classify entity for extraction strategy
    
    Args:
        entity: Dict with 'type' and 'name' keys
    
    Returns:
        'semantic' | 'academic' | 'skip'
    """
    entity_type = entity['type']
    entity_name = entity['name']
    
    # Skip list
    if is_skip(entity_type, entity_name):
        return 'skip'
    
    # Academic entities
    if is_academic(entity_type):
        return 'academic'
    
    # Default: semantic entities
    return 'semantic'


# ============================================================================
# STATISTICS FUNCTION
# ============================================================================

def print_classification_stats(entities: list):
    """Print classification statistics for verification"""
    from collections import Counter
    
    semantic_count = 0
    academic_count = 0
    skip_count = 0
    
    semantic_types = Counter()
    academic_types = Counter()
    skip_types = Counter()
    
    for entity in entities:
        strategy = get_extraction_strategy(entity)
        
        if strategy == 'semantic':
            semantic_count += 1
            semantic_types[entity['type']] += 1
        elif strategy == 'academic':
            academic_count += 1
            academic_types[entity['type']] += 1
        elif strategy == 'skip':
            skip_count += 1
            skip_types[entity['type']] += 1
    
    print("="*80)
    print("ENTITY CLASSIFICATION STATISTICS")
    print("="*80)
    print(f"Total entities: {len(entities):,}")
    print(f"\nSemantic entities (Track 1): {semantic_count:,} ({semantic_count/len(entities)*100:.1f}%)")
    print(f"Academic entities (Track 2): {academic_count:,} ({academic_count/len(entities)*100:.1f}%)")
    print(f"Skip entities:               {skip_count:,} ({skip_count/len(entities)*100:.1f}%)")
    
    print(f"\nTop 10 semantic types:")
    for ent_type, count in semantic_types.most_common(10):
        print(f"  {ent_type:40s} {count:>6,}")
    
    print(f"\nTop 10 academic types:")
    for ent_type, count in academic_types.most_common(10):
        print(f"  {ent_type:40s} {count:>6,}")
    
    print(f"\nSkip types:")
    for ent_type, count in skip_types.most_common():
        print(f"  {ent_type:40s} {count:>6,}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    import json
    
    # Load entities
    with open('data/interim/entities/normalized_entities.json', 'r') as f:
        entities = json.load(f)
    
    # Print statistics
    print_classification_stats(entities)
    
    # Show examples
    print("\n" + "="*80)
    print("EXAMPLE CLASSIFICATIONS")
    print("="*80)
    
    examples = [
        {'name': 'GDPR', 'type': 'Regulation'},
        {'name': 'transparency', 'type': 'Concept'},
        {'name': 'Smith et al. (2020)', 'type': 'Academic Citation'},
        {'name': 'Nature', 'type': 'Journal'},
        {'name': '10.1234/example', 'type': 'DOI'},
    ]
    
    for entity in examples:
        strategy = get_extraction_strategy(entity)
        is_sem = is_semantic(entity['type'], entity['name'])
        is_conc = is_concept(entity['type'])
        
        print(f"\nEntity: {entity['name']} ({entity['type']})")
        print(f"  Strategy: {strategy}")
        print(f"  In semantic matrix: {is_sem}")
        print(f"  In concept matrix: {is_conc}")
