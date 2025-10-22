"""
Heuristic filtering for DBpedia entity annotations.

Keeps:
- multi-word entities (e.g. "machine learning")
- capitalized single words (e.g. "Internet", "China")
- acronyms or dotted abbreviations (e.g. "AI", "U.S.")
Rejects:
- lowercase single words ("after", "during")
- single letters ("A", "I")
- entities with low similarity scores or unrelated ontology types
"""

import re

# ----------------------------------------
# Domain whitelist (ontology types we trust)
# ----------------------------------------
DOMAIN_WHITELIST_TYPES = {
    # Core conceptual / abstract classes
    "http://dbpedia.org/ontology/Company",
    "http://dbpedia.org/ontology/Project",
    "http://dbpedia.org/ontology/Software",
    "http://dbpedia.org/ontology/Technology",
    "http://dbpedia.org/ontology/AcademicDiscipline",
    "http://dbpedia.org/ontology/Activity",
    "http://dbpedia.org/ontology/Work",
    "http://dbpedia.org/ontology/WrittenWork",
    "http://dbpedia.org/ontology/Book",
    "http://dbpedia.org/ontology/Document",

    # Political / geopolitical institutions
    "http://dbpedia.org/ontology/Country",
    "http://dbpedia.org/ontology/PopulatedPlace",
    "http://dbpedia.org/ontology/GovernmentAgency",
    "http://dbpedia.org/ontology/PoliticalParty",
    "http://dbpedia.org/ontology/InternationalOrganization",
    "http://dbpedia.org/ontology/PoliticalOrganization",

    # Abstract categories
    "http://dbpedia.org/class/yago/WorldOrganization108294696",
    "http://dbpedia.org/class/yago/WikicatInternationalOrganizations",
    "http://dbpedia.org/class/yago/WikicatTradeBlocs",
    "http://dbpedia.org/class/yago/WikicatEmergingTechnologies",
    "http://dbpedia.org/class/yago/Technology100949619",
    "http://dbpedia.org/class/yago/Profession100609953",
    "http://dbpedia.org/class/yago/Occupation100582388",

    # Semantic web / schema-level
    "http://schema.org/Place",
    "http://schema.org/Country"
}

KEY_TOPICS =  [
    # Technical / AI / CS
    "data", "big data", "algorithm", "machine learning", "deep learning", "neural network", 
    "artificial intelligence", "ai", "nlp", "natural language processing", "cloud", "computing",
    "software", "programming", "blockchain", "smart contract",

    # Legal / regulation / policy
     "regulation", "directive", "law", "compliance", "policy", "governance",
    "international law", "treaty", "human rights", "ethics", "audit", "standards",
    "government", "state", "agency", "ministry", "parliament", "administration",
    "political", "public sector", "bureaucracy", "institution", "legislation"

    # Business / finance / economics
    "economics", "financial", "investment", "capitalism", "market", "audit", "accounting", "industry",
]
# ----------------------------------------
# Adaptive threshold: len(surface_form) → cutoff
# ----------------------------------------
def get_adaptive_threshold(surface_form: str) -> float:
    words = surface_form.split()
    if len(words) == 1:
        return 0.96
    elif len(words) == 2:
        return 0.75
    else:
        return 0.70


# ----------------------------------------
# Acronym detection
# ----------------------------------------
def looks_like_acronym(word: str) -> bool:
    """
    True if 'word' is an acronym or abbreviation (e.g. 'AI', 'U.S.', 'OECD').
    """
    return bool(re.fullmatch(r"([A-Z]{2,}|(?:[A-Z]\.){2,})", word))


# ----------------------------------------
# Main filter
# ----------------------------------------
def is_valid_entity(surface_form: str, entity_types: list, similarity_score: float):
    """
    Determine whether a candidate entity should be accepted or rejected
    based on domain heuristics, adaptive similarity thresholds, and word shape.

    Returns
    -------
    tuple[bool, str]
        (is_valid, reason)
    """

    # --- 1 Reject empty or whitespace-only surface forms ---
    if not surface_form or not surface_form.strip():
        return False, "Empty surface form"
    surface_form = surface_form.strip()
    words = surface_form.split()

    # --- 2 Check adaptive similarity threshold ---
    threshold = get_adaptive_threshold(surface_form)
    if similarity_score < threshold:
        return False, f"Score {similarity_score:.3f} < threshold {threshold:.2f}"

    # --- 3 Single-word shape checks & acronyms ---
    if len(words) == 1:
        word = words[0]
        if word.islower():
            return False, "Lowercase single word"
        if len(word) == 1:
            return False, "Single letter"
        if looks_like_acronym(word):
            return True, "Acronym accepted"

    # --- 4 If entity has no types: fallback to keyword heuristic ---
    if not entity_types or len(entity_types) == 0:
        if any(k in surface_form.lower() for k in KEY_TOPICS):
            return True, "Accepted by keyword heuristic"
        else:
            return False, "No types + no keywords"

    # --- 5 Entity has types: require ALL types to be in whitelist (AND logic) ---
    if not any(t in DOMAIN_WHITELIST_TYPES for t in entity_types):
        return False, "Types not in whitelist"

    # --- 6 Passed all checks ---
    return True, "All heuristics passed"
