# -*- coding: utf-8 -*-
"""
Test

Design: 6 categories × 6 queries = 36 total

"""
"""
# =============================================================================
# FULL TEST SET (36 queries)
# =============================================================================

FULL_QUERIES = [
    # =========================================================================
    # CATEGORY 1: REGULATION_ONLY (6 queries)
    # Tests: DLA Piper retrieval, jurisdiction filtering, regulatory knowledge
    # =========================================================================
    {
        'id': 'ro1',
        'query': 'EU AI Act definition and scope',
        'primary_category': 'regulation_only',
        'tags': {
            'complexity': 'single_entity',
            'style': 'keyword',
            'source_expected': 'regulation',
            'jurisdiction': 'EU'
        },
        'description': 'High-degree regulation, keyword style'
    },
    {
        'id': 'ro2',
        'query': 'Explain the prohibited AI practices under Article 5.',
        'primary_category': 'regulation_only',
        'tags': {
            'complexity': 'single_entity',
            'style': 'imperative',
            'source_expected': 'regulation',
            'jurisdiction': 'EU'
        },
        'description': 'Specific provision, imperative style'
    },
    {
        'id': 'ro3',
        'query': 'What are the conformity assessment requirements for high-risk AI?',
        'primary_category': 'regulation_only',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'wh_question',
            'source_expected': 'regulation',
            'jurisdiction': 'EU'
        },
        'description': 'Multi-hop: high-risk → conformity → requirements'
    },
    {
        'id': 'ro4',
        'query': 'Compare EU and US approaches to regulating generative AI.',
        'primary_category': 'regulation_only',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'imperative',
            'source_expected': 'regulation',
            'jurisdiction': 'multiple'
        },
        'description': 'Cross-jurisdictional within regulatory corpus'
    },
    {
        'id': 'ro5',
        'query': "China's algorithm recommendation regulations",
        'primary_category': 'regulation_only',
        'tags': {
            'complexity': 'single_entity',
            'style': 'keyword',
            'source_expected': 'regulation',
            'jurisdiction': 'CN'
        },
        'description': 'Non-EU regulation, keyword style'
    },
    {
        'id': 'ro6',
        'query': 'I need to understand GDPR Article 22 automated decision-making provisions.',
        'primary_category': 'regulation_only',
        'tags': {
            'complexity': 'single_entity',
            'style': 'declarative_need',
            'source_expected': 'regulation',
            'jurisdiction': 'EU'
        },
        'description': 'Cross-regulation reference, declarative need'
    },

    # =========================================================================
    # CATEGORY 2: ACADEMIC_ONLY (6 queries)
    # Tests: Scopus retrieval, research concepts, academic terminology
    # =========================================================================
    {
        'id': 'ao1',
        'query': 'Research methodologies for auditing algorithmic bias',
        'primary_category': 'academic_only',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'keyword',
            'source_expected': 'academic',
            'jurisdiction': None
        },
        'description': 'Methodological focus, paper-heavy'
    },
    {
        'id': 'ao2',
        'query': 'Explain the concept of explainable AI in machine learning literature.',
        'primary_category': 'academic_only',
        'tags': {
            'complexity': 'single_entity',
            'style': 'imperative',
            'source_expected': 'academic',
            'jurisdiction': None
        },
        'description': 'Core ML concept, imperative style'
    },
    {
        'id': 'ao3',
        'query': 'What fairness metrics do researchers use to evaluate AI systems?',
        'primary_category': 'academic_only',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'wh_question',
            'source_expected': 'academic',
            'jurisdiction': None
        },
        'description': 'Technical metrics from literature'
    },
    {
        'id': 'ao4',
        'query': 'Studies on environmental sustainability of large language models',
        'primary_category': 'academic_only',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'keyword',
            'source_expected': 'academic',
            'jurisdiction': None
        },
        'description': 'Specific research topic'
    },
    {
        'id': 'ao5',
        'query': 'Academic debate on AI consciousness and moral status',
        'primary_category': 'academic_only',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'keyword',
            'source_expected': 'academic',
            'jurisdiction': None
        },
        'description': 'Philosophical/conceptual debate'
    },
    {
        'id': 'ao6',
        'query': 'How do researchers define algorithmic transparency?',
        'primary_category': 'academic_only',
        'tags': {
            'complexity': 'single_entity',
            'style': 'wh_question',
            'source_expected': 'academic',
            'jurisdiction': None
        },
        'description': 'Definitional, academic perspective'
    },

    # =========================================================================
    # CATEGORY 3: CROSS_DOMAIN (6 queries) - KEY TEST
    # Tests: Regulation ↔ Academic bridging via 392 shared entities
    # Hypothesis: Graph mode should significantly outperform semantic
    # =========================================================================
    {
        'id': 'cd1',
        'query': 'Academic critiques of the EU AI Act risk classification system',
        'primary_category': 'cross_domain',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'keyword',
            'source_expected': 'both',
            'jurisdiction': 'EU'
        },
        'description': 'Regulation → Academic critique'
    },
    {
        'id': 'cd2',
        'query': 'How has algorithmic fairness research influenced AI regulation?',
        'primary_category': 'cross_domain',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'wh_question',
            'source_expected': 'both',
            'jurisdiction': None
        },
        'description': 'Academic → Regulatory implementation'
    },
    {
        'id': 'cd3',
        'query': 'Scholarly analysis of GDPR automated decision-making provisions',
        'primary_category': 'cross_domain',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'keyword',
            'source_expected': 'both',
            'jurisdiction': 'EU'
        },
        'description': 'Specific provision + academic treatment'
    },
    {
        'id': 'cd4',
        'query': 'Research papers discussing regulatory gaps in AI governance',
        'primary_category': 'cross_domain',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'keyword',
            'source_expected': 'both',
            'jurisdiction': None
        },
        'description': 'Meta-level: academic on regulatory landscape'
    },
    {
        'id': 'cd5',
        'query': 'Compare academic and regulatory definitions of high-risk AI systems.',
        'primary_category': 'cross_domain',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'imperative',
            'source_expected': 'both',
            'jurisdiction': None
        },
        'description': 'Explicit cross-domain comparison'
    },
    {
        'id': 'cd6',
        'query': 'Evidence base behind EU AI Act transparency requirements',
        'primary_category': 'cross_domain',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'keyword',
            'source_expected': 'both',
            'jurisdiction': 'EU'
        },
        'description': 'Regulation + supporting research'
    },

    # =========================================================================
    # CATEGORY 4: METADATA_PROVENANCE (6 queries) - KEY TEST
    # Tests: EXTRACTED_FROM links, citation chains, author traversal
    # Hypothesis: Graph mode should significantly outperform (semantic can't traverse)
    # =========================================================================
    {
        'id': 'mp1',
        'query': 'Which chunks discuss both transparency and the EU AI Act?',
        'primary_category': 'metadata_provenance',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'wh_question',
            'source_expected': 'both',
            'jurisdiction': 'EU'
        },
        'description': 'Tests EXTRACTED_FROM co-occurrence'
    },
    {
        'id': 'mp2',
        'query': 'Sources that connect GDPR and AI governance frameworks',
        'primary_category': 'metadata_provenance',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'keyword',
            'source_expected': 'both',
            'jurisdiction': 'EU'
        },
        'description': 'Cross-regulation via shared chunks'
    },
    {
        'id': 'mp3',
        'query': 'Documents discussing algorithmic accountability across jurisdictions',
        'primary_category': 'metadata_provenance',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'keyword',
            'source_expected': 'both',
            'jurisdiction': 'multiple'
        },
        'description': 'Multi-jurisdiction document discovery'
    },
    {
        'id': 'mp4',
        'query': 'Trace the regulatory references in AI ethics scholarship',
        'primary_category': 'metadata_provenance',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'imperative',
            'source_expected': 'both',
            'jurisdiction': None
        },
        'description': 'Citation/reference traversal'
    },
    {
        'id': 'mp5',
        'query': 'Primary sources for facial recognition regulation analysis',
        'primary_category': 'metadata_provenance',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'keyword',
            'source_expected': 'both',
            'jurisdiction': None
        },
        'description': 'Source provenance request'
    },
    {
        'id': 'mp6',
        'query': 'What documents cover both bias mitigation and compliance requirements?',
        'primary_category': 'metadata_provenance',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'wh_question',
            'source_expected': 'both',
            'jurisdiction': None
        },
        'description': 'Concept co-occurrence via provenance'
    },

    # =========================================================================
    # CATEGORY 5: APPLIED_SCENARIO (6 queries)
    # Tests: Practitioner perspective, real-world compliance
    # =========================================================================
    {
        'id': 'as1',
        'query': 'A startup deploying a chatbot in the EU - what regulations apply?',
        'primary_category': 'applied_scenario',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'scenario',
            'source_expected': 'regulation',
            'jurisdiction': 'EU'
        },
        'description': 'Business compliance scenario'
    },
    {
        'id': 'as2',
        'query': 'We are building a CV screening tool - what are the legal risks?',
        'primary_category': 'applied_scenario',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'scenario',
            'source_expected': 'both',
            'jurisdiction': None
        },
        'description': 'HR tech compliance, high-risk category'
    },
    {
        'id': 'as3',
        'query': 'Compliance checklist for medical AI device deployment',
        'primary_category': 'applied_scenario',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'keyword',
            'source_expected': 'regulation',
            'jurisdiction': None
        },
        'description': 'Sector-specific compliance'
    },
    {
        'id': 'as4',
        'query': 'Documentation requirements for AI providers under the EU AI Act',
        'primary_category': 'applied_scenario',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'keyword',
            'source_expected': 'regulation',
            'jurisdiction': 'EU'
        },
        'description': 'Specific compliance obligation'
    },
    {
        'id': 'as5',
        'query': 'How should a financial services firm approach AI model validation?',
        'primary_category': 'applied_scenario',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'wh_question',
            'source_expected': 'both',
            'jurisdiction': None
        },
        'description': 'Sector-specific, methodology question'
    },
    {
        'id': 'as6',
        'query': 'Implementing human oversight for automated decision systems',
        'primary_category': 'applied_scenario',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'keyword',
            'source_expected': 'both',
            'jurisdiction': None
        },
        'description': 'Implementation guidance'
    },

    # =========================================================================
    # CATEGORY 6: EDGE_CASES (6 queries)
    # Tests: OOD, alias resolution, specific legal, abbreviations, ambiguity
    # =========================================================================
    {
        'id': 'ec1',
        'query': "Who is Snoopy's arch enemy?",
        'primary_category': 'edge_cases',
        'subcategory': 'out_of_domain',
        'tags': {
            'complexity': 'single_entity',
            'style': 'keyword',
            'source_expected': 'none',
            'jurisdiction': None
        },
        'description': 'Completely OOD, graceful failure test'
    },
    {
        'id': 'ec2',
        'query': 'AI Act prohibited practices',
        'primary_category': 'edge_cases',
        'subcategory': 'alias_resolution',
        'tags': {
            'complexity': 'single_entity',
            'style': 'keyword',
            'source_expected': 'regulation',
            'jurisdiction': 'EU'
        },
        'description': 'Alias test: AI Act → EU AI Act'
    },
    {
        'id': 'ec3',
        'query': 'Tell me about AI.',
        'primary_category': 'edge_cases',
        'subcategory': 'ambiguous',
        'tags': {
            'complexity': 'single_entity',
            'style': 'imperative',
            'source_expected': 'both',
            'jurisdiction': None
        },
        'description': 'Maximally ambiguous, hub entity'
    },
    {
        'id': 'ec4',
        'query': 'Infopaq decision and its implications for originality in AI-generated works',
        'primary_category': 'edge_cases',
        'subcategory': 'specific_legal',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'keyword',
            'source_expected': 'both',
            'jurisdiction': 'EU'
        },
        'description': 'Specific ECJ case (C-5/08, 2009) - tests precise legal knowledge'
    },
    {
        'id': 'ec5',
        'query': 'DPA enforcement actions on algorithmic systems',
        'primary_category': 'edge_cases',
        'subcategory': 'abbreviation',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'keyword',
            'source_expected': 'regulation',
            'jurisdiction': 'multiple'
        },
        'description': 'Abbreviation: DPA → Data Protection Authority'
    },
    {
        'id': 'ec6',
        'query': 'ML transparency requirements',
        'primary_category': 'edge_cases',
        'subcategory': 'abbreviation',
        'tags': {
            'complexity': 'multi_hop',
            'style': 'keyword',
            'source_expected': 'both',
            'jurisdiction': None
        },
        'description': 'Abbreviation: ML → machine learning'
    }
]


# =============================================================================
# DETAILED TEST SET (6 queries) - one per category for qualitative analysis
# =============================================================================

DETAILED_QUERIES = [
    FULL_QUERIES[2],   # ro3: regulation_only - conformity assessment
    FULL_QUERIES[8],   # ao3: academic_only - fairness metrics
    FULL_QUERIES[12],  # cd1: cross_domain - academic critiques of AI Act
    FULL_QUERIES[18],  # mp1: metadata_provenance - chunk co-occurrence
    FULL_QUERIES[24],  # as1: applied_scenario - chatbot startup
    FULL_QUERIES[30],  # ec1: edge_cases - OOD (Snoopy's arch enemy)
    FULL_QUERIES[31],  # ec2: edge_cases - alias (AI Act → EU AI Act)
    FULL_QUERIES[33],  # ec4: edge_cases - specific_legal (Infopaq)
]


# =============================================================================
# ANALYSIS UTILITIES
# =============================================================================

def get_queries_by_tag(tag_name, tag_value):
    """Filter queries by a specific tag value."""
    return [q for q in FULL_QUERIES if q.get('tags', {}).get(tag_name) == tag_value]


def get_queries_by_category(category):
    """Filter queries by primary category."""
    return [q for q in FULL_QUERIES if q['primary_category'] == category]


def generate_analysis_tables():
    """Generate summary for thesis tables."""
    
    print("=" * 70)
    print("TABLE 1: BY PRIMARY CATEGORY (Domain/Layer)")
    print("=" * 70)
    categories = {}
    for q in FULL_QUERIES:
        cat = q['primary_category']
        categories[cat] = categories.get(cat, 0) + 1
    for cat, n in categories.items():
        print(f"  {cat:<25} n={n}")
    
    print("\n" + "=" * 70)
    print("TABLE 2: BY COMPLEXITY")
    print("=" * 70)
    complexities = {}
    for q in FULL_QUERIES:
        c = q.get('tags', {}).get('complexity', 'unknown')
        complexities[c] = complexities.get(c, 0) + 1
    for c, n in complexities.items():
        print(f"  {c:<25} n={n}")
    
    print("\n" + "=" * 70)
    print("TABLE 3: BY STYLE")
    print("=" * 70)
    styles = {}
    for q in FULL_QUERIES:
        s = q.get('tags', {}).get('style', 'unknown')
        styles[s] = styles.get(s, 0) + 1
    for s, n in sorted(styles.items(), key=lambda x: -x[1]):
        print(f"  {s:<25} n={n}")
    
    print("\n" + "=" * 70)
    print("TABLE 4: BY SOURCE EXPECTED")
    print("=" * 70)
    sources = {}
    for q in FULL_QUERIES:
        s = q.get('tags', {}).get('source_expected', 'unknown')
        sources[s] = sources.get(s, 0) + 1
    for s, n in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {s:<25} n={n}")


def get_queries(mode='detailed'):
    """Get test queries by mode."""
    if mode == 'full':
        return FULL_QUERIES
    elif mode == 'quick':
        return FULL_QUERIES[:2]
    return DETAILED_QUERIES


# Backward compatibility
QUICK_QUERIES = DETAILED_QUERIES
TEST_QUERIES = DETAILED_QUERIES


if __name__ == '__main__':
    generate_analysis_tables()
    
    print("\n" + "=" * 70)
    print("EXAMPLE QUERIES BY CATEGORY")
    print("=" * 70)
    seen = set()
    for q in FULL_QUERIES:
        cat = q['primary_category']
        if cat not in seen:
            seen.add(cat)
            print(f"\n{cat.upper()}:")
            print(f"  \"{q['query']}\"")