# -*- coding: utf-8 -*-
"""
Test queries for GraphRAG ablation study.

Coverage dimensions:
- Entity count: 1, 2, 3+
- Entity type: Technology, Regulation, Risk, Organization, Location, Concept
- Relation type: appliesto, requires, enforces, mitigates, poses
- Jurisdiction: Single, Multi, Cross-comparison
- Source: Paper-heavy, Regulation-heavy, Mixed
- Resolution: Exact, Alias, Fuzzy
- Difficulty: Simple, Medium, Complex, Expert

Categories (7, statistically valid n>=5):
- simple_factual (5)
- multi_entity (5)
- cross_jurisdictional (5)
- relation_specific (5)
- research (5)
- edge_cases (5)
- expert_specific (5)

Query Sets:
- FULL_QUERIES: 35 queries for comprehensive evaluation
- DETAILED_QUERIES: 8 queries (1 per category + 1) for qualitative analysis
"""

# =============================================================================
# FULL TEST SET (35 queries) - for comprehensive evaluation
# =============================================================================

FULL_QUERIES = [
    # -------------------------------------------------------------------------
    # CATEGORY 1: SIMPLE FACTUAL (Single Entity) - 5 queries
    # Tests: kNN expansion, entity resolution, single-entity retrieval
    # -------------------------------------------------------------------------
    {
        'id': 'sf1',
        'query': 'What is the EU AI Act?',
        'category': 'simple_factual',
        'entities': 1,
        'target_entity': 'EU AI Act',
        'description': 'High-degree regulation'
    },
    {
        'id': 'sf2',
        'query': 'What is ChatGPT?',
        'category': 'simple_factual',
        'entities': 1,
        'target_entity': 'ChatGPT',
        'description': 'High-degree technology'
    },
    {
        'id': 'sf3',
        'query': 'What does transparency mean in AI governance?',
        'category': 'simple_factual',
        'entities': 1,
        'target_entity': 'transparency',
        'description': 'High-degree concept (abstract)'
    },
    {
        'id': 'sf4',
        'query': 'What is the role of the European Commission in AI?',
        'category': 'simple_factual',
        'entities': 1,
        'target_entity': 'European Commission',
        'description': 'Organization entity'
    },
    {
        'id': 'sf5',
        'query': 'What are discriminatory impacts of AI?',
        'category': 'simple_factual',
        'entities': 1,
        'target_entity': 'discriminatory impacts',
        'description': 'Risk entity'
    },
    
    # -------------------------------------------------------------------------
    # CATEGORY 2: MULTI-ENTITY FACTUAL - 5 queries  
    # Tests: Steiner Tree, multi-entity resolution, relation traversal
    # -------------------------------------------------------------------------
    {
        'id': 'mf1',
        'query': 'What are high-risk AI systems?',
        'category': 'multi_entity',
        'entities': 2,
        'description': '2-entity, risk+technology'
    },
    {
        'id': 'mf2',
        'query': 'How does machine learning relate to privacy risks?',
        'category': 'multi_entity',
        'entities': 2,
        'description': '2-entity, technology→risk'
    },
    {
        'id': 'mf3',
        'query': 'What accountability requirements apply to AI systems?',
        'category': 'multi_entity',
        'entities': 2,
        'description': '2-entity, concept+technology'
    },
    {
        'id': 'mf4',
        'query': 'How do algorithms affect human rights?',
        'category': 'multi_entity',
        'entities': 2,
        'description': '2-entity, tech→political'
    },
    {
        'id': 'mf5',
        'query': 'What security risks do AI systems pose?',
        'category': 'multi_entity',
        'entities': 2,
        'description': '2-entity, security focus'
    },
    
    # -------------------------------------------------------------------------
    # CATEGORY 3: CROSS-JURISDICTIONAL - 5 queries
    # Tests: Location entities, jurisdiction diversity, regulatory comparison
    # -------------------------------------------------------------------------
    {
        'id': 'cj1',
        'query': 'Which jurisdictions regulate facial recognition?',
        'category': 'cross_jurisdictional',
        'entities': 2,
        'jurisdictions': 'multiple',
        'description': 'Tech across jurisdictions'
    },
    {
        'id': 'cj2',
        'query': 'How does the EU regulate AI compared to China?',
        'category': 'cross_jurisdictional',
        'entities': 3,
        'jurisdictions': ['EU', 'CN'],
        'description': 'EU vs China comparison'
    },
    {
        'id': 'cj3',
        'query': 'What AI regulations exist in Singapore?',
        'category': 'cross_jurisdictional',
        'entities': 2,
        'jurisdictions': ['SG'],
        'description': 'Single jurisdiction (Singapore)'
    },
    {
        'id': 'cj4',
        'query': 'Compare US and EU approaches to AI transparency',
        'category': 'cross_jurisdictional',
        'entities': 3,
        'jurisdictions': ['US', 'EU'],
        'description': 'US vs EU on concept'
    },
    {
        'id': 'cj5',
        'query': 'How does Japan address AI ethics?',
        'category': 'cross_jurisdictional',
        'entities': 2,
        'jurisdictions': ['JP'],
        'description': 'Single jurisdiction (Japan)'
    },
    
    # -------------------------------------------------------------------------
    # CATEGORY 4: RELATION-SPECIFIC - 5 queries
    # Tests: Specific relation traversal (appliesto, requires, enforces, etc.)
    # -------------------------------------------------------------------------
    {
        'id': 'rs1',
        'query': 'What requirements apply to AI in healthcare?',
        'category': 'relation_specific',
        'entities': 2,
        'description': 'appliesto relation'
    },
    {
        'id': 'rs2',
        'query': 'What does the EU AI Act require for high-risk systems?',
        'category': 'relation_specific',
        'entities': 3,
        'description': 'requires relation'
    },
    {
        'id': 'rs3',
        'query': 'How is AI regulation enforced?',
        'category': 'relation_specific',
        'entities': 2,
        'description': 'enforces relation'
    },
    {
        'id': 'rs4',
        'query': 'What measures mitigate AI bias?',
        'category': 'relation_specific',
        'entities': 2,
        'description': 'mitigates relation'
    },
    {
        'id': 'rs5',
        'query': 'What risks does automated decision-making pose?',
        'category': 'relation_specific',
        'entities': 2,
        'description': 'poses relation'
    },
    
    # -------------------------------------------------------------------------
    # CATEGORY 5: RESEARCH/ACADEMIC - 5 queries
    # Tests: Paper-heavy retrieval, citations, academic concepts
    # -------------------------------------------------------------------------
    {
        'id': 'ra1',
        'query': 'What academic research discusses algorithmic bias?',
        'category': 'research',
        'entities': 2,
        'source_bias': 'paper',
        'description': 'Academic focus, paper-heavy'
    },
    {
        'id': 'ra2',
        'query': 'What studies examine AI sustainability?',
        'category': 'research',
        'entities': 2,
        'source_bias': 'paper',
        'description': 'Sustainability research'
    },
    {
        'id': 'ra3',
        'query': 'How do researchers define explainable AI?',
        'category': 'research',
        'entities': 1,
        'source_bias': 'paper',
        'description': 'Technical concept from papers'
    },
    {
        'id': 'ra4',
        'query': 'What is the academic perspective on AI governance?',
        'category': 'research',
        'entities': 2,
        'source_bias': 'paper',
        'description': 'Governance from academic lens'
    },
    {
        'id': 'ra5',
        'query': 'What methodologies are used to evaluate AI fairness?',
        'category': 'research',
        'entities': 2,
        'source_bias': 'paper',
        'description': 'Methods/evaluation focus'
    },
    
    # -------------------------------------------------------------------------
    # CATEGORY 6: EDGE CASES - 5 queries
    # Tests: Out-of-domain, ambiguous, temporal boundary, alias resolution
    # -------------------------------------------------------------------------
    {
        'id': 'ec1',
        'query': "What is Snoopy's arch enemy?",
        'category': 'edge_cases',
        'subcategory': 'out_of_domain',
        'entities': 0,
        'description': 'Out-of-domain (graceful failure)'
    },
    {
        'id': 'ec2',
        'query': 'What is AI?',
        'category': 'edge_cases',
        'subcategory': 'ambiguous',
        'entities': 1,
        'description': 'Highest-degree entity, very broad'
    },
    {
        'id': 'ec3',
        'query': 'What is the AI Act?',
        'category': 'edge_cases',
        'subcategory': 'alias_resolution',
        'entities': 1,
        'description': 'Alias test (AI Act → EU AI Act)'
    },
    {
        'id': 'ec4',
        'query': 'What are the implications of the Infopak ruling for AI liability?',
        'category': 'edge_cases',
        'subcategory': 'temporal_boundary',
        'entities': 2,
        'description': 'Post-cutoff (2024 Belgian case)'
    },
    {
        'id': 'ec5',
        'query': 'How does ML affect data protection?',
        'category': 'edge_cases',
        'subcategory': 'alias_resolution',
        'entities': 2,
        'description': 'Abbreviation test (ML → machine learning)'
    },
    
    # -------------------------------------------------------------------------
    # CATEGORY 7: EXPERT SPECIFIC - 5 queries
    # Tests: Deep provision knowledge, specific articles, enforcement cases
    # -------------------------------------------------------------------------
    {
        'id': 'es1',
        'query': 'What practices are explicitly prohibited under Article 5 of the EU AI Act?',
        'category': 'expert_specific',
        'entities': 2,
        'description': 'Specific provision traversal'
    },
    {
        'id': 'es2',
        'query': 'How does GDPR Article 22 interact with AI automated decision-making?',
        'category': 'expert_specific',
        'entities': 3,
        'description': 'Cross-regulation linking'
    },
    {
        'id': 'es3',
        'query': 'What conformity assessment requirements apply to high-risk AI systems under the EU AI Act?',
        'category': 'expert_specific',
        'entities': 3,
        'description': 'Annex-level detail'
    },
    {
        'id': 'es4',
        'query': "How did Italy's Garante address algorithmic management in gig economy cases?",
        'category': 'expert_specific',
        'entities': 3,
        'description': 'Enforcement action (Deliveroo ~2021)'
    },
    {
        'id': 'es5',
        'query': "What is China's approach to regulating recommendation algorithms?",
        'category': 'expert_specific',
        'entities': 2,
        'description': '2021 provisions, DLA Piper coverage'
    }
]


# =============================================================================
# DETAILED TEST SET (8 queries) - for qualitative analysis with full outputs
# One representative per category + 1 extra expert query
# =============================================================================

DETAILED_QUERIES = [
    FULL_QUERIES[0],   # sf1: simple_factual - "What is the EU AI Act?"
    FULL_QUERIES[5],   # mf1: multi_entity - "What are high-risk AI systems?"
    FULL_QUERIES[10],  # cj1: cross_jurisdictional - "Which jurisdictions regulate facial recognition?"
    FULL_QUERIES[15],  # rs1: relation_specific - "What requirements apply to AI in healthcare?"
    FULL_QUERIES[20],  # ra1: research - "What academic research discusses algorithmic bias?"
    FULL_QUERIES[25],  # ec1: edge_cases - "What is Snoopy's arch enemy?"
    FULL_QUERIES[30],  # es1: expert_specific - "What practices are prohibited under Article 5?"
    FULL_QUERIES[31],  # es2: expert_specific - "How does GDPR Article 22 interact with AI?"
]


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

# Legacy alias
QUICK_QUERIES = DETAILED_QUERIES
TEST_QUERIES = DETAILED_QUERIES


def get_queries(mode='detailed'):
    """
    Get test queries by mode.
    
    Args:
        mode: 'detailed' (8 queries), 'full' (35 queries), or 'quick' (2 queries)
    
    Returns:
        List of query dictionaries
    """
    if mode == 'full':
        return FULL_QUERIES
    elif mode == 'quick':
        return FULL_QUERIES[:2]
    return DETAILED_QUERIES


def print_coverage_report():
    """Print coverage analysis of full query set."""
    print("=" * 60)
    print("TEST SUITE COVERAGE REPORT")
    print("=" * 60)
    
    # Category distribution
    categories = {}
    for q in FULL_QUERIES:
        cat = q['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\nTotal queries: {len(FULL_QUERIES)}")
    print(f"Detailed queries: {len(DETAILED_QUERIES)}")
    print(f"Categories: {len(categories)}")
    
    print("\nBy Category:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {count:2d} | {cat}")
    
    # Subcategory distribution (for edge_cases)
    print("\nEdge Cases Breakdown:")
    for q in FULL_QUERIES:
        if q['category'] == 'edge_cases':
            print(f"  - {q.get('subcategory', 'unknown')}: {q['query'][:50]}...")
    
    # Entity count distribution
    entity_counts = {}
    for q in FULL_QUERIES:
        ec = q.get('entities', 0)
        entity_counts[ec] = entity_counts.get(ec, 0) + 1
    
    print("\nBy Entity Count:")
    for ec, count in sorted(entity_counts.items()):
        print(f"  {count:2d} | {ec} entities")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    print_coverage_report()