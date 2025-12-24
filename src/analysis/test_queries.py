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
- Difficulty: Simple, Medium, Complex
"""

# =============================================================================
# QUICK TEST SET (6 queries) - for detailed analysis
# =============================================================================

QUICK_QUERIES = [
    # 1. Single entity, high-degree regulation
    {
        'id': 'q1',
        'query': 'What is the EU AI Act?',
        'category': 'simple_factual',
        'entities': 1,
        'target_entity': 'EU AI Act',
        'expected_types': ['Regulation'],
        'expected_mode': 'graph',
        'description': 'High-degree regulation (1025 relations)'
    },
    # 2. Single entity, high-degree technology (tests different entity type)
    {
        'id': 'q2',
        'query': 'How is ChatGPT regulated?',
        'category': 'technology_regulation',
        'entities': 1,
        'target_entity': 'ChatGPT',
        'expected_types': ['Technology'],
        'expected_mode': 'graph',
        'description': 'High-degree technology (1070 relations)'
    },
    # 3. Cross-jurisdictional (multi-location aggregation)
    {
        'id': 'q3',
        'query': 'Which jurisdictions regulate facial recognition?',
        'category': 'cross_jurisdictional',
        'entities': 2,
        'expected_types': ['Technology', 'Location'],
        'expected_mode': 'graph',
        'description': 'Technology across jurisdictions'
    },
    # 4. Comparison (Steiner Tree connecting multiple terminals)
    {
        'id': 'q4',
        'query': 'Compare China and EU approaches to AI governance',
        'category': 'comparison',
        'entities': 3,
        'expected_types': ['Location', 'Technology'],
        'expected_mode': 'dual',
        'description': 'Cross-jurisdiction comparison, tests Steiner Tree'
    },
    # 5. Relation-specific (tests `requires` predicate - 27k instances)
    {
        'id': 'q5',
        'query': 'What transparency requirements does the EU AI Act impose?',
        'category': 'relation_traversal',
        'entities': 2,
        'expected_types': ['Regulation', 'RegulatoryConcept'],
        'expected_relations': ['requires'],
        'expected_mode': 'graph',
        'description': 'Tests relation traversal via requires predicate'
    },
    # 6. Out-of-domain (graceful failure, Easter egg)
    {
        'id': 'q6',
        'query': "What is Snoopy's arch enemy?",
        'category': 'out_of_domain',
        'entities': 0,
        'expected_types': [],
        'expected_mode': 'semantic',
        'description': 'Out-of-domain, should fail gracefully (Red Baron!)'
    }
]


# =============================================================================
# FULL TEST SET (30 queries) - for comprehensive evaluation
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
        'target_degree': 1025,
        'expected_types': ['Regulation'],
        'expected_mode': 'graph',
        'description': 'High-degree regulation'
    },
    {
        'id': 'sf2',
        'query': 'What is ChatGPT?',
        'category': 'simple_factual',
        'entities': 1,
        'target_entity': 'ChatGPT',
        'target_degree': 1070,
        'expected_types': ['Technology'],
        'expected_mode': 'graph',
        'description': 'High-degree technology'
    },
    {
        'id': 'sf3',
        'query': 'What does transparency mean in AI governance?',
        'category': 'simple_factual',
        'entities': 1,
        'target_entity': 'transparency',
        'target_degree': 1811,
        'expected_types': ['RegulatoryConcept'],
        'expected_mode': 'semantic',
        'description': 'High-degree concept (abstract)'
    },
    {
        'id': 'sf4',
        'query': 'What is the role of the European Commission in AI?',
        'category': 'simple_factual',
        'entities': 1,
        'target_entity': 'European Commission',
        'target_degree': 764,
        'expected_types': ['Organization'],
        'expected_mode': 'dual',
        'description': 'Organization entity'
    },
    {
        'id': 'sf5',
        'query': 'What are discriminatory impacts of AI?',
        'category': 'simple_factual',
        'entities': 1,
        'target_entity': 'discriminatory impacts',
        'target_degree': 1482,
        'expected_types': ['Risk'],
        'expected_mode': 'semantic',
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
        'expected_types': ['Risk', 'Technology'],
        'expected_relations': ['appliesto', 'poses'],
        'expected_mode': 'dual',
        'description': '2-entity, risk+technology'
    },
    {
        'id': 'mf2',
        'query': 'How does machine learning relate to privacy risks?',
        'category': 'multi_entity',
        'entities': 2,
        'expected_types': ['Technology', 'Risk'],
        'expected_relations': ['poses', 'mitigates'],
        'expected_mode': 'graph',
        'description': '2-entity, technology→risk'
    },
    {
        'id': 'mf3',
        'query': 'What accountability requirements apply to AI systems?',
        'category': 'multi_entity',
        'entities': 2,
        'expected_types': ['RegulatoryConcept', 'Technology'],
        'expected_relations': ['requires', 'appliesto'],
        'expected_mode': 'dual',
        'description': '2-entity, concept+technology'
    },
    {
        'id': 'mf4',
        'query': 'How do algorithms affect human rights?',
        'category': 'multi_entity',
        'entities': 2,
        'expected_types': ['Technology', 'PoliticalConcept'],
        'expected_relations': ['affects', 'influences'],
        'expected_mode': 'semantic',
        'description': '2-entity, tech→political'
    },
    {
        'id': 'mf5',
        'query': 'What security risks do AI systems pose?',
        'category': 'multi_entity',
        'entities': 2,
        'expected_types': ['Risk', 'Technology'],
        'expected_relations': ['poses'],
        'expected_mode': 'graph',
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
        'expected_types': ['Technology'],
        'jurisdictions': 'multiple',
        'expected_mode': 'graph',
        'description': 'Tech across jurisdictions'
    },
    {
        'id': 'cj2',
        'query': 'How does the EU regulate AI compared to China?',
        'category': 'cross_jurisdictional',
        'entities': 3,
        'expected_types': ['Location', 'Technology'],
        'jurisdictions': ['EU', 'CN'],
        'expected_mode': 'dual',
        'description': 'EU vs China comparison'
    },
    {
        'id': 'cj3',
        'query': 'What AI regulations exist in Singapore?',
        'category': 'cross_jurisdictional',
        'entities': 2,
        'expected_types': ['Location', 'Regulation'],
        'jurisdictions': ['SG'],
        'expected_mode': 'graph',
        'description': 'Single jurisdiction (Singapore)'
    },
    {
        'id': 'cj4',
        'query': 'Compare US and EU approaches to AI transparency',
        'category': 'cross_jurisdictional',
        'entities': 4,
        'expected_types': ['Location', 'RegulatoryConcept'],
        'jurisdictions': ['US', 'EU'],
        'expected_mode': 'dual',
        'description': 'US vs EU on concept'
    },
    {
        'id': 'cj5',
        'query': 'How does Japan address AI ethics?',
        'category': 'cross_jurisdictional',
        'entities': 2,
        'expected_types': ['Location', 'RegulatoryConcept'],
        'jurisdictions': ['JP'],
        'expected_mode': 'semantic',
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
        'expected_relations': ['appliesto', 'requires'],
        'expected_mode': 'graph',
        'description': 'appliesto relation (51k instances)'
    },
    {
        'id': 'rs2',
        'query': 'What does the EU AI Act require for high-risk systems?',
        'category': 'relation_specific',
        'entities': 3,
        'expected_relations': ['requires'],
        'expected_mode': 'graph',
        'description': 'requires relation (27k instances)'
    },
    {
        'id': 'rs3',
        'query': 'How is AI regulation enforced?',
        'category': 'relation_specific',
        'entities': 2,
        'expected_relations': ['enforces'],
        'expected_mode': 'dual',
        'description': 'enforces relation (12k instances)'
    },
    {
        'id': 'rs4',
        'query': 'What measures mitigate AI bias?',
        'category': 'relation_specific',
        'entities': 2,
        'expected_relations': ['mitigates'],
        'expected_mode': 'graph',
        'description': 'mitigates relation (9k instances)'
    },
    {
        'id': 'rs5',
        'query': 'What risks does automated decision-making pose?',
        'category': 'relation_specific',
        'entities': 2,
        'expected_relations': ['poses'],
        'expected_mode': 'semantic',
        'description': 'poses relation (9k instances)'
    },
    
    # -------------------------------------------------------------------------
    # CATEGORY 5: RESEARCH/ACADEMIC - 4 queries
    # Tests: Paper-heavy retrieval, citations, academic concepts
    # -------------------------------------------------------------------------
    {
        'id': 'ra1',
        'query': 'What academic research discusses algorithmic bias?',
        'category': 'research',
        'entities': 2,
        'expected_types': ['Technology', 'Risk'],
        'source_bias': 'paper',
        'expected_mode': 'semantic',
        'description': 'Academic focus, paper-heavy'
    },
    {
        'id': 'ra2',
        'query': 'What studies examine AI sustainability?',
        'category': 'research',
        'entities': 2,
        'expected_types': ['Technology', 'EconomicConcept'],
        'source_bias': 'paper',
        'expected_mode': 'semantic',
        'description': 'Sustainability research'
    },
    {
        'id': 'ra3',
        'query': 'How do researchers define explainable AI?',
        'category': 'research',
        'entities': 1,
        'expected_types': ['TechnicalConcept'],
        'source_bias': 'paper',
        'expected_mode': 'semantic',
        'description': 'Technical concept from papers'
    },
    {
        'id': 'ra4',
        'query': 'What is the academic perspective on AI governance?',
        'category': 'research',
        'entities': 2,
        'expected_types': ['Technology', 'RegulatoryConcept'],
        'source_bias': 'paper',
        'expected_mode': 'semantic',
        'description': 'Governance from academic lens'
    },
    
    # -------------------------------------------------------------------------
    # CATEGORY 6: EDGE CASES - 6 queries
    # Tests: Out-of-domain, ambiguous, very specific, alias resolution
    # -------------------------------------------------------------------------
    {
        'id': 'ec1',
        'query': "What is Snoopy's arch enemy?",
        'category': 'out_of_domain',
        'entities': 0,
        'expected_mode': 'semantic',
        'description': 'Out-of-domain (graceful failure)'
    },
    {
        'id': 'ec2',
        'query': 'What is AI?',
        'category': 'ambiguous',
        'entities': 1,
        'target_entity': 'AI',
        'target_degree': 3706,
        'expected_mode': 'semantic',
        'description': 'Highest-degree entity, very broad'
    },
    {
        'id': 'ec3',
        'query': 'Explain GDPR implications for machine learning',
        'category': 'specific',
        'entities': 2,
        'expected_mode': 'graph',
        'description': 'Specific regulation + technology'
    },
    {
        'id': 'ec4',
        'query': 'What is the AI Act?',
        'category': 'alias_resolution',
        'entities': 1,
        'target_entity': 'EU AI Act',
        'resolution_type': 'alias',
        'expected_mode': 'graph',
        'description': 'Alias test (AI Act → EU AI Act)'
    },
    {
        'id': 'ec5',
        'query': 'How does ML affect data protection?',
        'category': 'alias_resolution',
        'entities': 2,
        'resolution_type': 'alias',
        'expected_mode': 'dual',
        'description': 'Abbreviation test (ML → machine learning)'
    },
    {
        'id': 'ec6',
        'query': 'What regulations mention neural networks in Austria?',
        'category': 'specific_jurisdiction',
        'entities': 2,
        'jurisdictions': ['AT'],
        'expected_mode': 'graph',
        'description': 'Very specific (Austria = high entity count)'
    }
]


# =============================================================================
# COMBINED for backward compatibility
# =============================================================================

TEST_QUERIES = QUICK_QUERIES  # Default to quick set


def get_queries(mode='quick'):
    """
    Get test queries by mode.
    
    Args:
        mode: 'quick' (6 queries) or 'full' (30 queries)
    
    Returns:
        List of query dictionaries
    """
    if mode == 'full':
        return FULL_QUERIES
    return QUICK_QUERIES


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
    
    print("\nBy Category:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {count:2d} | {cat}")
    
    # Entity count distribution
    entity_counts = {}
    for q in FULL_QUERIES:
        ec = q.get('entities', 0)
        entity_counts[ec] = entity_counts.get(ec, 0) + 1
    
    print("\nBy Entity Count:")
    for ec, count in sorted(entity_counts.items()):
        print(f"  {count:2d} | {ec} entities")
    
    # Expected mode distribution
    modes = {}
    for q in FULL_QUERIES:
        mode = q.get('expected_mode', 'unknown')
        modes[mode] = modes.get(mode, 0) + 1
    
    print("\nBy Expected Mode:")
    for mode, count in sorted(modes.items(), key=lambda x: -x[1]):
        print(f"  {count:2d} | {mode}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    print_coverage_report()