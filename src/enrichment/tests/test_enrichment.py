#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2A Enrichment Preflight Tests.

Quick sanity checks for Scopus parsing, citation matching, and jurisdiction linking.
Run before enrichment pipeline to catch configuration and import errors early.

Author: Pau Barba i Colomer
Created: 2025-12-21

Usage:
    python tests/test_enrichment.py
"""

# Standard library
import sys
from pathlib import Path
import tempfile
import json

# Project root - handles src/enrichment/tests/ location
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """Test all enrichment modules can be imported."""
    print("\n=== TEST: Enrichment Module Imports ===")
    
    modules = [
        ('src.enrichment.scopus_parser', 'ScopusParser'),
        ('src.enrichment.scopus_parser', 'ReferenceParser'),
        ('src.enrichment.citation_matcher', 'CitationEntityIdentifier'),
        ('src.enrichment.citation_matcher', 'CitationMatcher'),
        ('src.enrichment.jurisdiction_matcher', 'JurisdictionMatcher'),
        ('src.enrichment.enrichment_processor', 'EnrichmentProcessor'),
    ]
    
    all_passed = True
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"  ✓ {class_name}")
        except Exception as e:
            print(f"  ✗ {class_name}: {e}")
            all_passed = False
    
    return all_passed


def test_utils_imports():
    """Test utils modules can be imported."""
    print("\n=== TEST: Utils Imports ===")
    
    try:
        from src.utils.dataclasses import Entity, Relation, Chunk
        print("  ✓ dataclasses")
    except Exception as e:
        print(f"  ✗ dataclasses: {e}")
        return False
    
    try:
        from src.utils.id_generator import (
            generate_entity_id, generate_publication_id, 
            generate_author_id, generate_journal_id
        )
        print("  ✓ id_generator")
    except Exception as e:
        print(f"  ✗ id_generator: {e}")
        return False
    
    try:
        from src.utils.io import load_jsonl, save_jsonl
        print("  ✓ io")
    except Exception as e:
        print(f"  ✗ io: {e}")
        return False
    
    return True


def test_scopus_parser():
    """Test ScopusParser with mock CSV row."""
    print("\n=== TEST: ScopusParser ===")
    
    from src.enrichment.scopus_parser import ScopusParser, ReferenceParser
    
    # Create mock CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write('EID,Title,Year,DOI,Source title,Volume,Cited by,Author(s) ID,Author full names,ISSN,References\n')
        f.write('2-s2.0-123456789,Test Paper,2024,10.1234/test,Test Journal,1,5,111;222,Smith, John (111);Doe, Jane (222),1234-5678,"Floridi, L., Ethics, Phil J, (2018)"\n')
        csv_path = Path(f.name)
    
    try:
        parser = ScopusParser(csv_path)
        pubs, authors, journals = parser.parse_publications()
        
        assert len(pubs) == 1, f"Expected 1 publication, got {len(pubs)}"
        assert len(authors) == 2, f"Expected 2 authors, got {len(authors)}"
        assert len(journals) == 1, f"Expected 1 journal, got {len(journals)}"
        
        # Check publication fields
        pub = pubs[0]
        assert pub['scopus_id'] == '2-s2.0-123456789'
        assert pub['title'] == 'Test Paper'
        assert pub['year'] == 2024
        
        print(f"  ✓ Parsed 1 publication, 2 authors, 1 journal")
        
        # Test reference parsing
        ref_parser = ReferenceParser()
        refs_lookup = ref_parser.parse_all_references(pubs)
        
        assert '2-s2.0-123456789' in refs_lookup
        refs = refs_lookup['2-s2.0-123456789']
        assert len(refs) >= 1, f"Expected at least 1 reference, got {len(refs)}"
        
        print(f"  ✓ Parsed {len(refs)} reference(s)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    finally:
        csv_path.unlink()


def test_citation_matcher():
    """Test CitationMatcher with mock data."""
    print("\n=== TEST: CitationMatcher ===")
    
    from src.enrichment.citation_matcher import CitationEntityIdentifier, CitationMatcher
    
    # Mock entities
    entities = [
        {'entity_id': 'ent_001', 'name': 'Floridi (2018)', 'type': 'Citation', 'chunk_ids': ['chunk_001']},
        {'entity_id': 'ent_002', 'name': 'AI Governance', 'type': 'Concept', 'chunk_ids': ['chunk_001']},
    ]
    
    # Mock relations (discusses)
    relations = [
        {'subject_id': 'ent_001', 'predicate': 'discusses', 'object_id': 'ent_002'},
    ]
    
    try:
        # Test identifier
        identifier = CitationEntityIdentifier()
        citations = identifier.identify(entities, relations)
        
        assert len(citations) == 1, f"Expected 1 citation entity, got {len(citations)}"
        assert 'ent_001' in citations
        
        print(f"  ✓ Identified 1 citation entity")
        
        # Test matcher (provenance constraint)
        references_lookup = {
            'scopus_123': [
                {'author': 'Floridi, L.', 'title': 'Ethics', 'year': 2018, 'journal': 'Phil J', 'raw': 'Floridi...'}
            ]
        }
        chunk_to_l1 = {'chunk_001': 'scopus_123'}
        l1_publications = [{'scopus_id': 'scopus_123', 'title': 'Other Paper', 'year': 2024}]
        
        matcher = CitationMatcher(references_lookup, chunk_to_l1, l1_publications)
        
        result = matcher.match_entity_to_reference('ent_001', citations['ent_001'])
        
        assert result is not None, "Expected a match"
        matched_ref, match_result, confidence, method = result
        assert confidence > 0, "Expected positive confidence"
        
        print(f"  ✓ Matched citation with confidence {confidence:.2f} via {method}")
        
        # Test L2 deduplication
        l2_id_1 = matcher.get_or_create_l2(matched_ref)
        l2_id_2 = matcher.get_or_create_l2(matched_ref)  # Same ref
        assert l2_id_1 == l2_id_2, "L2 deduplication failed"
        
        print(f"  ✓ L2 deduplication working")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_jurisdiction_matcher():
    """Test JurisdictionMatcher with mock data."""
    print("\n=== TEST: JurisdictionMatcher ===")
    
    from src.enrichment.jurisdiction_matcher import JurisdictionMatcher
    
    valid_codes = {'US', 'EU', 'GB', 'DE'}
    
    entities = [
        {'entity_id': 'ent_001', 'name': 'United States', 'type': 'Country'},
        {'entity_id': 'ent_002', 'name': 'European Union', 'type': 'Region'},
        {'entity_id': 'ent_003', 'name': 'CNIL', 'type': 'Organization'},  # Should NOT match
    ]
    
    try:
        matcher = JurisdictionMatcher(valid_codes)
        matches = matcher.match_entities(entities)
        
        assert len(matches) == 2, f"Expected 2 matches, got {len(matches)}"
        
        codes_matched = {m['jurisdiction_code'] for m in matches}
        assert 'US' in codes_matched
        assert 'EU' in codes_matched
        
        print(f"  ✓ Matched 2 entities to jurisdictions (US, EU)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_id_generation():
    """Test ID generation functions."""
    print("\n=== TEST: ID Generation ===")
    
    from src.utils.id_generator import (
        generate_entity_id, generate_publication_id,
        generate_author_id, generate_journal_id, generate_l2_publication_id
    )
    
    try:
        # Entity ID
        eid1 = generate_entity_id("EU AI Act", "Regulation")
        eid2 = generate_entity_id("eu ai act", "regulation")  # Should be same
        assert eid1 == eid2, "Entity IDs should be case-insensitive"
        assert eid1.startswith("ent_"), f"Invalid entity ID format: {eid1}"
        print(f"  ✓ Entity ID: {eid1}")
        
        # Publication ID
        pid = generate_publication_id("2-s2.0-123456789", layer=1)
        assert pid.startswith("pub_l1_"), f"Invalid pub ID format: {pid}"
        print(f"  ✓ Publication ID: {pid}")
        
        # L2 Publication ID
        l2id = generate_l2_publication_id("Floridi, L. (2018). Ethics...")
        assert l2id.startswith("pub_l2_"), f"Invalid L2 pub ID format: {l2id}"
        print(f"  ✓ L2 Publication ID: {l2id}")
        
        # Author ID
        aid = generate_author_id("56291236900")
        assert aid == "author_56291236900", f"Invalid author ID: {aid}"
        print(f"  ✓ Author ID: {aid}")
        
        # Journal ID
        jid = generate_journal_id("Nature Machine Intelligence")
        assert jid.startswith("journal_"), f"Invalid journal ID format: {jid}"
        print(f"  ✓ Journal ID: {jid}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def run_all_tests():
    """Run all enrichment preflight tests."""
    print("=" * 60)
    print("PHASE 2A ENRICHMENT PREFLIGHT TESTS")
    print("=" * 60)
    
    results = {}
    
    results['imports'] = test_imports()
    results['foundation'] = test_foundation_imports()
    results['id_generation'] = test_id_generation()
    results['scopus_parser'] = test_scopus_parser()
    results['citation_matcher'] = test_citation_matcher()
    results['jurisdiction_matcher'] = test_jurisdiction_matcher()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL ENRICHMENT PREFLIGHT TESTS PASSED")
        print("Ready to run: python -m src.enrichment.enrichment_processor")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        print("Fix the issues above before running Phase 2A.")
        return 1


if __name__ == '__main__':
    exit(run_all_tests())