# -*- coding: utf-8 -*-
"""
Module: test_neo4j_import.py
Package: src.graph.tests
Purpose: Verify Neo4j import completeness and correctness

Author: Pau Barba i Colomer
Created: 2025-12-06
Modified: 2025-12-06

References:
    - PHASE_2B_DESIGN.md § Verification Checks
    - See docs/ARCHITECTURE.md for context
"""

# Standard library
import os
from pathlib import Path
import sys
import argparse

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Local
from src.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


class Neo4jImportVerifier:
    """Verify Neo4j import completeness and correctness."""
    
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize Neo4j connection for verification.
        
        Args:
            uri: Neo4j connection URI
            user: Username
            password: Password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Connected to Neo4j at {uri}")
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
        logger.info("Connection closed")
    
    def verify_node_counts(self, session) -> bool:
        """
        Verify node counts match expected values.
        
        Returns:
            True if all counts within 5% of expected
        """
        logger.info("\n=== NODE COUNT VERIFICATION ===")
        
        checks = {
            "Jurisdiction": ("MATCH (j:Jurisdiction) RETURN count(j)", 48),
            "Publication (L1)": ("MATCH (p:Publication) RETURN count(p)", 158),
            "Author": ("MATCH (a:Author) RETURN count(a)", 638),
            "Journal": ("MATCH (j:Journal) RETURN count(j)", 119),
            "Chunk": ("MATCH (c:Chunk) RETURN count(c)", 25131),
            "Entity": ("MATCH (e:Entity) RETURN count(e)", 76249),
            "L2Publication": ("MATCH (p:L2Publication) RETURN count(p)", 557),  # From Phase 2A
        }
        
        all_passed = True
        for name, (query, expected) in checks.items():
            result = session.run(query).single()[0]
            threshold = expected * 0.95  # Allow 5% variance
            passed = result >= threshold
            status = "✓" if passed else "✗"
            
            logger.info(f"{status} {name}: {result:,} (expected ~{expected:,})")
            
            if not passed:
                all_passed = False
                logger.warning(f"  Count below threshold: {result} < {threshold:.0f}")
        
        return all_passed
    
    def verify_relationship_counts(self, session) -> bool:
        """
        Verify relationship counts match expected values.
        
        Returns:
            True if all counts within reasonable range
        """
        logger.info("\n=== RELATIONSHIP COUNT VERIFICATION ===")
        
        checks = {
            "CONTAINS (Jur→Chunk)": ("MATCH (:Jurisdiction)-[r:CONTAINS]->(:Chunk) RETURN count(r)", 12000, 15000),
            "CONTAINS (Pub→Chunk)": ("MATCH (:Publication)-[r:CONTAINS]->(:Chunk) RETURN count(r)", 12000, 15000),
            "AUTHORED_BY": ("MATCH ()-[r:AUTHORED_BY]->() RETURN count(r)", 300, 500),
            "PUBLISHED_IN": ("MATCH ()-[r:PUBLISHED_IN]->() RETURN count(r)", 158, 158),
            "EXTRACTED_FROM": ("MATCH ()-[r:EXTRACTED_FROM]->() RETURN count(r)", 100000, 130000),
            "RELATION": ("MATCH ()-[r:RELATION]->() RETURN count(r)", 100000, 160000),
            "MATCHED_TO": ("MATCH ()-[r:MATCHED_TO]->() RETURN count(r)", 500, 1500),
            "CITES": ("MATCH ()-[r:CITES]->() RETURN count(r)", 800, 1200),
            "SAME_AS": ("MATCH ()-[r:SAME_AS]->() RETURN count(r)", 0, 50),  # Optional
        }
        
        all_passed = True
        for name, (query, min_expected, max_expected) in checks.items():
            result = session.run(query).single()[0]
            passed = min_expected <= result <= max_expected
            status = "✓" if passed else "✗"
            
            logger.info(f"{status} {name}: {result:,} (expected {min_expected:,}-{max_expected:,})")
            
            if not passed:
                all_passed = False
                logger.warning(f"  Count outside expected range")
        
        return all_passed
    
    def check_orphan_nodes(self, session) -> bool:
        """
        Find nodes with no relationships.
        
        Returns:
            True if no unexpected orphans found
        """
        logger.info("\n=== ORPHAN NODE CHECK ===")
        
        result = session.run("""
            MATCH (n)
            WHERE NOT (n)--()
            RETURN labels(n)[0] as type, count(n) as count
            ORDER BY count DESC
        """)
        
        orphans_found = False
        for record in result:
            node_type = record['type']
            count = record['count']
            
            # L2Publications might legitimately have no relationships if not cited/matched
            if node_type == 'L2Publication':
                logger.info(f"⚠ Orphan {node_type}: {count} (acceptable - not all citations matched)")
            else:
                logger.warning(f"✗ Orphan {node_type}: {count} (unexpected!)")
                orphans_found = True
        
        if not orphans_found:
            logger.info("✓ No unexpected orphan nodes found")
        
        return not orphans_found
    
    def verify_sample_queries(self, session) -> bool:
        """
        Test key query patterns work correctly.
        
        Returns:
            True if all sample queries return results
        """
        logger.info("\n=== SAMPLE QUERY VERIFICATION ===")
        
        all_passed = True
        
        # Test 1: Entity with relations
        logger.info("\n1. Entity with RELATION edges:")
        result = session.run("""
            MATCH (e:Entity)-[r:RELATION]->(target:Entity)
            RETURN e.name, r.predicate, target.name
            LIMIT 5
        """)
        
        count = 0
        for record in result:
            logger.info(f"   {record['e.name']} --[{record['r.predicate']}]--> {record['target.name']}")
            count += 1
        
        if count == 0:
            logger.warning("✗ No entity relations found!")
            all_passed = False
        else:
            logger.info(f"✓ Found {count} sample relations")
        
        # Test 2: Chunk provenance (regulation)
        logger.info("\n2. Chunk provenance (Jurisdiction→Chunk←Entity):")
        result = session.run("""
            MATCH (e:Entity)-[:EXTRACTED_FROM]->(c:Chunk)<-[:CONTAINS]-(j:Jurisdiction)
            RETURN j.code, c.chunk_id, e.name
            LIMIT 3
        """)
        
        count = 0
        for record in result:
            logger.info(f"   Jurisdiction: {record['j.code']}, Entity: {record['e.name']}")
            count += 1
        
        if count == 0:
            logger.warning("✗ No jurisdiction→chunk←entity paths found!")
            all_passed = False
        else:
            logger.info(f"✓ Found {count} sample provenance paths")
        
        # Test 3: Chunk provenance (academic)
        logger.info("\n3. Chunk provenance (Publication→Chunk←Entity):")
        result = session.run("""
            MATCH (e:Entity)-[:EXTRACTED_FROM]->(c:Chunk)<-[:CONTAINS]-(p:Publication)
            RETURN p.title, c.chunk_id, e.name
            LIMIT 3
        """)
        
        count = 0
        for record in result:
            logger.info(f"   Publication: {record['p.title'][:50]}...")
            count += 1
        
        if count == 0:
            logger.warning("✗ No publication→chunk←entity paths found!")
            all_passed = False
        else:
            logger.info(f"✓ Found {count} sample provenance paths")
        
        # Test 4: Citation enrichment
        logger.info("\n4. Citation enrichment (Entity→L2Publication):")
        result = session.run("""
            MATCH (e:Entity)-[:MATCHED_TO]->(l2:L2Publication)
            RETURN e.name, l2.title, l2.author
            LIMIT 3
        """)
        
        count = 0
        for record in result:
            logger.info(f"   Entity: {record['e.name']} → {record['l2.title'][:50]}...")
            count += 1
        
        if count == 0:
            logger.info("⚠ No MATCHED_TO relations found (acceptable if none matched)")
        else:
            logger.info(f"✓ Found {count} citation matches")
        
        # Test 5: Author/Journal metadata
        logger.info("\n5. Publication metadata (Author, Journal):")
        result = session.run("""
            MATCH (p:Publication)-[:AUTHORED_BY]->(a:Author)
            MATCH (p)-[:PUBLISHED_IN]->(j:Journal)
            RETURN p.title, a.name, j.name
            LIMIT 3
        """)
        
        count = 0
        for record in result:
            logger.info(f"   {record['p.title'][:40]}... by {record['a.name']}")
            count += 1
        
        if count == 0:
            logger.warning("✗ No publication metadata paths found!")
            all_passed = False
        else:
            logger.info(f"✓ Found {count} sample metadata paths")
        
        return all_passed
    
    def verify_constraints(self, session) -> bool:
        """
        Verify all constraints are created.
        
        Returns:
            True if all expected constraints exist
        """
        logger.info("\n=== CONSTRAINT VERIFICATION ===")
        
        result = session.run("SHOW CONSTRAINTS")
        constraints = [record['name'] for record in result]
        
        expected_constraints = [
            'jurisdiction_code',
            'publication_scopus',
            'l2pub_id',
            'entity_id',
            'chunk_id',
            'author_id',
            'journal_id',
        ]
        
        all_passed = True
        for constraint_name in expected_constraints:
            if constraint_name in constraints:
                logger.info(f"✓ {constraint_name}")
            else:
                logger.warning(f"✗ {constraint_name} missing!")
                all_passed = False
        
        return all_passed
    
    def run_all_verifications(self) -> bool:
        """
        Run complete verification suite.
        
        Returns:
            True if all verifications pass
        """
        logger.info("Starting Neo4j import verification...")
        
        results = {}
        
        with self.driver.session() as session:
            results['constraints'] = self.verify_constraints(session)
            results['node_counts'] = self.verify_node_counts(session)
            results['relationship_counts'] = self.verify_relationship_counts(session)
            results['orphan_check'] = self.check_orphan_nodes(session)
            results['sample_queries'] = self.verify_sample_queries(session)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("VERIFICATION SUMMARY")
        logger.info("="*60)
        
        for check, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"{status}: {check}")
        
        all_passed = all(results.values())
        
        if all_passed:
            logger.info("\n✓ ALL VERIFICATIONS PASSED")
        else:
            logger.warning("\n✗ SOME VERIFICATIONS FAILED - Review logs above")
        
        return all_passed


def main():
    """Main entry point for verification."""
    parser = argparse.ArgumentParser(
        description='Verify Neo4j import completeness'
    )
    parser.add_argument(
        '--uri',
        default=os.getenv('NEO4J_URI'),
        help='Neo4j URI (default: NEO4J_URI env var)'
    )
    parser.add_argument(
        '--user',
        default=os.getenv('NEO4J_USER', 'neo4j'),
        help='Neo4j username (default: NEO4J_USER env var or "neo4j")'
    )
    parser.add_argument(
        '--password',
        default=os.getenv('NEO4J_PASSWORD'),
        help='Neo4j password (default: NEO4J_PASSWORD env var)'
    )
    
    args = parser.parse_args()
    
    # Validate required args
    if not args.uri or not args.password:
        parser.error("--uri and --password required (or set NEO4J_URI and NEO4J_PASSWORD env vars)")
    
    # Run verification
    verifier = Neo4jImportVerifier(args.uri, args.user, args.password)
    
    try:
        success = verifier.run_all_verifications()
        exit(0 if success else 1)
    finally:
        verifier.close()


if __name__ == '__main__':
    main()