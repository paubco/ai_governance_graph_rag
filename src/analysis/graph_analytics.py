# -*- coding: utf-8 -*-
"""
Graph Analytics for GraphRAG Knowledge Graph (v1.1)

Computes thesis-relevant metrics:
1. Basic Counts - Nodes, relationships, coverage
2. RAKG Metrics - Relation density, predicate diversity
3. Network Structure - Degree distribution, power-law analysis
4. Cross-Domain Analysis - Academic-regulatory bridges (key contribution)
5. Citation Network - L2 publications, MATCHED_TO links
6. Quality Metrics - Provenance coverage, orphan detection

Usage:
    python -m src.analysis.graph_analytics
    python -m src.analysis.graph_analytics --output reports/graph_stats.json

Author: Pau Barba i Colomer
Created: 2025-12-15
Modified: 2025-12-22

References:
    - RAKG (Zhang et al. 2025) - Entity Density, Relation Richness
    - RAGulating (2025) - Navigation metrics
    - Xue & Zou (2022) - KG Quality Management survey
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neo4j import GraphDatabase, Driver
from dotenv import load_dotenv
from src.utils.logger import get_logger

load_dotenv(PROJECT_ROOT / '.env')
logger = get_logger(__name__)


# ============================================================================
# CORE METRICS
# ============================================================================

def get_coverage_stats(driver: Driver) -> Dict[str, int]:
    """Get overall coverage statistics."""
    query = """
    MATCH (j:Jurisdiction) WITH count(j) as jurisdictions
    MATCH (p:Publication) WITH jurisdictions, count(p) as publications
    MATCH (e:Entity) WITH jurisdictions, publications, count(e) as entities
    MATCH (c:Chunk) WITH jurisdictions, publications, entities, count(c) as chunks
    MATCH ()-[r:RELATION]->()
    RETURN jurisdictions, publications, entities, chunks, count(r) as relations
    """
    with driver.session() as session:
        result = session.run(query)
        return dict(result.single())


def get_node_counts(driver: Driver) -> Dict[str, int]:
    """Get count of each node label."""
    query = """
    CALL db.labels() YIELD label
    CALL { WITH label MATCH (n) WHERE label IN labels(n) RETURN count(n) AS count }
    RETURN label, count ORDER BY count DESC
    """
    with driver.session() as session:
        result = session.run(query)
        return {r["label"]: r["count"] for r in result}


def get_relationship_counts(driver: Driver) -> Dict[str, int]:
    """Get count of each relationship type."""
    query = """
    MATCH ()-[r]->()
    RETURN type(r) as rel_type, count(r) as count
    ORDER BY count DESC
    """
    with driver.session() as session:
        result = session.run(query)
        return {r["rel_type"]: r["count"] for r in result}


# ============================================================================
# RAKG METRICS
# ============================================================================

def get_relation_density(driver: Driver) -> Dict[str, Any]:
    """Calculate relation density (relations per entity) - RAKG metric."""
    query = """
    MATCH (e:Entity)
    WITH count(e) AS total_entities
    MATCH ()-[r:RELATION]->()
    WITH total_entities, count(r) AS total_relations
    RETURN 
        total_entities,
        total_relations,
        round(total_relations * 1.0 / total_entities, 2) AS relations_per_entity
    """
    with driver.session() as session:
        result = session.run(query)
        record = dict(result.single())
        
        density = record.get('relations_per_entity', 0)
        if density < 5:
            record['interpretation'] = 'Sparse (typical for regulatory KGs)'
        elif density < 10:
            record['interpretation'] = 'Moderate (good for domain KGs)'
        elif density < 20:
            record['interpretation'] = 'Dense (rich semantic network)'
        else:
            record['interpretation'] = 'Very dense (highly interconnected)'
        
        return record


def get_predicate_diversity(driver: Driver) -> Dict[str, Any]:
    """Calculate predicate diversity (unique predicates per entity)."""
    query = """
    MATCH (e:Entity)-[r:RELATION]->()
    WITH e, count(DISTINCT r.predicate) AS unique_predicates
    RETURN 
        avg(unique_predicates) AS avg_predicates_per_entity,
        max(unique_predicates) AS max_predicates,
        percentileCont(unique_predicates, 0.5) AS median_predicates
    """
    with driver.session() as session:
        result = session.run(query)
        return dict(result.single())


def get_predicate_distribution(driver: Driver, limit: int = 30) -> List[Dict]:
    """Get distribution of relation predicates."""
    query = f"""
    MATCH ()-[r:RELATION]->()
    WITH r.predicate as predicate, count(*) as count
    ORDER BY count DESC
    LIMIT {limit}
    RETURN predicate, count
    """
    with driver.session() as session:
        result = session.run(query)
        return [dict(r) for r in result]


# ============================================================================
# DEGREE ANALYSIS
# ============================================================================

def get_degree_stats(driver: Driver) -> Dict[str, float]:
    """Get aggregate degree statistics."""
    query = """
    MATCH (n)
    WITH n, size((n)--()) AS degree
    RETURN 
        avg(degree) AS avg_degree,
        min(degree) AS min_degree,
        max(degree) AS max_degree,
        stdev(degree) AS stdev_degree
    """
    with driver.session() as session:
        result = session.run(query)
        record = result.single()
        return {
            "avg_degree": round(record["avg_degree"], 2) if record["avg_degree"] else 0,
            "min_degree": record["min_degree"] or 0,
            "max_degree": record["max_degree"] or 0,
            "stdev_degree": round(record["stdev_degree"], 2) if record["stdev_degree"] else 0
        }


def get_degree_distribution_buckets(driver: Driver) -> List[Dict]:
    """Analyze degree distribution with buckets for power-law analysis."""
    query = """
    MATCH (e:Entity)
    OPTIONAL MATCH (e)-[r:RELATION]-()
    WITH e, count(r) AS degree
    WITH 
      CASE 
        WHEN degree = 0 THEN '0 (isolated)'
        WHEN degree <= 5 THEN '1-5'
        WHEN degree <= 20 THEN '6-20'
        WHEN degree <= 100 THEN '21-100'
        WHEN degree <= 500 THEN '101-500'
        ELSE '500+'
      END AS bucket,
      count(*) AS entity_count
    RETURN bucket, entity_count
    ORDER BY 
      CASE bucket
        WHEN '0 (isolated)' THEN 0
        WHEN '1-5' THEN 1
        WHEN '6-20' THEN 2
        WHEN '21-100' THEN 3
        WHEN '101-500' THEN 4
        ELSE 5
      END
    """
    with driver.session() as session:
        result = session.run(query)
        return [dict(r) for r in result]


def get_degree_distribution_raw(driver: Driver) -> List[Dict]:
    """Get raw degree distribution for plotting."""
    query = """
    MATCH (e:Entity)
    WITH e, size((e)-[:RELATION]-()) AS degree
    RETURN degree, count(e) AS count
    ORDER BY degree
    """
    with driver.session() as session:
        result = session.run(query)
        return [dict(r) for r in result]


def get_top_entities_by_degree(driver: Driver, limit: int = 50) -> List[Dict]:
    """Get top entities by total degree."""
    query = f"""
    MATCH (e:Entity)
    WITH e, size((e)-[:RELATION]-()) AS degree
    ORDER BY degree DESC
    LIMIT {limit}
    RETURN 
        e.name AS name,
        e.entity_id AS entity_id,
        e.type AS type,
        degree
    """
    with driver.session() as session:
        result = session.run(query)
        return [dict(r) for r in result]


# ============================================================================
# ENTITY TYPE ANALYSIS
# ============================================================================

def get_entity_type_distribution(driver: Driver) -> List[Dict]:
    """Get distribution of entity types (the 'type' property)."""
    query = """
    MATCH (e:Entity)
    WITH e.type as type, count(*) as count
    ORDER BY count DESC
    RETURN type, count
    """
    with driver.session() as session:
        result = session.run(query)
        return [dict(r) for r in result]


# ============================================================================
# CROSS-DOMAIN ANALYSIS (Key Thesis Contribution)
# ============================================================================

def get_chunk_source_distribution(driver: Driver) -> Dict[str, int]:
    """Get chunk counts by source type (regulatory vs academic)."""
    query = """
    MATCH (c:Chunk)
    RETURN c.doc_type AS doc_type, count(c) AS count
    """
    with driver.session() as session:
        result = session.run(query)
        return {r["doc_type"]: r["count"] for r in result}


def get_academic_regulatory_bridges(driver: Driver) -> Dict[str, Any]:
    """
    Analyze entities that bridge academic and regulatory domains.
    Key contribution: Cross-jurisdictional + cross-domain integration.
    """
    # Domain distribution
    distribution_query = """
    MATCH (e:Entity)-[:EXTRACTED_FROM]->(c:Chunk)
    WITH e, 
         sum(CASE WHEN c.doc_type = 'regulatory' THEN 1 ELSE 0 END) as in_reg,
         sum(CASE WHEN c.doc_type = 'academic' THEN 1 ELSE 0 END) as in_acad
    WITH 
      CASE 
        WHEN in_reg > 0 AND in_acad > 0 THEN 'bridging'
        WHEN in_reg > 0 THEN 'regulatory_only'
        WHEN in_acad > 0 THEN 'academic_only'
        ELSE 'no_chunks'
      END as category,
      count(e) as entity_count
    RETURN category, entity_count
    """
    
    # Top bridges
    top_bridges_query = """
    MATCH (e:Entity)-[:EXTRACTED_FROM]->(c:Chunk)
    WITH e, 
         sum(CASE WHEN c.doc_type = 'regulatory' THEN 1 ELSE 0 END) as reg_count,
         sum(CASE WHEN c.doc_type = 'academic' THEN 1 ELSE 0 END) as acad_count
    WHERE reg_count > 0 AND acad_count > 0
    WITH e, reg_count, acad_count, reg_count + acad_count as total
    ORDER BY total DESC
    LIMIT 20
    RETURN e.name as name, e.type as type, reg_count, acad_count, total
    """
    
    with driver.session() as session:
        dist_result = session.run(distribution_query)
        distribution = [dict(r) for r in dist_result]
        
        top_result = session.run(top_bridges_query)
        top_bridges = [dict(r) for r in top_result]
    
    # Calculate cohesion
    total = sum(d['entity_count'] for d in distribution)
    bridging = next((d['entity_count'] for d in distribution if d['category'] == 'bridging'), 0)
    
    return {
        'domain_distribution': distribution,
        'top_bridges': top_bridges,
        'cohesion': {
            'total_entities': total,
            'bridging_entities': bridging,
            'bridging_pct': round(100 * bridging / total, 2) if total > 0 else 0,
            'interpretation': 'Highly integrated' if (bridging / total if total > 0 else 0) > 0.1 else 'Domain-siloed'
        }
    }


def get_cross_jurisdictional_entities(driver: Driver, min_jurisdictions: int = 2) -> List[Dict]:
    """Get entities mentioned in multiple jurisdictions."""
    query = f"""
    MATCH (e:Entity)-[:SAME_AS]->(j:Jurisdiction)
    WITH e, collect(DISTINCT j.code) as jurisdictions
    WHERE size(jurisdictions) >= {min_jurisdictions}
    RETURN e.name as name, e.type as type, jurisdictions, size(jurisdictions) as coverage
    ORDER BY coverage DESC
    LIMIT 50
    """
    with driver.session() as session:
        result = session.run(query)
        return [dict(r) for r in result]


def get_jurisdiction_entity_counts(driver: Driver) -> List[Dict]:
    """Get entity counts per jurisdiction via SAME_AS links."""
    query = """
    MATCH (e:Entity)-[:SAME_AS]->(j:Jurisdiction)
    RETURN j.code as jurisdiction, j.name as name, count(e) as entity_count
    ORDER BY entity_count DESC
    """
    with driver.session() as session:
        result = session.run(query)
        return [dict(r) for r in result]


# ============================================================================
# CITATION NETWORK
# ============================================================================

def get_citation_stats(driver: Driver) -> Dict[str, Any]:
    """Get citation network statistics."""
    stats_query = """
    MATCH (p:Publication) WITH count(p) as l1_pubs
    MATCH (l2:L2Publication) WITH l1_pubs, count(l2) as l2_pubs
    MATCH ()-[m:MATCHED_TO]->() WITH l1_pubs, l2_pubs, count(m) as matched_to
    MATCH ()-[c:CITES]->()
    RETURN l1_pubs, l2_pubs, matched_to, count(c) as cites
    """
    
    top_cited_query = """
    MATCH (e:Entity)-[:MATCHED_TO]->(l2:L2Publication)
    WITH l2, count(e) as times_cited
    ORDER BY times_cited DESC
    LIMIT 10
    RETURN l2.title as title, l2.author as author, l2.year as year, times_cited
    """
    
    with driver.session() as session:
        stats = dict(session.run(stats_query).single())
        top_cited = [dict(r) for r in session.run(top_cited_query)]
    
    return {
        'counts': stats,
        'top_cited_l2': top_cited
    }


# ============================================================================
# ALIAS STATS (v1.1)
# ============================================================================

def get_alias_stats(driver: Driver) -> Dict[str, Any]:
    """Get statistics about entity aliases."""
    query = """
    MATCH (e:Entity)
    WHERE e.aliases IS NOT NULL AND size(e.aliases) > 0
    WITH e, size(e.aliases) AS alias_count
    RETURN 
        count(e) AS entities_with_aliases,
        sum(alias_count) AS total_aliases,
        avg(alias_count) AS avg_aliases,
        max(alias_count) AS max_aliases
    """
    with driver.session() as session:
        result = session.run(query)
        record = result.single()
        if record:
            return {
                "entities_with_aliases": record["entities_with_aliases"],
                "total_aliases": record["total_aliases"],
                "avg_aliases": round(record["avg_aliases"], 2) if record["avg_aliases"] else 0,
                "max_aliases": record["max_aliases"]
            }
        return {}


# ============================================================================
# QUALITY METRICS
# ============================================================================

def get_provenance_coverage(driver: Driver) -> Dict[str, Any]:
    """Calculate provenance coverage (entities with chunk links)."""
    query = """
    MATCH (e:Entity)
    WITH count(e) as total_entities
    MATCH (e:Entity)-[:EXTRACTED_FROM]->(:Chunk)
    WITH total_entities, count(DISTINCT e) as entities_with_chunks
    RETURN 
        total_entities,
        entities_with_chunks,
        round(entities_with_chunks * 100.0 / total_entities, 1) as coverage_pct
    """
    with driver.session() as session:
        result = session.run(query)
        return dict(result.single())


def get_orphan_stats(driver: Driver) -> Dict[str, int]:
    """Detect orphan nodes (nodes with no relationships)."""
    query = """
    MATCH (n)
    WHERE NOT (n)-[]-()
    RETURN labels(n)[0] as node_type, count(n) as orphan_count
    ORDER BY orphan_count DESC
    """
    with driver.session() as session:
        result = session.run(query)
        return {r['node_type']: r['orphan_count'] for r in result}


# ============================================================================
# CSV EXPORT
# ============================================================================

def export_to_csv(data: List[Dict], filepath: Path, fieldnames: List[str] = None):
    """Export list of dicts to CSV for pgfplots."""
    if not data:
        return
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    logger.info(f"Exported {len(data)} rows to {filepath}")


# ============================================================================
# MAIN PROCESSOR
# ============================================================================

class GraphAnalytics:
    """Runs all graph analytics and exports results."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or (PROJECT_ROOT / 'data' / 'processed' / 'graph_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = os.getenv('NEO4J_USER', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD')
        
        if not self.password:
            raise ValueError("NEO4J_PASSWORD not set")
        
        self.driver = None
    
    def connect(self):
        logger.info(f"Connecting to Neo4j at {self.uri}")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        with self.driver.session() as s:
            s.run("RETURN 1")
        logger.info("Connected")
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def run(self) -> Dict[str, Any]:
        """Run all analytics and export results."""
        start = datetime.now()
        
        try:
            self.connect()
            
            results = {
                'metadata': {
                    'timestamp': start.isoformat(),
                    'neo4j_uri': self.uri
                }
            }
            
            # 1. Coverage
            print("\n" + "="*60)
            print("GRAPH ANALYTICS (v1.1)")
            print("="*60)
            
            coverage = get_coverage_stats(self.driver)
            results['coverage'] = coverage
            print(f"\n📊 COVERAGE:")
            print(f"   Jurisdictions: {coverage['jurisdictions']}")
            print(f"   Publications:  {coverage['publications']}")
            print(f"   Entities:      {coverage['entities']:,}")
            print(f"   Chunks:        {coverage['chunks']:,}")
            print(f"   Relations:     {coverage['relations']:,}")
            
            # 2. Node/Relationship counts
            results['node_counts'] = get_node_counts(self.driver)
            results['relationship_counts'] = get_relationship_counts(self.driver)
            
            print(f"\n📦 NODE COUNTS:")
            for label, count in results['node_counts'].items():
                print(f"   {label}: {count:,}")
            
            print(f"\n🔗 RELATIONSHIP COUNTS:")
            for rel, count in results['relationship_counts'].items():
                print(f"   {rel}: {count:,}")
            
            # 3. RAKG Metrics
            density = get_relation_density(self.driver)
            diversity = get_predicate_diversity(self.driver)
            results['relation_density'] = density
            results['predicate_diversity'] = diversity
            
            print(f"\n📈 RAKG METRICS:")
            print(f"   Relation density: {density['relations_per_entity']:.2f} rels/entity")
            print(f"   → {density['interpretation']}")
            print(f"   Predicate diversity: {diversity['avg_predicates_per_entity']:.2f} predicates/entity")
            
            # 4. Degree distribution
            degree_stats = get_degree_stats(self.driver)
            degree_buckets = get_degree_distribution_buckets(self.driver)
            degree_raw = get_degree_distribution_raw(self.driver)
            results['degree_stats'] = degree_stats
            results['degree_buckets'] = degree_buckets
            
            print(f"\n📊 DEGREE DISTRIBUTION:")
            print(f"   Avg: {degree_stats['avg_degree']:.2f}, Max: {degree_stats['max_degree']}, Stdev: {degree_stats['stdev_degree']:.2f}")
            for bucket in degree_buckets:
                print(f"   {bucket['bucket']:15} → {bucket['entity_count']:,} entities")
            
            # Export for pgfplots
            export_to_csv(degree_raw, self.output_dir / 'degree_distribution.csv')
            
            # 5. Entity types
            entity_types = get_entity_type_distribution(self.driver)
            results['entity_types'] = entity_types
            export_to_csv(entity_types, self.output_dir / 'entity_type_distribution.csv')
            
            print(f"\n🏷️ TOP ENTITY TYPES:")
            for et in entity_types[:10]:
                print(f"   {et['type']}: {et['count']:,}")
            
            # 6. Top predicates
            predicates = get_predicate_distribution(self.driver)
            results['predicates'] = predicates
            export_to_csv(predicates, self.output_dir / 'predicate_distribution.csv')
            
            print(f"\n🔗 TOP PREDICATES:")
            for p in predicates[:10]:
                print(f"   {p['predicate']}: {p['count']:,}")
            
            # 7. Top entities
            top_entities = get_top_entities_by_degree(self.driver)
            results['top_entities'] = top_entities
            export_to_csv(top_entities, self.output_dir / 'top_entities.csv')
            
            print(f"\n🏆 TOP ENTITIES BY DEGREE:")
            for i, e in enumerate(top_entities[:10], 1):
                print(f"   {i}. {e['name'][:40]:40} ({e['type']}) → {e['degree']}")
            
            # 8. Cross-domain bridges (key contribution)
            bridges = get_academic_regulatory_bridges(self.driver)
            results['cross_domain_bridges'] = bridges
            
            print(f"\n🌉 ACADEMIC-REGULATORY BRIDGES:")
            cohesion = bridges['cohesion']
            print(f"   Bridging entities: {cohesion['bridging_entities']:,} ({cohesion['bridging_pct']:.1f}%)")
            print(f"   → {cohesion['interpretation']}")
            
            print(f"\n   Domain Distribution:")
            for d in bridges['domain_distribution']:
                print(f"     {d['category']:20} → {d['entity_count']:,}")
            
            # 9. Cross-jurisdictional
            cross_jur = get_cross_jurisdictional_entities(self.driver)
            results['cross_jurisdictional'] = cross_jur
            
            if cross_jur:
                print(f"\n🌍 CROSS-JURISDICTIONAL ENTITIES:")
                print(f"   Entities in 2+ jurisdictions: {len(cross_jur)}")
                print(f"   Max coverage: {max(e['coverage'] for e in cross_jur)} jurisdictions")
                for i, e in enumerate(cross_jur[:5], 1):
                    print(f"   {i}. {e['name'][:40]:40} → {e['coverage']} jurisdictions")
            
            # 10. Jurisdiction counts
            jur_counts = get_jurisdiction_entity_counts(self.driver)
            results['jurisdiction_counts'] = jur_counts
            export_to_csv(jur_counts, self.output_dir / 'jurisdiction_entity_counts.csv')
            
            # 11. Citation network
            citations = get_citation_stats(self.driver)
            results['citation_network'] = citations
            
            print(f"\n📖 CITATION NETWORK:")
            c = citations['counts']
            print(f"   L1 Publications: {c['l1_pubs']}")
            print(f"   L2 Publications: {c['l2_pubs']}")
            print(f"   MATCHED_TO: {c['matched_to']}")
            print(f"   CITES: {c['cites']}")
            
            # 12. Alias stats
            aliases = get_alias_stats(self.driver)
            results['alias_stats'] = aliases
            
            if aliases:
                print(f"\n🔤 ALIAS STATISTICS:")
                print(f"   Entities with aliases: {aliases['entities_with_aliases']:,}")
                print(f"   Total aliases: {aliases['total_aliases']:,}")
                print(f"   Avg per entity: {aliases['avg_aliases']:.2f}")
            
            # 13. Quality metrics
            provenance = get_provenance_coverage(self.driver)
            orphans = get_orphan_stats(self.driver)
            results['provenance'] = provenance
            results['orphans'] = orphans
            
            print(f"\n✓ QUALITY METRICS:")
            print(f"   Provenance coverage: {provenance['coverage_pct']:.1f}%")
            if orphans:
                print(f"   Orphan nodes: {sum(orphans.values())}")
            else:
                print(f"   No orphan nodes ✓")
            
            # Save full results
            with open(self.output_dir / 'graph_analytics.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            elapsed = (datetime.now() - start).total_seconds()
            print(f"\n{'='*60}")
            print(f"✓ Analysis complete in {elapsed:.1f}s")
            print(f"Results: {self.output_dir}")
            print("="*60)
            
            return results
            
        finally:
            self.close()


def main():
    parser = argparse.ArgumentParser(description='Graph analytics for GraphRAG KG')
    parser.add_argument('--output', '-o', type=str, help='Output directory')
    args = parser.parse_args()
    
    output_dir = Path(args.output) if args.output else None
    analytics = GraphAnalytics(output_dir)
    analytics.run()


if __name__ == '__main__':
    main()