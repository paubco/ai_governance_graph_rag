#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph Analytics for GraphRAG Knowledge Graph

Runs analytical queries on Neo4j to compute network statistics,
coverage metrics, and entity centrality measures.

Analytics Categories:
1. Basic Counts: Nodes, relationships, coverage
2. KG-Specific Metrics:
   - Relation density (relations per entity)
   - Predicate diversity (semantic richness)
   - Degree distribution (power-law analysis)
   - Property completeness (metadata quality)
   - Cross-type connectivity (integration quality)
3. Network Science Metrics:
   - Author collaboration network (avg degree, E-R comparison)
   - Citation network analysis (in-degree distribution, preferential attachment)
   - Academic-regulatory bridges (cohesion, domain distribution)
4. Domain Metrics:
   - Cross-jurisdictional coverage
   - Academic-regulatory bridges
   - Citation enrichment statistics
5. Quality Metrics:
   - Provenance coverage
   - Orphan detection

Usage:
    python tests/graph/analyze_graph.py
    python tests/graph/analyze_graph.py --output reports/graph_stats.json
"""

# Standard library
import os
from pathlib import Path
import sys
import json
import argparse
from typing import Dict, List, Any
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Local
from src.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


class GraphAnalyzer:
    """Analyzes GraphRAG knowledge graph structure and statistics."""
    
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Connected to Neo4j at {uri}")
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
        logger.info("Connection closed")
    
    def run_query(self, query: str, description: str) -> List[Dict]:
        """
        Execute Cypher query and return results.
        
        Args:
            query: Cypher query string
            description: Human-readable description for logging
            
        Returns:
            List of result dictionaries
        """
        logger.info(f"Running: {description}")
        with self.driver.session() as session:
            result = session.run(query)
            data = [dict(record) for record in result]
            logger.info(f"  → Returned {len(data)} results")
            return data
    
    def get_node_counts(self) -> List[Dict]:
        """Get count of each node type."""
        query = """
        MATCH (n)
        RETURN labels(n)[0] as node_type, count(n) as count
        ORDER BY count DESC
        """
        return self.run_query(query, "Node counts by type")
    
    def get_relationship_counts(self) -> List[Dict]:
        """Get count of each relationship type."""
        query = """
        MATCH ()-[r]->()
        RETURN type(r) as relationship_type, count(r) as count
        ORDER BY count DESC
        """
        return self.run_query(query, "Relationship counts by type")
    
    def get_coverage_stats(self) -> Dict:
        """Get overall coverage statistics."""
        query = """
        MATCH (j:Jurisdiction)
        WITH count(j) as jurisdictions
        MATCH (p:Publication)
        WITH jurisdictions, count(p) as publications
        MATCH (e:Entity)
        WITH jurisdictions, publications, count(e) as entities
        MATCH (c:Chunk)
        WITH jurisdictions, publications, entities, count(c) as chunks
        MATCH ()-[r:RELATION]->()
        RETURN 
            jurisdictions,
            publications,
            entities,
            chunks,
            count(r) as semantic_relations
        """
        results = self.run_query(query, "Overall coverage statistics")
        return results[0] if results else {}
    
    def get_top_connected_entities(self, limit: int = 20) -> List[Dict]:
        """Get most highly connected entities by degree."""
        query = f"""
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[r:RELATION]-()
        WITH e, count(r) as degree
        WHERE degree > 0
        ORDER BY degree DESC
        LIMIT {limit}
        RETURN e.name as entity, e.type as type, degree
        """
        return self.run_query(query, f"Top {limit} connected entities")
    
    def get_predicate_distribution(self, limit: int = 30) -> List[Dict]:
        """Get distribution of relation predicates."""
        query = f"""
        MATCH ()-[r:RELATION]->()
        WITH r.predicate as predicate, count(*) as frequency
        ORDER BY frequency DESC
        LIMIT {limit}
        RETURN predicate, frequency
        """
        return self.run_query(query, f"Top {limit} predicates")
    
    def get_predicate_diversity(self) -> Dict:
        """Calculate predicate diversity (unique predicates per entity)."""
        query = """
        MATCH (e:Entity)-[r:RELATION]->()
        WITH e, count(DISTINCT r.predicate) AS unique_predicates
        RETURN 
            avg(unique_predicates) AS avg_predicates_per_entity,
            max(unique_predicates) AS max_predicates,
            min(unique_predicates) AS min_predicates,
            percentileCont(unique_predicates, 0.5) AS median_predicates
        """
        results = self.run_query(query, "Predicate diversity (semantic richness)")
        return results[0] if results else {}
    
    def get_relation_density(self) -> Dict:
        """Calculate relation density (relations per entity)."""
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
        results = self.run_query(query, "Relation density")
        result = results[0] if results else {}
        
        # Add benchmark context
        if result:
            density = result.get('relations_per_entity', 0)
            if density < 1.5:
                result['interpretation'] = 'Sparse (typical for regulatory KGs)'
            elif density < 3.0:
                result['interpretation'] = 'Moderate (good for domain KGs)'
            elif density < 5.0:
                result['interpretation'] = 'Dense (rich semantic network)'
            else:
                result['interpretation'] = 'Very dense (highly interconnected)'
        
        return result
    
    def get_degree_distribution_buckets(self) -> List[Dict]:
        """Analyze degree distribution to detect power-law patterns."""
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
        results = self.run_query(query, "Degree distribution (power-law check)")
        
        # Check for power-law pattern
        if results and len(results) >= 3:
            # Power-law: most entities have low degree, few have high degree
            low_degree = sum(r['entity_count'] for r in results if r['bucket'] in ['1-5', '6-20'])
            high_degree = sum(r['entity_count'] for r in results if r['bucket'] in ['21-100', '101-500', '500+'])
            
            if low_degree > high_degree * 3:
                logger.info("  → Power-law distribution detected (typical for KGs) ")
        
        return results
    
    def get_property_completeness(self) -> Dict:
        """Calculate metadata completeness for structured nodes."""
        query = """
        MATCH (p:Publication)
        WITH p,
          (CASE WHEN p.title IS NOT NULL AND p.title <> '' THEN 1 ELSE 0 END +
           CASE WHEN p.year IS NOT NULL THEN 1 ELSE 0 END +
           CASE WHEN p.doi IS NOT NULL AND p.doi <> '' THEN 1 ELSE 0 END +
           CASE WHEN p.abstract IS NOT NULL AND p.abstract <> '' THEN 1 ELSE 0 END) AS filled_fields
        WITH avg(filled_fields) / 4.0 AS avg_completeness
        
        MATCH (a:Author)
        WITH avg_completeness, 
             count(CASE WHEN a.name IS NOT NULL AND a.name <> '' THEN 1 END) * 1.0 / count(a) AS author_completeness
        
        MATCH (j:Journal)
        WITH avg_completeness, author_completeness,
             count(CASE WHEN j.name IS NOT NULL AND j.name <> '' THEN 1 END) * 1.0 / count(j) AS journal_completeness
        
        RETURN 
            round(avg_completeness * 100, 1) AS publication_completeness_pct,
            round(author_completeness * 100, 1) AS author_completeness_pct,
            round(journal_completeness * 100, 1) AS journal_completeness_pct
        """
        results = self.run_query(query, "Property completeness (metadata quality)")
        return results[0] if results else {}
    
    def get_cross_type_connectivity(self) -> List[Dict]:
        """Analyze how different node types connect (integration quality)."""
        query = """
        MATCH (a)-[r]->(b)
        WHERE labels(a)[0] <> labels(b)[0]
        WITH labels(a)[0] AS from_type, 
             labels(b)[0] AS to_type,
             type(r) AS rel_type,
             count(*) AS connections
        ORDER BY connections DESC
        RETURN from_type, to_type, rel_type, connections
        LIMIT 30
        """
        return self.run_query(query, "Cross-type connectivity (integration quality)")
    
    # ========================================================================
    # GRAPH COHESION METRICS (New)
    # ========================================================================
    
    def get_connected_components(self) -> Dict:
        """
        Analyze connected components in the semantic entity graph.
        Shows graph fragmentation - ideally one giant component.
        
        Note: Uses only RELATION edges between entities for semantic cohesion.
        """
        # Count weakly connected components using simple BFS approximation
        # Full WCC needs GDS plugin, so we approximate with sampling
        
        # 1. Get total entities and isolated entities
        isolation_query = """
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[:RELATION]-()
        WITH e, count(*) as degree
        WHERE degree = 0
        RETURN count(e) as isolated_entities
        """
        isolated = self.run_query(isolation_query, "Isolated entities")
        isolated_count = isolated[0]['isolated_entities'] if isolated else 0
        
        # 2. Get total entities
        total_query = """
        MATCH (e:Entity)
        RETURN count(e) as total_entities
        """
        total = self.run_query(total_query, "Total entities")
        total_count = total[0]['total_entities'] if total else 0
        
        # 3. Sample reachability from top hubs to estimate giant component
        # Note: Full reachability requires GDS plugin, using simple connected count
        # Simpler query for approximation
        giant_query = """
        MATCH (e:Entity)-[:RELATION]-()
        WITH DISTINCT e
        RETURN count(e) as connected_entities
        """
        connected = self.run_query(giant_query, "Connected entities (in giant component)")
        connected_count = connected[0]['connected_entities'] if connected else 0
        
        result = {
            'total_entities': total_count,
            'isolated_entities': isolated_count,
            'connected_entities': connected_count,
            'isolation_rate_pct': round(100.0 * isolated_count / max(1, total_count), 2),
            'connectivity_rate_pct': round(100.0 * connected_count / max(1, total_count), 2)
        }
        
        # Interpretation
        if result['connectivity_rate_pct'] > 90:
            result['interpretation'] = 'Highly connected graph (single giant component)'
        elif result['connectivity_rate_pct'] > 70:
            result['interpretation'] = 'Well connected with some isolated clusters'
        elif result['connectivity_rate_pct'] > 50:
            result['interpretation'] = 'Moderately fragmented graph'
        else:
            result['interpretation'] = 'Highly fragmented - many disconnected components'
        
        return result
    
    def get_type_homophily(self) -> Dict:
        """
        Measure type homophily: do entities prefer to connect to same-type entities?
        
        Homophily ratio = intra-type relations / total relations
        High homophily suggests type-based clustering.
        """
        query = """
        MATCH (a:Entity)-[r:RELATION]->(b:Entity)
        WITH a.type as source_type, b.type as target_type, count(*) as rel_count
        WITH source_type, target_type, rel_count,
             CASE WHEN source_type = target_type THEN rel_count ELSE 0 END as intra_type_count
        WITH sum(rel_count) as total_relations,
             sum(intra_type_count) as intra_type_relations
        RETURN 
            total_relations,
            intra_type_relations,
            total_relations - intra_type_relations as inter_type_relations,
            round(100.0 * intra_type_relations / total_relations, 2) as homophily_pct
        """
        results = self.run_query(query, "Type homophily analysis")
        result = results[0] if results else {}
        
        if result:
            homophily = result.get('homophily_pct', 0)
            if homophily > 70:
                result['interpretation'] = 'High homophily - entities cluster by type'
            elif homophily > 40:
                result['interpretation'] = 'Moderate homophily - mixed clustering'
            else:
                result['interpretation'] = 'Low homophily - rich cross-type connections'
        
        return result
    
    def get_type_homophily_breakdown(self) -> List[Dict]:
        """
        Breakdown of homophily by entity type.
        Shows which types are most insular vs most integrative.
        """
        query = """
        MATCH (a:Entity)-[r:RELATION]->(b:Entity)
        WITH a.type as source_type, 
             CASE WHEN a.type = b.type THEN 'intra' ELSE 'inter' END as connection_type,
             count(*) as count
        WITH source_type, 
             sum(CASE WHEN connection_type = 'intra' THEN count ELSE 0 END) as intra,
             sum(CASE WHEN connection_type = 'inter' THEN count ELSE 0 END) as inter,
             sum(count) as total
        WHERE total >= 10
        RETURN 
            source_type,
            intra,
            inter,
            total,
            round(100.0 * intra / total, 1) as homophily_pct
        ORDER BY total DESC
        LIMIT 20
        """
        return self.run_query(query, "Type homophily breakdown")
    
    def get_intra_type_density(self) -> List[Dict]:
        """
        Calculate relation density within each entity type.
        
        Density = actual_relations / possible_relations
        For type with N entities: possible = N*(N-1)/2 (undirected)
        
        Higher density within a type suggests tighter semantic cohesion.
        """
        query = """
        MATCH (e:Entity)
        WITH e.type as entity_type, count(e) as type_count
        WHERE type_count >= 5
        
        MATCH (a:Entity)-[r:RELATION]-(b:Entity)
        WHERE a.type = b.type AND id(a) < id(b)
        WITH a.type as rel_type, count(*) as intra_relations
        
        MATCH (e:Entity)
        WITH e.type as entity_type, count(e) as type_count, 
             coalesce(
                 (SELECT r.intra_relations FROM (
                     MATCH (a:Entity)-[r:RELATION]-(b:Entity) 
                     WHERE a.type = e.type AND a.type = b.type AND id(a) < id(b)
                     RETURN a.type as t, count(*) as intra_relations
                 ) WHERE t = e.type), 
                 0
             ) as intra_rels
        RETURN entity_type, type_count, intra_rels,
               type_count * (type_count - 1) / 2 as possible_relations,
               round(100.0 * intra_rels / (type_count * (type_count - 1) / 2), 4) as density_pct
        ORDER BY type_count DESC
        LIMIT 20
        """
        # Simpler query that works without subqueries
        simple_query = """
        MATCH (e:Entity)
        WITH e.type as entity_type, collect(e) as entities, count(e) as type_count
        WHERE type_count >= 5
        UNWIND entities as a
        UNWIND entities as b
        WITH entity_type, type_count, a, b
        WHERE id(a) < id(b)
        OPTIONAL MATCH (a)-[r:RELATION]-(b)
        WITH entity_type, type_count, count(r) as intra_relations
        WITH entity_type, type_count, intra_relations,
             type_count * (type_count - 1) / 2 as possible_relations
        RETURN 
            entity_type,
            type_count,
            intra_relations,
            possible_relations,
            round(100.0 * intra_relations / possible_relations, 4) as density_pct
        ORDER BY type_count DESC
        LIMIT 20
        """
        # Even simpler - just count intra-type relations per type
        query = """
        MATCH (a:Entity)-[r:RELATION]-(b:Entity)
        WHERE a.type = b.type AND id(a) < id(b)
        WITH a.type as entity_type, count(*) as intra_relations
        
        MATCH (e:Entity)
        WHERE e.type = entity_type
        WITH entity_type, intra_relations, count(e) as type_count
        WHERE type_count >= 5
        WITH entity_type, type_count, intra_relations,
             type_count * (type_count - 1) / 2 as possible_relations
        RETURN 
            entity_type,
            type_count,
            intra_relations,
            possible_relations,
            round(100.0 * intra_relations / possible_relations, 4) as density_pct
        ORDER BY type_count DESC
        LIMIT 20
        """
        return self.run_query(query, "Intra-type density (type cohesion)")
    
    def get_layer_integration_score(self) -> Dict:
        """
        Measure how well semantic entities integrate with metadata layer.
        
        Checks:
        1. Entity -> Chunk (provenance)
        2. Entity -> Jurisdiction (via SAME_AS)
        3. Entity -> Publication (via MATCHED_TO or through Chunk)
        4. Entity -> Author/Journal (via SAME_AS)
        
        Higher score = better layer integration.
        """
        query = """
        MATCH (e:Entity)
        WITH count(e) as total_entities
        
        // Entities with provenance (EXTRACTED_FROM -> Chunk)
        OPTIONAL MATCH (e1:Entity)-[:EXTRACTED_FROM]->(:Chunk)
        WITH total_entities, count(DISTINCT e1) as with_provenance
        
        // Entities linked to jurisdictions
        OPTIONAL MATCH (e2:Entity)-[:SAME_AS]->(:Jurisdiction)
        WITH total_entities, with_provenance, count(DISTINCT e2) as with_jurisdiction
        
        // Entities linked to publications (MATCHED_TO)
        OPTIONAL MATCH (e3:Entity)-[:MATCHED_TO]->()
        WITH total_entities, with_provenance, with_jurisdiction, count(DISTINCT e3) as with_publication
        
        // Entities with any metadata link
        OPTIONAL MATCH (e4:Entity)-[:SAME_AS]->()
        WITH total_entities, with_provenance, with_jurisdiction, with_publication, 
             count(DISTINCT e4) as with_any_same_as
        
        RETURN 
            total_entities,
            with_provenance,
            with_jurisdiction,
            with_publication,
            with_any_same_as,
            round(100.0 * with_provenance / total_entities, 2) as provenance_coverage_pct,
            round(100.0 * with_jurisdiction / total_entities, 2) as jurisdiction_coverage_pct,
            round(100.0 * with_publication / total_entities, 2) as publication_coverage_pct,
            round(100.0 * with_any_same_as / total_entities, 2) as same_as_coverage_pct
        """
        results = self.run_query(query, "Layer integration score")
        result = results[0] if results else {}
        
        if result:
            # Calculate composite integration score
            provenance = result.get('provenance_coverage_pct', 0)
            jurisdiction = result.get('jurisdiction_coverage_pct', 0)
            publication = result.get('publication_coverage_pct', 0)
            same_as = result.get('same_as_coverage_pct', 0)
            
            # Weighted score: provenance most important
            composite = (provenance * 0.5 + jurisdiction * 0.2 + publication * 0.15 + same_as * 0.15)
            result['composite_integration_score'] = round(composite, 2)
            
            if composite > 80:
                result['interpretation'] = 'Excellent layer integration'
            elif composite > 60:
                result['interpretation'] = 'Good layer integration'
            elif composite > 40:
                result['interpretation'] = 'Moderate integration - some semantic islands'
            else:
                result['interpretation'] = 'Poor integration - layers largely disconnected'
        
        return result
    
    def get_hub_centrality_by_type(self) -> List[Dict]:
        """
        Analyze which entity types produce the most hubs (high-degree nodes).
        Shows which types are central to the knowledge graph.
        """
        query = """
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[r:RELATION]-()
        WITH e, count(r) as degree
        WHERE degree > 0
        WITH e.type as entity_type, degree,
             CASE 
                 WHEN degree >= 100 THEN 'mega_hub'
                 WHEN degree >= 50 THEN 'major_hub'
                 WHEN degree >= 20 THEN 'minor_hub'
                 ELSE 'regular'
             END as hub_category
        WITH entity_type, hub_category, count(*) as count
        WITH entity_type, 
             collect({category: hub_category, count: count}) as hub_dist,
             sum(count) as total_in_type
        UNWIND hub_dist as h
        WITH entity_type, total_in_type,
             sum(CASE WHEN h.category = 'mega_hub' THEN h.count ELSE 0 END) as mega_hubs,
             sum(CASE WHEN h.category = 'major_hub' THEN h.count ELSE 0 END) as major_hubs,
             sum(CASE WHEN h.category = 'minor_hub' THEN h.count ELSE 0 END) as minor_hubs
        WHERE total_in_type >= 10
        RETURN 
            entity_type,
            total_in_type,
            mega_hubs,
            major_hubs,
            minor_hubs,
            mega_hubs + major_hubs + minor_hubs as total_hubs,
            round(100.0 * (mega_hubs + major_hubs + minor_hubs) / total_in_type, 1) as hub_rate_pct
        ORDER BY total_hubs DESC
        LIMIT 15
        """
        return self.run_query(query, "Hub centrality by entity type")
    
    def analyze_author_collaboration_network(self) -> Dict[str, Any]:
        """Analyze author collaboration network structure."""
        # Get author collaboration stats
        collab_query = """
        MATCH (a1:Author)<-[:AUTHORED_BY]-(p:Publication)-[:AUTHORED_BY]->(a2:Author)
        WHERE id(a1) < id(a2)
        WITH a1, a2, count(p) as papers_together
        WITH count(*) as total_collaborations, 
             collect({author1: a1.name, author2: a2.name, papers: papers_together}) as collabs
        RETURN total_collaborations, collabs
        """
        collab_results = self.run_query(collab_query, "Author collaborations")
        
        # Get degree distribution
        degree_query = """
        MATCH (a:Author)<-[:AUTHORED_BY]-(p:Publication)-[:AUTHORED_BY]->(coauthor:Author)
        WHERE a <> coauthor
        WITH a, count(DISTINCT coauthor) as num_collaborators
        RETURN 
            avg(num_collaborators) as avg_collaborators,
            max(num_collaborators) as max_collaborators,
            min(num_collaborators) as min_collaborators,
            percentileCont(num_collaborators, 0.5) as median_collaborators,
            count(a) as authors_with_collaborators
        """
        degree_results = self.run_query(degree_query, "Author collaboration degrees")
        
        # Degree distribution histogram
        hist_query = """
        MATCH (a:Author)<-[:AUTHORED_BY]-(p:Publication)-[:AUTHORED_BY]->(coauthor:Author)
        WHERE a <> coauthor
        WITH a, count(DISTINCT coauthor) as num_collaborators
        WITH 
          CASE 
            WHEN num_collaborators = 1 THEN '1'
            WHEN num_collaborators <= 3 THEN '2-3'
            WHEN num_collaborators <= 5 THEN '4-5'
            WHEN num_collaborators <= 10 THEN '6-10'
            ELSE '10+'
          END as bucket,
          count(*) as author_count
        RETURN bucket, author_count
        ORDER BY 
          CASE bucket
            WHEN '1' THEN 1
            WHEN '2-3' THEN 2
            WHEN '4-5' THEN 3
            WHEN '6-10' THEN 4
            ELSE 5
          END
        """
        hist_results = self.run_query(hist_query, "Author collaboration degree distribution")
        
        # Total authors for E-R comparison
        total_authors_query = """
        MATCH (a:Author)
        RETURN count(a) as total_authors
        """
        author_count_results = self.run_query(total_authors_query, "Total author count")
        
        # Compile results
        result = {
            'degree_stats': degree_results[0] if degree_results else {},
            'degree_distribution': hist_results,
            'total_authors': author_count_results[0]['total_authors'] if author_count_results else 0
        }
        
        # Add E-R comparison
        if collab_results and author_count_results:
            n = author_count_results[0]['total_authors']
            e = collab_results[0]['total_collaborations']
            if n > 1:
                # Expected avg degree in E-R random graph with same N, E
                er_avg_degree = (2 * e) / n
                result['erdos_renyi_comparison'] = {
                    'actual_avg_degree': result['degree_stats'].get('avg_collaborators', 0),
                    'er_expected_avg_degree': round(er_avg_degree, 2),
                    'interpretation': 'Similar to random' if abs(result['degree_stats'].get('avg_collaborators', 0) - er_avg_degree) < 0.5 else 'Structured (non-random)'
                }
        
        return result
    
    def analyze_citation_network(self) -> Dict[str, Any]:
        """Analyze citation network structure (L2 publications)."""
        # In-degree distribution (most cited L2 papers)
        in_degree_query = """
        MATCH (l2:L2Publication)<-[:MATCHED_TO]-(e:Entity)
        WITH l2, count(e) as times_cited
        RETURN 
            avg(times_cited) as avg_citations,
            max(times_cited) as max_citations,
            percentileCont(times_cited, 0.5) as median_citations,
            count(l2) as cited_publications
        """
        in_degree_results = self.run_query(in_degree_query, "Citation in-degrees (L2 papers)")
        
        # In-degree distribution histogram
        in_hist_query = """
        MATCH (l2:L2Publication)<-[:MATCHED_TO]-(e:Entity)
        WITH l2, count(e) as times_cited
        WITH 
          CASE 
            WHEN times_cited = 1 THEN '1'
            WHEN times_cited <= 3 THEN '2-3'
            WHEN times_cited <= 5 THEN '4-5'
            WHEN times_cited <= 10 THEN '6-10'
            ELSE '10+'
          END as bucket,
          count(*) as pub_count
        RETURN bucket, pub_count
        ORDER BY 
          CASE bucket
            WHEN '1' THEN 1
            WHEN '2-3' THEN 2
            WHEN '4-5' THEN 3
            WHEN '6-10' THEN 4
            ELSE 5
          END
        """
        in_hist_results = self.run_query(in_hist_query, "Citation in-degree distribution")
        
        # Top cited L2 papers
        top_cited_query = """
        MATCH (l2:L2Publication)<-[:MATCHED_TO]-(e:Entity)
        WITH l2, count(e) as times_cited
        ORDER BY times_cited DESC
        LIMIT 10
        RETURN l2.title as title, l2.author as author, times_cited
        """
        top_cited_results = self.run_query(top_cited_query, "Most cited L2 publications")
        
        # Total L2 pubs for E-R comparison
        total_l2_query = """
        MATCH (l2:L2Publication)
        RETURN count(l2) as total_l2_pubs
        """
        l2_count_results = self.run_query(total_l2_query, "Total L2 publications")
        
        # Total citations
        total_citations_query = """
        MATCH (:L2Publication)<-[:MATCHED_TO]-(:Entity)
        RETURN count(*) as total_citations
        """
        citation_count_results = self.run_query(total_citations_query, "Total citation links")
        
        result = {
            'in_degree_stats': in_degree_results[0] if in_degree_results else {},
            'in_degree_distribution': in_hist_results,
            'top_cited': top_cited_results,
            'total_l2_publications': l2_count_results[0]['total_l2_pubs'] if l2_count_results else 0,
            'total_citations': citation_count_results[0]['total_citations'] if citation_count_results else 0
        }
        
        # E-R comparison
        if l2_count_results and citation_count_results:
            n = l2_count_results[0]['total_l2_pubs']
            e = citation_count_results[0]['total_citations']
            if n > 0:
                # Expected avg in-degree in E-R directed graph
                er_avg_in_degree = e / n
                result['erdos_renyi_comparison'] = {
                    'actual_avg_in_degree': result['in_degree_stats'].get('avg_citations', 0),
                    'er_expected_avg_in_degree': round(er_avg_in_degree, 2),
                    'interpretation': 'Similar to random' if abs(result['in_degree_stats'].get('avg_citations', 0) - er_avg_in_degree) < 0.5 else 'Preferential attachment (non-random)'
                }
        
        return result
    
    def analyze_academic_regulatory_bridges(self) -> Dict[str, Any]:
        """Analyze entities that bridge academic and regulatory domains."""
        # Entities in both domains
        bridge_query = """
        MATCH (e:Entity)-[:EXTRACTED_FROM]->(c1:Chunk)<-[:CONTAINS]-(j:Jurisdiction)
        MATCH (e)-[:EXTRACTED_FROM]->(c2:Chunk)<-[:CONTAINS]-(p:Publication)
        WITH e, 
             count(DISTINCT j) as num_jurisdictions,
             count(DISTINCT p) as num_papers
        RETURN 
            count(e) as bridging_entities,
            avg(num_jurisdictions) as avg_jurisdictions_per_bridge,
            avg(num_papers) as avg_papers_per_bridge,
            max(num_jurisdictions) as max_jurisdictions,
            max(num_papers) as max_papers
        """
        bridge_results = self.run_query(bridge_query, "Academic-regulatory bridge entities")
        
        # Top bridging entities
        top_bridges_query = """
        MATCH (e:Entity)-[:EXTRACTED_FROM]->(c1:Chunk)<-[:CONTAINS]-(j:Jurisdiction)
        MATCH (e)-[:EXTRACTED_FROM]->(c2:Chunk)<-[:CONTAINS]-(p:Publication)
        WITH e, 
             count(DISTINCT j) as num_jurisdictions,
             count(DISTINCT p) as num_papers
        WITH e, num_jurisdictions, num_papers,
             num_jurisdictions + num_papers as bridge_score
        ORDER BY bridge_score DESC
        LIMIT 20
        RETURN e.name as entity, e.type as type, num_jurisdictions, num_papers, bridge_score
        """
        top_bridges_results = self.run_query(top_bridges_query, "Top bridging entities")
        
        # Isolated entities (only in one domain)
        isolation_query = """
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[:EXTRACTED_FROM]->(:Chunk)<-[:CONTAINS]-(j:Jurisdiction)
        OPTIONAL MATCH (e)-[:EXTRACTED_FROM]->(:Chunk)<-[:CONTAINS]-(p:Publication)
        WITH e, 
             count(DISTINCT j) as in_regulations,
             count(DISTINCT p) as in_papers
        WITH 
          CASE 
            WHEN in_regulations > 0 AND in_papers > 0 THEN 'bridging'
            WHEN in_regulations > 0 THEN 'regulation_only'
            WHEN in_papers > 0 THEN 'academic_only'
            ELSE 'isolated'
          END as category,
          count(e) as entity_count
        RETURN category, entity_count
        """
        isolation_results = self.run_query(isolation_query, "Entity domain distribution")
        
        result = {
            'bridge_stats': bridge_results[0] if bridge_results else {},
            'top_bridges': top_bridges_results,
            'domain_distribution': isolation_results
        }
        
        # Calculate cohesion metrics
        if isolation_results:
            total = sum(r['entity_count'] for r in isolation_results)
            bridging = next((r['entity_count'] for r in isolation_results if r['category'] == 'bridging'), 0)
            result['cohesion'] = {
                'total_entities': total,
                'bridging_entities': bridging,
                'bridging_percentage': round(100 * bridging / total, 2) if total > 0 else 0,
                'interpretation': 'Highly integrated' if (bridging / total if total > 0 else 0) > 0.3 else 'Domain-siloed'
            }
        
        return result
    
    def get_cross_jurisdictional_entities(self, min_jurisdictions: int = 2) -> List[Dict]:
        """Get entities mentioned in multiple jurisdictions."""
        query = f"""
        MATCH (e:Entity)-[:EXTRACTED_FROM]->(:Chunk)<-[:CONTAINS]-(j:Jurisdiction)
        WITH e.name as entity, e.type as type, collect(DISTINCT j.code) as jurisdictions
        WHERE size(jurisdictions) >= {min_jurisdictions}
        RETURN entity, type, jurisdictions, size(jurisdictions) as coverage
        ORDER BY coverage DESC
        LIMIT 50
        """
        return self.run_query(query, f"Entities in {min_jurisdictions}+ jurisdictions")
    
    def get_academic_regulation_bridge(self) -> List[Dict]:
        """Get entities that bridge academic papers and regulations."""
        query = """
        MATCH (e:Entity)-[:EXTRACTED_FROM]->(c1:Chunk)<-[:CONTAINS]-(j:Jurisdiction)
        MATCH (e)-[:EXTRACTED_FROM]->(c2:Chunk)<-[:CONTAINS]-(p:Publication)
        WITH e.name as entity, e.type as type, 
             collect(DISTINCT j.code) as jurisdictions,
             collect(DISTINCT p.title)[..3] as papers
        RETURN entity, type, jurisdictions, papers, 
               size(jurisdictions) as jur_count,
               size(papers) as paper_count
        ORDER BY jur_count DESC, paper_count DESC
        LIMIT 30
        """
        return self.run_query(query, "Entities bridging academic + regulatory sources")
    
    def get_citation_enrichment_stats(self) -> Dict:
        """Get statistics on citation enrichment from Phase 2A."""
        query = """
        MATCH (e:Entity)-[:MATCHED_TO]->(l2:L2Publication)
        WITH count(DISTINCT e) as matched_entities, count(DISTINCT l2) as cited_pubs
        MATCH (:Publication)-[:CITES]->(:L2Publication)
        WITH matched_entities, cited_pubs, count(*) as citation_links
        RETURN matched_entities, cited_pubs, citation_links
        """
        results = self.run_query(query, "Citation enrichment statistics")
        return results[0] if results else {}
    
    def get_orphan_stats(self) -> Dict:
        """Detect orphan nodes (nodes with no relationships)."""
        query = """
        MATCH (n)
        WHERE NOT (n)-[]-()
        RETURN labels(n)[0] as node_type, count(n) as orphan_count
        ORDER BY orphan_count DESC
        """
        results = self.run_query(query, "Orphan node detection")
        return {r['node_type']: r['orphan_count'] for r in results}
    
    def get_entity_type_distribution(self) -> List[Dict]:
        """Get distribution of entity types."""
        query = """
        MATCH (e:Entity)
        WITH e.type as entity_type, count(*) as count
        ORDER BY count DESC
        RETURN entity_type, count
        """
        return self.run_query(query, "Entity type distribution")
    
    def get_provenance_coverage(self) -> Dict:
        """Calculate what percentage of entities/chunks have proper provenance."""
        query = """
        MATCH (e:Entity)
        WITH count(e) as total_entities
        MATCH (e:Entity)-[:EXTRACTED_FROM]->(:Chunk)
        WITH total_entities, count(DISTINCT e) as entities_with_chunks
        MATCH (c:Chunk)
        WITH total_entities, entities_with_chunks, count(c) as total_chunks
        MATCH (c:Chunk)<-[:CONTAINS]-()
        RETURN 
            total_entities,
            entities_with_chunks,
            total_chunks,
            count(DISTINCT c) as chunks_with_source,
            (entities_with_chunks * 100.0 / total_entities) as entity_coverage_pct,
            (count(DISTINCT c) * 100.0 / total_chunks) as chunk_coverage_pct
        """
        results = self.run_query(query, "Provenance coverage")
        return results[0] if results else {}
    
    def run_all_analyses(self) -> Dict[str, Any]:
        """
        Run all analytical queries and compile results.
        
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("\n" + "="*60)
        logger.info("GRAPH ANALYTICS REPORT")
        logger.info("="*60 + "\n")
        
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'neo4j_uri': self.driver._pool.address[0]
            },
            'node_counts': self.get_node_counts(),
            'relationship_counts': self.get_relationship_counts(),
            'coverage_stats': self.get_coverage_stats(),
            'top_connected_entities': self.get_top_connected_entities(20),
            'predicate_distribution': self.get_predicate_distribution(30),
            'cross_jurisdictional_entities': self.get_cross_jurisdictional_entities(2),
            'academic_regulation_bridge': self.get_academic_regulation_bridge(),
            'citation_enrichment': self.get_citation_enrichment_stats(),
            'entity_type_distribution': self.get_entity_type_distribution(),
            'provenance_coverage': self.get_provenance_coverage(),
            'orphan_stats': self.get_orphan_stats(),
            # KG-specific metrics
            'predicate_diversity': self.get_predicate_diversity(),
            'relation_density': self.get_relation_density(),
            'degree_distribution': self.get_degree_distribution_buckets(),
            'property_completeness': self.get_property_completeness(),
            'cross_type_connectivity': self.get_cross_type_connectivity(),
            # Graph cohesion metrics (NEW)
            'connected_components': self.get_connected_components(),
            'type_homophily': self.get_type_homophily(),
            'type_homophily_breakdown': self.get_type_homophily_breakdown(),
            'layer_integration': self.get_layer_integration_score(),
            'hub_centrality_by_type': self.get_hub_centrality_by_type(),
            # Network science metrics
            'author_collaboration_network': self.analyze_author_collaboration_network(),
            'citation_network': self.analyze_citation_network(),
            'bridge_network_analysis': self.analyze_academic_regulatory_bridges()
        }
        
        # Log summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print formatted summary to console."""
        logger.info("\n=== COVERAGE SUMMARY ===")
        cov = results['coverage_stats']
        logger.info(f"Jurisdictions: {cov.get('jurisdictions', 0)}")
        logger.info(f"Publications: {cov.get('publications', 0)}")
        logger.info(f"Entities: {cov.get('entities', 0):,}")
        logger.info(f"Chunks: {cov.get('chunks', 0):,}")
        logger.info(f"Semantic Relations: {cov.get('semantic_relations', 0):,}")
        
        logger.info("\n=== KG-SPECIFIC METRICS ===")
        
        # Relation Density
        density = results['relation_density']
        logger.info(f"\n Relation Density: {density.get('relations_per_entity', 0):.2f} relations/entity")
        logger.info(f"   Interpretation: {density.get('interpretation', 'N/A')}")
        logger.info(f"   Benchmark: Domain KGs typically 2-5 relations/entity")
        
        # Predicate Diversity
        pred_div = results['predicate_diversity']
        logger.info(f"\n Predicate Diversity (Semantic Richness):")
        logger.info(f"   Avg predicates per entity: {pred_div.get('avg_predicates_per_entity', 0):.2f}")
        logger.info(f"   Max predicates (richest entity): {pred_div.get('max_predicates', 0)}")
        logger.info(f"   Median: {pred_div.get('median_predicates', 0):.1f}")
        
        # Degree Distribution
        logger.info(f"\n Degree Distribution (Power-Law Analysis):")
        deg_dist = results['degree_distribution']
        for bucket in deg_dist:
            logger.info(f"   {bucket['bucket']:15} → {bucket['entity_count']:,} entities")
        
        # Property Completeness
        completeness = results['property_completeness']
        logger.info(f"\n Metadata Completeness:")
        logger.info(f"   Publications: {completeness.get('publication_completeness_pct', 0):.1f}%")
        logger.info(f"   Authors: {completeness.get('author_completeness_pct', 0):.1f}%")
        logger.info(f"   Journals: {completeness.get('journal_completeness_pct', 0):.1f}%")
        
        # Cross-Type Connectivity
        logger.info(f"\n Cross-Type Connectivity (Top 10 Integrations):")
        cross_type = results['cross_type_connectivity'][:10]
        for conn in cross_type:
            logger.info(f"   {conn['from_type']:15} →[{conn['rel_type']:15}]→ {conn['to_type']:15} ({conn['connections']:,})")
        
        logger.info("\n" + "="*60)
        logger.info("GRAPH COHESION METRICS")
        logger.info("="*60)
        
        # Connected Components
        cc = results.get('connected_components', {})
        logger.info(f"\n Connected Components:")
        logger.info(f"   Total entities: {cc.get('total_entities', 0):,}")
        logger.info(f"   Connected (in graph): {cc.get('connected_entities', 0):,} ({cc.get('connectivity_rate_pct', 0):.1f}%)")
        logger.info(f"   Isolated (no relations): {cc.get('isolated_entities', 0):,} ({cc.get('isolation_rate_pct', 0):.1f}%)")
        logger.info(f"   → {cc.get('interpretation', 'N/A')}")
        
        # Type Homophily
        homophily = results.get('type_homophily', {})
        logger.info(f"\n Type Homophily (Do same-type entities cluster?):")
        logger.info(f"   Total relations: {homophily.get('total_relations', 0):,}")
        logger.info(f"   Intra-type (same type): {homophily.get('intra_type_relations', 0):,} ({homophily.get('homophily_pct', 0):.1f}%)")
        logger.info(f"   Inter-type (cross type): {homophily.get('inter_type_relations', 0):,}")
        logger.info(f"   → {homophily.get('interpretation', 'N/A')}")
        
        # Type Homophily Breakdown (top 5)
        homophily_breakdown = results.get('type_homophily_breakdown', [])
        if homophily_breakdown:
            logger.info(f"\n   Top types by homophily:")
            for h in homophily_breakdown[:5]:
                logger.info(f"      {h['source_type']:20} → {h['homophily_pct']:.1f}% intra-type ({h['total']:,} total)")
        
        # Layer Integration
        integration = results.get('layer_integration', {})
        logger.info(f"\n Layer Integration Score:")
        logger.info(f"   Provenance (Entity→Chunk): {integration.get('provenance_coverage_pct', 0):.1f}%")
        logger.info(f"   Jurisdiction (Entity→Jur): {integration.get('jurisdiction_coverage_pct', 0):.1f}%")
        logger.info(f"   Publication (Entity→Pub): {integration.get('publication_coverage_pct', 0):.1f}%")
        logger.info(f"   Any SAME_AS link: {integration.get('same_as_coverage_pct', 0):.1f}%")
        logger.info(f"   Composite Score: {integration.get('composite_integration_score', 0):.1f}/100")
        logger.info(f"   → {integration.get('interpretation', 'N/A')}")
        
        # Hub Centrality by Type
        hub_types = results.get('hub_centrality_by_type', [])
        if hub_types:
            logger.info(f"\n Hub Centrality by Type (which types produce hubs):")
            for h in hub_types[:10]:
                logger.info(f"   {h['entity_type']:20} → {h['total_hubs']:3} hubs ({h['hub_rate_pct']:.1f}%) [mega:{h['mega_hubs']}, major:{h['major_hubs']}, minor:{h['minor_hubs']}]")
        
        logger.info("\n=== TOP CONNECTED ENTITIES ===")
        for i, entity in enumerate(results['top_connected_entities'][:10], 1):
            logger.info(f"{i:2}. {entity['entity'][:50]:50} ({entity['type']}) → {entity['degree']} connections")
        
        logger.info("\n=== CROSS-JURISDICTIONAL COVERAGE ===")
        cross_jur = results['cross_jurisdictional_entities']
        if cross_jur:
            logger.info(f"Entities in 2+ jurisdictions: {len([e for e in cross_jur if e['coverage'] >= 2])}")
            logger.info(f"Entities in 3+ jurisdictions: {len([e for e in cross_jur if e['coverage'] >= 3])}")
            logger.info(f"Max coverage: {max(e['coverage'] for e in cross_jur)} jurisdictions")
            logger.info(f"\nTop cross-jurisdictional concepts:")
            for i, entity in enumerate(cross_jur[:5], 1):
                logger.info(f"{i}. {entity['entity'][:40]:40} → {entity['coverage']} jurisdictions")
        
        logger.info("\n=== ACADEMIC-REGULATORY BRIDGES ===")
        bridges = results['academic_regulation_bridge'][:5]
        if bridges:
            logger.info(f"Entities connecting regulations + papers: {len(results['academic_regulation_bridge'])}")
            logger.info(f"\nTop bridging concepts:")
            for i, bridge in enumerate(bridges, 1):
                logger.info(f"{i}. {bridge['entity'][:40]:40} → {bridge['jur_count']} jurs, {bridge['paper_count']} papers")
        
        logger.info("\n=== PROVENANCE COVERAGE ===")
        prov = results['provenance_coverage']
        logger.info(f"Entities with chunks: {prov.get('entity_coverage_pct', 0):.1f}%")
        logger.info(f"Chunks with source: {prov.get('chunk_coverage_pct', 0):.1f}%")
        
        logger.info("\n=== ENTITY TYPE DISTRIBUTION (Top 10) ===")
        entity_types = results['entity_type_distribution'][:10]
        for et in entity_types:
            logger.info(f"{et['entity_type']:20} → {et['count']:,}")
        
        logger.info("\n=== ORPHAN NODES ===")
        orphans = results['orphan_stats']
        if orphans:
            for node_type, count in orphans.items():
                logger.info(f"{node_type}: {count} orphans")
        else:
            logger.info("No orphan nodes detected ")
        
        logger.info("\n" + "="*60)
        logger.info("NETWORK SCIENCE ANALYSIS")
        logger.info("="*60)
        
        # Author Collaboration Network
        logger.info("\n===  AUTHOR COLLABORATION NETWORK ===")
        author_net = results['author_collaboration_network']
        deg_stats = author_net.get('degree_stats', {})
        logger.info(f"Total authors: {author_net.get('total_authors', 0)}")
        logger.info(f"Authors with collaborators: {deg_stats.get('authors_with_collaborators', 0)}")
        logger.info(f"Avg collaborators per author: {deg_stats.get('avg_collaborators', 0):.2f}")
        logger.info(f"Max collaborators: {deg_stats.get('max_collaborators', 0)}")
        logger.info(f"Median collaborators: {deg_stats.get('median_collaborators', 0):.1f}")
        
        if 'erdos_renyi_comparison' in author_net:
            er = author_net['erdos_renyi_comparison']
            logger.info(f"\n Erdős-Rényi Comparison:")
            logger.info(f"   Actual avg degree: {er['actual_avg_degree']:.2f}")
            logger.info(f"   E-R expected (random): {er['er_expected_avg_degree']:.2f}")
            logger.info(f"   → {er['interpretation']}")
        
        logger.info(f"\n Collaboration Degree Distribution:")
        for bucket in author_net.get('degree_distribution', []):
            logger.info(f"   {bucket['bucket']:6} collaborators → {bucket['author_count']:3} authors")
        
        # Citation Network
        logger.info("\n===  CITATION NETWORK (L2 Publications) ===")
        cite_net = results['citation_network']
        in_stats = cite_net.get('in_degree_stats', {})
        logger.info(f"Total L2 publications: {cite_net.get('total_l2_publications', 0)}")
        logger.info(f"Total citation links: {cite_net.get('total_citations', 0)}")
        logger.info(f"Cited publications: {in_stats.get('cited_publications', 0)}")
        logger.info(f"Avg citations per L2 paper: {in_stats.get('avg_citations', 0):.2f}")
        logger.info(f"Max citations: {in_stats.get('max_citations', 0)}")
        logger.info(f"Median citations: {in_stats.get('median_citations', 0):.1f}")
        
        if 'erdos_renyi_comparison' in cite_net:
            er = cite_net['erdos_renyi_comparison']
            logger.info(f"\n Erdős-Rényi Comparison:")
            logger.info(f"   Actual avg in-degree: {er['actual_avg_in_degree']:.2f}")
            logger.info(f"   E-R expected (random): {er['er_expected_avg_in_degree']:.2f}")
            logger.info(f"   → {er['interpretation']}")
        
        logger.info(f"\n Citation In-Degree Distribution:")
        for bucket in cite_net.get('in_degree_distribution', []):
            logger.info(f"   {bucket['bucket']:6} citations → {bucket['pub_count']:3} publications")
        
        logger.info(f"\n Top 10 Most Cited L2 Publications:")
        for i, pub in enumerate(cite_net.get('top_cited', [])[:10], 1):
            title = pub['title'][:50] if pub['title'] else 'N/A'
            author = pub['author'][:30] if pub['author'] else 'N/A'
            logger.info(f"{i:2}. {title:50} by {author:30} ({pub['times_cited']} cites)")
        
        # Academic-Regulatory Bridges
        logger.info("\n===  ACADEMIC-REGULATORY BRIDGE NETWORK ===")
        bridge_net = results['bridge_network_analysis']
        bridge_stats = bridge_net.get('bridge_stats', {})
        logger.info(f"Bridging entities: {bridge_stats.get('bridging_entities', 0):,}")
        logger.info(f"Avg jurisdictions per bridge: {bridge_stats.get('avg_jurisdictions_per_bridge', 0):.2f}")
        logger.info(f"Avg papers per bridge: {bridge_stats.get('avg_papers_per_bridge', 0):.2f}")
        logger.info(f"Max jurisdictions: {bridge_stats.get('max_jurisdictions', 0)}")
        logger.info(f"Max papers: {bridge_stats.get('max_papers', 0)}")
        
        if 'cohesion' in bridge_net:
            cohesion = bridge_net['cohesion']
            logger.info(f"\n Network Cohesion:")
            logger.info(f"   Bridging entities: {cohesion['bridging_entities']:,} ({cohesion['bridging_percentage']:.1f}%)")
            logger.info(f"   → {cohesion['interpretation']}")
        
        logger.info(f"\n Domain Distribution:")
        for dist in bridge_net.get('domain_distribution', []):
            logger.info(f"   {dist['category']:20} → {dist['entity_count']:,} entities")
        
        logger.info(f"\n Top 10 Bridging Entities:")
        for i, bridge in enumerate(bridge_net.get('top_bridges', [])[:10], 1):
            logger.info(f"{i:2}. {bridge['entity'][:40]:40} ({bridge['type']:15}) → {bridge['num_jurisdictions']} jurs, {bridge['num_papers']} papers")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Analyze GraphRAG knowledge graph')
    parser.add_argument('--uri', type=str, help='Neo4j URI (default: env NEO4J_URI)')
    parser.add_argument('--user', type=str, default='neo4j', help='Neo4j user (default: neo4j)')
    parser.add_argument('--password', type=str, help='Neo4j password (default: env NEO4J_PASSWORD)')
    parser.add_argument('--output', type=str, help='Output JSON file path (optional)')
    
    args = parser.parse_args()
    
    # Get credentials
    uri = args.uri or os.getenv('NEO4J_URI')
    user = args.user or os.getenv('NEO4J_USER', 'neo4j')
    password = args.password or os.getenv('NEO4J_PASSWORD')
    
    if not uri or not password:
        logger.error("--uri and --password required (or set NEO4J_URI and NEO4J_PASSWORD env vars)")
        sys.exit(1)
    
    # Run analysis
    analyzer = GraphAnalyzer(uri, user, password)
    
    try:
        results = analyzer.run_all_analyses()
        
        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"\n Results saved to {output_path}")
        
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*60)
        
    finally:
        analyzer.close()


if __name__ == '__main__':
    main()