# -*- coding: utf-8 -*-
"""
Knowledge graph analytics and network science metrics for GraphRAG.

Comprehensive analysis toolkit for evaluating GraphRAG knowledge graph quality,
structure, and integration. Computes network science metrics (degree distribution,
connectivity, homophily), entity-type distributions, predicate concentration,
document coverage, chunk extraction statistics, and co-occurrence patterns. Provides
both high-level summaries for quick health checks and detailed breakdowns with
distributions and top-N lists for in-depth analysis.

The analyzer runs Cypher queries against Neo4j to extract graph statistics, calculates
network properties (connected components, clustering, assortativity), analyzes entity-
document linkages, and evaluates extraction quality. Supports verbose mode for detailed
output and exports JSON reports with timestamps. Key metrics include predicate
concentration (semantic diversity), entity coverage per document, and chunk-entity
provenance ratios.

Modes:
    --verbose    Show detailed metrics, distributions, and top-N lists
    (default)    Compact summary report with key findings

Examples:
    # Run basic analytics (compact summary)
    python -m src.analysis.graph_analytics

    # Detailed analytics with distributions
    python -m src.analysis.graph_analytics --verbose

    # Python API usage
    from src.analysis.graph_analytics import GraphAnalyzer

    analyzer = GraphAnalyzer(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        verbose=True
    )

    # Get node and relationship counts
    nodes = analyzer.get_node_counts()
    rels = analyzer.get_relationship_counts()

    # Coverage statistics
    coverage = analyzer.get_coverage_stats()
    print(f"Entities: {coverage['entities']}, Chunks: {coverage['chunks']}")

    # Network metrics
    network = analyzer.get_network_metrics()
    print(f"Connected components: {network['components']}")
    print(f"Clustering coefficient: {network['clustering']:.3f}")

    # Entity centrality
    top_entities = analyzer.get_top_connected_entities(limit=20)
    for entity in top_entities:
        print(f"{entity['entity']}: degree={entity['degree']}")

    # Predicate concentration (semantic diversity)
    concentration = analyzer.get_predicate_concentration()
    for pred in concentration[:10]:
        print(f"{pred['predicate']}: {pred['pct']}% (cumulative: {pred['cumulative_pct']}%)")

    # Clean up
    analyzer.close()

References:
    Neo4j: Graph database with Cypher query language for analytics
    NetworkX: Network science metrics (clustering, assortativity, components)
    RAGAS: Evaluation framework for retrieval-augmented generation
    config.extraction_config: Entity types and extraction parameters
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Third-party
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class GraphAnalyzer:
    """Analyzes GraphRAG knowledge graph structure and statistics."""
    
    def __init__(self, uri: str, user: str, password: str, verbose: bool = False):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            verbose: If True, show detailed query logs
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.verbose = verbose
        if verbose:
            print(f"Connected to Neo4j at {uri}")
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
        if self.verbose:
            print("Connection closed")
    
    def run_query(self, query: str, description: str) -> List[Dict]:
        """
        Execute Cypher query and return results.
        
        Args:
            query: Cypher query string
            description: Human-readable description for logging
            
        Returns:
            List of result dictionaries
        """
        if self.verbose:
            print(f"Running: {description}")
        with self.driver.session() as session:
            result = session.run(query)
            data = [dict(record) for record in result]
            if self.verbose:
                print(f"  -> Returned {len(data)} results")
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
        MATCH (e:Entity)-[r:RELATION]-()
        WITH e, count(r) as degree
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
    
    def get_predicate_concentration(self) -> Dict:
        """
        Calculate predicate concentration (top-N coverage).
        
        Returns cumulative coverage: e.g., "top 4 predicates cover 92.3%"
        Important for understanding semantic diversity vs. concentration.
        """
        query = """
        MATCH ()-[r:RELATION]->()
        WITH r.predicate as predicate, count(*) as cnt
        ORDER BY cnt DESC
        WITH collect({predicate: predicate, count: cnt}) as all_preds, sum(cnt) as total
        UNWIND range(0, size(all_preds)-1) as idx
        WITH all_preds[idx] as p, total, idx,
             reduce(s = 0, i IN range(0, idx) | s + all_preds[i].count) as cumulative
        RETURN 
            p.predicate as predicate, 
            p.count as count,
            round(100.0 * p.count / total, 2) as pct,
            round(100.0 * cumulative / total, 2) as cumulative_pct,
            total
        LIMIT 10
        """
        results = self.run_query(query, "Predicate concentration (cumulative coverage)")
        
        if not results:
            return {}
        
        total = results[0]['total'] if results else 0
        
        # Find top-4 cumulative coverage
        top_4_pct = results[3]['cumulative_pct'] if len(results) >= 4 else None
        top_10_pct = results[9]['cumulative_pct'] if len(results) >= 10 else None
        
        return {
            'predicates': results,
            'total_relations': total,
            'unique_predicates': len(results),  # approximation from limit
            'top_4_cumulative_pct': top_4_pct,
            'top_10_cumulative_pct': top_10_pct
        }
    
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
    
    def get_average_degree(self) -> Dict:
        """
        Calculate average node degree across ALL relationship types.
        
        This counts all edges (RELATION, EXTRACTED_FROM, etc.) per entity.
        Thesis metric: average degree of 14.9
        """
        query = """
        MATCH (e:Entity)
        WITH e, size((e)-[]-()) as degree
        RETURN 
            round(avg(degree), 2) as avg_degree,
            max(degree) as max_degree,
            min(degree) as min_degree,
            round(percentileCont(degree, 0.5), 1) as median_degree,
            count(e) as total_entities
        """
        results = self.run_query(query, "Average degree (all edge types)")
        return results[0] if results else {}
    
    def get_semantic_degree(self) -> Dict:
        """
        Calculate average degree using only RELATION edges.
        
        This is the "semantic connectivity" - how many other entities
        each entity is connected to through semantic relations.
        """
        query = """
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[r:RELATION]-()
        WITH e, count(r) as degree
        RETURN 
            round(avg(degree), 2) as avg_semantic_degree,
            max(degree) as max_semantic_degree,
            round(percentileCont(degree, 0.5), 1) as median_semantic_degree,
            count(CASE WHEN degree = 0 THEN 1 END) as zero_degree_entities,
            count(e) as total_entities
        """
        results = self.run_query(query, "Semantic degree (RELATION edges only)")
        return results[0] if results else {}
    
    def get_local_clustering(self) -> Dict:
        """
        Calculate local clustering coefficient for semantic relations.
        
        Measures how tightly connected neighborhoods are:
        - High (>0.3): Tight communities, entities form cliques
        - Low (<0.1): Hub-spoke structure, sparse local connections
        """
        query = """
        MATCH (a:Entity)-[:RELATION]-(b:Entity)-[:RELATION]-(c:Entity)-[:RELATION]-(a)
        WITH count(*) as triangles
        MATCH (a:Entity)-[:RELATION]-(b:Entity)-[:RELATION]-(c:Entity)
        WHERE a <> c
        WITH triangles, count(*) as triplets
        RETURN 
            triangles,
            triplets,
            CASE WHEN triplets > 0 
                 THEN round(3.0 * triangles / triplets, 4) 
                 ELSE 0 
            END as clustering_coef
        """
        results = self.run_query(query, "Local clustering coefficient")
        result = results[0] if results else {}
        
        if result:
            cc = result.get('clustering_coef', 0)
            if cc > 0.3:
                result['interpretation'] = 'High clustering - tight communities'
            elif cc > 0.1:
                result['interpretation'] = 'Moderate clustering'
            else:
                result['interpretation'] = 'Low clustering - hub-spoke structure'
        
        return result
    
    def get_type_affinity_matrix(self, limit: int = 20) -> List[Dict]:
        """
        Cross-tabulation of source_type × target_type for RELATION edges.
        
        Reveals extraction patterns: which entity types connect to which?
        """
        query = f"""
        MATCH (a:Entity)-[r:RELATION]->(b:Entity)
        RETURN a.type as source_type, b.type as target_type, count(*) as count
        ORDER BY count DESC
        LIMIT {limit}
        """
        return self.run_query(query, "Type affinity matrix")
    
    def get_path_length_sample(self, sample_size: int = 100) -> Dict:
        """
        Sample shortest paths between random entity pairs.
        
        Critical for Steiner tree performance - longer paths = more expensive.
        Uses sampling because full APSP is O(n²).
        """
        query = f"""
        MATCH (a:Entity)-[:RELATION]-()
        WITH a, rand() as r ORDER BY r LIMIT {sample_size}
        WITH collect(a) as sources
        MATCH (b:Entity)-[:RELATION]-()
        WITH sources, b, rand() as r ORDER BY r LIMIT {sample_size}
        WITH sources, collect(b) as targets
        UNWIND sources as src
        UNWIND targets as tgt
        WITH src, tgt WHERE src <> tgt
        WITH src, tgt LIMIT {sample_size}
        MATCH p = shortestPath((src)-[:RELATION*..10]-(tgt))
        WITH length(p) as path_len
        RETURN 
            round(avg(path_len), 2) as avg_path_length,
            max(path_len) as max_path_length,
            min(path_len) as min_path_length,
            count(*) as paths_found
        """
        results = self.run_query(query, f"Path length sample (n={sample_size})")
        result = results[0] if results else {}
        
        if result:
            avg_len = result.get('avg_path_length', 0)
            if avg_len <= 3:
                result['interpretation'] = 'Short paths - efficient Steiner tree'
            elif avg_len <= 5:
                result['interpretation'] = 'Medium paths - reasonable traversal'
            else:
                result['interpretation'] = 'Long paths - expensive graph expansion'
        
        return result
    
    def get_entity_quality_flags(self) -> Dict:
        """
        Flag potentially problematic entities from extraction.
        
        Detects:
        - Very long names (>100 chars) - likely extraction errors
        - Bracket names - JSON/code artifacts
        - Very short names (<3 chars) - likely noise
        """
        query = """
        MATCH (e:Entity)
        RETURN 
            count(e) as total_entities,
            sum(CASE WHEN size(e.name) > 100 THEN 1 ELSE 0 END) as very_long_names,
            sum(CASE WHEN e.name =~ '.*[\\\\[\\\\]{}].*' THEN 1 ELSE 0 END) as bracket_names,
            sum(CASE WHEN size(e.name) < 3 THEN 1 ELSE 0 END) as very_short_names,
            sum(CASE WHEN e.name =~ '^[0-9]+$' THEN 1 ELSE 0 END) as numeric_only_names,
            sum(CASE WHEN e.name IS NULL OR e.name = '' THEN 1 ELSE 0 END) as empty_names
        """
        results = self.run_query(query, "Entity quality flags")
        result = results[0] if results else {}
        
        if result:
            total = result.get('total_entities', 1)
            issues = (result.get('very_long_names', 0) + 
                     result.get('bracket_names', 0) + 
                     result.get('very_short_names', 0) +
                     result.get('numeric_only_names', 0) +
                     result.get('empty_names', 0))
            result['issue_rate_pct'] = round(100.0 * issues / max(1, total), 2)
            
            if result['issue_rate_pct'] < 1:
                result['interpretation'] = 'Clean - minimal quality issues'
            elif result['issue_rate_pct'] < 5:
                result['interpretation'] = 'Good - some quality issues'
            else:
                result['interpretation'] = 'Review needed - significant quality issues'
        
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
            
            if low_degree > high_degree * 3 and self.verbose:
                print("  -> Power-law distribution detected (typical for KGs)")
        
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
        Analyze connectivity in the semantic entity graph.
        
        Reports TWO connectivity metrics:
        1. "Connected to graph" = has ANY edge (data quality)
        2. "Has semantic relations" = has RELATION edge (retrieval effectiveness)
        
        For GraphRAG retrieval, RELATION edges matter because Steiner tree
        only traverses RELATION edges between entities.
        """
        # 1. Total entities
        total_query = """
        MATCH (e:Entity)
        RETURN count(e) as total_entities
        """
        total = self.run_query(total_query, "Total entities")
        total_count = total[0]['total_entities'] if total else 0
        
        # 2. Entities connected via ANY edge (data quality metric)
        any_edge_query = """
        MATCH (e:Entity)
        WHERE (e)-[]-()
        RETURN count(e) as connected_any
        """
        any_edge = self.run_query(any_edge_query, "Entities with any edge")
        connected_any = any_edge[0]['connected_any'] if any_edge else 0
        
        # 3. Entities with RELATION edges (retrieval effectiveness)
        relation_query = """
        MATCH (e:Entity)-[:RELATION]-()
        RETURN count(DISTINCT e) as connected_relation
        """
        relation = self.run_query(relation_query, "Entities with RELATION edges")
        connected_relation = relation[0]['connected_relation'] if relation else 0
        
        # 4. Truly isolated (no edges at all)
        truly_isolated_query = """
        MATCH (e:Entity)
        WHERE NOT (e)-[]-()
        RETURN count(e) as truly_isolated
        """
        truly_isolated = self.run_query(truly_isolated_query, "Truly isolated entities")
        truly_isolated_count = truly_isolated[0]['truly_isolated'] if truly_isolated else 0
        
        # 5. Has provenance but no relations (extraction worked, relation extraction missed)
        provenance_only_query = """
        MATCH (e:Entity)-[:EXTRACTED_FROM]->()
        WHERE NOT (e)-[:RELATION]-()
        RETURN count(e) as provenance_only
        """
        prov_only = self.run_query(provenance_only_query, "Entities with provenance but no relations")
        provenance_only_count = prov_only[0]['provenance_only'] if prov_only else 0
        
        # 6. Get isolated-from-relations by type (for diagnosis)
        isolated_by_type_query = """
        MATCH (e:Entity)
        WHERE NOT (e)-[:RELATION]-()
        RETURN e.type as entity_type, count(e) as count
        ORDER BY count DESC
        LIMIT 10
        """
        isolated_by_type = self.run_query(isolated_by_type_query, "Relation-isolated entities by type")
        
        no_relation_count = total_count - connected_relation
        
        result = {
            'total_entities': total_count,
            # Data quality metric
            'connected_any_edge': connected_any,
            'connected_any_pct': round(100.0 * connected_any / max(1, total_count), 1),
            'truly_isolated': truly_isolated_count,
            # Retrieval effectiveness metric  
            'has_semantic_relations': connected_relation,
            'has_relations_pct': round(100.0 * connected_relation / max(1, total_count), 1),
            'no_semantic_relations': no_relation_count,
            'no_relations_pct': round(100.0 * no_relation_count / max(1, total_count), 1),
            # Diagnosis
            'provenance_only': provenance_only_count,
            'isolated_by_type': isolated_by_type
        }
        
        # Interpretation for retrieval
        pct = result['has_relations_pct']
        if pct > 90:
            result['interpretation'] = 'Excellent - most entities reachable via graph traversal'
        elif pct > 70:
            result['interpretation'] = 'Good - majority reachable, some isolated'
        elif pct > 50:
            result['interpretation'] = 'Moderate - significant portion unreachable via graph'
        else:
            result['interpretation'] = 'Poor - most entities unreachable via graph traversal'
        
        # Note about provenance
        if provenance_only_count > 0:
            prov_pct = round(100.0 * provenance_only_count / max(1, no_relation_count), 1)
            result['provenance_note'] = f'{prov_pct}% of relation-isolated entities have chunk provenance (entity extraction OK, relation extraction missed them)'
        
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
        # Two separate queries to avoid Cypher scope issues
        # First get type counts
        type_query = """
        MATCH (e:Entity)
        WITH e.type as entity_type, count(e) as type_count
        WHERE type_count >= 5
        RETURN entity_type, type_count
        ORDER BY type_count DESC
        LIMIT 20
        """
        type_results = self.run_query(type_query, "Entity type counts")
        
        # Then get intra-type relation counts
        intra_query = """
        MATCH (a:Entity)-[r:RELATION]-(b:Entity)
        WHERE a.type = b.type AND elementId(a) < elementId(b)
        RETURN a.type as entity_type, count(r) as intra_relations
        """
        intra_results = self.run_query(intra_query, "Intra-type relation counts")
        
        # Join in Python
        intra_map = {r['entity_type']: r['intra_relations'] for r in intra_results}
        
        results = []
        for t in type_results:
            entity_type = t['entity_type']
            type_count = t['type_count']
            intra_relations = intra_map.get(entity_type, 0)
            possible = type_count * (type_count - 1) // 2
            density = round(100.0 * intra_relations / possible, 4) if possible > 0 else 0
            results.append({
                'entity_type': entity_type,
                'type_count': type_count,
                'intra_relations': intra_relations,
                'possible_relations': possible,
                'density_pct': density
            })
        
        return results
    
    def get_layer_integration_score(self) -> Dict:
        """
        Measure how well semantic entities integrate with metadata layer.
        
        Checks:
        1. Entity -> Chunk (provenance via EXTRACTED_FROM)
        2. Entity -> Jurisdiction (via SAME_AS)
        3. Entity -> Publication (via MATCHED_TO)
        4. Entity -> Author/Journal (via SAME_AS)
        
        Returns raw counts for clarity.
        """
        query = """
        // Count each cross-layer relation type independently
        CALL { MATCH ()-[r:EXTRACTED_FROM]->() RETURN count(r) as c } WITH c as extracted_from
        CALL { MATCH ()-[r:SAME_AS]->(:Jurisdiction) RETURN count(r) as c } WITH extracted_from, c as same_as_jurisdiction
        CALL { MATCH ()-[r:SAME_AS]->(:Author) RETURN count(r) as c } WITH extracted_from, same_as_jurisdiction, c as same_as_author
        CALL { MATCH ()-[r:SAME_AS]->(:Journal) RETURN count(r) as c } WITH extracted_from, same_as_jurisdiction, same_as_author, c as same_as_journal
        CALL { MATCH ()-[r:MATCHED_TO]->() RETURN count(r) as c } WITH extracted_from, same_as_jurisdiction, same_as_author, same_as_journal, c as matched_to
        CALL { MATCH ()-[r:CITES]->() RETURN count(r) as c } WITH extracted_from, same_as_jurisdiction, same_as_author, same_as_journal, matched_to, c as cites
        CALL { MATCH ()-[r:AUTHORED_BY]->() RETURN count(r) as c } WITH extracted_from, same_as_jurisdiction, same_as_author, same_as_journal, matched_to, cites, c as authored_by
        CALL { MATCH ()-[r:PUBLISHED_IN]->() RETURN count(r) as c } WITH extracted_from, same_as_jurisdiction, same_as_author, same_as_journal, matched_to, cites, authored_by, c as published_in
        CALL { MATCH (:Jurisdiction)-[r:CONTAINS]->(:Chunk) RETURN count(r) as c } WITH extracted_from, same_as_jurisdiction, same_as_author, same_as_journal, matched_to, cites, authored_by, published_in, c as jur_contains
        CALL { MATCH (:Publication)-[r:CONTAINS]->(:Chunk) RETURN count(r) as c } WITH extracted_from, same_as_jurisdiction, same_as_author, same_as_journal, matched_to, cites, authored_by, published_in, jur_contains, c as pub_contains
        
        RETURN 
            extracted_from,
            same_as_jurisdiction,
            same_as_author,
            same_as_journal,
            matched_to,
            cites,
            authored_by,
            published_in,
            jur_contains,
            pub_contains,
            (extracted_from + same_as_jurisdiction + same_as_author + same_as_journal + 
             matched_to + cites + authored_by + published_in + jur_contains + pub_contains) as total_cross_layer
        """
        results = self.run_query(query, "Layer integration counts")
        result = results[0] if results else {}
        
        return result
    
    def get_hub_centrality_by_type(self) -> List[Dict]:
        """
        Analyze which entity types produce the most hubs (high-degree nodes).
        Shows which types are central to the knowledge graph.
        """
        query = """
        MATCH (e:Entity)-[r:RELATION]-()
        WITH e, count(r) as degree
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
        WHERE elementId(a1) < elementId(a2)
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
                actual_avg = result['degree_stats'].get('avg_collaborators') or 0
                result['erdos_renyi_comparison'] = {
                    'actual_avg_degree': actual_avg,
                    'er_expected_avg_degree': round(er_avg_degree, 2),
                    'interpretation': 'Similar to random' if abs(actual_avg - er_avg_degree) < 0.5 else 'Structured (non-random)'
                }
        
        return result
    
    def analyze_citation_network(self) -> Dict[str, Any]:
        """Analyze citation network structure (L1 -> L2 citations via CITES)."""
        # In-degree distribution (most cited L2 papers)
        in_degree_query = """
        MATCH (l2:L2Publication)<-[:CITES]-(p:Publication)
        WITH l2, count(p) as times_cited
        RETURN 
            avg(times_cited) as avg_citations,
            max(times_cited) as max_citations,
            percentileCont(times_cited, 0.5) as median_citations,
            count(l2) as cited_publications
        """
        in_degree_results = self.run_query(in_degree_query, "Citation in-degrees (L2 papers)")
        
        # In-degree distribution histogram
        in_hist_query = """
        MATCH (l2:L2Publication)<-[:CITES]-(p:Publication)
        WITH l2, count(p) as times_cited
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
        MATCH (l2:L2Publication)<-[:CITES]-(p:Publication)
        WITH l2, count(p) as times_cited
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
        
        # Total citations (CITES relationships)
        total_citations_query = """
        MATCH (:Publication)-[c:CITES]->(:L2Publication)
        RETURN count(c) as total_citations
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
                actual_avg = result['in_degree_stats'].get('avg_citations') or 0
                result['erdos_renyi_comparison'] = {
                    'actual_avg_in_degree': actual_avg,
                    'er_expected_avg_in_degree': round(er_avg_in_degree, 2),
                    'interpretation': 'Similar to random' if abs(actual_avg - er_avg_in_degree) < 0.5 else 'Preferential attachment (non-random)'
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
            'predicate_concentration': self.get_predicate_concentration(),  # NEW
            'cross_jurisdictional_entities': self.get_cross_jurisdictional_entities(2),
            'academic_regulation_bridge': self.get_academic_regulation_bridge(),
            'citation_enrichment': self.get_citation_enrichment_stats(),
            'entity_type_distribution': self.get_entity_type_distribution(),
            'provenance_coverage': self.get_provenance_coverage(),
            'orphan_stats': self.get_orphan_stats(),
            # KG-specific metrics
            'predicate_diversity': self.get_predicate_diversity(),
            'relation_density': self.get_relation_density(),
            'average_degree': self.get_average_degree(),  # NEW - thesis claims 14.9
            'semantic_degree': self.get_semantic_degree(),  # NEW - RELATION edges only
            'local_clustering': self.get_local_clustering(),  # NEW - community structure
            'type_affinity': self.get_type_affinity_matrix(),  # NEW - type connections
            'path_length_sample': self.get_path_length_sample(100),  # NEW - Steiner tree cost
            'entity_quality': self.get_entity_quality_flags(),  # NEW - extraction quality
            'degree_distribution': self.get_degree_distribution_buckets(),
            'property_completeness': self.get_property_completeness(),
            'cross_type_connectivity': self.get_cross_type_connectivity(),
            # Graph cohesion metrics
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
        
        # Print report
        if self.verbose:
            self._print_verbose_report(results)
        else:
            self._print_report(results)
        
        return results
    
    def _print_report(self, results: Dict[str, Any]):
        """Print clean summary report with clear section separation."""
        cov = results['coverage_stats']
        cc = results.get('connected_components', {})
        homophily = results.get('type_homophily', {})
        integration = results.get('layer_integration', {})
        density = results['relation_density']
        cite_net = results.get('citation_network', {})
        bridge_net = results.get('bridge_network_analysis', {})
        
        print("\n" + "="*75)
        print("  GRAPHRAG KNOWLEDGE GRAPH ANALYTICS REPORT")
        print("="*75)
        
        # ===== OVERVIEW =====
        print(f"\n{'─'*75}")
        print("  OVERVIEW")
        print(f"{'─'*75}")
        print(f"  Documents: {cov.get('jurisdictions', 0)} jurisdictions + {cov.get('publications', 0)} academic papers")
        print(f"  Chunks: {cov.get('chunks', 0):,}")
        
        # ===== SEMANTIC LAYER =====
        print(f"\n{'─'*75}")
        print("  SEMANTIC LAYER  (Extracted entities and their relationships)")
        print(f"{'─'*75}")
        print(f"  Entities: {cov.get('entities', 0):,}")
        print(f"  Relations: {cov.get('semantic_relations', 0):,}")
        
        # Degree metrics
        avg_deg = results.get('average_degree', {})
        sem_deg = results.get('semantic_degree', {})
        print(f"  Average degree (all edges): {avg_deg.get('avg_degree', 0):.1f}")
        print(f"  Average semantic degree (RELATION only): {sem_deg.get('avg_semantic_degree', 0):.1f}")
        print(f"  Relation density: {density.get('relations_per_entity', 0):.1f} per entity  [{density.get('interpretation', '')}]")
        
        # Connectivity - two metrics
        print(f"\n  Connectivity:")
        print(f"    Connected (any edge): {cc.get('connected_any_pct', 0):.1f}%  [data quality]")
        print(f"    Has RELATION edges: {cc.get('has_relations_pct', 0):.1f}%  [retrieval effectiveness]")
        print(f"    Truly isolated: {cc.get('truly_isolated', 0):,}")
        print(f"    → {cc.get('interpretation', '')}")
        if cc.get('provenance_note'):
            print(f"    → {cc.get('provenance_note')}")
        
        # Predicate concentration
        pred_conc = results.get('predicate_concentration', {})
        if pred_conc.get('top_4_cumulative_pct'):
            print(f"\n  Predicate concentration:")
            print(f"    Top 4 predicates cover: {pred_conc.get('top_4_cumulative_pct')}%")
            if pred_conc.get('predicates'):
                for p in pred_conc['predicates'][:4]:
                    print(f"      {p['predicate']:<20} {p['count']:>6,} ({p['pct']}%)")
        
        print(f"\n  Type homophily: {homophily.get('homophily_pct', 0):.1f}% intra-type  [<30% = rich cross-type connections]")
        
        # Clustering and path length
        clustering = results.get('local_clustering', {})
        paths = results.get('path_length_sample', {})
        print(f"  Clustering coefficient: {clustering.get('clustering_coef', 0):.4f}  [{clustering.get('interpretation', '')}]")
        print(f"  Avg path length: {paths.get('avg_path_length', 'N/A')}  [{paths.get('interpretation', '')}]")
        
        # Entity quality
        quality = results.get('entity_quality', {})
        if quality.get('issue_rate_pct', 0) > 1:
            print(f"  Entity quality issues: {quality.get('issue_rate_pct', 0):.1f}%  [{quality.get('interpretation', '')}]")
        
        # Show isolated entities by type if significant isolation
        if cc.get('no_relations_pct', 0) > 20 and cc.get('isolated_by_type'):
            print(f"\n  Entities without RELATION edges by type:")
            for item in cc.get('isolated_by_type', [])[:5]:
                print(f"    {item['entity_type']:<20} {item['count']:>6,}")
        
        print(f"\n  Top Connected Entities:")
        for i, entity in enumerate(results['top_connected_entities'][:5], 1):
            print(f"    {i}. {entity['entity'][:35]:<35} ({entity['type']:<16}) {entity['degree']:,} connections")
        
        # ===== METADATA LAYER =====
        print(f"\n{'─'*75}")
        print("  METADATA LAYER  (Structured bibliographic data)")
        print(f"{'─'*75}")
        print(f"  L1 Publications (in corpus): {cov.get('publications', 0)}")
        print(f"  L2 Publications (cited):     {cite_net.get('total_l2_publications', 0)}")
        print(f"  Authors: {results.get('author_collaboration_network', {}).get('total_authors', 0)}")
        journal_count = next((n['count'] for n in results.get('node_counts', []) if n.get('node_type') == 'Journal'), 0)
        print(f"  Journals: {journal_count}")
        
        author_net = results.get('author_collaboration_network', {})
        if author_net.get('degree_stats'):
            print(f"\n  Author Collaboration Network:")
            print(f"    Avg co-authors per author: {author_net['degree_stats'].get('avg_collaborators', 0):.1f}")
            er = author_net.get('erdos_renyi_comparison', {})
            if er:
                print(f"    vs Erdős-Rényi random graph: {er.get('er_expected_avg_degree', 0):.1f} expected")
                print(f"    [{er.get('interpretation', '')}]")
        
        print(f"\n  Citation Network:")
        in_stats = cite_net.get('in_degree_stats', {})
        print(f"    Total citations: {cite_net.get('total_citations', 0)}")
        print(f"    Avg citations per L2: {in_stats.get('avg_citations', 0) or 0:.2f}")
        print(f"    Max citations: {in_stats.get('max_citations', 0) or 0}")
        if in_stats.get('max_citations', 0) <= 1:
            print(f"    [!] All L2s cited exactly once - likely deduplication issue")
        
        # ===== CROSS-LAYER INTEGRATION =====
        print(f"\n{'─'*75}")
        print("  CROSS-LAYER INTEGRATION  (Semantic ↔ Metadata connections)")
        print(f"{'─'*75}")
        print(f"")
        print(f"  {'Cross-Layer Relation':<40} {'Count':>10}")
        print(f"  {'-'*38} {'-'*10}")
        print(f"  {'EXTRACTED_FROM (Entity→Chunk)':<40} {integration.get('extracted_from', 0):>10,}")
        print(f"  {'CONTAINS (Jurisdiction→Chunk)':<40} {integration.get('jur_contains', 0):>10,}")
        print(f"  {'CONTAINS (Publication→Chunk)':<40} {integration.get('pub_contains', 0):>10,}")
        print(f"  {'SAME_AS (Entity→Jurisdiction)':<40} {integration.get('same_as_jurisdiction', 0):>10,}")
        print(f"  {'SAME_AS (Entity→Author)':<40} {integration.get('same_as_author', 0):>10,}")
        print(f"  {'SAME_AS (Entity→Journal)':<40} {integration.get('same_as_journal', 0):>10,}")
        print(f"  {'MATCHED_TO (Entity→Publication)':<40} {integration.get('matched_to', 0):>10,}")
        print(f"  {'CITES (Publication→L2)':<40} {integration.get('cites', 0):>10,}")
        print(f"  {'AUTHORED_BY (Publication→Author)':<40} {integration.get('authored_by', 0):>10,}")
        print(f"  {'PUBLISHED_IN (Publication→Journal)':<40} {integration.get('published_in', 0):>10,}")
        print(f"  {'-'*38} {'-'*10}")
        print(f"  {'TOTAL CROSS-LAYER RELATIONS':<40} {integration.get('total_cross_layer', 0):>10,}")
        
        # ===== DOMAIN ANALYSIS =====
        print(f"\n{'─'*75}")
        print("  DOMAIN ANALYSIS  (Cross-jurisdictional and academic-regulatory)")
        print(f"{'─'*75}")
        
        cross_jur = results['cross_jurisdictional_entities']
        if cross_jur:
            multi_jur = [e for e in cross_jur if e['coverage'] >= 2]
            print(f"  Cross-Jurisdictional Entities:")
            print(f"    Entities in 2+ jurisdictions: {len(multi_jur)}")
            print(f"    Max jurisdictions: {max(e['coverage'] for e in cross_jur)}")
            print(f"    Top: {cross_jur[0]['entity']} ({cross_jur[0]['coverage']} jurisdictions)")
        
        bridge_stats = bridge_net.get('bridge_stats', {})
        domain_dist = bridge_net.get('domain_distribution', [])
        cohesion = bridge_net.get('cohesion', {})
        
        print(f"\n  Academic-Regulatory Bridges:")
        print(f"    Entities appearing in BOTH regulatory docs AND academic papers")
        if domain_dist:
            for d in domain_dist:
                label = {'bridging': 'Bridging (both)', 'academic_only': 'Academic only', 'regulation_only': 'Regulatory only'}.get(d['category'], d['category'])
                print(f"      {label:<20} {d['entity_count']:>6,} entities")
        if cohesion:
            print(f"    Bridge rate: {cohesion.get('bridging_percentage', 0):.1f}%  [{cohesion.get('interpretation', '')}]")
        
        # ===== ISSUES =====
        orphans = results['orphan_stats']
        issues = []
        
        if orphans:
            for node_type, count in orphans.items():
                issues.append(f"WARNING: {count} orphan {node_type} nodes (no relationships)")
        
        if integration.get('jurisdiction_coverage_pct', 0) == 0 and integration.get('same_as_coverage_pct', 0) == 0:
            issues.append("WARNING: No SAME_AS relations imported - entity↔metadata linking broken")
        
        if cite_net.get('in_degree_stats', {}).get('max_citations', 0) <= 1 and cite_net.get('total_l2_publications', 0) > 100:
            issues.append("WARNING: L2 publications not deduplicated - each cited exactly once")
        
        print(f"\n{'─'*75}")
        print("  ISSUES & WARNINGS")
        print(f"{'─'*75}")
        if issues:
            for issue in issues:
                print(f"  {issue}")
        else:
            print(f"  None detected")
        
        print("\n" + "="*75)
        print("  Use --verbose for full metrics, distributions, and top-N lists")
        print("="*75 + "\n")
    
    def _print_verbose_report(self, results: Dict[str, Any]):
        """Print detailed report with clear section separation."""
        cov = results['coverage_stats']
        
        print("\n" + "="*80)
        print("  GRAPHRAG KNOWLEDGE GRAPH - DETAILED ANALYTICS REPORT")
        print("="*80)
        
        # ===================================================================
        # OVERVIEW
        # ===================================================================
        print(f"\n{'━'*80}")
        print("  1. OVERVIEW")
        print(f"{'━'*80}")
        print(f"  Jurisdictions: {cov.get('jurisdictions', 0)}")
        print(f"  Academic Publications: {cov.get('publications', 0)}")
        print(f"  Chunks: {cov.get('chunks', 0):,}")
        print(f"  Entities: {cov.get('entities', 0):,}")
        print(f"  Semantic Relations: {cov.get('semantic_relations', 0):,}")
        
        # ===================================================================
        # SEMANTIC LAYER
        # ===================================================================
        print(f"\n{'━'*80}")
        print("  2. SEMANTIC LAYER  (Extracted entities and relationships)")
        print(f"{'━'*80}")
        print("     Entities extracted from text via LLM, connected by semantic relations.")
        print("     This forms the core knowledge graph for question answering.")
        
        # Relation Density
        density = results['relation_density']
        print(f"\n  2.1 Relation Density")
        print(f"      {density.get('relations_per_entity', 0):.2f} relations/entity")
        print(f"      Interpretation: {density.get('interpretation', 'N/A')}")
        print(f"      Benchmark: Domain KGs typically have 2-5 relations/entity")
        
        # Predicate Diversity
        pred_div = results['predicate_diversity']
        print(f"\n  2.2 Predicate Diversity (semantic richness)")
        print(f"      Avg predicates per entity: {pred_div.get('avg_predicates_per_entity', 0):.2f}")
        print(f"      Max predicates: {pred_div.get('max_predicates', 0)}")
        print(f"      Median: {pred_div.get('median_predicates', 0):.1f}")
        
        # Degree Distribution
        print(f"\n  2.3 Degree Distribution (power-law analysis)")
        print(f"      Healthy KGs show power-law: many low-degree, few high-degree nodes")
        deg_dist = results['degree_distribution']
        for bucket in deg_dist:
            print(f"        {bucket['bucket']:15} {bucket['entity_count']:>6,} entities")
        
        # Average Degree
        avg_deg = results.get('average_degree', {})
        sem_deg = results.get('semantic_degree', {})
        print(f"\n  2.4 Average Degree")
        print(f"      All edges: {avg_deg.get('avg_degree', 0):.2f} (max: {avg_deg.get('max_degree', 0)}, median: {avg_deg.get('median_degree', 0)})")
        print(f"      RELATION only: {sem_deg.get('avg_semantic_degree', 0):.2f} (max: {sem_deg.get('max_semantic_degree', 0)}, median: {sem_deg.get('median_semantic_degree', 0)})")
        print(f"      Zero semantic degree: {sem_deg.get('zero_degree_entities', 0):,} entities")
        
        # Connectivity
        cc = results.get('connected_components', {})
        print(f"\n  2.5 Connectivity Analysis")
        print(f"      Total entities: {cc.get('total_entities', 0):,}")
        print(f"      Connected (any edge): {cc.get('connected_any', 0):,} ({cc.get('connected_any_pct', 0):.1f}%)")
        print(f"      Has RELATION edges: {cc.get('has_semantic_relations', 0):,} ({cc.get('has_relations_pct', 0):.1f}%)")
        print(f"      No RELATION edges: {cc.get('no_semantic_relations', 0):,} ({cc.get('no_relations_pct', 0):.1f}%)")
        print(f"      Truly isolated (no edges): {cc.get('truly_isolated', 0):,}")
        if cc.get('provenance_note'):
            print(f"      Note: {cc.get('provenance_note')}")
        print(f"      Interpretation: {cc.get('interpretation', 'N/A')}")
        
        # Show isolated by type if significant
        if cc.get('isolated_by_type'):
            print(f"\n      Entities without RELATION edges by type:")
            for item in cc.get('isolated_by_type', [])[:7]:
                print(f"        {item['entity_type']:<25} {item['count']:>6,}")
        
        # Predicate Concentration
        pred_conc = results.get('predicate_concentration', {})
        if pred_conc.get('predicates'):
            print(f"\n  2.6 Predicate Concentration")
            print(f"      Top 4 predicates cover: {pred_conc.get('top_4_cumulative_pct', 'N/A')}%")
            print(f"      Top 10 predicates cover: {pred_conc.get('top_10_cumulative_pct', 'N/A')}%")
            print(f"\n      {'Predicate':<25} {'Count':>8} {'%':>6} {'Cumul%':>8}")
            print(f"      {'-'*25} {'-'*8} {'-'*6} {'-'*8}")
            for p in pred_conc['predicates'][:10]:
                print(f"      {p['predicate']:<25} {p['count']:>8,} {p['pct']:>6.1f} {p['cumulative_pct']:>8.1f}")
        
        # Clustering Coefficient
        clustering = results.get('local_clustering', {})
        print(f"\n  2.7 Local Clustering Coefficient")
        print(f"      Triangles: {clustering.get('triangles', 0):,}")
        print(f"      Triplets: {clustering.get('triplets', 0):,}")
        print(f"      Clustering: {clustering.get('clustering_coef', 0):.4f}")
        print(f"      Interpretation: {clustering.get('interpretation', 'N/A')}")
        
        # Path Length
        paths = results.get('path_length_sample', {})
        print(f"\n  2.8 Path Length Analysis (Steiner tree cost)")
        print(f"      Sample size: {paths.get('paths_found', 0)} paths")
        print(f"      Avg path length: {paths.get('avg_path_length', 'N/A')}")
        print(f"      Max path length: {paths.get('max_path_length', 'N/A')}")
        print(f"      Interpretation: {paths.get('interpretation', 'N/A')}")
        
        # Type Homophily
        homophily = results.get('type_homophily', {})
        print(f"\n  2.9 Type Homophily (do same-type entities cluster together?)")
        print(f"      Total relations: {homophily.get('total_relations', 0):,}")
        print(f"      Intra-type (A→A): {homophily.get('intra_type_relations', 0):,} ({homophily.get('homophily_pct', 0):.1f}%)")
        print(f"      Inter-type (A→B): {homophily.get('inter_type_relations', 0):,}")
        print(f"      Interpretation: {homophily.get('interpretation', 'N/A')}")
        print(f"      [<30% = rich cross-type semantic connections]")
        
        homophily_breakdown = results.get('type_homophily_breakdown', [])
        if homophily_breakdown:
            print(f"\n      By entity type:")
            for h in homophily_breakdown[:5]:
                print(f"        {h['source_type']:20} {h['homophily_pct']:5.1f}% intra-type ({h['total']:,} relations)")
        
        # Type Affinity Matrix
        type_affinity = results.get('type_affinity', [])
        if type_affinity:
            print(f"\n  2.10 Type Affinity Matrix (which types connect?)")
            print(f"      {'Source':<20} {'Target':<20} {'Count':>8}")
            print(f"      {'-'*20} {'-'*20} {'-'*8}")
            for ta in type_affinity[:10]:
                print(f"      {ta['source_type']:<20} {ta['target_type']:<20} {ta['count']:>8,}")
        
        # Hub Analysis
        hub_types = results.get('hub_centrality_by_type', [])
        if hub_types:
            print(f"\n  2.11 Hub Centrality by Type")
            print(f"      Hub = highly connected entity. Categories: mega(≥100), major(≥50), minor(≥20)")
            for h in hub_types[:10]:
                print(f"        {h['entity_type']:20} {h['total_hubs']:>4} hubs ({h['hub_rate_pct']:>5.1f}%)  [mega:{h['mega_hubs']:>2}, major:{h['major_hubs']:>3}, minor:{h['minor_hubs']:>4}]")
        
        # Entity Quality
        quality = results.get('entity_quality', {})
        print(f"\n  2.12 Entity Quality Flags")
        print(f"      Total entities: {quality.get('total_entities', 0):,}")
        print(f"      Very long names (>100 chars): {quality.get('very_long_names', 0):,}")
        print(f"      Bracket names: {quality.get('bracket_names', 0):,}")
        print(f"      Very short names (<3 chars): {quality.get('very_short_names', 0):,}")
        print(f"      Numeric only: {quality.get('numeric_only_names', 0):,}")
        print(f"      Issue rate: {quality.get('issue_rate_pct', 0):.1f}%")
        print(f"      Interpretation: {quality.get('interpretation', 'N/A')}")
        
        # Top Connected Entities
        print(f"\n  2.13 Top 10 Connected Entities")
        for i, entity in enumerate(results['top_connected_entities'][:10], 1):
            print(f"      {i:2}. {entity['entity'][:45]:<45} ({entity['type']:<16}) {entity['degree']:>5,} conn")
        
        # Entity Type Distribution
        print(f"\n  2.14 Entity Type Distribution (Top 10)")
        entity_types = results['entity_type_distribution'][:10]
        for et in entity_types:
            print(f"        {et['entity_type']:20} {et['count']:>6,}")
        
        # ===================================================================
        # METADATA LAYER
        # ===================================================================
        print(f"\n{'━'*80}")
        print("  3. METADATA LAYER  (Structured bibliographic data)")
        print(f"{'━'*80}")
        print("     Publications, authors, journals - structured data from Scopus/references.")
        
        # Property Completeness
        completeness = results['property_completeness']
        print(f"\n  3.1 Metadata Completeness")
        print(f"      Publications: {completeness.get('publication_completeness_pct', 0):.1f}% (title, year, DOI, abstract)")
        print(f"      Authors: {completeness.get('author_completeness_pct', 0):.1f}% (name)")
        print(f"      Journals: {completeness.get('journal_completeness_pct', 0):.1f}% (name)")
        
        # Author Collaboration Network
        print(f"\n  3.2 Author Collaboration Network")
        print(f"      Two authors are 'collaborators' if they co-authored a paper.")
        author_net = results['author_collaboration_network']
        deg_stats = author_net.get('degree_stats', {})
        print(f"      Total authors: {author_net.get('total_authors', 0)}")
        print(f"      With collaborators: {deg_stats.get('authors_with_collaborators', 0)}")
        print(f"      Avg collaborators: {deg_stats.get('avg_collaborators', 0):.2f}")
        print(f"      Max collaborators: {deg_stats.get('max_collaborators', 0)}")
        print(f"      Median: {deg_stats.get('median_collaborators', 0):.1f}")
        
        if 'erdos_renyi_comparison' in author_net:
            er = author_net['erdos_renyi_comparison']
            print(f"\n      Erdős-Rényi comparison (is structure random or meaningful?):")
            print(f"        Actual avg degree: {er['actual_avg_degree']:.2f}")
            print(f"        Random graph expected: {er['er_expected_avg_degree']:.2f}")
            print(f"        Interpretation: {er['interpretation']}")
        
        print(f"\n      Collaboration degree distribution:")
        for bucket in author_net.get('degree_distribution', []):
            print(f"        {bucket['bucket']:>6} collaborators → {bucket['author_count']:>4} authors")
        
        # Citation Network
        print(f"\n  3.3 Citation Network")
        print(f"      L1 = papers in corpus, L2 = papers they cite (references).")
        print(f"      CITES relationship: L1 Publication → L2 Publication")
        cite_net = results['citation_network']
        in_stats = cite_net.get('in_degree_stats', {})
        print(f"      L1 publications: {cov.get('publications', 0)}")
        print(f"      L2 publications: {cite_net.get('total_l2_publications', 0)}")
        print(f"      Total citations: {cite_net.get('total_citations', 0)}")
        print(f"      Avg citations per L2: {in_stats.get('avg_citations') or 0:.2f}")
        print(f"      Max citations: {in_stats.get('max_citations') or 0}")
        print(f"      Median: {in_stats.get('median_citations') or 0:.1f}")
        
        if 'erdos_renyi_comparison' in cite_net:
            er = cite_net['erdos_renyi_comparison']
            print(f"\n      Erdős-Rényi comparison:")
            print(f"        Actual avg in-degree: {er['actual_avg_in_degree']:.2f}")
            print(f"        Random expected: {er['er_expected_avg_in_degree']:.2f}")
            print(f"        Interpretation: {er['interpretation']}")
        
        if cite_net.get('in_degree_distribution'):
            print(f"\n      Citation in-degree distribution:")
            for bucket in cite_net.get('in_degree_distribution', []):
                print(f"        {bucket['bucket']:>6} citations → {bucket['pub_count']:>4} publications")
        
        if cite_net.get('top_cited'):
            print(f"\n      Top 10 Most Cited L2 Publications:")
            for i, pub in enumerate(cite_net.get('top_cited', [])[:10], 1):
                title = (pub['title'][:45] + '...') if pub['title'] and len(pub['title']) > 45 else (pub['title'] or 'N/A')
                author = pub['author'][:25] if pub['author'] else 'N/A'
                print(f"        {i:2}. {title:<50} by {author:<25} ({pub['times_cited']} cites)")
        
        # ===================================================================
        # CROSS-LAYER INTEGRATION
        # ===================================================================
        print(f"\n{'━'*80}")
        print("  4. CROSS-LAYER INTEGRATION  (Semantic ↔ Metadata connections)")
        print(f"{'━'*80}")
        
        integration = results.get('layer_integration', {})
        print(f"\n  4.1 Cross-Layer Relations (counts)")
        print(f"      {'Relation Type':<45} {'Count':>10}")
        print(f"      {'-'*43} {'-'*10}")
        print(f"      {'EXTRACTED_FROM (Entity→Chunk)':<45} {integration.get('extracted_from', 0):>10,}")
        print(f"      {'CONTAINS (Jurisdiction→Chunk)':<45} {integration.get('jur_contains', 0):>10,}")
        print(f"      {'CONTAINS (Publication→Chunk)':<45} {integration.get('pub_contains', 0):>10,}")
        print(f"      {'SAME_AS (Entity→Jurisdiction)':<45} {integration.get('same_as_jurisdiction', 0):>10,}")
        print(f"      {'SAME_AS (Entity→Author)':<45} {integration.get('same_as_author', 0):>10,}")
        print(f"      {'SAME_AS (Entity→Journal)':<45} {integration.get('same_as_journal', 0):>10,}")
        print(f"      {'MATCHED_TO (Entity→Publication)':<45} {integration.get('matched_to', 0):>10,}")
        print(f"      {'CITES (Publication→L2)':<45} {integration.get('cites', 0):>10,}")
        print(f"      {'AUTHORED_BY (Publication→Author)':<45} {integration.get('authored_by', 0):>10,}")
        print(f"      {'PUBLISHED_IN (Publication→Journal)':<45} {integration.get('published_in', 0):>10,}")
        print(f"      {'-'*43} {'-'*10}")
        print(f"      {'TOTAL':<45} {integration.get('total_cross_layer', 0):>10,}")
        
        # Cross-Type Connectivity
        print(f"\n  4.2 Cross-Type Connectivity (top relationship patterns)")
        cross_type = results['cross_type_connectivity'][:10]
        for conn in cross_type:
            print(f"        {conn['from_type']:15} →[{conn['rel_type']:15}]→ {conn['to_type']:15} ({conn['connections']:,})")
        
        # Provenance
        prov = results['provenance_coverage']
        print(f"\n  4.3 Provenance Coverage")
        print(f"      Entities with chunk links: {prov.get('entity_coverage_pct', 0):.1f}%")
        print(f"      Chunks with source docs: {prov.get('chunk_coverage_pct', 0):.1f}%")
        
        # ===================================================================
        # DOMAIN ANALYSIS
        # ===================================================================
        print(f"\n{'━'*80}")
        print("  5. DOMAIN ANALYSIS  (Cross-jurisdictional & Academic-Regulatory)")
        print(f"{'━'*80}")
        
        # Cross-Jurisdictional
        print(f"\n  5.1 Cross-Jurisdictional Coverage")
        print(f"      Entities appearing in regulatory docs from multiple jurisdictions.")
        cross_jur = results['cross_jurisdictional_entities']
        if cross_jur:
            print(f"      Entities in 2+ jurisdictions: {len([e for e in cross_jur if e['coverage'] >= 2])}")
            print(f"      Entities in 3+ jurisdictions: {len([e for e in cross_jur if e['coverage'] >= 3])}")
            print(f"      Max coverage: {max(e['coverage'] for e in cross_jur)} jurisdictions")
            print(f"\n      Top cross-jurisdictional entities:")
            for i, entity in enumerate(cross_jur[:5], 1):
                print(f"        {i}. {entity['entity'][:40]:<40} {entity['coverage']:>2} jurisdictions")
        
        # Academic-Regulatory Bridges
        print(f"\n  5.2 Academic-Regulatory Bridges")
        print(f"      Entities appearing in BOTH regulatory docs AND academic papers.")
        bridge_net = results['bridge_network_analysis']
        bridge_stats = bridge_net.get('bridge_stats', {})
        print(f"      Bridging entities: {bridge_stats.get('bridging_entities', 0):,}")
        print(f"      Avg jurisdictions per bridge: {bridge_stats.get('avg_jurisdictions_per_bridge', 0):.2f}")
        print(f"      Avg papers per bridge: {bridge_stats.get('avg_papers_per_bridge', 0):.2f}")
        
        if 'cohesion' in bridge_net:
            cohesion = bridge_net['cohesion']
            print(f"\n      Domain Distribution:")
            print(f"        Bridging: {cohesion['bridging_entities']:,} ({cohesion['bridging_percentage']:.1f}%)")
        
        for dist in bridge_net.get('domain_distribution', []):
            label = {'bridging': 'Bridging (both)', 'academic_only': 'Academic only', 'regulation_only': 'Regulatory only'}.get(dist['category'], dist['category'])
            print(f"        {label:<20} {dist['entity_count']:>6,} entities")
        
        if bridge_net.get('cohesion'):
            print(f"      Interpretation: {bridge_net['cohesion']['interpretation']}")
        
        if bridge_net.get('top_bridges'):
            print(f"\n      Top 10 Bridging Entities:")
            for i, bridge in enumerate(bridge_net.get('top_bridges', [])[:10], 1):
                print(f"        {i:2}. {bridge['entity'][:40]:<40} ({bridge['type']:<15}) {bridge['num_jurisdictions']:>2} jurs, {bridge['num_papers']:>3} papers")
        
        # ===================================================================
        # ISSUES
        # ===================================================================
        print(f"\n{'━'*80}")
        print("  6. ISSUES & WARNINGS")
        print(f"{'━'*80}")
        
        orphans = results['orphan_stats']
        issues_found = False
        
        if orphans:
            for node_type, count in orphans.items():
                print(f"  WARNING: {count} orphan {node_type} nodes (no relationships)")
                issues_found = True
        
        # Check for missing SAME_AS relations - use actual counts
        same_as_total = (integration.get('same_as_jurisdiction', 0) + 
                        integration.get('same_as_author', 0) + 
                        integration.get('same_as_journal', 0))
        if same_as_total == 0:
            print(f"  WARNING: No SAME_AS relations found - entity↔metadata linking may be broken")
            issues_found = True
        
        # Check for high isolation rate
        components = results.get('connected_components', {})
        isolation_rate = components.get('isolation_rate_pct', 0)
        if isolation_rate > 30:
            print(f"  WARNING: {isolation_rate:.1f}% of entities have no semantic relations")
            print(f"           This limits graph traversal effectiveness")
            issues_found = True
        
        cite_net = results['citation_network']
        if cite_net.get('in_degree_stats', {}).get('max_citations', 0) <= 1 and cite_net.get('total_l2_publications', 0) > 100:
            print(f"  WARNING: L2 publications not deduplicated - each cited exactly once")
            issues_found = True
        
        if not issues_found:
            print(f"  None detected")
        
        print("\n" + "="*80 + "\n")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Analyze GraphRAG knowledge graph')
    parser.add_argument('--uri', type=str, help='Neo4j URI (default: env NEO4J_URI)')
    parser.add_argument('--user', type=str, default='neo4j', help='Neo4j user (default: neo4j)')
    parser.add_argument('--password', type=str, help='Neo4j password (default: env NEO4J_PASSWORD)')
    parser.add_argument('--output', type=str, help='Output JSON file path (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show query-by-query progress and full report')
    
    args = parser.parse_args()
    
    # Suppress neo4j driver warnings (deprecated functions, missing properties)
    logging.getLogger('neo4j').setLevel(logging.ERROR)
    
    # Get credentials
    uri = args.uri or os.getenv('NEO4J_URI')
    user = args.user or os.getenv('NEO4J_USER', 'neo4j')
    password = args.password or os.getenv('NEO4J_PASSWORD')
    
    if not uri or not password:
        print("ERROR: --uri and --password required (or set NEO4J_URI and NEO4J_PASSWORD env vars)")
        sys.exit(1)
    
    # Run analysis
    analyzer = GraphAnalyzer(uri, user, password, verbose=args.verbose)
    
    try:
        results = analyzer.run_all_analyses()
        
        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\nResults saved to {output_path}")
        
    finally:
        analyzer.close()


if __name__ == '__main__':
    main()