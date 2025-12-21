# -*- coding: utf-8 -*-
"""
Neo4j Importer for GraphRAG Knowledge Graph.

Core Neo4j import functionality with batched UNWIND pattern.
Handles connection management, constraint creation, and all node/relationship
imports using efficient batch processing.

Author: Pau Barba i Colomer
Created: 2025-12-21
Modified: 2025-12-21

References:
    - See ARCHITECTURE.md § 3.2.2 for Phase 2B context
    - See PHASE_2B_DESIGN.md for Neo4j schema
"""

# Standard library
from pathlib import Path
from typing import List, Dict
import sys

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
from neo4j import GraphDatabase, Session
from tqdm import tqdm

# Project imports
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Neo4jImporter:
    """
    Core Neo4j import functionality with batched UNWIND pattern.
    
    Handles connection, constraints, and all node/relationship imports.
    
    Example:
        importer = Neo4jImporter(uri, user, password)
        with importer.driver.session() as session:
            importer.import_entities(session, entities)
    """
    
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI (e.g., neo4j+s://xxx.databases.neo4j.io)
            user: Username (typically 'neo4j')
            password: Database password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Connected to Neo4j at {uri}")
    
    def close(self):
        """Close Neo4j driver connection."""
        self.driver.close()
        logger.info("Neo4j connection closed")
    
    def clear_database(self, session: Session) -> int:
        """
        Clear all nodes and relationships from database.
        
        Args:
            session: Neo4j session
            
        Returns:
            Number of nodes deleted
        """
        logger.warning("Clearing entire database...")
        result = session.run("MATCH (n) DETACH DELETE n RETURN count(n) as count")
        count = result.single()['count']
        logger.info(f"Deleted {count} nodes and all relationships")
        return count
    
    def create_constraints_and_indexes(self, session: Session):
        """
        Create unique constraints and performance indexes.
        
        Constraints auto-create indexes for constrained properties.
        """
        logger.info("Creating constraints and indexes...")
        
        constraints = [
            "CREATE CONSTRAINT jurisdiction_code IF NOT EXISTS FOR (j:Jurisdiction) REQUIRE j.code IS UNIQUE",
            "CREATE CONSTRAINT publication_scopus IF NOT EXISTS FOR (p:Publication) REQUIRE p.scopus_id IS UNIQUE",
            "CREATE CONSTRAINT l2pub_id IF NOT EXISTS FOR (p:L2Publication) REQUIRE p.publication_id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT author_id IF NOT EXISTS FOR (a:Author) REQUIRE a.author_id IS UNIQUE",
            "CREATE CONSTRAINT journal_id IF NOT EXISTS FOR (j:Journal) REQUIRE j.journal_id IS UNIQUE",
        ]
        
        indexes = [
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX chunk_doc_type IF NOT EXISTS FOR (c:Chunk) ON (c.doc_type)",
            "CREATE INDEX relation_predicate IF NOT EXISTS FOR ()-[r:RELATION]-() ON (r.predicate)",
        ]
        
        for constraint in constraints:
            session.run(constraint)
            logger.debug(f"Created: {constraint[:50]}...")
        
        for index in indexes:
            session.run(index)
            logger.debug(f"Created: {index[:50]}...")
        
        logger.info(f"Created {len(constraints)} constraints and {len(indexes)} indexes")
    
    def batch_import(self, session: Session, query: str, data: List[Dict], 
                     batch_size: int = 500, desc: str = "Importing") -> int:
        """
        Import data in batches using UNWIND pattern.
        
        Args:
            session: Neo4j session
            query: Cypher query with UNWIND $batch
            data: List of dicts to import
            batch_size: Number of items per batch
            desc: Description for progress bar
            
        Returns:
            Total number of items imported
        """
        total = len(data)
        if total == 0:
            logger.warning(f"{desc}: No data to import")
            return 0
        
        with tqdm(total=total, desc=desc) as pbar:
            for i in range(0, total, batch_size):
                batch = data[i:i+batch_size]
                session.run(query, batch=batch)
                pbar.update(len(batch))
        
        logger.info(f"{desc}: Imported {total} items")
        return total
    
    # =========================================================================
    # NODE IMPORT FUNCTIONS
    # =========================================================================
    
    def import_jurisdictions(self, session: Session, jurisdictions: List[Dict]) -> int:
        """
        Import Jurisdiction nodes.
        
        Args:
            session: Neo4j session
            jurisdictions: List of dicts with code, name, url, scraped_date
        """
        query = """
        UNWIND $batch AS j
        CREATE (n:Jurisdiction {
            code: j.code,
            name: j.name,
            url: j.url,
            scraped_date: j.scraped_date
        })
        """
        return self.batch_import(session, query, jurisdictions, desc="Jurisdictions")
    
    def import_publications(self, session: Session, publications: List[Dict]) -> int:
        """
        Import Publication nodes (L1 - source papers).
        
        Args:
            session: Neo4j session
            publications: List of dicts with scopus_id, title, year, doi, cited_by
        """
        query = """
        UNWIND $batch AS p
        CREATE (n:Publication {
            scopus_id: p.scopus_id,
            title: p.title,
            year: p.year,
            doi: p.doi,
            cited_by: p.cited_by
        })
        """
        return self.batch_import(session, query, publications, desc="Publications (L1)")
    
    def import_authors(self, session: Session, authors: List[Dict]) -> int:
        """
        Import Author nodes.
        
        Args:
            session: Neo4j session
            authors: List of dicts with author_id, name, scopus_author_id
        """
        query = """
        UNWIND $batch AS a
        CREATE (n:Author {
            author_id: a.author_id,
            name: a.name,
            scopus_author_id: a.scopus_author_id
        })
        """
        return self.batch_import(session, query, authors, desc="Authors")
    
    def import_journals(self, session: Session, journals: List[Dict]) -> int:
        """
        Import Journal nodes.
        
        Args:
            session: Neo4j session
            journals: List of dicts with journal_id, name, issn
        """
        query = """
        UNWIND $batch AS j
        CREATE (n:Journal {
            journal_id: j.journal_id,
            name: j.name,
            issn: j.issn
        })
        """
        return self.batch_import(session, query, journals, desc="Journals")
    
    def import_chunks(self, session: Session, chunks: List[Dict]) -> int:
        """
        Import Chunk nodes.
        
        Args:
            session: Neo4j session
            chunks: List of dicts with chunk_id, text, doc_type, jurisdiction,
                   scopus_id, section_title
        """
        query = """
        UNWIND $batch AS c
        CREATE (n:Chunk {
            chunk_id: c.chunk_id,
            text: c.text,
            doc_type: c.doc_type,
            jurisdiction: c.jurisdiction,
            scopus_id: c.scopus_id,
            section_title: c.section_title
        })
        """
        return self.batch_import(session, query, chunks, desc="Chunks")
    
    def import_entities(self, session: Session, entities: List[Dict]) -> int:
        """
        Import Entity nodes (WITHOUT embeddings, WITH aliases).
        
        v1.1: Added aliases property as string array.
        
        Args:
            session: Neo4j session
            entities: List of dicts with entity_id, name, type, description,
                     frequency, aliases (list of strings)
        """
        query = """
        UNWIND $batch AS e
        CREATE (n:Entity {
            entity_id: e.entity_id,
            name: e.name,
            type: e.type,
            description: e.description,
            frequency: e.frequency,
            aliases: e.aliases
        })
        """
        return self.batch_import(session, query, entities, desc="Entities")
    
    def import_l2_publications(self, session: Session, l2_pubs: List[Dict]) -> int:
        """
        Import L2Publication nodes (cited papers).
        
        Args:
            session: Neo4j session
            l2_pubs: List of dicts with publication_id, title, author, year, journal
        """
        query = """
        UNWIND $batch AS p
        CREATE (n:L2Publication {
            publication_id: p.publication_id,
            title: p.title,
            author: p.author,
            year: p.year,
            journal: p.journal
        })
        """
        return self.batch_import(session, query, l2_pubs, desc="L2 Publications")
    
    # =========================================================================
    # RELATIONSHIP IMPORT FUNCTIONS
    # =========================================================================
    
    def import_contains_jurisdiction(self, session: Session, relations: List[Dict]) -> int:
        """
        Import CONTAINS relationships: Jurisdiction -> Chunk.
        
        Args:
            session: Neo4j session
            relations: List of dicts with jurisdiction_code, chunk_id
        """
        query = """
        UNWIND $batch AS rel
        MATCH (j:Jurisdiction {code: rel.jurisdiction_code})
        MATCH (c:Chunk {chunk_id: rel.chunk_id})
        CREATE (j)-[:CONTAINS]->(c)
        """
        return self.batch_import(session, query, relations, desc="CONTAINS (Jurisdiction→Chunk)")
    
    def import_contains_publication(self, session: Session, relations: List[Dict]) -> int:
        """
        Import CONTAINS relationships: Publication -> Chunk.
        
        Args:
            session: Neo4j session
            relations: List of dicts with scopus_id, chunk_id
        """
        query = """
        UNWIND $batch AS rel
        MATCH (p:Publication {scopus_id: rel.scopus_id})
        MATCH (c:Chunk {chunk_id: rel.chunk_id})
        CREATE (p)-[:CONTAINS]->(c)
        """
        return self.batch_import(session, query, relations, desc="CONTAINS (Publication→Chunk)")
    
    def import_authored_by(self, session: Session, relations: List[Dict]) -> int:
        """
        Import AUTHORED_BY relationships: Publication -> Author.
        
        Args:
            session: Neo4j session
            relations: List of dicts with scopus_id, author_id
        """
        query = """
        UNWIND $batch AS rel
        MATCH (p:Publication {scopus_id: rel.scopus_id})
        MATCH (a:Author {author_id: rel.author_id})
        CREATE (p)-[:AUTHORED_BY]->(a)
        """
        return self.batch_import(session, query, relations, desc="AUTHORED_BY")
    
    def import_published_in(self, session: Session, relations: List[Dict]) -> int:
        """
        Import PUBLISHED_IN relationships: Publication -> Journal.
        
        Args:
            session: Neo4j session
            relations: List of dicts with scopus_id, journal_id
        """
        query = """
        UNWIND $batch AS rel
        MATCH (p:Publication {scopus_id: rel.scopus_id})
        MATCH (j:Journal {journal_id: rel.journal_id})
        CREATE (p)-[:PUBLISHED_IN]->(j)
        """
        return self.batch_import(session, query, relations, desc="PUBLISHED_IN")
    
    def import_extracted_from(self, session: Session, relations: List[Dict]) -> int:
        """
        Import EXTRACTED_FROM relationships: Entity -> Chunk.
        
        Args:
            session: Neo4j session
            relations: List of dicts with entity_id, chunk_id
        """
        query = """
        UNWIND $batch AS rel
        MATCH (e:Entity {entity_id: rel.entity_id})
        MATCH (c:Chunk {chunk_id: rel.chunk_id})
        CREATE (e)-[:EXTRACTED_FROM]->(c)
        """
        return self.batch_import(session, query, relations, desc="EXTRACTED_FROM")
    
    def import_relations(self, session: Session, relations: List[Dict]) -> int:
        """
        Import RELATION relationships: Entity -> Entity with properties.
        
        Args:
            session: Neo4j session
            relations: List of dicts with subject_id, predicate, object_id,
                      chunk_ids, confidence
        """
        query = """
        UNWIND $batch AS rel
        MATCH (s:Entity {entity_id: rel.subject_id})
        MATCH (o:Entity {entity_id: rel.object_id})
        CREATE (s)-[:RELATION {
            predicate: rel.predicate,
            chunk_ids: rel.chunk_ids,
            confidence: rel.confidence
        }]->(o)
        """
        return self.batch_import(session, query, relations, desc="RELATION")
    
    def import_part_of(self, session: Session, relations: List[Dict]) -> int:
        """
        Import PART_OF relationships: Entity -> Entity (alias clusters).
        
        v1.1 addition for entity disambiguation results.
        
        Args:
            session: Neo4j session
            relations: List of dicts with source_id, target_id
        """
        query = """
        UNWIND $batch AS rel
        MATCH (s:Entity {entity_id: rel.source_id})
        MATCH (t:Entity {entity_id: rel.target_id})
        CREATE (s)-[:PART_OF]->(t)
        """
        return self.batch_import(session, query, relations, desc="PART_OF")
    
    def import_same_as_entity(self, session: Session, relations: List[Dict]) -> int:
        """
        Import SAME_AS relationships: Entity -> Entity (disambiguation).
        
        v1.1 addition for entity disambiguation results.
        
        Args:
            session: Neo4j session
            relations: List of dicts with source_id, target_id
        """
        query = """
        UNWIND $batch AS rel
        MATCH (s:Entity {entity_id: rel.source_id})
        MATCH (t:Entity {entity_id: rel.target_id})
        CREATE (s)-[:SAME_AS]->(t)
        """
        return self.batch_import(session, query, relations, desc="SAME_AS (Entity→Entity)")
    
    def import_same_as_jurisdiction(self, session: Session, relations: List[Dict]) -> int:
        """
        Import SAME_AS relationships: Entity -> Jurisdiction.
        
        Args:
            session: Neo4j session
            relations: List of dicts with entity_id, jurisdiction_code
        """
        query = """
        UNWIND $batch AS rel
        MATCH (e:Entity {entity_id: rel.entity_id})
        MATCH (j:Jurisdiction {code: rel.jurisdiction_code})
        CREATE (e)-[:SAME_AS]->(j)
        """
        return self.batch_import(session, query, relations, desc="SAME_AS (Entity→Jurisdiction)")
    
    def import_matched_to(self, session: Session, relations: List[Dict]) -> int:
        """
        Import MATCHED_TO relationships: Entity -> L2Publication.
        
        Args:
            session: Neo4j session
            relations: List of dicts with entity_id, publication_id
        """
        query = """
        UNWIND $batch AS rel
        MATCH (e:Entity {entity_id: rel.entity_id})
        MATCH (p:L2Publication {publication_id: rel.publication_id})
        CREATE (e)-[:MATCHED_TO]->(p)
        """
        return self.batch_import(session, query, relations, desc="MATCHED_TO")
    
    def import_cites_l2(self, session: Session, relations: List[Dict]) -> int:
        """
        Import CITES relationships: Publication -> L2Publication.
        
        Args:
            session: Neo4j session
            relations: List of dicts with scopus_id, publication_id
        """
        query = """
        UNWIND $batch AS rel
        MATCH (p:Publication {scopus_id: rel.scopus_id})
        MATCH (l2:L2Publication {publication_id: rel.publication_id})
        CREATE (p)-[:CITES]->(l2)
        """
        return self.batch_import(session, query, relations, desc="CITES (L1→L2)")
    
    def import_cites_l1(self, session: Session, relations: List[Dict]) -> int:
        """
        Import CITES relationships: Publication -> Publication (L1 to L1).
        
        Args:
            session: Neo4j session
            relations: List of dicts with source_scopus_id, target_scopus_id
        """
        query = """
        UNWIND $batch AS rel
        MATCH (p1:Publication {scopus_id: rel.source_scopus_id})
        MATCH (p2:Publication {scopus_id: rel.target_scopus_id})
        CREATE (p1)-[:CITES]->(p2)
        """
        return self.batch_import(session, query, relations, desc="CITES (L1→L1)")