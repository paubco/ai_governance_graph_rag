#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo4j Import Processor for GraphRAG Knowledge Graph

Orchestrates complete Neo4j import with checkpointing and progress tracking.
Handles loading all input files, database clearing with confirmation,
import in correct dependency order, and checkpoint management for resume capability.

Features:
- Checkpointing for resumable imports
- Automatic file loading and validation
- Dependency-ordered node and relationship creation
- Support for both regulatory and academic data sources
- Enrichment relation integration (citations, entity matching)

Usage:
    python src/graph/neo4j_import_processor.py --clear --uri bolt://localhost:7687
    python src/graph/neo4j_import_processor.py --force-restart
"""

# Standard library
import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Set
import sys
import argparse

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Local
from src.graph.neo4j_importer import Neo4jImporter
from src.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


class Neo4jImportProcessor:
    """
    Orchestrates complete Neo4j import with checkpointing.
    
    Handles:
    - Loading all input files
    - Database clearing with confirmation
    - Import in correct dependency order
    - Checkpoint management for resume capability
    """
    
    def __init__(self, data_dir: Path, checkpoint_dir: Path, force_restart: bool = False):
        """
        Initialize import processor.
        
        Args:
            data_dir: Root data directory (contains raw/ and processed/)
            checkpoint_dir: Directory for checkpoint files
            force_restart: If True, ignore existing checkpoint
        """
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = checkpoint_dir / "neo4j_import_checkpoint.json"
        self.force_restart = force_restart
        
        # Load checkpoint
        self.completed_steps: Set[str] = set()
        if not force_restart and self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                self.completed_steps = set(checkpoint.get('completed_steps', []))
                logger.info(f"Loaded checkpoint: {len(self.completed_steps)} steps completed")
        else:
            logger.info("Starting fresh import (no checkpoint)")
    
    def mark_completed(self, step: str):
        """
        Mark a step as completed in checkpoint.
        
        Args:
            step: Step identifier
        """
        self.completed_steps.add(step)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({'completed_steps': list(self.completed_steps)}, f, indent=2)
        logger.debug(f"Checkpoint: {step} completed")
    
    def is_completed(self, step: str) -> bool:
        """Check if step already completed."""
        return step in self.completed_steps
    
    def load_json(self, relative_path: str) -> List[Dict]:
        """Load JSON file with error handling."""
        path = self.data_dir / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} items from {relative_path}")
        return data
    
    def load_jsonl(self, relative_path: str) -> List[Dict]:
        """Load JSONL file with error handling."""
        path = self.data_dir / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        logger.info(f"Loaded {len(data)} items from {relative_path}")
        return data
    
    def load_scopus_csv(self, relative_path: str) -> List[Dict]:
        """Load Scopus CSV with UTF-8-sig encoding (handles BOM)."""
        path = self.data_dir / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        
        data = []
        with open(path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        logger.info(f"Loaded {len(data)} items from {relative_path}")
        return data
    
    def prepare_jurisdictions(self) -> List[Dict]:
        """Load and prepare jurisdiction nodes."""
        scraping_summary = self.load_json('raw/dlapiper/scraping_summary.json')
        
        # Extract countries array from wrapper dict
        countries_list = scraping_summary.get('countries', scraping_summary)
        
        jurisdictions = []
        for item in countries_list:
            jurisdictions.append({
                'code': item['code'],
                'name': item['name'],
                'url': item.get('url', ''),
                'scraped_date': item.get('scraped_date', 'unknown')
            })
        
        return jurisdictions
    
    def prepare_publications(self, scopus_data: List[Dict]) -> List[Dict]:
        """Prepare L1 publication nodes from Scopus CSV with publication_id from enrichment."""
        # Load publication_id mappings from enrichment
        try:
            pubs_enrichment = self.load_json('processed/enrichment/publications.json')
            scopus_to_pubid = {}
            for pub in pubs_enrichment:
                if pub.get('publication_id', '').startswith('pub_l1_'):
                    scopus_id = pub.get('scopus_id', '')
                    if scopus_id:
                        scopus_to_pubid[scopus_id] = pub['publication_id']
            logger.info(f"Loaded {len(scopus_to_pubid)} L1 publication_id mappings")
        except Exception as e:
            logger.warning(f"Could not load publication_id mappings: {e}")
            scopus_to_pubid = {}
        
        publications = []
        for row in scopus_data:
            scopus_id = row['EID']
            publications.append({
                'scopus_id': scopus_id,
                'publication_id': scopus_to_pubid.get(scopus_id, ''),  # Add publication_id
                'title': row['Title'],
                'year': int(row['Year']) if row['Year'] else None,
                'doi': row.get('DOI', ''),
                'cited_by': int(row['Cited by']) if row.get('Cited by') else 0
            })
        
        return publications
    
    def prepare_authors(self) -> List[Dict]:
        """Load author nodes from Phase 2A."""
        authors_data = self.load_json('processed/enrichment/authors.json')
        
        authors = []
        for author in authors_data:
            authors.append({
                'author_id': author['author_id'],
                'name': author['name'],
                'scopus_author_id': author.get('scopus_author_id', '')
            })
        
        return authors
    
    def prepare_journals(self) -> List[Dict]:
        """Load journal nodes from Phase 2A."""
        journals_data = self.load_json('processed/enrichment/journals.json')
        
        journals = []
        for journal in journals_data:
            journals.append({
                'journal_id': journal['journal_id'],
                'name': journal['name'],
                'issn': journal.get('issn', '')
            })
        
        return journals
    
    def prepare_chunks(self) -> List[Dict]:
        """Load chunk nodes (without embeddings)."""
        # Try chunks_embedded.jsonl first, fallback to chunks.jsonl
        try:
            chunks_data = self.load_jsonl('processed/chunks/chunks_embedded.jsonl')
        except FileNotFoundError:
            chunks_data = self.load_jsonl('interim/chunks/chunks.jsonl')
        
        chunks = []
        for chunk in chunks_data:
            # Handle both chunk_id and chunk_ids formats
            chunk_id = chunk.get('chunk_id') or (chunk.get('chunk_ids', [''])[0])
            doc_id = chunk.get('document_id') or (chunk.get('document_ids', [''])[0])
            
            metadata = chunk.get('metadata', {})
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk.get('text', ''),
                'doc_type': metadata.get('source_type', ''),
                'jurisdiction': metadata.get('country_code', ''),
                'scopus_id': doc_id,
                'section_title': chunk.get('section_header', metadata.get('section_header', ''))
            })
        
        return chunks
    
    def prepare_entities(self) -> List[Dict]:
        """Load entity nodes (strip embeddings for Neo4j)."""
        entities_data = self.load_json('interim/entities/normalized_entities_with_ids.json')
        
        # Handle dict structure if present
        if isinstance(entities_data, dict):
            entities_list = list(entities_data.values())
        else:
            entities_list = entities_data
        
        entities = []
        for entity in entities_list:
            # Strip embedding - not needed in Neo4j
            entities.append({
                'entity_id': entity['entity_id'],
                'name': entity.get('name', ''),
                'type': entity.get('type', ''),
                'description': entity.get('description', ''),
                'frequency': entity.get('frequency', 0)
            })
        
        return entities
    
    def prepare_l2_publications(self) -> List[Dict]:
        """Load L2 publication nodes from Phase 2A."""
        publications_data = self.load_json('processed/enrichment/publications.json')
        
        # Filter for L2 only (cited_publication type or pub_l2_ prefix)
        l2_pubs = []
        for pub in publications_data:
            pub_id = pub.get('publication_id', '')
            is_l2 = (pub.get('node_type') == 'cited_publication' or 
                     pub_id.startswith('pub_l2_'))
            if is_l2:
                l2_pubs.append({
                    'publication_id': pub_id,
                    'title': pub.get('title', ''),
                    'author': pub.get('author', ''),
                    'year': pub.get('year', ''),
                    'journal': pub.get('journal', '')
                })
        
        return l2_pubs
    
    def prepare_contains_jurisdiction(self, chunks: List[Dict]) -> List[Dict]:
        """Prepare CONTAINS relationships: Jurisdiction -> Chunk."""
        relations = []
        
        for chunk in chunks:
            if chunk['doc_type'] == 'regulation' and chunk['jurisdiction']:
                relations.append({
                    'jurisdiction_code': chunk['jurisdiction'],
                    'chunk_id': chunk['chunk_id']
                })
        
        return relations
    
    def load_paper_scopus_mapping(self) -> Dict[str, str]:
        """Load mapping from paper_XXX to Scopus EID."""
        mapping_file = self.data_dir / 'interim' / 'academic' / 'paper_scopus_matches.csv'
        
        if not mapping_file.exists():
            logger.warning(f"Paper-Scopus mapping file not found: {mapping_file}")
            return {}
        
        mapping = {}
        with open(mapping_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                paper_id = row['paper_id']  # e.g., "paper_145"
                scopus_id = row['scopus_eid']  # e.g., "2-s2.0-85183649331"
                if scopus_id:  # Only if match exists
                    mapping[paper_id] = scopus_id
        
        logger.info(f"Loaded {len(mapping)} paper→scopus mappings")
        return mapping
    
    def prepare_contains_publication(self, chunks: List[Dict], paper_mapping: Dict[str, str]) -> List[Dict]:
        """Prepare CONTAINS relationships: Publication -> Chunk."""
        relations = []
        
        for chunk in chunks:
            if chunk['doc_type'] == 'academic_paper' and chunk['scopus_id']:
                # Convert paper_XXX to actual scopus EID
                paper_id = chunk['scopus_id']  # This is actually paper_XXX from document_id
                actual_scopus_id = paper_mapping.get(paper_id)
                
                if actual_scopus_id:
                    relations.append({
                        'scopus_id': actual_scopus_id,
                        'chunk_id': chunk['chunk_id']
                    })
        
        return relations
    
    def prepare_authored_by(self, scopus_data: List[Dict]) -> List[Dict]:
        """Prepare AUTHORED_BY relationships from Scopus CSV."""
        relations = []
        
        for row in scopus_data:
            scopus_id = row['EID']
            author_ids = row.get('Author(s) ID', '').split(';')
            
            for author_id in author_ids:
                author_id = author_id.strip()
                if author_id:
                    relations.append({
                        'scopus_id': scopus_id,
                        'author_id': f'author_{author_id}'  # Fixed: prepend "author_"
                    })
        
        return relations
    
    def load_issn_to_journal_mapping(self) -> Dict[str, str]:
        """Create mapping from ISSN to journal_id."""
        journals = self.load_json('processed/enrichment/journals.json')
        
        mapping = {}
        for journal in journals:
            issn = journal.get('issn', '')
            journal_id = journal['journal_id']
            
            # Handle multiple ISSNs (semicolon-separated)
            for single_issn in issn.split(';'):
                single_issn = single_issn.strip()
                if single_issn:
                    mapping[single_issn] = journal_id
        
        logger.info(f"Loaded {len(mapping)} ISSN→journal_id mappings")
        return mapping
    
    def prepare_published_in(self, scopus_data: List[Dict], issn_mapping: Dict[str, str]) -> List[Dict]:
        """Prepare PUBLISHED_IN relationships from Scopus CSV."""
        relations = []
        
        for row in scopus_data:
            scopus_id = row['EID']
            issn_str = row.get('ISSN', '').strip()
            
            if issn_str:
                # Split multiple ISSNs (semicolon-separated)
                for single_issn in issn_str.split(';'):
                    single_issn = single_issn.strip()
                    if single_issn:
                        # Look up journal_id from ISSN
                        journal_id = issn_mapping.get(single_issn)
                        if journal_id:
                            relations.append({
                                'scopus_id': scopus_id,
                                'journal_id': journal_id
                            })
                            break  # Found match, stop trying other ISSNs for this paper
        
        return relations
    
    def prepare_extracted_from(self) -> List[Dict]:
        """Prepare EXTRACTED_FROM relationships from normalized entities."""
        entities_data = self.load_json('interim/entities/normalized_entities_with_ids.json')
        
        relations = []
        for entity in entities_data:
            # Each entity has chunk_ids list
            for chunk_id in entity.get('chunk_ids', []):
                relations.append({
                    'entity_id': entity['entity_id'],
                    'chunk_id': chunk_id
                })
        
        return relations
    
    def prepare_relations(self) -> List[Dict]:
        """Load normalized relations with IDs."""
        relations_data = self.load_json('interim/relations/relations_normalized.json')
        
        relations = []
        for rel in relations_data:
            relations.append({
                'subject_id': rel['subject_id'],
                'predicate': rel['predicate'],
                'object_id': rel['object_id'],
                'chunk_ids': rel.get('chunk_ids', []),
                'confidence': rel.get('confidence', 1.0)
            })
        
        return relations
    
    def prepare_enrichment_relations(self) -> Dict[str, List[Dict]]:
        """Load all enrichment relations from Phase 2A."""
        enrichment_data = self.load_json('processed/enrichment/enrichment_relations.json')
        
        # Group by relation_type
        categorized = {
            'matched_to': [],
            'cites': [],
            'same_as_jurisdiction': [],
            'same_as_author': [],
            'same_as_journal': [],
            'authored': [],
            'published_in': [],
            'contains_publication': []
        }
        
        for rel in enrichment_data:
            rel_type = rel['relation_type']
            
            if rel_type == 'MATCHED_TO':
                categorized['matched_to'].append({
                    'entity_id': rel['source_id'],
                    'publication_id': rel['target_id']
                })
            elif rel_type == 'CITES':
                categorized['cites'].append({
                    'source_id': rel['source_id'],  # scopus_id or pub_l1_xxx
                    'target_id': rel['target_id']   # pub_l2_xxx
                })
            elif rel_type == 'SAME_AS':
                target_type = rel.get('target_type', '')
                if target_type == 'Jurisdiction':
                    # target_id is jur_CA, extract code
                    jur_code = rel['target_id'].replace('jur_', '')
                    categorized['same_as_jurisdiction'].append({
                        'entity_id': rel['source_id'],
                        'jurisdiction_code': jur_code
                    })
                elif target_type == 'Author':
                    categorized['same_as_author'].append({
                        'entity_id': rel['source_id'],
                        'author_id': rel['target_id']
                    })
                elif target_type == 'Journal':
                    categorized['same_as_journal'].append({
                        'entity_id': rel['source_id'],
                        'journal_id': rel['target_id']
                    })
            elif rel_type == 'AUTHORED':
                categorized['authored'].append({
                    'entity_id': rel['source_id'],
                    'publication_id': rel['target_id']
                })
            elif rel_type == 'PUBLISHED_IN':
                categorized['published_in'].append({
                    'publication_id': rel['source_id'],
                    'journal_id': rel['target_id']
                })
            elif rel_type == 'CONTAINS':
                categorized['contains_publication'].append({
                    'publication_id': rel['source_id'],
                    'chunk_id': rel['target_id']
                })
        
        return categorized
    
    def run_import(self, uri: str, user: str, password: str, clear_db: bool = False):
        """
        Execute complete import process.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            clear_db: If True, clear database before import
        """
        importer = Neo4jImporter(uri, user, password)
        
        try:
            with importer.driver.session() as session:
                # Step 0: Clear database if requested
                if clear_db:
                    if not self.is_completed('clear_database'):
                        logger.warning("Clearing database...")
                        importer.clear_database(session)
                        self.mark_completed('clear_database')
                    else:
                        logger.info("Database already cleared (checkpoint)")
                
                # Step 1: Create constraints and indexes
                if not self.is_completed('constraints'):
                    importer.create_constraints_and_indexes(session)
                    self.mark_completed('constraints')
                else:
                    logger.info("Constraints already created (checkpoint)")
                
                # Load data files once
                logger.info("Loading input files...")
                scopus_data = self.load_scopus_csv('raw/academic/scopus_2023/scopus_export_2023_raw.csv')
                chunks = self.prepare_chunks()
                paper_mapping = self.load_paper_scopus_mapping()  # Load paper_XXX → scopus_id mapping
                issn_mapping = self.load_issn_to_journal_mapping()  # Load ISSN → journal_id mapping
                
                # ============================================================
                # PHASE 1: IMPORT NODES (in dependency order)
                # ============================================================
                
                logger.info("\n=== PHASE 1: IMPORTING NODES ===")
                
                # 1. Jurisdictions
                if not self.is_completed('nodes_jurisdiction'):
                    jurisdictions = self.prepare_jurisdictions()
                    importer.import_jurisdictions(session, jurisdictions)
                    self.mark_completed('nodes_jurisdiction')
                else:
                    logger.info("Jurisdictions already imported (checkpoint)")
                
                # 2. Publications (L1)
                if not self.is_completed('nodes_publication'):
                    publications = self.prepare_publications(scopus_data)
                    importer.import_publications(session, publications)
                    self.mark_completed('nodes_publication')
                else:
                    logger.info("Publications (L1) already imported (checkpoint)")
                
                # 3. Authors
                if not self.is_completed('nodes_author'):
                    authors = self.prepare_authors()
                    importer.import_authors(session, authors)
                    self.mark_completed('nodes_author')
                else:
                    logger.info("Authors already imported (checkpoint)")
                
                # 4. Journals
                if not self.is_completed('nodes_journal'):
                    journals = self.prepare_journals()
                    importer.import_journals(session, journals)
                    self.mark_completed('nodes_journal')
                else:
                    logger.info("Journals already imported (checkpoint)")
                
                # 5. Chunks
                if not self.is_completed('nodes_chunk'):
                    importer.import_chunks(session, chunks)
                    self.mark_completed('nodes_chunk')
                else:
                    logger.info("Chunks already imported (checkpoint)")
                
                # 6. Entities
                if not self.is_completed('nodes_entity'):
                    entities = self.prepare_entities()
                    importer.import_entities(session, entities)
                    self.mark_completed('nodes_entity')
                else:
                    logger.info("Entities already imported (checkpoint)")
                
                # 7. L2 Publications
                if not self.is_completed('nodes_l2publication'):
                    l2_pubs = self.prepare_l2_publications()
                    importer.import_l2_publications(session, l2_pubs)
                    self.mark_completed('nodes_l2publication')
                else:
                    logger.info("L2 Publications already imported (checkpoint)")
                
                # ============================================================
                # PHASE 2: IMPORT RELATIONSHIPS (in dependency order)
                # ============================================================
                
                logger.info("\n=== PHASE 2: IMPORTING RELATIONSHIPS ===")
                
                # 8. CONTAINS: Jurisdiction -> Chunk
                if not self.is_completed('rels_contains_jurisdiction'):
                    rels = self.prepare_contains_jurisdiction(chunks)
                    importer.import_contains_jurisdiction(session, rels)
                    self.mark_completed('rels_contains_jurisdiction')
                else:
                    logger.info("CONTAINS (Jur→Chunk) already imported (checkpoint)")
                
                # 9. CONTAINS: Publication -> Chunk
                if not self.is_completed('rels_contains_publication'):
                    rels = self.prepare_contains_publication(chunks, paper_mapping)
                    importer.import_contains_publication(session, rels)
                    self.mark_completed('rels_contains_publication')
                else:
                    logger.info("CONTAINS (Pub→Chunk) already imported (checkpoint)")
                
                # 10. AUTHORED_BY
                if not self.is_completed('rels_authored_by'):
                    rels = self.prepare_authored_by(scopus_data)
                    importer.import_authored_by(session, rels)
                    self.mark_completed('rels_authored_by')
                else:
                    logger.info("AUTHORED_BY already imported (checkpoint)")
                
                # 11. PUBLISHED_IN
                if not self.is_completed('rels_published_in'):
                    rels = self.prepare_published_in(scopus_data, issn_mapping)
                    importer.import_published_in(session, rels)
                    self.mark_completed('rels_published_in')
                else:
                    logger.info("PUBLISHED_IN already imported (checkpoint)")
                
                # 12. EXTRACTED_FROM
                if not self.is_completed('rels_extracted_from'):
                    rels = self.prepare_extracted_from()
                    importer.import_extracted_from(session, rels)
                    self.mark_completed('rels_extracted_from')
                else:
                    logger.info("EXTRACTED_FROM already imported (checkpoint)")
                
                # 13. RELATION (largest - ~155K)
                if not self.is_completed('rels_relation'):
                    rels = self.prepare_relations()
                    importer.import_relations(session, rels)
                    self.mark_completed('rels_relation')
                else:
                    logger.info("RELATION already imported (checkpoint)")
                
                # 14-20. Enrichment relations
                enrichment_rels = self.prepare_enrichment_relations()
                
                # 14. MATCHED_TO
                if not self.is_completed('rels_matched_to'):
                    importer.import_matched_to(session, enrichment_rels['matched_to'])
                    self.mark_completed('rels_matched_to')
                else:
                    logger.info("MATCHED_TO already imported (checkpoint)")
                
                # 15. CITES (L1 -> L2)
                if not self.is_completed('rels_cites'):
                    importer.import_cites(session, enrichment_rels['cites'])
                    self.mark_completed('rels_cites')
                else:
                    logger.info("CITES already imported (checkpoint)")
                
                # 16. SAME_AS: Entity -> Jurisdiction
                if not self.is_completed('rels_same_as_jurisdiction'):
                    importer.import_same_as_jurisdiction(session, enrichment_rels['same_as_jurisdiction'])
                    self.mark_completed('rels_same_as_jurisdiction')
                else:
                    logger.info("SAME_AS (Jurisdiction) already imported (checkpoint)")
                
                # 17. SAME_AS: Entity -> Author
                if not self.is_completed('rels_same_as_author'):
                    importer.import_same_as_author(session, enrichment_rels['same_as_author'])
                    self.mark_completed('rels_same_as_author')
                else:
                    logger.info("SAME_AS (Author) already imported (checkpoint)")
                
                # 18. SAME_AS: Entity -> Journal
                if not self.is_completed('rels_same_as_journal'):
                    importer.import_same_as_journal(session, enrichment_rels['same_as_journal'])
                    self.mark_completed('rels_same_as_journal')
                else:
                    logger.info("SAME_AS (Journal) already imported (checkpoint)")
                
                # 19. AUTHORED: Entity -> L2Publication
                if not self.is_completed('rels_authored'):
                    importer.import_authored(session, enrichment_rels['authored'])
                    self.mark_completed('rels_authored')
                else:
                    logger.info("AUTHORED already imported (checkpoint)")
                
                # 20. PUBLISHED_IN from enrichment (using publication IDs)
                if not self.is_completed('rels_published_in_enrichment'):
                    importer.import_published_in_enrichment(session, enrichment_rels['published_in'])
                    self.mark_completed('rels_published_in_enrichment')
                else:
                    logger.info("PUBLISHED_IN (enrichment) already imported (checkpoint)")
                
                # 21. CONTAINS from enrichment (Publication -> Chunk)
                if not self.is_completed('rels_contains_enrichment'):
                    importer.import_contains_enrichment(session, enrichment_rels['contains_publication'])
                    self.mark_completed('rels_contains_enrichment')
                else:
                    logger.info("CONTAINS (enrichment) already imported (checkpoint)")
                
                logger.info("\n=== IMPORT COMPLETE ===")
        
        finally:
            importer.close()


def main():
    """Main entry point for Neo4j import."""
    parser = argparse.ArgumentParser(
        description='Import GraphRAG data into Neo4j with checkpointing'
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
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=PROJECT_ROOT / 'data',
        help='Data directory root (default: PROJECT_ROOT/data)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=Path,
        default=PROJECT_ROOT / 'checkpoints',
        help='Checkpoint directory (default: PROJECT_ROOT/checkpoints)'
    )
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear database before import (requires confirmation)'
    )
    parser.add_argument(
        '--force-restart',
        action='store_true',
        help='Ignore existing checkpoint and restart from beginning'
    )
    
    args = parser.parse_args()
    
    # Validate required args
    if not args.uri or not args.password:
        parser.error("--uri and --password required (or set NEO4J_URI and NEO4J_PASSWORD env vars)")
    
    # Confirm database clearing
    if args.clear:
        response = input("⚠️  This will DELETE all data in the Neo4j database. Continue? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Import cancelled")
            return
    
    # Run import
    processor = Neo4jImportProcessor(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        force_restart=args.force_restart
    )
    
    processor.run_import(
        uri=args.uri,
        user=args.user,
        password=args.password,
        clear_db=args.clear
    )


if __name__ == '__main__':
    main()