# -*- coding: utf-8 -*-
"""
Phase 2A Enrichment Pipeline Orchestrator.

Coordinates L1 metadata extraction, citation matching, jurisdiction linking,
and relation generation for the AI governance GraphRAG pipeline.

Author: Pau Barba i Colomer
Created: 2025-12-21
Modified: 2025-12-21

References:
    - See ARCHITECTURE.md § 3.2.1 for Phase 2A context
    - See PHASE_2A_DESIGN.md for matching pipeline
"""

# Standard library
import json
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from collections import defaultdict

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
from tqdm import tqdm

# Foundation imports
from src.utils.io import load_jsonl, save_jsonl

# Default config (can be overridden)
DEFAULT_ENRICHMENT_CONFIG = {
    # Input paths (relative to PROJECT_ROOT/data)
    'scopus_csv_path': 'data/raw/academic/scopus_2023/scopus_export_2023_raw.csv',
    'entities_path': 'data/processed/entities/entities_metadata.jsonl',
    'relations_path': 'data/processed/relations/relations_discusses.jsonl',
    'chunks_path': 'data/interim/chunks/chunks_embedded.jsonl',
    'scraping_summary_path': 'data/raw/dlapiper/scraping_summary.json',
    
    # Output directory
    'output_dir': 'data/processed/enrichment',
    
    # Matching thresholds
    'citation_match_threshold': 0.65,
    'title_similarity_threshold': 0.75,
    'l1_overlap_threshold': 0.90,
}

# Try to import from config, fall back to defaults
try:
    from config.extraction_config import ENRICHMENT_CONFIG as _cfg
    # Merge with defaults (config overrides defaults)
    ENRICHMENT_CONFIG = {**DEFAULT_ENRICHMENT_CONFIG, **_cfg}
except ImportError:
    ENRICHMENT_CONFIG = DEFAULT_ENRICHMENT_CONFIG

# Local imports
from src.enrichment.scopus_parser import ScopusParser, ReferenceParser
from src.enrichment.citation_matcher import CitationEntityIdentifier, CitationMatcher, MetadataMatcher
from src.enrichment.jurisdiction_matcher import JurisdictionMatcher
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# PIPELINE ORCHESTRATION
# =============================================================================

class EnrichmentProcessor:
    """
    Phase 2A enrichment pipeline orchestrator.
    
    Coordinates:
    1. L1 metadata extraction (Scopus CSV → Publications, Authors, Journals)
    2. Citation matching (Entity → L1/L2 Publications)
    3. Jurisdiction linking (Entity → Jurisdiction)
    4. Relation generation (MATCHED_TO, CITES, PUBLISHED_IN, CONTAINS, SAME_AS)
    
    Outputs JSON files for Phase 2B Neo4j construction.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize processor with configuration.
        
        Args:
            config: Optional config override (defaults to ENRICHMENT_CONFIG)
        """
        self.config = config or ENRICHMENT_CONFIG
        
        # Resolve paths relative to PROJECT_ROOT
        def resolve_path(p):
            path = Path(p)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            return path
        
        # Set paths from config
        self.scopus_csv = resolve_path(self.config['scopus_csv_path'])
        self.entities_file = resolve_path(self.config['entities_path'])
        self.relations_file = resolve_path(self.config['relations_path'])
        self.chunks_file = resolve_path(self.config['chunks_path'])
        self.scraping_summary = resolve_path(self.config['scraping_summary_path'])
        
        # Output directories
        self.output_dir = resolve_path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.l1_publications = []
        self.l2_publications = []
        self.authors = []
        self.journals = []
        self.relations = []
        self.quality_report = {}
    
    def run(self):
        """Execute complete enrichment pipeline."""
        logger.info("=" * 70)
        logger.info("PHASE 2A: SCOPUS ENRICHMENT & CITATION MATCHING")
        logger.info("=" * 70)
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        
        # Load input data
        entities, relations, chunks = self._load_input_data()
        
        # Step 1: Parse Scopus CSV
        self.l1_publications, self.authors, self.journals = self._parse_scopus_csv()
        
        # Step 2: Parse references
        references_lookup = self._parse_references()
        
        # Step 3: Identify citation entities
        citation_entities = self._identify_citation_entities(entities, relations)
        
        # Step 4: Build chunk-to-L1 mapping
        chunk_to_l1 = self._build_chunk_mapping(chunks)
        
        # Step 5: Match entities to references
        matched_entities = self._match_entities(
            citation_entities, references_lookup, chunk_to_l1
        )
        
        # Step 6: Match jurisdiction entities
        jurisdiction_links = self._match_jurisdictions(entities)
        
        # Step 6b: Match metadata entities (Author/Journal/Document)
        metadata_matches = self._match_metadata_entities(entities)
        
        # Step 7: Generate relations
        self._generate_relations(matched_entities, chunks, jurisdiction_links, metadata_matches)
        
        # Step 8: Generate quality report
        self._generate_quality_report(citation_entities, matched_entities, jurisdiction_links, metadata_matches)
        
        # Step 9: Save outputs
        self._save_outputs()
        
        logger.info("✓ Pipeline complete!")
    
    def _load_input_data(self) -> tuple:
        """Load entities, relations, and chunks."""
        logger.info("=" * 70)
        logger.info("LOADING INPUT DATA")
        logger.info("=" * 70)
        
        # Load entities (JSONL format for v1.1)
        if self.entities_file.suffix == '.jsonl':
            entities = list(load_jsonl(self.entities_file))
        else:
            with open(self.entities_file, 'r', encoding='utf-8') as f:
                entities = json.load(f)
        logger.info(f"✓ Loaded {len(entities)} entities")
        
        # Load relations (JSONL format for v1.1)
        if self.relations_file.suffix == '.jsonl':
            relations = list(load_jsonl(self.relations_file))
        else:
            with open(self.relations_file, 'r', encoding='utf-8') as f:
                relations = json.load(f)
        logger.info(f"✓ Loaded {len(relations)} relations")
        
        # Load chunks (JSONL format for v1.1)
        if self.chunks_file.suffix == '.jsonl':
            chunks = list(load_jsonl(self.chunks_file))
        else:
            with open(self.chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            # Handle dict structure: {chunk_id: chunk_object}
            if isinstance(chunks_data, dict):
                chunks = list(chunks_data.values())
            else:
                chunks = chunks_data
        logger.info(f"✓ Loaded {len(chunks)} chunks")
        
        return entities, relations, chunks
    
    def _parse_scopus_csv(self) -> tuple:
        """Parse Scopus CSV to extract L1 metadata."""
        logger.info("=" * 70)
        logger.info("STEP 1: Parsing Scopus CSV")
        logger.info("=" * 70)
        
        parser = ScopusParser(self.scopus_csv)
        l1_pubs, authors, journals = parser.parse_publications()
        
        logger.info(f"✓ Loaded {len(l1_pubs)} publications")
        logger.info(f"✓ Extracted {len(authors)} unique authors")
        logger.info(f"✓ Extracted {len(journals)} unique journals")
        
        return l1_pubs, authors, journals
    
    def _parse_references(self) -> Dict:
        """Parse References field for all L1 publications."""
        logger.info("=" * 70)
        logger.info("STEP 2: Parsing References")
        logger.info("=" * 70)
        
        parser = ReferenceParser()
        references_lookup = parser.parse_all_references(self.l1_publications)
        
        total_refs = sum(len(refs) for refs in references_lookup.values())
        logger.info(f"✓ Parsed {total_refs} references from {len(references_lookup)} publications")
        
        return references_lookup
    
    def _identify_citation_entities(
        self,
        entities: List[Dict],
        relations: List[Dict]
    ) -> Dict:
        """Identify which entities are academic citations with discusses relations."""
        logger.info("=" * 70)
        logger.info("STEP 3: Identifying Citation Entities")
        logger.info("=" * 70)
        
        identifier = CitationEntityIdentifier()
        citation_entities = identifier.identify(entities, relations)
        
        # Count by type
        type_counts = {}
        for ent_data in citation_entities.values():
            t = ent_data.get('type', 'unknown')
            type_counts[t] = type_counts.get(t, 0) + 1
        
        logger.info(f"✓ Identified {len(citation_entities)} citation entities:")
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  - {t}: {c}")
        
        return citation_entities
    
    def _build_chunk_mapping(self, chunks: List[Dict]) -> Dict:
        """Build mapping from chunk_id to scopus_id."""
        logger.info("=" * 70)
        logger.info("STEP 4: Building Chunk→L1 Mapping")
        logger.info("=" * 70)
        
        mapping = {}
        
        for chunk in chunks:
            # Handle both Chunk dataclass and dict
            chunk_id = chunk.get('chunk_id') or chunk.get('chunk_ids', [''])[0]
            metadata = chunk.get('metadata', {})
            
            # Academic chunks use 'eid' or 'scopus_id'
            scopus_id = metadata.get('eid') or metadata.get('scopus_id')
            
            if scopus_id:
                mapping[chunk_id] = scopus_id
        
        logger.info(f"✓ Mapped {len(mapping)} chunks to L1 papers")
        
        return mapping
    
    def _match_entities(
        self,
        citation_entities: Dict,
        references_lookup: Dict,
        chunk_to_l1: Dict
    ) -> List[Dict]:
        """Match citation entities to L1/L2 publications."""
        logger.info("=" * 70)
        logger.info("STEP 5: Matching Citations to References")
        logger.info("=" * 70)
        
        matcher = CitationMatcher(
            references_lookup=references_lookup,
            chunk_to_l1=chunk_to_l1,
            l1_publications=self.l1_publications
        )
        
        matched_entities = []
        stats = defaultdict(int)
        
        for entity_id, entity_data in tqdm(citation_entities.items(), desc="Matching"):
            result = matcher.match_entity_to_reference(entity_id, entity_data)
            
            if result is None:
                # Track why matching failed
                if not entity_data.get('chunk_ids'):
                    stats['no_chunks'] += 1
                elif not any(cid in chunk_to_l1 for cid in entity_data['chunk_ids']):
                    stats['no_l1_mapping'] += 1
                else:
                    stats['no_match'] += 1
                continue
            
            matched_ref, match_result, match_conf, match_method = result
            
            is_l1 = match_result['is_l1']
            
            if is_l1:
                target_id = f"pub_l1_{match_result['target_id']}"
                stats['l1_overlap'] += 1
            else:
                target_id = matcher.get_or_create_l2(matched_ref)
                stats['l2_matched'] += 1
            
            matched_entities.append({
                'entity_id': entity_id,
                'entity_name': entity_data['name'],
                'entity_type': entity_data.get('type', 'unknown'),
                'matched_to': target_id,
                'matched_type': 'L1' if is_l1 else 'L2',
                'source_l1': match_result['source_l1_id'],
                'match_confidence': match_conf,
                'match_method': match_method
            })
        
        self.l2_publications = matcher.get_l2_publications()
        
        logger.info(f"✓ Matching complete:")
        logger.info(f"  - Total citation entities: {len(citation_entities)}")
        logger.info(f"  - Successfully matched: {len(matched_entities)}")
        logger.info(f"  - L1 overlaps detected: {stats['l1_overlap']}")
        logger.info(f"  - L2 publications created: {len(self.l2_publications)}")
        logger.info(f"  Unmatched reasons:")
        logger.info(f"  - No chunk provenance: {stats['no_chunks']}")
        logger.info(f"  - Chunk not from L1 paper: {stats['no_l1_mapping']}")
        logger.info(f"  - No fuzzy match found: {stats['no_match']}")
        
        return matched_entities
    
    def _match_jurisdictions(self, entities: List[Dict]) -> List[Dict]:
        """Match country/region entities to jurisdiction codes."""
        logger.info("=" * 70)
        logger.info("STEP 6: Matching Jurisdiction Entities")
        logger.info("=" * 70)
        
        valid_codes = JurisdictionMatcher.load_valid_codes(self.scraping_summary)
        
        matcher = JurisdictionMatcher(valid_codes)
        jurisdiction_links = matcher.match_entities(entities)
        
        logger.info(f"✓ Matched {len(jurisdiction_links)} entities to jurisdictions")
        
        return jurisdiction_links
    
    def _match_metadata_entities(self, entities: List[Dict]) -> Dict:
        """Match Author/Journal/Document entities to Scopus nodes."""
        logger.info("=" * 70)
        logger.info("STEP 6b: Matching Metadata Entities")
        logger.info("=" * 70)
        
        # Load jurisdiction data for Document matching
        with open(self.scraping_summary, 'r', encoding='utf-8') as f:
            jur_data = json.load(f)
            if isinstance(jur_data, dict) and 'countries' in jur_data:
                jurisdictions = jur_data['countries']
            else:
                jurisdictions = jur_data
        
        # Initialize matcher with target pools
        matcher = MetadataMatcher(
            authors=self.authors,
            journals=self.journals,
            publications=self.l1_publications,
            jurisdictions=jurisdictions,
            threshold=0.85
        )
        
        # Filter to matchable types
        matchable_types = {'Author', 'Journal', 'Document'}
        matchable = [e for e in entities if e.get('type') in matchable_types]
        logger.info(f"Matchable entities: {len(matchable)}")
        
        # Run matching
        results = matcher.match_all(matchable)
        stats = results['stats']
        
        # Log results
        author_rate = stats['author_matched'] / max(1, stats['author_total']) * 100
        journal_rate = stats['journal_matched'] / max(1, stats['journal_total']) * 100
        document_rate = stats['document_matched'] / max(1, stats['document_total']) * 100
        
        logger.info(f"✓ Metadata matching complete:")
        logger.info(f"  - Authors:   {stats['author_matched']}/{stats['author_total']} ({author_rate:.1f}%)")
        logger.info(f"  - Journals:  {stats['journal_matched']}/{stats['journal_total']} ({journal_rate:.1f}%)")
        logger.info(f"  - Documents: {stats['document_matched']}/{stats['document_total']} ({document_rate:.1f}%)")
        
        # Generate SAME_AS relations
        same_as_relations = matcher.generate_same_as_relations(results)
        logger.info(f"  - SAME_AS relations: {len(same_as_relations)}")
        
        return {
            'results': results,
            'relations': same_as_relations,
            'stats': dict(stats)
        }
    
    def _generate_relations(
        self,
        matched_entities: List[Dict],
        chunks: List[Dict],
        jurisdiction_links: List[Dict],
        metadata_matches: Dict = None
    ):
        """Generate enrichment relationships."""
        logger.info("=" * 70)
        logger.info("STEP 7: Generating Enrichment Relations")
        logger.info("=" * 70)
        
        relations = []
        cites_set = set()
        
        # 1. MATCHED_TO: Entity → Publication
        for match in tqdm(matched_entities, desc="MATCHED_TO"):
            relations.append({
                'relation_type': 'MATCHED_TO',
                'source_id': match['entity_id'],
                'source_type': 'Entity',
                'target_id': match['matched_to'],
                'target_type': 'Publication',
                'properties': {
                    'confidence': match['match_confidence'],
                    'method': match['match_method'],
                    'entity_type': match['entity_type']
                }
            })
            
            cites_key = (match['source_l1'], match['matched_to'])
            cites_set.add(cites_key)
        
        # 2. CITES: L1 → L1/L2
        for source_l1, target_pub in tqdm(cites_set, desc="CITES"):
            relations.append({
                'relation_type': 'CITES',
                'source_id': source_l1,
                'source_type': 'Publication',
                'target_id': target_pub,
                'target_type': 'Publication'
            })
        
        # 3. PUBLISHED_IN: L1 → Journal
        for pub in tqdm(self.l1_publications, desc="PUBLISHED_IN"):
            journal_name = pub.get('source_title')
            if journal_name:
                from src.utils.id_generator import generate_journal_id
                journal_id = generate_journal_id(journal_name)
                relations.append({
                    'relation_type': 'PUBLISHED_IN',
                    'source_id': pub['publication_id'],
                    'source_type': 'Publication',
                    'target_id': journal_id,
                    'target_type': 'Journal'
                })
        
        # 4. CONTAINS: L1 → Chunk (for academic chunks)
        for chunk in tqdm(chunks, desc="CONTAINS"):
            chunk_id = chunk.get('chunk_id') or chunk.get('chunk_ids', [''])[0]
            metadata = chunk.get('metadata', {})
            scopus_id = metadata.get('eid') or metadata.get('scopus_id')
            
            if scopus_id:
                from src.utils.id_generator import generate_publication_id
                pub_id = generate_publication_id(scopus_id, layer=1)
                relations.append({
                    'relation_type': 'CONTAINS',
                    'source_id': pub_id,
                    'source_type': 'Publication',
                    'target_id': chunk_id,
                    'target_type': 'Chunk'
                })
        
        # 5. SAME_AS: Entity → Jurisdiction
        for jur_link in tqdm(jurisdiction_links, desc="SAME_AS (Jurisdiction)"):
            relations.append({
                'relation_type': 'SAME_AS',
                'source_id': jur_link['entity_id'],
                'source_type': 'Entity',
                'target_id': f"jur_{jur_link['jurisdiction_code']}",
                'target_type': 'Jurisdiction',
                'properties': {
                    'entity_name': jur_link['entity_name'],
                    'jurisdiction_code': jur_link['jurisdiction_code']
                }
            })
        
        # 6. SAME_AS: Entity → Author/Journal/Publication (from metadata matching)
        if metadata_matches and metadata_matches.get('relations'):
            for rel in tqdm(metadata_matches['relations'], desc="SAME_AS (Metadata)"):
                relations.append({
                    'relation_type': 'SAME_AS',
                    'source_id': rel['subject_id'],
                    'source_type': 'Entity',
                    'target_id': rel['object_id'],
                    'target_type': rel['object_type'],
                    'properties': {
                        'confidence': rel['confidence'],
                        'method': rel['method']
                    }
                })
        
        self.relations = relations
        
        # Log counts
        counts = defaultdict(int)
        for r in relations:
            counts[r['relation_type']] += 1
        
        logger.info(f"✓ Generated {len(relations)} enrichment relations:")
        for rel_type, count in sorted(counts.items()):
            logger.info(f"  - {rel_type}: {count}")
    
    def _generate_quality_report(
        self,
        citation_entities: Dict,
        matched_entities: List[Dict],
        jurisdiction_links: List[Dict],
        metadata_matches: Dict = None
    ):
        """Generate quality metrics."""
        logger.info("=" * 70)
        logger.info("STEP 8: Generating Quality Report")
        logger.info("=" * 70)
        
        type_counts = defaultdict(int)
        for ent_data in citation_entities.values():
            type_counts[ent_data['type']] += 1
        
        method_counts = defaultdict(int)
        for match in matched_entities:
            method_counts[match['match_method']] += 1
        
        l1_matches = len([m for m in matched_entities if m['matched_type'] == 'L1'])
        l2_matches = len([m for m in matched_entities if m['matched_type'] == 'L2'])
        
        match_rate = len(matched_entities) / len(citation_entities) * 100 if citation_entities else 0
        
        self.quality_report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_citation_entities': len(citation_entities),
                'matched_entities': len(matched_entities),
                'match_rate_pct': round(match_rate, 2),
                'l1_matches': l1_matches,
                'l2_matches': l2_matches,
                'unique_l2_publications': len(self.l2_publications),
                'l1_publications': len(self.l1_publications)
            },
            'entity_types': dict(type_counts),
            'match_methods': dict(method_counts),
            'coverage': {
                'l1_overlap_rate_pct': round(l1_matches / len(matched_entities) * 100, 2) if matched_entities else 0,
                'avg_cites_per_l1': round(len(matched_entities) / len(self.l1_publications), 2) if self.l1_publications else 0
            },
            'jurisdiction_linking': {
                'total_linked': len(jurisdiction_links),
                'unique_jurisdictions': len(set(j['jurisdiction_code'] for j in jurisdiction_links))
            }
        }
        
        # Add metadata matching stats
        if metadata_matches and metadata_matches.get('stats'):
            stats = metadata_matches['stats']
            self.quality_report['metadata_matching'] = {
                'author_matched': stats.get('author_matched', 0),
                'author_total': stats.get('author_total', 0),
                'journal_matched': stats.get('journal_matched', 0),
                'journal_total': stats.get('journal_total', 0),
                'document_matched': stats.get('document_matched', 0),
                'document_total': stats.get('document_total', 0),
                'total_same_as_relations': len(metadata_matches.get('relations', []))
            }
        
        logger.info("QUALITY SUMMARY:")
        logger.info(f"  Match rate: {self.quality_report['summary']['match_rate_pct']}%")
        logger.info(f"  L1 overlaps: {self.quality_report['summary']['l1_matches']}")
        logger.info(f"  L2 created: {self.quality_report['summary']['unique_l2_publications']}")
        logger.info(f"  Jurisdiction links: {self.quality_report['jurisdiction_linking']['total_linked']}")
        if metadata_matches:
            logger.info(f"  Metadata SAME_AS: {len(metadata_matches.get('relations', []))}")
    
    def _save_outputs(self):
        """Save all output files."""
        logger.info("=" * 70)
        logger.info("STEP 9: Saving Outputs")
        logger.info("=" * 70)
        
        # Merge L1 and L2 publications
        all_publications = self.l1_publications + self.l2_publications
        
        # Save publications
        pubs_path = self.output_dir / 'publications.json'
        with open(pubs_path, 'w', encoding='utf-8') as f:
            json.dump(all_publications, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved {pubs_path}")
        
        # Save authors
        authors_path = self.output_dir / 'authors.json'
        with open(authors_path, 'w', encoding='utf-8') as f:
            json.dump(self.authors, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved {authors_path}")
        
        # Save journals
        journals_path = self.output_dir / 'journals.json'
        with open(journals_path, 'w', encoding='utf-8') as f:
            json.dump(self.journals, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved {journals_path}")
        
        # Save enrichment relations
        relations_path = self.output_dir / 'enrichment_relations.json'
        with open(relations_path, 'w', encoding='utf-8') as f:
            json.dump(self.relations, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved {relations_path}")
        
        # Save quality report
        report_path = self.output_dir / 'enrichment_quality_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.quality_report, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved {report_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run enrichment pipeline."""
    processor = EnrichmentProcessor()
    processor.run()


if __name__ == "__main__":
    main()