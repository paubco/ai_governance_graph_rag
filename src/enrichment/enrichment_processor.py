# -*- coding: utf-8 -*-
"""
Pipeline orchestrator for Phase 2A Scopus enrichment.

Coordinates L1 metadata extraction, citation matching, jurisdiction linking, and
relation generation for the AI governance GraphRAG pipeline. Outputs JSON files
for Phase 2B Neo4j construction.

Example:
    python src/enrichment/enrichment_processor.py
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

# Local
from src.enrichment.scopus_enricher import (
    ScopusParser,
    ReferenceParser,
    CitationEntityIdentifier,
    CitationMatcher
)
from src.enrichment.jurisdiction_matcher import JurisdictionMatcher


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Input paths
SCOPUS_CSV = Path("data/raw/academic/scopus_2023/scopus_export_2023_raw.csv")
NORMALIZED_ENTITIES = Path("data/interim/entities/normalized_entities_with_ids.json")
RELATIONS_FILE = Path("data/interim/relations/relations_normalized.json")
CHUNKS_FILE = Path("data/interim/chunks/chunks_text.json")
SCRAPING_SUMMARY = Path("data/raw/dlapiper/scraping_summary.json")

# Output directories (organized by node type)
ENTITIES_DIR = Path("data/processed/entities")
RELATIONS_DIR = Path("data/processed/relations")
REPORTS_DIR = Path("data/processed/reports")

ENTITIES_DIR.mkdir(parents=True, exist_ok=True)
RELATIONS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# PIPELINE ORCHESTRATION
# ==============================================================================

class EnrichmentProcessor:
    """
    Phase 2A enrichment pipeline orchestrator.
    
    Coordinates:
    1. L1 metadata extraction (Scopus CSV → Publications, Authors, Journals)
    2. Citation matching (Entity → L1/L2 Publications)
    3. Jurisdiction linking (Entity → Jurisdiction)
    4. Relation generation (5 types: MATCHED_TO, CITES, PUBLISHED_IN, CONTAINS, SAME_AS)
    
    Outputs JSON files for Phase 2B Neo4j construction.
    """
    
    def __init__(self):
        """Initialize processor with empty state."""
        self.l1_publications = []
        self.l2_publications = []
        self.authors = []
        self.journals = []
        self.relations = []
        self.quality_stats = {}
    
    def run(self):
        """Execute complete enrichment pipeline."""
        print("\n" + "="*70)
        print("PHASE 2A: SCOPUS ENRICHMENT & CITATION MATCHING")
        print("="*70)
        print(f"\nTimestamp: {datetime.now().isoformat()}")
        
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
        
        # Step 7: Generate relations
        self._generate_relations(matched_entities, chunks, jurisdiction_links)
        
        # Step 8: Generate quality report
        self._generate_quality_report(citation_entities, matched_entities, jurisdiction_links)
        
        # Step 9: Save outputs
        self._save_outputs()
        
        print("\n✓ Pipeline complete!")
    
    def _load_input_data(self) -> tuple:
        """Load entities, relations, and chunks."""
        print(f"\n{'='*70}")
        print("LOADING INPUT DATA")
        print(f"{'='*70}")
        
        with open(NORMALIZED_ENTITIES, 'r', encoding='utf-8') as f:
            entities = json.load(f)
        print(f"✓ Loaded {len(entities)} entities")
        
        with open(RELATIONS_FILE, 'r', encoding='utf-8') as f:
            relations = json.load(f)
        print(f"✓ Loaded {len(relations)} relations")
        
        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            chunks_dict = json.load(f)
        
        # Convert dict to list
        chunks = list(chunks_dict.values())
        print(f"✓ Loaded {len(chunks)} chunks")
        
        return entities, relations, chunks
    
    def _parse_scopus_csv(self) -> tuple:
        """Parse Scopus CSV to extract L1 metadata."""
        print(f"\n{'='*70}")
        print("STEP 1: Parsing Scopus CSV")
        print(f"{'='*70}")
        
        parser = ScopusParser(SCOPUS_CSV)
        l1_pubs, authors, journals = parser.parse_publications()
        
        print(f"✓ Loaded {len(l1_pubs)} publications")
        print(f"✓ Extracted {len(authors)} unique authors")
        print(f"✓ Extracted {len(journals)} unique journals")
        
        return l1_pubs, authors, journals
    
    def _parse_references(self) -> Dict:
        """Parse References field for all L1 publications."""
        print(f"\n{'='*70}")
        print("STEP 2: Parsing References")
        print(f"{'='*70}")
        
        parser = ReferenceParser()
        references_lookup = parser.parse_all_references(self.l1_publications)
        
        total_refs = sum(len(refs) for refs in references_lookup.values())
        print(f"✓ Parsed {total_refs} references from {len(references_lookup)} publications")
        
        return references_lookup
    
    def _identify_citation_entities(
        self,
        entities: List[Dict],
        relations: List[Dict]
    ) -> Dict:
        """Identify which entities are academic citations with discusses relations."""
        print(f"\n{'='*70}")
        print("STEP 3: Identifying Citation Entities")
        print(f"{'='*70}")
        
        identifier = CitationEntityIdentifier()
        citation_entities = identifier.identify(entities, relations)
        
        # Count by type
        type_counts = {}
        for ent_data in citation_entities.values():
            t = ent_data.get('type', 'unknown')
            type_counts[t] = type_counts.get(t, 0) + 1
        
        print(f"\n✓ Identified {len(citation_entities)} citation entities with discusses relations:")
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  - {t}: {c}")
        
        return citation_entities
    
    def _build_chunk_mapping(self, chunks: List[Dict]) -> Dict:
        """Build mapping from chunk_id to scopus_id."""
        print(f"\n{'='*70}")
        print("STEP 4: Building Chunk→L1 Mapping")
        print(f"{'='*70}")
        
        mapping = {}
        
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            metadata = chunk.get('metadata', {})
            
            # Academic chunks use 'eid', regulations use 'country_code'
            scopus_id = metadata.get('eid') or metadata.get('scopus_id')
            
            if scopus_id:
                mapping[chunk_id] = scopus_id
        
        print(f"✓ Mapped {len(mapping)} chunks to L1 papers")
        
        return mapping
    
    def _match_entities(
        self,
        citation_entities: Dict,
        references_lookup: Dict,
        chunk_to_l1: Dict
    ) -> List[Dict]:
        """Match citation entities to L1/L2 publications."""
        print(f"\n{'='*70}")
        print("STEP 5: Matching Citations to References")
        print(f"{'='*70}")
        
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
        
        print(f"\n✓ Matching complete:")
        print(f"  - Total citation entities: {len(citation_entities)}")
        print(f"  - Successfully matched: {len(matched_entities)}")
        print(f"  - L1 overlaps detected: {stats['l1_overlap']}")
        print(f"  - L2 publications created: {len(self.l2_publications)}")
        print(f"\n  Unmatched reasons:")
        print(f"  - No chunk provenance: {stats['no_chunks']}")
        print(f"  - Chunk not from L1 paper: {stats['no_l1_mapping']}")
        print(f"  - No fuzzy match found: {stats['no_match']}")
        
        self.quality_stats.update(stats)
        
        return matched_entities
    
    def _match_jurisdictions(self, entities: List[Dict]) -> List[Dict]:
        """Match country/region entities to jurisdiction codes."""
        print(f"\n{'='*70}")
        print("STEP 6: Matching Jurisdiction Entities")
        print(f"{'='*70}")
        
        valid_codes = JurisdictionMatcher.load_valid_codes(SCRAPING_SUMMARY)
        print(f"✓ Loaded {len(valid_codes)} valid jurisdiction codes")
        
        matcher = JurisdictionMatcher(valid_codes)
        jurisdiction_links = matcher.match_entities(entities)
        
        print(f"\n✓ Matched {len(jurisdiction_links)} entities to jurisdictions")
        
        return jurisdiction_links
    
    def _generate_relations(
        self,
        matched_entities: List[Dict],
        chunks: List[Dict],
        jurisdiction_links: List[Dict]
    ):
        """Generate enrichment relationships."""
        print(f"\n{'='*70}")
        print("STEP 7: Generating Enrichment Relations")
        print(f"{'='*70}")
        
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
            journal_id = pub.get('journal_id')
            if journal_id:
                relations.append({
                    'relation_type': 'PUBLISHED_IN',
                    'source_id': f"pub_l1_{pub['scopus_id']}",
                    'source_type': 'Publication',
                    'target_id': journal_id,
                    'target_type': 'Journal'
                })
        
        # 4. CONTAINS: L1 → Chunk
        for chunk in tqdm(chunks, desc="CONTAINS"):
            metadata = chunk.get('metadata', {})
            scopus_id = metadata.get('scopus_id')
            
            if scopus_id:
                relations.append({
                    'relation_type': 'CONTAINS',
                    'source_id': f"pub_l1_{scopus_id}",
                    'source_type': 'Publication',
                    'target_id': chunk['chunk_id'],
                    'target_type': 'Chunk'
                })
        
        # 5. SAME_AS: Entity → Jurisdiction
        for jur_link in tqdm(jurisdiction_links, desc="SAME_AS"):
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
        
        self.relations = relations
        
        print(f"✓ Generated {len(relations)} enrichment relations:")
        print(f"  - MATCHED_TO: {len([r for r in relations if r['relation_type'] == 'MATCHED_TO'])}")
        print(f"  - CITES: {len([r for r in relations if r['relation_type'] == 'CITES'])}")
        print(f"  - PUBLISHED_IN: {len([r for r in relations if r['relation_type'] == 'PUBLISHED_IN'])}")
        print(f"  - CONTAINS: {len([r for r in relations if r['relation_type'] == 'CONTAINS'])}")
        print(f"  - SAME_AS: {len([r for r in relations if r['relation_type'] == 'SAME_AS'])}")
    
    def _generate_quality_report(
        self,
        citation_entities: Dict,
        matched_entities: List[Dict],
        jurisdiction_links: List[Dict]
    ):
        """Generate quality metrics."""
        print(f"\n{'='*70}")
        print("STEP 8: Generating Quality Report")
        print(f"{'='*70}")
        
        type_counts = defaultdict(int)
        for ent_data in citation_entities.values():
            type_counts[ent_data['type']] += 1
        
        method_counts = defaultdict(int)
        for match in matched_entities:
            method_counts[match['match_method']] += 1
        
        l1_matches = len([m for m in matched_entities if m['matched_type'] == 'L1'])
        l2_matches = len([m for m in matched_entities if m['matched_type'] == 'L2'])
        
        match_rate = len(matched_entities) / len(citation_entities) * 100 if citation_entities else 0
        
        report = {
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
        
        self.quality_report = report
        
        print("\nQUALITY SUMMARY:")
        print(f"  Match rate: {report['summary']['match_rate_pct']}%")
        print(f"  L1 overlaps: {report['summary']['l1_matches']}")
        print(f"  L2 created: {report['summary']['unique_l2_publications']}")
        print(f"  Jurisdiction links: {report['jurisdiction_linking']['total_linked']}")
    
    def _save_outputs(self):
        """Save all output files organized by node type."""
        print(f"\n{'='*70}")
        print("STEP 9: Saving Outputs")
        print(f"{'='*70}")
        
        # Merge L1 and L2 publications
        all_publications = self.l1_publications + self.l2_publications
        
        # Entity files
        entity_files = {
            'publications.json': all_publications,
            'authors.json': self.authors,
            'journals.json': self.journals
        }
        
        for filename, data in entity_files.items():
            output_path = ENTITIES_DIR / filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✓ Saved {output_path}")
        
        # Relation files
        relation_files = {
            'enrichment_relations.json': self.relations
        }
        
        for filename, data in relation_files.items():
            output_path = RELATIONS_DIR / filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✓ Saved {output_path}")
        
        # Report files
        report_files = {
            'enrichment_quality_report.json': self.quality_report
        }
        
        for filename, data in report_files.items():
            output_path = REPORTS_DIR / filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✓ Saved {output_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Run enrichment pipeline."""
    processor = EnrichmentProcessor()
    processor.run()


if __name__ == "__main__":
    main()