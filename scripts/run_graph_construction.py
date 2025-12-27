# -*- coding: utf-8 -*-
"""
Graph Construction Pipeline Orchestrator

This script orchestrates the complete document-to-knowledge-graph pipeline spanning
Phases 0-2: preprocessing (0A-0B), extraction (1A-1D), and enrichment/storage (2A-2B).
It provides a unified CLI interface for running individual phases or phase ranges,
enabling flexible pipeline execution from raw documents (DLA Piper regulations + Scopus
papers) to a populated Neo4j graph database with FAISS indices.

Phase 0A links MinerU-parsed PDFs to Scopus metadata via DOI/title matching. Phase 0B
cleans text and translates non-English documents. Phase 1A performs semantic chunking,
1B extracts entities via LLM, 1C disambiguates using FAISS+similarity, and 1D extracts
relations. Phase 2A enriches with Scopus author/journal/citation metadata, and 2B
imports to Neo4j and builds FAISS indices for retrieval.

Modes:
    --start-phase    First phase to execute (default: 0A)
    --end-phase      Last phase to execute (default: 2B)
    --list-phases    Display all available phases and exit

Examples:
    # Run full pipeline from documents to graph
    python run_graph_construction.py

    # Run only chunking phase
    python run_graph_construction.py --start-phase 1A --end-phase 1A

    # Resume from disambiguation through storage
    python run_graph_construction.py -s 1C -e 2B

    # Reprocess preprocessing only
    python run_graph_construction.py -s 0B -e 0B

References:
    ARCHITECTURE.md § 3-4 (Pipeline Phases 0-2)
    MinerU: PDF parsing library for academic papers
    Scopus API: Academic metadata enrichment source
    Neo4j: Graph database for storage
    FAISS: Vector index library for fast similarity search
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Phase definitions in execution order
PHASES = [
    ("0A", "Scopus Matching"),
    ("0B", "Preprocessing"),
    ("1A", "Chunking"),
    ("1B", "Entity Extraction"),
    ("1C", "Disambiguation"),
    ("1D", "Relation Extraction"),
    ("2A", "Scopus Enrichment"),
    ("2B", "Storage"),
]

PHASE_ORDER = [p[0] for p in PHASES]
PHASE_NAMES = {p[0]: p[1] for p in PHASES}


def run_phase_0a() -> bool:
    """
    Phase 0A: Scopus Matching
    
    Links MinerU-parsed PDFs to Scopus CSV metadata using DOI and title matching.
    
    Input:  data/raw/scopus/*.csv, data/raw/academic/*/
    Output: data/interim/paper_mapping.json
    
    Results (v1.1):
        - DOI exact: 117
        - Title (high confidence): 21
        - Title (low confidence): 3
        - Abstract fallback: 1
        - Total: 142/148 (95.9%)
    
    Returns True if successful.
    """
    logger.info("=" * 60)
    logger.info("PHASE 0A: Scopus Matching")
    logger.info("=" * 60)
    
    from src.ingestion.paper_to_scopus_metadata_matcher import main as run_matching
    run_matching()
    return True


def run_phase_0b() -> bool:
    """
    Phase 0B: Preprocessing
    
    Cleans text (encoding fixes, HTML/LaTeX removal), detects language,
    and translates non-English documents to English.
    
    Input:  data/raw/dlapiper/*.json, data/raw/academic/*/full.md
    Output: data/interim/preprocessed/documents_cleaned.jsonl
            data/interim/preprocessed/preprocessing_report.json
    
    Results (v1.1):
        - Documents: 190 (48 regulations, 142 papers)
        - Languages: en=184, es=5, de=1
        - Translated: 6
        - Chars: 10,410,535 → 9,927,342
        - Fixes: encoding=9,725,417, html=84,396, latex=7,228
    
    Returns True if successful.
    """
    logger.info("=" * 60)
    logger.info("PHASE 0B: Preprocessing")
    logger.info("=" * 60)
    
    from src.preprocessing.preprocessing_processor import main as run_preprocessing
    run_preprocessing()
    return True


def run_phase_1a() -> bool:
    """
    Phase 1A: Chunking
    
    Splits preprocessed documents into semantic chunks using BGE-small
    for boundary detection. v1.1 fix: threshold now actually works
    (v1.0 always fell back to min_sentences), producing longer chunks.
    
    Input:  data/interim/preprocessed/documents_cleaned.jsonl
    Output: data/processed/chunks/chunks_embedded.jsonl
    
    Results (v1.1):
        - Chunks: 3,914 (vs 25,131 in v1.0 - longer chunks, not more filtering)
        - Avg tokens/chunk: 334.1
        - Discarded: 536 (12.0%)
        - Duplicates merged: 452 (10.4%)
    
    Returns True if successful.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1A: Chunking")
    logger.info("=" * 60)
    
    from src.processing.chunks.chunk_processor import main as run_chunking
    run_chunking()
    return True


def run_phase_1b() -> bool:
    """
    Phase 1B: Entity Extraction
    
    Extracts pre-entities from chunks using Qwen-72B LLM.
    Each pre-entity has: name, type, description, chunk_id.
    
    Input:  data/processed/chunks/chunks_embedded.jsonl
    Output: data/interim/entities/pre_entities.jsonl
    
    Returns True if successful.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1B: Entity Extraction")
    logger.info("=" * 60)
    
    # TODO: Import and run entity_processor
    # from src.processing.entities.entity_processor import EntityProcessor
    # processor = EntityProcessor()
    # return processor.run()
    
    logger.warning("Phase 1B not yet implemented in runner")
    return True


def run_phase_1c() -> bool:
    """
    Phase 1C: Entity Disambiguation
    
    Merges duplicate pre-entities using FAISS blocking + tiered similarity thresholds.
    Assigns deterministic canonical IDs.
    
    Input:  data/interim/entities/pre_entities.jsonl
    Output: data/processed/entities/entities_canonical.jsonl
    
    Returns True if successful.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1C: Disambiguation")
    logger.info("=" * 60)
    
    # TODO: Import and run disambiguation
    # from src.processing.entities.disambiguation import DisambiguationProcessor
    # processor = DisambiguationProcessor()
    # return processor.run()
    
    logger.warning("Phase 1C not yet implemented in runner")
    return True


def run_phase_1d() -> bool:
    """
    Phase 1D: Relation Extraction
    
    Extracts relations between entities using Mistral-7B LLM.
    Two-track strategy: semantic (OpenIE) and academic (discusses only).
    
    Input:  data/processed/entities/entities_canonical.jsonl
            data/processed/chunks/chunks_embedded.jsonl
    Output: data/processed/relations/relations.jsonl
    
    Returns True if successful.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1D: Relation Extraction")
    logger.info("=" * 60)
    
    # TODO: Import and run relation extraction
    # from src.processing.relations.run_relation_extraction import RelationProcessor
    # processor = RelationProcessor()
    # return processor.run()
    
    logger.warning("Phase 1D not yet implemented in runner")
    return True


def run_phase_2a() -> bool:
    """
    Phase 2A: Scopus Enrichment
    
    Adds structured metadata from Scopus:
    - Author nodes with AUTHORED_BY relations
    - Journal nodes with PUBLISHED_IN relations
    - L2 Publication nodes for citations with CITES relations
    - Jurisdiction SAME_AS links
    
    Input:  data/processed/entities/entities_canonical.jsonl
            data/raw/scopus/*.csv
    Output: data/processed/entities/entities_enriched.jsonl
            data/processed/relations/relations_enriched.jsonl
    
    Returns True if successful.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2A: Scopus Enrichment")
    logger.info("=" * 60)
    
    # TODO: Import and run enrichment
    # from src.enrichment.enrichment_processor import EnrichmentProcessor
    # processor = EnrichmentProcessor()
    # return processor.run()
    
    logger.warning("Phase 2A not yet implemented in runner")
    return True


def run_phase_2b() -> bool:
    """
    Phase 2B: Storage
    
    Imports graph to Neo4j and builds FAISS indices.
    - Neo4j: nodes (entities, chunks) + edges (relations)
    - FAISS: HNSW indices for entity and chunk embeddings
    
    Input:  data/processed/entities/entities_enriched.jsonl
            data/processed/relations/relations_enriched.jsonl
            data/processed/chunks/chunks_embedded.jsonl
    Output: Neo4j database populated
            data/processed/faiss/entities.faiss
            data/processed/faiss/chunks.faiss
    
    Returns True if successful.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2B: Storage")
    logger.info("=" * 60)
    
    # TODO: Import and run storage
    # from src.graph.neo4j_import_processor import Neo4jImporter
    # from src.graph.faiss_builder import FaissBuilder
    # Neo4jImporter().run()
    # FaissBuilder().run()
    
    logger.warning("Phase 2B not yet implemented in runner")
    return True


# Phase runner dispatch
PHASE_RUNNERS = {
    "0A": run_phase_0a,
    "0B": run_phase_0b,
    "1A": run_phase_1a,
    "1B": run_phase_1b,
    "1C": run_phase_1c,
    "1D": run_phase_1d,
    "2A": run_phase_2a,
    "2B": run_phase_2b,
}


def get_phases_to_run(start: str, end: str) -> list[str]:
    """Get list of phases between start and end (inclusive)."""
    try:
        start_idx = PHASE_ORDER.index(start)
        end_idx = PHASE_ORDER.index(end)
    except ValueError as e:
        raise ValueError(f"Invalid phase: {e}. Valid phases: {PHASE_ORDER}")
    
    if start_idx > end_idx:
        raise ValueError(f"Start phase {start} comes after end phase {end}")
    
    return PHASE_ORDER[start_idx:end_idx + 1]


def run_pipeline(start_phase: str = "0A", end_phase: str = "2B") -> bool:
    """
    Run pipeline from start_phase to end_phase (inclusive).
    
    Args:
        start_phase: First phase to run (default: 0A)
        end_phase: Last phase to run (default: 2B)
    
    Returns:
        True if all phases completed successfully
    """
    phases = get_phases_to_run(start_phase, end_phase)
    
    logger.info("=" * 60)
    logger.info("GRAPH CONSTRUCTION PIPELINE")
    logger.info(f"Phases: {start_phase} → {end_phase}")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    for phase in phases:
        phase_name = PHASE_NAMES[phase]
        logger.info(f"\n>>> Starting {phase}: {phase_name}")
        
        try:
            success = PHASE_RUNNERS[phase]()
            if not success:
                logger.error(f"Phase {phase} failed")
                return False
            logger.info(f"<<< Completed {phase}: {phase_name}")
        except Exception as e:
            logger.exception(f"Phase {phase} raised exception: {e}")
            return False
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Finished: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run graph construction pipeline (Phases 0-2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  0A  Scopus Matching      Link papers to Scopus metadata
  0B  Preprocessing        Clean text, translate non-English
  1A  Chunking             Split into semantic chunks + embed
  1B  Entity Extraction    Extract pre-entities (LLM)
  1C  Disambiguation       Merge duplicates, assign IDs
  1D  Relation Extraction  Extract relations (LLM)
  2A  Scopus Enrichment    Add authors, journals, citations
  2B  Storage              Neo4j + FAISS import

Examples:
  python run_graph_construction.py                    # Full pipeline
  python run_graph_construction.py --start-phase 1A  # From chunking
  python run_graph_construction.py --end-phase 1D    # Through relations
  python run_graph_construction.py -s 0B -e 0B       # Just preprocessing
        """
    )
    
    parser.add_argument(
        "-s", "--start-phase",
        default="0A",
        choices=PHASE_ORDER,
        help="First phase to run (default: 0A)"
    )
    
    parser.add_argument(
        "-e", "--end-phase", 
        default="2B",
        choices=PHASE_ORDER,
        help="Last phase to run (default: 2B)"
    )
    
    parser.add_argument(
        "--list-phases",
        action="store_true",
        help="List all phases and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_phases:
        print("\nAvailable phases:")
        for code, name in PHASES:
            print(f"  {code}  {name}")
        sys.exit(0)
    
    success = run_pipeline(args.start_phase, args.end_phase)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()