"""
Relation extraction test suite.

Simplified test script for relation extraction. Edit the TEST_ENTITIES
configuration section to select entities for testing, then run the script
to validate relation extraction before processing the full dataset.

Run: python tests/processing/test_relation_extraction.py
"""

import sys
import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.processing.relations.relation_extractor import RAKGRelationExtractor
from src.utils.logger import setup_logging
from dotenv import load_dotenv
import os

load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG - EDIT THIS!
# ============================================================================

# Which entities to test? (name, type)
TEST_ENTITIES = [
    ("AI System", "Technology"),
    ("A Bottacin-Busolin", "Author"), 
    ("3DAthnicb CBOeChACOpeaYBaHNN", "Regulatory Body"),
    ("Artificial Intelligence and Corporate Responsibility Under International Human Rights Law","Citation")
]

# Parameters
NUM_CHUNKS = 6          # Chunks per stage (6 + optional 6 second round)
MMR_LAMBDA = 0.65       # Relevance weight (0.5=balanced, 1.0=pure relevance)
SAVE_PROMPTS = True     # Save prompts to logs/phase1d_prompts/?

# Files (don't usually need to change)
ENTITIES_FILE = "data/interim/entities/normalized_entities.json"
CHUNKS_FILE = "data/interim/chunks/chunks_embedded.json"
COOCCURRENCE_FILE = "data/interim/entities/cooccurrence_semantic.json"

# ============================================================================
# LOAD DATA
# ============================================================================

logger.info(f"Loading entities from {ENTITIES_FILE}")
with open(ENTITIES_FILE) as f:
    data = json.load(f)
    entities = list(data.values()) if isinstance(data, dict) else data

logger.info(f"Loading chunks from {CHUNKS_FILE}")
with open(CHUNKS_FILE) as f:
    data = json.load(f)
    chunks = list(data.values()) if isinstance(data, dict) else data

logger.info(f"✓ Loaded {len(entities)} entities, {len(chunks)} chunks\n")

# ============================================================================
# FIND TEST ENTITIES
# ============================================================================

entity_map = {(e['name'], e['type']): e for e in entities}
test_entities = []

for name, etype in TEST_ENTITIES:
    if (name, etype) in entity_map:
        entity = entity_map[(name, etype)]
        test_entities.append(entity)
        logger.info(f"✓ {name} [{etype}] - {len(entity.get('chunk_ids', []))} chunks")
    else:
        logger.warning(f"✗ Not found: {name} [{etype}]")

logger.info(f"\nTesting {len(test_entities)} entities\n")

# ============================================================================
# INITIALIZE EXTRACTOR
# ============================================================================

extractor = RAKGRelationExtractor(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    api_key=os.getenv('TOGETHER_API_KEY'),
    semantic_threshold=0.85,
    mmr_lambda=MMR_LAMBDA,
    num_chunks=NUM_CHUNKS,
    entity_cooccurrence_file=COOCCURRENCE_FILE,
    normalized_entities_file=ENTITIES_FILE
)

# ============================================================================
# RUN EXTRACTION
# ============================================================================

logger.info("=" * 80)
logger.info("EXTRACTING RELATIONS")
logger.info("=" * 80)

all_relations = []

for i, entity in enumerate(test_entities, 1):
    logger.info(f"\nEntity {i}/{len(test_entities)}: {entity['name']} [{entity['type']}]")
    
    try:
        relations = extractor.extract_relations_for_entity(
            entity, 
            chunks,
            save_prompt=SAVE_PROMPTS
        )
        
        logger.info(f"  ✓ Extracted {len(relations)} relations")
        all_relations.extend(relations)
        
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")

# ============================================================================
# RESULTS
# ============================================================================

logger.info("\n" + "=" * 80)
logger.info("RESULTS")
logger.info("=" * 80)
logger.info(f"Entities: {len(test_entities)}")
logger.info(f"Relations: {len(all_relations)}")
logger.info(f"Average: {len(all_relations)/len(test_entities):.1f} per entity" if test_entities else "N/A")

if all_relations:
    logger.info("\nSample (first 5):")
    for r in all_relations[:5]:
        logger.info(f"  ({r['subject']}, {r['predicate']}, {r['object']})")

logger.info("=" * 80)