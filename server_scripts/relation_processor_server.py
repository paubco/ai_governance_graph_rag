"""
Phase 1D: Relation Processor (Server Script)
Location: server_scripts/relation_processor_server.py
Parallel processing with Together.ai API (reuses Phase 1B patterns)

Usage:
    python server_scripts/relation_processor_server.py --workers 5 --test --test-size 10
    python server_scripts/relation_processor_server.py --workers 5  # Full run
"""

import json, os, sys, time, logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from src.phase1_graph_construction.relation_extractor import RAKGRelationExtractor

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

class RelationProcessor:
    def __init__(self, entities_path, chunks_path, output_path, 
                 num_workers=5, checkpoint_interval=100):
        self.entities_path = Path(entities_path)
        self.chunks_path = Path(chunks_path)
        self.output_path = Path(output_path)
        self.num_workers = num_workers
        self.checkpoint_interval = checkpoint_interval
        
        # Thread-safe storage (Phase 1B pattern)
        self.results = []
        self.results_lock = Lock()
        self.processed_ids = set()
        self.total_relations = 0
        self.errors = 0
        
        self.checkpoint_dir = self.output_path.parent / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize extractor (API key from .env)
        self.extractor = RAKGRelationExtractor(
            model_name=os.getenv('RELATION_MODEL', 'Qwen/Qwen2.5-7B-Instruct-Turbo'),
            semantic_threshold=float(os.getenv('SEMANTIC_THRESHOLD', '0.85')),
            mmr_lambda=float(os.getenv('MMR_LAMBDA', '0.55')),
            num_chunks=int(os.getenv('NUM_CHUNKS', '20'))
        )
        
        logger.info(f"Initialized (workers={num_workers})")
    
    def load_entities(self):
        with open(self.entities_path) as f:
            entities = json.load(f)
        if isinstance(entities, dict) and 'entities' in entities:
            entities = entities['entities']
        logger.info(f"Loaded {len(entities)} entities")
        return entities
    
    def load_chunks(self):
        with open(self.chunks_path) as f:
            chunks = json.load(f)
        if isinstance(chunks, dict):
            chunks = list(chunks.values()) if 'chunks' not in chunks else chunks['chunks']
        logger.info(f"Loaded {len(chunks)} chunks")
        return chunks
    
    def load_checkpoint(self):
        """Load checkpoint (Phase 1B pattern)"""
        ckpts = sorted(self.checkpoint_dir.glob("checkpoint_*.json"))
        if not ckpts:
            return 0
        
        logger.info(f"Loading {ckpts[-1].name}")
        with open(ckpts[-1]) as f:
            data = json.load(f)
        
        self.results = data.get('relations', [])
        self.processed_ids = set(data.get('processed_entity_ids', []))
        self.total_relations = len(self.results)
        
        logger.info(f"  Resumed: {len(self.processed_ids)} entities")
        return len(self.processed_ids)
    
    def save_checkpoint(self, num, final=False):
        """Atomic save (Phase 1B pattern - caller must hold lock!)"""
        out = self.output_path if final else self.checkpoint_dir / f"checkpoint_{num:04d}.json"
        data = {
            'metadata': {
                'final': final,
                'entities_processed': len(self.processed_ids),
                'total_relations': len(self.results),
                'errors': self.errors,
                'timestamp': datetime.now().isoformat()
            },
            'processed_entity_ids': list(self.processed_ids),
            'relations': self.results
        }
        
        temp = out.with_suffix('.tmp')
        try:
            with open(temp, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            with open(temp) as f:
                json.load(f)  # Validate
            temp.replace(out)
            if not final:
                logger.info(f"  ✓ Checkpoint: {out.name}")
        except Exception as e:
            logger.error(f"  ✗ Save failed: {e}")
            if temp.exists():
                temp.unlink()
            raise
    
    def extract_with_retry(self, entity, chunks, max_retries=3):
        """Extract with exponential backoff (Phase 1B pattern)"""
        name = entity.get('name', 'Unknown')
        
        for attempt in range(max_retries):
            try:
                return self.extractor.extract_relations_for_entity(entity, chunks)
            except Exception as e:
                if '429' in str(e) or 'rate limit' in str(e).lower():
                    wait = 2 ** attempt
                    logger.warning(f"  Rate limit for {name}, wait {wait}s")
                    time.sleep(wait)
                    continue
                else:
                    logger.error(f"  ✗ {name}: {e}")
                    with self.results_lock:
                        self.errors += 1
                    return []
        
        logger.error(f"  ✗ Failed {name} after {max_retries} attempts")
        with self.results_lock:
            self.errors += 1
        return []
    
    def process_all_entities(self, test_mode=False, test_size=100):
        """Main parallel processing (Phase 1B pattern)"""
        logger.info("=" * 80)
        logger.info("PHASE 1D: RELATION EXTRACTION")
        logger.info("=" * 80)
        
        entities = self.load_entities()
        chunks = self.load_chunks()
        offset = self.load_checkpoint()
        
        end = offset + test_size if test_mode else len(entities)
        
        to_process = []
        for i in range(offset, min(end, len(entities))):
            eid = entities[i].get('id', entities[i].get('name', f'entity_{i}'))
            if eid not in self.processed_ids:
                to_process.append(entities[i])
        
        if not to_process:
            logger.info("✓ All done!")
            return
        
        logger.info(f"Processing {len(to_process)} NEW entities")
        logger.info(f"Workers: {self.num_workers}")
        logger.info("")
        
        start = datetime.now()
        last_ckpt = len(self.processed_ids)
        
        # Parallel processing with ThreadPoolExecutor (Phase 1B pattern)
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            futures = {ex.submit(self.extract_with_retry, e, chunks): e for e in to_process}
            
            for future in as_completed(futures):
                entity = futures[future]
                eid = entity.get('id', entity.get('name'))
                rels = future.result()
                
                with self.results_lock:
                    self.results.extend(rels)
                    self.processed_ids.add(eid)
                    self.total_relations = len(self.results)
                    
                    done = len(self.processed_ids)
                    
                    if done % 10 == 0:
                        elapsed = (datetime.now() - start).total_seconds()
                        rate = (done - offset) / elapsed if elapsed > 0 else 0
                        logger.info(
                            f"Progress: {done}/{len(entities)} | "
                            f"Relations: {self.total_relations} | "
                            f"Rate: {rate:.1f}/sec"
                        )
                    
                    if done - last_ckpt >= self.checkpoint_interval:
                        logger.info("[CHECKPOINT] Saving...")
                        self.save_checkpoint(done)
                        last_ckpt = done
        
        # Final save
        logger.info("\nSaving final...")
        
        output = [
            {
                'triplet_id': f"rel_{i+1:06d}",
                'subject': r.get('subject'),
                'predicate': r.get('predicate'),
                'object': r.get('object'),
                'chunk_ids': r.get('chunk_ids', []),
                'extracted_from_entity': r.get('extracted_from_entity')
            }
            for i, r in enumerate(self.results)
        ]
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        # Stats
        stats = {
            'total_relations': len(output),
            'entities_processed': len(self.processed_ids),
            'avg_relations_per_entity': len(output) / len(self.processed_ids),
            'timestamp': datetime.now().isoformat()
        }
        with open(self.output_path.parent / "extraction_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        elapsed = (datetime.now() - start).total_seconds()
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Entities: {len(self.processed_ids)}")
        logger.info(f"Relations: {self.total_relations}")
        logger.info(f"Time: {elapsed/60:.1f} min")
        logger.info(f"Output: {self.output_path}")
        logger.info("=" * 80)

def main():
    import argparse
    p = argparse.ArgumentParser(description='Phase 1D Relation Extraction')
    p.add_argument('--workers', type=int, default=5, help='Number of parallel workers')
    p.add_argument('--test', action='store_true', help='Test mode')
    p.add_argument('--test-size', type=int, default=100, help='Test size')
    p.add_argument('--entities', default='data/processed/normalized_entities.json')
    p.add_argument('--chunks', default='data/interim/chunks/chunks_embedded.json')
    p.add_argument('--output', default='data/processed/relations/triplets.json')
    args = p.parse_args()
    
    proc = RelationProcessor(args.entities, args.chunks, args.output, args.workers)
    proc.process_all_entities(args.test, args.test_size)

if __name__ == "__main__":
    main()