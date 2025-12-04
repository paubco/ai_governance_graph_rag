#!/usr/bin/env python3
"""
Add hash-based entity IDs to normalized entities.

Hash-based IDs are deterministic: same (name, type) → same ID every time.
This ensures reproducibility across pipeline runs.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List

def generate_entity_id(name: str, entity_type: str) -> str:
    """
    Generate deterministic hash-based ID from entity name + type.
    
    Args:
        name: Entity name
        entity_type: Entity type (Technology, Organization, etc.)
    
    Returns:
        Hash-based ID like "ent_a3f4e9c2d5b1"
    """
    # Normalize to lowercase for consistency
    content = f"{name}|{entity_type}".lower()
    
    # SHA-256 hash (first 12 chars = 48 bits = ~281 trillion combinations)
    hash_val = hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]
    
    return f"ent_{hash_val}"


def add_ids_to_entities(input_file: str, output_file: str, lookup_file: str):
    """
    Add entity_id field to all entities and create name->id lookup.
    
    Args:
        input_file: Path to normalized_entities.json
        output_file: Path to save entities with IDs
        lookup_file: Path to save name->id lookup dict
    """
    print(f"📖 Loading entities from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        entities = json.load(f)
    
    print(f"🔑 Generating hash-based IDs for {len(entities)} entities...")
    
    # Add IDs to entities
    name_to_id = {}
    id_collisions = {}
    
    for entity in entities:
        name = entity['name']
        entity_type = entity['type']
        
        # Generate hash-based ID
        entity_id = generate_entity_id(name, entity_type)
        entity['entity_id'] = entity_id
        
        # Track for lookup and collision detection
        if entity_id in id_collisions:
            id_collisions[entity_id].append((name, entity_type))
        else:
            id_collisions[entity_id] = [(name, entity_type)]
        
        name_to_id[name] = entity_id
    
    # Check for ID collisions (extremely rare with 12-char hash)
    collisions = {k: v for k, v in id_collisions.items() if len(v) > 1}
    if collisions:
        print(f"⚠️  WARNING: {len(collisions)} ID collisions detected!")
        for entity_id, entries in list(collisions.items())[:5]:
            print(f"   {entity_id}: {entries}")
    else:
        print(f"✅ No ID collisions (all {len(entities)} IDs unique)")
    
    # Save entities with IDs
    print(f"💾 Saving entities with IDs to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(entities, f, indent=2, ensure_ascii=False)
    
    # Save name->id lookup
    print(f"💾 Saving name->id lookup to {lookup_file}...")
    with open(lookup_file, 'w', encoding='utf-8') as f:
        json.dump(name_to_id, f, indent=2, ensure_ascii=False)
    
    # Stats
    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"\n📊 Results:")
    print(f"   Entities processed: {len(entities):,}")
    print(f"   Unique IDs created: {len(name_to_id):,}")
    print(f"   Output file size: {file_size_mb:.1f} MB")
    print(f"\n✅ Done! Files created:")
    print(f"   - {output_file}")
    print(f"   - {lookup_file}")


if __name__ == "__main__":
    # File paths
    INPUT_FILE = "data/interim/entities/normalized_entities.json"
    OUTPUT_FILE = "data/processed/normalized_entities_with_ids.json"
    LOOKUP_FILE = "data/processed/entity_name_to_id.json"
    
    # Check input exists
    if not Path(INPUT_FILE).exists():
        print(f"❌ Error: Input file not found: {INPUT_FILE}")
        print(f"   Please adjust the INPUT_FILE path in the script.")
        exit(1)
    
    # Run
    add_ids_to_entities(INPUT_FILE, OUTPUT_FILE, LOOKUP_FILE)
    
    print("\n🎯 Next steps:")
    print("   1. Use normalized_entities_with_ids.json going forward")
    print("   2. Keep original normalized_entities.json as backup")
    print("   3. Use entity_name_to_id.json for relation enrichment")
