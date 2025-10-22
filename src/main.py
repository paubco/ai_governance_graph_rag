# src/main.py
import sys
import os
import pandas as pd
from utils.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DATA_PATH, DEBUG_MODE
from utils.logger import logger, cleaning_logger
from utils.neo4j_utils import get_driver, load_metadata_to_neo4j, close_driver
from data_intake import load_local_data
from text_preprocessing import chunk_dataframe
from dbpedia_pipeline import annotate_text_spotlight, fetch_dbpedia_entity, parse_dbpedia_entity, extract_and_clean_entities

def main():
    logger.info("Starting Graph RAG pipeline...")

    # Step 1 — Load data
    try:
        metadata_df, text_df = load_local_data()
        logger.info(f"Loaded {len(metadata_df)} metadata entries.")
        logger.info(f"Loaded {len(text_df)} text entries.")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # Step 1b — Chunk text for DBpedia annotation
    try:
        chunks_df = chunk_dataframe(text_df, text_col="description", id_col="scopus_id")
        logger.info(f"Text data chunked into {len(chunks_df)} chunks.")
    except Exception as e:
        logger.error(f"Failed to chunk text: {e}")
        sys.exit(1)

    # Step 2 — Annotate chunks with DBpedia Spotlight
    all_annotations = []
    try:
        for _, row in chunks_df.iterrows():
            annotations = annotate_text_spotlight(row["scopus_id"], row["chunk_id"], row["chunk_text"])
            for scopus_id, chunk_id, uri, surface, score in annotations:
                all_annotations.append({
                    "scopus_id": scopus_id,
                    "chunk_id": chunk_id,
                    "URI": uri,
                    "surface_form": surface,
                    "similarity_score": score
                })
        df_annotations = pd.DataFrame(all_annotations)
        raw_annotations_path = os.path.join(DATA_PATH, "annotations_raw.csv")
        df_annotations.to_csv(raw_annotations_path, index=False)
        logger.info(f"Saved raw annotations: {raw_annotations_path}")
    except Exception as e:
        logger.error(f"Error during DBpedia annotation: {e}")
        sys.exit(1)

    # Step 3 — Fetch, parse, and clean DBpedia entities
    try:
        entities_json = []
        for _, row in df_annotations.iterrows():
            uri = row["URI"]
            surface_form = row["surface_form"]
            score = row["similarity_score"]

            # Fetch DBpedia JSON
            entity_json = fetch_dbpedia_entity(uri)
            if entity_json:
                entity_info = parse_dbpedia_entity(entity_json)
            else:
                # Fallback if fetch fails
                entity_info = {
                    "label": surface_form,
                    "abstract": None,
                    "types": [],
                    "relations": {},
                    "wikidata_quids": []
                }

            # Add annotation metadata
            entity_info.update({
                "uri": uri,
                "surface_form": surface_form,
                "similarity_score": score
            })

            entities_json.append(entity_info)

        # Clean and validate entities
        cleaned_entities = extract_and_clean_entities(entities_json)
        entities_df = pd.DataFrame(cleaned_entities)

        cleaned_entities_path = os.path.join(DATA_PATH, "entities_clean.csv")
        entities_df.to_csv(cleaned_entities_path, index=False)
        logger.info(f"Saved cleaned DBpedia entities: {cleaned_entities_path}")
    except Exception as e:
        logger.error(f"Error extracting/cleaning DBpedia entities: {e}")
        sys.exit(1)

    # Step 4 — Connect to Neo4j
    try:
        driver = get_driver(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        logger.info(f"Connected to Neo4j at {NEO4J_URI}")
    except Exception as e:
        logger.error(f"Error connecting to Neo4j: {e}")
        sys.exit(1)

    # Step 5 — Load metadata into Neo4j
    try:
        load_metadata_to_neo4j(driver, metadata_df)
        logger.info("Metadata successfully loaded into Neo4j.")
    except Exception as e:
        logger.error(f"Error loading metadata into Neo4j: {e}")
    finally:
        close_driver(driver)
        logger.info("Neo4j connection closed.")

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
