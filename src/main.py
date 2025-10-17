# src/main.py
import sys
from utils.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DEBUG_MODE
from utils.logger import setup_logger
from utils.neo4j_utils import get_driver, load_metadata_to_neo4j, close_driver
from data_intake import load_local_data


def main():
    logger = setup_logger()
    logger.info("Starting Graph RAG pipeline...")

    # Step 1 — Load data
    try:
        metadata_df, text_df = load_local_data()  # <-- unpack tuple
        logger.info(f"Loaded {len(metadata_df)} metadata entries.")
        logger.info(f"Loaded {len(text_df)} text entries.")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # Step 2 — Connect to Neo4j
    try:
        driver = get_driver(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        logger.info(f"Connected to Neo4j at {NEO4J_URI}")
    except Exception as e:
        logger.error(f"Error connecting to Neo4j: {e}")
        sys.exit(1)

    # Step 3 — Load metadata into Neo4j
    try:
        load_metadata_to_neo4j(driver, metadata_df)
        logger.info("Data successfully loaded into Neo4j.")
    except Exception as e:
        logger.error(f"Error loading data into Neo4j: {e}")
    finally:
        close_driver(driver)
        logger.info("Neo4j connection closed.")

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
