import os
import pandas as pd
import ast
from utils.config import DATA_PATH
from utils.logger import logger

# -----------------------------
# Helper functions
# -----------------------------

def safe_eval(value):
    """Safely evaluate strings that look like Python literals (dicts/lists)."""
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            return value
    return value

def parse_affiliations(affiliations):
    """Normalize affiliations into a list of dicts with 'name'."""
    if not affiliations:
        return []
    if isinstance(affiliations, dict):
        return [affiliations]
    if isinstance(affiliations, list):
        cleaned = []
        for aff in affiliations:
            if isinstance(aff, dict):
                cleaned.append(aff)
            elif isinstance(aff, str):
                cleaned.append({"name": aff})
        return cleaned
    if isinstance(affiliations, str):
        return [{"name": affiliations}]
    return []

def parse_links(links):
    """Normalize links into dicts with at least 'self' if present."""
    if not links:
        return {}
    if isinstance(links, str):
        try:
            links = ast.literal_eval(links)
        except Exception:
            return {"self": links}
    if isinstance(links, dict):
        return links
    return {}

# -----------------------------
# Main loader
# -----------------------------

def load_local_data():
    """
    Load the temporary 24-paper dataset from CSV,
    split into metadata and text data, and clean nested fields.
    """
    file_path = os.path.join(DATA_PATH, "abstract_sample_data.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)

    # Split into metadata and text data (if abstract or similar exists)
    text_columns = [c for c in df.columns if "abstract" in c.lower() or "description" in c.lower()]
    text_df = df[["scopus_id"] + text_columns].copy() if text_columns else pd.DataFrame()

    metadata_df = df[
        [
            "scopus_id",
            "title",
            "creator",
            "publication_name",
            "cover_date",
            "subtype_desc",
            "links",
            "affiliations",
        ]
    ].copy()

    # Clean nested data
    metadata_df["affiliations"] = metadata_df["affiliations"].apply(safe_eval)
    metadata_df["links"] = metadata_df["links"].apply(safe_eval)

    metadata_df["affiliations_clean"] = metadata_df["affiliations"].apply(parse_affiliations)
    metadata_df["links_clean"] = metadata_df["links"].apply(parse_links)

    # Optional debug prints
    logger.info(f"Sample affiliations_clean: {metadata_df['affiliations_clean'].iloc[0]}")
    logger.info(f"Sample links_clean: {metadata_df['links_clean'].iloc[0]}")

    metadata_df.drop(columns=["affiliations", "links"], inplace=True)
    metadata_df.rename(
        columns={"affiliations_clean": "affiliations", "links_clean": "links"}, inplace=True
    )

    logger.info(f"Loaded {len(metadata_df)} metadata entries.")
    return metadata_df, text_df

