# src/config.py
import os
from dotenv import load_dotenv

# Load .env file (if present)
load_dotenv()

# -----------------------------
#Provisional data
# -----------------------------
DATA_PATH = os.getenv("DATA_PATH", "data/")

# -----------------------------
# Neo4j configuration
# -----------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://default.uri")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")  # must be set in env or .env

# -----------------------------
# Scopus API keys
# -----------------------------
SCOPUS_API_KEY = os.getenv("SCOPUS_API_KEY")
SCOPUS_INST_KEY = os.getenv("SCOPUS_INST_KEY")

# -----------------------------
# Other keys / modules (example)
# -----------------------------
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

# -----------------------------
# Embedding Settings
# -----------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 50))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 10))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# -----------------------------
# Helper for debug/logging
# -----------------------------
DEBUG_MODE = os.getenv("DEBUG_MODE", "True").lower() in ("true", "1")
