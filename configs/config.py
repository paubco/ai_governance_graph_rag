"""
Configuration for AI Governance GraphRAG Pipeline
Loads sensitive data from .env, defines application logic here
"""
from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ============================================================================
# BASE PATHS (from .env)
# ============================================================================
BASE_DIR = Path(__file__).parent.parent  # Project root
DATA_PATH = Path(os.getenv('DATA_PATH', 'data/'))

# Create absolute paths
if not DATA_PATH.is_absolute():
    DATA_PATH = BASE_DIR / DATA_PATH

# ============================================================================
# DERIVED PATHS (calculated from base)
# ============================================================================
# Data directories
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
INTERIM_DATA_PATH = DATA_PATH / "interim"
EXTERNAL_DATA_PATH = DATA_PATH / "external"

# Raw data sources
DLAPIPER_RAW_PATH = RAW_DATA_PATH / "dlapiper"
WIKIPEDIA_RAW_PATH = RAW_DATA_PATH / "wikipedia"
ACADEMIC_RAW_PATH = RAW_DATA_PATH / "academic"

# Processed data
ENTITIES_PATH = PROCESSED_DATA_PATH / "entities"
EMBEDDINGS_PATH = PROCESSED_DATA_PATH / "embeddings"
GRAPH_DATA_PATH = PROCESSED_DATA_PATH / "graph"

# Logs
LOGS_PATH = DATA_PATH / "logs"
SCRAPER_LOGS_PATH = LOGS_PATH / "scraper"
EXTRACTION_LOGS_PATH = LOGS_PATH / "extraction"

# Create all directories
for path in [RAW_DATA_PATH, PROCESSED_DATA_PATH, INTERIM_DATA_PATH, 
             EXTERNAL_DATA_PATH, LOGS_PATH, SCRAPER_LOGS_PATH, 
             EXTRACTION_LOGS_PATH, DLAPIPER_RAW_PATH, WIKIPEDIA_RAW_PATH,
             ACADEMIC_RAW_PATH, ENTITIES_PATH, EMBEDDINGS_PATH, GRAPH_DATA_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# SCRAPER CONFIGURATION (Application logic - NOT in .env)
# ============================================================================
SCRAPER_CONFIG = {
    "base_url": "https://intelligence.dlapiper.com/artificial-intelligence/",
    "output_dir": DLAPIPER_RAW_PATH,
    "delay_between_requests": 2,  # seconds - be respectful!
    "timeout": 10,  # seconds
    "retry_attempts": 3,
    "headers": {
        "User-Agent": "Mozilla/5.0 (Educational Research Bot)",
    }
}

# ============================================================================
# ENTITY EXTRACTION CONFIGURATION (for Week 1-2)
# ============================================================================
EXTRACTION_CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.1",
    "api_key": os.getenv('TOGETHER_API_KEY'),  # ← From .env
    "temperature": 0.0,  # Deterministic for entity extraction
    "max_tokens": 2048,
    "chunk_size": 512,
    "chunk_overlap": 50,
}

# ============================================================================
# EMBEDDING CONFIGURATION (for Week 2-3)
# ============================================================================
EMBEDDING_CONFIG = {
    "model_name": "BAAI/bge-m3",
    "dimension": 1024,
    "batch_size": 32,
    "device": "cuda",  # or "cpu"
}

# ============================================================================
# NEO4J CONFIGURATION (for Week 3-4)
# ============================================================================
NEO4J_CONFIG = {
    "uri": os.getenv('NEO4J_URI', 'bolt://localhost:7687'),  # ← From .env
    "user": os.getenv('NEO4J_USER', 'neo4j'),  # ← From .env
    "password": os.getenv('NEO4J_PASSWORD'),  # ← From .env (REQUIRED)
    "database": "neo4j",
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": LOGS_PATH / "pipeline.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"],
    },
}
