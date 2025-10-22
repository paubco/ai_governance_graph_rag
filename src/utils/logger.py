# utils/logger.py
import logging
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True, parents=True)

def setup_logger(name: str, file_name: str):
    """
    Sets up both file and console logging.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # --- File handler ---
        file_handler = logging.FileHandler(LOG_DIR / file_name, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # --- Console handler ---
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


# --- Initialize the main and cleaning loggers ---
logger = setup_logger("logger", "project.log")
cleaning_logger = setup_logger("cleaning_logger", "entity_cleaning.log")
