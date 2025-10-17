import logging
from pathlib import Path

def setup_logger(name="graph_rag", log_file="logs/app.log", level=logging.INFO):
    Path("logs").mkdir(exist_ok=True)
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler())

    logger.setLevel(level)
    return logger