import logging
from pathlib import Path

def setup_logger(name="graph_rag", log_file="logs/app.log", level=logging.INFO):
    Path("logs").mkdir(exist_ok=True)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    return logger
