# -*- coding: utf-8 -*-
"""
Centralized logging configuration for GraphRAG pipeline

Provides consistent logging setup across all modules with optional file output and
real-time log streaming. Each module should call setup_logging() once at the entry
point, then use logger = logging.getLogger(__name__) in individual modules.

Examples:
# In main script or entry point
    from src.utils.logger import setup_logging
    setup_logging(log_file="logs/my_process.log")

    # In any module
    import logging
    logger = logging.getLogger(__name__)
    logger.info("This will use the configured format")

"""
# Standard library
import logging
import sys
from pathlib import Path
from typing import Optional

# Global flag to prevent duplicate configuration
_logging_configured = False


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> None:
    """
    Configure logging for the application.

    Sets up console output and optional file output with consistent formatting.
    Should be called once at the application entry point. Safe to call multiple
    times (will only configure on first call).

    Args:
        level: Logging level (default: logging.INFO)
        log_file: Optional path to log file. If provided, creates logs directory
                  and writes to file in addition to console
        format_string: Log message format (default: includes timestamp, module, level, message)

    Example:
        >>> setup_logging(log_file="logs/processing.log")
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Ready to process")
    """
    global _logging_configured

    # Only configure once to avoid duplicate handlers
    if _logging_configured:
        return

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Setup handlers
    handlers = []

    # Console handler (always included)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # File handler (optional)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True  # Override any existing configuration
    )

    # Enable unbuffered output for real-time logs
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Convenience function that wraps logging.getLogger(). Use this or
    logging.getLogger(__name__) directly - both work the same way.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    return logging.getLogger(name)
