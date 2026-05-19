"""
Structured logging setup.
Outputs JSON in production (LOG_LEVEL != DEBUG) and human-readable in dev.
"""

import logging
import sys

from lead_scoring.config import settings


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured for the environment."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if settings.log_level.upper() == "DEBUG":
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt, datefmt=datefmt)
    else:
        # JSON-ish single-line format, easy to ingest into log aggregators
        fmt = '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
        formatter = logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S")

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
