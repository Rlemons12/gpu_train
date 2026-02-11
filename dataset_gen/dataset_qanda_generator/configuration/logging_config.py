#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Central logging utility for the Option-C Q&A pipeline.

Provides:
    - Rotating file logs (5MB × 5 backups)
    - Console logs
    - Consistent EMTAC-style formatting
    - Automatic directory creation using cfg.LOG_DIR
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dataset_gen.dataset_qanda_generator.configuration import cfg



# ============================================================
# LOG DIRECTORY
# ============================================================

LOG_DIR: Path = cfg.LOG_DIR
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "option_c_pipeline.log"


# ============================================================
# LOGGER FACTORY
# ============================================================
def get_qna_logger(name: str = "option_c_pipeline") -> logging.Logger:
    """
    Returns a configured logger instance.

    Features:
        • INFO level by default
        • File handler → 5MB rotating logs × 5 backups
        • Console handler
        • Prevents handler duplication on multiple imports
    """

    logger = logging.getLogger(name)

    # Avoid double-adding handlers if this logger already configured
    if getattr(logger, "_option_c_configured", False):
        return logger

    logger.setLevel(logging.INFO)

    # Unified EMTAC formatting
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # --------------------------------------------------------
    # FILE HANDLER
    # --------------------------------------------------------
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    # --------------------------------------------------------
    # CONSOLE HANDLER
    # --------------------------------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Attach handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Mark logger as configured to prevent duplication
    logger._option_c_configured = True

    return logger


# Default logger (import-friendly)
logger = get_qna_logger()
