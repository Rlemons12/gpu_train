"""
app/configuration/gpu_log_config.py
============================================================

GPU SERVICE LOGGING CONFIGURATION (UVICORN / FASTAPI)

WHAT THIS FILE DOES
-------------------
• Defines the **logging configuration dictionary** passed to uvicorn
• Controls:
    - uvicorn startup logs
    - uvicorn error logs
    - uvicorn access logs
• Uses **safe, minimal formatters**
• Avoids fragile LogRecord fields
• Compatible with:
    - FastAPI
    - uvicorn
    - reload mode
    - multiple workers

WHAT THIS FILE DOES *NOT* DO
----------------------------
• Does NOT define application-level loggers
• Does NOT log GPU activity or model events
• Does NOT implement request IDs
• Does NOT write to log files
• Does NOT replace domain loggers (see gpu_logger.py)

ARCHITECTURAL ROLE
------------------
This file exists ONLY to configure uvicorn’s logging system.

Domain logging (GPU activity, models, inference, routing, eviction)
MUST use:
    app/configuration/gpu_logger.py

INTENDED USAGE
--------------
Passed directly into uvicorn:

    uvicorn.run(
        "app.main:app",
        log_config=LOGGING_CONFIG,
        access_log=True,
    )

This file should remain **simple, stable, and boring**.
"""

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,

    # --------------------------------------------------
    # Formatters
    # --------------------------------------------------
    "formatters": {
        "default": {
            "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "access": {
            # NOTE:
            # uvicorn.access already formats client, method, path, status
            "format": "%(asctime)s | %(levelname)s | uvicorn.access | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },

    # --------------------------------------------------
    # Handlers
    # --------------------------------------------------
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "access": {
            "class": "logging.StreamHandler",
            "formatter": "access",
            "stream": "ext://sys.stdout",
        },
    },

    # --------------------------------------------------
    # Loggers
    # --------------------------------------------------
    "loggers": {
        "uvicorn": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.error": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["access"],
            "level": "INFO",
            "propagate": False,
        },
    },
}
