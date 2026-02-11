#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Folder Structure Processor - Production Version

Processes all supported documents inside:
    input_strct_run

Outputs structure JSON files into:
    claude_output

Uses the Stage-1 DocumentStructureExtractor (dict-based).

Works on:
    - Windows
    - WSL
    - Linux
    - macOS
"""

import sys
import traceback
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

try:
    from structure_extractor import DocumentStructureExtractor

except ImportError:
    print("[ERROR] Cannot import structure_extractor")
    print("Make sure structure_extractor.py is in the same directory")
    sys.exit(1)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input_strct_run"
OUTPUT_DIR = BASE_DIR / "claude_output"

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".txt",
    ".xlsx",
    ".xlsm",
    ".xls",
}

LOG_FILE = BASE_DIR / "extraction.log"
VERBOSE = True


# ==============================================================================
# SETUP LOGGING
# ==============================================================================

def setup_logging():
    logger = logging.getLogger("BatchExtractor")
    logger.setLevel(logging.DEBUG if VERBOSE else logging.INFO)

    logger.handlers.clear()

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ==============================================================================
# PROCESSING FUNCTIONS
# ==============================================================================

def process_single_file(
    file_path: Path,
    output_dir: Path,
    logger: logging.Logger
) -> Dict[str, any]:

    result = {
        "file": file_path.name,
        "path": str(file_path),
        "status": "unknown",
        "output": None,
        "chunks": 0,
        "pages": 0,
        "error": None,
    }

    try:
        # Create extractor (Stage-1 version)
        extractor = DocumentStructureExtractor(str(file_path))

        # Extract structure (dict)
        structure = extractor.extract()

        # Determine output path (preserve folder structure)
        relative_path = file_path.relative_to(INPUT_DIR)
        output_subdir = output_dir / relative_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)

        output_file = output_subdir / f"{file_path.stem}_structure.json"

        # Save structure (no pretty flag — Stage-1 aligned)
        extractor.save(structure, output_file)

        result["status"] = "success"
        result["output"] = str(output_file)
        result["chunks"] = len(structure.get("chunks", []))
        result["pages"] = structure.get("total_pages", 0)

        logger.info(
            f"✓ {file_path.name} -> "
            f"{result['pages']} pages, {result['chunks']} chunks"
        )

    except FileNotFoundError as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"✗ {file_path.name} - File not found")

    except ValueError as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"✗ {file_path.name} - Invalid file: {e}")

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"✗ {file_path.name} - Error: {e}")
        if VERBOSE:
            logger.debug(traceback.format_exc())

    return result


def process_folder(logger: logging.Logger):
    start_time = datetime.now()

    if not INPUT_DIR.exists():
        logger.error(f"Input folder not found: {INPUT_DIR}")
        logger.info("Please create the folder and add documents to process")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_files = 0
    processed = 0
    skipped = 0
    failed = 0
    results: List[Dict] = []

    logger.info("=" * 70)
    logger.info("DOCUMENT STRUCTURE EXTRACTION - BATCH PROCESSOR")
    logger.info("=" * 70)
    logger.info(f"Input:  {INPUT_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Log:    {LOG_FILE}")
    logger.info("")

    all_files = list(INPUT_DIR.rglob("*"))

    for file_path in all_files:
        if file_path.is_file():
            total_files += 1

    logger.info(f"Found {total_files} files")
    logger.info("")

    for idx, file_path in enumerate(all_files, 1):

        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            skipped += 1
            continue

        logger.info(f"[{idx}/{total_files}] Processing: {file_path.name}")

        result = process_single_file(file_path, OUTPUT_DIR, logger)
        results.append(result)

        if result["status"] == "success":
            processed += 1
        else:
            failed += 1

    elapsed = datetime.now() - start_time

    logger.info("")
    logger.info("=" * 70)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total Files Scanned       : {total_files}")
    logger.info(f"Successfully Processed    : {processed}")
    logger.info(f"Skipped (unsupported)     : {skipped}")
    logger.info(f"Failed                    : {failed}")
    logger.info(f"Elapsed Time              : {elapsed}")

    if processed > 0:
        total_chunks = sum(r["chunks"] for r in results if r["status"] == "success")
        total_pages = sum(r["pages"] for r in results if r["status"] == "success")

        logger.info("")
        logger.info("Extraction Statistics:")
        logger.info(f"  Total Chunks Extracted  : {total_chunks}")
        logger.info(f"  Total Pages Processed   : {total_pages}")

    logger.info("=" * 70)
    logger.info("")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    logger = setup_logging()

    try:
        process_folder(logger)

    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
