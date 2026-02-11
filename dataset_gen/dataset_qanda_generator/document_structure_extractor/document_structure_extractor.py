#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Folder Structure Processor - Production Version

Processes all supported documents inside:
    input_strct_run

Outputs structure JSON files into:
    claude_output

Extraction logic is fully delegated to DocumentStrategy.
This batch processor is intentionally simple and layer-clean.
"""

import sys
import traceback
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from dataset_gen.dataset_qanda_generator.document_structure_extractor.document_strategy import (
    DocumentStrategy,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input_strct_run"
OUTPUT_DIR = BASE_DIR / "claude_output"
LOG_FILE = BASE_DIR / "extraction.log"

VERBOSE = True
CONTINUE_ON_ERROR = True


# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("BatchExtractor")
    logger.setLevel(logging.DEBUG if VERBOSE else logging.INFO)
    logger.handlers = []

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ==============================================================================
# CORE PROCESSING
# ==============================================================================

def process_single_file(
    file_path: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> Dict:

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
        strategy = DocumentStrategy(file_path)
        structure = strategy.extract_structure()

        relative_path = file_path.relative_to(INPUT_DIR)
        output_subdir = output_dir / relative_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)

        output_file = output_subdir / f"{file_path.stem}_structure.json"

        with open(output_file, "w", encoding="utf-8") as f:
            import json
            json.dump(structure, f, indent=2, ensure_ascii=False)

        chunks = structure.get("chunks", [])
        total_pages = structure.get("total_pages", 0)

        result["status"] = "success"
        result["output"] = str(output_file)
        result["chunks"] = len(chunks)
        result["pages"] = total_pages

        logger.info(
            f"✓ {file_path.name} -> {total_pages} pages, {len(chunks)} chunks"
        )

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"✗ {file_path.name} - {e}")
        if VERBOSE:
            logger.debug(traceback.format_exc())

    return result


def process_folder(logger: logging.Logger):

    start_time = datetime.now()

    if not INPUT_DIR.exists():
        logger.error(f"Input folder not found: {INPUT_DIR}")
        logger.info("Create the folder and add documents to process.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_files = 0
    processed = 0
    failed = 0
    results: List[Dict] = []

    logger.info("")
    logger.info("=" * 70)
    logger.info("DOCUMENT STRUCTURE EXTRACTION - BATCH PROCESSOR")
    logger.info("=" * 70)
    logger.info(f"Input:  {INPUT_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Log:    {LOG_FILE}")
    logger.info("")

    all_files = [f for f in INPUT_DIR.rglob("*") if f.is_file()]
    total_files = len(all_files)

    logger.info(f"Found {total_files} files\n")

    for idx, file_path in enumerate(all_files, 1):
        logger.info(f"[{idx}/{total_files}] Processing: {file_path.name}")

        result = process_single_file(file_path, OUTPUT_DIR, logger)
        results.append(result)

        if result["status"] == "success":
            processed += 1
        else:
            failed += 1
            if not CONTINUE_ON_ERROR:
                break

    elapsed = datetime.now() - start_time

    logger.info("")
    logger.info("=" * 70)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total Files           : {total_files}")
    logger.info(f"Successfully Processed: {processed}")
    logger.info(f"Failed                : {failed}")
    logger.info(f"Elapsed Time          : {elapsed}")

    if processed > 0:
        total_chunks = sum(r["chunks"] for r in results if r["status"] == "success")
        total_pages = sum(r["pages"] for r in results if r["status"] == "success")

        logger.info("")
        logger.info("Extraction Statistics:")
        logger.info(f"  Total Chunks : {total_chunks}")
        logger.info(f"  Total Pages  : {total_pages}")
        logger.info(f"  Avg Chunks   : {total_chunks / processed:.1f}")
        logger.info(f"  Avg Pages    : {total_pages / processed:.1f}")

    if failed > 0:
        logger.info("\nFailed Files:")
        for r in results:
            if r["status"] == "failed":
                logger.info(f"  - {r['file']}: {r['error']}")

    logger.info("")
    logger.info(f"✓ Results saved to: {OUTPUT_DIR}")
    logger.info(f"✓ Full log saved to: {LOG_FILE}")
    logger.info("")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    logger = setup_logging()

    logger.info("=" * 70)
    logger.info("Document Structure Extractor - Batch Processor")
    logger.info("=" * 70)
    logger.info("Extraction controlled by DocumentStrategy")
    logger.info("")

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
