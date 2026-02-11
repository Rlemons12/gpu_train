from pathlib import Path
import argparse

from dataset_builder import DatasetBuilder
from dataset_gen.dataset_qanda_generator.configuration.logging_config import (
    get_qna_logger,
)

# GOVERNANCE (authoritative imports)
from app.config.pg_db_config import get_governance_session
from app.ml_governance.registry import DatasetRegistry

# Dataset naming helper (shared logic with UI)
from dataset_gen.app.routes.dataset_generator import (
    build_dataset_name_from_filters,
)

# ------------------------------------------------------------------
# LOGGER
# ------------------------------------------------------------------
log = get_qna_logger("dataset_export")


def main():
    """
    CLI entrypoint for exporting training datasets from the Q&A database.

    This command is AUTHORITATIVE for dataset creation and MUST:
      • derive dataset name from filters
      • export into dataset-scoped folders
      • register versions with ml_governance
    """

    parser = argparse.ArgumentParser(
        description="Export Q&A dataset from database"
    )

    # -----------------------------
    # Selection filters
    # -----------------------------
    parser.add_argument("--include-tags", nargs="*")
    parser.add_argument("--exclude-tags", nargs="*")
    parser.add_argument("--domain")
    parser.add_argument("--audience")
    parser.add_argument("--max-criticality")
    parser.add_argument("--document-ids", nargs="*", type=int)

    # -----------------------------
    # Output options
    # -----------------------------
    parser.add_argument(
        "--format",
        choices=["jsonl", "alpaca", "chatml", "prompt_response", "all"],
        default="all",
    )
    parser.add_argument(
        "--out",
        default="dataset_output",
        help="Root output directory",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Skip invalid documents instead of failing",
    )

    args = parser.parse_args()

    # -----------------------------
    # Resolve dataset name
    # -----------------------------
    base_name = "qna_training"

    dataset_name = build_dataset_name_from_filters(
        base_name,
        domain=args.domain,
        audience=args.audience,
        max_criticality=args.max_criticality,
        include_tags=args.include_tags or None,
        exclude_tags=args.exclude_tags or None,
        document_ids=args.document_ids or None,
    )

    dataset_dir = Path(args.out) / dataset_name

    # -----------------------------
    # Log configuration summary
    # -----------------------------
    log.info("===================================================")
    log.info("Dataset export started (CLI)")
    log.info("Dataset name     : %s", dataset_name)
    log.info("Dataset dir      : %s", dataset_dir)
    log.info("Format           : %s", args.format)
    log.info("Strict mode      : %s", not args.non_strict)
    log.info("Domain           : %s", args.domain)
    log.info("Audience         : %s", args.audience)
    log.info("Max criticality  : %s", args.max_criticality)
    log.info("Include tags     : %s", args.include_tags)
    log.info("Exclude tags     : %s", args.exclude_tags)
    log.info("Document IDs     : %s", args.document_ids)
    log.info("===================================================")

    # -----------------------------
    # Build dataset
    # -----------------------------
    builder = DatasetBuilder(
        output_dir=dataset_dir,
        strict=not args.non_strict,
    )

    records = builder.build(
        include_tags=args.include_tags,
        exclude_tags=args.exclude_tags,
        domain=args.domain,
        audience=args.audience,
        max_criticality=args.max_criticality,
        document_ids=args.document_ids,
    )

    if not records:
        log.warning("No dataset records generated — export skipped")
        return

    log.info("Dataset build complete: %d records", len(records))

    # -----------------------------
    # JSONL export + governance
    # -----------------------------
    with get_governance_session() as db:
        registry = DatasetRegistry(db, actor="dataset_export")

        # temp filename; governance decides version
        temp_path = builder.export_jsonl(
            records,
            filename="dataset.jsonl",
            output_dir=dataset_dir,
        )

        dataset = registry.register_dataset(
            name=dataset_name,
            path=str(temp_path),
            generator="dataset_export_cli",
            notes={
                "source": "dataset_export_cli",
                "record_count": len(records),
                "filters": {
                    "include_tags": args.include_tags,
                    "exclude_tags": args.exclude_tags,
                    "domain": args.domain,
                    "audience": args.audience,
                    "max_criticality": args.max_criticality,
                    "document_ids": args.document_ids,
                },
            },
        )

    # -----------------------------
    # Rename JSONL to versioned name
    # -----------------------------
    versioned_jsonl = (
        dataset_dir / f"{dataset.name}__{dataset.version}.jsonl"
    )
    temp_path.rename(versioned_jsonl)

    log.info(
        "Dataset registered: name=%s version=%s hash=%s",
        dataset.name,
        dataset.version,
        dataset.content_hash,
    )

    # -----------------------------
    # Additional formats (same folder, same version)
    # -----------------------------
    if args.format in ("alpaca", "all"):
        builder.export_alpaca(
            records,
            filename=f"alpaca__{dataset.version}.jsonl",
            output_dir=dataset_dir,
        )

    if args.format in ("chatml", "all"):
        builder.export_chatml(
            records,
            filename=f"chatml__{dataset.version}.jsonl",
            output_dir=dataset_dir,
        )

    if args.format in ("prompt_response", "all"):
        builder.export_prompt_response(
            records,
            filename=f"train_prompt_response__{dataset.version}.jsonl",
            output_dir=dataset_dir,
        )

    log.info(
        "Dataset export completed successfully (%d records)",
        len(records),
    )


if __name__ == "__main__":
    main()
