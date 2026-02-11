from pathlib import Path

from flask import Blueprint, render_template, request

# --------------------------------------------------
# DATASET BUILDER
# --------------------------------------------------
from dataset_gen.dataset_qanda_generator.dataset_builder.dataset_builder import (
    DatasetBuilder,
)
from dataset_gen.dataset_qanda_generator.configuration.logging_config import (
    get_qna_logger,
)

# --------------------------------------------------
# Q&A ORM MODELS (UI LOOKUPS)
# --------------------------------------------------
from dataset_gen.dataset_qanda_generator.qanda_db.qa_db import (
    Domain,
    Audience,
    Criticality,
    DocumentTag,
)

# --------------------------------------------------
# GOVERNANCE
# --------------------------------------------------
from app.config.pg_db_config import get_governance_session
from app.ml_governance.registry import DatasetRegistry


# ------------------------------------------------------------------
# Blueprint
# ------------------------------------------------------------------
dataset_generator_bp = Blueprint(
    "dataset_generator",
    __name__,
    url_prefix="/dataset",
)

log = get_qna_logger("dataset_generator")


# ------------------------------------------------------------------
# Dataset naming
# ------------------------------------------------------------------
def build_dataset_name_from_filters(
    base_name: str,
    *,
    domain: str | None,
    audience: str | None,
    max_criticality: str | None,
    include_tags: list[str] | None,
    exclude_tags: list[str] | None,
    document_ids: list[int] | None,
) -> str:
    parts: list[str] = []

    if domain:
        parts.append(f"domain_{domain}")
    if audience:
        parts.append(f"audience_{audience}")
    if max_criticality:
        parts.append(f"maxcrit_{max_criticality}")
    if include_tags:
        parts.append(f"tags_{'_'.join(sorted(include_tags))}")
    if exclude_tags:
        parts.append(f"exclude_{'_'.join(sorted(exclude_tags))}")
    if document_ids:
        parts.append(f"documents_{'_'.join(map(str, sorted(document_ids)))}")

    if not parts:
        parts.append("complete")

    return f"{base_name}__{'__'.join(parts)}"


# ------------------------------------------------------------------
# UI: Dataset Generator Page
# ------------------------------------------------------------------
@dataset_generator_bp.route("/", methods=["GET"])
def index():
    session = DatasetBuilder().session

    domains = session.query(Domain).order_by(Domain.code).all()
    audiences = session.query(Audience).order_by(Audience.code).all()
    criticalities = (
        session.query(Criticality)
        .order_by(Criticality.severity)
        .all()
    )
    tags = session.query(DocumentTag).order_by(DocumentTag.code).all()

    log.info(
        "Dataset UI loaded: %d domains, %d audiences, %d criticalities, %d tags",
        len(domains),
        len(audiences),
        len(criticalities),
        len(tags),
    )

    return render_template(
        "dataset/dataset_generator.html",
        domains=domains,
        audiences=audiences,
        criticalities=criticalities,
        tags=tags,
    )


# ------------------------------------------------------------------
# Action: Run Dataset Export
# ------------------------------------------------------------------
@dataset_generator_bp.route("/run", methods=["POST"])
def run():
    # -----------------------------
    # Parse form inputs
    # -----------------------------
    domain = request.form.get("domain") or None
    audience = request.form.get("audience") or None
    max_criticality = request.form.get("max_criticality") or None

    include_tags = request.form.getlist("include_tags") or None
    exclude_tags = request.form.getlist("exclude_tags") or None

    document_ids_raw = request.form.get("document_ids", "").split()
    document_ids = [int(x) for x in document_ids_raw if x.isdigit()] or None

    output_dir = Path(request.form.get("out", "dataset_output")).resolve()
    fmt = request.form.get("format", "all")
    strict = not bool(request.form.get("non_strict"))

    log.info("==============================================")
    log.info("[DATASET UI] Dataset export triggered")
    log.info("Domain          : %s", domain)
    log.info("Audience        : %s", audience)
    log.info("Max criticality : %s", max_criticality)
    log.info("Include tags    : %s", include_tags)
    log.info("Exclude tags    : %s", exclude_tags)
    log.info("Document IDs    : %s", document_ids)
    log.info("Output dir      : %s", output_dir)
    log.info("Format          : %s", fmt)
    log.info("Strict mode     : %s", strict)
    log.info("==============================================")

    # -----------------------------
    # Build dataset records
    # -----------------------------
    builder = DatasetBuilder(
        output_dir=output_dir,
        strict=strict,
    )

    records = builder.build(
        domain=domain,
        audience=audience,
        max_criticality=max_criticality,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
        document_ids=document_ids,
    )

    if not records:
        log.warning("[DATASET UI] No valid dataset records produced")
        return render_template(
            "dataset/dataset_generator.html",
            result="No valid dataset records generated. Check filters or pipeline status.",
        )

    log.info("[DATASET UI] Dataset records assembled: %d", len(records))

    # -----------------------------
    # Dataset identity
    # -----------------------------
    dataset_name = build_dataset_name_from_filters(
        base_name="qna_training",
        domain=domain,
        audience=audience,
        max_criticality=max_criticality,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
        document_ids=document_ids,
    )

    exported: list[Path] = []

    # --------------------------------------------------
    # JSONL EXPORT (AUTHORITATIVE, REGISTRY-ALIGNED)
    # --------------------------------------------------
    if fmt in ("jsonl", "all"):
        # 1) Export temporary JSONL
        temp_jsonl = builder.export_jsonl(
            records,
            f"{dataset_name}.jsonl",
        )

        # 2) Allocate version (registry-owned)
        with get_governance_session() as db:
            registry = DatasetRegistry(db, actor="dataset_ui")
            version = registry.allocate_version(name=dataset_name)

        # 3) Create canonical dataset directory
        dataset_dir = output_dir / dataset_name / version
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # 4) Move file into authoritative location
        final_filename = f"{dataset_name}.jsonl"
        final_jsonl = dataset_dir / final_filename
        temp_jsonl.replace(final_jsonl)

        # 5) Register FINAL dataset path
        with get_governance_session() as db:
            registry = DatasetRegistry(db, actor="dataset_ui")
            dataset = registry.register_dataset(
                name=dataset_name,
                path=str(final_jsonl),
                generator="dataset_ui",
                notes={
                    "source": "dataset_ui",
                    "filters": {
                        "domain": domain,
                        "audience": audience,
                        "max_criticality": max_criticality,
                        "include_tags": include_tags,
                        "exclude_tags": exclude_tags,
                        "document_ids": document_ids,
                    },
                    "record_count": len(records),
                },
            )

        exported.append(final_jsonl)

        log.info(
            "[DATASET UI] Dataset registered: name=%s version=%s hash=%s path=%s",
            dataset.name,
            dataset.version,
            dataset.content_hash,
            final_jsonl,
        )

    else:
        dataset_dir = None

    # -----------------------------
    # Remaining formats
    # -----------------------------
    if dataset_dir:
        if fmt in ("alpaca", "all"):
            exported.append(
                builder.export_alpaca(
                    records,
                    "alpaca.jsonl",
                    output_dir=dataset_dir,
                )
            )

        if fmt in ("chatml", "all"):
            exported.append(
                builder.export_chatml(
                    records,
                    "chatml.jsonl",
                    output_dir=dataset_dir,
                )
            )

        """if fmt in ("prompt_response", "all"):
            exported.append(
                builder.export_prompt_response(
                    records,
                    "train_prompt_response.jsonl",
                    output_dir=dataset_dir,
                )
            )"""

    for path in exported:
        log.info("[DATASET UI] Exported: %s", path)

    log.info("[DATASET UI] Dataset export completed successfully")

    return render_template(
        "dataset/dataset_generator.html",
        result=f"Dataset export completed successfully ({len(records)} records).",
        exported_files=[str(p) for p in exported],
    )
