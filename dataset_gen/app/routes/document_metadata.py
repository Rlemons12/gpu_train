# Thin UI routes for document metadata ingestion
# Drop-in replacement – request-safe, WSL-path normalized
from pathlib import Path
import threading
import uuid
from datetime import datetime
from dataset_gen.dataset_qanda_generator.pipeline.qanda_main_pipeline import run_full_pipeline
from dataset_gen.dataset_qanda_generator.qanda_db.qa_db import LLMModel
from flask import Blueprint, render_template, request, redirect, url_for

from dataset_gen.dataset_qanda_generator.configuration.pg_db_config import (
    get_qna_session,
)
from dataset_gen.dataset_qanda_generator.qanda_db.qa_db import (
    Document,
    Domain,
    Audience,
    Criticality,
    DocumentTag,
)
from dataset_gen.app.utils.path_utils import normalize_to_wsl_path


bp = Blueprint(
    "document_metadata",
    __name__,
    url_prefix="/documents",
)


# ------------------------------------------------------------
# NEW DOCUMENT FORM
# ------------------------------------------------------------
@bp.route("/new", methods=["GET"])
def new_document():
    session = get_qna_session()

    return render_template(
        "document_metadata/form.html",
        domains=session.query(Domain).order_by(Domain.code).all(),
        audiences=session.query(Audience).order_by(Audience.code).all(),
        criticalities=session.query(Criticality)
            .order_by(Criticality.severity)
            .all(),
        tags=session.query(DocumentTag).order_by(DocumentTag.code).all(),
    )


# ------------------------------------------------------------
# CREATE DOCUMENT + METADATA
# ------------------------------------------------------------
@bp.route("/create", methods=["POST"])
def create_document():
    session = get_qna_session()

    # --- Read form values (request context SAFE) ---
    title = request.form["title"].strip()
    file_name = request.form["file_name"].strip()
    raw_file_path = request.form["file_path"].strip()

    # --- Normalize Windows → WSL path ---
    normalized_path = normalize_to_wsl_path(raw_file_path)

    # --- Create document record ---
    doc = Document.create(
        session=session,
        title=title,
        file_name=file_name,
        file_path=normalized_path,
    )

    # --- Assign required metadata ---
    doc.domain_id = int(request.form["domain_id"])
    doc.audience_id = int(request.form["audience_id"])
    doc.criticality_id = int(request.form["criticality_id"])

    # --- Assign tags (optional) ---
    tag_ids = request.form.getlist("tag_ids")
    if tag_ids:
        doc.tags = (
            session.query(DocumentTag)
            .filter(DocumentTag.id.in_(tag_ids))
            .all()
        )

    session.commit()

    return redirect(url_for("document_metadata.new_document"))




# ------------------------------------------------------------
# BATCH INGEST + FULL PIPELINE + EMBEDDING
# ------------------------------------------------------------
@bp.route("/batch", methods=["GET", "POST"])
def batch_ingest():
    session = get_qna_session()

    if request.method == "GET":
        return render_template(
            "document_metadata/batch_form.html",
            domains=session.query(Domain).order_by(Domain.code).all(),
            audiences=session.query(Audience).order_by(Audience.code).all(),
            criticalities=session.query(Criticality)
                .order_by(Criticality.severity)
                .all(),
            tags=session.query(DocumentTag).order_by(DocumentTag.code).all(),
            models=session.query(LLMModel)
                .filter(LLMModel.enabled.is_(True))
                .order_by(LLMModel.name)
                .all(),
        )

    # -----------------------------
    # Parse inputs
    # -----------------------------
    raw_folder = request.form["folder_path"].strip()
    normalized_folder = normalize_to_wsl_path(raw_folder)
    folder = Path(normalized_folder)

    if not folder.exists():
        return f"Folder does not exist: {folder}", 400

    selected_model = request.form.get("model")
    if not selected_model:
        return "No model selected", 400

    domain_id = int(request.form["domain_id"])
    audience_id = int(request.form["audience_id"])
    criticality_id = int(request.form["criticality_id"])
    tag_ids = request.form.getlist("tag_ids")

    allowed_ext = {".pdf", ".docx", ".pptx", ".txt", ".xlsx", ".csv"}

    created_docs = []
    print(f"[BATCH] Raw path: {raw_folder}", flush=True)
    print(f"[BATCH] Normalized path: {normalized_folder}", flush=True)
    print(f"[BATCH] Exists? {Path(normalized_folder).exists()}", flush=True)

    # -----------------------------
    # Create document records
    # -----------------------------
    for file in folder.rglob("*"):
        if not file.is_file():
            continue
        if file.suffix.lower() not in allowed_ext:
            continue

        exists = session.query(Document).filter_by(
            file_path=str(file)
        ).first()

        if exists:
            continue

        doc = Document.create(
            session=session,
            title=file.stem,
            file_name=file.name,
            file_path=str(file),
        )

        doc.domain_id = domain_id
        doc.audience_id = audience_id
        doc.criticality_id = criticality_id

        if tag_ids:
            doc.tags = (
                session.query(DocumentTag)
                .filter(DocumentTag.id.in_(tag_ids))
                .all()
            )

        created_docs.append({
    "id": doc.id,
    "file_path": doc.file_path,
    "title": doc.title
})

    session.commit()

    # -----------------------------
    # Background full pipeline
    # -----------------------------
    run_id = str(uuid.uuid4())

    def background_job(documents_data, model_name, run_id):
        print("=" * 80, flush=True)
        print(f"[BATCH PIPELINE] START run_id={run_id}", flush=True)

        for doc in documents_data:
            print(
                f"[BATCH PIPELINE] Processing document_id={doc['id']} "
                f"title='{doc['title']}'",
                flush=True
            )

            run_full_pipeline(
                doc["file_path"],
                models=[model_name],
                max_chunks=None,
                min_context_len=60,
                num_questions=5,
                max_q_retries=5,
                similarity_threshold=0.85,
                embed=True,
                test_mode=False
            )

            print(
                f"[BATCH PIPELINE] Completed document_id={doc['id']}",
                flush=True
            )

        print(f"[BATCH PIPELINE] COMPLETE run_id={run_id}", flush=True)
        print("=" * 80, flush=True)

    threading.Thread(
        target=background_job,
        args=(created_docs, selected_model, run_id),
        daemon=True,
    ).start()

    return render_template(
        "document_metadata/batch_form.html",
        result=f"Batch ingest complete. Pipeline started for {len(created_docs)} documents.",
    )
