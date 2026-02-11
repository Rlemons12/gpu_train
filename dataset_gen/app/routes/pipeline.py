from flask import Blueprint, render_template, request, redirect, url_for

from dataset_gen.dataset_qanda_generator.configuration.pg_db_config import get_qna_session
from dataset_gen.dataset_qanda_generator.qanda_db.qa_db import (
    Document,
    Domain,
    Audience,
    Criticality,
)

from dataset_gen.dataset_qanda_generator.pipeline.qanda_main_pipeline import run_full_pipeline
from dataset_gen.dataset_qanda_generator.qanda_db.qa_db import LLMModel

bp = Blueprint(
    "pipeline",
    __name__,
    url_prefix="/pipeline",
)

# Pagination defaults
DEFAULT_PAGE_SIZE = 25

# ------------------------------------------------------------
# PIPELINE PRESETS
# ------------------------------------------------------------

PIPELINE_PRESETS = {
    "quick_test": {
        "label": "Quick Test (Fast, Minimal)",
        "description": "Runs a single chunk with one question. Use to verify wiring.",
        "values": {
            "max_chunks": 1,
            "min_context_len": 20,
            "num_questions": 1,
            "max_q_retries": 0,
            "similarity_threshold": 0.0,
            "embed": False,
            "test_mode": True,
        },
    },

    "balanced": {
        "label": "Balanced (Recommended)",
        "description": "Good quality dataset with reasonable runtime.",
        "values": {
            "max_chunks": None,
            "min_context_len": 40,
            "num_questions": 3,
            "max_q_retries": 3,
            "similarity_threshold": 0.7,
            "embed": False,
            "test_mode": False,
        },
    },

    "high_quality": {
        "label": "High Quality (Slow, Expensive)",
        "description": "Maximum question diversity and aggressive deduplication.",
        "values": {
            "max_chunks": None,
            "min_context_len": 60,
            "num_questions": 5,
            "max_q_retries": 5,
            "similarity_threshold": 0.85,
            "embed": True,
            "test_mode": False,
        },
    },
}


# ------------------------------------------------------------
# PIPELINE SELECTION PAGE (WITH PAGINATION)
# ------------------------------------------------------------
@bp.route("/run", methods=["GET"])
def select_documents():
    session = get_qna_session()

    # --- Filters ---
    title = request.args.get("title", "").strip()
    domain_id = request.args.get("domain_id")
    audience_id = request.args.get("audience_id")
    criticality_id = request.args.get("criticality_id")

    # --- Pagination ---
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("page_size", DEFAULT_PAGE_SIZE))
    offset = (page - 1) * page_size

    query = session.query(Document)

    if title:
        query = query.filter(Document.title.ilike(f"%{title}%"))
    if domain_id:
        query = query.filter(Document.domain_id == int(domain_id))
    if audience_id:
        query = query.filter(Document.audience_id == int(audience_id))
    if criticality_id:
        query = query.filter(Document.criticality_id == int(criticality_id))

    total = query.count()

    documents = (
        query
        .order_by(Document.created_at.desc())
        .limit(page_size)
        .offset(offset)
        .all()
    )

    total_pages = (total + page_size - 1) // page_size

    return render_template(
        "pipeline/run.html",
        documents=documents,
        domains=session.query(Domain).order_by(Domain.code).all(),
        audiences=session.query(Audience).order_by(Audience.code).all(),
        criticalities=session.query(Criticality).order_by(Criticality.severity).all(),
        filters={
            "title": title,
            "domain_id": domain_id,
            "audience_id": audience_id,
            "criticality_id": criticality_id,
        },
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
        },
    )


# ------------------------------------------------------------
# EXECUTE PIPELINE
# ------------------------------------------------------------
@bp.route("/execute", methods=["POST"])
def execute_pipeline():
    import threading
    import uuid
    from datetime import datetime
    import sys

    selected_model = request.form.get("models")

    if not selected_model:
        raise RuntimeError("No answer model selected")

    session = get_qna_session()

    run_id = str(uuid.uuid4())
    document_ids = request.form.getlist("document_ids")
    preset_key = request.form.get("preset")

    preset = PIPELINE_PRESETS.get(preset_key, {}).get("values", {})

    def _int(name, default):
        v = request.form.get(name)
        return int(v) if v not in (None, "", "null") else default

    options = {
        "max_chunks": _int("max_chunks", preset.get("max_chunks")),
        "min_context_len": _int("min_context_len", preset["min_context_len"]),
        "num_questions": _int("num_questions", preset["num_questions"]),
        "max_q_retries": _int("max_q_retries", preset["max_q_retries"]),
        "similarity_threshold": float(
            request.form.get("similarity_threshold", preset["similarity_threshold"])
        ),
        "embed": bool(request.form.get("embed")) or preset["embed"],
        "test_mode": bool(request.form.get("test_mode")) or preset["test_mode"],
    }

    # -----------------------------
    # HARD TERMINAL LOG (SYNC)
    # -----------------------------
    print("=" * 80, flush=True)
    print(f"[PIPELINE] EXECUTE CALLED @ {datetime.now().isoformat()}", flush=True)
    print(f"[PIPELINE] run_id={run_id}", flush=True)
    print(f"[PIPELINE] preset={preset_key}", flush=True)
    print(f"[PIPELINE] document_ids={document_ids}", flush=True)
    print(f"[PIPELINE] options={options}", flush=True)
    print("=" * 80, flush=True)

    def background_job(document_ids, options, selected_model, run_id):
        print(
            f"[PIPELINE] BACKGROUND THREAD STARTED "
            f"run_id={run_id} model={selected_model} "
            f"@ {datetime.now().isoformat()}",
            flush=True
        )

        for doc_id in document_ids:
            doc = session.query(Document).get(int(doc_id))
            if not doc:
                print(
                    f"[PIPELINE] WARNING: document_id={doc_id} not found",
                    flush=True
                )
                continue

            print(
                f"[PIPELINE] Processing document_id={doc.id} "
                f"title='{doc.title}'",
                flush=True
            )
            print(
                f"[PIPELINE] file_path={doc.file_path}",
                flush=True
            )

            # ---- ACTUAL PIPELINE CALL ----
            run_full_pipeline(
                doc.file_path,
                models=[selected_model] if selected_model else None,
                **options
            )

            print(
                f"[PIPELINE] Completed document_id={doc.id}",
                flush=True
            )

        print(
            f"[PIPELINE] BACKGROUND THREAD COMPLETE "
            f"run_id={run_id} @ {datetime.now().isoformat()}",
            flush=True
        )

    threading.Thread(
        target=background_job,
        args=(document_ids, options, selected_model, run_id),
        daemon=True,
    ).start()

    # Immediately return control to UI
    return redirect(url_for("pipeline.select_documents"))


@bp.route("/configure", methods=["POST"])
def configure_pipeline():
    """
    Receives selected document IDs and shows the
    pipeline configuration page (flags + presets + models).
    """
    document_ids = request.form.getlist("document_ids")

    if not document_ids:
        return redirect(url_for("pipeline.select_documents"))

    session = get_qna_session()

    try:
        documents = (
            session.query(Document)
            .filter(Document.id.in_([int(d) for d in document_ids]))
            .order_by(Document.title)
            .all()
        )

        # ------------------------------------
        # Load enabled answer models from DB
        # ------------------------------------
        answer_models = (
            session.query(LLMModel)
            .filter(LLMModel.enabled.is_(True))
            .order_by(LLMModel.name)
            .all()
        )

        return render_template(
            "pipeline/configure.html",
            documents=documents,
            presets=PIPELINE_PRESETS,
            answer_models=answer_models,
        )

    finally:
        session.close()

