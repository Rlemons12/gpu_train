from flask import Blueprint, render_template, request
from dataset_gen.dataset_qanda_generator.configuration.pg_db_config import get_qna_session
from dataset_gen.dataset_qanda_generator.qanda_db.qa_db import (
    Document, Domain, Audience, Criticality
)

bp = Blueprint("documents", __name__, url_prefix="/documents")


@bp.route("", methods=["GET"])
def list_documents():
    session = get_qna_session()

    # Filters
    title = request.args.get("title", "").strip()
    domain_id = request.args.get("domain_id")
    audience_id = request.args.get("audience_id")
    criticality_id = request.args.get("criticality_id")

    query = session.query(Document)

    if title:
        query = query.filter(Document.title.ilike(f"%{title}%"))
    if domain_id:
        query = query.filter(Document.domain_id == int(domain_id))
    if audience_id:
        query = query.filter(Document.audience_id == int(audience_id))
    if criticality_id:
        query = query.filter(Document.criticality_id == int(criticality_id))

    documents = query.order_by(Document.created_at.desc()).all()

    return render_template(
        "documents/list.html",
        documents=documents,
        domains=session.query(Domain).all(),
        audiences=session.query(Audience).all(),
        criticalities=session.query(Criticality).all(),
        filters={
            "title": title,
            "domain_id": domain_id,
            "audience_id": audience_id,
            "criticality_id": criticality_id,
        },
    )
