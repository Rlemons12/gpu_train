from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

STRUCTURE = {
    "routes": {
        "__init__.py": "",
        "document_metadata.py": """\
# Thin UI routes for document metadata ingestion
# Created by init_document_ui.py

from flask import Blueprint, render_template, request, redirect, url_for
from dataset_gen.dataset_qanda_generator.configuration.pg_db_config import get_qna_session
from dataset_gen.dataset_qanda_generator.qanda_db.qa_db import (
    Document, Domain, Audience, Criticality, DocumentTag
)

bp = Blueprint("document_metadata", __name__, url_prefix="/documents")

@bp.route("/new", methods=["GET"])
def new_document():
    session = get_qna_session()
    return render_template(
        "document_metadata/form.html",
        domains=session.query(Domain).order_by(Domain.code).all(),
        audiences=session.query(Audience).order_by(Audience.code).all(),
        criticalities=session.query(Criticality).order_by(Criticality.severity).all(),
        tags=session.query(DocumentTag).order_by(DocumentTag.code).all(),
    )

@bp.route("/create", methods=["POST"])
def create_document():
    session = get_qna_session()

    doc = Document.create(
        session=session,
        title=request.form["title"],
        file_name=request.form["file_name"],
        file_path=request.form["file_path"],
    )

    doc.domain_id = int(request.form["domain_id"])
    doc.audience_id = int(request.form["audience_id"])
    doc.criticality_id = int(request.form["criticality_id"])

    tag_ids = request.form.getlist("tag_ids")
    if tag_ids:
        doc.tags = session.query(DocumentTag).filter(
            DocumentTag.id.in_(tag_ids)
        ).all()

    session.commit()

    return redirect(url_for("document_metadata.new_document"))
"""
    },
    "templates": {
        "document_metadata": {
            "form.html": """\
<h2>New Document Metadata</h2>

<form method="post" action="{{ url_for('document_metadata.create_document') }}">

  <label>Title</label><br>
  <input name="title" required><br><br>

  <label>File name</label><br>
  <input name="file_name" required><br><br>

  <label>File path</label><br>
  <input name="file_path" required><br><br>

  <label>Domain</label><br>
  <select name="domain_id" required>
    {% for d in domains %}
      <option value="{{ d.id }}">{{ d.code }}</option>
    {% endfor %}
  </select><br><br>

  <label>Audience</label><br>
  <select name="audience_id" required>
    {% for a in audiences %}
      <option value="{{ a.id }}">{{ a.code }}</option>
    {% endfor %}
  </select><br><br>

  <label>Criticality</label><br>
  <select name="criticality_id" required>
    {% for c in criticalities %}
      <option value="{{ c.id }}">{{ c.code }}</option>
    {% endfor %}
  </select><br><br>

  <label>Tags</label><br>
  {% for t in tags %}
    <input type="checkbox" name="tag_ids" value="{{ t.id }}"> {{ t.code }}<br>
  {% endfor %}
  <br>

  <button type="submit">Save Document</button>
</form>
"""
        }
    },
    "static": {},
    "README_document_ui.md": """\
Document Metadata UI
====================

Purpose
-------
This UI provides a controlled ingestion gate for documents before
they enter the Q&A dataset pipeline.

Users must specify:
- Title
- Domain
- Audience
- Criticality
- Tags

Location
--------
Routes:
    /documents/new

Integration
-----------
The pipeline consumes metadata from qna_documents and related tables.
This UI ensures metadata completeness before processing.

Safe to extend.
"""
}

def ensure(path: Path, content: str | None = None):
    if path.exists():
        return
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content or "", encoding="utf-8")
    else:
        path.mkdir(parents=True, exist_ok=True)

def build(base: Path, tree: dict):
    for name, value in tree.items():
        p = base / name
        if isinstance(value, dict):
            ensure(p)
            build(p, value)
        else:
            ensure(p, value)

if __name__ == "__main__":
    build(BASE_DIR, STRUCTURE)
    print("✅ Document metadata UI initialized.")
