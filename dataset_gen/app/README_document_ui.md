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

DOTENV_PATH=/mnt/c/Users/operator/PycharmProjects/gpu_train/.env \
python -m dataset_gen.app.app