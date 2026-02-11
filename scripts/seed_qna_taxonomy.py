"""
ONE-OFF DEV SCRIPT
Seed QnA taxonomy tables.

Tables:
- qna_domains
- qna_audiences
- qna_criticalities
- qna_document_tags

Idempotent (safe to re-run).
"""

from dataset_gen.dataset_qanda_generator.qanda_db.qa_db import (
    Domain,
    Audience,
    Criticality,
    DocumentTag,
)
from dataset_gen.dataset_qanda_generator.configuration.pg_db_config import (
    get_qna_session,
)
from dataset_gen.dataset_qanda_generator.configuration.logging_config import (
    get_qna_logger,
)

log = get_qna_logger("seed_qna_taxonomy")


# ---------------------------------------------------------------------
# Seed definitions
# ---------------------------------------------------------------------

DOMAINS = [
    ("MAINT", "Maintenance procedures and troubleshooting"),
    ("OPS", "Operations and operator instructions"),
    ("SAFETY", "Safety procedures and risk controls"),
    ("QUALITY", "Quality systems and compliance"),
    ("ENGINEERING", "Engineering specifications and setup"),
    ("TRAINING", "Training and competency materials"),
]

AUDIENCES = [
    ("OPERATOR", "Machine operators"),
    ("TECH", "Maintenance technicians"),
    ("ENGINEER", "Manufacturing or controls engineers"),
    ("SUPERVISOR", "Supervisors and team leads"),
    ("QUALITY", "Quality assurance personnel"),
    ("TRAINEE", "New or in-training personnel"),
]

CRITICALITIES = [
    ("LOW", 1, "Low impact – informational or reference"),
    ("MEDIUM", 2, "Moderate impact – affects efficiency or quality"),
    ("HIGH", 3, "High impact – may cause downtime or defects"),
    ("CRITICAL", 4, "Critical – safety risk or major equipment damage"),
]

DOCUMENT_TAGS = [
    ("SOP", "Standard Operating Procedure"),
    ("WORK_INSTRUCTION", "Detailed work instruction"),
    ("MANUAL", "Equipment or vendor manual"),
    ("TROUBLESHOOTING", "Troubleshooting guide"),
    ("CHECKLIST", "Checklist or quick reference"),
    ("SPEC", "Specification or technical reference"),
    ("SAFETY_DOC", "Safety-related documentation"),
]


# ---------------------------------------------------------------------
# Generic helper
# ---------------------------------------------------------------------

def seed_unique(session, model, key_field, rows, factory):
    """
    Insert rows if they do not already exist.
    """
    for row in rows:
        key = row[0]
        exists = (
            session.query(model)
            .filter(getattr(model, key_field) == key)
            .first()
        )
        if not exists:
            session.add(factory(*row))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    log.info("=" * 80)
    log.info("QNA TAXONOMY SEED STARTED")
    log.info("=" * 80)

    session = get_qna_session()

    try:
        seed_unique(
            session,
            Domain,
            "code",
            DOMAINS,
            lambda code, desc: Domain(code=code, description=desc),
        )

        seed_unique(
            session,
            Audience,
            "code",
            AUDIENCES,
            lambda code, desc: Audience(code=code, description=desc),
        )

        seed_unique(
            session,
            Criticality,
            "code",
            CRITICALITIES,
            lambda code, sev, desc: Criticality(
                code=code,
                severity=sev,
                description=desc,
            ),
        )

        seed_unique(
            session,
            DocumentTag,
            "code",
            DOCUMENT_TAGS,
            lambda code, desc: DocumentTag(code=code, description=desc),
        )

        session.commit()
        log.info("✅ QnA taxonomy seeded successfully")

    except Exception:
        session.rollback()
        log.exception("❌ Failed to seed QnA taxonomy")
        raise

    finally:
        session.close()
        log.info("=" * 80)


if __name__ == "__main__":
    main()
