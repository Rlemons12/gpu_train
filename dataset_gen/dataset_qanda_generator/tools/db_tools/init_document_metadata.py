from sqlalchemy.exc import IntegrityError

from dataset_gen.dataset_qanda_generator.configuration.pg_db_config import (
    get_qna_session,
)
from dataset_gen.dataset_qanda_generator.configuration.logging_config import (
    get_qna_logger,
)

from dataset_gen.dataset_qanda_generator.qanda_db.qa_db import (
    Base,
    Domain,
    Audience,
    Criticality,
    DocumentTag,
)

log = get_qna_logger("init_document_metadata")


# ---------------------------------------------------------------------
# Seed data (authoritative)
# ---------------------------------------------------------------------

DOMAINS = [
    ("PLC", "Programmable Logic Controllers"),
    ("ELECTRICAL", "Electrical systems and wiring"),
    ("MECHANICAL", "Mechanical equipment and components"),
    ("CONTROLS", "Industrial control systems"),
    ("PACKAGING", "Packaging equipment and processes"),
    ("SAFETY", "Safety procedures and compliance"),
]

AUDIENCES = [
    ("TECHNICIAN", "Maintenance technicians"),
    ("OPERATOR", "Machine operators"),
    ("ENGINEER", "Engineers"),
    ("SUPERVISOR", "Supervisors and leads"),
    ("TRAINING", "Training and onboarding"),
]

CRITICALITIES = [
    ("LOW", 1, "Low impact"),
    ("NORMAL", 2, "Normal operational importance"),
    ("HIGH", 3, "High operational impact"),
    ("SAFETY", 4, "Safety critical"),
]

DOCUMENT_TAGS = [
    ("SOP", "Standard Operating Procedure"),
    ("PROCEDURAL", "Step-by-step procedural document"),
    ("TECHNICAL_TRAINING", "Technical training material"),
    ("OEM_MANUAL", "OEM supplied documentation"),
    ("SAFETY_CRITICAL", "Safety critical document"),
    ("POLICY", "Policy or governance document"),
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def seed_table(session, factory, rows):
    created = 0
    for row in rows:
        try:
            session.add(factory(row))
            session.flush()
            created += 1
        except IntegrityError:
            session.rollback()
    return created


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    session = get_qna_session()

    # IMPORTANT: use the session's bound engine
    engine = session.get_bind()

    log.info("Creating document metadata tables (if missing)")
    Base.metadata.create_all(bind=engine)

    log.info("Seeding Domains")
    domains_created = seed_table(
        session,
        lambda r: Domain(code=r[0], description=r[1]),
        DOMAINS,
    )

    log.info("Seeding Audiences")
    audiences_created = seed_table(
        session,
        lambda r: Audience(code=r[0], description=r[1]),
        AUDIENCES,
    )

    log.info("Seeding Criticalities")
    criticalities_created = seed_table(
        session,
        lambda r: Criticality(code=r[0], severity=r[1], description=r[2]),
        CRITICALITIES,
    )

    log.info("Seeding Document Tags")
    tags_created = seed_table(
        session,
        lambda r: DocumentTag(code=r[0], description=r[1]),
        DOCUMENT_TAGS,
    )

    session.commit()

    log.info(
        "Seed complete: domains=%d audiences=%d criticalities=%d tags=%d",
        domains_created,
        audiences_created,
        criticalities_created,
        tags_created,
    )

    session.close()


if __name__ == "__main__":
    main()
