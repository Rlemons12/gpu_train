"""
ONE-OFF DEV SCRIPT
Seed QnA taxonomy tables:
  - qna_domains
  - qna_audiences
  - qna_criticalities

Safe to re-run (idempotent).
"""

from dataset_gen.dataset_qanda_generator.qanda_db.qa_db import (
    Domain,
    Audience,
    Criticality,
)
from dataset_gen.dataset_qanda_generator.configuration.pg_db_config import (
    get_qna_session,
)
from dataset_gen.dataset_qanda_generator.configuration.logging_config import (
    get_qna_logger,
)

log = get_qna_logger("seed_qna_taxonomy")


# ---------------------------------------------------------------------
# Seed helpers (idempotent)
# ---------------------------------------------------------------------

def seed_domains(session):
    entries = [
        ("MAINT", "Maintenance procedures and troubleshooting"),
        ("OPS", "Operations and operator instructions"),
        ("SAFETY", "Safety procedures and risk controls"),
        ("QUALITY", "Quality systems and compliance"),
        ("ENGINEERING", "Engineering specifications and setup"),
        ("TRAINING", "Training and competency materials"),
    ]

    for code, desc in entries:
        exists = session.query(Domain).filter_by(code=code).first()
        if not exists:
            session.add(Domain(code=code, description=desc))
            log.info(f"[DOMAIN] inserted {code}")
        else:
            log.info(f"[DOMAIN] exists {code}")


def seed_audiences(session):
    entries = [
        ("OPERATOR", "Machine operators"),
        ("TECH", "Maintenance technicians"),
        ("ENGINEER", "Manufacturing or controls engineers"),
        ("SUPERVISOR", "Supervisors and team leads"),
        ("QUALITY", "Quality assurance personnel"),
        ("TRAINEE", "New or in-training personnel"),
    ]

    for code, desc in entries:
        exists = session.query(Audience).filter_by(code=code).first()
        if not exists:
            session.add(Audience(code=code, description=desc))
            log.info(f"[AUDIENCE] inserted {code}")
        else:
            log.info(f"[AUDIENCE] exists {code}")


def seed_criticalities(session):
    entries = [
        ("LOW", 1, "Low impact – informational or reference"),
        ("MEDIUM", 2, "Moderate impact – affects efficiency or quality"),
        ("HIGH", 3, "High impact – may cause downtime or defects"),
        ("CRITICAL", 4, "Critical – safety risk or major equipment damage"),
    ]

    for code, severity, desc in entries:
        exists = session.query(Criticality).filter_by(code=code).first()
        if not exists:
            session.add(
                Criticality(
                    code=code,
                    severity=severity,
                    description=desc,
                )
            )
            log.info(f"[CRITICALITY] inserted {code}")
        else:
            log.info(f"[CRITICALITY] exists {code}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    log.info("=" * 80)
    log.info("QNA TAXONOMY SEED STARTED")
    log.info("=" * 80)

    session = get_qna_session()

    try:
        seed_domains(session)
        seed_audiences(session)
        seed_criticalities(session)

        session.commit()
        log.info("SUCCESS: Seeded qna_domains, qna_audiences, qna_criticalities")

    except Exception:
        session.rollback()
        log.exception("FAILED: taxonomy seed")
        raise

    finally:
        session.close()
        log.info("=" * 80)


if __name__ == "__main__":
    main()
