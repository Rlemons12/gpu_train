"""
ONE-OFF DEV SCRIPT
Recreate all QnA database tables from SQLAlchemy models.
"""

from dataset_gen.dataset_qanda_generator.qanda_db.qa_db import Base
from dataset_gen.dataset_qanda_generator.configuration.pg_db_config import get_qna_session
from dataset_gen.dataset_qanda_generator.configuration.logging_config import get_qna_logger


log = get_qna_logger("schema_bootstrap")


def main():
    log.info("=" * 80)
    log.info("QNA SCHEMA BOOTSTRAP STARTED")
    log.info("=" * 80)

    # Create a session and extract its engine
    session = get_qna_session()
    engine = session.get_bind()

    log.info("Dropping all existing qna_* tables (DEV ONLY)")
    Base.metadata.drop_all(engine)

    log.info("Creating all qna_* tables from SQLAlchemy metadata")
    Base.metadata.create_all(engine)

    session.close()

    log.info("SUCCESS: QnA schema recreated")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
