from sqlalchemy import text
from app.config.pg_db_config import get_governance_session


REQUIRED_TABLES = {
    "datasets",
    "dataset_usage",
}


def check_governance_db():
    """
    Fail-fast check for governance DB health.

    Verifies:
    - DB connection
    - ml_governance schema exists
    - required tables exist
    """
    with get_governance_session() as db:
        # 1) Schema exists
        schema_exists = db.execute(
            text("""
                SELECT 1
                FROM information_schema.schemata
                WHERE schema_name = 'ml_governance'
            """)
        ).scalar()

        if not schema_exists:
            raise RuntimeError(
                "Schema 'ml_governance' does not exist. "
                "Run governance migrations before training."
            )

        # 2) Required tables exist
        rows = db.execute(
            text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'ml_governance'
            """)
        ).fetchall()

        existing_tables = {r[0] for r in rows}
        missing = REQUIRED_TABLES - existing_tables

        if missing:
            raise RuntimeError(
                f"Missing governance tables: {sorted(missing)}. "
                "Database is not ready for training."
            )
