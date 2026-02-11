from sqlalchemy.orm import Session
from app.ml_governance.models import DatasetAuditLog


class AuditLogger:
    """
    Centralized audit logging for ML governance events.
    """

    def __init__(self, db: Session, actor: str = "system"):
        self.db = db
        self.actor = actor

    def log(self, *, dataset_hash: str, action: str, context: dict | None = None):
        entry = DatasetAuditLog(
            dataset_hash=dataset_hash,
            action=action,
            actor=self.actor,
            context=context or {},
        )
        self.db.add(entry)
        self.db.commit()
