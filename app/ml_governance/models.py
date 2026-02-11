import uuid
from sqlalchemy import (
    Column,
    Text,
    Integer,
    TIMESTAMP,
    JSON,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
Base = declarative_base()


class Dataset(Base):
    """
    Immutable dataset registry entry.
    One row per unique dataset hash.
    """
    __tablename__ = "datasets"
    __table_args__ = {"schema": "ml_governance"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)                 # e.g. "sft"
    version = Column(Text, nullable=False)              # e.g. "v003"
    content_hash = Column(Text, nullable=False, unique=True)
    path = Column(Text, nullable=False)
    record_count = Column(Integer, nullable=False)
    schema = Column(JSON, nullable=False)
    generator = Column(Text, nullable=False)
    parent_hash = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now())
    created_by = Column(Text)
    notes = Column(JSONB, nullable=True)


class DatasetAuditLog(Base):
    """
    Append-only audit log for dataset events.
    """
    __tablename__ = "dataset_audit_log"
    __table_args__ = {"schema": "ml_governance"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_hash = Column(Text, nullable=False)
    action = Column(Text, nullable=False)    # REGISTERED | USED | DEPRECATED
    actor = Column(Text)
    context = Column(JSON)
    created_at = Column(TIMESTAMP, server_default=func.now())
