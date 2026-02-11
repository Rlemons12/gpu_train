#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q&A Pipeline Database Models (Option C)
---------------------------------------

This version:

    ✔ Uses unified QNA_ENGINE from pg_db_config
    ✔ Removes hardcoded DB URLs
    ✔ Provides clean ORM with built-in creation helpers
    ✔ Embeddings use correct meta_json field
    ✔ Correct polymorphic joins using foreign()
    ✔ Eliminates SAWarnings using overlaps=...
    ✔ Schema creation uses QNA_ENGINE
    ✔ Tracks pipeline runs + links every object back to exactly one run
"""

from datetime import datetime
from typing import List
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    func,
    Enum,
    Boolean,
    Float,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import declarative_base, relationship, foreign
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

# Unified DB connection for Option C
from dataset_gen.dataset_qanda_generator.configuration.pg_db_config import (
    QNA_ENGINE,
    QNA_SessionLocal,
)
from dataset_gen.dataset_qanda_generator.configuration.logging_config import get_qna_logger

# ----------------------------------------------------------------------
# SQLAlchemy Base
# ----------------------------------------------------------------------
Base = declarative_base()


def get_session():
    """Return a new QNA session."""
    return QNA_SessionLocal()

# =========================================================
# Document Classification Tables
# =========================================================

class Domain(Base):
    __tablename__ = "qna_domains"

    id = Column(Integer, primary_key=True)
    code = Column(String, unique=True, nullable=False)
    description = Column(String)

    def __repr__(self):
        return f"<Domain {self.code}>"

class Audience(Base):
    __tablename__ = "qna_audiences"

    id = Column(Integer, primary_key=True)
    code = Column(String, unique=True, nullable=False)
    description = Column(String)


class Criticality(Base):
    __tablename__ = "qna_criticalities"

    id = Column(Integer, primary_key=True)
    code = Column(String, unique=True, nullable=False)
    severity = Column(Integer, nullable=False)
    description = Column(String)


class DocumentTag(Base):
    __tablename__ = "qna_document_tags"

    id = Column(Integer, primary_key=True)
    code = Column(String, unique=True, nullable=False)
    description = Column(String)

    documents = relationship(
        "Document",
        secondary="qna_document_tag_assoc",
        back_populates="tags",
    )


class DocumentTagAssociation(Base):
    __tablename__ = "qna_document_tag_assoc"

    document_id = Column(
        Integer,
        ForeignKey("qna_documents.id", ondelete="CASCADE"),
        primary_key=True,
    )

    tag_id = Column(
        Integer,
        ForeignKey("qna_document_tags.id", ondelete="CASCADE"),
        primary_key=True,
    )


# ======================================================================
# DOCUMENT
# ======================================================================
class Document(Base):
    __tablename__ = "qna_documents"

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=True)
    file_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        overlaps="embeddings",
    )

    embeddings = relationship(
        "Embedding",
        primaryjoin="and_(foreign(Embedding.parent_id)==Document.id, "
                    "Embedding.parent_type=='document')",
        cascade="all, delete-orphan",
        lazy="dynamic",
        overlaps="embeddings,chunks",
    )

    # All runs that have been executed on this document
    runs = relationship(
        "PipelineRun",
        back_populates="document",
        cascade="all, delete-orphan",
    )

    run_item = relationship(
        "PipelineRunItem",
        uselist=False,
        primaryjoin="and_(foreign(PipelineRunItem.parent_id)==Document.id, "
                    "PipelineRunItem.parent_type=='document')",
        viewonly=False,
        overlaps="runs"
    )

    domain_id = Column(Integer, ForeignKey("qna_domains.id"), nullable=True)
    audience_id = Column(Integer, ForeignKey("qna_audiences.id"), nullable=True)
    criticality_id = Column(Integer, ForeignKey("qna_criticalities.id"), nullable=True)

    domain = relationship("Domain")
    audience = relationship("Audience")
    criticality = relationship("Criticality")

    tags = relationship(
        "DocumentTag",
        secondary="qna_document_tag_assoc",
        back_populates="documents",
    )

    @classmethod
    def create(cls, session, file_name: str, file_path: str, title: str | None = None):
        doc = cls(
            file_name=file_name,
            file_path=file_path,
            title=title or file_name,
        )
        session.add(doc)
        session.flush()
        return doc

    def add_chunk(self, session, chunk_id, context, page=None, section=None, subsection=None):
        chk = Chunk(
            document_id=self.id,
            chunk_id=chunk_id,
            context=context,
            page=page,
            section=section,
            subsection=subsection,
        )
        session.add(chk)
        session.flush()
        return chk

    def add_embedding(self, session, model_name, embedding, meta_json=None):
        return Embedding.create(
            session=session,
            parent_type="document",
            parent_id=self.id,
            model_name=model_name,
            embedding_vector=embedding,
            meta_json=meta_json,
        )


# ======================================================================
# CHUNK
# ======================================================================
class Chunk(Base):
    __tablename__ = "qna_chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("qna_documents.id"), nullable=True)
    chunk_id = Column(String, nullable=False)
    page = Column(Integer)
    section = Column(String)
    subsection = Column(String)
    context = Column(Text, nullable=False)

    document = relationship(
        "Document",
        back_populates="chunks",
        overlaps="embeddings",
    )

    questions = relationship(
        "Question",
        back_populates="chunk",
        cascade="all, delete-orphan",
        overlaps="embeddings",
    )

    embeddings = relationship(
        "Embedding",
        primaryjoin="and_(foreign(Embedding.parent_id)==Chunk.id, "
                    "Embedding.parent_type=='chunk')",
        cascade="all, delete-orphan",
        lazy="dynamic",
        overlaps="embeddings,document,questions",
    )

    # Each chunk is produced in exactly one run
    run_item = relationship(
        "PipelineRunItem",
        uselist=False,
        primaryjoin="and_(foreign(PipelineRunItem.parent_id)==Chunk.id, "
                    "PipelineRunItem.parent_type=='chunk')",
        viewonly=False,
    )

    def add_question(self, session, question_text: str, index=None):
        q = Question(
            chunk_id=self.id,
            question_index=index,
            question=question_text,
        )
        session.add(q)
        session.flush()
        return q

    def add_embedding(self, session, model_name, embedding, meta_json=None):
        return Embedding.create(
            session=session,
            parent_type="chunk",
            parent_id=self.id,
            model_name=model_name,
            embedding_vector=embedding,
            meta_json=meta_json,
        )


# ======================================================================
# QUESTION
# ======================================================================
class Question(Base):
    __tablename__ = "qna_questions"

    id = Column(Integer, primary_key=True)
    chunk_id = Column(Integer, ForeignKey("qna_chunks.id"), nullable=False)
    question_index = Column(Integer)
    question = Column(String, nullable=False)

    # One-to-one link to AnswerRanking
    ranking = relationship(
        "AnswerRanking",
        uselist=False,
        back_populates="question",
        overlaps="chunk,embeddings",
    )

    chunk = relationship(
        "Chunk",
        back_populates="questions",
        overlaps="embeddings",
    )

    answers = relationship(
        "Answer",
        back_populates="question",
        cascade="all, delete-orphan",
        overlaps="embeddings",
    )

    embeddings = relationship(
        "Embedding",
        primaryjoin="and_(foreign(Embedding.parent_id)==Question.id, "
                    "Embedding.parent_type=='question')",
        cascade="all, delete-orphan",
        lazy="dynamic",
        overlaps="embeddings,chunk,answers",
    )

    # Each question is produced in exactly one run
    run_item = relationship(
        "PipelineRunItem",
        uselist=False,
        primaryjoin="and_(foreign(PipelineRunItem.parent_id)==Question.id, "
                    "PipelineRunItem.parent_type=='question')",
        viewonly=False,
    )

    def add_answer(self, session, model_name, answer_text,
                   model_type: str = "causal_lm", model_path: str | None = None,
                   score: float | None = None):
        """
        Insert an answer from a specific model for this question.
        """
        answer = Answer(
            question_id=self.id,
            model_name=model_name,
            model_type=model_type,
            model_path=model_path,
            answer_text=answer_text,
            score=score,
        )
        session.add(answer)
        session.flush()
        return answer

    def add_embedding(self, session, model_name, embedding, meta_json=None):
        return Embedding.create(
            session=session,
            parent_type="question",
            parent_id=self.id,
            model_name=model_name,
            embedding_vector=embedding,
            meta_json=meta_json,
        )


# ======================================================================
# ANSWER
# ======================================================================
class Answer(Base):
    __tablename__ = "qna_answers"

    id = Column(Integer, primary_key=True)
    question_id = Column(Integer, ForeignKey("qna_questions.id"), nullable=False)

    model_name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    model_path = Column(String)

    answer_text = Column(Text, nullable=False)

    # Ranking metadata
    score = Column(Float)                    # similarity score or ranking signal
    is_best = Column(Boolean, default=False)
    is_worst = Column(Boolean, default=False)

    question = relationship("Question", back_populates="answers")

    # Each answer is produced in exactly one run
    run_item = relationship(
        "PipelineRunItem",
        uselist=False,
        primaryjoin="and_(foreign(PipelineRunItem.parent_id)==Answer.id, "
                    "PipelineRunItem.parent_type=='answer')",
        viewonly=False,
    )


# ======================================================================
# ANSWER RANKING
# ======================================================================
class AnswerRanking(Base):
    __tablename__ = "qna_answer_ranking"

    id = Column(Integer, primary_key=True)

    # Each question may have only one ranking record
    question_id = Column(
        Integer,
        ForeignKey("qna_questions.id"),
        nullable=False,
        unique=True,
        index=True,
    )

    # Best
    best_model = Column(String, nullable=False)
    best_answer = Column(Text, nullable=False)

    # Worst
    worst_model = Column(String, nullable=False)
    worst_answer = Column(Text, nullable=False)

    # JSON: {"flan": 0.82, "qwen": 0.77, ...}
    answer_scores = Column(JSONB, nullable=False)

    # Backref to Question
    question = relationship(
        "Question",
        back_populates="ranking",
        uselist=False,
        overlaps="chunk,embeddings",
    )

    # Each ranking is produced in exactly one run
    run_item = relationship(
        "PipelineRunItem",
        uselist=False,
        primaryjoin="and_(foreign(PipelineRunItem.parent_id)==AnswerRanking.id, "
                    "PipelineRunItem.parent_type=='ranking')",
        viewonly=False,
    )


# ======================================================================
# MODEL (LLM Model Registry with dynamic loading)
# ======================================================================
class LLMModel(Base):
    __tablename__ = "qna_models"

    id = Column(Integer, primary_key=True)

    # Unique model key: "flan", "qwen", "tinyllama", etc.
    name = Column(String, unique=True, nullable=False)

    # Optional descriptor: "causal_lm", "encoder", etc.
    model_type = Column(String)

    # Local file path to model directory (optional)
    model_path = Column(String)

    # Python class path for dynamic loading
    # Example: "dataset_qanda_generator.models.qwen.QwenAnswerGenerator"
    class_path = Column(String, nullable=False)

    # Toggle loading
    enabled = Column(Boolean, default=True)

    # -------- GLOBAL LOAD-ONCE CACHE --------
    _MODEL_CACHE = {}

    # ---------------------------------------------------------
    # get_or_create (kept for compatibility)
    # ---------------------------------------------------------
    @classmethod
    def get_or_create(cls, session, name, model_type=None,
                      model_path=None, class_path=None, enabled=True):
        """
        Fetch or create an LLMModel definition.
        Now supports class_path + enabled flags.
        """
        m = session.query(cls).filter_by(name=name).first()
        if m:
            return m

        m = cls(
            name=name,
            model_type=model_type,
            model_path=model_path,
            class_path=class_path or "",
            enabled=enabled,
        )
        session.add(m)
        session.flush()
        return m

    # ---------------------------------------------------------
    # Dynamic class import
    # ---------------------------------------------------------
    @classmethod
    def import_class(cls, class_path: str):
        """
        Import Python class from full dotted module path.
        """
        import importlib

        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    # ---------------------------------------------------------
    # Load a single model instance (per-row)
    # ---------------------------------------------------------
    def load_instance(self):
        """
        Load the model described by this DB row.
        Uses global class-level cache to ensure load-once behavior.
        """
        key = self.name.lower()

        # Return cached model if it exists
        if key in self._MODEL_CACHE:
            return self._MODEL_CACHE[key]

        # Import Python class
        try:
            model_cls = self.import_class(self.class_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to import model '{self.name}' "
                f"(class_path={self.class_path}): {exc}"
            )

        # Instantiate model
        try:
            if self.model_path:
                instance = model_cls(model_dir=self.model_path)
            else:
                instance = model_cls()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to instantiate model '{self.name}': {exc}"
            )

        # Cache the loaded instance
        self._MODEL_CACHE[key] = instance
        return instance

    # ---------------------------------------------------------
    # Load all enabled models (true load-once API)
    # ---------------------------------------------------------
    @classmethod
    def load_all_enabled(cls):
        """
        Loads ALL enabled models from qna_models and returns a dict:

            {
                "flan": <ModelInstance>,
                "qwen": <ModelInstance>,
                "mistral": <ModelInstance>,
                ...
            }

        Each instance is loaded EXACTLY ONCE.
        """
        # Already loaded?
        if cls._MODEL_CACHE:
            return cls._MODEL_CACHE

        from dataset_gen.dataset_qanda_generator.configuration.logging_config import get_qna_logger
        from dataset_gen.dataset_qanda_generator.configuration.pg_db_config import get_qna_session
        session = get_qna_session()

        try:
            rows = (
                session.query(cls)
                .filter(cls.enabled == True)
                .all()
            )
        finally:
            session.close()

        for row in rows:
            try:
                row.load_instance()
            except Exception as exc:
                print(f"[LLMModel] FAILED to load '{row.name}': {exc}")

        return cls._MODEL_CACHE

    # ---------------------------------------------------------
    # NEW: Return names only for enabled models
    # ---------------------------------------------------------
    @classmethod
    def get_default_model_names(cls) -> List[str]:
        """
        Returns the lowercase names of all enabled LLM models.
        Used by CLI when --models is not provided.
        """
        enabled = cls.load_all_enabled()  # dict: name -> model_instance
        return list(enabled.keys())


# ======================================================================
# EMBEDDING (Unified Polymorphic Table)
# ======================================================================
# parent_type here is "where the embedding attaches":
#   document / chunk / question / answer
PARENT_TYPES = ("document", "chunk", "question", "answer")

class Embedding(Base):
    __tablename__ = "qna_embeddings"

    id = Column(Integer, primary_key=True)

    parent_type = Column(
        Enum(*PARENT_TYPES, name="parent_type_enum"),
        nullable=False,
    )
    parent_id = Column(Integer, nullable=False)

    model_name = Column(String, nullable=False)
    embedding_vector = Column(Vector, nullable=True)
    actual_dimensions = Column(Integer)

    meta_json = Column(JSONB, nullable=True, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # -------------------------------------------------------------
    # Pipeline run linkage (polymorphic)
    # -------------------------------------------------------------
    run_item = relationship(
        "PipelineRunItem",
        uselist=False,
        primaryjoin=(
            "and_("
            "foreign(PipelineRunItem.parent_id) == Embedding.id, "
            "PipelineRunItem.parent_type == 'embedding'"
            ")"
        ),
        overlaps="run_item",
    )

    # -------------------------------------------------------------
    # Factory
    # -------------------------------------------------------------
    @classmethod
    def create(
        cls,
        session,
        parent_type,
        parent_id,
        model_name,
        embedding_vector,
        meta_json=None,
    ):
        if hasattr(embedding_vector, "tolist"):
            embedding_vector = embedding_vector.tolist()

        inst = cls(
            parent_type=parent_type,
            parent_id=parent_id,
            model_name=model_name,
            embedding_vector=embedding_vector,
            actual_dimensions=len(embedding_vector)
            if embedding_vector is not None
            else None,
            meta_json=meta_json or {},
        )

        session.add(inst)
        session.flush()
        return inst


# ======================================================================
# PIPELINE RUN + RUN ITEMS
# ======================================================================

class PipelineRun(Base):
    __tablename__ = "qna_pipeline_runs"

    id = Column(Integer, primary_key=True)

    document_id = Column(
        Integer,
        ForeignKey("qna_documents.id", ondelete="CASCADE"),
        nullable=True,
    )

    run_type = Column(String, nullable=False)

    options_json = Column(JSONB, nullable=False, default=dict)
    models_json = Column(JSONB, nullable=False, default=dict)
    env_json = Column(JSONB, nullable=False, default=dict)

    started_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )
    finished_at = Column(DateTime(timezone=True))

    success = Column(Boolean, default=False)
    error_message = Column(Text)

    # -------------------------------------------------
    # Relationships
    # -------------------------------------------------
    document = relationship(
        "Document",
        back_populates="runs",
    )

    items = relationship(
        "PipelineRunItem",
        back_populates="run",
        cascade="all, delete-orphan",
    )

class PipelineRunItem(Base):
    __tablename__ = "qna_pipeline_run_items"

    id = Column(Integer, primary_key=True)

    run_id = Column(Integer, ForeignKey("qna_pipeline_runs.id"), nullable=False)
    run = relationship("PipelineRun", back_populates="items")

    # polymorphic pointer to created item
    parent_type = Column(
        Enum(
            "document",  # NEW
            "chunk",
            "question",
            "answer",
            "embedding",
            "ranking",
            name="run_item_parent_type"
        ),
        nullable=False,
    )

    parent_id = Column(Integer, nullable=False)

    created_at = Column(DateTime, default=func.now())

    # Ensure exactly ONE run per object
    __table_args__ = (
        UniqueConstraint("parent_type", "parent_id", name="uq_run_item_parent"),
    )



# ======================================================================
# CREATE SCHEMA
# ======================================================================
def create_schema():
    print("Creating QNA schema...")
    Base.metadata.create_all(QNA_ENGINE)
    print("DONE.")


if __name__ == "__main__":
    create_schema()
