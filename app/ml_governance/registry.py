from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.ml_governance.models import Dataset
from app.ml_governance.audit import AuditLogger


class DatasetRegistry:
    """
    Authoritative registry enforcing:
    - Dataset immutability
    - Hash-based deduplication
    - Governance & auditing

    This registry OWNS:
    - Dataset version allocation
    - Canonical dataset paths
    - Finalization semantics

    Callers must NEVER register temporary paths.
    """

    def __init__(self, db: Session, actor: str = "system"):
        self.db = db
        self.actor = actor
        self.audit = AuditLogger(db, actor)

    # --------------------------------------------------
    # Static helpers
    # --------------------------------------------------
    @staticmethod
    def _hash_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _count_records(path: str) -> int:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    @staticmethod
    def _infer_schema(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            first = json.loads(next(f))
        return {k: type(v).__name__ for k, v in first.items()}

    # --------------------------------------------------
    # Version allocation (NO INSERT)
    # --------------------------------------------------
    def allocate_version(self, *, name: str) -> str:
        """
        Allocate the next dataset version for a dataset name.

        Versions are monotonically increasing:
        v001, v002, v003, ...
        """
        q = (
            self.db.query(Dataset.version)
            .filter(Dataset.name == name)
            .order_by(Dataset.created_at.desc())
            .limit(1)
        )

        last = q.scalar()
        if not last:
            return "v001"

        try:
            n = int(last.lstrip("v"))
        except ValueError:
            raise RuntimeError(f"Invalid dataset version format in DB: {last}")

        return f"v{n + 1:03d}"

    # --------------------------------------------------
    # Final dataset registration (CANONICAL PATH ONLY)
    # --------------------------------------------------
    def register_dataset(
        self,
        *,
        name: str,
        path: str,
        generator: str,
        parent_hash: Optional[str] = None,
        notes: Optional[dict] = None,
    ) -> Dataset:
        """
        Register a FINALIZED dataset.

        Rules:
        - `path` MUST be the canonical, versioned dataset file
        - Temporary / flat paths are rejected
        - Hash-based deduplication is enforced
        - Dataset entries are immutable once registered

        NOTE:
        Transaction scope is owned by the caller.
        """

        final_path = Path(path).resolve()

        # -----------------------------
        # Validation
        # -----------------------------
        if not final_path.exists():
            raise RuntimeError(f"Dataset path does not exist: {final_path}")

        # Enforce versioned directory structure
        # Expected: .../<dataset_name>/<v###>/<file>
        parts = final_path.parts
        version_part = None
        for p in parts:
            if p.startswith("v") and p[1:].isdigit():
                version_part = p
                break

        if not version_part:
            raise RuntimeError(
                f"Refusing to register non-versioned dataset path: {final_path}"
            )

        version = version_part

        # -----------------------------
        # Hash + deduplication
        # -----------------------------
        content_hash = self._hash_file(str(final_path))

        existing = (
            self.db.query(Dataset)
            .filter(Dataset.content_hash == content_hash)
            .one_or_none()
        )
        if existing:
            return existing

        # -----------------------------
        # Insert immutable dataset row
        # -----------------------------
        dataset = Dataset(
            name=name,
            version=version,
            content_hash=content_hash,
            path=str(final_path),
            record_count=self._count_records(str(final_path)),
            schema=self._infer_schema(str(final_path)),
            generator=generator,
            parent_hash=parent_hash,
            created_by=self.actor,
            notes=notes,
        )

        self.db.add(dataset)
        self.db.flush()  # ensure PK without committing

        # -----------------------------
        # Audit (same transaction)
        # -----------------------------
        self.audit.log(
            dataset_hash=content_hash,
            action="REGISTERED",
            context={
                "path": str(final_path),
                "version": version,
                "generator": generator,
            },
        )

        return dataset

    # --------------------------------------------------
    # Lifecycle tracking
    # --------------------------------------------------
    def mark_used(self, *, dataset_hash: str, run_id: str) -> None:
        """
        Called once training begins (after MLflow run exists).
        """
        self.audit.log(
            dataset_hash=dataset_hash,
            action="USED",
            context={"run_id": run_id},
        )

    def mark_completed(
        self,
        *,
        dataset_hash: str,
        run_id: str,
        status: str,
    ) -> None:
        """
        Called exactly once at the end of training.

        status ∈ {"success", "failed"}
        """
        if status not in {"success", "failed"}:
            raise ValueError(f"Invalid dataset completion status: {status}")

        self.audit.log(
            dataset_hash=dataset_hash,
            action="COMPLETED",
            context={
                "run_id": run_id,
                "status": status,
            },
        )

    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------
    def get_dataset(
        self,
        *,
        name: str,
        version: str,
    ) -> Dataset:
        """
        Retrieve an existing dataset by name + version.
        """
        dataset = (
            self.db.query(Dataset)
            .filter(
                Dataset.name == name,
                Dataset.version == version,
            )
            .one_or_none()
        )

        if not dataset:
            raise RuntimeError(
                f"Dataset not found: name={name} version={version}"
            )

        return dataset
