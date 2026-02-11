from __future__ import annotations

from pathlib import Path

from dataset_gen.dataset_qanda_generator.qna_db.qa_db import Document
from dataset_gen.dataset_qanda_generator.configuration.logging_config import (
    get_qna_logger,
)
from dataset_gen.dataset_qanda_generator.configuration.env_adapter import (
    normalize_path,
)

log = get_qna_logger(__name__)


class DocumentService:
    """
    Service-layer authority for Document validation and resolution.

    Guarantees:
      • Returned file paths are absolute
      • Returned file paths exist on disk
      • Returned file paths are environment-safe (Windows / WSL)
    """

    @staticmethod
    def resolve_file_path(document: Document) -> Path:
        """
        Resolve and validate the absolute filesystem path for a Document.

        Raises:
            FileNotFoundError
            ValueError
        """
        if not document.file_path:
            raise ValueError(
                f"Document id={document.id} has empty file_path"
            )

        # Normalize Windows → WSL paths if needed
        normalized = normalize_path(document.file_path)
        path = Path(normalized)

        if not path.is_absolute():
            raise FileNotFoundError(
                f"Document id={document.id} has non-absolute path: "
                f"'{document.file_path}'"
            )

        if not path.exists():
            raise FileNotFoundError(
                f"Document id={document.id} file not found: {path}"
            )

        if not path.is_file():
            raise FileNotFoundError(
                f"Document id={document.id} path is not a file: {path}"
            )

        log.debug(
            "[DOCUMENT] Resolved document_id=%s → %s",
            document.id,
            path,
        )

        return path
