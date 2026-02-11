import logging
from pathlib import Path
from typing import Dict, Any

from dataset_gen.dataset_qanda_generator.document_structure_extractor.structure_extractor import (
    DocumentStructureExtractor,
)

# Import your NuMarkdown adapter
from nu_markdown.vlm_adapter import VLMAdapter


log = logging.getLogger("document_strategy")


class UnsupportedDocumentType(Exception):
    pass


class DocumentStrategy:
    """
    Centralized document extraction strategy.

    Decides:
        - Native extraction
        - OCR fallback
        - Direct OCR (TIFF)
    """

    SUPPORTED_NATIVE = {".pdf",".doc",".docx",".pptx",".txt",".md",".xlsx",".xls",".xlsm",".csv"}

    SUPPORTED_OCR_ONLY = {".tif", ".tiff"}

    def __init__(self, doc_path: Path):
        if isinstance(doc_path, str):
            doc_path = Path(doc_path)

        if not doc_path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")

        self.doc_path = doc_path
        self.extension = doc_path.suffix.lower()

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def extract_structure(self) -> Dict[str, Any]:
        log.info("[DOC STRATEGY] Extracting: %s", self.doc_path.name)

        # -------------------------------------------------
        # OCR-ONLY FILE TYPES (TIFF, etc.)
        # -------------------------------------------------
        if self.extension in self.SUPPORTED_OCR_ONLY:
            log.info("[STRATEGY] Using TIFF OCR")
            return self._extract_via_ocr()

        # -------------------------------------------------
        # NATIVE-SUPPORTED FILE TYPES (PDF, DOCX, etc.)
        # -------------------------------------------------
        if self.extension in self.SUPPORTED_NATIVE:
            log.info("[STRATEGY] Using native extractor")

            structure = self._extract_native()

            # If native extraction produced no usable text,
            # fall back to OCR automatically
            if not self._has_text(structure):
                log.warning(
                    "[DOC STRATEGY] No native text detected. Falling back to NuMarkdown OCR."
                )
                log.info("[STRATEGY] Using NuMarkdown OCR")
                return self._extract_via_ocr()

            return structure

        # -------------------------------------------------
        # UNSUPPORTED TYPES
        # -------------------------------------------------
        raise UnsupportedDocumentType(
            f"Unsupported document type: {self.extension}"
        )

    # -----------------------------------------------------
    # Native Extraction
    # -----------------------------------------------------

    def _extract_native(self) -> Dict[str, Any]:
        extractor = DocumentStructureExtractor(str(self.doc_path))
        return extractor.extract()

    # -----------------------------------------------------
    # OCR Extraction
    # -----------------------------------------------------

    def _extract_via_ocr(self) -> Dict[str, Any]:
        """
        Use NuMarkdown VLM adapter for OCR-based extraction.
        Returns Stage-1 structure schema.
        """
        log.info("[DOC STRATEGY] Using NuMarkdown OCR extractor")

        from nu_markdown.vlm_adapter import VLMAdapter

        adapter = VLMAdapter()

        # For now we assume OCR is primarily for PDFs and images
        return adapter.extract_markdown_from_pdf(str(self.doc_path))

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    def _has_text(self, structure: Dict[str, Any]) -> bool:
        chunks = structure.get("chunks", [])
        if not chunks:
            return False

        for c in chunks:
            text = c.get("text") or c.get("markdown")
            if text and text.strip():
                return True

        return False

