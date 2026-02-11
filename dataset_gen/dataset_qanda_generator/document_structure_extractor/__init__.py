"""
Document structure extraction module for Option C.

Exposes:
    • DocumentStructureExtractor  – Stage 1 extraction (PDF, DOCX, TXT, PPTX)
    • StructureChunkLoader        – Stage 2 cleaning & pipeline chunk loader
"""

from .structure_extractor import DocumentStructureExtractor
from .structure_chunk_loader import StructureChunkLoader

__all__ = [
    "DocumentStructureExtractor",
    "StructureChunkLoader",
]
