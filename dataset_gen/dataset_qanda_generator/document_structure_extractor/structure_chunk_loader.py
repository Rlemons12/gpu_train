#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stage 2: Structure → Cleaned Chunks

Loads a structure.json created by Stage 1 and produces a cleaned,
Q&A-ready list of text chunks.

This does NOT generate Q&A — only prepares the clean chunks.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


class StructureChunkLoader:

    def __init__(self, structure_path: str,
                 min_length: int = 40,
                 dedupe: bool = True,
                 merge_headings: bool = True):
        """
        Args:
            structure_path: path to _structure.json
            min_length: ignore chunks shorter than this after cleaning
            dedupe: remove chunks that are exact duplicates
            merge_headings: headings get merged into the next chunk
        """
        self.structure_path = Path(structure_path)
        self.min_length = min_length
        self.dedupe = dedupe
        self.merge_headings_flag = merge_headings

        self.structure = self._load_json()

    # ----------------------------------------------------------
    # Load structure JSON
    # ----------------------------------------------------------
    def _load_json(self) -> dict:
        with open(self.structure_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ----------------------------------------------------------
    # Main entry: produce clean chunks
    # ----------------------------------------------------------
    def load_clean_chunks(self) -> List[Dict[str, Any]]:
        raw_chunks = self.structure.get("chunks", [])

        # Step 1 — normalize whitespace
        chunks = self._normalize_chunks(raw_chunks)

        # Step 2 — merge headings into next chunk
        if self.merge_headings_flag:
            chunks = self._merge_headings(chunks)

        # Step 3 — remove duplicates
        if self.dedupe:
            chunks = self._dedupe_chunks(chunks)

        # Step 4 — remove tiny chunks
        chunks = [c for c in chunks if len(c.get("text", "")) >= self.min_length]

        source_file = self.structure.get("source_file")
        source_type = self.structure.get("source_type")

        clean_chunks = []
        page_counters = {}

        current_section = None
        current_subsection = None

        # ----- heading detector -----
        def is_heading(text: str) -> bool:
            if not text:
                return False
            if len(text) <= 60 and text.isupper():
                return True
            if len(text) <= 80 and text.strip().endswith(":"):
                return True
            return False

        for c in chunks:
            txt = c.get("text", "").strip()
            page = c.get("page")

            # stable IDs
            if page is not None:
                page_counters.setdefault(page, 0)
                page_counters[page] += 1
                idx = page_counters[page]
                chunk_id = f"p{page}_c{idx}"
            else:
                idx = len(clean_chunks) + 1
                chunk_id = f"c{idx}"

            # heading logic
            if is_heading(txt):
                if len(txt.split()) <= 4:
                    current_section = txt
                    current_subsection = None
                    heading_level = "section"
                else:
                    current_subsection = txt
                    heading_level = "subsection"
            else:
                heading_level = "body"

            # ---------------------------------------------
            # PATCH #4 — Format chunk into pipeline-ready form
            # ---------------------------------------------
            safe_text = " ".join(txt.split())  # normalize whitespace
            has_heading = heading_level in ("section", "subsection")

            pipeline_context = safe_text
            context_tokens = len(safe_text.split())

            clean_chunks.append({
                "chunk_id": chunk_id,
                "text": safe_text,
                "pipeline_context": pipeline_context,
                "context_tokens": context_tokens,

                # hierarchy
                "heading_level": heading_level,
                "section": current_section,
                "subsection": current_subsection,
                "has_heading": has_heading,

                # metadata
                "page": page,
                "chunk_type": c.get("chunk_type", "text"),
                "source_file": source_file,
                "source_type": source_type,
                "merged_heading": c.get("merged_heading", False),

                # debugging
                "raw": c,
            })
        # ---------------------------------------------------------------------
        # PATCH D — Add Q&A-ready prompt blocks for pipeline consumption
        # ---------------------------------------------------------------------

        for c in clean_chunks:
            context = c["pipeline_context"]
            section = c.get("section")
            subsection = c.get("subsection")

            # Optional prefix for better question relevance
            hierarchy_prefix = ""
            if section:
                hierarchy_prefix += f"SECTION: {section}\n"
            if subsection:
                hierarchy_prefix += f"SUBSECTION: {subsection}\n"

            # Build question prompt
            c["question_prompt"] = (
                "You are generating factual knowledge-check questions from manufacturing documents.\n\n"
                f"{hierarchy_prefix}"
                f"CONTEXT:\n{context}\n\n"
                "TASK:\n"
                "Generate exactly 3 clear, factual questions that are:\n"
                "- based ONLY on information explicitly written in the context\n"
                "- short and direct (max 18 words)\n"
                "- answerable from the text\n"
                "- using question starters: What / Where / When / Which / How many / What is the purpose of\n\n"
                "Return ONLY the questions, each on its own line."
            )

            # Build answer prompt
            c["answer_prompt"] = (
                "You are answering a factual knowledge-check question.\n"
                "Answer ONLY using information explicitly stated in the context.\n"
                "Keep answers short and precise.\n\n"
                f"{hierarchy_prefix}"
                f"CONTEXT:\n{context}\n\n"
                "QUESTION: <question_here>\n"
                "ANSWER:"
            )

            # Mark that the chunk is fully ready for pipeline Q&A generation
            c["ready_for_pipeline"] = True

        return clean_chunks

    # ----------------------------------------------------------
    # Cleaning helpers
    # ----------------------------------------------------------
    def _normalize_chunks(self, chunks):
        """
        Normalize chunk schema from multiple extractors
        (native PDF, NuMarkdown OCR, future sources).
        """

        cleaned = []

        for c in chunks:

            # ---- Page normalization ----
            page = (
                    c.get("page")
                    or c.get("page_number")
                    or c.get("page_index")
            )

            if page is None:
                page = 1  # fallback safety

            # ---- Text normalization ----
            text = (
                    c.get("text")
                    or c.get("markdown")
                    or ""
            )

            text = text.strip()
            text = "\n".join(line.strip() for line in text.splitlines())
            text = text.replace("\u00a0", " ")

            # ---- Chunk type normalization ----
            chunk_type = c.get("chunk_type", "content")

            cleaned.append({
                "page": page,
                "chunk_type": chunk_type,
                "text": text
            })

        return cleaned

    def _merge_headings(self, chunks):
        """Merge heading chunks into their next text chunk."""
        merged = []
        skip_next = False

        for i, c in enumerate(chunks):
            if skip_next:
                skip_next = False
                continue

            # If this is the last chunk, just add it
            if i == len(chunks) - 1:
                merged.append(c)
                break

            if c["chunk_type"] == "heading":
                # merge into next chunk
                next_chunk = chunks[i + 1]
                combined_text = c["text"] + "\n" + next_chunk["text"]

                merged.append({
                    "page": next_chunk["page"],
                    "chunk_type": "text",
                    "text": combined_text
                })

                skip_next = True
            else:
                merged.append(c)

        return merged

    def _dedupe_chunks(self, chunks):
        """Remove chunks with identical text content."""
        seen = set()
        unique = []
        for c in chunks:
            key = c["text"]
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique
