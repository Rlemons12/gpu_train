#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
report_generators.py

Generates:
    - TXT reports (pretty, wrapped)
    - Excel reports (one or more sheets)

Uses dataset_queries.load_qna_rows_for_document as the main data source.
"""

import os
from typing import List, Dict, Any, Optional

import pandas as pd

from .formatting import wrap_text


# ---------------------------------------------------------------------------
# File-name helpers
# ---------------------------------------------------------------------------

def _sanitize_filename(name: str) -> str:
    """
    Very simple sanitizer: remove characters that cause trouble on Windows.
    """
    bad = '<>:"/\\|?*'
    cleaned = "".join(c for c in name if c not in bad)
    return cleaned.replace(" ", "_")


def build_output_paths(
    base_output_dir: str,
    document_name: str,
    suffix: str,
) -> str:
    """
    Build a default output path like:
        {base_output_dir}/{sanitized_doc_name}_{suffix}.ext

    The caller is responsible for adding an extension.
    """
    os.makedirs(base_output_dir, exist_ok=True)
    stem = _sanitize_filename(document_name)
    return os.path.join(base_output_dir, f"{stem}_{suffix}")


# ---------------------------------------------------------------------------
# TXT Report
# ---------------------------------------------------------------------------

def write_txt_qna_report(
    document_name: str,
    file_path: str,
    qna_rows: List[Dict[str, Any]],
    output_dir: str,
    width: int = 100,
) -> str:
    """
    Render a human-readable TXT report:
        - Document header
        - Chunk context
        - Questions + answers grouped by chunk
    """
    lines: List[str] = []

    lines.append("=" * 140)
    lines.append(f"DOCUMENT: {document_name}")
    lines.append(f"PATH:     {file_path}")
    lines.append("=" * 140)

    # Group rows by chunk_id
    by_chunk: Dict[str, List[Dict[str, Any]]] = {}
    for row in qna_rows:
        key = row.get("chunk_id")
        by_chunk.setdefault(key, []).append(row)

    # Sort chunks deterministically by page, then chunk_id
    sorted_chunks = sorted(
        by_chunk.items(),
        key=lambda kv: (
            kv[1][0].get("page", 0) or 0,
            kv[0] or "",
        ),
    )

    for chunk_id, rows in sorted_chunks:
        first = rows[0]
        page = first.get("page")
        section = first.get("section")
        subsection = first.get("subsection")
        context = first.get("context", "")

        lines.append("\n" + "-" * 140)
        lines.append(
            f"[CHUNK] chunk_id={chunk_id}   "
            f"(page={page}, section={section}, subsection={subsection})"
        )
        lines.append("-" * 140)
        lines.append(wrap_text(context, width=width))
        lines.append("-" * 140)

        # Group by question_id within the chunk
        by_question: Dict[int, List[Dict[str, Any]]] = {}
        for row in rows:
            qid = row.get("question_id")
            if qid is None:
                continue
            by_question.setdefault(qid, []).append(row)

        if not by_question:
            lines.append("  (No questions for this chunk)")
            continue

        # Sort questions by index
        sorted_questions = sorted(
            by_question.items(),
            key=lambda kv: kv[1][0].get("question_index", 0),
        )

        for q_id, q_rows in sorted_questions:
            q_first = q_rows[0]
            q_index = q_first.get("question_index")
            question = q_first.get("question", "")
            lines.append(f"\n  ► QUESTION {q_index}: {question}")

            # Sort answers by model name
            sorted_answers = sorted(
                q_rows,
                key=lambda r: (r.get("model") or ""),
            )

            if not sorted_answers or not sorted_answers[0].get("answer"):
                lines.append("      (No answers found for this question)")
                continue

            for row in sorted_answers:
                model_name = row.get("model")
                answer = row.get("answer") or ""
                best = row.get("is_best_model", False)

                if model_name:
                    tag = f"[{model_name}]"
                    if best:
                        tag += " (best)"
                    lines.append(f"      - {tag}")
                else:
                    lines.append("      - [NO_MODEL]")

                wrapped_answer = wrap_text(answer, width=width)
                for line in wrapped_answer.splitlines():
                    lines.append("          " + line)

    # Write to disk
    base_path = build_output_paths(output_dir, document_name, "qna_report")
    txt_path = base_path + ".txt"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return txt_path


# ---------------------------------------------------------------------------
# Excel Report(s)
# ---------------------------------------------------------------------------

def build_qna_dataframe(qna_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build a DataFrame from raw Q&A rows.
    You can easily tweak the column ordering here.
    """
    if not qna_rows:
        return pd.DataFrame()

    df = pd.DataFrame(qna_rows)

    preferred_cols = [
        "document_id",
        "document",
        "file_path",
        "page",
        "section",
        "subsection",
        "chunk_id",
        "chunk_db_id",
        "question_id",
        "question_index",
        "question",
        "model_id",
        "model",
        "answer_id",
        "answer",
        "is_best_model",
        "context",
    ]

    # Keep only columns that exist
    cols = [c for c in preferred_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in cols]
    return df[cols + remaining]


def write_excel_qna_table(
    document_name: str,
    qna_rows: List[Dict[str, Any]],
    output_dir: str,
    sheet_name: str = "QnA",
) -> str:
    """
    Write a single-sheet Excel file with the Q&A table.
    """
    base_path = build_output_paths(output_dir, document_name, "qna_table")
    xlsx_path = base_path + ".xlsx"

    df = build_qna_dataframe(qna_rows)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    return xlsx_path


def write_full_excel_report(
    document_name: str,
    qna_rows: List[Dict[str, Any]],
    output_dir: str,
) -> str:
    """
    Write an Excel file with multiple sheets, currently:
        - QnA: raw Q&A table
        - Questions: questions only (deduped)
        - Answers: answers only
    """
    base_path = build_output_paths(output_dir, document_name, "qna_report")
    xlsx_path = base_path + ".xlsx"

    df = build_qna_dataframe(qna_rows)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # Raw combined table
        df.to_excel(writer, sheet_name="QnA", index=False)

        if not df.empty:
            q_cols = [
                "document_id",
                "document",
                "file_path",
                "page",
                "section",
                "subsection",
                "chunk_id",
                "question_id",
                "question_index",
                "question",
                "context",
            ]
            q_cols = [c for c in q_cols if c in df.columns]
            df_q = df[q_cols].drop_duplicates()
            df_q.to_excel(writer, sheet_name="Questions", index=False)

            a_cols = [
                "document_id",
                "document",
                "chunk_id",
                "question_id",
                "model",
                "is_best_model",
                "answer",
            ]
            a_cols = [c for c in a_cols if c in df.columns]
            df_a = df[a_cols].drop_duplicates()
            df_a.to_excel(writer, sheet_name="Answers", index=False)

    return xlsx_path
