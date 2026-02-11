#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
export_qna_report.py

Command-line tool + interactive menu to export Q&A data
from the Option C training database.

Features:
    - List documents
    - List pipeline runs
    - Export Q&A (TXT, Excel table, full Excel)
"""

import os
import sys
import argparse
from typing import Optional

# --------------------------------------------------------------
# CONFIG & DB IMPORTS
# --------------------------------------------------------------
from dataset_gen.dataset_qanda_generator.configuration import cfg
from .dataset_queries import (
    open_training_session,
    list_documents,
    get_document_by_name_fragment,
    list_pipeline_runs_for_document,
    load_qna_rows_for_document,
)
from .report_generators import (
    write_txt_qna_report,
    write_excel_qna_table,
    write_full_excel_report,
)
from .formatting import print_table


# --------------------------------------------------------------
# DEFAULT OUTPUT DIRECTORY
# Now sourced from configuration
# --------------------------------------------------------------
DEFAULT_OUTPUT_DIR = str(cfg.DB_TOOLS_OUTPUT_DIR)


# --------------------------------------------------------------
# CORE EXPORT FUNCTION
# --------------------------------------------------------------

def export_qna_for_document(
    doc_name_fragment: str,
    output_dir: str,
    export_txt: bool,
    export_qna_table: bool,
    export_full_excel: bool,
    include_rankings: bool = True,
) -> None:

    session = open_training_session()

    try:
        doc = get_document_by_name_fragment(session, doc_name_fragment)
        if not doc:
            print(f"\nNo document found for: {doc_name_fragment}\n")
            return

        print(f"\nUsing document: {doc.file_name} (id={doc.id})\n")

        rows = load_qna_rows_for_document(
            session=session,
            document=doc,
            include_rankings=include_rankings,
        )

        if not rows:
            print("No Q&A rows found for this document.")
            return

        if export_txt:
            txt_path = write_txt_qna_report(
                document_name=doc.file_name,
                file_path=doc.file_path,
                qna_rows=rows,
                output_dir=output_dir,
                width=100,
            )
            print(f"TXT exported → {txt_path}")

        if export_qna_table:
            xlsx_table_path = write_excel_qna_table(
                document_name=doc.file_name,
                qna_rows=rows,
                output_dir=output_dir,
                sheet_name="QnA",
            )
            print(f"Excel QnA table exported → {xlsx_table_path}")

        if export_full_excel:
            xlsx_full_path = write_full_excel_report(
                document_name=doc.file_name,
                qna_rows=rows,
                output_dir=output_dir,
            )
            print(f"Full Excel report exported → {xlsx_full_path}")

    finally:
        session.close()


# --------------------------------------------------------------
# LISTING COMMANDS
# --------------------------------------------------------------

def cmd_list_documents(name_fragment: Optional[str] = None) -> None:
    session = open_training_session()
    try:
        docs = list_documents(session, name_fragment)
        if not docs:
            print("No documents found.")
            return

        rows = []
        for d in docs:
            rows.append([d.id, d.file_name, d.file_path])

        print_table(["ID", "File Name", "Path"], rows)
    finally:
        session.close()


def cmd_list_runs(doc_fragment: Optional[str] = None) -> None:
    """
    List all pipeline runs.
    Shows:
        - Run ID
        - Document Name
        - Run Type
        - Started At
        - Finished At
        - Duration
        - Status
    """
    from datetime import timedelta

    session = open_training_session()
    try:
        runs = list_pipeline_runs_for_document(
            session=session,
            document_id=None,
            document_name_fragment=doc_fragment,
        )

        if not runs:
            print("No pipeline runs found.")
            return

        rows = []
        for run, file_name in runs:
            started = run.started_at
            finished = run.finished_at

            if started and finished:
                duration = finished - started
            else:
                duration = None

            rows.append(
                [
                    run.id,
                    file_name or "",
                    run.run_type,
                    started.strftime("%Y-%m-%d %H:%M:%S") if started else "",
                    finished.strftime("%Y-%m-%d %H:%M:%S") if finished else "",
                    str(duration) if duration else "",
                    "OK" if run.success else "FAIL",
                ]
            )

        print_table(
            ["Run ID", "Document", "Type", "Started", "Finished", "Duration", "Status"],
            rows,
            max_width=32,
        )
    finally:
        session.close()



# --------------------------------------------------------------
# INTERACTIVE MENU
# --------------------------------------------------------------

_INTERACTIVE_OUTPUT_DIR = DEFAULT_OUTPUT_DIR


def _interactive_get_output_dir(force_prompt: bool = False) -> str:
    global _INTERACTIVE_OUTPUT_DIR

    if not force_prompt:
        return _INTERACTIVE_OUTPUT_DIR

    print(f"Current output directory: {_INTERACTIVE_OUTPUT_DIR}")
    new_dir = input("Enter new output directory (or press Enter to keep current): ").strip()

    if new_dir:
        _INTERACTIVE_OUTPUT_DIR = new_dir
        os.makedirs(_INTERACTIVE_OUTPUT_DIR, exist_ok=True)

    return _INTERACTIVE_OUTPUT_DIR


def interactive_menu() -> None:
    while True:
        print("\n==============================")
        print("OPTION C – Q&A REPORT TOOL")
        print("==============================")
        print("1) List all documents")
        print("2) Export Q&A report (TXT + Excel)")
        print("3) Export only Q&A table (Excel)")
        print("4) List pipeline runs")
        print("5) Change output directory")
        print("6) Exit")
        print("==============================")

        choice = input("Enter choice [1-6]: ").strip()

        if choice == "1":
            fragment = input("Filter by name (press Enter for all): ").strip() or None
            cmd_list_documents(fragment)

        elif choice == "2":
            doc_frag = input("Document name fragment: ").strip()
            if not doc_frag:
                print("Document fragment is required.")
                continue

            output_dir = _interactive_get_output_dir()
            export_qna_for_document(
                doc_name_fragment=doc_frag,
                output_dir=output_dir,
                export_txt=True,
                export_qna_table=True,
                export_full_excel=True,
                include_rankings=True,
            )

        elif choice == "3":
            doc_frag = input("Document name fragment: ").strip()
            if not doc_frag:
                print("Document fragment is required.")
                continue

            output_dir = _interactive_get_output_dir()
            export_qna_for_document(
                doc_name_fragment=doc_frag,
                output_dir=output_dir,
                export_txt=False,
                export_qna_table=True,
                export_full_excel=False,
                include_rankings=True,
            )

        elif choice == "4":
            frag = input("Filter by document name fragment: ").strip() or None
            cmd_list_runs(frag)

        elif choice == "5":
            _interactive_get_output_dir(force_prompt=True)

        elif choice == "6":
            print("Exiting.")
            break

        else:
            print("Invalid choice.")


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Q&A reports from the Option C training database."
    )

    parser.add_argument("--doc", "-d", dest="doc_name", help="Document name or fragment.")
    parser.add_argument("--output-dir", "-o", dest="output_dir",
                        default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")

    parser.add_argument("--list-docs", action="store_true", help="List documents and exit.")
    parser.add_argument("--list-runs", action="store_true", help="List pipeline runs and exit.")

    parser.add_argument("--txt", action="store_true", help="Export TXT report.")
    parser.add_argument("--xlsx", action="store_true", help="Export Excel Q&A table.")
    parser.add_argument("--full-xlsx", action="store_true", help="Export full multi-sheet XLSX.")
    parser.add_argument("--all", action="store_true", help="Export TXT + XLSX + full XLSX.")

    parser.add_argument("--no-rankings", action="store_true", help="Disable ranking info.")

    return parser


def main(argv=None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        interactive_menu()
        return

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_docs:
        cmd_list_documents()
        return

    if args.list_runs:
        cmd_list_runs(args.doc_name)
        return

    if not args.doc_name:
        print("You must provide --doc NAME when exporting.")
        parser.print_help()
        sys.exit(1)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    export_txt = args.txt or args.all
    export_qna_table = args.xlsx or args.all
    export_full_excel = args.full_xlsx or args.all

    include_rankings = not args.no_rankings

    export_qna_for_document(
        doc_name_fragment=args.doc_name,
        output_dir=output_dir,
        export_txt=export_txt,
        export_qna_table=export_qna_table,
        export_full_excel=export_full_excel,
        include_rankings=include_rankings,
    )


if __name__ == "__main__":
    main()
