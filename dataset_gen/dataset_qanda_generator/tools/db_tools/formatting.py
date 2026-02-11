#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
formatting.py

Small helpers for text wrapping and console display.
"""

import textwrap
from typing import Iterable, List, Tuple, Sequence, Optional


def wrap_text(text: Optional[str], width: int = 100) -> str:
    """
    Wrap a block of text to a fixed width.
    """
    if not text:
        return ""
    return "\n".join(textwrap.wrap(text, width=width))


def print_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
    max_width: int = 40,
) -> None:
    """
    Simple console table printer.

    Long cell values are truncated to max_width with "…".
    """
    headers = list(headers)
    rows = [list(r) for r in rows]

    def truncate(s: str) -> str:
        s = s or ""
        if len(s) <= max_width:
            return s
        return s[: max_width - 1] + "…"

    # Compute column widths
    cols = len(headers)
    col_widths: List[int] = [len(h) for h in headers]

    for row in rows:
        for i in range(cols):
            if i >= len(row):
                continue
            cell = truncate(str(row[i]))
            col_widths[i] = max(col_widths[i], len(cell))

    # Format row
    def fmt_row(values: Sequence[str]) -> str:
        cells = []
        for i, v in enumerate(values):
            text = truncate(str(v))
            cells.append(text.ljust(col_widths[i]))
        return " | ".join(cells)

    # Print header
    print(fmt_row(headers))
    print("-" * (sum(col_widths) + 3 * (len(col_widths) - 1)))

    # Print data rows
    for row in rows:
        print(fmt_row(row))
