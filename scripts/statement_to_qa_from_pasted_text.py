import json
import re
from typing import List


# ============================================================
# CONFIG
# ============================================================

OUTPUT_JSONL = "synthetic_qa_from_pasted_text.jsonl"

INSTRUCTION_TEXT = (
    "Answer the question using only the provided context. "
    "If the answer is not present, respond with 'Not found in the provided source.'"
)


# ============================================================
# BASIC NLP UTILITIES (DETERMINISTIC)
# ============================================================

def split_sentences(text: str) -> List[str]:
    """
    Very conservative sentence splitter.
    Avoids over-splitting technical text.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def is_fact_statement(sentence: str) -> bool:
    """
    Filters sentences likely to be factual and self-contained.
    """
    lowered = sentence.lower()

    bad_starts = ("this ", "that ", "these ", "those ", "it ")
    if lowered.startswith(bad_starts):
        return False

    fact_keywords = (
        " is ",
        " are ",
        " must ",
        " shall ",
        " allows ",
        " enable",
        " occurs ",
        " results ",
        " removes ",
        " requires ",
        " supports ",
    )

    return any(k in lowered for k in fact_keywords)


# ============================================================
# STATEMENT → QUESTION TRANSFORM
# ============================================================

def statement_to_question(statement: str) -> str:
    """
    Deterministic transformation rules.
    Safe, boring, and correct.
    """

    s = statement.strip().rstrip(".")

    # MUST / SHALL
    if " must " in s.lower():
        return "What must be done " + _after_keyword(s, "must") + "?"

    # ALLOWS / ENABLES / SUPPORTS
    for verb in ("allows", "enables", "supports"):
        if f" {verb} " in s.lower():
            subject = s[: s.lower().index(verb)].strip()
            return f"What does {subject} {verb}?"

    # OCCURS WHEN / RESULTS IN / REMOVES
    for verb in ("occurs", "results", "removes"):
        if f" {verb} " in s.lower():
            return f"What happens when {s[: s.lower().index(verb)].strip()}?"

    # FALLBACK
    return f"What is stated about {s.split()[0]}?"


def _after_keyword(sentence: str, keyword: str) -> str:
    idx = sentence.lower().index(keyword) + len(keyword)
    return sentence[idx:].strip()


# ============================================================
# MAIN PIPELINE
# ============================================================

def generate_qa_from_text(pasted_text: str):
    sentences = split_sentences(pasted_text)

    accepted = [
        s for s in sentences
        if is_fact_statement(s)
    ]

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for s in accepted:
            qa = {
                "instruction": INSTRUCTION_TEXT,
                "source": {
                    "document_title": "User Pasted Text",
                    "section": "Manual Input",
                    "page": None
                },
                "context": s,
                "question": statement_to_question(s),
                "response": s
            }

            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"[OK] Generated {len(accepted)} Q/A pairs")
    print(f"[OK] Output written to: {OUTPUT_JSONL}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("Paste your source text below. End with CTRL+D (Linux/macOS) or CTRL+Z + Enter (Windows):\n")

    try:
        pasted_text = ""
        while True:
            pasted_text += input() + "\n"
    except EOFError:
        pass

    if not pasted_text.strip():
        print("[ERROR] No text provided.")
    else:
        generate_qa_from_text(pasted_text)
