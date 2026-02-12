# Source‑Aware Q/A Generation Template

This document defines the **canonical question/answer generation contract** used in the training data pipeline.

It exists to keep design decisions **focused, consistent, and scalable** across documents, domains, and intents.

---

## 1. Core Principle

> **No source → no question → no answer**

All generated questions and answers **must** be grounded in an explicit source contract.

---

## 2. Conceptual Layers

### 2.1 Source (Truth Constraint)

Defines *where* the knowledge comes from.

Required fields:

* Document title
* Section
* Subsection
* Page (or equivalent locator)
* Reference text (context chunk)

The source constrains **what is allowed to be said**.

---

### 2.2 Domain (Reasoning Mode)

Defines *how the model should reason*.

Domain is **not document‑specific**.

Controlled vocabulary:

* Definition
* SOP
* StepByStep
* Troubleshooting
* Diagnostic
* Safety
* Configuration
* Inspection
* Training

Domain controls:

* Question phrasing
* Answer structure
* Level of procedural detail

---

### 2.3 Intent (Routing – not embedded in text)

Defines *what the user is trying to do*.

Examples:

* define_term
* how_to_perform
* resolve_fault
* safety_warning

Intent **selects the domain** but is not embedded directly into the LLM supervision text.

---

## 3. Question Generation Template

This template is used when generating **questions from a document chunk**.

```
You are generating questions from a technical reference.

SOURCE INFORMATION
Document: {DOCUMENT_TITLE}
Domain: {DOMAIN_TYPE}
Section: {SECTION}
Subsection: {SUBSECTION}
Page: {PAGE}

REFERENCE TEXT
{CONTEXT}

INSTRUCTIONS
Generate {N} questions that:
- Can be answered ONLY using the reference text
- Are appropriate for the specified domain
- Do NOT require outside knowledge
- Are unambiguous and specific

Return only the questions as a list.
```

Rules:

* Questions must be answerable from the source
* No hypothetical or speculative questions
* No cross‑document assumptions

---

## 4. Answer Generation Template

This template is used to generate **answers to generated questions**.

```
You are answering a technical question using an authoritative source.

SOURCE INFORMATION
Document: {DOCUMENT_TITLE}
Domain: {DOMAIN_TYPE}
Section: {SECTION}
Subsection: {SUBSECTION}
Page: {PAGE}

REFERENCE TEXT
{CONTEXT}

QUESTION
{QUESTION}

INSTRUCTIONS
- Answer using ONLY the reference text
- Do NOT introduce outside knowledge
- Explicitly cite the source
- Use language appropriate for the domain

The answer must begin with:
"According to {DOCUMENT_TITLE} ({DOMAIN_TYPE}, {SECTION}, Page {PAGE}), ..."
```

---

## 5. Refusal Rule (Boundary Enforcement)

If the reference text does not contain enough information to answer the question, respond with:

```
The provided source does not contain sufficient information to answer this question.
```

This rule prevents hallucination and teaches honest boundary behavior.

---

## 6. Training Export Alignment

All generated Q/A pairs **must already conform** to this contract before export.

Training formats (Alpaca, ChatML, etc.) are **wrappers**, not behavior definitions.

---

## 7. Invariants (Do Not Break These)

* One canonical template
* Source is mandatory
* Domain is mandatory
* Context is authoritative
* Citation is explicit

---

## 8. Design Rule of Thumb

> **Intent chooses the path.**
> **Domain governs reasoning.**
> **Source constrains truth.**

---

## 9. Scope

This document governs:

* Question generation
* Answer generation
* Supervision consistency
* Dataset quality control

It does **not** define:

* Model architecture
* Hyperparameters
* UI behavior

---

## 10. Change Policy

Changes to this document should be:

* Intentional
* Minimal
* Backward‑compatible

If a change alters behavior, update this file first.
