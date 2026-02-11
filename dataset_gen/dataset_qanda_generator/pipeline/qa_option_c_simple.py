#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Option C – Multi-model Q&A generator (OFFLINE)

Pipeline:
    Document → structure.json → cleaned chunks
             → FLAN-T5 for question generation
             → FLAN-T5 + TinyLlama + Qwen + Gemma + OpenELM for answers
             → JSONL output (one record per chunk with multiple answers)

No internet. No HuggingFace online. All models are loaded from local dirs.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,   # For FLAN (encoder-decoder)
    AutoModelForCausalLM     # For TinyLlama, Qwen, Gemma, OpenELM
)

# -------------------------------------------------------------------
# Import Stage 1 + Stage 2
# -------------------------------------------------------------------
from structure_extractor import DocumentStructureExtractor
from structure_chunk_loader import StructureChunkLoader
import time

from transformers.cache_utils import DynamicCache

# -------------------------------------------------------------------
# Compatibility shim for older/newer DynamicCache vs OpenELM code
# -------------------------------------------------------------------
if not hasattr(DynamicCache, "seen_tokens"):
    # For older transformers versions, add a default attribute so that
    # modeling_openelm can safely read/cache .seen_tokens
    DynamicCache.seen_tokens = 0

# -------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("optionC_multi_model")

# -------------------------------------------------------------------
# LOCAL MODEL PATHS
# -------------------------------------------------------------------
DEFAULT_FLAN = r"E:\emtac\models\llm\flan_t5_large\models--google--flan-t5-large\snapshots\0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
MODELS_TINY_LLAMA_DIR = r"E:\emtac\models\llm\TinyLlama_1_1B"
MODELS_QWEN_DIR       = r"E:\emtac\models\llm\Qwen2.5-3B-Instruct"
MODELS_GEMMA_DIR      = r"E:\emtac\models\llm\google_gemma-2-2b-it"
MODELS_APPLE_ELM_DIR  = r"E:\emtac\models\llm\apple_OpenELM-1_1B-Instruct"


# ===================================================================
# FLAN MODEL (Question + Answer)
# ===================================================================

class FLAN_QA_Model:
    """
    FLAN-T5-Large wrapper for:
        - generate_question(context)
        - generate_answer(context, question)

    The same model handles both.
    """

    def __init__(self, model_dir: str = DEFAULT_FLAN, device: str | None = None):
        model_path = Path(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"FLAN model not found: {model_path}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        log.info(f"[FLAN] Loading FLAN-T5-Large from: {model_path}")
        log.info(f"[FLAN] Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            str(model_path),
            local_files_only=True
        ).to(self.device)

        self.model.eval()

    # ---------------------------------------------------------------
    # QUESTION GENERATION
    # ---------------------------------------------------------------
    @torch.no_grad()
    def generate_question(self, context: str, max_len: int = 48) -> str:
        """
        Generates a factual question about the context.
        """
        prompt = (
            "You are generating a factual question based ONLY on the context.\n"
            "Ask one clear, specific question about information explicitly stated.\n\n"
            f"CONTEXT:\n{context}\n\n"
            "QUESTION:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768,
        ).to(self.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_len,
            num_beams=4,
            early_stopping=True,
        )

        q = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

        # Clean junk tokens
        q = q.replace("Question:", "").replace("QUESTION:", "").strip()

        if not q.endswith("?"):
            q += "?"

        return q

    def generate_questions(self, context: str, n: int = 3, max_len: int = 48) -> list[str]:
        """
        Generate N distinct factual questions based ONLY on the provided context.
        Returns a list of clean, unique questions.

        FLAN-T5 will output multiple lines, each containing a potential question.
        This method extracts, cleans, and limits them to N.
        """

        prompt = (
            "You are generating multiple factual questions based ONLY on the context.\n"
            f"Write {n} different short questions.\n"
            "Each question must be on its own line.\n"
            "Ask about facts explicitly stated in the context.\n\n"
            f"CONTEXT:\n{context}\n\n"
            "QUESTIONS:\n"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768,
        ).to(self.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_len * n,
            num_beams=5,
            early_stopping=True,
        )

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # -----------------------------------------------------------
        # Extract questions line-by-line
        # -----------------------------------------------------------
        raw_lines = text.split("\n")
        cleaned = []

        for line in raw_lines:
            q = line.strip()

            # Remove leading labels
            q = q.replace("Q:", "").replace("Q1:", "").replace("Q2:", "").replace("Question:", "").strip()

            # Skip headers, empty lines, echo, etc.
            if not q:
                continue
            if q.lower().startswith("questions"):
                continue
            if q.lower().startswith("context"):
                continue

            # Force question mark
            if not q.endswith("?"):
                q += "?"

            cleaned.append(q)

        # -----------------------------------------------------------
        # Enforce unique questions & cap to N
        # -----------------------------------------------------------
        uniq = []
        for q in cleaned:
            if q not in uniq:
                uniq.append(q)
            if len(uniq) == n:
                break

        # Safety fallback: if FLAN produced fewer than N, regenerate single-questions
        if len(uniq) < n:
            for _ in range(n - len(uniq)):
                q = self.generate_question(context)
                if q not in uniq:
                    uniq.append(q)

        return uniq

    # ---------------------------------------------------------------
    # ANSWER GENERATION
    # ---------------------------------------------------------------
    @torch.no_grad()
    def generate_answer(self, context: str, question: str, max_len: int = 64) -> str:
        """
        Answers the question concisely using ONLY information found in context.
        """
        prompt = (
            "You are answering a factual knowledge-check question.\n"
            "Use ONLY information explicitly stated in the context.\n"
            "Keep the answer short and precise.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{question}\n\n"
            "ANSWER:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768,
        ).to(self.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_len,
            num_beams=4,
            early_stopping=True,
        )

        a = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

        # Remove any echo of "ANSWER:"
        if a.lower().startswith("answer:"):
            a = a[7:].strip()

        return a


# ===================================================================
# BASE CLASS FOR CAUSAL-LM ANSWER GENERATORS
# ===================================================================

class CausalLMAnswerBase:
    """
    Shared logic for causal LMs (TinyLlama, Qwen, Gemma, OpenELM).
    Implements a common interface: generate_answer(context, question, ...).
    """

    def __init__(self, model_dir: str, name: str, device: str | None = None,
                 trust_remote_code: bool = False):
        model_path = Path(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"{name} model not found: {model_path}")

        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        log.info(f"[{name}] Loading from: {model_path}")
        log.info(f"[{name}] Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=trust_remote_code
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=trust_remote_code
        ).to(self.device)

        self.model.eval()

        # Fallback for eos token if missing
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

    @torch.no_grad()
    def generate_answer(
            self,
            context: str,
            question: str,
            max_new_tokens: int = 80,
            temperature: float = 0.7,
            top_p: float = 0.9,
    ) -> str:
        """
        Default prompt style for causal LMs.
        Subclasses can override `build_prompt` if needed.
        """
        prompt = self.build_prompt(context, question)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        generation_kwargs: dict[str, Any] = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            use_cache=False  # <── critical fix for OpenELM DynamicCache bug
        )
        if self.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = self.eos_token_id

        outputs = self.model.generate(**generation_kwargs)

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_text[len(prompt):].strip()
        answer = answer.split("\n")[0].strip()
        return answer

    def build_prompt(self, context: str, question: str) -> str:
        """
        Basic context + question → answer prompt.
        """
        return (
            "CONTEXT:\n"
            f"{context}\n\n"
            f"QUESTION: {question}\n"
            "ANSWER: "
        )


# ===================================================================
# TinyLlama Answer Generator
# ===================================================================

class TinyLlamaAnswerGenerator(CausalLMAnswerBase):
    def __init__(self, model_dir: str = MODELS_TINY_LLAMA_DIR, device: str | None = None):
        super().__init__(model_dir=model_dir, name="TINY", device=device, trust_remote_code=False)


# ===================================================================
# Qwen-2.5-3B-Instruct Answer Generator
# ===================================================================

class QwenAnswerGenerator(CausalLMAnswerBase):
    def __init__(self, model_dir: str = MODELS_QWEN_DIR, device: str | None = None):
        super().__init__(model_dir=model_dir, name="QWEN", device=device, trust_remote_code=True)


# ===================================================================
# Gemma-2-2B-it Answer Generator
# ===================================================================

class GemmaAnswerGenerator(CausalLMAnswerBase):
    """
    Gemma 2B/9B Instruct – plain prompt version.
    """

    def __init__(self, model_dir: str, device: str | None = None):
        super().__init__(
            model_dir=model_dir,
            name="GEMMA",
            device=device,
            trust_remote_code=True
        )

    def build_prompt(self, context: str, question: str) -> str:
        return (
            f"You are a helpful assistant. Use ONLY the information in the context.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n"
            f"ANSWER:"
        )

    @torch.no_grad()
    def generate_answer(
        self,
        context: str,
        question: str,
        max_new_tokens: int = 60,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> str:

        prompt = self.build_prompt(context, question)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.eos_token_id,
            eos_token_id=self.eos_token_id,
            use_cache=True,
        )

        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        answer = full[len(prompt):].strip()
        answer = answer.split("\n")[0].strip()

        return answer

# ===================================================================
# Apple OpenELM-1.1B-Instruct Answer Generator
# ===================================================================
class OpenELMAnswerGenerator:
    """
    Apple OpenELM-1.1B Instruct – Improved Answer Generator
    Handles:
        • Clean short answers
        • No <<<<< artifacts
        • Proper ChatML prompting
        • Deterministic behavior
        • DynamicCache compatibility (use_cache=False)
        • Basic repetition control
    """

    def __init__(self, model_dir: str = MODELS_APPLE_ELM_DIR, device: str | None = None):
        if not Path(model_dir).exists():
            raise FileNotFoundError(f"OpenELM model not found: {model_dir}")

        self.name = "OPENELM"
        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        log.info(f"[OPENELM] Loading Apple OpenELM-1.1B from: {model_dir}")
        log.info(f"[OPENELM] Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=True
        ).to(self.device)

        self.model.eval()
        self.eos_token_id = self.tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # IMPROVED ChatML Prompt (more stable, less verbose, more accurate)
    # ------------------------------------------------------------------
    def build_prompt(self, context: str, question: str) -> str:
        return (
            "<|im_start|>system\n"
            "Answer ONLY using facts from the context. "
            "Do not add outside knowledge. Keep the answer concise.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    # ------------------------------------------------------------------
    # POST-PROCESSING: clean tokens, remove artifacts, enforce clarity
    # ------------------------------------------------------------------
    def _clean_output(self, text: str) -> str:
        # Remove ChatML echo
        if "<|im_start|>assistant" in text:
            text = text.split("<|im_start|>assistant")[-1]

        # Remove OpenELM artifact characters
        for garb in ["<<<<<", "<<<<", "<<<", "<<", "<"]:
            text = text.replace(garb, "")

        # Remove extra whitespace
        text = " ".join(text.split())

        # Trim at first punctuation (OpenELM tends to babble after)
        for sep in [".", "?", "!"]:
            if sep in text:
                text = text.split(sep)[0] + sep
                break

        return text.strip()

    # ------------------------------------------------------------------
    # ANSWER GENERATION
    # ------------------------------------------------------------------
    @torch.no_grad()
    def answer(self, context: str, question: str, max_new_tokens: int = 80) -> str:
        """
        Generates a clean, short, factual answer.
        Improved with:
            - use_cache=False (Critical)
            - deterministic decoding
            - repetition_penalty
            - upgraded prompt
            - cleaned output
        """
        prompt = self.build_prompt(context, question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,               # Deterministic, stable answers
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.12,       # Reduce <<<<< artifacts
            use_cache=False,               # Critical fix for DynamicCache
            pad_token_id=self.eos_token_id,
            eos_token_id=self.eos_token_id,
        )

        raw_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        cleaned = self._clean_output(raw_text)
        return cleaned

# ===================================================================
# STRUCTURE JSON HANDLING
# ===================================================================

def get_structure_json(doc_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    structure_path = out_dir / f"{doc_path.stem}_structure.json"

    if structure_path.exists():
        log.info(f"[STRUCTURE] Using existing file: {structure_path}")
        return structure_path

    log.info(f"[STRUCTURE] Extracting → {structure_path}")
    extractor = DocumentStructureExtractor(str(doc_path))
    structure = extractor.extract()
    extractor.save(structure, structure_path)
    return structure_path


# ===================================================================
# MAIN Q&A PIPELINE
# ===================================================================

def run_option_c(
    structure_json: Path,
    out_dir: Path,
    max_chunks: int | None = None,
    min_context_len: int = 40,
) -> Path:

    import time  # timing added here

    log.info("[STAGE 2] Loading cleaned chunks…")

    loader = StructureChunkLoader(
        structure_path=str(structure_json),
        min_length=min_context_len,
        dedupe=True,
        merge_headings=True,
    )

    clean_chunks = loader.load_clean_chunks()
    log.info(f"[CHUNKS] Loaded {len(clean_chunks)} chunks")

    if max_chunks is not None:
        clean_chunks = clean_chunks[:max_chunks]
        log.info(f"[CHUNKS] Truncated to {max_chunks}")

    # ----------------------------------------------------
    # Load ALL models once
    # ----------------------------------------------------
    flan    = FLAN_QA_Model(DEFAULT_FLAN)
    tiny    = TinyLlamaAnswerGenerator(MODELS_TINY_LLAMA_DIR)
    qwen    = QwenAnswerGenerator(MODELS_QWEN_DIR)
    gemma   = GemmaAnswerGenerator(MODELS_GEMMA_DIR)
    openelm = OpenELMAnswerGenerator(MODELS_APPLE_ELM_DIR)

    # ----------------------------------------------------
    # Prepare output
    # ----------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    base = structure_json.stem.replace("_structure", "")
    out_path = out_dir / f"{base}_multi_model_optionC.jsonl"

    log.info(f"[OUTPUT] → {out_path}")

    written = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, chunk in enumerate(clean_chunks, start=1):

            context = (chunk.get("pipeline_context") or chunk.get("text") or "").strip()

            if not context or len(context) < min_context_len:
                log.debug(f"[SKIP {idx}] Too short ({len(context)} chars)")
                continue

            chunk_id = chunk.get("chunk_id")
            page = chunk.get("page")
            section = chunk.get("section")
            subsection = chunk.get("subsection")

            log.info("────────────────────────────────────")
            log.info(f"[CHUNK {idx}] id={chunk_id}, page={page}")
            log.debug(f"[CTX] {context[:200]}...")

            # --------------------------------------------------------
            # MULTI–QUESTION GENERATION (FLAN)
            # --------------------------------------------------------
            try:
                t0 = time.time()
                questions = flan.generate_questions(context, n=3)
                log.info(f"[Qx{len(questions)}] {questions}")
                log.info(f"[TIMING] Question generation took {time.time() - t0:.2f}s")
            except Exception as e:
                log.error(f"[ERROR] Failed to generate questions: {e}")
                continue

            # --------------------------------------------------------
            # LOOP: Run answers for each question
            # --------------------------------------------------------
            for q_index, q in enumerate(questions, start=1):
                log.info(f"[Q{q_index}] {q}")

                # --------------------------------------------------------
                # ANSWER – FLAN
                # --------------------------------------------------------
                try:
                    t0 = time.time()
                    a_flan = flan.generate_answer(context, q)
                    log.info(f"[A-FLAN] {a_flan}")
                    log.info(f"[TIMING] FLAN took {time.time() - t0:.2f}s")
                except Exception as e:
                    log.error(f"[ERROR] FLAN failed: {e}")
                    a_flan = None

                # --------------------------------------------------------
                # ANSWER – TinyLlama
                # --------------------------------------------------------
                try:
                    t0 = time.time()
                    a_tiny = tiny.generate_answer(context, q)
                    log.info(f"[A-TINY] {a_tiny}")
                    log.info(f"[TIMING] TinyLlama took {time.time() - t0:.2f}s")
                except Exception as e:
                    log.error(f"[ERROR] TinyLlama failed: {e}")
                    a_tiny = None

                # --------------------------------------------------------
                # ANSWER – Qwen 2.5–3B
                # --------------------------------------------------------
                try:
                    t0 = time.time()
                    a_qwen = qwen.generate_answer(context, q)
                    log.info(f"[A-QWEN] {a_qwen}")
                    log.info(f"[TIMING] Qwen took {time.time() - t0:.2f}s")
                except Exception as e:
                    log.error(f"[ERROR] Qwen failed: {e}")
                    a_qwen = None

                # --------------------------------------------------------
                # ANSWER – Gemma 2–2B IT
                # --------------------------------------------------------
                try:
                    t0 = time.time()
                    a_gemma = gemma.generate_answer(context, q)
                    log.info(f"[A-GEMMA] {a_gemma}")
                    log.info(f"[TIMING] Gemma took {time.time() - t0:.2f}s")
                except Exception as e:
                    log.error(f"[ERROR] Gemma failed: {e}")
                    a_gemma = None

                # --------------------------------------------------------
                # ANSWER – Apple OpenELM 1.1B
                # --------------------------------------------------------
                try:
                    t0 = time.time()
                    a_openelm = openelm.answer(context, q, max_new_tokens=40)
                    log.info(f"[A-OPENELM] {a_openelm}")
                    log.info(f"[TIMING] OpenELM took {time.time() - t0:.2f}s")
                except Exception as e:
                    log.error(f"[ERROR] OpenELM failed: {e}")
                    a_openelm = None

                # --------------------------------------------------------
                # WRITE RECORD (one per question)
                # --------------------------------------------------------
                record = {
                    "chunk_id": chunk_id,
                    "page": page,
                    "section": section,
                    "subsection": subsection,

                    "question_index": q_index,
                    "question": q,

                    "answer_flan": a_flan,
                    "answer_tiny": a_tiny,
                    "answer_qwen": a_qwen,
                    "answer_gemma": a_gemma,
                    "answer_openelm": a_openelm,

                    "context": context,
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    log.info("────────────────────────────────────")
    log.info(f"[DONE] Wrote {written} Q&A pairs → {out_path}")
    return out_path



# ===================================================================
# CLI
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Option C: Multi-model Q&A generator")
    p.add_argument("input", help="Document path or *_structure.json")
    p.add_argument("--out-dir", default="optionC_outputs")
    p.add_argument("--structure-dir", default="structure_maps")
    p.add_argument("--max-chunks", type=int, default=None)
    p.add_argument("--min-context-len", type=int, default=40)
    return p.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input)

    if not inp.exists():
        log.error(f"Input not found: {inp}")
        sys.exit(1)

    if inp.suffix.lower() == ".json" and inp.name.endswith("_structure.json"):
        structure_json = inp
    else:
        structure_json = get_structure_json(inp, Path(args.structure_dir))

    run_option_c(
        structure_json,
        Path(args.out_dir),
        max_chunks=args.max_chunks,
        min_context_len=args.min_context_len,
    )


if __name__ == "__main__":
    main()
