# dataset_qanda_generator/models/flan_qa.py

import os
import time
import json
import logging
import threading
import re
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

log = logging.getLogger(__name__)


class FLANCacheError(RuntimeError):
    pass


class FLAN_QA_Model:
    """
    FLAN-T5-Large offline model.

    This version:
      • Extracts FACTS (not questions)
      • Converts facts → questions deterministically
      • Is fully drop-in compatible with the existing pipeline
    """

    DEFAULT_CONTEXT_CHAR_CAP = 1500
    DEFAULT_GENERATE_TIMEOUT_S = 180

    _UNRENDERED_RE = re.compile(r"{\s*[a-zA-Z0-9_]+\s*}")

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------
    def __init__(
        self,
        device: str | None = "cpu",
        *,
        context_char_cap: int | None = None,
        generate_timeout_s: int | None = None,
    ):
        t0 = time.time()

        hf_home_raw = os.getenv("HF_HOME")
        if not hf_home_raw:
            raise RuntimeError("HF_HOME environment variable is not set")

        # Normalize Windows → WSL paths
        if len(hf_home_raw) > 2 and hf_home_raw[1:3] == ":\\":  # C:\...
            drive = hf_home_raw[0].lower()
            rest = hf_home_raw[3:].replace("\\", "/")
            hf_home = Path(f"/mnt/{drive}/{rest}")
            log.info("[FLAN] Normalized HF_HOME (win→wsl): %s -> %s", hf_home_raw, hf_home)
        else:
            hf_home = Path(hf_home_raw)

        hf_home = hf_home.resolve()

        self.device = (device or "cpu").lower()
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("[FLAN] CUDA requested but unavailable; forcing CPU.")
            self.device = "cpu"

        self.context_char_cap = context_char_cap or self.DEFAULT_CONTEXT_CHAR_CAP
        self.generate_timeout_s = generate_timeout_s or self.DEFAULT_GENERATE_TIMEOUT_S

        snapshot_root = hf_home / "models--google--flan-t5-large" / "snapshots"
        if not snapshot_root.exists():
            raise FileNotFoundError(f"FLAN snapshot root missing: {snapshot_root}")

        snapshot = sorted(snapshot_root.iterdir())[-1]
        self._verify_snapshot(snapshot)

        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            snapshot,
            local_files_only=True,
            torch_dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()

        log.info("[FLAN] Model ready in %.2fs", time.time() - t0)

    # ------------------------------------------------------------------
    def _verify_snapshot(self, snapshot: Path) -> None:
        required_files = [
            "config.json",
            "tokenizer.json",
        ]

        missing = [
            f for f in required_files
            if not (snapshot / f).exists()
        ]

        if missing:
            raise FLANCacheError(
                f"FLAN snapshot missing required files: {missing}"
            )

    # ------------------------------------------------------------------
    def _validate_prompt(self, prompt: str) -> None:
        if self._UNRENDERED_RE.search(prompt):
            raise RuntimeError(
                "[FLAN] Unrendered prompt placeholders detected.\n"
                f"Prompt preview:\n{prompt[:300]}"
            )

    # ------------------------------------------------------------------
    def _generate(self, *, inputs, gen_kwargs, label: str):
        stop = {"done": False}

        def heartbeat():
            start = time.time()
            while not stop["done"]:
                time.sleep(30)
                log.info("[FLAN:%s] generating… %ds", label, int(time.time() - start))

        threading.Thread(target=heartbeat, daemon=True).start()
        out = self.model.generate(**inputs, **gen_kwargs)
        stop["done"] = True
        return out

    # ------------------------------------------------------------------
    # FACT EXTRACTION
    # ------------------------------------------------------------------
    def _extract_facts(self, text: str) -> List[str]:
        lines = []
        for ln in text.splitlines():
            ln = ln.strip(" -*\t")
            if len(ln) > 20 and not ln.endswith("?"):
                lines.append(ln)
        return lines

    # ------------------------------------------------------------------
    # FACT → QUESTION
    # ------------------------------------------------------------------
    @staticmethod
    def _fact_to_question(fact: str) -> str:
        fact = fact.rstrip(".")

        lower = fact.lower()

        if lower.startswith("the person"):
            return "What must the person completing the action do?"

        if lower.startswith("the technician"):
            return "What must the technician do?"

        if "date of release" in lower:
            return "What is the date of release of the document?"

        if "procedure" in lower:
            return "What is the procedure described?"

        if lower.startswith("unused pages"):
            return "Why are unused pages required?"

        # fallback (still safe)
        return f"What is stated regarding {fact.lower()}?"

    # ------------------------------------------------------------------
    # QUESTION GENERATION (PIPELINE ENTRY POINT)
    # ------------------------------------------------------------------
    def generate_questions(
            self,
            prompt: str,
            *,
            n: int,
            max_len: int = 64,
            timeout_s: Optional[int] = None,
    ) -> List[str]:

        self._validate_prompt(prompt)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output = self._generate(
            inputs=inputs,
            gen_kwargs=dict(
                max_new_tokens=max_len,
                do_sample=False,
                num_beams=1,
            ),
            label="Q",
        )

        raw = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

        # --- HARD SANITY FILTERS ---
        q = raw.replace("\n", " ").strip()

        if not q.endswith("?"):
            log.warning("[FLAN:Q] Rejected (no '?'): %r", q)
            return []

        if len(q) > 200:
            log.warning("[FLAN:Q] Rejected (too long): %d chars", len(q))
            return []

        if re.search(r"\d+\.\d+\.\d+\.\d+", q):
            log.warning("[FLAN:Q] Rejected (clause ref): %r", q)
            return []

        if not re.match(r"^(what|who|when|where|why|how|which)\b", q, re.I):
            log.warning("[FLAN:Q] Rejected (not interrogative): %r", q)
            return []

        log.info("[FLAN:Q] Generated question: %r", q)
        return [q]

    # ------------------------------------------------------------------
    # ANSWER GENERATION (UNCHANGED)
    # ------------------------------------------------------------------
    def generate_answer(
        self,
        context: str,
        question: str,
        max_len: int = 64,
        *,
        timeout_s: Optional[int] = None,
    ) -> str:

        ctx = (context or "")[: self.context_char_cap]

        prompt = (
            "Use ONLY the context below to answer.\n\n"
            f"CONTEXT:\n{ctx}\n\n"
            f"QUESTION:\n{question}\n\n"
            "ANSWER:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output = self._generate(
            inputs=inputs,
            gen_kwargs=dict(
                max_new_tokens=max_len,
                do_sample=False,
                num_beams=1,
            ),
            label="A",
        )

        text = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return text[7:].strip() if text.lower().startswith("answer:") else text
