from pathlib import Path
from datetime import datetime
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class PortableModelExporter:

    def __init__(self, export_root: str = "exported_models"):
        self.export_root = Path(export_root)
        self.export_root.mkdir(exist_ok=True)

    def _get_next_version_path(self, model_name: str) -> Path:
        existing = sorted(self.export_root.glob(f"{model_name}_v*"))
        next_index = len(existing) + 1
        version_name = f"{model_name}_v{next_index:03d}"
        return self.export_root / version_name

    def export(
        self,
        base_model_path: str,
        trained_model_path: str,
        model_name: str,
        is_lora: bool = False,
        device: str = "cpu"
    ) -> str:

        target_dir = self._get_next_version_path(model_name)

        print(f"[EXPORT] Creating {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)

        print("[EXPORT] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(trained_model_path)

        print("[EXPORT] Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        if is_lora:
            print("[EXPORT] Merging LoRA adapter...")
            model = PeftModel.from_pretrained(model, trained_model_path)
            model = model.merge_and_unload()

        print("[EXPORT] Saving merged model...")
        model.save_pretrained(target_dir)
        tokenizer.save_pretrained(target_dir)

        print("[EXPORT] Complete.")
        return str(target_dir)
