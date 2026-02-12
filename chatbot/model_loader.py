import torch
import yaml
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------
# MLflow Models Root
# ---------------------------------------------------------

MLRUNS_PATH = Path(
    "/mnt/c/Users/operator/PycharmProjects/gpu_train/mlruns/models"
)


# ---------------------------------------------------------
# Model Manager
# ---------------------------------------------------------

class ModelManager:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.loaded = None  # (model_name, version)

        print(f"[ModelManager] Using device: {self.device}")

    # -----------------------------------------------------
    # List Available Versions
    # -----------------------------------------------------

    def list_versions(self, model_name: str):

        model_dir = MLRUNS_PATH / model_name

        if not model_dir.exists():
            return []

        versions = []

        for item in model_dir.iterdir():
            if item.is_dir() and item.name.startswith("version-"):
                try:
                    number = int(item.name.replace("version-", ""))
                    versions.append(number)
                except ValueError:
                    continue

        return sorted(versions)

    # -----------------------------------------------------
    # Load Specific Version
    # -----------------------------------------------------

    def load_version(self, model_name: str, version: int):

        # -----------------------------------------------------
        # Skip reload if already loaded
        # -----------------------------------------------------
        if self.loaded == (model_name, version):
            print(f"[ModelManager] {model_name} v{version} already loaded.")
            return

        print(f"[ModelManager] Loading {model_name} v{version}...")

        # -----------------------------------------------------
        # Locate MLflow version directory
        # -----------------------------------------------------
        version_dir = MLRUNS_PATH / model_name / f"version-{version}"
        meta_path = version_dir / "meta.yaml"

        if not meta_path.exists():
            raise FileNotFoundError(f"[ModelManager] meta.yaml not found at {meta_path}")

        # -----------------------------------------------------
        # Read MLflow meta.yaml
        # -----------------------------------------------------
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f)

        storage_location = meta.get("storage_location")

        if not storage_location:
            raise ValueError("[ModelManager] storage_location missing in meta.yaml")

        # -----------------------------------------------------
        # Properly resolve file:// URI
        # -----------------------------------------------------
        from urllib.parse import urlparse

        parsed = urlparse(storage_location)

        if parsed.scheme == "file":
            model_root = Path(parsed.path)  # preserves leading slash
        else:
            model_root = Path(storage_location)

        model_path = model_root / "model"

        # -----------------------------------------------------
        # Normalize Windows-style paths if needed
        # -----------------------------------------------------
        model_path = model_path.resolve()

        if not model_path.exists():
            raise FileNotFoundError(
                f"[ModelManager] Model folder not found at {model_path}"
            )

        print(f"[ModelManager] Resolved model path: {model_path}")

        # -----------------------------------------------------
        # Load tokenizer
        # -----------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        )

        # -----------------------------------------------------
        # Load model
        # -----------------------------------------------------
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
            local_files_only=True,
            trust_remote_code=True
        )

        if self.device.type == "cpu":
            self.model.to(self.device)

        self.model.eval()

        self.loaded = (model_name, version)

        print(f"[ModelManager] {model_name} v{version} loaded successfully.")

    # -----------------------------------------------------
    # Generate Response
    # -----------------------------------------------------

    def generate(self, prompt: str):

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_version() first.")

        start = time.time()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        latency = round(time.time() - start, 3)

        input_tokens = inputs["input_ids"].shape[1]
        output_tokens = outputs.shape[1]

        return {
            "response": response,
            "latency": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
