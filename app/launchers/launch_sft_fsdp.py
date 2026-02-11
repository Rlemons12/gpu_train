"""
SFT + FSDP launcher with MLflow tracking enabled.
(EXPLICIT trainer + LoRA enforcement)
"""

from app.launchers.launch_base import launch_training
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Launch SFT FSDP training job with MLflow"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without launching",
    )
    args = parser.parse_args()

    payload = {
        # ------------------------------
        # REQUIRED: Trainer selection
        # ------------------------------
        "trainer": "sft_fsdp",          # 🔒 NO fallback
        "enable_lora": True,            # 🔒 Forces SFT path

        # ------------------------------
        # Job identity
        # ------------------------------
        "job_name": "mistral7b_mini_lora",

        # ------------------------------
        # Model + dataset
        # ------------------------------
        "base_model_path": "/mnt/c/Users/operator/emtac/models/llm/mistral-7b-instruct",
        "train_data_path": "/mnt/c/Users/operator/PycharmProjects/gpu_train/dataset_output/train_prompt_response.jsonl",
        "output_dir": "/mnt/c/Users/operator/PycharmProjects/gpu_train/prod_out/mistral7b_mini",

        # ------------------------------
        # Training
        # ------------------------------
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "learning_rate": 2e-6,
        "max_seq_length": 1024,
        "mixed_precision": "bf16",

        # ------------------------------
        # LoRA (explicit, even though defaults exist)
        # ------------------------------
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        # Leave target modules auto-detected (OpenELM-safe)

        # ------------------------------
        # Training policy (model registry)
        # ------------------------------
        "training_policy": {
            "register_model": True,
            "registry_name": "emtac_mistral_sft",
        },
    }

    launch_training(
        payload,
        enable_mlflow=True,
        experiment="gpu_train",
        run_name=payload["job_name"],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
