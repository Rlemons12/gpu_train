"""
Simple learning-rate grid search launcher.
"""

from app.launchers.launch_base import launch_training

# Values to sweep over
LEARNING_RATES = [2e-6, 5e-6, 1e-5]

BASE_PAYLOAD = {
    "base_model_path": "/mnt/c/Users/operator/emtac/models/llm/mistral-7b-instruct",
    "train_data_path": "/mnt/c/Users/operator/PycharmProjects/gpu_train/dataset_output/dataset.jsonl",
    "output_dir": "/mnt/c/Users/operator/PycharmProjects/gpu_train/runs/grid",

    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "max_seq_length": 1024,
    "mixed_precision": "bf16",
}

for lr in LEARNING_RATES:
    payload = BASE_PAYLOAD | {
        "job_name": f"grid_lr_{lr}",
        "learning_rate": lr,
    }

    launch_training(
        payload,
        enable_mlflow=True,
        experiment="gpu_train",
        run_name=payload["job_name"],
    )\n