"""
Resume a previous MLflow run using its logged parameters.
"""

from app.launchers.launch_base import launch_training
import mlflow
import argparse


def main():
    parser = argparse.ArgumentParser(description="Resume training from an MLflow run")
    parser.add_argument("--run-id", required=True, help="MLflow run ID to resume")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load parameters from the previous run
    run = mlflow.get_run(args.run_id)
    payload = {k: v for k, v in run.data.params.items()}

    payload["resume"] = True  # Signal trainer to resume from checkpoint

    launch_training(
        payload,
        enable_mlflow=True,
        experiment=run.info.experiment_id,
        run_name=f"resume_{run.info.run_name}",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()\n