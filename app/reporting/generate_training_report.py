import os
from pathlib import Path
from datetime import datetime
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader


# -------------------------------------------------
# FIXED REPORT OUTPUT LOCATION
# -------------------------------------------------

# Resolve to: gpu_train/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Final path:
# C:\Users\operator\PycharmProjects\gpu_train\app\reporting\reports
REPORT_BASE_DIR = PROJECT_ROOT / "app" / "reporting" / "reports"
REPORT_BASE_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def get_metric_history(run_id: str, metric_name: str) -> pd.DataFrame:
    client = mlflow.tracking.MlflowClient()
    history = client.get_metric_history(run_id, metric_name)

    if not history:
        return pd.DataFrame(columns=["step", "value", "timestamp"])

    return pd.DataFrame(
        {
            "step": [m.step for m in history],
            "value": [m.value for m in history],
            "timestamp": [m.timestamp for m in history],
        }
    )


def plot_metric(df: pd.DataFrame, title: str, ylabel: str, output_path: Path) -> bool:
    if df.empty:
        return False

    plt.figure(figsize=(8, 4))
    plt.plot(df["step"], df["value"])
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return True


# -------------------------------------------------
# Main entry
# -------------------------------------------------

def generate_training_report(
    *,
    run_id: str,
    output_dir: Path | None = None,
    export_pdf: bool = False,
) -> Path:
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    # Default output directory: per-run folder
    if output_dir is None:
        output_dir = REPORT_BASE_DIR / run_id

    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Metadata
    # -----------------------------
    info = run.info
    data = run.data

    start = datetime.fromtimestamp(info.start_time / 1000)
    end = datetime.fromtimestamp(info.end_time / 1000) if info.end_time else None

    context = {
        "run_id": info.run_id,
        "status": info.status,
        "start_time": start,
        "end_time": end,
        "duration_sec": (end - start).total_seconds() if end else None,
        "params": data.params,
        "metrics": data.metrics,
    }

    # -----------------------------
    # Metrics
    # -----------------------------
    loss_df = get_metric_history(run_id, "train_loss")
    lr_df = get_metric_history(run_id, "lr")
    gpu_df = get_metric_history(run_id, "gpu.utilization_pct")

    # -----------------------------
    # Plots
    # -----------------------------
    plots = {
        "loss": plot_metric(loss_df, "Training Loss", "Loss", output_dir / "loss.png"),
        "lr": plot_metric(lr_df, "Learning Rate", "LR", output_dir / "lr.png"),
        "gpu": plot_metric(
            gpu_df,
            "GPU Utilization",
            "Utilization (%)",
            output_dir / "gpu.png",
        ),
    }

    context["plots"] = plots

    # -----------------------------
    # HTML rendering
    # -----------------------------
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent),
        autoescape=True,
    )

    template = env.from_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Training Report – {{ run_id }}</title>

        <style>
            @page {
                size: A4;
                margin: 20mm;
            }

            html, body {
                margin: 0 !important;
                padding: 0 !important;
                width: 100%;
            }

            body {
                font-family: Arial, sans-serif;
                font-size: 13px;
            }

            h1, h2 {
                color: #333;
                text-align: left;
                margin: 24px 0 12px 0;
            }

            h1 {
                margin-top: 0;
                font-size: 24px;
            }

            h2 {
                font-size: 18px;
            }

            .meta {
                margin-bottom: 24px;
                font-size: 12px;
            }

            img {
                width: 100%;
                max-width: 800px;
                margin: 12px 0 8px 0;
                display: block;
            }

            .plot-description {
                font-size: 11px;
                color: #555;
                line-height: 1.5;
                margin-bottom: 24px;
                max-width: 800px;
            }

            table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 24px;
                table-layout: fixed;
                font-size: 12px;
            }

            th, td {
                padding: 6px 8px;
                border: 1px solid #ccc;
                text-align: left;
                vertical-align: top;
                word-wrap: break-word;
            }

            th {
                background-color: #f3f3f3;
                font-weight: bold;
            }

            /* Parameters table specific widths */
            .params-table col:nth-child(1) {
                width: 30%;
            }
            .params-table col:nth-child(2) {
                width: 25%;
            }
            .params-table col:nth-child(3) {
                width: 45%;
            }

            /* Metrics table specific widths */
            .metrics-table col:nth-child(1) {
                width: 25%;
            }
            .metrics-table col:nth-child(2) {
                width: 20%;
            }
            .metrics-table col:nth-child(3) {
                width: 55%;
            }

            .param-desc, .metric-desc {
                font-size: 11px;
                color: #555;
                line-height: 1.3;
            }
        </style>
    </head>

    <body>

    <h1>Training Report</h1>

    <div class="meta">
        <b>Run ID:</b> {{ run_id }}<br>
        <b>Status:</b> {{ status }}<br>
        <b>Start:</b> {{ start_time }}<br>
        <b>End:</b> {{ end_time }}<br>
        <b>Duration (sec):</b> {{ duration_sec }}
    </div>

    <h2>Parameters</h2>
    <table class="params-table">
        <colgroup>
            <col style="width: 30%;">
            <col style="width: 25%;">
            <col style="width: 45%;">
        </colgroup>
        <tr>
            <th>Parameter</th>
            <th>Value</th>
            <th>Description</th>
        </tr>
        {% for k, v in params.items() %}
        <tr>
            <td>{{ k }}</td>
            <td>{{ v }}</td>
            <td class="param-desc">
                {% if k == 'learning_rate' %}
                    Step size for weight updates during training (lower = more cautious learning)
                {% elif k == 'max_seq_length' %}
                    Maximum input sequence length in tokens (longer = more context but more memory)
                {% elif k == 'mixed_precision' %}
                    Precision format for calculations (bf16 = faster training with lower memory usage)
                {% elif k == 'nnodes' %}
                    Number of machines/nodes used for distributed training
                {% elif k == 'node_rank' %}
                    Identifier for this node in multi-node training (0 = primary node)
                {% elif k == 'num_train_epochs' %}
                    Number of complete passes through the training dataset
                {% elif k == 'batch_size' or k == 'per_device_train_batch_size' %}
                    Number of training samples processed together per device
                {% elif k == 'gradient_accumulation_steps' %}
                    Number of steps to accumulate gradients before updating (simulates larger batch)
                {% elif k == 'warmup_ratio' %}
                    Fraction of training for learning rate warmup (stabilizes early training)
                {% elif k == 'weight_decay' %}
                    Regularization to prevent overfitting (penalizes large weights)
                {% elif k == 'seed' %}
                    Random seed for reproducible training runs
                {% elif k == 'epochs' %}
                    Total number of training epochs
                {% elif k == 'enable_lora' %}
                    Whether LoRA (Low-Rank Adaptation) fine-tuning is enabled
                {% elif k == 'fsdp_sharding_strategy' %}
                    How model parameters are split across GPUs (FULL_SHARD = maximum memory savings)
                {% else %}
                    &nbsp;
                {% endif %}
            </td>
        </tr>
        {% endfor %}
    </table>

    <h2>Metrics</h2>
    <table class="metrics-table">
        <colgroup>
            <col style="width: 25%;">
            <col style="width: 20%;">
            <col style="width: 55%;">
        </colgroup>
        <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Description</th>
        </tr>

        {% for k, v in metrics.items() %}
        <tr>
            <td>{{ k }}</td>
            <td>{{ v }}</td>
            <td class="metric-desc">
                {% if k == 'train_loss' %}
                    Final training loss value (lower is better)
                {% elif k == 'lr' %}
                    Learning rate at final training step
                {% elif k.startswith('gpu_') %}
                    GPU utilization / memory telemetry collected during training
                {% elif k == 'grad_norm' %}
                    Gradient norm magnitude (stability indicator)
                {% else %}
                    &nbsp;
                {% endif %}
            </td>
        </tr>
        {% endfor %}
    </table>

    {% if plots.loss %}
    <h2>Loss Curve</h2>
    <img src="loss.png">
    <p class="plot-description">
        <b>How to read this chart:</b> The loss curve shows how well the model is learning over time. 
        The y-axis shows the loss value (lower is better), and the x-axis shows training steps. 
        A good training run shows a general downward trend, indicating the model is improving. 
        Some fluctuation is normal, but large spikes or plateaus may indicate learning rate issues, 
        data quality problems, or the need for more training steps.
    </p>
    {% endif %}

    {% if plots.lr %}
    <h2>Learning Rate</h2>
    <img src="lr.png">
    <p class="plot-description">
        <b>How to read this chart:</b> The learning rate controls how much the model adjusts its weights during training. 
        This chart shows how the learning rate changes over training steps. 
        Most schedules start with a warmup phase (gradual increase), then decay (gradual decrease) toward the end. 
        The warmup helps stabilize early training, while the decay allows the model to fine-tune and converge. 
        A schedule that decays too quickly may prevent the model from learning effectively.
    </p>
    {% endif %}

    {% if plots.gpu %}
    <h2>GPU Utilization</h2>
    <img src="gpu.png">
    <p class="plot-description">
        <b>How to read this chart:</b> This chart shows GPU utilization percentage over time. 
        Higher utilization (closer to 100%) indicates more efficient use of your GPU resources. 
        Low utilization may suggest bottlenecks in data loading, small batch sizes, or inefficient code. 
        Consistent high utilization is ideal for minimizing training time and maximizing hardware investment.
    </p>
    {% endif %}

    </body>
    </html>
    """)

    html_path = output_dir / "training_report.html"
    html_path.write_text(template.render(**context), encoding="utf-8")

    # -----------------------------
    # Optional PDF
    # -----------------------------
    if export_pdf:
        try:
            from weasyprint import HTML
            pdf_path = output_dir / "training_report.pdf"
            HTML(filename=str(html_path)).write_pdf(pdf_path)
        except Exception as e:
            print(f"[WARN] PDF export failed: {e}")

        # -----------------------------
        # Attach report to MLflow run
        # -----------------------------
        try:
            import mlflow

            # Ensure we're pointing at the same tracking backend
            mlflow.set_tracking_uri(mlflow.get_tracking_uri())

            active_run = mlflow.active_run()

            if active_run is None:
                raise RuntimeError(
                    "No active MLflow run found. "
                    "Reports must be generated inside an existing run context."
                )

            # Sanity check: ensure we're attaching to the correct run
            if active_run.info.run_id != run_id:
                raise RuntimeError(
                    f"Active MLflow run mismatch: "
                    f"expected={run_id} actual={active_run.info.run_id}"
                )

            # --------------------------------------------------
            # SAFE artifact logging (NEVER to root)
            # --------------------------------------------------
            mlflow.log_artifact(
                str(html_path),
                artifact_path="reports"
            )

            if export_pdf:
                pdf_path = output_dir / "training_report.pdf"
                if pdf_path.exists():
                    mlflow.log_artifact(
                        str(pdf_path),
                        artifact_path="reports"
                    )

        except Exception as e:
            print(f"[WARN] Failed to attach report to MLflow run: {e}")


    return html_path

# -------------------------------------------------
# CLI
# -------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate MLflow training report")
    parser.add_argument("--run-id", required=True, help="MLflow run ID")
    parser.add_argument("--pdf", action="store_true", help="Export PDF report")

    args = parser.parse_args()

    report_path = generate_training_report(
        run_id=args.run_id,
        export_pdf=args.pdf,
    )

    print("[OK] Training report generated:")
    print(f"     {report_path}")

"""python app/reporting/generate_training_report.py \
  --run-id 0f6d2ada7a914580b7705fc82adec6d8 \
  --pdf
"""