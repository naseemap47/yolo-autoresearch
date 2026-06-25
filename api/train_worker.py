import os
import sys
import json
import argparse
import pandas as pd
from ultralytics import YOLO


# ── Supported YOLO model families ────────────────────────────────────────────
# The model_scale arg accepts a full model id like "yolov8n", "yolo11s", "yolo26m".
# If the user only passes a scale letter (n/s/m/l/x), we default to yolov8.

SCALE_LETTERS = {"n", "s", "m", "l", "x"}


def resolve_model_name(model_scale: str) -> str:
    """
    Normalize the model identifier to a .pt filename.

    Accepts any of:
      - Full name with extension:  "yolov8n.pt", "yolo11s.pt"
      - Full name without extension: "yolov8n", "yolo11s", "yolo26m"
      - Bare scale letter: "n", "s", "m", "l", "x"  → defaults to "yolov8{scale}.pt"
    """
    s = model_scale.strip()
    if s.endswith(".pt"):
        return s
    if s in SCALE_LETTERS:
        return f"yolov8{s}.pt"
    return f"{s}.pt"


def train_yolo(
    project_name: str,
    experiment_name: str,
    dataset_path: str,
    model_scale: str,
    hyperparams_json: str,
    output_dir: str,
    mlflow_uri: str,
):
    # Set MLflow env vars before importing ultralytics (it auto-integrates)
    os.environ["MLFLOW_TRACKING_URI"]  = mlflow_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = project_name
    os.environ["MLFLOW_RUN_NAME"]       = experiment_name

    print("--- Training Worker Started ---")
    print(f"Project     : {project_name}")
    print(f"Experiment  : {experiment_name}")
    print(f"Dataset     : {dataset_path}")
    print(f"Model input : {model_scale}")
    print(f"MLflow URI  : {mlflow_uri}")

    # Parse hyperparams
    try:
        hyperparams = json.loads(hyperparams_json)
    except Exception as e:
        print(f"[WARN] Failed to parse hyperparams JSON: {e}. Using defaults.")
        hyperparams = {}

    model_name = resolve_model_name(model_scale)
    print(f"Loading model: {model_name}")

    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"[ERROR] Could not load model '{model_name}': {e}")
        _write_status(output_dir, experiment_name, "failed", {}, str(e))
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    train_args = {
        "data":         dataset_path,
        "epochs":       hyperparams.get("epochs",       10),
        "batch":        hyperparams.get("batch",        16),
        "imgsz":        hyperparams.get("imgsz",        640),
        "lr0":          hyperparams.get("lr0",          0.01),
        "lrf":          hyperparams.get("lrf",          0.01),
        "optimizer":    hyperparams.get("optimizer",    "auto"),
        "weight_decay": hyperparams.get("weight_decay", 0.0005),
        "momentum":     hyperparams.get("momentum",     0.937),
        "mosaic":       hyperparams.get("mosaic",       1.0),
        "close_mosaic": hyperparams.get("close_mosaic", 10),
        "project":      output_dir,
        "name":         experiment_name,
        "exist_ok":     True,
        "verbose":      True,
    }
    print(f"Training args: {train_args}")

    try:
        results = model.train(**train_args)
        print("Training completed successfully!")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        _write_status(output_dir, experiment_name, "failed", {}, str(e))
        sys.exit(1)

    # ── Extract metrics ───────────────────────────────────────────────────────
    run_dir  = os.path.join(output_dir, experiment_name)
    csv_path = os.path.join(run_dir, "results.csv")
    metrics: dict = {}

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df.columns = [col.strip() for col in df.columns]
            if len(df) > 0:
                last = df.iloc[-1]
                metrics = {
                    "epoch":          int(last.get("epoch", len(df))),
                    "train_box_loss": float(last.get("train/box_loss",      0.0)),
                    "train_cls_loss": float(last.get("train/cls_loss",      0.0)),
                    "train_dfl_loss": float(last.get("train/dfl_loss",      0.0)),
                    "val_box_loss":   float(last.get("val/box_loss",        0.0)),
                    "val_cls_loss":   float(last.get("val/cls_loss",        0.0)),
                    "val_dfl_loss":   float(last.get("val/dfl_loss",        0.0)),
                    "precision":      float(last.get("metrics/precision(B)",  0.0)),
                    "recall":         float(last.get("metrics/recall(B)",     0.0)),
                    "map50":          float(last.get("metrics/mAP50(B)",      0.0)),
                    "map50_95":       float(last.get("metrics/mAP50-95(B)",   0.0)),
                }
                print(f"Final metrics: {metrics}")
        except Exception as e:
            print(f"[WARN] Error parsing results.csv: {e}")

    # Fallback to results object
    if not metrics and hasattr(results, "results_dict"):
        try:
            rd = results.results_dict
            metrics = {
                "precision": float(rd.get("metrics/precision(B)", 0.0)),
                "recall":    float(rd.get("metrics/recall(B)",    0.0)),
                "map50":     float(rd.get("metrics/mAP50(B)",     0.0)),
                "map50_95":  float(rd.get("metrics/mAP50-95(B)", 0.0)),
            }
        except Exception as e:
            print(f"[WARN] Could not read results_dict: {e}")

    best_weights = os.path.join(run_dir, "weights", "best.pt")
    _write_status(output_dir, experiment_name, "completed", metrics, best_model_path=best_weights)
    print(f"Status saved to {os.path.join(run_dir, 'status.json')}")


def _write_status(
    output_dir: str,
    experiment_name: str,
    status: str,
    metrics: dict,
    error: str = "",
    best_model_path: str = "",
):
    run_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(run_dir, exist_ok=True)
    payload = {"status": status, "metrics": metrics}
    if error:
        payload["error"] = error
    if best_model_path:
        payload["best_model_path"] = best_model_path
    with open(os.path.join(run_dir, "status.json"), "w") as f:
        json.dump(payload, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Training Subprocess Worker")
    parser.add_argument("--project-name",    required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--dataset-path",    required=True)
    parser.add_argument("--model-scale",     required=True,
                        help="Full model id, e.g. yolov8n, yolo11s, yolo26m, or bare scale n/s/m/l/x")
    parser.add_argument("--hyperparams",     required=True, help="JSON string of hyperparameters")
    parser.add_argument("--output-dir",      required=True)
    parser.add_argument("--mlflow-uri",      default="http://localhost:5000")
    args = parser.parse_args()

    train_yolo(
        project_name    = args.project_name,
        experiment_name = args.experiment_name,
        dataset_path    = args.dataset_path,
        model_scale     = args.model_scale,
        hyperparams_json= args.hyperparams,
        output_dir      = args.output_dir,
        mlflow_uri      = args.mlflow_uri,
    )
