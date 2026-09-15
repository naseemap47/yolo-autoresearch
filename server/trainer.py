import os
import yaml
import threading
from ultralytics import YOLO
import mlflow

# Simple dictionary to keep track of tasks (in-memory for this simple prototype)
# In production, use a database or Redis
tasks = {}

# ------------------------------------------------------------------ #
# MLflow helpers                                                       #
# ------------------------------------------------------------------ #

def _load_mlflow_cfg() -> dict:
    """Load mlflow config from settings.yaml if available."""
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")
    try:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("mlflow", {})
    except Exception:
        return {}


def _setup_mlflow(mlflow_cfg: dict):
    """Configure MLflow tracking URI and experiment."""
    tracking_uri = mlflow_cfg.get("tracking_uri", "http://127.0.0.1:5000")
    experiment_name = mlflow_cfg.get("experiment_name", "yolo-autoresearch")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def train_yolo_async(
    task_id: str,
    model_name: str,
    data_yaml: str,
    epochs: int,
    batch_size: int,
    imgsz: int,
    lr0: float,
    weight_decay: float = 0.0005,
    close_mosaic: int = 10,
):
    def training_thread():
        try:
            tasks[task_id] = {"status": "training", "progress": "started", "metrics": None, "error": None}

            # Load model
            model = YOLO(model_name)

            # Train model — save results in a per-task directory
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=imgsz,
                lr0=lr0,
                weight_decay=weight_decay,
                close_mosaic=close_mosaic,
                project="runs/train",
                name=task_id,
                device="0",
                exist_ok=True,
            )

            # Extract final metrics
            metrics_dict = {
                "mAP50-95": results.box.map,
                "mAP50": results.box.map50,
                "fitness": results.fitness,
            }

            # -------------------------------------------------------- #
            # MLflow — log this training run                            #
            # -------------------------------------------------------- #
            try:
                mlflow_cfg = _load_mlflow_cfg()
                _setup_mlflow(mlflow_cfg)

                with mlflow.start_run(run_name=f"gpu-{task_id[:8]}"):
                    # Hyperparameters
                    mlflow.log_params({
                        "model_name":   model_name,
                        "epochs":       epochs,
                        "batch_size":   batch_size,
                        "imgsz":        imgsz,
                        "lr0":          lr0,
                        "weight_decay": weight_decay,
                        "close_mosaic": close_mosaic,
                    })
                    # Metrics
                    mlflow.log_metrics({
                        "mAP50_95": metrics_dict["mAP50-95"],
                        "mAP50":    metrics_dict["mAP50"],
                        "fitness":  metrics_dict["fitness"],
                    })
                    # Tags
                    mlflow.set_tags({
                        "task_id": task_id,
                        "source":  "gpu_server",
                    })
            except Exception as mlflow_err:
                print(f"[MLflow] WARNING: Could not log GPU run ({mlflow_err}). Training results are still saved.")

            tasks[task_id]["status"] = "completed"
            tasks[task_id]["metrics"] = metrics_dict

        except Exception as e:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = str(e)

    thread = threading.Thread(target=training_thread)
    thread.start()
    return task_id


def get_task_status(task_id: str):
    return tasks.get(task_id, {"status": "not_found"})


def get_task_results(task_id: str):
    task = tasks.get(task_id)
    if task and task.get("status") == "completed":
        return task.get("metrics")
    return None
