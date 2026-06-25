from typing import List, Dict, Any, TypedDict, Optional


class AgentState(TypedDict):
    # ── User / Project Inputs ─────────────────────────────────────────────────
    project_name:        str
    project_description: str
    experiment_name:     str
    dataset_path:        str   # server-side path on the GPU machine
    target_accuracy:     float
    max_cycles:          int
    api_url:             str   # FastAPI training server URL
    mlflow_uri:          str

    # ── Agent State ───────────────────────────────────────────────────────────
    current_cycle:       int
    run_id:              str
    data_stats:          Dict[str, Any]
    current_model:       str   # full model id, e.g. "yolov8n", "yolo11s", "yolo26m"
    current_hyperparams: Dict[str, Any]
    # Each entry: {cycle, model, hyperparams, metrics, reasoning}
    history:             List[Dict[str, Any]]
    status:              str
    error:               Optional[str]
