import os
import json
import time
import requests
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from core.state import AgentState

# ── LLM Config (overridable from UI / env vars) ───────────────────────────────
OLLAMA_DEFAULT_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL          = os.environ.get("LLM_MODEL",   "qwen2.5:3b")

# ── YOLO model families the LLM can choose from ───────────────────────────────
YOLO_FAMILIES = [
    "yolov5", "yolov6", "yolov8",
    "yolo11",  "yolo12",
    "yolo26",
]
YOLO_SCALES = ["n", "s", "m", "l", "x"]


# ── Ollama helper ─────────────────────────────────────────────────────────────
def query_ollama(prompt: str, system_prompt: str, ollama_url: str = OLLAMA_DEFAULT_URL) -> Dict[str, Any]:
    """Query Ollama and return parsed JSON. Returns {} on failure."""
    url = f"{ollama_url}/api/generate"
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2},
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"[Ollama] Query failed: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# Node 1: Analyze Data
# ══════════════════════════════════════════════════════════════════════════════
def analyze_data_node(state: AgentState) -> Dict[str, Any]:
    print("--- Node: Analyze Data ---")
    api_url      = state.get("api_url", "http://localhost:8000")
    dataset_path = state.get("dataset_path", "")

    try:
        resp = requests.post(f"{api_url}/api/analyze-dataset",
                             json={"dataset_path": dataset_path}, timeout=60)
        resp.raise_for_status()
        return {"data_stats": resp.json(), "status": "Dataset analyzed successfully."}
    except Exception as e:
        return {"error": f"Dataset analysis failed: {e}", "status": "failed"}


# ══════════════════════════════════════════════════════════════════════════════
# Node 2: LLM Select Initial Model
# ══════════════════════════════════════════════════════════════════════════════
def llm_select_model_node(state: AgentState) -> Dict[str, Any]:
    print("--- Node: LLM Select Model ---")
    if state.get("error"):
        return {}

    stats    = state.get("data_stats", {})
    bbox_dist = stats.get("bbox_distribution", {})
    ollama_url = state.get("api_url", OLLAMA_DEFAULT_URL)   # reuse or store separately

    prompt = f"""
Dataset Statistics:
- Number of classes    : {stats.get("num_classes", 0)}
- Class Names          : {stats.get("class_names", [])}
- Total Training Images: {stats.get("total_images", 0)}
- BBox Size Distribution:
    * Small  bboxes: {bbox_dist.get("small", 0)}%
    * Medium bboxes: {bbox_dist.get("medium", 0)}%
    * Large  bboxes: {bbox_dist.get("large", 0)}%

Available YOLO families : {YOLO_FAMILIES}
Available scales        : {YOLO_SCALES}

Selection guidelines:
1. More small bboxes (>30%) → prefer larger imgsz (960) and bigger scale (m/l) or newer family (yolo11+).
2. Small dataset (<200 images) → prefer smaller scale (n/s), fewer epochs (10-15).
3. Large dataset (>2000 images) → prefer larger scale (m/l), more epochs (30-50).
4. For best accuracy on new datasets, prefer yolov8 or yolo11 family.
5. yolo26 is for cutting-edge experiments with large datasets.

Return ONLY valid JSON with this exact schema:
{{
  "reasoning": "short explanation",
  "model_family": "yolov8",
  "model_scale": "n",
  "hyperparams": {{
    "epochs": 10,
    "batch": 16,
    "imgsz": 640,
    "lr0": 0.01,
    "lrf": 0.01,
    "optimizer": "SGD",
    "weight_decay": 0.0005,
    "momentum": 0.937,
    "mosaic": 1.0,
    "close_mosaic": 10
  }}
}}
"""
    system = "You are an expert YOLO computer vision researcher. Output only valid JSON matching the requested schema."
    rec = query_ollama(prompt, system, state.get("api_url", OLLAMA_DEFAULT_URL).replace("/api", "").rstrip("/"))

    family     = rec.get("model_family", "yolov8")
    scale      = rec.get("model_scale",  "n")
    hyperparams = rec.get("hyperparams", {})
    reasoning  = rec.get("reasoning",   "Default initial selection.")

    # Validate
    if family not in YOLO_FAMILIES:
        family = "yolov8"
    if scale not in YOLO_SCALES:
        scale = "n"

    full_model = f"{family}{scale}"

    # Fill missing hyperparams with defaults
    defaults = {"epochs": 10, "batch": 16, "imgsz": 640, "lr0": 0.01, "lrf": 0.01,
                "optimizer": "SGD", "weight_decay": 0.0005, "momentum": 0.937,
                "mosaic": 1.0, "close_mosaic": 10}
    for k, v in defaults.items():
        hyperparams.setdefault(k, v)

    return {
        "current_model":      full_model,
        "current_hyperparams": hyperparams,
        "current_cycle":      1,
        "status": f"LLM selected {full_model}.pt — {reasoning}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Node 3: Train Model
# ══════════════════════════════════════════════════════════════════════════════
def train_model_node(state: AgentState) -> Dict[str, Any]:
    cycle = state.get("current_cycle", 1)
    print(f"--- Node: Train Model (Cycle {cycle}) ---")
    if state.get("error"):
        return {}

    api_url        = state.get("api_url", "http://localhost:8000")
    project_name   = state.get("project_name")
    experiment_name = state.get("experiment_name")
    dataset_path   = state.get("dataset_path")
    model          = state.get("current_model")
    hyperparams    = state.get("current_hyperparams")
    mlflow_uri     = state.get("mlflow_uri", "http://localhost:5000")

    cycle_run_name = f"{experiment_name}_cycle_{cycle}"

    payload = {
        "project_name":    project_name,
        "experiment_name": cycle_run_name,
        "dataset_path":    dataset_path,
        "model_scale":     model,
        "hyperparams":     hyperparams,
        "mlflow_uri":      mlflow_uri,
    }

    try:
        resp = requests.post(f"{api_url}/api/train", json=payload, timeout=30)
        resp.raise_for_status()
        run_id = resp.json().get("run_id")
        print(f"Training started, run_id={run_id}. Polling…")

        consecutive_failures = 0
        while True:
            time.sleep(10)
            try:
                sr = requests.get(f"{api_url}/api/train/status/{run_id}", timeout=30)
                sr.raise_for_status()
                sd = sr.json()
                status = sd.get("status", "running")
                prog   = sd.get("progress", {})
                print(f"  [{run_id}] {status} — "
                      f"{prog.get('pct', 0)}% "
                      f"(Epoch {prog.get('current_epoch')}/{prog.get('total_epochs')})")
                consecutive_failures = 0
                if status != "running":
                    break
            except Exception as pe:
                consecutive_failures += 1
                print(f"  [WARN] Polling failure {consecutive_failures}/5: {pe}")
                if consecutive_failures >= 5:
                    raise pe

        if status == "completed":
            return {"run_id": run_id, "status": f"Training completed (Cycle {cycle})."}
        else:
            return {"error": f"Training failed in cycle {cycle} (status={status})", "status": "failed"}

    except Exception as e:
        return {"error": f"Training process failed: {e}", "status": "failed"}


# ══════════════════════════════════════════════════════════════════════════════
# Node 4: Evaluate Results
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_results_node(state: AgentState) -> Dict[str, Any]:
    print("--- Node: Evaluate Results ---")
    if state.get("error"):
        return {}

    api_url = state.get("api_url", "http://localhost:8000")
    run_id  = state.get("run_id")
    cycle   = state.get("current_cycle", 1)

    try:
        resp = requests.get(f"{api_url}/api/train/results/{run_id}", timeout=30)
        resp.raise_for_status()
        metrics = resp.json().get("metrics", {})
    except Exception as e:
        return {"error": f"Failed to retrieve results: {e}", "status": "failed"}

    history_entry = {
        "cycle":      cycle,
        "model":      state.get("current_model"),
        "hyperparams": state.get("current_hyperparams"),
        "metrics":    metrics,
        "reasoning":  state.get("status", ""),
    }
    new_history = list(state.get("history", [])) + [history_entry]
    accuracy    = metrics.get("map50", 0.0)
    target      = state.get("target_accuracy", 0.8)
    print(f"  Cycle {cycle} — mAP50: {accuracy:.4f} (target: {target:.4f})")

    return {"history": new_history, "status": f"Cycle {cycle} evaluated. mAP50={accuracy:.4f}"}


# ══════════════════════════════════════════════════════════════════════════════
# Conditional Edge
# ══════════════════════════════════════════════════════════════════════════════
def should_continue(state: AgentState) -> str:
    if state.get("error"):
        return END

    history = state.get("history", [])
    if not history:
        return END

    latest_acc = history[-1].get("metrics", {}).get("map50", 0.0)
    target     = state.get("target_accuracy", 0.8)
    if latest_acc >= target:
        print(f"  Target reached ({latest_acc:.4f} >= {target:.4f}). Done.")
        return END

    cycle     = state.get("current_cycle", 1)
    max_cycles = state.get("max_cycles", 3)
    if cycle >= max_cycles:
        print(f"  Max cycles reached ({cycle}/{max_cycles}). Done.")
        return END

    return "continue"


# ══════════════════════════════════════════════════════════════════════════════
# Node 5: LLM Adjust Model
# ══════════════════════════════════════════════════════════════════════════════
def llm_adjust_model_node(state: AgentState) -> Dict[str, Any]:
    print("--- Node: LLM Adjust Model ---")
    if state.get("error"):
        return {}

    stats   = state.get("data_stats", {})
    history = state.get("history", [])
    target  = state.get("target_accuracy", 0.8)
    bbox    = stats.get("bbox_distribution", {})

    history_str = ""
    for h in history:
        m = h.get("metrics", {})
        history_str += f"""
  Cycle {h['cycle']}:
    Model       : {h['model']}.pt
    Hyperparams : {h['hyperparams']}
    mAP50       : {m.get('map50', 0.0):.4f}
    mAP50-95    : {m.get('map50_95', 0.0):.4f}
    Precision   : {m.get('precision', 0.0):.4f}
    Recall      : {m.get('recall', 0.0):.4f}
    Train BoxLoss: {m.get('train_box_loss', 0.0):.4f}
    Val   BoxLoss: {m.get('val_box_loss', 0.0):.4f}
"""

    prompt = f"""
Dataset:
  Classes : {stats.get("num_classes")} — {stats.get("class_names")}
  Images  : {stats.get("total_images")}
  BBox    : Small {bbox.get("small")}% | Medium {bbox.get("medium")}% | Large {bbox.get("large")}%

Target mAP50 : {target}

Previous training cycles:
{history_str}

Available YOLO families : {YOLO_FAMILIES}
Available scales        : {YOLO_SCALES}

Diagnostic rules:
1. Train loss LOW, Val loss HIGH  → overfitting: increase weight_decay, reduce lr0, reduce model_scale or epochs.
2. Both losses HIGH               → underfitting: increase epochs, lr0, scale, or try a newer family.
3. Recall LOW                     → increase mosaic augmentation, scale up model.
4. mAP50 near target but not there→ try next scale up or newer family.
5. No improvement after same model→ switch YOLO family.

Return ONLY valid JSON:
{{
  "reasoning": "diagnostic analysis and rationale",
  "model_family": "yolov8",
  "model_scale": "s",
  "hyperparams": {{
    "epochs": 20,
    "batch": 16,
    "imgsz": 640,
    "lr0": 0.01,
    "lrf": 0.01,
    "optimizer": "AdamW",
    "weight_decay": 0.0005,
    "momentum": 0.937,
    "mosaic": 1.0,
    "close_mosaic": 10
  }}
}}
"""
    system = "You are an expert YOLO computer vision researcher. Output only valid JSON."
    rec = query_ollama(prompt, system, state.get("api_url", OLLAMA_DEFAULT_URL).replace("/api", "").rstrip("/"))

    family      = rec.get("model_family", state.get("current_model", "yolov8n")[:-1] or "yolov8")
    scale       = rec.get("model_scale",  "n")
    hyperparams = rec.get("hyperparams",  {})
    reasoning   = rec.get("reasoning",   "Adjusted based on previous run.")

    if family not in YOLO_FAMILIES:
        # Try to extract family from current model name
        family = "yolov8"
    if scale not in YOLO_SCALES:
        scale = "n"

    full_model = f"{family}{scale}"

    # Merge with previous hyperparams as fallback
    prev_hp = state.get("current_hyperparams", {})
    for k, v in prev_hp.items():
        hyperparams.setdefault(k, v)

    next_cycle = state.get("current_cycle", 1) + 1

    return {
        "current_model":      full_model,
        "current_hyperparams": hyperparams,
        "current_cycle":      next_cycle,
        "status": f"LLM adjusted to {full_model}.pt for Cycle {next_cycle} — {reasoning}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Graph Compilation
# ══════════════════════════════════════════════════════════════════════════════
def compile_workflow():
    wf = StateGraph(AgentState)

    wf.add_node("analyze_data",     analyze_data_node)
    wf.add_node("llm_select_model", llm_select_model_node)
    wf.add_node("train_model",      train_model_node)
    wf.add_node("evaluate_results", evaluate_results_node)
    wf.add_node("llm_adjust_model", llm_adjust_model_node)

    wf.set_entry_point("analyze_data")
    wf.add_edge("analyze_data",     "llm_select_model")
    wf.add_edge("llm_select_model", "train_model")
    wf.add_edge("train_model",      "evaluate_results")
    wf.add_conditional_edges(
        "evaluate_results",
        should_continue,
        {"continue": "llm_adjust_model", END: END},
    )
    wf.add_edge("llm_adjust_model", "train_model")

    return wf.compile()


workflow_app = compile_workflow()
