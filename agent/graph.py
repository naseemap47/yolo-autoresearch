from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from .api_client import GPUClient
from .llm_planner import Planner
import mlflow


class AgentState(TypedDict):
    dataset_yaml_path: str
    target_map: float
    max_cycles: int
    current_cycle: int
    dataset_stats: Dict[str, Any]
    history: List[Dict[str, Any]]
    current_config: Dict[str, Any]
    final_result: str


def create_agent_graph(
    host: str = "192.168.0.84",
    model_name: str = "qwen3.5:9b",
    retriever: Optional[object] = None,
    target_map: float = 0.8,
    mlflow_cfg: Optional[Dict[str, Any]] = None,
):
    client = GPUClient(host=host)
    planner = Planner(model_name=model_name, retriever=retriever, target_map=target_map)

    # ------------------------------------------------------------------ #
    # MLflow setup (non-fatal — agent continues even if server is down)   #
    # ------------------------------------------------------------------ #
    mlflow_cfg = mlflow_cfg or {}
    tracking_uri    = mlflow_cfg.get("tracking_uri", "http://127.0.0.1:5000")
    experiment_name = mlflow_cfg.get("experiment_name", "yolo-autoresearch")
    mlflow_enabled  = bool(mlflow_cfg)

    if mlflow_enabled:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            print(f"[MLflow] Connected — experiment '{experiment_name}' at {tracking_uri}")
        except Exception as mlflow_init_err:
            print(f"[MLflow] WARNING: Could not connect to tracking server ({mlflow_init_err}). "
                  f"Running without experiment tracking. Start the server with: mlflow ui --port 5000")
            mlflow_enabled = False

    def analyze_data(state: AgentState):
        print(f"Cycle {state['current_cycle'] + 1}: Analyzing dataset...")
        if not state.get("dataset_stats"):
            stats = client.analyze_dataset(state["dataset_yaml_path"])
            state["dataset_stats"] = stats
        return state

    def plan_model(state: AgentState):
        print(f"Cycle {state['current_cycle'] + 1}: Planning next model...")
        config = planner.plan(state["dataset_stats"], state["history"])
        print(f"Proposed Config: {config}")
        state["current_config"] = config
        return state

    def train_model(state: AgentState):
        config = state["current_config"]
        batch_size = config.get("batch_size", 16)
        last_error = None

        while True:
            print(f"Cycle {state['current_cycle'] + 1}: Sending training request (batch_size={batch_size})...")
            try:
                task_id = client.start_training(
                    model_name=config.get("model_name", "yolov8n.pt"),
                    data_yaml_path=state["dataset_yaml_path"],
                    epochs=config.get("epochs", 10),
                    batch_size=batch_size,
                    imgsz=config.get("imgsz", 640),
                    lr0=config.get("lr0", 0.01),
                    weight_decay=config.get("weight_decay", 0.0005),
                    close_mosaic=config.get("close_mosaic", 10),
                )
                print(f"Training started with task ID: {task_id}")

                # Wait for training
                results = client.wait_for_training(task_id, poll_interval=10)
                print(f"Training results: {results}")

                # Update config with the batch size that actually succeeded
                config["batch_size"] = batch_size
                state["current_config"] = config

                # ---------------------------------------------------- #
                # MLflow — log orchestrator-side run for this cycle      #
                # ---------------------------------------------------- #
                if mlflow_enabled:
                    try:
                        cycle_num = state["current_cycle"] + 1
                        run_name  = f"cycle-{cycle_num}-{config.get('model_name', 'yolo').replace('.pt', '')}"
                        with mlflow.start_run(run_name=run_name):
                            mlflow.log_params({
                                "model_name":   config.get("model_name"),
                                "epochs":       config.get("epochs"),
                                "batch_size":   batch_size,
                                "imgsz":        config.get("imgsz"),
                                "lr0":          config.get("lr0"),
                                "weight_decay": config.get("weight_decay"),
                                "close_mosaic": config.get("close_mosaic"),
                            })
                            mlflow.log_metrics({
                                "mAP50_95": results.get("mAP50-95", 0),
                                "mAP50":    results.get("mAP50", 0),
                                "fitness":  results.get("fitness", 0),
                                "cycle":    cycle_num,
                            })
                            mlflow.set_tags({
                                "model_name": config.get("model_name"),
                                "cycle":      str(cycle_num),
                                "task_id":    task_id,
                                "source":     "agent_orchestrator",
                                "status":     "success",
                            })
                    except Exception as mlflow_err:
                        print(f"[MLflow] WARNING: Could not log cycle run ({mlflow_err}).")

                # Save to history
                record = {
                    "cycle":  state["current_cycle"] + 1,
                    "config": config,
                    "results": results,
                }
                state["history"].append(record)
                state["current_cycle"] += 1
                return state

            except Exception as e:
                last_error = e
                print(f"Training failed or connection error: {e}")

                if batch_size > 1:
                    new_batch_size = max(1, batch_size // 2)
                    print(f"Reducing batch size from {batch_size} to {new_batch_size} and retrying...")
                    batch_size = new_batch_size
                    import time
                    time.sleep(5)
                    continue
                else:
                    print("Batch size is already 1. Cannot reduce further.")
                    # Log failed run to MLflow
                    if mlflow_enabled:
                        try:
                            cycle_num = state["current_cycle"] + 1
                            with mlflow.start_run(run_name=f"cycle-{cycle_num}-FAILED"):
                                mlflow.log_params({
                                    "model_name": config.get("model_name"),
                                    "batch_size": batch_size,
                                })
                                mlflow.set_tags({
                                    "cycle":  str(cycle_num),
                                    "source": "agent_orchestrator",
                                    "status": "failed",
                                    "error":  str(last_error)[:250],
                                })
                        except Exception:
                            pass
                    state["final_result"] = f"Training failed even with batch_size=1. Reason: {last_error}"
                    return state

    def evaluate(state: AgentState):
        print("Evaluating results...")
        if state.get("final_result"):
            return state
        latest_results = state["history"][-1]["results"]
        map50_95 = latest_results.get("mAP50-95", 0)

        if map50_95 >= state["target_map"]:
            state["final_result"] = f"Success! Target reached with mAP50-95 = {map50_95}"
            return state

        if state["current_cycle"] >= state["max_cycles"]:
            state["final_result"] = f"Failed to reach target after {state['max_cycles']} cycles. Best mAP was not met."
            return state

        return state

    def should_continue(state: AgentState):
        if state.get("final_result"):
            return "end"
        return "continue"

    workflow = StateGraph(AgentState)
    workflow.add_node("analyze", analyze_data)
    workflow.add_node("plan", plan_model)
    workflow.add_node("train", train_model)
    workflow.add_node("evaluate", evaluate)

    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "plan")
    workflow.add_edge("plan", "train")
    workflow.add_edge("train", "evaluate")

    workflow.add_conditional_edges(
        "evaluate",
        should_continue,
        {
            "end": END,
            "continue": "plan",
        },
    )

    return workflow.compile()
