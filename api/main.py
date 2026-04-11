# api/main.py
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from core.graph import YOLOResearchGraph
import uuid

app = FastAPI()
research_graph = YOLOResearchGraph()

class ResearchRequest(BaseModel):
    research_goal: str
    dataset_path: str
    base_model: str = "yolov8n.pt"
    max_iterations: int = 10
    target_metric: str = "metrics/mAP50-95(B)"
    target_threshold: float = 0.85

@app.post("/research/start")
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    
    initial_state = {
        "messages": [],
        "iteration": 0,
        "research_goal": request.research_goal,
        "dataset_info": {"path": request.dataset_path},
        "base_model": request.base_model,
        "current_model_path": request.base_model,
        "best_model_path": "",
        "best_metrics": {},
        "experiment_history": [],
        "failed_attempts_count": 0,
        "consecutive_failures": 0,
        "should_continue": True,
        "max_iterations": request.max_iterations,
        "target_metric": request.target_metric,
        "target_threshold": request.target_threshold,
        "last_error": None,
        "recovery_attempts": 0,
        "thread_id": task_id
    }
    
    # Run research in background
    background_tasks.add_task(research_graph.run_research, initial_state)
    
    return {"task_id": task_id, "status": "started"}

@app.get("/research/status/{task_id}")
async def get_research_status(task_id: str):
    # Retrieve state from checkpoint
    config = {"configurable": {"thread_id": task_id}}
    state = await research_graph.graph.aget_state(config)
    
    if state:
        return {
            "task_id": task_id,
            "iteration": state.values.get("iteration", 0),
            "best_metrics": state.values.get("best_metrics", {}),
            "current_hypothesis": state.values.get("current_hypothesis", ""),
            "experiment_count": len(state.values.get("experiment_history", []))
        }
    
    return {"error": "Task not found"}