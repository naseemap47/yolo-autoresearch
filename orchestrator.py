import os
import asyncio
from typing import Dict, Any, Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import core.graph as cg
from core.graph import workflow_app

app = FastAPI(title="YOLO AutoResearch Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state to keep track of the running agent
agent_state_store = {
    "is_running": False,
    "state": {}
}

class StartAgentRequest(BaseModel):
    project_name: str
    project_description: str
    experiment_name: str
    dataset_path: str
    target_accuracy: float
    max_cycles: int
    api_url: str
    mlflow_uri: str
    ollama_url: str

def run_agent_sync(inputs: Dict[str, Any], ollama_url: str):
    """Runs the LangGraph workflow synchronously."""
    try:
        cg.OLLAMA_DEFAULT_URL = ollama_url
        agent_state_store["state"] = inputs
        agent_state_store["is_running"] = True
        
        for output in workflow_app.stream(inputs):
            for node_name, update in output.items():
                agent_state_store["state"].update(update)
                if update.get("error"):
                    break
        
        if not agent_state_store["state"].get("error"):
            agent_state_store["state"]["status"] = "Autoresearch completed."
            
    except Exception as e:
        agent_state_store["state"]["error"] = str(e)
        agent_state_store["state"]["status"] = f"Workflow error: {e}"
    finally:
        agent_state_store["is_running"] = False

@app.post("/api/agent/start")
def start_agent(req: StartAgentRequest, background_tasks: BackgroundTasks):
    if agent_state_store["is_running"]:
        raise HTTPException(status_code=400, detail="Agent is already running.")
    
    inputs = {
        "project_name": req.project_name,
        "project_description": req.project_description,
        "experiment_name": req.experiment_name,
        "dataset_path": req.dataset_path,
        "target_accuracy": req.target_accuracy,
        "max_cycles": req.max_cycles,
        "api_url": req.api_url,
        "mlflow_uri": req.mlflow_uri,
        "current_cycle": 0,
        "run_id": "",
        "data_stats": {},
        "current_model": "",
        "current_hyperparams": {},
        "history": [],
        "status": "Starting…",
        "error": None,
    }
    
    background_tasks.add_task(run_agent_sync, inputs, req.ollama_url)
    return {"message": "Agent started successfully"}

@app.get("/api/agent/state")
def get_agent_state():
    return {
        "is_running": agent_state_store["is_running"],
        "state": agent_state_store["state"]
    }

@app.delete("/api/agent/stop")
def stop_agent():
    # Currently, LangGraph doesn't support clean interruption mid-node natively without complex state management.
    # We will mark it as not running, but the current node will finish executing.
    agent_state_store["is_running"] = False
    agent_state_store["state"]["error"] = "Agent stopped by user."
    agent_state_store["state"]["status"] = "Stopped."
    return {"message": "Stop signal sent."}

# Mount the static frontend directory
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))
else:
    @app.get("/")
    def no_frontend():
        return {"message": "Frontend directory not found"}
