from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uuid
from .data_analyzer import analyze_yolo_dataset
from .trainer import train_yolo_async, get_task_status, get_task_results

app = FastAPI(title="YOLO Auto-Research GPU Server")

class AnalyzeRequest(BaseModel):
    data_yaml_path: str
    sample_size: int = 100

class TrainRequest(BaseModel):
    model_name: str
    data_yaml_path: str
    epochs: int = 10
    batch_size: int = 16
    imgsz: int = 640
    lr0: float = 0.01

@app.post("/dataset/analyze")
def analyze_dataset(req: AnalyzeRequest):
    stats = analyze_yolo_dataset(req.data_yaml_path, req.sample_size)
    if "error" in stats:
        raise HTTPException(status_code=400, detail=stats["error"])
    return stats

@app.post("/train")
def start_training(req: TrainRequest):
    task_id = str(uuid.uuid4())
    train_yolo_async(
        task_id=task_id,
        model_name=req.model_name,
        data_yaml=req.data_yaml_path,
        epochs=req.epochs,
        batch_size=req.batch_size,
        imgsz=req.imgsz,
        lr0=req.lr0
    )
    return {"task_id": task_id, "status": "started"}

@app.get("/status/{task_id}")
def check_status(task_id: str):
    return get_task_status(task_id)

@app.get("/results/{task_id}")
def get_results(task_id: str):
    res = get_task_results(task_id)
    if not res:
        status = get_task_status(task_id)
        if status.get("status") != "completed":
            raise HTTPException(status_code=400, detail="Task not completed or failed.")
        raise HTTPException(status_code=404, detail="Results not found.")
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
