import os
from ultralytics import YOLO
import threading

# Simple dictionary to keep track of tasks (in-memory for this simple prototype)
# In production, use a database or Redis
tasks = {}

def train_yolo_async(task_id: str, model_name: str, data_yaml: str, epochs: int, batch_size: int, imgsz: int, lr0: float):
    def training_thread():
        try:
            tasks[task_id] = {"status": "training", "progress": "started", "metrics": None, "error": None}
            
            # Load model
            model = YOLO(model_name)
            
            # Train model
            # Save results in a specific project directory
            project_dir = f"runs/train/{task_id}"
            
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=imgsz,
                lr0=lr0,
                project="runs/train",
                name=task_id,
                device="0", # Assuming GPU 0
                exist_ok=True
            )
            
            # Extract final metrics
            # the results object contains metrics
            metrics_dict = {
                "mAP50-95": results.box.map,
                "mAP50": results.box.map50,
                "fitness": results.fitness
            }
            
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
