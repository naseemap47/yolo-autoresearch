import os
from ultralytics import YOLO
import threading

# Simple dictionary to keep track of tasks (in-memory for this simple prototype)
# In production, use a database or Redis
tasks = {}

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
    # Additional train settings
    optimizer: str = "auto",
    lrf: float = 0.01,
    momentum: float = 0.937,
    warmup_epochs: float = 3.0,
    warmup_momentum: float = 0.8,
    warmup_bias_lr: float = 0.1,
    # distill_model: str = None,
    cos_lr: bool = False,
    amp: bool = True,
    patience: int = 20,
    single_cls: bool = False,
    multi_scale: float = 0.0,
    freeze: int = 0,
    dropout: float = 0.0,
    box: float = 7.5,
    cls: float = 0.5,
    dfl: float = 1.5,
    # Augmentation hyperparameters
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 0.0,
    perspective: float = 0.0,
    flipud: float = 0.0,
    fliplr: float = 0.5,
    bgr: float = 0.0,
    mosaic: float = 1.0,
    mixup: float = 0.0,
    cutmix: float = 0.0,
):
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
                optimizer=optimizer,
                lr0=lr0,
                lrf=lrf,
                momentum=momentum,
                weight_decay=weight_decay,
                warmup_epochs=warmup_epochs,
                warmup_momentum=warmup_momentum,
                warmup_bias_lr=warmup_bias_lr,
                # distill_model=distill_model,
                cos_lr=cos_lr,
                close_mosaic=close_mosaic,
                amp=amp,
                patience=patience,
                single_cls=single_cls,
                multi_scale=multi_scale,
                freeze=freeze,
                dropout=dropout,
                box=box,
                cls=cls,
                dfl=dfl,
                hsv_h=hsv_h,
                hsv_s=hsv_s,
                hsv_v=hsv_v,
                degrees=degrees,
                translate=translate,
                scale=scale,
                shear=shear,
                perspective=perspective,
                flipud=flipud,
                fliplr=fliplr,
                bgr=bgr,
                mosaic=mosaic,
                mixup=mixup,
                cutmix=cutmix,
                project="runs/train",
                name=task_id,
                device="0",
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
