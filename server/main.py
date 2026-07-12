from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uuid
from .data_analyzer import analyze_yolo_dataset
from .trainer import train_yolo_async, get_task_status, get_task_results

app = FastAPI(title="YOLO Auto-Research GPU Server")

class AnalyzeRequest(BaseModel):
    data_yaml_path: str

class TrainRequest(BaseModel):
    model_name: str
    data_yaml_path: str
    # Core train settings
    epochs: int = 50
    batch_size: int = 16
    imgsz: int = 640
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    distill_model: str = None,
    cos_lr: bool = False
    close_mosaic: int = 10
    amp: bool = True
    patience: int = 20
    single_cls: bool = False
    multi_scale: float = 0.0
    freeze: int = 0
    dropout: float = 0.0
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    # Augmentation hyperparameters
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    bgr: float = 0.0
    mosaic: float = 1.0
    mixup: float = 0.0
    cutmix: float = 0.0

@app.post("/dataset/analyze")
def analyze_dataset(req: AnalyzeRequest):
    stats = analyze_yolo_dataset(req.data_yaml_path)
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
        optimizer=req.optimizer,
        lr0=req.lr0,
        lrf=req.lrf,
        momentum=req.momentum,
        weight_decay=req.weight_decay,
        warmup_epochs=req.warmup_epochs,
        warmup_momentum=req.warmup_momentum,
        warmup_bias_lr=req.warmup_bias_lr,
        distill_model=req.distill_model,
        cos_lr=req.cos_lr,
        close_mosaic=req.close_mosaic,
        amp=req.amp,
        patience=req.patience,
        single_cls=req.single_cls,
        multi_scale=req.multi_scale,
        freeze=req.freeze,
        dropout=req.dropout,
        box=req.box,
        cls=req.cls,
        dfl=req.dfl,
        hsv_h=req.hsv_h,
        hsv_s=req.hsv_s,
        hsv_v=req.hsv_v,
        degrees=req.degrees,
        translate=req.translate,
        scale=req.scale,
        shear=req.shear,
        perspective=req.perspective,
        flipud=req.flipud,
        fliplr=req.fliplr,
        bgr=req.bgr,
        mosaic=req.mosaic,
        mixup=req.mixup,
        cutmix=req.cutmix,
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
