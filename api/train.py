from fastapi import FastAPI, HTTPException
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os


class TrainRequest(BaseModel):
    model_name: str
    data_path: str
    epochs: int
    patience: int
    batch: int | float
    imgsz: int
    device: int | str | list
    name: str
    optimizer: str = "auto"
    single_cls: bool = False
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    workers: int = 8

def make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    else:
        return obj

app = FastAPI(title="YOLO Training API")

@app.post("/train")
async def train_yolo_model(request: TrainRequest):
    """
    Endpoint to train YOLO model using ultralytics library.

    Args:
        model_name: The name of the YOLO model to train (eg. "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt")
        data_path: The path to the training data yaml file. The yaml file should be in the format required by ultralytics library. (eg. "data.yaml")
        epochs: The number of epochs to train the model. (eg. 100)
        patience: Number of epochs with no improvement after which training will be stopped. (eg. 10)
        batch: The batch size for training. It can be an integer or a float representing the fraction of the dataset to use as batch size. (eg. 16 or 0.1)
        imgsz: The image size for training. It can be an integer or a tuple representing the width and height of the images. (eg. 640 or (640, 640))
        device: The device to use for training. It can be an integer representing the GPU id, a string representing the device type ("cpu" or "cuda"), or a list of integers representing multiple GPU ids for distributed training. (eg. 0 or "cuda" or [0, 1])
        name: Name of the training run. Used for creating a subdirectory within the project folder, where training logs and outputs are stored. keep it unique foer each experiment. (eg. "experiment_1", "exp3")
        optimizer: The optimizer to use for training the YOLO model. (options: "SGD", "MuSGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "auto")
        single_cls: Treats all classes in multi-class datasets as a single class during training. Useful for binary classification tasks or when focusing on object presence rather than classification. (eg. False)
        lr0: Initial learning rate (i.e. SGD=1E-2, Adam=1E-3). Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated. (eg. 0.01)
        lrf: Final learning rate as a fraction of the initial rate = (lr0 * lrf), used in conjunction with schedulers to adjust the learning rate over time. (eg. 0.01)
        momentum: Momentum factor for SGD or beta1 for Adam optimizers, influencing the incorporation of past gradients in the current update. (eg. 0.937)
        weight_decay: L2 regularization term, penalizing large weights to prevent overfitting. (eg. 0.0005)
        workers: Number of worker threads for data loading (per RANK if Multi-GPU training). Influences the speed of data preprocessing and feeding into the model, especially useful in multi-GPU setups. (eg. 8)
        
    Returns:
        A dictionary containing the path to the best model weights, evaluation results.
    """
    try:
        model = YOLO(request.model_name)
        result = model.train(
            data=request.data_path, epochs=request.epochs,
            patience=request.patience, batch=request.batch,
            imgsz=request.imgsz, device=request.device,
            name=request.name, optimizer=request.optimizer,
            single_cls=request.single_cls, lr0=request.lr0,
            lrf=request.lrf, momentum=request.momentum,
            weight_decay=request.weight_decay, workers=request.workers,
        )
        model_path = os.path.join(result.save_dir, "weights", "best.pt")
        # Evaluate the trained model and return the results
        # You can implement your evaluation logic here and return the results in the desired format
        return {
            "model_path": model_path,
            "results": make_serializable(getattr(result, 'results_dict', {})),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "train:app",
        host="localhost",
        port=8080,
        reload=True
    )
