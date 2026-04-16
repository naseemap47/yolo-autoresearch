from langchain.tools import tool
from ultralytics import YOLO
from typing import Dict, Any
import httpx


async def train_model(
        model_name: str, data_path: str, epochs: int,
        patience: int, batch: int | float,
        imgsz: int, device: int | str | list,
        name: str, optimizer: str, single_cls: bool,
        lr0: float, lrf: float, momentum: float, weight_decay: float,
):
    model = YOLO(model_name)
    result =model.train(
        data=data_path, epochs=epochs,
        patience=patience, batch=batch,
        imgsz=imgsz, device=device,
        name=name, optimizer=optimizer,
        single_cls=single_cls, lr0=lr0,
        lrf=lrf, momentum=momentum,
        weight_decay=weight_decay,
    )
    return result

async def evaluate_model(model_path):
    model = YOLO(model_path)
    results = model.val()
    return results


# @mcp.tool()
# async def train_yolo_model(
#     model_name: str, data_path: str, epochs: int,
#     patience: int, batch: int | float,
#     imgsz: int, device: int | str | list,
#     name: str, optimizer: str, single_cls: bool,
#     lr0: float, lrf: float, momentum: float, weight_decay: float,
# ):
#     """
#     Tool to train YOLO model using ultralytics library.

#     Args:
#         model_name: The name of the YOLO model to train (eg. "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt")
#         data_path: The path to the training data (eg. "data.yaml")
#         epochs: Total number of training epochs. Each epoch represents a full pass over the entire dataset. Adjusting this value can affect training duration and model performance for (eg. 100)
#         patience: Number of epochs to wait without improvement in validation metrics before early stopping the training. Helps prevent overfitting by stopping training when performance plateaus. (eg. 10)
#         batch: Batch size, with three modes: set as an integer (e.g., batch=16), auto mode for 60% GPU memory utilization (batch=-1), or auto mode with specified utilization fraction (batch=0.70).
#         imgsz: Target image size for training. Images are resized to squares with sides equal to the specified value (if rect=False), preserving aspect ratio for YOLO models but not RT-DETR. Affects model accuracy and computational complexity. (eg. 640)
#         device: Specifies the computational device(s) for training: a single GPU (device=0), multiple GPUs (device=[0,1]), CPU (device=cpu), MPS for Apple silicon (device=mps), Huawei Ascend NPU (device=npu or device=npu:0), or auto-selection of most idle GPU (device=-1) or multiple idle GPUs (device=[-1,-1])
#         name: Name of the training run. Used for creating a subdirectory within the project folder, where training logs and outputs are stored. keep it unique foer each experiment. (eg. "experiment_1", "exp3")
#         optimizer: The optimizer to use for training the YOLO model. (options: "SGD", "MuSGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "auto")
#         single_cls: Treats all classes in multi-class datasets as a single class during training. Useful for binary classification tasks or when focusing on object presence rather than classification. (eg. False)
#         lr0: Initial learning rate (i.e. SGD=1E-2, Adam=1E-3). Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated. (eg. 0.01)
#         lrf: Final learning rate as a fraction of the initial rate = (lr0 * lrf), used in conjunction with schedulers to adjust the learning rate over time. (eg. 0.01)
#         momentum: Momentum factor for SGD or beta1 for Adam optimizers, influencing the incorporation of past gradients in the current update. (eg. 0.937)
#         weight_decay: L2 regularization term, penalizing large weights to prevent overfitting. (eg. 0.0005)

#     Returns:
#         trained model evaluation results.
#         {
#             "model_path": path to the best model weights (best.pt),
#             "curves": ['Precision-Recall(B)', 'F1-Confidence(B)', 'Precision-Confidence(B)', 'Recall-Confidence(B)'],
#             "curves_results": list of arrays containing the results for each curve,
#             "results": {
#                 'metrics/precision(B)': precision value, 
#                 'metrics/recall(B)': recall value, 
#                 'metrics/mAP50(B)': mAP50 value, 
#                 'metrics/mAP50-95(B)': mAP50-95 value, 
#                 'fitness': fitness value
#             },
#         }
#     """
#     train_result = await train_model(
#         model_name, data_path, epochs,
#         patience, batch, imgsz, device,
#         name, optimizer, single_cls,
#         lr0, lrf, momentum, weight_decay,
#     )
#     model_path = os.path.join(train_result.save_dir, "weights", "best.pt")
#     evaluate = await evaluate_model(model_path)

#     return {
#         "model_path": model_path,
#         "curves": evaluate.curves,
#         "curves_results": evaluate.curves_results,
#         "results": evaluate.results_dict,
#     }

@tool
async def train_model(
    model_name: str,
    data_path: str,
    epochs: int,
    patience: int,
    batch: int | float,
    imgsz: int,
    device: int | str | list,
    name: str,
    optimizer: str = "auto",
    single_cls: bool = False,
    lr0: float = 0.01,
    lrf: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
    workers: int = 8
) -> Dict[str, Any]:
    """
    Tool to train YOLO model using ultralytics library.

    Args:
        model_name: The name of the YOLO model to train (eg. "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt")
        data_path: The path to the training data (eg. "data.yaml")
        epochs: Total number of training epochs. Each epoch represents a full pass over the entire dataset. Adjusting this value can affect training duration and model performance for (eg. 100)
        patience: Number of epochs to wait without improvement in validation metrics before early stopping the training. Helps prevent overfitting by stopping training when performance plateaus. (eg. 10)
        batch: Batch size, with three modes: set as an integer (e.g., batch=16), auto mode for 60% GPU memory utilization (batch=-1), or auto mode with specified utilization fraction (batch=0.70).
        imgsz: Target image size for training. Images are resized to squares with sides equal to the specified value (if rect=False), preserving aspect ratio for YOLO models but not RT-DETR. Affects model accuracy and computational complexity. (eg. 640)
        device: Specifies the computational device(s) for training: a single GPU (device=0), multiple GPUs (device=[0,1]), CPU (device=cpu), MPS for Apple silicon (device=mps), Huawei Ascend NPU (device=npu or device=npu:0), or auto-selection of most idle GPU (device=-1) or multiple idle GPUs (device=[-1,-1])
        name: Name of the training run. Used for creating a subdirectory within the project folder, where training logs and outputs are stored. keep it unique foer each experiment. (eg. "experiment_1", "exp3")
        optimizer: The optimizer to use for training the YOLO model. (options: "SGD", "MuSGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "auto")
        single_cls: Treats all classes in multi-class datasets as a single class during training. Useful for binary classification tasks or when focusing on object presence rather than classification. (eg. False)
        lr0: Initial learning rate (i.e. SGD=1E-2, Adam=1E-3). Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated. (eg. 0.01)
        lrf: Final learning rate as a fraction of the initial rate = (lr0 * lrf), used in conjunction with schedulers to adjust the learning rate over time. (eg. 0.01)
        momentum: Momentum factor for SGD or beta1 for Adam optimizers, influencing the incorporation of past gradients in the current update. (eg. 0.937)
        weight_decay: L2 regularization term, penalizing large weights to prevent overfitting. (eg. 0.0005)
        workers: Number of worker threads for data loading. (eg. 8)

    Returns:
        trained model evaluation results.
    """
    request_data = {
        "model_name": model_name,
        "data_path": data_path,
        "epochs": epochs,
        "patience": patience,
        "batch": batch,
        "imgsz": imgsz,
        "device": device,
        "name": name,
        "optimizer": optimizer,
        "single_cls": single_cls,
        "lr0": lr0,
        "lrf": lrf,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "workers": workers
    }
    
    async with httpx.AsyncClient(timeout=10000) as client:
        response = await client.post("http://192.168.0.47:8080/train", json=request_data)
        return response.json()
