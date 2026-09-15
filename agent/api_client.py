import requests
import time

class GPUClient:
    def __init__(self, host: str = "192.168.0.84", port: int = 8000):
        self.base_url = f"http://{host}:{port}"
        
    def analyze_dataset(self, data_yaml_path: str, sample_size: int = 100):
        url = f"{self.base_url}/dataset/analyze"
        res = requests.post(url, json={"data_yaml_path": data_yaml_path, "sample_size": sample_size})
        res.raise_for_status()
        return res.json()
        
    def start_training(
        self,
        model_name: str,
        data_yaml_path: str,
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
        url = f"{self.base_url}/train"
        data = {
            "model_name": model_name,
            "data_yaml_path": data_yaml_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "imgsz": imgsz,
            "lr0": lr0,
            "weight_decay": weight_decay,
            "close_mosaic": close_mosaic,
            # Additional train settings
            "optimizer": optimizer,
            "lrf": lrf,
            "momentum": momentum,
            "warmup_epochs": warmup_epochs,
            "warmup_momentum": warmup_momentum,
            "warmup_bias_lr": warmup_bias_lr,
            "cos_lr": cos_lr,
            "amp": amp,
            "patience": patience,
            "single_cls": single_cls,
            "multi_scale": multi_scale,
            "freeze": freeze,
            "dropout": dropout,
            "box": box,
            "cls": cls,
            "dfl": dfl,
            # Augmentation hyperparameters
            "hsv_h": hsv_h,
            "hsv_s": hsv_s,
            "hsv_v": hsv_v,
            "degrees": degrees,
            "translate": translate,
            "scale": scale,
            "shear": shear,
            "perspective": perspective,
            "flipud": flipud,
            "fliplr": fliplr,
            "bgr": bgr,
            "mosaic": mosaic,
            "mixup": mixup,
            "cutmix": cutmix,
        }
        res = requests.post(url, json=data)
        res.raise_for_status()
        return res.json()["task_id"]

        
    def check_status(self, task_id: str):
        url = f"{self.base_url}/status/{task_id}"
        res = requests.get(url)
        res.raise_for_status()
        return res.json()
        
    def get_results(self, task_id: str):
        url = f"{self.base_url}/results/{task_id}"
        res = requests.get(url)
        res.raise_for_status()
        return res.json()
        
    def wait_for_training(self, task_id: str, poll_interval: int = 10):
        while True:
            status_data = self.check_status(task_id)
            status = status_data.get("status")
            if status == "completed":
                return self.get_results(task_id)
            elif status == "failed":
                raise Exception(f"Training failed: {status_data.get('error')}")
            time.sleep(poll_interval)
