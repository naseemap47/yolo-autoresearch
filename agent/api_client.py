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
        
    def start_training(self, model_name: str, data_yaml_path: str, epochs: int, batch_size: int, imgsz: int, lr0: float):
        url = f"{self.base_url}/train"
        data = {
            "model_name": model_name,
            "data_yaml_path": data_yaml_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "imgsz": imgsz,
            "lr0": lr0
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
