# yolo_autoresearch
AI agents running research on YOLO models

# YOLO Auto-Research Framework Setup and Usage

I have implemented the agentic framework based on your requirements. It is split into two components: the **Agent Node** (which runs locally using LangGraph and Ollama) and the **GPU Node** (which runs remotely on `192.168.0.84` using FastAPI and Ultralytics).

Here is a guide on how to set up and run the system.

## 1. Setup on the GPU PC (192.168.0.84)

Clone the repository to the GPU PC and navigate to the project directory.

> [!IMPORTANT]
> The dataset must be present on this machine, and you need to know the absolute path to its `data.yaml` file.

1. **Install Dependencies**
   Create a virtual environment and install the server requirements:
   ```bash
   pip install -r server/requirements.txt
   ```

2. **Start the FastAPI Server**
   Start the backend server on port `8000`:
   ```bash
   python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
   ```
   The server will now listen for data analysis and training requests.

## 2. Setup on the Agent Node (Local PC)

Ensure you have Ollama installed locally with the `qwen:3.5-0.8b` model pulled:
```bash
ollama run qwen:3.5-0.8b
```

1. **Install Dependencies**
   Create a virtual environment and install the agent requirements:
   ```bash
   pip install -r agent/requirements.txt
   ```

2. **Run the Agent Orchestrator**
   Start the auto-research agent by providing the path to the `data.yaml` *as it exists on the GPU PC*. For example, if the dataset is located at `/home/user/dataset/data.yaml` on the GPU PC, run:

   ```bash
   python -m agent.main --data /home/user/dataset/data.yaml --host 192.168.0.84 --target-map 0.8 --max-cycles 5
   ```

## System Flow

1. **Analysis (`agent/graph.py` -> `server/data_analyzer.py`)**: The agent asks the GPU server to analyze the specified YOLO dataset. The server samples the dataset, extracting image dimensions and bounding box statistics per class, and returns them to the agent.
2. **Planning (`agent/llm_planner.py`)**: The agent passes the dataset stats and any previous training history to the `qwen:3.5-0.8b` model via LangChain. The LLM acts as the "Brain", proposing a model (e.g., `yolov8n.pt`) and hyperparameters (epochs, learning rate, etc.).
3. **Training (`agent/api_client.py` -> `server/trainer.py`)**: The agent triggers the training on the GPU PC with the chosen configuration. The `server/trainer.py` executes YOLO asynchronously while the agent polls for status.
4. **Evaluation**: Once the training is complete, the server returns the metrics. The agent checks if `mAP50-95` >= `0.8`. If so, it stops. Otherwise, it feeds the results back into the LLM planner and starts the next cycle, up to `5` times.