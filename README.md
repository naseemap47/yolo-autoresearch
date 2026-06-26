# YOLO Auto-Research Agentic Framework

> Autonomous YOLO model selection, hyperparameter tuning, and iterative training — powered by LangGraph, Ollama, and FastAPI.

This framework uses an LLM-driven agent to automatically analyze YOLO datasets, propose optimal model variants (e.g., `yolov8n.pt`, `yolov8m.pt`) and hyperparameters, and train them repeatedly until a target accuracy (mAP50-95) is reached. 

Because LLM generation and YOLO training are resource-heavy, the system is designed to run across two nodes:
1. **Agent Node (Local)**: Runs the LLM inference (via Ollama) and LangGraph orchestrator.
2. **GPU Node (Remote)**: Runs a FastAPI server that handles heavy dataset analysis and Ultralytics YOLO training.

---

## 1. GPU Server Setup (Remote Machine)

This machine handles the YOLO model training. It must have GPU access and the dataset present on its local filesystem.

### Installation
1. Clone this repository on the GPU machine.
2. Install the required dependencies:
   ```bash
   uv sync
   ```

### Running the Server
The server reads `config/settings.yaml` for its host and port. Start it using the provided script:
```bash
python run_server.py
```

---

## 2. Agent Orchestrator Setup (Local Machine)

This machine controls the research loop.

### Prerequisites
1. **Ollama**: Ensure Ollama is installed and running locally.
2. **Pull LLM Model**: Pull the model you intend to use. The default is `qwen3.5:0.8b`.
   ```bash
   ollama pull qwen3.5:0.8b
   ```

### Installation
1. Clone this repository locally.
2. Install the required dependencies:
   ```bash
   uv sync
   ```

### Configuration
Edit the `config/settings.yaml` file to match your environment:
```yaml
server:
  host: "192.168.0.84"
  port: 8000

agent:
  gpu_host: "192.168.0.84"
  llm_model: "qwen3.5:0.8b"
  target_map: 0.8
  max_cycles: 5
  dataset_yaml_path: "/home/eternalcm/Downloads/Test/data.yaml"
```
**Important:** `dataset_yaml_path` must be the absolute path to `data.yaml` *as it exists on the remote GPU PC*.

### Running the Agent
Start the auto-research agent. It will automatically load your configuration from `config/settings.yaml`:

```bash
python -m agent.main
```

---

## Troubleshooting & Common Errors

Here are common issues you might encounter during setup and how to resolve them:

### 1. Agent Hangs on "Cycle 1: Analyzing dataset..."
**Issue**: The agent gets stuck trying to send the initial request and eventually throws a timeout or `urllib3` connection error.
**Cause**: The local Agent PC cannot reach port 8000 on the remote GPU PC.
**Solution**:
- Ensure the FastAPI server is running with the `--host 0.0.0.0` flag (not just default localhost).
- Ensure your firewall allows inbound TCP traffic on port 8000. For UFW on Ubuntu:
  ```bash
  sudo ufw allow 8000/tcp
  ```

### 2. HTTPError: 400 Client Error: Bad Request for url: .../dataset/analyze
**Issue**: The FastAPI server returns `{"detail":"Dataset yaml not found."}`.
**Cause**: The path you provided to the `--data` flag does not exist *on the remote GPU machine*.
**Solution**: Double-check the absolute path of the dataset on the GPU machine. The path provided to the agent must correspond exactly to where the file is located on the remote server's filesystem, not your local machine.

### 3. ollama._types.ResponseError: model '...' not found (status code: 404)
**Issue**: The LangGraph agent crashes during the "Planning next model" step with a 404 error from Ollama.
**Cause**: The LLM model name specified (either in the code or via `--llm`) has not been downloaded to your local Ollama registry, or the name is misspelled (e.g., `qwen:3.5-0.8b` instead of `qwen3.5:0.8b`).
**Solution**: 
- Run `ollama list` to see the exact names of the models you have pulled.
- Pass the correct name to the agent using the `--llm` flag, or pull the missing model using `ollama pull <model_name>`.

### 4. "Fallback configuration due to JSON parsing error" in Final History
**Issue**: The reasoning block in the output says the agent fell back to default parameters.
**Cause**: Small LLMs (like `0.5b` or `0.8b` parameters) sometimes fail to output strictly formatted JSON and may include conversational text. While the system attempts to cleanly extract the JSON, extreme hallucinations can cause parsing failures.
**Solution**: If you see this frequently, use a slightly larger model capable of stricter instruction following, such as `qwen2.5:3b` or `llama3`.