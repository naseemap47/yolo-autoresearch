# YOLO Auto-Research Agentic Framework

> Autonomous YOLO model selection, hyperparameter tuning, and iterative training — powered by LangGraph, Ollama, FastAPI, RAG, and **MLflow**.

This framework uses an LLM-driven agent to automatically analyze YOLO datasets, propose optimal model variants (e.g., `yolov8n.pt`, `yolov8m.pt`) and hyperparameters, and train them repeatedly until a target accuracy (mAP50-95) is reached.

A **RAG (Retrieval-Augmented Generation)** layer grounds the LLM planner in official Ultralytics documentation at every planning step, ensuring hyperparameter decisions reference real documentation rather than hallucinated defaults.

Every research cycle is tracked as a named **MLflow run** inside a single experiment, so you can compare all model variants and their results side-by-side in the MLflow UI without any manual bookkeeping.

Because LLM generation and YOLO training are resource-heavy, the system is designed to run across two nodes:
1. **Agent Node (Local)**: Runs LLM inference (via Ollama), the RAG retriever (ChromaDB), the LangGraph orchestrator, and logs MLflow runs.
2. **GPU Node (Remote)**: Runs a FastAPI server that handles dataset analysis and Ultralytics YOLO training, and also logs its own MLflow run per task.

---

## 1. How It Works

```
                          LANGGRAPH STATE MACHINE
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │                                                                                │
 │  AgentState                                                                    │
 │  ┌──────────────────────────────────────────────────────────────────────────┐  │
 │  │ dataset_yaml_path │ dataset_stats │ history │ current_config │ cycle ... │  │
 │  └──────────────────────────────────────────────────────────────────────────┘  │
 │                  │                                                              │
 │                  ▼                                                              │
 │         ┌────────────────┐                                                      │
 │         │  analyze_data  │  ← runs once; calls GPU server /dataset/analyze     │
 │         │                │    populates dataset_stats in state                  │
 │         └────────────────┘                                                      │
 │                  │                                                              │
 │                  ▼                                                              │
 │   ┌──────────────────────────┐        RAG RETRIEVER                            │
 │   │       plan_model         │◄──────────────────────────────────────────────┐ │
 │   │                          │  1. build semantic query from state            │ │
 │   │  Planner.plan(           │     (num_classes, total_images, last mAP)      │ │
 │   │    dataset_stats,        │                                                │ │
 │   │    history          )    │  2. UltralyticsRetriever.query(q)              │ │
 │   │                          │     ┌─────────────────────────────────────┐   │ │
 │   │  ┌────────────────────┐  │     │  ChromaDB  (agent/vectorstore/)     │   │ │
 │   │  │  OllamaLLM         │  │     │  ┌──────┐ ┌──────┐ ┌──────┐        │   │ │
 │   │  │  prompt:           │  │     │  │chunk │ │chunk │ │chunk │  ...   │   │ │
 │   │  │  {context}   ◄─────┼──┼─────┤  │ lr0  │ │ w_d  │ │ aug  │        │   │ │
 │   │  │  {dataset_stats}   │  │     │  └──────┘ └──────┘ └──────┘        │   │ │
 │   │  │  {history}         │  │     │  embedded from                      │   │ │
 │   │  └────────────────────┘  │     │  finetune/ultralytics_raw.txt       │   │ │
 │   │                          │     └─────────────────────────────────────┘   │ │
 │   │  → JSON config           │                                                │ │
 │   │    model_name            │  3. top-k chunks injected as {context}         │ │
 │   │    epochs                │     into LLM prompt                            │ │
 │   │    batch_size            │────────────────────────────────────────────────┘ │
 │   │    imgsz, lr0            │                                                  │
 │   │    weight_decay          │                                                  │
 │   │    close_mosaic          │                                                  │
 │   └──────────────────────────┘                                                  │
 │                  │                                                              │
 │                  ▼                                                              │
 │         ┌────────────────┐                                                      │
 │         │  train_model   │  ← sends config to GPU server /train                │
 │         │                │    polls /status/{task_id} until done                │
 │         │                │    (automatically halves batch size and retries      │
 │         │                │     if GPU runs out of memory or training fails)     │
 │         │                │    appends {cycle, config, results} to history       │
 │         │                │    ★ logs MLflow run (params + metrics + tags)       │
 │         └────────────────┘                                                      │
 │                  │                                                              │
 │                  ▼                                                              │
 │         ┌────────────────┐                                                      │
 │         │    evaluate    │  ← reads mAP50-95 from latest history entry          │
 │         └────────────────┘                                                      │
 │                  │                                                              │
 │       should_continue(state)?                                                   │
 │         ┌────────┴──────────┐                                                  │
 │    mAP >= target        cycle < max                                             │
 │    OR cycle >= max      AND mAP < target                                        │
 │         │                   │                                                  │
 │         ▼                   └────────────────────────────► back to plan_model  │
 │        END                                                (next research cycle) │
 │                                                                                │
 └────────────────────────────────────────────────────────────────────────────────┘

 GPU Server  (192.168.x.x:8000)
 ┌──────────────────────────────────────────────┐
 │  POST /dataset/analyze  →  dataset_stats      │
 │  POST /train            →  task_id            │
 │  GET  /status/{id}      →  {status}           │
 │  GET  /results/{id}     →  {mAP50-95, mAP50} │
 │  ★ logs MLflow run after each completed task  │
 └──────────────────────────────────────────────┘

 MLflow Tracking Server  (127.0.0.1:5000)
 ┌──────────────────────────────────────────────┐
 │  Experiment: yolo-autoresearch               │
 │  Run per cycle: cycle-N-<model_name>         │
 │  Params:  model_name, epochs, batch_size,    │
 │           imgsz, lr0, weight_decay,          │
 │           close_mosaic                       │
 │  Metrics: mAP50_95, mAP50, fitness, cycle   │
 │  Tags:    model_name, cycle, status,         │
 │           task_id, source                    │
 └──────────────────────────────────────────────┘
```

1. **Ingest (first run only)** — `agent/rag.py` loads `finetune/ultralytics_raw.txt` (14 scraped Ultralytics pages), splits it into ~218 overlapping chunks, embeds them via `OllamaEmbeddings`, and persists the index to `agent/vectorstore/` using ChromaDB.

2. **Load (subsequent runs)** — the persisted ChromaDB index is loaded instantly from disk. No re-embedding required.

3. **Retrieval (each planning step)** — the planner builds a natural-language query from dataset characteristics (image count, class count) and the last training result (low / medium / high mAP), then retrieves the 4 most relevant documentation excerpts.

4. **Grounded generation** — the LLM prompt includes real Ultralytics documentation covering learning rate schedules, weight decay, `close_mosaic` behaviour, batch size guidance, data augmentation, and overfitting prevention. The LLM must reference these excerpts in its `reasoning` field.

5. **MLflow logging** — after every successful training cycle the agent logs a named run to the MLflow tracking server, capturing all hyperparameters, the resulting mAP scores, and metadata tags so any two cycles can be compared in seconds via the UI.

## 2. GPU Server Setup (Remote Machine)

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

## 3. Agent Orchestrator Setup (Local Machine)

This machine controls the research loop.

### Prerequisites
1. **Ollama**: Ensure Ollama is installed and running locally.
2. **Pull LLM Model**: Pull the model you intend to use. The default is `qwen3.5:9b`.
   ```bash
   ollama pull qwen3.5:9b
   ```
3. **Pull Embedding Model** (required for RAG): Pull a local embedding model for the RAG retriever.
   ```bash
   ollama pull nomic-embed-text
   ```

### Installation
1. Clone this repository locally.
2. Install all dependencies (including MLflow, ChromaDB for RAG):
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
  llm_model: "qwen3.5:9b"
  target_map: 0.8
  max_cycles: 5
  dataset_yaml_path: "/path/to/dataset/data.yaml"  # path on the GPU machine

rag:
  docs_path: "finetune/ultralytics_raw.txt"   # pre-scraped Ultralytics docs
  persist_dir: "agent/vectorstore"            # ChromaDB index location
  embedding_model: "nomic-embed-text"         # Ollama embedding model
  top_k: 4                                    # doc chunks injected per planning step
  chunk_size: 800
  chunk_overlap: 100

mlflow:
  tracking_uri: "http://127.0.0.1:5000"       # MLflow server address
  experiment_name: "yolo-autoresearch"        # experiment name in the UI
```
**Important:** `dataset_yaml_path` must be the absolute path to `data.yaml` *as it exists on the remote GPU PC*.

### Running the Agent
```bash
python -m agent.main
```

On **first run**, the RAG module will embed and index the Ultralytics documentation (~30 s, one-time). On all subsequent runs the index is loaded instantly from disk.

---

## 4. MLflow Experiment Tracking

Every training cycle is automatically recorded as a **run** inside the `yolo-autoresearch` MLflow experiment. This gives you a full audit trail of every model variant and hyperparameter combination tried, plus a point-and-click comparison UI.

### What Is Logged

| Category | Fields |
|---|---|
| **Parameters** | `model_name`, `epochs`, `batch_size`, `imgsz`, `lr0`, `weight_decay`, `close_mosaic` |
| **Metrics** | `mAP50_95`, `mAP50`, `fitness`, `cycle` |
| **Tags** | `model_name`, `cycle`, `task_id`, `source` (`agent_orchestrator` / `gpu_server`), `status` (`success` / `failed`) |

Runs are named `cycle-N-<model_name>` (e.g. `cycle-2-yolov8m`) for instant identification in the UI.

### Starting the MLflow UI

On the **agent / local machine**, open a separate terminal and run:

```bash
mlflow ui --port 5000
```

Then open **http://127.0.0.1:5000** in your browser.

> You can also point the server to a shared network location so both the GPU node and agent node write to the same store:
> ```bash
> mlflow server --backend-store-uri sqlite:///mlflow.db \
>               --default-artifact-root ./mlruns \
>               --host 0.0.0.0 --port 5000
> ```
> Then set `tracking_uri: "http://<agent-ip>:5000"` in `settings.yaml` on both machines.

### Comparing Models in the UI

#### Step 1 — Open the Experiment
Navigate to **http://127.0.0.1:5000**, click **Experiments** in the left sidebar, and select **yolo-autoresearch**. You will see all runs listed in a table with their parameters and metrics at a glance.

#### Step 2 — Select Runs to Compare
Tick the checkbox next to the runs you want to compare (e.g. cycle-1-yolov8n vs cycle-3-yolov8m), then click the **Compare** button that appears at the top of the table.

#### Step 3 — View the Comparison Views

The MLflow compare page offers three views:

| View | How to use it |
|---|---|
| **Parallel Coordinates** | Each vertical axis is one parameter or metric. Drag the `mAP50_95` axis to the right end to see which parameter combinations produced the best mAP. Lines that reach a high mAP value on the right reveal winning configurations. |
| **Scatter Plot** | Select any two columns (e.g. `lr0` vs `mAP50_95`) to see correlation. Useful for spotting if a lower/higher learning rate consistently helps. |
| **Box Plot / Table** | Switch to the **Table** tab for a raw side-by-side view of every logged value. Sort by `mAP50_95` descending to rank models instantly. |

#### Step 4 — Drill into a Run
Click any run name to see its full detail page: parameters, metrics over time (if `step` logging is used), artifacts, and tags. The `task_id` tag links back to the GPU server log for that specific training job.

#### Tips
- **Filter by tag**: In the experiment view, use the search bar (`tags.model_name = "yolov8m.pt"`) to filter to a specific architecture.
- **Sort by metric**: Click the `mAP50_95` column header to sort all runs by accuracy instantly.
- **Download CSV**: Use **Download CSV** (top right of the runs table) to export all run data for offline analysis.
- **Failed runs**: Runs tagged `status=failed` appear in the list with no metrics — useful to see which configurations caused OOM errors.

---

### Dynamic Target mAP & Model Variety

- **Dynamic Target mAP**: The target accuracy (mAP50-95) is loaded dynamically from `config/settings.yaml` under `agent.target_map` and passed directly into the planning prompt template.
- **Diverse Model Architectures & Sizes**: The planner has access to a comprehensive table of modern YOLO variants (including YOLO26, YOLO12, YOLO11, YOLOv10, YOLOv9, YOLOv8, YOLOv5, YOLOv7, YOLOv6, YOLOv3, YOLOv4, YOLO-World, YOLOE, and RT-DETR) across multiple size categories (nano, small, medium, large, xlarge).

### Automatic Batch Size Reduction

To handle resource constraints on the remote GPU server, the agent automatically catches training failures (such as PyTorch CUDA Out Of Memory errors or connection resets due to crashes):
1. If training fails, the agent automatically halves the requested batch size (e.g., from 16 to 8, down to a minimum of 1).
2. It waits 5 seconds and resubmits the training request with the reduced batch size.
3. If training still fails even at `batch_size = 1`, the agent aborts execution and reports the original reason of the error.

### RAG-Enhanced Hyperparameters

In addition to `epochs`, `batch_size`, `imgsz`, and `lr0`, the RAG-informed planner also tunes:

| Parameter | Description | Tuning Range |
|---|---|---|
| `weight_decay` | L2 regularization to prevent overfitting | `0.0001 – 0.001` |
| `close_mosaic` | Epochs before end to disable mosaic augmentation | `0 – 15` |

### Updating the Documentation Source

The RAG source is `finetune/ultralytics_raw.txt`. To refresh it with updated or additional pages, delete `agent/vectorstore/` to force a re-index on the next run:

```bash
rm -rf agent/vectorstore/
python -m agent.main
```

---

## Troubleshooting & Common Errors

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
**Cause**: The path you provided does not exist *on the remote GPU machine*.
**Solution**: Double-check the absolute path of the dataset on the GPU machine. The path must correspond exactly to where the file is located on the remote server's filesystem, not your local machine.

### 3. ollama._types.ResponseError: model '...' not found (status code: 404)
**Issue**: The LangGraph agent crashes during the "Planning next model" step with a 404 error from Ollama.
**Cause**: The LLM model name has not been pulled, or the name is misspelled (e.g., `qwen:3.5-9b` instead of `qwen3.5:9b`).
**Solution**:
- Run `ollama list` to see the exact names of pulled models.
- Pull the missing model: `ollama pull <model_name>`.

### 4. "Fallback configuration due to JSON parsing error" in Final History
**Issue**: The reasoning block says the agent fell back to default parameters.
**Cause**: Small LLMs (like `0.5b` or `0.8b`) sometimes fail to produce strictly formatted JSON.
**Solution**: Use a slightly larger model capable of stricter instruction following, such as `qwen2.5:3b` or `llama3`.

### 5. [RAG] WARNING: Could not initialize RAG
**Issue**: The agent prints a RAG warning and continues without documentation context.
**Cause**: Either `chromadb` is not installed, the embedding model was not pulled, or `finetune/ultralytics_raw.txt` is missing.
**Solution**:
- Ensure dependencies are installed: `uv sync`
- Pull the embedding model: `ollama pull nomic-embed-text`
- Verify the docs file exists: `ls finetune/ultralytics_raw.txt`
- If the file is missing, run the scraping cell in `finetune/finetune.ipynb`

### 6. MLflow runs not appearing in the UI
**Issue**: The agent finishes but no runs show up at `http://127.0.0.1:5000`.
**Cause**: The MLflow tracking server is not running, or the `tracking_uri` in `settings.yaml` points to the wrong address.
**Solution**:
- Start the server **before** running the agent: `mlflow ui --port 5000`
- Confirm the URI in `config/settings.yaml → mlflow.tracking_uri` matches exactly.
- If the server is on a different machine, replace `127.0.0.1` with that machine's IP and ensure port 5000 is open in the firewall:
  ```bash
  sudo ufw allow 5000/tcp
  ```

### 7. mlflow.exceptions.MlflowException: Could not find experiment
**Issue**: The agent errors with an experiment-not-found message on first run.
**Cause**: Harmless — MLflow creates the experiment automatically on the first `mlflow.set_experiment()` call. If you see this error, the tracking server may not have been reachable at that moment.
**Solution**: Confirm the tracking server is running and retry. The experiment will be created on the next run.