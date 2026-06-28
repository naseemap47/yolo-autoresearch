# YOLO Auto-Research Agentic Framework

> Autonomous YOLO model selection, hyperparameter tuning, and iterative training — powered by LangGraph, Ollama, FastAPI, and RAG.

This framework uses an LLM-driven agent to automatically analyze YOLO datasets, propose optimal model variants (e.g., `yolov8n.pt`, `yolov8m.pt`) and hyperparameters, and train them repeatedly until a target accuracy (mAP50-95) is reached.

A **RAG (Retrieval-Augmented Generation)** layer grounds the LLM planner in official Ultralytics documentation at every planning step, ensuring hyperparameter decisions reference real documentation rather than hallucinated defaults.

Because LLM generation and YOLO training are resource-heavy, the system is designed to run across two nodes:
1. **Agent Node (Local)**: Runs LLM inference (via Ollama), the RAG retriever (ChromaDB), and the LangGraph orchestrator.
2. **GPU Node (Remote)**: Runs a FastAPI server that handles dataset analysis and Ultralytics YOLO training.

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
3. **Pull Embedding Model** (required for RAG): Pull a local embedding model for the RAG retriever.
   ```bash
   ollama pull nomic-embed-text
   ```

### Installation
1. Clone this repository locally.
2. Install all dependencies (including ChromaDB for RAG):
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
  dataset_yaml_path: "/path/to/dataset/data.yaml"  # path on the GPU machine

rag:
  docs_path: "finetune/ultralytics_raw.txt"   # pre-scraped Ultralytics docs
  persist_dir: "agent/vectorstore"            # ChromaDB index location
  embedding_model: "nomic-embed-text"         # Ollama embedding model
  top_k: 4                                    # doc chunks injected per planning step
  chunk_size: 800
  chunk_overlap: 100
```
**Important:** `dataset_yaml_path` must be the absolute path to `data.yaml` *as it exists on the remote GPU PC*.

### Running the Agent
```bash
python -m agent.main
```

On **first run**, the RAG module will embed and index the Ultralytics documentation (~30 s, one-time). On all subsequent runs the index is loaded instantly from disk.

---

## 3. How It Works

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
 │         │                │    appends {cycle, config, results} to history       │
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
 └──────────────────────────────────────────────┘
```

1. **Ingest (first run only)** — `agent/rag.py` loads `finetune/ultralytics_raw.txt` (14 scraped Ultralytics pages), splits it into ~218 overlapping chunks, embeds them via `OllamaEmbeddings`, and persists the index to `agent/vectorstore/` using ChromaDB.

2. **Load (subsequent runs)** — the persisted ChromaDB index is loaded instantly from disk. No re-embedding required.

3. **Retrieval (each planning step)** — the planner builds a natural-language query from dataset characteristics (image count, class count) and the last training result (low / medium / high mAP), then retrieves the 4 most relevant documentation excerpts.

4. **Grounded generation** — the LLM prompt includes real Ultralytics documentation covering learning rate schedules, weight decay, `close_mosaic` behaviour, batch size guidance, data augmentation, and overfitting prevention. The LLM must reference these excerpts in its `reasoning` field.

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
**Cause**: The LLM model name has not been pulled, or the name is misspelled (e.g., `qwen:3.5-0.8b` instead of `qwen3.5:0.8b`).
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